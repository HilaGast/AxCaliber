import nibabel as nib
import numpy as np
import os
from load_and_save_files import *
from scipy.ndimage import gaussian_filter
from calc_AxSI import *

def diff_data_registration(data, affine, bvec, bval, big_delta, small_delta, b0_ref=0):
    from dipy.align import register_dwi_series
    from dipy.core.gradients import gradient_table
    '''
    :param data: 4-D ndarray
    :param affine: (4,4) ndarray
    :param bvec: (v,3) ndarray, v is the number of volume
    :param bval: (v,) ndarray
    :param big_delta: float, diffusion time, in ms
    :param small_delta: float, diffusion gradient duration, in ms
    :param b0_ref: index of B=0 volume for reference
    :return:
        rd: 4-D ndarray, registered data
        tform: (4,4) ndarray, transformation matrix
        nbvec: (v,3) ndarray, adjusted bvec
    '''

    gtab = gradient_table(bval, bvec, big_delta=big_delta, small_delta=small_delta)
    nbvec = np.zeros(np.shape(bvec))


    rdata, tform = register_dwi_series(data, gtab=gtab, affine=affine, b0_ref = b0_ref)
    rd = rdata.get_fdata()
    for i in range(0,len(bval)):
        nbvec[i,:] = np.dot(bvec[i,:],tform[:,:,i].transpose()[:3,:3])

    return rd, tform, nbvec


def cart2sph(x,y,z):
    xy = np.sqrt(x ** 2 + y ** 2)  # sqrt(x² + y²)

    x_2 = x ** 2
    y_2 = y ** 2
    z_2 = z ** 2

    r = np.sqrt(xy ** 2 + z_2)  # r = sqrt(x² + y² + z²)
    phi = np.arctan2(z, xy)
    theta = np.arctan2(y, x)



    return r, theta, phi


def scan_param_dict(b0_locs, small_delta, big_delta, gmax, gamma_val, grad_dirs):
    r_q, phi_q, theta_q = cart2sph(grad_dirs[:, 0], grad_dirs[:, 1], -grad_dirs[:, 2])
    scan_param = {'nb0':b0_locs, 'small_delta':small_delta, 'big_delta':big_delta, 'gmax':gmax,'theta':theta_q, 'phi':phi_q} #Param
    scan_param['max_q'] = gamma_val * scan_param['small_delta'] * scan_param['gmax'] / 10e6
    scan_param['q_dirs'] = grad_dirs * scan_param['max_q']
    r_q = np.asarray(r_q) * scan_param['max_q']
    scan_param['bval'] = 4. * np.pi ** 2 * r_q ** 2 * (scan_param['big_delta'] - scan_param['small_delta'] / 3)
    scan_param['r'] = r_q

    return scan_param


def _data_prep_4_dti1000(data, bval, bvec, affine, big_delta, small_delta, preprocessed):
    b1000_locs = np.where(bval == 1)[0]
    data1000 = data[:, :, :, np.insert(b1000_locs,0,0)]
    bvec1000 = np.insert(bvec[b1000_locs,:],0,bvec[0,:],axis=0)
    bval1000 = np.insert(bval[b1000_locs],0,bval[0])

    if preprocessed:
        rdata1000 = data1000
        tform1000 = affine
        nbvec = bvec1000
    else:
        b0_locs = np.where(bval == 0)[0]
        rdata1000, tform1000, nbvec = diff_data_registration(data1000, affine, bvec1000, bval1000, big_delta, small_delta, b0_ref=b0_locs[0])

    # Smoothing the data:
    for i in range(0,len(bval1000)):
        rdata1000[:,:,:,i] = gaussian_filter(rdata1000[:,:,:,i],sigma=0.65,truncate=7)

    return rdata1000, tform1000, nbvec, bval1000, bvec1000


def axcaliber(subj_folder,file_names, small_delta, big_delta, gmax, gamma_val=4257, preprocessed=True, save_files=True):
    import ax_dti as ax_dti
    from scipy.stats import gamma
    from CHARMED import simulate_charmed_main
    '''
    :param subj_folder: 
    :param file_names: 
    :param small_delta: 
    :param big_delta: 
    :param gmax: 
    :param gamma_val: 
    :param preprocessed: 
    :return: 
    '''

    if not os.path.exists(f'{subj_folder}{os.sep}AxSI'):
        os.mkdir(f'{subj_folder}{os.sep}AxSI')
    data, affine, info_nii, bval, bvec, mask = load_diff_files(subj_folder, file_names)

    # Prepare data for DTI on shell bvalue=1000:
    rdata1000, tform1000, nbvec, bval1000, bvec1000 = _data_prep_4_dti1000(data, bval, bvec, affine, big_delta, small_delta, preprocessed)

    # DTI on shell bvalue=1000:
    fa1000, md1000, dt1000, eigvec1000 = ax_dti.dti(bval1000,bvec1000,rdata1000,1,mask,parallel_processing=False)[:4]

    # Save DTI results:
    if save_files:
        fa_name = f'{subj_folder}{os.sep}AxSI{os.sep}FA.nii.gz'
        save_nifti(fa_name, fa1000,affine)
        md_name = f'{subj_folder}{os.sep}AxSI{os.sep}MD.nii.gz'
        save_nifti(md_name, md1000, affine)

    b0_locs = np.where(bval == 0)[0]
    bval =  bval.astype('float64')
    bv_norm = np.sqrt(bval / np.max(bval)) #bvfac
    grad_dirs = bvec * bv_norm[np.newaxis].T

    scan_param = scan_param_dict(b0_locs, small_delta, big_delta, gmax, gamma_val, grad_dirs)

    add_vals = np.asarray([x / 10 for x in range(1, 320, 2)])
    gamma_dist = gamma.pdf(add_vals,a=2,scale=2) #gw3
    gamma_dist = gamma_dist / np.sum(gamma_dist)

    dwi_simulates = simulate_charmed_main(data, scan_param, fa1000, dt1000, eigvec1000, grad_dirs, mask, add_vals, gamma_dist, bval, md1000)
    dwi_simulates = np.nan_to_num(dwi_simulates, nan = 0)

    ##UNDISTORT
    smooth_dwi = np.zeros(dwi_simulates.shape)
    for i in range(0,grad_dirs.shape[0]):
        data_slice = data[:,:,:,i] #KK
        smooth_dwi[:,:,:,i] = gaussian_filter(data_slice,sigma=0.65,truncate=7) #rDWI

    grad_dirs[scan_param['nb0'],:] = 0



    #calculate DTI parameter for each b shel:

    bshell = np.asarray(list(set(bval))) #former - list

    sFA = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], len(bshell) - 1))
    sMD = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], len(bshell) - 1))
    sEigvec = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3, len(bshell) - 1))
    sEigval = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3, len(bshell) - 1))
    sDT = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 6, len(bshell) - 1))

    for i in range(1,len(bshell)):
        sFA[:,:,:, i - 1], sMD[:,:,:, i - 1], sDT[:,:,:,:, i-1], sEigvec[:,:,:,:, i - 1], sEigval[:,:,:,:,i-1]=ax_dti.dti(bval, grad_dirs, smooth_dwi, bshell[i], mask,parallel_processing=False)

    calc_add(subj_folder, mask, bval, affine, add_vals, smooth_dwi, scan_param, b0_locs, sEigval, dt1000, eigvec1000, grad_dirs,
              bshell, sMD)

