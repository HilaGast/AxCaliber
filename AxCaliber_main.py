import nibabel as nib
import numpy as np
import os
from AxCaliber.load_and_save_files import *

def diff_data_registration(data, affine, bvec, bval, big_delta, small_delta, b0_ref=0):
    from dipy.align import register_dwi_series
    from dipy.core.gradients import gradient_table
    from scipy.ndimage import gaussian_filter

    gtab = gradient_table(bval, bvec, big_delta=big_delta, small_delta=small_delta)
    nbvec = np.zeros(np.shape(bvec))


    rdata, tform = register_dwi_series(data, gtab=gtab, affine=affine, b0_ref = b0_ref)
    rd = rdata.get_fdata()
    rdata_smooth = np.zeros(np.shape(rd))
    for i in range(0,len(bval)):
        nbvec[i,:] = np.dot(bvec[i,:],tform[:,:,i].transpose()[:3,:3])
        rdata_smooth[:,:,:,i] = gaussian_filter(rd[:,:,:,i],sigma=0.65,truncate=7)


    return rdata_smooth, tform, nbvec


def cart2sph(x,y,z):
    xy = np.sqrt(x ** 2 + y ** 2)  # sqrt(x² + y²)

    x_2 = x ** 2
    y_2 = y ** 2
    z_2 = z ** 2

    r = np.sqrt(x_2 + y_2 + z_2)  # r = sqrt(x² + y² + z²)

    theta = np.arctan2(y, x)

    phi = np.arctan2(xy, z)

    return r, theta, phi


def axcaliber(subj_folder,file_names, small_delta, big_delta, gmax, gamma_val=4257):
    import AxCaliber.ax_dti as ax_dti
    from scipy.stats import gamma
    from AxCaliber.CHARMED import simulate_charmed_main
    '''

    :param subj_folder:
    :param file_names:
    :param small_delta: in miliseconds
    :param big_delta: in miliseconds
    :param gmax: in G/ms = 0.1*(mT/m) [calculate by: sqrt(bval * 100 / (7.178e8 * 0.03 ^ 2 * (0.06 - 0.01)))]
    :return:
    '''

    #mkdir(fullfile(subj_folder, 'AxCaliber'))
    data, affine, info_nii, bval, bvec, mask = load_diff_files(subj_folder, file_names)


    # DTI on shell bvalue=1000:
    b1000_locs = np.where(bval == 1)[0] #b1000locs
    data1000 = data[:, :, :, np.insert(b1000_locs,0,0)]
    bvec1000 = np.insert(bvec[b1000_locs,:],0,bvec[0,:],axis=0)
    bval1000 = np.insert(bval[b1000_locs],0,bval[0])

    rdata1000, tform1000, nbvec = diff_data_registration(data1000, affine, bvec1000, bval1000, big_delta, small_delta, b0_ref=0)

    fa1000, md1000, dt1000, eigvec1000 = ax_dti.dti(bval1000,bvec1000,rdata1000,1,mask,parallel_processing=False)

    fa_name = f'{subj_folder}{os.sep}FA.nii.gz'
    save_nifti(fa_name, fa1000,affine)
    md_name = f'{subj_folder}{os.sep}MD.nii.gz'
    save_nifti(md_name, md1000, affine)

    b0_locs = np.where(bval == 0)[0]
    bval =  bval.astype('float64')
    bv_norm = np.sqrt(bval / np.max(bval)) #bvfac
    grad_dirs = bvec * bv_norm[np.newaxis].T
    r_q, theta_q, phi_q = cart2sph(grad_dirs[:, 0], grad_dirs[:, 1], -grad_dirs[:, 2])


    scan_param = {'nb0':b0_locs, 'small_delta':small_delta, 'big_delta':big_delta, 'gmax':gmax,'theta':theta_q, 'phi':phi_q} #Param
    scan_param['maxq'] = gamma_val * scan_param['small_delta'] * scan_param['gmax'] / 10e6
    scan_param['qdirs'] = grad_dirs * scan_param['maxq']
    r_q = np.asarray(r_q) * scan_param['maxq']
    scan_param['bval'] = 4. * np.pi ** 2 * r_q ** 2 * (scan_param['big_delta'] - scan_param['small_delta'] / 3)
    scan_param['r'] = r_q

    add_vals = np.asarray([x / 20 for x in range(1, 320, 2)])
    gamma_dist = gamma.pdf(add_vals,2,0,2) #gw3
    gamma_dist = gamma_dist / np.sum(gamma_dist)

    dwi_simulates = simulate_charmed_main(data, scan_param, affine, fa1000, dt1000, eigvec1000, grad_dirs, mask,add_vals, gamma_dist, bval, md1000)


    ##UNDISTORT
    from dipy.align import affine_registration
    from scipy.ndimage import gaussian_filter

    registered_dwi = np.zeros(dwi_simulates.shape)
    for i in range(0,grad_dirs.shape[0]):
        dwi_sim_slice = np.real(dwi_simulates[:,:,:,i]) #A1
        data_slice = data[:,:,:,i] #KK
        rdata_slice,tform = affine_registration(data_slice, dwi_sim_slice, affine, affine)
        grad_dirs[i,:] = np.dot(grad_dirs[i,:], tform.transpose()[:3,:3])
        registered_dwi[:,:,:,i] = gaussian_filter(rdata_slice,sigma=0.65,truncate=7) #rDWI

    grad_dirs[scan_param['nb0'],:] = 0





    #calculate DTI parameter for each b shel:
