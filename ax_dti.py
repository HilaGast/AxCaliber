import numpy as np
import sys, os

def dti_calc(index_mask,signal_log,bval_mat,dt,fa,md,eigvec, eigval):
    for i in index_mask:
        signal_log_i = -1*(signal_log[i,:].squeeze()) #Zi
        signal_log_i_norm = np.linalg.lstsq(bval_mat,signal_log_i[np.newaxis].T,rcond=None)[0]
        dt_i = np.asarray([[signal_log_i_norm[0], signal_log_i_norm[1], signal_log_i_norm[2]],[signal_log_i_norm[1], signal_log_i_norm[3], signal_log_i_norm[4]],[signal_log_i_norm[2], signal_log_i_norm[4], signal_log_i_norm[5]]])

        [eigen_val_i,eigen_vec_i] = np.linalg.eig(dt_i.transpose()) #EigenValues, EigenVectors
        index = np.argsort(eigen_val_i)
        eigen_val_i = eigen_val_i[:,index]*1000
        eigen_val_i = eigen_val_i.squeeze()
        eigen_vec_i = eigen_vec_i[:,index]
        eigen_vec_i = eigen_vec_i.squeeze()
        eigen_val_i_org = eigen_val_i

        if (eigen_val_i < 0).all():
            eigen_val_i=np.abs(eigen_val_i)
        eigen_val_i[np.where(eigen_val_i<=0)[0]] = sys.float_info.epsilon


        md_i = (eigen_val_i[0] + eigen_val_i[1] + eigen_val_i[2]) / 3 #MDv

        fa_i = np.sqrt(1.5) * (np.sqrt((eigen_val_i[0] - md_i)**2 + (eigen_val_i[1] - md_i)**2 + (eigen_val_i[2] - md_i)**2) / np.sqrt(eigen_val_i[0]**2 + eigen_val_i[1]**2 + eigen_val_i[2]**2));

        fa[i] = fa_i
        eigvec[i,:] = eigen_vec_i[:, -1] * eigen_val_i_org[-1]
        eigval[i,:] = eigen_val_i
        md[i] = md_i
        dt_i = np.reshape(dt_i,-1)
        dt[i,:]=[dt_i[0],dt_i[1],dt_i[2],dt_i[4],dt_i[5],dt_i[8]]

    return dt,fa,md,eigvec, eigval


def dti(bval,bvec,data,bvalue,mask,parallel_processing=True):
    '''

    :param bval: the b0 + b-value vector of the shell divided by 1000 (list or ndarray of int or float)
    :param bvec:
    :param data:
    :param bvalue: the b-value shell divided by 1000 (int or float)
    :param mask:
    :return:
    '''
    bval = np.asarray(bval)
    bvec = np.asarray(bvec)
    data = np.asarray(data)
    mask = np.asarray(mask)

    norm_bvec = np.zeros((len(bval), np.shape(bvec)[1]))
    for i in range(0,len(bval)):
        norm_bvec[i,:] = bvec[i,:]/np.linalg.norm(bvec[i,:])

    bval_real = bval * 1000; #Bvalue
    b0locs = np.where(bval_real == 0)[0]
    bvlocs = np.where(bval_real == bvalue * 1000)[0]

    signal_0 = data[:,:,:, b0locs] #S0
    signal = data[:,:,:, bvlocs] #S
    bval_real = bval_real[bvlocs]
    norm_bvec = norm_bvec[bvlocs,:] #H

    signal_0 = np.nanmean(signal_0, axis=3)

    b = np.zeros((3, 3, np.shape(norm_bvec)[0]))
    for i in range(0,np.shape(norm_bvec)[0]):
        b[:,:, i] = np.dot(bval_real[i] , np.dot(norm_bvec[i,:][np.newaxis].T, norm_bvec[i,:][np.newaxis]))

    signal_log = np.zeros(np.shape(signal), dtype = np.float64)  #Slog
    eps = sys.float_info.epsilon
    for i in range(0,np.shape(norm_bvec)[0]):
        signal_log[:,:,:,i]=np.log(signal[:,:,:, i]/ signal_0 + eps)
        if np.isnan(signal_log[:,:,:,i]).all():
            signal_log[:, :, :, i] = 0
    signal_log[np.isnan(signal_log)] = 0
    signal_log[np.isinf(signal_log)] = 0
    bval_mat = np.asarray([b[0, 0,:], 2 * b[0, 1,:], 2 * b[0, 2,:], b[1, 1,:], 2 * b[1, 2,:], b[2, 2,:]]).transpose().squeeze() #Bv

    x_len, y_len, z_len = np.shape(mask)
    vec_len = x_len*y_len*z_len

    dt = np.zeros((vec_len, 6), dtype=np.float32)
    eigval = np.zeros((vec_len, 3), dtype=np.float32)
    fa = np.zeros((vec_len,1), dtype=np.float32)
    md = np.zeros((vec_len,1), dtype=np.float32)
    eigvec = np.zeros((vec_len, 3), dtype=np.float32)

    signal_log = np.reshape(signal_log, (vec_len,-1))

    mask_vec = mask.reshape(vec_len)
    index_mask = np.where(mask_vec>0)[0]
    if parallel_processing:
        from multiprocessing import Pool

        #pool = Pool(os.cpu_count())
        #pool.map(dti_func,input_data)
    else:
        dt,fa,md,eigvec, eigval = dti_calc(index_mask, signal_log, bval_mat, dt, fa,md,eigvec, eigval)

    dt = np.reshape(dt, (x_len, y_len, z_len, -1))
    fa = np.reshape(fa, (x_len, y_len, z_len))
    md = np.reshape(md, (x_len, y_len, z_len))
    eigvec = np.reshape(eigvec, (x_len, y_len, z_len, -1))
    eigval = np.reshape(eigval, (x_len, y_len, z_len, -1))

    return fa, md, dt, eigvec, eigval



