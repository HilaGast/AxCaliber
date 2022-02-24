import numpy as np

def simulate_charmed_main(rdata, scan_param, affine, fa1000, dt1000, eigvec1000, grad_dirs, mask,add_vals, gamma_dist, bval, md1000):

    rb0_data = rdata[:,:,:, scan_param['nb0']] #B0s

    b0_map = np.mean(rb0_data, axis=3)

    dwi_simulates = simchm(b0_map, fa1000, dt1000, eigvec1000, grad_dirs, mask, scan_param, add_vals, gamma_dist, bval, md1000)

    for i in range(0,len(bval)):
        dwi_sim1 = dwi_simulates[:,:,:, i] #A1
        dwi_sim1_vec = dwi_sim1.reshape(-1) #A2
        dwi_sim1_vec[dwi_sim1_vec<0] = 0
        dwi_sim1_vec_sorted = np.sort(dwi_sim1_vec) #mA2
        th_loc = round(0.995 * len(dwi_sim1_vec_sorted)) #loc1
        dwi_sim1[dwi_sim1 > (dwi_sim1_vec_sorted[th_loc])] = dwi_sim1_vec_sorted[th_loc]
        dwi_simulates[:,:,:, i] = dwi_sim1

    return dwi_simulates


def simchm(b0_map,fa,dt,eigvec, grad_dirs, mask, scan_param, add_vals, gamma_dist, bval, md):
    from AxCaliber_main import cart2sph

    x_len, y_len, z_len = np.shape(mask)
    vec_len = x_len*y_len*z_len
    mask_vec = mask.reshape(vec_len)
    index_mask = np.where(mask_vec>0)[0]

    xlocs, ylocs, zlocs = np.unravel_index(index_mask, shape=np.shape(mask))
    dwi_simulates = np.zeros((np.shape(mask)[0],np.shape(mask)[1],np.shape(mask)[2],len(bval))) #dwis


    len_av = len(add_vals) # l_a
    len_r = len(scan_param['r']) #l_q
    r_mat = np.repeat(add_vals[np.newaxis], len_r, axis=0) #R_mat

    gamma = gamma_dist / np.nansum(gamma_dist)
    gamma_matrix = np.repeat(gamma[np.newaxis], len_r, axis=0)

    for x,y,z in zip(xlocs,ylocs,zlocs):
        b0_signal = b0_map[x,y,z] #M
        hindered_fraction = 1 - fa[x,y,z] #f_h
        restricted_fraction = fa[x,y,z] #f_r
        r_sim, theta_sim, phi_sim = cart2sph(eigvec[x,y,z,0],eigvec[x,y,z,1],-eigvec[x,y,z,2])
        md_i = md[x,y,z] #D_r
        dt_mat = np.asarray([[dt[x,y,z,0], dt[x,y,z,1], dt[x,y,z,2]],[dt[x,y,z,1], dt[x,y,z,3], dt[x,y,z,4]],[dt[x,y,z,2], dt[x,y,z,4], dt[x,y,z,5]]]) #D_mat
        estimate_hindered = np.zeros(len(bval))

        for bi in range(0,len(bval)):
            estimate_hindered[bi] = hindered_fraction*np.exp(-4*np.dot(grad_dirs[0,:],np.dot(1000*dt_mat,grad_dirs[0,:][np.newaxis].T)))


        factor_angle_term_par = abs(np.cos(scan_param['theta']) * np.cos(theta_sim) * np.cos(scan_param['phi'] - phi_sim) +
                                np.sin(scan_param['theta']) * np.sin(theta_sim)) #figure out the names
        factor_angle_term_perp = np.sqrt(1 - factor_angle_term_par ** 2) #figure out the names
        q_par_sq = (scan_param['r'] * factor_angle_term_par) ** 2 #figure out the names
        q_par_sq_matrix = np.repeat(q_par_sq[np.newaxis].T, len_av, axis=1) #figure out the names
        q_perp_sq = (scan_param['r'] * factor_angle_term_perp) ** 2 #figure out the names
        q_perp_sq_matrix = np.repeat(q_perp_sq[np.newaxis].T, len_av, axis=1) #figure out the names
        exp_q_perp = np.exp(-4 * np.pi ** 2 * q_perp_sq_matrix * r_mat ** 2) #E
        exp_q_par = np.exp(-4 * np.pi ** 2 * q_par_sq_matrix * (scan_param['big_delta'] - scan_param['small_delta'] / 3) * md_i) * exp_q_perp #E1
        E_r = np.sum(exp_q_par * gamma_matrix, axis=1, keepdims=True) #figure out the names
        dwi_simulates[x,y,z,:] = b0_signal * (restricted_fraction * E_r.T + estimate_hindered)

    return dwi_simulates

