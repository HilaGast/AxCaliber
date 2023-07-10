import numpy as np


def exp1(x,a,b):
    y = a*np.exp(b*x)
    return y


def predict_csf(md_b1000): #pbrain
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm
    x,y,z = md_b1000.shape
    md_non0 = md_b1000.reshape(-1,1)#zi0
    md_non0[md_non0 <= 0] = np.nan
    md_non0 = md_non0[np.isfinite(md_non0)].reshape(-1,1)

    sigma = np.zeros(3)
    sigma[0] = 0.3 * np.nanmax(md_non0)*0.7
    sigma[1] = 0.3 * np.nanmax(md_non0)*0.2
    sigma[2] = 1-sigma[0]-sigma[1]

    params = {'n_comp':3,'mu':np.asarray([0.5,1,2]).reshape((3,1)),'sigma':sigma,'tol':1e-7,'max_iter':10000,'init_params':'random'} #s

    gm = GaussianMixture(n_components = params['n_comp'],covariance_type='full',tol=params['tol'],max_iter=params['max_iter'],init_params=params['init_params'],weights_init=params['sigma'],means_init=params['mu']).fit(md_non0) #obj

    comp_mu = gm.means_
    comp_sigma = gm.covariances_

    md_b0 = md_b1000.reshape(-1,1)
    iloc = np.where(md_b0 > 0)[0]
    p1 = np.zeros(md_b0.shape)
    p2 = np.zeros(md_b0.shape)
    p3 = np.zeros(md_b0.shape)

    for i in iloc:
        p1[i,0] = norm.pdf(md_b0[i,0], comp_mu[0,0], np.sqrt(comp_sigma[0,0]))
        p2[i,0] = norm.pdf(md_b0[i,0], comp_mu[1,0], np.sqrt(comp_sigma[1,0]))
        p3[i,0] = norm.pdf(md_b0[i,0], comp_mu[2,0], np.sqrt(comp_sigma[2,0]))

    tp = p1 + p2 + p3
    prcsf = p3 / tp
    prcsf = prcsf.reshape(x, y, z)
    prcsf[np.isnan(prcsf)] = 0

    return prcsf


def predict_hindered(dt1000, mask, bval, grad_dirs, sMDi, bshell):
    from scipy.optimize import curve_fit
    len_bval = len(bval)
    X, Y, Z = mask.shape
    md0 = np.zeros((X*Y*Z,1))
    dt1000_maps = 1000 * dt1000 #DTmaps1
    decay = np.zeros((X*Y*Z, len_bval))

    dt1000_maps = np.asarray([[dt1000_maps[:,:,:,0],dt1000_maps[:,:,:,1],dt1000_maps[:,:,:,2]],[dt1000_maps[:,:,:,1],dt1000_maps[:,:,:,3],dt1000_maps[:,:,:,4]],[dt1000_maps[:,:,:,2],dt1000_maps[:,:,:,4],dt1000_maps[:,:,:,5]]])
    dt1000_maps = np.moveaxis(dt1000_maps,[0,1],[-1,-2])

    bloc = np.where(bshell > 0.99)[0]
    bshell_nonzero = bshell[bloc]
    midi = sMDi[:,:,:, bloc - 1]
    midi = midi.reshape((X * Y * Z, len(bloc)))
    dt1000_maps = dt1000_maps.reshape((X * Y * Z, dt1000_maps.shape[3], dt1000_maps.shape[4]))
    mask = mask.reshape((X*Y*Z,1))

    for i in range(0,X*Y*Z):
        if mask[i]:
            f_a = curve_fit(exp1, bshell_nonzero, midi[i,:])[0][0]
            fac = f_a / midi[i, 0]
            #MD0i = 1.0 * f.a;
            D_mat = fac * dt1000_maps[i,:,:]
            for j in range(0,len_bval):
                decay[i, j] = np.exp(-np.max(bval) * (np.dot(grad_dirs[j,:], np.dot(D_mat, grad_dirs[j,:].T))))
            md0[i] = f_a

    decay = decay.reshape((X, Y, Z, len_bval))
    md0 = md0.reshape((X, Y, Z))

    return decay, md0


def predict_restricted(theta_q, phi_q, vec, R_q, bigdel, smalldel, B0sp, R, MD):
    from AxSI_main import cart2sph

    a = R
    l_q = len(R_q)
    l_a = len(a)
    R_mat = np.tile(a, (l_q, 1))
    gamma = np.ones(a.shape)
    gamma_matrix = np.tile(gamma, (l_q, 1))
    D_r = np.tile(MD, (1, 160))
    D_r = np.tile(D_r, (l_q, 1))
    M0 = B0sp
    r_n, phi_n, theta_n = cart2sph(vec[0], vec[1], -vec[2])
    factor_angle_term_par = abs(np.cos(theta_q) * np.cos(theta_n) * np.cos(phi_q - phi_n) + np.sin(theta_q) * np.sin(theta_n))
    factor_angle_term_perp = np.sqrt(1 - factor_angle_term_par ** 2)
    q_par_sq = (R_q * factor_angle_term_par)** 2
    q_par_sq_matrix = np.tile(q_par_sq, (l_a,1)).T
    q_perp_sq = (R_q * factor_angle_term_perp)** 2
    q_perp_sq_matrix = np.tile(q_perp_sq, (l_a,1)).T
    E = np.exp(-4 * np.pi ** 2 * q_perp_sq_matrix * R_mat** 2)
    E = np.exp(-4 * np.pi ** 2 * q_par_sq_matrix * (bigdel - smalldel / 3) * D_r) * E
    decay = M0 * (E * gamma_matrix)

    return decay





