import numpy as np
import os
from scipy.stats import gamma
from scipy.optimize import least_squares, lsq_linear
from prediction_functions import predict_hindered, predict_csf, predict_restricted
from load_and_save_files import save_nifti


def lin_least_squares_with_constraints(A, b, lb, ub):
    import cvxpy as cp
    x = cp.Variable(A.shape)[1]
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [lb <= x, x <= ub, sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(warm_start=True)
    except cp.SolverError:
        try:
            prob.solve(warm_start=True, solver=cp.ECOS)
        except cp.SolverError:
            x = np.zeros(x.shape)
            return x

    return x.value


def reg_func(x, ydata, pixpredictH, pixpredictR, pixpredictCSF,prcsf):
    #xt = 1 - x[0] - prcsf
    #newdata = x[1] * (x[0] * pixpredictH + xt * pixpredictR + prcsf * pixpredictCSF)
    newdata = x[1] * (x[0] * pixpredictH + (1-x[0]-prcsf) * pixpredictR + prcsf * pixpredictCSF)
    err = [newdata.T - ydata]
    err = np.matrix.flatten(err[0])

    return err


def jac_calc(x, ydata, pixpredictH, pixpredictR, pixpredictCSF,prcsf):
    jac = np.zeros((len(ydata),2))
    jac[:,0] = x[1]*pixpredictH - x[1]*pixpredictR
    jac[:,1] = x[0]*pixpredictH+(1-x[0]-prcsf) * pixpredictR + prcsf * pixpredictCSF

    return jac


def calc_add(subj_folder, mask, bval, affine, add_vals, smooth_dwi, scan_param, b0_locs, sEigval, dt1000, eigvec1000, grad_dirs, bshell, sMD):
    [X, Y, Z] = mask.shape

    pfr = np.zeros((X * Y * Z, 1))
    ph = np.zeros((X * Y * Z, 1))
    pcsf = np.zeros((X * Y * Z, 1))
    pasi = np.zeros((X * Y * Z, 1))
    CMDfr = np.zeros((X * Y * Z, 1))
    CMDfh = np.zeros((X * Y * Z, 1))
    CMDcsf = np.zeros((X * Y * Z, 1))
    pixpredictCSF = np.zeros((1, len(bval)))

    b0_dwi_mean = np.mean(smooth_dwi[:, :, :, b0_locs], axis=3, keepdims=False)  # B0s

    sMDi = sEigval[:, :, :, 2, :]
    dt1000 = np.real(dt1000)
    eigvec1000 = np.real(eigvec1000)


    decay_hindered, MD0 = predict_hindered(dt1000, mask, bval, grad_dirs, sMDi, bshell)

    D_mat = np.eye(3) * 4
    for k in range(0, len(bval)):
        pixpredictCSF[0, k] = np.exp(-4 * (np.dot(grad_dirs[k, :], np.dot(D_mat, grad_dirs[k, :].T))))

    prcsf = predict_csf(sMD[:, :, :, 0])

    n_spec = 160  # N
    L = np.eye(n_spec) + (0 - np.tril(np.ones(n_spec), -1) * np.triu(np.ones(n_spec), -1))
    L = np.append(L, np.zeros((n_spec, 2)), axis=1)

    alpha = 3
    beta = 2
    gamma_pdf = gamma.pdf(add_vals, a=alpha, scale=beta)

    yd = gamma_pdf * (np.pi * (add_vals / 2) ** 2)
    yd = yd / np.sum(yd)

    eigvec1000 = np.reshape(eigvec1000, [X * Y * Z, eigvec1000.shape[3]])
    b0_dwi_mean = np.reshape(b0_dwi_mean, [X * Y * Z, 1]);
    smooth_dwi = np.reshape(smooth_dwi, [X * Y * Z, smooth_dwi.shape[3]])
    decay_hindered = np.reshape(decay_hindered, [X * Y * Z, decay_hindered.shape[3]])
    MDi0 = np.reshape(MD0, [X * Y * Z, 1])
    prcsf = np.reshape(prcsf, [X * Y * Z, 1])
    paxsi = np.zeros((X * Y * Z, n_spec))
    mask = mask.reshape((X * Y * Z, 1))
    vCSF = pixpredictCSF[:]

    for i in range(0, X * Y * Z):
        if mask[i]:
            fvec = np.squeeze(eigvec1000[i, :])
            b0_mean = np.squeeze(b0_dwi_mean[i])
            ydata = np.squeeze(smooth_dwi[i, :]).T
            pixpredictH = np.squeeze(decay_hindered[i, :])
            vH = pixpredictH[:]
            vH[vH > 1] = 0
            decayR = predict_restricted(scan_param['theta'], scan_param['phi'], fvec, scan_param['r'],
                                        scan_param['big_delta'], scan_param['small_delta'], b0_mean, add_vals / 2,
                                        MDi0[i])
            vR = decayR / np.nanmax(decayR)
            vRes = np.dot(vR, yd)
            x0 = (0.5, 5000)
            min_val = (0, 0)
            max_val = (1, 20000)

            vRes = np.nan_to_num(vRes, nan=0)
            vR = np.nan_to_num(vR, nan=0)

            parameter_hat = \
                least_squares(reg_func, x0, bounds=(min_val, max_val), ftol=1e-6, xtol=1e-6, diff_step=1e-3,
                              jac = jac_calc, max_nfev=20000, args=(ydata, vH, vRes, np.matrix.flatten(vCSF.T), prcsf[i]))['x'] #jac = jac_calc
            CMDfh[i] = parameter_hat[0]
            CMDfr[i] = 1 - parameter_hat[0] - prcsf[i]
            CMDcsf[i] = prcsf[i]

            vdata = ydata / parameter_hat[1]
            preds = np.zeros((vR.shape[0], vR.shape[1] + 2))
            preds[:vR.shape[0], :vR.shape[1]] = vR
            preds[:, vR.shape[1]] = vH
            preds[:, vR.shape[1] + 1] = vCSF

            lb = np.zeros(len(yd) + 2)
            ub = np.ones(len(yd) + 2)
            lb[161] = prcsf[i] - 0.02
            ub[161] = prcsf[i] + 0.02

            Lambda = 1;

            Xprim = np.concatenate((preds, np.sqrt(Lambda) * L))
            yprim = np.concatenate((vdata, np.zeros((160))))
            # x = lsq_linear(Xprim, yprim.flatten(), bounds=(lb,ub))['x']
            x = lin_least_squares_with_constraints(Xprim, yprim, lb, ub)
            x[x < 0] = 0
            x = x / np.nansum(x)

            a_h = x[160]
            a_csf = prcsf[i]
            a_fr = 1 - a_csf - a_h
            if a_fr < 0:
                a_fr = 0

            nx = x[0:130]
            nx = nx / np.sum(nx)

            ph[i] = a_h
            pcsf[i] = a_csf
            pfr[i] = a_fr
            pasi[i] = np.sum(nx * add_vals[0: 130])
            paxsi[i, :] = x[0: 160]

    eigvec1000 = np.reshape(eigvec1000, (X, Y, Z, np.shape(eigvec1000)[1]))
    b0_dwi_mean = np.reshape(b0_dwi_mean, (X, Y, Z))
    smooth_dwi = np.reshape(smooth_dwi, (X, Y, Z, np.shape(smooth_dwi[1])))
    decay_hindered = np.reshape(decay_hindered, (X, Y, Z, np.shape(decay_hindered)[1]))
    MDi0 = np.reshape(MDi0, (X, Y, Z))
    prcsf = np.reshape(prcsf, (X, Y, Z))
    ph = np.reshape(ph, (X, Y, Z))
    pcsf = np.reshape(pcsf, (X, Y, Z))
    pfr = np.reshape(pfr, (X, Y, Z))
    pasi = np.reshape(pasi, (X, Y, Z))  # eMAD
    paxsi = np.reshape(paxsi, (X, Y, Z, np.shape(paxsi, 1)))  # eADD, probability of each value

    CMDfh = np.reshape(CMDfh, (X, Y, Z))
    CMDfr = np.reshape(CMDfr, (X, Y, Z))
    CMDcsf = np.reshape(CMDcsf, (X, Y, Z))

    pasi_name = f'{subj_folder}{os.sep}AxSI{os.sep}eMAD.nii.gz'
    save_nifti(pasi_name, pasi, affine)

    pfr_name = f'{subj_folder}{os.sep}AxSI{os.sep}pfr.nii.gz'
    save_nifti(pfr_name, pfr, affine)

    ph_name = f'{subj_folder}{os.sep}AxSI{os.sep}ph.nii.gz'
    save_nifti(ph_name, ph, affine)

    pcsf_name = f'{subj_folder}{os.sep}AxSI{os.sep}pcsf.nii.gz'
    save_nifti(pcsf_name, pcsf, affine)
