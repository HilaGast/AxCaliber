import numpy as np


def predict_csf(md_b0): #pbrain
    from sklearn.mixture import GaussianMixture
    md_b0 = md_b0.reshape(-1)#zi0
    md_b0 = md_b0[md_b0 > 0]

    sigma = np.zeros([1,3])
    sigma[0] = 0.3 * np.nanmax(md_b0)*0.7
    sigma[1] = 0.3 * np.nanmax(md_b0)*0.2
    sigma[2] = 0.3 * np.nanmax(md_b0)*0.1

    params = {'n_comp':3,'mu':[0.5,1,2],'sigma':sigma,'tol':1e-7,'max_iter':10000,'init_params':'random'} #s

    gm = GaussianMixture(n_components = params['n_comp'],covariance_type='full',tol=params['tol'],max_iter=params['max_iter'],init_params=params['init_params'],weights_init=params['sigma'],means_init=params['mu']).fit(md_b0) #obj


###########################################################################################
    ################################################################
    ##############################################################################

    p1 = zeros(size(im));
    p2 = zeros(size(im));
    p3 = zeros(size(im));

    objp1 = gmdistribution(obj.mu(1), obj.Sigma(1, 1, 1), 1);
    objp2 = gmdistribution(obj.mu(2), obj.Sigma(1, 1, 2), 1);
    objp3 = gmdistribution(obj.mu(3), obj.Sigma(1, 1, 3), 1);

    [xlocs, ylocs, zlocs] = ind2sub(size(im), find(im > 0));

    for i=1:length(xlocs)
    p1(xlocs(i), ylocs(i), zlocs(i)) = pdf(objp1, im(xlocs(i), ylocs(i), zlocs(i)));
    p2(xlocs(i), ylocs(i), zlocs(i)) = pdf(objp2, im(xlocs(i), ylocs(i), zlocs(i)));
    p3(xlocs(i), ylocs(i), zlocs(i)) = pdf(objp3, im(xlocs(i), ylocs(i), zlocs(i)));
    end

    tp = p1 + p2 + p3;
    % p1 = p1. / tp;
    % p2 = p2. / tp;
    p3 = p3. / tp;

    p4 = p3;
    % p4(find(p4 > 0.4)) = 0.4;
    % p4 = p4. / 0.4;
    prcsf = p4;





def predict_hindered():





def predict_restricted():