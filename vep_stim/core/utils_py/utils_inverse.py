import numpy as np
from numpy.linalg import pinv, inv, norm

def wMNE(gain, sensor, alpha, weights=None):
    """
    weighted minimum norm estimate (Hämallinen ?)
    """
    N,M = gain.shape
    # if weights matrix is the identity one obtains the classical MNE (i.e. w/o depth weighting)
    # has the problem of wrongly localising deep sources on the surfaces
    if weights == None:
        weights = np.eye(M)
    # TODO: implement depth weighting. see Grech et al. 2008
    tmp1 = np.dot(gain, inv(weights))
    tmp2 = np.dot(tmp1,gain.T)
    tmp3 = pinv(tmp2 + alpha * np.eye(N))
    tmp4 = np.dot(inv(weights), gain.T)
    tmp5  = np.dot(tmp4, tmp3)
    source = np.dot(tmp5, sensor)
    return source

def eLORETA(gain, sensor, alpha):
    """
    exact low resolution brain electromagnetic tomography (Pascal-Marqui 2007)
    """
    # estimate diagonal weights matrix iteratively
    N, M = gain.shape
    weights = np.eye(M)
    for i in range(30):
        weights_last = weights.copy()
        tmp = pinv(np.dot(np.dot(gain, inv(weights)),gain.T) + alpha * np.eye(N))
        weights[np.diag_indices_from(weights)] = np.diag(np.sqrt(np.dot(np.dot(gain.T, tmp),gain)))
        # check if converged
        d = norm(weights.ravel() - weights_last.ravel()) / norm(weights_last.ravel())
        if d < 1e-6:
            print("eLORETA weights estimation converged.")
            break
    else: 
        print("WARNING : eLORETA weights estimation did not converge !")
    
    tmp1 = np.dot(gain, inv(weights))
    tmp2 = np.dot(tmp1,gain.T)
    tmp3 = pinv(tmp2 + alpha * np.eye(N))
    tmp4 = np.dot(inv(weights), gain.T)
    tmp5 = np.dot(tmp4, tmp3)
    source = np.dot(tmp5, sensor)    

    return source

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1]
    eval(cmd)(*sys.argv[2:])