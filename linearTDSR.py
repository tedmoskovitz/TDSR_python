import numpy as np


def linearTDSR(X, r, u=None, alpha=None, opto=None, optotype='none'):
    """
     Linear TD-SR algorithm
    
     USAGE: results = linearTDRL(X,r,W,alpha,opto,optotype)
    
     INPUTS:
       X - [N x D] stimulus sequence (N = # trials, D = # features)
       r - [N x 1] reward sequence
       u - (optional) - initial values of the features (default: all zeros)
       alpha (optional) - 2 element vector with the SR Matrix alpha first,
                          and the TDRL alpha second (default: 0.06, 0.03)
       opto (optional) - sequence of optogenetic perturbations (default: all zeros)
       optotype (optional) - string of 'none' (default) or 'tonic' for
                             prolonged stimulation or 'phasic' for short
                             stimulation
    
     OUTPUTS:
       results - [N x 1] structure with the following fields:
                   .R - reward estimate
                   .V - value estimate
                   .dt - prediction error
                   .W - weight matrix
    
     Matt Gardner, 2018 based on Sam Gershman's code; Ported to Python3 by Ted Moskovitz, 2019
    """
    N, D = X.shape
    X = np.vstack([X, np.zeros([1, D])]) # add buffer at the end
    W = np.zeros([D,D])
    r = np.array([list(r) + [0]]).reshape(-1,1)

    if u is None: u = np.zeros([D,]); 
    if alpha is None: 
        alpha_W = 0.2
        alpha_u = 0.1
    else:
        alpha_W = alpha[0]
        alpha_u = alpha[1]
    if opto is None: opto = np.zeros(N); 

    gamma = 0.95
    results = []

    # For these experiments, we're only using currently active elements for
    # opto stimulation such that lambda = 0
    Lambda = 0
    E = 0 

    for n in range(0, N):
        
        if optotype == "none":
            PE = gamma * X[n+1,:] - X[n,:]
            if len(W.shape) == 1: W = W.reshape(-1,1);
            dt = X[n,:] + np.dot(PE.reshape(1,-1), W)

        elif optotype == "tonic":
            dt = (1 + opto[n]) * (X[n,:] + np.dot(gamma * X[n+1,:] - X[n,:], W))
              
        elif optotype == "phasic":
            E = Lambda * E + X[n,:]
            dt = X[n,:] + np.dot(gamma * X[n+1,:] - X[n,:], W) + opto[n] * E 

        
       
        # this gets the error for U
        du = r[n] - np.dot(X[n,:], u.reshape(-1,))

        # This corrects for the excitatory/inhibitory assymetry by reducing the inhibitory components
        # Inh to Exc ratio currently set at 1:4
        dt[dt < 0] = dt[dt < 0] / 4 
        du[du < 0] = du[du < 0] / 4

        # store results 
        result = {}
        result['R'] = np.dot(X[n,:], u)
        #print (X.shape, W.shape, u.shape)
        result['V'] = np.dot(np.dot(X[n,:].reshape(1,-1), W), u.reshape(-1,1))[0]
        result['W'] = W 
        result['U'] = u
        result['dt'] = dt 
        result['AllV'] = np.dot(W, u)
        results.append(result)

        # update 
        W = W + alpha_W * np.dot(X[n,:].reshape(-1,1), dt.reshape(1,-1))
        u = u + alpha_u * np.dot(X[n,:].reshape(-1,1), du.reshape(1,-1))

    return results 







    
