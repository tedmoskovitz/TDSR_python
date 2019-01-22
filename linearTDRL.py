import numpy as np


def linearTDRL(X, r, W=None, alpha=None, opto=None, optotype='none'):
    """
     Linear TD-RL algorithm
    
     USAGE: results = linearTDRL(X,r,W,alpha,opto,optotype)
    
     INPUTS:
       X - [N x D] stimulus sequence (N = # trials, D = # features)
       r - [N x 1] reward sequence
       W (optional) - initial values of the features (default: all zeros)
       alpha (optional) - learning rate (default: 0.05)
       opto (optional) - sequence of optogenetic perturbations (default: all zeros)
       optotype (optional) - string of 'none' (default) or 'tonic' for
                             prolonged stimulation or 'phasic' for short
                             stimulation
    
     OUTPUTS:
       results - [N x 1] structure with the following fields:
                   .V - value estimate
                   .dt - prediction error
                   .W - weight matrix
    
     Matt Gardner, 2018 based on Sam Gershman's code; Ported to Python3 by Ted Moskovitz, 2019
    """
    N, D = X.shape
    X = np.vstack([X, np.zeros([1, D])])

    if W is None: W = np.zeros([D,1]); 
    if not alpha: alpha = 0.05;
    if opto is None: opto = np.zeros(N); 

    gamma = 0.95
    results = []

    # For these experiments, we're only using currently active elements for
    # opto stimulation such that lambda = 0
    Lambda = 0
    E = 0 

    for n in range(0, N):
        
        if optotype == "none":
            RPE = gamma * X[n+1,:] - X[n,:]
            if len(W.shape) == 1: W = W.reshape(-1,1);
            dt = r[n] + np.dot(RPE.reshape(1,-1), W)
        elif optotype == "tonic":
            dt = (1 + opto[n]) * (r[n] + np.dot(gamma * X[n+1,:] - X[n,:], W))  
        elif optotype == "phasic":
            E = Lambda * E + X[n,:]
            dt = r[n] + np.dot(gamma * X[n+1,:] - X[n,:], W) + opto[n] * E 

        # This corrects for the excitatory/inhibitory assymetry by reducing the inhibitory components
        # Inh to Exc ratio currently set at 1:4
        dt[dt < 0] = dt[dt < 0] / 4 

        # store results 
        result = {}
        result['V'] = np.dot(X[n,:], W)
        result['W'] = W 
        result['dW'] = dt 
        results.append(result)

        # update 
        W = W + alpha * np.dot(X[n,:].reshape(-1,1), dt.reshape(1,-1))

    return results 







    
