import numpy as np

def randTrials(n, *args):
    """
    This function randomizes presentations of stimuli for the linearTDSR function

    INPUTS:
        n is the number of presentations for each stimulus sequence

        the varargin are the unique stimulus sequences to be included 
        each stimulus is an [s X D] matrix. (s = # of states within a single
        stimulus sequence, D = # of features 

    OUTPUT: [n*s*(nargin - 1) X D] of randomized stimulus presentations
    
    Note that zeros(1 X D) are added between each presentation
    """
    varargin = args
    nargin = len(varargin)
    #s, D = varargin[0].shape
    out = []
    ncue = nargin 
    curr = 0

    for i in range(n):

        A = np.random.permutation(ncue)

        for j in range(ncue):

            st = varargin[A[j]].shape[0]

            for k in range(st):

                curr = curr + 1
                #out[curr, :] = varargin[A[j]][k,:]
                out.append(varargin[A[j]][k,:])

            curr = curr + 1
            #out[curr, :] = np.zeros(varargin[A[j]].shape[1])
            out.append(np.zeros(varargin[A[j]].shape[1]))

    #r = len(out)
    #out_arr = np.zeros([r, len(out[0])])
    #for i in range(r): out_arr[i,:] = out[i]; 
    return np.vstack(out)