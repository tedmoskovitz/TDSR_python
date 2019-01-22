import numpy as np
from linearTDSR import linearTDSR
from randTrials import randTrials
import matplotlib.pyplot as plt 
import seaborn; seaborn.set() 
import sys 

def sim_TDSR(sim):
    """
    Simulate experimental paradigms.
    
    USAGE: results = sim_TDRL(sim)
    
    INPUTS:
       sim - string specifying simulation:
               'sharpe17_opto' - activation experiment from Sharpe et al (2017)
               'sharpe17_deval' - devaluation experiment from Sharpe et al (2017)
               'sharpe17_inhib' - inhibition experiment from Sharpe et al (2017)
               'takahashi17_identity' - identity change experiment from Takahashi et al (2017)
               'chang17_identity' - identity unblocking experiment from Chang et al (2017)
    
    OUTPUTS:
       results - see linearTDRL.py 
    
    Sam Gershman, Dec 2017, edited by Matt Gardner April 2018; Ported to Python3 by Ted Moskovitz 2019
    """
    if sim == "sharpe17_opto":
    
        # A = 1
        # C = 2
        # D = 3
        # E = 4
        # F = 5
        # X = 6
        # food = 7

        A_X = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0]]) # 24
        EF_X = np.array([[0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0]]) # 8
        AD_X = np.array([[1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0]]) # 8
        AC_X = np.array([[1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0]]) # 8
        X = np.array([[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1]]) # 4*24
        F = np.array([0, 0, 0, 0, 1, 0, 0]).reshape(1,-1)
        D = np.array([0, 0, 1, 0, 0, 0, 0]).reshape(1,-1)
        C = np.array([0, 1, 0, 0, 0, 0, 0]).reshape(1,-1)

        # this sets the alphas for the SR and U learning
        alpha = np.array([0.06, 0.03])

        # this function provides the logicals of the first state of stimulus
        def st1(x, Y):
            test_arr = Y == x[0,:]
            return np.all(test_arr, 1)

        # this function provides the second state of a simulation
        def st2(x):
            tmp = [False] + list(x[:-1])
            return np.asarray(tmp)

        # stimuli are randomized within each execution of the TDSR model.
        results = {}
        f_m, d_m, c_m = {}, {}, {}
        f_m[1] = []; f_m[2] = []; d_m[1] = []; d_m[2] = []; c_m[1] = []; c_m[2] = [];
        for i in range(100):

            # this randomizes trial order for each stage
            s1 = randTrials(24, A_X)
            s2 = randTrials(8, EF_X, AD_X, AC_X)
            s3 = randTrials(4*24, X)
            s4 = randTrials(6, F, D, C)

            # matrix with all stages included
            M = np.vstack([s1, s2, s3, s4])

            # this keeps track of trials from each stage
            sM = np.vstack([np.ones([s1.shape[0], 1]), 2 * np.ones([s2.shape[0], 1]), \
                  3 * np.ones([s3.shape[0], 1]), 4 * np.ones([s4.shape[0], 1])])
            sM = sM.reshape(-1,)

            # logicals of the first state of each stimulus. the logicals
            # are conditioned on the stage in which the cue occurred. 
            a_x = np.logical_and(st1(A_X, M), sM == 1)
            ef_x = np.logical_and(st1(EF_X, M), sM == 2)
            ad_x = np.logical_and(st1(AD_X, M), sM == 2)
            ac_x = np.logical_and(st1(AC_X, M), sM == 2)
            x = np.logical_and(st1(X, M), sM == 3)
            f = st1(F, M).astype(int)
            d = st1(D, M).astype(int)
            c = st1(C, M).astype(int)

            # rewarded states
            r = st2(x).astype(float)
            opto = 1 * np.logical_or(st2(ac_x), st2(st2(ad_x)))

            # this sets the initial value of the US
            ui = np.array([0, 0, 0, 0, 0, 0, 0.5]).reshape(-1,1)

            # this runs the model for the current iteration and saves 
            # the results 
            results[1] = np.array(linearTDSR(M, r, u=ui, alpha=alpha, opto=opto, optotype='phasic'))
            f_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(f == 1)[0]]]))
            d_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(d == 1)[0]]]))
            c_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(c == 1)[0]]]))

            results[2] = np.array(linearTDSR(M, r, u=ui, alpha=alpha))
            f_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(f == 1)[0]]]))
            d_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(d == 1)[0]]]))
            c_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(c == 1)[0]]]))

        # convert to arrays
        f_m[1] = np.vstack(f_m[1]); f_m[2] = np.vstack(f_m[2]);
        d_m[1] = np.vstack(d_m[1]); d_m[2] = np.vstack(d_m[2]); 
        c_m[1] = np.vstack(c_m[1]); c_m[2] = np.vstack(c_m[2]);  

        V = np.zeros([2,3])
        V[0,:] = np.mean(np.hstack([f_m[1][:,0].reshape(-1,1), \
            d_m[1][:,0].reshape(-1,1), c_m[1][:,0].reshape(-1,1)]), axis=0)
        V[1,:] =  np.mean(np.hstack([f_m[2][:,0].reshape(-1,1), \
            d_m[2][:,0].reshape(-1,1), c_m[2][:,0].reshape(-1,1)]), axis=0)

        # plot results
        colors = ['b', 'r', 'y']
        barwidth = 0.25
        fig, axes = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.6)
        plt.setp(axes, xticks=[r + barwidth for r in [0.25, 1.5]], \
        	xticklabels=["ChR2", "eYFP"]) #yticks=[1, 2, 3]

        bars = axes[1].bar((barwidth + 0.05) * np.arange(3) + 0.25, V[0,:], width=barwidth)
        for i,b in enumerate(bars):
            b.set_color(colors[i])

        bars2 = axes[1].bar((barwidth + 0.05) * np.arange(3) + 1.5, V[1,:], width=barwidth)
        for i,b in enumerate(bars2):
            b.set_color(colors[i])

        axes[1].set_title("Model", fontsize=22, fontweight="bold")
        axes[1].set_ylabel("V", fontsize=22)

        # data
        V = np.array([[2.05, 1.66, 2.66], [2.15, 1.47, 1.52]])
        bars = axes[0].bar((barwidth + 0.05) * np.arange(3) + 0.25, V[0,:], width=barwidth)
        for i,b in enumerate(bars):
            b.set_color(colors[i])

        bars2 = axes[0].bar((barwidth + 0.05) * np.arange(3) + 1.5, V[1,:], width=barwidth)
        for i,b in enumerate(bars2):
            b.set_color(colors[i])

        axes[0].set_title("Data", fontsize=22, fontweight="bold")
        axes[0].set_ylabel("Magazine Entries", fontsize=22)
        axes[0].legend([bars[0], bars[1], bars[2]], ["F", "D", "C"], fontsize=9)

        plt.show() 
    
    elif sim == "sharpe17_deval":
        
        # A = 1
        # C = 2
        # D = 3
        # E = 4
        # F = 5
        # X = 6
        # food = 7

        A_X = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0]]) # 24
        EF_X = np.array([[0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0]]) # 8
        AD_X = np.array([[1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0]]) # 8
        AC_X = np.array([[1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0]]) # 8
        X = np.array([[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1]]) # 4*24
        C = np.array([0, 1, 0, 0, 0, 0, 0]).reshape(1,-1)
        FOOD = np.array([0, 0, 0, 0, 0, 0, 1]).reshape(1,-1)

        # this sets the alphas for the SR and U learning
        alpha = np.array([0.06, 0.03])

        # this function provides the logicals of the first state of stimulus
        def st1(x, Y):
            test_arr = Y == x[0,:]
            return np.all(test_arr, 1)

        # this function provides the second state of a simulation
        def st2(x):
            tmp = [False] + list(x[:-1])
            return np.asarray(tmp)

        # stimuli are randomized within each execution of the TDSR model.
        results = {}
        c_m = {}
        c_m[1] = []; c_m[2] = [];
        for i in range(100):

            # this randomizes trial order for each stage
            s1 = randTrials(24, A_X)
            s2 = randTrials(8, EF_X, AD_X, AC_X)
            s3 = randTrials(4 * 24, X)
            s4 = np.tile(FOOD, [1,1]) 
            s5 = np.tile(C, [6,1])

            # matrix with all stages included
            M = np.vstack([s1, s2, s3, s4, s5])

            # this keeps track of trials from each stage 
            sM = np.vstack([np.ones([s1.shape[0], 1]), 2 * np.ones([s2.shape[0], 1]), \
                  3 * np.ones([s3.shape[0], 1]), 4 * np.ones([s4.shape[0], 1]), \
                  5 * np.ones([s5.shape[0], 1])])
            sM = sM.reshape(-1,)

            # logicals of the first state of each stimulus. the logicals
            # are conditioned on the stage in which the cue occurred. (X
            # occurs in stage 1 and 2 as part of the other cues)
            a_x = np.logical_and(st1(A_X, M), sM == 1)
            ef_x = np.logical_and(st1(EF_X, M), sM == 2)
            ad_x = np.logical_and(st1(AD_X, M), sM == 2)
            ac_x = np.logical_and(st1(AC_X, M), sM == 2)
            x = np.logical_and(st1(X, M), sM == 3)
            food = np.logical_and(st1(FOOD, M), sM == 4)
            c = np.logical_and(st1(C, M), sM == 5) 

            # rewarded states
            r1 = st2(x).astype(float) + food.astype(float)

            # the -10 devalues the outcome 
            r2 = st2(x).astype(float) - 10 * food.astype(float)

            opto = 1 * np.logical_or(st2(ac_x), st2(st2(ad_x)))

            # this sets the initial value of the US to 1
            ui = np.array([0, 0, 0, 0, 0, 0, 0.5]).reshape(-1,1)

            # this runs the model for the current iteration and saves 
            # the results 
            results[1] = np.array(linearTDSR(M, r1, u=ui, alpha=alpha, opto=opto, optotype='phasic'))
            c_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(c == 1)[0]]]))

            results[2] = np.array(linearTDSR(M, r2, u=ui, alpha=alpha, opto=opto, optotype='phasic'))
            c_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(c == 1)[0]]]))

        # convert to array
        c_m[1] = np.vstack(c_m[1]); c_m[2] = np.vstack(c_m[2]);  

        V = np.array([np.mean(c_m[1][:,0]), np.mean(c_m[2][:,0])])

        # plot results
        barwidth = 0.25
        fig, axes = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.6)
        centers = (barwidth + 0.05) * np.arange(2) + 0.25
        plt.setp(axes, xticks=centers, \
            xticklabels=["Non-devalued", "Devalued"]) #yticks=[1, 2, 3]

        bars = axes[1].bar((barwidth + 0.05) * np.arange(2) + 0.25, V, width=barwidth)

        axes[1].set_title("Model", fontsize=20, fontweight="bold")
        axes[1].set_ylabel("V", fontsize=20)

        # data
        V = np.array([3.22, 1.23])
        bars = axes[0].bar((barwidth + 0.05) * np.arange(2) + 0.25, V, width=barwidth)

        axes[0].set_title("Data", fontsize=20, fontweight="bold")
        axes[0].set_ylabel("CR", fontsize=20)

        plt.show()

    elif sim == "sharpe17_deval_rehearsal":

        # A = 1
        # C = 2
        # D = 3
        # E = 4
        # F = 5
        # X = 6
        # food = 7

        A_X = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0]]) # 24
        EF_X = np.array([[0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0]]) # 8
        AD_X = np.array([[1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0]]) # 8
        AC_X = np.array([[1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0]]) # 8
        X = np.array([[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1]]) # 4*24
        C = np.array([0, 1, 0, 0, 0, 0, 0]).reshape(1,-1)
        FOOD = np.array([0, 0, 0, 0, 0, 1, 1]).reshape(1,-1)

        # this sets the alphas for the SR and U learning
        alpha = np.array([0.06, 0.03])

        # this function provides the logicals of the first state of stimulus
        def st1(x, Y):
            test_arr = Y == x[0,:]
            return np.all(test_arr, 1)

        # this function provides the second state of a simulation
        def st2(x):
            tmp = [False] + list(x[:-1])
            return np.asarray(tmp)

        # stimuli are randomized within each execution of the TDSR model.
        results = {}
        c_m = {}
        c_m[1] = []; c_m[2] = [];
        for i in range(100):

            # this randomizes trial order for each stage. note that randTrials
            # adds a zeros vector between trials (models the ITI)
            s1 = randTrials(24, A_X)
            s2 = randTrials(8, EF_X, AD_X, AC_X)
            s3 = randTrials(4 * 24, X)
            s4 = np.tile(FOOD, [1,1]) 
            s5 = np.tile(C, [6,1])

            # matrix with all stages included
            M = np.vstack([s1, s2, s3, s4, s5])

            # this keeps track of trials from each stage 
            sM = np.vstack([np.ones([s1.shape[0], 1]), 2 * np.ones([s2.shape[0], 1]), \
                  3 * np.ones([s3.shape[0], 1]), 4 * np.ones([s4.shape[0], 1]), \
                  5 * np.ones([s5.shape[0], 1])])
            sM = sM.reshape(-1,)

            # logicals of the first state of each stimulus. the logicals
            # are conditioned on the stage in which the cue occurred. (X
            # occurs in stage 1 and 2 as part of the cues)
            a_x = np.logical_and(st1(A_X, M), sM == 1)
            ef_x = np.logical_and(st1(EF_X, M), sM == 2)
            ad_x = np.logical_and(st1(AD_X, M), sM == 2)
            ac_x = np.logical_and(st1(AC_X, M), sM == 2)
            x = np.logical_and(st1(X, M), sM == 3)
            food = np.logical_and(st1(FOOD, M), sM == 4)
            c = np.logical_and(st1(C, M), sM == 5) 

            # rewarded states
            r1 = st2(x).astype(float) + food.astype(float)

            # the -10 devalues the outcome 
            r2 = st2(x).astype(float) - 10 * food.astype(float)

            opto = 1 * np.logical_or(st2(ac_x), st2(st2(ad_x)))

            # this sets the initial value of the US to 1
            ui = np.array([0, 0, 0, 0, 0, 0, 0.5]).reshape(-1,1)

            # this runs the model for the current iteration and saves 
            # the results 
            results[1] = np.array(linearTDSR(M, r1, u=ui, alpha=alpha, opto=opto, optotype='phasic'))
            c_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(c == 1)[0]]]))

            results[2] = np.array(linearTDSR(M, r2, u=ui, alpha=alpha, opto=opto, optotype='phasic'))
            c_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(c == 1)[0]]]))

        # convert to array
        c_m[1] = np.vstack(c_m[1]); c_m[2] = np.vstack(c_m[2]);  

        V = np.array([np.mean(c_m[1][:,0]), np.mean(c_m[2][:,0])])

        # plot results
        barwidth = 0.25
        fig, axes = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.6)
        centers = (barwidth + 0.05) * np.arange(2) + 0.25
        plt.setp(axes, xticks=centers, \
            xticklabels=["Non-devalued", "Devalued"]) #yticks=[1, 2, 3]

        bars = axes[1].bar((barwidth + 0.05) * np.arange(2) + 0.25, V, width=barwidth)

        axes[1].set_title("Model", fontsize=20, fontweight="bold")
        axes[1].set_ylabel("V", fontsize=20)

        # data
        V = np.array([3.22, 1.23])
        bars = axes[0].bar((barwidth + 0.05) * np.arange(2) + 0.25, V, width=barwidth)

        axes[0].set_title("Data", fontsize=20, fontweight="bold")
        axes[0].set_ylabel("CR", fontsize=20)

        plt.show()

    elif sim == "sharpe17_inhib":

        # A = 1
        # B = 2
        # X = 3
        # Y = 4
        # flavor1 = 5
        # flavor2 = 6

        A_X = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]]) # 12
        B_Y = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0]]) # 12
        X = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]]) # 4*12
        Y = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1]]) # 4*12
        A = np.array([1, 0, 0, 0, 0, 0]).reshape(1,-1)
        B = np.array([0, 1, 0, 0, 0, 0]).reshape(1,-1)

        # this sets the alphas for the SR and U learning
        alpha = np.array([0.06, 0.03])

        # this function provides the logicals of the first state of stimulus
        def st1(x, Y):
            test_arr = Y == x[0,:]
            return np.all(test_arr, 1)

        # this function provides the second state of a simulus
        def st2(x):
            tmp = [False] + list(x[:-1])
            return np.asarray(tmp)

        # this gets the prior state
        def stpre(x):
            return np.asarray(list(x[1:]) + [False])

        # stimuli are randomized within each execution of the TDSR model.
        results = {}
        a_m = {}; b_m = {}; 
        a_m[1] = []; a_m[2] = [];
        b_m[1] = []; b_m[2] = [];
        for i in range(100):

            # this randomizes trial order for each stage
            s1 = randTrials(12, A_X, B_Y)
            s2 = randTrials(24 * 4, X, Y)
            s3 = randTrials(6, A, B)

            # matrix with all stages included
            M = np.vstack([s1, s2, s3])

            # this keeps track of trials from each stage 
            sM = np.vstack([np.ones([s1.shape[0], 1]), 2 * np.ones([s2.shape[0], 1]), \
                  3 * np.ones([s3.shape[0], 1])])
            sM = sM.reshape(-1,)

            # logicals of each stimulus
            b_y = np.logical_and(st1(B_Y, M), sM == 1)
            a = np.logical_and(st1(A, M), sM == 3)
            b = np.logical_and(st1(B, M), sM == 3)
            x = np.logical_and(st1(X, M), sM == 2)
            y = np.logical_and(st1(Y, M), sM == 2)

            # rewarded states
            r = st2(np.logical_or(x, y)).astype(float)

            opto = -0.8 * np.logical_or(np.logical_or(stpre(b_y), b_y), st2(b_y))

            # this sets the initial value of the US to 1
            ui = np.array([0, 0, 0, 0, 0.5, 0.5]).reshape(-1,1)

            # this runs the model for the current iteration and saves 
            # the results 
            results[1] = np.array(linearTDSR(M, r, u=ui, alpha=alpha, opto=opto, optotype='tonic'))
            a_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(a == 1)[0]]]))
            b_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(b == 1)[0]]]))

            results[2] = np.array(linearTDSR(M, r, u=ui, alpha=alpha))
            a_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(a == 1)[0]]]))
            b_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(b == 1)[0]]]))

        # convert to arrays
        a_m[1] = np.vstack(a_m[1]); a_m[2] = np.vstack(a_m[2]);
        b_m[1] = np.vstack(b_m[1]); b_m[2] = np.vstack(b_m[2]); 

        V = np.zeros([2,2])
        V[0,:] = np.mean(np.hstack([a_m[1][:,0].reshape(-1,1), b_m[1][:,0].reshape(-1,1)]), axis=0)
        V[1,:] = np.mean(np.hstack([a_m[2][:,0].reshape(-1,1), b_m[2][:,0].reshape(-1,1)]), axis=0)

        print (V)

        # plot results
        colors = ['b', 'r']
        barwidth = 0.25
        fig, axes = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.6)
        plt.setp(axes, xticks=[r + barwidth for r in [0.25, 1.5]], \
        	xticklabels=["NpHR", "eYFP"]) #yticks=[1, 2, 3]

        bars = axes[1].bar((barwidth + 0.05) * np.arange(2) + 0.25, V[0,:], width=barwidth)
        for i,b in enumerate(bars):
            b.set_color(colors[i])

        bars2 = axes[1].bar((barwidth + 0.05) * np.arange(2) + 1.5, V[1,:], width=barwidth)
        for i,b in enumerate(bars2):
            b.set_color(colors[i])

        axes[1].set_title("Model", fontsize=22, fontweight="bold")
        axes[1].set_ylabel("V", fontsize=22)

        # data
        V = np.array([[51.4, 29.3], [38.5, 36.9]])
        bars = axes[0].bar((barwidth + 0.05) * np.arange(2) + 0.25, V[0,:], width=barwidth)
        for i,b in enumerate(bars):
            b.set_color(colors[i])

        bars2 = axes[0].bar((barwidth + 0.05) * np.arange(2) + 1.5, V[1,:], width=barwidth)
        for i,b in enumerate(bars2):
            b.set_color(colors[i])

        axes[0].set_title("Data", fontsize=22, fontweight="bold")
        axes[0].set_ylabel("CR", fontsize=22)
        axes[0].legend([bars[0], bars[1]], ["A", "B"], fontsize=9)

        plt.show() 

    elif sim == "takahashi17_identity":

        # A (odor 1) = 1
        # B (odor 2) = 2
        # flavor 1   = 3
        # flavor 2   = 4
        # i: identity block shift

        n = 30
        A = np.array([[1, 0, 0, 0], [1, 0, 1, 0]]) 
        B = np.array([[0, 1, 0, 0], [0, 1, 0, 1]]) 
        Ai = np.array([[1, 0, 0, 0], [1, 0, 0, 1]])
        Bi = np.array([[0, 1, 0, 0], [0, 1, 1, 0]])

        # this sets the alphas for SR and U learning
        alpha = np.array([0.06, 0.03])

        # this function provides the logicals of the first 
        def st1(x, Y):
            test_arr = np.diff(np.vstack([np.zeros([1,4]), Y]), axis=0) == x[0,:] ######
            return np.all(test_arr, 1)

        # this runs identity then the upshift / downshift
        rev_reps = 4
        M = np.tile(np.vstack([randTrials(n, A, B), randTrials(n, Ai, Bi)]), [rev_reps, 1])

        r = np.tile(np.array([0, 1, 0]).reshape(-1,1), [n * 2 * 2 * rev_reps, 1])

        ai = st1(A, M)
        bi = st1(B, M)

        ui = np.array([0, 0, 0.5, 0.5]).reshape(-1,1)

        # first get the identity errors
        results = linearTDSR(M, r, u=ui, alpha=alpha)

        # this gets the last two blocks
        ni = -1
        dt = []
        for i in range(len(results) - n*2*rev_reps*3, len(results)):
            ni += 1
            dt.append(results[i]['dt'])
        dt = np.vstack(dt)
        
        # use A to assess the identity error over the final two blocks
        A_dt = dt[ai[len(results) - n*2*rev_reps*3 : len(results)], :]

        # use 5 trials
        k = 5

        D = [np.sum(np.mean(A_dt[:k,:], axis=0)), \
            np.sum(np.mean(A_dt[n - k : n, :], axis=0))]
        D = np.array(D).reshape(-1,1)
        
        # this is the actual data 
        T_Data = np.array([6.82, 2.34]) # identity 

        # plot results
        barwidth = 0.25
        fig, axes = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.6)
        centers = (barwidth + 0.05) * np.arange(2) + 0.25
        plt.setp(axes, xticks=centers, \
            xticklabels=["Early", "Late"]) #yticks=[1, 2, 3]

        bars = axes[0].bar((barwidth + 0.05) * np.arange(2) + 0.25, D, width=barwidth)

        axes[0].set_title("Model", fontsize=20, fontweight="bold")
        axes[0].set_ylabel("Prediction error", fontsize=20)
        axes[0].set_ylim([-0.5, 1.0])

        # data
        bars = axes[1].bar((barwidth + 0.05) * np.arange(2) + 0.25, T_Data, width=barwidth)

        axes[1].set_title("Data", fontsize=20, fontweight="bold")
        axes[1].set_ylabel("Spikes/sec", fontsize=20)
        axes[1].set_ylim([-5,15])

        plt.show()

    elif sim == "chang17_identity":
        
        # randTrials adds the ITI between trials
        A = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0]]) 
        B = np.array([[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1]]) 
        AX = np.array([[1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]])
        BY = np.array([[0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0]])
        X = np.array([0, 0, 1, 0, 0, 0]).reshape(1,-1)
        Y = np.array([0, 0, 0, 1, 0, 0]).reshape(1,-1)

        # this sets the alphas for the SR and U learning
        alpha = np.array([0.06, 0.03])

        # this function provides the logicals of the first state of a stimulus
        def st1(x, Y):
            test_arr = Y == x[0,:]
            return np.all(test_arr, 1)

        # this function provides the second state of a simulus
        def st2(x):
            tmp = [False] + list(x[:-1])
            return np.asarray(tmp)

        # this gets the prior state
        def stpre(x):
            return np.asarray(list(x[1:]) + [False])

        # stimuli are randomized within each executation of the TDRL model
        results = {}
        x_m = {}; y_m = {}; 
        x_m[1] = []; x_m[2] = [];
        y_m[1] = []; y_m[2] = [];
        for i in range(10):

            # this randomizes trial order for each stage
            s1 = randTrials(48, A, B)
            s2 = randTrials(16, AX, BY)
            s3 = randTrials(6, X, Y)

            # matrix with all stages included
            M = np.vstack([s1, s2, s3])

            # this keeps track of trials from each stage 
            sM = np.vstack([np.ones([s1.shape[0], 1]), 2 * np.ones([s2.shape[0], 1]), \
                  3 * np.ones([s3.shape[0], 1])])
            sM = sM.reshape(-1,)

            # logicals of each stimulus
            a = np.logical_and(st1(A, M), sM == 1)
            b = np.logical_and(st1(B, M), sM == 1)
            ax = np.logical_and(st1(AX, M), sM == 2)
            by = np.logical_and(st1(BY, M), sM == 2)
            x = np.logical_and(st1(X, M), sM == 3)
            y = np.logical_and(st1(Y, M), sM == 3)

            # rewarded states
            r = np.logical_or(np.logical_or(np.logical_or(st2(a), st2(b)), st2(ax)), \
                st2(by)).astype(float)

            # this implements tonic inhibitiom which occurs through the
            # full trial: ITI, CS, CS + US
            opto = {}
            opto[1] = -0.8 * np.logical_or(np.logical_or(stpre(by), by), st2(by))
            opto[2] = -0.8 * np.array([list(by[1:]) + [0]]).reshape(-1,1)

            # this sets the initial value of the US to 1
            ui = np.array([0, 0, 0, 0, 0.5, 0.5]).reshape(-1,1)

            results[1] = np.array(linearTDSR(M, r, u=ui, alpha=alpha, opto=opto[1], optotype="tonic"))
            x_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(x == 1)[0]]]))
            y_m[1].append(np.hstack([res['V'].reshape(1,-1) for res in results[1][np.where(y == 1)[0]]]))
  
            results[2] = np.array(linearTDSR(M, r, u=ui, alpha=alpha, opto=opto[2], optotype="tonic"))
            x_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(x == 1)[0]]]))
            y_m[2].append(np.hstack([res['V'].reshape(1,-1) for res in results[2][np.where(y == 1)[0]]]))

        
        # convert to arrays
        x_m[1] = np.vstack(x_m[1]); x_m[2] = np.vstack(x_m[2]);
        y_m[1] = np.vstack(y_m[1]); y_m[2] = np.vstack(y_m[2]); 

        V = np.zeros([2,2])
        V[0,:] = np.mean(np.hstack([x_m[1][:,0].reshape(-1,1), y_m[1][:,0].reshape(-1,1)]), axis=0)
        V[1,:] = np.mean(np.hstack([x_m[2][:,0].reshape(-1,1), y_m[2][:,0].reshape(-1,1)]), axis=0)

        # plot results
        colors = ['b', 'r']
        barwidth = 0.25
        fig, axes = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.6)
        plt.setp(axes, xticks=[r + barwidth for r in [0.25, 1.5]], \
        	xticklabels=["Exp", "ITI"]) #yticks=[1, 2, 3]

        bars = axes[1].bar((barwidth + 0.05) * np.arange(2) + 0.25, V[0,:], width=barwidth)
        for i,b in enumerate(bars):
            b.set_color(colors[i])

        bars2 = axes[1].bar((barwidth + 0.05) * np.arange(2) + 1.5, V[1,:], width=barwidth)
        for i,b in enumerate(bars2):
            b.set_color(colors[i])

        axes[1].set_title("Model", fontsize=22, fontweight="bold")
        axes[1].set_ylabel("V", fontsize=22)
        axes[1].set_ylim([0, 0.25])

        # data
        V = np.array([[27.6, 17.1], [26.7, 47.4]])
        bars = axes[0].bar((barwidth + 0.05) * np.arange(2) + 0.25, V[0,:], width=barwidth)
        for i,b in enumerate(bars):
            b.set_color(colors[i])

        bars2 = axes[0].bar((barwidth + 0.05) * np.arange(2) + 1.5, V[1,:], width=barwidth)
        for i,b in enumerate(bars2):
            b.set_color(colors[i])

        axes[0].set_title("Data", fontsize=22, fontweight="bold")
        axes[0].set_ylabel("CR", fontsize=22)
        axes[0].legend([bars[0], bars[1]], ["A_B", "A_{UB}"], fontsize=9)
        axes[0].set_ylim([0, 50])

        plt.show() 
        
    else:

        print ("Invalid experiment. The options are:")
        print ("sharpe17_opto \nsharpe17_deval \nsharpe17_inhib \ntakahashi17_identity \nchang17_identity")

if __name__=="__main__":
    experiment = sys.argv[1]
    sim_TDSR(experiment)




