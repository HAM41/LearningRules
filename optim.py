import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import ibl
import samplers
import models

def grid_search(X, Y, N_bootstrap=200):
    alpha_range = np.concatenate(([0.0], np.exp(np.linspace(-7,1,10))))
    sigma_range = np.exp(np.linspace(-8,1,10))

    vals = []
    max_value = -float('inf')
    for alpha in alpha_range:
        # _vals = []
        for sigma in sigma_range:
            model = models.GLMLearn(alpha, sigma)
            # model = models.QLearningModel(alpha, sigma, softmax=True)

            log_liks = []
            for batch_X, batch_Y in zip(X,Y):
                _, log_lik = samplers.bootstrap_filter(N_bootstrap, batch_X, batch_Y, model)
                _, log_lik2 = samplers.bootstrap_filter(N_bootstrap, batch_X, batch_Y, model)
                print(log_lik2 - log_lik)
                log_liks.append(log_lik)

            # Return sum of evidence estimates as evidence estimate of the dataset
            out = np.sum(log_liks)
            # _vals.append(out)
            if out > max_value:
                max_value = out
                argmin = (alpha, sigma)

            entry = {'alpha':round(alpha,3), 'sigma':round(sigma,3), 'evidence':out}
            print(entry)
            vals.append(entry)

    vals_df = pd.DataFrame(vals).pivot(index="sigma", columns="alpha", values="evidence")
    # print(vals_df)
    fig, ax = plt.subplots(constrained_layout=True)
    sns.heatmap(data=vals_df, ax=ax, cmap='magma')
    # ax.contourf(*np.meshgrid(alpha_range, sigma_range), vals_df.values)
    # ax.scatter(*argmin)
    # ax.plot(true_alpha, true_sigma, 'o')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()


    print(argmin)
    return 

def simplex(X, Y, N_bootstrap=200, B=1):
    '''
    Simplex method to maximize the evidence.
    Implemented for now for GLMLearn model. 
    Args:
        X: (B,T,)
        Y: (B,T,)
    '''

    # Define objective function
    def neg_evidence(log_theta):
        # Instantiate model for SMC
        print(np.exp(log_theta))
        model = models.GLMLearn(*np.exp(log_theta))
        # model = models.QLearningModel(*np.exp(log_theta))

        # Compute evidence estimates for each batch of (X,Y)
        log_liks = []
        for batch_X, batch_Y in zip(X,Y):

            # Average over batch
            _log_liks = []
            for _ in range(B):
                _, _log_lik = samplers.bootstrap_filter(N_bootstrap, batch_X, batch_Y, model)
                _log_liks.append(_log_lik)
            log_lik = np.mean(_log_liks)

            log_liks.append(log_lik)

        # Return sum of evidence estimates as evidence estimate of the dataset
        out = np.sum(log_liks)
        print(out)
        return -out
    
    # Define intial condition
    log_theta_0 = [-2., -2.] # alpha, sigma_w

    # Perform optimization
    result = sp.optimize.minimize(neg_evidence, log_theta_0, method='Powell')
    assert result.success
    return np.exp(result.x)

if __name__=='__main__':
    # Generate dummy data
    true_alpha = 0.05
    true_sigma = 0.01
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha)

    T = 200
    N_runs = 10

    Xs, Ys = [], []
    for _ in range(N_runs):
        _X, _Y, _ = true_model.simulate(T)
        Xs.append(_X.reshape(-1,1))
        Ys.append(_Y)

    X = np.stack(Xs)
    Y = np.stack(Ys)
    print(X.shape)

    # Evaluate 
    N = 1000
    # X, Y = ibl.get_behavioral_data(subject='ibl_witten_02')
    # grid_search([X[0][:270]], [Y[0][:270]], N_bootstrap=N)
    # print(simplex([X[0][-1000:]], [Y[0][-1000:]], N_bootstrap=N))

    print(simplex(X,Y,N_bootstrap=N,B=5))
    
    # [0.00762551 0.00136533] , -3510.52603770583

    