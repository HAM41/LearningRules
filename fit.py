
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import pickle

# import ibl
import sys
sys.path.append('../')
import samplers
import models, optim

import sys
import jax 
import jax.numpy as jnp
import jax.scipy as jsp
# import os
# os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from parameters import ParamsGLMLearn, ParameterProperties

def callback_f(intermediate_result):
    val = intermediate_result.fun
    params = intermediate_result.x
    logging.info(f'Likelihood: {-val:5.2f}, params: {ParamsGLMLearn(*params)}')
    # print(intermediate_result.nit, intermediate_result.x, intermediate_result.fun)

def fit(X, Y):
    def neg_MLL(params_array):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        # Instantiate model
        model = models.GLMLearn(seed=seed)
        model.update_params_from_array(params_array)

        # Compute marginal log-likelihood with SMC
        _, lik = samplers.bootstrap_filter(N_particles, X, Y, model, return_history=False, verbose=False)
        return -lik

    from scipy.optimize import minimize
    res = minimize(
        neg_MLL,
        x0 = jnp.array([-3.0, 0.0]),
        method='Nelder-Mead',
        tol=1e-3,
        options={'disp':True, 'return_all':True, 'initial_simplex':jnp.array([[-4.0, 0.0], [-2.0, 0.5], [0.0, 0.0]])},
        callback=callback_f,
        bounds=[(-6,2), (0,1)]
        )
    return res

def fit_single_trajectory_search():
    T = 10000
    alpha_values = [0.0, 0.01, 0.1, 0.5]
    sigma_values = [-5.0, -3.0, -1.0]

    convergence_dicts = []
    for true_alpha in alpha_values:
        for true_logsigma in sigma_values:
            conv_dict = fit_single_trajectory(true_alpha, true_logsigma, T=T)
            print('-'*50)
            print(conv_dict)
            print('-'*50)

            convergence_dicts.append(conv_dict)

    with open(f'saves/convergence_dicts_w_initial_simplex_T{T}.pkl', 'wb') as f:
        pickle.dump(convergence_dicts, f)

def fit_single_trajectory(true_alpha, true_logsigma, T=1000):
    true_model = models.GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)
    logging.info(f'True model: {true_model.params}')
            
    X, Y, _ = true_model.sample(T)

    res = fit(X,Y)
    conv_dict = {'T':T, 'true_alpha': true_alpha, 'true_log_sigma': true_logsigma, 'res':res}
    return conv_dict

# true_alpha = 0.05
# true_logsigma = -1.0
# true_model = models.GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)

# T = 1000
# X, Y, _ = true_model.sample(T)

# res = fit(X,Y)
if __name__=='__main__':
    N_particles = 10000
    seed = 0
    key = jax.random.PRNGKey(seed)

    print(fit_single_trajectory(0.1, -3.0))