import numpy as np
import matplotlib.pyplot as plt
import models
from models import QLearningModel, GLMLearn, reinforce
import scipy as sp
from tqdm import tqdm
from typing import Callable

import jax
import jax.numpy as jnp

from functools import partial
import time 

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# @partial(jax.jit, static_argnums=(0,4,)) #! jitting raises OOM errors. Because of N?
def bootstrap_filter(
        N: int, 
        X, Y, 
        model, 
        seed=0, 
        return_history=True, 
        R=None, 
        session_indices=[],
        verbose=True
        ):
    '''
    Bootstrap Filter / Particle filtering algorithm from [1], along with likelihood evaluation,
    specifically tailored for models regression models from X to Y.
    [1]: An Introduction to Sequential Monte Carlo. A. Doucet, N. de Freitas, N. Gordon.
    
    Args:
        N: int. Number of particles
        X: array, (T,M,). Regressors.
        Y: array, (T,). Data/emissions. 
        R: array, (T,). Rewards.
        model: needs to implement some form of forward method, and some for of emission likelihood
    Returns:
        z_history: array, (N, T, M+1).
            Equally weighted latent samples to p(z_{1:T} | y_{1:T}) to evalute integrals with MC methods
        log_lik: scalar,
            Estimate of the marginal log-likelihood
    '''
    if X.ndim == 1:
        T = len(X)
        M = 1
    else:
        T, M = X.shape
    if R is None:
        R = [None for _ in range(T)]

    # day_flags = jnp.zeros(T, dtype=bool)
    # day_flags = day_flags.at[session_indices].set(True)
    day_flags = models.set_day_flags(T, session_indices)

    # if len(session_indices) > 0:
    #     # day_flags = [False for _ in range(T)]
    #     day_flags = jnp.zeros(T, dtype=bool)
    # else:
    #     # if np.max(session_indices) > T:
    #     #     logging.warning("Session indices exceed length of data.")
    #     # day_flags = [True if i in session_indices else False for i in range(T)]
    key = jax.random.PRNGKey(seed)

    filtering_history = []
    log_lik = 0.
    p_t = jnp.ones(N)
    ps = [p_t]
    if return_history:
        # Block out N x T x M+1 array (float32, x 4 in bytes) to store z_t samples.
        # A lot of memory, but much faster. 
        posterior_history = jnp.zeros((N, T, M+1), dtype=jnp.float32) # float16 ?
    else:
        posterior_history = []

    if verbose:
        pbar = tqdm(range(0,T), desc='Bootstrap filter')
    
    for t in range(0,T):
        key, subkey = jax.random.split(key)

        # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
        #   Sample proposal N particles from previous N particles
        #   Outcome: {tilde z_t^i, 1/N}, an approximation to p(z_t|y_{1:t-1})

        if t == 0:
            tilde_z_t = jax.random.normal(subkey, shape=(N, M+1,))
        else:
            # tilde_z_t = models.policy_gradient(z_t, x=X[t-1], y=Y[t-1], r=R[t-1])
            tilde_z_t = model.update_weights(z_t, x=X[t-1], y=Y[t-1], r=R[t-1], day_flag=day_flags[t])
            # tilde_z_t = models.policy_gradient(z_t, x=X[t-1], r=R[t-1])

        # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
        #   Outcome: {tilde z_t^i, tilde w^i}, an approximation to p(z_t|y_{1:t})
            
        # tilde_w_t = model.emission_likelihood(y=Y[t], w=tilde_z_t, x=X[t]) 
        tilde_w_t = models.bernoulli_GLM_likelihood(y=Y[t], w=tilde_z_t, x=X[t]) 
        
        # 3. Resample with replacement N particles according the importance weights
        #   Outcome: {z_t, 1/N}, an approximation to p(z_t|y_{1:t})
        
        _, latent_dim = tilde_z_t.shape
        if return_history:
            # If return_history, also keep track and resample entire z trajectories 
            # #! Slow.
            # if t==0:
            #     posterior_history = jax.random.choice(subkey, tilde_z_t, shape=(N,), p=tilde_w_t)
            #     posterior_history = posterior_history.reshape(N, 1, latent_dim)
            # else:
            #     posterior_history = jnp.concatenate((posterior_history, tilde_z_t[:, np.newaxis, :]), axis=1)
            #     posterior_history = jax.random.choice(subkey, posterior_history, shape=(N,), p=tilde_w_t)
            # z_t = posterior_history[:,-1,:]

            # ? Faster?? Much faster.
            posterior_history = posterior_history.at[:,t,:].set(tilde_z_t)
            posterior_history = jax.random.choice(subkey, posterior_history, shape=(N,), p=tilde_w_t)
            z_t = posterior_history[:,t,:]
        else:
            z_t = jax.random.choice(subkey, tilde_z_t, shape=(N,), p=tilde_w_t)
        z_t = jax.random.choice(subkey, tilde_z_t, shape=(N,), p=tilde_w_t)
        # if return_history:
        #     filtering_history.append(z_t)

        # 4. Update log-likelihood estimate
        # p(y_t | y_{1:t-1}) = \int p(y_t | z_t) p(z_t | y_{1:t-1}) dz_t
        #                    ≈ 1/N \sum_i p(y_t | tilde z_t^i)
        #                    = 1/N \sum_i tilde w_t^i
            
        filtering_estimate = jnp.mean(tilde_w_t)
        log_lik += jnp.log(filtering_estimate)

        # # Check for NaN or Inf
        # if jnp.isnan(log_lik).any() or jnp.isinf(log_lik).any():
        #     raise ValueError("Normalized importance weights contain NaN or Inf.")

        if verbose:
            pbar.update(1)

    # if return_history:
    #     filtering_history = jnp.stack(filtering_history, axis=0)
    #     assert filtering_history.shape == (T, N, M+1)

    #     filtering_history = filtering_history.transpose(1,0,2)

    return (posterior_history, filtering_history), log_lik

def test_bootstrap_filter():
    true_alpha = 0.05
    true_logsigma = -1.0
    # true_model = QLearningModel(sigma=true_sigma, alpha=true_alpha, softmax=True)
    true_model = GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)


    T = 500
    for i in range(10):
        key = jax.random.PRNGKey(i)
        X, Y, Z = true_model.sample(T, key=key)

        log_liks, Z_errors = [], []
        N_particles = 10000

        _, log_lik = bootstrap_filter(N_particles, X=X, Y=Y, model=true_model, return_history=False, verbose=True)
        # _, log_lik = bootstrap_filter(N_particles, X=X, Y=Y, return_history=False, verbose=True)
    # for N_particles in particles_range:
    # _, log_lik = bootstrap_filter_2(N_particles, X=X, Y=Y, update_weights=true_model.update_weights, emission_likelihood=true_model.emission_likelihood, return_history=False, verbose=True)

    #     # Z_error = jnp.linalg.norm(z_history.mean(axis=0) - Z)
    #     # Z_error = jnp.linalg.norm(jnp.median(z_history, axis=0) - Z)
    #     # Z_errors.append(Z_error)
    #     log_liks.append(log_lik)

    # fig, axs = plt.subplots(ncols=2, constrained_layout=True)
    # axs[0].plot(particles_range, log_liks)
    # # axs[1].plot(particles_range, Z_errors)
    # for ax in axs:
    #     ax.set_xscale('log')
    # plt.show()
    # print(z_history.shape, log_lik)
    # # print(z_history, log_lik)

    # print(jnp.linalg.norm(z_history.mean(axis=0) - Z))

    # alpha_range = np.exp(np.linspace(-5,0,20))
    # sigma_range = np.exp(np.linspace(-3,1,10))

    # vals = []
    # for _ in range(N_runs):
    #     _vals = []
    #     for alpha in alpha_range:
    #         # model = QLearningModel(sigma=true_sigma, alpha=alpha, softmax=True)
    #         model = GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)

    #         _, log_liks = bootstrap_filter(N, inputs=X, data=Y, model=model)
    #         _vals.append(log_liks)
    #         print(round(alpha,3), log_liks)
    #     vals.append(_vals)

    # vals = np.array(vals)

    # fig, ax = plt.subplots()
    # ax.plot(alpha_range, np.sum(vals, axis=0), c='tab:grey')
    # ax.axvline(x=true_alpha, c='k', label=r'True $\alpha$')
    # # ax.axvline(x=true_sigma, c='k', label=r'True $\sigma$')
    # # ax.plot(alpha_range, vals.T, c='tab:grey')
    # # ax.fill_between(alpha_range, mean-std, mean+std, alpha=0.3, color='tab:blue')
    # ax.legend()
    # ax.set_xlabel(r'Learning rate $\alpha$')
    # # ax.set_xlabel(r'Percept noise scale $\sigma$')
    # ax.set_ylabel('Log-likelihood')
    # # ax.set_title(f'SMC marginal likelihood estimates. $\sigma=${true_sigma:1.2f}')
    # ax.set_title(r'SMC marginal likelihood estimates. $\alpha=$'+f'{true_alpha:1.2f}')
    # ax.set_xscale('log')
    # plt.savefig('test.png')

    # # -------------

    # true_alpha = 0.1
    # true_sigma = 0.1

    # true_model = PolicyGradientGLM(sigma_w=true_sigma, alpha=true_alpha)
    # X, Y, _ = true_model.simulate(T)
    # vals = []
    # for alpha in alpha_range:
    #     model = PolicyGradientGLM(sigma_w=true_sigma, alpha=alpha)
    #     _, log_lik = bootstrap_filter(N, inputs=X, data=Y, model=model)
    #     vals.append(log_lik)

    # fig, ax = plt.subplots()
    # ax.plot(alpha_range, vals)
    # ax.axvline(x=true_alpha, c='k', label=r'True $\alpha$')
    # ax.set_xscale('log')
    # plt.savefig('test2.png')
    # plt.close()

if __name__=='__main__':
    test_bootstrap_filter()