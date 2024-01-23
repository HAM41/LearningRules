import numpy as np
import matplotlib.pyplot as plt
from models import QLearningModel, GLMLearn
import scipy as sp
from tqdm import tqdm

import jax
import jax.numpy as jnp

def bootstrap_filter(N: int, X, Y, model, seed=0, return_history=True, R=None, verbose=True):
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
        z_history: List (T,), each element containing N samples, 
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
    key = jax.random.PRNGKey(seed)

    z_history = []
    log_lik = 0.

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
            tilde_z_t = model.update_weights(z_t, x=X[t-1], y=Y[t-1], r=R[t-1])

        # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
        #   Outcome: {tilde z_t^i, tilde w^i}, an approximation to p(z_t|y_{1:t})
            
        tilde_w_t = model.emission_likelihood(y=Y[t], w=tilde_z_t, x=X[t]) 
        
        # 3. Resample with replacement N particles according the importance weights
        #   Outcome: {z_t, 1/N}, an approximation to p(z_t|y_{1:t})
        
        _, latent_dim = tilde_z_t.shape
        if return_history:
            # If return_history, also keep track and resample entire z trajectories 
            if t==0:
                z_history = jax.random.choice(subkey, tilde_z_t, shape=(N,), p=tilde_w_t)
                z_history = z_history.reshape(N, 1, latent_dim)
            else:
                z_history = jnp.concatenate((z_history, tilde_z_t[:, np.newaxis, :]), axis=1)
                z_history = jax.random.choice(subkey, z_history, shape=(N,), p=tilde_w_t)
            z_t = z_history[:,-1,:]
        else:
            z_t = jax.random.choice(subkey, tilde_z_t, shape=(N,), p=tilde_w_t)

        # 4. Update log-likelihood estimate
        # p(y_t | y_{1:t-1}) = \int p(y_t | z_t) p(z_t | y_{1:t-1}) dz_t
        #                    ≈ 1/N \sum_i p(y_t | tilde z_t^i)
        #                    = 1/N \sum_i tilde w_t^i
            
        filtering_estimate = jnp.mean(tilde_w_t)
        log_lik += jnp.log(filtering_estimate)

        # # Check for NaN or Inf
        # if jnp.isnan(normalized_tilde_w_t).any() or jnp.isinf(normalized_tilde_w_t).any():
        #     raise ValueError("Normalized importance weights contain NaN or Inf.")

        if verbose:
            pbar.update(1)

    return z_history, log_lik

def test_bootstrap_filter():
    true_alpha = 0.05
    true_logsigma = -1.0
    # true_model = QLearningModel(sigma=true_sigma, alpha=true_alpha, softmax=True)
    true_model = GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)


    T = 500
    X, Y, Z = true_model.sample(T)

    log_liks, Z_errors = [], []
    particles_range = [2, 10, 100, 1000, 10000, 100000]
    for N_particles in particles_range:
        _, log_lik = bootstrap_filter(N_particles, X=X, Y=Y, model=true_model, return_history=False, verbose=True)

        # Z_error = jnp.linalg.norm(z_history.mean(axis=0) - Z)
        # Z_error = jnp.linalg.norm(jnp.median(z_history, axis=0) - Z)
        # Z_errors.append(Z_error)
        log_liks.append(log_lik)

    fig, axs = plt.subplots(ncols=2, constrained_layout=True)
    axs[0].plot(particles_range, log_liks)
    # axs[1].plot(particles_range, Z_errors)
    for ax in axs:
        ax.set_xscale('log')
    plt.show()
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