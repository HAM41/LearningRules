import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from typing import Tuple, Optional, Iterable, Union

import os
os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
# sys.path.append('../')
sys.path.append('/home/vg0233/PillowLab/LearningRules/')
from fit_ibl import fit_optax, fit_EM, find_initial
from parameters import ParamsGLMLearn, ParameterProperties
import models

import argparse

if __name__=='__main__':
    # Hyperparams ------------------------------------
    parser = argparse.ArgumentParser(description="Argument parser for IBL fitting.")

    # Add arguments
    parser.add_argument("--T", type=int, default=500, 
                        help="Number of time-steps per trajectory.")
    parser.add_argument("--B", type=int, default=1,
                        help="Number of batches (trajectories / subjects).")
    parser.add_argument("--N-particles", "-N", type=int, default=1000,
                        help="Number of particles for the fitting.")
    parser.add_argument("--learning-rule", type=str, default="policy_gradient", choices=["policy_gradient", "reinforce"], 
                        help="Learning rule for the model.")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Seed for random number generator")
    parser.add_argument("--fix-true", action='store_true',
                        help="Fix the true parameters.")
    d = 1   # number of regressors

    # Parse the command-line arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    # Format params and create dataset ---------------

    key = jax.random.PRNGKey(args.seed)
    logging.info(f'Number of particles: {args.N_particles}. Seed: {args.seed}. T: {args.T}, B: {args.B}. d: {d}. Learning rule: {args.learning_rule}.')

    # Instantiate true model

    if args.fix_true:
        true_key = jax.random.PRNGKey(101)
    else:
        true_key = key
    
    subkey1, subkey2, subkey3 = jax.random.split(true_key, 3)
    true_log_sigma = jax.random.uniform(subkey1, minval=-4.0, maxval=-1.0)
    true_log_sigma_day = jax.random.uniform(subkey2, minval=-2.0, maxval=0.0)
    true_log_alpha = jax.random.uniform(subkey3, minval=-6, maxval=-2, shape=(d+1,))
    true_params = ParamsGLMLearn(log_sigma=true_log_sigma, log_sigma_day=true_log_sigma_day, log_alpha=true_log_alpha)

    logging.info(f"True parameters: {true_params}")
    true_model = models.GLMLearn(seed=args.seed, learning_rule=args.learning_rule) # **true_params._asdict(),

    # Create dataset
    X, Y, Z = [], [], []
    for _ in range(args.B):
        key, subkey = jax.random.split(key)
        X_i, Y_i, Z_i, _ = true_model.sample(T=args.T, params=true_params, key=subkey)
        X.append(X_i)
        Y.append(Y_i)
        Z.append(Z_i)
    session_indices = [jnp.arange(0, args.T, min(200,int(args.T/10))).astype(int) for _ in range(args.B)]


    if args.learning_rule == 'reinforce':
        R = [jnp.asarray(models.reward(X_sub[:,0], Y_sub)) for X_sub, Y_sub in zip(X, Y)]
    elif args.learning_rule == 'policy_gradient':
        R = [jnp.asarray(models.effective_reward(X_sub[:,0])) for X_sub, Y_sub in zip(X, Y)]

    # true_loglik = true_model.marginal_log_likelihood(
    #     key, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
    #     verbose=True
    # )
    # logging.info(f"True log-likelihood: {true_loglik}")

    partition = int(len(X[0]) * 0.6)
    key, subkey = jax.random.split(key)
    prediction_score = true_model.score_predict(
        key, true_params, 
        X_hist=X[0][:partition], Y_hist=Y[0][:partition], R_hist=R[0][:partition], session_indices=session_indices[0], 
        X_pred=X[0][partition:partition+100], Y_pred=Y[0][partition:partition+100],
        N_particles=args.N_particles,
    )
    logging.info(f"Prediction score: {prediction_score:.4f}")

    # Start fitting ----------------------------------
        
    logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")
    # res = fit_optax(X, Y, R=R, session_indices=session_indices, N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})

    gridsearch_params = find_initial(
        X, Y, R=R, session_indices=session_indices,
        N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule}, seed=args.seed,
        vmap=True, metric='prediction_score',
        )
    # gridsearch_params = [0.5, -1.0, -1.0]

    initial_params = ParamsGLMLearn(
        log_sigma=gridsearch_params[1], 
        log_sigma_day=gridsearch_params[2], 
        log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1)
        )#._asdict()
    logging.info(f'Initial params: {initial_params}')

    # res = fit_optax(
    #     X[0], Y[0], R=R[0], session_indices=session_indices[0],
    #     N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': 0.},
    #     initial_params=initial_params
    #     )
    params, results_dict = fit_EM(
        X[0], Y[0], R=R[0], session_indices=session_indices[0],
        N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': 0.},
        initial_params=initial_params, seed=args.seed, n_iters=10, m_step_iters=500,
        posterior_type='smooth',
    )
    # logging.info(f"Results: {results_dict}")
        
        # res = fit_EM(X[0], Y[0], R=R[0], session_indices=session_indices[0], N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})

    # # # Plot recovery

    # import matplotlib.pyplot as plt

    # print(jnp.unique(X[0]))
    # def posterior_samples(key, params):
    #     Z_post, lik = true_model.posterior_samples(
    #         key=key, params=params, 
    #         X=X[0], Y=Y[0], R=R[0], session_indices=session_indices[0],
    #         return_samples=True, verbose=True,
    #         posterior_type='smooth',
    #         N_particles=args.N_particles
    #     )
    #     return Z_post, lik

    # # recovered_params = ParamsGLMLearn(log_sigma=-1.7749162, log_sigma_day=-0.02403495, log_alpha=jnp.array([-7.796574 , -4.2768683]))
    # fig, ax = plt.subplots(figsize=(10,5));
    # for rep in range(args.B):
    #     key, subkey1, subkey2 = jax.random.split(key, 3)
    #     # Z_post = posterior_samples(subkey1, recovered_params)
    #     # Z_post_attrue, lik = posterior_samples(subkey2, true_params)
    #     Z_post_attrue, lik = true_model.posterior_samples(
    #         key=subkey1, params=true_params, 
    #         X=X[rep], Y=Y[rep], R=R[rep], session_indices=session_indices[rep],
    #         return_samples=True, verbose=True,
    #         posterior_type='smooth',
    #         N_particles=args.N_particles
    #     )
    #     print(rep, lik)

    #     # Z_error = jnp.linalg.norm(Z_post.mean(axis=0) - Z[0])
    #     # print(Z_error)

    #     # for i in range(2):
    #     #     ax.plot(Z_post.mean(axis=0)[:,i], c='tab:blue', label='Smoothed')
    #     #     ax.fill_between(
    #     #         np.arange(args.T), 
    #     #         Z_post.mean(axis=0)[:,i]-Z_post.std(axis=0)[:,i],
    #     #         Z_post.mean(axis=0)[:,i]+Z_post.std(axis=0)[:,i], 
    #     #         alpha=0.3, color='tab:blue')
            
    #     ax.plot(Z_post_attrue.mean(axis=0)[:,0], label=f'Smooth {rep}')
    #     ax.fill_between(
    #         np.arange(args.T), 
    #         Z_post_attrue.mean(axis=0)[:,0]-Z_post_attrue.std(axis=0)[:,0],
    #         Z_post_attrue.mean(axis=0)[:,0]+Z_post_attrue.std(axis=0)[:,0], 
    #         alpha=0.3, color='tab:orange')
            
    #     ax.plot(Z[rep][:,0], c='k', label='True')
    # ax.legend()
    # ax.axvline(5000)
    # plt.savefig('tests/figures/Z_recovery.png')
    # plt.close()
    
