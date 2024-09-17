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
from fit_ibl import fit_optax, fit_EM, find_initial, fit_MLL
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
    # true_log_sigma = jax.random.uniform(subkey1, minval=-4.0, maxval=-1.0)
    # true_log_sigma_day = jax.random.uniform(subkey2, minval=-2.0, maxval=0.0)

    # needs small log sigma for alpha to be learned. PsyTrack found log_sigma ~= -5.5, log_sigma_day ~= -3.5
    true_log_sigma = jax.random.uniform(subkey1, minval=-8, maxval=-5)
    true_log_sigma_day = jax.random.uniform(subkey2, minval=-5, maxval=-3)
    true_log_alpha = jax.random.uniform(subkey3, minval=-6, maxval=-2, shape=(d+1,))
    true_params = ParamsGLMLearn(log_sigma=true_log_sigma, log_sigma_day=true_log_sigma_day, log_alpha=true_log_alpha)

    logging.info(f"True parameters: {true_params}")
    true_model = models.GLMLearn(learning_rule=args.learning_rule) # **true_params._asdict(),

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

    import time
    start = time.time()
    true_loglik = true_model.marginal_log_likelihood(
        key, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
        verbose=True
    )

    # _, true_loglik = true_model.posterior_samples(
    #     key, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
    #     return_history=False, verbose=False, posterior_type='filt'
    # )
    logging.info(f"True log-likelihood (scan): {true_loglik}, time: {time.time()-start}")

    key, scores_key1, scores_key2 = jax.random.split(key, 3)
    scores, log_lik = true_model.next_step_prediction_score(
        scores_key1, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
    )
    logging.info(f"True next-step prediction score: {scores.mean():.3f} ± {scores.std():.3f}, log-lik: {log_lik}")

    scores2t, _ = true_model.two_step_prediction_score(
        scores_key2, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
    )
    logging.info(f"True two-step prediction score: {scores2t.mean():.3f} ± {scores2t.std():.3f}")

    
    # start = time.time()
    # true_loglik = true_model.marginal_log_likelihood_1(
    #     key, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
    #     verbose=True
    # )
    # logging.info(f"True log-likelihood: {true_loglik}, time: {time.time()-start}")
    # sys.exit()

    # partition = int(len(X[0]) * 0.6)
    # key, subkey = jax.random.split(key)
    # prediction_score = true_model.score_predict(
    #     key, true_params, 
    #     X_hist=X[0][:partition], Y_hist=Y[0][:partition], R_hist=R[0][:partition], session_indices=session_indices[0], 
    #     X_pred=X[0][partition:partition+100], Y_pred=Y[0][partition:partition+100],
    #     N_particles=args.N_particles,
    # )
    # logging.info(f"Prediction score: {prediction_score:.4f}")

    # Start fitting ----------------------------------
        
    logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")
    # res = fit_optax(X, Y, R=R, session_indices=session_indices, N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})

    gridsearch_params = find_initial(
        X, Y, R=R, session_indices=session_indices,
        N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule}, seed=args.seed,
        vmap=True, metric='mll', return_top_n=1
        )
    gridsearch_params = gridsearch_params[0]
    logging.info(f"Gridsearch params: {gridsearch_params}")

    # gridsearch_params = [-1.0, -5.0, -4.0]

    # for i, top_grid_point in enumerate(gridsearch_params):
    #     logging.info(f"Grid point #{i}: {top_grid_point}")

        
    initial_params = ParamsGLMLearn(
        log_sigma=gridsearch_params[1], 
        log_sigma_day=gridsearch_params[2], 
        log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1)
        )
    logging.info(f'Initial params: {initial_params}')

    all_log_alpha_samples, accepts = true_model.alpha_mcmc(
            key, initial_params, 
            X[0], Y[0], R=R[0], session_indices=session_indices[0],
            N_particles=args.N_particles, n_iters=50, verbose=False, proposal_scale=0.5,
            )
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,4), ncols=2, constrained_layout=True)
    for j in range(2):
        ax[j].plot(all_log_alpha_samples[:,:,j], alpha=0.1, c='tab:blue')
        ax[j].plot(jnp.percentile(all_log_alpha_samples[:,:,j], q=jnp.array([2.5, 50.0, 97.5]), axis=1).T, c='tab:blue')
        ax[j].axhline(true_log_alpha[j], c='k', ls='--')
    plt.savefig('tests/figures/alpha_samples.png')
    
    # log_alpha_samples = all_log_alpha_samples[-1]
    log_alpha_samples = all_log_alpha_samples.reshape(-1, all_log_alpha_samples.shape[-1])
    ci = jnp.percentile(log_alpha_samples, q=jnp.array([2.5, 97.5]), axis=0).T
    log_geo_means = jnp.mean(log_alpha_samples, axis=0)
    log_ari_means = jax.scipy.special.logsumexp(log_alpha_samples, axis=0) - jnp.log(log_alpha_samples.shape[0])
    log_alpha_meds = jnp.median(log_alpha_samples, axis=0)

    logging.info("Posterior alpha:")
    for i in range(len(initial_params.log_alpha)):
        logging.info(f"alpha_{i}: log geo mean = {log_geo_means[i]:.2f}, log ari mean = {log_ari_means[i]:.2f}, med = {log_alpha_meds[i]:.2f}, CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")
    sys.exit()
    
    res = fit_optax(
        X[0], Y[0], R=R[0], session_indices=session_indices[0],
        N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': 0.},
        initial_params=initial_params
        )

    # params, res = fit_MLL(
    #     key,
    #     X[0], Y[0], R=R[0], session_indices=session_indices[0],
    #     N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': 0.},
    #     initial_params=initial_params
    # )
    # params, results_dict = fit_EM(
    #     X[0], Y[0], R=R[0], session_indices=session_indices[0],
    #     N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': 0.},
    #     initial_params=initial_params, seed=args.seed, n_iters=50, m_step_iters=500,
    #     posterior_type='smooth',
    # )
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
    
