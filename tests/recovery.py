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
    d = 1   # number of regressors

    # Parse the command-line arguments
    args = parser.parse_args()

    # Format params and create dataset ---------------

    key = jax.random.PRNGKey(args.seed)
    
    logging.info(f'Number of particles: {args.N_particles}. Seed: {args.seed}. T: {args.T}, B: {args.B}. d: {d}. Learning rule: {args.learning_rule}.')

    # Instantiate true model

    true_log_sigma = jax.random.uniform(key, minval=-4.0, maxval=-1.0)
    true_log_sigma_day = jax.random.uniform(key, minval=-2.0, maxval=0.0)
    true_log_alpha = jax.random.uniform(key, minval=-6, maxval=np.log(0.5), shape=(d+1,))
    true_params = ParamsGLMLearn(log_sigma=true_log_sigma, log_sigma_day=true_log_sigma_day, log_alpha=true_log_alpha)

    logging.info(f"True parameters: {true_params}")
    true_model = models.GLMLearn(seed=args.seed, **true_params._asdict(), learning_rule=args.learning_rule)

    # Create dataset
    X, Y = [], []
    for _ in range(args.B):
        key, subkey = jax.random.split(key)
        X_i, Y_i, _, _ = true_model.sample(T=args.T, key=subkey)
        X.append(X_i)
        Y.append(Y_i)
    session_indices = [jnp.arange(0, args.T, min(200,int(args.T/10))).astype(int) for _ in range(args.B)]

    if args.learning_rule == 'reinforce':
        R = [jnp.asarray(models.reward(X_sub[:,0], Y_sub)) for X_sub, Y_sub in zip(X, Y)]
    elif args.learning_rule == 'policy_gradient':
        R = [jnp.asarray(models.effective_reward(X_sub[:,0])) for X_sub, Y_sub in zip(X, Y)]

    # Start fitting ----------------------------------
        
    logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")
    # res = fit_optax(X, Y, R=R, session_indices=session_indices, N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})

    gridsearch_params = find_initial(
        X, Y, R=R, session_indices=session_indices,
        N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule}, seed=args.seed
        )
    
    initial_params = ParamsGLMLearn(
        log_sigma=gridsearch_params[1], 
        log_sigma_day=gridsearch_params[2], 
        log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1)
        )._asdict()
    logging.info(f'Initial params: {initial_params}')

    # res = fit_optax(
    #     X_train, Y_train, R=R_train, session_indices=session_indices_train, 
    #     N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
    #     initial_params=initial_params
    #     )
    params, results_dict = fit_EM(
        X[0], Y[0], R=R[0], session_indices=session_indices[0],
        N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': 0.},
        initial_params=initial_params, seed=args.seed, n_iters=100, m_step_iters=500,
        posterior_type='smooth',
    )
    logging.info(f"Results: {results_dict}")
        
        # res = fit_EM(X[0], Y[0], R=R[0], session_indices=session_indices[0], N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})