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
from fit_ibl import fit_optax, fit_EM
from parameters import ParamsGLMLearn, ParameterProperties
import models

def test_recovery():
    # Hyperparams ----------------------------
    N_particles = 10000
    seed = 0
    T = 100
    d = 1   # number of regressors
    B = 1   # number of trajectories (/ subjects)
    learning_rule = 'policy_gradient'

    # Format params and create dataset -------

    key = jax.random.PRNGKey(seed)
    logging.info(f'Number of particles: {N_particles}. Seed: {seed}. T: {T}, B: {B}. d: {d}. Learning rule: {learning_rule}.')

    # Instantiate true model

    true_log_sigma = jax.random.uniform(key, minval=-4.0, maxval=-1.0)
    true_log_sigma_day = jax.random.uniform(key, minval=-2.0, maxval=0.0)
    true_alpha = jax.random.uniform(key, minval=0.0, maxval=0.5, shape=(d+1,))
    true_params = ParamsGLMLearn(log_sigma=true_log_sigma, log_sigma_day=true_log_sigma_day, alpha=true_alpha)

    logging.info(f"True parameters: {true_params}")
    true_model = models.GLMLearn(seed=seed, **true_params._asdict(), learning_rule=learning_rule)

    # Create dataset
    X, Y = [], []
    for _ in range(B):
        key, subkey = jax.random.split(key)
        X_i, Y_i, _, _ = true_model.sample(T=T, key=subkey)
        X.append(X_i)
        Y.append(Y_i)
    session_indices = [jnp.arange(0,T,min(200,int(T/10))).astype(int) for _ in range(B)]

    if learning_rule == 'reinforce':
        R = [jnp.asarray(models.reward(X_sub[:,0], Y_sub)) for X_sub, Y_sub in zip(X, Y)]
    elif learning_rule == 'policy_gradient':
        R = [jnp.asarray(models.effective_reward(X_sub[:,0])) for X_sub, Y_sub in zip(X, Y)]

    # Start fitting --------------------------
    logging.info(f"Starting fitting. Model: GLM with '{learning_rule}' learning rule")
    res = fit_optax(X, Y, R=R, session_indices=session_indices, N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})
    # res = fit_EM(X[0], Y[0], R=R[0], session_indices=session_indices[0], N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})

if __name__=='__main__':
    test_recovery()