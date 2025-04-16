
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import pickle
import argparse

import ibl
import sys
sys.path.append('../')
import fit_utils
import samplers
import models, optim

import sys
import jax 
import jax.numpy as jnp
import jax.scipy as jsp
# from scipy.optimize import minimize
import jaxopt
from itertools import combinations
import psutil

import tensorflow_probability.substrates.jax.distributions as tfd
from functools import partial

# import os
# os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import parameters
from parameters import ParamsGLMLearn, ParameterProperties
import os, psutil
process = psutil.Process()
from multiprocessing import Pool

from typing import List, Optional

import optax 

def callback_f(intermediate_result):
    val = intermediate_result.fun
    params = intermediate_result.x
    logging.info(f'Likelihood: {-val:5.2f}, params: {ParamsGLMLearn(*params)}')
    # print(intermediate_result.nit, intermediate_result.x, intermediate_result.fun)

def pairs_to_triples(pairs):
    triples = []
    for pair in pairs:
        expanded_vertex = [pair[0], pair[0], pair[1]]
        triples.append(expanded_vertex)
    return jnp.array(triples)

def is_ndimensional_space(points):
    # Convert the list of points to a NumPy array for easier manipulation
    points_array = np.array(points)

    # Check if the number of points is at least N+1
    if len(points) < len(points_array[0]) + 1:
        return False

    # Compute vectors between the points
    vectors = jnp.array([i[0]-i[1] for i in combinations(points_array, 2)])

    # Compute the matrix rank to check for linear independence
    rank = np.linalg.matrix_rank(vectors)

    # If the rank is equal to N, the points form an N-dimensional space
    return rank == len(points_array[0])

def make_n_dimensional(points, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    # Convert the list of points to a NumPy array for easier manipulation
    points_array = np.array(points)

    # Check if the number of points is at least N+1
    if len(points) < len(points_array[0]) + 1:
        raise ValueError("The number of points must be at least N+1")

    # If already n dimensional, just return the points
    if is_ndimensional_space(points):
        return points

    # If the points do not form an N-dimensional space, perturb the last points

    # Compute vectors between the points
    vectors = jnp.array([i[0]-i[1] for i in combinations(points_array, 2)])

    # Compute the matrix rank to check for linear independence
    rank = np.linalg.matrix_rank(vectors)

    rank_deficit = len(points_array[0]) - rank
    for j in range(rank_deficit):
        key, _ = jax.random.split(key)
        perturbation = jax.random.normal(key, shape=(len(points_array[0]),))
        points_array[-j-1] += perturbation

    return points_array

def fit_EM(
        model,
        X: List[jnp.ndarray], Y: List[jnp.ndarray], R: List[jnp.ndarray], day_flags: List[jnp.ndarray],
        n_iters: int=200, m_step_iters=500, N_particles: int=1000, 
        seed: int=0, initial_params: Optional[dict] = None,
        ):
    '''
    Fit model with SMC EM: posterior samples are obtained by SMC, and used to evalutate with Monte-Carlo the ELBO. 
    '''
    logging.info(f'Fitting model with SMC-EM.')
    key = jax.random.PRNGKey(seed)

    learning_rate = 0.01
    m_step_optimizer = optax.adam(learning_rate=learning_rate)
    logging.info(f'M-step optimizer: adam, initial learning rate: {learning_rate}.')

    liks = [] 
    params = initial_params

    def get_samples(params, subject_id):
        Zs, lik = model.posterior_samples(
            key, params, 
            X[subject_id], Y[subject_id], R[subject_id], day_flags[subject_id],
            N_particles=N_particles, verbose=False, LAG=True
            )
        return Zs.mean(0), lik
    
    # def log_joint(params):
    #     def log_joint_per_particle(z):
    #         return model.log_joint(
    #             X, Y, z, R=R, 
    #             # params = ParamsGLMLearn(**_params),
    #             params=params,
    #             day_flags=day_flags,
    #             )
    
    #     log_joint_MCvalues = jax.vmap(log_joint_per_particle)(Zs) 
    #     # log_joint_MCvalues = jax.pmap(log_joint_per_particle)(Zs) 
    #     val = jnp.mean(log_joint_MCvalues, axis=0)
    #     return -val

    # @jax.jit
    # @partial(jax.jit, static_argnums=(2,))
    def neg_log_joint(params, Z, subject_id):
        # def _func(_Z):
        out = -model.log_joint(
            X[subject_id], Y[subject_id], Z, R=R[subject_id], day_flags=day_flags[subject_id],
            params=params,
            )
        return out
        # return jax.nn.logsumexp(jax.vmap(_func)(Z)) - jnp.log(N_particles)
        # return jnp.mean(jax.vmap(_func)(Z))
    
    # def prediction_score(params, split=0.6, window=500):
    #     split_ind = int(len(X) * split)
    #     if split_ind + window > len(X):
    #         window = len(X) - split - 1
    #     score = model.score_predict(
    #         key, params,
    #         X_hist=X[:split_ind], Y_hist=Y[:split_ind], R_hist=R[:split_ind], day_flags=day_flags,
    #         X_pred=X[split_ind:split_ind+window], Y_pred=Y[split_ind:split_ind+window],
    #         N_particles=N_particles
    #     )
    #     return score

    def compute_noise_component(params, Z, subject_id):
        learning_updates = jax.vmap(lambda t: model.update_weights(
            key, Z[t], 
            x=X[subject_id][t], y=Y[subject_id][t], r=R[subject_id][t], day_flag=day_flags[subject_id][t],
            params=params, return_learning_signal=True,
            )[1])(jnp.arange(len(X[subject_id])))
        # learning_updates = learning_updates.mean(1)

        if isinstance(model, models.TimeVarGLMLearn):
            _, w_post = model.split_latent(Z)
        else:
            w_post = Z

        learning_component = w_post[0] + jnp.cumsum(learning_updates, axis=0)
        noise_component = w_post[1:] - learning_component[:-1]
        return noise_component

    logging.info("Starting EM procedure.")
    best_val_per_subject = [-jnp.inf]*len(X)
    best_params_per_subject = [params]*len(X)
    for iter_id in jnp.arange(n_iters):
        if len(X) > 1:
            subject_id = jax.random.randint(jax.random.PRNGKey(iter_id), shape=(), minval=0, maxval=len(X)-1).item()
        else:
            subject_id = 0

        # E-step: Get posterior samples
        Z, lik = get_samples(params, subject_id)
        logging.info(f'[{iter_id} - E] Subject {subject_id}, LL: {lik:5.2f}, params: {params}')

        noise_component = compute_noise_component(params, Z, subject_id)
        logging.info(f'Noise component norm: {jnp.linalg.norm(noise_component):.2f}')

        def objective(params):
            return neg_log_joint(params, Z=Z, subject_id=subject_id)

        # M-step: Optimize log joint. Use optax for optimization
        opt_state = m_step_optimizer.init(params)
        for m_iter_id in range(m_step_iters):

            # Grad of vmap over particles
            neg_val, grad = jax.value_and_grad(objective)(params)
            val = -neg_val
            if val > best_val_per_subject[subject_id]:
                best_val_per_subject[subject_id] = val
                best_params_per_subject[subject_id] = params

            # Update
            updates, opt_state = m_step_optimizer.update(grad, opt_state, loss=neg_val)
            params = optax.apply_updates(params, updates)
            if m_iter_id % 50 == 0:
                logging.info(f'[{iter_id} - M - {m_iter_id}] Subject {subject_id}, joint: {val:.2f}' + (f', log_alpha grad norm = {jnp.linalg.norm(grad.log_alpha):.2e}' if 'log_alpha' in grad._fields else ''))

        logging.info(f"[{iter_id} - M] Optim result: {params}")
        logging.info(f'Memory: {process.memory_info().rss/1e6:5.2f} MB')

        # # ELBO:
        # def log_joint_per_particle(z):
        #     return model.log_joint(
        #         X, Y, z, R=R,
        #         session_indices=session_indices
        #         )
        # log_joint_MCvalues = jax.vmap(log_joint_per_particle)(Zs)
        # entropy_term = jnp.mean(log_joint_MCvalues, axis=0)

        # ELBO = -res.fun - entropy_term + lik
        # approx_lik_diff = ELBO - lik
        # # if iter_id > 0:
        # #     ELBO_delta = ELBO - ELBOs[-1]
        # #     logging.info(f'[{iter_id} / M] Optim success: {res.success}. ELBO: {ELBO:.2f} (Δ: {ELBO_delta:.2f}), log-joint: {-res.fun:.2f}.')
        # # else:
        # logging.info(f'[{iter_id} / M] Optim success: {res.success}. ELBO: {ELBO:.2f}. params: {res.x}')
        # ELBOs.append(ELBO)
        # # logging.info(f'[{iter_id}] Params: {res.x}')
        
        # liks.append(lik.item())

    # result_dict = {
    #     'params': params,
    #     'T': T,
    #     'N_particles': N_particles,
    #     'seed': seed,
    #     'liks': liks
    # }

    return params, (best_val_per_subject, best_params_per_subject), opt_state

def fit_LaplaceEM(X, Y, R=None, n_iters=200, model_kwargs={}, seed=0, N_particles=1000, session_indices=[]):
    from jax.scipy.optimize import minimize
    T, D = X.shape

    current_params = [-1.0, -1.0, 0.5]

    model = models.GLMLearn(seed=seed, **model_kwargs)
    model.update_params_from_array(current_params)
    logging.info("Getting initial MAP estimate")
    (Zs, _), lik = samplers.bootstrap_filter(
            N_particles, 
            X, Y, 
            model, 
            R=R, session_indices=session_indices, 
            return_history=True, verbose=True,
    )
    Z_MAP = jnp.mean(Zs, axis=0) # Initial estimate for Z_MAP

    for iter_id in jnp.arange(n_iters):
        model = models.GLMLearn(seed=seed, **model_kwargs)
        model.update_params_from_array(current_params)

        # E-step: 
        def neg_log_posterior(Z_flat):
            '''
            posterior is proportional to joint
            '''
            Z = Z_flat.reshape((T, D+1))
            val = model.log_joint(X, Y, Z, R=R, session_indices=session_indices)
            return - val
        
        res = minimize(
            neg_log_posterior,
            x0 = Z_MAP.flatten(),
            method='BFGS',
            jac = jax.grad(neg_log_posterior),
            options={'disp':101},
            )
        Z_MAP_flat = res.x
        print(res)

        Hinv = jnp.linalg.pinv(jax.hessian(neg_log_posterior)(Z_MAP_flat))
        print(Hinv.shape)

        # M-Step
        def neg_log_joint(params_array):
            val = model.log_joint(
                X, Y, Z=Z_MAP_flat.reshape((T, D+1)), R=R, session_indices=session_indices,
                params=ParamsGLMLearn._make(params_array), #ParamsGLMLearn(**params),
                )
            return - val

        res = minimize(
            neg_log_joint,
            x0 = current_params,
            method='L-BFGS-B',
            jac = jax.grad(neg_log_joint),
            bounds=[(-6,2), (-6,2), (0,1)],
            )

        logging.info(f'[{iter_id} - M] Optim success: {res.success}, Log-joint: {-res.fun:5.2f}, params: {res.x}')
        current_params = res.x
    return current_params     

def fit_MLL(key, X, Y, R=None, model_kwargs={}, N_particles=10000, session_indices=[], initial_params=None):
    model = models.GLMLearn(**model_kwargs)
    T, D = X.shape

    if initial_params is None:
        initial_params = ParamsGLMLearn(log_sigma=-3.0, log_sigma_day=-1.0, alpha=0.5 * jnp.ones(D+1))

    # Define objective function
    @jax.jit
    def neg_MLL(params):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        loglik = model.marginal_log_likelihood(
                key, params, X, Y, R=R, session_indices=session_indices,
                N_particles=N_particles,
            )
        # logging.info(f'Likelihood: {lik:5.2f}, memory: {process.memory_info().rss/1e6:5.2f} MB')
        # logging.info(f'Func eval. Likelihood: {loglik}, params: {params}')
        return - loglik

    def callback(xk):
        lik = -neg_MLL(xk)
        logging.info(f'Marginal log-lik: {lik}, Params: {xk}')

    # solver = jaxopt.ScipyMinimize(
    #     method='Nelder-Mead',
    #     jit=True,
    #     fun=neg_MLL,
    #     callback=callback,
    #     )
    logging.info(f'Fitting model with direct optimization of MLL.')
    solver = jaxopt.BFGS(neg_MLL, verbose=True, maxiter=20)
    res = solver.run(init_params=initial_params)
    logging.info(f'Final params: {res.params}')
    logging.info(f'Final state: {res.state}')
    return res.params, res

def find_initial(
        model, 
        X: list[jnp.ndarray], Y: list[jnp.ndarray], R: list[jnp.ndarray] = None, day_flags: list[jnp.ndarray] = None,
        N_particles: int=1000, seed: int=0,
        vmap=True, return_top_n=1,
        vector_alpha=False,
        ) -> list:
    '''
    Perform grid search to find initial parameters for optimization.
    Returns list of `return_top_n` best parameters, in the sense of marginal log-likelihood, of type determined by the `model`.
    '''
    d = X[0].shape[-1]
    key = jax.random.PRNGKey(seed)
    
    # Define grid
    if isinstance(model, models.TimeVarGLMLearn):
        betas = jnp.linspace(-5.0, 5.0, 6) if model.lapse else jnp.linspace(0.0, 1.0, 5)
        log_sigmas = jnp.linspace(-8.0, -3.0, 6)
        log_qs = jnp.linspace(-9.0, -4.0, 4)
        log_alphas = jnp.linspace(-10.0, -1.0, 8)
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_qs, betas), axis=-1).reshape(-1,4)
    elif isinstance(model, models.AC):
        log_alphas = jnp.linspace(-8.0, -2.0, 6)
        log_sigmas = jnp.linspace(-8.0, -3.0, 6)
        logit_betas = jnp.linspace(-5.0, 5.0, 6) if model.sigmoid else jnp.linspace(-1.0, 1.0, 6)
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, logit_betas, log_sigmas), axis=-1).reshape(-1,4)
    elif isinstance(model, models.GLMRegLearn):
        log_alphas = jnp.linspace(-10.0, -1.0, 8)
        log_sigmas = jnp.linspace(-8.0, -3.0, 6)
        baselines = jnp.array([-1.0, 0.0, 1.0])
        betas = jnp.array([0., 1.])
        gammas = jnp.array([0., 1.])
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, baselines, betas, gammas), axis=-1).reshape(-1,5)
    elif isinstance(model, models.QLearning):
        log_alphas = jnp.linspace(-8.0, -2.0, 6)
        log_sigmas = jnp.linspace(-8.0, -3.0, 4)
        percept_log_scales = jnp.linspace(-3.0, 0.0, 6)
        log_temps = jnp.linspace(-2.0, 3.0, 6)
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, percept_log_scales, log_temps), axis=-1).reshape(-1,4)
    elif isinstance(model, models.GLMBaseLearn):
        log_alphas = jnp.linspace(-8.0, -2.0, 8)
        log_sigmas = jnp.linspace(-8.0, -3.0, 6)
        log_sigma_days = jnp.linspace(-5.0, -2.0, 3)
        betas = jnp.linspace(-1.0, 1.0, 5)
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_sigma_days, betas), axis=-1).reshape(-1,4)
    elif isinstance(model, models.DynamicGLMHMM):
        log_sigma = jnp.linspace(-8.0, 2.0, 8)
        alpha = jnp.linspace(0, 3, 8)
        grid_points = jnp.stack(jnp.meshgrid(log_sigma, alpha), axis=-1).reshape(-1,2)
    elif isinstance(model, models.RVBF):
        log_alphas = jnp.linspace(-8.0, -2.0, 6)
        log_sigmas = jnp.linspace(-8.0, -3.0, 5)
        log_sigma_days = jnp.linspace(-5.0, -2.0, 3)
        log_qs = jnp.linspace(-9.0, -4.0, 3)
        baselines = jnp.array([-1.0, 0.0, 1.0])
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_sigma_days, log_qs, baselines), axis=-1).reshape(-1,5)
    elif isinstance(model, models.TimeVarRVBF):
        log_alphas = jnp.linspace(-8.0, -2.0, 6)
        log_sigmas = jnp.linspace(-8.0, -4.0, 5)
        log_sigma_days = jnp.linspace(-5.0, -3.0, 3)
        log_sigmas_0 = jnp.linspace(-8.0, -4.0, 5)
        baselines = jnp.array([-1.0, 0.0, 1.0])
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_sigma_days, log_sigmas_0, baselines), axis=-1).reshape(-1,5)
    else:
        log_sigmas = jnp.linspace(-8.0, -3.0, 8)
        log_sigma_days = jnp.linspace(-5.0, -2.0, 5)
        log_alphas = jnp.linspace(-10.0, -1.0, 10)
        # ps = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_sigma_days), axis=-1).reshape(-1,3)
    logging.info(f'Finding initialization from grid of {grid_points.shape} points.')

    def array_to_params(params_array):
        if isinstance(model, models.TimeVarGLMLearn):
            log_alpha, log_sigma, log_q, beta_0 = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            baseline = jnp.zeros((d+1,)) if vector_alpha else 0.0
            # baseline = jnp.zeros((d+1,)) if vector_alpha else 0.0
            # params = parameters.ParamsTimeVarGLMLearn(
            #     beta_0=beta_0, log_alpha=log_alpha_val, log_sigma_0=log_sigma, log_sigma=log_sigma, log_sigma_day=log_sigma_day # * jnp.ones(model.latent_dim)
            #     )
            params = parameters.ParamsTimeVarGLMLearn(
                beta_0=beta_0, log_alpha=log_alpha_val, log_sigma_0=log_sigma, log_sigma=log_sigma, log_sigma_day=-3.0,
                log_Q = log_q* jnp.ones((d+1,)), baseline=baseline
                # baseline=baseline
            )
            
        elif isinstance(model, models.GLMHMMLearn):
            log_alpha, log_sigma, log_sigma_day = params_array
            params = parameters.ParamsGLMHMMLearn(logit_pi0=0.0, logit_a_1=5.0, logit_a_2=5.0, log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day)
        elif isinstance(model, models.Psytrack):
            log_alpha, log_sigma, log_sigma_day = params_array
            params = parameters.ParamsPsytrack(log_sigma=log_sigma, log_sigma_day=log_sigma_day)
        elif isinstance(model, models.GLMRegLearn):
            log_alpha, log_sigma, baseline, beta, gamma = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            params = parameters.ParamsGLMRegLearn(
                log_alpha=log_alpha_val, 
                log_sigma=log_sigma, log_sigma_day=-2.0, 
                Q=jnp.zeros((d+1,)),
                A=jnp.zeros((d+1,)), kappa=0., 
                gamma=gamma, beta=beta,
                baseline=baseline,
            )
        elif isinstance(model, models.GLMInterpLearn):
            log_alpha, log_sigma, log_sigma_day = params_array
            params = parameters.ParamsGLMInterpLearn(
                log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day, 
                Q=jnp.zeros((d+1,)),)
        elif isinstance(model, models.QLearning):
            log_alpha, log_sigma, percept_log_scale, log_temp = params_array
            params = parameters.ParamsQLearning(
                log_alpha=log_alpha, log_sigma=log_sigma, percept_log_scale=percept_log_scale, log_temp=log_temp,
            )
        elif isinstance(model, models.GLMBaseLearn):
            log_alpha, log_sigma, log_sigma_day, beta = params_array
            params = parameters.ParamsGLMBaseLearn(log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day, baseline_weights=beta * jnp.ones((5,5,)))
        elif isinstance(model, models.DynamicGLMHMM):
            log_sigma, alpha = params_array
            params = parameters.ParamsDynamicGLMHMM(log_sigma=log_sigma, alpha=alpha)
        elif isinstance(model, models.AC):
            log_alpha, log_sigma, beta_0, log_sigma_0 = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            # beta_0 = jnp.zeros((d+1,)) if vector_alpha else 0.0
            beta_0_val = beta_0 * jnp.ones((d+1,))
            log_q = -9.0
            params = parameters.ParamsAC(
                beta_0=beta_0_val, log_alpha=log_alpha_val, log_sigma_0=log_sigma_0, log_sigma=log_sigma, log_sigma_day=-3.0,
                log_Q=log_q * jnp.ones((d+1,))
            )
        elif isinstance(model, models.RVBF):
            log_alpha, log_sigma, log_sigma_day, log_q, baseline = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            baseline_val = baseline * jnp.ones((d+1,)) if vector_alpha else baseline
            log_sigma = log_sigma * jnp.ones((d+1,)) if vector_alpha else log_sigma
            log_sigma_day = log_sigma_day * jnp.ones((d+1,)) if vector_alpha else log_sigma_day
            params = parameters.ParamsRVBF(
                log_alpha=log_alpha_val, log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_Q=log_q * jnp.ones((d+1,)), baseline=baseline_val
            )
        elif isinstance(model, models.TimeVarRVBF):
            log_alpha, log_sigma, log_sigma_day, log_sigma_0, baseline = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            baseline_val = baseline * jnp.ones((d+1,)) if vector_alpha else baseline
            log_sigma = log_sigma * jnp.ones((d+1,)) if vector_alpha else log_sigma
            log_sigma_day = log_sigma_day * jnp.ones((d+1,)) if vector_alpha else log_sigma_day
            log_q = -5.0
            params = parameters.ParamsTimeVarRVBF(
                log_alpha=log_alpha_val, log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_Q=log_q * jnp.ones((d+1,)), 
                baseline=baseline_val, log_sigma_0=log_sigma_0, 
            )
        else:
            log_alpha, log_sigma, log_sigma_day = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            params = ParamsGLMLearn(log_alpha=log_alpha_val, log_sigma=log_sigma, log_sigma_day=log_sigma_day)#, p=p)
        return params
    
    logging.info(f"Memory available: {psutil.virtual_memory().available/1e6} MB")

    # Define loss
    def neg_MLL(params_array):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        params = array_to_params(params_array)

        # Compute marginal log-likelihood with SMC, summing over subjects
        loglik = 0.
        for _X, _Y, _R, _day_flags in zip(X, Y, R, day_flags):
            _loglik = model.marginal_log_likelihood(
                key, params, _X, _Y, R=_R, day_flags=_day_flags,
                N_particles=N_particles,
            )
            loglik += _loglik
        return -loglik
    
    # Compute neg MLL over grid
    if vmap:
        vals = jax.vmap(neg_MLL)(grid_points)
    else:
        vals = []
        for params in tqdm(grid_points, desc='Grid search'):
            vals.append(neg_MLL(params))
        vals = jnp.array(vals)

    # Print 5 highest MLL grid points
    sorted_indices = jnp.argsort(vals)
    logging.info('Top 5 grid points ([log_alpha, log_sigma, log_sigma_day]):')
    for i in range(5):
        logging.info(f'\tMLL: {-vals[sorted_indices[i]]:.2f}, Grid point: {grid_points[sorted_indices[i]]}')

    # # Make initial simplex for nelder mead
    # vertices = []
    # for i in range(3+1):
    #     vertex = grid_points[sorted_indices[i]]
    #     vertices.append(vertex)
    # # vertex_2 = grid_points[sorted_indices[1]]
    # # vertex_3 = grid_points[sorted_indices[2]] + jax.random.normal(key, shape=(3,))*0.1 # add noise to ensure non-collinearity
    # # vertex_3 = grid_points[sorted_indices[2]] + jax.random.normal(key, shape=(3,))*0.1 # add noise to ensure non-collinearity
    # initial_simplex = jnp.stack(vertices, axis=0)
    # initial_simplex = make_n_dimensional(initial_simplex, key=key)

    # Take lowest
    best_param_arrays = grid_points[sorted_indices[:return_top_n]]
    best_params = [array_to_params(params_array) for params_array in best_param_arrays]

    # # Fix sigma
    # logging.info("Fixing log sigma")
    # best_params = [params._replace(log_sigma=-5.0) for params in best_params]

    # logging.info(f'Best params: [log_alpha, log_sigma, log_sigma_day]={best_param_arrays[0]}, neg MLL: {vals[sorted_indices[0]]:.2f}, ')
    return best_params


def fit_optax(
        model,
        X: List[jnp.ndarray], Y: List[jnp.ndarray], R: List[jnp.ndarray], day_flags: List[jnp.ndarray],
        n_iters: int=200, N_particles: int=1000, 
        seed: int=0, initial_params: Optional[dict] = None, correct_bias=True,
        ):
    '''
    Fit model via optimization of the marginal log-likelihood (MLL) using optax. 
    MLL is computed via SMC, and summed across subjects.
    Args:
        X: list of jnp arrays, each of shape (T_i, d), where T_i is the number of trials for subject i and d is the number of regressors.
        Y: list of jnp arrays, each of shape (T_i,), decisions
        R: list of jnp arrays, each of shape (T_i,), rewards
        n_iters: number of optimization iterations,
        model_kwargs: dictionary of model parameters,
        seed: int, random seed,
        N_particles: int, number of particles for SMC,
        session_indices: list of lists, session indices for each subject.
    Returns:
        params: optimized parameters
    '''
    # T, d = X.shape
    # model = models.GLMLearn(**model_kwargs)
    # model = models.TimeVarGLMLearn(**model_kwargs)
    key = jax.random.PRNGKey(seed)
    logging.info(f'Fitting model with optax gradient-ascent of MLL.')

    learning_rate = 0.01
    scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=int(n_iters/2), # 2 cycles of decay
            decay_rate=0.1,
            transition_begin=10)
    
    optimizer = optax.amsgrad(learning_rate=learning_rate)
    logging.info(f'Optimizer: amsgrad. Minimize neg MLL. Learning rate: {learning_rate}.')
    # optimizer = optax.chain(
    #     # optax.scale_by_adam(),
    #     optax.scale_by_schedule(scheduler),
    # )
    # logging.info(f'Starting optimization. Optimizer: adam + reduce_on_plateau, learning rate: {learning_rate}')

    # Define loss
    # @partial(jax.jit, static_argnums=(1,))
    def neg_MLL(params, subject_id):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        # # Might need to add a loop over subjects
        # loglik = model.marginal_log_likelihood(
        #     key, params,
        #     X, Y, R=R, day_flags=day_flags,
        #     N_particles=N_particles,
        #     verbose=False,
        #     )

        # Compute marginal log-likelihood with SMC, summing over subjects
        loglik = 0.
        # for X_sub, Y_sub, R_sub, day_flags_sub in zip(X, Y, R, day_flags):
        loglik_sub = model.marginal_log_likelihood(
            key, params, 
            X[subject_id], Y[subject_id], R[subject_id], day_flags[subject_id],
            N_particles=N_particles,
            verbose=False,
        )
        loglik += loglik_sub
        return -loglik
    
    @partial(jax.jit, static_argnums=(1,))
    def neg_MLL2(params, subject_id):
        log_lik = model.filtering_MLL(
            key, params,
            X[subject_id], Y[subject_id], R=R[subject_id], day_flags=day_flags[subject_id],
            N_particles=N_particles,
            )
        return -log_lik
    
    def compute_noise_component(params, Z, subject_id):
        learning_updates = jax.vmap(lambda t: model.update_weights(
            key, Z[t], 
            x=X[subject_id][t], y=Y[subject_id][t], r=R[subject_id][t], day_flag=day_flags[subject_id][t],
            params=params, return_learning_signal=True,
            )[1])(jnp.arange(len(X[subject_id])))
        # learning_updates = learning_updates.mean(1)

        if isinstance(model, models.TimeVarGLMLearn) or isinstance(model, models.AC):
            _, w_post = model.split_latent(Z)
        else:
            w_post = Z

        if learning_updates.ndim == 1:
            learning_updates = learning_updates[:, jnp.newaxis]

        learning_component = w_post[0] + jnp.cumsum(learning_updates, axis=0)
        noise_component = w_post[1:] - learning_component[:-1]
        return noise_component
    
    # @jax.jit
    # def next_step_predict_score(params):
    #     scores, _ = model.next_step_prediction_score(
    #         key, params,
    #         X, Y, R=R, day_flags=day_flags,
    #         N_particles=N_particles,
    #         verbose=False,
    #     )
    #     return -jnp.mean(scores)
    
    # Initialize
    params = initial_params
    opt_state = optimizer.init(params)  # Obtain the `opt_state` that contains statistics for the optimizer.

    def evaluate(params):
        for i, (X_sub, Y_sub, R_sub, day_flags_sub) in enumerate(zip(X, Y, R, day_flags)):
            t2_score, _ = model.two_step_prediction_score(
                key, params,
                X_sub, Y_sub, R=R_sub, day_flags=day_flags_sub,
                N_particles=N_particles,
            )
            logging.info(f"[Subject {i}] t2 score: {t2_score.mean()}")

   
    # Optimize
    best_val_per_subject = [-jnp.inf]*len(X)
    best_params_per_subject = [params]*len(X)
    for i in range(n_iters):
        # key, _ = jax.random.split(key)
        # with Pool(10) as pool:
        #     grads = pool.map(neg_evidence_value_and_grad, zip(X, Y, R, session_indices))
        # print(grads)
        if len(X) > 1:
            subject_id = jax.random.randint(jax.random.PRNGKey(i), shape=(), minval=0, maxval=len(X)-1).item()
        else:
            subject_id = 0

        neg_val, grad = jax.value_and_grad(lambda p: neg_MLL(p, subject_id))(params)
        val = -neg_val
        if val > best_val_per_subject[subject_id]:
            best_val_per_subject[subject_id] = val
            best_params_per_subject[subject_id] = params

        logging.info(f'[{i}] Subject {subject_id}, LL: {val:.4f}, L per trial: {jnp.exp(val/len(Y[subject_id])):.4f}, params: {params}')
        # if i % 10 == 0:
        #     evaluate(params)
            # logging.info(f'[{i}] \t2step score: {evaluate(params):.4f}')
        # val, grad = jax.value_and_grad(next_step_predict_score)(params)

        updates, opt_state = optimizer.update(grad, opt_state, loss=neg_val)
        
        # # Fix sigma
        # updates = updates._replace(log_sigma=0.0)
        params = optax.apply_updates(params, updates)

        if i % 20 == 0:
            # if not isinstance(model, models.QLearning): # not implemented for Q-learning
            #     Zs, _ = model.posterior_samples_scan(
            #         key, params, 
            #         X[subject_id], Y[subject_id], R[subject_id], day_flags[subject_id],
            #         N_particles=N_particles, verbose=False, correct_bias=correct_bias,
            #         )
            #     Z = Zs.mean(0)
            #     noise_component = compute_noise_component(params, Z, subject_id)
            #     logging.info(f'[{i}] Noise component norm: {jnp.linalg.norm(noise_component):.2f}')

            # Forward pass liks and predicitions

            _, forward_loglik = model.forward_pass(
                key, params,
                X[subject_id], Y[subject_id], R[subject_id], day_flags[subject_id],
                N_particles=N_particles, predict_Y=False, correct_bias=correct_bias,
            )
            logging.info(f'[{i}] Forward pass loglik: {forward_loglik:.2f}, L per trial: {jnp.exp(forward_loglik/len(Y[subject_id])):.4f}')

            _, prediction_loglik = model.forward_pass(
                key, params,
                X[subject_id], Y[subject_id], R[subject_id], day_flags[subject_id],
                N_particles=N_particles, predict_Y=True, correct_bias=correct_bias,
            )
            logging.info(f'[{i}] Prediction loglik: {prediction_loglik:.2f}, L per trial: {jnp.exp(prediction_loglik/len(Y[subject_id])):.4f}')


        # if opt_state[1].lr != lr_scale:
        #     lr_scale = opt_state[1].lr
        #     logging.info(f'[{i}] ReduceLROnPlateau: Learning rate: {lr_scale * learning_rate:.2e}')

    return params, (best_val_per_subject, best_params_per_subject), opt_state

def parallel_fit():
    # Load IBL data
    df = pd.read_csv('./data/ibl_learning_processed.csv')
    
    lab = 'wittenlab'
    lab_df = df[df['lab']=='wittenlab']
    logging.info(f'Loaded IBL {lab} data.')

    lab_subjects = np.unique(lab_df['subject'].values)

    Xs, Ys, Rs = [], [], []
    for subject in lab_subjects:
        X, Y, sess_ind = ibl.get_mouse_design(
            lab_df, subject=subject, 
            regressors=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'], # 'stimIntensity' #'contrastLeft', 'contrastRight',
            )
        # R = jnp.asarray(models.reward(X[:,1]-X[:,0], Y))
        ER = jnp.asarray(models.effective_reward(X[:,1]-X[:,0]))
        logging.info(f"Loaded subject {subject} data. T={len(Y)}.")

        Xs.append(X)
        Ys.append(Y)
        Rs.append(ER)

    # Start parallel fit
    logging.info('Starting parallel fitting. PG learning rule.')
    logging.info(f'CPU count: {os.cpu_count()}.')
    with Pool() as pool:
        results = pool.map(fit_MLL, zip(Xs, Ys, Rs))

    for i, res in enumerate(results):
        logging.info(f'Subject {lab_subjects[i]}: {res.x} at {res.nit} iterations. Likelihood: {-res.fun}')
    
    return results

def plot_landscape(X, Y, R=None, model_kwargs={}, seed=0, N_particles=10000, day_flags=None):
    def neg_MLL(params_array):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        # Instantiate model
        model = models.GLMLearn(seed=seed, **model_kwargs)#, learning_rule='reinforce')
        model.update_params_from_array(params_array)

        # Compute marginal log-likelihood with SMC
        _, lik = samplers.bootstrap_filter(
            N_particles, 
            X, Y, 
            model, 
            R=R, day_flags=day_flags, 
            return_history=False, verbose=True
            )
        return -lik / len(Y)
    
    key = jax.random.PRNGKey(seed)

    logging.info('Finding NM initial simplex.')
    alphas = jnp.array([0.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.5])
    logsigmas = jnp.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    logsigma_days = jnp.array([-2.0, -1.0, 0.0])

    # Compute neg MLL over grid
    grid_points = jnp.stack(jnp.meshgrid(logsigmas, logsigma_days, alphas), axis=-1).reshape(-1,3)
    vals = jax.vmap(neg_MLL)(grid_points)

    # for i, val in enumerate(vals):
    #     logging.info(f'Params: {grid_points[i]}, neg MLL: {val:.2f}')

    df = pd.DataFrame({'log_sigma':grid_points[:,0], 'log_sigma_day':grid_points[:,1], 'alpha':grid_points[:,-1], 'neg_MLL':vals})
    print(df)

    fig, axs = plt.subplots(figsize=[12,4], ncols=3, constrained_layout=True)

    for i, log_sigma_day in enumerate(logsigma_days):
        sub_df = df[df['log_sigma_day']==log_sigma_day]
        plot_df = sub_df.pivot(index='log_sigma', columns='alpha', values='neg_MLL')
        sns.heatmap(plot_df, annot=True, fmt=".3f", ax=axs[i], cmap='viridis_r')
        axs[i].set_title(f'log_sigma_day: {log_sigma_day}')
        # axs[].plot(plot_df['log_sigma'], plot_df['neg_MLL'], label=f'log_sigma_day: {log_sigma_day}')
    # print(plot_df)
    plt.savefig('landscape.png', dpi=300)
    return vals

def test_held_out_sessions(
        model, params,
        held_out_sessions: list,
        X: jnp.ndarray, Y: jnp.ndarray, R: jnp.ndarray = None, day_flags: jnp.ndarray = None,
        session_indices: list=None,
        N_particles: int=10000, 
        seed: int=0,
        ):
    '''
    Evaluate model on held-out sessions.
    Return p(y_{t1:t2} | y_{1:t1-1}, x_{1:t2-1}) for each held-out session of trials interval [t1:t2]
    '''
    key = jax.random.PRNGKey(seed)

    # Compute all filtering log-likelihood p(y_t | y_{1:t-1}, x_{1:t})
    _, all_log_liks = model.marginal_log_likelihood(
        key, params, 
        X, Y, R, day_flags, N_particles=N_particles, 
        return_logliks=True
        )
    
    # test LL is sum of (predictive) log-likelihoods of held-out sessions
    T1 = [session_indices[i] for i in held_out_sessions]
    T2 = [session_indices[i+1] for i in held_out_sessions]
    test_lls = []
    for t1, t2 in zip(T1, T2):
        test_lls.append(all_log_liks[t1:t2])

    test_lls = jnp.concatenate(test_lls, axis=0)
    return test_lls.sum(), test_lls

def test(
        model, params,
        held_out_tials: jnp.ndarray,
        X: jnp.ndarray, Y: jnp.ndarray, R: jnp.ndarray = None, day_flags: jnp.ndarray = None,
        N_particles: int=10000, 
        seed: int=0,
        ):
    '''
    Evaluate model on held-out sessions.
    Return p(y_{t1:t2} | y_{1:t1-1}, x_{1:t2-1}) for each held-out session of trials interval [t1:t2]
    '''
    key = jax.random.PRNGKey(seed)

    # Compute all filtering log-likelihood p(y_t | y_{1:t-1}, x_{1:t})
    test_mlls = model.held_out_trials_marginal_log_likelihood(
        key, params, 
        held_out_tials,
        X, Y, R, day_flags, N_particles=N_particles,
        )
    
    return test_mlls.sum(), test_mlls

def posterior_mcmc(
        model, 
        key, initial_params, 
        X: list, Y: list, R: list, day_flags: list,
        N_particles=1000, n_iters=100, N_samples=100,
        verbose=True, proposal_scale=1.0,
        ):
    '''Metropolis hastings to sample from posterior of alpha.'''
    proposal = lambda x: tfd.Normal(loc=x, scale=proposal_scale)
    initial_params_array, lengths = parameters.params_to_array(initial_params)

    def array_to_params(array):
        return parameters.array_to_params(initial_params, array, lengths)
    
    @partial(jax.vmap, in_axes=(0, None))
    def log_target_func(params_array, key) -> float:
        # params_prop = params._replace(log_alpha=log_alpha)
        # params_prop = initial_params.from_array(params_array, lengths)
        params_prop = array_to_params(params_array)

        mll = 0.
        for X_sub, Y_sub, R_sub, day_flags_sub in zip(X, Y, R, day_flags):
            mll += model.marginal_log_likelihood(key, params_prop, X_sub, Y_sub, R_sub, day_flags_sub, N_particles)
            # mll += model.filtering_MLL(key, params_prop, X_sub, Y_sub, R_sub, day_flags_sub, N_particles)
        return mll
    
    @jax.jit
    def metropolis_hastings_step(carry, inputs):
        '''
        Single step of Metropolis-Hastings. 
        Written in form amendable to jax.lax.scan.
        '''
        # Unpack
        params_array, log_fx, best_log_fx, best_params_array = carry
        key = inputs
        key, proposal_key, accept_key, eval_key = jax.random.split(key, 4)

        # Proposal step
        params_array_prop = proposal(params_array).sample(seed=proposal_key)

        # Acceptance step
        log_fx_prop = log_target_func(params_array_prop, eval_key)

        log_ratio = log_fx_prop - log_fx
        accept = jnp.log(jax.random.uniform(accept_key)) < log_ratio

        params_array = jnp.where(accept[:, None], params_array_prop, params_array)
        log_fx = jnp.where(accept, log_fx_prop, log_fx)

        # Keep best params
        best_params_array = jnp.where(log_fx[:, None] > best_log_fx[:, None], params_array, best_params_array)
        best_log_fx = jnp.where(log_fx > best_log_fx, log_fx, best_log_fx)

        return (params_array, log_fx, best_log_fx, best_params_array), (log_fx, params_array, accept)
    
    # Initialize
    keys = jax.random.split(key, n_iters)
    # log_alpha = params.log_alpha

    params_array_samples_t = jnp.tile(initial_params_array, (N_samples, 1))
    log_fx = log_target_func(params_array_samples_t, key)
    
    best_params_array = params_array_samples_t
    best_log_fx = log_fx

    # Start loop
    all_accepts, log_lik_samples, params_array_samples = [], [], []
    MH_state = (params_array_samples_t, log_fx, best_log_fx, best_params_array)
    for i in range(n_iters):
        # Run Metropolis-Hastings step
        MH_state, (_, _, accepts) = metropolis_hastings_step(MH_state, keys[i])

        # Print confidence intervals
        if i % 10 == 0 and verbose:
            # med_params = initial_params.from_array(jnp.median(MH_state[0], axis=0), lengths)
            med_params = array_to_params(jnp.median(MH_state[0], axis=0))
            logging.info(f"[{i}/{n_iters}] Med params: {med_params}, accept frac: {accepts.sum()/N_samples:.4f}")
            logging.info(f"CI: {jnp.percentile(MH_state[0], q=jnp.array([2.5, 97.5]), axis=0).T}")
            logging.info(f"Marginal log prob estimate: {MH_state[1].mean():.4f}")

        all_accepts.append(accepts)
        log_lik_samples.append(MH_state[1])
        params_array_samples.append(MH_state[0])
    
    all_accepts = jnp.stack(all_accepts)
    log_lik_samples = jnp.stack(log_lik_samples)
    params_array_samples = jnp.stack(params_array_samples)

    # # Wrap in scan
    # _, (log_lik_samples, params_array_samples, all_accepts) = jax.lax.scan(
    #     metropolis_hastings_step,
    #     params_array_samples_t, keys, length=n_iters
    #     )
        
    assert params_array_samples.shape[0] == n_iters and params_array_samples.shape[1] == N_samples
        
    logging.info("*"*40)
    logging.info("Posterior sampling results:")
    
    params_array_samples_t, log_fx, best_log_fx, best_params_array = MH_state

    best_ind = jnp.argmax(best_log_fx)
    best_log_fx = best_log_fx[best_ind]
    best_params_array = best_params_array[best_ind]
    best_params = array_to_params(best_params_array)

    # best_params_array = best_params_array.mean(0)
    # best_log_fx = best_log_fx.mean()
    logging.info(f"Final best params: {best_params}")
    logging.info(f"Final best log-lik: {best_log_fx}")

    # med_params = initial_params.from_array(jnp.median(params_array_samples_t, axis=0), lengths)
    med_params = array_to_params(jnp.median(params_array_samples_t, axis=0))
    logging.info(f"Final median params: {med_params}, mean accept frac: {all_accepts.mean():.2f}")

    logging.info(f"Final params CI: {jnp.percentile(params_array_samples_t, q=jnp.array([2.5, 97.5]), axis=0).T}")
    logging.info(f"Marginal log-lik estimate: {log_fx.mean():.4f}")
    logging.info("*"*40)
    return all_accepts, log_lik_samples, params_array_samples, (best_params, best_log_fx)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Argument parser for IBL fitting.")

    # Add arguments
    parser.add_argument("--lab", type=str, default="wittenlab", 
                        choices=["angelakilab", "churchlandlab", "cortexlab", "danlab", "hoferlab", "mainenlab", "mrsicflogellab", "wittenlab", "zadorlab"],
                        help="IBL lab name")
    parser.add_argument("--subject-id", type=int, default=0, 
                        help="Subject index in the lab data.")
    parser.add_argument("--all-subjects", action='store_true',
                        help="Override subject ID and group all subjects together from lab.")
    parser.add_argument("--learning-rule", type=str, default="reinforce", choices=["policy_gradient", "reinforce", "regression_gradient", "max_ent", "max_ent_MC"], 
                        help="Learning rule for the model.")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Seed for random number generator")
    parser.add_argument("-N", "--N-particles", type=int, default=1000, 
                        help="Number of particles for SMC.")
    parser.add_argument("--model-class", type=str, default="GLMLearn", 
                        choices=["GLMLearn", "TimeVarGLMLearn", "Psytrack", "GLMRegLearn", "GLMHMMLearn", "GLMInterpLearn", "QLearning", "GLMBaseLearn", 'DynamicGLMHMM', "AC", "RVBF", "TimeVarRVBF"],
                        help="Model class to use for fitting.")
    parser.add_argument("--vector-alpha", action='store_true',
                        help="Use vector alpha for GLM.")
    parser.add_argument("--lapse", action='store_true',
                        help="lapse for timevar GLM")
    parser.add_argument("--EM", action='store_true', default=False,
                        help="Use EM for inference. Default is to use optimization of MLL.")
    parser.add_argument("--regressors", type=str, nargs='+', default=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'],
                        help="Regressors to use for the model.")
    parser.add_argument("--modulator", type=str, default='lr', choices=['lr', 'baseline'],
                        help="Modulator for the TimeVarRVBF model.")
    parser.add_argument("--protocol", type=str, default='training', choices=['training', 'no_curriculum'],
                        help="Protocol of the IBL mouse data. See `ibl.py' for details.")

    # Parse the command-line arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    model = fit_utils.load_model(args)

    try:
        # Override subject index with SLURM array task ID
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        idx = args.subject_id

    seed = args.seed
    key = jax.random.PRNGKey(seed)
    logging.info(f'Number of particles: {args.N_particles}. Seed: {seed}.')

    # Load IBL data
    # regressors = ['stimIntensity']
    regressors = args.regressors # ['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'] #'stimIntensity',

    X_train, Y_train, R_train, day_flags_train = [], [], [], []
    X_full, Y_full, R_full, day_flags_full = [], [], [], []
    all_held_out_trials = []
    
    if args.all_subjects:
        n_subjects = ibl.get_number_subjects(args.lab)
        logging.info(f"Fitting on all {n_subjects} subjects in lab {args.lab}")
        subjects = range(ibl.get_number_subjects(args.lab)-1)
    else:
        logging.info(f"Fitting on single subject {idx} in lab {args.lab}")
        subjects = [idx]

    for subject_id in subjects:
        loader_params = {
            'lab': args.lab,
            'subject_id': subject_id,
            'regressors': regressors,
            'learning_rule': args.learning_rule,
            'seed': seed,
            'protocol': args.protocol,
        }
        loader = ibl.IBLSingleTrajectoryLoader(loader_params)
        train_trajectory = loader.load_train_data()
        logging.info(f"Loaded IBL train data, subject id {subject_id}. T={len(train_trajectory.X)}. Params: {loader_params}")

        if args.model_class == "QLearning":
            X_train.append(train_trajectory.X[:, 1] - train_trajectory.X[:, 0])
        else:
            X_train.append(jnp.array(train_trajectory.X))
        Y_train.append(jnp.array(train_trajectory.Y))
        R_train.append(jnp.array(train_trajectory.R))
        day_flags_train.append(train_trajectory.day_flags)

        trajectory, held_out_trials = loader.load_test_data()
        if args.model_class == "QLearning" and ('stimIntensity' not in regressors):
            X_full.append(trajectory.X[:, 1] - trajectory.X[:, 0])
        else:
            X_full.append(trajectory.X)
        Y_full.append(trajectory.Y)
        R_full.append(trajectory.R)
        day_flags_full.append(trajectory.day_flags)
        all_held_out_trials.append(held_out_trials)

    if isinstance(model, models.TimeVarGLMLearn):
        beta_dim = len(regressors) + 1 if args.vector_alpha else 1
        model.beta_dim = beta_dim
        model.latent_dim = len(regressors) + beta_dim + 1
        logging.info(f"Model latent dim: {model.latent_dim}, beta dim: {beta_dim}")
    elif isinstance(model, models.AC):
        beta_dim = len(regressors) + 1
        model.beta_dim = beta_dim
        model.latent_dim = len(regressors) + beta_dim + 1
        model.reward_func = ibl.format_reward_function(regressors, learning_rule='reinforce')
        logging.info(f"Model latent dim: {model.latent_dim}, beta dim: {beta_dim}")
    elif isinstance(model, models.GLMHMMLearn):
        model.latent_dim = len(regressors) + 1
    elif isinstance(model, models.TimeVarRVBF):
        model.modulator = args.modulator
        if model.modulator == 'lr':
            beta_dim = 1
        elif model.modulator == 'baseline':
            beta_dim = len(regressors) + 1
        model.beta_dim = beta_dim
        model.latent_dim = len(regressors) + beta_dim + 1
        model.reward_func = ibl.format_reward_function(regressors, learning_rule='reinforce')
        logging.info(f"Model latent dim: {model.latent_dim}, beta dim: {beta_dim}")
    else:
        model.latent_dim = len(regressors) + 1
        model.reward_func = ibl.format_reward_function(regressors, learning_rule=args.learning_rule)

    logging.info(f"Model: {model}")

    # ----------------------------------------------------------------

    # Step 1: Grid search

    logging.info('Starting fitting.')
    logging.info("-"*80)
    logging.info('Step 1: Grid search for initialization.')
    top_gridsearch_params = find_initial(
        model,
        X_train, Y_train, R=R_train, day_flags=day_flags_train,
        N_particles=args.N_particles, seed=seed,
        return_top_n=-1,
        vector_alpha=args.vector_alpha, vmap=True
        )
    initial_params = top_gridsearch_params[0]
    # initial_params = top_gridsearch_params[-1]

    # Step 2

    # if args.all_subjects:
    #     logging.info("Gradient computation is too expensive for all subjects. Skipping optimization.")
    #     params = initial_params
    # else:
    logging.info("-"*80)
    logging.info('Step 2: Optimization of MLL with gradient ascent.')
    if args.EM:
        fit_method = fit_EM
    else:
        fit_method = fit_optax

    optax_final_params, (best_val_per_subject, best_params_per_subject), res = fit_method(
        model,
        X_train, Y_train, R=R_train, day_flags=day_flags_train, 
        N_particles=args.N_particles,
        initial_params=initial_params, n_iters=100,
        )
    logging.info(f'Final params: {optax_final_params}')
    for sub in range(len(X_train)):
        best_val, sub_params = best_val_per_subject[sub], best_params_per_subject[sub]
        logging.info(f'Subject {subject_id}, Best: MLL: {best_val:.2f}, params: {sub_params}')
    params = optax_final_params if args.all_subjects else best_params_per_subject[0]

    for i in range(3):
        key = jax.random.PRNGKey(seed+i)
        log_lik = model.marginal_log_likelihood(
            key, params, 
            X_train[0], Y_train[0], R=R_train[0], day_flags=day_flags_train[0],
            N_particles=args.N_particles
            )
        logging.info(f"'marginal_log_likelihood' Log-lik: {log_lik:.2f}")

        log_lik_filter = model.filtering_MLL(
            key, params, 
            X_train[0], Y_train[0], R=R_train[0], day_flags=day_flags_train[0],
            N_particles=args.N_particles
            )
        logging.info(f"'filter' Log-lik: {log_lik_filter:.2f}")

    # Step 3

    logging.info("-"*80)
    logging.info('Step 3: Parameter posterior sampling with MCMC.')
    all_accepts, log_lik_samples, posterior_samples, _ = posterior_mcmc(
            model, key, params, 
            X_train, Y_train, R=R_train, day_flags=day_flags_train,
            N_particles=args.N_particles, n_iters=100, N_samples=500,
            verbose=True, proposal_scale=0.5
            )
    
    BURN_IN = 50
    logging.info(f"")
    logging.info(f"Marginal log-likelihood estimate: {log_lik_samples[-BURN_IN:].mean():.2f}")

    posterior_samples = posterior_samples[-BURN_IN:].reshape(-1, posterior_samples.shape[-1])
    ci = jnp.percentile(posterior_samples, q=jnp.array([2.5, 97.5]), axis=0).T
    posterior_means = jnp.mean(posterior_samples, axis=0)
    posterior_meds = jnp.median(posterior_samples, axis=0)
    # log_ari_means = jax.scipy.special.logsumexp(log_alpha_samples, axis=0) - jnp.log(log_alpha_samples.shape[0])

    logging.info("Posterior alpha:")
    for i in range(len(posterior_means)):
        logging.info(f"alpha_{i}: mean = {posterior_means[i]:.2f}, med = {posterior_meds[i]:.2f}, CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")

    # Step 4 : Evaluation

    logging.info("-"*80)
    logging.info("Evaluating model (first animal).")

    lengths = parameters.params_to_array(params)[1]
    for eval_params_array in [posterior_means, posterior_meds]:
        # eval_params = params._replace(log_alpha=eval_log_alpha)
        # eval_params = params.from_array(eval_params_array, lengths)
        eval_params = parameters.array_to_params(params, eval_params_array, lengths)
        logging.info(f'Params: {eval_params}')

        loglik = model.marginal_log_likelihood(
            key, eval_params, 
            X_full[0], Y_full[0], R=R_full[0], day_flags=day_flags_full[0],
            N_particles=args.N_particles
            )
        logging.info(f"Complete trajectory 'marginal_log_likelihood': {loglik:.2f}, per trial: {loglik/len(Y_full[0]):.4f}, L per trial: {jnp.exp(loglik/len(Y_full[0])):.4f}")

        # loglik_filter = model.filtering_MLL(
        #     key, eval_params, 
        #     X_full[0], Y_full[0], R=R_full[0], day_flags=day_flags_full[0],
        #     N_particles=args.N_particles
        #     )
        # logging.info(f"Filtering (prior predictive) log-likelihood: {loglik_filter:.2f}, per trial: {loglik_filter/len(Y_full[0]):.4f}, L per trial: {jnp.exp(loglik_filter/len(Y_full[0])):.4f}")

        _, forward_loglik = model.forward_pass(
            key, params,
            X_full[0], Y_full[0], R=R_full[0], day_flags=day_flags_full[0],
            N_particles=args.N_particles, predict_Y=False,
        )
        logging.info(f'Forward pass (prior predictive, use Y) loglik: {forward_loglik:.2f}, L per trial: {jnp.exp(forward_loglik/len(Y_full[0])):.4f}')

        _, prediction_loglik = model.forward_pass(
            key, params,
            X_full[0], Y_full[0], R=R_full[0], day_flags=day_flags_full[0],
            N_particles=args.N_particles, predict_Y=True
        )
        logging.info(f'Prediction (prior predictive, sample Y) loglik: {prediction_loglik:.2f}, L per trial: {jnp.exp(prediction_loglik/len(Y_full[0])):.4f}')

        if not isinstance(model, models.QLearning):
            Zs, loglik2 = model.posterior_samples(
                key, params, 
                X_full[0], Y_full[0], R=R_full[0], day_flags=day_flags_full[0],
                N_particles=args.N_particles, verbose=False, LAG=True
                )
            Z = Zs.mean(0)
            
            learning_updates = jax.vmap(lambda t: model.update_weights(
                key, Z[t], 
                x=jnp.array(X_full[0])[t], y=jnp.array(Y_full[0])[t], r=jnp.array(R_full[0])[t], day_flag=jnp.array(day_flags_full[0])[t],
                params=params, return_learning_signal=True,
                )[1])(jnp.arange(len(X_full[0])))
            # learning_updates = learning_updates.mean(1)

            if isinstance(model, models.TimeVarGLMLearn) or isinstance(model, models.AC) or isinstance(model, models.TimeVarRVBF):
                _, w_post = model.split_latent(Z)
            else:
                w_post = Z

            if learning_updates.ndim == 1:
                learning_updates = learning_updates[:, None]

            learning_component = w_post[0] + jnp.cumsum(learning_updates, axis=0)
            noise_component = w_post[1:] - learning_component[:-1]
            logging.info(f'Noise component norm: {jnp.linalg.norm(noise_component):.2f}, per trial: {jnp.linalg.norm(noise_component)/len(Y_full[0]):.4f}')

        # scores, loglik = model.next_step_prediction_score(
        #     key, eval_params, 
        #     X_train[0], Y_train[0], R=R_train[0], day_flags=day_flags_train[0],
        #     N_particles=args.N_particles, verbose=True
        #     )
        # logging.info(f"Total log-likelihood: {loglik:.2f}, per trial: {loglik/len(Y_train[0]):.4f}")
        # logging.info(f"Prediction score: {scores.mean():.4f} +/- {scores.std():.4f}, quartiles: {jnp.percentile(scores, jnp.array([25, 50, 75]))}")
        
        # scores2, _ = model.two_step_prediction_score(
        #     key, eval_params,
        #     X_train[0], Y_train[0], R=R_train[0], day_flags=day_flags_train[0], 
        #     N_particles=args.N_particles, verbose=True
        #     )
        # logging.info(f"Two-step prediction score: {scores2.mean():.2f} +/- {scores2.std():.2f}, quartiles: {jnp.percentile(scores2, jnp.array([25, 50, 75]))}")

        # # Test

        # test_ll, test_lls = test(
        #     model, eval_params, all_held_out_trials[0],
        #     all_trajectories[0].X, all_trajectories[0].Y, R=all_trajectories[0].R, day_flags=all_trajectories[0].day_flags,
        #     N_particles=args.N_particles
        #     )
        # logging.info(f"Test log-likelihood: {test_ll:.2f}, per trial: {test_lls.mean():.4f}")

    # sys.exit()
    
    # for i, gridsearch_params in enumerate(top_gridsearch_params):
    #     logging.info(f"Grid point #{i}: {gridsearch_params}")
    
    #     # # gridsearch_params = [-3.0, -5.0, -4.0]
    #     # if isinstance(model, models.TimeVarGLMLearn):
    #     #     initial_params = models.ParamsTimeVarGLMLearn(
    #     #         beta_0=1.0,
    #     #         log_alpha=gridsearch_params[0], 
    #     #         log_sigma_0=gridsearch_params[1], 
    #     #         log_sigma=gridsearch_params[1], 
    #     #         log_sigma_day=gridsearch_params[2]
    #     #         )
    #     # else:
    #     #     initial_params = ParamsGLMLearn(
    #     #         log_sigma=gridsearch_params[1], 
    #     #         log_sigma_day=gridsearch_params[2], 
    #     #         log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1)
    #     #         )
    #     logging.info(f'Initial params: {initial_params}')

    #     logging.info("Evaluating model prediction.")
    #     scores, loglik = model.next_step_prediction_score(
    #         key, initial_params, 
    #         X_train[0], Y_train[0], R=R_train[0], day_flags=day_flags_train[0], 
    #         N_particles=args.N_particles, verbose=True
    #         )
        
    #     scores2, _ = model.two_step_prediction_score(
    #         key, initial_params, 
    #         X_train[0], Y_train[0], R=R_train[0], day_flags=day_flags_train[0],
    #         N_particles=args.N_particles, verbose=True
    #         )

    #     logging.info(f"Prediction score: {scores.mean():.2f} +/- {scores.std():.2f}, quartiles: {jnp.percentile(scores, jnp.array([25, 50, 75]))}")
    #     logging.info(f"Two-step prediction score: {scores2.mean():.2f} +/- {scores2.std():.2f}, quartiles: {jnp.percentile(scores2, jnp.array([25, 50, 75]))}")
    #     logging.info(f"Total log-likelihood: {loglik:.2f}, likelihood per trial: {jnp.exp(loglik/len(Y)):.2f}")

    #     # logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")
    #     # params, res = fit_optax(
    #     #     X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
    #     #     N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
    #     #     initial_params=initial_params, n_iters=200
    #     #     )
    #     # params, result_dict = fit_EM(
    #     #     X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
    #     #     N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
    #     #     initial_params=initial_params,
    #     #     posterior_type='smooth',
    #     #     n_iters=20,
    #     # )
    #     # params, _ = fit_MLL(
    #     #     key, X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0],
    #     #     N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
    #     #     initial_params=initial_params,
    #     #     )
