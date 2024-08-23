
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
import samplers
import models, optim

import sys
import jax 
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.optimize import minimize
from itertools import combinations
# import os
# os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        X, Y, R=None, 
        n_iters=200, model_kwargs={}, 
        seed=0, N_particles=1000, 
        session_indices=[], initial_params=None, 
        m_step_iters=500,
        posterior_type='smooth'
        ):
    '''
    Fit model with SMC EM: posterior samples are obtained by SMC, and used to evalutate with Monte-Carlo the ELBO. 
    '''
    logging.info(f'Fitting model with SMC-EM.')
    logging.info(f"Posterior type: {posterior_type}. N_particles: {N_particles}.")
    T, D = X.shape
    key = jax.random.PRNGKey(seed)
    model = models.GLMLearn(seed=seed, **model_kwargs)

    # current_params = [-5.0, -1.0, 0.5]

    if initial_params is None:
        initial_params = ParamsGLMLearn(log_sigma=-3.0, log_sigma_day=-1.0, alpha=0.5 * jnp.ones(D+1))._asdict()

    learning_rate = 0.01
    scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=int(m_step_iters/2), # 2 cycles of decay
            decay_rate=0.1,
            transition_begin=10)
    
    # opt = optax.adam(learning_rate)
    m_step_optimizer = optax.amsgrad(learning_rate=learning_rate)
    # m_step_optimizer = optax.chain(
    #     optax.scale_by_adam(),
    #     optax.scale_by_learning_rate(learning_rate=learning_rate),
    # #     optax.scale_by_schedule(scheduler),
    # )
    logging.info(f'M-step optimizer: amsgrad, initial learning rate: {learning_rate}.')



    liks = [] 
    params = initial_params

    def get_samples(params):
        Zs, lik = model.posterior_samples(
            key, params,
            X, Y, R=R, session_indices=session_indices,
            N_particles=N_particles, 
            return_history=True,
            verbose=False,
            posterior_type=posterior_type
            )
        return Zs, lik
    
    def log_joint(params):
        def log_joint_per_particle(z):
            return model.log_joint(
                X, Y, z, R=R, 
                # params = ParamsGLMLearn(**_params),
                params=params,
                session_indices=session_indices,
                )
    
        log_joint_MCvalues = jax.vmap(log_joint_per_particle)(Zs) 
        # log_joint_MCvalues = jax.pmap(log_joint_per_particle)(Zs) 
        val = jnp.mean(log_joint_MCvalues, axis=0)
        return -val
    
    def prediction_score(params, split=0.6, window=500):
        split_ind = int(len(X) * split)
        if split_ind + window > len(X):
            window = len(X) - split - 1
        score = model.score_predict(
            key, params,
            X_hist=X[:split_ind], Y_hist=Y[:split_ind], R_hist=R[:split_ind], session_indices=session_indices,
            X_pred=X[split_ind:split_ind+window], Y_pred=Y[split_ind:split_ind+window],
            N_particles=N_particles
        )
        return score

    logging.info("Starting EM procedure.")
    for iter_id in jnp.arange(n_iters):
        # model = models.GLMLearn(seed=seed, **params, **model_kwargs)
        # model.update_params_from_array(params)

        # E-step: MC samples from the posterior
        # (Zs_smooth, Zs_filt), lik = samplers.bootstrap_filter(
        #     N_particles, 
        #     X, Y, 
        #     model, 
        #     R=R, session_indices=session_indices, 
        #     return_history=True if posterior_type == 'smooth' else False, 
        #     verbose=False,
        #     )
        # if posterior_type == 'smooth':
        #     Zs = Zs_smooth
        #     del Zs_filt
        # elif posterior_type == 'filt':
        #     Zs = Zs_filt
        #     del Zs_smooth

        Zs, lik = get_samples(params)
        assert Zs.shape == (N_particles, T, D+1)
        logging.info(f'[{iter_id} - E] Lik: {lik:5.2f}')

        for split in [0.2, 0.5, 0.8]: 
            logging.info(f"[{iter_id}] T500 Prediction score with {split} history: {prediction_score(params, split):.4f}")

        # Z_mean = jnp.mean(Zs, axis=0)
        # print('Average std over particles', jnp.std(Zs, axis=0).mean(axis=0))

        # M-Step
        # def neg_log_joint(params_array):
        #     def log_joint_per_particle(z):
        #         return model.log_joint(
        #             X, Y, z, R=R, 
        #             params = ParamsGLMLearn._make(params_array),
        #             session_indices=session_indices
        #             )
        
        #     log_joint_MCvalues = jax.vmap(log_joint_per_particle)(Zs) #! try vectorizing directly in log_joint
        #     val = jnp.mean(log_joint_MCvalues, axis=0)
        #     # logging.info(f'Log-joint: {val} pm {jnp.std(log_joint_MCvalues, axis=0)}')

        #     # model = models.GLMLearn(seed=seed, **params, **model_kwargs)
        #     # val = model.log_joint(
        #     #     X, Y,
        #     #     Z=Z_mean,
        #     #     R=R, 
        #     #     params=ParamsGLMLearn._make(params_array), #ParamsGLMLearn(**params),
        #     #     session_indices=session_indices
        #     #     )
        #     # logging.info(f'Log-joint value: {val:5.2f}, params: {params_array}')
        #     return - val

        # def log_joint(params):
        #     def log_joint_per_particle(z):
        #         return model.log_joint(
        #             X, Y, z, R=R, 
        #             # params = ParamsGLMLearn(**_params),
        #             params=params,
        #             session_indices=session_indices,
        #             )
        
        #     log_joint_MCvalues = jax.vmap(log_joint_per_particle)(Zs) 
        #     # log_joint_MCvalues = jax.pmap(log_joint_per_particle)(Zs) 
        #     val = jnp.mean(log_joint_MCvalues, axis=0)
        #     return val


        # res = minimize(
        #     neg_log_joint,
        #     x0 = current_params,
        #     method='Nelder-Mead',
        #     tol=1e-3,
        #     options={'disp':True, 'return_all':True},
        #     callback=callback_f,
        #     bounds=[(-6,2), (-6,2), (0,1)]
        #     )

        # res = minimize(
        #     neg_log_joint,
        #     x0 = current_params,
        #     method='L-BFGS-B',
        #     jac = jax.grad(neg_log_joint),
        #     bounds=[(-6,2), (-6,2), (0,1)],
        #     # options={'disp':101}, # iprint number
        #     )

        # use optax for optimization
        opt_state = m_step_optimizer.init(params)
        for m_iter_id in range(m_step_iters):

            # Grad of vmap over particles
            val, grad = jax.value_and_grad(log_joint)(params)

            # # Vmap of grads over particles
            # grads = jax.vmap(
            #     lambda z: jax.grad(lambda _params: neg_log_joint2(_params, z))
            #     )(Zs)
            # print(grad['log_alpha'].shape, grads.shape)
            # sys.exit()

            # Update
            updates, opt_state = m_step_optimizer.update(grad, opt_state, loss=val)
            params = optax.apply_updates(params, updates)
            if m_iter_id % 50 == 0:
                logging.info(f'[{iter_id} - M - {m_iter_id}] joint: {-val:.2f}, log_alpha grad norm = {jnp.linalg.norm(grad.log_alpha):.2e}')

        logging.info(f"[{iter_id} - M] Optim result: {params}")
        logging.info(f'Memory: {process.memory_info().rss/1e6:5.2f} MB')

        # if learning_rate * opt_state[1].lr < 1e-06:
        #     logging.info(f"Learning rate below threshold. Exiting.")
        #     break

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

        # # Update
        # current_params = res.x
        
        liks.append(lik.item())

    result_dict = {
        'params': params,
        'T': T,
        'N_particles': N_particles,
        'seed': seed,
        'liks': liks
    }

    return params, result_dict

def fit_LaplaceEM(X, Y, R=None, n_iters=200, model_kwargs={}, seed=0, N_particles=1000, session_indices=[]):
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

def fit_MLL(X, Y, R=None, model_kwargs={}, seed=0, N_particles=10000, session_indices=[]):
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
            R=R, session_indices=session_indices, 
            return_history=False, verbose=True
            )
        # _, lik = samplers.bootstrap_filter(N_particles, X=X, Y=Y, R=R, return_history=False, verbose=True)
        # logging.info(f'Likelihood: {lik:5.2f}, memory: {process.memory_info().rss/1e6:5.2f} MB')
        # logging.info(f'Func eval. Likelihood: {lik}, params: {params_array}')
        return -lik
    
    def find_initial_simplex(key=None):
        if key is None:
            key = jax.random.PRNGKey(seed)

        logging.info('Finding NM initial simplex.')
        alphas = jnp.array([0.0, 0.01, 0.05, 0.1, 0.5])
        logsigmas = jnp.array([-5.0, -4.0, -3.0, -2.0, -1.0])
        N = 3

        # Compute neg MLL over grid
        grid_points = jnp.stack(jnp.meshgrid(logsigmas, logsigmas, alphas), axis=-1).reshape(-1,N)
        # grid_points = pairs_to_triples(grid_points)
        vals = jax.vmap(neg_MLL)(grid_points)

        # Take lowest 3 grid points to define initial simplex
        sorted_indices = jnp.argsort(vals)
        
        vertices = []
        for i in range(N+1):
            vertex = grid_points[sorted_indices[i]]
            vertices.append(vertex)
        # vertex_2 = grid_points[sorted_indices[1]]
        # vertex_3 = grid_points[sorted_indices[2]] + jax.random.normal(key, shape=(3,))*0.1 # add noise to ensure non-collinearity
        # vertex_3 = grid_points[sorted_indices[2]] + jax.random.normal(key, shape=(3,))*0.1 # add noise to ensure non-collinearity
        initial_simplex = jnp.stack(vertices, axis=0)
        initial_simplex = make_n_dimensional(initial_simplex, key=key)
        
        logging.info('Initial simplex:')
        for i, vertex in enumerate(initial_simplex):
            logging.info(f'\tparams: {vertex}, neg MLL: {vals[sorted_indices[i]]:.2f}')
        return initial_simplex

    # initial_simplex = find_initial_simplex()
    # initial_simplex = jnp.array([
    #     [-5.0, -1.0, 0.0],
    #     [-1.0, -1.0, 0.0],
    #     [-3.0, -5.0, 0.0],
    #     [-3.0, -1.0, 0.5]
    # ])
    # logging.info(f'Initial simplex: {initial_simplex}')

    # vertices =[]
    # for vertex in initial_simplex:
    #     expanded_vertex = [vertex[0], vertex[0], vertex[1]]
    #     vertices.append(expanded_vertex)
    # initial_simplex = jnp.array(vertices)
    # print(initial_simplex)

    # res = minimize(
    #     neg_MLL,
    #     x0 = initial_simplex[0],
    #     method='Nelder-Mead',
    #     tol=1e-3,
    #     options={'disp':True, 'return_all':True, 'initial_simplex':initial_simplex},
    #     callback=callback_f,
    #     bounds=[(-6,2), (-6,2), (0,1)]
    #     )

    res = minimize(
        neg_MLL,
        x0 = [-5.0, -1.0, 0.5],
        method='L-BFGS-B',
        jac = jax.grad(neg_MLL),
        bounds=[(-6,2), (-6,2), (0,1)],
        options={'disp':101}, # iprint number
        )
    return res

def find_initial(
        X: list[jnp.ndarray], Y: list[jnp.ndarray], R: list[jnp.ndarray] = None, session_indices: list[list]=[[]],
        N_particles: int=1000, 
        model_kwargs: dict={}, seed: int=0,
        vmap=True, metric='mll',
        ) -> jnp.ndarray:
    log_sigmas = jnp.linspace(-4.0, -1.0, 5)
    log_sigma_days = jnp.linspace(-2.0, 0.0, 3)
    log_alphas = jnp.linspace(-8.0, -2.0, 5)
    grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_sigma_days), axis=-1).reshape(-1,3)
    logging.info(f'Finding initialization from grid of {grid_points.shape} points.')
    model = models.GLMLearn(seed=seed, **model_kwargs)
    key = jax.random.PRNGKey(seed)
    
    import psutil
    logging.info(f"Memory available: {psutil.virtual_memory().available/1e6} MB")

    # Define loss
    def neg_MLL(params_array):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        log_alpha, log_sigma, log_sigma_day = params_array
        params = ParamsGLMLearn(log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day)

        # # Instantiate model
        # model = models.GLMLearn(
        #     log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day, 
        #     seed=seed, 
        #     **model_kwargs
        #     )

        # Compute marginal log-likelihood with SMC, summing over subjects
        loglik = 0.
        for _X, _Y, _R, _sess_ind in zip(X, Y, R, session_indices):
            # _, _loglik = samplers.bootstrap_filter(
            #     N_particles, 
            #     _X, _Y, 
            #     model, 
            #     R=_R, session_indices=_sess_ind, 
            #     return_history=False, verbose=False
            #     )
            _loglik = model.marginal_log_likelihood(
                key, params, _X, _Y, R=_R, session_indices=_sess_ind,
                N_particles=N_particles,
            )
            loglik += _loglik
        return -loglik
    
    split = 0.5
    window = 500
    split_ind = int(len(X[0]) * split)
    if split_ind + window > len(X[0]):
        window = len(X[0]) - split - 1

    def score_predict(params_array):
        log_alpha, log_sigma, log_sigma_day = params_array
        params = ParamsGLMLearn(log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day)

        score = model.score_predict(
            key, params,
            X_hist=X[0][:split_ind], Y_hist=Y[0][:split_ind], R_hist=R[0][:split_ind], session_indices=session_indices[0], 
            X_pred=X[0][split_ind:split_ind+window], Y_pred=Y[0][split_ind:split_ind+window],
            N_particles=N_particles,
        )
        return -score

    # Compute neg MLL over grid
    if metric == 'mll':
        func = neg_MLL
    elif metric == 'prediction_score':
        func = score_predict
    
    if vmap:
        vals = jax.vmap(func)(grid_points)
    else:
        vals = []
        for params in tqdm(grid_points, desc='Grid search'):
            vals.append(func(params))
        vals = jnp.array(vals)
    # vals = jax.pmap(neg_MLL)(grid_points)

    # Print 5 highest MLL grid points
    sorted_indices = jnp.argsort(vals)
    logging.info('Top 5 grid points ([log_alpha, log_sigma, log_sigma_day]):')
    for i in range(5):
        logging.info(f'\tneg MLL: {vals[sorted_indices[i]]:.2f}, Grid point: {grid_points[sorted_indices[i]]}')

    # Take lowest
    best_params = grid_points[sorted_indices[0]]
    # logging.info(f'Best params: [log_alpha, log_sigma, log_sigma_day]={best_params}, neg MLL: {vals[sorted_indices[0]]:.2f}, ')
    return best_params


def fit_optax(
        X: list[jnp.ndarray], Y: list[jnp.ndarray], R: list[jnp.ndarray] = None, session_indices: list[list]=[[]],
        n_iters: int=200, N_particles: int=10000, 
        model_kwargs: dict={}, seed: int=0, initial_params: Optional[dict] = None,
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
    T, d = X.shape
    model = models.GLMLearn(seed=seed, **model_kwargs)
    key = jax.random.PRNGKey(seed)
    logging.info(f'Fitting model with direct gradient optimization of MLL.')

    if initial_params is None:
        initial_params = ParamsGLMLearn(log_sigma=-3.0, log_sigma_day=-1.0, log_alpha= -0.5 * jnp.ones(d+1))._asdict()

    learning_rate = 0.05
    # optimizer = optax.adam(learning_rate)
    # optimizer = optax.noisy_sgd(learning_rate=learning_rate)
    # optimizer = optax.chain(
    #     optax.adam(learning_rate),
    #     optax.contrib.reduce_on_plateau(
    #         patience=10, # Number of epochs with no improvement after which learning rate will be reduced
    #         factor=0.1, # Factor by which to reduce the learning rate
    #         # rtol=1e-3, 
    #     ),
    # )
    scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=int(n_iters/2), # 2 cycles of decay
            decay_rate=0.1,
            transition_begin=10)
    
    optimizer = optax.amsgrad(learning_rate=learning_rate)
    logging.info(f'Optimizer: amsgrad. Minimize neg MLL. Learning rate: {learning_rate}.')
    # optimizer = optax.chain(
    #     # optax.scale_by_adam(),
    #     optax.scale_by_
    #     optax.scale_by_schedule(scheduler),
    # )
    # logging.info(f'Starting optimization. Optimizer: adam + reduce_on_plateau, learning rate: {learning_rate}')



    # Define loss
    def neg_MLL(params):#, _X, _Y, _R, _sess_ind):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        # Instantiate model
        # model = models.GLMLearn(seed=seed, **params, **model_kwargs)#, learning_rule='reinforce')
        # print(model.params)
        # model.update_params_from_array(params)

        # Compute marginal log-likelihood with SMC
        # loglik = 0.
        # for _X, _Y, _R, _sess_ind in zip(X, Y, R, session_indices):
        #! Removing loop over subjects
        # _, loglik = samplers.bootstrap_filter(
        #     N_particles, 
        #     X, Y, 
        #     model, 
        #     R=R, session_indices=session_indices, 
        #     return_history=False, verbose=False
        #     )
        # loglik += _loglik
        # model.update_params_from_array(params)
        # loglik = _loglik

        # X = jnp.array(X)

        # logliks = jax.pmap(
        #     lambda _X: samplers.bootstrap_filter(
        #         N_particles, 
        #         _X, Y[0], 
        #         model, 
        #         R=R[0], session_indices=session_indices[0], 
        #         return_history=False, verbose=False
        #         )[1]
        #     )(
        #         jnp.asarray(X),
        #         )
        # print(logliks)

        # _, lik = samplers.bootstrap_filter(N_particles, X=X, Y=Y, R=R, return_history=False, verbose=True)
        # logging.info(f'Likelihood: {lik:5.2f}, memory: {process.memory_info().rss/1e6:5.2f} MB')
        # logging.info(f'Func eval. Likelihood: {lik}, params: {params_array}')

        loglik = model.marginal_log_likelihood(
            key, params,
            X, Y, R=R, session_indices=session_indices,
            N_particles=N_particles,
            verbose=False,
            )
        return -loglik
    

    # Obtain the `opt_state` that contains statistics for the optimizer.
    params = initial_params #._asdict()
    opt_state = optimizer.init(params)
    # lr_scale = opt_state[1].lr
    for i in range(n_iters):
        # with Pool(10) as pool:
        #     grads = pool.map(neg_evidence_value_and_grad, zip(X, Y, R, session_indices))
        # print(grads)
        # sys.exit()
        val, grad = jax.value_and_grad(neg_MLL)(params)

        # updates, opt_state = optimizer.update(grad, opt_state)
        updates, opt_state = optimizer.update(grad, opt_state, loss=val)
        params = optax.apply_updates(params, updates)
        # params['alpha'] = optax.projections.projection_non_negative(params['alpha'])

        # if opt_state[1].lr != lr_scale:
        #     lr_scale = opt_state[1].lr
        #     logging.info(f'[{i}] ReduceLROnPlateau: Learning rate: {lr_scale * learning_rate:.2e}')
        logging.info(f'[{i}] lik: {-val:.2f}, params: {params}')

        # Ensure alpha is non-negative
    return params

# def neg_evidence_value_and_grad(args):
#     _X, _Y, _R, _sess_ind = args
#     print(_X, _Y, _R, _sess_ind)
#     def neg_MLL(params):
#         return samplers.bootstrap_filter(
#                 N_particles, 
#                 _X, _Y, 
#                 model = models.GLMLearn(seed=seed, **params, learning_rule='reinforce'), #! fixing learning rule
#                 R=_R, session_indices=_sess_ind, 
#                 return_history=False, verbose=True
#                 )[1]
#     grad = jax.grad(neg_MLL)(params)
#     return grad

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
            regressors=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']
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

# def fit_single_trajectory_search():
#     T = 10000
#     alpha_values = [0.0, 0.01, 0.1, 0.5]
#     sigma_values = [-5.0, -3.0, -1.0]

#     convergence_dicts = []
#     for true_alpha in alpha_values:
#         for true_logsigma in sigma_values:
#             conv_dict = fit_single_trajectory(true_alpha, true_logsigma, T=T)
#             print('-'*50)
#             print(conv_dict)
#             print('-'*50)

#             convergence_dicts.append(conv_dict)

#     with open(f'saves/convergence_dicts_w_initial_simplex_T{T}.pkl', 'wb') as f:
#         pickle.dump(convergence_dicts, f)

# def fit_single_trajectory(true_alpha, true_logsigma, T=1000):
#     true_model = models.GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)
#     logging.info(f'True model: {true_model.params}')
            
#     X, Y, _ = true_model.sample(T)

#     res = fit(X,Y)
#     conv_dict = {'T':T, 'true_alpha': true_alpha, 'true_log_sigma': true_logsigma, 'res':res}
#     return conv_dict

def plot_landscape(X, Y, R=None, model_kwargs={}, seed=0, N_particles=10000, session_indices=[]):
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
            R=R, session_indices=session_indices, 
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
    # grid_points = jnp.array([grid_points[:,0], jnp.ones_like(grid_points[:,0])*-2.0, grid_points[:,1]]).T
    # grid_points = pairs_to_triples(grid_points)
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
    
    # ax.set_xlabel('alpha')
    # ax.set_ylabel('log_sigma')
    plt.savefig('landscape.png', dpi=300)
    return vals

# def predict_score(
#         params,
#         X_hist: list[jnp.ndarray], Y_hist: list[jnp.ndarray],
#         X_pred: list[jnp.ndarray], Y_pred: list[jnp.ndarray],
#         R_hist: list[jnp.ndarray] = None, session_indices: list[list]=[[]],
#         N_particles: int=10000, 
#         model_kwargs: dict={}, seed: int=0, initial_params: Optional[dict] = None,
#         ):
#     '''
#     Do filtering to obtain last weights, then sample weights trajectories from there and compare
#     sampled decisions with true decisions.
#     '''
#     T = len(X_hist) + len(X_pred)
#     day_flags = models.set_day_flags(T, session_indices)
#     model = models.GLMLearn(seed=seed, **model_kwargs)
#     key = jax.random.PRNGKey(seed)

#     # Step 1: filtering to obtain last weights
#     # (_, Zs_filt), _ = samplers.bootstrap_filter(
#     #     N_particles, 
#     #     X_hist, Y_hist, 
#     #     model, 
#     #     R=R_hist, session_indices=session_indices, 
#     #     return_history=False, verbose=True,
#     #     )
#     Zs_filt, _ = model.posterior_samples(
#         key, params, X_hist, Y_hist, R=R_hist, session_indices=session_indices,
#         N_particles=N_particles, return_=True, posterior_type='filt',
#     )
#     w = Zs_filt.mean(0)[-1]

#     # Step 2: sample weights trajectories
#     Ys, Ws = [], []
#     for t in range(len(X_pred)):
#         key, decision_key, update_key = jax.random.split(key, 3)

#         # Decision
#         y = model.decision(w, X_pred[t], key=decision_key)

#         if model.learning_rule == 'reinforce':
#             r = models.reward(X_pred[t,1]-X_pred[t,0], y)
#         elif model.learning_rule == 'policy_gradient':
#             r = models.effective_reward(X_pred[t,1]-X_pred[t,0])

#         # Update
#         w = model.update_weights(w, params=params, x=X_pred[t], y=y, key=update_key, r=r, return_noise=False, day_flag=day_flags[t])

#         Ys.append(y)
#         Ws.append(w)

#     # Compute score 
#     score = jnp.mean(jnp.array(Ys) == jnp.array(Y_pred))
#     return score


# true_alpha = 0.05
# true_logsigma = -1.0
# true_model = models.GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)

# T = 1000
# X, Y, _ = true_model.sample(T)

# res = fit(X,Y)
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
    parser.add_argument("--learning-rule", type=str, default="policy_gradient", choices=["policy_gradient", "reinforce"], 
                        help="Learning rule for the model.")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Seed for random number generator")
    parser.add_argument("--N-particles", "-N", type=int, default=10000, 
                        help="Number of particles for SMC.")

    # Parse the command-line arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")
    model = models.GLMLearn(seed=args.seed, learning_rule=args.learning_rule)

    try:
        # Override subject index with SLURM array task ID
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        idx = args.subject_id

    N_particles = 10000
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    logging.info(f'Number of particles: {args.N_particles}. Seed: {seed}.')

    # parallel_fit()

    # Load IBL data
    df = pd.read_csv('./data/ibl_learning_processed.csv')
    regressors = ['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']
    
    lab = args.lab
    lab_df = df[df['lab']==lab]
    logging.info(f"Loaded IBL '{lab}' data.")

    lab_subjects = np.unique(lab_df['subject'].values)
    fit_subjects = lab_subjects if args.all_subjects else [lab_subjects[idx]]

    X, Y, sess_ind = [], [], []
    for subject in fit_subjects:
        X_sub, Y_sub, sess_ind_sub = ibl.get_mouse_design(lab_df, subject=subject, regressors=regressors)
        X.append(X_sub)
        Y.append(Y_sub)
        sess_ind.append(jnp.array(sess_ind_sub))

    # if not args.all_subjects:
    #     logging.info(f"Loaded subject '{fit_subjects[0]}' data. T={len(Y[0])}.")
    logging.info(f"Regressors: {['bias'] + regressors}")

    # else:
    #     try:
    #         # Override subject index with SLURM array task ID
    #         idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    #     except KeyError:
    #         idx = args.subject_id
    #     subject = lab_subjects[idx]
    #     X, Y, sess_ind = ibl.get_mouse_design(
    #         lab_df, subject=subject, 
    #         regressors=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']
    #         )
    #     logging.info(f"Loaded subject '{subject}' data. T={len(Y)}.")
    #     X = [X]
    #     Y = [Y]
    #     sess_ind = [sess_ind]
    
    # if args.all_subjects:
    #     raise KeyboardInterrupt

    # log_sigma, log_sigma_day, alpha = [-3.023e+00, -4.213e-01,  0.02]
    # model = models.GLMLearn(seed=0, 
    #                         log_sigma=log_sigma, 
    #                         log_sigma_day=log_sigma_day, 
    #                         alpha=alpha, 
    #                         learning_rule=args.learning_rule)
    # X, Y, true_Z, true_sample_noises = model.sample(T=5000, key=key)

    if args.learning_rule == 'reinforce':
        R = [jnp.asarray(models.reward(X_sub[:,1]-X_sub[:,0], Y_sub)) for X_sub, Y_sub in zip(X, Y)]
    elif args.learning_rule == 'policy_gradient':
        R = [jnp.asarray(models.effective_reward(X_sub[:,1]-X_sub[:,0])) for X_sub, Y_sub in zip(X, Y)]

    # Format initial condition 
    T_start = 0
    T = len(X[0])

    if T_start > 0:
        #! Hard coded
        # rec_params = {'log_alpha': jnp.array([-1.8199868 , -7.543031  , -4.49377   , -1.1154436 , -0.75007665], dtype=jnp.float32), 'log_sigma': 4.6615686, 'log_sigma_day': 0.09152629}
        
        # 0 : 2000
        rec_params_2000 = {'log_alpha': jnp.array([-0.8376785 , -0.9085973 , -0.90883476, -0.90600324, -0.86133814],      dtype=jnp.float32), 'log_sigma': -2.5854561, 'log_sigma_day': -0.49938372}

        # 2000 : 4000
        rec_params_20004000 = {'log_alpha': jnp.array([-1.3416517 , -1.414401  , -1.392542  , -1.4192349 , -0.63191247], dtype=jnp.float32), 'log_sigma': -2.0918186, 'log_sigma_day': -0.2745998 }

        # 4000 : 6000
        rec_params_40006000 = {'log_alpha': jnp.array([-1.6269428, -1.5575756, -1.6327392, -0.8933461, -0.6904897],      dtype=jnp.float32), 'log_sigma':-1.892599, 'log_sigma_day': 0.42951426}

        # 6000 : 8000
        rec_params_60008000 = {'log_alpha': jnp.array([-1.7647194, -1.6801634, -1.7461658, -1.0577695, -1.0274256],      dtype=jnp.float32), 'log_sigma': -1.7569481, 'log_sigma_day': 0.5056702}
    else:
        rec_params = None

    if T_start > 0:

        # First, 0 : 2000
        (Zs, _), _ = samplers.bootstrap_filter(
            N_particles,
            X[0][:2000], Y[0][:2000],
            models.GLMLearn(seed=seed, **rec_params_2000, learning_rule=args.learning_rule),
            R=R[0][:2000], session_indices=sess_ind[0],
            return_history=True, verbose=True
            )
        z_0 = jnp.mean(Zs[:,-1,:], axis=0)

        # Second, 2000 : 4000
        (Zs, _), _ = samplers.bootstrap_filter(
            N_particles,
            X[0][2000:4000], Y[0][2000:4000],
            models.GLMLearn(seed=seed, **rec_params_20004000, learning_rule=args.learning_rule, z_0=z_0),
            R=R[0][2000:4000], session_indices=sess_ind[0],
            return_history=True, verbose=True
            )
        z_0 = jnp.mean(Zs[:,-1,:], axis=0)

        # Third, 4000 : 6000
        (Zs, _), _ = samplers.bootstrap_filter(
            N_particles,
            X[0][4000:6000], Y[0][4000:6000],
            models.GLMLearn(seed=seed, **rec_params_40006000, learning_rule=args.learning_rule, z_0=z_0),
            R=R[0][4000:6000], session_indices=sess_ind[0],
            return_history=True, verbose=True
            )
        z_0 = jnp.mean(Zs[:,-1,:], axis=0)

        # Third, 6000 : 8000
        (Zs, _), _ = samplers.bootstrap_filter(
            N_particles,
            X[0][6000:8000], Y[0][6000:8000],
            models.GLMLearn(seed=seed, **rec_params_60008000, learning_rule=args.learning_rule, z_0=z_0),
            R=R[0][6000:8000], session_indices=sess_ind[0],
            return_history=True, verbose=True
            )
        z_0 = jnp.mean(Zs[:,-1,:], axis=0)

        logging.info(f"Starting weights at t={T_start} : {jnp.mean(Zs[:,-1,:], axis=0)} pm {jnp.std(Zs[:,-1,:], axis=0)}")
    else:
        z_0 = 0.

    # if T_start > 0:
    X_train = [_X[T_start:T_start+T] for _X in X]
    Y_train = [_Y[T_start:T_start+T] for _Y in Y]
    R_train = [_R[T_start:T_start+T] for _R in R]
    session_indices_train = [sess - T_start for sess in sess_ind]
    # else:
    #     X_train = [_X[:T] for _X in X]
    #     Y_train = [_Y[:T] for _Y in Y]
    #     R_train = [_R[:T] for _R in R]
    #     session_indices_train = sess_ind

    if not args.all_subjects:
        if T_start >0:
            logging.info(f"Loaded subject '{fit_subjects[0]}' data. Training data: T={T}, start: {T_start}.")
        else:
            logging.info(f"Loaded subject '{fit_subjects[0]}' data. T={len(Y_train[0])}.")

    # Start fit
    logging.info('Starting fitting.')
    gridsearch_params = find_initial(
        X_train, Y_train, R=R_train, session_indices=session_indices_train,
        N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule}, seed=seed
        )
    # gridsearch_params = [-3.0, -1.0, -1.0]
    initial_params = ParamsGLMLearn(
        log_sigma=gridsearch_params[1], 
        log_sigma_day=gridsearch_params[2], 
        log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1)
        )#._asdict()
    logging.info(f'Initial params: {initial_params}')

    # logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")

    # res = fit_optax(
    #     X_train, Y_train, R=R_train, session_indices=session_indices_train, 
    #     N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
    #     initial_params=initial_params
    #     )
    params, result_dict = fit_EM(
        X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
        N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
        initial_params=initial_params,
        posterior_type='smooth',
        n_iters=20,
    )
    # res = fit_optax(X, Y, R=R, session_indices=sess_ind, 
    #                 N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule})
    # res = fit_optax(X[:], Y[:], R=R[:], session_indices=sess_ind, 
    #                 N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule})
    # params_array = fit_EM(X[:], Y[:], R=R[:], n_iters=200, model_kwargs={'learning_rule':args.learning_rule}, 
    #                       seed=seed, N_particles=N_particles, session_indices=sess_ind)
    # params_array = fit_LaplaceEM(X[:1000], Y[:1000], R=R[:1000], n_iters=200, model_kwargs={'learning_rule':args.learning_rule}, 
    #                       seed=seed, N_particles=N_particles, session_indices=sess_ind)
    
    # vals = plot_landscape(X[:], Y[:], R=R[:], N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule},
    #                       session_indices=sess_ind)


    # Prediction score
    logging.info("Evaluating model prediction.")
    # params = {
    #     'log_alpha': jnp.array([-3.967024 , -6.1233783, -5.3488607, -4.191792 , -3.1133502]), 
    #     'log_sigma': -3.2669349, 'log_sigma_day': -0.82476026
    #     }
    
    for partition in jnp.linspace(0.1, 0.8, 10):
        split = int(len(X_train[0]) * partition)
        score = model.score_predict(
            key, params,
            X_train[0][:split], Y_train[0][:split],
            X_train[0][split:split+100], Y_train[0][split:split+100],
            R_hist=R_train[0][:split], session_indices=session_indices_train[0],
            N_particles=1000,
            )
        logging.info(f"T100-Prediction score with {partition*100:2.0f}% history: {score:.2f}")
