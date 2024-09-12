
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
# from scipy.optimize import minimize
import jaxopt
from itertools import combinations
import psutil

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
    model = models.GLMLearn(**model_kwargs)

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

        # E-step: Get posterior samples
        Zs, lik = get_samples(params)
        assert Zs.shape == (N_particles, T, D+1)
        logging.info(f'[{iter_id} - E] Lik: {lik:5.2f}')

        for split in [0.2, 0.5, 0.8]: 
            logging.info(f"[{iter_id}] T500 Prediction score with {split} history: {prediction_score(params, split):.4f}")\
        
        # M-step: Optimize log joint. Use optax for optimization
        opt_state = m_step_optimizer.init(params)
        for m_iter_id in range(m_step_iters):

            # Grad of vmap over particles
            val, grad = jax.value_and_grad(log_joint)(params)

            # Update
            updates, opt_state = m_step_optimizer.update(grad, opt_state, loss=val)
            params = optax.apply_updates(params, updates)
            if m_iter_id % 50 == 0:
                logging.info(f'[{iter_id} - M - {m_iter_id}] joint: {-val:.2f}, log_alpha grad norm = {jnp.linalg.norm(grad.log_alpha):.2e}')

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
        X: list[jnp.ndarray], Y: list[jnp.ndarray], R: list[jnp.ndarray] = None, session_indices: list[list]=[[]],
        N_particles: int=1000, 
        model_kwargs: dict={}, seed: int=0,
        vmap=True, metric='mll', return_top_n=1
        ) -> jnp.ndarray:
    # Define grid
    log_sigmas = jnp.linspace(-8.0, -3.0, 8)
    log_sigma_days = jnp.linspace(-5.0, -2.0, 6)
    log_alphas = jnp.linspace(-8.0, -3.0, 8)
    grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_sigma_days), axis=-1).reshape(-1,3)
    logging.info(f'Finding initialization from grid of {grid_points.shape} points.')

    def array_to_params(params_array):
        log_alpha, log_sigma, log_sigma_day = params_array
        if isinstance(model, models.TimeVarGLMLearn):
            params = models.ParamsTimeVarGLMLearn(
                beta_0=1.0, log_alpha=log_alpha, log_sigma_0=log_sigma, log_sigma=log_sigma, log_sigma_day=log_sigma_day
                )
        elif isinstance(model, models.GLMHMMLearn):
            params = models.ParamsGLMHMMLearn(A=jnp.array([[0.98, 0.02], [0.02, 0.98]]), log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day)
        else:
            params = ParamsGLMLearn(log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day)
        return params
    
    # Define model and key
    # model = models.GLMLearn(**model_kwargs)
    # model = models.TimeVarGLMLearn(**model_kwargs)
    key = jax.random.PRNGKey(seed)
    
    logging.info(f"Memory available: {psutil.virtual_memory().available/1e6} MB")

    # Define loss
    def neg_MLL(params_array):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        log_alpha, log_sigma, log_sigma_day = params_array
        params = array_to_params(params_array)

        # Compute marginal log-likelihood with SMC, summing over subjects
        loglik = 0.
        for _X, _Y, _R, _sess_ind in zip(X, Y, R, session_indices):
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
        # params = ParamsGLMLearn(log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day)
        # params = models.ParamsTimeVarGLMLearn(alpha_0=log_alpha, log_sigma_0=log_sigma, log_sigma=log_sigma, log_sigma_day=log_sigma_day)

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

    # logging.info(f'Best params: [log_alpha, log_sigma, log_sigma_day]={best_param_arrays[0]}, neg MLL: {vals[sorted_indices[0]]:.2f}, ')
    return best_params


def fit_optax(
        model,
        X: jnp.ndarray, Y: jnp.ndarray, R: jnp.ndarray = None, session_indices: list=[],
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
    # model = models.GLMLearn(**model_kwargs)
    # model = models.TimeVarGLMLearn(**model_kwargs)
    key = jax.random.PRNGKey(seed)
    logging.info(f'Fitting model with direct gradient optimization of MLL.')

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
    @jax.jit
    def neg_MLL(params):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        # Might need to add a loop over subjects
        loglik = model.marginal_log_likelihood(
            key, params,
            X, Y, R=R, session_indices=session_indices,
            N_particles=N_particles,
            verbose=False,
            )
        return -loglik
    
    # @jax.jit
    # def next_step_predict_score(params):
    #     scores, _ = model.next_step_prediction_score(
    #         key, params,
    #         X, Y, R=R, session_indices=session_indices,
    #         N_particles=N_particles,
    #         verbose=False,
    #     )
    #     return -jnp.mean(scores)
    
    # Initialize
    params = initial_params
    opt_state = optimizer.init(params)  # Obtain the `opt_state` that contains statistics for the optimizer.

    def evaluate(params):
        t2_score, _ = model.two_step_prediction_score(
            key, params,
            X, Y, R=R, session_indices=session_indices,
            N_particles=N_particles,
        )
        return t2_score.mean()

   
    # Optimize
    for i in range(n_iters):
        # with Pool(10) as pool:
        #     grads = pool.map(neg_evidence_value_and_grad, zip(X, Y, R, session_indices))
        # print(grads)
        val, grad = jax.value_and_grad(neg_MLL)(params)
        logging.info(f'[{i}] lik: {-val:.2f}, params: {params}')
        logging.info(f'[{i}] \t2step score: {evaluate(params):.4f}')
        # val, grad = jax.value_and_grad(next_step_predict_score)(params)

        # updates, opt_state = optimizer.update(grad, opt_state)
        updates, opt_state = optimizer.update(grad, opt_state, loss=val)
        params = optax.apply_updates(params, updates)

        # if opt_state[1].lr != lr_scale:
        #     lr_scale = opt_state[1].lr
        #     logging.info(f'[{i}] ReduceLROnPlateau: Learning rate: {lr_scale * learning_rate:.2e}')

    return params, opt_state

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
    plt.savefig('landscape.png', dpi=300)
    return vals

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
    parser.add_argument("-N", "--N-particles", type=int, default=1000, 
                        help="Number of particles for SMC.")

    # Parse the command-line arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    # model = models.GLMLearn(learning_rule=args.learning_rule)
    model = models.TimeVarGLMLearn(learning_rule=args.learning_rule)
    # model = models.GLMHMMLearn(learning_rule=args.learning_rule)
    logging.info(f"Model: {model}")

    try:
        # Override subject index with SLURM array task ID
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        idx = args.subject_id

    seed = args.seed
    key = jax.random.PRNGKey(seed)
    logging.info(f'Number of particles: {args.N_particles}. Seed: {seed}.')

    # parallel_fit()

    # Load IBL data
    df = pd.read_csv('./data/ibl_learning_processed.csv')
    regressors = ['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'] #  'stimIntensity', 
    # model.latent_dim = len(regressors) + 1
    
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

    if args.learning_rule == 'reinforce':
        R = [jnp.asarray(models.reward(X_sub[:,1]-X_sub[:,0], Y_sub)) for X_sub, Y_sub in zip(X, Y)]
    elif args.learning_rule == 'policy_gradient':
        R = [jnp.asarray(models.effective_reward(X_sub[:,1]-X_sub[:,0])) for X_sub, Y_sub in zip(X, Y)]

    # Format initial condition 
    T_start = 0
    T = len(X[0])
    
    X_train = [_X[T_start:T_start+T] for _X in X]
    Y_train = [_Y[T_start:T_start+T] for _Y in Y]
    R_train = [_R[T_start:T_start+T] for _R in R]
    session_indices_train = [sess - T_start for sess in sess_ind]

    if not args.all_subjects:
        if T_start > 0:
            logging.info(f"Loaded subject '{fit_subjects[0]}' data. Training data: T={T}, start: {T_start}.")
        else:
            logging.info(f"Loaded subject '{fit_subjects[0]}' data. T={len(Y_train[0])}.")

    # Start fit
    logging.info('Starting fitting.')
    top_gridsearch_params = find_initial(
        model,
        X_train, Y_train, R=R_train, session_indices=session_indices_train,
        N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule}, seed=seed,
        return_top_n=1
        )
    initial_params = top_gridsearch_params[0]
    # gridsearch_params = [-5.142857, -3.,       -2.      ]

    # if isinstance(model, models.TimeVarGLMLearn):
    #     initial_params = models.ParamsTimeVarGLMLearn(
    #             beta_0=1.0,
    #             log_alpha=gridsearch_params[0], 
    #             log_sigma_0=gridsearch_params[1], 
    #             log_sigma=gridsearch_params[1], 
    #             log_sigma_day=gridsearch_params[2]
    #             )
    # elif isinstance(model, models.GLMHMMLearn):
    #     initial_params = models.ParamsGLMHMMLearn(
    #         A = jnp.array([[0.99, 0.01], [0.01, 0.99]]),
    #         log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1),
    #         log_sigma=gridsearch_params[1], 
    #         log_sigma_day=gridsearch_params[2], 
    #     )
    # else:
    #     initial_params = ParamsGLMLearn(
    #         log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1),
    #         log_sigma=gridsearch_params[1], 
    #         log_sigma_day=gridsearch_params[2], 
    #         )

    log_alpha_samples, _ = model.alpha_mcmc(
            key, initial_params, 
            X[0], Y[0], R=R[0], session_indices=session_indices_train[0],
            N_particles=args.N_particles, n_iters=100, N_samples=100,
            verbose=True, proposal_scale=0.5
            )
    
    BURN_IN = 50
    log_alpha_samples = log_alpha_samples[-BURN_IN:].reshape(-1, log_alpha_samples.shape[-1])
    ci = jnp.percentile(log_alpha_samples, q=jnp.array([2.5, 97.5]), axis=0).T
    log_geo_means = jnp.mean(log_alpha_samples, axis=0)
    log_ari_means = jax.scipy.special.logsumexp(log_alpha_samples, axis=0) - jnp.log(log_alpha_samples.shape[0])
    log_alpha_meds = jnp.median(log_alpha_samples, axis=0)

    logging.info("Posterior alpha:")
    for i in range(len(log_geo_means)):
        logging.info(f"alpha_{i}: log geo mean = {log_geo_means[i]:.2f}, log ari mean = {log_ari_means[i]:.2f}, med = {log_alpha_meds[i]:.2f}, CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")

    logging.info("Evaluating model prediction.")
    for eval_log_alpha in [log_geo_means, log_ari_means, log_alpha_meds]:
        # eval_params = ParamsGLMLearn(
        #     log_sigma=gridsearch_params[1], 
        #     log_sigma_day=gridsearch_params[2], 
        #     log_alpha=eval_log_alpha
        #     )
        
        # gridsearch_params = [-3.0, -5.0, -4.0]
        # if isinstance(model, models.TimeVarGLMLearn):
        #     eval_params = models.ParamsTimeVarGLMLearn(
        #         beta_0=1.0,
        #         log_alpha=jnp.array(eval_log_alpha).mean(), 
        #         log_sigma_0=gridsearch_params[1], 
        #         log_sigma=gridsearch_params[1], 
        #         log_sigma_day=gridsearch_params[2]
        #         )
        # else:
        #     eval_params = ParamsGLMLearn(
        #         log_sigma=gridsearch_params[1], 
        #         log_sigma_day=gridsearch_params[2], 
        #         log_alpha=eval_log_alpha
        #         )
        eval_params = initial_params._replace(log_alpha=eval_log_alpha)
        logging.info(f'Params: {eval_params}')

        scores, loglik = model.next_step_prediction_score(
            key, eval_params, 
            X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
            N_particles=args.N_particles, verbose=True
            )
        logging.info(f"Total log-likelihood: {loglik:.2f}, likelihood per trial: {jnp.exp(loglik/len(Y)):.2f}")
        logging.info(f"Prediction score: {scores.mean():.2f} +/- {scores.std():.2f}, quartiles: {jnp.percentile(scores, jnp.array([25, 50, 75]))}")
        
        scores2, _ = model.two_step_prediction_score(
            key, eval_params, 
            X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
            N_particles=args.N_particles, verbose=True
            )
        logging.info(f"Two-step prediction score: {scores2.mean():.2f} +/- {scores2.std():.2f}, quartiles: {jnp.percentile(scores2, jnp.array([25, 50, 75]))}")

    sys.exit()
    # top_gridsearch_params = [[-5.0, -5.0, -4.0]]
    
    for i, gridsearch_params in enumerate(top_gridsearch_params):
        logging.info(f"Grid point #{i}: {gridsearch_params}")
    
        # # gridsearch_params = [-3.0, -5.0, -4.0]
        # if isinstance(model, models.TimeVarGLMLearn):
        #     initial_params = models.ParamsTimeVarGLMLearn(
        #         beta_0=1.0,
        #         log_alpha=gridsearch_params[0], 
        #         log_sigma_0=gridsearch_params[1], 
        #         log_sigma=gridsearch_params[1], 
        #         log_sigma_day=gridsearch_params[2]
        #         )
        # else:
        #     initial_params = ParamsGLMLearn(
        #         log_sigma=gridsearch_params[1], 
        #         log_sigma_day=gridsearch_params[2], 
        #         log_alpha=gridsearch_params[0] * jnp.ones(X[0].shape[1]+1)
        #         )
        logging.info(f'Initial params: {initial_params}')

        logging.info("Evaluating model prediction.")
        scores, loglik = model.next_step_prediction_score(
            key, initial_params, 
            X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
            N_particles=args.N_particles, verbose=True
            )
        
        scores2, _ = model.two_step_prediction_score(
            key, initial_params, 
            X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
            N_particles=args.N_particles, verbose=True
            )

        logging.info(f"Prediction score: {scores.mean():.2f} +/- {scores.std():.2f}, quartiles: {jnp.percentile(scores, jnp.array([25, 50, 75]))}")
        logging.info(f"Two-step prediction score: {scores2.mean():.2f} +/- {scores2.std():.2f}, quartiles: {jnp.percentile(scores2, jnp.array([25, 50, 75]))}")
        logging.info(f"Total log-likelihood: {loglik:.2f}, likelihood per trial: {jnp.exp(loglik/len(Y)):.2f}")

        # logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")
        # params, res = fit_optax(
        #     X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
        #     N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
        #     initial_params=initial_params, n_iters=200
        #     )
        # params, result_dict = fit_EM(
        #     X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0], 
        #     N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
        #     initial_params=initial_params,
        #     posterior_type='smooth',
        #     n_iters=20,
        # )
        # params, _ = fit_MLL(
        #     key, X_train[0], Y_train[0], R=R_train[0], session_indices=session_indices_train[0],
        #     N_particles=args.N_particles, model_kwargs={'learning_rule':args.learning_rule, 'z_0': z_0},
        #     initial_params=initial_params,
        #     )


    # # Prediction score
    # logging.info("Evaluating model prediction.")
    # # params = {
    # #     'log_alpha': jnp.array([-3.967024 , -6.1233783, -5.3488607, -4.191792 , -3.1133502]), 
    # #     'log_sigma': -3.2669349, 'log_sigma_day': -0.82476026
    # #     }
    
    # for partition in jnp.linspace(0.1, 0.8, 10):
    #     split = int(len(X_train[0]) * partition)
    #     score = model.score_predict(
    #         key, params,
    #         X_train[0][:split], Y_train[0][:split],
    #         X_train[0][split:split+100], Y_train[0][split:split+100],
    #         R_hist=R_train[0][:split], session_indices=session_indices_train[0],
    #         N_particles=1000,
    #         )
    #     logging.info(f"T100-Prediction score with {partition*100:2.0f}% history: {score:.2f}")
