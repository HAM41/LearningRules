
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

from typing import List

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

def fit_EM(X, Y, R=None, n_iters=200, model_kwargs={}, seed=0, N_particles=1000, session_indices=[]):
    '''
    Fit model with SMC EM: posterior samples are obtained by SMC, and used to evalutate with Monte-Carlo the ELBO. 
    '''
    logging.info(f'Fitting model with SMC-EM.')
    T, D = X.shape

    current_params = [-5.0, -1.0, 0.5]

    # optimizer = optax.adam(learning_rate)
    # logging.info(f'Starting optimization. Optimizer: Adam, learning rate: {learning_rate}.')

    # # Obtain the `opt_state` that contains statistics for the optimizer.
    # params = initial_params._asdict()
    # opt_state = optimizer.init(params)

    # @jax.jit
    # def neg_log_joint(params_array, Z):
    #     val = model.log_joint(
    #         X, Y, Z=Z, R=R, 
    #         params=ParamsGLMLearn._make(params_array), #ParamsGLMLearn(**params),
    #         session_indices=session_indices
    #         )
    #     return - val

    ELBOs = [] 

    for iter_id in jnp.arange(n_iters):
        model = models.GLMLearn(seed=seed, **model_kwargs)
        model.update_params_from_array(current_params)

        # E-step: MC samples from the posterior
        (Zs, _), lik = samplers.bootstrap_filter(
            N_particles, 
            X, Y, 
            model, 
            R=R, session_indices=session_indices, 
            return_history=True, verbose=False,
            )
        assert Zs.shape == (N_particles, T, D+1)
        logging.info(f'[{iter_id} / E] Lik: {lik:5.2f}')

        Z_mean = jnp.mean(Zs, axis=0)

        # M-Step
        def neg_log_joint(params_array):
            def log_joint_per_particle(z):
                return model.log_joint(
                    X, Y, z, R=R, 
                    params = ParamsGLMLearn._make(params_array),
                    session_indices=session_indices
                    )
        
            log_joint_MCvalues = jax.vmap(log_joint_per_particle)(Zs) #! try vectorizing directly in log_joint
            val = jnp.mean(log_joint_MCvalues, axis=0)
            # logging.info(f'Log-joint: {val} pm {jnp.std(log_joint_MCvalues, axis=0)}')

            # model = models.GLMLearn(seed=seed, **params, **model_kwargs)
            # val = model.log_joint(
            #     X, Y,
            #     Z=Z_mean,
            #     R=R, 
            #     params=ParamsGLMLearn._make(params_array), #ParamsGLMLearn(**params),
            #     session_indices=session_indices
            #     )
            # logging.info(f'Log-joint value: {val:5.2f}, params: {params_array}')
            return - val

        # res = minimize(
        #     neg_log_joint,
        #     x0 = current_params,
        #     method='Nelder-Mead',
        #     tol=1e-3,
        #     options={'disp':True, 'return_all':True},
        #     callback=callback_f,
        #     bounds=[(-6,2), (-6,2), (0,1)]
        #     )

        res = minimize(
            neg_log_joint,
            x0 = current_params,
            method='L-BFGS-B',
            jac = jax.grad(neg_log_joint),
            bounds=[(-6,2), (-6,2), (0,1)],
            # options={'disp':101}, # iprint number
            )
        # print(res)


        # ELBO:
        def log_joint_per_particle(z):
            return model.log_joint(
                X, Y, z, R=R,
                session_indices=session_indices
                )
        log_joint_MCvalues = jax.vmap(log_joint_per_particle)(Zs)
        entropy_term = jnp.mean(log_joint_MCvalues, axis=0)

        ELBO = -res.fun - entropy_term + lik
        approx_lik_diff = ELBO - lik
        # if iter_id > 0:
        #     ELBO_delta = ELBO - ELBOs[-1]
        #     logging.info(f'[{iter_id} / M] Optim success: {res.success}. ELBO: {ELBO:.2f} (Δ: {ELBO_delta:.2f}), log-joint: {-res.fun:.2f}.')
        # else:
        logging.info(f'[{iter_id} / M] Optim success: {res.success}. ELBO: {ELBO:.2f}. params: {res.x}')
        ELBOs.append(ELBO)
        # logging.info(f'[{iter_id}] Params: {res.x}')

        # Update
        current_params = res.x

    return current_params, ELBOs

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

def fit_optax(X: List, Y: List, R=None, n_iters=200, model_kwargs={}, seed=0, N_particles=10000, session_indices: List=[[]]):
    learning_rate = 1e-02
    initial_params = ParamsGLMLearn(log_sigma=-5.0, log_sigma_day=-1.0, alpha=0.5)

    optimizer = optax.adam(learning_rate)
    logging.info(f'Starting optimization. Optimizer: Adam, learning rate: {learning_rate}.')

    # Obtain the `opt_state` that contains statistics for the optimizer.
    params = initial_params._asdict()
    opt_state = optimizer.init(params)

    # Define loss
    def neg_MLL(params):#, _X, _Y, _R, _sess_ind):
        print(params)
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        # Instantiate model
        model = models.GLMLearn(seed=seed, **params, **model_kwargs)#, learning_rule='reinforce')
        # print(model.params)
        # model.update_params_from_array(params)

        # Compute marginal log-likelihood with SMC
        loglik = 0.
        for _X, _Y, _R, _sess_ind in zip(X, Y, R, session_indices):
            _, _loglik = samplers.bootstrap_filter(
                N_particles, 
                _X, _Y, 
                model, 
                R=_R, session_indices=_sess_ind, 
                return_history=False, verbose=True
                )
            loglik += _loglik
        # _, lik = samplers.bootstrap_filter(N_particles, X=X, Y=Y, R=R, return_history=False, verbose=True)
        # logging.info(f'Likelihood: {lik:5.2f}, memory: {process.memory_info().rss/1e6:5.2f} MB')
        # logging.info(f'Func eval. Likelihood: {lik}, params: {params_array}')

        return -loglik
    
    for i in range(n_iters):
        # for _X, _Y, _sess_ind in zip(X,Y, session_indices):
        #     print(_X.shape, _Y.shape)
        #     func = lambda params: neg_MLL(params, _X, _Y, _sess_ind) 
        val, grad = jax.value_and_grad(neg_MLL)(params)
            # print(val, grad)

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        logging.info(f'[{i}] lik: {-val:.2f}, params: {params}')
    return params


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

    # Parse the command-line arguments
    args = parser.parse_args()

    try:
        # Override subject index with SLURM array task ID
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        idx = args.subject_id

    N_particles = 10000
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    logging.info(f'Number of particles: {N_particles}. Seed: {seed}.')

    # parallel_fit()

    # Load IBL data
    df = pd.read_csv('./data/ibl_learning_processed.csv')
    
    lab = args.lab
    lab_df = df[df['lab']==lab]
    logging.info(f"Loaded IBL '{lab}' data.")

    lab_subjects = np.unique(lab_df['subject'].values)
    fit_subjects = lab_subjects if args.all_subjects else [lab_subjects[idx]]

    X, Y, sess_ind = [], [], []
    for subject in fit_subjects:
        X_sub, Y_sub, sess_ind_sub = ibl.get_mouse_design(
            lab_df, subject=subject, 
            regressors=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']
            )
        X.append(X_sub[:1000])
        Y.append(Y_sub[:1000])
        sess_ind.append(sess_ind_sub)

    if not args.all_subjects:
        logging.info(f"Loaded subject '{fit_subjects[0]}' data. T={len(Y[0])}.")

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

    # if args.all_subjects:
    if args.learning_rule == 'reinforce':
        R = [jnp.asarray(models.reward(X_sub[:,1]-X_sub[:,0], Y_sub)) for X_sub, Y_sub in zip(X, Y)]
    elif args.learning_rule == 'policy_gradient':
        R = [jnp.asarray(models.effective_reward(X_sub[:,1]-X_sub[:,0])) for X_sub, Y_sub in zip(X, Y)]
    # else:
    #     if args.learning_rule == 'reinforce':
    #         R = jnp.asarray(models.reward(X[:,1]-X[:,0], Y))
    #     elif args.learning_rule == 'policy_gradient':
    #         R = jnp.asarray(models.effective_reward(X[:,1]-X[:,0]))
    #     R = [R]

    # Start fit
    # logging.info('Starting fitting. Model: GLM with learning rule: REINFORCE.')
    logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")
    res = fit_optax(X, Y, R=R, session_indices=sess_ind, 
                    N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule})
    # res = fit_optax(X[:], Y[:], R=R[:], session_indices=sess_ind, 
    #                 N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule})
    # params_array = fit_EM(X[:], Y[:], R=R[:], n_iters=200, model_kwargs={'learning_rule':args.learning_rule}, 
    #                       seed=seed, N_particles=N_particles, session_indices=sess_ind)
    # params_array = fit_LaplaceEM(X[:1000], Y[:1000], R=R[:1000], n_iters=200, model_kwargs={'learning_rule':args.learning_rule}, 
    #                       seed=seed, N_particles=N_particles, session_indices=sess_ind)
    
    # vals = plot_landscape(X[:], Y[:], R=R[:], N_particles=N_particles, model_kwargs={'learning_rule':args.learning_rule},
    #                       session_indices=sess_ind)


    # logging.info(f'Optimization result: {res}')
    # logging.info(f'Final result: {res.x} at {res.nit} iterations. Likelihood: {-res.fun}')

    # final_dict = {'subject':subject, 'learning_rule': args.learning_rule,
    #               'log_sigma': res.x[0], 'log_sigma_day': res.x[1],
    #               'alpha':res.x[2], 'likelihood': -res.fun}
    # logging.info(final_dict)

    # Xs.append(X)
    # Ys.append(Y)
    # Rs.append(ER)

    # # Start parallel fit
    # logging.info('Starting parallel fitting. PG learning rule.')
    # results = parallel_fit(Xs, Ys, Rs)

    # for i, res in enumerate(results):
    #     logging.info(f'Subject {lab_subjects[i]}: {res.x} at {res.nit} iterations. Likelihood: {-res.fun}')

    # subject = lab_subjects[3]
    # X, Y, sess_ind = ibl.get_mouse_design(
    #     lab_df, subject=subject, 
    #     regressors=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']
    #     )

    # R = jnp.asarray(models.reward(X[:,1]-X[:,0], Y))

    # assert len(R) == len(Y)
    # assert len(ER) == len(Y)
    # logging.info(f"Loaded subject {subject} data. T={len(Y)}.")