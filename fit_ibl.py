
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
# import os
# os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from parameters import ParamsGLMLearn, ParameterProperties
import os, psutil
process = psutil.Process()
from multiprocessing import Pool

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

    # Compute the matrix rank to check for linear independence
    rank = np.linalg.matrix_rank(points_array)

    # If the rank is equal to N, the points form an N-dimensional space
    return rank == len(points_array[0])

def fit_EM(X, Y, R=None, n_iters=200, model_kwargs={}, seed=0, N_particles=1000, session_indices=[]):
    T, D = X.shape

    current_params = [-3.0, -1.0, 0.0]
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
        logging.info(f'[{iter_id}] Lik: {lik:5.2f}')

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
            # logging.info(f'Log-joint value: {val:5.2f}, params: {params_array}')
            return - val
        
        def callback_f(intermediate_result):
            val = intermediate_result.fun
            params = intermediate_result.x
            logging.info(f'Log-joint: {-val:5.2f}, params: {ParamsGLMLearn(*params)}')

        res = minimize(
            neg_log_joint,
            x0 = current_params,
            method='Nelder-Mead',
            tol=1e-3,
            options={'disp':True, 'return_all':True},
            callback=callback_f,
            bounds=[(-6,2), (-6,2), (0,1)]
            )
        current_params = res.x
    return params_array
            

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
            return_history=False, verbose=False
            )
        # _, lik = samplers.bootstrap_filter(N_particles, X=X, Y=Y, R=R, return_history=False, verbose=True)
        # logging.info(f'Likelihood: {lik:5.2f}, memory: {process.memory_info().rss/1e6:5.2f} MB')
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
        assert is_ndimensional_space(initial_simplex)
        
        logging.info('Initial simplex:')
        for i, vertex in enumerate(initial_simplex):
            logging.info(f'\tparams: {vertex}, neg MLL: {vals[sorted_indices[i]]:.2f}')
        return initial_simplex

    # initial_simplex = find_initial_simplex()
    initial_simplex = jnp.array([
        [-5.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [-3.0, -5.0, 0.0],
        [-3.0, -1.0, 0.5]
    ])
    logging.info(f'Initial simplex: {initial_simplex}')

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
        x0 = initial_simplex[0],
        method='L-BFGS-B',
        jac = jax.grad(neg_MLL),
        bounds=[(-6,2), (-6,2), (0,1)],
        options={'disp':101}, # iprint number
        )
    return res

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
        results = pool.map(fit, zip(Xs, Ys, Rs))

    for i, res in enumerate(results):
        logging.info(f'Subject {lab_subjects[i]}: {res.x} at {res.nit} iterations. Likelihood: {-res.fun}')
    
    return results

def fit_single_trajectory_search():
    T = 10000
    alpha_values = [0.0, 0.01, 0.1, 0.5]
    sigma_values = [-5.0, -3.0, -1.0]

    convergence_dicts = []
    for true_alpha in alpha_values:
        for true_logsigma in sigma_values:
            conv_dict = fit_single_trajectory(true_alpha, true_logsigma, T=T)
            print('-'*50)
            print(conv_dict)
            print('-'*50)

            convergence_dicts.append(conv_dict)

    with open(f'saves/convergence_dicts_w_initial_simplex_T{T}.pkl', 'wb') as f:
        pickle.dump(convergence_dicts, f)

def fit_single_trajectory(true_alpha, true_logsigma, T=1000):
    true_model = models.GLMLearn(dynamics_logscale=true_logsigma, alpha=true_alpha)
    logging.info(f'True model: {true_model.params}')
            
    X, Y, _ = true_model.sample(T)

    res = fit(X,Y)
    conv_dict = {'T':T, 'true_alpha': true_alpha, 'true_log_sigma': true_logsigma, 'res':res}
    return conv_dict


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
    parser.add_argument("--subject_id", type=int, default=0, 
                        help="Subject index in the lab data.")
    parser.add_argument("--learning_rule", type=str, default="policy_gradient", choices=["policy_gradient", "reinforce"], 
                        help="Learning rule for the model.")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Seed for random number generator")

    # Parse the command-line arguments
    args = parser.parse_args()

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

    try:
        # Override subject index with SLURM array task ID
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        idx = args.subject_id
    subject = lab_subjects[idx]

    X, Y, sess_ind = ibl.get_mouse_design(
        lab_df, subject=subject, 
        regressors=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']
        )
        
    if args.learning_rule == 'reinforce':
        R = jnp.asarray(models.reward(X[:,1]-X[:,0], Y))
    elif args.learning_rule == 'policy_gradient':
        R = jnp.asarray(models.effective_reward(X[:,1]-X[:,0]))
    logging.info(f"Loaded subject '{subject}' data. T={len(Y)}.")

    # Start fit
    # logging.info('Starting fitting. Model: GLM with learning rule: REINFORCE.')
    logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule}' learning rule")
    res = fit_MLL(X, Y, R=R, session_indices=sess_ind, model_kwargs={'learning_rule':args.learning_rule})
    # params_array = fit_EM(X[:], Y[:], R=R[:], n_iters=200, model_kwargs={'learning_rule':args.learning_rule}, 
    #                       seed=seed, N_particles=N_particles, session_indices=sess_ind)

    logging.info(f'Optimization result: {res}')
    logging.info(f'Final result: {res.x} at {res.nit} iterations. Likelihood: {-res.fun}')

    final_dict = {'subject':subject, 'learning_rule': args.learning_rule,
                  'log_sigma': res.x[0], 'log_sigma_day': res.x[1],
                  'alpha':res.x[2], 'likelihood': -res.fun}
    logging.info(final_dict)

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