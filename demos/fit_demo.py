import pandas as pd
from tqdm import tqdm
import argparse

import sys
sys.path.append('../')
HOMEDIR = '/home/vg0233/PillowLab/LearningRules'
sys.path.append(HOMEDIR)
import fit_utils
import models

import sys
import jax 
import jax.numpy as jnp
import jax.random as random
import psutil

# import os
# os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import parameters
from parameters import Trajectory
import os, psutil
process = psutil.Process()

from fit import find_initial, fit_optax, posterior_mcmc
from ibl import hold_out_trials, format_reward_function

class DemoSingleTrajectoryLoader():
    def __init__(self, params):
        """
        Initialize the IBL data handler.
        
        Parameters:
            params : dict
                Dictionary containing the following keys:
                - 'lab': str, name of the lab.
                - 'subject_id': int, index of the subject.
                - 'regressors': list, list of regressors to be used.
                - 'learning_rule': str, learning rule to be applied.
                - 'seed': int, random seed for reproducibility.
            DOWNLOAD : bool, optional
                If True, download the IBL data from the ONE API. If False, load the data from a local file. Default is False.
        Attributes:
            data : dict
                Dictionary to store the processed data, including:
                - 'session_indices': jnp.array, session indices.
                - 'trajectory': Trajectory, trajectory data.
                - 'held_out_trials': jnp.array, indices of held-out trials.
                - 'train_trajectory': Trajectory, training trajectory data.
        Raises:
            FileNotFoundError
                If the local data file is not found and DOWNLOAD is set to False.
            AssertionError
                If the specified lab or subject index is not found in the data.
        """
        regressors = params['regressors']
        seed = params['seed']

        self.data = {}

        # Sample random design matrix data
        keys = random.split(random.PRNGKey(seed), 5)

        T = 1000  # Number of trials
        n_regressors = len(regressors)
        X = random.normal(keys[0], shape=(T, n_regressors))
        Y = random.choice(keys[1], jnp.array([0, 1]), shape=(T,))
        session_indices = jnp.sort(random.choice(keys[2], jnp.arange(T), shape=(T//10,), replace=False))
        R = random.choice(keys[3], jnp.array([0.0, 1.0]), shape=(T,))
        
        day_flags = models.set_day_flags(len(X), jnp.array(session_indices))
        self.data['session_indices'] = jnp.array(session_indices, dtype=jnp.int32)
        self.data['trajectory']  = Trajectory(X, Y, R, day_flags)


        # Held-out trials
        n_trials = len(X)
        held_out_trials = jax.random.choice(keys[4], n_trials, shape=(int(n_trials/10),), replace=False)
        held_out_trials = jnp.sort(held_out_trials)
        self.data['held_out_trials']  = held_out_trials

        self.data['train_trajectory'] = hold_out_trials(X, Y, R, day_flags, held_out_trials)
        
    def load_data(self):
        return self.data
        
    def load_train_data(self):
        return self.data['train_trajectory']
    
    def load_test_data(self):
        '''load data needed for testing'''
        return self.data['trajectory'], self.data['held_out_trials']

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Argument parser for IBL fitting.")

    # Add arguments
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
                        help="Model class to use for fitting.")
    parser.add_argument("--vector-alpha", action='store_true',
                        help="Use vector alpha for GLM.")
    parser.add_argument("--lapse", action='store_true',
                        help="lapse for timevar GLM")
    parser.add_argument("--regressors", type=str, nargs='+', default=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'],
                        help="Regressors to use for the model.")
    parser.add_argument("--modulator", type=str, default='lr', choices=['lr', 'baseline'],
                        help="Modulator for the TimeVarRVBF model.")
    
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

    # Load demo data
    regressors = args.regressors

    train_trajectories, full_trajectories = [], []
    all_held_out_trials = []
    
    loader_params = {
        'regressors': regressors,
        'seed': seed,
    }
    loader = DemoSingleTrajectoryLoader(loader_params)
    train_trajectory = loader.load_train_data()

    train_trajectories.append(train_trajectory)
    logging.info(f"Loaded train data, T={len(train_trajectory.X)}. Params: {loader_params}")

    trajectory, held_out_trials = loader.load_test_data()
    full_trajectories.append(trajectory)
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
        model.reward_func = format_reward_function(regressors, learning_rule='reinforce')
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
        model.reward_func = format_reward_function(regressors, learning_rule='reinforce')
        logging.info(f"Model latent dim: {model.latent_dim}, beta dim: {beta_dim}")
    elif isinstance(model, models.HRL):
        model.latent_dim = len(regressors) + 1 + 2
        model.reward_func = format_reward_function(regressors, learning_rule='reinforce')
    else:
        model.latent_dim = len(regressors) + 1
        model.reward_func = format_reward_function(regressors, learning_rule=args.learning_rule)

    logging.info(f"Model: {model}")
    logging.info('Starting fitting.')

    # ================================================================================
    # Step 1: Grid search

    logging.info("-"*80)

    top_gridsearch_params = find_initial(
        model,
        train_trajectories,
        N_particles=args.N_particles, seed=seed,
        return_top_n=-1,
        vector_alpha=args.vector_alpha, vmap=True
        )
    initial_params = top_gridsearch_params[0]

    # ================================================================================
    # Step 2: Optimization of MLL with gradient ascent

    logging.info("-"*80)
    logging.info('Step 2: Optimization of MLL with gradient ascent.')

    params, (best_val_per_subject, best_params_per_subject), res = fit_optax(
        model,
        train_trajectories,
        N_particles=args.N_particles,
        initial_params=initial_params, n_iters=200,
        )
    logging.info(f'Final params: {params}')
    for sub in range(len(train_trajectories)):
        best_val, sub_params = best_val_per_subject[sub], best_params_per_subject[sub]
        logging.info(f'Subject {sub}, Best: MLL: {best_val:.2f}, params: {sub_params}')

    # ================================================================================
    # Step 3: Parameter posterior sampling with MCMC

    logging.info("-"*80)
    logging.info('Step 3: Parameter posterior sampling with MCMC.')
    all_accepts, log_lik_samples, posterior_samples, _ = posterior_mcmc(
            model, key, params, 
            train_trajectories,
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

    logging.info("Posterior alpha:")
    for i in range(len(posterior_means)):
        logging.info(f"alpha_{i}: mean = {posterior_means[i]:.2f}, med = {posterior_meds[i]:.2f}, CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")