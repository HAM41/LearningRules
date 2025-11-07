import pandas as pd
from tqdm import tqdm
import argparse

import VR
import ibl
from ibl import Trajectory
import sys
sys.path.append('../')
import fit_utils
import models

import sys
import jax 
import jax.numpy as jnp
import psutil

import tensorflow_probability.substrates.jax.distributions as tfd
from functools import partial

# import os
# os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import parameters
from parameters import ParamsGLMLearn
import os, psutil
process = psutil.Process()

from typing import List, Optional

import optax 

def fit_EM(
        model,
        trajectories: List[Trajectory],
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

    params = initial_params

    def get_samples(params, subject_id):
        Zs, lik = model.posterior_samples(
            key, params, 
            trajectories[subject_id],
            N_particles=N_particles, verbose=False, LAG=True
            )
        return Zs.mean(0), lik
    
    # @partial(jax.jit, static_argnums=(2,))
    def neg_log_joint(params, Z, subject_id):
        out = -model.log_joint(trajectories[subject_id], params=params)
        return out

    def compute_noise_component(params, Z, subject_id):
        learning_updates = jax.vmap(lambda t: model.update_weights(
            key, Z[t], 
            x=trajectories[subject_id].X[t], y=trajectories[subject_id].Y[t], r=trajectories[subject_id].R[t], day_flag=trajectories[subject_id].day_flags[t],
            params=params, return_learning_signal=True,
            )[1])(jnp.arange(len(trajectories[subject_id].X)))
        # learning_updates = learning_updates.mean(1)

        # if isinstance(model, models.TimeVarGLMLearn):
        if hasattr(model, 'split_latent'):
            w_post = model.split_latent(Z)[-1]
        else:
            w_post = Z

        learning_component = w_post[0] + jnp.cumsum(learning_updates, axis=0)
        noise_component = w_post[1:] - learning_component[:-1]
        return noise_component

    logging.info("Starting EM procedure.")
    best_val_per_subject = [-jnp.inf]*len(trajectories)
    best_params_per_subject = [params]*len(trajectories)
    for iter_id in jnp.arange(n_iters):
        if len(trajectories) > 1:
            subject_id = jax.random.randint(jax.random.PRNGKey(iter_id), shape=(), minval=0, maxval=len(trajectories)-1).item()
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

    return params, (best_val_per_subject, best_params_per_subject), opt_state

def find_initial(
        model, 
        trajectories: List[Trajectory],
        N_particles: int=1000, seed: int=0,
        vmap=True, return_top_n=1,
        vector_alpha=False,
        ) -> list:
    '''
    Perform grid search to find initial parameters for optimization.
    Returns list of `return_top_n` best parameters, in the sense of marginal log-likelihood, of type determined by the `model`.
    '''
    d = trajectories[0].X.shape[-1]
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
    elif isinstance(model, models.HRL):
        log_alphas = jnp.linspace(-8.0, -2.0, 6)
        log_sigmas = jnp.linspace(-8.0, -3.0, 5)
        log_sigma_days = jnp.linspace(-5.0, -2.0, 3)
        log_sigmas_0 = jnp.linspace(-8.0, -4.0, 5)
        q0s = jnp.linspace(-2, 2., 5)
        grid_points = jnp.stack(jnp.meshgrid(log_alphas, log_sigmas, log_sigma_days, log_sigmas_0, q0s), axis=-1).reshape(-1,5)
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
            # w0_val = jnp.zeros((d+1,)) if vector_alpha else jnp.zeros((1,))
            params = parameters.ParamsRVBF(
                log_alpha=log_alpha_val, log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_Q=log_q * jnp.ones((d+1,)), baseline=baseline_val,
            )
        elif isinstance(model, models.TimeVarRVBF):
            log_alpha, log_sigma, log_sigma_day, log_sigma_0, baseline = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            baseline_val = baseline * jnp.ones((d+1,)) if vector_alpha else baseline
            log_sigma = log_sigma * jnp.ones((d+1,)) if vector_alpha else log_sigma
            log_sigma_day = log_sigma_day * jnp.ones((d+1,)) if vector_alpha else log_sigma_day
            log_q = -5.0
            # w0_val = jnp.zeros((d+1,)) if vector_alpha else jnp.zeros((1,))
            params = parameters.ParamsTimeVarRVBF(
                log_alpha=log_alpha_val, log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_Q=log_q * jnp.ones((d+1,)), 
                baseline=baseline_val, log_sigma_0=log_sigma_0, #w0=w0_val
            )
        elif isinstance(model, models.HRL):
            log_alpha, log_sigma, log_sigma_day, log_sigma_0, q0 = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            log_sigma = log_sigma * jnp.ones((d+1,)) if vector_alpha else log_sigma
            log_sigma_day = log_sigma_day * jnp.ones((d+1,)) if vector_alpha else log_sigma_day
            baseline_val = jnp.zeros((d+1,)) if vector_alpha else 0.0
            params = parameters.ParamsHRL(
                log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_sigma_0=log_sigma_0,
                log_alpha_0=log_alpha, log_alpha_1=log_alpha_val,
                baseline_0=0., baseline_1=baseline_val,
                q0=q0,
            )
        elif isinstance(model, models.POGLMLearn):
            log_alpha, log_sigma, log_sigma_day = params_array
            log_alpha_val = log_alpha * jnp.ones((d+1,)) if vector_alpha else log_alpha
            log_sigma_m = log_sigma * jnp.ones((d+1,)) if vector_alpha else log_sigma
            params = parameters.ParamsPOGLMLearn(log_alpha=log_alpha_val, log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_sigma_m=log_sigma_m)
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
        for traj in trajectories:
            _loglik = model.marginal_log_likelihood(
                key, params, traj, N_particles=N_particles,
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

    # Take lowest
    best_param_arrays = grid_points[sorted_indices[:return_top_n]]
    best_params = [array_to_params(params_array) for params_array in best_param_arrays]
    return best_params


def fit_optax(
        model,
        trajectories: List[Trajectory],
        n_iters: int=200, N_particles: int=1000, 
        seed: int=0, initial_params: Optional[dict] = None, correct_bias=True,
        ):
    '''
    Fit model via optimization of the marginal log-likelihood (MLL) using optax. 
    MLL is computed via SMC, and summed across subjects.
    Args:
        model: model instance,
        trajectories: list of Trajectory instances, one per subject,
        n_iters: number of optimization iterations,
        model_kwargs: dictionary of model parameters,
        seed: int, random seed,
        N_particles: int, number of particles for SMC,
        session_indices: list of lists, session indices for each subject.
    Returns:
        params: optimized parameters
    '''
    key = jax.random.PRNGKey(seed)
    logging.info(f'Fitting model with optax gradient-ascent of MLL.')

    learning_rate = 0.01
    
    # scheduler = optax.exponential_decay(
    #         init_value=learning_rate,
    #         transition_steps=int(n_iters/2), # 2 cycles of decay
    #         decay_rate=0.1,
    #         transition_begin=10)
    # optimizer = optax.chain(
    #     # optax.scale_by_adam(),
    #     optax.scale_by_schedule(scheduler),
    # )
    optimizer = optax.amsgrad(learning_rate=learning_rate)
    logging.info(f'Optimizer: amsgrad. Minimize neg MLL. Learning rate: {learning_rate}.')

    # Define loss
    # @partial(jax.jit, static_argnums=(1,))
    def neg_MLL(params, subject_id):
        '''Negative marginal log-likelihood, as a function of the parameters.'''
        # # Might need to add a loop over subjects, summing over subjects
        # for X_sub, Y_sub, R_sub, day_flags_sub in zip(X, Y, R, day_flags):

        # Compute marginal log-likelihood with SMC
        loglik = 0.
        loglik_sub = model.marginal_log_likelihood(
            key, params, 
            trajectories[subject_id],
            N_particles=N_particles,
            verbose=False,
        )
        loglik += loglik_sub
        return -loglik
    
    @partial(jax.jit, static_argnums=(1,))
    def neg_MLL2(params, subject_id):
        log_lik = model.filtering_MLL(
            key, params,
            trajectories[subject_id],
            N_particles=N_particles,
            )
        return -log_lik
    
    def compute_noise_component(params, Z, subject_id):
        learning_updates = jax.vmap(lambda t: model.update_weights(
            key, Z[t], 
            x=trajectories[subject_id].X[t], y=trajectories[subject_id].Y[t], r=trajectories[subject_id].R[t], day_flag=trajectories[subject_id].day_flags[t],
            params=params, return_learning_signal=True,
            )[1])(jnp.arange(len(trajectories[subject_id].X)))
        # learning_updates = learning_updates.mean(1)

        # if isinstance(model, models.TimeVarGLMLearn) or isinstance(model, models.AC):
        if hasattr(model, 'split_latent'):
            w_post = model.split_latent(Z)[-1]
        else:
            w_post = Z

        if learning_updates.ndim == 1:
            learning_updates = learning_updates[:, jnp.newaxis]

        learning_component = w_post[0] + jnp.cumsum(learning_updates, axis=0)
        noise_component = w_post[1:] - learning_component[:-1]
        return noise_component
    
    # Initialize
    params = initial_params
    opt_state = optimizer.init(params)  # Obtain the `opt_state` that contains statistics for the optimizer.

    def evaluate(params):
        for i, traj in enumerate(trajectories):
            t2_score, _ = model.two_step_prediction_score(
                key, params,
                traj,
                N_particles=N_particles,
            )
            logging.info(f"[Subject {i}] t2 score: {t2_score.mean()}")

   
    # Optimize
    best_val_per_subject = [-jnp.inf]*len(trajectories)
    best_params_per_subject = [params]*len(trajectories)
    for i in range(n_iters):
        # Select subject for SGD
        if len(trajectories) > 1:
            subject_id = jax.random.randint(jax.random.PRNGKey(i), shape=(), minval=0, maxval=len(trajectories)-1).item()
        else:
            subject_id = 0

        # Compute grad
        neg_val, grad = jax.value_and_grad(lambda p: neg_MLL(p, subject_id))(params)
        val = -neg_val
        if val > best_val_per_subject[subject_id]:
            best_val_per_subject[subject_id] = val
            best_params_per_subject[subject_id] = params

        logging.info(f'[{i}] Subject {subject_id}, LL: {val:.4f}, L per trial: {jnp.exp(val/len(trajectories[subject_id].Y)):.4f}, params: {params}')
        updates, opt_state = optimizer.update(grad, opt_state, loss=neg_val)
        
        # Apply updates
        params = optax.apply_updates(params, updates)

        # Diagnostics
        if i % 20 == 0:

            # Forward pass liks and predicitions

            _, forward_loglik = model.forward_pass(
                key, params, trajectories[subject_id],
                N_particles=N_particles, predict_Y=False, correct_bias=correct_bias,
            )
            logging.info(f'[{i}] Forward pass loglik: {forward_loglik:.2f}, L per trial: {jnp.exp(forward_loglik/len(trajectories[subject_id].Y)):.4f}')

            _, prediction_loglik = model.forward_pass(
                key, params,
                trajectories[subject_id],
                N_particles=N_particles, predict_Y=True, correct_bias=correct_bias,
            )
            logging.info(f'[{i}] Prediction loglik: {prediction_loglik:.2f}, L per trial: {jnp.exp(prediction_loglik/len(trajectories[subject_id].Y)):.4f}')


        # if opt_state[1].lr != lr_scale:
        #     lr_scale = opt_state[1].lr
        #     logging.info(f'[{i}] ReduceLROnPlateau: Learning rate: {lr_scale * learning_rate:.2e}')

    return params, (best_val_per_subject, best_params_per_subject), opt_state

def test_held_out_sessions(
        model, params,
        held_out_sessions: list,
        trajectory: Trajectory,
        session_indices: list=None,
        N_particles: int=5000, 
        seed: int=0,
        ):
    '''
    Evaluate model on held-out sessions.
    Return p(y_{t1:t2} | y_{1:t1-1}, x_{1:t2-1}) for each held-out session of trials interval [t1:t2]
    '''
    key = jax.random.PRNGKey(seed)

    # Compute all filtering log-likelihood p(y_t | y_{1:t-1}, x_{1:t})
    _, all_log_liks = model.marginal_log_likelihood(
        key, params, trajectory,
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
        held_out_trials: jnp.ndarray,
        trajectory: Trajectory,
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
        held_out_trials,
        trajectory,
        N_particles=N_particles,
        )
    
    return test_mlls.sum(), test_mlls

def posterior_mcmc(
        model, 
        key, initial_params, 
        trajectories: List[Trajectory],
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
        for traj in trajectories:
            mll += model.marginal_log_likelihood(key, params_prop, traj, N_particles)
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
                        # choices=["GLMLearn", "TimeVarGLMLearn", "Psytrack", "GLMRegLearn", "GLMHMMLearn", "GLMInterpLearn", "QLearning", "GLMBaseLearn", 'DynamicGLMHMM', "AC", "RVBF", "TimeVarRVBF"],
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
    parser.add_argument("--initialize-at-learned", action='store_true', default=False,
                        help="Initialize at learned parameters.")

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

    # X_train, Y_train, R_train, day_flags_train = [], [], [], []
    # X_full, Y_full, R_full, day_flags_full = [], [], [], []
    train_trajectories, full_trajectories = [], []
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
        logging.info(f"Loaded train data, subject id {subject_id}. T={len(train_trajectory.X)}. Params: {loader_params}")

        if args.model_class == "QLearning":
            train_trajectories.append(train_trajectory._replace(X=train_trajectory.X[:, 1] - train_trajectory.X[:, 0]))
        else:
            train_trajectories.append(train_trajectory)

        trajectory, held_out_trials = loader.load_test_data()
        if args.model_class == "QLearning" and ('stimIntensity' not in regressors):
            full_trajectories.append(trajectory._replace(X=trajectory.X[:, 1] - trajectory.X[:, 0]))
        else:
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
    elif isinstance(model, models.HRL):
        model.latent_dim = len(regressors) + 1 + 2
        model.reward_func = ibl.format_reward_function(regressors, learning_rule='reinforce')
    else:
        model.latent_dim = len(regressors) + 1
        model.reward_func = ibl.format_reward_function(regressors, learning_rule=args.learning_rule)

    logging.info(f"Model: {model}")

    # ================================================================================
    # Step 1: Grid search

    logging.info('Starting fitting.')
    logging.info("-"*80)

    if not args.initialize_at_learned:
        logging.info('Step 1: Grid search for initialization.')
        top_gridsearch_params = find_initial(
            model,
            train_trajectories,
            N_particles=args.N_particles, seed=seed,
            return_top_n=-1,
            vector_alpha=args.vector_alpha, vmap=True
            )
        initial_params = top_gridsearch_params[0]
    else:
        logging.info('Step 1: Using learned parameters for initialization.')

        if isinstance(model, models.TimeVarRVBF):
            logging.warning("Overriding this step. Initialize with learned parameters from RVBF")
            load_model = models.RVBF()
        else:
            load_model = model

        
        # Load params from compiled entries table
        if 'stimIntensity' in regressors:
            parsed_entries_file = './postprocessing/parsed_slurm_entries_wnoisecomponent_stimIntensity.pkl'
        else:
            parsed_entries_file = './postprocessing/parsed_slurm_entries_3.pkl'
        entries = pd.read_pickle(parsed_entries_file)

        sub_entries = entries.query(
            f"lab == '{args.lab}' and model == '{load_model}' and subject_id == {args.subject_id} and protocol == '{args.protocol}'"
            ).iloc[0]

        params_array = jnp.array(sub_entries['params_array'])
        lengths = sub_entries['params_lengths']
        params_name = parameters.get_param_name(load_model.__repr__())
        loaded_params = parameters.array_to_params(params_name, params_array, lengths)

        if isinstance(model, models.TimeVarRVBF):
            initial_params = parameters.ParamsTimeVarRVBF(
                log_alpha=loaded_params.log_alpha, 
                log_sigma=loaded_params.log_sigma, 
                log_sigma_day=loaded_params.log_sigma_day,
                log_Q=loaded_params.log_Q,
                baseline=loaded_params.baseline,
                # w0=loaded_params.w0,
            )
        else:
            initial_params = loaded_params            

        logging.info(f"Loaded params: {initial_params}")

    # ================================================================================
    # Step 2: Optimization of MLL with gradient ascent

    logging.info("-"*80)
    logging.info('Step 2: Optimization of MLL with gradient ascent.')
    if args.EM:
        fit_method = fit_EM
    else:
        fit_method = fit_optax

    optax_final_params, (best_val_per_subject, best_params_per_subject), res = fit_method(
        model,
        train_trajectories,
        N_particles=args.N_particles,
        initial_params=initial_params, n_iters=20,
        )
    logging.info(f'Final params: {optax_final_params}')
    for sub in range(len(train_trajectories)):
        best_val, sub_params = best_val_per_subject[sub], best_params_per_subject[sub]
        logging.info(f'Subject {sub}, Best: MLL: {best_val:.2f}, params: {sub_params}')
    params = optax_final_params if args.all_subjects else best_params_per_subject[0]

    for i in range(3):
        key = jax.random.PRNGKey(seed+i)
        log_lik = model.marginal_log_likelihood(
            key, params, 
            train_trajectories[0],
            N_particles=args.N_particles
            )
        logging.info(f"'marginal_log_likelihood' Log-lik: {log_lik:.2f}")

        log_lik_filter = model.filtering_MLL(
            key, params, 
            train_trajectories[0],
            N_particles=args.N_particles
            )
        logging.info(f"'filter' Log-lik: {log_lik_filter:.2f}")

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
    # log_ari_means = jax.scipy.special.logsumexp(log_alpha_samples, axis=0) - jnp.log(log_alpha_samples.shape[0])

    logging.info("Posterior alpha:")
    for i in range(len(posterior_means)):
        logging.info(f"alpha_{i}: mean = {posterior_means[i]:.2f}, med = {posterior_meds[i]:.2f}, CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")

    # ================================================================================
    # Step 4 : Evaluation

    logging.info("-"*80)
    logging.info("Evaluating model (first animal).")

    lengths = parameters.params_to_array(params)[1]
    for eval_params_array in [posterior_means, posterior_meds]:
        eval_params = parameters.array_to_params(params, eval_params_array, lengths)
        logging.info(f'Params: {eval_params}')

        loglik = model.marginal_log_likelihood(
            key, eval_params, 
            full_trajectories[0],
            N_particles=args.N_particles
            )
        logging.info(f"Complete trajectory 'marginal_log_likelihood': {loglik:.2f}, per trial: {loglik/len(full_trajectories[0].Y):.4f}, L per trial: {jnp.exp(loglik/len(full_trajectories[0].Y)):.4f}")

        # loglik_filter = model.filtering_MLL(
        #     key, eval_params, 
        #     X_full[0], Y_full[0], R=R_full[0], day_flags=day_flags_full[0],
        #     N_particles=args.N_particles
        #     )
        # logging.info(f"Filtering (prior predictive) log-likelihood: {loglik_filter:.2f}, per trial: {loglik_filter/len(Y_full[0]):.4f}, L per trial: {jnp.exp(loglik_filter/len(Y_full[0])):.4f}")

        _, forward_loglik = model.forward_pass(
            key, params,
            full_trajectories[0],
            N_particles=args.N_particles, predict_Y=False,
        )
        logging.info(f'Forward pass (prior predictive, use Y) loglik: {forward_loglik:.2f}, L per trial: {jnp.exp(forward_loglik/len(full_trajectories[0].Y)):.4f}')

        _, prediction_loglik = model.forward_pass(
            key, params,
            full_trajectories[0],
            N_particles=args.N_particles, predict_Y=True
        )
        logging.info(f'Prediction (prior predictive, sample Y) loglik: {prediction_loglik:.2f}, L per trial: {jnp.exp(prediction_loglik/len(full_trajectories[0].Y)):.4f}')

        # # Test

        # test_ll, test_lls = test(
        #     model, eval_params, all_held_out_trials[0],
        #     all_trajectories[0].X, all_trajectories[0].Y, R=all_trajectories[0].R, day_flags=all_trajectories[0].day_flags,
        #     N_particles=args.N_particles
        #     )
        # logging.info(f"Test log-likelihood: {test_ll:.2f}, per trial: {test_lls.mean():.4f}")