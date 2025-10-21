
import pandas as pd
import jax 
import jax.numpy as jnp
import argparse
from tqdm import tqdm
from copy import deepcopy
import time

import os
import sys
sys.path.append('../')
HOMEDIR = '/home/vg0233/PillowLab/LearningRules'
sys.path.append(HOMEDIR)
import pickle
import fit_utils
import ibl
import models, parameters
from fit_ibl import posterior_mcmc

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(args, regressors):
    model = fit_utils.load_model(args)

    if isinstance(model, models.TimeVarGLMLearn):
        beta_dim = len(regressors) + 1 if args.vector_alpha else 1
        model.beta_dim = beta_dim
        model.latent_dim = len(regressors) + beta_dim + 1
        model.reward_func = ibl.format_reward_function(regressors, learning_rule=None)
        logging.info(f"Model latent dim: {model.latent_dim}, beta dim: {beta_dim}")
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
    elif isinstance(model, models.AC):
        beta_dim = len(regressors) + 1
        model.beta_dim = beta_dim
        model.latent_dim = len(regressors) + beta_dim + 1
        model.reward_func = ibl.format_reward_function(regressors, learning_rule='reinforce')
        logging.info(f"Model latent dim: {model.latent_dim}, beta dim: {beta_dim}")
    elif isinstance(model, models.GLMHMMLearn):
        model.latent_dim = len(regressors) + 1
    elif isinstance(model, models.GLMBaseLearn):
        beta_dim = len(regressors) + 1 if args.vector_alpha else 1
        model.latent_dim = beta_dim + len(regressors) + 1
    elif isinstance(model, models.HRL):
        model.latent_dim = len(regressors) + 1 + 2
        model.reward_func = ibl.format_reward_function(regressors, learning_rule='reinforce')
    else:
        model.latent_dim = len(regressors) + 1
        model.reward_func = ibl.format_reward_function(regressors, learning_rule=args.learning_rule)
    return model

def compute_posterior(args, model, params, X, Y, R, day_flags):
    #%% Compute posterior mean
    key = jax.random.PRNGKey(args.posterior_seed)
    def posterior_samples(_key):
        post_weights, log_lik = model.posterior_samples(
            _key, params, 
            X, Y, R, day_flags,
            N_particles=args.N_particles, return_history=True, LAG=True, verbose=args.verbose
            )
        return post_weights, log_lik

    n_iters = 5
    if args.loop_post_samples:
        post_weights, log_liks = [], []
        subkeys = jax.random.split(key, n_iters)
        for i in range(n_iters):
            post_weight, log_lik = posterior_samples(subkeys[i])
            post_weights.append(post_weight)
            log_liks.append(log_lik)
        post_weights = jnp.stack(post_weights)
        log_liks = jnp.stack(log_liks)
    else:
        post_weights, log_liks = jax.vmap(posterior_samples)(jax.random.split(key, n_iters))
    # post_weights = post_weights.mean(0)
    # log_lik = jnp.mean(log_liks)

    # post_dict = {'post_weights': post_weights, 'LL': log_lik}
    # pickle.dump(post_dict, open(SAVEDIR+f'posterior_{model}_T{T}.pkl', 'wb'))

    jnp.save(MODELDIR+f'posterior_weights_mean_N{args.N_particles}x{n_iters}_ps{args.posterior_seed}.npy', post_weights.mean(0).mean(0))

    learning_components = []
    noise_components = []
    min_noise_iter = -1
    min_noise_val = jnp.inf
    for i in range(n_iters):
        logging.info(f"Total log-likelihood post ({i}): {log_liks[i]:.2f}, likelihood per trial: {jnp.exp(log_liks[i]/len(Y)):.3f}")

        #%% Compute learning and noise components

        # if isinstance(model, models.TimeVarGLMLearn):
        #     beta_post, w_post = model.split_latent(post_weights)
        #     beta_post_mean = jnp.tile(beta_post.mean(0), (len(w_post), 1, 1))
        #     w_post_bmean = model.merge_latent(beta_post_mean, w_post)

        learning_updates = jax.vmap(lambda t: model.update_weights(
            key, post_weights[i,:,t], x=X[t], y=Y[t], r=R[t], day_flag=day_flags[t], params=params, return_learning_signal=True,
            )[1])(jnp.arange(T))
        learning_updates = learning_updates.mean(1)

        if hasattr(model, 'split_latent'):
            w_post = model.split_latent(post_weights[i])[-1]
        else:
            w_post = post_weights[i]

        learning_component = w_post.mean(0)[0] + jnp.cumsum(learning_updates, axis=0)
        noise_component = w_post.mean(0)[1:] - learning_component[:-1]

        learning_components.append(learning_component)
        noise_components.append(noise_component)
        norm_noise_component = jnp.linalg.norm(noise_component)

        if norm_noise_component < min_noise_val:
            min_noise_val = norm_noise_component
            min_noise_iter = i
        logging.info(f"Norm of noise_component: {norm_noise_component:.2f}, per trial: {norm_noise_component/len(Y):.3f}")

    learning_component = jnp.stack(learning_components).mean(0)
    noise_component = jnp.stack(noise_components).mean(0)
    jnp.save(MODELDIR+f'learning_component_N{args.N_particles}x{n_iters}_ps{args.posterior_seed}.npy', learning_component)
    jnp.save(MODELDIR+f'noise_component_N{args.N_particles}x{n_iters}_ps{args.posterior_seed}.npy', noise_component)

    min_learning_component = learning_components[min_noise_iter]
    min_noise_component = noise_components[min_noise_iter]
    jnp.save(MODELDIR+f'min_learning_component_N{args.N_particles}x{n_iters}_ps{args.posterior_seed}.npy', min_learning_component)
    jnp.save(MODELDIR+f'min_noise_component_N{args.N_particles}x{n_iters}_ps{args.posterior_seed}.npy', min_noise_component)
    return

def modeldir(args, model):
    MODELDIR = HOMEDIR + f'/postprocessing/posterior/{args.lab}/{args.subject_id}/{model}/{args.protocol}/'
    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)
    return MODELDIR

def generate_prior_trajectory(
        key, model, params, X, Y, R, day_flags, 
        use_emissions=True, use_noise=True, posterior_latents=None, N_samples=1, correct_bias=True):
    '''
    Generate a latent prior trajectory, with and without emissions.
    '''
    # key = jax.random.PRNGKey(args.posterior_seed)
    T = len(X)
    
    if posterior_latents is not None:
        # if isinstance(model, (models.TimeVarGLMLearn, models.AC, models.TimeVarRVBF)):
        if hasattr(model, 'split_latent'):
            assert posterior_latents.shape[0] == len(X), "Posterior latents shape mismatch."
            beta_post = model.split_latent(posterior_latents)[:-1]
        else:
            logging.warning("Posterior latents unnecessary, no other latents than w.")
        #     post_weights, _ = model.posterior_samples(
        #         key, params, X, Y, R, day_flags, 
        #         N_particles=args.N_particles, return_history=True, verbose=args.verbose
        #         )
        # else:
    
    _params = deepcopy(params)
    # if not use_noise:
    #     _params = _params._replace(log_sigma=-10, log_sigma_day=-10)

    # Initial conditions
    z_t = model.sample_initial(key, params, N_samples, d=len(regressors)) #! global regressors
    W_out = jnp.zeros((T, len(regressors)+1)) #! global regressors

    # if isinstance(model, (models.TimeVarGLMLearn, models.AC, models.TimeVarRVBF)):
    if hasattr(model, 'split_latent'):
        w_t = model.split_latent(z_t)[-1]
    else:
        w_t = z_t
    W_out = W_out.at[0].set(w_t.mean(0))

    if correct_bias and len(regressors) == 4:
        correct_bias_flags = jnp.cumprod(jnp.abs(X[:,1] - X[:,0]) >= 0.9).astype(bool)
    else:
        correct_bias_flags = jnp.zeros(T, dtype=bool)

    if hasattr(model, 'split_latent'):
        if posterior_latents is not None:
            if isinstance(beta_post, tuple):
                def keep_posterior_latents(t, z_next):
                    w_next = model.split_latent(z_next)[-1]
                    beta_next_list = []
                    for beta_elem in beta_post:
                        beta_next_list.append(jnp.tile(beta_elem[t+1], (N_samples, 1)))
                    z_next = model.merge_latent(*beta_next_list, w_next)
                    return w_next, z_next
            else:
                assert len(beta_post) == len(X), "Posterior latents shape mismatch."
                def keep_posterior_latents(t, z_next):
                    w_next = model.split_latent(z_next)[-1]
                    beta_next = jnp.tile(beta_post[t+1], (N_samples, 1))
                    z_next = model.merge_latent(beta_next, w_next)
                    return w_next, z_next
    else:
        def keep_posterior_latents(t, z_next):
            w_next = z_next.copy()
            return w_next, z_next

    def scan_fn(carry, inputs):
        t, z_carry, W_out = carry
        X_t, Y_t, R_t, next_day_flag, correct_bias_flag, key = inputs
        
        # Update the latents p(z_{t+1} | z_t, y_t, x_t)
        if use_emissions:
            # Use the true, data emissions Y_t and corresponding rewards R_t
            # Update the latents p(z_{t+1} | z_t, y_{t,data}, x_t)
            z_next = model.update_weights(
                        key, z_carry, x=X_t, y=Y_t, r=R_t, day_flag=next_day_flag, 
                        params=_params, return_learning_signal=False, correct_bias=correct_bias_flag,
                        )
        else:
            # Sample emissions y_{t,sample} ~ p(y_t | z_t, x_t)
            Y_prior_t = model.decision(key, z_carry, X_t)
            # jax.debug.print('z_carry.shape = {}, Y_prior_t.shape = {}, Y_prior_t.std = {}', z_carry.shape, Y_prior_t.shape, Y_prior_t.std())
            R_prior_t = jnp.where(Y_prior_t == Y_t, R_t, 0.)
            subkeys = jax.random.split(key, N_samples)

            # Update the latents p(z_{t+1} | z_t, y_{t,sample}, x_t) for each particle
            z_next = jax.vmap(
                lambda subkey, z, y, r: model.update_weights(
                        subkey, z, x=X_t, y=y, r=r, day_flag=next_day_flag, 
                        params=_params, return_learning_signal=False, correct_bias=correct_bias_flag,
                        )
            )(subkeys, z_carry, Y_prior_t, R_prior_t)
            # jax.debug.print('z_next.shape = {}, z_next.std = {}', z_next.shape, z_next.std())
        
        # Keep the posterior mean for the non-weight latents. 
        # This step has no effect if beta_dim = 0 
        w_next, z_next = keep_posterior_latents(t, z_next)
        W_out = W_out.at[t+1].set(w_next.mean(0))
        return (t+1, z_next, W_out), None
    
    # Scan over time
    keys = jax.random.split(key, T-1)
    inputs = jax.vmap(lambda t: (X[t], Y[t], R[t], day_flags[t], correct_bias_flags[t], keys[t]))(jnp.arange(T-1))
    carry = (0, z_t, W_out)
    W_out = jax.lax.scan(scan_fn, carry, inputs)[0][-1]

    return W_out

def generate_all_prior_trajectories(args, model, params, X, Y, R, day_flags, use_posterior_latents=True):
    MODELDIR = modeldir(args, model)
    # Compute posterior latents ahead of loops
    if use_posterior_latents:
        generate_posterior_trajectories(args, model, params, X, Y, R, day_flags)
        # try:
        #     posterior_latents = jnp.load(MODELDIR+f'posterior_mean_N{args.N_particles}_ps{args.posterior_seed}_scan.npy')
        # except FileNotFoundError:
        posterior_latents = jnp.load(MODELDIR+f'posterior_mean_N{args.N_particles}_ps{args.posterior_seed}_scan.npy')
        # for i in range(args.n_posterior_samples):
        #     key = jax.random.PRNGKey(args.posterior_seed+i)
        #     post_weights, _ = model.posterior_samples_scan(
        #         key, params, 
        #         X, Y, R, day_flags,
        #         N_particles=args.N_particles,
        #         )
        #     posterior_latents = post_weights.mean(0)
        #     posterior_CI = jnp.percentile(post_weights, jnp.array([2.5, 97.5]), axis=0)
        #     jnp.save(MODELDIR+f'posterior_mean_N{args.N_particles}_ps{args.posterior_seed+i}_scan.npy', posterior_latents)
        #     jnp.save(MODELDIR+f'posterior_CI_N{args.N_particles}_ps{args.posterior_seed+i}_scan.npy', posterior_CI)
    else:
        posterior_latents = None
    
    key = jax.random.PRNGKey(args.posterior_seed)
    for use_emissions in [True, False]:
        for use_noise in [True, False]:
            logging.info(f"Use emissions: {use_emissions}, use noise: {use_noise}")
            if use_noise:
                _N_samples = 1 # since we average over samples
            else:
                _N_samples = args.N_particles

            key, subkey = jax.random.split(key)
            Ws_theoretical = generate_prior_trajectory(
                subkey, model, params, X, Y, R, day_flags, 
                use_emissions=use_emissions, use_noise=use_noise, posterior_latents=posterior_latents, N_samples=_N_samples
                )
    
            file_name = 'prior_useY{}_noise{}_postZ{}_N{}_ps{}_scan.npy'.format(
                use_emissions, use_noise, use_posterior_latents, args.N_particles, args.posterior_seed
                )
            jnp.save(MODELDIR+file_name, Ws_theoretical)
            logging.info(f"Last weights: {Ws_theoretical[-1]}")
    return

def generate_posterior_trajectories(args, model, params, X, Y, R, day_flags) -> None:
    for i in range(args.n_posterior_samples):
        logging.info(f"Posterior sample {i+1}/{args.n_posterior_samples}")
        key = jax.random.PRNGKey(args.posterior_seed+i)
        if hasattr(model, 'posterior_samples_scan'):
            posterior_samples_func = model.posterior_samples_scan
        else:
            posterior_samples_func = model.posterior_samples
        post_weights, _ = posterior_samples_func(
            key, params, 
            X, Y, R, day_flags,
            N_particles=args.N_particles, verbose=args.verbose, # LAG=True,
            )
        posterior_latents = post_weights.mean(0)
        logging.info(f"First latents: {posterior_latents[:10].mean(0)}")
        logging.info(f"Last latents: {posterior_latents[-10:].mean(0)}")
        posterior_CI = jnp.percentile(post_weights, jnp.array([2.5, 97.5]), axis=0)
        posterior_std = jnp.std(post_weights, axis=0)
        jnp.save(MODELDIR+f'posterior_mean_N{args.N_particles}_ps{args.posterior_seed+i}_scan.npy', posterior_latents)
        jnp.save(MODELDIR+f'posterior_CI_N{args.N_particles}_ps{args.posterior_seed+i}_scan.npy', posterior_CI)
        jnp.save(MODELDIR+f'posterior_std_N{args.N_particles}_ps{args.posterior_seed+i}_scan.npy', posterior_std)

def evaluate_model(args, model, params, X, Y, R, day_flags, held_out_trials=None):
    '''
    Evaluate model on data.
    '''
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(args.posterior_seed), 3)
    
    MLL, MLLs = model.marginal_log_likelihood(
        key1, params, 
        X, Y, R=R, day_flags=day_flags,
        N_particles=args.N_particles, 
        return_logliks=True
        )
    logging.info(f"Complete trajectory 'marginal_log_likelihood': {MLL:.2f}, avg per trial: {MLLs.mean():.4f}, avg L per trial: {jnp.exp(MLLs).mean():.4f}")
    if held_out_trials is not None:
        test_MLL = MLLs[held_out_trials].sum()
        logging.info(f"Held out trials 'marginal_log_likelihood': {test_MLL:.2f}, avg per trial: {MLLs[held_out_trials].mean():.4f}, avg L per trial: {jnp.exp(MLLs[held_out_trials]).mean():.4f}")

    (forward_LLs, _), forward_LL = model.forward_pass(
        key2, params,
        X, Y, R=R, day_flags=day_flags,
        N_particles=args.N_particles, predict_Y=False, return_Z=False,
    )
    logging.info(f'Forward pass (prior predictive, use Y) loglik: {forward_LL:.2f}, avg per trial: {forward_LLs.mean():.4f}, avg L per trial: {jnp.exp(forward_LLs).mean():.4f}')

    (pred_LLs, _), pred_LL = model.forward_pass(
        key3, params,
        X, Y, R=R, day_flags=day_flags,
        N_particles=args.N_particles, predict_Y=True, return_Z=False,
    )
    logging.info(f'Prediction (prior predictive, sample Y) loglik: {pred_LL:.2f}, avg per trial: {pred_LLs.mean():.4f}, avg L per trial: {jnp.exp(pred_LLs).mean():.4f}')
    return MLL, forward_LL, pred_LL

def decompose_learning_noise(args, model, params, X, Y, R, day_flags, correct_bias=True):
    '''
    Decompose learning and noise components of the posterior.
    '''
    key = jax.random.PRNGKey(args.posterior_seed)
    T, M = X.shape

    # Get posterior samples
    post_weights, _ = model.posterior_samples_scan(
            key, params, 
            X, Y, R, day_flags,
            N_particles=args.N_particles, verbose=args.verbose, #LAG=True,
            )
    Z = post_weights.mean(0)
    logging.warning(f"Considering only posterior mean. Posterior weights shape: {Z.shape}")
    jnp.save(MODELDIR+f'posterior_mean_N{args.N_particles}_ps{args.posterior_seed}_scan_2.npy', Z)
    
    # Learning component
    keys = jax.random.split(key, T)
    if correct_bias and M == 4:
        correct_bias_flags = jnp.cumprod(jnp.abs(X[:,1] - X[:,0]) >= 0.9).astype(bool) # True until stim intensity goes below 0.9 in abs
    else:
        correct_bias_flags = jnp.zeros(T, dtype=bool)

    # def bias_correction(w):
    #     bias = w[0]
    #     correction = bias * jnp.array([-1, 1, 1, 0, 0]) # [remove, add, add, 0, 0]
    #     return w + correction

    def learning_update(t):
        _, learning_signal = model.update_weights(
            keys[t], Z[t], x=X[t], y=Y[t], r=R[t], 
            day_flag=day_flags[t], correct_bias=correct_bias_flags[t],
            params=params, return_learning_signal=True,
        )

        # learning_signal_bias_corrected = learning_signal
        # jnp.where(
        #     correct_bias_flags[t],
        #     model.bias_correction(learning_signal), 
        #     learning_signal
        #     )
        return learning_signal
    
    learning_updates = jax.vmap(learning_update)(jnp.arange(T)) # shape (T, weight_dim)
    # learning_updates = learning_updates.transpose(1, 0, 2) # shape (N, T, weight_dim)
    
    # # Make sure updates are of shape (N, T, weight_dim)
    # if learning_updates.ndim == 2:
    #     learning_updates = learning_updates[:, :, None]
    
    # Get weight component of posterior samples
    # if isinstance(model, (models.TimeVarGLMLearn, models.AC, models.TimeVarRVBF)):
    if hasattr(model, 'split_latent'):
        w_post = model.split_latent(Z)[-1]
    else:
        w_post = Z
    weight_updates = w_post[1:] - w_post[:-1] # shape (T-1, weight_dim)
    noise_updates = weight_updates - learning_updates[:-1] # shape (T-1, weight_dim)
    assert jnp.allclose(noise_updates + learning_updates[:-1], weight_updates), "Weight updates do not match learning and noise updates."
    # learning_along_weight = jnp.divide(
    #     jnp.einsum('ntk,ntk->nt', learning_updates[:,:-1], weight_updates), # shape (N, T-1)
    #     # jnp.multiply(learning_updates[:,:-1], weight_updates).sum(axis=2), # shape (N, T-1)
    #     jnp.linalg.norm(weight_updates, axis=-1) # shape (N, T-1)
    #     )
    # logging.info(f"Fraction of learning along weight update: {learning_along_weight.mean():.4f}")


    # # Cosine sim from mean
    # cosine_learning_weight_mean = jnp.divide(
    #     jnp.einsum('tk,tk->t', learning_updates.mean(0)[:-1], weight_updates.mean(0)),
    #     jnp.linalg.norm(learning_updates.mean(0)[:-1], axis=-1) * jnp.linalg.norm(weight_updates.mean(0), axis=-1)
    #     )
    # logging.info(f"Cosine similarity of learning and weight update from mean: {cosine_learning_weight_mean.mean():.4f}")

    # learning_along_weight_mean = jnp.divide(
    #     jnp.einsum('tk,tk->t', learning_updates.mean(0)[:-1], weight_updates.mean(0)), # shape (N, T-1)
    #     # jnp.multiply(learning_updates[:,:-1], weight_updates).sum(axis=2), # shape (N, T-1)
    #     jnp.linalg.norm(weight_updates.mean(0), axis=-1) # shape (N, T-1)
    #     )
    # logging.info(f"Fraction of learning along weight update mean: {learning_along_weight_mean.mean():.4f}")

    # learning_along_weight_per_reg = jnp.divide(
    #     jnp.multiply(learning_updates[:,:-1], weight_updates),
    #     jnp.clip(jnp.abs(weight_updates), 1e-08)
    # ).mean(0).mean(0)
    # logging.info(f"Fraction of learning along weight update per regressor: {learning_along_weight_per_reg}")


    @jax.jit
    def cosine_similarity(a, b):
        '''Cosine similarity between two vectors.'''
        return jnp.dot(a, b) / jnp.clip(jnp.linalg.norm(a) * jnp.linalg.norm(b), 1e-10)

    @jax.jit
    def project_fraction(a, b):
        '''Fraction of the projection of a onto b, signed by orientation of a with respect to b.'''
        return jnp.dot(a, b) / jnp.dot(b, b)

    cosine_learning_weight = jax.vmap(cosine_similarity)(learning_updates[:-1], weight_updates)
    logging.info(f"Cosine similarity between learning and weight update: {cosine_learning_weight.mean():.8f}")

    learning_projections = jax.vmap(project_fraction)(learning_updates[:-1], weight_updates)
    logging.info(f"Fraction of learning along weight update: {learning_projections.mean():.8f}")
    logging.info(f"Fraction of learning along weight update, abs: {jnp.abs(learning_projections).mean():.8f}")

    # try:
    #     learning_projections_mean = jax.vmap(project_fraction)(learning_updates.mean(0)[:-1], weight_updates.mean(0))
    #     logging.info(f"Fraction of learning along weight update mean: {learning_projections_mean.mean():.4f}")
    # except Exception as e:
    #     logging.warning(f"Error in computing learning projections mean: {e}")

    noise_projections = jax.vmap(project_fraction)(noise_updates, weight_updates)
    logging.info(f"Fraction of noise along weight update: {noise_projections.mean():.8f}")
    
    # Cumulative learning component
    learning_component = w_post[0][None, :] + jnp.cumsum(learning_updates, axis=0) # shape (N, T, weight_dim)
    noise_component = w_post[1:] - learning_component[:-1]

    # # Average over particles
    # learning_component = learning_component.mean(0)
    # noise_component = noise_component.mean(0)
    logging.info(f'Noise component norm: {jnp.linalg.norm(noise_component):.4f}, per trial: {jnp.linalg.norm(noise_component)/len(Y):.4f}')

    # Report signal to noise ratio per regressor
    SNRs = []
    for i in range(learning_component.shape[1]):
        learning_component_i = learning_component[:,i]
        noise_component_i = noise_component[:,i]
        snr = jnp.linalg.norm(learning_component_i)/jnp.linalg.norm(noise_component_i)
        SNRs.append(snr.item())
        logging.info(f'SNR for regressor {i}: {snr:.4f}')
    logging.info(f'SNRs: {SNRs}')

    # Save
    jnp.save(MODELDIR+f'learning_component_N{args.N_particles}_ps{args.posterior_seed}_2.npy', learning_component)
    jnp.save(MODELDIR+f'noise_component_N{args.N_particles}_ps{args.posterior_seed}_2.npy', noise_component)
    
    return learning_component, noise_component

def get_parameter_posterior(args, model, params, X, Y, R, day_flags):
    '''
    Get parameter posterior samples.
    '''
    key = jax.random.PRNGKey(args.posterior_seed)
    logging.info(f"Getting parameter MCMC posterior samples for {model}...")
    _, log_lik_samples, posterior_samples, _ = posterior_mcmc(
            model, key, params, 
            [X], [Y], R=[R], day_flags=[day_flags],
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

    # Summary stats
    n = posterior_samples.shape[0]
    sample_mean = jnp.mean(posterior_samples, axis=0)
    sample_var = jnp.var(posterior_samples, axis=0, ddof=1)
    summary_stats_dict = {'n': n, 'sample_mean': sample_mean, 'sample_var': sample_var}
    logging.info(f"Summary stats dict: {summary_stats_dict}")
    pickle.dump(summary_stats_dict, open(MODELDIR+f'posterior_summary_stats_N{args.N_particles}_ps{args.posterior_seed}.pkl', 'wb'))

    # log_ari_means = jax.scipy.special.logsumexp(log_alpha_samples, axis=0) - jnp.log(log_alpha_samples.shape[0])

    logging.info("Posterior alpha:")
    for i in range(len(posterior_means)):
        logging.info(f"alpha_{i}: mean = {posterior_means[i]:.2f}, med = {posterior_meds[i]:.2f}, CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")
    return summary_stats_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Argument parser for IBL fitting.")

    # Add arguments
    parser.add_argument("--lab", type=str, default="wittenlab", 
                        choices=["angelakilab", "churchlandlab", "cortexlab", "danlab", "hoferlab", "mainenlab", "mrsicflogellab", "wittenlab", "zadorlab"],
                        help="IBL lab name")
    parser.add_argument("--subject-id", type=int, default=0, 
                        help="Subject index in the lab data.")
    parser.add_argument("--learning-rule", type=str, default="reinforce", choices=["policy_gradient", "reinforce", "regression_gradient"], 
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
    parser.add_argument("--posterior-seed", type=int, default=0)
    parser.add_argument("--loop-post-samples", action='store_true', default=False)
    parser.add_argument("-v", "--verbose", action='store_true', default=False, 
                        help='Verbose during posterior sampling.')
    parser.add_argument("--regressors", type=str, nargs='+', default=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'],
                        help="Regressors to use for the model.")
    parser.add_argument("--n-posterior-samples", type=int, default=1,
                        help="Number of posterior samples to compute.")
    parser.add_argument("--modulator", type=str, default='lr', choices=['lr', 'baseline'],
                        help="Modulator for the TimeVarRVBF model.")
    parser.add_argument("--protocol", type=str, default='training', choices=['training', 'no_curriculum', 'training_biasedChoiceWorld'],
                        help="Protocol of the IBL mouse data. See `ibl.py' for details.")
    parser.add_argument("--remove-Q", type=int, default=0, choices=[0, 1],
                        help="Remove forgetting parameter.")
    parser.add_argument("--remove-baseline", type=int, default=0, choices=[0, 1],
                        help="Remove baseline parameter.")

    # Parse the command-line arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")


    regressors = args.regressors # ['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'] #  'stimIntensity', 


    # Data
    loader_params = {
        'lab': args.lab,
        'subject_id': args.subject_id,
        'regressors': regressors,
        'learning_rule': args.learning_rule,
        'seed': args.seed,
        'protocol': args.protocol,
    }

    loader = ibl.IBLSingleTrajectoryLoader(loader_params)
    data = loader.load_data()
    # X, Y, R, day_flags = jnp.array(data['train_trajectory'].X), jnp.array(data['train_trajectory'].Y), data['train_trajectory'].R, data['train_trajectory'].day_flags
    X, Y, R, day_flags = jnp.array(data['trajectory'].X), jnp.array(data['trajectory'].Y), data['trajectory'].R, data['trajectory'].day_flags
    if args.model_class == "QLearning" and ('stimIntensity' not in regressors):
        X = X[:, 1] - X[:, 0]
    data_dict = {'X': X, 'Y': Y, 'R': R, 'day_flags': day_flags}

    if 'stimIntensity' in regressors:
        parsed_entries_file = './postprocessing/parsed_slurm_entries_stimIntensity.pkl'
        args.protocol = 'stimIntensity_regressors'
    # Model
    model = load_model(args, regressors)
    MODELDIR = modeldir(args, model)

    if args.protocol == 'training_biasedChoiceWorld':
        DATA_SAVEDIR = HOMEDIR + f'/data/processed/{args.lab}/{args.subject_id}/{args.protocol}'
        if not os.path.exists(DATA_SAVEDIR): 
            os.makedirs(DATA_SAVEDIR)
        pickle.dump(data_dict, open(DATA_SAVEDIR+'/data.pkl', 'wb'))
    # sys.exit()

    # train_trajectory = loader.load_train_data()
    # X_train, Y_train, R_train, day_flags_train = train_trajectory.X, train_trajectory.Y, train_trajectory.R, train_trajectory.day_flags
    # session_indices = data['session_indices']
    # logging.info(f"Loaded subject '{args.subject_id}' data. T={len(Y)}.")

    # Format data 
    T = len(X)
    X = X[:T]
    Y = Y[:T]
    R = R[:T]
    day_flags = day_flags[:T]
    if args.model_class == "QLearning":
        correct_choice = models.correct_choice(X)
    else:
        correct_choice = models.correct_choice(X[:,1]-X[:,0])

    # Load params from compiled entries table
    if 'stimIntensity' in regressors:
        parsed_entries_file = './postprocessing/parsed_slurm_entries_stimIntensity.pkl'
        args.protocol = 'stimIntensity_regressors'
    else:
        parsed_entries_file = './postprocessing/parsed_slurm_entries_2.pkl'
    entries = pd.read_pickle(parsed_entries_file)
    sub_entries = entries.query(
        f"lab == '{args.lab}' and model == '{model}' and subject_id == {args.subject_id} and protocol == '{"training" if "training" in args.protocol else args.protocol}'"
        ).iloc[0]

    params_array = jnp.array(sub_entries['params_array'])
    lengths = sub_entries['params_lengths']
    params_name = parameters.get_param_name(model.__repr__())
    params = parameters.array_to_params(params_name, params_array, lengths)

    if bool(args.remove_Q):
        try:
            params = params._replace(log_Q=-10.0 * jnp.ones_like(params.log_Q))
        except AttributeError:
            logging.warning(f"Model {model} does not have Q parameter to remove.")
    if bool(args.remove_baseline):
        try:
            params = params._replace(baseline=jnp.zeros_like(params.baseline))
        except AttributeError:
            logging.warning(f"Model {model} does not have baseline parameter to remove.")
    logging.info(f"Loaded params: {params}")

    evaluate_model(args, model, params, X, Y, R, day_flags, held_out_trials = data['held_out_trials'])

    generate_posterior_trajectories(args, model, params, X, Y, R, day_flags)

    # generate_all_prior_trajectories(args, model, params, X, Y, R, day_flags, use_posterior_latents=True)

    # decompose_learning_noise(args, model, params, X, Y, R, day_flags)

    # get_parameter_posterior(args, model, params, X, Y, R, day_flags)