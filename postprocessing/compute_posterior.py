
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

        if isinstance(model, models.TimeVarGLMLearn) or isinstance(model, models.AC):
            beta_post, w_post = model.split_latent(post_weights[i])
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
    MODELDIR = HOMEDIR + f'/postprocessing/posterior/{args.lab}/{args.subject_id}/{model}/'
    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)
    return MODELDIR

def generate_prior_trajectory(
        args, model, params, X, Y, R, day_flags, 
        use_emissions=True, use_noise=True, posterior_latents=None, N_samples=1):
    '''
    Generate a latent prior trajectory, with and without emissions.
    '''
    key = jax.random.PRNGKey(args.posterior_seed)
    
    if posterior_latents is not None:
        if isinstance(model, (models.TimeVarGLMLearn, models.AC)):
            assert posterior_latents.shape[0] == len(X), "Posterior latents shape mismatch."
            beta_post, w_post = model.split_latent(posterior_latents)
        else:
            logging.warning("Posterior latents unnecessary, no other latents than w.")
        #     post_weights, _ = model.posterior_samples(
        #         key, params, X, Y, R, day_flags, 
        #         N_particles=args.N_particles, return_history=True, verbose=args.verbose
        #         )
        # else:
    
    _params = deepcopy(params)
    if not use_noise:
        _params = _params._replace(log_sigma=-10, log_sigma_day=-10)

    # Initial conditions
    z_t = model.sample_initial(key, params, N_samples, d=len(regressors)) #! global regressors
    W_out = jnp.zeros((T, len(regressors)+1)) #! global regressors

    if isinstance(model, (models.TimeVarGLMLearn, models.AC)):
        w_t = model.split_latent(z_t)[1]
    else:
        w_t = z_t
    W_out = W_out.at[0].set(w_t.mean(0))

    # Generate dynamics
    for t in tqdm(range(T-1)):
        key, _ = jax.random.split(key)

        if use_emissions:
            Yt = Y[t]
            Rt = R[t]
        else:
            Yt = model.decision(key, z_t, X[t])
            Rt = R[t] if Yt == Y[t] else 0

        # Update weights
        z_next = model.update_weights(
            key, z_t, x=X[t], y=Yt, r=Rt, day_flag=day_flags[t], 
            params=_params, return_learning_signal=False,
            )
        z_t = z_next
        
        # Append w component
        if isinstance(model, (models.TimeVarGLMLearn, models.AC)):
            w_next = model.split_latent(z_next)[1]
            if posterior_latents is not None:
                beta_next = beta_post[:,t+1]
                beta_next = jnp.tile(beta_next, (N_samples, 1))
                z_next = model.merge_latent(beta_next, w_next)
        else:
            w_next = z_next
        W_out = W_out.at[t+1].set(w_next.mean(0))
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
        
    for use_emissions in [True, False]:
        for use_noise in [True, False]:
            logging.info(f"Use emissions: {use_emissions}, use noise: {use_noise}")
            Ws_theoretical = generate_prior_trajectory(
                args, model, params, X, Y, R, day_flags, 
                use_emissions=use_emissions, use_noise=use_noise, posterior_latents=posterior_latents, N_samples=1
                )
    
            file_name = 'prior_useY{}_noise{}_postZ{}_N{}_ps{}.npy'.format(
                use_emissions, use_noise, use_posterior_latents, args.N_particles, args.posterior_seed
                )
            jnp.save(MODELDIR+file_name, Ws_theoretical)
            logging.info(f"Last weights: {Ws_theoretical[-1]}")
    return

def generate_posterior_trajectories(args, model, params, X, Y, R, day_flags) -> None:
    for i in range(args.n_posterior_samples):
        key = jax.random.PRNGKey(args.posterior_seed+i)
        post_weights, _ = model.posterior_samples(
            key, params, 
            X, Y, R, day_flags,
            N_particles=args.N_particles, LAG=True, verbose=True,
            )
        posterior_latents = post_weights.mean(0)
        print(posterior_latents[:10])
        print(posterior_latents[-10:])
        posterior_CI = jnp.percentile(post_weights, jnp.array([2.5, 97.5]), axis=0)
        jnp.save(MODELDIR+f'posterior_mean_N{args.N_particles}_ps{args.posterior_seed+i}_scan.npy', posterior_latents)
        jnp.save(MODELDIR+f'posterior_CI_N{args.N_particles}_ps{args.posterior_seed+i}_scan.npy', posterior_CI)

def evaluate_model(args, model, params, X, Y, R, day_flags):
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

    
    # Parse the command-line arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    regressors = args.regressors # ['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'] #  'stimIntensity', 
    # Model
    model = load_model(args, regressors)
    MODELDIR = modeldir(args, model)

    # Data
    loader_params = {
        'lab': args.lab,
        'subject_id': args.subject_id,
        'regressors': regressors,
        'learning_rule': args.learning_rule,
        'seed': args.seed,
    }

    loader = ibl.IBLSingleTrajectoryLoader(loader_params)
    data = loader.load_data()
    X, Y, R, day_flags = jnp.array(data['train_trajectory'].X), jnp.array(data['train_trajectory'].Y), data['train_trajectory'].R, data['train_trajectory'].day_flags
    if args.model_class == "QLearning" and ('stimIntensity' not in regressors):
        X = X[:, 1] - X[:, 0]
    dict = {'X': X, 'Y': Y, 'R': R, 'day_flags': day_flags}
    # pickle.dump(dict, open(SAVEDIR+f'data.pkl', 'wb'))
    
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
        parsed_entries_file = './postprocessing/parsed_slurm_entries_wnoisecomponent_stimIntensity.pkl'
    else:
        parsed_entries_file = './postprocessing/parsed_slurm_entries_wnoisecomponent.pkl'
    entries = pd.read_pickle(parsed_entries_file)
    sub_entries = entries.query(
        f"lab == '{args.lab}' and model == '{model}' and subject_id == {args.subject_id}"
        ).iloc[0]

    params_array = jnp.array(sub_entries['params_array'])
    lengths = sub_entries['params_lengths']
    params_name = parameters.get_param_name(model.__repr__())
    params = parameters.array_to_params(params_name, params_array, lengths)
    logging.info(f"Loaded params: {params}")

    evaluate_model(args, model, params, X, Y, R, day_flags)
    # sys.exit()

    generate_all_prior_trajectories(args, model, params, X, Y, R, day_flags, use_posterior_latents=True)