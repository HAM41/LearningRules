
import pandas as pd
import jax 
import jax.numpy as jnp
import argparse

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

    if isinstance(model, models.TimeVarGLMLearn) or isinstance(model, models.AC):
        beta_dim = len(regressors) + 1 if args.vector_alpha else 1
        model.beta_dim = beta_dim
        model.latent_dim = len(regressors) + beta_dim + 1
        logging.info(f"Model latent dim: {model.latent_dim}, beta dim: {beta_dim}")
    elif isinstance(model, models.GLMHMMLearn):
        model.latent_dim = len(regressors) + 1
    elif isinstance(model, models.GLMBaseLearn):
        beta_dim = len(regressors) + 1 if args.vector_alpha else 1
        model.latent_dim = beta_dim + len(regressors) + 1
    else:
        model.latent_dim = len(regressors) + 1
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
                        choices=["GLMLearn", "TimeVarGLMLearn", "Psytrack", "GLMRegLearn", "GLMHMMLearn", "GLMInterpLearn", "QLearning", "GLMBaseLearn", "AC"],
                        help="Model class to use for fitting.")
    parser.add_argument("--vector-alpha", action='store_true',
                        help="Use vector alpha for GLM.")
    parser.add_argument("--lapse", action='store_true',
                        help="lapse for timevar GLM")
    parser.add_argument("--posterior-seed", type=int, default=0)
    parser.add_argument("--loop-post-samples", action='store_true', default=False)
    parser.add_argument("-v", "--verbose", action='store_true', default=False, 
                        help='Verbose during posterior sampling.')

    regressors = ['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded'] #  'stimIntensity', 
    
    # Parse the command-line arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    # Model
    model = load_model(args, regressors)
    MODELDIR = HOMEDIR + f'/postprocessing/posterior/{args.lab}/{args.subject_id}/{model}/'
    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)

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
    X, Y, R, day_flags = jnp.array(data['trajectory'].X), jnp.array(data['trajectory'].Y), data['trajectory'].R, data['trajectory'].day_flags
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
    correct_choice = models.correct_choice(X[:,1]-X[:,0])

    # Load params from compiled entries table
    entries = pd.read_pickle('./postprocessing/parsed_slurm_entries_wnoisecomponent.pkl')
    sub_entries = entries.query(f"lab == '{args.lab}' and model == '{model}'").iloc[args.subject_id]

    params_array = jnp.array(sub_entries['params_array'])
    lengths = sub_entries['params_lengths']
    params_name = parameters.get_param_name(model.__repr__())
    params = parameters.array_to_params(params_name, params_array, lengths)
    logging.info(f"Loaded params: {params}")

    compute_posterior(args, model, params, X, Y, R, day_flags)