import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from typing import Tuple, Optional, Iterable, Union
import seaborn as sns

import os
os.environ['JAX_PLATFORMS']='cpu'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
# sys.path.append('../')
sys.path.append('/home/vg0233/PillowLab/LearningRules/')
from fit import fit_optax, fit_EM, find_initial, posterior_mcmc
import parameters
from parameters import ParamsGLMLearn, ParamsQLearning, Trajectory
import models

import argparse

TRAINING_ACCURACY_THRESHOLD = 0.7

def learning_rule_to_reward_func(learning_rule):
    if learning_rule == 'reinforce' or 'QLearning':
        return models.reward
    elif learning_rule == 'policy_gradient':
        return lambda x, y: models.effective_reward(x)
    elif learning_rule == 'psytrack':
        return lambda x, y: 0. * models.reward(x, y)
    else:
        raise ValueError(f"Invalid learning rule: {learning_rule}")

if __name__=='__main__':
    # Hyperparams ------------------------------------
    parser = argparse.ArgumentParser(description="Argument parser for IBL fitting.")

    # Add arguments
    parser.add_argument("--T", type=int, default=500, 
                        help="Number of time-steps per trajectory.")
    parser.add_argument("--B", type=int, default=1,
                        help="Number of batches (trajectories / subjects).")
    parser.add_argument("--N-particles", "-N", type=int, default=1000,
                        help="Number of particles for the fitting.")
    parser.add_argument("--learning-rule-true", type=str, default="policy_gradient", choices=["policy_gradient", "reinforce", "psytrack", "QLearning"], 
                        help="Learning rule for the true model.")
    parser.add_argument("--learning-rule-fit", type=str, default="policy_gradient", choices=["policy_gradient", "reinforce", "psytrack", "QLearning"], 
                        help="Learning rule for the fitting model.")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Seed for random number generator")
    parser.add_argument("--fix-true", action='store_true',
                        help="Fix the true parameters.")
    d = 1   # number of regressors

    # Parse the command-line arguments
    args = parser.parse_args()
    results_dict = args.__dict__
    logging.info(f"Arguments: {args}")

    # Format params and instantiate models ---------------

    key = jax.random.PRNGKey(args.seed)
    logging.info(f'Number of particles: {args.N_particles}. Seed: {args.seed}. T: {args.T}, B: {args.B}. d: {d}. Learning rule true: {args.learning_rule_true}. Learning rule fit: {args.learning_rule_fit}')
    
    try:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

        learning_rules = ["policy_gradient", "reinforce", "psytrack"]
        args.learning_rule_true = learning_rules[idx % 3]
        args.learning_rule_fit = learning_rules[idx // 3]
        logging.info("SLURM_ARRAY_TASK_ID found. Using learning rules: true={}, fit={}".format(args.learning_rule_true, args.learning_rule_fit))
    except KeyError:
        logging.info("SLURM_ARRAY_TASK_ID not found. Using command-line arguments.")
        pass

    if args.learning_rule_true == 'psytrack':
        true_model = models.Psytrack(latent_dim=2)
    elif args.learning_rule_true == 'QLearning':
        true_model = models.QLearning()
    else:
        true_model = models.GLMLearn(learning_rule=args.learning_rule_true, latent_dim=2)
    true_model.reward_func = learning_rule_to_reward_func(args.learning_rule_true)

    if args.learning_rule_fit == 'psytrack':
        fit_model = models.Psytrack(latent_dim=2)
    elif args.learning_rule_fit == 'QLearning':
        fit_model = models.QLearning()
    else:
        fit_model = models.GLMLearn(learning_rule=args.learning_rule_fit, latent_dim=2)
    fit_model.reward_func = learning_rule_to_reward_func(args.learning_rule_fit)

    # Define true parameters and generate data ------------

    if args.fix_true:
        true_key = jax.random.PRNGKey(101)
        true_params = ParamsGLMLearn(log_sigma=-2, log_sigma_day=-1, log_alpha=-2.5) # closer to real IBL data
    else:
        true_key = key
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(true_key, 4)
        true_log_sigma = jax.random.uniform(subkey1, minval=-8, maxval=-3)
        true_log_sigma_day = jax.random.uniform(subkey2, minval=-5, maxval=-2)
        if args.learning_rule_true == 'QLearning':
            true_log_alpha =  jax.random.uniform(subkey3, minval=-6, maxval=-3) # jnp.log(jnp.array([0.5])) # from Lak
            percept_log_scale = jnp.log(jnp.sqrt(jnp.array([0.2]))) # from Lak
            true_log_temp = -1.0 # lower is shaper, higher lik
            # true_log_sigma = -8.0
            true_params = ParamsQLearning(
                log_sigma=true_log_sigma, log_sigma_day=true_log_sigma_day, percept_log_scale=percept_log_scale,
                log_alpha=true_log_alpha, log_temp=true_log_temp
                )
        else:
            true_log_alpha = jax.random.uniform(subkey3, minval=-6, maxval=-2, shape=(d+1,)) + 1
            true_w0 = jax.random.uniform(subkey4, shape=(2,), minval=-2, maxval=2)
            true_params = ParamsGLMLearn(log_sigma=true_log_sigma, log_sigma_day=true_log_sigma_day, log_alpha=true_log_alpha, w0=true_w0)

    logging.info(f"True parameters: {true_params}")

    true_params_array, _ = parameters.params_to_array(true_params)
    for i in range(len(true_params_array)):
        results_dict[f'true_param_{i}'] = true_params_array[i].item()

    # Create dataset
    day_flags = models.set_day_flags(args.T, jnp.arange(0, args.T, min(200, int(args.T/10))).astype(int))
    stim_values = jnp.array([-1., -0.848, -0.555, -0.303, 0., 0.303, 0.555, 0.848, 1.]) # from IBL loader, so post tanh

    X, Y, Z, R_true, R_fit = [], [], [], [], []
    trajectories_fit, trajectories_true = [], []
    for b in range(args.B):
        key, subkey = jax.random.split(key)
        X_i = jax.random.choice(subkey, stim_values, shape=(args.T,), replace=True).reshape(-1,1)

        sample_traj = Trajectory(X=X_i, Y=jnp.zeros((args.T,)), R=jnp.zeros((args.T,)), day_flags=day_flags)
        Y_i, Z_i = true_model.sample_forward(key, true_params, sample_traj)
        print(Z_i[-50:].mean(0))
        
        # Use the reward function of the fitting model
        R_fit_i = jnp.asarray(fit_model.reward_func(X_i, Y_i))
        R_true_i = jnp.asarray(true_model.reward_func(X_i, Y_i))
        R_i = models.reward(X_i, Y_i)
        if R_i[-args.T//50:].mean() < TRAINING_ACCURACY_THRESHOLD:
            raise ValueError("Mean final reward too low, resample data by changing the seed or fixing true params.")
        
        # X.append(X_i); Y.append(Y_i); Z.append(Z_i); R_fit.append(R_fit_i); R_true.append(R_true_i)
        traj_fit = Trajectory(X=X_i, Y=Y_i, R=R_fit_i, day_flags=day_flags)
        traj_true = Trajectory(X=X_i, Y=Y_i, R=R_true_i, day_flags=day_flags)

        trajectories_fit.append(traj_fit)
        trajectories_true.append(traj_true)
        
    day_flags = jnp.tile(day_flags, (args.B, 1))

    # if args.learning_rule == 'reinforce':
    #     R = [jnp.asarray(models.reward(X_sub[:,0], Y_sub)) for X_sub, Y_sub in zip(X, Y)]
    # elif args.learning_rule == 'policy_gradient':
    #     R = [jnp.asarray(models.effective_reward(X_sub[:,0])) for X_sub, Y_sub in zip(X, Y)]

    import time
    start = time.time()
    true_loglik = 0.
    total_length = 0
    for b in range(args.B):
        true_loglik_b = true_model.marginal_log_likelihood(
            key, true_params, 
            trajectories_true[b],
            N_particles=args.N_particles,
        )
        # true_loglik_b = true_model.filtering_MLL(
        #     key, true_params, 
        #     X[b], Y[b], R_true[b], day_flags[b], 
        #     N_particles=args.N_particles,
        # )
        true_loglik += true_loglik_b
        total_length += len(trajectories_true[b])
    logging.info(f"True log-likelihood (scan): {true_loglik}, per trial {true_loglik/total_length}, time: {time.time()-start}")

    results_dict['true_LL'] = true_loglik.item()

    # key, scores_key1, scores_key2 = jax.random.split(key, 3)
    # scores, log_lik = true_model.next_step_prediction_score(
    #     scores_key1, true_params, X[0], Y[0], R[0], day_flags, N_particles=args.N_particles,
    # )
    # logging.info(f"True next-step prediction score: {scores.mean():.3f} ± {scores.std():.3f}, log-lik: {log_lik}")

    # scores2t, _ = true_model.two_step_prediction_score(
    #     scores_key2, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
    # )
    # logging.info(f"True two-step prediction score: {scores2t.mean():.3f} ± {scores2t.std():.3f}")

    
    # start = time.time()
    # true_loglik = true_model.marginal_log_likelihood_1(
    #     key, true_params, X[0], Y[0], R[0], session_indices[0], N_particles=args.N_particles,
    #     verbose=True
    # )
    # logging.info(f"True log-likelihood: {true_loglik}, time: {time.time()-start}")
    # sys.exit()

    # partition = int(len(X[0]) * 0.6)
    # key, subkey = jax.random.split(key)
    # prediction_score = true_model.score_predict(
    #     key, true_params, 
    #     X_hist=X[0][:partition], Y_hist=Y[0][:partition], R_hist=R[0][:partition], session_indices=session_indices[0], 
    #     X_pred=X[0][partition:partition+100], Y_pred=Y[0][partition:partition+100],
    #     N_particles=args.N_particles,
    # )
    # logging.info(f"Prediction score: {prediction_score:.4f}")
    # sys.exit()
    # Start fitting ----------------------------------
        
    logging.info(f"Starting fitting. Model: GLM with '{args.learning_rule_fit}' learning rule")
    # res = fit_optax(X, Y, R=R, session_indices=session_indices, N_particles=N_particles, model_kwargs={'learning_rule':learning_rule})

    # Step 1: Grid search

    logging.info('Starting fitting.')
    logging.info("-"*80)
    logging.info('Step 1: Grid search for initialization.')
    top_gridsearch_params = find_initial(
        fit_model,
        trajectories_fit,
        N_particles=args.N_particles, 
        seed=args.seed,
        return_top_n=-1,
        vector_alpha=True,
        )
    initial_params = top_gridsearch_params[0]

    # # ------------------------------------------------------------

    # # Plot true against best initial params

    # # params_rec = ParamsQLearning(percept_log_scale=-1.0791758, log_sigma=-8.026407, log_sigma_day=-2.0328515, log_alpha=-3.2256474, log_temp=0.2703311)
    # Z_post, _logliks = fit_model.posterior_samples(
    #     key, initial_params, 
    #     X[0], Y[0], R=R_fit[0], day_flags=day_flags[0],
    #     N_particles=args.N_particles, LAG=True, verbose=True,
    #     correct_bias=False
    #     )
    # # print(_logliks.sum(), _logliks.mean())

    
    # Z_post_fromtrue, _logliks = true_model.posterior_samples(
    #     key, true_params, 
    #     X[0], Y[0], R=R_true[0], day_flags=day_flags[0],
    #     N_particles=args.N_particles, LAG=True, verbose=True,
    #     correct_bias=False
    #     )
    
    # # Z_filt_fromtrue, _ = true_model.filter(
    # #     key, true_params, 
    # #     X[0], Y[0], R=R_true[0], day_flags=day_flags[0],
    # #     N_particles=args.N_particles,
    # # )
    
    # Z_true = Z[0]

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(figsize=(8,5), nrows=2, constrained_layout=True);
    # axs[0].plot(Z_true[:,1], label='True', c='k')
    # axs[0].plot(Z_post.mean(axis=0)[:,1], label='Initial', c='tab:blue')
    # axs[0].plot(Z_post_fromtrue.mean(axis=0)[:,1], label='From true params', c='tab:orange')
    # axs[0].fill_between(jnp.arange(args.T), 
    #                     Z_post_fromtrue.mean(axis=0)[:,1] - Z_post_fromtrue.std(axis=0)[:,1], 
    #                     Z_post_fromtrue.mean(axis=0)[:,1] + Z_post_fromtrue.std(axis=0)[:,1], 
    #                     alpha=0.3, color='tab:orange')
    
    # # axs[0].plot(Z_filt_fromtrue.mean(axis=0)[:,0], label='Filtering from true params', c='tab:red')
    # axs[1].plot(Z_true[:,2], label='True', c='k')
    # axs[1].plot(Z_post.mean(axis=0)[:,2], label='Initial', c='tab:blue')
    # axs[1].plot(Z_post_fromtrue.mean(axis=0)[:,2], label='From true params', c='tab:orange')
    # axs[1].fill_between(jnp.arange(args.T), 
    #                     Z_post_fromtrue.mean(axis=0)[:,2] - Z_post_fromtrue.std(axis=0)[:,2], 
    #                     Z_post_fromtrue.mean(axis=0)[:,2] + Z_post_fromtrue.std(axis=0)[:,2], 
    #                     alpha=0.3, color='tab:orange')
    # # axs[1].plot(Z_filt_fromtrue.mean(axis=0)[:,1:], label='Filtering from true params', c='tab:red')
    # axs[0].set_title('Weight 1')
    # axs[1].set_title('Weight 2')
    # # plt.fill_between(jnp.arange(args.T ), Z_post.mean(axis=0) - Z_post.std(axis=0), Z_post.mean(axis=0) + Z_post.std(axis=0), alpha=0.3)
    # plt.legend()
    # plt.savefig(f'/home/vg0233/PillowLab/LearningRules/tests/figures/post_true{true_model}_fit{fit_model}_T{args.T}.png', dpi=300)
    # plt.savefig(f'/home/vg0233/PillowLab/LearningRules/tests/figures/post_true{true_model}_fit{fit_model}_T{args.T}.eps', format='eps')

    # # ------------------------------------------------------------

    # Step 2

    # if args.all_subjects:
    #     logging.info("Gradient computation is too expensive for all subjects. Skipping optimization.")
    #     params = initial_params
    # else:
    logging.info("-"*80)
    logging.info('Step 2: Optimization of MLL with gradient ascent, 200 iters.')
    optax_final_params, (best_val_per_subject, best_params_per_subject), res = fit_optax(
        fit_model,
        trajectories_fit,
        N_particles=args.N_particles,
        initial_params=initial_params, n_iters=50, correct_bias=False,
        )
    logging.info(f'Final params: {optax_final_params}')
    params = best_params_per_subject[0]
    best_params_array, lengths = parameters.params_to_array(params)
    results_dict['LL_optim'] = np.sum(best_val_per_subject)

    for i in range(3):
        key = jax.random.PRNGKey(i)
        log_lik = fit_model.marginal_log_likelihood(
            key, params, 
            trajectories_fit[0],
            N_particles=args.N_particles
            )
        # log_lik = fit_model.filtering_MLL(
        #     key, params, 
        #     X[0], Y[0], R=R_fit[0], day_flags=day_flags[0], 
        #     N_particles=args.N_particles
        #     )
        logging.info(f"Log-lik: {log_lik:.2f}")

    # Step 3

    logging.info("-"*80)
    logging.info('Step 3: Parameter posterior sampling with MCMC.')
    all_accepts, log_lik_samples, posterior_samples, (post_best_params, best_log_fx) = posterior_mcmc(
            fit_model, key, params, 
            trajectories_fit,
            N_particles=args.N_particles, n_iters=100, N_samples=500,
            verbose=True, proposal_scale=0.5
            )

    results_dict['LL_MAP'] = best_log_fx.item()
    
    
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
        logging.info(f"param_{i}: mean = {posterior_means[i]:.2f}, med = {posterior_meds[i]:.2f}, CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]")
        
        results_dict[f'param_{i}_mean'] = posterior_means[i].item()
        results_dict[f'param_{i}_med'] = posterior_meds[i].item()
        results_dict[f'param_{i}_MAP'] = best_params_array[i].item()
        results_dict[f'param_{i}_CI_0'] = ci[i,0].item()
        results_dict[f'param_{i}_CI_1'] = ci[i,1].item()

    posterior_med_params = parameters.array_to_params(params, posterior_meds, lengths)


    for eval_params_array, label in zip([posterior_means, posterior_meds], ['mean', 'med']):
        # eval_params = params._replace(log_alpha=eval_log_alpha)
        # eval_params = params.from_array(eval_params_array, lengths)
        eval_params = parameters.array_to_params(params, eval_params_array, lengths)
        logging.info(f'Params: {eval_params}')

        log_lik = 0.
        total_length = 0
        for subject_id in range(len(trajectories_fit)):
            # log_lik_sub = fit_model.marginal_log_likelihood(
            #     key, eval_params, 
            #     X[subject_id], Y[subject_id], R[subject_id], day_flags[subject_id], 
            #     N_particles=args.N_particles,
            #     )
            log_lik_sub = fit_model.filtering_MLL(
                key, eval_params, 
                trajectories_fit[subject_id],
                N_particles=args.N_particles,
                )
            log_lik += log_lik_sub
            total_length += len(trajectories_fit[subject_id])
        logging.info(f"{label}: Total log-likelihood: {log_lik:.2f}, per trial: {log_lik/total_length:.4f}")

        results_dict[f'LL_{label}'] = log_lik

    logging.info("-"*80)
    logging.info(f"Results: {results_dict}")
    sys.exit()

    # # Plot results ==================================

    # ------------------------------------------------------------

    # Plot true against best initial params

    # params_rec = ParamsQLearning(percept_log_scale=-1.0791758, log_sigma=-8.026407, log_sigma_day=-2.0328515, log_alpha=-3.2256474, log_temp=0.2703311)
    key1, key2 = jax.random.split(key, 2)

    post_mean_params = parameters.array_to_params(params, posterior_means, lengths)
    Z_post, _logliks = fit_model.posterior_samples(
        key, post_mean_params, 
        trajectories_fit[0],
        N_particles=args.N_particles, LAG=True, verbose=True,
        correct_bias=False
        )
    Z_post_CI = jnp.percentile(Z_post, q=jnp.array([2.5, 97.5]), axis=0)

    Z_post_best, _ = fit_model.posterior_samples(
        key, post_best_params, 
        trajectories_fit[0],
        N_particles=args.N_particles, LAG=True, verbose=True,
        correct_bias=False
        )
    Z_post_best_CI = jnp.percentile(Z_post_best, q=jnp.array([2.5, 97.5]), axis=0)
    
    
    Z_true = Z[0]

    if args.learning_rule_fit == 'QLearning':
        z_index_start = 1
    else:
        z_index_start = 0

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,2.5), constrained_layout=True);
    ax.plot(Z_true[:,z_index_start], label='Simulated', c='tab:blue', ls='-')
    ax.plot(Z_post.mean(axis=0)[:,z_index_start], label='Recovered', c='tab:blue', ls='--', alpha=0.5)
    ax.fill_between(jnp.arange(args.T), Z_post_CI[0,:,z_index_start], Z_post_CI[1,:,z_index_start], alpha=0.3, color='tab:blue')

    # ax.plot(Z_post_best.mean(axis=0)[:,0], label='MAP', c='tab:blue', ls='-.', alpha=0.5)
    # ax.fill_between(jnp.arange(args.T), Z_post_best_CI[0,:,0], Z_post_best_CI[1,:,0], alpha=0.3, color='k')
    
    # axs[0].plot(Z_filt_fromtrue.mean(axis=0)[:,0], label='Filtering from true params', c='tab:red')
    ax.plot(Z_true[:,z_index_start+1], label='Simulated', c='tab:orange', ls='-')
    ax.plot(Z_post.mean(axis=0)[:,z_index_start+1], label='Recovered', c='tab:orange', ls='--', alpha=0.5)
    ax.fill_between(jnp.arange(args.T), Z_post_CI[0,:,z_index_start+1], Z_post_CI[1,:,z_index_start+1], alpha=0.3, color='tab:orange')

    # ax.plot(Z_post_best.mean(axis=0)[:,1], label='MAP', c='tab:orange', ls='-.', alpha=0.5)
    # ax.fill_between(jnp.arange(args.T), Z_post_best_CI[0,:,1], Z_post_best_CI[1,:,1], alpha=0.3, color='k')

    # axs[1].fill_between(jnp.arange(args.T), 
    #                     Z_post.mean(axis=0)[:,1] - Z_post.std(axis=0)[:,1], 
    #                     Z_post.mean(axis=0)[:,1] + Z_post.std(axis=0)[:,1], 
    #                     alpha=0.3, color='tab:orange')
    # axs[1].plot(Z_filt_fromtrue.mean(axis=0)[:,1:], label='Filtering from true params', c='tab:red')
    # plt.fill_between(jnp.arange(args.T ), Z_post.mean(axis=0) - Z_post.std(axis=0), Z_post.mean(axis=0) + Z_post.std(axis=0), alpha=0.3)
    plt.legend()
    sns.despine()
    plt.savefig(f'/home/vg0233/PillowLab/LearningRules/tests/figures/post_true{true_model}_fit{fit_model}_T{args.T}_seed{args.seed}.png', dpi=300)
    plt.savefig(f'/home/vg0233/PillowLab/LearningRules/tests/figures/post_true{true_model}_fit{fit_model}_T{args.T}_seed{args.seed}.eps', format='eps')

    # ------------------------------------------------------------

    # import matplotlib.pyplot as plt

    # def posterior_samples(key, params):
    #     Z_post, lik = fit_model.posterior_samples_scan(
    #         key, params, 
    #         X[0], Y[0], R=R[0], day_flags=day_flags[0],
    #         N_particles=args.N_particles
    #         )
    #     return Z_post, lik

    # # # recovered_params = ParamsGLMLearn(log_sigma=-1.7749162, log_sigma_day=-0.02403495, log_alpha=jnp.array([-7.796574 , -4.2768683]))
    # fig, ax = plt.subplots(figsize=(10,5));

    # Z_post = jnp.zeros((args.B, args.N_particles, args.T, 2))
    # for rep in range(args.B):
    #     key, subkey1, subkey2 = jax.random.split(key, 3)
        
    #     # Z_post_rep, lik = fit_model.posterior_samples_scan(
    #     #     key, params, 
    #     #     X[rep], Y[rep], R=R[rep], day_flags=day_flags[rep],
    #     #     N_particles=args.N_particles
    #     #     )
    #     Z_post_rep, lik = fit_model.posterior_samples(
    #         key, params, 
    #         X[rep], Y[rep], R=R[rep], day_flags=day_flags[rep],
    #         N_particles=args.N_particles, LAG=True, verbose=True
    #         )
    #     Z_post = Z_post.at[rep].set(Z_post_rep)
    #     # Z_error = jnp.linalg.norm(Z_post.mean(axis=0) - Z[rep])

    # Z = jnp.asarray(Z)
    # # Only plot first batch
    # Z_post = jnp.asarray(Z_post[0])
    # for i in range(2):
    #     post_mean, post_std = Z_post.mean(axis=0)[:,i], Z_post.std(axis=0)[:,i]
    #     print(post_std)
    #     ax.plot(jnp.arange(args.T), Z.mean(0)[:,i], c='k', label='True')
    #     ax.plot(jnp.arange(args.T), post_mean, c='tab:orange', label='Recovered')

    #     if args.B > 2:
    #         ax.fill_between(jnp.arange(args.T), Z_post.mean(axis=0)[:,i] - Z_post.std(axis=0)[:,i], Z_post.mean(axis=0)[:,i] + Z_post.std(axis=0)[:,i], alpha=0.3, color='tab:blue')
    #         ax.fill_between(jnp.arange(args.T), Z.mean(0)[:,i] - Z.std(0)[:,i], Z.mean(0)[:,i] + Z.std(0)[:,i], alpha=0.3, color='tab:orange')
    #     else:
    #         # Error bar over particles
    #         ax.fill_between(jnp.arange(args.T), post_mean - post_std, post_mean + post_std, alpha=0.3, color='tab:orange')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Weight')
    # fig.suptitle(f'Posterior weights from rec params. mean-std over B.\nB={args.B}, T={args.T}, N={args.N_particles}')

    # ax.legend()
    # plt.savefig(f'/home/vg0233/PillowLab/LearningRules/tests/figures/weight_recovery/weights_B{args.B}_T{args.T}_N{args.N_particles}_wLAG.png', dpi=300)
    # plt.close()
    
