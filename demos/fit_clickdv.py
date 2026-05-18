'''
Fit LearningRules models on ClickDV (Brody-lab Poisson-clicks) data.

Mirrors `demos/fit_demo.py` but plugs in `ClickDVSingleTrajectoryLoader`.
'''
import argparse
import logging
import os
import sys

sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

import fit_utils
import models
from clickdv import ClickDVSingleTrajectoryLoader
from clickdv import format_reward_function as clickdv_reward_fn
from fit import find_initial, fit_optax, posterior_mcmc

logging.basicConfig(level=logging.INFO,
                    format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='LearningRules fit on ClickDV data.')
    parser.add_argument('--subject', type=str, default='A324',
                        help='ClickDV subject id (e.g. A324, A327, C211).')
    parser.add_argument('--learning-rule', type=str, default='reinforce',
                        choices=['policy_gradient', 'reinforce', 'regression_gradient',
                                 'max_ent', 'max_ent_MC'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-N', '--N-particles', type=int, default=1000)
    parser.add_argument('--model-class', type=str, default='GLMLearn')
    parser.add_argument('--vector-alpha', action='store_true')
    parser.add_argument('--lapse', action='store_true')
    parser.add_argument('--regressors', type=str, nargs='+',
                        default=['clickLogRatio', 'previousChoice', 'previousRewarded'],
                        help='First regressor must be a signed stimulus '
                             '(clickDiff/clickLogRatio/clickAsymmetry).')
    parser.add_argument('--modulator', type=str, default='lr', choices=['lr', 'baseline'])
    parser.add_argument('--n-iters', type=int, default=200,
                        help='fit_optax iterations.')
    parser.add_argument('--mcmc-iters', type=int, default=100)
    parser.add_argument('--mcmc-samples', type=int, default=500)
    parser.add_argument('--skip-mcmc', action='store_true')
    parser.add_argument('--use-stimulus-reward', action='store_true',
                        help='Compute reward from sign(stimulus) vs choice (IBL convention) '
                             'instead of using the recorded reward column.')
    args = parser.parse_args()
    logger.info(f'Arguments: {args}')

    model = fit_utils.load_model(args)

    seed = args.seed
    key = jax.random.PRNGKey(seed)
    logger.info(f'Number of particles: {args.N_particles}. Seed: {seed}.')

    loader = ClickDVSingleTrajectoryLoader({
        'subject': args.subject,
        'regressors': args.regressors,
        'learning_rule': args.learning_rule,
        'seed': seed,
        'use_recorded_reward': not args.use_stimulus_reward,
    })
    train_trajectory = loader.load_train_data()
    trajectory, held_out_trials = loader.load_test_data()

    train_trajectories = [train_trajectory]
    full_trajectories = [trajectory]
    logger.info(f'Loaded {args.subject} train data, T={len(train_trajectory.X)}, '
                f'held out {len(held_out_trials)} trials.')

    # Wire model latent dims / reward fn (same dispatch as fit_demo.py).
    if isinstance(model, models.TimeVarGLMLearn):
        beta_dim = len(args.regressors) + 1 if args.vector_alpha else 1
        model.beta_dim = beta_dim
        model.latent_dim = len(args.regressors) + beta_dim + 1
        model.reward_func = clickdv_reward_fn(args.regressors, learning_rule=args.learning_rule)
    elif isinstance(model, models.AC):
        beta_dim = len(args.regressors) + 1
        model.beta_dim = beta_dim
        model.latent_dim = len(args.regressors) + beta_dim + 1
        model.reward_func = clickdv_reward_fn(args.regressors, learning_rule='reinforce')
    elif isinstance(model, models.GLMHMMLearn):
        model.latent_dim = len(args.regressors) + 1
    elif isinstance(model, models.TimeVarRVBF):
        model.modulator = args.modulator
        beta_dim = 1 if model.modulator == 'lr' else len(args.regressors) + 1
        model.beta_dim = beta_dim
        model.latent_dim = len(args.regressors) + beta_dim + 1
        model.reward_func = clickdv_reward_fn(args.regressors, learning_rule='reinforce')
    elif isinstance(model, models.HRL):
        model.latent_dim = len(args.regressors) + 1 + 2
        model.reward_func = clickdv_reward_fn(args.regressors, learning_rule='reinforce')
    else:
        model.latent_dim = len(args.regressors) + 1
        model.reward_func = clickdv_reward_fn(args.regressors, learning_rule=args.learning_rule)

    logger.info(f'Model: {model}')
    logger.info('-' * 80)

    top_gridsearch_params = find_initial(
        model, train_trajectories,
        N_particles=args.N_particles, seed=seed,
        return_top_n=-1, vector_alpha=args.vector_alpha, vmap=True,
    )
    initial_params = top_gridsearch_params[0]

    logger.info('-' * 80)
    logger.info('Step 2: MLL gradient ascent.')
    params, (best_val_per_subject, best_params_per_subject), _ = fit_optax(
        model, train_trajectories,
        N_particles=args.N_particles,
        initial_params=initial_params, n_iters=args.n_iters,
    )
    logger.info(f'Final params: {params}')
    for sub in range(len(train_trajectories)):
        best_val, sub_params = best_val_per_subject[sub], best_params_per_subject[sub]
        logger.info(f'Subject {sub}, Best: MLL: {best_val:.2f}, params: {sub_params}')

    if args.skip_mcmc:
        return

    logger.info('-' * 80)
    logger.info('Step 3: Posterior MCMC.')
    _, log_lik_samples, posterior_samples, _ = posterior_mcmc(
        model, key, params, train_trajectories,
        N_particles=args.N_particles,
        n_iters=args.mcmc_iters, N_samples=args.mcmc_samples,
        verbose=True, proposal_scale=0.5,
    )
    BURN_IN = 50
    logger.info(f'Marginal log-likelihood estimate: {log_lik_samples[-BURN_IN:].mean():.2f}')

    posterior_samples = posterior_samples[-BURN_IN:].reshape(-1, posterior_samples.shape[-1])
    ci = jnp.percentile(posterior_samples, q=jnp.array([2.5, 97.5]), axis=0).T
    posterior_means = posterior_samples.mean(axis=0)
    posterior_meds = jnp.median(posterior_samples, axis=0)
    for i in range(len(posterior_means)):
        logger.info(f'param_{i}: mean = {posterior_means[i]:.2f}, '
                    f'med = {posterior_meds[i]:.2f}, '
                    f'CI = [{ci[i,0]:.2f}, {ci[i,1]:.2f}]')


if __name__ == '__main__':
    main()
