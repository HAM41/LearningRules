'''
Plot the posterior-mean weight trajectory from a fitted GLMLearn ClickDV run.

Reads the "Final params" line out of a fit log, re-loads the trajectory with
ClickDVSingleTrajectoryLoader, runs one SMC posterior pass, and draws each
regressor's weight (including the bias) as a function of trial, with session
boundaries marked.
'''
import argparse
import logging
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import models
import fit_utils
from clickdv import ClickDVSingleTrajectoryLoader
from parameters import ParamsGLMLearn

logging.basicConfig(level=logging.INFO,
                    format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_FLOAT = r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'


def parse_final_params_glmlearn(log_path: str) -> ParamsGLMLearn:
    '''Extract the last "Final params: ParamsGLMLearn(...)" from a fit log.'''
    with open(log_path) as f:
        lines = f.readlines()
    body = None
    for line in reversed(lines):
        if 'Final params:' in line and 'ParamsGLMLearn(' in line:
            body = line[line.index('ParamsGLMLearn(') + len('ParamsGLMLearn('):]
            break
    if body is None:
        raise ValueError(f'No "Final params: ParamsGLMLearn(...)" found in {log_path}')

    def grab_scalar(name):
        m = re.search(rf'{name}=Array\(({_FLOAT}),', body)
        if not m:
            raise ValueError(f'{name} not found in params block')
        return float(m.group(1))

    def grab_vector(name):
        m = re.search(rf'{name}=Array\(\[([^\]]*)\]', body)
        if not m:
            raise ValueError(f'{name} not found in params block')
        return jnp.array([float(x) for x in m.group(1).split(',')])

    return ParamsGLMLearn(
        log_sigma=jnp.float32(grab_scalar('log_sigma')),
        log_sigma_day=jnp.float32(grab_scalar('log_sigma_day')),
        log_alpha=jnp.float32(grab_scalar('log_alpha')),
        w_0=grab_vector('w_0'),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True,
                        help='Path to a fit_clickdv.py log file.')
    parser.add_argument('--subject', type=str, default='A324')
    parser.add_argument('--regressors', type=str, nargs='+',
                        default=['clickLogRatio', 'previousChoice', 'previousRewarded'])
    parser.add_argument('--learning-rule', type=str, default='reinforce')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-N', '--N-particles', type=int, default=500)
    parser.add_argument('--out', type=str, default='figures/clickdv_weights.png')
    args = parser.parse_args()

    params = parse_final_params_glmlearn(args.log)
    logger.info(f'Loaded params from {args.log}: {params}')

    loader = ClickDVSingleTrajectoryLoader({
        'subject': args.subject,
        'regressors': args.regressors,
        'learning_rule': args.learning_rule,
        'seed': args.seed,
    })
    full_trajectory = loader.load_data()['trajectory']
    train_trajectory = loader.load_train_data()
    session_indices = np.array(loader.load_data()['session_indices'])

    # Use training trajectory (held-out trials have Y=NaN, treated as missing).
    # latent_dim = M + 1 (bias + regressors)
    class _Args:
        model_class = 'GLMLearn'
        learning_rule = args.learning_rule
        lapse = False
        vector_alpha = False
    model = fit_utils.load_model(_Args())
    model.reward_func = lambda x, y: models.reward(x[0], y)  # clickLogRatio first
    model.latent_dim = len(args.regressors) + 1

    key = jax.random.PRNGKey(args.seed)
    logger.info('Running posterior SMC sweep...')
    z_hist, log_lik = model.posterior_samples(
        key, params, train_trajectory,
        N_particles=args.N_particles, return_history=True, LAG=True, verbose=False,
    )
    logger.info(f'Posterior MLL: {float(log_lik):.2f}')

    w_mean = np.array(z_hist).mean(axis=0)   # (T, M+1)
    w_std = np.array(z_hist).std(axis=0)

    labels = ['bias'] + args.regressors
    T = w_mean.shape[0]
    trial_idx = np.arange(T)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2.3 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    sb = [int(x) for x in session_indices if 0 < int(x) < T]
    for ax, k, name in zip(axes, range(w_mean.shape[1]), labels):
        ax.plot(trial_idx, w_mean[:, k], lw=1.5, color='C0')
        ax.fill_between(trial_idx,
                        w_mean[:, k] - w_std[:, k],
                        w_mean[:, k] + w_std[:, k],
                        alpha=0.2, color='C0', linewidth=0)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        for b in sb:
            ax.axvline(b, color='red', lw=0.6, alpha=0.7)
        ax.set_ylabel(name)
    axes[-1].set_xlabel('Trial')
    axes[0].set_title(f'{args.subject} — GLMLearn ({args.learning_rule}) — posterior weight trajectory'
                      f'\n(red lines: session boundaries)')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    logger.info(f'Saved {args.out}')


if __name__ == '__main__':
    main()
