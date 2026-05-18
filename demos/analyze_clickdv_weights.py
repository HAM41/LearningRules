'''
A1 + A2 analyses on a fitted ClickDV GLMLearn run.

A1: Decompose the posterior weight trajectory into a learning component
    (cumulative REINFORCE updates applied at the posterior mean) and a noise
    component (residual = posterior - learning component). Tells us whether
    drifts in the weights are driven by the learning rule or by the random-walk
    noise (sigma / sigma_day).

A2: Per-trial, per-regressor contribution to the choice logit
    (eta[t] = w_mean[t, 0] + sum_k w_mean[t, k+1] * X[t, k]). Lets us read off
    "clicks explain X% of choice logit variance in session 1 vs Y% in session 2",
    via the covariance decomposition (sums to 100% by linearity).
'''
import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import models
import fit_utils
from clickdv import ClickDVSingleTrajectoryLoader
from demos.plot_clickdv_weights import parse_final_params_glmlearn

logging.basicConfig(level=logging.INFO,
                    format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_model(args):
    class _Args:
        model_class = 'GLMLearn'
        learning_rule = args.learning_rule
        lapse = False
        vector_alpha = False
    model = fit_utils.load_model(_Args())
    model.reward_func = lambda x, y: models.reward(x[0], y)
    model.latent_dim = len(args.regressors) + 1
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    parser.add_argument('--subject', type=str, default='A324')
    parser.add_argument('--regressors', type=str, nargs='+',
                        default=['clickLogRatio', 'previousChoice', 'previousRewarded'])
    parser.add_argument('--learning-rule', type=str, default='reinforce')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-N', '--N-particles', type=int, default=2000)
    parser.add_argument('--out-dir', type=str, default='figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    params = parse_final_params_glmlearn(args.log)
    logger.info(f'Loaded params: {params}')

    loader = ClickDVSingleTrajectoryLoader({
        'subject': args.subject,
        'regressors': args.regressors,
        'learning_rule': args.learning_rule,
        'seed': args.seed,
    })
    train_trajectory = loader.load_train_data()
    session_indices = np.array(loader.load_data()['session_indices'])
    X = np.array(train_trajectory.X)
    Y = np.array(train_trajectory.Y)
    R = np.array(train_trajectory.R)
    day_flags = np.array(train_trajectory.day_flags)
    T = X.shape[0]

    model = build_model(args)
    key = jax.random.PRNGKey(args.seed)

    logger.info('Running posterior SMC sweep...')
    z_hist, log_lik = model.posterior_samples(
        key, params, train_trajectory,
        N_particles=args.N_particles, return_history=True, LAG=True, verbose=False,
    )
    logger.info(f'Posterior MLL: {float(log_lik):.2f}')

    w_mean = np.array(z_hist).mean(axis=0)             # (T, M+1)
    M_plus_1 = w_mean.shape[1]

    # ------------------------------------------------------------------ A1
    logger.info('Computing learning vs noise decomposition...')
    keys = jax.random.split(key, T)
    w_mean_j = jnp.asarray(w_mean)
    X_j = jnp.asarray(X)
    Y_j = jnp.asarray(Y)
    R_j = jnp.asarray(R)
    day_flags_j = jnp.asarray(day_flags)

    def _learning_at(t):
        _, learning_signal = model.update_weights(
            keys[t], w_mean_j[t],
            x=X_j[t], y=Y_j[t], r=R_j[t], day_flag=day_flags_j[t],
            params=params, return_learning_signal=True,
        )
        return learning_signal

    learning_updates = np.array(jax.vmap(_learning_at)(jnp.arange(T)))  # (T, M+1)

    learning_component = np.zeros_like(w_mean)
    learning_component[0] = w_mean[0]
    learning_component[1:] = w_mean[0] + np.cumsum(learning_updates[:-1], axis=0)
    noise_component = w_mean - learning_component

    # Identity check
    recon = learning_component + noise_component
    assert np.allclose(recon, w_mean, atol=1e-5), \
        f'Decomposition identity violated, max err={np.abs(recon-w_mean).max():.3e}'
    logger.info('Decomposition identity check passed.')

    labels = ['bias'] + args.regressors
    sb = [int(b) for b in session_indices if 0 < int(b) < T]

    fig, axes = plt.subplots(M_plus_1, 1, figsize=(11, 2.4 * M_plus_1), sharex=True)
    if M_plus_1 == 1:
        axes = [axes]
    trial_idx = np.arange(T)
    for ax, k, name in zip(axes, range(M_plus_1), labels):
        ax.plot(trial_idx, w_mean[:, k], lw=1.5, color='black', label='posterior mean')
        ax.plot(trial_idx, learning_component[:, k], lw=1.2, color='C2', alpha=0.85, label='learning component')
        ax.plot(trial_idx, noise_component[:, k], lw=1.0, color='C3', alpha=0.85, label='noise component')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        for b in sb:
            ax.axvline(b, color='red', lw=0.6, alpha=0.4)
        ax.set_ylabel(name)
        ax.legend(loc='upper left', fontsize=8, ncol=3, framealpha=0.9)
    axes[-1].set_xlabel('Trial')
    axes[0].set_title(f'{args.subject} — GLMLearn ({args.learning_rule}) — '
                      f'learning vs noise decomposition (REINFORCE updates at posterior mean)')
    fig.tight_layout()
    decomp_path = os.path.join(args.out_dir, f'analyze_{args.subject}_decomp.png')
    fig.savefig(decomp_path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {decomp_path}')

    # ------------------------------------------------------------------ A2
    logger.info('Computing per-trial logit contributions...')
    contrib = np.zeros((T, M_plus_1))
    contrib[:, 0] = w_mean[:, 0]                          # bias
    for k, name in enumerate(args.regressors):
        contrib[:, k + 1] = w_mean[:, k + 1] * X[:, k]
    eta = contrib.sum(axis=1)

    # Sanity: eta == w_mean[:,0] + sum_k w_mean[:,k+1]*X[:,k]
    expected = w_mean[:, 0] + np.einsum('ij,ij->i', w_mean[:, 1:], X)
    assert np.allclose(eta, expected, atol=1e-5), \
        f'Logit identity violated, max err={np.abs(eta-expected).max():.3e}'
    logger.info('Logit identity check passed.')

    # Per-session covariance-based fractions: f_k = cov(contrib_k, eta) / var(eta)
    sess_bounds = list(session_indices) + ([T] if session_indices[-1] != T else [])
    sess_bounds = sorted(set(int(b) for b in sess_bounds))
    if sess_bounds[0] != 0:
        sess_bounds = [0] + sess_bounds
    if sess_bounds[-1] != T:
        sess_bounds = sess_bounds + [T]

    print()
    print(f'{"Session":>10} {"trials":>10} ' + ' '.join(f'{n:>14}' for n in labels))
    for s, (t0, t1) in enumerate(zip(sess_bounds[:-1], sess_bounds[1:])):
        eta_s = eta[t0:t1]
        if eta_s.std() < 1e-8:
            fractions = ['n/a'] * M_plus_1
        else:
            v = np.var(eta_s)
            fractions = [
                f'{(np.cov(contrib[t0:t1, k], eta_s, ddof=0)[0, 1] / v):>14.3f}'
                for k in range(M_plus_1)
            ]
        print(f'{s:>10} {t1-t0:>10} ' + ' '.join(fractions))
    print('(Each row: fraction of logit variance attributable to each regressor; rows sum to 1.0)')
    print()

    fig, axes = plt.subplots(M_plus_1 + 1, 1, figsize=(11, 2.2 * (M_plus_1 + 1)), sharex=True)
    for ax, k, name in zip(axes[:M_plus_1], range(M_plus_1), labels):
        ax.plot(trial_idx, contrib[:, k], lw=1.0, color=f'C{k}')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        for b in sb:
            ax.axvline(b, color='red', lw=0.6, alpha=0.4)
        ax.set_ylabel(f'{name}\ncontrib')
    axes[-1].plot(trial_idx, eta, lw=1.0, color='black', label='total η')
    axes[-1].axhline(0, color='gray', lw=0.5, ls='--')
    for b in sb:
        axes[-1].axvline(b, color='red', lw=0.6, alpha=0.4)
    axes[-1].set_ylabel('total η')
    axes[-1].set_xlabel('Trial')
    axes[0].set_title(f'{args.subject} — per-trial contribution to choice logit '
                      f'(η = bias + Σ w·x); red lines: session boundaries')
    fig.tight_layout()
    contrib_path = os.path.join(args.out_dir, f'analyze_{args.subject}_contributions.png')
    fig.savefig(contrib_path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {contrib_path}')


if __name__ == '__main__':
    main()
