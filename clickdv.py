r'''
Loader for ClickDV (Brody-lab Poisson-clicks) data into the LearningRules
`Trajectory(X, Y, R, day_flags)` format.

Mirrors the structure of `ibl.py` but reads per-session HDF5 files produced by
the ClickDV preprocessing pipeline. Click trains are reduced to one signed
scalar stimulus per trial (clickDiff / clickLogRatio / clickAsymmetry).

Note: the per-trial summary columns in the processed h5 (`n_left_clicks`,
`n_right_clicks`, `click_asymmetry`) appear to be stale in some sessions
(always 1, always 0). Click counts are recomputed from the `clicks` table.
'''
import os
import sys
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

import models
from constants import Choice
from parameters import Trajectory
from ibl import hold_out_trials

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLICKDV_DATA_ROOT = Path(os.getenv(
    "CLICKDV_DATA_ROOT",
    "/home/ham/SF/Personal/Education/07 Princeton/Brody-Daw/ClickDV/data/processed",
))

CLICKDV_REGRESSORS = ['clickDiff', 'clickLogRatio', 'clickAsymmetry',
                      'previousChoice', 'previousRewarded']


def _trials_with_click_counts(h5_path: Path) -> pd.DataFrame:
    '''Read the `trials` table and overwrite click counts using the `clicks` table.'''
    trials = pd.read_hdf(h5_path, 'trials')
    clicks = pd.read_hdf(h5_path, 'clicks')

    counts = clicks.groupby(['trial_id', 'click_side']).size().unstack(fill_value=0)
    counts = counts.rename(columns={'left': 'n_left_clicks_true',
                                    'right': 'n_right_clicks_true'})
    for col in ('n_left_clicks_true', 'n_right_clicks_true'):
        if col not in counts.columns:
            counts[col] = 0
    out = trials.merge(counts.reset_index(), on='trial_id', how='left').fillna(
        {'n_left_clicks_true': 0, 'n_right_clicks_true': 0}
    )
    out['n_left_clicks'] = out['n_left_clicks_true'].astype(int)
    out['n_right_clicks'] = out['n_right_clicks_true'].astype(int)
    out = out.drop(columns=['n_left_clicks_true', 'n_right_clicks_true'])
    return out


def load_clickdv_behavioral_data(subject: str,
                                 data_root: Path = CLICKDV_DATA_ROOT) -> pd.DataFrame:
    '''
    Load all available sessions for `subject` and return a single sorted DataFrame.

    One row per trial, with columns:
        ['subject', 'date', 'session', 'trial_id', 'original_trial_num',
         'choice', 'rewarded', 'violated',
         'n_left_clicks', 'n_right_clicks',
         'trial_duration', 'click_duration', 'click_rate']
    Sorted by (date, original_trial_num). 'session' is always 1 since ClickDV
    files are one session per date.
    '''
    pattern = str(Path(data_root) / subject / '*' / f'{subject}_*_session_data.h5')
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No session h5 files found for {subject} under {data_root}')

    frames = []
    for p in paths:
        date = Path(p).parent.name  # e.g. '2023-07-21'
        df = _trials_with_click_counts(Path(p))
        df['subject'] = subject
        df['date'] = date
        df['session'] = 1
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(by=['date', 'original_trial_num'], ignore_index=True)
    logger.info(f'Loaded {subject}: {len(out)} trials across {len(paths)} sessions.')
    return out


def format_clickdv_regressor_data(df: pd.DataFrame, regressor_name: str) -> np.ndarray:
    '''
    Build a per-trial regressor vector. Sign convention: positive = right.

    Supported names:
        'clickDiff'       -> n_right - n_left
        'clickLogRatio'   -> log(n_right + 1) - log(n_left + 1)
        'clickAsymmetry'  -> (n_right - n_left) / (n_right + n_left), 0 if both zero
        'previousChoice'  -> previous trial's choice in {-1, +1}, 0 at trial 0
        'previousRewarded'-> previous trial's reward in {0, 1}, 0 at trial 0
    '''
    if regressor_name not in CLICKDV_REGRESSORS:
        raise ValueError(f'Unknown regressor {regressor_name!r}; expected one of {CLICKDV_REGRESSORS}')

    nL = df['n_left_clicks'].to_numpy(dtype=float)
    nR = df['n_right_clicks'].to_numpy(dtype=float)

    if regressor_name == 'clickDiff':
        data = nR - nL
    elif regressor_name == 'clickLogRatio':
        data = np.log(nR + 1.0) - np.log(nL + 1.0)
    elif regressor_name == 'clickAsymmetry':
        total = nR + nL
        data = np.where(total > 0, (nR - nL) / np.where(total > 0, total, 1.0), 0.0)
    elif regressor_name == 'previousChoice':
        # ClickDV choice: 0=left, 1=right -> remap to {-1, +1}
        y = 2.0 * df['choice'].to_numpy(dtype=float) - 1.0
        # NaN-mask violated trials so prev-of-violated is 0 (handled by nan-to-0 below)
        viol = df['violated'].to_numpy(dtype=bool)
        y = np.where(viol, np.nan, y)
        prev = np.concatenate(([0.0], y[:-1]))
        prev = np.where(np.isnan(prev), 0.0, prev)
        data = prev
    elif regressor_name == 'previousRewarded':
        r = df['rewarded'].to_numpy(dtype=float)
        # If the prior trial was violated, treat as unrewarded.
        viol = df['violated'].to_numpy(dtype=bool)
        r = np.where(viol, 0.0, r)
        data = np.concatenate(([0.0], r[:-1]))

    assert len(data) == len(df), f'Length mismatch: {len(data)} vs {len(df)}'
    return data


def get_subject_design(df: pd.DataFrame,
                       regressors: List[str]
                       ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    '''
    Build (X, Y, session_indices) for a single subject's concatenated DataFrame.

    X is (T, M). Y is (T,) in {Choice.LEFT.value, Choice.RIGHT.value, NaN};
    violated trials get Y=NaN so the likelihood ignores them while keeping the
    sequence (and day_flags / previous-trial regressors) aligned.
    session_indices is a list of cumulative trial counts at the start of each
    session, including 0 and len(df) sentinels (matches `ibl.get_mouse_design`).
    '''
    T = len(df)
    X = np.zeros((T, len(regressors)))
    for i, name in enumerate(regressors):
        X[:, i] = format_clickdv_regressor_data(df, name)

    # Remap choice 0/1 -> -1/+1, NaN-mask violated trials
    Y = 2.0 * df['choice'].to_numpy(dtype=float) - 1.0
    viol = df['violated'].to_numpy(dtype=bool)
    Y = np.where(viol, np.nan, Y)

    # Sanity: non-NaN Y is exactly {Choice.LEFT.value, Choice.RIGHT.value}
    valid = ~np.isnan(Y)
    assert np.isin(Y[valid], np.array([Choice.LEFT.value, Choice.RIGHT.value])).all(), \
        f'Unexpected Y values: {np.unique(Y[valid])}'

    # Session boundaries from sorted date column
    sess_ind = [0]
    for date in df['date'].drop_duplicates().tolist():
        sess_ind.append(sess_ind[-1] + int((df['date'] == date).sum()))
    return X, Y, sess_ind


def format_reward_function(regressors: List[str], learning_rule: str):
    '''
    Reward function keyed off the first (signed-stimulus) regressor.
    Mirrors `ibl.format_reward_function`. Assumes regressors[0] is the
    signed stimulus.
    '''
    stim_name = regressors[0]
    if stim_name not in ('clickDiff', 'clickLogRatio', 'clickAsymmetry'):
        raise ValueError(
            f'First regressor must be a signed stimulus '
            f'(clickDiff/clickLogRatio/clickAsymmetry); got {stim_name!r}'
        )
    if learning_rule == 'policy_gradient':
        return lambda x, y: models.effective_reward(x[0])
    else:
        return lambda x, y: models.reward(x[0], y)


def format_reward(X, Y, regressors, learning_rule) -> jnp.ndarray:
    reward_func = format_reward_function(regressors, learning_rule)
    return jax.vmap(reward_func)(X, Y)


class ClickDVSingleTrajectoryLoader:
    '''
    Same interface as `ibl.IBLSingleTrajectoryLoader` and
    `demos.fit_demo.DemoSingleTrajectoryLoader`: exposes
    `load_data`, `load_train_data`, `load_test_data`.

    params dict keys:
        'subject'        : str, e.g. 'A324'
        'regressors'     : list[str], from CLICKDV_REGRESSORS
        'learning_rule'  : str, used to pick the reward function shape
        'seed'           : int, controls the held-out trial split
        'use_recorded_reward' (optional, default True):
            If True, R is the animal's actually-recorded reward; if False, R
            is computed from sign(X[:,0]) vs Y via models.reward (matches IBL
            behaviour when the recorded reward is unavailable).
        'data_root' (optional): path override for the processed ClickDV root.
    '''

    def __init__(self, params: dict):
        subject = params['subject']
        regressors = params['regressors']
        learning_rule = params['learning_rule']
        seed = params['seed']
        use_recorded_reward = params.get('use_recorded_reward', True)
        data_root = Path(params.get('data_root', CLICKDV_DATA_ROOT))

        self.data = {}

        df = load_clickdv_behavioral_data(subject, data_root=data_root)
        X, Y, session_indices = get_subject_design(df, regressors)

        if use_recorded_reward:
            # Use the animal's recorded reward; for violated trials it stays 0.
            R = df['rewarded'].to_numpy(dtype=float)
            R = np.where(df['violated'].to_numpy(dtype=bool), 0.0, R)
            R = jnp.asarray(R)
        else:
            R = format_reward(jnp.asarray(X),
                              jnp.where(np.isnan(Y), 0.0, Y),  # avoid NaN in vmap reward
                              regressors, learning_rule)

        X = jnp.asarray(X)
        Y = jnp.asarray(Y)
        day_flags = models.set_day_flags(len(X), jnp.array(session_indices))

        self.data['session_indices'] = jnp.array(session_indices, dtype=jnp.int32)
        self.data['trajectory'] = Trajectory(X, Y, R, day_flags)

        key = jax.random.PRNGKey(seed)
        n_trials = len(X)
        held_out_trials = jax.random.choice(key, n_trials,
                                            shape=(int(n_trials / 10),), replace=False)
        held_out_trials = jnp.sort(held_out_trials)
        self.data['held_out_trials'] = held_out_trials

        self.data['train_trajectory'] = hold_out_trials(X, Y, R, day_flags, held_out_trials)

    def load_data(self):
        return self.data

    def load_train_data(self):
        return self.data['train_trajectory']

    def load_test_data(self):
        return self.data['trajectory'], self.data['held_out_trials']


if __name__ == '__main__':
    subject = 'A324'
    regressors = ['clickLogRatio', 'previousChoice', 'previousRewarded']

    df = load_clickdv_behavioral_data(subject)
    print(f'Sessions: {df["date"].nunique()}, trials: {len(df)}')
    print(f'Violated trials: {int(df["violated"].sum())}')
    print(f'Choice value counts: {df["choice"].value_counts().to_dict()}')
    print(f'Reward rate: {df["rewarded"].mean():.3f}')

    loader = ClickDVSingleTrajectoryLoader({
        'subject': subject,
        'regressors': regressors,
        'learning_rule': 'reinforce',
        'seed': 0,
    })
    data = loader.load_data()
    traj = data['trajectory']
    print()
    print(f'Trajectory: X {traj.X.shape}, Y {traj.Y.shape}, R {traj.R.shape}, day_flags {traj.day_flags.shape}')
    print(f'Y unique (non-nan): {np.unique(np.array(traj.Y)[~np.isnan(np.array(traj.Y))])}')
    print(f'R unique: {np.unique(np.array(traj.R))}')
    print(f'NaN-masked Y trials: {int(np.isnan(np.array(traj.Y)).sum())}')
    print(f'Day flags True at: {np.where(np.array(traj.day_flags))[0]}')
    print(f'Held-out trials: {len(data["held_out_trials"])}')
    print(f'X[:5]:\n{np.array(traj.X)[:5]}')

    # Sanity: agreement between sign(X[:,0]) and Y on non-violated trials.
    X0 = np.array(traj.X)[:, 0]
    Y_arr = np.array(traj.Y)
    valid = ~np.isnan(Y_arr)
    pred_side = np.where(X0[valid] > 0, Choice.RIGHT.value, Choice.LEFT.value)
    agree = (pred_side == Y_arr[valid]).mean()
    rew_rate = np.array(traj.R)[valid].mean()
    print(f'Choice agrees with sign(stim): {agree:.3f}; mean reward: {rew_rate:.3f}')
