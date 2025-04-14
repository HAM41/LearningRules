r'''
Author: Victor Geadah

This script contains functions to load and preprocess behavioral data from the IBL database.

Useful links: 
- IBL protocol details: https://figshare.com/articles/preprint/A_standardized_and_reproducible_method_to_measure_decision-making_in_mice_Appendix_2_IBL_protocol_for_mice_training/11634729/3?file=24954497
- IBL ONE API searching: https://int-brain-lab.github.io/ONE/notebooks/one_search/one_search.html
'''
import numpy as np
import pandas as pd
from typing import NamedTuple, List, Tuple, Optional
from jaxtyping import Array, Float, Bool
import jax
import jax.numpy as jnp
import models
import os; import sys

HOMEDIR = '/home/vg0233/PillowLab/LearningRules/'

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Y_L, Y_R = -1.0, 1.0 # Numerical value for the left, null, and right choices

def tanh_transform(x, p=5):
    return np.tanh(p*x)/np.tanh(p)

def tanh_inv_transform(y, p=5):
    return np.arctanh(y*np.tanh(p))/p

def load_IBL_behavioral_data(protocol: str='training') -> Tuple[pd.DataFrame, list]:
    '''
    Load data from IBL database for a given protocol, into a pandas dataframe.
    Select the `trainable` the subjects that moved to biasedChoiceWorld, thus attained status `Trained 1b`. 
    Args:
        protocol: str, the protocol to load data from. Select from ['training', 'biasedChoiceWorld']
    returns: 
        entries: pd.DataFrame, the data from the IBL database, with columns:
            ['lab', 'subject', 'date', 'contrastRight', 'choice', 'probabilityLeft', 'feedbackType', 'rewardVolume', 'contrastLeft']
    '''
    from one.api import ONE
    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    one = ONE(password='international')

    # Select only the subjects that moved to biasedChoiceWorld
    _, infos_biasedCW = one.search(task_protocol='biasedChoiceWorld', details=True)
    subjects = np.unique([info['subject'] for info in infos_biasedCW])

    # Get eids and infos for these subjects
    eids, infos = one.search(subject=subjects, task_protocol=protocol, details=True)

    # Select keys for trial data that are not time dependent (e.g. 'goCue_times')
    keys = ['contrastLeft', 'contrastRight', 'choice', 'probabilityLeft', 'feedbackType', 'rewardVolume']

    def check_keys(trial_data) -> bool:
        return np.all([key in trial_data.keys() for key in keys])

    # Compile data into a pandas dataframe
    entries = {'lab': [], 'subject': [], 'date': [], 'session': []}
    for key in keys:
        entries[key] = []
    
    error_animals = []
    for eid, info in zip(eids, infos):
        try:
            trial_data = one.load_object(eid, 'trials')
            assert check_keys(trial_data), f'Keys missing.'

            for t in range(len(trial_data['choice'])):
                entries['lab'].append(info['lab'])
                entries['subject'].append(info['subject'])
                entries['date'].append(str(info['date']))
                entries['session'].append(info['number'])
                
                for key in keys:
                    entries[key].append(float(trial_data[key][t]))
        
        except Exception as e: # Catch any errors, leave out the data
            logger.info(f'Error loading trials for {eid}: {e}')
            error_animals.append(info['subject'])
            continue

    entries_df = pd.DataFrame(entries)
    entries_df = entries_df.fillna(0) # NaNs are on the contrast information, which is 0 when not present
    
    if protocol == 'training':
        # Remove the animals that loaded with errors on some sessions in their training data.
        # This removed 38/100 amimals, 347037/1171032 trials (30 %)
        error_animal_ids = np.unique(error_animals)
        logger.warning(f'WARNING: removing {len(error_animal_ids)} animals with trial loading errors.')
        entries_df = entries_df[~entries_df['subject'].isin(error_animal_ids)]
    
    return entries_df, error_animals

def format_IBL_behavioral_data(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    out_df['choice'] = -out_df['choice'] # choice from wheel direction to decision
    out_df['choice'] = (out_df['choice'] + 1)/2 # Convert choice to 0, 1
    out_df['feedbackType'] = (out_df['feedbackType'] + 1)/2 # Convert feedbackType to 0, 1
    return out_df

def z_score(array):
    return (array - np.mean(array))/np.std(array)

def format_regressor_data(df: pd.DataFrame, regressor_name:str) -> np.ndarray:
    regressors_list = ['stimIntensity', 'contrastLeft', 'contrastRight', 
                       'correctSide', 'previousChoice', 'previousRewarded']
    
    # Columns in df: ['contrastLeft', 'choice', 'contrastRight', 'probabilityLeft', 'feedbackType', 'rewardVolume']

    if regressor_name  not in regressors_list:
        raise Exception('Regressor name invalid or not implemented.')
    
    if regressor_name == 'stimIntensity':
        _contrast_right = tanh_transform(df['contrastRight'], p=5)
        _contrast_left =  tanh_transform(df['contrastLeft'], p=5)
        # data = z_score(stim_intensity)
        data = _contrast_right - _contrast_left
    elif regressor_name == 'contrastLeft':
        p=5 # as used in Psytrack paper
        data = tanh_transform(df['contrastLeft'], p) # tanh transformation of left contrasts
    elif regressor_name == 'contrastRight':
        p=5 # as used in Psytrack paper
        data = tanh_transform(df['contrastRight'], p) # tanh transformation of right contrasts
    elif regressor_name == 'correctSide':
        # data = np.array(df['correctSide'])
        data = np.where(df['contrastRight'] > df['contrastLeft'], Y_R, Y_L)
    elif regressor_name == 'previousChoice':
        y = np.array(df['choice'])
        data_temp = y[0:-1]
        data = np.concatenate(([0.0], data_temp))
    elif regressor_name == 'previousRewarded':
        data_temp = np.where(df['contrastRight'] > df['contrastLeft'], Y_R, Y_L)[0:-1] # previous rewarded
        data = np.concatenate(([0.0], data_temp))
    
    assert len(data) == len(df), f"Data length mismatch: {len(data)} vs {len(df)}"
    return data

def get_mouse_design(
        dfAll: pd.DataFrame, subject: str, regressors=Optional[List[str]],
        ) -> Tuple[np.ndarray, np.ndarray, List]:
    '''
    Returns design matrix X and vector Y of outputs (decisions) for a given subject. 
    The regressors that consistute the design matrix are passed as a list. Options include:
        ['stimIntensity', 'contrastLeft', 'contrastRight', 'correctSide', 'previousChoice', 'previousRewarded']
    '''
    data = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified

    # # Keep specified number of sessions
    dates_to_keep = np.unique(data['date'])
    data_temp = pd.DataFrame(data.loc[data['date'].isin(list(dates_to_keep))])

    # Design and choice matrices
    if regressors is None:
        regressors = ['stimIntensity'] # Default regressor
    X = np.zeros((data_temp.shape[0], len(regressors)))
    Y = np.array(data_temp['choice'])
    
    if Y_L == -1.0:
        Y = 2 * Y - 1
        if np.sum(Y == 0.) > 0:
            logger.warning(f"Warning: {np.sum(Y == 0.)}/{len(Y)} choices are 0.0, replaced with {Y_R}")
            Y = np.where(Y==0., Y_R, Y)
    else:
        if np.sum(Y == 0.5) > 0:
            logger.warning(f"Warning: {np.sum(Y == 0.5)}/{len(Y)} choices are 0.0, replaced with {Y_R}")
            Y = np.where(Y==0.5, Y_R, Y)

    # Y = np.where(Y == 0., Y_L, Y) # make decision = 0 as Y_L

    # Ensure that notations match 
    assert np.isin(Y, np.array([Y_L, Y_R])).all(), f"Choices must in {[Y_L, Y_R]} but got {np.unique(Y, return_counts=True)}"
    
    _sub_df = data_temp.query("contrastRight != 0.0 and contrastLeft != 0.0")
    _correct_choice = np.where(_sub_df["contrastRight"] > _sub_df["contrastLeft"], Y_R, Y_L)
    _feedback = _correct_choice == np.array(_sub_df['choice']).astype(float)
    assert (_feedback == _sub_df["feedbackType"]).all(), f"Correct choice and feedbackType mismatch {np.sum(_feedback != data_temp["feedbackType"])}"

    for i, regressor in enumerate(regressors):
        X[:,i] = format_regressor_data(data_temp, regressor)

    # Session start indicies
    sess_ind = [0]
    for date in dates_to_keep:
        d = data_temp[data_temp['date']==date]
        for sess in np.unique(d['session']):
            dTemp = d[d['session'] == sess]
            dLength = len(dTemp.index.tolist())
            sess_ind.append(sess_ind[-1] + dLength)
    
    return X, Y, sess_ind

def load_session_indices(lab, subject_id):
    # Load IBL data
    DOWNLOAD = False
    if DOWNLOAD:
        protocol = 'training'
        df, _ = load_IBL_behavioral_data(protocol)
        df.to_csv(f'IBL_{protocol}_protocol.csv', index=False)
        df = format_IBL_behavioral_data(df)
    else:
        df = pd.read_csv(HOMEDIR + 'data/IBL_training_protocol.csv')
        df = format_IBL_behavioral_data(df)

    assert lab in np.unique(df['lab'].values), f"Lab {lab} not found in data."
    lab_df = df[df['lab']==lab]
    lab_subjects = np.unique(lab_df['subject'].values)
    assert subject_id < len(lab_subjects), f"Subject index {subject_id} out of bounds."
    subject = lab_subjects[subject_id]
    
    data = lab_df[lab_df['subject']==subject]   # Restrict data to the subject specified

    # # Keep specified number of sessions
    dates_to_keep = np.unique(data['date'])
    data_temp = pd.DataFrame(data.loc[data['date'].isin(list(dates_to_keep))])

    # Session start indicies
    sess_ind = [0]
    for date in dates_to_keep:
        d = data_temp[data_temp['date']==date]
        for sess in np.unique(d['session']):
            dTemp = d[d['session'] == sess]
            dLength = len(dTemp.index.tolist())
            sess_ind.append(sess_ind[-1] + dLength)
    
    return sess_ind

def format_reward_function(regressors, learning_rule):
    if regressors[0] == 'stimIntensity':
        if learning_rule == 'policy_gradient':
            return lambda x, y: models.effective_reward(x[0])
        else:
            return lambda x, y: models.reward(x[0], y)
    elif regressors[1] == 'contrastRight' and regressors[0] == 'contrastLeft':
        if learning_rule == 'policy_gradient':
            return lambda x, y: models.effective_reward(x[1]-x[0])
        else:
            return lambda x, y: models.reward(x[1]-x[0], y)
    else:
        raise Exception('Reward function not implemented for this regressor set.')
    
def format_reward(X, Y, regressors, learning_rule) -> jnp.ndarray:
    reward_func = format_reward_function(regressors, learning_rule)
    R = jax.vmap(reward_func)(X, Y)
    return R

class IBLDataTrajectory(NamedTuple):
    '''IBL single trajectory data, with T trials.'''
    X: Float[Array, "T M"]      # Regressors, M-dimensional
    Y: Float[Array, "T"]        # Choices, in {Y_L, Y_R}
    R: Float[Array, "T"]        # Rewards
    day_flags: Bool[Array, "T"] # Day flags, True if new day/session

def split_train_test_sessions(X, Y, R, day_flags, session_indices, held_out_sessions):
    '''
    Split per session, then concatenate held_out_sessions into test set, and others into training set.
    Args:
        X: (T, M), Y: (T,), R: (T,), day_flags: (T,), session_indices: list of session time indices (in (0,T))
        held_out_sessions: list of session indices (in (0, len(session_indices))) to hold out.
    '''
    # Split data into training and test sets
    X_train, Y_train, R_train, day_flags_train = [], [], [], []
    X_test, Y_test, R_test, day_flags_test = [], [], [], []

    for session_id in range(len(session_indices) - 1):
        t1, t2 = session_indices[session_id], session_indices[session_id + 1]
        if session_id in held_out_sessions:
            X_test.append(X[t1:t2])
            Y_test.append(Y[t1:t2])
            R_test.append(R[t1:t2])
            day_flags_test.append(day_flags[t1:t2])
        else:
            X_train.append(X[t1:t2])
            Y_train.append(Y[t1:t2])
            R_train.append(R[t1:t2])
            day_flags_train.append(day_flags[t1:t2])

    X_train = jnp.concatenate(X_train, axis=0)
    Y_train = jnp.concatenate(Y_train, axis=0)
    R_train = jnp.concatenate(R_train, axis=0)
    day_flags_train = jnp.concatenate(day_flags_train, axis=0)

    X_test = jnp.concatenate(X_test, axis=0)
    Y_test = jnp.concatenate(Y_test, axis=0)
    R_test = jnp.concatenate(R_test, axis=0)
    day_flags_test = jnp.concatenate(day_flags_test, axis=0)

    train_trajectory = IBLDataTrajectory(X_train, Y_train, R_train, day_flags_train)
    test_trajectory = IBLDataTrajectory(X_test, Y_test, R_test, day_flags_test)
    return train_trajectory, test_trajectory

def hold_out_trials(X, Y, R, day_flags, held_out_trials):
    '''
    Split per session, then concatenate held_out_sessions into test set, and others into training set.
    Args:
        X: (T, M), Y: (T,), R: (T,), day_flags: (T,), session_indices: list of session time indices (in (0,T))
        held_out_sessions: list of session indices (in (0, len(session_indices))) to hold out.
    '''
    # Split data into training and test sets
    X_train = jnp.asarray(X).copy()

    # Mask held out trials in Y
    Y_train = jnp.asarray(Y).copy()
    Y_train = Y_train.at[held_out_trials].set(jnp.nan)

    R_train = jnp.asarray(R).copy()

    day_flags_train = jnp.asarray(day_flags).copy()

    train_trajectory = IBLDataTrajectory(X_train, Y_train, R_train, day_flags_train)
    return train_trajectory


class IBLSingleTrajectoryLoader():
    def __init__(self, params, DOWNLOAD=False):
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
                - 'trajectory': IBLDataTrajectory, trajectory data.
                - 'held_out_trials': jnp.array, indices of held-out trials.
                - 'train_trajectory': IBLDataTrajectory, training trajectory data.
        Raises:
            FileNotFoundError
                If the local data file is not found and DOWNLOAD is set to False.
            AssertionError
                If the specified lab or subject index is not found in the data.
        """
        lab = params['lab']
        idx = params['subject_id']
        regressors = params['regressors']
        learning_rule = params['learning_rule']
        seed = params['seed']

        self.data = {}

        # Load IBL data
        if DOWNLOAD:
            protocol = 'training'
            df, _ = load_IBL_behavioral_data(protocol)
            df.to_csv(f'IBL_{protocol}_protocol.csv', index=False)
            df = format_IBL_behavioral_data(df)
        else:
            try:
                df = pd.read_csv(HOMEDIR + 'data/IBL_training_protocol.csv')
            except FileNotFoundError:
                logger.error("IBL training protocol data not found. Use DOWNLOAD=True to download the data from the ONE api.")
                sys.exit(1)
            df = format_IBL_behavioral_data(df)

        assert lab in np.unique(df['lab'].values), f"Lab {lab} not found in data."
        lab_df = df[df['lab']==lab]
        lab_subjects = np.unique(lab_df['subject'].values)
        assert idx < len(lab_subjects), f"Subject index {idx} out of bounds."
        subject = lab_subjects[idx]
        logger.info(f"Loading data for subject {subject}.")

        # Get design matrix data
        X, Y, session_indices = get_mouse_design(lab_df, subject=subject, regressors=regressors)
        R = format_reward(X, Y, regressors, learning_rule)
        day_flags = models.set_day_flags(len(X), jnp.array(session_indices))
        self.data['session_indices'] = jnp.array(session_indices, dtype=jnp.int32)
        self.data['trajectory']  = IBLDataTrajectory(X, Y, R, day_flags)

        #TODO : check that reward and (choice vs correctchoice) are correctly aligned, or reward and rewardVolume

        # Split data into training and test sets
        key = jax.random.PRNGKey(seed)

        # n_sessions = len(session_indices)
        # held_out_sessions = jax.random.choice(key, n_sessions, shape=(int(n_sessions/10),), replace=False)
        # held_out_sessions = jnp.sort(held_out_sessions)
        # self.data['held_out_sessions']  = held_out_sessions

        # Focus on held-out trials
        n_trials = len(X)
        held_out_trials = jax.random.choice(key, n_trials, shape=(int(n_trials/10),), replace=False)
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
    
def get_number_subjects(lab):
    df = pd.read_csv(HOMEDIR + 'data/IBL_training_protocol.csv')
    df = format_IBL_behavioral_data(df)
    assert lab in np.unique(df['lab'].values), f"Lab {lab} not found in data."
    lab_df = df[df['lab']==lab]
    lab_subjects = np.unique(lab_df['subject'].values)
    return len(lab_subjects)

def load_session_indices(lab, subject_id):
    df = pd.read_csv(HOMEDIR + 'data/IBL_training_protocol.csv')
    df = format_IBL_behavioral_data(df)
    assert lab in np.unique(df['lab'].values), f"Lab {lab} not found in data."
    lab_df = df[df['lab']==lab]
    lab_subjects = np.unique(lab_df['subject'].values)
    assert subject_id < len(lab_subjects), f"Subject index {subject_id} out of bounds."
    subject = lab_subjects[subject_id]
    logger.info(f"Loading data for subject {subject}.")

    X, Y, session_indices = get_mouse_design(lab_df, subject=subject, regressors=['stimIntensity', 'previousChoice', 'previousRewarded'])
    return jnp.array(session_indices, dtype=jnp.int32)

def trajectory_length(lab, subject_id):
    loader = IBLSingleTrajectoryLoader({
        'lab': lab,
        'subject_id': subject_id,
        'regressors': ['stimIntensity', 'previousChoice', 'previousRewarded'],
        'learning_rule': 'policy_gradient',
        'seed': 0
    })
    data = loader.load_data()
    return len(data['trajectory'].X)

if __name__=='__main__':
    HOMEDIR = '/home/vg0233/PillowLab/LearningRules/'
    logging.warning(f"Y_L and Y_R are set to {Y_L} and {Y_R} as global variables.")

    df = pd.read_csv(HOMEDIR + 'data/IBL_training_protocol.csv')
    num_unique_animals = df['subject'].nunique()
    print(f"Number of unique animals: {num_unique_animals}")

    # Print all lengths of trajectories
    all_n_sessions = []
    for lab in np.unique(df['lab'].values):
        for subject_id in range(get_number_subjects(lab)):
            try:
                N_sessions = len(load_session_indices(lab, subject_id))
                print(f"Number of sessions for {lab} subject {subject_id}: {N_sessions}")
                all_n_sessions.append(N_sessions)
            except:
                print(f"Error loading data for {lab} subject {subject_id}.")
    print(all_n_sessions)
    # jnp.save(HOMEDIR + 'data/IBL_trajectory_lengths.npy', jnp.array(all_n_sessions))

    # loader = IBLSingleTrajectoryLoader(
    #     params ={
    #     'lab': 'wittenlab',
    #     'subject_id': 1,
    #     'regressors': ['stimIntensity', 'previousChoice', 'previousRewarded'],
    #     'learning_rule': 'policy_gradient',
    #     'seed': 0
    #     }, 
    #     DOWNLOAD=False
    #     )
    
    # data = loader.load_data()
    # print(data)