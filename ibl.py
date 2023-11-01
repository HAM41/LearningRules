import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def z_score(array):
    return (array - np.mean(array))/np.std(array)

# def _get_mouse_design(dfAll, subject, sessStop=-1, D=4):
#     '''
#     function to give design matrix x and output vector y for a given subject until session sessStpo
#     '''
#     data = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified
#     # keeping first 40 sessions
#     dateToKeep = np.unique(data['date'])[0:sessStop]
#     dataTemp = pd.DataFrame(data.loc[data['date'].isin(list(dateToKeep))])
#     # getting correct answer for each trial
#     correctSide = np.array(dataTemp['correctSide'])
#     # design and out matrix
#     x = np.zeros((dataTemp.shape[0], D))
#     y = np.array(dataTemp['choice'])
#     x[:,0] = 1 # bias
#     if (D==2):
#         x[:,1] = dataTemp['contrastRight'] - dataTemp['contrastLeft'] # 'stimulus intensity'
#         x[:,1] = (x[:,1] - np.mean(x[:,1])) / np.std(x[:,1]) # z-scored
#     elif (D==4):
#         x[:,1] = dataTemp['contrastRight'] - dataTemp['contrastLeft'] # 'stimulus intensity'
#         x[:,1] = (x[:,1] - np.mean(x[:,1])) / np.std(x[:,1]) # z-scored
#         x[1:,2] = 2 * y[0:-1] - 1 # previous chioce as in Zoe's
#         x[1:,3] = 2 * np.array(dataTemp['correctSide'])[0:-1] - 1 # previous reward as in Zoe's
#     elif (D==3):
#         p=5 # as used in Psytrack paper
#         data['cL'] = np.tanh(p*data['contrastLeft'])/np.tanh(p) # tanh transformation of left contrasts
#         data['cR'] = np.tanh(p*data['contrastRight'])/np.tanh(p) # tanh transformation of right contrasts
#         x[:,1] = dataTemp['cL'] # contrast left transformed
#         x[:,2] = dataTemp['cR'] # contrast right transformed
#     elif (D==5):
#         p=5 # as used in Psytrack paper
#         data['cL'] = np.tanh(p*data['contrastLeft'])/np.tanh(p) # tanh transformation of left contrasts
#         data['cR'] = np.tanh(p*data['contrastRight'])/np.tanh(p) # tanh transformation of right contrasts
#         x[:,1] = dataTemp['cL'] # contrast left transformed
#         x[:,2] = dataTemp['cR'] # contrast right transformed
#         # not taking into account first and last of each session (probably no effect of that)Z2
#         x[1:,3] = y[0:-1] # previous choice
#         x[1:,4] = np.array(dataTemp['correctSide'])[0:-1] # previous rewarded
   
#     # session start indicies
#     sessInd = [0]
#     for date in dateToKeep:
#         d = dataTemp[dataTemp['date']==date]
#         for sess in np.unique(d['session']):
#             dTemp = d[d['session'] == sess]
#             dLength = len(dTemp.index.tolist())
#             sessInd.append(sessInd[-1] + dLength)
#     return x, y, sessInd, correctSide

# def get_behavioral_data(df=None, subject: str='IBL-T1'):
#     if df is None:
#         df = pd.read_csv('./data/ibl_learning_processed.csv')
#     df = df[df['subject']==subject]

#     dates = np.unique(df['date'].values)
#     sessions = np.unique(df['session'].values)

#     X, Y = [], []
#     # for date in dates:
#     #     for session in sessions:
#     #         sub_df = df.query(f"session == {session} and date == '{date}'")
#     #         stim_intensity = sub_df['contrastRight'].values - sub_df['contrastLeft'].values
#     #         if len(stim_intensity) > 0:
#     #             X.append(list(z_score(stim_intensity)))
#     #             Y.append(list(sub_df['choice'].values))

#     stim_intensity = df['contrastRight'].values - df['contrastLeft'].values 
#     X.append(list(z_score(stim_intensity)))
#     Y.append(list(df['choice'].values))
#     return X, Y

def format_regressor_data(df: pd.DataFrame, regressor_name:str) -> np.ndarray:
    regressors_list = ['stimIntensity', 'contrastLeft', 'contrastRight', 
                       'correctSide', 'previousChoice', 'previousRewarded']
    if regressor_name  not in regressors_list:
        raise Exception('Regressor name invalid or not implemented.')
    
    if regressor_name == 'stimIntensity':
        stim_intensity = df['contrastRight'] - df['contrastLeft']
        data = z_score(stim_intensity)
    elif regressor_name == 'contrastLeft':
        p=5 # as used in Psytrack paper
        data = np.tanh(p*df['contrastLeft'])/np.tanh(p) # tanh transformation of left contrasts
    elif regressor_name == 'contrastRight':
        p=5 # as used in Psytrack paper
        data = np.tanh(p*df['contrastRight'])/np.tanh(p) # tanh transformation of right contrasts
    elif regressor_name == 'correctSide':
        data = np.array(df['correctSide'])
    elif regressor_name == 'previousChoice':
        y = np.array(df['choice'])
        data_temp = y[0:-1]
        data = np.concatenate(([0.0], data_temp))
    elif regressor_name == 'previousRewarded':
        data_temp = np.array(df['correctSide'])[0:-1] # previous rewarded
        data = np.concatenate(([0.0], data_temp))
    
    assert len(data) == len(df)
    return data

def get_mouse_design(
        dfAll: pd.DataFrame, subject: str, regressors=Optional[List[str]], sess_stop: int=-1,
        ) -> Tuple[np.ndarray, np.ndarray, List]:
    '''
    Returns design matrix X and vector Y of outputs (decisions) for a given subject. 
    The regressors that consistute the design matrix are passed as a list. Options include:
        ['stimIntensity', 'contrastLeft', 'contrastRight', 'correctSide', 'previousChoice', 'previousRewarded']
    '''
    data = dfAll[dfAll['subject']==subject]   # Restrict data to the subject specified
    default_regressors = ['stimIntensity']

    # Keep specified number of sessions
    dates_to_keep = np.unique(data['date'])[0:sess_stop]
    data_temp = pd.DataFrame(data.loc[data['date'].isin(list(dates_to_keep))])

    # Design and choice matrices
    if regressors is None:
        regressors = default_regressors
    X = np.zeros((data_temp.shape[0], len(regressors)))
    Y = np.array(data_temp['choice'])

    X[:,0] = 1. # bias
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

if __name__=='__main__':
    df = pd.read_csv('./data/ibl_learning_processed.csv')
    df = df[df['lab']=='wittenlab']
    print(get_mouse_design(df, subject='ibl_witten_02', 
                           regressors=['contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']))