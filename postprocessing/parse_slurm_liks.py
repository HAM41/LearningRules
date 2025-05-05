import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
import itertools
from scipy import stats

slurm_outs_path = '/home/vg0233/PillowLab/LearningRules/slurm_outs/'

def rename_model(name):
    if name == 'TimeVarGLMLearn':
        return 'dRL'
    elif name == 'GLMLearn':
        return 'RL'
    elif name == 'Psytrack':
        return 'Psytrack'
    elif name == 'QLearning':
        return 'QL'
    elif name == 'TimeVarRVBF':
        return 'dRVBF'
    else:
        return name
    
def safe_float(value):
    """
    Convert a value to float, returning NaN if conversion fails.
    
    Parameters:
        value (str): The value to convert.
    
    Returns:
        float: The converted float value or NaN if conversion fails.
    """
    try:
        return float(value)
    except ValueError:
        return math.nan
    
def parse_float(line, pattern):
    regex = rf"{pattern}:\s*([-\w.]+)"
    match = re.search(regex, line)
    if match:
        value = match.group(1)
        if value.lower() == "nan":
            return math.nan
        return safe_float(value)
    return math.nan


def parse_line(line, block):
    """
    Parse a single line and update the block dictionary with values if patterns match.
    
    Parameters:
        line (str): A line from the log file.
        block (dict): Dictionary holding extracted data for the current block.
    
    Returns:
        bool: True if the current block is complete (i.e. the prediction line was found), False otherwise.
    """
    # Extract model name from the Arguments line
    if "Arguments: Namespace(" in line:
        # Extract model_class value
        m_model = re.search(r"model_class='([^']+)'", line)
        if m_model:
            block['model'] = rename_model(m_model.group(1))
        
        # Extract subject_id value
        m_subject = re.search(r"subject_id=([^,)\s]+)", line)
        if m_subject:
            try:
                block['subject_id'] = int(m_subject.group(1))
            except ValueError:
                block['subject_id'] = m_subject.group(1)

        # Extract posterior seed
        m_seed = re.search(r"posterior_seed=([-\d]+)", line)
        if m_seed:
            block['posterior_seed'] = int(m_seed.group(1))

        # Extract modulator
        m_modulator = re.search(r"modulator='([^']+)'", line)
        if m_modulator and block['model'] == 'dRVBF':
            block['model'] += "-"+m_modulator.group(1)[0]

        # Extract lab name
        m_lab = re.search(r"lab='([^']+)'", line)
        if m_lab:
            block['lab'] = m_lab.group(1)

    
    # Extract marginal log-likelihood per trial
    elif "Complete trajectory 'marginal_log_likelihood'" in line:
        m = re.search(r"avg L per trial: \s*([-\d.]+)", line)
        if m:
            block['marginal_L_per_trial'] = safe_float(m.group(1))

        m = re.search(r"'marginal_log_likelihood': \s*([-\d.]+)", line)
        if m:
            block['marginal_LL'] = safe_float(m.group(1))
        
    elif "Held out trials 'marginal_log_likelihood'" in line:
        m = re.search(r"avg L per trial: \s*([-\d.]+)", line)
        if m:
            block['test_marginal_L_per_trial'] = safe_float(m.group(1))

        m = re.search(r"'marginal_log_likelihood': \s*([-\d.]+)", line)
        if m:
            block['test_marginal_LL'] = safe_float(m.group(1))
    
    # Extract forward pass loglik per trial
    elif "Forward pass" in line:
        # m = re.search(r"avg L per trial: \s*([-\d.]+)", line)
        # if m:
        #     block['forward_L_per_trial'] = safe_float(m.group(1))

        # m = re.search(r"loglik: \s*([-\d.]+)", line)
        # if m:
        #     block['forward_LL'] = safe_float(m.group(1))
        block['forward_L_per_trial'] = parse_float(line, "avg L per trial")
        block['forward_LL'] = parse_float(line, "loglik")
    
    # Extract prediction loglik per trial and mark block as complete
    elif "Prediction" in line:
        block['prediction_L_per_trial'] = parse_float(line, "avg L per trial")
        block['prediction_LL'] = parse_float(line, "loglik")
        return True  # End of current block
    
    # # Extract SNRs
    # elif "SNRs:" in line:
    #     m = re.search(r"SNRs:\s*\[([^\]]+)\]", line)
    #     if m:
    #         try:
    #             block['SNRs'] = [float(x) for x in m.group(1).split(',')]
    #         except ValueError:
    #             block['SNRs'] = []

    # Extract noise component

    # elif "Noise component" in line:
    #     m = re.search(r"norm: \s*([-\d.]+)", line)
    #     print(m)
    #     if m:
    #         block['noise_norm'] = safe_float(m.group(1))
    
    return False

def compute_differences(df, baseline_model='psytrack'):
    """
    For each subject_id, compute the difference in log-likelihood metrics relative to Psytrack.
    
    For every subject_id group, this function subtracts the Psytrack values from the
    'marginal', 'forward', and 'prediction' columns for all models.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns: subject_id, model, marginal, forward, prediction.
    
    Returns:
        pd.DataFrame: Updated DataFrame with new columns for the differences.
                      The new columns are 'marginal_diff', 'forward_diff', and 'prediction_diff'.
    """
    baseline_model = baseline_model.lower()
    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()
    # Create a helper column with lower-case model names for comparison
    df['model_lower'] = df['model'].str.lower()

    def subtract_baseline_model(group):
        # Find the row corresponding to Psytrack for this subject
        baseline_row = group[group['model_lower'] == baseline_model]
        if baseline_row.empty:
            # If there's no baseline_model entry, set differences to NaN.
            group['marginal_LL_diff'] = pd.NA
            group['test_marginal_LL_diff'] = pd.NA
            group['forward_LL_diff'] = pd.NA
            group['prediction_LL_diff'] = pd.NA
        else:
            base = baseline_row.iloc[0]
            group['marginal_LL_diff'] = group['marginal_LL'] - base['marginal_LL']
            group['test_marginal_LL_diff'] = group['test_marginal_LL'] - base['test_marginal_LL']
            group['forward_LL_diff'] = group['forward_LL'] - base['forward_LL']
            group['prediction_LL_diff'] = group['prediction_LL'] - base['prediction_LL']
        return group

    df = df.groupby(['subject_id', 'lab'], group_keys=False).apply(subtract_baseline_model)
    df.drop(columns=['model_lower'], inplace=True)
    return df

def parse_file(filename):
    """
    Parse the log file and extract data for each block.
    
    Parameters:
        filename (str): The path to the log file.
    
    Returns:
        list: A list of dictionaries, each containing extracted data from one block.
    """
    data = []
    block = {}
    with open(filename, 'r') as f:
        for line in f:
            # Update the block dictionary with data from the current line.
            block_complete = parse_line(line, block)
            if block_complete:
                data.append(block)
                block = {}  # Reset block for the next set of log lines
    return data

def main():
    # filenames = ['slurm-62440998.out', 'slurm-62441358.out', 'slurm-62441487.out']
    # filenames = ['psytrack_churchland_liks.txt', 'glmlearn_churchland_liks.txt', 'timevarglmlearn_churchland_liks.txt', 'ac_churchland_liks.txt']
    # filenames = ['slurm-62675717.out', 'qlearning_churchland_v2_liks.txt'] 

    # filenames = [f'slurm-63457070/{i}.err' for i in range(6)] + [f'slurm-63457066/{i}.err' for i in range(9)] + [f'slurm-63454786/{i}.err' for i in range(16)] + \
    #             [f'slurm-63454822/{i}.err' for i in range(6)] + [f'slurm-63454762/{i}.err' for i in range(9)] + [f'slurm-63454792/{i}.err' for i in range(16)] + \
    #             [f'slurm-63473769/{i}.err' for i in range(6)] + [f'slurm-63473706/{i}.err' for i in range(9)] + [f'slurm-63473750/{i}.err' for i in range(16)]
    filenames = [
        # 'slurm-63994008.out', # 'slurm-63502940.out', TimeVarRVBF-a, churchland
        # 'slurm-63994013.out', # slurm-63503279 TimeVarRVBF-a, angelaki
        # 'slurm-63994003.out', # slurm-63503285, TimeVarRVBF-a, witten
        '', #  TimeVarRVBF-a, w0
        'slurm-63502859.out', # RVBF, churchland
        'slurm-64090174.out', # slurm-63503268  TimeVarRVBF-b, churchland
        'slurm-64090175.out', # TimeVarRVBF-b, witten
        'slurm-64090177.out', # slurm-63503273, TimeVarRVBF-b, angelaki
        'slurm-63503292.out', # RVBF, witten
        'slurm-63503296.out', # RVBF, angelaki
        'slurm-63504647.out', # psytrack
        'slurm-63504774.out', # glmlearn
        'slurm-64090544.out', # HRL
        ]
        # [f'slurm-63709636/{i}.err' for i in range(16)] # HRL

    # No curriculum
    # filenames = [f'slurm-63567996/{i}.err' for i in range(10)] + [f'slurm-63567978/{i}.err' for i in range(10)] + [f'slurm-63577147/{i+10}.err' for i in range(10)] # OLD
    # filenames = [f'slurm-63781462/{i}.err' for i in range(30)] 

    # # No curriculum, [RVBF, dRVBF-b, dRVBF-a]
    # filenames = [f'slurm-63844696/{i}.err' for i in range(30)] +\
    #     [f'slurm-63844649/{i}.err' for i in range(30)] +\
    #     [f'slurm-63781462/{i}.err' for i in range(30)]

    # Create dataframe
    data = []
    for filename in filenames:
        try:
            data += parse_file(slurm_outs_path + filename)
        except FileNotFoundError:
            print(f"File {filename} not found. Skipping.")
            continue
    df_full = pd.DataFrame(data) #.dropna()
    print('dataframe: ', df_full)
    for model in df_full['model'].unique():
        for lab in df_full['lab'].unique():
            print(f"{model} ({lab}): {df_full.query(f'model == \"{model}\" and lab == \"{lab}\"')['subject_id'].nunique()} subjects")
    # print(df_full['SNRs'])
    
    try:
        df = df_full.drop(columns=['SNRs', 'noise_norm'])
    except KeyError:
        df = df_full.copy()

    # Average over posterior seeds
    df = df.groupby(['model', 'subject_id', 'lab']).mean().reset_index()

    # Compute difference with Psytrack per model and subject_id
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # print(df.query("lab == 'churchlandlab' and model == 'RVBF'")['test_marginal_LL'].values - df.query("lab == 'churchlandlab' and model == 'Psytrack'")['test_marginal_LL'].values)
    # print(df.query("lab == 'churchlandlab' and model == 'dRVBF-b'")['test_marginal_LL'].values - df.query("lab == 'churchlandlab' and model == 'Psytrack'")['test_marginal_LL'].values)
    df = compute_differences(df, baseline_model='HRL')

    # Plot
    fig, axs = plt.subplots(figsize=[10,3], ncols=4, constrained_layout=True)
    # for i, label in enumerate(['marginal_L_per_trial', 'test_marginal_L_per_trial', 'forward_L_per_trial', 'prediction_L_per_trial']):
    for i, label in enumerate(['marginal_LL_diff', 'test_marginal_LL_diff', 'forward_LL_diff', 'prediction_LL_diff']):
        sns.boxplot(x='model', y=label, data=df, ax=axs[i], boxprops=dict(facecolor='white'), showfliers=False)
        # sns.swarmplot(x='model', y=label, data=df, ax=axs[i], color='black', size=2, alpha=0.5)
        axs[i].set_title(label)
    for ax in axs:
        ax.tick_params(axis='x', labelsize=6)

    # for ax in axs:
    #     # ax.set_yscale('log')
    #     # ax.axhline(0.5, ls='-', color='red', alpha=0.5, zorder=-1)   
    #     sns.despine(ax=ax)
    # axs[0].set_ylabel('Likelihood')
    plt.savefig('./postprocessing/figures/TimeVarRVBF_Ls_per_trial.png', dpi=300)

    # Statistical tests
    # List of log-likelihood metrics to test
    metrics = ['marginal_LL', 'test_marginal_LL', 'forward_LL', 'prediction_LL']

    # Get the unique models in the dataframe
    models = df['model'].unique()
    # models = ['RVBF', 'dRVBF-b', 'dRVBF-l']

    # Dictionary to hold results
    results = {}

    # For each metric and for every two distinct models, perform paired one-tailed t-tests (testing if model_1 > model_2)
    for metric in metrics:
        for model1, model2 in itertools.permutations(models, 2):
            # Merge data on subject_id and lab so that the comparison is paired
            df1 = df[df['model'] == model1][['subject_id', 'lab', metric]]
            df2 = df[df['model'] == model2][['subject_id', 'lab', metric]]
            merged = pd.merge(df1, df2, on=['subject_id', 'lab'], suffixes=(f'_{model1}', f'_{model2}'))
            merged = merged.dropna(subset=[f'{metric}_{model1}', f'{metric}_{model2}'])
            if merged.empty:
                p_val = None
            else:
                stat, p_val = stats.ttest_rel(merged[f'{metric}_{model1}'], merged[f'{metric}_{model2}'])
                # One-tailed test: check if model1 > model2.
                if stat > 0:
                    p_val = p_val / 2
                else:
                    p_val = 1.0
            results[(model1, model2, metric)] = p_val

    # Report the p values for each comparison
    print("Statistical test p-values (paired t-test), showing p < 0.05:")
    for (model1, model2, metric), p_val in results.items():
        if p_val < 0.05:
            print(f"Comparison {model1:8s} > {model2:8s} for {metric}: p = {p_val:.4f}")

    # # ---------------------
    # # SNRs

    # df_SNRs_expanded = pd.DataFrame(
    #     [(row['model'], snr, i) for _, row in df_full.iterrows() for i, snr in enumerate(row['SNRs'])],
    #     columns=['model', 'SNR', 'Regressor']
    # )

    # # snr_id = 2
    # # df_full[f'snr_{snr_id}'] = df_full['SNRs'].apply(lambda x: x[snr_id] if isinstance(x, list) and len(x) > 0 else np.nan)
    # fig, ax = plt.subplots(figsize=(6,5))
    # sns.boxplot(x='model', y='SNR', hue='Regressor', data=df_SNRs_expanded, ax=ax)#, boxprops=dict(facecolor='white'), showfliers=False)
    # plt.title(f'SNR (S/R) per regressor per model')
    # ax.set_yscale('log')
    # ax.axhline(1.0, ls='-', color='b', alpha=0.5, zorder=-1)
    # plt.savefig(f'./postprocessing/figures/SNRs_per_model.png', dpi=300)

    # # ---------------------

    # fig, ax = plt.subplots(figsize=[4,4], constrained_layout=True)
    # ax.plot(df.query('model == "Psytrack"')['marginal'], df.query('model == "RL"')['marginal'], 'o', label='RL')
    # ax.plot(df.query('model == "Psytrack"')['marginal'], df.query('model == "QL"')['marginal'], 'o', label='QL')
    # ax.plot(
    #     [min(df.query('model == "Psytrack"')['marginal']), max(df.query('model == "Psytrack"')['marginal'])], 
    #     [min(df.query('model == "Psytrack"')['marginal']), max(df.query('model == "Psytrack"')['marginal'])], 
    #     'k--', zorder=-1)
    # ax.set_xlabel('Psytrack marginal log-likelihood')
    # ax.set_ylabel('Model marginal log-likelihood')
    # ax.legend()
    # plt.savefig('./postprocessing/figures/marginal_loglik_psytrack_vs_rl.png', dpi=300)


if __name__ == '__main__':
    main()