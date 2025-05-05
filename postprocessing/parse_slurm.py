import re 
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import ttest_1samp, ttest_rel

def parse_loglik(file):
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r"Final best log-lik: (-?\d+\.\d+)", line)
            if match:
                return float(match.group(1))
    return np.nan

def parse_T(file):
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r"T=(\d+)", line)
            if match:
                return int(match.group(1))
    return np.nan

def parse_params(file, params_name):
    with open(file, 'r') as f:
        for line in f:
            match = re.search(f"Final best params: ({params_name}"+r"\(.*\))", line)
            if match:
                return match.group(1)
    return None

def parse_model(file):
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r"Model: (\w+\(.*\))", line)
            if match:
                return match.group(1)
    return None

def parse_totaloglik(file):
    out = []
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r"Total log-likelihood: (-?\d+\.\d+)", line)
            match2 = re.search(r"Complete trajectory log-likelihood: (-?\d+\.\d+)", line)
            match3 = re.search(r"Filtering log-likelihood: (-?\d+\.\d+)", line)
            if match:
                 out.append(float(match.group(1)))
            elif match2:
                out.append(float(match2.group(1)))
            elif match3:
                out.append(float(match3.group(1)))
    return out[:2]

def parse_optim_loglik(file):
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r"Best: MLL: (-?\d+\.\d+)", line)
            if match:
                return float(match.group(1))
    return None

def parse_noise_norm(file):
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r"INFO - Noise component norm: (-?\d+\.\d+)", line)
            if match:
                return float(match.group(1))
    return np.nan

import ast
def parse_ParamsReg(params_str):
    # Regular expression to match the pattern for floats and arrays, excluding ", dtype=float32"
    # pattern = re.compile(r"Array\((.*?)(?:, dtype=float32)?\)")
    # pattern = re.compile(r"Array\((\[.*?\]|-?\d+\.?\d*)\)")
    pattern = re.compile(r"Array\((.*?)\s*,\s*dtype=float32\)", re.DOTALL)

    # Find all matches
    matches = pattern.findall(params_str)

    # Convert matches to floats or lists of floats
    result = []
    lengths = []
    for match in matches:
        try:
            if match == 'nan':
                result.append(np.nan)
                lengths.append(1)
                continue
            # Use ast.literal_eval to safely evaluate the string as a Python literal
            value = ast.literal_eval(match)
            if isinstance(value, (float, int)):
                result.append(value)
                lengths.append(1)
            elif isinstance(value, list):
                for val in value:
                    result.append(val)
                lengths.append(len(value))
        except (ValueError, SyntaxError):
            print(f"Failed to parse {match}")
            continue
    return result, lengths

def parse_final_params_ci(text):
    """
    Extract the 2D array of confidence intervals from a log string.

    Args:
        text (str): The complete text of the log file.

    Returns:
        List[List[float]]: A list of [lower, upper] pairs.
    """
    # 1) Locate the block inside the double square-brackets after "Final params CI:"
    m = re.search(r'Final params CI:\s*\[\[(.*?)\]\]', text, re.DOTALL)
    if not m:
        raise ValueError("Couldn't find a 'Final params CI' block in the input text.")
    block = m.group(1)

    # 2) Grab every floating-point number (including negatives and decimals)
    nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', block)
    nums = [float(n) for n in nums]

    # 3) Group into rows of two
    if len(nums) % 2 != 0:
        raise ValueError("Expected an even number of floats; got %d." % len(nums))
    ci = [nums[i:i+2] for i in range(0, len(nums), 2)]

    return ci


def parse_CI(filepath):
    """
    Read the entire file at `filepath`, then extract the CI matrix.
    """
    with open(filepath, 'r') as f:
        text = f.read()
    try:
        return parse_final_params_ci(text)
    except ValueError as e:
        return None

# def parse_CI(file):
#     with open(file, 'r') as f:
#         for line in f:
#             match = re.search(r"Final params CI: \[\[(\d+\.\d+), (\d+\.\d+)\], \[(\d+\.\d+), (\d+\.\d+)\]\]", line)
#             if match:
#                 return match.group(1)
#     return None

def parse_alpha_CI(file):
    with open(file, 'r') as f:
        for line in f:
            match = re.search(r"alpha_2: mean = (-?\d+\.\d+), med = (-?\d+\.\d+), CI = \[(-?\d+\.\d+), (-?\d+\.\d+)\]", line)
            if match:
                return np.array([float(match.group(3)), float(match.group(4))])
    return None

def parse_lab(file):
    with open(file, 'r') as f:
        for line in f:
            # Parse from "Arguments: Namespace(lab='churchlandlab',"
            match = re.search(r"Arguments: Namespace\(lab='(\w+)", line)
            if match:
                return match.group(1)
    return None

if __name__=='__main__':    
    slurm_outs_path = '/home/vg0233/PillowLab/LearningRules/slurm_outs/'
    file_name_dicts_old = [
        {'model': 'Psytrack()', 'folder': 'slurm-59678306', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'wittenlab'},
        {'model': 'Psytrack()', 'folder': 'slurm-59856144', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'angelakilab'},
        {'model': 'Psytrack()', 'folder': 'slurm-59856167', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'churchlandlab'},
        # {'model': 'QLearning()', 'folder': 'slurm-60520956', 'params_name': 'ParamsQLearning', 'label':'QL', 'lab': 'wittenlab'},
        # {'model': 'QLearning()', 'folder': 'slurm-60521783', 'params_name': 'ParamsQLearning', 'label':'QL', 'lab': 'angelakilab'},
        # {'model': 'QLearning()', 'folder': 'slurm-60521800', 'params_name': 'ParamsQLearning', 'label':'QL', 'lab': 'churchlandlab'},
        {'model': 'GLMLearn(policy_gradient)', 'folder': 'slurm-59678305', 'params_name': 'ParamsGLMLearn', 'label':'RL-PG', 'lab': 'wittenlab'},
        {'model': 'GLMLearn(policy_gradient)', 'folder': 'slurm-59859138', 'params_name': 'ParamsGLMLearn', 'label':'RL-PG', 'lab': 'angelakilab'},
        {'model': 'GLMLearn(policy_gradient)', 'folder': 'slurm-59859132', 'params_name': 'ParamsGLMLearn', 'label':'RL-PG', 'lab': 'churchlandlab'},
        # {'model': 'GLMLearn(policy_gradient)', 'folder': 'slurm-59686208', 'params_name': 'ParamsGLMRegLearn', 'label':'RL-PG-vec'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-59752871', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'wittenlab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-59852940', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'angelakilab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-59859129', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'churchlandlab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-59755153', 'params_name': 'ParamsGLMLearn', 'label':'RL-R-vec', 'lab': 'wittenlab'}, #slightly unsure about this one
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-60230939', 'params_name': 'ParamsGLMLearn', 'label':'RL-R-vec', 'lab': 'angelakilab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-60230934', 'params_name': 'ParamsGLMLearn', 'label':'RL-R-vec', 'lab': 'churchlandlab'},
        # {'model': 'GLMRegLearn()', 'folder': 'slurm-59726416', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg'},
        # {'model': 'GLMRegLearn()', 'folder': 'slurm-59821794', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'wittenlab'},
        # {'model': 'GLMRegLearn()', 'folder': 'slurm-59877065', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'wittenlab'}, # changed to diag(Q)
        {'model': 'GLMRegLearn()', 'folder': 'slurm-59884496', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'wittenlab'}, # Diag(Q), baseline
        # {'model': 'GLMRegLearn()', 'folder': 'slurm-59851865', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'angelakilab'},
        # {'model': 'GLMRegLearn()', 'folder': 'slurm-59875909', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'angelakilab'}, # changed to diag(Q)
        {'model': 'GLMRegLearn()', 'folder': 'slurm-59884649', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'angelakilab'}, # Diag(Q), baseline
        # {'model': 'GLMRegLearn()', 'folder': 'slurm-59850900-2', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'churchlandlab'}, 
        # {'model': 'GLMRegLearn()', 'folder': 'slurm-59877064', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'churchlandlab'}, # changed to diag(Q)
        {'model': 'GLMRegLearn()', 'folder': 'slurm-59884934', 'params_name': 'ParamsGLMRegLearn', 'label':'RLReg', 'lab': 'churchlandlab'}, # Diag(Q), baseline
        # {'model': 'TimeVarGLMLearn(lapse=False)', 'folder': 'slurm-59737337', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL'}, # 20 iters optim
        # {'model': 'TimeVarGLMLearn(lapse=True)', 'folder': 'slurm-59737845', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'Lapse'},
        # {'model': 'TimeVarGLMLearn(lapse=False)', 'folder': 'slurm-59741856', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-PG'}, # 200 iters optim
        # {'model': 'TimeVarGLMLearn(lapse=False)', 'folder': 'slurm-59742118', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-PG-vec'},
        # {'model': 'TimeVarGLMLearn(lapse=False)', 'folder': 'slurm-59752861', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-R-vec'},
        # {'model': 'TimeVarGLMLearn(lapse=True)', 'folder': 'slurm-59756743', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'Lapse-R'},
        # {'model': 'GLMHMMLearn(learning_rule=reinforce)', 'folder': 'slurm-59757310', 'params_name': 'ParamsGLMHMMLearn', 'label':'HMM-R'},
        # {'model': 'TimeVarGLMLearn(learning_rule=reinforce)', 'folder': 'slurm-60046747', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-R-vec-bf', 'lab': 'wittenlab'},
        # {'model': 'TimeVarGLMLearn(learning_rule=reinforce)', 'folder': 'slurm-60050644', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-R-vec-bf', 'lab': 'angelakilab'},
        # {'model': 'TimeVarGLMLearn(learning_rule=reinforce)', 'folder': 'slurm-60050640', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-R-vec-bf', 'lab': 'churchlandlab'},
        # {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-60202966', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-Reg-bf', 'lab': 'wittenlab'},
        # {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-60203908', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-Reg-bf', 'lab': 'angelakilab'},
        # {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-60203913', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-Reg-bf', 'lab': 'churchlandlab'},
        # {'model': 'GLMBaseLearn(time_var=True)', 'folder': 'slurm-60630888', 'params_name': 'ParamsGLMBaseLearn', 'label':'R-b-Var', 'lab': 'wittenlab'},
        # {'model': 'GLMBaseLearn(time_var=True)', 'folder': 'slurm-60631402', 'params_name': 'ParamsGLMBaseLearn', 'label':'R-b-Var', 'lab': 'angelakilab'},
        # {'model': 'GLMBaseLearn(time_var=True)', 'folder': 'slurm-60631379', 'params_name': 'ParamsGLMBaseLearn', 'label':'R-b-Var', 'lab': 'churchlandlab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=5)', 'folder': 'slurm-60748048', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-Reg-vec-bf', 'lab': 'wittenlab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=5)', 'folder': 'slurm-60748057', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-Reg-vec-bf', 'lab': 'angelakilab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=5)', 'folder': 'slurm-60748046', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-Reg-vec-bf', 'lab': 'churchlandlab'},
        # {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-60815547', 'params_name': 'ParamsGLMLearn', 'label':'RL-R-EM', 'lab': 'wittenlab'},
        # {'model': 'Psytrack()', 'folder': 'slurm-60815579', 'params_name': 'ParamsPsytrack', 'label':'Psytrack-EM', 'lab': 'wittenlab'},
    ]

    # Including noise norm
    file_name_dicts_training = [
        {'model': 'Psytrack()', 'folder': 'slurm-61828749', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'wittenlab'},
        {'model': 'Psytrack()', 'folder': 'slurm-61840618', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'churchlandlab'},
        {'model': 'Psytrack()', 'folder': 'slurm-63844490', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'angelakilab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-61735144', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'wittenlab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-61840608', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'churchlandlab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-63844494', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'angelakilab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-61735146', 'params_name': 'ParamsGLMLearn', 'label':'RL-R-vec', 'lab': 'wittenlab'},
        {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-61832314', 'params_name': 'ParamsGLMLearn', 'label':'RL-R-vec', 'lab': 'churchlandlab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-61735059', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL', 'lab': 'wittenlab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-61840617', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL', 'lab': 'churchlandlab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-62364458', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL', 'lab': 'angelakilab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=5)', 'folder': 'slurm-61828727', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-vec', 'lab': 'wittenlab'},
        {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=5)', 'folder': 'slurm-61840616', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-vec', 'lab': 'churchlandlab'},
        {'model': 'TimeVarGLMLearn(lapse=True, beta_dim=1)', 'folder': 'slurm-61828807', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-lapse', 'lab': 'wittenlab'},
        {'model': 'TimeVarGLMLearn(lapse=True, beta_dim=1)', 'folder': 'slurm-61832306', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL-lapse', 'lab': 'churchlandlab'},
        # {'model': 'AC(beta_dim=1)', 'folder': 'slurm-62154062', 'params_name': 'ParamsAC', 'label':'AC', 'lab': 'wittenlab'}, # slurm-62154062, slurm-61984520
        # {'model': 'AC(beta_dim=5, sigmoid=True)', 'folder': 'slurm-61995108', 'params_name': 'ParamsAC', 'label':'AC-sigmoid', 'lab': 'churchlandlab'},
        # {'model': 'AC(beta_dim=5)', 'folder': 'slurm-61970338', 'params_name': 'ParamsAC', 'label':'AC-vec', 'lab': 'wittenlab'},
        # {'model': 'AC(beta_dim=5)', 'folder': 'slurm-61984517', 'params_name': 'ParamsAC', 'label':'AC-vec', 'lab': 'churchlandlab'},
        # {'model': 'AC(beta_dim=5, sigmoid=False)', 'folder': 'slurm-62154062', 'params_name': 'ParamsAC', 'label':'AC', 'lab': 'wittenlab'},
        # {'model': 'AC(beta_dim=5, sigmoid=False)', 'folder': 'slurm-62164851', 'params_name': 'ParamsAC', 'label':'AC', 'lab': 'churchlandlab'},
        # {'model': 'QLearning()', 'folder': 'slurm-62025462', 'params_name': 'ParamsQLearning', 'label':'TDRL', 'lab': 'wittenlab'}, #! re-run -- taken from with filter below
        {'model': 'QLearning()', 'folder': 'slurm-62861103', 'params_name': 'ParamsQLearning', 'label':'TDRL', 'lab': 'churchlandlab'},
        # {'model': 'RVBF()', 'folder': 'slurm-62897457', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'wittenlab'},
        # {'model': 'RVBF()', 'folder': 'slurm-62897458', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'churchlandlab'},
        # {'model': 'RVBF()', 'folder': 'slurm-62909732', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'angelakilab'},
        {'model': 'RVBF()', 'folder': 'slurm-64092569', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'wittenlab'}, # with w0
        {'model': 'RVBF()', 'folder': 'slurm-64092578', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'churchlandlab'}, # with w0
        {'model': 'RVBF()', 'folder': 'slurm-64092600', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'angelakilab'}, # with w0
        {'model': 'AC(beta_dim=5, sigmoid=False)', 'folder': 'slurm-63377415', 'params_name': 'ParamsAC', 'label':'AC', 'lab': 'angelakilab'},
        {'model': 'AC(beta_dim=5, sigmoid=False)', 'folder': 'slurm-63377414', 'params_name': 'ParamsAC', 'label':'AC', 'lab': 'wittenlab'},
        {'model': 'AC(beta_dim=5, sigmoid=False)', 'folder': 'slurm-63376705', 'params_name': 'ParamsAC', 'label':'AC', 'lab': 'churchlandlab'},
        # {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63429466', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'churchlandlab'},
        # {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63434568', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'wittenlab'},
        # {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63434732', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'angelakilab'},
        # {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63960290', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'churchlandlab'}, # --initialize-at-learned
        # {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63981335', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'wittenlab'}, # --initialize-at-learned
        # {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63981336', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'angelakilab'},# --initialize-at-learned
        {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-64097586', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'churchlandlab'}, # --initialize-at-learned, w0
        {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-64097587', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'wittenlab'}, # --initialize-at-learned, w0
        {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-64097585', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'angelakilab'},# --initialize-at-learned, w0
        # {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63429367', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'churchlandlab'},
        {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63994423', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'churchlandlab'}, # --initialize-at-learned
        # {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63434579', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'wittenlab'},
        {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63994437', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'wittenlab'}, # --initialize-at-learned
        # {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63434721', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'angelakilab'},
        {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63994433', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'angelakilab'}, # --initialize-at-learned
        {'model': 'HRL()', 'folder': 'slurm-63609314', 'params_name': 'ParamsHRL', 'label':'HRL', 'lab': 'churchlandlab'},
        {'model': 'HRL()', 'folder': 'slurm-63709874', 'params_name': 'ParamsHRL', 'label':'HRL', 'lab': 'wittenlab'},
        {'model': 'HRL()', 'folder': 'slurm-63709885', 'params_name': 'ParamsHRL', 'label':'HRL', 'lab': 'angelakilab'},
    ]
    for entry in file_name_dicts_training:
        entry['protocol'] = 'training'

    # file_name_dicts_nocurriculum = [ # OLD unsorted
    #     {'model': 'Psytrack()', 'folder': 'slurm-63555221', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'wittenlab'},
    #     {'model': 'RVBF()', 'folder': 'slurm-63556547', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'wittenlab'},
    #     {'model': 'RVBF()', 'folder': 'slurm-63570193', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'wittenlab'}, # subjects 10-30
    #     {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63556546', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'wittenlab'},
    #     {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63570187', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'wittenlab'}, # subjects 10-30
    # ]

    file_name_dicts_nocurriculum = [
        {'model': 'Psytrack()', 'folder': 'slurm-63773990', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'wittenlab'},
        {'model': 'Psytrack()', 'folder': 'slurm-63844578', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'wittenlab'}, # subjects 11-30
        {'model': 'RVBF()', 'folder': 'slurm-63764770', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'wittenlab'},
        {'model': 'RVBF()', 'folder': 'slurm-63774149', 'params_name': 'ParamsRVBF', 'label':'RVBF', 'lab': 'wittenlab'}, # subjects 11-30
        {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63764763', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'wittenlab'},
        {'model': 'TimeVarRVBF(modulator=lr, beta_dim=1)', 'folder': 'slurm-63774125', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-a', 'lab': 'wittenlab'}, # subjects 11-30
        # {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63787254', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'wittenlab'},
        {'model': 'TimeVarRVBF(modulator=baseline, beta_dim=5)', 'folder': 'slurm-63787254', 'params_name': 'ParamsTimeVarRVBF', 'label':'RVBF-b', 'lab': 'wittenlab'}, # subjects 0-30
    ]
    for entry in file_name_dicts_nocurriculum:
        entry['protocol'] = 'no_curriculum'

    file_name_dicts = file_name_dicts_training + file_name_dicts_nocurriculum

    # # Stim intensity regresors only. QLearning is unchanged.
    # file_name_dicts = [
    #     {'model': 'Psytrack()', 'folder': 'slurm-62652442', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'churchlandlab'},
    #     {'model': 'QLearning()', 'folder': 'slurm-62861103', 'params_name': 'ParamsQLearning', 'label':'TDRL', 'lab': 'churchlandlab'},
    #     {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-62652434', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'churchlandlab'},
    # ]

    # # With filter
    # file_name_dicts = [
    #     {'model': 'Psytrack()', 'folder': 'slurm-62025450', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'wittenlab'},
    #     {'model': 'Psytrack()', 'folder': 'slurm-62028397', 'params_name': 'ParamsPsytrack', 'label':'Psytrack', 'lab': 'churchlandlab'},
    #     {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-62025459', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'wittenlab'},
    #     {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-62028400', 'params_name': 'ParamsGLMLearn', 'label':'RL-R', 'lab': 'churchlandlab'},
    #     {'model': 'GLMLearn(reinforce)', 'folder': 'slurm-62025470', 'params_name': 'ParamsGLMLearn', 'label':'RL-R-vec', 'lab': 'wittenlab'},
    #     {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-62025622', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL', 'lab': 'wittenlab'},
    #     {'model': 'TimeVarGLMLearn(lapse=False, beta_dim=1)', 'folder': 'slurm-62028402', 'params_name': 'ParamsTimeVarGLMLearn', 'label':'IRL', 'lab': 'churchlandlab'},
    # ]
    all_log_liks = []
    all_log_liks_mean = []
    models = []
    all_params = [] 
    entries = []
    for file_name_dict in file_name_dicts:
        model = file_name_dict['model']
        folder = file_name_dict['folder']
        params_name = file_name_dict['params_name']
        # print(model, folder)

        models.append(file_name_dict['label'])
        log_liks = []
        log_liks_mean = []
        Ts = []
        for i in range(31):
            file = f'/home/vg0233/PillowLab/LearningRules/slurm_outs/{folder}/{i}.err'
            if not os.path.exists(file):
                # print(f'File {file} not found')
                continue
            if model != parse_model(file):
                print(parse_model(file))
            T = parse_T(file)
            params_str = parse_params(file, params_name)
            if params_str is not None:
                params_array, params_lengths = parse_ParamsReg(params_str)
            else:
                params_array, params_lengths = None, None
            log_lik_map = parse_loglik(file)
            log_lik_optim = parse_optim_loglik(file)

            noise_norm = parse_noise_norm(file) / T

            log_lik = log_lik_map#, log_lik_optim, log_lik_mean, log_lik_med)
            log_lik = log_lik if log_lik is not None else np.nan

            # print(i, params)
            # print(f'[{i}] T {T}, log_lik {log_lik:.2f}, per trial: {log_lik/T:.4f}')#, params: {params_str}')
            # if model == 'GLMRegLearn()':
            #     try:
            #         all_params.append(parse_ParamsReg(params_str))
            #     except TypeError:
            #         print(f'Failed to parse {params_str}')
            #         continue
            
            lab = parse_lab(file)
            # if lab == 'wittenlab':
            try:
                log_lik_mean, log_lik_med = tuple(parse_totaloglik(file))
            except ValueError:
                log_lik_mean, log_lik_med = np.nan, np.nan
            #     log_liks.append(log_lik)
            #     log_liks_mean.append(log_lik_mean)
            #     Ts.append(T)

            CI = parse_CI(file)

            lls_dict = {'log_lik_map': log_lik_map, 'log_lik_mean': log_lik_mean, 'log_lik_optim': log_lik_optim, 'CI': CI, 'noise_norm': noise_norm, 'log_lik_map_per_trial': log_lik/T, 'log_lik_mean_per_trial': log_lik_mean/T}
            # print(lls_dict)
            subject_dict = {'lab': lab, 'T': T, 'params_str': params_str, 'params_array': params_array, 'params_lengths': params_lengths, 'subject_id': i}
            entry = file_name_dict | subject_dict | lls_dict

            entries.append(entry)

        if lab == 'wittenlab':
            all_log_liks.append(log_liks)
            all_log_liks_mean.append(log_liks_mean)

    entries = pd.DataFrame(entries)
    # entries = entries[entries['lab'] == 'wittenlab']
    # print(entries[entries['label'] == 'AC']['params_str'].iloc[0])
    # print(entries[entries['label'] == 'IRL-lapse'])

    # Pickle save entries
    entries.to_pickle('./postprocessing/parsed_slurm_entries_3.pkl') # v3 = inclusion of w0

    for protocol in ['training', 'no_curriculum']:
        print(f'Protocol: {protocol}')
        for label in entries.query('protocol == @protocol')['label'].unique():
            for lab in entries.query('protocol == @protocol')['lab'].unique():
                print(f'{label} {lab}: {entries.query("protocol == @protocol and label == @label and lab == @lab")["subject_id"].nunique()} subjects')
    # sys.exit()
    protocol = 'training'
    entries = entries[entries['protocol'] == protocol]

    selected_models = ['Psytrack', 'RL-R', 'RVBF', 'RVBF-a', 'RVBF-b'] # 'AC',

    # entries = entries[entries['lab'] == 'wittenlab']

    # Plot noise components
    fig, ax = plt.subplots(figsize=[len(selected_models),3], constrained_layout=True)
    for j, label in enumerate(selected_models):
    #     # if model_label == 'Psytrack':
    #     #     continue
        noise_norms = entries[entries['label'] == label]['noise_norm']
        noise_norms = noise_norms[~np.isnan(noise_norms)]
        ax.boxplot(noise_norms[~np.isnan(noise_norms)], positions=[j], labels=[label], widths=0.6, showfliers=False)
    # sns.swarmplot(x='label', y='noise_norm', data=entries[entries['label'] != 'Psytrack'], ax=ax, color='k', alpha=0.5) # [entries['label'] != 'Psytrack']
    # sns.boxplot(x='label', y='noise_norm', data=entries[entries['label'] != 'Psytrack'], ax=ax, showfliers=False, boxprops=dict(facecolor='white')) # [entries['label'] != 'Psytrack']
    ax.set_ylabel('Noise norm')
    ax.set_yscale('log')
    sns.despine(ax=ax)
    plt.savefig(f'./figures/noise_norm_{len(selected_models)}models3.png', dpi=300)
    plt.savefig(f'./figures/noise_norm_{len(selected_models)}models3.eps', format='eps')
        

    # Group by lab and plot difference in log_lik_map per label, compared to label=PysTrack
    _ll_label = 'log_lik_map'

    fig, ax = plt.subplots()
    sns.boxplot(x='label', y='log_lik_map_per_trial', data=entries, ax=ax, boxprops=dict(facecolor='white')) # [entries['label'] != 'Psytrack']
    sns.swarmplot(x='label', y='log_lik_map_per_trial', data=entries, ax=ax, color='k', alpha=0.5) # [entries['label'] != 'Psytrack']
    ax.axhline(np.log(0.5), c='r', ls='--', zorder=-1, label='Chance')
    plt.savefig(f'./figures/log_lik_map_per_trial.png', dpi=300)

    fig, ax = plt.subplots(figsize=[len(selected_models),4], constrained_layout=True)
    sub_entries = entries[entries['label'].isin(selected_models)]
    sns.boxplot(x='label', y='log_lik_map_per_trial', data=sub_entries, ax=ax, boxprops=dict(facecolor='white')) # [entries['label'] != 'Psytrack']
    # sns.swarmplot(x='label', y='log_lik_map_per_trial', data=sub_entries, ax=ax, color='k', alpha=0.5, size=3) # [entries['label'] != 'Psytrack']
    for (lab, subject_id), group in sub_entries.groupby(['lab', 'subject_id']):
        ax.plot(group['label'], group['log_lik_map_per_trial'], 'o-', markersize=2, c='k', alpha=0.2, zorder=2)
    # ax.axhline(np.log(0.5), c='r', ls='--', zorder=-1, label='Chance')
    ax.set_xticklabels(sub_entries['label'].unique(), rotation=45)
    sns.despine(ax=ax)
    plt.savefig(f'./figures/log_lik_map_per_trial{len(selected_models)}models.png', dpi=300)
    # plt.savefig(f'./figures/log_lik_map_per_trial{len(selected_models)}models.eps', format='eps')

    # # Plot difference between TDRL and RL-R
    # fig, ax = plt.subplots(figsize=[2,4], constrained_layout=True)
    # self_diffs = diffs = sub_entries[sub_entries['label'] == 'TDRL'][_ll_label].values - sub_entries[sub_entries['label'] == 'TDRL'][_ll_label].values
    # diffs = sub_entries[sub_entries['label'] == 'RL-R'][_ll_label].values - sub_entries[sub_entries['label'] == 'TDRL'][_ll_label].values
    # ax.boxplot(self_diffs, positions=[0], labels=['TDRL'], widths=0.6, showfliers=False)
    # ax.boxplot(diffs, positions=[1], labels=['RL-R'], widths=0.6, showfliers=False)
    # ax.axhline(0, c='k', ls='--', zorder=-1)
    # ax.set_ylabel('MLL difference')
    # sns.despine(ax=ax)
    # plt.savefig(f'./figures/TDRL-RL-R_diff.png', dpi=300)
    # plt.savefig(f'./figures/TDRL-RL-R_diff.eps', format='eps')

    labs = entries['lab'].unique()
    fig, ax = plt.subplots(figsize=[len(models)+2,6], constrained_layout=True)
    for i, lab in enumerate(labs):
        lab_entries = entries[entries['lab'] == lab]
        for j, label in enumerate(lab_entries['label'].unique()):
            try:
                diffs = lab_entries[lab_entries['label'] == label][_ll_label].values - lab_entries[lab_entries['label'] == 'Psytrack'][_ll_label].values
                ax.boxplot(diffs, positions=[3*j - 2+i], labels=[label], widths=0.6)
            except ValueError:
                print(f'Missing entries for {label}')
                continue

    # print(entries.query("label == 'Psytrack' and lab == 'wittenlab'"))
    # print(entries.query("label == 'IRL-Reg-vec-bf'"))
    
    ax.axhline(0, c='k', ls='--', zorder=-1)
    plt.savefig(f'./figures/{_ll_label}_diff_per_lab.png', dpi=300)

    labs = entries['lab'].unique()
    fig, ax = plt.subplots(figsize=[1.5*len(selected_models),3], constrained_layout=True)

    all_diffs = []
    print("T-tests for log-lik difference")

    # for i, baseline_model in enumerate(selected_models):
    #     for j, label in enumerate(selected_models):
    #         try:
    #             diffs = entries[entries['label'] == label][_ll_label].values - entries[entries['label'] == baseline_model][_ll_label].values
    #             # diffs = diffs / entries[entries['label'] == label]['T'].values
    #         except ValueError:
    #             print(f'No entries for {label}')
    #             continue

    #         if baseline_model=='RVBF':
    #             ax.boxplot(diffs[~np.isnan(diffs)], positions=[j], labels=[label], widths=0.6, showfliers=False)
    #         # ax.errorbar(j, np.mean(diffs), yerr=np.std(diffs), fmt='o', label=label)

    #         all_diffs.append(diffs)

    #         t_statistic, p_value = ttest_1samp(diffs[~np.isnan(diffs)], 0)
    #         # Adjust the p-value for a one-sided test (greater than 0)
    #         if t_statistic > 0:
    #             p_value_one_sided = p_value / 2
    #         else:
    #             p_value_one_sided = 1  # If t-statistic is negative, no need to check right-tailed test
    #         print(f"{label}, one-sided p-value: {p_value_one_sided:.2e}")
        
    # all_diffs = np.array(all_diffs)
    # # ax.plot(all_diffs, '-o', c='k', alpha=0.1, zorder=-1)
    # ax.axhline(0, c='k', ls='--', zorder=-1)
    # ax.set_ylabel('LL difference')
    # sns.despine(ax=ax)
    
    # plt.savefig(f'./figures/{_ll_label}_diff_{len(selected_models)}models2.png', dpi=300)
    # # plt.savefig(f'./figures/{_ll_label}_diff_{len(selected_models)}models2.eps', format='eps')
    # # import sys; sys.exit()


    # Plot params --------------------------------

    for model_class in selected_models:
        print(f'Params for {model_class}')

        all_params = []
        non_error_is = []
        for i, params_str in enumerate(entries[entries['label']==model_class]['params_str']):
            try:
                _params_array, lengths = parse_ParamsReg(params_str)
                if len(_params_array) != sum(lengths):
                    raise TypeError("Parameter array length does not match the expected sum of lengths.")
                all_params.append(_params_array)
                non_error_is.append(i)
            except TypeError:
                print(f'Failed to parse #{i}, params_str={params_str}')
                continue

        try:
            all_params = np.array(all_params)
        except ValueError:
            print("Error in :")
            print(entries[entries['label']==model_class].iloc[non_error_is[np.argmin([len(p) for p in all_params])]])
            continue

        if model_class == 'RVBF':
            np.save(f'./postprocessing/RVBF_{protocol}_params.npy', all_params)
            fig, axs = plt.subplots(figsize=[4,7], nrows=6, ncols=2, width_ratios=[4,1], constrained_layout=True)
            titles = ['log sigma', 'log sigma_day', r'Learning rate $\log \alpha$', r'Forgetting $\log Q$', 'baseline', 'Initial weight $w_0$']
            for i in range(6):
                ax = axs[i,0]
                if i < 4:
                    ax.boxplot(np.exp(all_params)[:,5*i:5*(i+1)], showfliers=False)
                    ax.set_yscale('log')
                    ax.set_xticklabels([])
                    axs[i,1].boxplot(np.exp(all_params)[:,5*i:5*(i+1)].mean(1), showfliers=False)
                    axs[i,1].set_yscale('log')
                else:
                    ax.boxplot(all_params[:,5*i:5*(i+1)], showfliers=False)
                    ax.axhline(0, c='k', ls='--', zorder=-1)
                    axs[i,1].boxplot(all_params[:,5*i:5*(i+1)].mean(1), showfliers=False)


                # ax.plot(np.arange(5)+1, all_params[:,5*i:5*(i+1)].T, '-o', c='k', alpha=0.1, markersize=2)
                ax.set_xticks(range(1,6))
                ax.set_title(titles[i])
                sns.despine(ax=ax)

                axs[i,1].sharey(ax)
                axs[i,1].set_xticklabels([])
                # axs[i,1].set_yticklabels([])
                sns.despine(ax=axs[i,1])
            axs[-1,0].set_xticklabels(['bias', 'left', 'right', 'prevChoice', 'prevRewarded'], rotation=30, ha='right')
            axs[-1,0].set_xlabel('Regressor')
        elif model_class == 'TDRL':
            all_params = np.exp(all_params)
            fig, ax = plt.subplots(figsize=[len(all_params[0])/2,2], constrained_layout=True)
            ax.boxplot(all_params)
            ax.set_yscale('log')
            sns.despine(ax=ax)
        else:
            fig, ax = plt.subplots(figsize=[len(all_params[0]),2], ncols=len(lengths), constrained_layout=True, width_ratios=lengths)
            # ax.errorbar(range(len(all_params[0])), np.mean(all_params, axis=0), yerr=np.std(all_params, axis=0), fmt='o', label=model_class)
            cum_lengths = np.concatenate(([0], np.cumsum(lengths)))

            for i in range(len(lengths)):
                sub_params = all_params[:,cum_lengths[i]:cum_lengths[i+1]]
                # sub_params = sub_params[~np.isnan(sub_params)]
                ax[i].boxplot(sub_params, showfliers=False, widths=0.5)  
                # ax[i].set_yscale('log')
                # ax[i].set_xticklabels([])
                # ax[i].set_xticks(range(1,len(lengths)+1))
                sns.despine(ax=ax[i])
            # ax.boxplot(all_params, widths=0.6, showfliers=False)
            # ax.set_yscale('log')
            # ax.plot(np.arange(len(all_params[0]))+1, all_params.T, '-o', c='k', alpha=0.1)
            # ax.set_xticks(range(len(all_params[0])))
            # sns.despine(ax=ax)
        plt.savefig(f'./figures/params_{model_class}.png', dpi=300)
        # plt.savefig(f'./figures/params_{model_class}.eps', format='eps')


        if model_class == 'RL-R-vec':
            L_R_indices = [3,4]
        elif model_class == 'RVBF':
            L_R_indices = [11, 12]
        
        if model_class in ['RL-R-vec', 'RVBF']:
            # Do a test to determine whether the learning rates for each side differ
            alpha_L = all_params[:,L_R_indices[0]]
            alpha_R = all_params[:,L_R_indices[1]]


            t_statistic, p_value = ttest_rel(alpha_L, alpha_R)
            print("\nAlpha_L vs Alpha_R, model = ", model_class)
            print(f"Paired t-test: is the mean difference between L and R different from 0?")
            print(f"\tp-value: {p_value:.2e}, (significant? {p_value < 0.05}) t-statistic: {t_statistic:.2f}, N = {len(alpha_L)}")

            L_R_difference = np.abs(alpha_L - alpha_R)
            t_statistic, p_value = ttest_1samp(L_R_difference, 0, alternative='greater')
            print("ttest_1samp: Is the absolute difference between L and R different from 0?")
            print(f"\tp-value: {p_value:.2e}, (significant? {p_value < 0.05}) t-statistic: {t_statistic:.2f}, N = {len(alpha_L)}")
    sys.exit()

    # --------------------------------------------

    Ts = np.array(Ts)
    all_log_liks = np.array(all_log_liks)
    all_log_liks_mean = np.array(all_log_liks_mean)

    all_params = []
    for params_str in entries[entries['label']=='RLReg']['params_str']:
        try:
            all_params.append(parse_ParamsReg(params_str)[0])
        except TypeError:
            print(f'Failed to parse {params_str}')
            continue
    
    all_params = np.array(all_params)
    param_labels = [r'$\sigma$', r'$\sigma_{day}$', r'$\alpha$', '$Q_1$', '$Q_2$', '$Q_3$', '$Q_4$', '$Q_5$', '$A_1$', '$A_2$', '$A_3$', '$A_4$', '$A_5$', r'$b$', r'$\gamma$', r'$\kappa$', r'$\beta$']
    print("N subject {}, N params {}".format(*all_params.shape))
    print(param_labels)
    print(np.median(all_params, axis=0))
    print("log sigma: ", all_params[:,0])
    # all_params[:,:8] = np.exp(all_params[:,:8])
    # all_params[:, 8:] = all_params[:, 8:] / np.exp(all_params[:,2][:, None])

    # # Correct for gamma, = 1
    # all_params[2:, :8] = all_params[2:, :8] - np.log(all_params[:,-2])[2:,None]
    # all_params[:, 8:] = all_params[:, 8:] / all_params[:,-2][:, None]

    # all_params = all_params / np.exp(all_params[:,2])[:,None]
    all_params_means = np.mean(all_params, axis=0)
    all_params_stds = np.std(all_params, axis=0)


    print('\n'+r'T-tests for difference w 0, before ($\alpha$ * param) normalization')
    for i, param_dist in enumerate(all_params.T):
        if i <= 2:
            t_statistic_log, p_value_log = ttest_1samp(param_dist, -10.)
            t_statistic, p_value = ttest_1samp(np.exp(param_dist), 0)
        else:
            t_statistic, p_value = ttest_1samp(param_dist, 0)
            t_statistic_log, p_value_log = ttest_1samp(np.log(param_dist)[~np.isnan(np.log(param_dist))], -10.)
        print(f"{param_labels[i]}, p-value: {p_value:.2e}, < 0.05: {p_value < 0.05}, p-value log: {p_value_log:.2e}, < 0.05: {p_value_log < 0.05}")
    print('')

    fig, ax  = plt.subplots(figsize=[3,2.5], constrained_layout=True)
    ax.plot(entries[entries['label']=='RLReg']['T'], all_params[:,2], 'o', c='k')
    # print(np.stack(entries[entries['label']=='RLReg']['CI']))
    yerr = np.abs(np.stack(entries[entries['label']=='RLReg']['CI']).T - all_params[:,2])
    ax.errorbar(entries[entries['label']=='RLReg']['T'], all_params[:,2], yerr=yerr, fmt='o', c='k', capsize=5, zorder=-1)
    ax.set_xlabel('T')
    ax.set_ylabel(r'$\alpha$')

    # Plot regression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(entries[entries['label']=='RLReg']['T'].values.reshape(-1,1), all_params[:,2])
    x = np.unique(entries[entries['label']=='RLReg']['T'])
    y = reg.predict(x.reshape(-1,1))
    ax.plot(x, y, c='r', ls='--', zorder=1)
    print(reg.score(entries[entries['label']=='RLReg']['T'].values.reshape(-1,1), all_params[:,2]))
    # print(entries[entries['label']=='RLReg']['CI'])
    sns.despine(ax=ax)
    plt.savefig('./figures/alpha_vs_T.png', dpi=300)
    plt.savefig('./figures/alpha_vs_T.eps', format='eps')

    # fig, axs = plt.subplots(figsize=[len(all_params_means)/2,5], ncols=2, nrows=2, constrained_layout=True)
    # # ax.errorbar(range(len(all_params_means)), all_params_means, yerr=all_params_stds, fmt='o')
    # axs[0,0].boxplot(all_params[:,:8])
    # # axs[0,0].set_xticklabels([r'$\sigma$', r'$\sigma_{day}$', r'$\alpha$', '$Q_1$', '$Q_2$', '$Q_3$', '$Q_4$', '$Q_5$'])
    # axs[0,0].set_ylabel('log value')

    # # axs[0].set_yscale('log')
    # axs[0,1].boxplot(all_params[:,8:])
    # # axs[0,1].set_xticklabels(['$A_1$', '$A_2$', '$A_3$', '$A_4$', '$A_5$', r'$\kappa$', r'$\gamma$', r'$\beta$'])
    # axs[0,1].axhline(0, c='k', ls='--', zorder=-1)
    # axs[0,0].set_title(f'From individual fits ({len(all_params)} subjects, 3 labs)')


    # lab_wide_params_str = 'ParamsGLMRegLearn(log_sigma=Array(-3.5113401, dtype=float32), log_sigma_day=Array(-0.6770225, dtype=float32), log_alpha=Array(-7.78724, dtype=float32), log_Q=Array([-1.2062161, -2.6502953, -3.2131824, -5.2243137, -4.4338512],      dtype=float32), A=Array([-0.23430336,  0.7535554 ,  0.75474846,  0.724375  ,  1.6699619 ],      dtype=float32), kappa=Array(-0.5994488, dtype=float32), gamma=Array(1.9287833, dtype=float32), beta=Array(2.7450528, dtype=float32))'
    # lab_wide_params_array = parse_ParamsReg(lab_wide_params_str)
    # lab_wide_params_mean = parse_ParamsReg('ParamsGLMRegLearn(log_sigma=Array(-3.3490317, dtype=float32), log_sigma_day=Array(-0.50399244, dtype=float32), log_alpha=Array(-8.042228, dtype=float32), log_Q=Array([-2.012658 , -3.336878 , -2.1640806, -5.373836 , -3.9208875],      dtype=float32), A=Array([ 0.10940534, -0.7521002 ,  0.5298442 ,  0.65056694, -0.25763687],      dtype=float32), kappa=Array(0.12343716, dtype=float32), gamma=Array(4.3142095, dtype=float32), beta=Array(1.7879734, dtype=float32))')
    # CI = np.array([[-3.6316333,  -3.0951262 ], [-1.030373,   -0.07394695], [-9.936462,   -6.7510166 ], [-4.298006,    0.04626565], [-5.852873,   -1.2783043 ], [-4.429428,   -0.28309795], [-7.888959,   -3.0668821 ], [-5.995984,   -1.5884788 ], [-1.9508822,   2.272294  ], [-3.1023917,   1.4654257 ], [-1.6975002,   2.8129752 ], [-1.4672774,   2.7501564 ], [-2.4770107,   1.8826354 ], [-1.6232564,   1.7358222 ], [ 2.020223,    6.807311  ], [-0.49435645,  3.8700585 ]])
    # axs[1,0].errorbar(
    #     x=range(8),
    #     y=lab_wide_params_mean[:8],
    #     yerr=[lab_wide_params_mean[:8] - CI[:8,0], CI[:8,1] - lab_wide_params_mean[:8]],
    #     fmt='o', capsize=5, c='k', zorder=0,
    # )
    # axs[1,0].plot(range(8), lab_wide_params_array[:8], 'o', c='tab:green', label='best', zorder=1)
    # axs[1,0].sharey(axs[0,0])
    # axs[1,0].set_title('Posterior mean and CI from whole-lab (wittenlab) fit')

    # # axs[1,1].plot(range(8), CI[8:], '_', c='k')
    # axs[1,1].errorbar(
    #     x=range(8),
    #     y=lab_wide_params_mean[8:],
    #     yerr=[lab_wide_params_mean[8:] - CI[8:,0], CI[8:,1] - lab_wide_params_mean[8:]],
    #     fmt='o', capsize=5, c='k', zorder=0,
    # )
    # axs[1,1].plot(range(8), lab_wide_params_array[8:], 'o', c='tab:green', label='best', zorder=1)
    # axs[1,1].legend()
    # axs[1,1].axhline(0, c='k', ls='--', zorder=-1)
    # axs[1,1].sharey(axs[0,1])


    # fig.suptitle('Regression gradient learning rule parameters')
    # # ax.set_yscale('log')
    # plt.savefig('./figures/params.png', dpi=300)
    
    # --------------------------------

    # Panel 3, per lab
    lab_lengths = entries[entries['label'] == 'RLReg'].groupby('lab').size()
    print(lab_lengths, lab_lengths.values)
    lab_lengths = lab_lengths.values
    n_params = all_params.shape[1]
    flierprops = dict(marker='.')
    fig, axs = plt.subplots(figsize=[len(models),3], ncols=2, width_ratios=[3, n_params-3], constrained_layout=True)
    axs[0].boxplot(np.exp(all_params)[:lab_lengths[0],:3], positions=3*np.arange(3)-0.5, widths=0.5, flierprops=flierprops)
    axs[0].boxplot(np.exp(all_params)[lab_lengths[0]:lab_lengths[0]+lab_lengths[1], :3], positions=3*np.arange(3), widths=0.5, flierprops=flierprops)
    axs[0].boxplot(np.exp(all_params)[lab_lengths[0]+lab_lengths[1]:, :3], positions=3*np.arange(3)+0.5, widths=0.5, flierprops=flierprops)
    axs[0].set_xticks(3*np.arange(3))
    axs[0].set_yscale('log')
    axs[0].set_xticklabels([r'$\sigma$', r'$\sigma_{day}$', r'$\alpha$'])

    axs[1].boxplot(all_params[:lab_lengths[0], 3:], positions=3*np.arange(n_params-3)-0.5, widths=0.5, flierprops=flierprops)
    axs[1].boxplot(all_params[lab_lengths[0]:lab_lengths[0]+lab_lengths[1], 3:], positions=3*np.arange(n_params-3), widths=0.5, flierprops=flierprops)
    axs[1].boxplot(all_params[lab_lengths[0]+lab_lengths[1]:, 3:], positions=3*np.arange(n_params-3)+0.5, widths=0.5, flierprops=flierprops)
    axs[1].set_xticks(3*np.arange(n_params-3))
    axs[1].set_xticklabels(['$Q_1$', '$Q_2$', '$Q_3$', '$Q_4$', '$Q_5$', '$A_1$', '$A_2$', '$A_3$', '$A_4$', '$A_5$', r'$b$', r'$\gamma$', r'$\kappa$', r'$\beta$'])# using notation of paper
    axs[1].axhline(0, c='k', ls='--', zorder=-1)
    # axs[0].set_xticklabels([r'$\sigma$', r'$\sigma_{day}$', r'$\alpha$', '$Q_1$', '$Q_2$', '$Q_3$', '$Q_4$', '$Q_5$'])
    # axs[0].set_ylabel('log value')
    
    # axs[1].boxplot(all_params[:lab_lengths[0],8:], positions=3*np.arange(8)-0.5, widths=0.5, flierprops=flierprops)
    # axs[1].boxplot(all_params[lab_lengths[0]:lab_lengths[0]+lab_lengths[1],8:], positions=3*np.arange(8), widths=0.5, flierprops=flierprops)
    # axs[1].boxplot(all_params[lab_lengths[0]+lab_lengths[1]:,8:], positions=3*np.arange(8)+0.5, widths=0.5, flierprops=flierprops)
    # axs[1].set_xticks(3*np.arange(8))
    # axs[1].set_xticklabels(['$A_1$', '$A_2$', '$A_3$', '$A_4$', '$A_5$', r'$\kappa$', r'$\gamma$', r'$\beta$'])
    # axs[1].axhline(0, c='k', ls='--', zorder=-1)

    fig.suptitle('Regression gradient learning rule parameters\n Intra and inter-lab variability')
    plt.savefig('./figures/params_per_lab.png', dpi=300)

    # --------------------------------


    # alpha scales multiplicatively every term. Normalize
    log_alphas = all_params[:,2]
    # all_params[:, 2:8] = all_params[:, 2:8] + log_alphas[:,None]
    # assert np.allclose(all_params[:, 2], 0)
    all_params[:, 3:] = all_params[:, 3:] * np.exp(log_alphas)[:,None]

    fig, axs = plt.subplots(
        figsize=[len(all_params_means)/2,3], ncols=2, constrained_layout=True, width_ratios=[3, all_params.shape[1]-3]
    )
    # ax.errorbar(range(len(all_params_means)), all_params_means, yerr=all_params_stds, fmt='o')
    axs[0].boxplot(np.exp(all_params)[:,:3], showfliers=True, widths=0.6)
    axs[0].set_yscale('log')
    axs[0].set_xticklabels([r'$\sigma$', r'$\sigma_{day}$', r'$\alpha$'])#, '$Q_1$', '$Q_2$', '$Q_3$', '$Q_4$', '$Q_5$'])
    # axs[0].set_ylabel('log value')
    sns.despine(ax=axs[0])

    # axs[0].set_yscale('log')
    # all_params[:, 2:8] = np.exp(all_params[:, 2:8])
    axs[1].boxplot(all_params[:,3:], showfliers=False)
    # axs[1].set_yscale('log')
    axs[1].set_xticklabels(['$Q_1$', '$Q_2$', '$Q_3$', '$Q_4$', '$Q_5$', '$A_1$', '$A_2$', '$A_3$', '$A_4$', '$A_5$', r'$b$', r'$\gamma$', r'$\kappa$', r'$\beta$'])# using notation of paper

    axs[1].axhline(0, c='k', ls='--', zorder=-1)
    axs[1].set_title(r'Effective ($\alpha *$ param) values')
    sns.despine(ax=axs[1])

    fig.suptitle(f'From individual fits ({len(all_params)} subjects, 3 labs)')
    plt.savefig('./figures/params_normalized.png', dpi=300)
    plt.savefig('./figures/params_normalized.eps', format='eps')

    # Tests 
    print('\n'+r'T-tests for difference w 0, after ($\alpha$ * param) normalization')
    for i, param_dist in enumerate(all_params.T):
        if i <= 2:
            t_statistic_log, p_value_log = ttest_1samp(param_dist, -10.)
            t_statistic, p_value = ttest_1samp(np.exp(param_dist), 0)
        else:
            t_statistic, p_value = ttest_1samp(param_dist, 0)
            t_statistic_log, p_value_log = ttest_1samp(np.where(param_dist < 0, -8.0, np.log(param_dist)), -8.)
        print(f"{param_labels[i]}, p-value: {p_value:.2e}, < 0.05: {p_value < 0.05}, p-value log: {p_value_log:.2e}, < 0.05: {p_value_log < 0.05}")
    print('')

    import sys; sys.exit()
    # diffs = all_log_liks[2] - all_log_liks[0]

    diffs = all_log_liks - all_log_liks[0] # w.r.t Psytrack
    diffs_mean = all_log_liks_mean - all_log_liks_mean[0] # w.r.t Psytrack
    
    fig, (ax, ax2) = plt.subplots(figsize=[len(models)+2,6], nrows=2, constrained_layout=True)

    # logliks_per_trial = all_log_liks / Ts[None, :]
    # ax.boxplot(logliks_per_trial.T, positions=range(len(models)), labels=models)
    # ax.plot(logliks_per_trial, '-o', c='k', alpha=0.2, zorder=-1)
    # ax.set_title('MLL per trial')
    # ax.set_ylabel('Log-lik')
    # ax.set_xticklabels([])

    ax.boxplot(diffs_mean.T, positions=range(len(models)), labels=models)
    ax.plot(diffs_mean, '-o', c='k', alpha=0.2, zorder=-1)
    ax.set_title('MLL (mean)')
    ax.set_ylabel('log-lik difference')
    ax.axhline(0, c='k', ls='--', zorder=-1)


    ax2.boxplot(diffs.T, positions=range(len(models)), labels=models)
    ax2.plot(diffs, '-o', c='k', alpha=0.2, zorder=-1)
    ax2.set_title('MLL (max)')
    ax2.set_ylabel('log-lik difference')
    ax2.axhline(0, c='k', ls='--', zorder=-1)

    fig.suptitle('Marginal log-lik (MLL) difference w.r.t Psytrack')
    # fig.suptitle('Marginal log-lik (MLL) model comparison')

    plt.savefig('./figures/log_lik_per_trial.png', dpi=300)
    plt.show()
    plt.close()