from typing import NamedTuple, Type, Union, Tuple, List
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Bool

# Data

class Trajectory(NamedTuple):
    '''IBL single trajectory data, with T trials.'''
    X: Float[Array, "T M"]      # Regressors, M-dimensional
    Y: Float[Array, "T"]        # Choices
    R: Float[Array, "T"]        # Rewards
    day_flags: Bool[Array, "T"] # Day flags, True if new day/session

    def __len__(self) -> int:
        """Return number of trials (length of Y)."""
        return len(self.Y)

def trim_trajectory(trajectory: Trajectory, T: int) -> Trajectory:
    '''Trim trajectory up to T trials. If trajectory shorter than T, return original trajectory.'''
    curr_T = trajectory.X.shape[0]
    if curr_T <= T:
        return trajectory
    else:
        return Trajectory(
            X=trajectory.X[:T],
            Y=trajectory.Y[:T],
            R=trajectory.R[:T],
            day_flags=trajectory.day_flags[:T]
        )

# Params

class ParameterProperties():
    def __init__(self, trainable: bool=True) -> None:
        self.trainable = trainable

class ParamsGLMLearn(NamedTuple):
    log_sigma: Union[float, ParameterProperties]
    log_sigma_day: Union[float, ParameterProperties]
    log_alpha: Union[float, jnp.ndarray, ParameterProperties]
    w_0: jnp.ndarray = jnp.zeros((2,))

class ParamsPsytrack(NamedTuple):
    log_sigma: float
    log_sigma_day: float

class ParamsTimeVarGLMLearn(NamedTuple):
    beta_0: float = 0.0
    log_alpha: float = -5.0
    log_sigma_0: float = -5.0
    log_sigma: float = -3.0
    log_sigma_day: float = -2.0
    # baseline: float = 0.0
    # log_forget: float = -4.0
    # log_forget_day: float = -4.0
    log_Q: jnp.ndarray = jnp.array([-4.0])
    # r1: float = 1.0
    # r0: float = 1.0
    gamma: float = 1.0
    kappa: float = 0.0
    baseline: float = 0.0

class ParamsGLMRegLearn(NamedTuple):
    log_sigma: float
    log_sigma_day: float
    log_alpha: Union[float, jnp.ndarray]
    Q: jnp.ndarray # for diag
    A: jnp.ndarray # for diag
    kappa: float
    gamma: float
    beta: float
    baseline: Union[float, jnp.ndarray]
    
class ParamsGLMHMMLearn(NamedTuple):
    logit_pi0: float
    logit_a_1: float
    logit_a_2: float
    log_alpha: float
    log_sigma: float
    log_sigma_day: float

class ParamsGLMInterpLearn(NamedTuple):
    log_sigma: float
    log_sigma_day: float
    log_alpha: float
    Q: jnp.ndarray
    r1: float = 1.0
    r0: float = 1.0
    gamma: float = 0.0
    baseline: float = 0.0

class ParamsQLearning(NamedTuple):
    percept_log_scale: float = -2.0
    log_sigma: float = -3.0
    log_sigma_day: float = -2.0
    log_alpha: float = -5.0
    log_temp: float = 0.0

class ParamsGLMBaseLearn(NamedTuple):
    log_sigma: Union[float, ParameterProperties]
    log_sigma_day: Union[float, ParameterProperties]
    log_alpha: Union[float, jnp.ndarray, ParameterProperties]
    baseline_weights: jnp.ndarray = jnp.zeros((5,5,))
    log_sigma_0: float = -5.0
    log_Q: jnp.ndarray = -5.0 * jnp.ones((5,))

class ParamsDynamicGLMHMM(NamedTuple):
    log_sigma: float
    log_sigma_day: float = 1.0
    alpha: float = 2.0

class ParamsAC(NamedTuple):
    beta_0: Union[float, jnp.ndarray] = 0.0
    log_alpha: float = -5.0
    log_sigma_0: float = -5.0
    log_sigma: float = -3.0
    log_sigma_day: float = -2.0
    log_Q: jnp.ndarray = jnp.array([-4.0])

class ParamsRVBF(NamedTuple):
    log_sigma: Union[float, jnp.ndarray]
    log_sigma_day: Union[float, jnp.ndarray]
    log_alpha: Union[float, jnp.ndarray]
    log_Q: jnp.ndarray
    baseline: Union[float, jnp.ndarray]
    w_0: jnp.ndarray = jnp.zeros((5,)) #! magic dim val

class ParamsTimeVarRVBF(NamedTuple):
    log_sigma: Union[float, jnp.ndarray]
    log_sigma_day: Union[float, jnp.ndarray]
    log_alpha: Union[float, jnp.ndarray]
    log_Q: jnp.ndarray
    baseline: Union[float, jnp.ndarray]
    beta_0: Union[float, jnp.ndarray] = 0.0
    log_sigma_0: float = -5.0
    w_0: jnp.ndarray = jnp.zeros((5,)) #! magic dim val

class ParamsHRL(NamedTuple):
    log_sigma: Union[float, jnp.ndarray]
    log_sigma_day: Union[float, jnp.ndarray]
    log_alpha_0: float
    log_alpha_1: Union[float, jnp.ndarray]
    baseline_1: Union[float, jnp.ndarray]
    baseline_0: float = 0.0
    log_sigma_0: float = -5.0
    q0: float = 0.0

def handle_none_params(func):
    def wrapper(self, *args, params=None, **kwargs):
        if params is None:
            params = self.params
        return func(self, *args, params=params, **kwargs)
    return wrapper

def params_to_array(named_tuple) -> Tuple[jnp.ndarray, List[int]]:
    arrays = []
    lengths = []

    # Loop through each attribute of the namedtuple
    for value in named_tuple._asdict().values():
        if jnp.isscalar(value) or (isinstance(value, jnp.ndarray) and value.ndim == 0):
            # If the value is a scalar, convert it to a 1D ndarray
            arrays.append(jnp.array([value]))
            lengths.append(1)
        elif isinstance(value, jnp.ndarray) and value.ndim == 1:
            # If it's a 1D array, append it directly
            arrays.append(value)
            lengths.append(len(value))
        else:
            raise ValueError("All attributes must be scalars or 1D arrays")
    
    # Concatenate all arrays into a single 1D numpy array
    return jnp.concatenate(arrays), lengths

def array_to_params(name, array, lengths) -> NamedTuple:
    # Unpack the lengths of each attribute
    L = np.cumsum(lengths)

    # Slice the array based on the original lengths of attributes
    sliced_arrays = [array[0:L[0]].squeeze()]
    for i in range(1, len(L)):
        sliced_arrays.append(array[L[i-1]:L[i]].squeeze())
    
    # Reconstruct the NamedTuple
    if isinstance(name, str):
        named_tuple = eval(name)
    else:
        named_tuple = name
    return named_tuple._make(sliced_arrays)

def get_param_name(model_repr) -> str:
    # Use regular expressions to capture any class name before the opening parenthesis
    import re
    match = re.match(r"(\w+)\(", model_repr)
    if match:
        # Prepend "Params" to the matched class name
        return f"Params{match.group(1)}"
    return ""

def label_to_parameterclass(name: str) -> Type[NamedTuple]:
    name = name.lower()
    if name == 'ac':
        return ParamsAC
    elif name == 'glmbaselearn':
        return ParamsGLMBaseLearn
    elif name == 'glmhmmlearn':
        return ParamsGLMHMMLearn
    elif name == 'glminterplearn':
        return ParamsGLMInterpLearn
    elif name == 'glmreglearn':
        return ParamsGLMRegLearn
    elif name == 'glmlearn':
        return ParamsGLMLearn
    elif name == 'psytrack':
        return ParamsPsytrack
    elif name == 'timevarglmlearn':
        return ParamsTimeVarGLMLearn
    elif name == 'dynamicglmhmm':
        return ParamsDynamicGLMHMM
    elif name == 'qlearning':
        return ParamsQLearning
    elif name == 'rvbf':
        return ParamsRVBF
    else:
        raise ValueError(f"Unknown parameter class name: {name}")

def get_params_labels(name: str) -> tuple[str, ...]:
    return label_to_parameterclass(name)._fields
        
if __name__=='__main__':
    # model = GLMLearn_(alpha=0.1, log_sigma=-2, not_trainable=[])

    # model_2 = GLMLearn_(alpha=0.3)

    # params_update = {'log_sigma': 2}
    # model.update_params(**params_update)
    # print(model.params)
    # print(model.some_output(1.0))

    params = ParamsPsytrack(log_sigma=-2.0, log_sigma_day=-2.0)
    params_array, lengths = params_to_array(params)
    print(params_array, lengths)
    # print(params.from_array(params_array, lengths))

    print(array_to_params(params, params_array, lengths))

    print("Attributes of ParamsGLMLearn:", ParamsGLMLearn._fields)
    print(label_to_parameterclass("psytrack").__name__)
    # print(get_param_name("ParamsGLMLearn(log_sigma=0.0, log_alpha=0.0)"))

