from typing import NamedTuple, Union, Mapping, Optional, Tuple, List
import jax.numpy as jnp
import numpy as np

class ParameterProperties():
    def __init__(self, trainable: bool=True) -> None:
        self.trainable = trainable

class ParamsGLMLearn(NamedTuple):
    log_sigma: Union[float, ParameterProperties]
    log_sigma_day: Union[float, ParameterProperties]
    log_alpha: Union[float, jnp.ndarray, ParameterProperties]

    # @staticmethod
    # def from_array(array, lengths):
    #     # Unpack the lengths of each attribute

    #     # Slice the array based on the original lengths of attributes
    #     log_sigma = array[0:lengths[0]].squeeze()
    #     log_sigma_day = array[lengths[0]:lengths[0]+lengths[1]].squeeze()
    #     log_alpha = array[lengths[0]+lengths[1]:].squeeze()
        
    #     # Reconstruct the NamedTuple
    #     return ParamsGLMLearn(log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_alpha=log_alpha)

class ParamsPsytrack(NamedTuple):
    log_sigma: float
    log_sigma_day: float

    # @staticmethod
    # def from_array(array, lengths):
    #     # Unpack the lengths of each attribute

    #     # Slice the array based on the original lengths of attributes
    #     log_sigma = array[0:lengths[0]].squeeze()
    #     log_sigma_day = array[lengths[0]:].squeeze()
        
    #     # Reconstruct the NamedTuple
    #     return ParamsPsytrack(log_sigma=log_sigma, log_sigma_day=log_sigma_day)

class ParamsTimeVarGLMLearn(NamedTuple):
    beta_0: float
    log_alpha: float
    log_sigma_0: float
    log_sigma: float
    log_sigma_day: float

    # @staticmethod
    # def from_array(array, lengths):
    #     # Unpack the lengths of each attribute

    #     # Slice the array based on the original lengths of attributes
    #     beta_0 = array[0:lengths[0]].squeeze()
    #     log_alpha = array[lengths[0]:lengths[0]+lengths[1]].squeeze()
    #     log_sigma_0 = array[lengths[0]+lengths[1]:lengths[0]+lengths[1]+lengths[2]].squeeze()
    #     log_sigma = array[lengths[0]+lengths[1]+lengths[2]:lengths[0]+lengths[1]+lengths[2]+lengths[3]].squeeze()
    #     log_sigma_day = array[lengths[0]+lengths[1]+lengths[2]+lengths[3]:].squeeze()
        
    #     # Reconstruct the NamedTuple
    #     return ParamsTimeVarGLMLearn(beta_0=beta_0, log_alpha=log_alpha, log_sigma_0=log_sigma_0, log_sigma=log_sigma, log_sigma_day=log_sigma_day)

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

    # @staticmethod
    # def from_array(array, lengths):
    #     # Unpack the lengths of each attribute

    #     L = np.cumsum(lengths)

    #     # Slice the array based on the original lengths of attributes
    #     log_sigma = array[0:L[0]].squeeze()
    #     log_sigma_day = array[L[0]:L[1]].squeeze()
    #     log_alpha = array[L[1]:L[2]].squeeze()
    #     log_Q = array[L[2]:L[3]].squeeze()
    #     A = array[L[3]:L[4]].squeeze()
    #     kappa = array[L[4]:L[5]].squeeze()
    #     gamma = array[L[5]:L[6]].squeeze()
    #     beta = array[L[6]:].squeeze()

    #     # Reconstruct the NamedTuple
    #     return ParamsGLMRegLearn(
    #         log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day, log_Q=log_Q, A=A, kappa=kappa, gamma=gamma, beta=beta,
    #         )
    
class ParamsGLMHMMLearn(NamedTuple):
    logit_pi0: float
    logit_a_1: float
    logit_a_2: float
    log_alpha: float
    log_sigma: float
    log_sigma_day: float

    # @staticmethod
    # def from_array(array, lengths):
    #     # Unpack the lengths of each attribute

    #     L = np.cumsum(lengths)

    #     # Slice the array based on the original lengths of attributes
    #     logit_pi0 = array[0:L[0]].squeeze()
    #     logit_a_1 = array[L[0]:L[1]].squeeze()
    #     logit_a_2 = array[L[1]:L[2]].squeeze()
    #     log_alpha = array[L[2]:L[3]].squeeze()
    #     log_sigma = array[L[3]:L[4]].squeeze()
    #     log_sigma_day = array[L[4]:].squeeze()

    #     # Reconstruct the NamedTuple
    #     return ParamsGLMHMMLearn(
    #         logit_pi0=logit_pi0, logit_a_1=logit_a_1, logit_a_2=logit_a_2, log_alpha=log_alpha, log_sigma=log_sigma, log_sigma_day=log_sigma_day
    #         )

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

def array_to_params(named_tuple, array, lengths) -> NamedTuple:
    # Unpack the lengths of each attribute
    L = np.cumsum(lengths)

    # Slice the array based on the original lengths of attributes
    sliced_arrays = [array[0:L[0]].squeeze()]
    for i in range(1, len(L)):
        sliced_arrays.append(array[L[i-1]:L[i]].squeeze())
    
    # Reconstruct the NamedTuple
    return named_tuple._make(sliced_arrays)

class GLMLearn_():
    def __init__(self, log_sigma: float=-1.0, alpha: float=0.0, not_trainable: list=[]) -> None:
        self.params = ParamsGLMLearn(log_sigma=log_sigma, alpha=alpha)
        self.props = ParamsGLMLearn(log_sigma=ParameterProperties(), alpha=ParameterProperties())
        for param in not_trainable:
            getattr(self.props, param).trainable = False
        
    def update_params(self, **kwargs) -> None:
        '''passes trainable kwargs to self.params._replace'''
        for key in kwargs:
            if not getattr(self.props, key).trainable:
                raise ValueError(f"Parameter '{key}' is not trainable")
        
        self.params = self.params._replace(**kwargs)

    @handle_none_params
    def some_output(self, x: float, params: Optional[ParamsGLMLearn]=None) -> jnp.ndarray:
        return jnp.array([x, params.alpha, params.log_sigma])
    
        
if __name__=='__main__':
    # model = GLMLearn_(alpha=0.1, log_sigma=-2, not_trainable=[])

    # model_2 = GLMLearn_(alpha=0.3)

    # params_update = {'log_sigma': 2}
    # model.update_params(**params_update)
    # print(model.params)
    # print(model.some_output(1.0))

    params = ParamsGLMLearn(log_sigma=jnp.array([-2.0]), log_alpha=jnp.array([-3.5, -2.0]), log_sigma_day=-2.0)
    params_array, lengths = params_to_array(params)
    print(params_array, lengths)
    # print(params.from_array(params_array, lengths))

    print(array_to_params(params, params_array, lengths))

