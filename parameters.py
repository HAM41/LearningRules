from typing import NamedTuple, Union, Mapping, Optional
import jax.numpy as jnp

class ParameterProperties():
    def __init__(self, trainable: bool=True) -> None:
        self.trainable = trainable

class ParamsGLMLearn(NamedTuple):
    log_sigma: Union[float, ParameterProperties]
    alpha: Union[float, ParameterProperties]
    
    @property
    def alpha(self) -> float:
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float):
        if value < 0:
            raise ValueError('Learning rate must be non-negative')
        self._alpha = value

    def to_array(self) -> jnp.ndarray:
        return jnp.array([self.log_sigma, self.alpha])
    
def handle_none_params(func):
    def wrapper(self, *args, params=None, **kwargs):
        if params is None:
            params = self.params
        return func(self, *args, params=params, **kwargs)
    return wrapper

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

    params = ParamsGLMLearn(log_sigma=-2.0, alpha=0.1)
    print(params)
    print(params._asdict())
    print(ParamsGLMLearn._make([-2.0, 0.0]))

