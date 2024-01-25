import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from typing import Tuple, Optional, Iterable

import os
os.environ['JAX_PLATFORMS']='cpu'

from parameters import ParamsGLMLearn, ParameterProperties, handle_none_params

def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def safe_sigmoid(X, threshold=100.):
    return jnp.where(X > threshold, jnp.ones_like(X), jnp.where(X < -threshold, jnp.zeros_like(X), sigmoid(X)))

def vec(x):
    x = jnp.asarray(x)
    if x.ndim == 0:
        return jnp.array([1,x])
    else:
        if x.ndim > 1:
            raise NotImplementedError
        return jnp.concatenate((jnp.array([1.0]),x))
    
def unvec(x):
    return x[1:]

def sign(y):
    y = jnp.asarray(y).astype(bool)
    return jnp.where(y == 0., -1., 1.)

def reward(X, Y, r1=1.0, r0=0.0) -> np.ndarray:
    '''
    Returns reward 
        r(x,y)= r1 if (x < 0 and y == 0) or (x > 0 and y==1)
                r0 else
    for all (x,y) pairs in X, Y
    '''
    X = jnp.array(X)
    Y = jnp.array(Y).astype(int)
    
    # Broadcast scalars to arrays if needed
    if X.size == 1 and Y.size > 1:
        X = jnp.full(Y.shape, X)
    elif Y.size == 1 and X.size > 1:
        Y = jnp.full(X.shape, Y).astype(int)

    # Initialize an array for the rewards with the same shape as x and y
    r = r0 * jnp.ones_like(X, dtype=float)

    # Calculate rewards element-wise
    mask_condition = (X < 0) & (Y == 0) | (X > 0) & (Y == 1)
    # r = r.at[mask_condition].set(1.0)
    r = jnp.where(mask_condition, r1, r)

    return r

def effective_reward(X):
    r'''
    Returns effective reward R(x) = \sum_y r(x,y) sign(y), for all x in X. 
    Output `RX` is of same shape as `X`.
    '''
    RX = jnp.sum(jnp.array([sign(y) * reward(X, y) for y in [0,1]]), axis=0)
    return RX

def cumulative_gaussian(x, sigma=1.0, mu=0.0):
    '''Cumulative distribution function for the Gaussian distribution.'''
    return 0.5*(1+jsp.special.erf((x-mu)/(sigma*jnp.sqrt(2))))

def Phi(x):
    '''Cumulative distribution function for the standard Gaussian.'''
    return cumulative_gaussian(x)

class QLearningModel():
    def __init__(self, alpha: float, sigma: float, beta: float=1.0, softmax: bool=True) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta        # "Inverse temperature" parameter for the softmax decision.

        self.V_init = 0.2       # Initialize values at 0.2

        # Use softmax for stochastic decisions. Defaults to True.
        self.softmax = softmax

    def p_R(self, m):
        '''p_R(m): Belief state p(s > 0 | m)'''
        return Phi(m/self.sigma)
    
    def encode(self, X):
        N = X.shape
        noise = jax.random.normal(key, shape=X.shape)
        return X + self.sigma * noise
    
    def transition_point(self, V):
        V_L, V_R = V
        return self.sigma * jsp.special.erfinv(np.divide(V_L - V_R, V_R + V_L))
    
    def decision(self, m, V):
        if self.softmax:
            V_L, V_R = V
            p_R = self.p_R(m)

            # Compute Q values for each state
            Qs = jnp.stack([jnp.multiply(1-p_R, V_L), jnp.multiply(p_R, V_R)])
            
            p = jax.nn.softmax(self.beta * Qs, axis=0)
            if p.ndim > 1:
                Y = np.array([np.random.binomial(1, p=_p[1]) for _p in p]).astype(int) #! make jax
            else:
                Y = jax.random.bernoulli(key, p[1]).astype(int)
        else:
            a = self.transition_point(V)
            Y = jnp.array(m > a).astype(int)
        return Y
    
    def update_values(self, V, x, y, m):
        '''
        Learning rule update
        Args:
            V: np.ndarray (2,N), left V[0,:] and right V[1,:] values
            x: np.ndarray (N), 
        '''
        V_L, V_R = V
        p_R = self.p_R(m)
        if y==1:
            V_R = V_R + self.alpha * (reward(x,y) - jnp.multiply(p_R, V_R))
        else:
            V_L = V_L + self.alpha * (reward(x,y) - jnp.multiply(1-p_R, V_L))
        return jnp.stack([V_L, V_R])

    def emission_likelihood(self, y, m, V):
        r'''p(y | x_hat, V)'''
        V_L, V_R = V
        p_R = self.p_R(m)
        y = jnp.array(y).astype(int)

        # Compute Q values for each state
        Qs = jnp.stack([jnp.multiply(1-p_R, V_L), jnp.multiply(p_R, V_R)])

        # Compute decision likelihood p(y|Qs)
        if self.softmax:
            p = jax.nn.softmax(Qs, axis=0)[y]
            return p
        else:
            return jnp.array(y==jnp.argmax(Qs, axis=0), dtype=float)

    def joint_dynamics_loglikelihood(self, M, X, **params):
        '''log p(m_{1:T} | x_{1:T})'''
        pass

    def joint_emission_loglikelihood(self, M, X, Y,  **params):
        '''log p(y_{1:T} | m_{1:T}, x_{1:T})'''
        pass

    def log_joint(self, X, Y, M, **params):
        '''log p(Y, M | X, theta)'''
        log_pM = self.joint_dynamics_loglikelihood(M, X, **params)
        log_pyM = self.joint_emission_loglikelihood(M, Y, X, **params)
        return log_pyM + log_pM

    def forward(self, t, N_samples, prev_latent, X_prev, Y_prev, X):
        '''
        Treating the values V and the percept m as the latents
        Return N_samples from one forward step 
            p(V_t, m_t | V_{t-1}, m_{t-1}, x_t) = p(m_t | x_t)p(V_t | V_{t-1}, m_{t-1}, x_t)
        '''
        if t == 0:
            V = self.V_init * jnp.ones((2, N_samples))
        else:
            V_prev, m_prev = prev_latent
            V = self.update_values(V_prev, X_prev, Y_prev, m_prev)

        if X.size != N_samples:
            m = jnp.array([self.encode(X) for _ in range(N_samples)])
        else:
            m = self.encode(X)
        return V, m
    
    def sample(self, T):
        # Generate stimulus uniformly from range
        x_range = jnp.linspace(-1,1,12)
        X = jax.random.choice(key, x_range, shape=(T,), replace=True)

        # Encode percept and define initial values
        m = self.encode(X)
        V = self.V_init * jnp.ones(2)

        # Generate decisions and values sequentially
        Y, Vs = [], []
        for t in range(T):
            y = self.decision(m[t], V)

            Vs.append(V)
            Y.append(y)

            V = self.update_values(V, x=X[t], y=Y[t], m=m[t])
        return X, jnp.stack(Y), m, jnp.stack(Vs)
    
class GLMLearn():
    r'''
    GLM-Learn behavioral model. 

    This models decision (`y`) making as a Bernoulli-GLM with regressors `x` and weigths `w`. 
    The weights evolve according to a learning rule. We consider here either the REINFORCE 
        learning rule, or a closed-form policy gradient update.
    '''
    def __init__(self, 
                 alpha: float=0.0, dynamics_logscale: float=-1.0, 
                 not_trainable: list=[],
                 learning_rule: str='policy_gradient', seed: int=0,
                 ) -> None:
        # self.alpha = alpha
        # self.sigma_w = sigma_w
        self.learning_rule = learning_rule

        self.params = ParamsGLMLearn(log_sigma=dynamics_logscale, alpha=alpha)
        self.props = ParamsGLMLearn(log_sigma=ParameterProperties(), alpha=ParameterProperties())
        for param in not_trainable:
            getattr(self.props, param).trainable = False

        # Initialization for latents and key for reproducibility
        self.key = jax.random.PRNGKey(seed)

    def update_params(self, **kwargs) -> None:
        '''passes trainable kwargs to self.params._replace
        #? Add to parent class?
        '''
        for key in kwargs:
            if not getattr(self.props, key).trainable:
                raise ValueError(f"Parameter '{key}' is not trainable")
        
        self.params = self.params._replace(**kwargs)

    def update_params_from_array(self, params_array: Iterable) -> None:
        '''Standardize way to set parameters from array.'''
        # params_dict = {'log_sigma': params_array[0], 'alpha': params_array[1]}
        # self.update_params(**params_dict)
        self.params = ParamsGLMLearn._make(params_array)
        
    def emission_likelihood(self, w, x, y=1):
        '''p(y | w, x), default p(y=1 | w, x)'''
        vx = vec(x)
        LM = jnp.dot(w, vx)
        p = safe_sigmoid(sign(y) * LM)
        return p

    def decision(self, w, x, key=None):
        if key is None:
            self.key, subkey = jax.random.split(self.key)
        else:
            subkey = key

        p_R = self.emission_likelihood(w, x, y=1.0)
        y = jax.random.bernoulli(subkey, p_R).astype(int)
        return y

    def policy_gradient(self, w, x, r=None):
        p_R = self.emission_likelihood(w, x, y=1)
        if r is None:
            r = effective_reward(x)
        return r * jnp.outer(jnp.multiply(p_R, 1 - p_R), vec(x)).squeeze()
    
    def reinforce(self, w, x, y, r=None):
        p = self.emission_likelihood(w, x, y)
        if r is None:
            r = reward(x,y)
        return r * jnp.outer(1-p, sign(y) * vec(x)).squeeze()
    
    @handle_none_params
    def update_weights(
            self, 
            w, x, 
            params: Optional[ParamsGLMLearn]=None, y=None, r=None,
            key=None):
        if key is None:
            self.key, subkey = jax.random.split(self.key)
        else:
            subkey = key
        
        if self.learning_rule == 'reinforce':
            learning_signal = params.alpha * self.reinforce(w, x, y, r)
        else: # use Policy gradient
            learning_signal = params.alpha * self.policy_gradient(w, x, r)

        update_noise = jax.random.normal(subkey, shape=w.shape)
        update_noise = jnp.multiply(jnp.exp(params.log_sigma), update_noise)
        return w + learning_signal + update_noise
    
    def initial_loglikelihood(self, z_0):
        """log p(z_0). We use p(z_0) = N(0,I)."""
        if z_0.shape == (1,):
            log_lik = lambda z: jsp.stats.norm.logpdf(z)#, loc=z_0_mean, scale=1.0)
        else:
            # N = w_init_mean.shape[0]
            log_lik = lambda z: jsp.stats.multivariate_normal.logpdf(z)#, mean=w_init_mean, cov=np.eye(N))
        return log_lik(z_0)
    
    @handle_none_params
    def dynamics_loglikelihood(self, z_next, z_prev, inputs, data, 
                               params: Optional[ParamsGLMLearn]=None, r=None):
        '''p(z_t | z_{t-1})
        In our case, the latents z are the GLM weights w
        '''
        if self.learning_rule == 'reinforce':
            learning_signal = params.alpha * self.reinforce(z_prev, inputs, data, r=r)
        else: # use Policy gradient
            learning_signal = params.alpha * self.policy_gradient(z_prev, inputs, r=r)
        mean = z_prev + learning_signal
        N = mean.shape[0]
        cov = jnp.multiply(jnp.square(jnp.exp(params.log_sigma)), jnp.eye(N))
        log_lik = lambda z: jsp.stats.multivariate_normal.logpdf(z, mean=mean, cov=cov)
        return log_lik(z_next)
    
    def sample(self, T, key=None):
        '''
        Samples from the model, focusing only on univariate stimuli (stimulus intensity).
        Returns:
            X: array, stimulus, of shape (T,)
            Y: array, decisions, of shape (T,)
            W: array, weights, of shape (T, 2)
        '''
        if key is None:
            key = self.key

        key, init_key = jax.random.split(key)

        # Generate stimulus uniformly from range
        x_range = jnp.linspace(-1,1,12)
        X = jax.random.choice(init_key, x_range, shape=(T,), replace=True)

        # Encode percept and define initial values
        w = jax.random.normal(init_key, shape=(2,)) # w_0 ~ N(0, 1)

        # Generate decisions and weights sequentially
        Y, Ws = [], []
        for t in range(T):
            key, decision_key, update_key = jax.random.split(key, num=3)

            # Sample
            y = self.decision(w, X[t], key=decision_key)
            Y.append(y)
            Ws.append(w)

            # Update
            w = self.update_weights(w, x=X[t], y=Y[t], key=update_key)
        return X, jnp.stack(Y), jnp.stack(Ws)
    
    @handle_none_params
    def log_joint(
            self, X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray, params: Optional[ParamsGLMLearn]=None
            ) -> float:
        '''
        #! add rewards
        Evaluate `log p(Y, Z | X, theta) = log p(y_{1:T}, z_{1:T} | x_{1:T}, theta)`. 

        parameters:
            X: array, stimulus, of shape (T, input_dim)
            Y: array, decisions, of shape (T, output_dim)
            Z: array, latent variables, of shape (T, latent_dim)
            theta: array of floats, the parameters {w_init, alpha, log_sigma}

        returns: 
            log_joint: float, value of log joint likelihood
        '''
        T = len(Y)
        # w_init = self.w_init_mean
        # w_init = theta[:2]
        # alpha = theta[-2]
        # sigma = jnp.exp(theta[-1])

        # Initial t=0 joint likelihood terms
        # Dynamics
        log_pz0 = self.initial_loglikelihood(Z[0]) #, w_init_mean=w_init)

        # Emissions
        log_pyz0 = jnp.log(self.emission_likelihood(Z[0], X[0], Y[0]))

        log_joint = log_pz0 + log_pyz0

        # Loop over time steps 
        for t in range(1,T):
            # Dynamics
            log_pzz = self.dynamics_loglikelihood(Z[t], Z[t-1], X[t-1], Y[t-1], params=params)

            # Emissions
            log_pyz = jnp.log(self.emission_likelihood(Z[t], X[t], Y[t]))

            # Update value and gradient of log joint
            log_joint += log_pzz + log_pyz

        return log_joint

if __name__=='__main__':
    # Seed for reproducibility
    seed = 0
    key = jax.random.PRNGKey(seed)

    # true_model = QLearningModel(sigma=0.3, alpha=0.5, softmax=False)
    # X, Y, m, Vs = true_model.sample(10)
    # print(X, Y, m, Vs)

    true_model = GLMLearn(dynamics_logscale=-1.0, alpha=1.0, seed=seed)
    X, Y, Ws = true_model.sample(10)
    print(X, Y, Ws)