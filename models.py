import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.random import PRNGKey
import numpy as np
import scipy as sp
from typing import Tuple, Optional, Iterable, Union
from functools import partial
from tqdm import tqdm

import tensorflow_probability.substrates.jax.distributions as tfd

import os
os.environ['JAX_PLATFORMS']='cpu'

from parameters import ParamsGLMLearn, ParameterProperties #, handle_none_params

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@jax.jit
def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

@jax.jit
def safe_sigmoid(X, threshold=80.):
    return jnp.where(
        X > threshold, 
        jnp.ones_like(X), 
        jnp.where(
            X < -threshold, 
            jnp.zeros_like(X), 
            sigmoid(X)
            )
            )

def vec(x):
    x = jnp.asarray(x)
    if x.ndim == 0:
        return jnp.array([1,x])
    elif x.ndim == 1:
        return jnp.concatenate((jnp.array([1.0]),x))
    else:
        print(f"Input x has shape {x.shape}. Assuming (T,D)")
        if x.ndim > 2:
            raise NotImplementedError
        return jnp.concatenate((jnp.ones((x.shape[0],1)),x), axis=1)

def softmax_forward(x):
    return jax.nn.softmax(jnp.concatenate([x, jnp.zeros((1,))]))

# def unvec(x): # has to be consistent with vec(). remove if not used
#     return x[1:]

@jax.jit
def sign(y: jnp.ndarray):
    '''
    y: array-like, 
    '''
    y = jnp.asarray(y).astype(bool).astype(float)
    # return jnp.where(y == 0., -1., 1.)
    return 2*y - 1

def correct_choice(x: jnp.ndarray):
    '''
    Returns the side {0,1} of the stimulus. 
    Also corresponds to the correct choice.
    '''
    # x = jnp.asarray(x).astype(float)
    return jnp.where(x < 0, 0, 1)

def reward(X, Y, r1=1.0, r0=0.0) -> np.ndarray:
    '''
    Returns reward 
        r(x,y)= r1 if (x < 0 and y == 0) or (x > 0 and y==1)
                r0 else
    for (x,y) pairs in zip(X, Y)
    '''
    X = jnp.array(X).squeeze()
    Y = jnp.array(Y).squeeze().astype(int)
    
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

# @partial(jax.jit, static_argnums=(0,))
def set_day_flags(T, session_indices):
    day_flags = jnp.zeros(T, dtype=bool)
    day_flags = day_flags.at[session_indices].set(True)
    day_flags = day_flags.at[0].set(False) # do not use 0 session index as a new day for transitions
    return day_flags

@jax.jit
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

@jax.jit
def bernoulli_GLM_likelihood(w, x, y):
    '''Log-likelihood for a Bernoulli GLM'''
    vx = vec(x)
    # if vx.ndim == 1:
    LM = jnp.dot(w, vx)
    # else:
    #     # print(w.shape, vx.shape)
    #     # LM = w @ vx.T
    #     LM = jnp.einsum('ij,ij->i', w, vx)
    #     print(LM.shape)
    # p = safe_sigmoid(sign(y) * LM)
    p = sigmoid(sign(y) * LM)
    return p

@jax.jit
def policy_gradient(w, x, r=None):
    p_R = bernoulli_GLM_likelihood(w, x, y=1.0)
    if r is None:
        r = effective_reward(x)
    return r * jnp.outer(jnp.multiply(p_R, 1 - p_R), vec(x)).squeeze()

@jax.jit
def reinforce(w, x, y, r=None):
    p = bernoulli_GLM_likelihood(w, x, y)
    if r is None:
        r = reward(x,y)
    return r * jnp.outer(1-p, sign(y) * vec(x)).squeeze()

@jax.jit
def maximum_likelihood(w, x):
    z = correct_choice(x)
    p = bernoulli_GLM_likelihood(w, x, z)
    return jnp.outer(1-p, sign(z) * vec(x)).squeeze()

def linear_regression_gradient(w, x, y, params):
    p_c = bernoulli_GLM_likelihood(w, x, y)
    r = reward(x, y)
    return params.Q @ w + r * (params.A @ vec(x) + params.kappa) + params.b + \
            r * jnp.outer(params.alpha * (1 - p_c) + params.beta * jnp.multiply(p_c, 1 - p_c), vec(x)).squeeze()

class LearningRule():
    '''
    A learning rule is a probability distribution over the next weights given the current weights and the data.
        p(w_t | w_{t-1}, x_t, y_t)
    '''
    def update_weights(self, 
                       weights: jnp.ndarray, 
                       inputs: jnp.ndarray, emissions: jnp.ndarray, rewards: jnp.ndarray, 
                       params
                       ):
        raise NotImplementedError
    
    def log_likelihood(self, weights, inputs, emissions, rewards, params):
        raise NotImplementedError

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
                #  log_alpha: Union[float, jnp.ndarray] = 0.0, log_sigma: float=-1.0, log_sigma_day=-1.0,
                #  not_trainable: list=[], 
                 z_0: Union[float, jnp.ndarray] = 0.0,
                 learning_rule: str='policy_gradient', # seed: int=0,
                 latent_dim = None,
                 ) -> None:
        self.learning_rule = learning_rule.lower()

        # self.params = ParamsGLMLearn(log_sigma=log_sigma, log_alpha=log_alpha, log_sigma_day=log_sigma_day)
        # self.props = ParamsGLMLearn(
        #     log_sigma=ParameterProperties(), 
        #     log_alpha=ParameterProperties(),
        #     log_sigma_day=ParameterProperties()
        #     )
        # for param in not_trainable:
        #     getattr(self.props, param).trainable = False

        # Initialization for latents and key for reproducibility
        # self.key = PRNGKey(seed)
        self.z_0 = z_0
        self.latent_dim = latent_dim

    # def update_params(self, **kwargs) -> None:
    #     '''passes trainable kwargs to self.params._replace
    #     #? Add to parent class?
    #     '''
    #     for key in kwargs:
    #         if not getattr(self.props, key).trainable:
    #             raise ValueError(f"Parameter '{key}' is not trainable")
        
    #     self.params = self.params._replace(**kwargs)

    # def update_params_from_array(self, params_array: Iterable) -> None:
    #     '''Standardize way to set parameters from array.'''
    #     # params_dict = {'log_sigma': params_array[0], 'alpha': params_array[1]}
    #     # self.update_params(**params_dict)
    #     self.params = ParamsGLMLearn._make(params_array)
        
    def decision(self, key: PRNGKey, w, x):
        # if key is None:
        #     self.key, subkey = jax.random.split(self.key)
        # else:
        #     subkey = key

        p_R = bernoulli_GLM_likelihood(w, x, y=1)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y
    
    # @handle_none_params
    def update_weights(
            self, 
            key: PRNGKey,
            w, x, 
            params: ParamsGLMLearn, y=None, r=None,
            day_flag: bool=False,
            return_noise=False):
        
        # Change in mean weights from learning rule
        if self.learning_rule == 'reinforce':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), reinforce(w, x, y, r))
        elif self.learning_rule == 'policy_gradient':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), policy_gradient(w, x, r)) # alpha * jnp.ones(w.shape[1])
        elif self.learning_rule == 'maximum_likelihood':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), maximum_likelihood(w, x))
        else:
            raise ValueError(f"Learning rule {self.learning_rule} not implemented.")

        # Add noise
        update_noise = jax.random.normal(key, shape=w.shape)
        update_noise = jnp.where(day_flag, 
                                 jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
                                 jnp.multiply(jnp.exp(params.log_sigma), update_noise)
                                 )
        
        if return_noise:
            return w + learning_signal + update_noise, update_noise
        else:
            return w + learning_signal + update_noise
    
    def initial_loglikelihood(self, z_0):
        """log p(z_0). We use p(z_0) = N(0,I)."""
        if z_0.shape == (1,):
            log_lik = lambda z: jsp.stats.norm.logpdf(z, loc=self.z_0, scale=1.0)
        else:
            D = z_0.shape[0]
            assert z_0.shape == (D,), f"z_0 shape {z_0.shape} does not match expected shape ({D},)"
            log_lik = lambda z: jsp.stats.multivariate_normal.logpdf(
                z, mean=self.z_0 * jnp.ones(D), cov=jnp.diag(jnp.ones(D))
                )
        return log_lik(z_0)

    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        # if key is None:
        #     self.key, subkey = jax.random.split(self.key)
        # else:
        #     subkey = key

        # if z_0 is None:
        #     z_0 = jnp.zeros((d+1))

        # key, subkey = jax.random.split(self.key)
        return self.z_0 + jax.random.normal(key, shape=(N, d+1,))
    
    # @handle_none_params
    def dynamics_loglikelihood(self, z_next, z_prev, inputs, data, 
                               params: ParamsGLMLearn, day_flag=False, r=None):
        '''p(z_t | z_{t-1})
        In our case, the latents z are the GLM weights w
        '''
        if self.learning_rule == 'reinforce':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), reinforce(z_prev, inputs, data, r=r))
        else: # use Policy gradient
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), policy_gradient(z_prev, inputs, r=r))
        mean = z_prev + learning_signal
        N = mean.shape[0]

        # if day_flag:
        #     cov = jnp.multiply(jnp.square(jnp.exp(params.log_sigma_day)), jnp.eye(N))
        # else:
        #     cov = jnp.multiply(jnp.square(jnp.exp(params.log_sigma)), jnp.eye(N))

        cov = jnp.where(day_flag, 
            jnp.multiply(jnp.square(jnp.exp(params.log_sigma_day)), jnp.eye(N)),
            jnp.multiply(jnp.square(jnp.exp(params.log_sigma)), jnp.eye(N))
            )
        
        log_lik = lambda z: jsp.stats.multivariate_normal.logpdf(z, mean=mean, cov=cov)
        return log_lik(z_next)
    
    def emission_likelihood(self, z, x, y):
        '''p(y | z, x)'''
        return bernoulli_GLM_likelihood(z, x, y)
    
    def sample(self, key, params, T):
        '''
        Samples from the model, focusing only on univariate stimuli (stimulus intensity).
        Returns:
            X: array, stimulus, of shape (T, 1,)
            Y: array, decisions, of shape (T,)
            W: array, weights, of shape (T, 2, )
        '''
        key, init_key = jax.random.split(key)

        # Generate stimulus uniformly from range
        x_range = jnp.linspace(-1,1,8)
        X = jax.random.choice(init_key, x_range, shape=(T,1,), replace=True)

        # Encode percept and define initial values
        w = jax.random.normal(init_key, shape=(2,)) # w_0 ~ N(0, 1)

        # Generate decisions and weights sequentially
        Y, Ws, noises = [], [], []
        for t in range(T):
            key, decision_key, update_key = jax.random.split(key, num=3)

            # Sample
            y = self.decision(decision_key, w, X[t])
            Y.append(y)
            Ws.append(w)

            if self.learning_rule == 'reinforce':
                r = reward(X[t], y)
            elif self.learning_rule == 'policy_gradient':
                r = effective_reward(X[t])

            # Update
            w, eps = self.update_weights(update_key, w, params=params, x=X[t], y=Y[t], r=r, return_noise=True)
            noises.append(eps)
        return X, jnp.stack(Y), jnp.stack(Ws), noises
    
    # @handle_none_params
    def log_joint(
            self, 
            X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray, 
            params: ParamsGLMLearn, R: Optional[jnp.ndarray]=None,
            session_indices: Optional[list]=[],
            ) -> float:
        '''
        Evaluate `log p(Y, Z | X, R, theta) = log p(y_{1:T}, z_{1:T} | x_{1:T}, r_{1:T}, theta)`. 

        parameters:
            X: array, stimulus, of shape (T, input_dim)
            Y: array, decisions, of shape (T, output_dim)
            Z: array, latent variables, of shape (T, latent_dim) = (T, input_dim + 1)
            params: ParamsGLMLearn, model parameters

        returns: 
            log_joint: float, value of log joint likelihood
        #? Add potentially to parent class
        '''
        # Format arguments
        T = len(Y)
        if R is None:
            R = [None]*T
        day_flags = set_day_flags(T, session_indices)

        # Initial t=0 dynamics likelihood terms
        log_pz0 = self.initial_loglikelihood(Z[0])

        # # Evaluate dynamics likelihoods
        log_pzz = jax.vmap(
            lambda z1, z0, x0, y0, day_flag, r0: self.dynamics_loglikelihood(z1, z0, x0, y0, params=params, day_flag=day_flag, r=r0)
        )(
            Z[1:], Z[:-1], X[:-1], Y[:-1], day_flags[1:], R[:-1], 
        )

        # Evaluate emissions likelihood log p(y_{0:T} | z_{0:T})
        log_pyz = jax.vmap(
            lambda z, x, y: jnp.log(bernoulli_GLM_likelihood(z, x, y)), #TODO write as self.emission_loglikelihood
        )(Z, X, Y)
        
        # Combine to obtain log p(y_{0:T}, z_{0:T})
        log_joint = log_pyz.sum() + log_pz0 + log_pzz.sum()

        return log_joint
    
    def posterior_samples(self, 
            key, 
            params, 
            X, Y, R=None, session_indices=[], 
            N_particles=1000, return_history=True, posterior_type='smooth', verbose=False
            ):
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        if self.latent_dim is None:
            self.latent_dim = M + 1
        day_flags = set_day_flags(T, session_indices)

        if return_history:
            # Block out N x T x M+1 array (float32, x 4 in bytes) to store z_t samples.
            # A lot of memory, but much faster. 
            z_history = jnp.zeros((N_particles, T, self.latent_dim), dtype=jnp.float32) # float16

        if verbose:
            pbar = tqdm(range(0,T), desc='Bootstrap filter')
    
        log_lik = 0.
        for t in range(0,T):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            #   Sample proposal N particles from previous N particles
            #   Outcome: {tilde z_t^i, 1/N}, an approximation to p(z_t|y_{1:t-1})
            if t == 0:
                tilde_z_t = self.sample_initial(subkey1, params=params, N=N_particles, d=M)
            else:
                tilde_z_t = self.update_weights(subkey1, z_t, params=params, x=X[t-1], y=Y[t-1], r=R[t-1], day_flag=day_flags[t])

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            #   Outcome: {tilde z_t^i, tilde w^i}, an approximation to p(z_t|y_{1:t})
            # tilde_w_t = bernoulli_GLM_likelihood(y=Y[t], w=tilde_z_t, x=X[t]) 
            tilde_w_t = self.emission_likelihood(tilde_z_t, X[t], Y[t])
            
            # 3. Resample with replacement N particles according the importance weights
            #   Outcome: {z_t, 1/N}, an approximation to p(z_t|y_{1:t})
            if return_history:
                if posterior_type == 'smooth':
                    z_history = z_history.at[:,t,:].set(tilde_z_t)
                    z_history = jax.random.choice(subkey2, z_history, shape=(N_particles,), p=tilde_w_t)
                    z_t = z_history[:,t,:]

                elif posterior_type == 'filt':
                    z_t = jax.random.choice(subkey2, tilde_z_t, shape=(N_particles,), p=tilde_w_t)
                    z_history = z_history.at[:,t,:].set(z_t)
            else:
                z_t = jax.random.choice(subkey2, tilde_z_t, shape=(N_particles,), p=tilde_w_t)

            # 4. Update log-likelihood estimate
            log_lik += jnp.log(jnp.mean(tilde_w_t))

            if verbose:
                pbar.update(1)

        if not return_history:
            z_history = z_t

        return z_history, log_lik

    def marginal_log_likelihood(self, key, params, X, Y, R=None, session_indices=[],
                                N_particles=1000, verbose=False):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        day_flags = set_day_flags(T, session_indices)

        def scan_fn(carry, inputs):
            tilde_z_t, log_lik = carry
            X_t, Y_t, R_t, next_day_flag, (subkey1, subkey2) = inputs

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            tilde_w_t = self.emission_likelihood(tilde_z_t, X_t, Y_t) 
            
            # 3. Resample with replacement N particles according the importance weights
            z_t = jax.random.choice(subkey1, tilde_z_t, shape=(N_particles,), p=tilde_w_t)

            # 4. Update log-likelihood estimate
            log_lik += jnp.log(jnp.mean(tilde_w_t))
            
            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            tilde_z_t = self.update_weights(subkey2, z_t, params=params, x=X_t, y=Y_t, r=R_t, day_flag=next_day_flag)
            
            return (tilde_z_t, log_lik), None
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=1)
        subkeys = jax.random.split(key, num=(T, 2))
        inputs = (X, Y, R, next_day_flags, subkeys)
        
        (_, log_lik), _ = jax.lax.scan(scan_fn, carry, inputs, length=T)
        return log_lik
    
    
    def score_predict(self,
            key, params,
            X_hist: jnp.ndarray, Y_hist: jnp.ndarray,
            X_pred: jnp.ndarray, Y_pred: jnp.ndarray,
            R_hist: jnp.ndarray = None, session_indices: list=[],
            N_particles: int=10000,
            ):
        '''
        Do filtering to obtain last weights, then sample weights trajectories from there and compare
        sampled decisions with true decisions.
        '''
        T = len(X_hist) + len(X_pred)
        day_flags = set_day_flags(T, session_indices)

        # Step 1: filtering to obtain last weights
        # (_, Zs_filt), _ = samplers.bootstrap_filter(
        #     N_particles, 
        #     X_hist, Y_hist, 
        #     model, 
        #     R=R_hist, session_indices=session_indices, 
        #     return_history=False, verbose=True,
        #     )
        Zs_filt_T, _ = self.posterior_samples(
            key, params, X_hist, Y_hist, R=R_hist, session_indices=session_indices,
            N_particles=N_particles, return_history=False, posterior_type='filt',
        )
        w = Zs_filt_T.mean(0)

        # Step 2: sample weights trajectories
        Ys, Ws = [], []
        for t in range(len(X_pred)):
            key, decision_key, update_key = jax.random.split(key, 3)

            # Decision
            y = self.decision(decision_key, w, X_pred[t])

            if self.learning_rule == 'reinforce':
                r = reward(X_pred[t,1]-X_pred[t,0], y)
            elif self.learning_rule == 'policy_gradient':
                r = effective_reward(X_pred[t,1]-X_pred[t,0])

            # Update
            w = self.update_weights(update_key, w, params=params, x=X_pred[t], y=y, r=r, return_noise=False, day_flag=day_flags[t])

            Ys.append(y)
            Ws.append(w)

        # Compute score 
        score = jnp.mean(jnp.array(Ys) == jnp.array(Y_pred))
        return score
    
    def next_step_prediction_score(
            self, key, params, X, Y, R=None, session_indices=[],
            N_particles=1000, verbose=False):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        day_flags = set_day_flags(T, session_indices)

        def scan_fn(carry, inputs):
            tilde_z_t, log_lik = carry
            X_t, Y_t, R_t, next_day_flag, (subkey1, subkey2, subkey3) = inputs

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            tilde_w_t = self.emission_likelihood(tilde_z_t, X_t, Y_t)

            # 3. Resample with replacement N particles according the importance weights
            z_t = jax.random.choice(subkey2, tilde_z_t, shape=(N_particles,), p=tilde_w_t)

            # 4. Update log-likelihood estimate
            lik = jnp.mean(tilde_w_t)
            log_lik += jnp.log(lik)
            
            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            #    Outcome: {tilde z_t^i, 1/N}, an approximation to p(z_t|y_{1:t-1})
            tilde_z_t = self.update_weights(subkey3, z_t, params=params, x=X_t, y=Y_t, r=R_t, day_flag=next_day_flag)
            
            return (tilde_z_t, log_lik), lik
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=1)
        subkeys = jax.random.split(key, num=(T, 3))
        inputs = (X, Y, R, next_day_flags, subkeys)
        
        (_, log_lik), scores = jax.lax.scan(scan_fn, carry, inputs, length=T)
        return scores, log_lik
    
    
    def two_step_prediction_score(
            self, key, params, X, Y, R=None, session_indices=[],
            N_particles=1000, verbose=False):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        day_flags = set_day_flags(T, session_indices)

        def scan_fn(carry, inputs):
            tilde_z_t, log_lik = carry
            X_t1, Y_t1, R_t1, X_t2, Y_t2, next_day_flag, (forward_key, subkey2, subkey3) = inputs

            # Emission predicition
            # y_pred = self.decision(subkey1, tilde_z_t, X_t)
            # score = jnp.mean(y_pred == Y_t)

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            tilde_w_t = self.emission_likelihood(tilde_z_t, X_t1, Y_t1)

            # 2 step look ahead
            z_forward = tilde_z_t.copy()
            z_forward = self.update_weights(
                forward_key, z_forward, params=params, x=X_t1, y=Y_t1, r=R_t1, day_flag=next_day_flag
                )
            pz2 = self.emission_likelihood(z_forward, X_t2, Y_t2)
            
            # 3. Resample with replacement N particles according the importance weights
            z_t = jax.random.choice(subkey2, tilde_z_t, shape=(N_particles,), p=tilde_w_t)

            # 4. Update log-likelihood estimate
            lik = jnp.mean(tilde_w_t)
            log_lik += jnp.log(lik)
            
            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            tilde_z_t = self.update_weights(subkey3, z_t, params=params, x=X_t1, y=Y_t1, r=R_t1, day_flag=next_day_flag)
            
            return (tilde_z_t, log_lik), pz2
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=1)
        subkeys = jax.random.split(key, num=(T-1, 3))
        inputs = (X[:-1], Y[:-1], R[:-1], X[1:], Y[1:], next_day_flags[:-1], subkeys)
        (_, log_lik), scores = jax.lax.scan(scan_fn, carry, inputs, length=T-1)
        return scores, log_lik
    
    def alpha_mcmc(self, 
                   key, params, 
                   X, Y, R=None, session_indices=[],
                   N_particles=1000, n_iters=100, N_samples=100,
                   verbose=True, proposal_scale=1.0,
                   ):
        '''Metropolis hastings to sample from posterior of alpha.'''
        proposal = lambda x: tfd.Normal(loc=x, scale=proposal_scale)

        @partial(jax.vmap, in_axes=(0,))
        def log_target(log_alpha) -> float:
            params_prop = params._replace(log_alpha=log_alpha)
            return self.marginal_log_likelihood(key, params_prop, X, Y, R, session_indices, N_particles)
        
        @jax.jit
        def metropolis_hastings_step(log_alpha, key):
            '''
            Single step of Metropolis-Hastings. 
            Written in form amendable to jax.lax.scan.
            '''
            key, proposal_key, accept_key = jax.random.split(key, 3)

            # Proposal step
            log_alpha_prop = proposal(log_alpha).sample(seed=proposal_key)
            log_alpha_prop = jnp.clip(log_alpha_prop, -10, 2)

            # Acceptance step
            log_ratio = log_target(log_alpha_prop) - log_target(log_alpha)
            accept = jnp.log(jax.random.uniform(accept_key)) < log_ratio

            log_alpha = jnp.where(accept[:, None], log_alpha_prop, log_alpha)
            return log_alpha, (log_alpha, accept)
        
        # Initialize
        keys = jax.random.split(key, n_iters)
        log_alpha = params.log_alpha
        log_alpha_samples_t = jnp.tile(log_alpha, (N_samples, 1))
        
        if verbose:
            log_alpha_samples = []
            for i in range(n_iters):
                _, (log_alpha_samples_t, accepts) = metropolis_hastings_step(log_alpha_samples_t, keys[i])
                logging.info(f"[{i}/{n_iters}] log alpha mean: {log_alpha_samples_t.mean(0)}, accept frac: {accepts.sum()/N_samples:.2f}")

                # Print confidence intervals
                if i % 10 == 0:
                    logging.info(f"log alpha CI: {jnp.percentile(log_alpha_samples_t, q=jnp.array([2.5, 97.5]), axis=0).T}")

                log_alpha_samples.append(log_alpha_samples_t)
            log_alpha_samples = jnp.stack(log_alpha_samples)
        else:
            # Wrap in scan
            _, (log_alpha_samples, accepts) = jax.lax.scan(
                metropolis_hastings_step,
                log_alpha_samples_t, keys, length=n_iters
                )
            accepts = accepts[-1]
            
        assert log_alpha_samples.shape[0] == n_iters and log_alpha_samples.shape[1] == N_samples
        
        logging.info(f"log alpha mean: {log_alpha_samples[-1].mean(0)}, accept frac: {accepts.mean():.2f}")
        logging.info(f"log alpha CI: {jnp.percentile(log_alpha_samples[-1], q=jnp.array([2.5, 97.5]), axis=0).T}")
        # return log_alpha_samples
        return log_alpha_samples, accepts

from typing import NamedTuple
class ParamsTimeVarGLMLearn(NamedTuple):
    beta_0: float
    log_alpha: float
    log_sigma_0: float
    log_sigma: float
    log_sigma_day: float

class TimeVarGLMLearn(GLMLearn):
    r'''
    We posit a random walk prior for the learning rate, 
    $$
        \beta_{t+1} = \beta_t + \epsilon_t
    $$
    and have the learning rule now be
    $$
        \w_{t+1} = \w_t + \alpha \beta_t \mathcal{L} + \epsilon_t
    $$
    and with a decision making model of the form,
    $$
        p(y = R\mid \w, \vec x, \alpha) = (1-2*\sigma(\beta)) * \sigma(\w^\top \vec x)   + \sigma(\beta)
    $$
    so that $\sigma(\alpha)$ represents the lapse rate. 
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def split_latent(self, z):
        if z.ndim == 1:
            beta, w = jnp.array(z[0]), z[1:]
        else:
            beta, w = z[:,0], z[:,1:]
        return beta, w
    
    def merge_latent(self, beta, w):
        if w.ndim == 1:
            # alpha is scalar
            return jnp.concatenate([jnp.array([beta]), w])
        else:
            return jnp.concatenate([beta[:, None], w], axis=1)

    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        beta = params.beta_0 + jnp.exp(params.log_sigma_0) * jax.random.normal(key, shape=(N,))
        w = super().sample_initial(key, params=None, N=N, d=d)
        return self.merge_latent(beta, w)
        
    def update_weights(self, key: PRNGKey, z, x, params, y=None, r=None, day_flag=False, return_noise=False):
        beta_t, w_t = self.split_latent(z)

        key, beta_key, w_key = jax.random.split(key, 3)
        beta_t = beta_t + jnp.exp(params.log_sigma_0) * jax.random.normal(beta_key, shape=beta_t.shape)

        log_learning_rate = params.log_alpha + jnp.log(jnp.clip(beta_t, 0, 1)) #jnp.log((1-sigmoid(beta_t)))
        # jnp.log(jnp.clip(beta_t, 0, 1))[:, None] if w_t.ndim > 1 else jnp.log(jnp.clip(alpha_t, 0, 1))
        params_GLM = ParamsGLMLearn(
            log_sigma=params.log_sigma, 
            log_alpha=log_learning_rate[:, None] if w_t.ndim > 1 else log_learning_rate, 
            log_sigma_day=params.log_sigma_day
            )
        w_t = super().update_weights(w_key, w_t, x, params_GLM, y, r, day_flag)

        z_t = self.merge_latent(beta_t, w_t)
        return z_t
    
    def emission_likelihood(self, z, x, y):
        beta, w = self.split_latent(z)
        p = bernoulli_GLM_likelihood(w, x, y)
        # p = (1-sigmoid(beta)) * p_GLM + sigmoid(beta)
        return p
    
    def decision(self, key: PRNGKey, z, x):
        p_R = self.emission_likelihood(z, x, y=1)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y
    
    def dynamics_loglikelihood(self, z_next, z_prev, inputs, data, params: ParamsGLMLearn, day_flag=False, r=None):
        raise NotImplementedError
    
    def log_joint(self, X, Y, Z, params, R=None, session_indices=[]):
        raise NotImplementedError


class ParamsGLMHMMLearn(NamedTuple):
    A: jnp.ndarray
    log_alpha: float
    log_sigma: float
    log_sigma_day: float

class GLMHMMLearn(GLMLearn):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def split_latent(self, z):
        if z.ndim == 1:
            state, w = jnp.array(z[0], dtype=int), z[1:]
        else:
            state, w = z[:,0], z[:,1:]
        return state, w
    
    def merge_latent(self, state, w):
        if w.ndim == 1:
            # alpha is scalar
            return jnp.concatenate([jnp.array([state], dtype=int), w])
        else:
            return jnp.concatenate([state[:, None], w], axis=1)
        
    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        state = jax.random.choice(key, 2, shape=(N,))
        w = super().sample_initial(key, params=None, N=N, d=d)
        return self.merge_latent(state, w)
    
    def update_weights(self, key: PRNGKey, z, x, params, y=None, r=None, day_flag=False, return_noise=False):
        '''
        State dependent weight updates. In one state, we follow a policy gradient update, 
        in the other there is no learning. 
        '''
        state_t, w_t = self.split_latent(z)
        key, state_key, w_key = jax.random.split(key, 3)

        params_GLM = ParamsGLMLearn(log_sigma=params.log_sigma, log_alpha=params.log_alpha, log_sigma_day=params.log_sigma_day)
        w_t = jnp.where(state_t[:, None], 
                  super().update_weights(
                      w_key, w_t, x, 
                      ParamsGLMLearn(log_sigma=params.log_sigma, log_alpha=-20., log_sigma_day=params.log_sigma_day), 
                      y, r, day_flag
                      ),
                  super().update_weights(
                      w_key, w_t, x, 
                      ParamsGLMLearn(log_sigma=params.log_sigma, log_alpha=params.log_alpha, log_sigma_day=params.log_sigma_day), 
                      y, r, day_flag
                      )
        )

        state_t = jax.vmap(lambda A: jax.random.choice(state_key, 2, p=A))(params.A[state_t.astype(int)])
        z_t = self.merge_latent(state_t, w_t)
        return z_t
    
    def emission_likelihood(self, z, x, y):
        _, w = self.split_latent(z)
        p = bernoulli_GLM_likelihood(w, x, y)
        return p
    
    def decision(self, key: PRNGKey, z, x):
        p_R = self.emission_likelihood(z, x, y=1)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y



if __name__=='__main__':
    # Seed for reproducibility
    seed = 1
    key = PRNGKey(seed)

    array = jnp.array([0.2, 0.3, 0.4])
    print(softmax_forward(array))

    # X = jax.random.uniform(key, shape=(10, 1))
    # Y = jax.random.bernoulli(key, p=0.5, shape=(10,)).astype(int)

    # print(reward(X, Y))

    # true_model = QLearningModel(sigma=0.3, alpha=0.5, softmax=False)
    # X, Y, m, Vs = true_model.sample(10)
    # print(X, Y, m, Vs)

    # true_params = ParamsGLMLearn(log_sigma=-2.744629, log_sigma_day=-1.1630859, log_alpha=jnp.array([-4.8521523, -1.7326317]))
    # true_model_PG = GLMLearn(**true_params._asdict(), seed=seed, learning_rule='policy_gradient')
    # _, _, Ws_PG, _ = true_model_PG.sample(1000)

    # true_model_R = GLMLearn(**true_params._asdict(), seed=seed, learning_rule='reinforce')
    # _, _, Ws_R, _ = true_model_R.sample(1000)

    # import matplotlib.pyplot as plt 
    # fig, axs = plt.subplots(nrows=2, constrained_layout=True)
    # axs[0].plot(Ws_PG[:,0], c="tab:blue", label='Policy Gradient')
    # axs[0].plot(Ws_PG[:,1], c="tab:orange")
    # axs[0].plot(Ws_R[:,0], c="tab:blue", ls='--', label='REINFORCE')
    # axs[0].plot(Ws_R[:,1], c="tab:orange", ls='--',)
    # # axs[0].plot(Ws_R, label='REINFORCE')
    # axs[1].plot(Ws_PG - Ws_R)
    # plt.savefig('figures/weights_logalpha-2.png', dpi=300)
    # plt.close()

    # for log_sigma in np.linspace(-5,-1,10):
    #     print(log_sigma, true_model.log_joint(X, Y, Ws, params=ParamsGLMLearn(log_sigma, -1.0, 0.5)))
    # print(X, Y, Ws)