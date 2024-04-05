import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from typing import Tuple, Optional, Iterable, Union

import os
os.environ['JAX_PLATFORMS']='cpu'

from parameters import ParamsGLMLearn, ParameterProperties, handle_none_params

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

    
# def unvec(x): # has to be consistent with vec(). remove if not used
#     return x[1:]

@jax.jit
def sign(y):
    '''
    y: array-like, 
    '''
    y = jnp.asarray(y).astype(bool)
    return jnp.where(y == 0., -1., 1.)

def correct_choice(x):
    '''
    Returns the side {0,1} of the stimulus. 
    Also corresponds to the correct choice.
    '''
    x = jnp.asarray(x).astype(float)
    return jnp.where(x < 0, 0, 1)

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
    p = safe_sigmoid(sign(y) * LM)
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
    # print(r.shape, p.shape, vec(x).shape, sign(y).shape)

    # signed_vx = jnp.multiply(sign(y)[:,None], vec(x))
    # print(signed_vx.shape)
    # return r * 1-p * sign(y) * vec(x)
    return r * jnp.outer(1-p, sign(y) * vec(x)).squeeze()

@jax.jit
def maximum_likelihood(w, x):
    z = correct_choice(x)
    p = bernoulli_GLM_likelihood(w, x, z)
    return jnp.outer(1-p, sign(z) * vec(x)).squeeze()

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
        self.learning_rule = learning_rule.lower()

        self.params = ParamsGLMLearn(log_sigma=log_sigma, alpha=alpha, log_sigma_day=log_sigma_day)
        self.props = ParamsGLMLearn(
            log_sigma=ParameterProperties(), 
            alpha=ParameterProperties(),
            log_sigma_day=ParameterProperties()
            )
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
        
    def decision(self, w, x, key=None):
        if key is None:
            self.key, subkey = jax.random.split(self.key)
        else:
            subkey = key

        p_R = bernoulli_GLM_likelihood(w, x, y=1)
        y = jax.random.bernoulli(subkey, p_R).astype(int)
        return y
    
    @handle_none_params
    def update_weights(
            self, 
            w, x, 
            params: Optional[ParamsGLMLearn]=None, y=None, r=None,
            day_flag: bool=False,
            key=None, return_noise=False):
        if key is None:
            self.key, subkey = jax.random.split(self.key)
        else:
            subkey = key
        
        # Change in mean weights from learning rule
        if self.learning_rule == 'reinforce':
            learning_signal = params.alpha * reinforce(w, x, y, r)
        elif self.learning_rule == 'policy_gradient':
            learning_signal = jnp.multiply(params.alpha, policy_gradient(w, x, r)) # alpha * jnp.ones(w.shape[1])
        elif self.learning_rule == 'maximum_likelihood':
            learning_signal = params.alpha * maximum_likelihood(w, x)
        else:
            raise ValueError(f"Learning rule {self.learning_rule} not implemented.")

        # Add noise
        update_noise = jax.random.normal(subkey, shape=w.shape)
        if day_flag:
            update_noise = jnp.multiply(jnp.exp(params.log_sigma_day), update_noise)
        else:
            update_noise = jnp.multiply(jnp.exp(params.log_sigma), update_noise)
        
        if return_noise:
            return w + learning_signal + update_noise, update_noise
        else:
            return w + learning_signal + update_noise
    
    def initial_loglikelihood(self, z_0):
        """log p(z_0). We use p(z_0) = N(0,I)."""
        if z_0.shape == (1,):
            log_lik = lambda z: jsp.stats.norm.logpdf(z)#, loc=z_0_mean, scale=1.0)
        else:
            # N = w_init_mean.shape[0]
            log_lik = lambda z: jsp.stats.multivariate_normal.logpdf(
                z, mean=jnp.zeros_like(z), cov=jnp.diag(jnp.ones_like(z))
                )
        return log_lik(z_0)
    
    @handle_none_params
    def dynamics_loglikelihood(self, z_next, z_prev, inputs, data, 
                               params: Optional[ParamsGLMLearn]=None, day_flag=False, r=None):
        '''p(z_t | z_{t-1})
        In our case, the latents z are the GLM weights w
        '''
        if self.learning_rule == 'reinforce':
            learning_signal = params.alpha * reinforce(z_prev, inputs, data, r=r)
        else: # use Policy gradient
            learning_signal = params.alpha * policy_gradient(z_prev, inputs, r=r)
        mean = z_prev + learning_signal
        N = mean.shape[0]

        if day_flag:
            cov = jnp.multiply(jnp.square(jnp.exp(params.log_sigma_day)), jnp.eye(N))
        else:
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
        Y, Ws, noises = [], [], []
        for t in range(T):
            key, decision_key, update_key = jax.random.split(key, num=3)

            # Sample
            y = self.decision(w, X[t], key=decision_key)
            Y.append(y)
            Ws.append(w)

            # Update
            w, eps = self.update_weights(w, x=X[t], y=Y[t], key=update_key, return_noise=True)
            noises.append(eps)
        return X, jnp.stack(Y), jnp.stack(Ws), noises
    
    @handle_none_params
    def log_joint(
            self, 
            X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray, 
            params: Optional[ParamsGLMLearn]=None, R: Optional[jnp.ndarray]=None,
            session_indices: Optional[list]=[],
            ) -> float:
        '''
        Evaluate `log p(Y, Z | X, R, theta) = log p(y_{1:T}, z_{1:T} | x_{1:T}, r_{1:T}, theta)`. 

        parameters:
            X: array, stimulus, of shape (T, input_dim)
            Y: array, decisions, of shape (T, output_dim)
            Z: array, latent variables, of shape (T, latent_dim)
            params: ParamsGLMLearn, model parameters

        returns: 
            log_joint: float, value of log joint likelihood
        '''
        # Format arguments
        T = len(Y)
        if R is None:
            R = [None]*T
        if session_indices == []:
            day_flags = jnp.array([False for _ in range(T)], dtype=bool)
        else:
            # assert np.max(session_indices) <= T, "Session indices exceed length of data."
            # if np.max(session_indices) > T:
            #     logging.warning("Session indices exceed length of data.")
            day_flags = jnp.array([True if i in session_indices else False for i in range(T)], dtype=bool)
            day_flags = day_flags.at[0].set(False) #! do not use 0 session index as a new day for transitions

        # Initial t=0 joint likelihood terms
        # Dynamics
        log_pz0 = self.initial_loglikelihood(Z[0])

        # # Emissions
        # log_pyz0 = jnp.log(bernoulli_GLM_likelihood(Z[0], X[0], Y[0]))

        # log_joint_0 = log_pyz0 + log_pz0

        # # # Loop over time steps 
        # # logpzzdays = []
        # # log_pzz0 = 0.
        # #? vectorize with vmap?

        # day_Zs = []
        # prevday_Zs = []
        # for t in range(1,T):
        #     # Dynamics
        #     _log_pzz = self.dynamics_loglikelihood(
        #         Z[t], Z[t-1], X[t-1], Y[t-1], 
        #         params=params, day_flag=day_flags[t], r=R[t-1]
        #         )
        #     if day_flags[t]:
        #         day_Zs.append(Z[t])
        #         prevday_Zs.append(Z[t-1])
            
        #     # if day_flags[t]:
        #     #     logpzzdays.append(_log_pzz)

        #     # # Emissions
        #     log_pyz = jnp.log(bernoulli_GLM_likelihood(Z[t], X[t], Y[t]))

        #     # Update value and gradient of log joint
        #     log_joint_0 += log_pyz + _log_pzz

        # print(Z[1:].shape, Z[:-1].shape, X[:-1].shape, Y[:-1].shape, day_flags[:-1].shape, R[:-1].shape)
            
        # day_flags[0] = False
        prev_day_flags = jnp.roll(day_flags, shift=-1)
        # print(jnp.sum(day_flags))
        
        # print(Z.shape)
        # print('day_Zs', day_Zs)
        # print('Z[day_flags]', Z[day_flags], Z[day_flags])

        log_pzz_days = jax.vmap(
            lambda z1, z0, x0, y0, r0: self.dynamics_loglikelihood(
                z1, z0, x0, y0, params=params, day_flag=True, r=r0
                )
        )(
            Z[day_flags],
            Z[prev_day_flags], 
            X[prev_day_flags], 
            Y[prev_day_flags],
            R[prev_day_flags]
            )
        log_pzz_days = jnp.sum(log_pzz_days)

        log_pzz_trials = jax.vmap(
            lambda z1, z0, x0, y0, r0: self.dynamics_loglikelihood(
                z1, z0, x0, y0, params=params, day_flag=False, r=r0
                )
        )(
            Z[~day_flags][1:], 
            Z[~prev_day_flags][:-1], 
            X[~prev_day_flags][:-1], 
            Y[~prev_day_flags][:-1],
            R[~prev_day_flags][:-1]
            )
        log_pzz_trials = jnp.sum(log_pzz_trials)
        log_pz = log_pz0 + log_pzz_days + log_pzz_trials

            
        # def log_pzz_t(t):
        #     log_pzz = self.dynamics_loglikelihood(
        #         Z[t], Z[t-1], X[t-1], Y[t-1], 
        #         params=params, day_flag=day_flags[t], r=R[t-1]
        #         )
        
        # dynamics_loglikelihood_t = lambda t: \
        #     self.dynamics_loglikelihood(Z[t], Z[t-1], X[t-1], day_flag=t in session_indices)
        # log_pzz1 = jnp.sum(jax.vmap(dynamics_loglikelihood_t)(jnp.arange(T)))
        # print(log_pzz1, log_pzz0)

        # log_pzz = jnp.sum(log_pzz_days)

        # log_pzz_trials = jax.vmap(
        #     lambda z1, z0, x0, y0, r0: self.dynamics_loglikelihood(
        #         z1, z0, x0, y0, params=params, day_flag=False, r=r0
        #         )
        # )(
        #     Z[~day_flags], 
        #     Z[~prev_day_flags], 
        #     X[~prev_day_flags], 
        #     Y[~prev_day_flags],
        #     R[~prev_day_flags]
        #     )
        # log_pzz += jnp.sum(log_pzz_trials)

        log_pyz = jax.vmap(
            lambda z, x, y: jnp.log(bernoulli_GLM_likelihood(z, x, y)),
        )(Z, X, Y)
        log_pyz = jnp.sum(log_pyz)
        
        log_joint = log_pyz + log_pz

        # print(log_joint_0, log_joint_1)

        # print(log_joint)
        # import sys
        # sys.exit()

        return log_joint

if __name__=='__main__':
    # Seed for reproducibility
    seed = 0
    key = jax.random.PRNGKey(seed)

    # true_model = QLearningModel(sigma=0.3, alpha=0.5, softmax=False)
    # X, Y, m, Vs = true_model.sample(10)
    # print(X, Y, m, Vs)

    true_model = GLMLearn(log_sigma=-3.0, alpha=1.0, seed=seed, learning_rule='reinforce')
    X, Y, Ws = true_model.sample(1000)

    for log_sigma in np.linspace(-5,-1,10):
        print(log_sigma, true_model.log_joint(X, Y, Ws, params=ParamsGLMLearn(log_sigma, -1.0, 0.5)))
    # print(X, Y, Ws)