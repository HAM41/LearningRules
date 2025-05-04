import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.random import PRNGKey
import numpy as np
import scipy as sp
from typing import Tuple, Optional, Iterable, Union, NamedTuple
from jaxtyping import Array, Float, Bool
from functools import partial
from tqdm import tqdm



import tensorflow_probability.substrates.jax.distributions as tfd

import os
os.environ['JAX_PLATFORMS']='cpu'

import parameters
from parameters import ParamsGLMLearn, ParamsPsytrack, ParamsTimeVarGLMLearn #, handle_none_params

import logging
logging.basicConfig(level=logging.INFO, format='[%(filename)s][%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Y_L, Y_R = -1.0, 1.0 # Numerical value for the left, null, and right choices
Y_vals = jnp.array([Y_L, Y_R])

@jax.jit
def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

@jax.jit
def softplus(x):
    return jnp.log(1 + jnp.exp(x))

def tanh_transform(x, p=5):
    return jnp.tanh(p*x)/jnp.tanh(p)

def tanh_inv_transform(y, p=5):
    return jnp.arctanh(y*jnp.tanh(p))/p

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

def logmeanexp(x):
    '''
    Compute log(mean(exp(x)))
    '''
    return jsp.special.logsumexp(x) - jnp.log(x.size)

# if Y_L == -1.0:
@jax.jit
def sign(y: jnp.ndarray):
    y = jnp.asarray(y, dtype=float)
    return jnp.where(jnp.isnan(y), 0., y)
# else:
#     def sign(y: jnp.ndarray):
#         y = jnp.asarray(y, dtype=float)
#         out = 2*jnp.where(jnp.isnan(y), 0., y) - 1
#         return out

# def sign(y: jnp.ndarray):
#     '''
#     y: array-like, 
#     '''
#     out = jnp.asarray(y, dtype=float)
#     # return jnp.where(y == 0., -1., 1.)
#     # out = 2*jnp.where(jnp.isnan(y), Y_N, y) - 1
#     return out

def correct_choice(x: jnp.ndarray):
    '''
    Returns the side {Y_L,Y_R} of the stimulus. 
    Also corresponds to the correct choice.
    '''
    # x = jnp.asarray(x).astype(float)
    return jnp.where(x < 0, Y_L, Y_R)

def flip(y):
    '''
    Flips the choice y
    '''
    return jnp.where(y == Y_L, Y_R, Y_L)

def reward(X, Y, r1=1.0, r0=0.0) -> np.ndarray:
    '''
    Returns reward 
        r(x,y)= r1 if (x < 0 and y == 0) or (x > 0 and y==1)
                r0 else
    for (x,y) pairs in zip(X, Y)
    '''
    X = jnp.array(X).squeeze()
    Y = jnp.array(Y).squeeze().astype(float)
    
    # Broadcast scalars to arrays if needed
    if X.size == 1 and Y.size > 1:
        X = jnp.full(Y.shape, X)
    elif Y.size == 1 and X.size > 1:
        Y = jnp.full(X.shape, Y).astype(float)

    # Initialize an array for the rewards with the same shape as x and y
    r = r0 * jnp.ones_like(X, dtype=float)

    # Calculate rewards element-wise
    mask_condition = jnp.logical_or(
        jnp.logical_and(X < 0, Y == Y_L), 
        jnp.logical_and(X > 0, Y == Y_R)
        )
    # r = r.at[mask_condition].set(1.0)
    r = jnp.where(mask_condition, r1, r)

    return r

@partial(jax.vmap, in_axes=(0,))
def bias_correction(w: jnp.ndarray) -> jnp.ndarray:
    _correction = w[0] * jnp.array([-1, 1, 1, 0, 0])
    return w + _correction

# @partial(jax.jit, static_argnums=(0,))
def set_day_flags(T, session_indices):
    day_flags = jnp.zeros(T, dtype=bool)
    day_flags = day_flags.at[session_indices].set(True)
    day_flags = day_flags.at[0].set(False) # do not use 0 session index as a new day for transitions
    return day_flags

@jax.jit
def effective_reward(X, r1=1.0, r0=0.0):
    r'''
    Returns effective reward R(x) = \sum_y r(x,y) sign(y), for all x in X. 
    Output `RX` is of same shape as `X`.
    '''
    RX = jnp.sum(jnp.array([sign(y) * reward(X, y, r1=r1, r0=r0) for y in [Y_L, Y_R]]), axis=0)
    return RX

def cumulative_gaussian(x, sigma=1.0, mu=0.0):
    '''Cumulative distribution function for the Gaussian distribution.'''
    return 0.5*(1+jsp.special.erf((x-mu)/(sigma*jnp.sqrt(2))))

def Phi(x):
    '''Cumulative distribution function for the standard Gaussian.'''
    return cumulative_gaussian(x)

@jax.jit
def bernoulli_GLM_likelihood(w, x, y):
    '''
    Log-likelihood for a Bernoulli GLM
        p(y | x, w) = sigmoid(sign(y) * w^T x)
    
    WARNING: 
        Setting NaN y values to p=1, so that log p(y=NaN) = 0.
        This is to handle held-out or missing data, & 
        makes uncertainty-based (i.e. contains p or 1-p) learning rule updates equal 0.
    '''
    vx = vec(x)
    LM = jnp.dot(w, vx)
    p = jnp.where(jnp.isnan(y), 1.0, sigmoid(sign(y) * LM))
    return p

@jax.jit
def bernoulli_GLM_loglikelihood(w, x, y):
    '''
    Log-likelihood for a Bernoulli GLM
        log p(y | x, w) = log(sigmoid(sign(y) * w^T x))
    For missing y values (NaN), returns 0 (i.e. log 1).
    '''
    vx = vec(x)
    LM = jnp.dot(w, vx)
    # For non-NaN y, compute log(sigmoid(sign(y) * LM))
    # using the identity log(sigmoid(z)) = -softplus(-z)
    logp = jnp.where(jnp.isnan(y), 0.0, -softplus(-sign(y) * LM))
    return logp

@jax.jit
def policy_gradient(w, x, r=None):
    p_R = bernoulli_GLM_likelihood(w, x, y=Y_R)
    if r is None:
        r = effective_reward(x)
    if r.ndim == 1:
        r = r[:, None]
    return jnp.multiply(r, jnp.outer(jnp.multiply(p_R, 1 - p_R), vec(x)).squeeze())

@jax.jit
def reinforce(w, x, y, r):
    p = bernoulli_GLM_likelihood(w, x, y)
    return r * jnp.outer(1-p, sign(y) * vec(x)).squeeze()

@jax.jit
def maximum_likelihood(w, x):
    z = correct_choice(x)
    p = bernoulli_GLM_likelihood(w, x, z)
    return jnp.outer(1-p, sign(z) * vec(x)).squeeze()

def max_ent_MC(w, x, y, r):
    t1 = reinforce(w, x, y, r)
    p = bernoulli_GLM_likelihood(w, x, y)
    t2 = jnp.outer(jnp.multiply(p, 1 - p), sign(y) * vec(x))
    return t1 + t2

def max_ent(w, x, y, r):
    Er = effective_reward_2(r, y)
    t1 = policy_gradient(w, x, Er)

    p = bernoulli_GLM_likelihood(w, x, y)
    t2 = jnp.outer(jnp.multiply(jnp.multiply(p, 1 - p), 2*p - 1), sign(y) * vec(x))
    return t1 + t2

# @jax.jit
# def regression_gradient(w, x, y, r, params):
#     p_c = bernoulli_GLM_likelihood(w, x, y)
#     if r is None:
#         r = reward(x,y)
        
#     self_term = - w @ jnp.diag(params.Q) #jnp.exp(params.log_Q))
#     # self_term = self_term[:, None] if w.ndim == 2 else self_term
#     reward_stim_term = (r - params.baseline) * (jnp.diag(params.A) @ vec(x) + params.kappa)
#     reward_prob_term = (r - params.baseline) * jnp.outer(params.gamma * (1 - p_c) + params.beta * jnp.multiply(p_c, 1 - p_c), sign(y) * vec(x)).squeeze()
#     return self_term + reward_stim_term[None, :] + reward_prob_term

def effective_reward_2(r, y, r1=1.0, r0=0.0):
    '''
    Model effective reward from r_t=R(x_t, y_t) and y_t directly
    '''
    return (2*r - r1 - r0) * sign(y)


@jax.jit
def regression_gradient(w, x, y, r, params, r1=1.0, r0=0.0):
    # Poligy gradient term
    Er = effective_reward_2(r, y, r1=r1, r0=r0)
    PG_term = policy_gradient(w, x, r=Er)

    # REINFORCE term
    REINFORCE_term = reinforce(w, x, y, r=r-params.baseline)

    return params.gamma * REINFORCE_term + params.kappa * PG_term

def REINFORCE_with_baseline(w, x, y, r, baseline_weights):
    r = r - baseline_weights @ vec(x)
    return reinforce(w, x, y, r)

# @partial(jax.jit, static_argnums=(3,))
# def sample_particles_gumbel(key, tilde_z_t, log_w_t, N_particles):
#     # Generate a batch of keys for vectorized sampling
#     keys = jax.random.split(key, N_particles)
    
#     # Define a function to sample one index using the Gumbel-max trick
#     def sample_one(k):
#         # Generate Gumbel noise with the same shape as log_w_t.
#         gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(k, shape=(N_particles,))))
#         return jnp.argmax(log_w_t + gumbel_noise)
    
#     # Vectorize the sampling process across the batch of keys.
#     indices = jax.vmap(sample_one)(keys)
#     return tilde_z_t[indices]

@partial(jax.jit, static_argnums=(3,))
def sample_particles_gumbel(key, tilde_z_t, log_w_t, N_particles):
    # Generate an N_particles x N_particles matrix of independent uniform samples.
    U = jax.random.uniform(key, shape=(N_particles, N_particles))
    # Compute Gumbel noise for each entry.
    noise = -jnp.log(-jnp.log(U))
    # For each row, add the log weights and take the argmax.
    indices = jnp.argmax(log_w_t + noise, axis=-1)
    return tilde_z_t[indices]

# def gumbel_softmax_sample(rng, p, temperature=1.0, eps=1e-10):
#     """
#     Draw a relaxed one-hot sample from categorical with probs p via Gumbel-Softmax.
    
#     Args:
#         rng: JAX PRNG key
#         p: 1D array of shape (K, ) with categorical probabilities (sum to 1)
#         temperature: scalar temperature
#         eps: small constant for numerical stability
        
#     Returns:
#         A 'soft' one-hot vector of shape (K, ), i.e. sums to 1 but with continuous entries.
#     """
#     # Sample Gumbel noise
#     g = jax.random.gumbel(rng, shape=p.shape)
#     # Compute 'logits' = log(p) + Gumbel noise
#     logits = jnp.log(p + eps) + g
#     # Softmax
#     y = jax.nn.softmax(logits / temperature)
#     return y


# def gumbel_softmax_sample_straight_through(rng, p, temperature=1.0):
#     y_soft = gumbel_softmax_sample(rng, p, temperature)
#     # Forward pass: pick discrete argmax
#     idx = jnp.argmax(y_soft)
#     y_hard = jnp.zeros_like(y_soft).at[idx].set(1.0)
    
#     # Straight-through gradient: 
#     # forward uses y_hard, backward uses y_soft
#     return y_hard + jax.lax.stop_gradient(y_soft - y_hard)


# def multinomial_gumbel_softmax(rng, p, n_draws, temperature=1.0, straight_through=False):
#     """
#     Returns shape (n_draws, K) array of relaxed (or hard-straight-through) one-hot vectors.
#     """
#     subkeys = jax.random.split(rng, n_draws)
    
#     if straight_through:
#         sampler = lambda k: gumbel_softmax_sample_straight_through(k, p, temperature)
#     else:
#         sampler = lambda k: gumbel_softmax_sample(k, p, temperature)
    
#     # vmap over multiple draws
#     return jax.vmap(sampler)(subkeys)

# import jax
# import jax.numpy as jnp

def sample_one_particle_gumbel(
    rng,
    p: jnp.ndarray,            # shape (N,)
    tilde_z_t: jnp.ndarray,    # shape (N, M)
    temperature: float = 1.0,
    straight_through: bool = True,
    eps: float = 1e-10
) -> jnp.ndarray:
    """
    Draw one Gumbel-Softmax sample (relaxed or ST) from categorical 'p'
    and map it onto a single new particle via a dot product with tilde_z_t.

    Returns:
      z_new: shape (M,)
    """
    # 1) sample Gumbel noise of same shape as p
    g = jax.random.gumbel(rng, shape=p.shape)
    logits = jnp.log(p + eps) + g

    # 2) continuous "soft" one-hot
    y_soft = jax.nn.softmax(logits / temperature)

    if straight_through:
        # Hard selection in forward pass
        idx = jnp.argmax(y_soft)
        y_hard = jnp.zeros_like(y_soft).at[idx].set(1.0)
        # Straight-through gradient: forward uses y_hard, backward uses y_soft
        y = y_hard + jax.lax.stop_gradient(y_soft - y_hard)
    else:
        # purely soft
        y = y_soft

    # 3) map one-hot to the actual (M,) state by dot product
    #    shape: (N,) dot (N, M) => (M,)
    z_new = y @ tilde_z_t
    return z_new


def resample_particles_gumbel_vmap(
    rng,
    tilde_z_t: jnp.ndarray,    # shape (N, M)
    log_tilde_w_t: jnp.ndarray,# shape (N,)
    temperature: float = 0.5,
    straight_through: bool = True
) -> jnp.ndarray:
    """
    Resamples N new particles via Gumbel-Softmax, returning shape (N, M).
    Does so without building an (N x N) matrix.
    """
    # Convert log-weights to normalized probabilities
    max_log_w = jnp.max(log_tilde_w_t)
    logw_stable = log_tilde_w_t - max_log_w
    w = jnp.exp(logw_stable)
    p = w / jnp.sum(w)  # shape (N,)

    N = tilde_z_t.shape[0]
    subkeys = jax.random.split(rng, N)

    # We'll vmap over 'N' draws:
    def single_draw(subkey):
        return sample_one_particle_gumbel(
            subkey,
            p=p,
            tilde_z_t=tilde_z_t,
            temperature=temperature,
            straight_through=straight_through
        )

    # shape: (N, M)
    z_new = jax.vmap(single_draw)(subkeys)
    return z_new

from functools import partial
import jax.lax as lax

def resample_particles_gumbel_scan(
    rng,
    tilde_z_t: jnp.ndarray,
    log_tilde_w_t: jnp.ndarray,
    temperature: float = 0.5,
    straight_through: bool = True
) -> jnp.ndarray:
    # same setup to get p
    max_log_w = jnp.max(log_tilde_w_t)
    logw_stable = log_tilde_w_t - max_log_w
    w = jnp.exp(logw_stable)
    p = w / jnp.sum(w)
    
    N = tilde_z_t.shape[0]
    subkeys = jax.random.split(rng, N)

    def scan_body(carry, subkey):
        # carry is not used except to hold the partial results
        z_new = sample_one_particle_gumbel(
            subkey,
            p=p,
            tilde_z_t=tilde_z_t,
            temperature=temperature,
            straight_through=straight_through
        )
        return carry, z_new

    init_carry = None
    # out shape: (N, M)
    _, z_new_all = lax.scan(scan_body, init_carry, subkeys)
    return z_new_all



class LearningModel():
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
    
    def emission_likelihood(self, z, x, y, params):
        '''p(y_t | x_t, z_t, theta)'''
        raise NotImplementedError
    
    def inital_z_likelihood(self, z, x, params):
        raise NotImplementedError
    
    def decision(self, key: PRNGKey, z, x, params):
        p_R = self.emission_likelihood(z, x, y=Y_R, params=params)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y

class QLearning():
    def __init__(self, softmax: bool=True) -> None:
        # self.alpha = alpha
        # self.sigma = sigma
        # self.beta = beta        # "Inverse temperature" parameter for the softmax decision.

        # self.V_init = 0.2       # Initialize values at 0.2

        # Use softmax for stochastic decisions. Defaults to True.
        self.softmax = softmax
    
    def __repr__(self) -> str:
        return f"QLearning()"

    def split_latent(self, z):
        # Split latent variable into beta and weights, over last axis
        if z.ndim == 1:
            m, V_L, V_R = jnp.array([z[0]]), jnp.array([z[1]]), jnp.array([z[2]])
        else:
            m, V_L, V_R = z[..., 0], z[..., 1], z[..., 2]
        return m, V_L, V_R
    
    def merge_latent(self, m, V_L, V_R):
        return jnp.vstack([m, V_L, V_R]).T

    def p_R(self, m, params):
        '''p_R(m): Belief state p(s > 0 | m)'''
        return cumulative_gaussian(m, sigma=jnp.exp(params.percept_log_scale))
    
    def encode(self, key, X, params, N=1):
        noise = jax.random.normal(key, shape=X.shape + (N,))
        percept = X + jnp.exp(params.percept_log_scale) * noise
        return percept.squeeze()
    
    def transition_point(self, V, params):
        V_L, V_R = V
        return jnp.exp(params.percept_log_scale) * jsp.special.erfinv(np.divide(V_L - V_R, V_R + V_L))

    def decision(self, key: PRNGKey, z, params):
        p_R = self.emission_likelihood(z, y=Y_R, params=params)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return Y_vals[y]

    def update_weights(
            self, 
            key: PRNGKey,
            w, x, x_next, 
            params, y, r=None,
            day_flag: bool=False, # unused
            return_learning_signal=False # unused
            ):
        m, V_L, V_R = self.split_latent(w)

        p_R = self.p_R(m, params)
        key_m, key_R, key_L = jax.random.split(key, 3)
        
        def update_value(key, V, condition, p):
            RPE = reward(x, y) - p * V
            update = jnp.exp(params.log_alpha) * RPE

            update_noise = jax.random.normal(key, shape=V.shape)
            update_noise = jnp.where(day_flag, 
                                 jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
                                 jnp.multiply(jnp.exp(params.log_sigma), update_noise)
                                 )

            # Compute the update if condition is true; else add zero.
            return V + jnp.where(condition, update, 0.0) + update_noise

        # Update V_R when y equals Y_R; update V_L when y equals Y_L.
        V_R = update_value(key_R, V_R, y == Y_R, p_R)
        V_L = update_value(key_L, V_L, y == Y_L, 1 - p_R)

        # Update m
        m = self.encode(key_m, x_next, params, N=len(m) if m.ndim > 0 else 1)
        
        w = self.merge_latent(m, V_L, V_R)
        return w

    def emission_likelihood(self, z, y, params):
        r'''p(y | x_hat, V)'''
        m, V_L, V_R = self.split_latent(z)
        p_R = self.p_R(m, params)

        # Compute Q values for each state
        Qs = jnp.stack([jnp.multiply(1-p_R, V_L), jnp.multiply(p_R, V_R)])
        Qs = Qs / jnp.exp(params.log_temp) # temperature

        # Compute decision likelihood p(y|Qs)
        if self.softmax:
            Ps = jax.nn.softmax(Qs, axis=0)
            p = jnp.where(y==Y_R, Ps[1], Ps[0])
        else:
            Y_vals = jnp.array([Y_L, Y_R])
            choices = Y_vals[jnp.argmax(Qs, axis=0)]
            p = jnp.equal(y, choices).astype(float)
        return p
    
    def emission_loglikelihood(self, z, y, params):
        '''Log-likelihood for the emission model p(y | x_hat, V). TODO: leverage softmax for numerical stability.'''
        return jnp.log(self.emission_likelihood(z, y, params))
        
    def sample_initial(self, key, X, params, N=1):
        '''Sample from the initial distribution p(z_0).
        Args:
            N: int, number of samples
        '''
        # m = jax.random.normal(key, shape=(1,N,))
        m_key, V_key = jax.random.split(key)
        # m = jax.vmap(lambda _key: self.encode(_key, X, params))(jax.random.split(m_key, num=N))
        m = self.encode(m_key, X, params, N)
        V = 0.2 + 0.1 * jax.random.normal(V_key, shape=(2, N))
        # V = 0.2 * jnp.ones((2, N))
        return self.merge_latent(m, V[0], V[1])
        
    def marginal_log_likelihood(self, key, params, X, Y, R=None, day_flags=None,
                                N_particles=1000, verbose=False, return_logliks = False):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
        else:
            T, _ = X.shape

        def scan_fn(carry, inputs):
            tilde_z_t, marginal_log_lik = carry
            X_t, X_t_next, Y_t, R_t, next_day_flag, (subkey1, subkey2) = inputs

            # 2. Evaluate importance weights p(y_t | tilde z_t)
            #   Outcome: {tilde z_t^i, tilde w^i}, an approximation to p(z_t|y_{1:t})
            # tilde_w_t = self.emission_likelihood(tilde_z_t, Y_t, params=params)
            tilde_log_w_t = self.emission_loglikelihood(tilde_z_t, Y_t, params=params)
            p = jnp.exp(tilde_log_w_t - jnp.max(tilde_log_w_t))

            # 3. Update step: resample with replacement N particles according the importance weights
            #   Outcome: {z_t, 1/N}, an approximation to p(z_t|y_{1:t})
            z_t = jax.random.choice(subkey1, tilde_z_t, shape=(N_particles,), p=p)
            # z_t = sample_particles_gumbel(subkey1, tilde_z_t, tilde_log_w_t, N_particles)

            # 4. Update log-likelihood estimate
            log_lik = logmeanexp(tilde_log_w_t)
            # log_lik = jnp.log(jnp.mean(tilde_w_t))
            marginal_log_lik += log_lik
            
            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            #   Sample proposal N particles from previous N particles
            #   Outcome: {tilde z_t^i, 1/N}, an approximation to p(z_t|y_{1:t-1})
            tilde_z_next = self.update_weights(subkey2, z_t, params=params, x=X_t, x_next=X_t_next, y=Y_t, r=R_t, day_flag=next_day_flag)
            
            return (tilde_z_next, marginal_log_lik), log_lik
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, X[0], params=params, N=N_particles)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        X_next = jnp.roll(X, shift=-1)
        subkeys = jax.random.split(key, num=(T, 2))
        inputs = (X, X_next, Y, R, next_day_flags, subkeys)
        
        (_, marginal_log_lik), log_liks = jax.lax.scan(scan_fn, carry, inputs, length=T)
        # marginal_log_lik = 0.
        # for t in range(T):
        #     carry, log_lik = scan_fn(carry, (inp[t] for inp in inputs))
        #     marginal_log_lik += log_lik

        if return_logliks:
            return marginal_log_lik, log_liks
        else:
            return marginal_log_lik

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
    
    def sample_forward(self, key, params, X: Float[Array, "T M"], day_flags: Bool[Array, "T"], z_0=None):
        '''
        Forward pass through the model from initial state, sampling decisions.
        '''
        assert self.reward_func is not None, "Reward function, (x_t, y_t) -> r_t, must be defined."
        if z_0 is None:
            z_0 = self.sample_initial(key, X=X[0], params=params).squeeze()
       
        T = len(X)
        latent_dim = z_0.shape[0]

        # Forward pass
        def scan_fn(carry, inputs):
            y_t, z_t = carry

            # Process inputs
            subkey, X_t, X_next, day_flag = inputs
            r_t = self.reward_func(X_t, y_t)  

            # Update state
            z_pred = self.update_weights(subkey, z_t, X_t, X_next, params, y=y_t, day_flag=day_flag).squeeze()
            y_pred = self.decision(subkey, z_pred, params)

            return (y_pred, z_pred), (y_pred, z_pred)
        
        # Initialize
        subkeys = jax.random.split(key, T)
        y_0 = self.decision(subkeys[0], z_0, params)
        init = (y_0, z_0)
        inputs = (subkeys[1:], X[:-1], X[1:], day_flags[:-1])

        # Forward pass
        _, (Y_pred, Z_pred) = jax.lax.scan(scan_fn, init, inputs, length=T-1)
        Y = jnp.concatenate([jnp.array([y_0]), Y_pred]).squeeze(); assert Y.shape == (T,), f"Y shape {Y.shape} does not match expected shape ({T},)"
        Z = jnp.concatenate([jnp.array([z_0]), Z_pred]); assert Z.shape == (T, latent_dim), f"Z shape {Z.shape} does not match expected shape ({T}, {latent_dim})"
        return Y, Z
    
    def posterior_samples(self, 
            key, 
            params, 
            X, Y, R=None, day_flags=None,
            N_particles=1000, return_history=True, posterior_type='smooth', verbose=False,
            LAG=False, correct_bias=False
            ):
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)
        self.latent_dim = 3
        if LAG: L = 50

        if return_history:
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
                tilde_z_t = self.sample_initial(subkey1, X[t], params=params, N=N_particles)
            else:
                tilde_z_t = self.update_weights(subkey1, z_t, params=params, x=X[t-1], x_next=X[t], y=Y[t-1], r=None, day_flag=day_flags[t])

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            #   Outcome: {tilde z_t^i, tilde w^i}, an approximation to p(z_t|y_{1:t})
            # tilde_w_t = bernoulli_GLM_likelihood(y=Y[t], w=tilde_z_t, x=X[t]) 
            tilde_w_t = self.emission_likelihood(tilde_z_t, Y[t], params=params)
            
            # 3. Resample with replacement N particles according the importance weights
            #   Outcome: {z_t, 1/N}, an approximation to p(z_t|y_{1:t})
            if return_history:
                if LAG:
                    z_history = z_history.at[:,t,:].set(tilde_z_t)
                    z_history_window = z_history[:,max(0,t-L):t+1,:]
                    z_history_window = jax.random.choice(subkey2, z_history_window, shape=(N_particles,), p=tilde_w_t)
                    z_history = z_history.at[:,max(0,t-L):t+1,:].set(z_history_window)
                    z_t = z_history[:,t,:]
                else:
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
    
    def filter(self, 
            key, params, 
            X: Float[Array, "T M"], Y: Float[Array, "T"], R: Float[Array, "T"], day_flags: Bool[Array, "T"], 
            N_particles=1000
            ):
        '''
        return p(w_{1:T} | y_{1:T}, x_{1:T}) under the prior, using the true data.
        '''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape

        def scan_fn(carry, inputs):
            z_t, log_lik = carry
            X_t, X_next, Y_t, next_day_flag, subkey = inputs

            # Log-lik
            lik = self.emission_likelihood(z_t, Y_t, params=params) 
            log_lik += jnp.log(jnp.mean(lik))

            # Update z_t ~ p(z_t | z_{t-1})
            z_next = self.update_weights(subkey, z_t, params=params, x=X_t, x_next=X_next, y=Y_t, r=None, day_flag=next_day_flag)
            
            return (z_next, log_lik), z_next
        
        key, subkey = jax.random.split(key)
        z_0 = self.sample_initial(subkey, params=params, X=X[0], N=N_particles)
        carry = (z_0, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=T)
        X_next = jnp.roll(X, shift=-1)
        inputs = (X[:-1], X[1:], Y[:-1], next_day_flags[:-1], subkeys[:-1])
        
        (_, log_lik), Z = jax.lax.scan(scan_fn, carry, inputs)

        # Replace Z[0] and shift other Zs
        Z = jnp.concatenate([jnp.array([z_0]), Z[:-1]])

        return Z, log_lik
    
    def filtering_MLL(self, 
            key, params, 
            X: Float[Array, "T M"], Y: Float[Array, "T"], R: Float[Array, "T"], day_flags: Bool[Array, "T"], 
            N_particles=1000
            ):
        '''
        return p(w_{1:T} | y_{1:T}, x_{1:T}) under the prior, using the true data.
        '''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape

        def scan_fn(carry, inputs):
            z_t, log_lik = carry
            X_t, X_next, Y_t, next_day_flag, subkey = inputs

            # Log-lik
            lik = self.emission_likelihood(z_t, Y_t, params=params) 
            log_lik += jnp.log(jnp.mean(lik))

            # Update z_t ~ p(z_t | z_{t-1})
            z_next = self.update_weights(subkey, z_t, params=params, x=X_t, x_next=X_next, y=Y_t, r=None, day_flag=next_day_flag)
            
            return (z_next, log_lik), None
        
        key, subkey = jax.random.split(key)
        z_0 = self.sample_initial(subkey, params=params, X=X[0], N=N_particles)
        carry = (z_0, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=T)
        inputs = (X[:-1], X[1:], Y[:-1], next_day_flags[:-1], subkeys[:-1])
        
        (_, log_lik), _ = jax.lax.scan(scan_fn, carry, inputs)

        # # Replace Z[0] and shift other Zs
        # Z = jnp.concatenate([jnp.array([z_0]), Z[:-1]])

        return log_lik

    def forward_pass(self, 
            key, params, 
            X: Float[Array, "T M"], Y: Float[Array, "T"], R: Float[Array, "T"], day_flags: Bool[Array, "T"], 
            N_particles: int=1000, predict_Y: bool=False, correct_bias=None, return_Z: bool=False,
            ):
        '''
        Make a forward pass in the model, either using the animal decisions (predict_Y=False) or sampling decisions
        (predict_Y=True). Predicition time-step to time-step likelihoods are returned.
        '''
        T = len(X)

        if predict_Y:
            assert self.reward_func is not None, "Reward function, (x_t, y_t) -> r_t, must be defined."

            def scan_fn(carry, inputs):
                z_t, log_lik = carry
                X_t, X_next, Y_t, next_day_flag, subkey = inputs

                # Sample Y
                decision_key, update_key = jax.random.split(subkey)
                Y_pred = self.decision(decision_key, z_t, params=params)
                # R_pred = jax.vmap(lambda y: reward(X_t, y))(Y_pred)

                # Log-lik
                # w_true = self.emission_likelihood(z_t, Y_t, params=params)
                log_w_true = self.emission_loglikelihood(z_t, Y_t, params=params)
                # w_pred = self.emission_likelihood(z_t, Y_pred, params=params)
                log_w_pred = self.emission_loglikelihood(z_t, Y_pred, params=params)

                log_lik_t = logmeanexp(log_w_true)
                log_lik += log_lik_t

                # # Update step
                # p = jnp.exp(log_w_pred - jnp.max(log_w_pred))
                # z_t = jax.random.choice(update_key, z_t, shape=(N_particles,), p=p)
                # # z_t = sample_particles_gumbel(update_key, z_t, log_w_pred, N_particles)

                # Update z_t ~ p(z_t | z_{t-1})
                z_next = jax.vmap(
                    lambda z, y: self.update_weights(update_key, z, params=params, x=X_t, x_next=X_next, y=y, r=None, day_flag=next_day_flag)
                    )(z_t, Y_pred).squeeze()
                
                if return_Z:
                    return (z_next, log_lik), (log_lik_t, z_next)
                else:
                    return (z_next, log_lik), log_lik_t
        else:
            def scan_fn(carry, inputs):
                z_t, log_lik = carry
                X_t, X_next, Y_t, next_day_flag, subkey = inputs

                # Log-lik
                # w_t = self.emission_likelihood(z_t, Y_t, params=params)
                log_w_t = self.emission_loglikelihood(z_t, Y_t, params=params)

                log_lik_t = logmeanexp(log_w_t)
                log_lik += log_lik_t

                # Update step
                p = jnp.exp(log_w_t - jnp.max(log_w_t))
                z_t = jax.random.choice(subkey, z_t, shape=(N_particles,), p=p)
                # z_t = sample_particles_gumbel(subkey, z_t, log_w_t, N_particles)

                # Update z_t ~ p(z_t | z_{t-1})
                z_next = self.update_weights(subkey, z_t, params=params, x=X_t, x_next=X_next, y=Y_t, r=None, day_flag=next_day_flag).squeeze()
                
                if return_Z:
                    return (z_next, log_lik), (log_lik_t, z_next)
                else:
                    return (z_next, log_lik), log_lik_t
        
        key, subkey = jax.random.split(key)
        z_0 = self.sample_initial(subkey, params=params, X=X[0], N=N_particles)
        carry = (z_0, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=T)
        inputs = (X[:-1], X[1:], Y[:-1], next_day_flags[:-1], subkeys[:-1])
        
        if return_Z:
            (_, log_lik), (logliks, Z) = jax.lax.scan(scan_fn, carry, inputs) #, length=T)

            # Replace Z[0] and shift other Zs
            Z = jnp.concatenate([jnp.array([z_0]), Z[:-1]])
            Z = Z.transpose(1,0,2)
        else:
            (_, log_lik), logliks = jax.lax.scan(scan_fn, carry, inputs)
            Z = None

        return (logliks, Z), log_lik
    
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
                 z_0: float = 0.0,
                 learning_rule: str='policy_gradient', # seed: int=0,
                 latent_dim = None,
                 ) -> None:
        self.learning_rule = learning_rule.lower()
        self.reward_func = None

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

    def __repr__(self) -> str:
        return f"GLMLearn({self.learning_rule})"
        
    def decision(self, key: PRNGKey, w, x):
        p_R = self.emission_likelihood(w, x, y=Y_R)
        y = Y_vals[jax.random.bernoulli(key, p_R).astype(int)]
        return y
    
    # @handle_none_params
    def update_weights(
            self, 
            key: PRNGKey,
            w, x, 
            params, y=None, r=None,
            day_flag: bool=False,
            return_learning_signal=False, 
            return_mean=False,
            correct_bias=False,
            ):
        
        # Change in mean weights from learning rule
        if self.learning_rule == 'reinforce':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), reinforce(w, x, y, r))
        elif self.learning_rule == 'policy_gradient':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), policy_gradient(w, x, r)) # alpha * jnp.ones(w.shape[1])
        elif self.learning_rule == 'maximum_likelihood':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), maximum_likelihood(w, x))
        elif self.learning_rule == 'max_ent':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), max_ent(w, x, y, r))
        elif self.learning_rule == 'max_ent_mc':
            learning_signal = jnp.multiply(jnp.exp(params.log_alpha), max_ent_MC(w, x, y, r))
        else:
            raise ValueError(f"Learning rule {self.learning_rule} not implemented.")

        # Add noise
        update_noise = jax.random.normal(key, shape=w.shape)
        # update_noise = jnp.where(day_flag, 
        #                          jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
        #                          jnp.multiply(jnp.exp(params.log_sigma), update_noise)
        #                          )
        update_noise = jax.lax.select(day_flag,
            jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
            jnp.multiply(jnp.exp(params.log_sigma), update_noise)
        )

        # if correct_bias:
        #     learning_signal = learning_signal.at[..., 0].set(0.0)
        #     update_noise = update_noise.at[..., 0].set(0.0)
        learning_signal = jnp.where(correct_bias, learning_signal.at[..., 0].set(0.0), learning_signal)
        update_noise = jnp.where(correct_bias, update_noise.at[..., 0].set(0.0), update_noise)
        
        if return_learning_signal:
            return w + learning_signal + update_noise, learning_signal
        elif return_mean:
            return w + learning_signal
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
        '''Sample from the initial distribution p(z_0).
        Args:
            N: int, number of samples
            d: int, number of regressors. Weights are of shape (d+1,), for the bias. 
        '''
        return jax.random.normal(key, shape=(N, d+1,))
    
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

        cov = jnp.where(day_flag, 
            jnp.multiply(jnp.square(jnp.exp(params.log_sigma_day)), jnp.eye(N)),
            jnp.multiply(jnp.square(jnp.exp(params.log_sigma)), jnp.eye(N))
            )
        
        log_lik = lambda z: jsp.stats.multivariate_normal.logpdf(z, mean=mean, cov=cov)
        return log_lik(z_next)
    
    def emission_likelihood(self, z, x, y, params=None):
        '''p(y | z, x)'''
        # _x = x.copy()
        # _x = _x.at[..., 0].set(tanh_inv_transform(_x[..., 0]))
        # _x = _x.at[..., 1].set(tanh_inv_transform(_x[..., 1]))

        # _x = _x.at[..., 0].set(tanh_transform(_x[..., 0], params.p))
        # _x = _x.at[..., 1].set(tanh_transform(_x[..., 1], params.p))
        return bernoulli_GLM_likelihood(z, x, y)

    def emission_loglikelihood(self, z, x, y, params=None):
        return bernoulli_GLM_loglikelihood(z, x, y)
    
    # @handle_none_params
    def log_joint(
            self, 
            X: jnp.ndarray, Y: jnp.ndarray, Z: jnp.ndarray, 
            params: ParamsGLMLearn, R: Optional[jnp.ndarray]=None,
            day_flags: Optional[jnp.ndarray]=None,
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
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)

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
            X, Y, R=None, day_flags=None,
            N_particles=1000, return_history=True, posterior_type='smooth', verbose=False,
            LAG=False, correct_bias=True,
            ):
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)
        if self.latent_dim is None:
            self.latent_dim = M + 1

        if LAG: L = 50

        if return_history:
            # Block out N x T x M+1 array (float32, x 4 in bytes) to store z_t samples.
            # A lot of memory, but much faster. 
            # if LAG: L = 50
            #     z_history = jnp.zeros((N_particles, L, self.latent_dim), dtype=jnp.float32) # float16
            # else:
            z_history = jnp.zeros((N_particles, T, self.latent_dim), dtype=jnp.float32) # float16

        if verbose:
            pbar = tqdm(range(0,T), desc='Bootstrap filter')

        if correct_bias and M == 4:
            correct_bias_flags = jnp.cumprod(jnp.abs(X[:,1] - X[:,0]) >= 0.9).astype(bool) # True until stim intensity goes below 0.9 in abs
        else:
            correct_bias_flags = jnp.zeros(T, dtype=bool)
    
        log_lik = 0.
        for t in range(0,T):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            #   Sample proposal N particles from previous N particles
            #   Outcome: {tilde z_t^i, 1/N}, an approximation to p(z_t|y_{1:t-1})
            if t == 0:
                tilde_z_t = self.sample_initial(subkey1, params=params, N=N_particles, d=M)
                if correct_bias_flags[t]:
                    tilde_z_t = self.bias_correction(tilde_z_t)
            else:
                tilde_z_t = self.update_weights(subkey1, z_t, params=params, x=X[t-1], y=Y[t-1], r=R[t-1], 
                                                day_flag=day_flags[t], correct_bias=correct_bias_flags[t])
            
            # Correct the non-identifiability while we have the non_one_X_flag
            # if correct_bias_flags[t]:
            #     tilde_z_t = self.bias_correction(tilde_z_t)

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            #   Outcome: {tilde z_t^i, tilde w^i}, an approximation to p(z_t|y_{1:t})
            # tilde_w_t = bernoulli_GLM_likelihood(y=Y[t], w=tilde_z_t, x=X[t]) 
            # tilde_w_t = self.emission_likelihood(tilde_z_t, X[t], Y[t])
            log_tilde_w_t = self.emission_loglikelihood(tilde_z_t, X[t], Y[t])
            
            # 3. Resample with replacement N particles according the importance weights
            #   Outcome: {z_t, 1/N}, an approximation to p(z_t|y_{1:t})
            p = jnp.exp(log_tilde_w_t - jnp.max(log_tilde_w_t))
            if return_history:
                if LAG:
                    z_history = z_history.at[:,t,:].set(tilde_z_t)
                    z_history_window = z_history[:,max(0,t-L):t+1,:]
                    z_history_window = jax.random.choice(subkey2, z_history_window, shape=(N_particles,), p=p)
                    z_history = z_history.at[:,max(0,t-L):t+1,:].set(z_history_window)
                    z_t = z_history[:,t,:]
                else:
                    if posterior_type == 'smooth':
                        z_history = z_history.at[:,t,:].set(tilde_z_t)
                        z_history = jax.random.choice(subkey2, z_history, shape=(N_particles,), p=p)
                        z_t = z_history[:,t,:]

                    elif posterior_type == 'filt':
                        z_t = jax.random.choice(subkey2, tilde_z_t, shape=(N_particles,), p=p)
                        z_history = z_history.at[:,t,:].set(z_t)
            else:
                z_t = jax.random.choice(subkey2, tilde_z_t, shape=(N_particles,), p=p)

            # 4. Update log-likelihood estimate
            # log_lik += jnp.log(jnp.mean(tilde_w_t))
            log_lik += logmeanexp(log_tilde_w_t)

            if verbose:
                pbar.update(1)



        if not return_history:
            z_history = z_t

        return z_history, log_lik

    def marginal_log_likelihood(self, key, params, X, Y, R=None, day_flags=None,
                                N_particles=1000, verbose=False, return_logliks = False):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)

        def scan_fn(carry, inputs):
            tilde_z_t, marginal_log_lik = carry
            X_t, Y_t, R_t, next_day_flag, (subkey1, subkey2) = inputs

            # 1. Evaluate importance weights p(y_t | xhat_t, V_t)
            # tilde_w_t = self.emission_likelihood(tilde_z_t, X_t, Y_t, params=params)
            log_tilde_w_t = self.emission_loglikelihood(tilde_z_t, X_t, Y_t, params=params)

            # -- Normalize and handle numerical stability
            #   jax.random.choice(replace=True) is invariant to scaling of p of the form a * p
            p = jnp.exp(log_tilde_w_t - jnp.max(log_tilde_w_t))
            
            # 2. Update step : resample with replacement N particles according the importance weights
            z_t = jax.random.choice(subkey1, tilde_z_t, shape=(N_particles,), p=p)
            # z_t = resample_particles_gumbel_scan(
            #     rng=subkey1,
            #     tilde_z_t=tilde_z_t,
            #     log_tilde_w_t=log_tilde_w_t,
            #     temperature=0.5,
            #     straight_through=True
            # )

            # -- Update log-likelihood estimate
            log_lik = logmeanexp(log_tilde_w_t)
            marginal_log_lik += log_lik
            
            # 3. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            tilde_z_t = self.update_weights(subkey2, z_t, params=params, x=X_t, y=Y_t, r=R_t, day_flag=next_day_flag)

            # jax.debug.print("log tilde w_t = {}, p = {}, log_lik = {}", log_tilde_w_t, p, log_lik)
            
            return (tilde_z_t, marginal_log_lik), log_lik
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=(T, 2))
        inputs = (X, Y, R, next_day_flags, subkeys)

        # Scan
        (_, marginal_log_lik), log_liks = jax.lax.scan(scan_fn, carry, inputs, length=T)

        # # For loop
        # marginal_log_lik, log_liks = 0., []
        # for t in range(T):
        #     carry, log_lik = scan_fn(carry, (inp[t] for inp in inputs))
        #     marginal_log_lik += log_lik
        #     log_liks.append(log_lik)
        # log_liks = jnp.array(log_liks)

        if return_logliks:
            return marginal_log_lik, log_liks
        else:
            return marginal_log_lik
    
    def posterior_samples_scan(self, key, params, X, Y, R=None, day_flags=None,
                                N_particles=1000, verbose=False, correct_bias=True):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)

        def scan_fn(carry, inputs):
            t, tilde_z_t, log_lik, z_history = carry
            X_t, Y_t, R_t, next_day_flag, correct_bias_flag, (subkey1, subkey2) = inputs

            # Correct tilde_z_t for nonidentifability
            # if correct_bias_flag:
            #     tilde_z_t = bias_correction(tilde_z_t)
            # jax.debug.print('correct_bias_flag = {}', correct_bias_flag)
            # tilde_z_t = jax.lax.select(correct_bias_flag, self.bias_correction(tilde_z_t), tilde_z_t)

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            # tilde_w_t = self.emission_likelihood(tilde_z_t, X_t, Y_t)
            log_tilde_w_t = self.emission_loglikelihood(tilde_z_t, X_t, Y_t, params=params)
            
            # 3. Resample with replacement N particles according the importance weights
            # z_t = jax.random.choice(subkey1, tilde_z_t, shape=(N_particles,), p=tilde_w_t)
            p = jnp.exp(log_tilde_w_t - jnp.max(log_tilde_w_t))
            z_history = z_history.at[:,t,:].set(tilde_z_t)
            z_history = jax.random.choice(subkey1, z_history, shape=(N_particles,), p=p)
            z_t = z_history[:,t,:]

            # 4. Update log-likelihood estimate
            # log_lik += jnp.log(jnp.mean(tilde_w_t))
            log_lik += logmeanexp(log_tilde_w_t)
            
            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            tilde_z_t = self.update_weights(subkey2, z_t, params=params, x=X_t, y=Y_t, r=R_t, day_flag=next_day_flag,
                                            correct_bias=correct_bias_flag)
            
            return (t+1, tilde_z_t, log_lik, z_history), None
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        assert self.latent_dim is not None, "Latent dimension not defined."
        z_history = jnp.zeros((N_particles, T, self.latent_dim), dtype=jnp.float32) # float16
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=(T, 2))

        if correct_bias:
            correct_bias_flags = jnp.cumprod(jnp.abs(X[:,1] - X[:,0]) >= 0.9).astype(bool) # True until stim intensity goes below 0.9 in abs
        else:
            correct_bias_flags = jnp.zeros(T, dtype=bool)
        
        if correct_bias_flags[0]:
            tilde_z_t = self.bias_correction(tilde_z_t)

        carry = (0, tilde_z_t, 0., z_history)
        inputs = (X, Y, R, next_day_flags, correct_bias_flags, subkeys)
        
        (_, _, log_lik, z_history), _ = jax.lax.scan(scan_fn, carry, inputs, length=T)
        return z_history, log_lik

    def bias_correction(self, w):
        if w.shape[-1] == 5:
            return bias_correction(w)
        else:
            return w
    
    def score_predict(self,
            key, params,
            X_hist: jnp.ndarray, Y_hist: jnp.ndarray,
            X_pred: jnp.ndarray, Y_pred: jnp.ndarray,
            R_hist: jnp.ndarray = None, day_flags: jnp.ndarray = None,
            N_particles: int=10000,
            ):
        '''
        Do filtering to obtain last weights, then sample weights trajectories from there and compare
        sampled decisions with true decisions.
        '''
        T = len(X_hist) + len(X_pred)
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)

        # Step 1: filtering to obtain last weights
        # (_, Zs_filt), _ = samplers.bootstrap_filter(
        #     N_particles, 
        #     X_hist, Y_hist, 
        #     model, 
        #     R=R_hist, session_indices=session_indices, 
        #     return_history=False, verbose=True,
        #     )
        Zs_filt_T, _ = self.posterior_samples(
            key, params, X_hist, Y_hist, R=R_hist, day_flags=day_flags,
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
            w = self.update_weights(update_key, w, params=params, x=X_pred[t], y=y, r=r, day_flag=day_flags[t])

            Ys.append(y)
            Ws.append(w)

        # Compute score 
        score = jnp.mean(jnp.array(Ys) == jnp.array(Y_pred))
        return score
    
    def next_step_prediction_score(
            self, key, params, X, Y, R=None, day_flags=None,
            N_particles=1000, verbose=False):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)

        def scan_fn(carry, inputs):
            tilde_z_t, log_lik = carry
            X_t, Y_t, R_t, next_day_flag, (subkey1, subkey2) = inputs

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            tilde_w_t = self.emission_likelihood(tilde_z_t, X_t, Y_t)

            # 3. Resample with replacement N particles according the importance weights
            z_t = jax.random.choice(subkey1, tilde_z_t, shape=(N_particles,), p=tilde_w_t)

            # 4. Update log-likelihood estimate
            lik = jnp.mean(tilde_w_t)
            log_lik += jnp.log(lik)
            
            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            #    Outcome: {tilde z_t^i, 1/N}, an approximation to p(z_t|y_{1:t-1})
            tilde_z_t = self.update_weights(subkey2, z_t, params=params, x=X_t, y=Y_t, r=R_t, day_flag=next_day_flag)
            
            return (tilde_z_t, log_lik), lik
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=(T, 2))
        inputs = (X, Y, R, next_day_flags, subkeys)
        
        (_, log_lik), scores = jax.lax.scan(scan_fn, carry, inputs, length=T)
        return scores, log_lik
    
    
    def two_step_prediction_score(
            self, key, params, X, Y, R=None, day_flags=None,
            N_particles=1000, verbose=False):
        '''Use scan to make computation more efficient.'''
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        if R is None:
            R = [None for _ in range(T)]
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)

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

            # # 4. Update log-likelihood estimate
            lik = jnp.mean(tilde_w_t)
            log_lik += jnp.log(lik)

            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            tilde_z_t = self.update_weights(subkey3, z_t, params=params, x=X_t1, y=Y_t1, r=R_t1, day_flag=next_day_flag)
            
            return (tilde_z_t, log_lik), pz2
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=(T, 3))
        inputs = (X[:-1], Y[:-1], R[:-1], X[1:], Y[1:], next_day_flags[:-1], subkeys[:-1])
        (tilde_z_t, log_lik), scores = jax.lax.scan(scan_fn, carry, inputs, length=T-1)

        # Add last step loglik
        log_lik += jnp.log(jnp.mean(self.emission_likelihood(tilde_z_t, X[-1], Y[-1])))

        return scores, log_lik
    
    def sample_forward(self, key, params, X: Float[Array, "T M"], day_flags: Bool[Array, "T"], z_0=None):
        '''
        Forward pass through the model from initial state, sampling decisions.
        '''
        assert self.reward_func is not None, "Reward function, (x_t, y_t) -> r_t, must be defined."
        if z_0 is None:
            z_0 = self.sample_initial(key, params, N=1, d=self.latent_dim-1).squeeze()
            assert z_0.shape == (self.latent_dim,)
       
        T = len(X)
        latent_dim = z_0.shape[0]

        # Forward pass
        def scan_fn(carry, inputs):
            y_t, z_t, log_lik = carry

            # Process inputs
            subkey, X_t, X_next, day_flag = inputs
            r_t = self.reward_func(X_t, y_t)  

            # Update state
            z_pred = self.update_weights(subkey, z_t, X_t, params, y_t, r_t, day_flag)
            y_pred = self.decision(subkey, z_pred, X_next)

            pred_lik = self.emission_likelihood(z_pred, X_next, y_pred)
            log_lik += jnp.log(jnp.mean(pred_lik))

            return (y_pred, z_pred, log_lik), (y_pred, z_pred)
        
        # Initialize
        subkeys = jax.random.split(key, T)
        y_0 = self.decision(subkeys[0], z_0, X[0])
        log_lik = jnp.log(jnp.mean(self.emission_likelihood(z_0, X[0], y_0)))
        init = (y_0, z_0, log_lik)
        inputs = (subkeys[1:], X[:-1], X[1:], day_flags[:-1])

        # Forward pass
        (_, _, log_lik), (Y_pred, Z_pred) = jax.lax.scan(scan_fn, init, inputs, length=T-1)
        Y = jnp.concatenate([jnp.array([y_0]), Y_pred]); assert Y.shape == (T,)
        Z = jnp.concatenate([jnp.array([z_0]), Z_pred]); assert Z.shape == (T, latent_dim)
        return Y, Z

    def forward_pass(self, 
            key, params, 
            X: Float[Array, "T M"], Y: Float[Array, "T"], R: Float[Array, "T"], day_flags: Bool[Array, "T"], 
            N_particles: int=1000, predict_Y: bool=False, correct_bias: bool=True, return_Z: bool=False
            ):
        '''
        Make a forward pass in the model, either using the animal decisions (predict_Y=False) or sampling decisions
        (predict_Y=True). Predicition time-step to time-step likelihoods are returned.
        '''
        T, M = X.shape

        if predict_Y:
            assert self.reward_func is not None, "Reward function, (x_t, y_t) -> r_t, must be defined."

            def scan_fn(carry, inputs):
                tilde_z_t, log_lik = carry
                X_t, Y_t, R_t, next_day_flag, correct_bias_flag, subkey = inputs

                # # Correct bias
                # tilde_z_t = jax.lax.select(correct_bias_flag, self.bias_correction(tilde_z_t), tilde_z_t)

                # Evaluate likelihood/importance weights with sampled decision
                decision_key, update_key = jax.random.split(subkey)
                Y_pred = self.decision(decision_key, tilde_z_t, X_t)
                R_pred = jax.vmap(lambda y: self.reward_func(X_t, y))(Y_pred)

                # Log-lik
                # w_true = self.emission_likelihood(tilde_z_t, X_t, Y_t)
                log_w_true = self.emission_loglikelihood(tilde_z_t, X_t, Y_t)
                # w_pred = self.emission_likelihood(tilde_z_t, X_t, Y_pred)
                log_w_pred = self.emission_loglikelihood(tilde_z_t, X_t, Y_pred)
                # log_lik += jnp.log(jnp.mean(w_true)) # store log-lik of true data
                log_lik_t = logmeanexp(log_w_true)
                log_lik += log_lik_t

                # Update step
                # # z_t = jax.random.choice(update_key, tilde_z_t, shape=(N_particles,), p=w_pred)
                # z_t = sample_particles_gumbel(update_key, tilde_z_t, log_w_pred, N_particles) #! shouldn't really impact, right?
                z_t = tilde_z_t

                # Prediction step: z_t ~ p(z_t | z_{t-1})
                z_next = jax.vmap(
                    lambda z, y, r: self.update_weights(
                        update_key, z, params=params, x=X_t, y=y, r=r, 
                        day_flag=next_day_flag, correct_bias=correct_bias_flag
                        )
                    )(z_t, Y_pred, R_pred)
                
                return (z_next, log_lik), (log_lik_t, z_next)
        else:
            def scan_fn(carry, inputs):
                tilde_z_t, log_lik = carry
                X_t, Y_t, R_t, next_day_flag, correct_bias_flag, subkey = inputs

                # # Correct bias
                # tilde_z_t = jax.lax.select(correct_bias_flag, self.bias_correction(tilde_z_t), tilde_z_t)

                # Evaluate likelihood/importance weights with true decision
                w_t = self.emission_likelihood(tilde_z_t, X_t, Y_t)
                log_w_t = self.emission_loglikelihood(tilde_z_t, X_t, Y_t)
                # log_lik += jnp.log(jnp.mean(w_t))
                log_lik_t = logmeanexp(log_w_t)
                log_lik += log_lik_t

                # Update step
                p = jnp.exp(log_w_t - jnp.max(log_w_t))
                z_t = jax.random.choice(subkey, tilde_z_t, shape=(N_particles,), p=p)
                # z_t = sample_particles_gumbel(subkey, tilde_z_t, log_w_t, N_particles)

                # Prediction step: z_t ~ p(z_t | z_{t-1})
                z_next = self.update_weights(
                    subkey, z_t, params=params, x=X_t, y=Y_t, r=R_t, 
                    day_flag=next_day_flag, correct_bias=correct_bias_flag
                    )

                # jax.debug.print("log tilde w_t = {}, p = {}, log_lik = {}, tilde_z_t = {}, z_t = {}, Y_t={}, z_next = {}", log_w_t, jnp.exp(log_w_t - log_w_t.max()), log_lik_t, tilde_z_t[0], z_t[0],Y_t, z_next[0])
                
                return (z_next, log_lik), (log_lik_t, z_next)
        
        key, subkey = jax.random.split(key)
        z_0 = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (z_0, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        if correct_bias and X.shape[1] == 5:
            correct_bias_flags = jnp.cumprod(jnp.abs(X[:,1] - X[:,0]) >= 0.9).astype(bool) # True until stim intensity goes below 0.9 in abs
        else:
            correct_bias_flags = jnp.zeros(T, dtype=bool)
        subkeys = jax.random.split(key, num=T)
        inputs = (X, Y, R, next_day_flags, correct_bias_flags, subkeys)
        
        if return_Z:
            (_, log_lik), (logliks, Z) = jax.lax.scan(scan_fn, carry, inputs) #, length=T)

            # Replace Z[0] and shift other Zs
            Z = jnp.concatenate([jnp.array([z_0]), Z[:-1]])
            Z = Z.transpose(1,0,2)
        else:
            (_, log_lik), (logliks, _) = jax.lax.scan(scan_fn, carry, inputs) #, length=T)
            Z = None

        return (logliks, Z), log_lik

    def filtering_MLL(self, 
            key, params, 
            X: Float[Array, "T M"], Y: Float[Array, "T"], R: Float[Array, "T"], day_flags: Bool[Array, "T"], 
            N_particles=1000
            ):
        '''
        return p(w_{1:T} | y_{1:T}, x_{1:T}) under the prior, using the true data.
        '''
        T, M = X.shape

        def scan_fn(carry, inputs):
            z_t, log_lik = carry
            X_t, Y_t, R_t, next_day_flag, subkey = inputs

            # Log-lik
            lik = self.emission_likelihood(z_t, X_t, Y_t)
            log_lik += jnp.log(jnp.mean(lik))

            # Update z_t ~ p(z_t | z_{t-1})
            z_next = self.update_weights(subkey, z_t, params=params, x=X_t, y=Y_t, r=R_t, day_flag=next_day_flag)
            
            return (z_next, log_lik), None
        
        key, subkey = jax.random.split(key)
        z_0 = self.sample_initial(subkey, params=params, N=N_particles, d=M)
        carry = (z_0, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=T)
        inputs = (X, Y, R, next_day_flags, subkeys)
        
        (_, log_lik), _ = jax.lax.scan(scan_fn, carry, inputs) #, length=T)

        return log_lik
    
    def held_out_session_marginal_log_likelihood(self, 
            key, params, 
            t1, t2,
            X, Y, R=None, day_flags=None,
            N_particles=1000):
        '''Compute the predictive marginal log likelihood of held out data.
            log p(y_{t1:t2} | y_{1:t1}, x_{1:t2}, theta)
        '''
        _, logliks = self.marginal_log_likelihood(
            key, params, X[:t2], Y[:t2], R[:t2], day_flags, N_particles,
            return_logliks = True,
            )
        return logliks[t1:t2].sum()
    
    def held_out_trials_marginal_log_likelihood(self, 
            key, params, 
            held_out_trials,
            X, Y, R=None, day_flags=None,
            N_particles=1000):
        '''Compute the predictive marginal log likelihood of held out data for each held out trial t,
            log p(y_{t} | y_{1:t-1}, x_{1:t}, theta)
        '''
        T = max(held_out_trials)
        _, logliks = self.marginal_log_likelihood(
            key, params, X[:T], Y[:T], R[:T], day_flags[:T], N_particles,
            return_logliks = True,
            )
        return logliks[held_out_trials]
    
    def predict_trials_score(
            self, key, params,
            X, Y, R, day_flags,
            held_out_interval, 
            N_particles=1000, verbose=False
            ):
        T_in, T_out = held_out_interval
        assert T_out > 0 and T_out <= len(X), "Invalid held out interval."

        Y_masked = Y.copy()
        Y_masked[T_in:T_out] = jnp.nan
        Z_post, _ = self.posterior_samples(
            key, params, 
            X[:T_out], Y_masked[:T_out], R[:T_out], day_flags[:T_out],
            N_particles=N_particles, verbose=verbose, LAG=True,
            )
        
        # liks = []
        # for t in range(T_in, T_out):
        #     lik = self.emission_likelihood(Z_post[:,t,:], X[t], Y[t])
        #     liks.append(jnp.mean(lik))
        # liks = jnp.array(liks)
        liks = jax.vmap(
            lambda z, x, y: jnp.mean(self.emission_likelihood(z, x, y))
            )(Z_post[:,T_in:,:].transpose(1,0,2), X[T_in:T_out], Y[T_in:T_out])

        # logliks = jax.vmap(
        #     lambda t: jnp.log(self.emission_likelihood(Z_post[:,t,:], X[t], Y[t]).mean())
        #     )(jnp.arange(T_in, T_out))
        return liks

class Psytrack(GLMLearn):
    def __init__(self, 
                 z_0=0.0,
                 latent_dim = None,
                 ) -> None:
        self.z_0 = z_0
        self.latent_dim = latent_dim
        self.reward_func = None

    def __repr__(self) -> str:
        return f"Psytrack()"

    def update_weights(self, key: PRNGKey, w, x, params: ParamsPsytrack, y=None, r=None, day_flag: bool = False, 
                       return_learning_signal=False, correct_bias=False):
        # Add noise
        update_noise = jax.random.normal(key, shape=w.shape)
        update_noise = jnp.where(day_flag, 
                                 jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
                                 jnp.multiply(jnp.exp(params.log_sigma), update_noise)
                                 )
        return w + update_noise
    
    def dynamics_loglikelihood(self, z_next, z_prev, inputs, data, 
                               params: ParamsGLMLearn, day_flag=False, r=None):
        '''p(z_t | z_{t-1})
        In our case, the latents z are the GLM weights w
        '''
        N = z_prev.shape[0]

        cov = jnp.where(day_flag, 
            jnp.multiply(jnp.square(jnp.exp(params.log_sigma_day)), jnp.eye(N)),
            jnp.multiply(jnp.square(jnp.exp(params.log_sigma)), jnp.eye(N))
            )
        
        log_lik = lambda z: jsp.stats.multivariate_normal.logpdf(z, mean=jnp.zeros_like(z_prev), cov=cov)
        return log_lik(z_next)

class TimeVarGLMLearn(GLMLearn):
    r'''
    If lapse, beta = lapse state, sigmoid(beta) is the lapse rate. 
        beta_0 should be set negative, so that sigmoid(beta) is close to 0.
    If not lapse, beta = reward value (IRL) setting. 
        beta_0 should be set to 1.
    '''
    def __init__(self, lapse, beta_dim, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lapse = lapse
        self.beta_dim = beta_dim # len(beta), 1 for scalar.

    def __repr__(self) -> str:
        return f"TimeVarGLMLearn(lapse={self.lapse}, beta_dim={self.beta_dim})" 

    def split_latent(self, z):
        # Split latent variable into beta and weights, over last axis
        if z.ndim == 1:
            beta, w = jnp.array(z[:self.beta_dim]), z[self.beta_dim:]
        else:
            beta, w = z[..., :self.beta_dim], z[..., self.beta_dim:]
        return beta, w
    
    def merge_latent(self, beta, w):
        # if w.ndim == 1:
        #     return jnp.concatenate([jnp.array([beta]), w])
        # else:
        return jnp.concatenate([beta, w], axis=-1)

    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        beta = params.beta_0 + jnp.exp(params.log_sigma_0) * jax.random.normal(key, shape=(N, self.beta_dim))
        w = super().sample_initial(key, params=None, N=N, d=d)
        z = self.merge_latent(beta, w)
        assert z.shape == (N, self.beta_dim + d + 1)
        return z
        
    def update_weights(self, key: PRNGKey, z, x, params, y=None, r=None, day_flag=False, return_learning_signal=False):
        beta_t, w_t = self.split_latent(z)
        key, beta_key, w_key = jax.random.split(key, 3)
        
        if self.lapse:
            log_learning_rate = params.log_alpha + jnp.log(1 - sigmoid(beta_t))
        else:
            log_learning_rate = params.log_alpha + beta_t
        
        # learning_signal = jnp.multiply(jnp.exp(log_learning_rate), regression_gradient(w_t, x, y, r, params))
        # learning_signal = jnp.multiply(jnp.exp(params.log_alpha), regression_gradient(w_t, x, y, params, r1=beta_t[:,1], r0=beta_t[:,0]))

         # Add noise
        update_noise = jax.random.normal(w_key, shape=w_t.shape)
        update_noise = jnp.where(day_flag, 
                                 jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
                                 jnp.multiply(jnp.exp(params.log_sigma), update_noise)
                                 )
        
        # if return_learning_signal:
        #     w_t, learning_signal = super().update_weights(w_key, w_t, x, params_GLM, y, r, day_flag, 
        #                                                   return_learning_signal=return_learning_signal)
        # else:
        #     w_t = super().update_weights(w_key, w_t, x, params_GLM, y, r, day_flag)

        # Forgetting
        # forget_term = jnp.where(day_flag, w_t * jnp.exp(params.log_forget_day), w_t * jnp.exp(params.log_forget))
        # forget_term = jnp.where(day_flag, w_t @ jnp.diag(params.Q_day), w_t @ jnp.diag(params.Q))
        forget_term = w_t @ jnp.diag(jnp.exp(params.log_Q))
        if self.lapse:
            learning_signal = learning_signal - jnp.multiply(sigmoid(beta_t), forget_term)
        else:
            learning_signal = learning_signal - forget_term

        w_t = w_t + learning_signal + update_noise
        beta_t = beta_t + jnp.exp(params.log_sigma_0) * jax.random.normal(beta_key, shape=beta_t.shape)
        z_t = self.merge_latent(beta_t, w_t)

        if return_learning_signal:
            return z_t, learning_signal
        else:
            return z_t
    
    def emission_likelihood(self, z, x, y, params=None):
        beta, w = self.split_latent(z)
        p = bernoulli_GLM_likelihood(w, x, y)
        if self.lapse:
            lapse_rate = sigmoid(beta).mean(1).squeeze()
            p = jnp.multiply(1 - lapse_rate, p) + 0.5 * lapse_rate

        # p = jnp.where(
        #     self.lapse,
        #     (1 - sigmoid(beta)) * bernoulli_GLM_likelihood(w, x, y) + 0.5 * sigmoid(beta),
        #     bernoulli_GLM_likelihood(w, x, y),
        # )
        return p
    
    def emission_loglikelihood(self, z, x, y, params=None):
        if self.lapse:
            return jnp.log(self.emission_likelihood(z, x, y, params))
        else:
            _, w = self.split_latent(z)
            log_p = bernoulli_GLM_loglikelihood(w, x, y)
            return log_p
    
    def decision(self, key: PRNGKey, z, x):
        p_R = self.emission_likelihood(z, x, y=Y_R)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y
    
    def dynamics_loglikelihood(self, z_next, z_prev, inputs, data, params: ParamsGLMLearn, day_flag=False, r=None):
        raise NotImplementedError
    
    def log_joint(self, X, Y, Z, params, R=None, day_flags=None):
        raise NotImplementedError
    
    def bias_correction(self, z):
        beta, w = self.split_latent(z)
        w_corrected = bias_correction(w)
        return self.merge_latent(beta, w_corrected)
    
class AC(GLMLearn):
    def __init__(self, beta_dim, sigmoid=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sigmoid = sigmoid
        self.beta_dim = beta_dim

    def __repr__(self):
        return f"AC(beta_dim={self.beta_dim}, sigmoid={self.sigmoid})" 
    
    def split_latent(self, z):
        # Split latent variable into beta and weights, over last axis
        if z.ndim == 1:
            beta, w = jnp.array(z[:self.beta_dim]), z[self.beta_dim:]
        else:
            beta, w = z[..., :self.beta_dim], z[..., self.beta_dim:]
        return beta, w
    
    def merge_latent(self, beta, w):
        return jnp.concatenate([beta, w], axis=-1)
    
    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        beta = params.beta_0 + jnp.exp(params.log_sigma_0) * jax.random.normal(key, shape=(N, self.beta_dim))
        w = super().sample_initial(key, params=None, N=N, d=d)
        z = self.merge_latent(beta, w)
        assert z.shape == (N, self.beta_dim + d + 1)
        return z
    
    def update_weights(self, key: PRNGKey, z, x, params, y=None, r=None, 
                       day_flag=False, return_learning_signal=False, correct_bias=False):
        beta_t, w_t = self.split_latent(z)
        key, beta_key, w_key = jax.random.split(key, 3)

        if self.sigmoid:
            baseline_t = sigmoid(beta_t)
        else:
            baseline_t = beta_t
        
        # Change in mean weights from learning rule
        learning_signal = jnp.multiply(jnp.exp(params.log_alpha), reinforce(w_t, x, y, r=r-baseline_t))
        
        forget_term = w_t @ jnp.diag(jnp.exp(params.log_Q))
        learning_signal = learning_signal - forget_term

         # Add noise
        update_noise = jax.random.normal(w_key, shape=w_t.shape)
        sigma_t = jnp.where(
            day_flag,
            jnp.exp(params.log_sigma_day),
            jnp.exp(params.log_sigma)
        )
        update_noise = update_noise * sigma_t

        # Correct bias
        # if correct_bias:
        #     learning_signal = learning_signal.at[..., 0].set(0.0)
        #     update_noise = update_noise.at[..., 0].set(0.0)
        learning_signal = jnp.where(correct_bias, learning_signal.at[..., 0].set(0.0), learning_signal)
        update_noise = jnp.where(correct_bias, update_noise.at[..., 0].set(0.0), update_noise)
        
        # Combine and update
        w_next = w_t + learning_signal + update_noise
        beta_next = beta_t + jnp.exp(params.log_sigma_0) * jax.random.normal(beta_key, shape=beta_t.shape)
        z_next = self.merge_latent(beta_next, w_next)

        if return_learning_signal:
            return z_next, learning_signal
        else:
            return z_next
    
    def emission_likelihood(self, z, x, y, params=None):
        _, w = self.split_latent(z)
        p = bernoulli_GLM_likelihood(w, x, y)
        return p
    
    def emission_loglikelihood(self, z, x, y, params=None):
        _, w = self.split_latent(z)
        log_p = bernoulli_GLM_loglikelihood(w, x, y)
        return log_p

    def decision(self, key: PRNGKey, z, x):
        p_R = self.emission_likelihood(z, x, y=Y_R)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y
    
    def bias_correction(self, z):
        beta, w = self.split_latent(z)
        w_corrected = bias_correction(w)
        return self.merge_latent(beta, w_corrected)


class GLMRegLearn(GLMLearn):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"GLMRegLearn()"

    def update_weights(
            self, 
            key: PRNGKey, w, x, 
            params: parameters.ParamsGLMRegLearn, y=None, r=None,
            day_flag: bool=False,
            return_noise=False):
        
        # Change in mean weights from learning rule
        learning_signal = jnp.multiply(jnp.exp(params.log_alpha), regression_gradient(w, x, y, r, params))

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

class GLMBaseLearn(GLMLearn):
    def __init__(self, time_var=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.time_var = time_var
        self.beta_dim = 5

    def __repr__(self) -> str:
        return f"GLMBaseLearn(time_var={self.time_var})"
    
    def split_latent(self, z):
        # Split latent variable into beta and weights, over last axis
        if z.ndim == 1:
            beta, w = jnp.array(z[:self.beta_dim]), z[self.beta_dim:]
        else:
            beta, w = z[..., :self.beta_dim], z[..., self.beta_dim:]
        return beta, w
    
    def merge_latent(self, beta, w):
        # if w.ndim == 1:
        #     return jnp.concatenate([jnp.array([beta]), w])
        # else:
        return jnp.concatenate([beta, w], axis=-1)

    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        w = super().sample_initial(key, params=None, N=N, d=d)
        if self.time_var:
            beta = params.baseline_weights + jnp.exp(params.log_sigma_0) * jax.random.normal(key, shape=(N, self.beta_dim))
            z = self.merge_latent(beta, w)
            assert z.shape == (N, self.beta_dim + d + 1)
        else:
            z = w
        return z

    def update_weights(
            self, 
            key: PRNGKey, z, x, 
            params: parameters.ParamsGLMBaseLearn, y=None, r=None,
            day_flag: bool=False,
            return_noise=False):
        
        if self.time_var:
            beta_t, w_t = self.split_latent(z)
        else:
            w_t = z
            beta_t = params.baseline_weights

        key, w_key, beta_key = jax.random.split(key, 3)
        
        # Change in mean weights from learning rule
        learning_signal = jnp.multiply(jnp.exp(params.log_alpha), REINFORCE_with_baseline(w_t, x, y, r, params, baseline_weights=beta_t))
        
        forget_term = w_t @ jnp.diag(jnp.exp(params.log_Q))
        learning_signal = learning_signal - forget_term

        # Add noise
        update_noise = jax.random.normal(w_key, shape=w_t.shape)
        update_noise = jnp.where(day_flag, 
                                 jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
                                 jnp.multiply(jnp.exp(params.log_sigma), update_noise)
                                 )
        w_t = w_t + learning_signal + update_noise
        
        if self.time_var:
            beta_t = beta_t + jnp.exp(params.log_sigma_0) * jax.random.normal(beta_key, shape=beta_t.shape)
            z_t = self.merge_latent(beta_t, w_t)
        else:
            z_t = w_t
        
        if return_noise:
            return z_t, update_noise
        else:
            return z_t
        
    def emission_likelihood(self, z, x, y):
        if self.time_var:
            _, w = self.split_latent(z)
        else:
            w = z
        p = bernoulli_GLM_likelihood(w, x, y)
        return p
        
class GLMInterpLearn(GLMLearn):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"GLMInterpLearn()"

    def update_weights(
            self, 
            key: PRNGKey, w, x, 
            params: parameters.ParamsGLMInterpLearn, y=None, r=None,
            day_flag: bool=False,
            return_noise=False):
        
        # Change in mean weights from learning rule
        learning_signal = jnp.multiply(jnp.exp(params.log_alpha), regression_gradient(w, x, y, params))

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


class RVBF(GLMLearn):
    r'''
    Policy: Bernoulli GLM
    Learning rule: Reinforce (R) with vectorized (V) parameters, including baseline (B) and forgetting (F) terms. 
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"RVBF()"
    
    def update_weights(
            self, 
            key: PRNGKey, w, x, 
            params: parameters.ParamsGLMInterpLearn, y=None, r=None,
            day_flag: bool=False,
            return_noise=False, return_learning_signal=False, correct_bias=False):
        
        # Change in mean weights from learning rule
        REINFORCE_term = reinforce(w, x, y, r=r-params.baseline)
        learning_signal = jnp.multiply(jnp.exp(params.log_alpha), REINFORCE_term)

        # Forgetting
        forget_term = w @ jnp.diag(jnp.exp(params.log_Q))
        learning_signal = learning_signal - forget_term

        # Add noise with per-dimension scales
        update_noise = jax.random.normal(key, shape=w.shape)
        sigma = jnp.where(
            day_flag,
            jnp.exp(params.log_sigma_day),
            jnp.exp(params.log_sigma)
        )
        update_noise = update_noise * sigma

        # Correct for bias if needed
        learning_signal = jnp.where(correct_bias, learning_signal.at[..., 0].set(0.0), learning_signal)
        update_noise = jnp.where(correct_bias, update_noise.at[..., 0].set(0.0), update_noise)

        if return_learning_signal:
            return w + learning_signal + update_noise, learning_signal
        else:
            return w + learning_signal + update_noise
        
class TimeVarRVBF(GLMLearn):
    r'''
    RVBF model with time-varying learning rate (modulator = 'lr') or time-varying baseline (modulator = 'baseline').
    Modulator is a function of the latent variable beta_t, which is a scalar or vector.
        beta_t = log(alpha_t) or beta_t = baseline_t
        beta_0 = 0. by default
    '''
    def __init__(self, modulator: str='lr', beta_dim: int=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.modulator = modulator
        self.beta_dim = beta_dim # len(beta), 1 for scalar.

    def __repr__(self) -> str:
        return f"TimeVarRVBF(modulator={self.modulator}, beta_dim={self.beta_dim})"

    def split_latent(self, z):
        # Split latent variable into beta and weights, over last axis
        if z.ndim == 1:
            beta, w = jnp.array(z[:self.beta_dim]), z[self.beta_dim:]
        else:
            beta, w = z[..., :self.beta_dim], z[..., self.beta_dim:]
        return beta, w
    
    def merge_latent(self, beta, w):
        return jnp.concatenate([beta, w], axis=-1)

    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        beta = params.beta_0 + jnp.exp(params.log_sigma_0) * jax.random.normal(key, shape=(N, self.beta_dim))
        w = super().sample_initial(key, params=None, N=N, d=d)
        z = self.merge_latent(beta, w)
        assert z.shape == (N, self.beta_dim + d + 1)
        return z
        
    def update_weights(self, key: PRNGKey, z, x, params, y=None, r=None, 
                       day_flag=False, return_learning_signal=False, correct_bias=False):
        beta_t, w_t = self.split_latent(z)
        key, beta_key, w_key = jax.random.split(key, 3)

        if self.modulator == 'lr':
            log_alpha_t = params.log_alpha + beta_t
            baseline_t = params.baseline
        elif self.modulator == 'baseline':
            log_alpha_t = params.log_alpha
            baseline_t = beta_t

        # Change in mean weights from learning rule
        REINFORCE_term = reinforce(w_t, x, y, r=r-baseline_t)
        learning_signal = jnp.multiply(jnp.exp(log_alpha_t), REINFORCE_term)

        # Forgetting
        forget_term = w_t @ jnp.diag(jnp.exp(params.log_Q))
        learning_signal = learning_signal - forget_term

        # Add noise with per-dimension scales
        update_noise = jax.random.normal(w_key, shape=w_t.shape)
        sigma = jnp.where(
            day_flag,
            jnp.exp(params.log_sigma_day),
            jnp.exp(params.log_sigma)
        )
        update_noise = update_noise * sigma

        # Correct for bias if needed
        # if correct_bias:
        #     learning_signal = learning_signal.at[..., 0].set(0.0)
        #     update_noise = update_noise.at[..., 0].set(0.0)
        learning_signal = jnp.where(correct_bias, learning_signal.at[..., 0].set(0.0), learning_signal)
        update_noise = jnp.where(correct_bias, update_noise.at[..., 0].set(0.0), update_noise)

        # Combine and update
        beta_next = beta_t + jnp.exp(params.log_sigma_0) * jax.random.normal(beta_key, shape=beta_t.shape)
        w_next = w_t + learning_signal + update_noise
        z_next = self.merge_latent(beta_next, w_next)

        if return_learning_signal:
            return z_next, learning_signal
        else:
            return z_next
    
    def emission_likelihood(self, z, x, y, params=None):
        _, w = self.split_latent(z)
        p = bernoulli_GLM_likelihood(w, x, y)
        return p
    
    def emission_loglikelihood(self, z, x, y, params=None):
        _, w = self.split_latent(z)
        log_p = bernoulli_GLM_loglikelihood(w, x, y)
        return log_p
    
    def bias_correction(self, z):
        beta, w = self.split_latent(z)
        w_corrected = bias_correction(w)
        return self.merge_latent(beta, w_corrected)

class GLMHMMLearn(GLMLearn):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"GLMHMMLearn(learning_rule={self.learning_rule})"

    def split_latent(self, z):
        if z.ndim == 1:
            state, w = jnp.array(z[0], dtype=int), z[1:]
        else:
            state, w = z[..., 0], z[..., 1:]
        return state, w
    
    def merge_latent(self, state, w):
        if w.ndim == 1:
            # alpha is scalar
            return jnp.concatenate([jnp.array([state], dtype=int), w])
        else:
            return jnp.concatenate([state[:, None], w], axis=1)
        
    def construct_pi0(self, params):
        pi0 = jnp.array([sigmoid(params.logit_pi0), 1 - sigmoid(params.logit_pi0)])
        return pi0
    
    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        state = jax.random.choice(key, 2, shape=(N,), p=self.construct_pi0(params))
        w = super().sample_initial(key, params=None, N=N, d=d)
        return self.merge_latent(state, w)
    
    def construct_A(self, params):
        A = jnp.array([[sigmoid(params.logit_a_1), 1 - sigmoid(params.logit_a_1)], 
                       [1 - sigmoid(params.logit_a_2), sigmoid(params.logit_a_2)]])
        return A
    
    def update_weights(self, key: PRNGKey, z, x, params, y=None, r=None, day_flag=False, return_noise=False):
        '''
        State dependent weight updates. In one state, we follow a policy gradient update, 
        in the other there is no learning. 
        '''
        state_t, w_t = self.split_latent(z)
        key, state_key, w_key = jax.random.split(key, 3)

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

        transition_matrix = self.construct_A(params)
        state_t = jax.vmap(lambda A_row: jax.random.choice(state_key, 2, p=A_row))(transition_matrix[state_t.astype(int)])
        z_t = self.merge_latent(state_t, w_t)
        return z_t
    
    def emission_likelihood(self, z, x, y):
        state, w = self.split_latent(z)
        p = jnp.where(state, 
                        0.5,
                        bernoulli_GLM_likelihood(w, x, y),
                        )
        # p = bernoulli_GLM_likelihood(w, x, y)
        return p
    
    def decision(self, key: PRNGKey, z, x):
        p_R = self.emission_likelihood(z, x, y=1)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y

class DynamicGLMHMM():
    def __init__(self, A=None, K=2, w_dim=5) -> None:
        self.A = A # Global transition matrix
        if self.A is None:
            self.A = jax.nn.softmax(jnp.eye(K), axis=1)
            self.A = jnp.array([[0.975, 0.013, 0.011], [0.02, 0.971, 0.009], [0.018, 0.011, 0.972]]) # from paper
        self.K = K
        self.w_dim = w_dim

    def __repr__(self) -> str:
        return f"DynamicGLMHMM(K={self.K})"

    def split_latent(self, z):
        if z.ndim == 1:
            state, P_flat, W_flat = jnp.array(z[:1], dtype=int), z[1:-self.w_dim * self.K], z[-self.w_dim * self.K:]
            P = jnp.reshape(P_flat, (self.K, self.K))
            W = jnp.reshape(W_flat, (self.K, self.w_dim))
        else:
            state, P_flat, W_flat = z[..., :1], z[..., 1:-self.w_dim * self.K], z[..., -self.w_dim * self.K:]
            P = jnp.reshape(P_flat, (*P_flat.shape[:-1], self.K, self.K))
            W = jnp.reshape(W_flat, (*W_flat.shape[:-1], self.K, self.w_dim))
        return jnp.asarray(state, dtype=int).squeeze(), P, W
    
    def merge_latent(self, state, P, W):
        P_flat = P.reshape(*P.shape[:-2], -1)
        W_flat = W.reshape(*W.shape[:-2], -1)
        # if state.shape[-1] != 1:
        #     state = state[..., None]
        return jnp.concatenate([state[..., None], P_flat, W_flat], axis=-1)
        
    # def construct_pi0(self, params):
    #     # pi0 = jnp.array([sigmoid(params.logit_pi0), 1 - sigmoid(params.logit_pi0)])
    #     pi0 = jax.nn.softmax(params.logit_pi0)
    #     return pi0

    # def _pi0(self, params):
    #     return jax.nn.softmax(params.logit_pi0)
    
    
    # def _transition_matrix(self, params):
    #     A = jax.nn.softmax(params.logit_A, axis=1)
    #     assert A.shape == (self.K, self.K)
    #     assert jnp.allclose(jnp.sum(A, axis=1), 1)
    #     return A

    def sample_transition_matrix(self, key, params, N=1):
        P = jnp.empty((N, self.K, self.K))
        for k in range(self.K):
            key, subkey = jax.random.split(key)
            P_k = jax.random.dirichlet(subkey, params.alpha * self.A[k] + 1, shape=(N,))
            P = P.at[:,k].set(P_k)
        return P.squeeze() # removes N = 1 if present

    def sample_initial(self, key: PRNGKey, params, N: int):
        '''Sample from the initial distribution p(z_0)'''
        state_key, w_key, P_key = jax.random.split(key, 3)
        
        state = jax.random.choice(state_key, self.K, shape=(N,)) # draw uniformly from states
        P = self.sample_transition_matrix(P_key, params, N=N)
        w = jax.random.normal(w_key, shape=(N, self.K, self.w_dim,))
        return self.merge_latent(state, P, w)
    
    def update_weights(self, key: PRNGKey, z, params, day_flag=False):
        state_t, P_t, W_t = self.split_latent(z)
        key, state_key, P_key, w_key = jax.random.split(key, 4)
        N = len(state_t)

        # Psytrack evolution over sessions
        update_noise = jax.random.normal(w_key, shape=W_t.shape)
        # w_t = jnp.where(day_flag, w_t + jnp.multiply(jnp.exp(params.log_sigma), update_noise), w_t)
        W_t = jnp.where(day_flag,
            W_t + jnp.multiply(jnp.exp(params.log_sigma_day), update_noise),
            W_t + jnp.multiply(jnp.exp(params.log_sigma), update_noise),
        )
        
        # Transition matrices sampled from dirichlet prior
        assert P_t.shape == (N, self.K, self.K)
        P_t = jnp.where(day_flag, self.sample_transition_matrix(P_key, params, N=N), P_t)

        # Transition to next state
        state_t = jax.vmap( # vmap over N
            lambda subkey, state, P: jax.random.choice(subkey, self.K, p=P[state])
            )(jax.random.split(state_key, N), state_t, P_t)
        z_t = self.merge_latent(state_t, P_t, W_t)
        return z_t
    
    def emission_likelihood(self, z, x, y):
        state, _, W = self.split_latent(z)
        W_k = jax.vmap(lambda w, k: w[k])(W, state)
        p = bernoulli_GLM_likelihood(W_k, x, y)
        return p
    
    def decision(self, key: PRNGKey, z, x):
        state, _, W = self.split_latent(z)
        p_R = self.emission_likelihood(W[:,state], x, y=1)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y

    def marginal_log_likelihood(self, key, params, X, Y, R=None, day_flags=None,
                            N_particles=1000, verbose=False, return_logliks = False):
        if X.ndim == 1:
            T = len(X)
            M = 1
        else:
            T, M = X.shape
        assert self.w_dim == M + 1, "Dimension of weights must match dimension of data."

        if R is None:
            R = [None for _ in range(T)]
        if day_flags is None:
            day_flags = jnp.zeros(T, dtype=bool)

        def scan_fn(carry, inputs):
            tilde_z_t, marginal_log_lik = carry
            X_t, Y_t, next_day_flag, (subkey1, subkey2) = inputs

            # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
            tilde_w_t = self.emission_likelihood(tilde_z_t, X_t, Y_t)
            
            # 3. Resample with replacement N particles according the importance weights
            z_t = jax.random.choice(subkey1, tilde_z_t, shape=(N_particles,), p=tilde_w_t)

            # 4. Update log-likelihood estimate
            log_lik = jnp.log(jnp.mean(tilde_w_t))
            marginal_log_lik += log_lik
            
            # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
            tilde_z_t = self.update_weights(subkey2, z_t, params=params, day_flag=next_day_flag)
            
            return (tilde_z_t, marginal_log_lik), log_lik
        
        key, subkey = jax.random.split(key)
        tilde_z_t = self.sample_initial(subkey, params=params, N=N_particles)
        carry = (tilde_z_t, 0.)
        
        next_day_flags = jnp.roll(day_flags, shift=-1)
        subkeys = jax.random.split(key, num=(T, 2))
        inputs = (X, Y, next_day_flags, subkeys)

        # Scan
        (_, marginal_log_lik), log_liks = jax.lax.scan(scan_fn, carry, inputs, length=T)

        # # For loop
        # marginal_log_lik, log_liks = 0., []
        # for t in range(T):
        #     carry, log_lik = scan_fn(carry, (inp[t] for inp in inputs))
        #     marginal_log_lik += log_lik
        #     log_liks.append(log_lik)
        # log_liks = jnp.array(log_liks)

        if return_logliks:
            return marginal_log_lik, log_liks
        else:
            return marginal_log_lik

def reinforce_bernoulli(v, q, r, baseline=0.):
    return -(r-baseline) * 1/((1-v) - q)

class HRL(GLMLearn):
    def __init__(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"HRL()"

    def split_latent(self, z):
        # Split latent variable into option, q, and weights over the last axis.
        if z.ndim == 1:
            option, q, w = z[0], z[1], z[2:]
        else:
            option, q, w = z[..., 0], z[..., 1], z[..., 2:]
        return option, q, w

    def merge_latent(self, option, q, w):
        # Ensure option and q are at least 1D so that concatenation reverses split_latent.
        option = jnp.atleast_1d(option)
        q = jnp.atleast_1d(q)
        # If w is 3D but option or q are only 2D, unsqueeze them on the last axis.
        if option.ndim == w.ndim - 1:
            option = jnp.expand_dims(option, axis=-1)
        if q.ndim == w.ndim - 1:
            q = jnp.expand_dims(q, axis=-1)
        return jnp.concatenate([option, q, w], axis=-1)
    
    def update_weights(self, key: PRNGKey, z, x, params, y=None, r=None, 
                       day_flag=False, return_learning_signal=False, correct_bias=False):
        option_t, q_t, w_t = self.split_latent(z)
        key, option_key, w_key = jax.random.split(key, 3)

        # Top level option PG
        RF_top = reinforce_bernoulli(option_t, q_t, r, baseline=params.baseline_0)
        learning_signal_0 = jnp.multiply(jnp.exp(params.log_alpha_0), RF_top)

        # Bottom level weights PG
        RF_bottom = reinforce(w_t, x, y, r=r-params.baseline_1)
        learning_signal_1 = jnp.multiply(jnp.exp(params.log_alpha_1), RF_bottom)

        # Update noise
        update_noise_1 = jax.random.normal(w_key, shape=w_t.shape)
        sigma_1 = jnp.where(day_flag, jnp.exp(params.log_sigma_day), jnp.exp(params.log_sigma))
        update_noise_1 = update_noise_1 * sigma_1
        
        update_noise_0 = jax.random.normal(option_key, shape=option_t.shape) * jnp.exp(params.log_sigma_0)
        
        # Correct for bias if needed
        learning_signal_1 = jnp.where(correct_bias, learning_signal_1.at[..., 0].set(0.0), learning_signal_1)
        update_noise_1 = jnp.where(correct_bias, update_noise_1.at[..., 0].set(0.0), update_noise_1)
        
        # Combine and update
        q_next = q_t + learning_signal_0 + update_noise_0
        option_next = jax.random.bernoulli(option_key, p=jax.nn.sigmoid(q_next)).astype(int)
        w_next = w_t + learning_signal_1 + update_noise_1

        z_next = self.merge_latent(option_next, q_next, w_next)
        if return_learning_signal:
            return z_next, learning_signal_1
        else:
            return z_next
        
    def emission_likelihood(self, z, x, y, params=None):
        option, _, w = self.split_latent(z)
        p = jnp.where(option, 
                        0.5,
                        bernoulli_GLM_likelihood(w, x, y),
                        )
        return p
    
    def emission_loglikelihood(self, z, x, y, params=None):
        return jnp.log(self.emission_likelihood(z, x, y))

    def sample_initial(self, key: PRNGKey, params, N: int, d: int=1):
        '''Sample from the initial distribution p(z_0)'''
        q0 = params.q0 + jnp.exp(params.log_sigma_0) * jax.random.normal(key, shape=(N,))
        option = jax.random.bernoulli(key, p=jax.nn.sigmoid(q0)).astype(int)
        w = super().sample_initial(key, params=None, N=N, d=d)
        return self.merge_latent(option, q0, w)

    def bias_correction(self, z):
        option, q, w = self.split_latent(z)
        if w.shape[-1] == 5:
            w_corrected = bias_correction(w)
        else:
            w_corrected = w
        return self.merge_latent(option, q, w_corrected)

if __name__=='__main__':
    # Seed for reproducibility
    seed = 1
    key = PRNGKey(seed)

    # array = jnp.array([0.2, 0.3, 0.4])
    # print(softmax_forward(array))

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