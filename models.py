import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp

# def sigmoid(x, w=1.0, b=0.0, g=1.0):
#     return 1/(1+np.exp(-g*(w*x + b)))
def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

def safe_sigmoid(X, threshold=80.):
    return jnp.where(X > threshold, jnp.ones_like(X), jnp.where(X < -threshold, jnp.zeros_like(X), sigmoid(X)))

def vec(x):
    x = jnp.asarray(x)
    if x.ndim == 0:
        return jnp.array([1,x])
    else:
        if x.ndim > 1:
            raise NotImplementedError
        return jnp.concatenate(([1.0],x))
    
def unvec(x):
    return x[1:]

def sign(y) -> float:
    y = float(y)
    assert y in [0., 1.], 'y must be 0 or 1'
    if y==0.:
        return -1.
    elif y==1:
        return 1.

def reward(X, Y) -> np.ndarray:
    '''
    Returns reward 
        r(x,y)= 1. if (x < 0 and y == 0) or (x > 0 and y==1)
                0. else
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
    r = jnp.zeros_like(X, dtype=float)

    # Calculate rewards element-wise
    mask_condition = (X < 0) & (Y == 0) | (X > 0) & (Y == 1)
    # r = r.at[mask_condition].set(1.0)
    r = jnp.where(mask_condition, 1.0, r)

    return r

def effective_reward(X):
    r'''
    Returns effective reward R(x) = \sum_y r(x,y) sign(y), for all x in X. 
    Output `RX` is of same shape as `X`.
    '''
    RX = jnp.sum(jnp.array([sign(y) * reward(X, y) for y in [0,1]]))
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
                Y = np.array([np.random.binomial(1, p=_p[1]) for _p in p]).astype(int) #! jax
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
    
    def simulate(self, T):
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
                 alpha: float, sigma_w: np.ndarray, w_init: np.ndarray=np.array([0.0, 1.0]), 
                 learning_rule='policy_gradient'
                 ) -> None:
        self.alpha = alpha
        self.sigma_w = sigma_w
        self.w_0 = w_init # np.array([0.0, 1.0]) # bias = 0, weight = 1
        self.learning_rule = learning_rule

    def emission_likelihood(self, w, x, y=1):
        '''p(y | w, x), default p(y=1 | w, x)'''
        vx = vec(x)
        LM = jnp.dot(w, vx)
        p = safe_sigmoid(sign(y) * LM)
        return p

    def decision(self, w, x):
        p_R = self.emission_likelihood(w, x, y=1.0)
        y = jax.random.bernoulli(key, p_R).astype(int)
        return y

    def policy_gradient(self, w, x):
        p_R = self.emission_likelihood(w, x, y=1)
        return effective_reward(x) * jnp.outer(jnp.multiply(p_R, 1 - p_R), vec(x)).squeeze()
    
    def reinforce(self, w, x, y):
        p = self.emission_likelihood(w, x, y)
        return reward(x,y) * jnp.outer(1-p, vec(x)).squeeze()

    def update_weights(self, w, x, y=None):
        if self.learning_rule == 'reinforce':
            learning_signal = self.alpha * self.reinforce(w, x, y)
        else: # use Policy gradient
            learning_signal = self.alpha * self.policy_gradient(w, x)
        update_noise = jax.random.normal(key, shape=w.shape)
        update_noise = jnp.multiply(self.sigma_w, update_noise)
        return w + learning_signal + update_noise
    
    def dynamics_loglikelihood(self, z_next, z_prev, inputs, data):
        '''log p(z_t | z_{t-1})
        In our case, the latents z are the GLM weights w
        '''
        if self.learning_rule == 'reinforce':
            learning_signal = self.alpha * self.reinforce(z_prev, inputs, data)
        else: # use Policy gradient
            learning_signal =self.alpha * self.policy_gradient(z_prev, inputs)
        mean = z_prev + learning_signal
        cov = jnp.multiply(self.sigma_w, jnp.eye(N))
        N = mean.shape[0]
        log_lik = lambda z: jsp.stats.multivariate_normal.pdf(z, mean=mean, cov=cov)
        return log_lik(z_next)
    
    def complete_data_loglikelihood(self, data, Z, inputs):
        T = len(data)
        #! Handle initial

        L_CD = 0.
        for t in range(1,T):
            log_pzz = self.dynamics_loglikelihood(Z[t], Z[t-1], inputs[t-1], data[t-1])
            log_pyz = jnp.log(self.emission_likelihood(Z[t], inputs[t], data[t]))
            L_CD += log_pzz + log_pyz
        return L_CD
    
    def simulate(self, T):
        # Generate stimulus uniformly from range
        x_range = jnp.linspace(-1,1,12)
        X = jax.random.choice(key, x_range, shape=(T,), replace=True)

        # Encode percept and define initial values
        w = self.w_0

        # Generate decisions and weights sequentially
        Y, Ws = [], []
        for t in range(T):
            # Sample
            y = self.decision(w, X[t])
            Y.append(y)
            Ws.append(w)

            # Update
            w = self.update_weights(w, x=X[t], y=Y[t])
        return X, jnp.stack(Y), jnp.stack(Ws)

if __name__=='__main__':
    # Seed for reproducibility
    key = jax.random.PRNGKey(0)

    # true_model = QLearningModel(sigma=0.3, alpha=0.5, softmax=False)
    # X, Y, m, Vs = true_model.simulate(10)
    # print(X, Y, m, Vs)

    true_model = GLMLearn(sigma_w=0.1, alpha=1.0)
    X, Y, Ws = true_model.simulate(10)
    print(X, Y, Ws)

    t = 5