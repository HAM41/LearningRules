import numpy as np
import scipy as sp

def sigmoid(x, w=1.0, b=0.0, g=1.0):
    return 1/(1+np.exp(-g*(w*x + b)))

def vec(x):
    return np.array([1,x])
def unvec(x):
    return x[1]

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
    for all x,y pairs in X, Y
    '''
    X = np.array(X)
    Y = np.array(Y).astype(int)
    
    # Broadcast scalars to arrays if needed
    if X.size == 1 and Y.size > 1:
        X = np.full(Y.shape, X)
    elif Y.size == 1 and X.size > 1:
        Y = np.full(X.shape, Y).astype(int)

    # Initialize an array for the rewards with the same shape as x and y
    r = np.zeros_like(X, dtype=float)

    # Calculate rewards element-wise
    mask_condition = (X < 0) & (Y == 0) | (X > 0) & (Y == 1)
    r[mask_condition] = 1.0

    return r

def cumulative_gaussian(x, sigma=1.0, mu=0.0):
    return 0.5*(1+sp.special.erf((x-mu)/(sigma*np.sqrt(2))))

def Phi(x):
    return cumulative_gaussian(x)

class QLearningModel():
    def __init__(self, sigma, alpha, softmax):
        self.sigma = sigma
        self.alpha = alpha

        self.V_init = 0.2       # Initialize values at 0.2

        # Use softmax for stochastic decisions 
        self.softmax = softmax

    def p_R(self, m):
        '''p_R(m): Belief state p(s > 0 | m)'''
        return Phi(m/self.sigma)
    
    def encode(self, X):
        N = X.shape
        return X + self.sigma * np.random.randn(*N)
    
    def transition_point(self, V):
        V_L, V_R = V
        return self.sigma * sp.special.erfinv(np.divide(V_L - V_R, V_R + V_L))
    
    def decision(self, m, V):
        if self.softmax:
            V_L, V_R = V
            p_R = self.p_R(m)

            # Compute Q values for each state
            Qs = np.stack([np.multiply(1-p_R, V_L), np.multiply(p_R, V_R)])
            
            p = sp.special.softmax(Qs, axis=0)
            if p.ndim > 1:
                Y = np.array([np.random.binomial(1, p=_p[1]) for _p in p]).astype(int)
            else:
                Y = np.random.binomial(1, p=p[1])
        else:
            a = self.transition_point(V)
            Y = np.array(m > a).astype(int)
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
            V_R = V_R + self.alpha * (reward(x,y) - np.multiply(p_R, V_R))
        else:
            V_L = V_L + self.alpha * (reward(x,y) - np.multiply(1-p_R, V_L))
        return np.stack([V_L, V_R])

    def emission_likelihood(self, y, m, V):
        r'''p(y | x_hat, V)'''
        V_L, V_R = V
        p_R = self.p_R(m)
        y = np.array(y).astype(int)

        # Compute Q values for each state
        Qs = np.stack([np.multiply(1-p_R, V_L), np.multiply(p_R, V_R)])

        # Compute decision likelihood p(y|Qs)
        if self.softmax:
            p = sp.special.softmax(Qs, axis=0)[y]
            return p
        else:
            return np.array(y==np.argmax(Qs, axis=0), dtype=float)
        
    def forward(self, t, N_samples, prev_latent, X_prev, Y_prev, X):
        '''
        Treating the values V and the percept m as the latents
        Return N_samples from one forward step 
            p(V_t, m_t | V_{t-1}, m_{t-1}, x_t) = p(m_t | x_t)p(V_t | V_{t-1}, m_{t-1}, x_t)
        '''
        if t == 0:
            V = self.V_init * np.ones((2, N_samples))
        else:
            V_prev, m_prev = prev_latent
            V = self.update_values(V_prev, X_prev, Y_prev, m_prev)

        if X.size != N_samples:
            m = np.array([self.encode(X) for _ in range(N_samples)])
        else:
            m = self.encode(X)
        return V, m
    
    def simulate(self, T):
        # Generate stimulus uniformly from range
        x_range = np.linspace(-1,1,12)
        X = np.random.choice(x_range, size=T, replace=True)

        # Encode percept and define initial values
        m = self.encode(X)
        V = self.V_init * np.ones(2)

        # Generate decisions and values sequentially
        Y, Vs = [], []
        for t in range(T):
            y = self.decision(m[t], V)

            Vs.append(V)
            Y.append(y)

            V = self.update_values(V, x=X[t], y=Y[t], m=m[t])
        return X, Y, m, Vs
    
class PolicyGradientGLM():
    def __init__(self, alpha):
        self.alpha = alpha 

    def emission_likelihood(self, w, x, y=1):
        '''p(y | w, x), default p(y=1 | w, x)'''
        vx = vec(x)
        LM = np.dot(w, vx)
        p = sigmoid(sign(y) * LM)
        return p

    def decision(self, w, x):
        p_R = self.emission_likelihood(y=1, w, x)
        y = np.random.binomial(1, p=p_R)
        return y

    def policy_gradient_vectorized(self, w, x):
        p_R = self.emission_likelihood(w, x, y=1)
        p_L = 1 - p_R 

        x = unvec(vx)
        effective_reward = np.sum([
            eps(y) * reward(x, y) for y in [0,1]
        ])
        return effective_reward * p_R * p_L * np.array(vx)


if __name__=='__main__':
    true_model = QLearningModel(sigma=0.3, alpha=0.5, softmax=True)
    X, Y, m, Vs = true_model.simulate(10)
