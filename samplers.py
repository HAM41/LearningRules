import numpy as np
from models import QLearningModel

def bootstrap_filter(N, inputs, data, model):
    '''
    returns N samples, equally weighted, to evaluate integrals
    '''
    X = inputs
    Y = data
    T = len(X)

    z_history = []

    loglik_running_estimate = 0.
    z_t = None

    for t in range(0,T):

        # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
        #   Sample proposal N particles from previous N particles
        #   {tilde z_t^i, 1/N} is an approximation to p(z_t|y_{1:t-1})
        tilde_z_t = model.forward(t, N, z_t, X[t], Y[t])

        # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
        #   {tilde z_t^i, tilde w^i} is an approximation to p(z_t|y_{1:t})
        if isinstance(model, QLearningModel):
            tilde_V_t, tilde_m_t = tilde_z_t
            tilde_w_t = model.emission_likelihood(y=Y[t], m=tilde_m_t, V=tilde_V_t) 
        
        # Normalize
        # normalized_tilde_w_t = sp.special.softmax(tilde_w_t)
        normalized_tilde_w_t = tilde_w_t/np.sum(tilde_w_t)
        # if np.sum(tilde_w_t) == 0.:
        #     xhat_history.append(tilde_xhat_t)
        #     V_history.append(tilde_V_t)
        #     continue

        # 3. Resampling step: 
        #   Resample with replacement N particles according the importance weights
        choices = np.random.choice(N, size=N, p=normalized_tilde_w_t)
        if isinstance(model, QLearningModel):
            # print(tilde_m_t.shape)
            V_t = np.array([list(tilde_V_t[:,c]) for c in choices]).T
            m_t = np.array([tilde_m_t[c] for c in choices])
            z_t = V_t, m_t

        # print(np.mean([emission_likelihood(y=Y[t], x_hat=_xhat, V=_V, sigma=sigma, softmax=softmax) for _xhat, _V in zip(xhat_t, V_t)]))

        # lik_estimate = np.mean([emission_likelihood(y=Y[t], x_hat=_xhat, V=_V, sigma=sigma, softmax=softmax) for _xhat, _V in zip(xhat_t, V_t)])
        lik_estimate = np.mean(tilde_w_t)
        loglik_running_estimate += np.log(lik_estimate)

        z_history.append(z_t)

    return z_history, loglik_running_estimate

if __name__=='__main__':
    model = QLearningModel(sigma=0.3, alpha=0.5, softmax=True)
    T = 200
    N = 10

    X = np.random.randn(T)
    Y =  np.random.randint(0, 2, size=T)
    print(bootstrap_filter(N, inputs=X, data=Y, model=model))