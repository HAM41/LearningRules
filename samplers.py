import numpy as np
import matplotlib.pyplot as plt
from models import QLearningModel
import scipy as sp

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
        if t == 0:
            tilde_z_t = model.forward(t, N, z_t, None, Y[t], X[t])
        else:
            tilde_z_t = model.forward(t, N, z_t, X[t-1], Y[t], X[t])

        # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
        #   {tilde z_t^i, tilde w^i} is an approximation to p(z_t|y_{1:t})
        if isinstance(model, QLearningModel):
            tilde_V_t, tilde_m_t = tilde_z_t
            tilde_w_t = model.emission_likelihood(y=Y[t], m=tilde_m_t, V=tilde_V_t) 
        else:
            raise NotImplementedError
        
        # Normalize
        if np.sum(tilde_w_t) == 0.:
            choices = np.arange(N)
        else:
            normalized_tilde_w_t = sp.special.softmax(tilde_w_t)
            # normalized_tilde_w_t = tilde_w_t/np.sum(tilde_w_t)
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
        else:
            raise NotImplementedError

        # print(np.mean([emission_likelihood(y=Y[t], x_hat=_xhat, V=_V, sigma=sigma, softmax=softmax) for _xhat, _V in zip(xhat_t, V_t)]))

        # lik_estimate = np.mean([emission_likelihood(y=Y[t], x_hat=_xhat, V=_V, sigma=sigma, softmax=softmax) for _xhat, _V in zip(xhat_t, V_t)])
        lik_estimate = np.mean(tilde_w_t)
        loglik_running_estimate += np.log(lik_estimate)

        z_history.append(z_t)

    return z_history, loglik_running_estimate

if __name__=='__main__':
    true_alpha = 0.05
    true_sigma = 0.3
    true_model = QLearningModel(sigma=true_sigma, alpha=true_alpha, softmax=True)
    
    T = 200
    N = 500
    N_runs = 10
    
    alpha_range = np.exp(np.linspace(-5,0,20))
    sigma_range = np.exp(np.linspace(-3,1,10))

    vals = []
    for _ in range(N_runs):
        X, Y, _, _ = true_model.simulate(T)
        _vals = []
        for alpha in alpha_range:
            model = QLearningModel(sigma=true_sigma, alpha=alpha, softmax=True)

            _, log_liks = bootstrap_filter(N, inputs=X, data=Y, model=model)
            _vals.append(log_liks)
            print(round(alpha,3), log_liks)
        vals.append(_vals)

    vals = np.array(vals)

    fig, ax = plt.subplots()
    ax.plot(alpha_range, np.sum(vals, axis=0), c='tab:grey')
    ax.axvline(x=true_alpha, c='k', label=r'True $\alpha$')
    # ax.axvline(x=true_sigma, c='k', label=r'True $\sigma$')
    # ax.plot(alpha_range, vals.T, c='tab:grey')
    # ax.fill_between(alpha_range, mean-std, mean+std, alpha=0.3, color='tab:blue')
    ax.legend()
    ax.set_xlabel(r'Learning rate $\alpha$')
    # ax.set_xlabel(r'Percept noise scale $\sigma$')
    ax.set_ylabel('Log-likelihood')
    # ax.set_title(f'SMC marginal likelihood estimates. $\sigma=${true_sigma:1.2f}')
    ax.set_title(r'SMC marginal likelihood estimates. $\alpha=$'+f'{true_alpha:1.2f}')
    ax.set_xscale('log')
    plt.savefig('test.png')