import numpy as np
import matplotlib.pyplot as plt
from models import QLearningModel, GLMLearn
import scipy as sp
from tqdm import tqdm

import jax
import jax.numpy as jnp

def bootstrap_filter(N: int, X, Y, model, seed=0, return_history=True, R=None):
    '''
    Bootstrap Filter / Particle filtering algorithm from [1], along with likelihood evaluation,
    specifically tailored for models regression models from X to Y.
    [1]: An Introduction to Sequential Monte Carlo. A. Doucet, N. de Freitas, N. Gordon.
    
    Args:
        N: int. Number of particles
        X: array, (T,M,). Inputs
        Y: array, (T,). Data/emissions. 
        model: needs to implement some form of forward method, and some for of emission likelihood
    Returns:
        z_history: List (T,), each element containing N samples, 
            Equally weighted latent samples to evalute integrals with MC methods
        loglik: scalar,
            Estimate of the marginal log-likelihood
    '''
    if X.ndim == 1:
        T = len(X)
        M = 1
    else:
        T, M = X.shape
    if R is None:
        R = [None for _ in range(T)]
    key = jax.random.PRNGKey(seed)

    z_history = []
    loglik_running_estimate = 0.
    
    for t in tqdm(range(0,T), desc='Bootstrap filter'):
        key, subkey = jax.random.split(key)

        # 1. Prediction step : tilde z_t ~ p(z_t | z_{t-1})
        #   Sample proposal N particles from previous N particles
        #   {tilde z_t^i, 1/N} is an approximation to p(z_t|y_{1:t-1})
        if t == 0:
            # if isinstance(model, GLMLearn):
            tilde_z_t = jax.random.normal(subkey, shape=(N, M+1,))
                # tilde_z_t = np.stack([model.w_0 for _ in range(N)])
            # else:
            #     tilde_z_t = jnp.stack([model.V_init for _ in range(N)])
        else:
            # if isinstance(model, QLearningModel):
                # tilde_z_t = model.forward(t, N, z_t, X[t-1], Y[t], X[t])
            # elif isinstance(model, GLMLearn):
            tilde_z_t = model.update_weights(z_t, x=X[t-1], y=Y[t-1], r=R[t-1])

        # 2. Evaluate importance weights p(y_t | xhat_t, V_t)
        #   {tilde z_t^i, tilde w^i} is an approximation to p(z_t|y_{1:t})
        # if isinstance(model, QLearningModel):
        #     tilde_V_t, tilde_m_t = tilde_z_t
        #     tilde_w_t = model.emission_likelihood(y=Y[t], m=tilde_m_t, V=tilde_V_t) 
        # elif isinstance(model, GLMLearn):
        tilde_w_t = model.emission_likelihood(y=Y[t], w=tilde_z_t, x=X[t]) 
        # else:
        #     raise NotImplementedError
        
        # 3. Resampling step: 

        # 3.1. Calculate importance weights
        # if jnp.sum(tilde_w_t) == 0.:
        #     choices = jnp.arange(N) # Nil likelihood, so no resampling
        # else:
        # Normalize importance weights
        # normalized_tilde_w_t = jax.nn.softmax(tilde_w_t) # Softmax
        normalized_tilde_w_t = tilde_w_t/np.sum(tilde_w_t) # L1 normalization


        # choices = np.random.choice(N, size=N, p=normalized_tilde_w_t)
        choices = jax.random.choice(subkey, N, shape=(N,), p=normalized_tilde_w_t)

        # 3.2. Resample with replacement N particles according the importance weights
        # if t==0:
        # if isinstance(model, QLearningModel):
        #     V_t = np.array([list(tilde_V_t[:,c]) for c in choices]).T
        #     m_t = np.array([tilde_m_t[c] for c in choices])
        #     z_t = V_t, m_t
        # elif isinstance(model, GLMLearn):
        #     z_t = jnp.array([tilde_z_t[c] for c in choices])
        # else:
        #     raise NotImplementedError
        #     z_history = jnp.array([z_t])
        # else:
        #     # z_history.append(tilde_z_t)
        #     z_history = jnp.concatenate((z_history, z_t[jnp.newaxis,:]), axis=0)
        #     if isinstance(model, GLMLearn):
        #         temp = []
        #         for c in choices:
        #             # Select c'th trajectory
        #             # print([z_n.shape for z_n in z_history])
        #             _traj = np.array([z_n[c,:] for z_n in z_history]) # Length t
        #             # print(len(_traj))
        #             temp.append(_traj)
        #         z_history = jnp.transpose(jnp.stack(temp), (1,0,2))
        #         # print(_z_history.shape)
        #         # print([z_n.shape for z_n in z_history])
        #     else:
        #         raise NotImplementedError
        #         z_history of shape (N, t+1, latent_dim)
        _, latent_dim = tilde_z_t.shape

        if return_history:
            if t==0:
                z_history = jnp.array([tilde_z_t[c] for c in choices])
                z_history = z_history.reshape(N, 1, latent_dim)
            else:
                # z_history of shape (N, t+1, latent_dim)
                z_history = jnp.concatenate((z_history, tilde_z_t[:, np.newaxis, :]), axis=1)
                z_history = jnp.array([z_history[c] for c in choices])
            z_t = z_history[:,-1,:]
        else:
            z_t = jnp.array([tilde_z_t[c] for c in choices])
            # z_history.append(z_t)
        
        # lik_estimate = np.mean([emission_likelihood(y=Y[t], x_hat=_xhat, V=_V, sigma=sigma, softmax=softmax) for _xhat, _V in zip(xhat_t, V_t)])
        lik_estimate = jnp.mean(tilde_w_t)
        # if jnp.linalg.norm(lik_estimate) == 0.:
        #     loglik_running_estimate = np.nan
        #     break
        # else:
        loglik_running_estimate += jnp.log(lik_estimate)

        # # Check for NaN or Inf
        # if jnp.isnan(normalized_tilde_w_t).any() or jnp.isinf(normalized_tilde_w_t).any():
        #     raise ValueError("Normalized importance weights contain NaN or Inf.")

        

    # if not return_history:
    #     z_history = jnp.stack(z_history).transpose(1,0,2)

    return z_history, loglik_running_estimate

if __name__=='__main__':
    true_alpha = 0.05
    true_sigma = 0.01
    # true_model = QLearningModel(sigma=true_sigma, alpha=true_alpha, softmax=True)
    true_model = GLMLearn(sigma_w=true_sigma, alpha=true_alpha)

    T = 200
    N = 500
    N_runs = 10

    alpha_range = np.exp(np.linspace(-5,0,20))
    sigma_range = np.exp(np.linspace(-3,1,10))

    vals = []
    for _ in range(N_runs):
        X, Y, _ = true_model.simulate(T)
        _vals = []
        for alpha in alpha_range:
            # model = QLearningModel(sigma=true_sigma, alpha=alpha, softmax=True)
            model = GLMLearn(sigma_w=true_sigma, alpha=alpha)

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

    # # -------------

    # true_alpha = 0.1
    # true_sigma = 0.1

    # true_model = PolicyGradientGLM(sigma_w=true_sigma, alpha=true_alpha)
    # X, Y, _ = true_model.simulate(T)
    # vals = []
    # for alpha in alpha_range:
    #     model = PolicyGradientGLM(sigma_w=true_sigma, alpha=alpha)
    #     _, log_lik = bootstrap_filter(N, inputs=X, data=Y, model=model)
    #     vals.append(log_lik)

    # fig, ax = plt.subplots()
    # ax.plot(alpha_range, vals)
    # ax.axvline(x=true_alpha, c='k', label=r'True $\alpha$')
    # ax.set_xscale('log')
    # plt.savefig('test2.png')
    # plt.close()