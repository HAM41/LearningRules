import jax
import jax.numpy as jnp
import numpy as np
import models
import scipy as sp

def learning_signal(Zs, X, Y, R, learning_rule):
    """
    Args:
        Zs: (N_particles, T, D+1) or (1, T, D+1,)
        X: (T, D)
        Y: (T,)
        R: (T,)
        learning_rule: str
    Returns:
        vec_out: (N_particles, T-1, D+1)
            Learning rule update for each particle at each time step
    """
    learning_rule = learning_rule.lower()
    print("Zs shape", Zs.shape)

    # Apply learning rule update, vmap over time axis
    if learning_rule=="reinforce":
        vec_out = jax.vmap(
            lambda z, x, y, r: models.reinforce(z, x, y, r)
            )(Zs.transpose(1,0,2), X, Y, R)
    elif learning_rule=="policy_gradient":
        vec_out = jax.vmap(
            lambda z, x, r: models.policy_gradient(z, x, r)
            )(Zs.transpose(1,0,2), X, R)
    else:
        raise ValueError(f"Learning rule {learning_rule} not implemented")
    
    if vec_out.ndim == 2:
        return vec_out[np.newaxis, :-1, ...]
    else:
        vec_out = vec_out.transpose(1,0,2)
        assert vec_out.shape == Zs.shape
        return vec_out[:, :-1, :]

def decompose_learning_noise(Zs, X, Y, R, learning_rule, alpha=0.1):
    """
    Args:
        Zs: (N_particles, T, D+1) or (T, D+1,)
        X: (T, D)
        Y: (T,)
        R: (T,)
        learning_rule: str
    Returns:
        learning_signals: (N_particles, T-1, D+1)
            Learning signal for each particle at each time step
        noise_signals: (N_particles, T-1, D+1)
            Noise signals for each particle at each time step
    """
    if Zs.ndim == 2:
        Zs = Zs[np.newaxis, ...]
    updates = Zs[:,1:,:] - Zs[:,:-1,:]
    learning_signals = learning_signal(Zs, X, Y, R, learning_rule)
    noise_signals = updates - alpha*learning_signals
    return learning_signals, noise_signals

def confidence_bounds(
        T, log_sigma, log_sigma_day=-1.0, session_indices=[], confidence_level=0.95
        ):
    ub_q = 1 - (1 - confidence_level) / 2
    lb_q = (1 - confidence_level) / 2

    scales = jnp.array([jnp.exp(log_sigma_day) if t in session_indices else jnp.exp(log_sigma) for t in range(T)])
    scales = jnp.sqrt(jnp.cumsum(jnp.square(scales)))

    def ub(t):
        return sp.stats.norm.ppf(ub_q, scale=scales[t])
    def lb(t):
        return sp.stats.norm.ppf(lb_q, scale=scales[t])
    
    ub_vals, lb_vals = [], []
    for t in range(T):
        ub_vals.append(ub(t))
        lb_vals.append(lb(t))
    
    # ub_vals = jax.vmap(ub)(jnp.arange(T))
    # lb_vals = jax.vmap(lb)(jnp.arange(T))
    return ub_vals, lb_vals
    

if __name__=="__main__":
    key = jax.random.PRNGKey(3)
    N_particles, T, D = 5000, 5000, 4

    # key, k1, k2, k3 = jax.random.split(key, 4)
    # X = jax.random.normal(k1, shape=(T, D))
    # Y = jax.random.bernoulli(k2, p=0.5, shape=(T,))
    # Zs = jax.random.normal(k3, shape=(N_particles, T, D+1))
    # R = models.reward(X[:,0], Y)

    # Vectorized over particles
    # learning_signal_per_particle = models.reinforce(Zs[:,0], X[0], Y[0], r=R[0])

    learning_rule = 'reinforce'
    log_sigma, log_sigma_day, alpha = [-3.023e+00, -4.213e-01,  2.609e-01]
    model = models.GLMLearn(seed=0, 
                            log_sigma=log_sigma, 
                            log_sigma_day=log_sigma_day, 
                            alpha=alpha, 
                            learning_rule=learning_rule)
    X, Y, true_Z, true_sample_noises = model.sample(T=T, key=key)
    R = models.reward(X, Y)

    import samplers
    (Zs, _), _ = samplers.bootstrap_filter(N_particles, X, Y, model, R=R, return_history=True, verbose=True)

    # Learning_signal
    # vec_out = np.zeros((N_particles, T, D+1))

    # # Vmap over time axis
    # vec_out = jax.vmap(
    #     lambda z, x, y, r: models.reinforce(z, x, y, r) # already vectorized over particles
    #     )(Zs.transpose(1,0,2), X, Y, R)
    # vec_out = vec_out.transpose(1,0,2)

    # # Sequential 
    # seq_out = np.zeros((N_particles, D+1))
    # for i in range(N_particles):
    #     seq_out[i] = models.reinforce(Zs[i,0], X[0], Y[0], r=R[0])
    
    # assert (seq_out == vec_out).all()
    # print(vec_out)

    # learning_signals = learning_signal(Zs, X, Y, R, learning_rule="reinforce")
    # assert (learning_signal_per_particle == learning_signals[:,0]).all()

    # # Differences
    # noise_signals = Zs[:,1:,:] - learning_signals

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(jnp.mean(noise_signals, axis=-1))
    # plt.savefig("learning_signal_noise.png")

    # alpha = 0.01
    # learning_rule = 'reinforce'


    learning_signals, noise_signals = decompose_learning_noise(Zs, X[:T], Y[:T], R[:T], 
                                                               learning_rule, alpha=alpha)
    true_learning_signals, true_noise_signals = decompose_learning_noise(true_Z, X[:T], Y[:T], R[:T], 
                                                               learning_rule, alpha=alpha)

    true_learning_cumsum = alpha * jnp.cumsum(true_learning_signals, axis=1)
    true_learning_mean, true_learning_std = jnp.mean(true_learning_cumsum, axis=0), jnp.std(true_learning_cumsum, axis=0)

    learning_cumsum = alpha * jnp.cumsum(learning_signals, axis=1)
    learning_mean, learning_std = jnp.mean(learning_cumsum, axis=0), jnp.std(learning_cumsum, axis=0)

    true_noise_cumsum = jnp.cumsum(true_noise_signals, axis=1)
    true_noise_mean, true_noise_std = jnp.mean(true_noise_cumsum, axis=0), jnp.std(true_noise_cumsum, axis=0)

    noise_cumsum = jnp.cumsum(noise_signals, axis=1)
    noise_mean, noise_std = jnp.mean(noise_cumsum, axis=0), jnp.std(noise_cumsum, axis=0)

    # def confidence_bounds(t, log_sigma):
    #     ub = sp.stats.norm.ppf(0.95, scale=jnp.sqrt( (t+1) * jnp.exp(log_sigma)**2))
    #     lb = sp.stats.norm.ppf(0.05, scale=jnp.sqrt( (t+1) * jnp.exp(log_sigma)**2))
    #     return lb, ub
    

    Z = jnp.mean(Zs, axis=0)

    import matplotlib.pyplot as plt

    labels = ['bias', 'stimIntensity']# 'contrastLeft', 'contrastRight', 'previousChoice', 'previousRewarded']
    colors = ['tab:blue', 'tab:orange']#, 'tab:green', 'tab:red', 'tab:purple']
    fig, axs = plt.subplots(figsize=[6,6], nrows=3, constrained_layout=True)
    print(jnp.cumsum(jnp.mean(learning_signals, axis=0), axis=0).shape)

    ub, lb = confidence_bounds(T-1, log_sigma)
    for i in range(2):
        axs[0].plot(np.arange(T-1)[::10], Z[::10,i] - Z[0,i], label=labels[i], c=colors[i])
        axs[0].plot(np.arange(T-1)[::10], true_Z[::10,i] - true_Z[0,i], ls='--', c=colors[i])#, label=labels[i])
        
        axs[1].plot(np.arange(T-1)[::10], learning_mean[::10,i], label=labels[i],  c=colors[i])
        axs[1].plot(np.arange(T-1)[::10], true_learning_mean[::10,i], label=labels[i], ls='--', c=colors[i])
        axs[1].fill_between(np.arange(T-1)[::10], learning_mean[::10,i]-learning_std[::10,i], learning_mean[::10,i]+learning_std[::10,i], alpha=0.3, color=colors[i])
        
        axs[2].plot(np.arange(T-1)[::10], noise_mean[::10,i], label=labels[i],  c=colors[i])
        axs[2].plot(np.arange(T-1)[::10], true_noise_mean[::10,i], label=labels[i], ls='--', c=colors[i])

        # axs[2].plot(np.arange(T-1)[::10],  jnp.cumsum(jnp.stack(true_sample_noises))[::10,i], c='k', zorder=-1)

    axs[2].plot(np.arange(T-1)[::10], ub[::10], c='k', zorder=-1)
    axs[2].plot(np.arange(T-1)[::10], lb[::10], c='k', zorder=-1)
    axs[2].set_xlabel('Trial number')

    axs[0].legend()
    axs[0].set_title('Posterior')
    axs[1].set_title('Learning signal')
    axs[2].set_title('Noise signal')
    axs[2].axhline(y=0)
    fig.suptitle('Decomposition into learning and noise signals for simulated data')
    plt.savefig('utils_learning_signal_noise.png', dpi=300)