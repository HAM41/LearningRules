import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import ibl
import samplers
import models

import sys
import jax 
import jax.numpy as jnp
import jax.scipy as jsp
# import os
# os.environ['JAX_PLATFORMS']='cpu'

def grid_search(X, Y, N_bootstrap=200):
    alpha_range = np.concatenate(([0.0], np.exp(np.linspace(-7,1,10))))
    sigma_range = np.exp(np.linspace(-8,1,10))

    vals = []
    max_value = -float('inf')
    for alpha in alpha_range:
        # _vals = []
        for sigma in sigma_range:
            model = models.GLMLearn(alpha, sigma)
            # model = models.QLearningModel(alpha, sigma, softmax=True)

            log_liks = []
            for batch_X, batch_Y in zip(X,Y):
                _, log_lik = samplers.bootstrap_filter(N_bootstrap, batch_X, batch_Y, model)
                _, log_lik2 = samplers.bootstrap_filter(N_bootstrap, batch_X, batch_Y, model)
                print(log_lik2 - log_lik)
                log_liks.append(log_lik)

            # Return sum of evidence estimates as evidence estimate of the dataset
            out = np.sum(log_liks)
            # _vals.append(out)
            if out > max_value:
                max_value = out
                argmin = (alpha, sigma)

            entry = {'alpha':round(alpha,3), 'sigma':round(sigma,3), 'evidence':out}
            print(entry)
            vals.append(entry)

    vals_df = pd.DataFrame(vals).pivot(index="sigma", columns="alpha", values="evidence")
    # print(vals_df)
    fig, ax = plt.subplots(constrained_layout=True)
    sns.heatmap(data=vals_df, ax=ax, cmap='magma')
    # ax.contourf(*np.meshgrid(alpha_range, sigma_range), vals_df.values)
    # ax.scatter(*argmin)
    # ax.plot(true_alpha, true_sigma, 'o')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()


    print(argmin)
    return 

def simplex(X, Y, N_bootstrap=200, B=1):
    '''
    Simplex method to maximize the evidence.
    Implemented for now for GLMLearn model. 
    Args:
        X: (B,T,)
        Y: (B,T,)
    '''

    # Define objective function
    def neg_evidence(log_theta):
        # Instantiate model for SMC
        print(np.exp(log_theta))
        model = models.GLMLearn(*np.exp(log_theta))
        # model = models.QLearningModel(*np.exp(log_theta))

        # Compute evidence estimates for each batch of (X,Y)
        log_liks = []
        for batch_X, batch_Y in zip(X,Y):

            # Average over batch
            _log_liks = []
            for _ in range(B):
                _, _log_lik = samplers.bootstrap_filter(N_bootstrap, batch_X, batch_Y, model)
                _log_liks.append(_log_lik)
            log_lik = np.mean(_log_liks)

            log_liks.append(log_lik)

        # Return sum of evidence estimates as evidence estimate of the dataset
        out = np.sum(log_liks)
        print(out)
        return -out
    
    # Define intial condition
    log_theta_0 = [-2., -2.] # alpha, sigma_w

    # Perform optimization
    result = sp.optimize.minimize(neg_evidence, log_theta_0, method='Powell')
    assert result.success
    return np.exp(result.x)

def grad_log_joint(model, Z, X, Y, theta):
    '''
    [w_init, log_alpha, log_sigma] = theta
    '''
    T = len(Y)
    grad_log_joint = jnp.zeros_like(theta)
    log_joint = 0.

    # Initial t=0 joint likelihood terms
    # Dynamics
    log_pz0 = lambda _theta: model.initial_loglikelihood(Z[0], w_init_mean=_theta[:2])
    log_pz0_val, log_pz0_grad = jax.value_and_grad(log_pz0)(theta)

    # Emissions
    log_pyz0_val = jnp.log(model.emission_likelihood(Z[0], X[0], Y[0]))
    grad_log_pyz0_val = jnp.zeros_like(grad_log_joint)
    
    log_joint += log_pz0_val + log_pyz0_val
    grad_log_joint += log_pz0_grad + grad_log_pyz0_val

    # Loop over time steps 
    for t in range(1,T):
        # Dynamics
        log_pzz = lambda _theta: model.dynamics_loglikelihood(
            Z[t], Z[t-1], X[t-1], Y[t-1], 
            alpha=jnp.exp(_theta[-2]), sigma_w=jnp.exp(_theta[-1])
            )
        log_pzz_val, log_pzz_grad = jax.value_and_grad(log_pzz)(theta)

        # Emissions, do not depend on hyper-parameters
        log_pyz_val = jnp.log(model.emission_likelihood(Z[t], X[t], Y[t]))
        grad_log_pyz_val = jnp.zeros_like(grad_log_joint)

        # Update value and gradient of log joint
        log_joint += log_pzz_val + log_pyz_val
        grad_log_joint += log_pzz_grad + grad_log_pyz_val

    return log_joint, grad_log_joint

def test_grad_loglik():
    # Generate simulated data
    true_alpha = 1.0
    true_sigma = 0.01
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    true_theta = jnp.array([true_model.w_init_mean[0], true_model.w_init_mean[1], true_alpha, true_sigma])

    T = 200
    for i in range(1):
        X, Y, Z_true = true_model.sample(T, key=jax.random.PRNGKey(i))
        _, true_ll = samplers.bootstrap_filter(1000, X, Y, true_model)
        val, _ = true_model.value_and_grad_joint(X, Y, Z_true, true_theta)
        print(true_theta, true_ll, val)

    theta_init = jnp.array([0.0, 1.0, np.exp(-1.0), np.exp(-1.0)])
    # model = models.GLMLearn(sigma_w=np.exp(theta_init[-1]), alpha=np.exp(theta_init[-2]), w_init_mean=theta_init[:2], seed=seed)
    # print(_l)

    theta = theta_init
    # model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    # SMC_Z_samples, _l = samplers.bootstrap_filter(100, X, Y, model)
    # values, gradients = jax.vmap(lambda Z: model.joint_loglikelihood(X, Y, Z, theta))(SMC_Z_samples)
    # print(values, gradients)
    # sys.exit()

    # theta_init = jnp.array([true_model.w_init_mean[0], true_model.w_init_mean[1], np.log(true_alpha), np.log(true_sigma)])
    learning_rate = 1e-05
    for k in range(1,100):
        # learning_rate = np.power(k, -2/3)
        model = models.GLMLearn(sigma_w=theta[-1], alpha=theta[-2], w_init_mean=theta[:2], seed=seed)
        SMC_Z_samples, _l = samplers.bootstrap_filter(1000, X, Y, model, return_history=False)
        # print(_l)
        # for Z in SMC_Z_samples:ç
        #     val, grad = grad_log_joint(model, Z, X, Y, theta)
        #     print(val, grad)s
        # Apply grad_log_joint to all elements of SMC_Z_samples in parallel
        # values, gradients = jax.vmap(lambda Z: grad_log_joint(model, Z, X, Y, theta))(SMC_Z_samples)
        values, gradients = jax.vmap(lambda Z: model.value_and_grad_joint(X, Y, Z, theta))(SMC_Z_samples)

        value = jnp.mean(values)
        grad = jnp.mean(gradients, axis=0)

        # grad = grad.at[-1].set(grad[-1] * theta[-1])
        # grad = grad.at[-2].set(grad[-2] * theta[-2])
        # print(gradients)
        # # Extract values and gradients from the results
        # values = jnp.array([result[0] for result in results])
        # gradients = jnp.array([result[1] for result in results])
        # print(values, gradients)

        # Update theta
        theta += learning_rate * grad
        print(_l, value, theta)
        # print(values, theta, gradients)
    
    # print(grad_log_joint(model, Z, X, Y, theta=np.array([0.0, 1.0, -1.0, -1.0])))#, theta=jnp.array([0.01, 0.01, 0.01, 0.01])))

def compare_models():
    true_alpha = 0.05
    true_sigma = 0.01
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    true_theta = true_model.w_init_mean, np.log(true_alpha), np.log(true_sigma)
    print(true_theta)

    T = 200
    X, Y, _ = true_model.sample(T)

    # Model 1: GLM
    GLM_theta_init = jnp.array([0.0, 1.0, -3.0, -3.0])
    model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    SMC_Z_samples, _l = samplers.bootstrap_filter(100, X, Y, model)
    # print(_l)

    GLM_theta = theta_init
    GLM
    learning_rate = 1e-03
    for k in range(1,100):
        model = models.GLMLearn(sigma_w=jnp.exp(theta[-1]), alpha=jnp.exp(theta[-2]), w_init_mean=theta[:2], seed=seed)
        values, gradients = jax.vmap(lambda Z: model.value_and_grad_joint(X, Y, Z, theta))(SMC_Z_samples)

        value = jnp.mean(values)
        grad = jnp.mean(gradients, axis=0)
        theta += learning_rate * grad
        print(value, theta)


def test_grad_SMC():
    true_alpha = 0.05
    true_sigma = 0.02
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    true_theta = true_model.w_init_mean, np.log(true_alpha), np.log(true_sigma)
    print(true_theta)

    T = 500
    X, Y, _ = true_model.sample(T)

    def loglik(log_theta):
        model = models.GLMLearn(sigma_w=jnp.exp(log_theta[0]), alpha=jnp.exp(log_theta[1]), seed=seed)
        _, value = samplers.bootstrap_filter(100, X, Y, model)
        return value
    
    # print(jsp.optimize.minimize(loglik, jnp.array([-1., -1.]), method='Powell'))

    # # Define the function that returns the gradient at a given point
    # def gradient_func(log_theta):
    #     # Compute the gradient of the log-likelihood function at the given point
    #     value, gradient = jax.value_and_grad(jax.grad_and_value(loglik)(log_theta))
    #     # Scale the gradient by the exponential of the log_theta values
    #     gradient = jnp.divide(gradient, jnp.exp(log_theta))
    #     return gradient

    # Define the range of log_theta values to plot
    log_sigmas = np.linspace(-5, 0, 10)
    log_alphas = np.linspace(-5, 0, 10)
    # sigmas, alphas = np.meshgrid(_sigmas, _alphas)
    # log_theta_values = np.stack([sigmas, alphas], axis=-1)

    entries =[]
    for log_sigma in log_sigmas:
        for log_alpha in log_alphas:
            value, gradient = jax.value_and_grad(loglik)([log_sigma, log_alpha])
            entry = {'log_sigma': log_sigma, 'log_alpha': log_alpha, 'value': float(value), 'gradient_x': float(gradient[0]), 'gradient_y': float(gradient[1])}
            entries.append(entry)

    df = pd.DataFrame(entries)
    df_heatmap = df.pivot(columns="log_sigma", index="log_alpha", values="value")
    print(df)

    fig, axs = plt.subplots(figsize=[8,4], ncols=2, constrained_layout=True)
    sns.heatmap(df_heatmap, ax=axs[0])
    axs[1].quiver(df['log_sigma'].values, df['log_alpha'].values, df['gradient_x'].values, df['gradient_y'].values)
    plt.savefig('figures/grad_field_pd.png', dpi=300)
    plt.close()

    # # Compute the gradient at each point in the range
    # gradients = np.stack([gradient_func(log_theta) for log_theta in log_theta_values.reshape(-1, 2)], axis=0)
    # values = np.stack([loglik(log_theta) for log_theta in log_theta_values.reshape(-1, 2)], axis=0)
    # gradients = gradients.reshape(len(_sigmas), len(_alphas), 2)
    # values = values.reshape(len(_alphas), len(_sigmas))

    # # Plot the gradient field
    # fig, axs = plt.subplots(figsize=[8,4], ncols=2, constrained_layout=True)
    # axs[0].contourf(sigmas, alphas, values, levels=20)
    # axs[1].quiver(sigmas, alphas, gradients[..., 0], gradients[..., 1])
    # for ax in axs:
    #     ax.set_xlabel('log(sigma_w)')
    #     ax.set_ylabel('log(alpha)')
    # plt.savefig('figures/grad_field.png', dpi=300)
    # plt.close()

    # log_theta_init = jnp.array([-1., -1.])
    # theta = log_theta_init
    # for _ in range(10):
    #     gradient = jnp.asarray(jax.grad(loglik)(theta))
    #     print(gradient)
    #     gradient = jnp.multiply(jnp.exp(theta), gradient)
    #     theta = theta + 0.01 * gradient
    #     print(theta)
    # # print(jax.grad(loglik)([-1., -1.]))
    # print(np.multiply(np.exp([-1., -1.]), jax.grad(loglik)([-1., -1.])))
    
def test_simplex():
    # Generate dummy data
    true_alpha = 0.05
    true_sigma = 0.01
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha)

    T = 200
    N_runs = 10

    Xs, Ys = [], []
    for _ in range(N_runs):
        _X, _Y, _ = true_model.sample(T)
        Xs.append(_X.reshape(-1,1))
        Ys.append(_Y)

    X = np.stack(Xs)
    Y = np.stack(Ys)
    print(X.shape)

    # Evaluate 
    N = 1000
    # X, Y = ibl.get_behavioral_data(subject='ibl_witten_02')
    # grid_search([X[0][:270]], [Y[0][:270]], N_bootstrap=N)
    # print(simplex([X[0][-1000:]], [Y[0][-1000:]], N_bootstrap=N))

    print(simplex(X,Y,N_bootstrap=N,B=5))
    
    # [0.00762551 0.00136533] , -3510.52603770583

def test_Laplace():
    # Generate simulated data
    true_log_alpha = np.log(0.1)
    true_log_sigma = np.log(0.01)
    true_model = models.GLMLearn(sigma_w=np.exp(true_log_sigma), alpha=np.exp(true_log_alpha), seed=seed)
    true_theta = jnp.array([true_model.w_init_mean[0], true_model.w_init_mean[1], true_log_alpha, true_log_sigma])
    print('True: ', true_theta)

    factor = jnp.array([0.1, 0.1, 1.0, 1.0])
    theta_init = true_theta + jnp.multiply(factor, jnp.abs(jax.random.normal(key, shape=true_theta.shape))) # perturb true theta
    # theta_init = jnp.array([0.0, 1.0, -3.0, -3.0])
    print('Theta init: ', theta_init)
    
    T = 200
    N_runs = 10
    X = np.zeros((N_runs, T))
    Y = np.zeros((N_runs, T))
    for n in range(10):
        x, y, _ = true_model.sample(T)
        X[n] = x
        Y[n] = y
    
    _, _, Z_init = true_model.sample(T, key=jax.random.PRNGKey(2))

    theta = theta_init
    Z_estimate = Z_init
    for _ in range(10):
        # Find MAP
        def objective(Z):
            def log_joint(x,y):
                return true_model.log_joint(x, y, Z.reshape(T,2), theta)
            
            evidence_per_trial = jax.vmap(log_joint, (0, 0))(X, Y)
            evidence = jnp.sum(evidence_per_trial)
            return -evidence

        # objective = lambda Z: -true_model.log_joint(X, Y, Z.reshape(T,2), theta)
        res = sp.optimize.minimize(objective, Z_estimate.flatten(), jac=jax.grad(objective), hess=jax.hessian(objective), method='Newton-CG', options={'disp':True, 'xtol':0.1})
        Z_estimate = res.x
        # print(np.linalg.norm(Z_estimate - Z_true))

        # Use MAP to update parameters
        # def objective(_theta):
        #     out = 0.
        #     for x, y in zip(X,Y):
        #         out = out + true_model.log_joint(x, y, Z_estimate, theta)
        #     return out

        Hinv = jnp.linalg.inv(jax.hessian(objective)(Z_estimate))

        N_MC_samples = 1
        Z_samples = jax.random.multivariate_normal(key, Z_estimate, Hinv, shape=(N_MC_samples,))
        Z_samples = Z_samples.reshape(N_MC_samples, T, 2)
        # Z_estimate = Z_estimate.reshape(T,2)

        def objective(_theta):
            value = 0.
            for Z in Z_samples:
                def log_joint(x,y):
                    return true_model.log_joint(x, y, Z, _theta)
                
                evidence_per_trial = jax.vmap(log_joint, (0, 0))(X, Y)
                evidence = jnp.sum(evidence_per_trial)
                value += evidence
            value /= N_MC_samples
            return value
        # objective = lambda _theta: true_model.log_joint(X, Y, Z_estimate, _theta)
        
        for _ in range(10):
            theta = theta + 1e-05 * jax.grad(objective)(theta)
            print(theta)
        
        print('---', objective(theta))
    # res = sp.optimize.minimize(objective, theta_init, jac=jax.grad(objective), options={'disp':True})#, hess=jax.hessian(objective), method='Newton-CG', options={'disp':True, 'xtol':0.001})
    # print(res)
    # print(res.x)

    # for _ in range(100):
    #     Z_init = Z_init + 1e-04 * jax.grad(objective)(Z_init)
    #     print(objective(Z_init))
    # print(jax.value_and_grad(objective)(Z_init))


if __name__=='__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    test_Laplace()
    # test_grad_loglik()
    # test_grad_SMC()

    # true_alpha = 0.05
    # true_sigma = 0.01
    # T = 100
    # true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha)
    # X, Y, _ = true_model.sample(T)

    # z_hist, _l = samplers.bootstrap_filter(10, X, Y, true_model)
    # # print(jnp.stack(z_hist).shape)
    # print(jnp.stack(z_hist)[:,0,:].shape)
    # # SMC_Z_samples = jnp.array([jnp.stack(z_hist)[:,i,:] for i in range(10)])
    # # sys.exit()
    # # SMC_Z_samples = jnp.stack(z_hist)
    # SMC_Z_samples = jnp.transpose(jnp.stack(z_hist), (1,0,2))
    # print(SMC_Z_samples.shape)
    # values, gradients = jax.vmap(
    #     lambda Z: true_model.joint_loglikelihood(
    #         X, Y, Z, jnp.array([true_model.w_init_mean[0], true_model.w_init_mean[1], np.log(true_alpha), np.log(true_sigma)])
    #         ))(SMC_Z_samples)
    
    # print(_l, values)