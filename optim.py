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
    true_alpha = 0.05
    true_sigma = 0.01
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    true_theta = true_model.w_init_mean, np.log(true_alpha), np.log(true_sigma)
    print(true_theta)

    T = 200
    X, Y, _ = true_model.simulate(T)


    theta_init = jnp.array([0.0, 1.0, -2.0, -2.0])
    model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    # model = models.GLMLearn(sigma_w=np.exp(theta_init[-1]), alpha=np.exp(theta_init[-2]), w_init_mean=theta_init[:2], seed=seed)
    z_hist, _l = samplers.bootstrap_filter(10, X, Y, model)
    SMC_Z_samples = jnp.transpose(jnp.stack(z_hist), (1,0,2))

    # theta_init = jnp.array([true_model.w_init_mean[0], true_model.w_init_mean[1], np.log(true_alpha), np.log(true_sigma)])
    theta = theta_init
    learning_rate = 1e-02
    for k in range(1,100):
        # learning_rate = np.power(k, -2/3)
        # model = models.GLMLearn(sigma_w=jnp.exp(theta[-1]), alpha=jnp.exp(theta[-2]), w_init_mean=theta[:2], seed=seed)
        # print(_l)
        # for Z in SMC_Z_samples:
        #     val, grad = grad_log_joint(model, Z, X, Y, theta)
        #     print(val, grad)
        # Apply grad_log_joint to all elements of SMC_Z_samples in parallel
        # values, gradients = jax.vmap(lambda Z: grad_log_joint(model, Z, X, Y, theta))(SMC_Z_samples)
        values, gradients = jax.vmap(lambda Z: model.joint_loglikelihood(X, Y, Z, theta))(SMC_Z_samples)

        value = jnp.mean(values)
        grad = jnp.mean(gradients, axis=0)
        # print(gradients)
        # # Extract values and gradients from the results
        # values = jnp.array([result[0] for result in results])
        # gradients = jnp.array([result[1] for result in results])
        # print(values, gradients)
        theta += learning_rate * grad
        print(value, theta)
        # print(values, theta, gradients)
    
    # print(grad_log_joint(model, Z, X, Y, theta=np.array([0.0, 1.0, -1.0, -1.0])))#, theta=jnp.array([0.01, 0.01, 0.01, 0.01])))

def test_grad_SMC():
    true_alpha = 0.05
    true_sigma = 0.01
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha, seed=seed)
    true_theta = true_model.w_init_mean, np.log(true_alpha), np.log(true_sigma)
    print(true_theta)

    T = 100
    X, Y, _ = true_model.simulate(T)

    def loglik(log_theta):
        model = models.GLMLearn(sigma_w=jnp.exp(log_theta[0]), alpha=jnp.exp(log_theta[1]), seed=seed)
        _, value = samplers.bootstrap_filter(1000, X, Y, model)
        return value
    
    # print(jsp.optimize.minimize(loglik, jnp.array([-1., -1.]), method='Powell'))
    
    log_theta_init = jnp.array([-1., -1.])
    theta = log_theta_init
    for _ in range(10):
        gradient = jnp.asarray(jax.grad(loglik)(theta))
        gradient = jnp.multiply(jnp.exp(theta), gradient)
        theta = theta + 0.01 * gradient
        print(theta)
    # print(jax.grad(loglik)([-1., -1.]))
    print(np.multiply(np.exp([-1., -1.]), jax.grad(loglik)([-1., -1.])))
    
def test_simplex():
    # Generate dummy data
    true_alpha = 0.05
    true_sigma = 0.01
    true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha)

    T = 200
    N_runs = 10

    Xs, Ys = [], []
    for _ in range(N_runs):
        _X, _Y, _ = true_model.simulate(T)
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

if __name__=='__main__':
    seed = 0
    key = jax.random.PRNGKey(seed)
    # test_grad_loglik()
    test_grad_SMC()

    # true_alpha = 0.05
    # true_sigma = 0.01
    # T = 100
    # true_model = models.GLMLearn(sigma_w=true_sigma, alpha=true_alpha)
    # X, Y, _ = true_model.simulate(T)

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