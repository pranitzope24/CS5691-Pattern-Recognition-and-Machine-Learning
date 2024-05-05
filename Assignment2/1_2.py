import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('A2Q1.csv', delimiter=',')
K = 4
n_iter = 10

def expectation(mu_k, sigma_k_sq, pi_k, X):
    sigma_k=np.sqrt(sigma_k_sq)
    numerator= np.reciprocal((np.sqrt(2 * np.pi)*sigma_k))* np.exp(-0.5*(((X-mu_k)/sigma_k))**2) * pi_k
    denominator=np.sum(numerator, axis=1).reshape(-1, 1)
    lambda_ik = numerator/denominator
    return lambda_ik, numerator

def maximization(lambda_ik, X):
    num_datapts = X.shape[0]
    pi_k = np.sum(lambda_ik, axis=0).reshape(1,-1)/ num_datapts
    mu_k = (np.dot(X.T,lambda_ik)/np.sum(lambda_ik, axis=0).reshape(1,-1)).reshape(1,-1)
    sigma_k_sq = np.sum((((X-mu_k)**2)*lambda_ik),axis=0)/(np.sum(lambda_ik, axis=0).reshape(1,-1)).reshape(1,-1)
    return pi_k,mu_k,sigma_k_sq

def exclude_nans(mu_k, sigma_k_sq, pi_k):  
    if np.any(np.isnan(mu_k)):
        mu_k[np.where(np.isnan(mu_k))] = float(np.random.uniform(0.05, 1, 1))
    if np.any(mu_k < 1e-3):
        mu_k[np.where(mu_k < 1e-3)] = float(np.random.uniform(0.05, 1, 1))
    if np.any(np.isnan(sigma_k_sq)):
        sigma_k_sq[np.where(np.isnan(sigma_k_sq))] = float(np.random.uniform(0.05, 1, 1))
    if np.any(sigma_k_sq < 1e-3):
        sigma_k_sq[np.where(sigma_k_sq < 1e-3)] = float(np.random.uniform(0.05, 1, 1))
    if np.any(np.isnan(pi_k)):
        pi_k[np.where(np.isnan(pi_k))] = float(np.random.uniform(0.05, 0.5, 1))
    if np.any(pi_k < 1e-3):
        pi_k[np.where(pi_k < 1e-3)] = float(np.random.uniform(0.05, 0.5, 1))
    return mu_k, sigma_k_sq, pi_k

def EM_Algorithm(n_iter,X,rand_init=100):
    X = X.reshape(-1, 1)
    n_iter_LL = []
    n_iter_mu_k = []
    n_iter_sigma_k_sq = []

    for i in range(rand_init):
        LL_iter = []

        mu_k = np.random.uniform(0., 3., K).reshape(1, -1)
        sigma_k_sq = np.random.uniform(0.05, 1, K).reshape(1, -1)
        pi_k = np.random.dirichlet(np.ones(K)*100, size=1)

        for _ in range(n_iter):
            lambda_ik,intermediate_term = expectation(mu_k, sigma_k_sq, pi_k, X)          
            pi_k,mu_k,sigma_k_sq = maximization(lambda_ik,X)                              
            mu_k, sigma_k_sq, pi_k = exclude_nans(mu_k, sigma_k_sq, pi_k)                       
            log_intermediate_term = np.log(np.sum(intermediate_term, axis=1))
            lambda_k = np.sum(lambda_ik,axis=1)
            LL = np.sum(lambda_k * (log_intermediate_term - np.log(lambda_k)))
            LL_iter.append(LL)

        LL_iter = np.array(LL_iter).reshape(1,-1)
        mu_k = mu_k.reshape(1,-1)
        sigma_k_sq = sigma_k_sq.reshape(1,-1)

        n_iter_LL.append(LL_iter)
        n_iter_mu_k.append(mu_k)
        n_iter_sigma_k_sq.append(sigma_k_sq)

    LL = np.mean(np.squeeze(np.array(n_iter_LL)),axis=0)
    avg_mu_k = np.mean(np.squeeze(np.array(n_iter_mu_k)),axis=0)
    avg_sigma_k = np.mean(np.squeeze(np.array(n_iter_sigma_k_sq)),axis=0)
    return LL, avg_mu_k, avg_sigma_k


loglikelihood, mu_k, sigma_k_sq = EM_Algorithm(n_iter, X)

plt.figure()
plt.plot(range(1, len(loglikelihood)+1), loglikelihood, marker='o')
plt.title('Log Likelihood vs Iterations Plot for GMM')
plt.xlabel('Iterations')
plt.ylabel('Log-Likelihood')
plt.grid(linestyle = '--')
plt.show()

print("mu of K Gaussian mixtures: ",mu_k)
print("sigma^2 of K Gaussian mixtures: ",sigma_k_sq)