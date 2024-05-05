import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.loadtxt('A2Q1.csv', delimiter=',')

mean = X.mean(axis=0)
X_c = X - mean
cov_mat = (1/400.0) * np.dot(X_c.T,X_c)
eigenval, eigenvec = np.linalg.eig(cov_mat)
print("Eigen Values : \n",eigenval,"\n")

pr_num1 = X.sum(axis=1) / X.shape[1]
print("Unique probability of number of ones in a Datapoint: \n", np.unique(pr_num1,return_counts=True)[0]) 
    #This proves that there are only 15 unique p values implies that the points come from a bernoulli mixture

plt.figure()
plt.plot(range(1,len(np.unique(pr_num1))+1),np.unique(pr_num1),marker="o")
plt.title("p trend (Bernoulli)") 
plt.xlabel("No. of unique p-values")
plt.ylabel("p")
plt.grid(linestyle = '--')
plt.show()

plt.figure()
plt.plot(np.unique(pr_num1,return_counts=True)[0],np.unique(pr_num1,return_counts=True)[1],marker="o")
plt.title("Number of Datapoints vs p") 
plt.xlabel("p-value")
plt.ylabel("Occurence Frequency")
plt.grid(linestyle = '--')
plt.show()

K = 4
n_iter = 10

def expectation(p_k, pi_k, X):
    p_k_term = p_k ** X
    one_minus_p_k_term = (1 - p_k) ** (1 - X) 
    numerator= pi_k * p_k_term * one_minus_p_k_term
    denominator=np.sum(numerator, axis=1).reshape(-1, 1)
    lambda_ik_updated = numerator/denominator
    return lambda_ik_updated, numerator

# Maximization step: Updating parameters using lambdas calculated in Expectation step
def maximization(lambda_ik, X):
    num_datapts = X.shape[0]
    pi_k = np.sum(lambda_ik, axis=0).reshape(1,-1)/ num_datapts
    p_k = (np.dot(X.T,lambda_ik)/np.sum(lambda_ik, axis=0).reshape(1,-1)).reshape(1,-1)
    return pi_k,p_k


def EM_Algorithm(n_iter,X,n_init=100):
    X = X.reshape(-1, 1)
    n_iter_LL = []
    n_iter_p_k = []
  
    for i in range(n_init):
        LL_iter = []

        p_k = np.random.uniform(0.001, 1, K).reshape(1, -1)
        pi_k = np.random.dirichlet(np.ones(K)*1000, size=1)

        for _ in range(n_iter):
            lambda_ik,intermediate_term = expectation(p_k, pi_k, X)           
            pi_k,p_k = maximization(lambda_ik,X)                              
            log_intermediate_term = np.log(np.sum(intermediate_term, axis=1))
            lambda_k = np.sum(lambda_ik,axis=1)
            LL = np.sum(lambda_k * (log_intermediate_term - np.log(lambda_k)))
            LL_iter.append(LL)

        LL_iter = np.array(LL_iter).reshape(1,-1)
        p_k = p_k.reshape(1,-1)
        n_iter_LL.append(LL_iter)
        n_iter_p_k.append(p_k)

    LL = np.mean(np.squeeze(np.array(n_iter_LL)),axis=0)
    avg_p_k = np.mean(np.squeeze(np.array(n_iter_p_k)),axis=0)
    return LL, avg_p_k


loglikelihood, p_k = EM_Algorithm(n_iter, X)

plt.figure()
plt.plot(range(1, len(loglikelihood)+1), loglikelihood,marker='o')
plt.title('Log Likelihood vs Iterations Plot for BMM')
plt.xlabel('Iterations')
plt.ylabel('Log-Likelihood')
plt.grid(linestyle = '--')
plt.show()
print("p of K Bernoulli mixtures: ",p_k)