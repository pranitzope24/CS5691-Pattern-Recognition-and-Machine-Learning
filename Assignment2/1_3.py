import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('A2Q1.csv', delimiter=',')
K = 4

def k_means(data, k = 4, iterations = 100, seed = 1):
    error = []
    np.random.seed(seed)
    Z = np.random.randint(1,K+1,len(data))
    cl = {}
    for i in range(1,K+1):
        cl[i] = []
    for i in range(len(data)):
        cl = cl_upd(data[i],Z[i], cl,cl_a_prev = 0)
    Z = np.array(Z)
    Z_mu = cl_mu_calc(cl)
    error.append(compute_error(cl,Z_mu))
    a_prev =  np.array(Z)
  
    for i in range(iterations):
        a_next = []
        for k in range(len(data)):
            a = []
            for mean in Z_mu:
                a.append(np.sum((np.array(data[k]) - np.array(mean))**2))
            i  =  np.argmin(np.array(a))
            cl_closest_idx = i+1 
            a_next.append(cl_closest_idx)
            updated_cl = cl_upd(data[k],cl_closest_idx,cl,a_prev[k])

        cl_mu_new = cl_mu_calc(updated_cl)
        Z_mu = cl_mu_new
        error.append(compute_error(updated_cl,Z_mu))
        cl = updated_cl.copy()
        if np.sum(np.squeeze(a_prev) - np.squeeze(np.array(a_next))) != 0:
            a_prev =  np.array(a_next)
            continue
        else:
            last_iter = i
            print("Last Iteration = ",i)
            break
    
    return np.array(a_next),error,last_iter,cl_mu_new

def cl_mu_calc(cl):
    means = []
    dimension_of_each_data_pt = 50
    for i in cl.keys():
        if len(cl[i]) != 0:
            m = np.mean(cl[i],axis = 0)
        else:
            m = np.array([0]*dimension_of_each_data_pt)
        means.append(m)
    return np.array(means)

def cl_upd(dp,Z_cur, cl,cl_a_prev = 0):
    if cl_a_prev == 0:
        cl[Z_cur].append(dp)
    else:
        C = []
        for i,pt in enumerate(cl[cl_a_prev]):
            if (pt[0] != dp[0]) and (pt[1] != dp[1]) :
                C.append(pt)
        cl[cl_a_prev] = C
        cl[Z_cur].append(dp)
    return cl

def compute_error(cl,mu): 
    err = 0
    mu = cl_mu_calc(cl)
    for i in range(len(mu)):
        if len(cl[i+1]) != 0:
            err_cl_i = np.sum((np.array(cl[i+1]) - np.array(mu[i]))**2)
        else:
            err_cl_i = 0
        err +=  err_cl_i
    return err

k_assignment, err, last_iter, means = k_means(X,K)

plt.figure()
plt.plot(range(1,len(err)+1),err,marker = 'o')
plt.xlabel("Iterations")
plt.ylabel("K-Means Error")
plt.title("K-Means Error Plot")
plt.grid()
plt.show()

print(k_assignment)
print("Cluster means  =  ", means)