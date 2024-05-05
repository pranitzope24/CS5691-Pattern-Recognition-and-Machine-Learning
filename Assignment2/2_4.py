import numpy as np
import matplotlib.pyplot as plt
import time

st = time.time()
step_size=0.01
num_iterations=5000
num_folds=5

data = np.loadtxt('A2Q2Data_train.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

data_t = np.loadtxt('A2Q2Data_test.csv', delimiter=',')
Xt = data_t[:, :-1]
yt = data_t[:, -1]

def ridge_regression(Xt, yt, L) :
    n_ = Xt.shape[0]
    d_ = Xt.shape[1]
    XtTX =  Xt.T @ Xt
    w_t = np.zeros((d_))
    for _ in np.arange(num_iterations):
      grad = (2*( XtTX @ w_t - Xt.T @ yt) + 2 * L* w_t) / n_  
      w_t1 = w_t - step_size * grad
      w_t = w_t1
    return w_t

def k_fold_cv(ds, L):
    ds_1 = ds.copy()
    np.random.shuffle(ds_1)
    F = np.split(ds_1, num_folds, axis = 0)

    err_F = []
    for i in range(num_folds):
        f_test = F[i].copy()

        initialized = False
        for j in range(num_folds):
            if j == i:
                continue
            
            if initialized == False:
                f_train = F[j].copy()
                initialized = True
            else:
                f_train = np.concatenate((f_train, F[j]), axis=0)
        
        Xt = f_train[:, :-1]
        yt = f_train[:, -1]
        w_r = ridge_regression(Xt, yt, L)

        X_test = f_test[:, :-1]
        y_test = f_test[:, -1]

        err = X_test @ w_r - y_test
        err_F.append(np.dot(err,err))
    return np.average(err_F)


w_ML = np.linalg.inv(X.T @ X) @ X.T @ y

err_L = []
lambdas = np.linspace(0.01, 3, 50)

for L in lambdas:
    print('Processing Lambda Value = ' + str(L))
    err = k_fold_cv(data, L)
    err_L.append(err)

L_idx = np.argmin(err_L)
L_optimal = lambdas[L_idx]

print("Optimal Lambda : ", L_optimal)
w_R = ridge_regression(X, y, L_optimal)
print(w_R)

err_ML = Xt @ w_ML - yt
err_R = Xt @ w_R - yt

w_ML_err = err_ML.T @ err_ML
w_Ridge_err = err_R.T @ err_R

print(" w_ML : error = " + str(w_ML_err))
print(" w_Ridge: error = " + str(w_Ridge_err))

plt.figure()
plt.plot(lambdas, err_L, 'o-')  
plt.title('Validation Error vs Lambda')
plt.xlabel('Lambda')
plt.ylabel('Validation Error')
plt.grid()
plt.show()

en = time.time()

print("Time elapsed (ms) = " + str(en-st))  # runtime is very high due to massive calculations (1509 sec in this case)
