import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('A2Q2Data_train.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

n, d = X.shape 
wML = np.linalg.inv(X.T @ X) @ X.T @ y
print(wML)

alpha = 0.02
num_iters = 25
b = 100
w = np.zeros(d)
mse = []

for i in range(num_iters):
    idx = np.random.permutation(n)
    X_r = X[idx]
    y_r = y[idx]
    for j in range(0, n, b):
        X_b = X_r[j:j+b]
        y_b = y_r[j:j+b]
        y_pred = X_b @ w 
        grad = 2 * X_b.T @ (y_pred - y_b) / b
        w -= alpha * grad
    mse.append(np.linalg.norm(wML - w) ** 2)
print(w)

plt.figure()
plt.title("Mean Squared Error : Stochastic Gradient Descent")
plt.xlabel('Iterations')
plt.ylabel('$||w_{ML} - w||^2$')
plt.grid('--')
plt.plot(range(num_iters), mse, label='MSE')
plt.show()


