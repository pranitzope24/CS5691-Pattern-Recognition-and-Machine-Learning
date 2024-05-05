import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt('A2Q2Data_train.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

n, d = X.shape 
wML = np.linalg.inv(X.T @ X) @ X.T @ y
print(wML)

alpha = 0.02
num_iters = 2500
w = np.zeros(d)
mse = []

for i in range(num_iters):
    y_pred = X @ w
    mse.append(np.linalg.norm(wML-w)**2)
    grad = 2 * X.T @ (y_pred - y) / n
    w -= alpha * grad

print(w)

plt.figure()
plt.title("Mean Squared Error : Gradient Descent")
plt.xlabel('Iterations')
plt.ylabel('$||w_{ML} - w||^2$')
plt.grid('--')
plt.plot(range(num_iters), mse, label='MSE')
plt.show()
