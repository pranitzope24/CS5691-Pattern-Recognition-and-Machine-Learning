import numpy as np

data = np.loadtxt('A2Q2Data_train.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1] 
print(np.shape(X))

wML = np.linalg.inv(X.T @ X) @ X.T @ y
print(wML)