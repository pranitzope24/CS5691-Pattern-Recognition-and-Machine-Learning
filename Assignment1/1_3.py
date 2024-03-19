import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
L = data['label']
features = data.drop('label', axis=1)
sampled_data = pd.DataFrame()
np.random.seed(24)
for i in range(10):
    indices = L.index[L == i]
    smp_idx = np.random.choice(indices, size=100, replace=False)
    sampled_data = pd.concat([sampled_data, features.loc[smp_idx]])
sampled_data.reset_index(drop=True, inplace=True)

X = sampled_data.to_numpy()
X = X.astype(np.float64)

def polynomial_kernel(X, Y, degree=3):
    return (np.dot(X, Y.T) + 1) ** degree

def rbf_kernel(x, y, sigma = 1.0):
    delta = np.matrix(abs(np.subtract(x, y)))
    return np.exp(-(np.square(delta).sum(axis=1))/(2*sigma**2))

def kernel_pca(X, kernel_function, n_components=2):
    n_samples = X.shape[0]
    K = kernel_function(X, X)
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_c = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigval, eigvec = np.linalg.eigh(K_c)
    srt_idx = np.argsort(eigval)[::-1]
    eigval = eigval[srt_idx]
    eigvec = eigvec[:, srt_idx]
    top_components = eigvec[:, :n_components]
    return top_components

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
degrees = [2, 3, 4]

for i, degree in enumerate(degrees):
    poly_kernel_components = kernel_pca(X, lambda X, Y: polynomial_kernel(X, Y, degree=degree), n_components=2)
    axes[i].scatter(poly_kernel_components[:, 0], poly_kernel_components[:, 1], c=L[:1000], cmap='viridis', s=10)
    axes[i].set_title(f'Polynomial Kernel PCA (Degree {degree})')
    axes[i].set_xlabel('Principal Component 1')
    axes[i].set_ylabel('Principal Component 2')

plt.show()

# Perform Kernel PCA with RBF Kernel for sigmas 1, 5, 10
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sigmas = [0.5, 1, 5]

for i, sigma in enumerate(sigmas):
    rbf_kernel_components = kernel_pca(X, lambda X, Y: rbf_kernel(X, Y, sigma=sigma), n_components=2)
    axes[i].scatter(rbf_kernel_components[:, 0], rbf_kernel_components[:, 1], c=L[:1000], cmap='viridis', s=10)
    axes[i].set_title(f'RBF Kernel PCA (Sigma {sigma})')
    axes[i].set_xlabel('Principal Component 1')
    axes[i].set_ylabel('Principal Component 2')

plt.show()
