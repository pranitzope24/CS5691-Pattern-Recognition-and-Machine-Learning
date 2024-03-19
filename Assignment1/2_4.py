import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

data = np.loadtxt("cm_dataset_2.csv", delimiter=",")
n, d = data.shape
k = 2 
iterationCounter = 0

var = 0.1 
max_iter = 1

def rbf_kernel(x, y, sigma):
    delta = np.matrix(abs(np.subtract(x, y)))
    return np.exp(-(np.square(delta).sum(axis=1))/(2*sigma**2))

def poly_kernel(x,y, d) :
    return (np.dot(x,y) + 1) ** d

np.random.seed(42)
centroids = data[np.random.choice(range(len(data)), k, replace=False)]
print(centroids)
og_centroids = centroids
def cluster(func,arg):
    global centroids
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = func(data[i], data[j],arg)
    
    for _ in range(max_iter):
        eigvals, eigvecs = np.linalg.eigh(K)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        top_eigvecs = eigvecs[:, :k]
        labels = np.argmax(top_eigvecs, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    colors = plt.cm.Spectral(np.linspace(0, 1, k))
    for cluster_label, color in zip(range(k), colors):
        cluster_points = data[labels == cluster_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], marker='o', label=f'Cluster {cluster_label + 1} - Initialization')
    plt.scatter(og_centroids[:, 0], og_centroids[:, 1], c='black', marker='o', label='init')
    if func == poly_kernel:
        plt.title(f'Polynomial Kernel, d={arg}')
    else :
        plt.title(f'RBF Kernel, var={arg}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

cluster(rbf_kernel,0.5)
cluster(rbf_kernel,1.5)
cluster(rbf_kernel,5)
cluster(poly_kernel,2)
cluster(poly_kernel,3)
cluster(poly_kernel,4)