import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

data = np.loadtxt("cm_dataset_2.csv", delimiter=",")
n, d = data.shape
k = 2 
iterationCounter = 0
input = data
var = 1.5 
max_iter = 100

def rbf_kernel(x, y, sigma):
    delta = np.matrix(abs(np.subtract(x, y)))
    return np.exp(-(np.square(delta).sum(axis=1))/(2*sigma**2))

def poly_kernel(x,y, d) :
    return (np.dot(x,y) + 1) ** d


def compute(func,arg):
    S = np.ones((n,n), dtype = np.float64)
    for i in range(0,n) :
        for j in range(0,n) :
            S[i,j] = func(data[i], data[j], arg)
    # S = S - np.mean(S)
    D = np.diag(np.array(S.sum(axis=1)).ravel())
    L = D-S

    e_vals, e_vecs = np.linalg.eigh(L)  
    ind = e_vals.real.argsort()[:k]
    H = np.ndarray(shape=(L.shape[0],0))
    for i in range(1, ind.shape[0]):
        cor_e_vec = np.transpose(np.matrix(e_vecs[:,ind[i]]))
        H = np.concatenate((H, cor_e_vec), axis=1)

    centriods = H[np.random.choice(H.shape[0], k, replace=False)]

    while(iterationCounter < max_iter):
        iterationCounter +=1
        euclideanMatrixAllCluster = np.ndarray(shape=(H.shape[0], 0))
        for i in range(0, k):
            centroidRepeated = np.repeat(centriods[i, :], H.shape[0], axis=0)
            deltaMatrix = abs(np.subtract(H, centroidRepeated))
            euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
            euclideanMatrixAllCluster = \
                np.concatenate((euclideanMatrixAllCluster, euclideanMatrix), axis=1)
        clusterMatrix = np.ravel(np.argmin(np.matrix(euclideanMatrixAllCluster), axis=1))
        listClusterMemberTransf = [[] for i in range(k)]
        listClusterMemberOri = [[] for i in range(k)]
        for i in range(0, H.shape[0]):#assign data to cluster regarding cluster matrix
            listClusterMemberTransf[(clusterMatrix[i])].append(np.array(H[i, :]).ravel())
            listClusterMemberOri[(clusterMatrix[i])].append(np.array(data[i, :]).ravel())
        #calculate new centroid
        newCentroidTransf = np.ndarray(shape=(0, centriods.shape[1]))
        newCentroidOri = np.ndarray(shape=(0, data.shape[1]))
        # print("iteration: ", iterationCounter) 
        for i in range(0,k):
            memberClusterTransf = np.asmatrix(listClusterMemberTransf[i])
            memberClusterOri = np.asmatrix(listClusterMemberOri[i])
            # print("cluster members number-", i+1, ": ", memberClusterTransf.shape)
            centroidClusterTransf = memberClusterTransf.mean(axis=0)
            centroidClusterOri = memberClusterOri.mean(axis=0)
            newCentroidTransf = np.concatenate((newCentroidTransf, centroidClusterTransf), axis=0)
            newCentroidOri = np.concatenate((newCentroidOri, centroidClusterOri), axis=0)
        if((centriods == newCentroidTransf).all()):
            break
        centriods = newCentroidTransf

    result = listClusterMemberOri
    mu = newCentroidOri

    colors = plt.cm.Spectral(np.linspace(0, 1, k))
    plt.figure("result")
    plt.clf()
    if func == rbf_kernel :
        plt.title(f"Spectral Clustering with RBF Kernel, var = {arg}")
    else :
        plt.title(f"Spectral Clustering with Polynomial Kernel, d={arg}")

    for i in range(k):
        memberCluster = np.asmatrix(result[i])
        plt.scatter(np.ravel(memberCluster[:, 0]), np.ravel(memberCluster[:, 1]), marker=".", s=100, color=colors[i])

    for i in range(k):
        plt.scatter(np.ravel(mu[:, 0]), np.ravel(mu[:, 1]), color='black', marker='*', s=150, label='mu')

    plt.show()


compute(rbf_kernel,0.5)
compute(rbf_kernel,1.5)
compute(rbf_kernel,5)
compute(poly_kernel,2)
compute(poly_kernel,3)
compute(poly_kernel,4)

