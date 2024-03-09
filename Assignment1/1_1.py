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
mu = np.mean(X, axis=0)
X_c = X - mu
XX_T = np.cov(X_c, rowvar=False)
eigval, eigvec = np.linalg.eigh(XX_T)
srt_idx = np.argsort(eigval)[::-1]
eigval = eigval[srt_idx]
eigvec = eigvec[:, srt_idx]
proxy = X_c.dot(eigvec)
var_ratio = eigval / np.sum(eigval)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    r = i // 5 
    c = i % 5
    pc = eigvec[:, i].reshape(28, 28)
    axes[r, c].imshow(pc, cmap='gray')
            axes[r, c].set_title(f'PC {i+1}')
            axes[r, c].axis('off')
plt.show()

print("Explained Variance Ratio:")
for i in range(10):
    print(f"PC {i+1}: {var_ratio[i]*100:.3f}%")