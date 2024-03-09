import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Importing Data
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



## PCA and Variance Calculation
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


## Calculating d which would be suitable for sownstream tasks
req_var = 0
d = 1
while(req_var < 95) :
    req_var += var_ratio[d-1]*100
    d+=1
print(f'Required dimensions d = {d} with a variance of {req_var}')
selected_pcs = eigvec[:, :d]


## Plotting a random datapoint for different dimesions considered
# idx = np.random.choice(1000,1)
idx = 410 # random
comp_dict = {}
disp_pcs = [0,1,3,10,50,100,150,300,400,500,784]
disp_pcs.append(d)
disp_pcs.sort()
for i in range(0,11):
    sel_pcs = eigvec[:,:disp_pcs[i]]
    og_data = features.loc[idx].to_numpy()
    label_proxy = og_data - mu
    rec_data = label_proxy.dot(sel_pcs)
    rec_data = rec_data.dot(sel_pcs.T) + mu
    comp_dict[i] = rec_data.reshape(-1, 28, 28)

fig, axes = plt.subplots(3, 4, figsize=(10, 10))
for i in range(3):
    for j in range(4):
        if i == 0 and j == 0 :
            axes[i,j].imshow(og_data.reshape(-1,28,28)[0],cmap='gray')
            axes[i,j].set_title('Original Image')
            axes[i,j].axis('off')
        else :
            axes[i,j].imshow(comp_dict[4*i+j-1][0],cmap='gray')
            axes[i,j].set_title(f'd = {disp_pcs[4*i+j]}')
            axes[i,j].axis('off')
# plt.show()



## Using the obtained d to visualize some points from the data
rec_dict = {}
og_dict = {}
for i in range(10):
    label_idx = L.index[L == i]
    smp_idx = np.random.choice(label_idx, size=5, replace=False)
    og_data = features.loc[smp_idx].to_numpy()
    label_proxy = og_data - mu
    rec_data = label_proxy.dot(selected_pcs)
    rec_data = rec_data.dot(selected_pcs.T) + mu
    rec_dict[i] = rec_data.reshape(-1, 28, 28)
    og_dict[i] = og_data.reshape(-1, 28, 28)

fig, axes = plt.subplots(10, 10, figsize=(10, 15))
for i in range(10):
    for j in range(5):
        axes[i, 2*j].imshow(og_dict[i][j], cmap='gray')
        axes[i, 2*j].set_title(f'OG:L{i},I{j+1}')
        axes[i, 2*j].axis('off')
        axes[i, 2*j+1].imshow(rec_dict[i][j], cmap='gray')
        axes[i, 2*j+1].set_title(f'R:L{i},I{j+1}')
        axes[i, 2*j+1].axis('off')
plt.show()