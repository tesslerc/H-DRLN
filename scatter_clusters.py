import numpy as np
import matplotlib.pyplot as plt
import h5py

clust_num = 0
single_cluster = True

#with h5py.File('./tsneData.h5', 'r') as hf:
with h5py.File('./tsneMatch.h5', 'r') as hf:
    # print('List of arrays in this file: \n', hf.keys())
    data = hf.get('data')
    np_data = np.array(data)
    # print('Shape of the array data: \n', np_data.shape)
    x = data[0, :]
    y = data[1, :]
    color = data[2, :]

if single_cluster == True:
    indices = np.nonzero(color == clust_num)
    x = np_data[0, indices]
    y = np_data[1, indices]
    color = np_data[2, indices]

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(111)

if single_cluster == False:
    ax1.scatter(x, y, c=color, s=1, marker='o')
    #ax1.scatter(np_data[0, np.nonzero(color == 0)], np_data[1, np.nonzero(color == 0)], c='b', s=1, alpha=0.3, marker='o', label='first')
    #ax1.scatter(np_data[0, np.nonzero(color == 1)], np_data[1, np.nonzero(color == 1)], c='r', s=1.5, alpha=0.3, marker="x", label='second')
    #ax1.scatter(np_data[0, np.nonzero(color == 2)], np_data[1, np.nonzero(color == 2)], c='g', s=2, alpha=0.3, marker="v", label='third')
    #plt.legend(loc='upper left')
else:
    colors = ['b', 'r', 'g']
    ax1.scatter(x, y, c=colors[clust_num], s=1, marker='o')

plt.xlim(-90, 90)
plt.ylim(-90, 90)

if single_cluster == True:
    plt.savefig('t-SNE_800perplexity_clusters_heatmap_cluster' + str(clust_num) + '.png')
else:
    plt.savefig('t-SNE_800perplexity_clusters2_heatmap.png')

plt.show()
