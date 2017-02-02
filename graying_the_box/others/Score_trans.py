import sys
sys.path.append('/home/tom/OpenBox/bhtsne/')

import numpy as np
import h5py
import matplotlib.image as mpimg

import numpy as np
import matplotlib.pyplot as plt

numframes = 13000
ind1=6323
ind2=1315

im_size        = 84


print "loading states... "



Seaquest_state_file = h5py.File('/home/tom/OpenBox/tsne_res/seaquest/13k/screens.h5', 'r')
Seaquest_state_mat = Seaquest_state_file['data']
Seaquest_states = Seaquest_state_mat[:numframes]
Seaquest_states = np.reshape(np.transpose(Seaquest_states), (3,210,160,-1))
Seaquest_states=np.transpose(Seaquest_states,(3,1,2,0))


fig, axs = plt.subplots(nrows=1, ncols=3)

for ax in axs.flat:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

axs.flat[0].imshow(Seaquest_states[ind1], interpolation='none')
axs.flat[2].imshow(Seaquest_states[ind1+1], interpolation='none')
Seaquest_img=mpimg.imread('/home/tom/Desktop/score/transition1.png')
axs.flat[1].imshow(Seaquest_img)

plt.show()
