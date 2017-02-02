import sys
sys.path.append('/home/tom/OpenBox/bhtsne/')

import numpy as np
import h5py
import matplotlib.image as mpimg

import numpy as np
import matplotlib.pyplot as plt

numframes = 3000
Seaquestind1 = 53
Seaquestind2 = 193
Seaquestind3 = 203

im_size        = 84


print "loading states... "

# Breakout_state_file = h5py.File('/home/tom/OpenBox/tsne_res/6k/states.h5', 'r')
# Breakout_state_mat = Breakout_state_file['data']
# Breakout_states = Breakout_state_mat[:numframes]
# Breakout_states = np.reshape(Breakout_states, (Breakout_states.shape[0], im_size,im_size))

Seaquest_state_file = h5py.File('/home/tom/OpenBox/tsne_res/seaquest/13k/states.h5', 'r')
Seaquest_state_mat = Seaquest_state_file['data']
Seaquest_states = Seaquest_state_mat[:numframes]
Seaquest_states = np.reshape(Seaquest_states, (Seaquest_states.shape[0], im_size,im_size))

# Pacman_state_file = h5py.File('/home/tom/OpenBox/tsne_res/pacman/7k/states.h5', 'r')
# Pacman_state_mat = Pacman_state_file['data']
# Pacman_states = Pacman_state_mat[:numframes]
# Pacman_states = np.reshape(Pacman_states, (Pacman_states.shape[0], im_size,im_size))
print "loading grads... "
thresh     = 0.1

# Breakout_grad_file = h5py.File('/home/tom/OpenBox/tsne_res/6k/grads.h5', 'r')
# Breakout_grad_mat = Breakout_grad_file['data']
# Breakout_grads = Breakout_grad_mat[:numframes]
# Breakout_grads[np.abs(Breakout_grads)<thresh] = 0
# Breakout_grads = np.reshape(Breakout_grads, (Breakout_grads.shape[0], im_size,im_size))

thresh     = 0.05

Seaquest_grad_file = h5py.File('/home/tom/OpenBox/tsne_res/seaquest/13k/grads.h5', 'r')
Seaquest_grad_mat = Seaquest_grad_file['data']
Seaquest_grads = Seaquest_grad_mat[:numframes]
Seaquest_grads[np.abs(Seaquest_grads)<thresh] = 0
Seaquest_grads = np.reshape(Seaquest_grads, (Seaquest_grads.shape[0], im_size,im_size))

# Pacman_grad_file = h5py.File('/home/tom/OpenBox/tsne_res/pacman/7k/grads.h5', 'r')
# Pacman_grad_mat = Pacman_grad_file['data']
# Pacman_grads = Pacman_grad_mat[:numframes]
# Pacman_grads[np.abs(Pacman_grads)<thresh] = 0
# Pacman_grads = np.reshape(Pacman_grads, (Pacman_grads.shape[0], im_size,im_size))


fig, axs = plt.subplots(nrows=1, ncols=3)

for ax in axs.flat:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False

axs.flat[0].imshow(Seaquest_states[Seaquestind1], interpolation='none', cmap='gray')
my_cmap = plt.cm.get_cmap('hot')
my_cmap.set_bad('w')
my_cmap.set_under('w')
axs.flat[0].imshow(Seaquest_grads[Seaquestind1], interpolation='none', cmap=my_cmap,vmin=.001,alpha=0.4)
axs.flat[0].set_title("a")

axs.flat[1].imshow(Seaquest_states[Seaquestind2], interpolation='none', cmap='gray')
my_cmap = plt.cm.get_cmap('hot')
my_cmap.set_bad('w')
my_cmap.set_under('w')
axs.flat[1].imshow(Seaquest_grads[Seaquestind2], interpolation='none', cmap=my_cmap,vmin=.001,alpha=0.4)
axs.flat[1].set_title("b")

axs.flat[2].imshow(Seaquest_states[Seaquestind3], interpolation='none', cmap='gray')
my_cmap = plt.cm.get_cmap('hot')
my_cmap.set_bad('w')
my_cmap.set_under('w')
axs.flat[2].imshow(Seaquest_grads[Seaquestind3], interpolation='none', cmap=my_cmap,vmin=.001,alpha=0.4)
axs.flat[2].set_title("c")
plt.show()
