import sys
sys.path.append('/home/tom/OpenBox/bhtsne/')

import numpy as np
import h5py
import matplotlib.image as mpimg

import numpy as np
import matplotlib.pyplot as plt

run = '120k_B'
numframes = 30000


Breakout_tsne = np.zeros(shape=(numframes,2))
Breakout_term = np.zeros(shape=numframes)
Seaquest_tsne = np.zeros(shape=(numframes,2))
Seaquest_term = np.zeros(shape=numframes)
Pacman_tsne = np.zeros(shape=(numframes,2))
Pacman_term = np.zeros(shape=numframes)

print "loading activations... "
Breakout_tsne_file = h5py.File('/home/tom/OpenBox/tsne_res/breakout/120k_b/lowd_activations.h5', 'r')
Breakout_activation_mat = Breakout_tsne_file['data']

Seaquest_tsne_file = h5py.File('/home/tom/OpenBox/tsne_res/seaquest/120k_b/lowd_activations.h5', 'r')
Seaquest_activation_mat = Seaquest_tsne_file['data']

Pacman_tsne_file = h5py.File('/home/tom/OpenBox/tsne_res/pacman/120k_b/lowd_activations.h5', 'r')
Pacman_activation_mat = Pacman_tsne_file['data']
print "loading terminations... "
Breakout_term_file = h5py.File('/home/tom/OpenBox/tsne_res/breakout/120k_b/termination.h5', 'r')
Breakout_term_mat = Breakout_term_file['data']

Seaquest_term_file = h5py.File('/home/tom/OpenBox/tsne_res/seaquest/120k_b/termination.h5', 'r')
Seaquest_term_mat = Seaquest_term_file['data']

Pacman_term_file = h5py.File('/home/tom/OpenBox/tsne_res/pacman/120k_b/termination.h5', 'r')
Pacman_term_mat = Pacman_term_file['data']


for i in range(numframes):
    Breakout_tsne[i] = Breakout_activation_mat[i]
    Breakout_term[i] = Breakout_term_mat[i]
    Seaquest_tsne[i] = Seaquest_activation_mat[i]
    Seaquest_term[i] = Seaquest_term_mat[i]
    Pacman_tsne[i] = Pacman_activation_mat[i]
    Pacman_term[i] = Pacman_term_mat[i]
Breakout_term = Breakout_term/Breakout_term.max()
Breakout_tsne = Breakout_tsne.T
Seaquest_term = Seaquest_term/Seaquest_term.max()
Seaquest_tsne = Seaquest_tsne.T
Pacman_term = Pacman_term/Pacman_term.max()
Pacman_tsne = Pacman_tsne.T
fig, axs = plt.subplots(nrows=2, ncols=3)

for ax in axs.flat:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False


Breakout_img=mpimg.imread('/home/tom/Desktop/TerminalFigure/breakout_initial.png')
Seaquest_img=mpimg.imread('/home/tom/Desktop/TerminalFigure/seaquest_initial.png')
Pacman_img=mpimg.imread('/home/tom/Desktop/TerminalFigure/pacman_initial.png')
mycmap = 'seismic'


Breakout_scalars = np.ones(Breakout_tsne.shape[1])*5
Breakout_scalars[Breakout_term==0] /= 20
Seaquest_scalars = np.ones(Seaquest_tsne.shape[1])*10
Seaquest_scalars[Seaquest_term==0] /= 40
Pacman_scalars   = np.ones(Pacman_tsne.shape[1])*5
Pacman_scalars[Pacman_term==0] /= 20

axs.flat[0].scatter(Breakout_tsne[0], Breakout_tsne[1], s= Breakout_scalars,facecolor=Breakout_term, edgecolor='none',cmap=mycmap)
axs.flat[3].imshow(Breakout_img)
axs.flat[3].set_title("Breakout",fontsize=20)

axs.flat[1].scatter(Seaquest_tsne[0], Seaquest_tsne[1], s= Seaquest_scalars,facecolor=Seaquest_term, edgecolor='none',cmap=mycmap)
axs.flat[4].imshow(Seaquest_img)
axs.flat[4].set_title("Seaquest",fontsize=20)

axs.flat[2].scatter(Pacman_tsne[0], Pacman_tsne[1], s= Pacman_scalars,facecolor=Pacman_term, edgecolor='none',cmap=mycmap)
axs.flat[5].imshow(Pacman_img)
axs.flat[5].set_title("Pacman",fontsize=20)

Breakout_term = Breakout_tsne[:,Breakout_term>0]
Seaquest_term = Seaquest_tsne[:,Seaquest_term>0]
Pacman_term = Pacman_tsne[:,Pacman_term>0]

axs.flat[0].annotate(
        'Termination',
        xy = (Breakout_term[0,0], Breakout_term[1,0]), xytext = (-20, 20),size=8,
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

axs.flat[1].annotate(
        'Termination',
        xy = (Seaquest_term[0,0], Seaquest_term[1,0]), xytext = (-20, 20),size=8,
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

axs.flat[2].annotate(
        'Termination',
        xy = (Pacman_term[0,0], Pacman_term[1,0]), xytext = (-20, 20),size=8,
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()
