import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
from add_global_features import add_buttons as add_global_buttons
from control_buttons import add_buttons as add_control_buttons
import common

class VIS_TOOL(object):

    def __init__(self, global_feats, hand_craft_feats, game_id, cluster_params):

        # 0. connect arguments
        self.global_feats = global_feats
        self.game_id = game_id
        self.hand_craft_feats = hand_craft_feats
        self.num_points = global_feats['tsne'].shape[0]
        self.data_t = global_feats['tsne'].T
        screens = np.reshape(np.transpose(global_feats['screens']), (3,210,160,-1))
        self.screens = np.transpose(screens,(3,1,2,0))
        self.im_size = np.sqrt(global_feats['states'].shape[1])
        self.states = np.reshape(global_feats['states'], (global_feats['states'].shape[0], self.im_size,self.im_size))
        self.tsne3d_next = global_feats['tsne3d_next'].T
        self.tsnedata = global_feats['tsne3d'].T
        self.traj_index = global_feats['trajectory_index']
        self.tsne3d_norm = global_feats['tsne3d_norm']
	tmp = global_feats['value'] - np.amin(np.array(global_feats['value']))
        self.color = tmp / np.amax(tmp)
        self.cluster_params = cluster_params

        # 1. Constants
        self.pnt_size = 2
        self.ind      = 0
        self.prev_ind = 0

        # 2. Plots
        self.fig = plt.figure('tSNE')

        # 2.1 t-SNE
        self.ax_tsne = plt.subplot2grid((3,5),(0,0), rowspan=3, colspan=3)

        self.tsne_scat = self.ax_tsne.scatter(self.data_t[0], self.data_t[1], s= np.ones(self.num_points)*self.pnt_size,c = self.color,edgecolor='none',picker=5)

        self.ax_tsne.set_xticklabels([])
        self.ax_tsne.set_yticklabels([])

        # 2.1.5 colorbar
        cb_axes = self.fig.add_axes([0.253,0.13,0.2,0.01])
        cbar = self.fig.colorbar(self.tsne_scat, cax=cb_axes, orientation='horizontal', ticks=[0,1])
        cbar.ax.set_xticklabels(['Low','High'])

        # 2.2 game screen (state)
        self.ax_screen = plt.subplot2grid((3,5),(2,3), rowspan=1, colspan=1)

        self.screenplot = self.ax_screen.imshow(self.screens[self.ind], interpolation='none')

        self.ax_screen.set_xticklabels([])
        self.ax_screen.set_yticklabels([])

        # 2.3 gradient image (saliency map)
        self.ax_state = plt.subplot2grid((3,5),(2,4), rowspan=1, colspan=1)

        self.stateplot = self.ax_state.imshow(self.states[self.ind], interpolation='none', cmap='gray',picker=5)

        self.ax_state.set_xticklabels([])
        self.ax_state.set_yticklabels([])

        # 3. Global Features
        add_global_buttons(self, global_feats)

        # 4. Control buttons
        add_control_buttons(self)

        # 4.1 add game buttons
        if game_id == 0: # breakout
            from add_breakout_buttons import add_game_buttons, update_cond_vector
        if game_id == 1: # seaquest
            from add_seaquest_buttons import add_game_buttons, update_cond_vector
        if game_id == 2: # pacman
            from add_pacman_buttons import add_game_buttons, update_cond_vector

        add_game_buttons(self)

        self.update_cond_vector = update_cond_vector

    def add_color_button(self, pos, name, color):
        def set_color(event):
            self.color = self.COLORS[id(event.inaxes)]
            self.tsne_scat.set_array(self.color)

        ax = plt.axes(pos)
        setattr(self, name+'_button', Button(ax, name))
        getattr(self, name+'_button').on_clicked(set_color)
        color = np.array(color) - np.amin(np.array(color))
        self.COLORS[id(ax)] = color/np.amax(color)

    def update_sliders(self, val):
        for f in self.SLIDER_FUNCS:
            f()
        # self.update_cond_vector_breakout()

    def add_slider_button(self, pos, name, v_min, v_max):

        def update_slider(self, name, slider):
            def f():
                setattr(self, name, slider.val)
            return f

        ax_min = plt.axes(pos)
        ax_max = plt.axes([pos[0], pos[1]-0.02, pos[2], pos[3]])

        slider_min = Slider(ax_min, name+'_min', valmin=v_min, valmax=v_max, valinit=v_min)
        slider_max = Slider(ax_max, name+'_max', valmin=v_min, valmax=v_max, valinit=v_max)

        self.SLIDER_FUNCS.append(update_slider(self, name+'_min', slider_min))
        self.SLIDER_FUNCS.append(update_slider(self, name+'_max', slider_max))

        slider_min.on_changed(self.update_sliders)
        slider_max.on_changed(self.update_sliders)

    def add_check_button(self, pos, name, options, init_vals):
        def set_options(label):
            pass

        ax = plt.axes(pos)
        setattr(self, name+'_check_button', CheckButtons(ax, options, init_vals))
        getattr(self, name+'_check_button').on_clicked(set_options)

    def on_scatter_pick(self,event):
        self.ind = event.ind[0]
        self.update_plot()
        self.prev_ind = self.ind

    def update_plot(self):
        self.screenplot.set_array(self.screens[self.ind])
        self.stateplot.set_array(self.states[self.ind])
        sizes = self.tsne_scat.get_sizes()
        sizes[self.ind] = 250
        sizes[self.prev_ind] = self.pnt_size
        self.tsne_scat.set_sizes(sizes)
        self.fig.canvas.draw()
        print 'chosen point: %d' % self.ind

    def show(self):
        plt.show(block=True)
