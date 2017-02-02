import numpy as np

def add_game_buttons(self, top_y=0.69, x_left = 0.68):

    self.add_slider_button([x_left, top_y, 0.08, 0.01], 'ball_x', 0, 160)
    self.add_slider_button([x_left, top_y-0.04, 0.08, 0.01], 'ball_y', 0, 210)
    self.add_slider_button([x_left, top_y-0.08, 0.08, 0.01], 'racket_x', 0, 160)
    self.add_slider_button([x_left, top_y-0.12, 0.08, 0.01], 'missing_bricks', 0, 130)
    self.add_check_button([x_left, top_y-0.20, 0.06, 0.04], 'hole', ('no-hole','hole'), (True,True))
    self.add_check_button([x_left, top_y-0.30, 0.07, 0.08], 'ball_dir', ('down-right','up-right','down-left','up-left'), (True,True,True,True))
    self.ball_dir_mat = np.zeros(shape=(self.num_points,4), dtype='bool')
    for i,dir in enumerate(self.hand_craft_feats['ball_dir']):
        self.ball_dir_mat[i,int(dir)] = 1

        ##############################################
        # marking points along trajectories from a to b

        # self.cond_vector_mark_trajs = np.zeros(shape=(self.num_points,), dtype='int8')
        #
        # self.fig_mark_trajs = plt.figure('mark trajectories')
        # self.fig_mark_trajs.canvas.mpl_connect('pick_event', self.on_scatter_pick_mark_trajs)
        # self.ax_mark_trajs = self.fig_mark_trajs.add_subplot(111)
        # self.scat_mark_trajs = self.ax_mark_trajs.scatter(self.data_t[0],
        #                                                   self.data_t[1],
        #                                                   s = 5 * self.cond_vector_mark_trajs + np.ones(self.num_points)*self.pnt_size,
        #                                                   facecolor = self.V,
        #                                                   edgecolor='none',
        #                                                   picker=5)

        ###############################################

def update_cond_vector(self):

    ball_x = np.asarray([row[0] for row in self.hand_craft_feats['ball_pos']])
    ball_y = np.asarray([row[1] for row in self.hand_craft_feats['ball_pos']])
    ball_dir = np.asarray(self.hand_craft_feats['ball_dir'])
    racket_x = np.asarray(self.hand_craft_feats['racket'])
    missing_bricks = np.asarray(self.hand_craft_feats['missing_bricks'])
    has_hole = np.asarray(self.hand_craft_feats['hole'])

    self.cond_vector =  (ball_x >= self.ball_x_min) * (ball_x <= self.ball_x_max) * \
                        (ball_y >= self.ball_y_min) * (ball_y <= self.ball_y_max) * \
                        (racket_x >= self.racket_x_min) * (racket_x <= self.racket_x_max) * \
                        (missing_bricks >= self.missing_bricks_min) * (missing_bricks <= self.missing_bricks_max)

    # ball dir
    dirs_mask = np.zeros_like(self.cond_vector)
    for i,val in enumerate(self.ball_dir_check_button.get_status()):
        if val:
            dirs_mask += self.ball_dir_mat[:,i]
    self.cond_vector = self.cond_vector * dirs_mask

    # has a hole
    if self.hole_check_button.get_status()[0] == 0: # filter out states with no-hole
        self.cond_vector = self.cond_vector * (has_hole)
    elif self.hole_check_button.get_status()[1] == 0: # filter out states with a hole
        self.cond_vector = self.cond_vector * (1-has_hole)
    self.cond_vector = self.cond_vector.astype(int)

def on_scatter_pick_mark_trajs(self,event):
    if hasattr(event,'ind'):
        ind = event.ind[0]
    traj_ids = self.state_labels[:,6]
    times = self.state_labels[:,7]
    has_hole = self.state_labels[:,5]

    traj_id = traj_ids[ind]
    time = times[ind]

    # if current point is already marked then un-mark the entire trajectory
    if self.cond_vector_mark_trajs[ind] == 1:
        self.cond_vector_mark_trajs[np.nonzero(traj_ids==traj_id)] = 0
    else:
        # mark the entire point on the current trajectory from time until has_hole = 1
        cond = 1 * (traj_ids == traj_id) * (times >=time) * (has_hole == 0)
        self.cond_vector_mark_trajs[np.nonzero(cond)] = 1
        self.cond_vector_mark_trajs = self.cond_vector_mark_trajs.astype(int)

    sizes = 5 * self.cond_vector_mark_trajs + np.ones(self.num_points)*self.pnt_size
    self.scat_mark_trajs.set_array(self.cond_vector_mark_trajs)
    self.scat_mark_trajs.set_sizes(sizes)
    plt.pause(0.01)