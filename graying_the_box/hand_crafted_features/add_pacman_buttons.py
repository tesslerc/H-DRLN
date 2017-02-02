import matplotlib.pyplot as plt
import numpy as np
import math

def add_game_buttons(self, top_y=0.69, x_left = 0.68):
    self.add_slider_button([x_left, top_y, 0.08, 0.01], 'player_x', 0, 160)
    self.add_slider_button([x_left, top_y-0.04, 0.08, 0.01], 'player_y', 0, 210)
    self.add_slider_button([x_left, top_y-0.08, 0.08, 0.01], 'bricks', 0, 121)
    self.add_slider_button([x_left, top_y-0.12, 0.08, 0.01], 'enemy_distance', 0, 160+210)
    self.add_slider_button([x_left, top_y-0.16, 0.08, 0.01], 'lives', 0, 3)
    self.add_check_button([x_left, top_y-0.22, 0.06, 0.03], 'ghost', ('no-ghost','ghost'), (True,True))
    self.add_check_button([x_left, top_y-0.26, 0.06, 0.03], 'box', ('no-box','box'), (True,True))
    self.add_check_button([x_left, top_y-0.35, 0.07, 0.08], 'player_dir', ('stand','right','left','bottom','top'), (True, True, True, True, True))
    self.player_dir_mat = np.zeros(shape=(self.num_points,5), dtype='bool')
    for i,dir in enumerate(self.hand_craft_feats['player_dir']):
        self.player_dir_mat[i,int(dir)] = 1

def update_cond_vector(self):
    player_x = np.asarray([row[0] for row in self.hand_craft_feats['player_pos']])
    player_y = np.asarray([row[1] for row in self.hand_craft_feats['player_pos']])
    player_dir = np.asarray(self.hand_craft_feats['player_dir'])
    bricks = np.asarray(self.hand_craft_feats['bricks'])
    nb_lives = np.asarray(self.hand_craft_feats['lives'])
    ghost_mode = np.asarray(self.hand_craft_feats['ghost'])
    enemies_dist = np.asarray(self.hand_craft_feats['enemy_distance'])
    bonus_box = np.asarray(self.hand_craft_feats['box'])

    self.cond_vector =  (player_x >= self.player_x_min) * (player_x <= self.player_x_max) * \
                        (player_y >= self.player_y_min) * (player_y <= self.player_y_max) * \
                        (bricks >= self.bricks_min) * (bricks <= self.bricks_max) * \
                        (enemies_dist >= self.enemy_distance_min) * (enemies_dist <= self.enemy_distance_max) * \
                        (nb_lives >= self.lives_min) * (nb_lives <= self.lives_max)

    # player dir
    dirs_mask = np.zeros_like(self.cond_vector)
    for i,val in enumerate(self.player_dir_check_button.get_status()):
        if val:
            dirs_mask += self.player_dir_mat[:,i]

    self.cond_vector = self.cond_vector * dirs_mask

    # ghost mode
    if self.ghost_check_button.get_status()[0] == 0: # filter out states with "no-ghost"
        self.cond_vector = self.cond_vector * ghost_mode
    elif self.ghost_check_button.get_status()[1] == 0: # filter out states with "ghost" mode
        self.cond_vector = self.cond_vector * (1-ghost_mode)

    # bonus box
    if self.box_check_button.get_status()[0] == 0: # filter out states with "no-box"
        self.cond_vector = self.cond_vector * bonus_box
    elif self.box_check_button.get_status()[1] == 0: # filter out states with "box"
        self.cond_vector = self.cond_vector * (1-bonus_box)

    self.cond_vector = self.cond_vector.astype(int)

def value_colored_frame(self):
    # value colored frame
    fig5 = plt.figure('value colored pacman frame')
    ax_5 = fig5.add_subplot(111)

    v_mat = np.zeros((210,160))
    count_mat = np.zeros((210,160))
    player_x = self.state_labels[:,0]
    player_y = self.state_labels[:,1]

    for x,y,v in zip(player_x, player_y, self.V):
        if math.isnan(x) or math.isnan(y):
            continue
        v_mat[int(y),int(x)] += v
        count_mat[int(y),int(x)] += 1

    v_mat = v_mat / count_mat

    ax_5.imshow(v_mat, interpolation='spline36')