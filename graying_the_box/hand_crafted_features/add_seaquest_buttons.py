import  numpy as np

def add_game_buttons(self, top_y=0.69, x_left = 0.68):
    self.add_slider_button([x_left, top_y , 0.08, 0.01], 'shooter_x', 0, 160)
    self.add_slider_button([x_left, top_y-0.04 , 0.08, 0.01], 'shooter_y', 0, 210)
    self.add_slider_button([x_left, top_y-0.08 , 0.08, 0.01], 'oxygen', 0, 1)
    self.add_slider_button([x_left, top_y-0.12 , 0.08, 0.01], 'divers', 0, 6)
    self.add_slider_button([x_left, top_y-0.16 , 0.08, 0.01], 'taken_divers', 0, 3)
    self.add_slider_button([x_left, top_y-0.20 , 0.08, 0.01], 'enemies', 0, 8)
    self.add_slider_button([x_left, top_y-0.24 , 0.08, 0.01], 'lives', 0, 3)
    self.add_check_button([x_left, top_y-0.32 , 0.06, 0.04], 'shooter_dir', ('dont-care','down','up'), (True, True, True))

    self.shooter_dir_mat = np.zeros(shape=(self.num_points,3), dtype='bool')

    for i,dir in enumerate(self.hand_craft_feats['shooter_dir']):
        self.shooter_dir_mat[i,int(dir)] = 1

def update_cond_vector(self):
    shooter_x = np.asarray([row[0] for row in self.hand_craft_feats['shooter_pos']])
    shooter_y = np.asarray([row[1] for row in self.hand_craft_feats['shooter_pos']])
    shooter_dir = self.hand_craft_feats['shooter_dir']
    oxygen = np.asarray(self.hand_craft_feats['oxygen'])
    nb_divers = np.asarray(self.hand_craft_feats['divers'])
    nb_taken_divers = np.asarray(self.hand_craft_feats['taken_divers'])
    nb_enemies = np.asarray(self.hand_craft_feats['enemies'])
    nb_lives = np.asarray(self.hand_craft_feats['lives'])

    self.cond_vector =  (shooter_x >= self.shooter_x_min) * (shooter_x <= self.shooter_x_max) * \
                        (shooter_y >= self.shooter_y_min) * (shooter_y <= self.shooter_y_max) * \
                        (oxygen >= self.oxygen_min) * (oxygen <= self.oxygen_max) * \
                        (nb_divers >= self.divers_min) * (nb_divers <= self.divers_max) * \
                        (nb_taken_divers >= self.taken_divers_min) * (nb_taken_divers <= self.taken_divers_max) *\
                        (nb_enemies >= self.enemies_min) * (nb_enemies <= self.enemies_max) * \
                        (nb_lives >= self.lives_min) * (nb_lives <= self.lives_max)

    # shtr dir
    dirs_mask = np.zeros_like(self.cond_vector)
    for i,val in enumerate(self.shooter_dir_check_button.get_status()):
        if val:
            dirs_mask += self.shooter_dir_mat[:,i]

    self.cond_vector = self.cond_vector * dirs_mask

    self.cond_vector = self.cond_vector.astype(int)
