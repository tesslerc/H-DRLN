import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage

def label_states(states, screens, termination_mat, debug_mode, num_lives):

    screens = np.reshape(np.transpose(screens), (3,210,160,-1))

    screens = np.transpose(screens,(3,1,2,0))

    features = {
        'player_pos': [[-1,-1],[-1,-1]],
        'player_dir': [-1,-1],
        'bricks': [-1,-1],
        'lives': [3,3],
        'sweets': [[1,1,1,1],[1,1,1,1]],
        'ghost': [-1,-1],
        'enemy_distance': [-1,-1],
        'box': [0,0]
    }

    # player mask
    player_mask = np.ones((210,160))
    player_mask[91:96,78:82] = 0

    # number of bricks
    brown_frame =  1 * (screens[0,:,:,0]==162)

    small_brick_mask = np.asarray([[45,45, 45, 45, 45, 45],
                                   [45,162,162,162,162,45],
                                   [45,162,162,162,162,45],
                                   [45,45, 45, 45, 45, 45]])

    wide_brick_mask = np.asarray([[45,45, 45, 45, 45, 45, 45, 45, 45, 45],
                                  [45,162,162,162,162,162,162,162,162,45],
                                  [45,162,162,162,162,162,162,162,162,45],
                                  [45,45, 45, 45, 45, 45, 45, 45, 45, 45]])

    # sweets
    sweets_y = [22,142,22,142]
    sweets_x = [6,6,151,151]

    hist_depth = 16
    sweets_mat = np.zeros((hist_depth,4))
    sweets_vec_ = np.zeros((4,))
    bonus_times = [[] for i in range(4)]

    # left bottom brick mask = [139:149,5:11,0]
    # top left brick mask = [19:29,5:11,0]
    # top right brick mask = [19:29,149:155,0]
    # bottom right brick mask = [139:149,149:155,0]
    brick_color_1 = 45 * np.ones((10,6))
    brick_color_1[1:-1,1:-1] = 180
    brick_color_2 = 45 * np.ones((10,6))
    brick_color_2[1:-1,1:-1] = 149
    brick_color_3 = 45 * np.ones((10,6))
    brick_color_3[1:-1,1:-1] = 212
    brick_color_4 = 45 * np.ones((10,6))
    brick_color_4[1:-1,1:-1] = 232
    brick_color_5 = 45 * np.ones((10,6))
    brick_color_5[1:-1,1:-1] = 204
    #enemies mask
    enemies_mask = np.ones((210,160))
    enemies_mask[20:28,6:10] = 0
    enemies_mask[140:148,6:10] = 0
    enemies_mask[20:28,150:154] = 0
    enemies_mask[140:148,150:154] = 0

    # bonus box
    bonus_color_mask = 162 * np.ones((7,6))
    bonus_color_mask[1:-1,1:-1] = 210

    if debug_mode:
        fig1 = plt.figure('screens')
        ax1 = fig1.add_subplot(111)
        screen_plt = ax1.imshow(screens[0], interpolation='none')
        fig2 = plt.figure('enemies')
        ax2 = fig2.add_subplot(111)
        brown_plt = ax2.imshow(brown_frame)
        # fig3 = plt.figure('brick heat map')
        # ax3 = fig3.add_subplot(111)
        # bricks_plt = ax3.imshow(brown_frame, interpolation='none')
        plt.ion()
        plt.show()

    player_x_ = 79
    player_y_ = 133

    restarted_flag = 0
    ttime = 0

    for i,s in enumerate(screens[2:]):
    # for i,s in enumerate(screens[750:]):

        # 1. player location
        player_frame = s[:,:,0]==210 * player_mask
        row_ind, col_ind = np.nonzero(player_frame)
        player_y = np.mean(row_ind)
        player_x = np.mean(col_ind)

        # 2. player direction
        dx = player_x - player_x_
        dy = player_y - player_y_

        player_dir = 0
        if abs(dx) >= abs(dy):
            if dx > 0:
                player_dir = 1
            elif dx<0:
                player_dir = 2
        else:
            if dy > 0:
                player_dir = 3
            elif dy < 0:
                player_dir = 4

        player_x_ = player_x
        player_y_ = player_y

        # 3. number of bricks to eat
        # approximated (fast)
        brown_frame =  1 * (s[:,:,0]==162)
        brown_sum = np.sum(brown_frame)
        nb_bricks_apprx = (brown_sum - 6042)/8.
        nb_bricks_apprx = np.maximum(nb_bricks_apprx,0)

        # exact (slow)
        # nb_bricks = 0
        # for r in range(160):
        #     for c in range (150):
        #         small_slice = s[r:r+4,c:c+6,0]
        #         wide_slice = s[r:r+4,c:c+10,0]
        #         if (small_slice == small_brick_mask).all():
        #             nb_bricks += 1
        #         elif (wide_slice == wide_brick_mask).all():
        #             nb_bricks += 1

        # 4. number of lives
        lives_strip = s[184:189,:,0]
        _, nb_lives = scipy.ndimage.label(lives_strip)

        # 5. sweets
        # look for bottom left brick
        sweets_mat[i%hist_depth,0] =    1 * np.all(s[139:149,5:11,0] == brick_color_1) + \
                                1 * np.all(s[139:149,5:11,0] == brick_color_2) + \
                                1 * np.all(s[139:149,5:11,0] == brick_color_3) + \
                                1 * np.all(s[139:149,5:11,0] == brick_color_4) + \
                                1 * np.all(s[139:149,5:11,0] == brick_color_5)


        # look for top right brick
        sweets_mat[i%hist_depth,1] =    1 * np.all(s[19:29,149:155,0] == brick_color_1) + \
                                1 * np.all(s[19:29,149:155,0] == brick_color_2) + \
                                1 * np.all(s[19:29,149:155,0] == brick_color_3) + \
                                1 * np.all(s[19:29,149:155,0] == brick_color_4) + \
                                1 * np.all(s[19:29,149:155,0] == brick_color_5)

        # look for bottom right brick
        sweets_mat[i%hist_depth,2] =    1 * np.all(s[139:149,149:155,0] == brick_color_1) + \
                                1 * np.all(s[139:149,149:155,0] == brick_color_2) + \
                                1 * np.all(s[139:149,149:155,0] == brick_color_3) + \
                                1 * np.all(s[139:149,149:155,0] == brick_color_4) + \
                                1 * np.all(s[139:149,149:155,0] == brick_color_5)

        # look for top left brick
        sweets_mat[i%hist_depth,3] =    1 * np.all(s[19:29,5:11,0] == brick_color_1) + \
                                1 * np.all(s[19:29,5:11,0] == brick_color_2) + \
                                1 * np.all(s[19:29,5:11,0] == brick_color_3) + \
                                1 * np.all(s[19:29,5:11,0] == brick_color_4) + \
                                1 * np.all(s[19:29,5:11,0] == brick_color_5)

        sweets_vec = 1 * np.any(sweets_mat, axis=0)

        if np.all(sweets_vec_ == 1) and restarted_flag == 0:
            ttime = 0
            restarted_flag = 1
        if not np.all(sweets_vec_ == 1):
            restarted_flag = 0
        ttime += 1

        # 6. ghost mode
        ghost_mode = np.any(1 * (s[:,:,0] == 149) + 1 * (s[:,:,0] == 212))

        # 7. enemies map
        enemies_map = 1 * (s[:,:,0] == 180) + \
                      1 * (s[:,:,0] == 149) + \
                      1 * (s[:,:,0] == 212) + \
                      1 * (s[:,:,0] == 128) + \
                      1 * (s[:,:,0] == 232) + \
                      1 * (s[:,:,0] == 204)
        enemies_map = enemies_map * enemies_mask

        enemies_dilate_map = scipy.ndimage.binary_dilation(enemies_map, iterations=1)
        labeled_array, nb_enemies = scipy.ndimage.label(enemies_dilate_map)

        min_dist = 999
        for j in range(nb_enemies):
            enemy_j_rows, enemy_j_cols = np.nonzero(labeled_array==j+1)
            e_j_y = np.mean(enemy_j_rows)
            e_j_x = np.mean(enemy_j_cols)
            dist_j = abs(player_x - e_j_x) + abs(player_y - e_j_y)
            if dist_j < min_dist:
                min_dist = dist_j

        # 8. bonus box
        # has_box = np.any(s[91:96,78:82,0]==210)
        has_box = 1 * np.all(s[90:97,77:83,0] == bonus_color_mask)

        #6. bonus bricks option
        # bottom left brick
        if sweets_vec[0] == 0 and sweets_vec_[0] == 1:
            bonus_times[0].append(ttime)
        # top right brick
        elif sweets_vec[1] == 0 and sweets_vec_[1] == 1:
            bonus_times[1].append(ttime)
        # bottom right brick
        elif sweets_vec[2] == 0 and sweets_vec_[2] == 1:
            bonus_times[2].append(ttime)
        # top left brick
        elif sweets_vec[3] == 0 and sweets_vec_[3] == 1:
            bonus_times[3].append(ttime)

        sweets_vec_ = sweets_vec

        if debug_mode:
            screen_plt.set_data(s)
            brown_plt.set_data(brown_frame)
            # bricks_plt.set_data(brown_frame)
            buf_line = ('Exqample %d: Player (x,y): (%0.2f,%0.2f), Player dir: %d, number of bricks: %d, number of lives: %d, sweets_vec: %s, ghost mode: %d, minimal enemy distance: %0.2f, bonus box: %d, time: %d') %\
                        (i, player_x, player_y, player_dir, nb_bricks_apprx, nb_lives, sweets_vec, ghost_mode, min_dist, has_box, ttime)
            print buf_line
            plt.pause(0.01)

        features['player_pos'].append([player_x, player_y])
        features['player_dir'].append(player_dir)
        features['bricks'].append(nb_bricks_apprx)
        features['sweets'].append(sweets_vec)
        features['ghost'].append(ghost_mode)
        features['lives'].append(nb_lives)
        features['enemy_distance'].append(min_dist)
        features['box'].append(has_box)

    # histogram of sweets cllection times
    if 0:
        brick_bl_times = np.asarray(bonus_times[0])
        brick_tr_times = np.asarray(bonus_times[1])
        brick_br_times = np.asarray(bonus_times[2])
        brick_tl_times = np.asarray(bonus_times[3])

        h_bl = plt.hist(brick_bl_times, bins = 70, range=(0,800), normed=1, facecolor='black', alpha=0.75, label='bottom left')
        h_tr = plt.hist(brick_tr_times, bins = 70, range=(0,800), normed=1, facecolor='red', alpha=0.75, label='top right')
        h_br = plt.hist(brick_br_times, bins = 70, range=(0,800), normed=1, facecolor='green', alpha=0.75, label='bottom right')
        h_tl = plt.hist(brick_tl_times, bins = 70, range=(0,800), normed=1, facecolor='blue', alpha=0.75, label='top left')

        plt.legend(prop={'size':20})
        plt.tick_params(axis='both',which='both',left='off',right='off', labelsize=20)
        plt.locator_params(axis='x',nbins=4)
        plt.locator_params(axis='y',nbins=8)

        plt.show()

    return features