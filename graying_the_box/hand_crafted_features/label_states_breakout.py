import numpy as np
import matplotlib.pyplot as plt

def label_states(states, screens, termination_mat, debug_mode, num_lives):

    im_size = np.sqrt(states.shape[1])
    states = np.reshape(states, (states.shape[0], im_size, im_size)).astype('int16')

    screens = np.reshape(np.transpose(screens), (3,210,160,-1))

    screens = np.transpose(screens,(3,1,2,0))

    # masks
    ball_mask = np.ones_like(screens[0])
    ball_mask[189:] = 0
    ball_mask[57:63] = 0

    ball_x_ = 80
    ball_y_ = 105

    td_mask = np.ones_like(screens[0])
    td_mask[189:] = 0
    td_mask[:25] = 0

    features = {
        'ball_pos': [[-1,-1],[-1,-1]],
        'ball_dir': [-1,-1],
        'racket': [-1,-1],
        'missing_bricks': [0,0],
        'hole': [0,0],
        'traj': [0,0],
        'time': [0,0]
    }

    if debug_mode:
        fig1 = plt.figure('screens')
        ax1 = fig1.add_subplot(111)
        screen_plt = ax1.imshow(screens[0], interpolation='none')
        plt.ion()
        plt.show()

    traj_id = 0
    time = 0
    strike_counter = 0
    s_ = screens[1]
    for i,s in enumerate(screens[2:]):

        #0. TD
        tdiff = (s - s_) * td_mask
        s_ = s

        row_ind, col_ind = np.nonzero(tdiff[:,:,0])
        ball_y = np.mean(row_ind)
        ball_x = np.mean(col_ind)

        # #1. ball location
        red_ch = s[:,:,0]
        # is_red = 255 * (red_ch == 200)
        # ball_filtered = np.zeros_like(s)
        # ball_filtered[:,:,0] = is_red
        # ball_filtered = ball_mask * ball_filtered
        #
        # row_ind, col_ind = np.nonzero(ball_filtered[:,:,0])
        #
        # ball_y = np.mean(row_ind)
        # ball_x = np.mean(col_ind)

        #2. ball direction
        ball_dir = 0 * (ball_x >= ball_x_ and ball_y >= ball_y_) +\
                   1 * (ball_x >= ball_x_ and ball_y < ball_y_) +\
                   2 * (ball_x < ball_x_ and ball_y >= ball_y_) +\
                   3 * (ball_x < ball_x_ and ball_y < ball_y_)

        ball_x_ = ball_x
        ball_y_ = ball_y

        #3. racket position
        is_red = 255 * (red_ch[190,8:-8] == 200)
        racket_x = np.mean(np.nonzero(is_red)) + 8

        #4. number of bricks
        z = red_ch[57:92,8:-8].flatten()
        is_brick = np.sum(1*(z>0) + 0*(z==0))

        missing_bricks = (len(z) - is_brick)/40.

        #5. holes
        brick_strip = red_ch[57:92,8:-8]
        brick_row_sum = brick_strip.sum(axis=0)
        has_hole = np.any((brick_row_sum==0))

        #6. traj_id
        if termination_mat[i] > 0:
            strike_counter+=1
            if strike_counter%num_lives==0:
                traj_id += 1
                time = 0
        time += 1

        if debug_mode:
            screen_plt.set_data(s)
            buf_line = ('Exqample %d: ball pos (x,y): (%0.2f, %0.2f), ball direct: %d, racket pos: (%0.2f), number of missing bricks: %d, has a hole: %d, traj id: %d, time: %d, st_cnt: %d') % \
                       (i, ball_x, ball_y, ball_dir, racket_x, missing_bricks, has_hole, traj_id, time, strike_counter)
            print buf_line
            plt.pause(0.001)

        # labels[i] = (ball_x, ball_y, ball_dir, racket_x, missing_bricks, has_hole, traj_id, time)

        features['ball_pos'].append([ball_x, ball_y])
        features['ball_dir'].append(ball_dir)
        features['racket'].append(racket_x)
        features['missing_bricks'].append(missing_bricks)
        features['hole'].append(has_hole)
        features['traj'].append(traj_id)
        features['time'].append(time)
        features['n_trajs'] = traj_id

    return features