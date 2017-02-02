import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage

def label_states(states, screens, termination_mat, debug_mode, num_lives):

    screens = np.reshape(np.transpose(screens), (3,210,160,-1))

    screens = np.transpose(screens,(3,1,2,0))

    features = {
        'shooter_pos': [[-1,-1],[-1,-1]],
        'shooter_dir': [-1,-1],
        'racket': [-1,-1],
        'oxygen': [0,0],
        'divers': [0,0],
        'taken_divers': [0,0],
        'enemies': [0,0],
        'lives': [3,3],
    }

    # shooter convolution
    yellow_frame = screens[0,:,:,0]==187
    shtr_mask = np.asarray([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    shooter_conv_map = scipy.signal.convolve2d(yellow_frame, shtr_mask, mode='same')
    shtr_y_, shtr_x_ = np.unravel_index(np.argmax(shooter_conv_map),(210,160))

    #divers
    divers_frame = np.ones_like(screens[0,:,:,0])
    divers_frame[:160,:] = 1 * (screens[0,:160,:,0]==66)

    if debug_mode:
        fig1 = plt.figure('screens')
        ax1 = fig1.add_subplot(111)
        screen_plt = ax1.imshow(screens[0], interpolation='none')
        fig2 = plt.figure('divers')
        ax2 = fig2.add_subplot(111)
        divers_plt = ax2.imshow(divers_frame)
        # fig3 = plt.figure('enemies conv')
        # ax3 = fig3.add_subplot(111)
        # diver_conv_plt = ax3.imshow(diver_conv_map, interpolation='none')
        plt.ion()
        plt.show()

    shtr_dir_ = 0
    shtr_dir__ = 0

    for i,s in enumerate(screens[2:]):
    # for i,s in enumerate(screens[13210:]):

        # 1. shooter location
        yellow_frame = s[:,:,0]==187
        shooter_conv_map = scipy.signal.convolve2d(yellow_frame, shtr_mask,mode='same')
        shtr_y, shtr_x = np.unravel_index(np.argmax(shooter_conv_map),(210,160))

        # if shtr_x==0 :
        #     shtr_x = -1
        # if shtr_y==0:
        #     shtr_y = -1

        # 2. shooter direction
        shtr_dir = 1 * (shtr_y >= shtr_y_) +\
                   2 * (shtr_y < shtr_y_)

        coherent = (shtr_dir == shtr_dir_) * (shtr_dir == shtr_dir__)
        shtr_dir__ = shtr_dir_
        shtr_dir_ = shtr_dir

        shtr_dir = shtr_dir * coherent

        shtr_x_ = shtr_x
        shtr_y_ = shtr_y

        # 3. oxygen
        oxgn_line = s[172,49:111,0]
        oxgn_lvl = np.sum(1*(oxgn_line==214))/float((111-49))

        # 4. avaiable divers
        divers_frame = 0 * divers_frame
        divers_frame[:160,:] = 1 * (s[:160,:,0] == 66)
        erote_map = scipy.ndimage.binary_erosion(divers_frame, structure=np.ones((2,2)), iterations=1)
        diver_conv_map = scipy.ndimage.binary_dilation(erote_map, iterations=3)
        _, nb_divers = scipy.ndimage.label(diver_conv_map)

        # 5. taken divers
        # divers_frame = 0 * divers_frame
        # divers_frame[178:188,:] = 1 * (s[178:188,:,0] == 24)
        # diver_conv_map = scipy.ndimage.binary_dilation(divers_frame, iterations=1)
        # _, nb_taken_divers = scipy.ndimage.label(diver_conv_map)
        nb_taken_divers = np.sum(1 * s[178,:,0] == 24)

        # 6. enemies
        enemies_frame = 0 * divers_frame
        enemies_frame[:160,:] = 1 * (s[:160,:,0] == 92) + \
                                2 * (s[:160,:,0] == 160) + \
                                3 * (s[:160,:,0] == 170) + \
                                4 * (s[:160,:,0] == 198)

        enemies_conv_map = scipy.ndimage.binary_dilation(enemies_frame, iterations=2)
        _, nb_enemies = scipy.ndimage.label(enemies_conv_map)

        # lives
        lives_slice = 1 * (s[18:30,:,0] == 210)
        _, nb_lives = scipy.ndimage.label(lives_slice)


        if debug_mode:
            screen_plt.set_data(s)
            # divers_plt.set_data(diver_conv_map)
            # diver_conv_plt.set_data(diver_conv_map)
            buf_line = ('Exqample %d: shooter (x,y): (%0.2f, %0.2f), shooter dir: %d, oxygen level: %0.2f, divers: %d, taken divers: %d, enemies: %d, lives: %d') %\
                        (i, shtr_x, shtr_y, shtr_dir, oxgn_lvl, nb_divers, nb_taken_divers, nb_enemies, nb_lives)
            print buf_line
            plt.pause(0.01)

        features['shooter_pos'].append([shtr_x, shtr_y])
        features['shooter_dir'].append(shtr_dir)
        features['oxygen'].append(oxgn_lvl)
        features['divers'].append(nb_divers)
        features['taken_divers'].append(nb_taken_divers)
        features['enemies'].append(nb_enemies)
        features['lives'].append(nb_lives)

    return features