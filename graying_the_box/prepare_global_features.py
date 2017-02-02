import common
import numpy as np
import h5py

def prepare_global_features(data_dir, num_frames, num_actions, num_lives, grad_thresh):

    print "Preparing global features... "

    # load hdf5 files
    print "Loading from hdf5 files... "
    states_hd5f = common.load_hdf5('statesClean', data_dir, num_frames)
    screens_hdf5 = np.ones((3, states_hd5f.shape[0], states_hd5f.shape[1]))
    screens_hdf5[0, :, :] = states_hd5f[:, :]
    screens_hdf5[1, :, :] = states_hd5f[:, :]
    screens_hdf5[2, :, :] = states_hd5f[:, :]
    #termination_hdf5 = common.load_hdf5('terminationClean', data_dir, num_frames)
    lowd_activation_hdf5 = common.load_hdf5('lowd_activations_800', data_dir, num_frames)
    lowd_activation_3d_hdf5 = np.zeros((lowd_activation_hdf5.shape[0], 3))  # common.load_hdf5('lowd_activations3d', data_dir, num_frames)
    q_hdf5 = common.load_hdf5('qvalsClean', data_dir, num_frames)
    a_hdf5 = np.array(common.load_hdf5('actionsClean', data_dir, num_frames))
    reward_hdf5 = common.load_hdf5('rewardClean', data_dir, num_frames)
    with h5py.File('/home/deep5/DQN_Shahar_Chen_oldpc/dqn_distill/tsneMatch.h5', 'r') as hf:
        gaus_clust = (np.array(hf.get('data')))
    with h5py.File('/home/deep5/DQN_Shahar_Chen_oldpc/dqn_distill/tsneRooms.h5', 'r') as hf:
        rooms = (np.array(hf.get('data')))
    with h5py.File('/home/deep5/DQN_Shahar_Chen_oldpc/dqn_distill/tsneData.h5', 'r') as hf:
        single_clusters = (np.array(hf.get('data')))

    # grads_hdf5 = common.load_hdf5('grads', data_dir, num_frames)
    grads_hdf5 = states_hd5f  # take state image instead of saliency image until we have saliency for all data

    termination_hdf5 = np.sign(np.array(reward_hdf5)) + 1


    # 9. get Q and action, translate to numpy
    V                   = np.zeros(shape=num_frames)
    Q                   = np.zeros(shape=(num_frames,num_actions))
    tsne                = np.zeros(shape=(num_frames,2))
    a                   = np.zeros(shape=num_frames)
    term                = np.zeros(shape=num_frames)
    reward              = np.zeros(shape=num_frames)
    TD                  = np.zeros(shape=num_frames)
    tsne3d              = np.zeros(shape=(num_frames,3))
    tsne3d_next         = np.zeros(shape=(num_frames,3))
    time                = np.zeros(shape=num_frames)
    act_rep             = np.zeros(shape=num_frames)
    trajectory_index    = np.zeros(shape=num_frames)
    tsne3d_norm         = np.zeros(shape=num_frames)

    counter = 0
    term_counter = 0
    trajectory_counter = 1

    for i in range(1,num_frames-1):
        V[i] = q_hdf5[i]
        a[i] = float(a_hdf5[i])-1
        reward[i] = reward_hdf5[i]
        term[i] = termination_hdf5[i]
        Q[i] = q_hdf5[i]
        tsne[i] = lowd_activation_hdf5[i]
        tsne3d[i] = lowd_activation_3d_hdf5[i]
        tsne3d_next[i] = lowd_activation_3d_hdf5[i+1]-lowd_activation_3d_hdf5[i]
        tsne3d_norm[i] = np.linalg.norm(tsne3d_next[i])
        TD[i] = abs(Q[i-1,int(a[i-1])]-0.99*(Q[i,int(a[i])]+reward[i-1]))

        # calculate time and trajecttory index
        time[i] = counter
        trajectory_index[i] = trajectory_counter

        if (term[i] > 0):#the fifth terminal
            term_counter += 1
            if term_counter % num_lives == 0:
                counter = 0
                trajectory_counter+=1
            else:
                counter += 1
        else:
            counter += 1

    Advantage = V-Q.T
    risk = np.sum(Advantage,axis=0)
    term_binary = term
    term_binary[np.nonzero(term_binary!=0)]=1

    global_feats = {
        'tsne': tsne,
        'states':states_hd5f,
        'screens':screens_hdf5,
        'value':V,
        'actions':a,
        'termination':term_binary,
        'risk':risk,
        'tsne3d':tsne3d,
        'tsne3d_norm':tsne3d_norm,
        'Advantage':Advantage,
        'time':time,
        'TD':TD,
        'reward':reward,
        'act_rep':act_rep,
        'tsne3d_next':tsne3d_next,
        'grads':grads_hdf5,
        'trajectory_index':trajectory_index,
        'data_dir':data_dir,
        'gauss_clust':gaus_clust,
        'rooms':rooms,
        'single_clusters':single_clusters
        }

    return global_feats
