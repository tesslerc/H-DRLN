import cPickle as pickle
from prepare_global_features import prepare_global_features

def prepare_data(game_id, run_dir, num_frames, load_data, debug_mode):
    # 1. switch games TEST
    if game_id == 0: #'breakout'
        num_actions = 10
        num_lives   = 1
        data_dir = run_dir
        from label_states_breakout import label_states
        grad_thresh = 0.1

    elif game_id == 1: #'seaquest'
        num_actions = 18
        num_lives   = 3
        data_dir = 'data/' + 'seaquest' + '/' + run_dir + '/'
        from label_states_seaquest import label_states
        grad_thresh = 0.05

    elif game_id == 2: #'pacman'
        num_actions = 5
        num_lives   = 3
        data_dir = 'data/' + 'pacman' + '/' + run_dir + '/'
        from label_states_packman import label_states
        grad_thresh = 0.05

    # 2. load data
    if load_data:
        # 2.1 global features
        global_feats = pickle.load(file(data_dir + 'global_features.bin','rb'))

        # 2.2 hand craft features
        hand_craft_feats = pickle.load(file(data_dir + 'hand_craft_features.bin','rb'))

    # 3. prepare data
    else:
        # 3.1 global features
        global_feats = prepare_global_features(data_dir, num_frames, num_actions, num_lives, grad_thresh)
        # pickle.dump(global_feats,file(data_dir + 'global_features.bin','wb'))

        # 3.2 hand craft features
        hand_craft_feats = label_states(global_feats['states'], global_feats['screens'], global_feats['termination'], debug_mode=debug_mode, num_lives=num_lives)
        # pickle.dump(hand_craft_feats,file(data_dir + 'hand_craft_features.bin','wb'))

    return global_feats, hand_craft_feats
