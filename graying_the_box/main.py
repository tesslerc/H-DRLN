import sys

sys.path.append('hand_crafted_features')

from prepare_data import prepare_data
from vis_tool import VIS_TOOL

# Parameters
run_dir         = '/home/deep5/DQN_Shahar_Chen_oldpc/dqn_distill/'
num_frames      = 100000
game_id         = 0 # 0-breakout, 1-seaquest, 2-pacman
load_data       = 0
debug_mode      = 0
cluster_method  = 0 # 0-kmeans, 1-spectral_clustering, 2-EMHC (entropy minimization hierarchical clustering)
n_clusters      = 4
window_size     = 2
n_iters         = 8
entropy_iters   = 0

cluster_params = {
    'method': cluster_method,
    'n_clusters': n_clusters,
    'window_size': window_size,
    'n_iters': n_iters,
    'entropy_iters': entropy_iters
}
global_feats, hand_crafted_feats = prepare_data(game_id, run_dir, num_frames, load_data, debug_mode)

vis_tool = VIS_TOOL(global_feats=global_feats, hand_craft_feats=hand_crafted_feats, game_id=game_id, cluster_params=cluster_params)

vis_tool.show()
