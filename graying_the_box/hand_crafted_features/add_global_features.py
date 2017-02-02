import math

def add_buttons(self, global_feats):

    #############################
    # 3.1 global coloring buttons
    #############################
    self.COLORS = {}
    self.add_color_button([0.60, 0.95, 0.09, 0.02], 'value', global_feats['value'])
    self.add_color_button([0.70, 0.95, 0.09, 0.02], 'actions', global_feats['actions'])
    self.add_color_button([0.60, 0.92, 0.09, 0.02], 'rooms', global_feats['rooms'])
    self.add_color_button([0.60, 0.89, 0.09, 0.02], 'gauss_clust', global_feats['gauss_clust'])
    self.add_color_button([0.70, 0.89, 0.09, 0.02], 'TD', global_feats['TD'])
    self.add_color_button([0.60, 0.86, 0.09, 0.02], 'action repetition', global_feats['act_rep'])
    self.add_color_button([0.70, 0.86, 0.09, 0.02], 'reward', global_feats['reward'])

    for i in range((global_feats['single_clusters']).shape[0]):	
        self.add_color_button([0.80 + 0.10 * math.floor(i / 4), 0.95 - (i % 4) * 0.03, 0.09, 0.02], 'cluster_' + str(i), (global_feats['single_clusters'])[i])


    self.SLIDER_FUNCS = []
    self.CHECK_BUTTONS = []
