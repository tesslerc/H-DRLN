import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from smdp import SMDP
from digraph import draw_transition_table

def load_hdf5(object_name, path, num_frames=None):
    print "    Loading " +  object_name
    obj_file = h5py.File(path + object_name + '.h5', 'r')
    obj_mat = obj_file['data']
    return obj_mat[:num_frames]

def save_hdf5(object_name, path, object):
    print "    Saving " +  object_name
    with h5py.File(path + object_name +'.h5', 'w') as hf:
        hf.create_dataset('data', data=object)

def create_trajectory_data(self, reward, traj_ids):
    self.traj_list = []
    n_trajs = np.asarray(traj_ids).max() - 1
    for traj_id in xrange(n_trajs):
        traj_point_mask = np.asarray(traj_ids)==traj_id
        traj_labels = self.clustering_labels[np.nonzero(traj_point_mask)]
        traj_length = np.sum(traj_point_mask)
        R = np.sum(reward[np.nonzero(traj_point_mask)])
        moves = []
        for i in xrange(traj_length-1):
            if traj_labels[i] != traj_labels[i+1] and self.smdp.P[traj_labels[i],traj_labels[i+1]]>0:
                moves.append((traj_labels[i],traj_labels[i+1]))
        self.traj_list.append(
            {
                'R': R.astype(np.float32),
                'length': traj_length,
                'moves': moves,
                'points': traj_point_mask,
             }
        )

def visualize(self):
    m = self.smdp
    m.evaluate_greedy_policy(0)

    plt.figure('Value function consistency')
    ax_1 = plt.subplot('211')
    ax_1.plot((m.v_dqn-m.v_dqn.mean())/m.v_dqn.std(),'b',label='DQN')
    ax_1.plot((m.v_smdp-m.v_smdp.mean())/m.v_smdp.std(),'r',label='Semi-MDP')
    ax_1.plot((m.v_greedy-m.v_greedy.mean())/m.v_greedy.std(),'g',label='Greedy policy')
    ax_1.legend()
    ax_1.set_xlabel('cluster index')
    ax_1.set_ylabel('value')
    title1 = 'White - n_clusters: %d ' % m.n_clusters
    ax_1.set_title(title1)

    ax_2 = plt.subplot('212')
    ax_2.plot(m.v_dqn,'b',label='DQN')
    ax_2.plot(m.v_smdp,'r',label='Semi-MDP')
    ax_2.plot(m.v_greedy,'g',label='Greedy policy')
    ax_2.legend()
    title2 = 'Reg - n_clusters: %d ' % m.n_clusters
    ax_2.set_title(title2)
    ax_2.set_xlabel('cluster index')
    ax_2.set_ylabel('value')

    # ll = np.arange(start=0.0,stop=0.05,step=0.05)
    # title = 'greedy policy improvement'
    # plt.figure(title)
    # plt.title(title)
    # plt.plot(m.v_smdp,'r--',label='smdp')
    # plt.plot(m.v_dqn,'g--',label='dqn')
    #
    # for l in ll:
    #     m.evaluate_greedy_policy(l)
    #     plt.plot(m.v_greedy,c=np.random.rand(3,1),label=l)
    #     # plt.plot(m.v_greedy,'b',label='improved policy')
    # plt.legend()
    # plt.show(block=True)

    plt.figure('Greedy policy consistency')
    # PI
    ax_1 = plt.subplot('211')
    markerline, stemlines, baseline = ax_1.stem(self.state_pi_correlation,'b-.')
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    ax_1.set_title('Policy Improvement - Reward Correlation')
    ax_1.set_xlabel('Cluster index')

    # Top-greedy correlation vs. bottom-Greedy correlation
    ax_2 = plt.subplot('212')
    ax_2.plot(np.asarray([x[0] for x in self.top_greedy_vec],dtype=np.float32)/len(self.traj_list), np.asarray([x[1] for x in self.top_greedy_vec]), '-b')
    ax_2.plot(np.asarray([x[0] for x in self.bottom_greedy_vec],dtype=np.float32)/len(self.traj_list), np.asarray([x[1] for x in self.bottom_greedy_vec]), '-r')
    ax_2.set_title('Greedy Policy Weight')
    ax_2.set_xlabel('Percentage of Extermum trajectories used')

    # # Top-Bottom discrepency
    # ax = plt.subplot('412')
    # markerline, stemlines, baseline = ax.stem(self.tb_trajs_discr['tb_disc'],'b-.')
    # plt.setp(markerline, 'markerfacecolor', 'b')
    # plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    # ax.set_title('Top Bottom Discrepency')
    # # Bottom-Top discrepency
    # ax = plt.subplot('413')
    # markerline, stemlines, baseline = ax.stem(self.tb_trajs_discr['bt_disc'],'b-.')
    # plt.setp(markerline, 'markerfacecolor', 'b')
    # plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    # ax.set_title('Bottom Top Discrepency')

    # value_diff =  (m.v_greedy[:,0]-m.v_greedy[:,0].mean())/m.v_greedy[:,0].std() - (m.v_smdp-m.v_smdp.mean())/m.v_smdp.std()
    # value_diff =  (m.v_greedy[:,0]-m.v_smdp)/m.v_smdp
    # plt.plot(value_diff,'r--',label='V greedy - V smdp')
    # corr = np.corrcoef(value_diff, self.state_pi_correlation)[0,1]
    # plt.legend()

    # draw_transition_table(transition_table=self.tb_trajs_discr['top_model'].P, cluster_centers=self.cluster_centers,
    #                       meanscreen=self.meanscreen, tsne=self.data_t.T, color=self.color, black_edges=self.tb_trajs_discr['top_model'].edges, title='Top Model')
    #
    # draw_transition_table(transition_table=self.tb_trajs_discr['bottom_model'].P, cluster_centers=self.cluster_centers,
    #                       meanscreen=self.meanscreen, tsne=self.data_t.T, color=self.color, black_edges=self.tb_trajs_discr['bottom_model'].edges, title='Bottom Model')

    plt.show()

def reward_policy_correlation(traj_list, policy, smdp):
    N = len(traj_list)
    corr = np.zeros(smdp.n_clusters)

    for c in xrange(smdp.n_clusters):
        rewards = np.zeros(N)
        good_moves = np.zeros(N)
        for t_ind,traj in enumerate(traj_list):
            rewards[t_ind] = traj['R']
            count = 0
            total_cluster_visitations = 0
            for move in traj['moves']:
                if move[0] != c:
                    continue
                total_cluster_visitations += 1
                if policy[int(move[0])][1]==move[1]: # policy[i] = pi_i .
                    count += 1
            if total_cluster_visitations == 0:
                continue
            good_moves[t_ind] = float(count)/total_cluster_visitations

        corr[c] = np.corrcoef(rewards, good_moves)[0,1]

    return corr

def draw_skills(self,n_clusters,plt):
    plt.figure('Skills')
    subplot_ind = 0
    N_sub_plots = 12
    for cluster_ind in xrange(n_clusters):
        for i,l in enumerate(self.smdp.skill_indices[cluster_ind]):
            subplot_ind += 1
            ax = plt.subplot(N_sub_plots,N_sub_plots,subplot_ind)
            ax.set_title('Cluster %d, Skill %d' %(cluster_ind,self.smdp.skill_list[cluster_ind][i]),fontsize=8)#per skill
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            skill_hist = plt.hist(self.smdp.skill_time[cluster_ind][i], bins=100) #per skill
            # most_likely_length = np.round(skill_hist[1][np.argmax(skill_hist[0])])
            skill_mean_screen = calc_cluster_im(self,l)#per skill
            subplot_ind += 1
            ax = plt.subplot(N_sub_plots,N_sub_plots,subplot_ind)
            ax.imshow(skill_mean_screen)#per skill
            ax.set_title('Cluster %d, Skill %d' %(cluster_ind,self.smdp.skill_list[cluster_ind][i]),fontsize=8)#per skill
            ax.axis('off')


def extermum_trajs_discrepency(traj_list, labels, termination, rewards, values, n_clusters, pi_analytic, d=30):

    def unite_trajs_mask(traj_list, traj_indices, n_points):
        unite_mask = np.zeros(n_points, dtype=bool)
        for t_ind in traj_indices:
            unite_mask = np.logical_or(unite_mask, traj_list[t_ind]['points'])
        return unite_mask

    n_trajs = len(traj_list)
    n_points = len(traj_list[0]['points'])
    reward = np.zeros(n_trajs)

    # sort trajectories by reward
    for t_ind, t in enumerate(traj_list):
        reward[t_ind] = t['R']

    traj_order = np.argsort(reward)
    bottom_trajs = traj_order[:d]
    top_trajs = traj_order[-d:]
    top_mask = unite_trajs_mask(traj_list, top_trajs, n_points)
    bottom_mask = unite_trajs_mask(traj_list, bottom_trajs, n_points)

    top_model = SMDP(labels[top_mask], termination[top_mask], rewards[top_mask], values[top_mask], n_clusters, tb=1)
    bottom_model = SMDP(labels[bottom_mask], termination[bottom_mask], rewards[bottom_mask], values[bottom_mask], n_clusters, tb=1)

    # print 'top_model.n_clusters: %d' % top_model.n_clusters
    # print 'bottom_model.n_clusters: %d' % bottom_model.n_clusters

    # pi_empiric_1 = []
    # pi_empiric_2 = []
    # pi_empiric_3 = []
    # for i,(p_top,p_bottom) in enumerate(zip(top_model.P,bottom_model.P)):
    #     pi_empiric_1.append([i,np.argmax(p_top-p_bottom)])
    #     pi_empiric_2.append([i,np.argmax(p_top)])
    #     pi_empiric_3.append([i,np.argmax(p_bottom)])
    #
    # match_count_1 = 0
    # match_count_2 = 0
    # match_count_3 = 0
    # for pi_e_1, pi_e_2, pi_e_3, pi_a in zip(pi_empiric_1, pi_empiric_2, pi_empiric_3, pi_analytic):
    #     if pi_e_1[1]==pi_a[1]:
    #         match_count_1 +=1
    #     if pi_e_2[1]==pi_a[1]:
    #         match_count_2 +=1
    #     if pi_e_3[1]==pi_a[1]:
    #         match_count_3 +=1
    #
    # print 'match_count_1: %d' % match_count_1
    # print 'match_count_2: %d' % match_count_2
    # print 'match_count_3: %d' % match_count_3

    top_greedy_sum = 0
    top_cluster_sum = 0
    for i,pi in enumerate(top_model.P):
        top_greedy_sum += pi[pi_analytic[i][1]]
        if pi.sum()>0:
            top_cluster_sum += 1

    top_greedy_sum = top_greedy_sum / top_cluster_sum

    bottom_greedy_sum = 0
    bottom_cluster_sum = 0
    for i,pi in enumerate(bottom_model.P):
        bottom_greedy_sum += pi[pi_analytic[i][1]]
        if pi.sum()>0:
            bottom_cluster_sum += 1

    bottom_greedy_sum = bottom_greedy_sum / bottom_cluster_sum

    tb_disc = np.zeros(n_clusters)
    bt_disc = np.zeros(n_clusters)
    for c in xrange(n_clusters):
        tb_disc[c] = scipy.stats.entropy(top_model.P[c], bottom_model.P[c])
        bt_disc[c] = scipy.stats.entropy(bottom_model.P[c], top_model.P[c])
                           # - scipy.stats.entropy(top_model.P[c]) - scipy.stats.entropy(bottom_model.P[c])

    result = {'top_model': top_model,
                      'bottom_model': bottom_model,
                      'tb_disc': tb_disc,
                      'bt_disc': bt_disc,
                      'top_greedy_sum': top_greedy_sum,
                      'bottom_greedy_sum': bottom_greedy_sum,
              }
    return result

def calc_cluster_im(self,indices):
    screens = np.copy(self.screens[indices])
    if self.game_id  == 2: #pacman
        for s in screens:

            enemies_map = 1 * (s[:,:,0] == 180) + \
                          1 * (s[:,:,0] == 149) + \
                          1 * (s[:,:,0] == 212) + \
                          1 * (s[:,:,0] == 128) + \
                          1 * (s[:,:,0] == 232) + \
                          1 * (s[:,:,0] == 204)

            enemies_mask = np.ones((210,160),dtype=bool)
            enemies_mask[20:28,6:10] = 0
            enemies_mask[140:148,6:10] = 0
            enemies_mask[20:28,150:154] = 0
            enemies_mask[140:148,150:154] = 0
            enemies_map = enemies_map * enemies_mask
            r_ch = s[:,:,0]
            g_ch = s[:,:,1]
            b_ch = s[:,:,2]
            r_ch[np.nonzero(enemies_map)] = 45
            g_ch[np.nonzero(enemies_map)] = 50
            b_ch[np.nonzero(enemies_map)] = 184
    meanscreen=np.mean(screens,axis=0)

    return meanscreen
#############################
# 8. color outliers
#############################
# def outliers(event):
#     if self.outlier_color is None:
#         # run your algorithm once
#         from sos import sos
#         import argparse
#         import sys
#         parser = argparse.ArgumentParser(description="Stochastic Outlier Selection")
#         parser.add_argument('-b', '--binding-matrix', action='store_true',
#         default=False, help="Print binding matrix", dest="binding_matrix")
#         parser.add_argument('-t', '--threshold', type=float, default=None,
#         help=("Float between 0.0 and 1.0 to use as threshold for selecting "
#             "outliers. By default, this is not set, causing the outlier "
#             "probabilities instead of the classification to be outputted"))
#         parser.add_argument('-d', '--delimiter', type=str, default=',', help=(
#         "String to use to separate values. By default, this is a comma."))
#         parser.add_argument('-i', '--input', type=argparse.FileType('rb'),
#         default=sys.stdin, help=("File to read data set from. By default, "
#             "this is <stdin>."))
#         parser.add_argument('-m', '--metric', type=str, default='euclidean', help=(
#         "String indicating the metric to use to compute the dissimilarity "
#         "matrix. By default, this is 'euclidean'. Use 'none' if the data set "
#         "is a dissimilarity matrix."))
#         parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
#         default=sys.stdout, help=("File to write the computed outlier "
#             "probabilities to. By default, this is <stdout>."))
#         parser.add_argument('-p', '--perplexity', type=float, default=30.0,
#         help="Float to use as perpexity. By default, this is 30.0.")
#         parser.add_argument('-v', '--verbose', action='store_true', default=False,
#         help="Print debug messages to <stderr>.")
#         args = parser.parse_args()
#         self.outlier_color = sos(self.global_feats['tsne'], 'euclidean', 50,args)
#
#
#
#     self.tsne_scat.set_array(self.outlier_color)
#     sizes = np.ones(self.num_points)*self.pnt_size
#     sizes[self.outlier_color>self.slider_outlier_thresh.val] = 250
#
#     self.tsne_scat.set_sizes(sizes)
#     self.fig.canvas.draw()
#

# self.ax_otlyr = plt.axes([0.80, 0.77, 0.09, 0.02])
# self.b_otlyr = Button(self.ax_otlyr, 'Outliers')
# self.outlier_color = None
# self.b_otlyr.on_clicked(outliers)
# self.slider_outlier_thresh = Slider(plt.axes([0.80, 0.74, 0.09, 0.02]), 'outlier_thresh', valmin=0, valmax=1, valinit=0.75)
# self.SLIDER_FUNCS.append(update_slider(self, 'outlier_thresh', self.slider_outlier_thresh))
# self.slider_outlier_thresh.on_changed(self.update_sliders)
