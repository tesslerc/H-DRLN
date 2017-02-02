import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import path
from matplotlib.patches import Polygon
from clustering import clustering_
import common
import pickle
from digraph import draw_transition_table

def add_buttons(self):

    #############################
    # 1. play 1 b/w step
    #############################
    def BW(event):
        self.ind = (self.ind - 1) % self.num_points
        self.update_plot()
        self.prev_ind = self.ind

    self.ax_bw = plt.axes([0.60, 0.80, 0.09, 0.02])
    self.b_bw = Button(self.ax_bw, 'B/W')
    self.b_bw.on_clicked(BW)

    #############################
    # 2. play 1 f/w step
    #############################
    def FW(event):
        self.ind = (self.ind + 1) % self.num_points
        self.update_plot()
        self.prev_ind = self.ind

    self.ax_fw = plt.axes([0.70, 0.80, 0.09, 0.02])
    self.b_fw = Button(self.ax_fw, 'F/W')
    self.b_fw.on_clicked(FW)

    #############################
    # 3. Color by condition
    #############################
    def set_color_by_cond(event):
        self.update_cond_vector(self)
        self.tsne_scat.set_array(self.cond_vector)
        sizes = 5 * self.cond_vector + self.pnt_size * np.ones_like(self.tsne_scat.get_sizes())
        self.tsne_scat.set_sizes(sizes)
        print ('number of valid points: %d') % (np.sum(self.cond_vector))

    self.ax_cond = plt.axes([0.60, 0.77, 0.09, 0.02])
    self.cond_vector = np.ones(shape=(self.num_points,1), dtype='int8')
    self.b_color_by_cond = Button(self.ax_cond, 'color by cond')
    self.b_color_by_cond.on_clicked(set_color_by_cond)

    self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)

    #############################
    # 4. save figure
    #############################
    def save_figure(event):

        # save figure
        plt.savefig(self.global_feats['data_dir'] + '/knn/tsne_figure.png')

        # save clusters
        common.save_hdf5('cluster_ids', self.global_feats['data_dir'] + '/knn/', self.clusters['cluster_ids'])

        # save clusters (pickle)
        pickle.dump(self.clusters, file((self.global_feats['data_dir'] + '/knn/' + 'clusters.bin'),'w'))

        print 'saved figure to %s' %  (self.global_feats['data_dir'] + '/knn/tsne_figure.png')

    self.ax_save = plt.axes([0.70, 0.77, 0.09, 0.02])
    self.b_save = Button(self.ax_save, 'Save figure')
    self.b_save.on_clicked(save_figure)

    #############################
    # 5.0 update cluster (helper function)
    #############################
    def update_cluster(self, marked_points):
        # create path object from marked points
        poly_path = path.Path(marked_points)

        # recieve the points that are inside the marked area
        cluster_points = poly_path.contains_points(self.data_t.T)

        # update the list of polygons
        self.clusters['polygons'].append(Polygon(marked_points, alpha=0.2))

        # draw new cluster
        self.ax_tsne.add_patch(self.clusters['polygons'][-1])

        # update cluster points
        self.clusters['cluster_number'] += 1

        self.clusters['cluster_ids'][cluster_points] = self.clusters['cluster_number']

        # update marked_points
        self.clusters['marked_points'].append(marked_points)

        # annotate cluster
        self.clusters['annotations'].append(self.ax_tsne.annotate(self.clusters['cluster_number'], xy=marked_points[0], size=20, color='r'))

    #############################
    # 5. mark cluster
    #############################
    def mark_cluster(event):
        # user marks cluster area
        marked_points = plt.ginput(0, timeout=-1)

        update_cluster(self, marked_points)

        # save cluster ids to hdf5 vector
        common.save_hdf5('cluster_ids', self.global_feats['data_dir'] + '/knn/', self.clusters['cluster_ids'])

        # save clusters (pickle)
        pickle.dump(self.clusters, file((self.global_feats['data_dir'] + '/knn/' + 'clusters.bin'),'w'))

    self.ax_mc = plt.axes([0.60, 0.74, 0.09, 0.02])
    self.b_mc = Button(self.ax_mc, 'Mark Cluster')
    self.b_mc.on_clicked(mark_cluster)
    self.clusters = {
            'polygons' : [],
            'annotations' : [],
            'cluster_ids' : np.zeros(self.num_points),
            'cluster_number' : 0,
            'cluster_points' : [],
            'marked_points' : [],
    }

    #############################
    # 6. delete cluster
    #############################
    def delete_cluster(event):
        # remove cluster points
        self.clusters['cluster_ids'][self.clusters['cluster_ids']==self.clusters['cluster_number']] = 0

        # decrease cluster number by 1
        self.clusters['cluster_number'] -= 1

        # delete cluster from figure
        self.clusters['polygons'][-1].remove()
        self.clusters['annotations'][-1].remove()

        # delete cluster from list
        self.clusters['polygons'].pop()
        self.clusters['annotations'].pop()
        self.clusters['marked_points'].pop()

    self.ax_dc = plt.axes([0.70, 0.74, 0.09, 0.02])
    self.b_dc = Button(self.ax_dc, 'Delete Cluster')
    self.b_dc.on_clicked(delete_cluster)

    #############################
    # 6. load clusters
    #############################
    def load_clusters(event):
        self.clusters = pickle.load(file(self.global_feats['data_dir'] + '/knn/' + 'clusters.bin'))

        marked_points_list = self.clusters['marked_points']

        self.clusters = {
            'polygons' : [],
            'annotations' : [],
            'cluster_ids' : np.zeros(self.num_points),
            'cluster_number' : 0,
            'cluster_points' : [],
            'marked_points' : [],
        }

        # draw clusters
        for marked_points in marked_points_list:
            update_cluster(self, marked_points)

    self.ax_lc = plt.axes([0.60, 0.71, 0.09, 0.02])
    self.b_lc = Button(self.ax_lc, 'Load Clusters')
    self.b_lc.on_clicked(load_clusters)

    #############################
    # 7. cluster states
    #############################
    def clustering(event):
        clustering_(self,plt)

    self.ax_clstr = plt.axes([0.70, 0.71, 0.09, 0.02])
    self.b_clstr = Button(self.ax_clstr, 'Clustering')
    self.clustering_labels = None
    self.b_clstr.on_clicked(clustering)

    #############################
    # 8. Mark trajectory
    #############################
    def mark_trajectory(event):

        # 1. Display trajectory points over t-SNE
        traj_point_mask = np.asarray(self.hand_craft_feats['traj'])==self.traj_id
        self.tsne_scat.set_array(traj_point_mask)
        sizes = 5 * traj_point_mask + self.pnt_size * np.ones(self.num_points)
        self.tsne_scat.set_sizes(sizes)

        # 2. Display trajectory moves over SMDP
        if self.clustering_labels is not None:
            traj = self.traj_list[self.traj_id]
            moves = traj['moves']
            R = traj['R']
            length = traj['length']
            traj_moves = list(set(moves))
            draw_transition_table(transition_table=self.smdp.P, cluster_centers=self.cluster_centers,
                          meanscreen=self.meanscreen, tsne=self.global_feats['tsne'], color=self.color, black_edges=self.smdp.edges, red_edges=traj_moves)

            print ('Trajectory id: %d, Number of points: %d, R: %d') % (self.traj_id, length, R)

        self.traj_id = (self.traj_id + 1) % self.hand_craft_feats['n_trajs']

    self.traj_id = 0
    self.ax_mt = plt.axes([0.80, 0.80, 0.09, 0.02])
    self.b_mt = Button(self.ax_mt, 'Mark Traj.')
    self.b_mt.on_clicked(mark_trajectory)

    #############################
    # 9. Policy improvement
    #############################
    def policy_improvement(event):
        if not hasattr(self,'clustering_labels'):
            return

        policy = self.smdp.greedy_policy
        draw_transition_table(transition_table=self.smdp.P, cluster_centers=self.cluster_centers,
                          meanscreen=self.meanscreen, tsne=self.global_feats['tsne'], color=self.color, black_edges=None, red_edges=policy)

    self.ax_pi = plt.axes([0.80, 0.77, 0.09, 0.02])
    self.b_pi = Button(self.ax_pi, 'Policy improve.')
    self.b_pi.on_clicked(policy_improvement)


    #############################
    # 10. Eject
    #############################
    def eject(event):
        if not hasattr(self,'clustering_labels'):
            return

        tt_ratio = 0.2
        n_points = self.global_feats['termination'].shape[0]

        n_points_model = int(n_points*tt_ratio)
        last_model_traj = self.hand_craft_feats['traj'][n_points_model]
        n_points_model = np.flatnonzero(np.asarray(self.hand_craft_feats['traj']) == last_model_traj)[-1]

        n_points_held_out = n_points - n_points_model - 1

        # 1. Average score before eject
        R = 0
        count = 0
        for traj in self.traj_list[last_model_traj+1:]:
            R += traj['R']
            count += 1

        print 'Average score before eject: %f' % (R / count)

        # 2. Build model on first part of the data
        # DEBUG
        n_points_model=n_points-1
        clustering_(self, plt, n_points=n_points_model+1, force=1)

        # 3. define eject set
        eject_indices = np.nonzero((self.tb_trajs_discr['top_model'].P==0) * (self.tb_trajs_discr['bottom_model'].P>0))

        # 4. create held_out set
        data_held_out = np.zeros(shape=(n_points_held_out, 3))
        data_held_out[:, 0:2] = self.global_feats['tsne'][-n_points_held_out:]
        data_held_out[:, 2] = self.global_feats['value'][-n_points_held_out:]

        rewards_held_out = self.global_feats['reward'][-n_points_held_out:]

        labels_held_out = -1*np.ones(n_points_held_out)

        # 5. cluster held-out points using NN
        for i,d in enumerate(data_held_out):
            labels_held_out[i] = np.argmin(((d - self.cluster_centers) ** 2).sum(axis=1))

        # 6. find average score on held-out trajectories, excluding ejected trajectories
        held_out_trajs = self.hand_craft_feats['traj'][-n_points_held_out:]
        held_out_trajs -= held_out_trajs.min()
        max_traj_id = np.asarray(held_out_trajs).max()

        eject_indices_ = np.asarray(eject_indices).T

        valid_trajs = np.ones(max_traj_id)

        for i in xrange(n_points_held_out-1):
            move_i = np.asarray([labels_held_out[i],labels_held_out[i+1]])
            diff_i = (abs((move_i - eject_indices_))).sum(axis=1)
            if diff_i.min()==0:
                valid_trajs[held_out_trajs[i]] = 0

        R_ = 0
        curr_traj = held_out_trajs[0]
        for i in xrange(n_points_held_out):
            if curr_traj == max_traj_id:
                break
            if (held_out_trajs[i]==curr_traj) and valid_trajs[curr_traj]==1:
                R_t += rewards_held_out[i]
            else:
                R_ += R_t
                R_t = 0
                curr_traj = held_out_trajs[i]

        print 'Average score after eject: %f' % (R_ / valid_trajs.sum())

    self.ax_ej = plt.axes([0.80, 0.74, 0.09, 0.02])
    self.b_ej = Button(self.ax_ej, 'Eject')
    self.b_ej.on_clicked(eject)