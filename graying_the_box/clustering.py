from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans as Kmeans_st
# from sklearn.cluster import KMeans_st as Kmeans_st

from emhc import EMHC
from smdp import SMDP
import numpy as np
import common
from digraph import draw_transition_table

def perpare_features(self, n_features=3):

    data = np.zeros(shape=(self.global_feats['tsne'].shape[0],n_features))
    data[:,0:2] = self.global_feats['tsne']
    data[:,2] = self.global_feats['value']
    # data[:,3] = self.global_feats['time']
    # data[:,4] = self.global_feats['termination']
    # data[:,5] = self.global_feats['tsne3d_norm']
    # data[:,6] = self.hand_craft_feats['missing_bricks']
    # data[:,6] = self.hand_craft_feats['hole']
    # data[:,7] = self.hand_craft_feats['racket']
    # data[:,8] = self.hand_craft_feats['ball_dir']
    # data[:,9] = self.hand_craft_feats['traj']
    # data[:,9:11] = self.hand_craft_feats['ball_pos']
    data[np.isnan(data)] = 0
    # 1.2 data standartization
    # scaler = preprocessing.StandardScaler(with_centering=False).fit(data)
    # data = scaler.fit_transform(data)

    # data_mean  = data.mean(axis=0)
    # data -= data_mean
    return data

def clustering_(self, plt, n_points=None, force=0):

    if n_points==None:
        n_points = self.global_feats['termination'].shape[0]

    if self.clustering_labels is not None:
        self.tsne_scat.set_array(self.clustering_labels.astype(np.float32)/self.clustering_labels.max())
        draw_transition_table(transition_table=self.smdp.P, cluster_centers=self.cluster_centers,
                          meanscreen=self.meanscreen, tsne=self.global_feats['tsne'], color=self.color, black_edges=self.smdp.edges)
        plt.show()
        if force==0:
            return

    n_clusters = self.cluster_params['n_clusters']
    W = self.cluster_params['window_size']
    n_iters = self.cluster_params['n_iters']
    entropy_iters = self.cluster_params['entropy_iters']

    # slice data by given indices
    term = self.global_feats['termination'][:n_points]
    reward = self.global_feats['reward'][:n_points]
    value = self.global_feats['value'][:n_points]
    tsne = self.global_feats['tsne'][:n_points]
    traj_ids = self.hand_craft_feats['traj'][:n_points]

    # 1. create data for clustering
    data = perpare_features(self)
    data = data[:n_points]
    data_scale = data.max(axis=0)
    data /= data_scale

    # 2. Build cluster model
    # 2.1 spatio-temporal K-means
    if self.cluster_params['method'] == 0:
        windows_vec = np.arange(start=W,stop=W+1,step=1)
        clusters_vec = np.arange(start=n_clusters,stop=n_clusters+1,step=1)
        models_vec = []
        scores = np.zeros(shape=(len(clusters_vec),1))
        for i,n_w in enumerate(windows_vec):
            for j,n_c in enumerate(clusters_vec):
                cluster_model = Kmeans_st(n_clusters=n_clusters,window_size=n_w,n_jobs=8,n_init=n_iters,entropy_iters=entropy_iters)
                cluster_model.fit(data, rewards=reward, termination=term, values=value)
                labels = cluster_model.labels_
                models_vec.append(cluster_model.smdp)
                scores[j] = cluster_model.smdp.score
                print 'window size: %d , Value mse: %f' % (n_w, cluster_model.smdp.score)
            best = np.argmin(scores)
            self.cluster_params['n_clusters'] +=best
            self.smdp = models_vec[best]

    # 2.1 Spectral clustering
    elif self.cluster_params['method'] == 1:
        import scipy.spatial.distance
        import scipy.sparse
        dists = scipy.spatial.distance.pdist(tsne, 'euclidean')
        similarity = np.exp(-dists/10)
        similarity[similarity<1e-2] = 0
        print 'Created similarity matrix'
        affine_mat = scipy.spatial.distance.squareform(similarity)
        cluster_model = SpectralClustering(n_clusters=n_clusters,affinity='precomputed')
        labels = cluster_model.fit_predict(affine_mat)

    # 2.2 EMHC
    elif self.cluster_params['method'] == 2:
        # cluster with k means down to n_clusters + D
        n_clusters_ = n_clusters + 5
        kmeans_st_model = Kmeans_st(n_clusters=n_clusters_,window_size=W,n_jobs=8,n_init=n_iters,entropy_iters=entropy_iters, random_state=123)
        kmeans_st_model.fit(data, rewards=reward, termination=term, values=value)
        cluster_model = EMHC(X=data, labels=kmeans_st_model.labels_, termination=term, min_clusters=n_clusters, max_entropy=np.inf)
        cluster_model.fit()
        labels = cluster_model.labels_
        self.smdp = SMDP(labels=labels, termination=term, rewards=reward, values=value, n_clusters=n_clusters)

    self.smdp.complete_smdp()
    self.clustering_labels = self.smdp.labels
    common.create_trajectory_data(self, reward, traj_ids)
    self.state_pi_correlation = common.reward_policy_correlation(self.traj_list, self.smdp.greedy_policy, self.smdp)

    top_greedy_vec = []
    bottom_greedy_vec = []
    max_diff = 0
    best_d = 1
    for i,d in enumerate(xrange(1,30)):
        tb_trajs_discr = common.extermum_trajs_discrepency(self.traj_list, self.clustering_labels, term, reward, value, self.smdp.n_clusters, self.smdp.greedy_policy, d=d)
        top_greedy_vec.append([i,tb_trajs_discr['top_greedy_sum']])
        bottom_greedy_vec.append([i,tb_trajs_discr['bottom_greedy_sum']])
        diff_i = tb_trajs_discr['top_greedy_sum'] - tb_trajs_discr['bottom_greedy_sum']
        if diff_i > max_diff:
            max_diff = diff_i
            best_d = d

    self.tb_trajs_discr = common.extermum_trajs_discrepency(self.traj_list, self.clustering_labels, term, reward, value, self.smdp.n_clusters, self.smdp.greedy_policy, d=best_d)
    self.top_greedy_vec = top_greedy_vec
    self.bottom_greedy_vec = bottom_greedy_vec

    common.draw_skills(self,self.smdp.n_clusters,plt)


    # 4. collect statistics
    cluster_centers = cluster_model.cluster_centers_
    cluster_centers *= data_scale

    screen_size = self.screens.shape
    meanscreen  = np.zeros(shape=(n_clusters,screen_size[1],screen_size[2],screen_size[3]))
    cluster_time = np.zeros(shape=(n_clusters,1))
    width = int(np.floor(np.sqrt(n_clusters)))
    length = int(n_clusters/width)
    # f, ax = plt.subplots(length,width)

    for cluster_ind in range(n_clusters):
        indices = (labels==cluster_ind)
        cluster_data = data[indices]
        cluster_time[cluster_ind] = np.mean(self.global_feats['time'][indices])
        meanscreen[cluster_ind,:,:,:] = common.calc_cluster_im(self,indices)

    # 5. draw cluster indices
    plt.figure(self.fig.number)
    data *= data_scale
    for i in range(n_clusters):
        self.ax_tsne.annotate(i, xy=cluster_centers[i,0:2], size=20, color='r')
    draw_transition_table(transition_table=self.smdp.P, cluster_centers=cluster_centers,
                          meanscreen=meanscreen, tsne=data[:,0:2], color=self.color, black_edges=self.smdp.edges)

    self.cluster_centers = cluster_centers
    self.meanscreen =meanscreen
    self.cluster_time =cluster_time
    common.visualize(self)

def update_slider(self, name, slider):
    def f():
        setattr(self, name, slider.val)
    return f
