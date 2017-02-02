import numpy as np
import scipy.linalg
import itertools
import sys

class EMHC(object):
    def __init__ (self, X, termination, labels=None, min_clusters=20, max_entropy=5, w_entropy=0.01, w_distance=1):

        # 0. scale features
        # TODO

        # DEBUG
        # N = 11
        # X = X[:N]
        # termination = termination[:N]
        # labels = np.random.randint(low=0, high=300, size=N)

        # debug
        # X = np.zeros((11,3))
        # X[:5,:] = np.random.normal(loc=0,size=(5,3))
        # X[5:,:] = np.random.normal(loc=100,size=(6,3))
        # labels = np.asarray([x for x in xrange(11)])

        # 2. remove last point (it is loaded with zeros)
        self.X = X[:-1,:]
        self.termination = termination[:-1]
        self.termination[-1] = 1
        self.labels_ = labels
        self.n_features = self.X.shape[1]
        if self.labels_ is not None:
            self.labels_ = self.labels_[:-1]

        # 1. parameters
        self.w_entropy = w_entropy
        self.w_distance = w_distance
        self.n_samples = self.X.shape[0]
        self.n_clusters = self.n_samples
        self.min_clusters = min_clusters
        self.max_entropy = max_entropy

        # 2. remove empty clusters: check that all clusters exist
        self.remove_empty_clusters()

        # 3. total transition matrix TT
        self.init_TT()

        # 4. clusters size vector s
        self.init_cluster_sizes()

        # 5. entropy list e
        self.init_entropy()

        # 6. initalize connectivity lists : in_list, out_list
        self.init_connectivity_lists()

        # 7. pairwise distance matrix D
        self.init_D()

        # 8. pairwise entropy gain matrix EG
        self.init_pw_ent_gain_mat()

        # 9. score matrix SCORE
        self.update_score_mat()

        # 10. i_min, j_min
        self.i_min, self.j_min = np.unravel_index(np.argmin(self.SCORE),self.SCORE.shape)

        # 11. print
        print 'Finished preparing model...'

    # 10. fit function
    def fit(self):
        self.models_likelihood = [[],[]]
        while self.n_clusters > self.min_clusters and self.mean_wighted_entropy() < self.max_entropy:
            print 'n_clusters: %d, entropy: %f' % (self.n_clusters, self.mean_wighted_entropy())
            # print '(i,j): (%d,%d)' % (self.i_min, self.j_min)
            self.models_likelihood[0].append(self.n_clusters)
            self.models_likelihood[1].append(self.mean_wighted_entropy())

            assert(self.i_min!=self.j_min)
            assert(self.i_min<self.SCORE.shape[0]-1)

            # 10.0 decrease number of clusters by 1
            self.n_clusters -= 1

            # 10.1 unite TT i,j rows and columns
            self.TT = self.unite_rows_cols(self.TT, self.i_min, self.j_min)

            # 10.2 update connectivity lists
            self.update_connectivity_lists()

            # 10.3 update vector size
            self.update_cluster_size()

            # 10.4 change list
            self.change_list = [self.i_min] + self.in_list[self.i_min]

            # 10.5 update indices due to removal of j_min
            # removed for the meantime: we operate on the upper triangular part of the pairwise matrix.
            # Therefore i is always smaller than j
            # self.change_list = self.update_indices(self.change_list, self.j_min)

            # 10.6 update entropy
            self.update_entropy()

            # 10.7 update pairwise entropy gain matrix EG
            # self.update_pw_ent_gain_mat(self.change_list)
            # DEBUG: check if we bug exists in the fast entropy gain matrix by comparing to full calculation
            self.update_pw_ent_gain_mat_full()

            # 10.8 update pairwise distance matrix D
            self.update_pw_distances_mat()

            # 10.9 update score matrix SCORE
            self.update_score_mat()

            # 10.10 update best pair
            self.i_min, self.j_min = np.unravel_index(np.argmin(self.SCORE), self.SCORE.shape)

        # 11. assign labels
        for i, cluster in enumerate(self.member_list):
            for c in cluster:
                self.labels_[c] = i

        # 12. cluster centers
        self.cluster_centers_ = np.zeros((self.n_clusters,self.n_features))
        for i in xrange(self.n_clusters):
            self.cluster_centers_[i,:] = np.mean(self.X[self.member_list[i],:],axis=0)

    def add_unique_excluding_i(self, list_a, list_b, i):
        in_a = set(list_a)
        in_b = set(list_b)
        in_b_but_not_in_a = in_b - in_a
        result = list_a + list(in_b_but_not_in_a)
        if i in result: result.remove(i)
        return result

    def assert_no_self_links(self, my_list):
        for ii, cluster in enumerate(my_list):
            for c in cluster:
                assert(c!=ii)

    def assert_max_in_range(self, my_list):
        max_cluster = 0
        for c in my_list:
            if c:
                if max(c) > max_cluster:
                    max_cluster = max(c)
        assert (max_cluster <= self.n_clusters-1)

    def dist(self, l, m, L=2):
        u = self.X[l]
        v = self.X[m]
        if L == 1:
            return (u - v).abs().sum()
        if L == 2:
            return ((u - v) ** 2).sum()

    def clusters_dist(self, i, j):
        d = 0
        k = 0
        for pair in itertools.product(self.member_list[i], self.member_list[j]):
            # d_pair = self.dist(*pair)
            if pair[0]<pair[1]:
                d_pair = self.ED[pair[0],pair[1]]
            else:
                d_pair = self.ED[pair[1],pair[0]]

            d += d_pair
            k += 1
        mean_dist = d/k

        return mean_dist

    def build_clusters_vec(d, n_samples):
        clusters_vec = np.zeros(n_samples)
        for cluster_id, cluster in enumerate(d):
            for member in cluster:
                clusters_vec[member] = cluster_id
        return clusters_vec

    def remove_empty_clusters(self):
        if self.labels_ is not None:
            n_clusters = np.max(self.labels_)+1
            cluster_flags = np.zeros(n_clusters, dtype=np.bool)
            for l in self.labels_:
                cluster_flags[l] = True
            shift_vec = np.cumsum(~cluster_flags)
            for ind, l in enumerate(self.labels_):
                self.labels_[ind] -= shift_vec[l]
            self.n_clusters = np.max(self.labels_)+1
            print "Removed empty clusters. total number of cluster is now: %d..." % self.n_clusters

    def init_TT(self):
        self.TT = np.zeros(shape=(self.n_clusters, self.n_clusters))
        if self.labels_ is None:
            for i, t in enumerate(self.termination[:-1]):
                if t:
                    self.TT[i, i] = 1
                else:
                    self.TT[i, i + 1] = 1
            self.TT[-1,-1] = 1
        else:
            for i, (t,l) in enumerate(zip(self.termination[:-1],self.labels_[:-1])):
                if t:
                    self.TT[self.labels_[i], self.labels_[i]] += 1
                else:
                    self.TT[self.labels_[i], self.labels_[i+1]] += 1
            self.TT[self.labels_[-1], self.labels_[-1]] += 1
            if (np.any(self.TT.sum(axis=1)==0)):
                a=1
        print "Initialized transition table..."

    def init_cluster_sizes(self):
        if self.labels_ is None:
            self.s = np.ones(self.n_clusters)
        else:
            self.s = self.TT.sum(axis=1)

    def init_entropy(self):
        self.e = np.zeros(self.n_clusters)
        if self.labels_ is not None:
            for i in xrange(self.n_clusters):
                TTi = np.copy(self.TT[i])
                TTi[i] = 0
                self.e[i] = scipy.stats.entropy(TTi/TTi.sum())
                if np.isnan(self.e[i]) or np.isinf(self.e[i]):
                    self.e[i] = 0
        print "Initialized entropy vector..."

    def init_D(self):
        print "Creating pairwise distance matrix..."

        #1. pairwise-example distance matrix
        dists = scipy.spatial.distance.pdist(self.X, 'euclidean')
        self.ED = scipy.spatial.distance.squareform(dists)
        self.ED[np.tril_indices(self.n_clusters)] = np.infty
        self.ED[np.diag_indices(self.n_clusters)] = np.infty

        #2. pairwise-cluster distance matrix
        self.CD = np.infty * np.ones(shape=(self.n_clusters, self.n_clusters))
        for i in xrange(self.n_clusters-1):
            for j in xrange(i + 1, self.n_clusters):
                self.CD[i, j] = self.clusters_dist(i, j)
            print "processing cluster %d/%d" % (i,self.n_clusters)
        print "Initialized pairwise distance matrix..."

    def init_connectivity_lists(self):
        self.in_list = [[] for x in xrange(self.n_clusters)]
        self.out_list = [[] for x in xrange(self.n_clusters)]
        self.member_list = [[] for x in xrange(self.n_clusters)]

        if self.labels_ is None:
            self.member_list = [[x] for x in xrange(self.n_clusters)]
            for i, t in enumerate(self.termination):
                if t == 0 and i<(self.n_clusters-1):
                    self.in_list[i + 1] = [i]
                    self.out_list[i] = [i+1]
        else:
            for i,l in enumerate(self.labels_):
                self.member_list[l].append(i)
            for i,line in enumerate(self.TT):
                self.out_list[i] = list(np.flatnonzero(line))
                if i in self.out_list[i]: self.out_list[i].remove(i)
            for i, col in enumerate(self.TT.T):
                self.in_list[i] = list(np.flatnonzero(col))
                if i in self.in_list[i]: self.in_list[i].remove(i)
        print "Initialized connectivity lists..."

    def init_pw_ent_gain_mat(self):
        self.EG = np.zeros(shape=(self.n_clusters, self.n_clusters))

        self.EG[np.tril_indices(self.n_clusters)] = np.infty
        if self.labels_ is None:
            unite_zero_ent = 2*scipy.stats.entropy(np.asarray([0.5,0.5]))
            for i in xrange(self.n_samples):
                for j in xrange(i+1,self.n_samples):
                    if self.termination[i]==0 and self.termination[j]==0:
                        if abs(i-j)!=1:
                            self.EG[i,j]=unite_zero_ent
        else:
            change_list = [x for x in xrange(self.n_clusters)]
            self.update_pw_ent_gain_mat(change_list)

    def update_score_mat(self):
        self.SCORE = self.w_entropy*self.EG + self.w_distance*self.CD

    def mean_wighted_entropy(self):
        return np.average(a=self.e,weights=self.s)

    def update_connectivity_lists(self):
        # 1. j_min outputs
        # 1.1 in-links
        for c in self.out_list[self.j_min]:
            self.in_list[c].remove(self.j_min) # remove j_min
            if c != self.i_min and (self.i_min not in self.in_list[c]):
                self.in_list[c].append(self.i_min) # add i_min

        # 1.2 out-links
        self.out_list[self.i_min] = self.add_unique_excluding_i(self.out_list[self.i_min], self.out_list[self.j_min], self.i_min)
        self.out_list[self.j_min] = []

        # 2. j_min inputs
        # 2.1 out-links
        for c in self.in_list[self.j_min]:
            self.out_list[c].remove(self.j_min)  # remove j_min
            if c != self.i_min and (self.i_min not in self.out_list[c]):
                self.out_list[c].append(self.i_min)  # add i_min

        # 2.2 in-links
        self.in_list[self.i_min] = self.add_unique_excluding_i(self.in_list[self.i_min], self.in_list[self.j_min], self.i_min)
        self.in_list[self.j_min] = []

        # 3. decrease cluster numbers larger than j_min
        # 3.1 in list
        for ii, cluster in enumerate(self.in_list):
            for jj, c in enumerate(cluster):
                if c > self.j_min:
                    self.in_list[ii][jj] -= 1

        # 3.2 out list
        for ii, cluster in enumerate(self.out_list):
            for jj, c in enumerate(cluster):
                if c > self.j_min:
                    self.out_list[ii][jj] -= 1

        # 4. remove j_min from connectivity lists
        del self.out_list[self.j_min]
        del self.in_list[self.j_min]

        # 5. member list
        self.member_list[self.i_min] += self.member_list[self.j_min]
        del self.member_list[self.j_min]

        # 6. assert lists value are on range
        # self.assert_max_in_range(self.in_list)
        # self.assert_max_in_range(self.out_list)
        # self.assert_no_self_links(self.in_list)

    def unite_rows_cols(self, A, i, j):
        B = np.copy(A)
        B[i] += B[j]
        B[:,i] += B[:,j]
        B = np.delete(B,j,axis=0)
        B = np.delete(B,j,axis=1)
        return B

    def update_cluster_size(self):
        self.s[self.i_min] += self.s[self.j_min]
        self.s = np.delete(self.s,self.j_min)

    def update_indices(self, arry, j):
        arry_shift = arry
        for a in arry_shift:
            if a>j:
                a -= 1
        return arry_shift

    def update_entropy(self):
        for i in self.change_list:
            TTi = np.copy(self.TT[i])
            TTi[i] = 0
            TTi_normalized = TTi/TTi.sum()
            self.e[i] = scipy.stats.entropy(TTi_normalized)
            if np.isnan(self.e[i]) or np.isinf(self.e[i]):
                self.e[i] = 0

        self.e = np.delete(self.e,self.j_min)

    def update_pw_ent_gain_mat(self, change_list):
        if hasattr(self,'j_min'):
            self.EG = np.delete(self.EG, self.j_min, axis=0)
            self.EG = np.delete(self.EG, self.j_min, axis=1)

        for i in change_list:
            for j in xrange(self.n_clusters):
                if i==j:
                    continue

                # 1. create TT matrix based on i,j union
                TTij = self.unite_rows_cols(self.TT, i, j)

                # 2. copy current in_lists
                in_list_i = list(self.in_list[i])
                in_list_j = list(self.in_list[j])

                # 3. remove duplicates from in_list_i,in_list_j
                in_list = set(in_list_i+in_list_j)

                # 3.1 remove i,j references from in_list_unique
                if i in in_list: in_list.remove(i)
                if j in in_list: in_list.remove(j)

                in_list = list(in_list)

                # 3. index change: decrease cluster numbers greater than j by 1 to access TTij
                # 3.1 in_list
                in_list_reduced = list(in_list)
                for ii,c in enumerate(in_list_reduced):
                    if c > j:
                        in_list_reduced[ii] -= 1

                # 3.2 i
                i_reduced = i
                if i>j:
                    i_reduced -= 1

                # 4. entropy gain
                d_ent = 0

                # 4.1 add entropy of the newly created cluster
                cij_line = np.copy(TTij[i_reduced])
                cij_line[i_reduced] = 0
                eij = scipy.stats.entropy(cij_line/cij_line.sum())
                if np.isnan(eij) or np.isinf(eij):
                    eij = 0

                d_ent += eij # * (self.s[i]+self.s[j])

                # 4.2 remove old entropy of i,j
                d_ent -= self.e[i] # * self.s[i]
                d_ent -= self.e[j] # * self.s[j]

                # 4.3 remove old entropy of all clusters pointing to i,j
                d_ent -= np.sum(self.e[in_list] ) # * self.s[in_list] )

                # 2.4 add new entropy of all clusters pointing to i,j
                for l in (in_list_reduced):
                    TTij_l = np.copy(TTij[l])
                    TTij_l[l] = 0
                    eij_l = scipy.stats.entropy(TTij_l/TTij_l.sum())
                    if np.isnan(eij_l):
                        eij_l = 0
                    d_ent += eij_l # * TTij[l].sum()

                if j>i:
                    self.EG[i,j] = d_ent
                else:
                    self.EG[j,i] = d_ent

    def update_pw_ent_gain_mat_full(self):
        if hasattr(self,'j_min'):
            self.EG = np.delete(self.EG, self.j_min, axis=0)
            self.EG = np.delete(self.EG, self.j_min, axis=1)

        for i in xrange(self.n_clusters):
            for j in xrange(i + 1, self.n_clusters):

                # 1. create TT matrix based on i,j union
                TTij = self.unite_rows_cols(self.TT, i, j)
                TTij[np.diag_indices(TTij.shape[0])] = 0

                sij = np.copy(self.s)
                sij[i] += sij[j]
                sij = np.delete(sij,j)

                # 2. calculate entropy
                TTij_normal = TTij/TTij.sum(axis=1)[:,np.newaxis]
                # eij = np.sum(scipy.stats.entropy(TTij_normal.T) * sij)
                ent = scipy.stats.entropy(TTij_normal.T)
                ent[np.nonzero(np.isinf(ent))] = 0
                ent[np.nonzero(np.isnan(ent))] = 0
                # eij = np.average(a=ent, weights=sij)
                eij = np.sum(ent)

                if np.isinf(eij) or np.isnan(eij):
                    eij = 0

                # 3.2 decrease i
                if i>j:
                    i -= 1

                if j>i:
                    self.EG[i,j] = eij
                else:
                    self.EG[j,i] = eij

    def update_pw_distances_mat(self):
        self.CD = np.delete(self.CD, self.j_min, axis=0)
        self.CD = np.delete(self.CD, self.j_min, axis=1)

        for j in xrange(self.n_clusters):
            if j == self.i_min:
                continue

            d_pair = self.clusters_dist(self.i_min,j)
            if j>self.i_min:
                self.CD[self.i_min,j] = d_pair
            else:
                self.CD[j,self.i_min] = d_pair
