import numpy as np
import scipy.linalg


def divide_tt(X, tt_ratio):
    N = X.shape[0]
    X_train = X[:int(tt_ratio*N)]
    X_test = X[int(tt_ratio*N):]
    return X_train, X_test

class SMDP(object):
    def __init__(self, labels, termination, rewards, values, n_clusters, tb=0, gamma=0.99, trunc_th = 0.1, k=5):

        self.k = k
        self.gamma = gamma
        self.trunc_th = trunc_th
        self.rewards = rewards
        if tb == 0:
            self.labels,self.n_clusters = self.remove_empty_clusters(labels,n_clusters)
        else:
            self.labels = labels
            self.n_clusters = n_clusters
        self.termination = termination
        self.TT = self.calculate_transition_matrix()
        self.P = self.calculate_prob_transition_matrix(self.TT)
        if tb == 0:
            self.check_empty_P()
            self.r, self.skill_time = self.smdp_reward()
            self.v_smdp = self.calc_v_smdp(self.P)
            self.v_dqn = self.calc_v_dqn(values)
            self.score = self.value_score()
            self.clusters_count = self.count_clusters()
            self.entropy = self.calc_entropy(self.P)
        self.edges = self.get_smdp_edges()

    ####### Methods ######

    def check_empty_P(self):
        cluster_ind = 0
        for p in (self.P):
            if p.sum()==0:
                indices = np.nonzero(self.labels==cluster_ind)[0]
                for i in indices:
                    self.labels[i]=self.labels[i-1]
            cluster_ind+=1
        self.labels, self.n_clusters = self.remove_empty_clusters(self.labels,self.n_clusters)
        self.TT = self.calculate_transition_matrix()
        self.P = self.calculate_prob_transition_matrix(self.TT)

    def count_clusters(self):
        cluster_count = np.zeros(self.n_clusters)
        for l in self.labels:
            cluster_count[l]+=1
        return cluster_count

    def remove_empty_clusters(self,labels,n_c):
        # remove empty clusters
        labels_i = np.copy(labels)
        cluster_flags = np.zeros(n_c,dtype=np.bool)
        cluster_count =  np.zeros(n_c)
        for l in labels_i:
            cluster_flags[l] = True
            cluster_count[l]+=1
        shift_vec = np.cumsum(~cluster_flags)
        for ind,l in enumerate(labels_i):
            labels_i[ind] -= shift_vec[l]
        new_n_clusters = np.max(labels_i)+1

        return labels_i,new_n_clusters

    def calc_entropy(self,P):
        e               = scipy.stats.entropy(P.T)
        e_finite_ind    = np.isfinite(e)
        entropy         = np.average(a=e[e_finite_ind],weights=self.clusters_count)
        return entropy

    def truncate(self, M, th):
        M[np.nonzero(M<th)] = 0
        M_t = M / M.sum(axis=1)[:, np.newaxis]
        M_t[np.isnan(M_t)]=0
        return M_t

    def calculate_transition_matrix(self):
        TT = np.zeros((self.n_clusters, self.n_clusters))
        for i, (t, l) in enumerate(zip(self.termination[:-1], self.labels[:-1])):
            if t:
                TT[self.labels[i], self.labels[i]] += 1
            else:
                if self.labels[i]!=self.labels[i+1] and np.all(self.labels[i+1:i+2+self.k]==self.labels[i+1]):
                    TT[self.labels[i], self.labels[i + 1]] += 1
        return TT

    def calculate_prob_transition_matrix(self,TT):
        P = np.copy(TT)
        for i in xrange(self.n_clusters):
            P[i,i]=0
        P = P / P.sum(axis=1)[:, np.newaxis]
        P = self.truncate(P, self.trunc_th)
        return P

    def smdp_reward(self):
        l_p = self.labels[0]
        total_r = 0
        t = 0
        rewards_clip = np.clip(self.rewards,-1,1)
        mean_rewards = np.zeros(shape=(self.n_clusters, 2))
        mean_times = np.zeros(shape=(self.n_clusters, 2))

        for i, (l, r) in enumerate(zip(self.labels[1:], rewards_clip[1:])):
            total_r += self.gamma**t * r
            if l == l_p:
                t += 1
            else:
                if t>self.k:
                    mean_rewards[l_p,0] += total_r #/ (t+1)
                    mean_rewards[l_p,1] += 1
                    mean_times[l_p,0] += t
                    mean_times[l_p,1] += 1
                l_p = l
                total_r = 0
                t = 0

        for mr,mt in zip(mean_rewards, mean_times):
            mr[0] = mr[0] / mr[1]
            mt[0] = mt[0] / mt[1]

        return mean_rewards[:,0], mean_times[:,0]

    def calc_v_smdp(self,P):
        GAMMA = np.diag(self.gamma**self.skill_time)
        v = np.dot(np.linalg.pinv(np.eye(self.n_clusters)-np.dot(GAMMA,P)),self.r)
        return v

    def calc_v_policy(self,P):
        skill_time = np.zeros(shape=(self.n_clusters,1))
        r = np.zeros(shape=(self.n_clusters,1))
        for cluster_ind in xrange(self.n_clusters):
            cluster_policy_skills = np.nonzero(P[cluster_ind,:])
            cluster_skills = np.nonzero(self.P[cluster_ind,:])
            n_skills = len(cluster_policy_skills[0])

            for policy_skill_ind in xrange(n_skills):
                next_ind = cluster_policy_skills[0][policy_skill_ind]
                skill_ind = np.nonzero(cluster_skills[0]==next_ind)[0][0]
                r[cluster_ind] += P[cluster_ind,next_ind]*self.R_skills[cluster_ind][skill_ind,0]
                skill_time[cluster_ind] += P[cluster_ind,next_ind]*self.k_skills[cluster_ind][skill_ind,0]
        GAMMA = np.diag((self.gamma**skill_time)[:,0])
        v = np.dot(np.linalg.pinv(np.eye(self.n_clusters)-np.dot(GAMMA,P)),r)
        return v

    def calc_v_dqn(self, values):

        value_vec = np.zeros(shape=(self.n_clusters,2))
        for i,(l,v) in enumerate(zip(self.labels, values)):
            value_vec[l, 0] += v
            value_vec[l, 1] += 1

        # 1.5 normalize rewards
        for val in value_vec:
            val[0] = val[0]/val[1]

        return value_vec[:,0]

    def value_score(self):
        # v_dqn = (self.v_dqn-self.v_dqn.mean())/self.v_dqn.std()
        # v_smdp = (self.v_smdp-self.v_smdp.mean())/self.v_smdp.std()
        v_dqn = self.v_dqn
        v_smdp = self.v_smdp
        return np.linalg.norm(v_dqn-v_smdp)/np.linalg.norm(v_dqn)

    def policy_improvement(self):
        policy = []
        for cluster_ind in xrange(self.n_clusters):
            n_skills = len(self.skills[cluster_ind][0])
            val = np.zeros(shape=(n_skills,1))
            for skill_ind in xrange(n_skills):
                r = self.R_skills[cluster_ind][skill_ind,0]
                k = self.k_skills[cluster_ind][skill_ind,0]
                next_ind = self.skills[cluster_ind][0][skill_ind]
                val[skill_ind] = r+(self.gamma**k)*self.v_smdp[next_ind]
            policy.append((cluster_ind,self.skills[cluster_ind][0][np.argmax(val)]))
        return policy

    def get_smdp_edges(self):
        edges = []
        for i in xrange(self.n_clusters):
            for j in xrange(self.n_clusters):
                if self.P[i,j]>0:
                    edges.append((i,j))
        return edges

    def evaluate_greedy_policy(self,l):
        PP = np.copy(self.P)
        for trans in self.greedy_policy:
            p = PP[trans[0],:]
            p_greedy = np.zeros_like(p)
            p_greedy[trans[1]]=1
            p = l*p+(1-l)*p_greedy
            PP[trans[0],:]=p
        self.v_greedy = self.calc_v_policy(PP)

    def create_skills_model(self,P):
        skills = []
        for cluster_ind in xrange(self.n_clusters):
            skills.append(np.nonzero(P[cluster_ind,:]))
        l_p = int(self.labels[0])
        total_r = 0
        t = 0
        R_skills = []
        k_skills = []
        for cluster_ind in xrange(len(skills)):
            R_skills.append(np.zeros(shape=(len(skills[cluster_ind][0]),2)))
            k_skills.append(np.zeros(shape=(len(skills[cluster_ind][0]),2)))

        rewards_clip = np.clip(self.rewards,-1,1)
        for i, (l, r) in enumerate(zip(self.labels[1:], rewards_clip[1:])):
            total_r += self.gamma**t * r
            if l == l_p:
                t += 1
            else:
                if self.P[l_p,l]>0:
                    skill_index = np.flatnonzero(np.asarray(skills[l_p][0])==l)[0]
                    R_skills[l_p][skill_index,0] += total_r #/ (t+1)
                    R_skills[l_p][skill_index,1] += 1
                    k_skills[l_p][skill_index,0] += t
                    k_skills[l_p][skill_index,1] += 1
                l_p = int(l)
                total_r = 0
                t = 0

        for skill_r,skill_k in zip(R_skills, k_skills):
             skill_r[:,0] /= skill_r[:,1]
             skill_k[:,0] /= skill_k[:,1]

        return skills,R_skills,k_skills

    def calc_skill_indices(self):
        l_p = int(self.labels[0])
        current_skill = []
        skill_indices = [[] for i in range(self.n_clusters)]
        skill_list    = [[] for i in range(self.n_clusters)]
        skill_time    = [[] for i in range(self.n_clusters)]

        for i, l in enumerate(zip(self.labels[1:])):
            current_skill.append(i)
            if l[0] != l_p:
                if self.P[l_p,l[0]]>0:
                    skill_index = np.nonzero(skill_list[l_p]==l[0])[0] #find skill index in list
                    curr_length = len(current_skill)
                    if curr_length > self.k:
                        length = []
                        length.append(curr_length)

                        if len(skill_index) == 0: #if not found - append
                            skill_list[l_p].append(l[0])
                            skill_indices[l_p].append(current_skill)
                            skill_time[l_p].append(length)
                        else:
                            skill_indices[l_p][skill_index].extend(current_skill)
                            skill_time[l_p][skill_index].extend(length)

                l_p = l[0]
                current_skill = []
        return skill_indices,skill_list,skill_time

    def complete_smdp(self):
        # not needed in spatio-temporal
        self.skills, self.R_skills, self.k_skills = self.create_skills_model(self.P)
        self.greedy_policy = self.policy_improvement()
        self.v_greedy = self.evaluate_greedy_policy(0)
        self.skill_indices, self.skill_list,self.skill_time = self.calc_skill_indices()
