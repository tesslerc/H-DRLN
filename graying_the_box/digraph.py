import networkx as nx
import matplotlib.pyplot as plt
import pylab
import pickle
import numpy as np
import common

def draw_transition_table(transition_table, cluster_centers, meanscreen, tsne ,color, black_edges=None, red_edges=None, title=None):
    G  = nx.DiGraph()
    edge_colors = []

    if red_edges is not None:
        for e in red_edges:
            G.add_edges_from([e], weight=np.round(transition_table[e[0],e[1]]*100)/100)
            edge_colors.append('red')

    if black_edges is not None:
        if red_edges is not None:
            black_edges = list(set(black_edges)-set(red_edges))

        for e in black_edges:
            G.add_edges_from([e], weight=np.round(transition_table[e[0],e[1]]*100)/100)
            edge_colors.append('black')


    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])

    node_labels = {node:node for node in G.nodes()};
    counter=0
    for key in node_labels.keys():
        node_labels[key] =  counter
        counter+=1

    if title is None:
        fig = plt.figure('SMDP')
        fig.clear()
    else:
        fig = plt.figure(title)

    plt.scatter(tsne[:,0],tsne[:,1],s= np.ones(tsne.shape[0])*2,facecolor=color, edgecolor='none')
    pos = cluster_centers[:,0:2]
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,label_pos=0.65,font_size=9)
    nx.draw_networkx_labels(G, pos, labels=node_labels,font_color='w',font_size=8)
    nx.draw(G,pos,cmap=plt.cm.brg,edge_color=edge_colors)


    ######Present images on nodes
    ax = plt.subplot(111)
    plt.axis('off')
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform
    cut = 1.01
    xmax = cut * max(tsne[:,0])
    ymax = cut * max(tsne[:,1])
    xmin = cut * min(tsne[:,0])
    ymin = cut * min(tsne[:,1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    h = 70.0
    w  = 70.0
    counter= 0
    for node in G:
        xx, yy = trans(pos[node])
        # axes coordinates
        xa, ya = trans2((xx, yy))

        # this is the image size
        piesize_1 = (300.0 / (h*80))
        piesize_2 = (300.0 / (w*80))
        p2_2 = piesize_2 / 2
        p2_1 = piesize_1 / 2
        a = plt.axes([xa - p2_2, ya - p2_1, piesize_2, piesize_1])
        G.node[node]['image'] = meanscreen[counter]
        #display it
        a.imshow(G.node[node]['image'])
        a.set_title(node_labels[counter])
        #turn off the axis from minor plot
        a.axis('off')
        counter+=1
    plt.draw()

def draw_transition_table_no_image(transition_table,cluster_centers):
    G  = nx.DiGraph()
    G2 = nx.DiGraph()

    # print transition_table.sum(axis=1)

    transition_table = (transition_table.transpose()/transition_table.sum(axis=1)).transpose()
    transition_table[np.isnan(transition_table)]=0
    # print(transition_table)
    # transition_table = (transition_table.transpose()/transition_table.sum(axis=1)).transpose()
    # print transition_table
    # print transition_table.sum(axis=0)
    # assert(np.all(transition_table.sum(axis=0)!=0))
    transition_table[transition_table<0.1]=0

    pos = cluster_centers[:,0:2]
    m,n = transition_table.shape

    for i in range(m):
        for j in range(n):
            if transition_table[i,j]!=0:
                G.add_edges_from([(i, j)], weight=np.round(transition_table[i,j]*100)/100)
                G2.add_edges_from([(i, j)], weight=np.round(transition_table[i,j]*100)/100)
    values = cluster_centers[:,2]

    red_edges = []
    edges_sizes =[]
    for i in range(n):
        trans = transition_table[i,:]
        indices = (trans!=0)
        index = np.argmax(cluster_centers[indices,2])
        counter = 0
        for j in range(len(indices)):
            if indices[j]:
                if counter == index:
                    ind = j
                    break
                else:
                    counter+=1
        edges_sizes.append(ind)
        red_edges.append((i,ind))
    # print(red_edges)
    # sizes = 3000*cluster_centers[:,3]
    sizes = np.ones_like(values)*500
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
    edge_colors = ['black' for edge in G.edges()]
    # edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]


    node_labels = {node:node for node in G.nodes()};
    counter=0
    for key in node_labels.keys():
        # node_labels[key] =  np.round(100*cluster_centers[counter,3])/100
        node_labels[key] =  counter
        counter+=1

    fig = plt.figure()
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,label_pos=0.65,font_size=9)
    nx.draw_networkx_labels(G, pos, labels=node_labels,font_color='w',font_size=8)
    nx.draw(G,pos, node_color = values,cmap=plt.cm.brg, node_size=np.round(sizes),edge_color=edge_colors,edge_cmap=plt.cm.Reds)


    ######Present images on nodes
    # plt.show()

#
def test():
    gamename = 'breakout' #breakout pacman
    transition_table = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'transition_table.bin'))
    cluster_centers = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'cluster_centers.bin'))
    cluster_std = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'cluster_std.bin'))
    cluster_med = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'cluster_med.bin'))
    cluster_min = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'cluster_min.bin'))
    cluster_max = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'cluster_max.bin'))
    meanscreen = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'meanscreen.bin'))
    cluster_time = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'cluster_time.bin'))
    tsne = common.load_hdf5('lowd_activations', 'data/' + 'breakout' + '/'+'120k/')
    q_hdf5 = common.load_hdf5('qvals', 'data/' + 'breakout' + '/'+'120k/')

    num_frames = 120000
    V          = np.zeros(shape=(num_frames))

    for i in range(0,num_frames):
        V[i] = max(q_hdf5[i])
    V = V/V.max()
    draw_transition_table(transition_table,cluster_centers,meanscreen,cluster_time,tsne,V)
    plt.show()
# test()



# stdscreen = pickle.load(file('/home/tom/git/graying_the_box/data/'+gamename+'/120k' + '/knn/' + 'stdscreen.bin'))
# #
# a = 1
# b = 0
# c = 0
# screen = a*meanscreen  + c*stdscreen

#                                      facecolor = self.color,
#                                      edgecolor='none',picker=5)
# draw_transition_table_no_image(transition_table,cluster_centers)

    # transition_table = pickle.load(file('/home/tom/git/graying_the_box/data/seaquest/120k' + '/knn/' + 'transition_table.bin'))
    # transition_table[transition_table<0.1]=0
    # cluster_centers = pickle.load(file('/home/tom/git/graying_the_box/data/seaquest/120k' + '/knn/' + 'cluster_centers.bin'))

    # pos2 = np.zeros(shape=(cluster_centers.shape[0],2))
    # pos2[:,0] = cluster_time[:,0]
    # pos2[:,1] =  cluster_centers[:,1]
    # plt.figure()
    # nx.draw_networkx_edge_labels(G2,pos2,edge_labels=edge_labels,label_pos=0.8,font_size=8)
    # nx.draw_networkx_labels(G2, pos2, labels=node_labels,font_color='w',font_size=8)
    # nx.draw(G2,pos2, node_color = values,cmap=plt.cm.brg, node_size=np.round(sizes),edge_color=edge_colors,edge_cmap=plt.cm.Reds)
