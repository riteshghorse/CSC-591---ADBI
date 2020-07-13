import sys
import pandas as pd
from igraph import *
import numpy as np
from scipy.spatial import distance



def create_graph():
    attrlist = pd.read_csv('./data/fb_caltech_small_attrlist.csv')
    with open('./data/fb_caltech_small_edgelist.txt', 'r') as f:
        edges = f.readlines()
        edges = [(int(edge.strip().split()[0]), int(edge.strip().split()[1])) for edge in edges]
    
    vertices = attrlist.shape[0]
    print(vertices)
    print(len(edges))
    g = Graph()
    g.add_vertices(vertices)
    g.add_edges(edges)
    for attribute in attrlist.columns:
        g.vs[attribute] = attrlist[attribute]
    g.es['weight'] = [1] * len(edges)
    return g

def similarity(i, j, g):
    xi = g.vs[i].attributes().values()
    xj = g.vs[j].attributes().values()
    # xi = xi.reshape([-1,1])
    # xj = xj.reshape([-1,1])
    return 1 - distance.cosine(xi, xj)

def DQ_Newman(g, i, j, community):
    # when x is in its original community
    q_original = g.modularity(community, weights='weight')
    # move x to community c
    t = community[i]
    community[i] = j
    q_new = g.modularity(community, weights='weight')
    community[i] = t
    # gain
    delta_q_newman = q_new - q_original
    return delta_q_newman

def DQ_Attr(g, i, j, sim, community):
    # delta_q_attr = similarity(j, i, g)
    delta = 0.0
    clusters = []
    for m,c in enumerate(community):
        if c == j:
            clusters.append(m)
    for c in clusters:
        delta += sim[i][c]
    delta = delta/(len(clusters)*len(set(community)))
    return delta

def delta_q(g, i, j, sim, alpha, community):
    dq_newman = DQ_Newman(g, i, j, community)
    dq_attr = DQ_Attr(g, i, j, sim, community)
    delta = (alpha * dq_newman) + ((1-alpha) * dq_attr)
    return delta

def phase_1(g, sim, alpha, community):
    flag = 0
    iterations = 0
    vcount = g.vcount()
    while flag == 0 and iterations < 15:
        flag = 1
        for i in range(vcount):
            x = -1      # node for which gain is maximum
            dq_x = 0.0
            communities = list(set(community))    # unique set of communities
            for j in communities:
                if community[i] != j:
                    dq = delta_q(g, i, j, sim, alpha, community)
                    if dq > dq_x:
                        dq_x = dq
                        x = j
            if dq_x > 0.0 and x != -1:
                community[i] = x
                flag = 0
        iterations += 1
    return community

def combine_communities(communities):
    combined = []
    indexing = {}
    count = 0
    for community in communities:
        if community in indexing:
            combined.append(indexing[community])
        else:
            combined.append(count)
            indexing[community] = count
            count += 1
    
    return combined

def phase_2(g, sim_dup, community):
    communities = combine_communities(community)
    communities_count = len(set(communities))
    sim = np.zeros((communities_count, communities_count))
    clusters = list(Clustering(communities))

    for i in range(communities_count):
        for j in range(communities_count):
            score = 0.0
            for m in clusters[i]:
                for n in clusters[j]:
                    score += sim_dup[m][n]
            sim[i][j] = score

    g.contract_vertices(communities)
    g.simplify(combine_edges=sum)

def Q_Attr(g, sim, community):
    clusters = list(Clustering(community))
    total = 0.0
    for k in clusters:
        temp = 0.0
        for i in k:
            for j in k:
                if i != j:
                    temp += sim[i][j]
        temp /= len(k)
        total += temp
    q_attr = total/(len(set(c00000000000000000000000000000000000000000000000
    return q_attr

def composite_modularity(g, sim, alpha, community):
    Q = g.modularity(community, weights='weight') + Q_Attr(g, sim, community)
    return Q

def output(clusters, alpha):
    with open('communities_'+alpha+'.txt', 'w+') as f:
        for cluster in clusters:
            for i in range(len(cluster)-1):
                f.write(str(cluster[i])+',')
            f.write(str(cluster[-1])+'\n')
    
        f.close()
            


if __name__ == "__main__":
    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])  
        g = create_graph()
        vcount = g.vcount()

        #initialize similarity matrix by cosine similarity
        sim = np.zeros((vcount, vcount))
        for i in range(vcount):
            for j in range(vcount):
                sim[i][j] = similarity(i, j, g)

        # randomly assign communities
        communities = [i for i in range(vcount)]

        sim_dup = np.array(sim)
        
        # run phase 1 of the algorithm
        communities = phase_1(g, sim, alpha, communities)
        print("Number of communities in phase 1: "+str(len(set(communities))))
        
        communities = combine_communities(communities)
        clusters_phase_1 = list(Clustering(communities))
        # print(len(set(communities)))

        # composite modularity for phase 1
        Q_phase_1 = composite_modularity(g, sim, alpha, communities)


        # phase 2 starts
        # combine the nodes in a community
        phase_2(g, sim_dup, communities)

        # apply phase_1 on new grouped clusters
        vcount = g.vcount()
        larger_communities = [i for i in range(vcount)]
        larger_communities = phase_1(g, sim, alpha, larger_communities)
        compact_larger_communities = combine_communities(larger_communities)
        clusters_phase_2 = list(Clustering(compact_larger_communities))
        Q_phase_2 = composite_modularity(g, sim, alpha, communities)


        community = []
        communities_new = combine_communities(communities)
        clusters_phase_1 = list(Clustering(communities_new))
        for cluster in clusters_phase_2:
            temp = list()
            for node in cluster:
                temp.extend(clusters_phase_1[node])
            community.append(temp)

        alp = 1
        if alpha == 0.0:
            alp = 0
        elif alpha == 0.5:
            alp = 5
        elif alpha == 1.0:
            alp = 1
        
        if Q_phase_1 > Q_phase_2:
            output(clusters_phase_1, str(alp))
            print clusters_phase_1
        else:
            output(clusters_phase_2, str(alp))
            print clusters_phase_2


