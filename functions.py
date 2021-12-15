import matplotlib.pyplot as plt
import time
import networkx as nx
import numpy.linalg as LA
from statistics import mean
import numpy as np
import math

start_time = time.time()

# -------------------------------------------------------------------------------------------


def illustrate_graph(G):

    pos = nx.spring_layout(G)
    nx.draw(G, pos, nodelist=[n for n in G.nodes if nx.get_node_attributes(G,'node_label')[n] == 0],
            node_color='tomato', edge_color='grey', node_size=10, arrows=True)
    nx.draw(G, pos, nodelist=[n for n in G.nodes if nx.get_node_attributes(G,'node_label')[n] == 1],
            node_color='limegreen', edge_color='grey', node_size=10, arrows=True)

    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def degree_distribution(G):

    in_degrees = {n: G.in_degree(n, weight='weight') for n in G.nodes()}
    out_degrees = {n: G.out_degree(n, weight='weight') for n in G.nodes()}
    # nx.in_degree_centrality(G)

    figure, axis = plt.subplots(1, 2)

    axis[0].hist([[in_degrees.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
                  [in_degrees.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
                 bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    axis[0].legend(loc='upper right')
    axis[0].set_xlabel('in-degree')
    axis[0].set_ylabel('node count')
    axis[0].set_title('in-degree distribution')

    axis[1].hist([[out_degrees.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
                  [out_degrees.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
                 bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    axis[1].legend(loc='upper right')
    axis[1].set_xlabel('out-degree')
    axis[1].set_ylabel('node count')
    axis[1].set_title('out-degree distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def betweenness_centrality(G):

    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

    plt.hist([[betweenness_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [betweenness_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('betweenness centrality')
    plt.ylabel('node count')
    plt.title('betweenness centrality distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def pagerank(G):

    pagerank = nx.pagerank(G, weight='weight')

    plt.hist([[pagerank.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [pagerank.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('PageRank value')
    plt.ylabel('node count')
    plt.title('PageRank distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def closeness_centrality(G):

    closeness_centrality = nx.closeness_centrality(G, distance='weight')

    plt.hist([[closeness_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [closeness_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('closeness centrality')
    plt.ylabel('node count')
    plt.title('closeness centrality distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def eigenvector_centrality(G):

    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=200)

    plt.hist([[eigenvector_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [eigenvector_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('eigenVector centrality')
    plt.ylabel('node count')
    plt.title('eigenVector centrality distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def katz_centrality(G):

    A = nx.adjacency_matrix(G)
    eigenvalues = LA.eigvals(A.todense())
    max_eigval = max(eigenvalues)
    katz_centrality = nx.katz_centrality(G, alpha=1/max_eigval-0.004, weight='weight')
    # beta=1, max_iter=1000 (default values)

    plt.hist([[katz_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [katz_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('katz centrality')
    plt.ylabel('node count')
    plt.title('katz centrality distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def clustering_coefficient(G):

    clustering_coefficient = nx.clustering(G, weight='weight')

    plt.hist([[clustering_coefficient.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [clustering_coefficient.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('clustering coefficient value')
    plt.ylabel('node count')
    plt.title('clustering coefficient distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def Average(lst):

    avg = 0
    if len(lst) > 0:
        avg = mean(lst)
    return avg

# -------------------------------------------------------------------------------------------


def average_path_length(G):

    shortest_path_len = dict(nx.shortest_path_length(G, weight='weight'))  # This is calculated among every pair of nodes
    avg_incoming_path_len = {v: Average([shortest_path_len[u][v] for u in G.nodes if u != v and nx.has_path(G,u,v)]) for v in G.nodes}
    avg_outgoing_path_len = {u: Average([shortest_path_len[u][v] for v in G.nodes if v != u and nx.has_path(G,u,v)]) for u in G.nodes}
    avg_path_len = {n: mean([avg_incoming_path_len[n], avg_outgoing_path_len[n]])
                       if (avg_incoming_path_len[n] > 0 and avg_outgoing_path_len[n] > 0)
                       else max(avg_incoming_path_len[n], avg_outgoing_path_len[n]) for n in G.nodes}

    plt.hist([[avg_path_len.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 0],
              [avg_path_len.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('average path length')
    plt.ylabel('node count')
    plt.title('average path length distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()


# -------------------------------------------------------------------------------------------


def harmonic_centrality(G):

    harmonic_centrality = nx.harmonic_centrality(G, distance='weight')

    plt.hist([[harmonic_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [harmonic_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('harmonic centrality')
    plt.ylabel('node count')
    plt.title('harmonic centrality distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def local_reaching_centrality(G):

    if nx.is_weighted(G, weight='weight'):
        l_reaching_centrality = {n: nx.local_reaching_centrality(G, n, weight='weight') for n in G.nodes()}
    else:
        l_reaching_centrality = {n: nx.local_reaching_centrality(G, n) for n in G.nodes()}

    plt.hist([[l_reaching_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 0],
              [l_reaching_centrality.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('local reaching centrality')
    plt.ylabel('node count')
    plt.title('local reaching centrality distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def average_neighbor_degree(G):

    avg_neighbor_degree = nx.average_neighbor_degree(G, weight='weight')

    plt.hist([[avg_neighbor_degree.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0],
              [avg_neighbor_degree.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]],
             bins=32, color=['tomato', 'limegreen'], label=['fake', 'genuine'])
    plt.legend(loc='upper right')
    plt.xlabel('average neighbor degree')
    plt.ylabel('node count')
    plt.title('average neighbor degree distribution')

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def measure_criteria_vec(index, G, G2):

    dic = {}
    vec = [[], []]

    if index == 1:
        dic = {n: G.in_degree(n, weight='weight') for n in G.nodes()}
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]
    elif index == 2:
        dic = {n: G.out_degree(n, weight='weight') for n in G.nodes()}
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]
    elif index == 3:
        dic = nx.betweenness_centrality(G2, weight='weight')
        vec[0] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2, 'node_label')[k] == 1]
    elif index == 4:
        dic = nx.pagerank(G, weight='weight')
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]
    elif index == 5:
        dic = nx.closeness_centrality(G2, distance='weight')
        vec[0] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2,'node_label')[k] == 1]
    elif index == 6:
        dic = nx.eigenvector_centrality(G, weight='weight', max_iter=200)
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]
    elif index == 7:
        A = nx.adjacency_matrix(G)
        eigenvalues = LA.eigvals(A.todense())
        max_eigval = max(eigenvalues)
        dic = nx.katz_centrality(G, alpha=1/max_eigval-0.004, weight='weight')
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 1]
    elif index == 8:
        dic = nx.clustering(G, weight='weight')
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 1]
    elif index == 9:
        shortest_path_len = dict(nx.shortest_path_length(G2, weight='weight'))
        avg_incoming_path_len = {v: Average([shortest_path_len[u][v]
                                             for u in G2.nodes if u != v and nx.has_path(G2, u, v)]) for v in G2.nodes}
        avg_outgoing_path_len = {u: Average([shortest_path_len[u][v]
                                             for v in G2.nodes if v != u and nx.has_path(G2, u, v)]) for u in G2.nodes}
        dic = {n: mean([avg_incoming_path_len[n], avg_outgoing_path_len[n]])
                  if (avg_incoming_path_len[n] > 0 and avg_outgoing_path_len[n] > 0)
                  else max(avg_incoming_path_len[n], avg_outgoing_path_len[n]) for n in G2.nodes}
        vec[0] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2, 'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2, 'node_label')[k] == 1]
    elif index == 10:
        dic = nx.harmonic_centrality(G2, distance='weight')
        vec[0] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G2.nodes if nx.get_node_attributes(G2,'node_label')[k] == 1]
    elif index == 11:
        if nx.is_weighted(G, weight='weight'):
            dic = {n: nx.local_reaching_centrality(G, n, weight='weight') for n in G.nodes()}
        else:
            dic= {n: nx.local_reaching_centrality(G, n) for n in G.nodes()}
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 1]
    elif index == 12:
        dic = nx.average_neighbor_degree(G, weight='weight')
        vec[0] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G,'node_label')[k] == 0]
        vec[1] = [dic.get(k) for k in G.nodes if nx.get_node_attributes(G, 'node_label')[k] == 1]

    return vec

# -------------------------------------------------------------------------------------------


def removing_outliers(x, y, num_of_removals):

    x_median = np.median(x)
    y_median = np.median(y)
    x_normalizer = (np.percentile(x, 25) + np.percentile(x, 75)) / 2
    y_normalizer = (np.percentile(y, 25) + np.percentile(y, 75)) / 2    
    n = len(x)
    # d = [math.sqrt(pow(x[i] - x_median, 2) + pow(y[i] - y_median, 2)) for i in range(n)]
    d = [math.sqrt(pow((x[i] - x_median)/x_normalizer, 2) + pow((y[i] - y_median)/y_normalizer, 2)) for i in range(n)]      # normalized distance
    for i in range(num_of_removals):
        max_index = d.index(max(d))
        del d[max_index]
        del x[max_index]
        del y[max_index]
    return x, y

# -------------------------------------------------------------------------------------------


def correlation(G, G2):

    criteria = ['in-degree', 'out-degree', 'betweenness centrality', 'PageRank', 'closeness centrality',
                'eigenVector centrality', 'Katz centrality', 'clustering coefficient', 'average path length',
                'harmonic centrality', 'local reaching centrality', 'average neighbor degree']
    print('Which two criteria do you want to plot a correlation between?')
    print('Criteria List:')
    for i in range(len(criteria)):
        print(str(i+1) + '. '+ criteria[i])
    x_index = int(input("\nFirst criteria: "))
    y_index = int(input("Second criteria: "))

    x = measure_criteria_vec(x_index, G, G2)
    y = measure_criteria_vec(y_index, G, G2)

    x[1], y[1] = removing_outliers(x[1], y[1], 1)   # Change the number of removals if needed

    plt.scatter(x[1], y[1], facecolors='none', edgecolors='limegreen', s=20, label='genuine')
    plt.scatter(x[0], y[0], facecolors='none', edgecolors='tomato', s=20, label='fake')

    plt.legend(loc='upper right')
    plt.xlabel(criteria[x_index-1])
    plt.ylabel(criteria[y_index-1])
    plt.title('correlation between ' + criteria[x_index-1] + ' and ' + criteria[y_index-1])

    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " sec")

    plt.show()

# -------------------------------------------------------------------------------------------


def in_deg_2ndVer(G, n):

    in_deg = 0
    edges = G.edges.data("weight", default=1)
    for item in edges:
        if item[1] == n:
            in_deg += item[2]

    return in_deg

# -------------------------------------------------------------------------------------------


def out_deg_2ndVer(G, n):

    out_deg = 0
    edges = G.edges.data("weight", default=1)
    for item in edges:
        if item[0] == n:
            out_deg += item[2]

    return out_deg

# -------------------------------------------------------------------------------------------


def deg_2ndVer(G, n):

    deg = in_deg_2ndVer(G, n) + out_deg_2ndVer(G, n)
    return deg

# -------------------------------------------------------------------------------------------


def avg_neighbor_deg_2ndVer(G):     # source & target= 'in+out'

    edges = G.edges.data("weight", default=1)
    neighbors = {n : set.union({item[1] for item in edges if item[0] == n},
                               {item[0] for item in edges if item[1] == n}) for n in G.nodes}
    avg_neighbor_deg = {i : 0 if deg_2ndVer(G, i) == 0 else
                            sum([deg_2ndVer(G, j) for j in neighbors[i]]) / deg_2ndVer(G, i) for i in G.nodes}

    return avg_neighbor_deg

# -------------------------------------------------------------------------------------------


def minDistance(G, dist, sptSet):

    min = sys.maxsize
    min_index = None

    for v in G.nodes:
        if dist[v] < min and sptSet[v] is False:
            min = dist[v]
            min_index = v

    return min_index

# -------------------------------------------------------------------------------------------


def dijkstra(G, src):

    edge_list = G.edges.data("weight", default=1)   # [(u1, v1, w1), (u2, v2, w2), ...]
    edge_dic = {}   # {(u1, v1): w1, (u2, v2): w2, ...}
    for item in edge_list:
        edge_dic[(item[0], item[1])] = item[2]

    dist = {n: sys.maxsize for n in G.nodes}
    dist[src] = 0
    sptSet = {n: False for n in G.nodes}

    for count in range(len(G.nodes)):

        u = minDistance(G, dist, sptSet)
        sptSet[u] = True

        for v in G.nodes:
            if (u, v) in edge_dic and sptSet[v] is False and dist[v] > dist[u] + edge_dic[(u, v)]:
                dist[v] = dist[u] + edge_dic[(u, v)]

    return {n: dist[n] for n in dist if dist[n] < sys.maxsize}

# -------------------------------------------------------------------------------------------


def shortest_path_len_2ndVer(G):

    result = {n: dijkstra(G, n) for n in G.nodes}
    return result

# -------------------------------------------------------------------------------------------


def has_path(G, u, v):

    if v in shortest_path_len_2ndVer(G)[u]:
        return True
    else:
        return False
    
# -------------------------------------------------------------------------------------------


def harmonic_centrality_2ndVer(G):

    return {u: sum([1/shortest_path_len_2ndVer(G)[v][u]
                    for v in G.nodes if v != u and has_path(G, v, u)])
            for u in G.nodes}
