import functions
import networkx as nx


def unweightedGraph_runFunctions(G):

    functions.illustrate_graph(G)
    functions.degree_distribution(G)
    functions.betweenness_centrality(G)
    functions.pagerank(G)
    functions.closeness_centrality(G)
    functions.eigenvector_centrality(G)
    functions.katz_centrality(G)
    functions.clustering_coefficient(G)
    functions.average_path_length(G)
    
    functions.harmonic_centrality(G)
    functions.local_reaching_centrality(G)
    functions.average_neighbor_degree(G)

    functions.correlation(G, G)

# -------------------------------------------------------------------------------------------


def weightedGraph_runFunctions(G, G2):

    functions.illustrate_graph(G2)
    functions.degree_distribution(G)
    functions.betweenness_centrality(G2)
    functions.pagerank(G)
    functions.closeness_centrality(G2)
    try:
        functions.eigenvector_centrality(G)
    except Exception as e:
        print('eigenVector_centrality function: ' + str(e))
    try:
        functions.katz_centrality(G)
    except Exception as e:
        print('katz_centrality function: ' + str(e))
    functions.clustering_coefficient(G)
    functions.average_path_length(G2)
    
    functions.harmonic_centrality(G2)
    functions.local_reaching_centrality(G)
    functions.average_neighbor_degree(G)

    functions.correlation(G, G2)

# -------------------------------------------------------------------------------------------


print("Which graph do you want to check in detail?")
print("1. FollowerFollowingGraph")
print("2. commentGraph")
print("3. retweetGraph")
which_graph = input()

if which_graph == '1':
    G = nx.read_gexf(r'C:\Users\Sara\Desktop\twitterProject\results\storage\FollowerFollowingGraph.gexf')
    unweightedGraph_runFunctions(G)
elif which_graph == '2':
    G = nx.read_gexf(r'C:\Users\Sara\Desktop\twitterProject\results\storage\commentGraph.gexf')
    G2 = nx.read_gexf(r'C:\Users\Sara\Desktop\twitterProject\results\storage\commentGraph2.gexf')
    weightedGraph_runFunctions(G, G2)
elif which_graph == '3':
    G = nx.read_gexf(r'C:\Users\Sara\Desktop\twitterProject\results\storage\retweetGraph.gexf')
    G2 = nx.read_gexf(r'C:\Users\Sara\Desktop\twitterProject\results\storage\retweetGraph2.gexf')
    weightedGraph_runFunctions(G, G2)
else:
    print("Unsupported input")
