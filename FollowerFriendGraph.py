import networkx as nx
import pandas as pd

# -------------------------------------------------------------------------------------------
# read files

user_fields = ['id']

fake_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\FSF\users.csv', usecols=user_fields)
fake_follower_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\FSF\followers.csv')
fake_friend_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\FSF\friends.csv')
genuine_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\E13\users.csv', usecols=user_fields)
genuine_follower_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\E13\followers.csv')
genuine_friend_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2015\E13\friends.csv')

user_dic = {u: 0 for u in fake_user_df['id']}
user_dic.update({u: 1 for u in genuine_user_df['id']})

follow_list = fake_follower_df.values.tolist() + fake_friend_df.values.tolist()\
              + genuine_follower_df.values.tolist() + genuine_friend_df.values.tolist()

# -------------------------------------------------------------------------------------------
# remove unknown followers/friends from the follow_list

M = len(follow_list)
i = 0
while i < M:
    if not(follow_list[i][0] in user_dic) or not(follow_list[i][1] in user_dic):
        del follow_list[i]
        i -= 1
        M -= 1
    i += 1

# -------------------------------------------------------------------------------------------
# create object

G = nx.DiGraph()
ndxs = list(user_dic.keys())
edges = []
for i in range(M):
    edges.append((follow_list[i][0], follow_list[i][1]))
G.add_nodes_from(ndxs)
nx.set_node_attributes(G, user_dic, name='node_label')
G.add_edges_from(edges)

# -------------------------------------------------------------------------------------------
# remove nodes with zero degrees

remove_list = [n for n in G.nodes() if G.degree(n) == 0]
G.remove_nodes_from(remove_list)

# -------------------------------------------------------------------------------------------
# store the graph in a file

nx.write_gexf(G, r'C:\Users\Sara\Desktop\twitterProject\results\storage\FollowerFollowingGraph.gexf')
