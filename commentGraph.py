import networkx as nx
import pandas as pd

# -------------------------------------------------------------------------------------------
# read files

user_fields = ['id']
tweet_fields = ['user_id', 'in_reply_to_user_id']

fake_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\social_spambots_2\users.csv', usecols=user_fields)
fake_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\social_spambots_2\tweets.csv', usecols=tweet_fields, encoding='latin1')
genuine_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\genuine_accounts\users.csv', usecols=user_fields)
genuine_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\genuine_accounts\tweets.csv', usecols=tweet_fields, encoding='latin1')

user_dic = {u: 0 for u in fake_user_df['id']}
user_dic.update({u: 1 for u in genuine_user_df['id']})

# -------------------------------------------------------------------------------------------
# create a dictionary of edges along with their weights

comment_dic = {}

for i in range(len(fake_tweet_df)):
    if fake_tweet_df['in_reply_to_user_id'][i] == 0 or \
            not(fake_tweet_df['in_reply_to_user_id'][i] in user_dic):    # unknown users
        continue
    if not ((fake_tweet_df['in_reply_to_user_id'][i], fake_tweet_df['user_id'][i]) in comment_dic):
        comment_dic[(fake_tweet_df['in_reply_to_user_id'][i], fake_tweet_df['user_id'][i])] = 1
    else:
        comment_dic[(fake_tweet_df['in_reply_to_user_id'][i], fake_tweet_df['user_id'][i])] += 1

for i in range(len(genuine_tweet_df)):
    if genuine_tweet_df['in_reply_to_user_id'][i] == 0 or \
            not(genuine_tweet_df['in_reply_to_user_id'][i] in user_dic):  # unknown users
        continue
    if not ((genuine_tweet_df['in_reply_to_user_id'][i], genuine_tweet_df['user_id'][i]) in comment_dic):
        comment_dic[(genuine_tweet_df['in_reply_to_user_id'][i], genuine_tweet_df['user_id'][i])] = 1
    else:
        comment_dic[(genuine_tweet_df['in_reply_to_user_id'][i], genuine_tweet_df['user_id'][i])] += 1

# -------------------------------------------------------------------------------------------
# convert comment_dic to comment_list

comment_list = []   # tweet_to_comment_list
comment_list2 = []

for i in comment_dic:
    comment_list.append((i[0], i[1], comment_dic[i]))
    comment_list2.append((i[0], i[1], 1/comment_dic[i]))

M = len(comment_list)

# -------------------------------------------------------------------------------------------
# create object

G = nx.DiGraph()
G2 = nx.DiGraph()

ndxs = list(user_dic.keys())
edges = comment_list
edges2 = comment_list2

G.add_nodes_from(ndxs)
nx.set_node_attributes(G, user_dic, name='node_label')
G.add_weighted_edges_from(edges)
G2.add_nodes_from(ndxs)
nx.set_node_attributes(G2, user_dic, name='node_label')
G2.add_weighted_edges_from(edges2)

# -------------------------------------------------------------------------------------------
# remove nodes with zero degrees

remove_list = [n for n in G.nodes() if G.degree(n) == 0]
G.remove_nodes_from(remove_list)
G2.remove_nodes_from(remove_list)

# -------------------------------------------------------------------------------------------
# store the graphs in a file

nx.write_gexf(G, r'C:\Users\Sara\Desktop\twitterProject\results\storage\commentGraph.gexf')
nx.write_gexf(G2, r'C:\Users\Sara\Desktop\twitterProject\results\storage\commentGraph2.gexf')
