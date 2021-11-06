import networkx as nx
import pandas as pd

# -------------------------------------------------------------------------------------------
# read files

user_fields = ['id']
tweet_fields = ['id', 'user_id', 'retweeted_status_id']

fake_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\social_spambots_2\users.csv', usecols=user_fields)
fake_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\social_spambots_2\tweets.csv', usecols=tweet_fields, encoding='latin1')
genuine_user_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\genuine_accounts\users.csv', usecols=user_fields)
genuine_tweet_df = pd.read_csv(r'C:\Users\Sara\Desktop\twitterProject\data\cresci-2017\genuine_accounts\tweets.csv', usecols=tweet_fields, encoding='latin1')

user_dic = {u: 0 for u in fake_user_df['id']}
user_dic.update({u: 1 for u in genuine_user_df['id']})

fake_tweet_df = fake_tweet_df[fake_tweet_df['retweeted_status_id'] != 0]    # len=15,485
fake_tweet_df.reset_index(drop=True, inplace=True)
genuine_tweet_df = genuine_tweet_df[genuine_tweet_df['retweeted_status_id'] != 0]   # len=738,089
genuine_tweet_df.reset_index(drop=True, inplace=True)

tweet_list = [[], []]   # tweet_to_user_list
tweet_list[0] = fake_tweet_df['id'].values.tolist() + genuine_tweet_df['id'].values.tolist()
tweet_list[1] = fake_tweet_df['user_id'].values.tolist() + genuine_tweet_df['user_id'].values.tolist()

# -------------------------------------------------------------------------------------------
# create a dictionary of edges along with their weights

retweet_dic = {}

for i in range(len(fake_tweet_df)):
    try:
        t_index = tweet_list[0].index(fake_tweet_df['retweeted_status_id'][i])
        u_id = tweet_list[1][t_index]
        if not ((u_id, fake_tweet_df['user_id'][i]) in retweet_dic):
            retweet_dic[(u_id, fake_tweet_df['user_id'][i])] = 1
        else:
            retweet_dic[(u_id, fake_tweet_df['user_id'][i])] += 1
    except:     # unknown retweeted users
        continue

for i in range(len(genuine_tweet_df)):
    try:
        t_index = tweet_list[0].index(genuine_tweet_df['retweeted_status_id'][i])
        u_id = tweet_list[1][t_index]
        if not ((u_id, genuine_tweet_df['user_id'][i]) in retweet_dic):
            retweet_dic[(u_id, genuine_tweet_df['user_id'][i])] = 1
        else:
            retweet_dic[(u_id, genuine_tweet_df['user_id'][i])] += 1
    except:     # unknown retweeted users
        continue

# -------------------------------------------------------------------------------------------
# convert retweet_dic to retweet_list

retweet_list = []   # tweet_to_retweet_list
retweet_list2 = []

for i in retweet_dic:
    retweet_list.append((i[0], i[1], retweet_dic[i]))
    retweet_list2.append((i[0], i[1], 1 / retweet_dic[i]))

M = len(retweet_list)

# -------------------------------------------------------------------------------------------
# create object

G = nx.DiGraph()
G2 = nx.DiGraph()

ndxs = list(user_dic.keys())
edges = retweet_list
edges2 = retweet_list2

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

nx.write_gexf(G, r'C:\Users\Sara\Desktop\twitterProject\results\storage\retweetGraph.gexf')
nx.write_gexf(G2, r'C:\Users\Sara\Desktop\twitterProject\results\storage\retweetGraph2.gexf')
