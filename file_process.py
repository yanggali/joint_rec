import pandas as pd
import numpy as np
import os
import scipy.io as sio
import pickle as pk
import random

test_ratio = 0.2


def get_user_item_list(inputfile,col_names=["user","item"],sep="\t"):
    df = pd.read_csv(inputfile, sep=sep, names=col_names, engine="python")
    user_list = list(df["user"].unique())
    item_list = list(df["item"].unique())
    return user_list,item_list


# item_degree, user_nei, all_edges
def read_pkl_file(inputfile,train_user_nei_file,train_item_degree_file,train_all_edges_file,test_user_nei_file):
    user_nei = dict()
    matrix = pk.load(open(inputfile,'rb'))
    for row in range(matrix.shape[0]):
        if matrix[row,0] not in user_nei:
            user_nei[matrix[row,0]] = set()
        user_nei[matrix[row,0]].add(matrix[row,1])
    all_items = set(matrix[:,1])
    train_user_nei = dict()
    train_item_degree = dict()
    train_edges = list()
    test_user_item_neg = dict()
    for user,neis in user_nei.items():
        if len(neis)*test_ratio >= 1:
            test_len = int(len(neis)*test_ratio)
        elif len(neis)*test_ratio > test_ratio:
            test_len = 1
        else: continue
        test_items = set(random.sample(neis,test_len))
        train_items = neis - test_items
        for item in train_items:
            train_edges.append((user,item))
            if item not in train_item_degree:
                train_item_degree[item] = 0
            train_item_degree[item] += 1
        # generate 1000 negative samples for a user
        u_neg_items = all_items - neis
        neg_items = set(random.sample(u_neg_items, 1000))
        train_user_nei[user] = train_items
        test_user_item_neg[user] = dict()
        test_user_item_neg[user]["pos"] = test_items
        test_user_item_neg[user]["neg"] = neg_items
    print("user size:%d, review size:%d" % (len(user_nei),matrix.shape[0]))
    pk.dump(train_user_nei,open(train_user_nei_file,'wb'))
    pk.dump(train_item_degree,open(train_item_degree_file,'wb'))
    pk.dump(train_edges,open(train_all_edges_file,'wb'))
    pk.dump(test_user_item_neg,open(test_user_nei_file,'wb'))


def read_file(inputfile,col_names,sep="\t"):
    df = pd.read_csv(inputfile,sep=sep,names=col_names,engine="python")
    all_tuples = list()
    # col1 = set(df[col_names[0]])
    # col2 = set(df[col_names[1]])
    ver1_num = dict()
    ver2_num = dict()
    ver1_neighbours = dict()
    ver2_neighbours = dict()
    for index,row in df.iterrows():
        all_tuples.append((row[col_names[0]],row[col_names[1]]))
        if row[col_names[0]] not in ver1_neighbours:
            ver1_neighbours[row[col_names[0]]] = [row[col_names[1]]]
            ver1_num[row[col_names[0]]] = 1
        else:
            ver1_neighbours[row[col_names[0]]].append(row[col_names[1]])
            ver1_num[row[col_names[0]]] += 1
        if row[col_names[1]] not in ver2_neighbours:
            ver2_neighbours[row[col_names[1]]] = [row[col_names[0]]]
            ver2_num[row[col_names[1]]] = 1
        else:
            ver2_neighbours[row[col_names[1]]].append(row[col_names[0]])
            ver2_num[row[col_names[1]]] += 1
    return ver1_num,ver2_num,ver1_neighbours,ver2_neighbours,all_tuples


def get_group_users(inputfile,col_names=["groupid","users"],sep="\t"):
    df = pd.read_csv(inputfile,sep=sep,names=col_names,engine="python")
    group_users = dict()
    for index,row in df.iterrows():
        group_users[row["groupid"]] = [int(member) for member in list(str(row["users"]).strip().split(" "))]
    return group_users


def get_user_groups(inputfile,col_names=["groupid","users"],sep="\t"):
    df = pd.read_csv(inputfile,sep=sep,names=col_names,engine="python")
    user_groups = dict()
    user_groups_degree = dict()
    for index,row in df.iterrows():
        users = [int(user) for user in str(row["users"]).split(" ")]
        for user in users:
            if user not in user_groups:
                user_groups[user] = set()
                user_groups_degree[user] = 0
            user_groups[user].add(row["groupid"])
            user_groups_degree[user] += 1
    return user_groups,user_groups_degree


def get_emb(vertex_emb_file,vertex_emb):
    df = pd.read_csv(vertex_emb_file, sep="\t", names=["vertex", "emb"], engine="python")
    for index, row in df.iterrows():
        vertex_emb[row["vertex"]] = np.array(str(row["emb"]).strip().split(" ")).astype(np.float32)


def emb_to_file(filename,ver_emb):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename,'w') as fw:
        for k,v in ver_emb.items():
            write_str = str(k)+"\t"
            for e in v:
                write_str+=str(e)+" "
            write_str = write_str.strip()+"\n"
            fw.write(write_str)


def userweight_to_file(filename,user_weight):
    if os.path.exists(filename):
        os.remove(filename)
    user_weight_str = ""
    with open(filename, 'a') as fw:
        for user,weight in user_weight.items():
            user_weight_str += str(user)+"\t"+str(weight)+"\n"
        fw.write(user_weight_str)


def featureweight_to_file(filename,feature_weight):
    if os.path.exists(filename):
        os.remove(filename)
    feature_weight_str = ""
    with open(filename, 'a') as fw:
        for feature,weight in feature_weight.items():
            feature_weight_str += str(feature)+"\t"+str(weight)+"\n"
        fw.write(feature_weight_str)


def read_int_to_list(filename):
    read_list = list()
    read_str = ""
    with open(filename,'r') as fr:
        read_str = fr.readline()

    read_list = [int(node) for node in read_str.strip().split(" ")]
    return read_list

def read_float_to_list(filename):
    read_list = list()
    read_str = ""
    with open(filename, 'r') as fr:
        read_str = fr.readline().strip()
    read_list = [float(node) for node in read_str.strip().split(" ")]
    return read_list


def read_to_dict(filename):
    read_dict = dict()
    with open(filename,'r') as fr:
        for line in fr.readlines():
            line_str = line.split("\t")
            read_dict[int(line_str[0])] = [float(node) for node in line_str[1].split(" ")]
    return read_dict


def write_to_file(filename,str):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename,'w') as fw:
        fw.write(str)


def append_to_file(filename,str):
    with open(filename,'a') as fw:
        fw.write(str)


def read_to_map(filename,sep="\t",col_names=["col1","col2"]):
    read_map = dict()
    df = pd.read_csv(filename,sep=sep,names=col_names,engine="python")
    for index,row in df.iterrows():
        read_map[row[col_names[0]]] = row[col_names[1]]
    return  read_map


# return neg vertices degree and neg vertices list
def initial_negtable_data(user_user_view):
    user_nei = dict()
    neg_vertices_degree = dict()
    for t in user_user_view:
        if t[0] not in user_nei:
            user_nei[t[0]] = list()
        user_nei[t[0]].append(t[1])
        if t[1] not in neg_vertices_degree:
            neg_vertices_degree[t[1]] = 0
        neg_vertices_degree[t[1]] += 1
    neg_vertices_list = list(neg_vertices_degree.keys())
    return user_nei,neg_vertices_degree,neg_vertices_list


# return neg vertices degree and neg vertices list
def initial_realedges_data(user_user_view):
    user_nei = dict()
    neg_vertices_degree = dict()
    all_vertices_set = set()
    for t in user_user_view:
        if t[0] not in all_vertices_set:
            all_vertices_set.add(t[0])
        if t[0] not in user_nei:
            user_nei[t[0]] = list()
        user_nei[t[0]].append(t[1])
        if t[1] not in all_vertices_set:
            all_vertices_set.add(t[1])
        if t[1] not in neg_vertices_degree:
            neg_vertices_degree[t[1]] = 0
        neg_vertices_degree[t[1]] += 1
    neg_vertices_list = list(neg_vertices_degree.keys())
    return user_nei,neg_vertices_degree,neg_vertices_list,all_vertices_set


# prepare data for train model
def initial_data(rating_file,trust_file,train_user_pkl,train_item_degree_pkl,test_user_pkl,edges_1,edges_2):
    train_user_dict = dict()
    train_item_degree_dict = dict()
    test_user_dict = dict()
    train_edges_1 = list()
    train_edges_2 = list()
    # all user_items
    user_nei = dict()
    rating_matrix = pk.load(open(rating_file, 'rb'))
    for row in range(rating_matrix.shape[0]):
        if rating_matrix[row,0] not in user_nei:
            user_nei[rating_matrix[row,0]] = dict()
        user_nei[rating_matrix[row,0]][rating_matrix[row,1]] = rating_matrix[row,2]
    all_items = set(rating_matrix[:,1])
    for user,neis in user_nei.items():
        train_user_dict[user] = dict()
        test_user_dict[user] = dict()
        if len(neis)*test_ratio >= 1:
            test_len = int(len(neis)*test_ratio)
        elif len(neis)*test_ratio > test_ratio:
            test_len = 1
        else: continue
        test_items = set(random.sample(neis.keys(),test_len))
        train_items = neis.keys() - test_items
        for item in train_items:
            train_edges_1.append((user,item))
            if item not in train_item_degree_dict:
                train_item_degree_dict[item] = 0
            train_item_degree_dict[item] += 1
        # generate 1000 negative samples for a user
        u_neg_items = all_items - neis.keys()
        neg_items = set(random.sample(u_neg_items, 1000))
        train_user_dict[user]["item"] = {item:neis[item] for item in train_items}
        test_user_dict[user]["pos_item"] = {item:neis[item] for item in test_items}
        test_user_dict[user]["neg_item"] = neg_items

    # input trust_network
    trust_matrix = pk.load(open(trust_file,'rb'))
    user_friends = dict()
    for row in range(trust_matrix.shape[0]):
        user1,user2 = trust_matrix[row, 0],trust_matrix[row, 1]
        if user1 not in user_friends:
            user_friends[user1] = dict()
        if train_user_dict.get(user1,0)!=0 and train_user_dict.get(user2,0)!=0:
            user_friends[user1][user2] = cal_trust(train_user_dict[user1].get("item",0),
                                               train_user_dict[user2].get("item",0))
    all_friends = set(trust_matrix[:, 1])
    for user,neis in user_friends.items():
        if len(neis)*test_ratio >= 1:
            test_len = int(len(neis)*test_ratio)
        elif len(neis)*test_ratio > test_ratio:
            test_len = 1
        else: continue
        test_friends = set(random.sample(neis.keys(),test_len))
        train_friends = neis.keys() - test_friends
        for friend in train_friends:
            train_edges_2.append((user,friend))
            if friend not in train_user_dict:
                train_user_dict[friend] = dict()
            if "degree" not in train_user_dict[friend]:
                train_user_dict[friend]["degree"] = 0
            train_user_dict[friend]["degree"] += 1
        # generate 1000 negative samples for a user
        u_neg_fris = all_friends - neis.keys()
        neg_fris = set(random.sample(u_neg_fris, 1000))
        train_user_dict[user]["friend"] = {friend:neis[friend] for friend in train_friends}
        test_user_dict[user]["pos_friend"] = {friend:neis[friend] for friend in test_friends}
        test_user_dict[user]["neg_friend"] = neg_fris
    pk.dump(train_user_dict,open(train_user_pkl,'wb'))
    pk.dump(test_user_dict,open(test_user_pkl,'wb'))
    pk.dump(train_items,open(train_item_degree_pkl,'wb'))
    pk.dump(train_edges_1,open(edges_1,'wb'))
    pk.dump(train_edges_2,open(edges_2,'wb'))


def cal_trust(user1_items,user2_items):
    if user1_items == 0 or user2_items == 0:
        return 0.001
    inters = user1_items.keys()&user2_items.keys()
    if len(inters) == 0:
        return 0.001
    fenzi = sum([user1_items[i] for i in inters])
    fenmu = sum(user1_items.values())
    return fenzi/fenmu
# rating_file = "./raw data/ciao/ratings_data.txt"
# trust_file = "./raw data/ciao/trust_data.txt"
# out_rating_file = "./data/dataset/ciao/rating.pkl"
# out_trust_file = "./data/dataset/ciao/trustnetwork.pkl"
# out_rating_file = "./data/dataset/ciao/rating.pkl"
# out_trust_file = "./data/dataset/ciao/trustnetwork.pkl"
# train_user_file = "./data/dataset/ciao/train/train_user.pkl"
# train_item_degree = "./data/dataset/ciao/train/train_item_degree.pkl"
# train_edges_1 = "./data/dataset/ciao/train/train_edges_1.pkl"
# train_edges_2 = "./data/dataset/ciao/train/train_edges_2.pkl"
# test_user_file = "./data/dataset/ciao/test/test_user.pkl"


# rating_file = "./raw data/epinions/ratings_data.txt"
# trust_file = "./raw data/epinions/trust_data.txt"
# out_rating_file = "./data/dataset/epinions/rating.pkl"
# out_trust_file = "./data/dataset/epinions/trustnetwork.pkl"
# train_user_file = "./data/dataset/epinions/train/train_user.pkl"
# train_item_degree = "./data/dataset/epinions/train/train_item_degree.pkl"
# train_edges_1 = "./data/dataset/epinions/train/train_edges_1.pkl"
# train_edges_2 = "./data/dataset/epinions/train/train_edges_2.pkl"
# test_user_file = "./data/dataset/epinions/test/test_user.pkl"
out_rating_file = "./data/dataset/ciao/rating.pkl"
out_trust_file = "./data/dataset/ciao/trustnetwork.pkl"
train_user_file = "./data/dataset/ciao/train/train_user.pkl"
train_item_degree = "./data/dataset/ciao/train/train_item_degree.pkl"
train_edges_1 = "./data/dataset/ciao/train/train_edges_1.pkl"
train_edges_2 = "./data/dataset/ciao/train/train_edges_2.pkl"
test_user_file = "./data/dataset/ciao/test/test_user.pkl"
if __name__ == "__main__":
    initial_data(out_rating_file,out_trust_file,train_user_file,train_item_degree,test_user_file,
                 train_edges_1,train_edges_2)
#     asd


