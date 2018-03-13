# input:test_group_event
#       group_users
#       user_emb,luser_emb,ruser_emb,item_emb
# output:hit@n, MRR
import pandas as pd
import numpy as np
import math
import pickle as pk
from file_process import get_group_users

test_user_file = "./data/dataset/test/test_user.pkl"
user_emb_file = "./data/output/user"
item_emb_file = "./data/output/item"
map_matrix_file = "./data/output/map_matrix"

DIM = 50


def get_emb(vertex_emb_file):
    df = pd.read_csv(vertex_emb_file, sep="\t", names=["vertex", "emb"], engine="python")
    vertex_emb = dict()
    for index, row in df.iterrows():
        vertex_emb[row["vertex"]] = np.array(str(row["emb"]).strip().split(" ")).astype(np.float32)
    return vertex_emb


user_emb = get_emb(user_emb_file)
item_emb = get_emb(item_emb_file)
map_matrix = pk.load(open(map_matrix_file,'rb'))


def cal_user_friend_topk(user,friend,candi_list):
    global user_emb
    user_emb = user_emb.get(user)
    friend_emb = user_emb.get(friend)
    rec_friends_dict = dict()
    rec_friends_dict[friend] = user_emb.dot(friend_emb)
    for candi in candi_list:
        rec_friends_dict[candi] = user_emb.dot(user_emb.get(candi))
    # sort recommendation list
    sorted_rec_friends_dict = sorted(rec_friends_dict.items(), key=lambda d: d[1], reverse=True)
    user_k_map = dict()
    rr = 0
    for k in top_k:
        rank = hit = 0
        for t in sorted_rec_friends_dict:
            rank += 1
            if friend == t[0]:
                if rank <= k:
                    hit = 1
                rr = 1 / float(rank)
                break
        user_k_map[k] = hit
    return user_k_map, rr


def cal_user_item_topk(user,item,candi_list):
    global user_emb,item_emb
    user_emb = user_emb.get(user)
    item_emb = item_emb.get(item)
    rec_items_dict = dict()
    rec_items_dict[item] = user_emb.dot(map_matrix).dot(item_emb)
    for candi in candi_list:
        rec_items_dict[candi] = user_emb.dot(map_matrix).dot(item_emb.get(candi))
    # sort recommendation list
    sorted_rec_items_dict = sorted(rec_items_dict.items(), key=lambda d: d[1], reverse=True)
    user_k_map = dict()
    rr = 0
    for k in top_k:
        rank = hit = 0
        for t in sorted_rec_items_dict:
            rank += 1
            if item == t[0]:
                if rank <= k:
                    hit = 1
                rr = 1 / float(rank)
                break
        user_k_map[k] = hit
    return user_k_map, rr


if __name__ == "__main__":
    top_k = [1,5,10,15,20,40,60,80,100]
    friends_hit_at_k_map = dict()
    items_hit_at_k_map = dict()
    for k in top_k:
        friends_hit_at_k_map[k] = items_hit_at_k_map[k] = 0.0
    MRR = 0.0
    test_user = pk.load(open(test_user_file,'rb'))
    friend_avg_hit = friend_MRR = pos_friend_num = 0.0
    item_avg_hit = item_MRR = pos_item_num = 0.0
    for user, info in test_user.items():
        friends_candi_list = info.get("neg_users")
        items_candi_list = info.get("neg_items")
        for pos_friend in info.get("pos_users"):
            pos_friend_num += 1
            hit_at_k_map, rr = cal_user_friend_topk(user, pos_friend, friends_candi_list)
            for k in top_k:
                friends_hit_at_k_map[k] += hit_at_k_map[k]
            friend_MRR += rr
        for pos_item in info.get("pos_items"):
            pos_item_num += 1
            hit_at_k_map, rr = cal_user_item_topk(user, pos_item, items_candi_list)
            for k in top_k:
                items_hit_at_k_map[k] += hit_at_k_map[k]
            item_MRR += rr

    for k in top_k:
        print("friend hit@%d is %f" % (k,friends_hit_at_k_map[k]/(pos_friend_num)))
    print("friend mrr is:%f" % (friend_MRR / pos_friend_num))

    for k in top_k:
        print("item hit@d is %f" % (items_hit_at_k_map[k]/(pos_item_num)))
    print("item mrr is:%f" % (item_MRR / pos_item_num))
