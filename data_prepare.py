# 准备user-user的两个view，user-user关系，user-item关系
# user-user的两个view：一个根据user-item评分信息，一个根据user-friend信息
import pickle as pk
import scipy.io as scio
import os

train_user_file = "./data/dataset/ciao/train/train_user.pkl"
user_social_view = "./data/dataset/ciao/train/train_user_social_view.pkl"
user_pre_view = "./data/dataset/ciao/train/train_user_pre_view.pkl"
user_social_edges = "./data/dataset/ciao/train/train_user_social_edges"
user_prefer_edges = "./data/dataset/ciao/train/train_user_prefer_edges"
# sample_user_social_view = "./data/dataset/ciao/train/sample_train_user_social_view.pkl"
# sample_user_pre_view = "./data/dataset/ciao/train/sample_train_user_pre_view.pkl"
train_edges_1_view = "./data/dataset/ciao/train/train_edges_1.pkl"
train_edges_2_view = "./data/dataset/ciao/train/train_edges_2.pkl"
# sample_user_social_edges = "./data/dataset/ciao/train/sample_train_user_social_edges"
# sample_user_prefer_edges = "./data/dataset/ciao/train/sample_train_user_prefer_edges"
train_edges_1 = "./data/dataset/ciao/train/train_edges_1"
train_edges_2 = "./data/dataset/ciao/train/train_edges_2"
rating_mat = "F:/dataset/ciao/rating.mat"
trust_mat = "F:/dataset/ciao/trustnetwork.mat"
rating_file = "F:/dataset/ciao/rating.dat"
trustnetwork_file = "F:/dataset/ciao/trustnetwork.dat"
train_user = pk.load(open(train_user_file, 'rb'))


def get_user_user_relation():
    user_s_relation = list()
    user_p_relation = list()
    max_user_social_weight = 0
    min_user_social_weight = 10000
    max_user_pre_weight = 0
    min_user_pre_weight = 10000
    count = 0
    for user1, info1 in train_user.items():
        for user2, info2 in train_user.items():
            # if user2 == 6875:
            #     print(6875)
            if user1 == user2: continue
            if info1.get("item", None) is not None and info2.get("item", None) is not None:
                w1 = 0
                for item in info1["item"].keys() & info2["item"].keys():
                    w1 += info1["item"][item] * info2["item"][item] / 25
                if w1 != 0:
                    user_p_relation.append((user1, user2, w1))
                min_user_pre_weight = min(min_user_pre_weight, w1)
                max_user_pre_weight = max(max_user_pre_weight, w1)
            if info1.get("friend", None) is not None and info2.get("friend", None) is not None:
                inter = info1["friend"].keys() & info2["friend"].keys()
                w2 = len(inter)
                if w2 != 0:
                    user_s_relation.append((user1, user2, w2))
                min_user_social_weight = min(min_user_social_weight, w2)
                max_user_social_weight = max(max_user_social_weight, w2)
            count += 1
            # if count > 1000:
            #     break
    reg_user_p_relation = list()
    diff1 = max_user_pre_weight - min_user_pre_weight
    for t in user_p_relation:
        reg_user_p_relation.append((t[0], t[1], (t[2] - min_user_pre_weight) / diff1))
        #reg_user_p_relation.append((t[0], t[1], t[2]))
    reg_user_s_relation = list()
    diff2 = max_user_social_weight - min_user_social_weight
    for t in user_s_relation:
        reg_user_s_relation.append((t[0], t[1], (t[2] - min_user_social_weight) / diff2))
        # reg_user_s_relation.append((t[0], t[1], t[2]))
    pk.dump(reg_user_p_relation, open(user_pre_view, 'wb'))
    pk.dump(reg_user_s_relation, open(user_social_view, 'wb'))


def append_to_file(filename, str):
    with open(filename, 'a') as fw:
        fw.write(str)


def write_to_file(view, view_file):
    if os.path.exists(view_file):
        os.remove(view_file)
    edges = pk.load(open(view, 'rb'))
    edges_str = ""
    count = 0
    for edge in edges:
        if count % 1000 == 0:
            append_to_file(view_file, edges_str)
            edges_str = ""
        edges_str += str(edge[0]) + " " + str(edge[1]) + " " + str(edge[2]) + "\n"
        count += 1
    append_to_file(view_file, edges_str)


def rating_mat_to_file(mat_file,edges_file):
    mat = scio.loadmat(mat_file)
    rating = mat["rating"]
    edges_str = ""
    count = 0
    for i in range(rating.shape[0]):
        if count%10000 == 0:
            append_to_file(edges_file,edges_str)
            edges_str = ""
        edges_str += str(rating[i,0])+" "+str(rating[i,1])+" "+str(rating[i,3])+"\n"
        count += 1
    append_to_file(edges_file,edges_str)


def trust_mat_to_file(mat_file,edges_file):
    mat = scio.loadmat(mat_file)
    trust = mat["trustnetwork"]
    edges_str = ""
    count = 0
    for i in range(trust.shape[0]):
        if count%10000 == 0:
            append_to_file(edges_file,edges_str)
            edges_str = ""
        edges_str += str(trust[i,0])+" "+str(trust[i,1])+"\n"
        count += 1
    append_to_file(edges_file,edges_str)


if __name__ == "__main__":
    #get_user_user_relation()
    write_to_file(user_social_view, user_social_edges)
    write_to_file(user_pre_view, user_prefer_edges)
    # edges1 = pk.load(open(train_edges_2_view,'rb'))
    # print(len(edges1))
    # rating_mat_to_file(rating_mat,rating_file)
    # trust_mat_to_file(trust_mat,trustnetwork_file)
