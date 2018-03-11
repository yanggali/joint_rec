from file_process import read_file, get_group_users, emb_to_file, \
    userweight_to_file,get_emb,featureweight_to_file,write_to_file,\
    append_to_file,initial_negtable_data,initial_realedges_data
import numpy as np
import random
import math
import pickle as pk


DIM = 50
NEG_N = 2  # number of negative samples
init_lr = 0.025
lr = init_lr
reg1 = reg2 = 0.1
gama = 1
lower_bound = 0.1
reg = 0.1
ITER_NUM = 10
sample_T = 10000000

# all vertices set
all_user_v = set()
all_item_v = set()


# vertex sample table
pop_item_sample = list()
pop_user_sample = list()
# neg vertex sample table
view1_friend_neg_table = list()
view2_friend_neg_table = list()
friend_neg_table = list()
item_neg_table = list()

neg_table_size = int(1e8)
neg_sampling_power = 0.75

# vertex and its neighbours
user_user_nei = dict()
user_item_nei = dict()
item_user_nei = dict()

# all vertices' embedding
user_social_emb = dict()
user_prefer_emb = dict()
user_context_emb = dict()
user_emb = dict()
user_weight_dict = dict()
item_emb = dict()
user_item_map_matrix = np.random.rand(DIM,DIM)

SIGMOID_BOUND = 6
sigmoid_table_size = 1000
sig_map = dict()
# input
train_user_file = "./data/dataset/ciao/train/train_user.pkl"
train_item_degree = "./data/dataset/ciao/train/train_item_degree.pkl"
train_user_s_relation = "./data/dataset/ciao/train/train_user_social_view.pkl"
train_user_p_relation = "./data/dataset/ciao/train/train_user_pre_view.pkl"
# train_user_s_relation = "./data/dataset/ciao/train/sample_train_user_social_view.pkl"
# train_user_p_relation = "./data/dataset/ciao/train/sample_train_user_pre_view.pkl"
train_user_friend_edges = "./data/dataset/ciao/train/train_edges_1.pkl"
train_user_item_edges = "./data/dataset/ciao/train/train_edges_2.pkl"
test_user_file = "./data/dataset/ciao/test/test_user.pkl"
# output file
out_user = "./data/dataset/ciao/output/user"
out_item = "./data/dataset/ciao/output/item"
# initial two views and two realtions
train_user = pk.load(open(train_user_file,'rb'))
user_user_view1 = pk.load(open(train_user_s_relation,'rb'))
user_user_view2 = pk.load(open(train_user_p_relation,'rb'))
view_iter_num = max(len(user_user_view1),len(user_user_view2))
user_friend_edges = pk.load(open(train_user_friend_edges,'rb'))
user_item_edges = pk.load(open(train_user_item_edges,'rb'))
link_iter_num = max(len(user_friend_edges),len(user_item_edges))
print("input dataset finished")

# 获取二部图用户的邻居，朋友的度数，朋友列表，（所有边）
view1_user_nei,view1_friend_degree,view1_friend_list = initial_negtable_data(user_user_view1)
view2_user_nei,view2_friend_degree,view2_friend_list = initial_negtable_data(user_user_view2)
user_nei,friend_degree,friend_list,all_user_set = initial_realedges_data(user_friend_edges)
item_nei,item_degree,item_list,all_item_set = initial_realedges_data(user_item_edges)
view1_user_list = list(view1_user_nei.keys())
view2_user_list = list(view2_user_nei.keys())


# initial negative sampling table
def init_vertex_neg_table(vertex_neg_table, vertex_degree, vertex_list):
    sum = cur_num = por = 0.0
    vid = 0
    for vertex, degree in vertex_degree.items():
        sum += math.pow(degree, neg_sampling_power)
    for k in range(neg_table_size):
        if float(k + 1) / neg_table_size > por:
            cur_num += math.pow(vertex_degree.get(vertex_list[vid]), neg_sampling_power)
            por = cur_num / sum
            vid += 1
        vertex_neg_table.append(vertex_list[vid - 1])


# initial neg sampling table
def init_neg_table():
    # initial friend context in view1 neg table
    init_vertex_neg_table(view1_friend_neg_table, view1_friend_degree, view1_friend_list)
    # initial friend context in view2 neg table
    init_vertex_neg_table(view2_friend_neg_table,view2_friend_degree,view2_friend_list)
    # initial friend neg table
    init_vertex_neg_table(friend_neg_table, friend_degree, friend_list)
    # initial item neg table
    init_vertex_neg_table(item_neg_table, item_degree, item_list)


def cal_pop_pro(vertex_list, vertex_degree, sample_list):
    totalweight = 0.0
    for vertex, degree in vertex_degree.items():
        totalweight += math.pow(degree, neg_sampling_power)
    sample_list.append(float(math.pow(vertex_degree[vertex_list[0]], neg_sampling_power) / totalweight))
    for vertex in vertex_list[1:]:
        sample_list.append(sample_list[-1] + math.pow(float(vertex_degree[vertex]), neg_sampling_power) / totalweight)
    sample_list[-1] = 1.0


def gen_gaussian():
    max_value = 32767
    vector = np.zeros(DIM, )
    for i in range(DIM):
        vector[i] = (random.randint(0, max_value) * 1.0 / max_value - 0.5) / DIM
    return vector


# initialize all nodes' embedding
def init_vec(vec_list, vec_emb_dict):
    for vec in vec_list:
        if vec not in vec_emb_dict:
            vec_emb_dict[vec] = gen_gaussian()


def init_user(user_set,vec_emb_dict):
    for user in user_set:
        if user in user_social_emb and user in user_prefer_emb:
            w1 = user_weight_dict[user]["view1"]
            w2 = user_weight_dict[user]["view2"]
            vec_emb_dict[user] = w1*user_social_emb.get(user)/(w1+w2) + w2*user_prefer_emb.get(user)/(w1+w2)
        else:
            vec_emb_dict[user] = gen_gaussian()


# initialize two user views' weight
def init_user_view_weight(vec_list,view_weight_dict):
    for node in vec_list:
        view_weight_dict[node] = dict()
        view_weight_dict[node]["view1"] = view_weight_dict[node]["view2"] = 0.5


def math_exp(x):
    try:
        ans = math.exp(x)
    except OverflowError:
        ans = float('inf')
        # ans = 1.79769313e+308
    return ans


def init_user_weight(users_at_group, user_weight):
    user_len = len(users_at_group)
    max_value = 32767
    for user in users_at_group:
        user_weight[user] = (random.randint(0, max_value) * 1.0 / max_value)/user_len


# initial vertices' embedding
def init_all_vec():
    # 初始化user的context embedding
    init_vec(view1_friend_list, user_context_emb)
    init_vec(view2_friend_list, user_context_emb)
    # 初始化view1和view2中user的embedding
    init_vec(view1_user_list,user_social_emb)
    init_vec(view2_user_list,user_prefer_emb)
    # 初始化user的embeddding，根据两种view的user加权和
    init_user_view_weight(all_user_set,user_weight_dict)
    init_user(all_user_set, user_emb)
    # 初始化item的embedding
    init_vec(all_item_set,item_emb)
    emb_to_file(out_user+"_r"+str(reg)+"N"+str(NEG_N)+"_init", user_emb)
    emb_to_file(out_item+"_r"+str(reg)+"N"+str(NEG_N)+"_init", item_emb)


def init_sigmod_table():
    for k in range(sigmoid_table_size):
        x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND
        sig_map[k] = 1 / (1 + math_exp(-x))


# sample an edge randomly
def draw_tuple(tuple_list):
    return tuple_list[random.randint(0, len(tuple_list) - 1)]


# sample vertex
def draw_vertex(vertex_list, vertex_sample_pro):
    r = random.random()
    for i in range(len(vertex_sample_pro)):
        if vertex_sample_pro[i] > r:
            break
    return vertex_list[i]


def sigmoid(x):
    if x > SIGMOID_BOUND: return 1
    if x < -SIGMOID_BOUND: return 0
    k = int((x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2)
    return sig_map.get(k)


# update user_context target vertex
def update_user_friend_vertex(source, target, weight,error, label,type):
    if type == 1:
        source_emb = user_social_emb.get(source)
    else:
        source_emb = user_prefer_emb.get(source)
    target_emb = user_context_emb.get(target)
    score = sigmoid(math.pow(-1, label) * source_emb.dot(target_emb))
    g = weight * math.pow(-1, label) * score * lr
    # total error for source vertex
    error += g * target_emb
    new_vec = target_emb + g * source_emb
    item_emb[target] = new_vec


# update vertices according to an edge
def update_user_friend(source, target, weight,neg_vertices,type):
    error = np.zeros(DIM, )
    update_user_friend_vertex(source, target, weight,error, 1,type)
    M = len(neg_vertices)
    if M != 0:
        for i in range(M):
            update_user_friend_vertex(source, neg_vertices[i], weight,error, 0,type)

    w1 = user_weight_dict[source]["view1"]
    w2 = user_weight_dict[source]["view2"]
    lamda1 = w1/(w1+w2)
    lamda2 = w2/(w1+w2)
    if type == 1:
        if (source, target) in user_friend_edges:
            source_error = lr * user_emb.get(target) * lamda1
        else:
            source_error = 0
        # 正则项
        default_emb = np.zeros(DIM,)
        reg_part_error = lr*2*reg*(lamda1*(1-lamda1)*(user_social_emb.get(source)-user_emb.get(source))-
                                           lamda2*lamda1*(user_prefer_emb.get(source,default_emb) - user_emb.get(source)))
        user_social_emb[source] = user_social_emb.get(source) + error + source_error + reg_part_error
    else:
        if (source, target) in user_friend_edges:
            source_error = lr * user_emb.get(target) * lamda2
        else:
            source_error = 0
        # 正则项
        default_emb = np.zeros(DIM, )
        reg_part_error = lr * 2 * reg * (lamda2 * (1 - lamda2)* (user_prefer_emb.get(source) - user_emb.get(source)) -
                                                        lamda1* lamda2* (user_social_emb.get(source,default_emb) - user_emb.get(source)))
        user_prefer_emb[source] = user_prefer_emb.get(source) +error + source_error + reg_part_error


def update_item_vertex_in_group_stage1(source_emb, target_emb, error, label):
    if math.isnan(source_emb.dot(target_emb)):
        print("not a number")
    score = sigmoid(math.pow(-1, label) * source_emb.dot(target_emb))
    g = math.pow(-1, label) * score * lr
    # total error for puser , luser and ruser vertex
    error += g * target_emb


# fix source, sample targets
def neg_sample_user_friend(source, target, weight,source_nei,target_list,target_neg_table,type):
    # sample M negative vertices
    neg_vertices = list()
    record = 0
    while len(neg_vertices) < NEG_N:
        if record < len(target_list):
            sample_v = target_neg_table[random.randint(0,neg_table_size-1)]
            if sample_v not in source_nei.get(source) and sample_v not in neg_vertices and sample_v != source:
                neg_vertices.append(sample_v)
        else:
            break
        record += 1
    update_user_friend(source, target, weight,neg_vertices,type)


def training_user_in_view(type):
    # train view1
    if type == 1:
        t = draw_tuple(user_user_view1)
        v1 = t[0]
        v2 = t[1]
        w = t[2]
        # fix user, sample items
        neg_sample_user_friend(v1, v2,w, view1_user_nei,view1_friend_list,view1_friend_neg_table,type)
    # train view2
    else:
        t = draw_tuple(user_user_view2)
        v1 = t[0]
        v2 = t[1]
        w = t[2]
        # fix user, sample items
        neg_sample_user_friend(v1, v2, w, view2_user_nei, view2_friend_list, view2_friend_neg_table,type)


# 训练user-friend关系,更新user的两个view的weight
def train_user_user_relation():
    t = draw_tuple(user_friend_edges)
    v1 = t[0]
    v2 = t[1]
    if (v1 in user_social_emb and v1 in user_prefer_emb) and (v2 in user_social_emb and v2 in user_prefer_emb):
        v1_w1 = user_weight_dict.get(v1).get("view1")
        v1_w2 = user_weight_dict.get(v1).get("view2")
        v2_w1 = user_weight_dict.get(v2).get("view1")
        v2_w2 = user_weight_dict.get(v2).get("view2")
        # 更新ui的weight
        user_weight_dict[v1]["view1"] += lr*user_emb.get(v2)/((v1_w1+v1_w2)*(v1_w1+v1_w2))*\
                                         (v1_w2*user_social_emb.get(v1)- user_prefer_emb.get(v1))
        user_weight_dict[v1]["view2"] += lr * user_emb.get(v2)/((v1_w1 + v1_w2) * (v1_w1 + v1_w2))*\
                                         (v1_w1 * user_prefer_emb.get(v1) - user_social_emb.get(v1))
        # 更新uj的weight
        user_weight_dict[v2]["view1"] += lr * user_emb.get(v1)/((v2_w1 + v2_w2) * (v2_w1 + v2_w2)) * (
                                        v2_w2 * user_social_emb.get(v2) - user_prefer_emb.get(v2))
        user_weight_dict[v2]["view2"] += lr * user_emb.get(v1)/((v2_w1 + v2_w2) * (v2_w1 + v2_w2)) * (
                                        v2_w1 * user_prefer_emb.get(v2) - user_social_emb.get(v2))
        user_emb[v1] = (user_weight_dict[v1]["view1"]*user_social_emb[v1] + user_weight_dict[v1]["view2"]*
                        user_prefer_emb[v1])/(user_weight_dict[v1]["view1"]+user_weight_dict[v1]["view2"])
        user_emb[v2] = (user_weight_dict[v2]["view1"] * user_social_emb[v2] + user_weight_dict[v2]["view2"] *
                        user_prefer_emb[v2]) / (user_weight_dict[v2]["view1"] + user_weight_dict[v2]["view2"])
    else:
        new_user_v1 = user_emb.get(v1) + lr*user_emb[v2]
        new_user_v2 = user_emb.get(v2) + lr*user_emb[v1]
        user_emb[v1] = new_user_v1
        user_emb[v2] = new_user_v2


# 训练user-item关系
def train_user_item_relation():
    global user_item_map_matrix
    t = draw_tuple(user_item_edges)
    user = t[0]
    item = t[1]
    w = train_user.get(user).get(item) / 5
    if user in user_social_emb and user in user_prefer_emb:
        v1_w1 = user_weight_dict.get(user).get("view1")
        v1_w2 = user_weight_dict.get(user).get("view2")
        # 更新ui的weight
        user_weight_dict[user]["view1"] += lr * item_emb.get(item).dot(user_item_map_matrix) / ((v1_w1 + v1_w2) * (v1_w1 + v1_w2)) * \
                                         (v1_w2 * user_social_emb.get(user) - user_prefer_emb.get(user))
        user_weight_dict[user]["view2"] += lr * item_emb.get(item).dot(user_item_map_matrix) / ((v1_w1 + v1_w2) * (v1_w1 + v1_w2)) * \
                                         (v1_w1 * user_prefer_emb.get(user) - user_social_emb.get(user))
        # 更新vj的embedding
        new_item_emb = item_emb.get(item) + lr*w*user_emb.get(user).dot(user_item_map_matrix)
        # 更新map_matrix
        user_item_map_matrix += lr*w*user_emb[user].reshape(DIM,1).dot(item_emb[item].reshape(1,DIM))
        # 更新vj的embedding
        item_emb[item] = new_item_emb
        # 更新ui的embedding
        user_emb[user] = (user_weight_dict[user]["view1"]*user_social_emb[user] + user_weight_dict[user]["view2"]*
                        user_prefer_emb[user])/(user_weight_dict[user]["view1"]+user_weight_dict[user]["view2"])
    else:
        new_user_emb = user_emb.get(user) + lr*w*user_item_map_matrix.dot(item_emb.get(item))
        new_item_emb = item_emb.get(item) + lr*w*user_item_map_matrix.dot(user_emb.get(user))
        user_item_map_matrix += lr*w*user_emb[user].reshape(DIM,1).dot(item_emb[item].reshape(1,DIM))
        user_emb[user] = new_user_emb
        item_emb[item] = new_item_emb


def bernoilli():
    r = random.random()
    if r < 0.5:
        return 1
    else:
        return 0


def train_data():
    iter = 0
    last_count = 0
    current_sample_count = 0
    while iter <= ITER_NUM:
        if iter - last_count > 10000:
            current_sample_count += iter - last_count
            last_count = iter
            lr = init_lr * (1 - current_sample_count / (1.0 * (view_iter_num + 1)))
            print("Iteration i:   " + str(iter) + "   ##########lr  " + str(lr))
            if lr < init_lr * 0.0001:
                lr = init_lr * 0.0001

        # 更新两个view中的节点
        # 随机选择一个view更新user emb和user context emb
        view_iter = 0
        while view_iter < sample_T:
            # social user user relation
            training_user_in_view(1)
            # preference user user relation
            training_user_in_view(2)
            if view_iter%10000 == 0:
                print("view iter %d finished." % view_iter)
            view_iter += 1
        # 更新真实的user-user和user-item节点
        # 随机选择user-friend图或者user-item图
        link_iter = 0
        while link_iter < link_iter_num:
            train_user_user_relation()
            train_user_item_relation()
            if link_iter%1000 == 0:
                print("real edge iter %d finished." % link_iter)
            link_iter += 1
        # write embedding into file
        emb_to_file(out_user + "_r" + str(reg) + "N" + str(NEG_N) + "_" + str(iter), user_emb)
        emb_to_file(out_item + "_r" + str(reg) + "N" + str(NEG_N) + "_" + str(iter), item_emb)
        print("Round i:  %d finished." % iter)
        iter += 1


if __name__ == "__main__":
    init_neg_table()
    print("initial neg table")
    init_all_vec()
    init_sigmod_table()
    print("training starting")
    train_data()
    print("training finished")
    emb_to_file(out_user+"_finished", user_emb)
    emb_to_file(out_item+"_finished", item_emb)