import pickle
import numpy as np
import pandas as pd
import argparse
import sys

np.random.seed(2022)

def get_args():
    parser = argparse.ArgumentParser('Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')
    # select dataset
    parser.add_argument('-d', '--data', type=str, help='data sources to use', default='math')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

args, sys_argv = get_args()

if args.data == 'math': # max node index =  3135 max edge index =  16455
    data = open('../data/sx-mathoverflow.txt')
    data_name = 'math'
    day_num = 90
elif args.data == 'ask': # max node index =  8159 max edge index =  20389
    data = open('../data/sx-askubuntu.txt')
    data_name = 'ask'
    day_num = 30
elif args.data == 'stack': # max node index =  51242 max edge index =  124828
    data = open('../data/sx-stackoverflow.txt')
    data_name = 'stack'
    day_num = 3
else:
    print ('error')
    exit()

src_list, dst_list, ts_list = [], [], []

for line in data.readlines():
    src, dst, ts = line.split(' ')
    src_list.append(src)
    dst_list.append(dst)
    ts_list.append(int(ts))

reidx = np.argsort(ts_list)
src_list = np.array(src_list)[reidx]
dst_list = np.array(dst_list)[reidx]
ts_list = np.array(ts_list)[reidx]

max_ts = max(ts_list)
cut_ts = max_ts - 86400 * day_num
cut_src_list, cut_dst_list, cut_ts_list, cut_idx_list = [], [], [], []
node_dict = {}
reidx = 0
idx = 0
for src, dst, ts in zip(src_list, dst_list, ts_list):
    if ts > cut_ts:
        idx += 1
        if int(src) not in node_dict:
            reidx += 1
            node_dict[int(src)] = reidx
        if int(dst) not in node_dict:
            reidx += 1
            node_dict[int(dst)] = reidx
        cut_src_list.append(node_dict[int(src)])
        cut_dst_list.append(node_dict[int(dst)])
        cut_ts_list.append(ts)
        cut_idx_list.append(idx)

min_ts = min(cut_ts_list)
for i in range(len(cut_ts_list)):
    cut_ts_list[i] -= min_ts

cut_src_list = np.array(cut_src_list)
cut_dst_list = np.array(cut_dst_list)
cut_ts_list = np.array(cut_ts_list)

print ('max node index = ', np.max(cut_src_list), 'max edge index = ', cut_src_list.shape[0])

df = pd.DataFrame({'u': cut_src_list, 'i': cut_dst_list, 'ts': cut_ts_list, 'idx': cut_idx_list})

OUT_DF = './datasets/ml_{}.csv'.format(data_name)
df.to_csv(OUT_DF)



# sampling for validation and test
class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        src_list = np.concatenate(src_list)
        dst_list = np.concatenate(dst_list)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

g_df = pd.read_csv('./datasets/ml_{}.csv'.format(data_name))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
ts_l = g_df.ts.values

### data processing begin ###
max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())
valid_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))

train_flag = (ts_l <= valid_time)
valid_flag = (ts_l > valid_time) * (ts_l <= test_time)
test_flag = (ts_l > test_time)

train_src_l, train_dst_l, train_ts_l, train_e_idx_l = src_l[train_flag], dst_l[train_flag], ts_l[train_flag], e_idx_l[train_flag]
valid_src_l, valid_dst_l, valid_ts_l, valid_e_idx_l = src_l[valid_flag], dst_l[valid_flag], ts_l[valid_flag], e_idx_l[valid_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l = src_l[test_flag], dst_l[test_flag], ts_l[test_flag], e_idx_l[test_flag]

valid_rand_sampler = RandEdgeSampler((train_src_l, valid_src_l, ), (train_dst_l, valid_dst_l, ))
test_rand_sampler = RandEdgeSampler((train_src_l, valid_src_l, test_src_l, ), (train_dst_l, valid_dst_l, test_dst_l, ))
valid_size, test_size = len(valid_src_l), len(test_src_l)
_, valid_dst_l_fake = valid_rand_sampler.sample(valid_size)
_, test_dst_l_fake = test_rand_sampler.sample(test_size)
pickle.dump([valid_dst_l_fake, test_dst_l_fake], open('./datasets/all_{}_negs.pkl'.format(data_name), 'wb'))

train_node_set = set(train_src_l).union(train_dst_l)
valid_node_set = set(valid_src_l).union(valid_dst_l)
new_valid_node_set = valid_node_set - train_node_set
is_new_node_edge_valid = np.array([(a in new_valid_node_set or b in new_valid_node_set) for a, b in zip(src_l, dst_l)])
not_is_new_node_edge_valid = np.array([bool(1 - flag) for flag in is_new_node_edge_valid])
trans_valid_flag = valid_flag * not_is_new_node_edge_valid
ind_valid_flag = valid_flag * is_new_node_edge_valid

test_node_set = set(test_src_l).union(test_dst_l)
new_test_node_set = test_node_set - train_node_set
is_new_node_edge_test = np.array([(a in new_test_node_set or b in new_test_node_set) for a, b in zip(src_l, dst_l)])
not_is_new_node_edge_test = np.array([bool(1 - flag) for flag in is_new_node_edge_test])
trans_test_flag = test_flag * not_is_new_node_edge_test
ind_test_flag = test_flag * is_new_node_edge_test

trans_valid_src_l, trans_valid_dst_l, trans_valid_ts_l, trans_valid_e_idx_l = src_l[trans_valid_flag], dst_l[trans_valid_flag], ts_l[trans_valid_flag], e_idx_l[trans_valid_flag]
trans_test_src_l, trans_test_dst_l, trans_test_ts_l, trans_test_e_idx_l = src_l[trans_test_flag], dst_l[trans_test_flag], ts_l[trans_test_flag], e_idx_l[trans_test_flag]
trans_valid_rand_sampler = RandEdgeSampler((train_src_l, trans_valid_src_l, ), (train_dst_l, trans_valid_dst_l, ))
trans_test_rand_sampler = RandEdgeSampler((train_src_l, trans_valid_src_l, trans_test_src_l, ), (train_dst_l, trans_valid_dst_l, trans_test_dst_l, ))
#trans_valid_rand_sampler = RandEdgeSampler((trans_valid_src_l, ), (trans_valid_dst_l, ))
#trans_test_rand_sampler = RandEdgeSampler((trans_test_src_l, ), (trans_test_dst_l, ))
trans_valid_size, trans_test_size = len(trans_valid_src_l), len(trans_test_src_l)
_, trans_valid_dst_l_fake = trans_valid_rand_sampler.sample(trans_valid_size)
_, trans_test_dst_l_fake = trans_test_rand_sampler.sample(trans_test_size)
pickle.dump([trans_valid_dst_l_fake, trans_test_dst_l_fake], open('./datasets/trans_{}_negs.pkl'.format(data_name), 'wb'))

ind_valid_src_l, ind_valid_dst_l, ind_valid_ts_l, ind_valid_e_idx_l = src_l[ind_valid_flag], dst_l[ind_valid_flag], ts_l[ind_valid_flag], e_idx_l[ind_valid_flag]
ind_test_src_l, ind_test_dst_l, ind_test_ts_l, ind_test_e_idx_l = src_l[ind_test_flag], dst_l[ind_test_flag], ts_l[ind_test_flag], e_idx_l[ind_test_flag]
ind_valid_rand_sampler = RandEdgeSampler((train_src_l, ind_valid_src_l, ), (train_dst_l, ind_valid_dst_l, ))
ind_test_rand_sampler = RandEdgeSampler((train_src_l, ind_valid_src_l, ind_test_src_l, ), (train_dst_l, ind_valid_dst_l, ind_test_dst_l, ))
#ind_valid_rand_sampler = RandEdgeSampler((ind_valid_src_l, ), (ind_valid_dst_l, ))
#ind_test_rand_sampler = RandEdgeSampler((ind_test_src_l, ), (ind_test_dst_l, ))
ind_valid_size, ind_test_size = len(ind_valid_src_l), len(ind_test_src_l)
_, ind_valid_dst_l_fake = ind_valid_rand_sampler.sample(ind_valid_size)
_, ind_test_dst_l_fake = ind_test_rand_sampler.sample(ind_test_size)
pickle.dump([ind_valid_dst_l_fake, ind_test_dst_l_fake], open('./datasets/ind_{}_negs.pkl'.format(data_name), 'wb'))
