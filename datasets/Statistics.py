import pandas as pd

import argparse
import sys
import numpy as np


def get_args():
    parser = argparse.ArgumentParser('Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')
    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use', default='math')
    parser.add_argument('-dm', '--dmode', type=str, default='i', choices=['t', 'i', 'ti'], help='transductive (t) or inductive (i) or both')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

args, sys_argv = get_args()

# Load data and sanity check
g_df = pd.read_csv('./datasets/ml_{}.csv'.format(args.data))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
ts_l = g_df.ts.values

### data processing begin ###
max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())
print ('num of nodes = ', max_idx)
print ('num of edges = ', len(src_l))
valid_time, test_time = list(np.quantile(ts_l, [0.70, 0.85]))

train_flag = (ts_l <= valid_time)
valid_flag = (ts_l > valid_time) * (ts_l <= test_time)
test_flag = (ts_l > test_time)

train_src_l, train_dst_l, train_ts_l, train_e_idx_l = src_l[train_flag], dst_l[train_flag], ts_l[train_flag], e_idx_l[train_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l

valid_src_l, valid_dst_l = src_l[valid_flag], dst_l[valid_flag]
test_src_l, test_dst_l = src_l[test_flag], dst_l[test_flag]

train_node_set = set(train_src_l).union(train_dst_l)
print ('num of training nodes = ', len(list(train_node_set)))
valid_node_set = set(valid_src_l).union(valid_dst_l)
new_valid_node_set = valid_node_set - train_node_set
print ('num of new valid nodes = ', len(list(new_valid_node_set)))
is_new_node_edge_valid = np.array([(a in new_valid_node_set or b in new_valid_node_set) for a, b in zip(src_l, dst_l)])
not_is_new_node_edge_valid = np.array([bool(1 - flag) for flag in is_new_node_edge_valid])

test_node_set = set(test_src_l).union(test_dst_l)
new_test_node_set = test_node_set - train_node_set
print ('num of new test nodes = ', len(list(new_test_node_set)))
is_new_node_edge_test = np.array([(a in new_test_node_set or b in new_test_node_set) for a, b in zip(src_l, dst_l)])
not_is_new_node_edge_test = np.array([bool(1 - flag) for flag in is_new_node_edge_test])

if args.dmode == 't':
    valid_flag = valid_flag * not_is_new_node_edge_valid
    test_flag = test_flag * not_is_new_node_edge_test
elif args.dmode == 'i':
    valid_flag = valid_flag * is_new_node_edge_valid
    test_flag = test_flag * is_new_node_edge_test
if args.dmode == 'ti':
    valid_flag = valid_flag
    test_flag = test_flag

valid_src_l, valid_dst_l, valid_ts_l, valid_e_idx_l = src_l[valid_flag], dst_l[valid_flag], ts_l[valid_flag], e_idx_l[valid_flag]
valid_data = valid_src_l, valid_dst_l, valid_ts_l, valid_e_idx_l
test_src_l, test_dst_l, test_ts_l, test_e_idx_l = src_l[test_flag], dst_l[test_flag], ts_l[test_flag], e_idx_l[test_flag]
test_data = test_src_l, test_dst_l, test_ts_l, test_e_idx_l

print ('num of train edges = ', len(train_src_l))
print ('num of valid edges = ', len(valid_src_l))
print ('num of test edges = ', len(test_src_l))
