import pandas as pd
from funcs import *
from train import *
#import numba
from caw import CAWN
from graph import NeighborFinder

import argparse
import sys
import pickle


def get_args():
    parser = argparse.ArgumentParser('Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use', default='math')
    parser.add_argument('--data_usage', default=1, type=float, help='fraction of data to use (0-1)')

    # method-related hyper-parameters
    parser.add_argument('--n_degree', nargs='*', default=['16', '1'], help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--bias', default=0.0, type=float, help='the hyperparameter alpha controlling sampling preference in recent time, default to 0 which is uniform sampling')
    parser.add_argument('--feat_dim', type=int, default=100, help='dimension of the positional embedding')
    parser.add_argument('--pos_dim', type=int, default=100, help='dimension of the positional embedding')
    parser.add_argument('--walk_pool', type=str, default='attn', choices=['attn', 'sum'], help='how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other walk_ arguments')
    parser.add_argument('--walk_n_head', type=int, default=4, help="number of heads to use for walk attention")

    parser.add_argument('-dm', '--dmode', type=str, default='i', choices=['t', 'i', 'ti'], help='transductive (t) or inductive (i) or both')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=0, help='tolerated marginal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

args, sys_argv = get_args()

if args.data_usage < 1:
    args.n_epoch = 1

if args.data == 'stack':
    args.n_epoch = 7

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_DIM = args.pos_dim
FEAT_DIM = args.feat_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
TOLERANCE = args.tolerance
SEED = args.seed
set_random_seed(SEED)
logger, get_checkpoint_path, pretrain_model_path = set_up_logger(args, sys_argv)

# Load data and sanity check
g_df = pd.read_csv('./datasets/ml_{}.csv'.format(DATA))
if args.dmode == 't':
    valid_negs, test_negs = pickle.load(open('./datasets/'+'trans_{}_negs.pkl'.format(args.data), 'rb'))
elif args.dmode == 'i':
    valid_negs, test_negs = pickle.load(open('./datasets/'+'ind_{}_negs.pkl'.format(args.data), 'rb'))
elif args.dmode == 'ti':
    valid_negs, test_negs = pickle.load(open('./datasets/'+'all_{}_negs.pkl'.format(args.data), 'rb'))
if args.data_usage < 1:
    g_df = g_df.iloc[:int(args.data_usage*g_df.shape[0])]
    logger.info('use partial data, ratio: {}'.format(args.data_usage))
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

adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(valid_src_l, valid_dst_l, valid_e_idx_l, valid_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(test_src_l, test_dst_l, test_e_idx_l, test_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
ngh_finder = NeighborFinder(adj_list, bias=args.bias)

train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
valid_rand_sampler = RandEdgeSampler((train_src_l, valid_src_l, ), (train_dst_l, valid_dst_l, ))
test_rand_sampler = RandEdgeSampler((train_src_l, valid_src_l, test_src_l, ), (train_dst_l, valid_dst_l, test_dst_l, ))

train_valid_data = train_data, valid_data
train_valid_rand_samplers = train_rand_sampler, valid_rand_sampler
### data processing done ###

# model initialization
# device = torch.device('cuda:{}'.format(GPU))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cawn = CAWN(device,
            ngh_finder, train_node_set, max_idx, num_layers=NUM_LAYER,
            drop_out=DROP_OUT, pos_dim=POS_DIM, feat_dim=FEAT_DIM,
            num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_pool=args.walk_pool,
            get_checkpoint_path=get_checkpoint_path)
cawn.to(device)
#cawn.load_state_dict(torch.load(''))
optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(max_round=10, tolerance=TOLERANCE)

# start train and val phases
train_val(train_valid_data, cawn, args.data, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, train_valid_rand_samplers, valid_negs, logger)

# final testing
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test', args.data, cawn, test_rand_sampler, test_data, -1, test_negs, logger)
logger.info('Test statistics: all nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))

# save model
logger.info('Saving CAWN model ...')
torch.save(cawn.state_dict(), pretrain_model_path)
logger.info('CAWN model saved')
