# funcs in utils.

import numpy as np
import torch
import os
import random


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


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


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers



# funcs in log.py

import logging
import time
import sys
import os


def set_up_logger(args, sys_argv):
    # set up running log
    n_degree, n_layer = process_sampling_numbers(args.n_degree, args.n_layer)
    n_degree = [str(n) for n in n_degree]
    runtime_id = '{}-{}-{}-{}-{}'.format(str(time.time()), args.data, n_layer, 'k'.join(n_degree), args.pos_dim)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_root = './DEAL/log/'
    if not os.path.exists(log_root):
        os.mkdir(log_root)
        logger.info('Create directory {}'.format(log_root))
    file_path = './DEAL/log/{}.log'.format(runtime_id) #TODO: improve log name
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)

    # set up model parameters log
    checkpoint_root = './DEAL/saved_checkpoints/'
    checkpoint_dir = checkpoint_root + runtime_id + '/'
    best_model_root = './DEAL/best_models/'
    best_model_dir = best_model_root + runtime_id + '/'
    if not os.path.exists(checkpoint_root):
        os.mkdir(checkpoint_root)
        logger.info('Create directory {}'.format(checkpoint_root))
    if not os.path.exists(best_model_root):
        os.mkdir(best_model_root)
        logger.info('Create directory'.format(best_model_root))
    os.mkdir(checkpoint_dir)
    os.mkdir(best_model_dir)
    logger.info('Create checkpoint directory {}'.format(checkpoint_dir))
    logger.info('Create best model directory {}'.format(best_model_dir))

    get_checkpoint_path = lambda epoch: (checkpoint_dir + 'checkpoint-epoch-{}.pth'.format(epoch))
    best_model_path = best_model_dir + 'best-model.pth'

    return logger, get_checkpoint_path, best_model_path



# funcs in position.py

import numpy as np
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def nodets2key(batch: int, node: int, ts: float):
    key = '-'.join([str(batch), str(node), float2str(ts)])
    return key


def float2str(ts):
    return str(int(round(ts)))


def make_batched_keys(node_record, t_record):
    batch = node_record.shape[0]
    support = node_record.shape[1]
    batched_keys = make_batched_keys_l(node_record, t_record, batch, support)
    batched_keys = np.array(batched_keys).reshape((batch, support))
    # batched_keys = np.array([nodets2key(b, n, t) for b, n, t in zip(batch_matrix.ravel(), node_record.ravel(), t_record.ravel())]).reshape(batch, support)
    return batched_keys


def make_batched_keys_l(node_record, t_record, batch, support):
    batch_matrix = np.arange(batch).repeat(support).reshape((-1, support))
    # batch_matrix = np.tile(np.expand_dims(np.arange(batch), 1), (1, support))
    batched_keys = []
    for i in range(batch):
        for j in range(support):
            b = batch_matrix[i, j]
            n = node_record[i, j]
            t = t_record[i, j]
            batched_keys.append(nodets2key(b, n, t))
    return batched_keys



# funcs in sample.py

import random
import numpy as np
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def seq_binary_sample(ngh_binomial_prob, num_neighbor):
    sampled_idx = []
    for j in range(num_neighbor):
        idx = seq_binary_sample_one(ngh_binomial_prob)
        sampled_idx.append(idx)
    sampled_idx = np.array(sampled_idx)  # not necessary but just for type alignment with the other branch
    return sampled_idx


def seq_binary_sample_one(ngh_binomial_prob):
    seg_len = 10
    a_l_seg = np.random.random((seg_len,))
    seg_idx = 0
    for idx in range(len(ngh_binomial_prob)-1, -1, -1):
        a = a_l_seg[seg_idx]
        seg_idx += 1 # move one step forward
        if seg_idx >= seg_len:
            a_l_seg = np.random.random((seg_len,))  # regenerate a batch of new random values
            seg_idx = 0  # and reset the seg_idx
        if a < ngh_binomial_prob[idx]:
            # print('=' * 50)
            # print(a, len(ngh_binomial_prob) - idx, len(ngh_binomial_prob),
            #       (len(ngh_binomial_prob) - idx) / len(ngh_binomial_prob), ngh_binomial_prob)
            return idx
    return 0  # very extreme case due to float rounding error


def bisect_left_adapt(a, x):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    lo = 0
    hi = len(a)
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo
