import logging
import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from funcs import *
from torch.nn import MultiheadAttention
import torch.nn.functional as F
PRECISION = 5
POS_DIM_ALTER = 100
from module import *


class CAWN(torch.nn.Module):
    def __init__(self, device, ngh_finder, train_node_set, max_idx,
                 pos_dim=0, feat_dim=0, walk_pool='attn', walk_n_head=8,
                 num_layers=3, drop_out=0.1, num_neighbors=20,
                 get_checkpoint_path=None):
        super(CAWN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.max_train_node_set = max(train_node_set)
        assert(self.max_train_node_set == len(train_node_set))
        self.train_node_set = train_node_set
        self.node_embedding = nn.Embedding(num_embeddings=max_idx+1, embedding_dim=pos_dim)
        torch.nn.init.xavier_normal_(self.node_embedding.weight)

        # subgraph extraction hyper-parameters
        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors, num_layers)
        self.ngh_finder = ngh_finder

        self.get_checkpoint_path = get_checkpoint_path

    def contrast(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l=None, sample_flag=True):
        '''
        1. grab subgraph for src, tgt, bgd
        2. add positional encoding for src & tgt nodes
        3. forward propagate to get src embeddings and tgt embeddings (and finally pos_score (shape: [batch, ]))
        4. forward propagate to get src embeddings and bgd embeddings (and finally neg_score (shape: [batch, ]))
        '''
        subgraph_src = self.grab_subgraph(src_idx_l, cut_time_l, e_idx_l=e_idx_l)
        subgraph_tgt = self.grab_subgraph(tgt_idx_l, cut_time_l, e_idx_l=e_idx_l)
        subgraph_bgd = self.grab_subgraph(bgd_idx_l, cut_time_l, e_idx_l=None)
        pos_score = self.forward(src_idx_l, tgt_idx_l, cut_time_l, (subgraph_src, subgraph_tgt))
        neg_score = self.forward(src_idx_l, bgd_idx_l, cut_time_l, (subgraph_src, subgraph_bgd))
        return pos_score.sigmoid(), neg_score.sigmoid()

    def forward(self, src_idx_l, tgt_idx_l, cut_time_l, subgraphs=None):
        subgraph_src, subgraph_tgt = subgraphs
        subgraph_src_walk = self.subgraph_tree2walk(src_idx_l, cut_time_l, subgraph_src)
        subgraph_tgt_walk = self.subgraph_tree2walk(tgt_idx_l, cut_time_l, subgraph_tgt)
        emb_src_embed = self.full_gat(src_idx_l, subgraph_src_walk[0]) # nodes: bs * 16 * 3
        emb_tgt_embed = self.full_gat(tgt_idx_l, subgraph_tgt_walk[0])

        score = torch.norm(emb_src_embed-emb_tgt_embed, p=2, dim=-1)
        return score
    
    def gat_layer_mean(self, nghs):
        masks = (nghs <= self.max_train_node_set) & (nghs != 0) # bs * walk_num
        emb_nghs = self.node_embedding(nghs) # bs * walk_num * emb_dim
        output = torch.sum(masks.unsqueeze(-1)*emb_nghs, -2) / (torch.sum(masks, -1, keepdim=True) + 1e-12)
        return output

    def full_gat(self, src_idx_l, subgraph_src_node):
        subgraph_src_node = torch.from_numpy(subgraph_src_node).long().to(self.device) # bs * walk_num * walk_len
        src_idx_l = torch.from_numpy(src_idx_l).long().to(self.device) # 64
        src_emb = self.node_embedding(src_idx_l) # bs * emb_dim
        ngh_hop_1, ngh_hop_2 = subgraph_src_node[:, :, 1], subgraph_src_node[:, :, 2]
        emb_hop_1 = self.gat_layer_mean(ngh_hop_1)
        emb_hop_2 = self.gat_layer_mean(ngh_hop_2)
        output = (src_emb + emb_hop_1 + emb_hop_2) / 3
        return output
    
    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None):
        subgraph = self.ngh_finder.find_k_hop(self.num_layers, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, e_idx_l=e_idx_l)
        return subgraph

    def subgraph_tree2walk(self, src_idx_l, cut_time_l, subgraph_src):
        # put src nodes and extracted subgraph together
        node_records, t_records = subgraph_src
        node_records_tmp = [np.expand_dims(src_idx_l, 1)] + node_records
        t_records_tmp = [np.expand_dims(cut_time_l, 1)] + t_records

        # use the list to construct a new matrix
        new_node_records = self.subgraph_tree2walk_one_component(node_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)
        return new_node_records, new_t_records

    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert(n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix
