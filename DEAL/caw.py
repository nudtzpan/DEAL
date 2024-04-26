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
    def __init__(self, device, ngh_finder, train_node_set, max_idx, mmode, data,
                 pos_dim=0, feat_dim=0, walk_pool='attn', walk_n_head=8,
                 num_layers=3, drop_out=0.1, num_neighbors=20,
                 get_checkpoint_path=None):
        super(CAWN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.max_train_node_set = max(train_node_set)
        assert(self.max_train_node_set == len(train_node_set))
        self.train_node_set = train_node_set
        load_paras = torch.load('./PretrainEmbs/'+'pretrain_paras_{}.pth'.format(data))
        self.load_node_embedding = torch.nn.Embedding.from_pretrained(load_paras['node_embedding.weight'], freeze=True)
        self.node_embedding = nn.Embedding(num_embeddings=max_idx+1, embedding_dim=pos_dim)
        torch.nn.init.xavier_normal_(self.node_embedding.weight)
        self.mmode = mmode

        # subgraph extraction hyper-parameters
        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors, num_layers)
        self.ngh_finder = ngh_finder

        # dimensions of 4 elements: node, edge, time, position
        self.feat_dim = feat_dim  # node feature dimension
        self.time_dim = self.feat_dim  # default to be time feature dimension
        self.pos_dim = pos_dim  # position feature dimension
        self.model_dim = self.time_dim + self.pos_dim
        self.logger.info('neighbors: {}, pos dim: {}, time dim: {}'.format(self.num_neighbors, self.pos_dim, self.time_dim))

        # walk-based attention/summation model hyperparameters
        self.walk_pool = walk_pool
        self.walk_n_head = walk_n_head

        # dropout for both tree and walk based model
        self.dropout_p = drop_out

        # encoders
        self.logger.info('Using time encoding')
        self.time_encoder = TimeEncode(expand_dim=self.time_dim)
        self.position_encoder = PositionEncoder(enc_dim=self.pos_dim, num_layers=self.num_layers, ngh_finder=self.ngh_finder,
                                                logger=self.logger)

        # attention model
        self.random_walk_attn_model = RandomWalkAttention(feat_dim=self.model_dim, pos_dim=self.pos_dim,
                                                          model_dim=self.model_dim, out_dim=self.feat_dim,
                                                          walk_pool=self.walk_pool,
                                                          n_head=self.walk_n_head,
                                                          dropout_p=self.dropout_p, logger=self.logger)

        # final projection layer
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, non_linear=True) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
        self.score_linear = nn.Sequential(nn.Linear(2, self.pos_dim), nn.Linear(self.pos_dim, 1))

        self.q = torch.nn.Linear(self.pos_dim, self.pos_dim)
        self.k = torch.nn.Linear(self.pos_dim, self.pos_dim)
        self.v = torch.nn.Linear(self.pos_dim, self.pos_dim)
        self.w = torch.nn.Linear(2*self.pos_dim, 1)

        self.get_checkpoint_path = get_checkpoint_path

    def contrast(self, src_idx_l, tgt_idx_l, bgd_idx_l, cut_time_l, e_idx_l=None):
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

        emb_src_embed = self.full_gat(src_idx_l, subgraph_src_walk[0], 'mean') # nodes: bs * 16 * 3
        emb_tgt_embed = self.full_gat(tgt_idx_l, subgraph_tgt_walk[0], 'mean')

        sim = torch.norm(emb_src_embed-emb_tgt_embed, p=2, dim=-1)
        sim_numpy = sim.cpu().detach().numpy()
        sample_flag = np.zeros(sim_numpy.shape[0],)
        sample_flag[sim_numpy < 0.2] = 0
        sample_flag[sim_numpy > 0.2] = 1
        #np.random.shuffle(norm_flag)
        
        re_subgraph_src = self.re_grab_subgraph(src_idx_l, cut_time_l, sample_flag)
        re_subgraph_tgt = self.re_grab_subgraph(tgt_idx_l, cut_time_l, sample_flag)

        self.position_encoder.init_internal_data(src_idx_l, tgt_idx_l, cut_time_l, re_subgraph_src, re_subgraph_tgt)
        re_subgraph_src_walk = self.subgraph_tree2walk(src_idx_l, cut_time_l, re_subgraph_src)
        re_subgraph_tgt_walk = self.subgraph_tree2walk(tgt_idx_l, cut_time_l, re_subgraph_tgt)

        emb_src_embed = self.full_gat(src_idx_l, re_subgraph_src_walk[0], 'att') # nodes: bs * 16 * 3
        emb_tgt_embed = self.full_gat(tgt_idx_l, re_subgraph_tgt_walk[0], 'att')

        src_embed = self.forward_msg(cut_time_l, re_subgraph_src_walk)
        tgt_embed = self.forward_msg(cut_time_l, re_subgraph_tgt_walk)
        
        if self.mmode == 'c':
            score, score_walk = self.affinity_score(src_embed, tgt_embed) # score_walk shape: [B, M]
            score.squeeze_(dim=-1)
        elif self.mmode == 'e':
            score, score_walk = self.affinity_score(emb_src_embed, emb_tgt_embed) # score_walk shape: [B, M]
            score.squeeze_(dim=-1)
        elif self.mmode == 'b':
            caw_score, caw_score_walk = self.affinity_score(src_embed, tgt_embed) # score_walk shape: [B, M]
            emb_score = torch.sum(emb_src_embed*emb_tgt_embed, -1).unsqueeze(-1) # self.emb_affinity_score(emb_src_embed, emb_tgt_embed) # score_walk shape: [B, M]
            score = self.score_linear(torch.cat([caw_score, emb_score], -1)).squeeze(-1) # bs

        return score
    
    def gat_layer_mean(self, nghs):
        masks = (nghs <= self.max_train_node_set) & (nghs != 0) # bs * walk_num
        emb_nghs = self.load_node_embedding(nghs) # bs * walk_num * emb_dim
        output = torch.sum(masks.unsqueeze(-1)*emb_nghs, -2) / (torch.sum(masks, -1, keepdim=True) + 1e-12)
        return output

    def gat_layer_att(self, src_emb, nghs):
        masks = (nghs <= self.max_train_node_set) & (nghs != 0) # bs * walk_num
        emb_nghs = self.node_embedding(nghs) # bs * walk_num * emb_dim
        weights = self.w(torch.cat([self.q(src_emb), self.k(emb_nghs)], -1)).squeeze(-1) # bs * walk_num
        weights = torch.exp(weights) # bs * walk_num
        weights = (masks*weights) / (torch.sum(masks*weights, -1, keepdim=True)+1e-12) # bs * walk_num
        output = torch.sum(weights.unsqueeze(-1)*self.v(emb_nghs), -2) # bs * emb_dim
        return output

    def full_gat(self, src_idx_l, subgraph_src_node, mode):
        subgraph_src_node = torch.from_numpy(subgraph_src_node).long().to(self.device) # bs * walk_num * walk_len
        walk_num = subgraph_src_node.shape[1]
        src_idx_l = torch.from_numpy(src_idx_l).long().to(self.device) # 64
        if mode == 'att':
            src_emb = self.node_embedding(src_idx_l) # bs * emb_dim
        elif mode == 'mean':
            src_emb = self.load_node_embedding(src_idx_l) # bs * emb_dim
        src_emb_repeat = src_emb.unsqueeze(1).repeat(1, walk_num, 1) # bs * walk_num * emb_dim
        ngh_hop_1, ngh_hop_2 = subgraph_src_node[:, :, 1], subgraph_src_node[:, :, 2]
        if mode == 'att':
            emb_hop_1 = self.gat_layer_att(src_emb_repeat, ngh_hop_1)
            emb_hop_2 = self.gat_layer_att(src_emb_repeat, ngh_hop_2)
        elif mode == 'mean':
            emb_hop_1 = self.gat_layer_mean(ngh_hop_1)
            emb_hop_2 = self.gat_layer_mean(ngh_hop_2)
        output = (src_emb + emb_hop_1 + emb_hop_2) / 3
        return output

    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None):
        subgraph = self.ngh_finder.find_k_hop(self.num_layers, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, e_idx_l=e_idx_l)
        return subgraph

    def re_grab_subgraph(self, src_idx_l, cut_time_l, flag):
        subgraph = self.ngh_finder.re_find_k_hop(self.num_layers, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, flag=flag)
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

    def forward_msg(self, cut_time_l, subgraph_src):
        node_records, t_records = subgraph_src
        masks = self.get_masks(node_records)  # length self.num_layers+1
        time_features = self.retrieve_time_features(cut_time_l, t_records)  # length self.num_layers+1
        position_features = self.retrieve_position_features(node_records, t_records)  # length self.num_layers+1, core contribution
        final_node_embeddings = self.random_walk_attn_model.forward_one_node(time_features, position_features, masks)
        return final_node_embeddings

    def get_masks(self, node_records):
        node_records_th = torch.from_numpy(node_records).long().to(self.device)
        masks = (node_records_th != 0).sum(dim=-1).long()  # shape [batch, n_walk], here the masks means differently: it records the valid length of each walk
        return masks

    def retrieve_time_features(self, cut_time_l, t_records):
        batch = len(cut_time_l)
        t_records_th = torch.from_numpy(t_records).float().to(self.device)
        t_records_th = t_records_th.select(dim=-1, index=0).unsqueeze(dim=2) - t_records_th
        n_walk, len_walk = t_records_th.size(1), t_records_th.size(2)
        time_features = self.time_encoder(t_records_th.view(batch, -1)).view(batch, n_walk, len_walk,
                                                                             self.time_encoder.time_dim)
        return time_features

    def retrieve_position_features(self, node_records, t_records):
        batch, n_walk, len_walk = node_records.shape
        node_records_r, t_records_r = node_records.reshape(batch, -1), t_records.reshape(batch, -1)
        position_features = self.position_encoder(node_records_r, t_records_r)
        position_features = position_features.view(batch, n_walk, len_walk, self.pos_dim)
        return position_features


class PositionEncoder(nn.Module):
    '''
    Note that encoding initialization and lookup is done on cpu but encoding (post) projection is on device
    '''
    def __init__(self, num_layers, enc_dim=2, ngh_finder=None, logger=None):
        super(PositionEncoder, self).__init__()
        self.enc_dim = enc_dim
        self.num_layers = num_layers
        self.nodetime2emb_maps = None
        self.projection = nn.Linear(1, 1)  # reserved for when the internal position encoding does not match input
        self.ngh_finder = ngh_finder
        self.logger = logger
        self.trainable_embedding = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))  # landing prob at [0, 1, ... num_layers]

    def init_internal_data(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
        # initialize internal data structure to index node positions
        self.nodetime2emb_maps = self.collect_pos_mapping_ptree(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src,
                                                                subgraph_tgt)

    def collect_pos_mapping_ptree(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
        # Return:
        # nodetime2idx_maps: a list of dict {(node index, rounded time string) -> index in embedding look up matrix}
        
        subgraph_src_node, subgraph_src_ts = subgraph_src  # only use node index and timestamp to identify a node in temporal graph
        subgraph_tgt_node, subgraph_tgt_ts = subgraph_tgt
        nodetime2emb_maps = {}
        for row in range(len(src_idx_l)):
            src = src_idx_l[row]
            tgt = tgt_idx_l[row]
            cut_time = cut_time_l[row]
            src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
            src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
            tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
            tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
            nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
                                                                                src_neighbors_node, src_neighbors_ts,
                                                                                tgt_neighbors_node, tgt_neighbors_ts, batch_idx=row)
            nodetime2emb_maps.update(nodetime2emb_map)

        return nodetime2emb_maps

    @staticmethod
    def collect_pos_mapping_ptree_sample(src, tgt, cut_time, src_neighbors_node, src_neighbors_ts,
                                         tgt_neighbors_node, tgt_neighbors_ts, batch_idx):
        """
        This function has the potential of being written in numba by using numba.typed.Dict!
        """
        n_hop = len(src_neighbors_node)
        makekey = nodets2key
        nodetime2emb = {}
        
        # landing probability encoding, n_hop+1 types of probabilities for each node
        src_neighbors_node, src_neighbors_ts = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts
        tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
        for k in range(n_hop+1):
            for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                                                          tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
                if src_key not in nodetime2emb:
                    nodetime2emb[src_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                if tgt_key not in nodetime2emb:
                    nodetime2emb[tgt_key] = np.zeros((2, n_hop+1), dtype=np.float32)
                nodetime2emb[src_key][0, k] += 1
                nodetime2emb[tgt_key][1, k] += 1
        null_key = makekey(batch_idx, 0, 0.0)
        nodetime2emb[null_key] = np.zeros((2, n_hop + 1), dtype=np.float32)
        return nodetime2emb

    def forward(self, node_record, t_record):
        '''
        accept two numpy arrays each of shape [batch, k-hop-support-number], corresponding to node indices and timestamps respectively
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        '''
        # encodings = []
        device = next(self.projection.parameters()).device
        # float2str = PositionEncoder.float2str
        batched_keys = make_batched_keys(node_record, t_record)
        unique, inv = np.unique(batched_keys, return_inverse=True)
        unordered_encodings = np.array([self.nodetime2emb_maps[key] for key in unique])
        encodings = unordered_encodings[inv, :]
        encodings = torch.tensor(encodings).to(device)
        encodings = self.get_trainable_encodings(encodings)
        return encodings

    def get_trainable_encodings(self, encodings):
        '''
        Args:
            encodings: a device tensor of shape [batch, support_n, 2] / [batch, support_n, 2, L+1]
        Returns:  a device tensor of shape [batch, pos_dim]
        '''
        encodings = self.trainable_embedding(encodings.float())   # now shape [batch, support_n, 2, pos_dim]
        encodings = encodings.sum(dim=-2)  # now shape [batch, support_n, pos_dim]
        return encodings


class RandomWalkAttention(nn.Module):
    '''
    RandomWalkAttention have two modules: lstm + tranformer-self-attention
    '''
    def __init__(self, feat_dim, pos_dim, model_dim, out_dim, logger, walk_pool='attn', n_head=8, dropout_p=0.1):
        '''
        masked flags whether or not use only valid temporal walks instead of full walks including null nodes
        '''
        super(RandomWalkAttention, self).__init__()
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim//2  # half the model dim to save computation cost for attention
        self.out_dim = out_dim
        self.walk_pool = walk_pool
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.logger = logger

        self.feature_encoder = FeatureEncoder(self.feat_dim, self.model_dim, self.dropout_p)  # encode all types of features along each temporal walk
        self.self_attention = TransformerEncoderLayer(d_model=self.model_dim, nhead=self.n_head,
                                                      dim_feedforward=4*self.attn_dim, dropout=self.dropout_p,
                                                      activation='relu')
        self.pooler = SetPooler(n_features=self.model_dim, out_features=self.out_dim, dropout_p=self.dropout_p)
        self.logger.info('bi-lstm actual encoding dim: {}, attention dim: {}, attention heads: {}'.format(self.feature_encoder.model_dim, self.attn_dim, self.n_head))

    def forward_one_node(self, time_features, position_features, masks=None):
        '''
        Input shape [batch, n_walk, len_walk, *_dim]
        Return shape [batch, n_walk, feat_dim]
        '''
        combined_features = self.aggregate(time_features, position_features)
        X = self.feature_encoder(combined_features, masks)
        X = self.self_attention(X)
        X = self.pooler(X, agg='mean')  # we are actually doing mean pooling since sum has numerical issues
        return X

    def aggregate(self, time_features, position_features):
        device = time_features.device
        combined_features = torch.cat([time_features, position_features], dim=-1)
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.feat_dim)
        return combined_features
