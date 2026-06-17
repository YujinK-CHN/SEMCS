import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_policy.utils.util import init_, init_gru
from base_policy.components.transformers import EncoderBlock, Mod_Sequential
from base_policy.utils.entity_util import decode_skill

from base_policy.algorithms.mcs.base.aggregation import TransformerAgg, GRUSAgg


class MergeTransformerAgg(TransformerAgg):
    
    def forward(self, skill_list, h_state, active_masks, n_agents, n_entites, deterministic=False):
        """
        @param skill_list: the skills from other agents via communication or their own entities in each task
        """
        x_size_list = [sk.size() for sk in skill_list]  # size (bs_na, num_skills)
        bs_na_list = [sk.size(0) for sk in skill_list]  # size (bs_na, num_skills)
        skill_dot = None
                
        skill_embed = [self.input_embedding(s_i) for s_i in skill_list]
        if self.args.share_tblocks:   # share skill features across tasks
            skill_embed, skill_dot = self.tblocks(skill_embed)
            skill_embed = decode_skill(self.args, skill_embed, n_agents, n_entites, bs_na_list, self.n_embd*self.n_head)  
        else:  # do not share features across tasks
            skill_embed, skill_dot = self.tblocks.forward_per_task(skill_embed)

        skill_embed = [tb_i.view(bs_na[0], bs_na[1], -1) for tb_i, bs_na in zip(skill_embed, x_size_list)]

        ####### mean or sum over messages #######
        if "Mean" in self.op_aggregate:
            skill_embed = [tb_i.mean(-2) for tb_i in skill_embed]
        elif "Sum" in self.op_aggregate:
            skill_embed = [tb_i.sum(-2) for tb_i in skill_embed]  
        # otherwise all received messages will be preserved  
        ####### mean or sum over messages #######
        
        skill_list = [self.toprobs(tb_i) for tb_i in skill_embed]

        return skill_list, skill_embed, h_state, skill_dot


class MergeGRUSAgg(GRUSAgg):
    def __init__(self, args, input_dim, output_dim, device, use_orthogonal) -> None:
        super(MergeGRUSAgg, self).__init__(args, input_dim, output_dim, device, use_orthogonal)    
        
        # output aggregated skills
        self.fc1 = init_(args, nn.Linear(input_dim, self.hidden_size*self.n_head))
        self.rnn = init_gru(args, nn.GRU(self.hidden_size*self.n_head, self.hidden_size*self.n_head, num_layers=args.recurrent_N, batch_first=True))
        self.fc2 = init_(args, nn.Linear(self.hidden_size*self.n_head, output_dim))


    def forward(self, skill_list, h_state, active_masks, n_agents, n_entites, deterministic=False):
        '''
        use rnn to process communicated skills
            sk_i: (chunk_length*bs*na, (na+ne), feat_dim) / (threads*na, (na+ne), feat_dim)
            p_s: (chunk_length*bs*na, subtask_dim) / (threads*na, subtask_dim)
            h_i: (bs*na, recurrent_N, rnn_hidden_dim) / (threads*na, recurrent_N, rnn_hidden_dim)
            active_masks: (chunk_length*bs*na, 1) / (threads*na, 1)
        '''
        agg_skill_list, skill_embed_list, h_state_list = [], [], []
        for sk_i, h_i, mask_i in zip(skill_list, h_state, active_masks):
            [x_bs, _, _] = sk_i.size()
            [hxs_bs, rnn_layers, _] = h_i.size()
            assert mask_i.size(0) == x_bs
            sk_i = self.fc1(sk_i) # (x_bs, num_agents, num_skills) --> (x_bs, num_agents, n_emb)
            if x_bs == hxs_bs:
                h_i = h_i.repeat(1, 1, self.n_head)
                if self.comm_use_active_masks:
                    mask_i_re = mask_i.unsqueeze(-2).repeat(1, rnn_layers, 1) # (x_bs, rnn_layers, 1)
                    sk_i, h_i = self.rnn(sk_i, (h_i * mask_i_re).transpose(0, 1).contiguous())
                else:
                    sk_i, h_i = self.rnn(sk_i, h_i.transpose(0, 1).contiguous())
                h_i = h_i.transpose(0, 1)    
            else:
                # sk_i is a (T, N, -1) tensor that has been flatten to (T * N, nae, feat_dim)
                N = hxs_bs
                T = int(x_bs / N)  # T chunk
                h_i = h_i.repeat(T, 1, self.n_head)  # repeat for each chunk
                if self.comm_use_active_masks:
                    mask_i_re = mask_i.unsqueeze(-2).repeat(1, rnn_layers, 1) # (x_bs, rnn_layers, 1)
                    sk_i, h_i = self.rnn(sk_i, (h_i * mask_i_re).transpose(0, 1).contiguous())
                else:
                    sk_i, h_i = self.rnn(sk_i, h_i.transpose(0, 1).contiguous())
                h_i = h_i.transpose(0, 1)     
            # mean pool to aggregate features from agents
            sk_i = sk_i.mean(1)
            # to skills
            fc_sk_i = self.fc2(sk_i)
            h_i = h_i[:, :, ::self.n_head]   
            
            agg_skill_list.append(fc_sk_i)
            skill_embed_list.append(sk_i)
            h_state_list.append(h_i)

        return agg_skill_list, skill_embed_list, h_state_list, None
