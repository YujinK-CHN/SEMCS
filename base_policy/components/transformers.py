import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base_policy.utils.util import init_


###################################################################################################
######################################### Transformers ############################################
###################################################################################################

class SelfAttention(nn.Module):
    def __init__(self, args, n_emb=32, n_head=4, use_orth=True) -> None:
        super(SelfAttention, self).__init__()
        # q, k, v is same at the dim
        self.n_emb = n_emb
        self.n_head = n_head
        self.sqrt_n_emb = self.n_emb ** 0.5
        
        self.q_w = init_(args, nn.Linear(self.n_emb * self.n_head, self.n_emb * self.n_head))
        self.k_w = init_(args, nn.Linear(self.n_emb * self.n_head, self.n_emb * self.n_head))
        self.v_w = init_(args, nn.Linear(self.n_emb * self.n_head, self.n_emb * self.n_head))
        # self.head2emb = init_(nn.Linear(self.n_emb * self.n_head, self.n_emb))
        # self.talking_w = init_(nn.Linear(args.n_agents+args.n_enemies, args.n_agents+args.n_enemies))
    
    def forward_i(self, features):
        # bs: batch_size, ne: num_eneities, fd: feature_dim, hs: n_head
        bs, ne, _ = features.size()
        hs, fd = self.n_head, self.n_emb
        q_wave = self.q_w(features).view(bs, ne, hs, fd)
        k_wave = self.k_w(features).view(bs, ne, hs, fd)
        v_wave = self.v_w(features).view(bs, ne, hs, fd)

        # swap the dimensions
        q_wave = q_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        k_wave = k_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        v_wave = v_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)

        # dot = F.softmax(self.talking_w(torch.matmul(q_wave, k_wave.transpose(-1, -2))/self.sqrt_n_emb), dim=-1)
        # the matmul here is actually doing batch matrix-matrix product: ((bs*hs)×ne×fd),((bs*hs)×fd×ne) -->((bs*hs)×ne×ne)
        features_dot = F.softmax(torch.matmul(q_wave, k_wave.transpose(-1, -2))/self.sqrt_n_emb, dim=-1)  # agents-to-agents
        features_attention = torch.matmul(features_dot, v_wave).view(bs, hs, ne, fd)  # agents-to-features
        features_attention = features_attention.transpose(1, 2).contiguous().view(bs, ne, hs*fd)

        # ? do we need this head embedding?
        # features_attention = self.head2emb(features_attention)
        return features_attention, features_dot

    
    def forward(self, features):
        """
        concate all features from tasks
        """
        hs, fd = self.n_head, self.n_emb
        features_attention, features_dot = zip(*[self.forward_i(feature) for feature in features])
        features_attention = torch.cat([f_i.view(-1, hs*fd) for f_i in features_attention], dim=0)  # ! cat features from all tasks
        return features_attention, features_dot
    

    def forward_per_task(self, features):
        """
        process features from different tasks
        """
        features_attention, features_dot = zip(*[self.forward_i(feature) for feature in features])
        features_attention, features_dot = features_attention, features_dot
        
        return features_attention, features_dot


    def forward_4_skill_i(self, x_emb, skill_emb):
        # bs: batch_size, ne: num_eneities, fd: feature_dim, hs: n_head
        bs, ne, _ = x_emb.size()  # e.g., x_emb torch.Size([12, 6, 192])
        hs, fd = self.n_head, self.n_emb
        q_skill = self.q_w(skill_emb).unsqueeze(1).view(bs, ne, hs, fd)
        q_wave = self.q_w(x_emb).view(bs, ne, hs, fd)
        
        k_wave = self.k_w(x_emb).view(bs, ne, hs, fd)
        v_wave = self.v_w(x_emb).view(bs, ne, hs, fd)

        q_skill = q_skill.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        q_wave = q_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        k_wave = k_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        v_wave = v_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)

        skill_dot = F.softmax(torch.matmul(q_skill, k_wave.transpose(-1, -2)) / self.sqrt_n_emb, dim=-1)
        skill_attention = torch.matmul(skill_dot, v_wave).view(bs, hs, ne, fd)
        skill_attention = skill_attention.transpose(1, 2).contiguous().view(bs, ne, hs*fd)

        x_dot = F.softmax(torch.matmul(q_wave, k_wave.transpose(-1, -2)) / self.sqrt_n_emb, dim=-1)
        x_attention = torch.matmul(x_dot, v_wave).view(bs, hs, ne, fd)
        x_attention = x_attention.transpose(1, 2).contiguous().view(bs, ne, hs*fd)
        # features_attention = self.head2emb(features_attention) 
        
        return x_attention, x_dot, skill_attention, skill_dot


    def forward_4_skills(self, x_emb, skill_emb):
        """
        process features together with skills
        """
        x_emb_tuple, x_dot, skill_emb_tuple, skill_dot_tuple = zip(*[self.forward_4_skill_i(i_x, i_s) for i_x, i_s in zip(x_emb, skill_emb)])
        x_emb, x_dot, skill_emb, skill_dot = list(x_emb_tuple), list(x_dot), list(skill_emb_tuple), list(skill_dot_tuple)
        return x_emb, x_dot, skill_emb, skill_dot


    def forward_4_action_i(self, key, value, query):
        # bs: batch_size, ne: num_eneities, fd: feature_dim, hs: n_head
        bs, ne, _ = key.size()
        hs, fd = self.n_head, self.n_emb
        
        k_wave = self.k_w(key).view(bs, ne, hs, fd)
        v_wave = self.v_w(value).view(bs, ne, hs, fd)
        q_wave = self.q_w(query).view(bs, ne, hs, fd)

        # swap the dimensions
        k_wave = k_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        v_wave = v_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)
        q_wave = q_wave.transpose(1, 2).contiguous().view(bs*hs, ne, fd)

        features_dot = F.softmax(torch.matmul(q_wave, k_wave.transpose(-1, -2))/self.sqrt_n_emb, dim=-1)  # agents-to-agents
                
        features_attention = torch.matmul(features_dot, v_wave).view(bs, hs, ne, fd)  # agents-to-features
        features_attention = features_attention.transpose(1, 2).contiguous().view(bs, ne, hs*fd)

        return features_attention, features_dot
    
    
    def forward_4_actions(self, key_list, value_list, query_list):
        """
        process features from different tasks
        """
        features_attention, features_dot = zip(*[self.forward_4_action_i(key_i, value_i, query_i) for key_i, value_i, query_i in zip(key_list, value_list, query_list)])
        features_attention, features_dot = features_attention, features_dot
        
        return features_attention, features_dot



class EncoderBlock(nn.Module):
    def __init__(self, args, n_emb, n_head, use_orth, ff_hidden_mult=4, dropout=0.0) -> None:
        super(EncoderBlock, self).__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        
        self.attention = SelfAttention(args, n_emb, n_head, use_orth)

        # prior norm
        self.norm1 = nn.LayerNorm(n_emb*n_head)
        self.norm2 = nn.LayerNorm(n_emb*n_head)
        
        self.ff = nn.Sequential(
            init_(args, nn.Linear(n_emb*n_head, ff_hidden_mult*n_emb)),
            nn.GELU(),  # better than RELU
            init_(args, nn.Linear(ff_hidden_mult*n_emb, n_emb*n_head))
        )

    def forward(self, x):
        '''
        using concatenated features for all tasks
        '''
        x_attention, x_dot = self.attention(x)
        x = torch.cat([x_i.view(-1, self.n_emb*self.n_head) for x_i in x])        
        x = self.norm1(x+ x_attention)
        x = self.norm2(x + self.ff(x))
        return x, x_dot

    def forward_per_task(self, x):
        """
        using separate features for each task, and this will flatten the enitites
        """
        x_attention, x_dot = self.attention.forward_per_task(x)        
        x = [self.norm1(x_i + x_att_i) for x_i, x_att_i in zip(x, x_attention)]
        x_fedforward = [self.ff(x_i) for x_i in x]
        x = [self.norm2(x_i + x_f_i) for x_i, x_f_i in zip(x, x_fedforward)]
        return x, x_dot
    

    def forward_4_skills(self, x_emb, skill_emb):
        """
            forward to incorporate skills
        """
        x_emb_attended, x_dot, skill_emb_attended, skill_dot = self.attention.forward_4_skills(x_emb, skill_emb)
        x_emb = [self.norm1(i_x_a + i_x) for i_x, i_x_a in zip(x_emb, x_emb_attended)]
        x_fedforward = [self.ff(i_x) for i_x in x_emb]
        x_emb = [self.norm2(i_xf + i_x) for i_xf, i_x in zip(x_fedforward, x_emb)]
        skill_emb = [self.norm1(i_s_a + i_s) for i_s, i_s_a in zip(skill_emb, skill_emb_attended)]
        s_fedforward = [self.ff(i_s) for i_s in skill_emb]
        skill_emb = [self.norm2(i_sf + i_s) for i_sf, i_s in zip(s_fedforward, skill_emb)]
        return x_emb, x_dot, skill_emb, skill_dot




class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, args, n_emb, n_head, use_orth, ff_hidden_mult=4, dropout=0.0) -> None:
        super(DecodeBlock, self).__init__()

        self.n_emb = n_emb
        self.n_head = n_head

        self.attn1 = SelfAttention(args, n_emb, n_head, use_orth)
        self.attn2 = SelfAttention(args, n_emb, n_head, use_orth)
        
        self.norm1 = nn.LayerNorm(n_emb*n_head)
        self.norm2 = nn.LayerNorm(n_emb*n_head)
        self.norm3 = nn.LayerNorm(n_emb*n_head)
        
        self.ff = nn.Sequential(
            init_(args, nn.Linear(n_emb*n_head, ff_hidden_mult*n_emb)),
            nn.GELU(),  # better than RELU
            init_(args, nn.Linear(ff_hidden_mult*n_emb, n_emb*n_head))
        )

    def forward(self, x, rep_enc):
        x = self.norm1(x + self.attn1(x, x, x))
        # in the paper, the rep_enc will not be added in the normalization layer
        x = self.norm2(rep_enc + self.attn2(x, x, rep_enc))
        x = self.norm3(x + self.ff(x))
        return x

    def forward_4_actions(self, x, rep_enc):
        """
        using separate features for each task, and this will flatten the enitites
        """
        x_attention, _ = self.attn1.forward_4_actions(x, x, x)        
        x = [self.norm1(x_i + x_att_i) for x_i, x_att_i in zip(x, x_attention)]
        x_attention_two, x_dot = self.attn2.forward_4_actions(x, x, rep_enc)   
        x = [self.norm2(rep_i + x_att_i) for rep_i, x_att_i in zip(rep_enc, x_attention_two)]
        x_fedforward = [self.ff(x_i) for x_i in x]
        x = [self.norm3(x_i + x_f_i) for x_i, x_f_i in zip(x, x_fedforward)]
        return x, x_dot
    


class Mod_Sequential(nn.Sequential):
    def forward_4_skills(self, x_emb, skill_emb):
        for module in self:
            x_emb, x_dot, skill_emb, skill_dot = module.forward_4_skills(x_emb, skill_emb)
        return x_emb, x_dot, skill_emb, skill_dot

    def forward_per_task(self, x_emb):
        for module in self:
            x_emb, x_dot = module.forward_per_task(x_emb)
        return x_emb, x_dot

    def forward_4_actions(self, x_emb, rep_enc):
        for module in self:
            x_emb, x_dot = module.forward_4_actions(x_emb, rep_enc)
        return x_emb, x_dot
