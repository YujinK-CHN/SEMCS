import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

import numpy as np
from base_policy.utils.util import init_
from base_policy.components.transformers import DecodeBlock, Mod_Sequential
from itertools import combinations
# from geomloss import SamplesLoss

class SkillAutoPredictor(nn.Module):
    
    def __init__(self, args, action_space, use_orthogonal, device) -> None:
        super(SkillAutoPredictor, self).__init__()
        
        self.args = args
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_block = args.n_block
        
        self.act_dim_list = [act_i[0].n for act_i in action_space]
        self.max_action_dim = max(self.act_dim_list) + 1
        self.num_skills = args.num_skills
        
        # predict actions for how many time steps 
        self.n_future_steps = args.n_future_steps
        self.padded_values = args.padded_values
        
        if self.args.use_norm_init:
            self.action_encoder = nn.Sequential(init_(args, nn.Linear(self.max_action_dim, self.n_embd*self.n_head)), nn.GELU())
            self.skill_encoder = nn.Sequential(init_(args, nn.Linear(self.num_skills, self.n_embd*self.n_head)), nn.GELU())
        else:
            self.action_encoder = init_(args, nn.Linear(self.max_action_dim, self.n_embd*self.n_head))
            self.skill_encoder = init_(args, nn.Linear(self.num_skills, self.n_embd*self.n_head))

        tblocks = []
        for _ in range(self.n_block):
            tblocks.append(
                DecodeBlock(args, self.n_embd, self.n_head, use_orthogonal)
            )
        self.tblocks = nn.Sequential(*tblocks)
        self.tblocks = Mod_Sequential(*tblocks)
        self.toprobs = init_(args, nn.Linear(self.n_embd*self.n_head, self.max_action_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)


    def decoder(self, action_list, skill_list):
        action_embd_list = [self.action_encoder(a_i) for a_i in action_list]
        skill_list = [self.skill_encoder(sk_i) for sk_i in skill_list]
        # skill_embd as the query
        tb, _ = self.tblocks.forward_4_actions(action_embd_list, skill_list) 
        logit_list = [self.toprobs(tb_i) for tb_i in tb]
        return logit_list


    def forward(self, skill_embed_list, future_available_actions):        
        # the shape of x in x_list is (bs_na, na+ne, fd)
        batch_size_list = [a_i.size(0) for a_i in skill_embed_list]
        n_steps = self.n_future_steps
        
        # available_actions: (batch, n_agent+1, action_dim)
        if self.args.skill_to_obs == "merge":
            skill_embed_list = [sk_i.repeat(1, n_steps).view(ba_i, n_steps, -1) for ba_i, sk_i in zip(batch_size_list, skill_embed_list)]
        elif self.args.skill_to_obs == "entity":
            skill_embed_list = [sk_i.mean(-2).repeat(1, n_steps).view(ba_i, n_steps, -1) for ba_i, sk_i in zip(batch_size_list, skill_embed_list)]

        shifted_action = [torch.zeros((batch_size, n_steps, act_dim_i+1)).to(**self.tpdv) for batch_size, act_dim_i in zip(batch_size_list, self.act_dim_list)]
        # initialize actions
        for shifted_act in shifted_action:
            shifted_act[:, 0, 0] = 1
        
        # padding shifted_action for the rest of dimensions
        shifted_action = [F.pad(t, (0, self.max_action_dim - t.shape[-1]), value=self.padded_values) for t in shifted_action]
        idx_tuple = range(len(batch_size_list))

        output_action = [torch.zeros((batch_size, n_steps, 1), dtype=torch.long) for batch_size in batch_size_list]
        output_action_log = [torch.zeros((batch_size, n_steps, act_dim_i)).to(**self.tpdv) for batch_size, act_dim_i in zip(batch_size_list, self.act_dim_list)]

        for i in range(n_steps):
            logit_list = self.decoder(shifted_action, skill_embed_list)
            
            for idx, logit, available_acts, act_dim_i in zip(idx_tuple, logit_list, future_available_actions, self.act_dim_list):
                logit = logit[:, i, :act_dim_i]
                if available_acts is not None:
                    value = -1e10
                    if logit.dtype == torch.float16:
                        value = max(min(value, 65504), -65504)  # clamp within float16 range 
                    logit[available_acts[:, i, :] == 0] = value

                distri = Categorical(logits=logit)
                action = distri.probs.argmax(dim=-1)
                action_log = F.log_softmax(logit, dim=-1)

                output_action[idx][:, i, :] = action.unsqueeze(-1)
                output_action_log[idx][:, i, :] = action_log
                if i + 1 < n_steps:
                    new_shifted_action = shifted_action[idx].clone()
                    new_shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=act_dim_i)
                    shifted_action[idx] = new_shifted_action

        return output_action_log # (batch_size, n_agent, 1), (batch_size, n_agent, 1)
