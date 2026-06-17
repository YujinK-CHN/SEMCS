import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

import numpy as np
from base_policy.utils.util import init_
from base_policy.components.transformers import DecodeBlock, Mod_Sequential
from base_policy.utils.entity_util import encode_entity

class SkillAllPredictor(nn.Module):
    
    def __init__(self, args, input_dim, hidden_size, action_space, use_orthogonal, device) -> None:
        super(SkillAllPredictor, self).__init__()
        
        self.args = args
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_block = args.n_block
        
        self.input_dim = input_dim
        self.act_dim_list = [act_i[0].n for act_i in action_space]
        self.max_action_dim = max(self.act_dim_list)
        self.num_skills = args.num_skills
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain

        # predict actions for how many time steps 
        self.n_future_steps = args.n_future_steps
        self.padded_values = args.padded_values
                
        if self.args.use_norm_init:
            self.obs_encoder = nn.Sequential(init_(args, nn.Linear(input_dim, self.n_embd*self.n_head)), nn.GELU())
            self.skill_encoder = nn.Sequential(init_(args, nn.Linear(self.num_skills*2, self.n_embd*self.n_head)), nn.GELU())
        else:
            self.obs_encoder = init_(args, nn.Linear(input_dim, self.n_embd*self.n_head))
            self.skill_encoder = init_(args, nn.Linear(self.num_skills*2, self.n_embd*self.n_head))

        tblocks = []
        for _ in range(self.n_block):
            tblocks.append(
                DecodeBlock(args, self.n_embd, self.n_head, use_orthogonal)
            )
        self.tblocks = nn.Sequential(*tblocks)
        self.tblocks = Mod_Sequential(*tblocks)
        self.toprobs = init_(args, nn.Linear(self.n_embd*self.n_head, hidden_size))
        self.tpdv = dict(dtype=torch.float32, device=device)

        # action layers for domains
        if "StarCraft" in args.env_name:
            from base_policy.components.act_entity import EntityVAEACTLayer
            self.act_layer = EntityVAEACTLayer(args, action_space, hidden_size, self._use_orthogonal, self._gain)
        elif "AliceBob" in args.env_name:
            from base_policy.components.act_multi import MultiACTLayer
            self.act_layer = MultiACTLayer(args, action_space[0][0], hidden_size, self._use_orthogonal, self._gain)
        elif "Football" in args.env_name:
            from base_policy.components.act_multi import MultiACTLayer
            self.act_layer = MultiACTLayer(args, action_space[0][0], hidden_size, self._use_orthogonal, self._gain)


    def decoder(self, action_list, skill_list):
        action_embd_list = [self.obs_encoder(a_i) for a_i in action_list]
        skill_list = [self.skill_encoder(sk_i) for sk_i in skill_list]
        # skill_embd as the query
        tb, _ = self.tblocks.forward_4_actions(action_embd_list, skill_list) 
        x_list = [self.toprobs(tb_i) for tb_i in tb]
        return x_list


    def forward(self, obs, skill_list, comm_skill_list, available_actions, active_masks, n_agents, n_enemies, n_entities):        
        entity_ob_list, _, _ = encode_entity(self.args, obs, n_agents, n_entities, self.input_dim)        
        
        skill_list = [sk.unsqueeze(-2).repeat(1, n_en, 1) for sk, n_en in zip(skill_list, n_entities)]
        comm_skill_list = [c_sk.unsqueeze(-2).repeat(1, n_en, 1).detach() for c_sk, n_en in zip(comm_skill_list, n_entities)]

        combined_skill_list = [torch.cat([i_s, i_c], dim=-1) for i_s, i_c in zip(skill_list, comm_skill_list)]
        
        feature_list = self.decoder(entity_ob_list, combined_skill_list)
        
        output_action_log = self.act_layer.evaluate_preds(feature_list, available_actions, active_masks, n_enemies=n_enemies)
        
        output_action_log = [output.unsqueeze(1) for output in output_action_log]
        
        return output_action_log # (batch_size, n_agent, 1), (batch_size, n_agent, 1)

