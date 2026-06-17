import torch
import numpy as np
import torch.nn as nn
from base_policy.utils.util import init_, print_cuda
from base_policy.components.transformers import EncoderBlock, Mod_Sequential
from base_policy.components.multi_head_hard_attention import MultiHeadHardAttention
from base_policy.utils.entity_util import encode_entity
"""
Skill-based policy interagte observations and skills into a unifed embedding
"""

from base_policy.algorithms.mcs.base.integation_comm import IntegrationComm

class MergeIntegrationComm(IntegrationComm):
    def __init__(self, args, input_dim, output_dim, use_orth, device) -> None:
        super(MergeIntegrationComm, self).__init__(args, input_dim, output_dim, use_orth, device)

        self.comm_threshold = args.comm_threshold
        if self.comm_threshold == 0:
            print("do not use threshold")

        # aggregate messages if communication is used
        if "Trans" in self.op_aggregate:
            from base_policy.algorithms.mcs.mcs_merge.merge_aggregation import MergeTransformerAgg as Aggregator
            self.aggregation = Aggregator(args, self.num_skills, self.num_skills, device, use_orth)
        elif "GRU" in self.op_aggregate:
            from base_policy.algorithms.mcs.mcs_merge.merge_aggregation import MergeGRUSAgg as Aggregator
            self.aggregation = Aggregator(args, self.num_skills, self.num_skills, device, use_orth)

        if "CommMask" == self.comm_channel:
            self.att_input_shape = args.num_skills
            self.hard_attention = MultiHeadHardAttention(args, self.att_input_shape, self.att_input_shape,
                                                     self.args.n_head, args.n_embd)

    def communication(self, skills, active_masks, n_agents, train_info, record_info):
        """
        prepare messages before aggregating messages from other agents
        active_masks: mask dead agents during communication
        """
        ### reshape the skills
        assert self.comm_channel != "None"
        train_info["comm_links"] = 1
        record_info["comm_weights"] = [torch.zeros(0) for _ in skills]

        # mask dead agents
        if self.comm_use_active_masks:
            skills = [sk.view(-1, n_a, self.args.num_skills) * mk.view(-1, n_a, 1) for sk, mk, n_a in zip(skills, active_masks, n_agents)]    
        else:
            skills = [sk.view(-1, n_a, self.args.num_skills) for sk, n_a in zip(skills, n_agents)]  
        
        # use communication
        if "All" == self.comm_channel:
            comm_skills = [c_skill.unsqueeze(1).repeat(1, n_a, 1, 1).view(-1, n_a, self.args.num_skills) for c_skill, n_a in zip(skills, n_agents)]
        elif "ExcludeSelf" == self.comm_channel: 
            comm_skills = []
            for idx, n_a in enumerate(n_agents):  # per task
                skill_per_task = []
                for i in range(n_a):  # per agent
                    comm_s = torch.cat((skills[idx][:, :i, :], skills[idx][:, i+1:, :]), 1)
                    skill_per_task.append(comm_s)
                skill_per_task = torch.stack(skill_per_task, 1).view(-1, n_a-1, self.args.num_skills)
                comm_skills.append(skill_per_task)
        elif "CommMask" == self.comm_channel:
            # prepare attention inputs for agent communication
            queries_list = []
            mask_list = []
            comm_skill_list = []
            for skill_tensor in skills:
                B, N, D = skill_tensor.shape
                # use self skills as queries
                queries = skill_tensor.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, D]
                queries = queries.view(-1, N, D)
                # mask out self skills
                mask = (1 - torch.eye(N, device=self.device)).unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]
                mask = mask.view(-1, N)  # [B*N, N]
                # all agents' skills
                comm_skill = skill_tensor.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, D]
                comm_skill = comm_skill.view(-1, N, D)
                queries_list.append(queries)
                mask_list.append(mask)
                comm_skill_list.append(comm_skill)

            # hard-attention
            soft_weight_list = [self.hard_attention(q_sk, c_sk, c_mask) for q_sk, c_sk, c_mask in zip(queries_list, comm_skill_list, mask_list)]            
            soft_weight_list = [soft_weight.reshape(-1, self.args.n_head, n_a) for soft_weight, n_a in zip(soft_weight_list, n_agents)]
            soft_weight_list = [soft_weight.mean(dim=1) for soft_weight in soft_weight_list]
            # when self.comm_threshold == 0, do not use threshold
            if self.comm_threshold > 0:
                soft_weight_list = [(soft_weight > self.comm_threshold).float() * soft_weight for soft_weight in soft_weight_list]            

            # combine attention and skills
            comm_skills = [c_sk * soft_weight.unsqueeze(-1) for c_sk, soft_weight in zip(comm_skill_list, soft_weight_list)]

            # record comm links and weights
            all_soft_weights = torch.cat([hw.view(-1) for hw in soft_weight_list], dim=0)
            num_nonzero = (all_soft_weights != 0).sum().item()
            num_total = all_soft_weights.numel()
            percentage_nonzero = 100.0 * num_nonzero / num_total
            # print(f"Percentage of active communication links: {percentage_nonzero:.2f}%")
            train_info["comm_links"] = percentage_nonzero
            train_info["comm_loss"] = torch.norm(torch.cat([c.reshape(-1) for c in soft_weight_list], dim=0), p=2)
            record_info["comm_weights"] = soft_weight_list
        return comm_skills


    def forward(self, x_list, h_state, masks, active_masks, n_agents, n_enemies, n_entites, is_training, deterministic=False):
        record_info = {}
        skill_list, skill_dot, skill_emb, comm_skill_list, comm_skill_dot, comm_skill_emb, train_info = None, None, None, None, None, None, None
        
        entity_ob_list, past_skill_list, _ = encode_entity(self.args, x_list, n_agents, n_entites, self.input_dim)        
        if self.skill_generator and self.pi_use_latent:
            skill_list, skill_dot, skill_emb, h_state, train_info = self.skill_generator(entity_ob_list, past_skill_list, h_state, masks, active_masks, n_agents, n_entites, is_training, deterministic)
            
        if self.comm_channel != "None": 
            comm_skill_list = self.communication(skill_list, active_masks, n_agents, train_info, record_info)
            comm_skill_list, comm_skill_emb, h_state, _ = self.aggregation(comm_skill_list, h_state, active_masks, n_agents, n_entites)
        else:
            # when communication is not used and skill is used, sender == receiver
            if self.pi_use_latent:
                comm_skill_list = skill_list
                comm_skill_emb = skill_emb

        # filter emb if do not reduce connection between encoder and decoder
        if not self.args.reduce_connection and self.pi_use_latent:
            comm_skill_emb = [self.skill_embedding(sk) for sk in comm_skill_list]
        
        if self.pi_use_obs and not self.pi_use_latent:  # pi_use_obs=1, pi_use_latent=0
            x_emb = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
            # using skills to attend observation features
            x_emb, _ = self.tblocks.forward_per_task(x_emb)
            fea_list = [self.toprobs(i_x) for i_x in x_emb]
        elif not self.pi_use_obs and self.pi_use_latent:   # pi_use_obs=0, pi_use_latent=1
            # when merge two embeddings, repeat skills for entities when communication used
            comm_skill_emb = [s_emb.unsqueeze(-2).repeat(1, n_en, 1) for s_emb, n_en in zip(comm_skill_emb, n_entites)]
            # using skills to attend observation features
            comm_skill_emb, comm_skill_dot = self.tblocks.forward_per_task(comm_skill_emb)
            fea_list = [self.toprobs(i_s) for i_s in comm_skill_emb]
        else:  # pi_use_obs=1, pi_use_latent=1
            x_emb = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
            # when merge two embeddings, repeat skills for entities when communication used
            comm_skill_emb = [s_emb.unsqueeze(-2).repeat(1, n_en, 1) for s_emb, n_en in zip(comm_skill_emb, n_entites)]
            # using skills to attend observation features
            x_emb, _, comm_skill_emb, comm_skill_dot = self.tblocks.forward_4_skills(x_emb, comm_skill_emb) 
            fea_list = [self.toprobs(torch.cat([i_x, i_s], dim=-1)) for i_x, i_s in zip(x_emb, comm_skill_emb)]

        record_info["skill_dot"] = skill_dot
        record_info["comm_skill_dot"] = comm_skill_dot

        return fea_list, skill_list, skill_emb, comm_skill_list, comm_skill_emb, entity_ob_list, h_state, train_info, record_info
