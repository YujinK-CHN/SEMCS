import torch
import numpy as np
import torch.nn as nn
from base_policy.utils.util import init_
from base_policy.components.transformers import EncoderBlock, Mod_Sequential
from base_policy.utils.entity_util import encode_entity

"""
Skill-based policy interagte observations and skills into a unifed embedding
"""


class IntegrationCritic(nn.Module):
    def __init__(self, args, input_dim, output_dim, use_orth, device) -> None:
        super(IntegrationCritic, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.num_skills = args.num_skills
        self.output_dim = output_dim
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.n_block = args.n_block

        self.pi_use_latent = args.pi_use_latent   # only use latent not using obs

        tblocks = []
        for _ in range(self.n_block):
            tblocks.append(
                EncoderBlock(args, self.n_embd, self.n_head, use_orth)
            )
        self.tblocks = Mod_Sequential(*tblocks)
        self.input_embedding = nn.Sequential(nn.LayerNorm(input_dim), init_(args, nn.Linear(input_dim, self.n_embd*self.n_head)), nn.GELU())
        self.skill_embedding = nn.Sequential(
                    nn.LayerNorm(self.num_skills),
                    init_(args, nn.Linear(self.num_skills, self.n_embd)), 
                    nn.GELU(),
                    init_(args, nn.Linear(self.n_embd, self.n_embd*self.n_head)), 
                    nn.Tanh()
                )
        if self.pi_use_latent and self.args.skill_to_obs != "None":
            self.toprobs = nn.Sequential(nn.LayerNorm(2*self.n_embd*self.n_head), init_(args, nn.Linear(2*self.n_embd*self.n_head, output_dim)))
        else:
            self.toprobs = init_(args, nn.Linear(self.n_embd*self.n_head, output_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)
    

    def forward(self, x_list, n_agents, n_enemies, n_entities, skill_actor=None):    
        entity_ob_list, past_skill_actor, _ = encode_entity(self.args, x_list, n_agents, n_entities, self.input_dim)        
        if self.pi_use_latent and self.args.skill_to_obs != "None":
            if skill_actor is None:
                skill_actor = [past_sk.detach() for past_sk in past_skill_actor]

            x_emb = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
            skill_emb = [self.skill_embedding(skill) for skill in skill_actor]
            if self.args.skill_to_obs == "merge":
                skill_emb = [s_emb.unsqueeze(-2).repeat(1, n_en, 1) for s_emb, n_en in zip(skill_emb, n_entities)]
            # using skills to attend observation features
            x_emb, _, skill_emb, _ = self.tblocks.forward_4_skills(x_emb, skill_emb) 
            x_list = [self.toprobs(torch.cat([i_x, i_s], dim=-1)) for i_x, i_s in zip(x_emb, skill_emb)]
        else:
            x_emb = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
            # using skills to attend observation features
            x_emb, _ = self.tblocks.forward_per_task(x_emb) 
            x_list = [self.toprobs(i_x) for i_x in x_emb]
        return x_list

