import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod

from base_policy.utils.util import init_, print_cuda
from base_policy.components.transformers import EncoderBlock, Mod_Sequential
from base_policy.utils.entity_util import encode_entity
"""
Skill-based policy interagte observations and skills into a unifed embedding
"""


class IntegrationComm(nn.Module):
    def __init__(self, args, input_dim, output_dim, use_orth, device) -> None:
        super(IntegrationComm, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.num_skills = args.num_skills
        self.output_dim = output_dim
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.n_block = args.n_block
                
        self.skill_choice = args.skill_choice
        self.pi_choice = args.pi_choice # the choice of how to utilize skills in the action policy
        self.use_action_predictor = args.use_action_predictor
        self.op_aggregate = args.op_aggregate   # the way to aggregate messages

        self.comm_channel = args.comm_channel   # the way to communicate; may extend to communication graph
        self.comm_use_active_masks = args.comm_use_active_masks   # whether to use active mask during communication
        
        self.pi_use_obs = args.pi_use_obs   # only use obs not using latent
        self.pi_use_latent = args.pi_use_latent   # only use latent not using obs

        tblocks = []
        for _ in range(self.n_block):
            tblocks.append(
                EncoderBlock(args, self.n_embd, self.n_head, use_orth)
            )
        # self.tblocks = nn.Sequential(*tblocks)
        self.tblocks = Mod_Sequential(*tblocks)

        """
        Generate skills by using VAE, GRU, or Transformer
        """
        if self.skill_choice == "UseVAE":
            from base_policy.algorithms.mcs.skill_generator import VAESkillGenerator
            self.skill_generator = VAESkillGenerator(args, input_dim, self.num_skills, device)
        elif self.skill_choice == "UseGRU":
            from base_policy.algorithms.mcs.skill_generator import GRUSkillGenerator
            self.skill_generator = GRUSkillGenerator(args, input_dim, self.num_skills, device, use_orth)
        elif self.skill_choice == "UseTrans":
            from base_policy.algorithms.mcs.skill_generator import TransformerSkillGenerator
            self.skill_generator = TransformerSkillGenerator(args, input_dim, self.num_skills, device, use_orth)
        else:
            self.skill_generator = None
        
        """
        Several ways can be used to integrate observations and skills
        1. CatTrans: concatenate the transformer's encoded embeddings of observations and skills (with self-attention for each embeddings)
        2. CatDecoder: simply use a decoder where skills is used as query vectors (but autoregressive is not used)
        """
        if self.pi_choice == "CatTrans":
            if self.args.use_norm_init:
                self.input_embedding = nn.Sequential(nn.LayerNorm(input_dim), init_(args, nn.Linear(input_dim, self.n_embd*self.n_head)), nn.GELU())
                self.skill_embedding = nn.Sequential(
                    nn.LayerNorm(self.num_skills),
                    init_(args, nn.Linear(self.num_skills, self.n_embd)), 
                    nn.GELU(),
                    init_(args, nn.Linear(self.n_embd, self.n_embd*self.n_head)), 
                    nn.Tanh()
                )
            else:
                self.input_embedding = init_(args, nn.Linear(input_dim, self.n_embd*self.n_head))
                self.skill_embedding = nn.Sequential(
                    init_(args, nn.Linear(self.num_skills, self.n_embd)), 
                    init_(args, nn.Linear(self.n_embd, self.n_embd*self.n_head)), 
                    nn.Tanh()
                )
            # set for only use obs or only use skills
            if self.pi_use_obs and not self.pi_use_latent:  # pi_use_obs=1, pi_use_latent=0
                self.toprobs = init_(args, nn.Linear(self.n_embd*self.n_head, output_dim))
            elif not self.pi_use_obs and self.pi_use_latent:   # pi_use_obs=0, pi_use_latent=1
                self.toprobs = init_(args, nn.Linear(self.n_embd*self.n_head, output_dim))
            else:  # pi_use_obs=1, pi_use_latent=1
                self.toprobs = nn.Sequential(nn.LayerNorm(2*self.n_embd*self.n_head), init_(args, nn.Linear(2*self.n_embd*self.n_head, output_dim)))
        else:
            pass
        
        # self.toprobs = init_(nn.Linear(n_embd*self.n_head+self.num_skills, output_dim))
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

    
    @abstractmethod
    def communication(self, skills, active_masks, n_agents):
        pass

    @abstractmethod
    def forward(self, x_list, h_state, masks, active_masks, n_agents, n_enemies, is_training, deterministic=False):
        pass