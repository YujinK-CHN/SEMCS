import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod

from base_policy.utils.util import init_, init_gru
from base_policy.components.transformers import EncoderBlock, Mod_Sequential
from base_policy.utils.entity_util import decode_skill


class TransformerAgg(nn.Module):
    def __init__(self, args, input_dim, output_dim, device, use_orthogonal) -> None:
        super(TransformerAgg, self).__init__()
        
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_block = args.n_block

        self.comm_channel = args.comm_channel   # the way to communicate; may extend to communication graph
        self.op_aggregate = args.op_aggregate   # the way to aggregate messages

        if self.args.use_norm_init:
            self.input_embedding = nn.Sequential(nn.LayerNorm(input_dim), init_(args, nn.Linear(input_dim, self.n_embd*self.n_head)), nn.GELU())
        else:
            self.input_embedding = init_(args, nn.Linear(input_dim, self.n_embd*self.n_head))

        tblocks = []
        for _ in range(self.n_block):
            tblocks.append(
                EncoderBlock(args, self.n_embd, self.n_head, use_orthogonal)
            )
        self.tblocks = nn.Sequential(*tblocks)
        self.tblocks = Mod_Sequential(*tblocks)
        self.toprobs = init_(args, nn.Linear(self.n_embd*self.n_head, output_dim))

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.all_skill_onehot = torch.eye(output_dim).to(**self.tpdv)

    @abstractmethod
    def forward(self, skill_list, h_state, active_masks, n_agents, n_enemies, deterministic=False):
        pass


class GRUSAgg(nn.Module):
    def __init__(self, args, input_dim, output_dim, device, use_orthogonal) -> None:
        super(GRUSAgg, self).__init__()

        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.comm_use_active_masks = args.comm_use_active_masks   # whether to use active mask during communication
        
        self.hidden_size = args.hidden_size # this is hidden_size for GRU, but enforced to be equal to n_embd to produced skill embedding
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.all_skill_onehot = torch.eye(output_dim).to(**self.tpdv)

    @abstractmethod
    def forward(self, x, hxs, active_masks):
        pass
