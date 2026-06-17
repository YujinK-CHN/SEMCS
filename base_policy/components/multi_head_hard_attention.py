# -*- coding: utf-8 -*-
# @Time    : 2023/12/29 20:57
# @File    : multi_head_hard_attention.py
# @Software: PyCharm
# @mail    : zhangzh.grey@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_policy.utils.util import init_


def multi_mask_gumbel_softmax(x, mask):
    if mask is None:
        return F.gumbel_softmax(x, dim=-1, tau=0.01)[:, :, 1]
    else:
        shape = x.shape
        mask = mask.reshape(-1)        
        x = sequence_mask(x.reshape(-1, shape[-1]), mask, value=-1e10)
        return F.gumbel_softmax(x.reshape(shape), dim=-1, tau=0.01)[:, :, 1]

def single_mask_gumbel_softmax(x, mask):
    if mask is None:
        return F.gumbel_softmax(x, dim=-1, tau=0.01)
    else:
        shape = x.shape
        mask = mask.reshape(-1)        
        x = sequence_mask(x.reshape(-1), mask, value=-1e10)
        return F.gumbel_softmax(x.reshape(shape), dim=-1, tau=0.01)


def transpose_task_input(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def sequence_mask(x, mask, value):
    if x.dtype == torch.float16:
        value = max(min(value, 65504), -65504)  # clamp within float16 range
    mask = mask.to(torch.bool)
    x[~mask] = value
    return x


class AdditiveAttention(nn.Module):
    def __init__(self, args, key_size, query_size, num_hiddens):
        super(AdditiveAttention, self).__init__()
        self.hidden_dim = num_hiddens
        self.single_comm = args.single_comm
        self.w_k = init_(args, nn.Linear(key_size, num_hiddens, bias=False))
        self.w_q = init_(args, nn.Linear(query_size, num_hiddens, bias=False))
        if self.single_comm:
            print("Use single comm....")
            self.w_v = init_(args, nn.Linear(num_hiddens, 1, bias=False))
        else:
            print("Use multi comm....")
            self.w_v = init_(args, nn.Linear(num_hiddens, 2, bias=False))

    def forward(self, queries, keys, mask):
        queries, keys = self.w_q(queries), self.w_k(keys)        
        features = queries + keys
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)  
        if self.single_comm:
            weights = single_mask_gumbel_softmax(scores, mask)
        else:
            weights = multi_mask_gumbel_softmax(scores, mask)

        return weights


class MultiHeadHardAttention(nn.Module):
    def __init__(self, args, key_size, query_size, num_heads, embed_dim):
        super(MultiHeadHardAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = AdditiveAttention(args, embed_dim, embed_dim, embed_dim)
        self.w_q = init_(args, nn.Linear(query_size, embed_dim * num_heads))
        self.w_k = init_(args, nn.Linear(key_size, embed_dim * num_heads))

    def forward(self, queries, keys, mask):
        queries = transpose_task_input(self.w_q(queries), self.num_heads)
        keys = transpose_task_input(self.w_k(keys), self.num_heads)
        if mask is not None:
            mask = torch.repeat_interleave(mask, repeats=self.num_heads, dim=0)
        hard_attention_weights = self.attention(queries, keys, mask)

        return hard_attention_weights
