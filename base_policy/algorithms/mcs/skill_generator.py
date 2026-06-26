import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

from base_policy.utils.util import init_, init_gru, check, pearson_loss, print_cuda
from base_policy.components.transformers import EncoderBlock, Mod_Sequential
from itertools import combinations
from base_policy.utils.entity_util import decode_entity

class VAESkillGenerator(nn.Module):
    def __init__(self, args, input_dim, output_dim, device) -> None:
        super(VAESkillGenerator, self).__init__()
        
        self.args = args
        if args.use_obs_instead_of_state:
            self.critic_feat_dim = args.critic_feat_dim_low
        else:
            self.critic_feat_dim = args.critic_feat_dim_high
        self.actor_feat_dim = args.actor_feat_dim
        
        self.hidden_size = args.hidden_size
        self.num_skills = output_dim
        self.encoder = nn.Sequential(
            init_(args, nn.Linear(input_dim, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, 2*output_dim))
        )

        self.decoder = nn.Sequential(
            init_(args, nn.Linear(output_dim, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            init_(args, nn.Linear(self.hidden_size, input_dim))
        )

        self.tpdv = dict(dtype=torch.float32, device=device)

    def rp_sample(self, mus, sigmas):
        '''
        Reparameter sampling
        z ~ N(mu, sigma) --> (z-mu)/sqrt(sigma) ~ N(0, 1)
        '''
        epsilon = torch.distributions.Normal(
            check(torch.zeros(self.num_skills), self.tpdv), check(torch.ones(self.num_skills), self.tpdv)
        ).sample(mus.shape[:-1])
        
        # reparametrization like VAE
        z = mus + epsilon * torch.sqrt(sigmas)
        return z
    
    def forward(self, x_list, past_x_list, h_list, masks, active_masks, n_agents=None, n_entites=None, is_training=False, deterministic=False):
        '''
            x: (..., input_dim)
            TODO: use past_x and masks when producing skills
        '''
        params = [self.encoder(x_i) for x_i in x_list]
        # print("parmas:", [par.shape for par in params])
        z_list = []
        mu_list = []
        sigma_list = []
        for mus, log_sigmas in [torch.split(param, [self.num_skills, self.num_skills], dim=-1) for param in params]:
            sigmas = torch.exp(log_sigmas)
            z = self.rp_sample(mus, sigmas)
            z_list.append(z)
            mu_list.append(mus)
            sigma_list.append(sigmas)

        # use VAE loss for training
        if is_training:
            loss_kl, loss_re, x_list_re = self.vae_loss(x_list, z_list, mu_list, sigma_list)
        else:
            loss_kl, loss_re = None, None
            x_list_re = self.decode(z_list)
        
        train_info = {
            "vae_loss_re": loss_re,
            "vae_loss_kl": loss_kl,
            "x_list_re": x_list_re
        }
        return z_list, h_list, train_info

    def decode(self, z):
        '''
        z: (..., output_dim)
        '''
        x = [self.decoder(z_i) for z_i in z]
        return x

    def kl_loss(self, mus_list, sigmas_list):
        '''
        mus_: (..., self.output_dim)
        sigmas_: (..., self.output_dim)
        '''
        loss_kl_list = []
        for mus, sigmas in zip(mus_list, sigmas_list):
            priors = torch.distributions.Normal(
                check(torch.zeros_like(mus), self.tpdv), check(torch.ones_like(sigmas), self.tpdv)
            )
            posteriors = torch.distributions.Normal(mus, sigmas)
            # kl between Standard normal distribution and the current distribution
            loss_kl = self.args.coef_kl_loss * torch.sum(torch.distributions.kl.kl_divergence(posteriors, priors), dim=-1)
            loss_kl_list.append(loss_kl)
        return loss_kl_list

    def re_loss(self, x, z):
        '''
        x: (..., input_dim)
        z: (..., output_dim) for each subtask
        '''
        x_re = self.decode(z)
        # average reconstruction loss
        loss_re = [torch.sum((x_re_i - x_i) ** 2, dim=-1).mean() for x_re_i, x_i in zip(x_re, x)]
        return loss_re, x_re

    def vae_loss(self, x, z_list, mus_list, sigmas_list):
        loss_re, x_re = self.re_loss(x, z_list)
        if self.args.skill_kl_loss:
            loss_kl =  self.kl_loss(mus_list, sigmas_list)
        else:
            loss_kl = [torch.tensor(0.).to(**self.tpdv) for _ in loss_re]
        x_re = self.decode(z_list)
        return loss_kl, loss_re, x_re



class GRUSkillGenerator(nn.Module):
    def __init__(self, args, input_dim, output_dim, device) -> None:
        super(GRUSkillGenerator, self).__init__()

        self.num_skills = self.args.num_skills
        self.hidden_size = args.hidden_size

        self.fc1 = init_(args, nn.Linear(input_dim, self.hidden_size))
        
        if self.args.skill_to_obs != "None":
            self.rnn = init_(args, nn.GRU(self.hidden_size + self.num_skills, self.hidden_size, num_layers=self.args.recurrent_N))
        else:
            self.rnn = init_gru(args, nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.args.recurrent_N))
        # output layer for generating skills
        self.fc2 = init_(nn.Linear(self.hidden_size, output_dim))

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.all_skill_onehot = torch.eye(output_dim).to(**self.tpdv)


    def encoder(self, x, p_s, hxs, masks):
        '''
            # train/intreaction
            x: (chunk_length*bs*na, (na+ne), feat_dim) / (threads*na, (na+ne), feat_dim)
            p_s: (chunk_length*bs*na, subtask_dim) / (threads*na, subtask_dim)
            hxs: (bs*na, recurrent_N, rnn_hidden_dim) / (threads*na, recurrent_N, rnn_hidden_dim)
            masks: (chunk_length*bs*na, 1) / (threads*na, 1)
        '''
        [x_bs, _, _] = x.size()
        [hxs_bs, rnn_layers, _] = hxs.size()
        assert masks.size(0) == x_bs

        # fc1
        x = self.fc1(x).mean(dim=-2) # (x_bs, num_skills or feat_dim)
        if p_s is not None:
            x = torch.cat([x, p_s], dim=-1)

        # * Use RNN for encoding x 
        if x_bs == hxs_bs:
            # print(f"Interactive: x's 1st shape is {x_bs} | h's 1st shape is {hxs_bs}")
            masks_re = masks.unsqueeze(-2).repeat(1, rnn_layers, 1) # (x_bs, rnn_layers, 1)
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks_re).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, nae, feat_dim)
            # print(f"Training: x's 1st shape is {x_bs} | h's 1st shape is {hxs_bs}")
            N = hxs_bs
            T = int(x_bs / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]   # has_zeros = [0, 4, 7, ..., T]

            # recurrent layer
            hxs = hxs.transpose(0, 1) # (recurrent_N, N, rnn_hidden_dim)

            # * x[start_idx:end_idx] which will choose the input for corresponding time step
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(rnn_layers, 1, 1)).contiguous() # (rnn_layers, N, rnn_hidden_dim)
                # (end-start, N, num_skills) | (rnn_layers, N, rnn_hidden_dim)
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1) # (x_bs, num_skills)
            hxs = hxs.transpose(0, 1) # (N, rnn_layers, rnn_hidden_dim) actually: rnn_hidden_dim = num_skills

        # x = self.norm(x)
        return x, hxs


    def forward(self, x_list, past_x_list, h_list, masks, active_masks, n_agents=None, n_entites=None, is_training=False, deterministic=False):
        """
        @param x: this is the input embedding for all agents
        @param past_skills: this might be replaced by other agents' skills, which will be concatenated with the agent's own skills
        """
        # the shape of z in z_list is (bs_na, na+ne, fd)
        r_x_tuple, h_tuple = tuple(zip(*[self.encoder(x, p_s, h, mask) for x, p_s, h, mask in zip(x_list, past_x_list, h_list, masks)]))
        r_x_list = list(r_x_tuple) # (bs_na, na+ne, num_skills) --> (bs_na, num_skills)
        h_list = list(h_tuple) # (bs_na, hidden_size) --> (bs_na, hidden_size)

        # skill_list = [torch.softmax(self.perm_invar_fc2(r_x), dim=-1) for r_x in r_x_list]
        skill_list = [self.fc2(r_x) for r_x in r_x_list] # directly give num_skills

        if deterministic:
            skill_list = [self.all_skill_onehot[skill.argmax(dim=-1)] for skill in skill_list]
        else:
            skill_list = [self.all_skill_onehot[F.gumbel_softmax(skill, hard=True).argmax(dim=-1)] for skill in skill_list]
        
        train_info = None
        
        return skill_list, h_list, train_info

    

class TransformerSkillGenerator(nn.Module):
    def __init__(self, args, input_dim, output_dim, device, use_orthogonal) -> None:
        super(TransformerSkillGenerator, self).__init__()
        
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.n_block = args.n_block

        self.op_entity = args.op_entity  # the way to operate entity features, may use concate, mean or sum
        self.sim_metrics = args.sim_metrics

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
    
    
    def compute_similarity(self, x_list, skill_list, n_agents, n_entites):
        """
        compute the similarity between input observations and generated skills
        """
        x_size_list = [ob.size() for ob in x_list]
        skill_size_list = [skill.size() for skill in skill_list]      

        x_list = [x_i.view(-1, n_agent, n_entity, x_size[-1]) for x_i, n_agent, n_entity, x_size in zip(x_list, n_agents, n_entites, x_size_list)]
        skill_list = [s_i.view(-1, n_agent, s_size[-1]) for s_i, n_agent, s_size in zip(skill_list, n_agents, skill_size_list)]
        
        simm_loss = []
        x_similarity = []
        skill_similarity = []
        for i, j in combinations(range(len(x_list)), 2):  # unique pairs only                                
            if self.sim_metrics == "Cosin":
                # input
                x_i_flattened = x_list[i].view(x_list[i].shape[0], -1, x_list[i].shape[-1])  # x shape: [1200, 6, 16]
                x_j_flattened = x_list[j].view(x_list[j].shape[0], -1, x_list[j].shape[-1])
                # latent
                s_i_flattened = skill_list[i].view(skill_list[i].shape[0], -1, skill_list[i].shape[-1])
                s_j_flattened = skill_list[j].view(skill_list[j].shape[0], -1, skill_list[j].shape[-1])
                x_sim = torch.bmm(F.normalize(x_i_flattened, dim=-1), F.normalize(x_j_flattened, dim=-1).transpose(1, 2)) 
                skill_sim = torch.bmm(F.normalize(s_i_flattened, dim=-1), F.normalize(s_j_flattened, dim=-1).transpose(1, 2))  
                
                # print("Cosin:", x_sim.size(), skill_sim.size())   
                loss = F.mse_loss(x_sim.mean().detach(), skill_sim.mean())
            elif self.sim_metrics == "Covariance":
                # input
                x_i_flattened = x_list[i].view(x_list[i].shape[0], -1, x_list[i].shape[-1])  # x shape: [1200, 6, 16]
                x_j_flattened = x_list[j].view(x_list[j].shape[0], -1, x_list[j].shape[-1])
                # latent
                s_i_flattened = skill_list[i].view(skill_list[i].shape[0], -1, skill_list[i].shape[-1])
                s_j_flattened = skill_list[j].view(skill_list[j].shape[0], -1, skill_list[j].shape[-1])

                x_sim = torch.bmm(F.normalize(x_i_flattened, dim=-1), F.normalize(x_j_flattened, dim=-1).transpose(1, 2)) 
                skill_sim = torch.bmm(F.normalize(s_i_flattened, dim=-1), F.normalize(s_j_flattened, dim=-1).transpose(1, 2))  
                
                # averagre over entities                    
                x_sim = x_sim.view(-1, x_list[i].size()[1], x_list[i].size()[2], x_list[j].size()[2], x_list[j].size()[1])
                x_sim = x_sim.mean(dim=(2,3)) 
                skill_sim = skill_sim.view(-1, x_list[i].size()[1], x_list[j].size()[1])

                # print("Covariance:", x_sim.size(), skill_sim.size())
                loss = pearson_loss(x_sim.detach(), skill_sim)
            elif self.sim_metrics == "Contrastive":  # contrastive loss from Siming Lan et al. NeurIPS 2023
                pass
            simm_loss.append(loss)
            x_similarity.append(x_sim)
            skill_similarity.append(skill_sim)
        return simm_loss, x_similarity, skill_similarity


    def forward(self, x_list, past_x_list, h_list, masks, active_masks, n_agents=None, n_entites=None, is_training=False, deterministic=False):
        """
        @param z_list: where does it come from; it is from the embedding features
        @param past_skills: this might be replaced by other agents' skills, which will be concatenated with the agent's own skills
        """        
        x_size_list = [x_i.size() for x_i in x_list]
        bs_na_list = [x_i.size(0) for x_i in x_list]
        embd_x = [self.input_embedding(x_i) for x_i in x_list]
        
        if self.args.share_tblocks:   # use shared transformer blocks
            tb, x_dot = self.tblocks(embd_x)
            tb = decode_entity(tb, n_agents, n_entites, bs_na_list, self.n_embd*self.n_head)  
        else:   # use different transformer blocks
            tb, x_dot = self.tblocks.forward_per_task(embd_x) 

        tb = [tb_i.view(bs_na[0], bs_na[1], -1) for tb_i, bs_na in zip(tb, x_size_list)]

        ####### mean or sum over entities #######
        if self.args.skill_to_obs == "merge":
            if "Mean" in self.op_entity:
                tb = [tb_i.mean(-2) for tb_i in tb]
            elif "Sum" in self.op_entity:
                tb = [tb_i.sum(-2) for tb_i in tb]
        # otherwise the entity will be preserved
        ####### mean or sum over entities #######

        skill_list = [self.toprobs(tb_i) for tb_i in tb]

        # use gumbel_softmax to make it differentiable when skill is discrete
        if self.args.skill_type == "Discrete":
            if deterministic:
                skill_list = [self.all_skill_onehot[skill.argmax(dim=-1)] for skill in skill_list]
            else:
                # import pdb; pdb.set_trace()
                skill_list = [F.gumbel_softmax(skill, hard=True) for skill in skill_list]
                
        # "Cosin", "Wasserstein", "Covariance", "Contrastive", "None"
        if is_training and self.args.use_similarity:
            simm_loss, x_similarity, skill_similarity = self.compute_similarity(x_list, skill_list, n_agents, n_entites)
            train_info = {"simm_loss": simm_loss,
                          "obs_sim": x_similarity,
                          "skill_sim": skill_similarity}
        else:
            train_info = {}

        return skill_list, x_dot, tb, h_list, train_info

