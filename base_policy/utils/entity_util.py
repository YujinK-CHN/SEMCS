import torch
import numpy as np

def encode_entity(args, obs, n_agents, n_entites, feat_dim):
    '''
    obs: a list of ob, in which ob's shape is (bs_na, obs_dim)
    @override: incorporates past skills
    '''
    bs_na_list = [ob.size(0) for ob in obs]
     
    # 4 for move features
    if "StarCraft" in args.env_name:
        if args.skill_to_obs == "merge":
            split_lists = [[n_entity*feat_dim, 4, args.num_skills] for n_entity in n_entites]
            past_skill_list = [ob[:, -args.num_skills:] for ob in obs]
        elif args.skill_to_obs == "entity":
            split_lists = [[n_entity*feat_dim, 4, args.num_skills*n_entity] for n_entity in n_entites]
            past_skill_list = [ob[:, -args.num_skills*n_entity:].view(-1, n_entity, args.num_skills) for ob, n_entity in zip(obs, n_entites)]
        else:
            split_lists = [[n_entity*feat_dim, 4] for n_entity in n_entites]
            past_skill_list = [None for _ in n_agents]
        
        entity_ob_list = [torch.split(ob, split_list, dim=-1)[0].contiguous().view(bs_na, n_entity, feat_dim) \
            for ob, split_list, bs_na, n_entity in \
                zip(obs, split_lists, bs_na_list, n_entites)]
    elif "AliceBob" in args.env_name or "Football" in args.env_name:
        if args.skill_to_obs == "merge":
            split_lists = [[n_entity*feat_dim, args.num_skills] for n_entity in n_entites]
            past_skill_list = [ob[:, -args.num_skills:] for ob in obs]
        elif args.skill_to_obs == "entity":
            split_lists = [[n_entity*feat_dim, args.num_skills*n_entity] for n_entity in n_entites]
            past_skill_list = [ob[:, -args.num_skills*n_entity:].view(-1, n_entity, args.num_skills) for ob, n_entity in zip(obs, n_entites)]
        else:
            split_lists = [[n_entity*feat_dim] for n_entity in n_entites]
            past_skill_list = [None for _ in n_agents]
        
        entity_ob_list = [torch.split(ob, split_list, dim=-1)[0].contiguous().view(bs_na, n_entity, feat_dim) \
            for ob, split_list, bs_na, n_entity in \
                zip(obs, split_lists, bs_na_list, n_entites)]        
        
    return entity_ob_list, past_skill_list, bs_na_list


def decode_entity(entity_obs, n_agents, n_entites, bs_na_list, feat_dim):
    '''
    obs is a tensor which shape is (-1, feat_dim)
    '''
    split_list = [bs_na * n_entity for bs_na, n_entity in zip(bs_na_list, n_entites)]
    entity_ob_list = torch.split(entity_obs, split_list, dim=0)
    entity_ob_list = [entity_ob.contiguous().view(bs_na, n_entity, feat_dim) \
            for entity_ob, bs_na, n_entity in zip(entity_ob_list, bs_na_list, n_entites)]
    return entity_ob_list


def decode_skill(args, skills, n_agents, n_entites, bs_na_list, feat_dim):
    '''
    obs is a tensor which shape is (-1, feat_dim)
    '''
    if "All" == args.comm_channel:
        split_list = [bs_na * n_agent for bs_na, n_agent in zip(bs_na_list, n_agents)]
        skills = torch.split(skills, split_list, dim=0)
        skills_list = [entity_ob.contiguous().view(bs_na, n_agent, feat_dim) \
                for entity_ob, bs_na, n_agent in zip(skills, bs_na_list, n_agents)]
    elif "ExcludeSelf" == args.comm_channel:
        split_list = [bs_na * (n_agent -1) for bs_na, n_agent in zip(bs_na_list, n_agents)]
        skills = torch.split(skills, split_list, dim=0)
        skills_list = [entity_ob.contiguous().view(bs_na, (n_agent-1), feat_dim) \
                for entity_ob, bs_na, n_agent in zip(skills, bs_na_list, n_agents)]
    else:
        split_list = [bs_na * n_entity for bs_na, n_entity in zip(bs_na_list, n_entites)]
        skills = torch.split(skills, split_list, dim=0)
        skills_list = [entity_ob.contiguous().view(bs_na, n_entity, feat_dim) \
                for entity_ob, bs_na, n_entity in zip(skills, bs_na_list, n_entites)]
    return skills_list
