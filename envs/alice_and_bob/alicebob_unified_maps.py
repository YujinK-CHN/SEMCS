import random

STARTID = 2
MAX_AGENTS = 5
MAX_GOALS = 5
MAX_KEYS = 5

ENTITY_IDS = {
    "agent": list(range(STARTID, STARTID+MAX_AGENTS)),
    "goal": list(range(STARTID+MAX_AGENTS, STARTID+MAX_AGENTS+MAX_GOALS)),
    "key": list(range(STARTID+MAX_AGENTS+MAX_GOALS, STARTID+MAX_AGENTS+MAX_GOALS+MAX_KEYS))
}

ALL_IDS = ENTITY_IDS["agent"] + ENTITY_IDS["goal"] + ENTITY_IDS["key"]

def to_one_hot(id_value, all_ids):
    index = all_ids.index(id_value)
    one_hot = [0] * len(all_ids)
    one_hot[index] = 1    
    return one_hot

ENTITY_ONE_HOT_IDS = {
        "agent": {id_: to_one_hot(id_, ALL_IDS) for id_ in ENTITY_IDS["agent"]},
        "goal": {id_: to_one_hot(id_, ALL_IDS) for id_ in ENTITY_IDS["goal"]},
        "key":  {id_: to_one_hot(id_, ALL_IDS) for id_ in ENTITY_IDS["key"]}
    }

ID_TO_ONE_HOT = {
    id_: one_hot
    for category in ENTITY_ONE_HOT_IDS.values()
    for id_, one_hot in category.items()
}

ENTITY_IDS_TASKS = {
    "222-0": {"agent": [2, 3], "goal": [7, 8], "key": [12, 13]},
    "222-1": {"agent": [2, 3], "goal": [8, 9], "key": [13, 14]},
    "222-2": {"agent": [2, 3], "goal": [7, 9], "key": [12, 14]},
    "233-0": {"agent": [2, 3], "goal": [7, 8, 9], "key": [12, 13, 14]},
    "233-1": {"agent": [2, 3], "goal": [7, 8, 11], "key": [12, 13, 15]},
    "233-2": {"agent": [2, 3], "goal": [7, 9, 11], "key": [12, 14, 15]},
    "233-3": {"agent": [2, 3], "goal": [8, 9, 11], "key": [13, 14, 15]},
    "244-0": {"agent": [2, 3], "goal": [7, 8, 9, 10], "key": [12, 13, 14, 15]},
    "244-1": {"agent": [2, 3], "goal": [7, 8, 9, 11], "key": [12, 13, 14, 16]},
    "244-2": {"agent": [2, 3], "goal": [7, 8, 10, 11], "key": [12, 13, 15, 16]},
    "244-3": {"agent": [2, 3], "goal": [8, 9, 10, 11], "key": [13, 14, 15, 16]},
    
    "333-0": {"agent": [2, 3, 4], "goal": [7, 8, 9], "key": [12, 13, 14]},
    "333-1": {"agent": [2, 3, 5], "goal": [7, 8, 11], "key": [12, 13, 15]},
    "333-2": {"agent": [3, 4, 5], "goal": [7, 9, 11], "key": [12, 14, 15]},
    "333-3": {"agent": [2, 4, 5], "goal": [8, 9, 11], "key": [13, 14, 15]},
    "344-0": {"agent": [2, 3, 4], "goal": [7, 8, 9, 10], "key": [12, 13, 14, 15]},
    "344-1": {"agent": [3, 4, 6], "goal": [7, 8, 9, 11], "key": [12, 13, 14, 16]},
    "344-2": {"agent": [4, 5, 6], "goal": [7, 8, 10, 11], "key": [12, 13, 15, 16]},
    "344-3": {"agent": [2, 5, 6], "goal": [8, 9, 10, 11], "key": [13, 14, 15, 16]},

    "444-0": {"agent": [2, 3, 4, 5], "goal": [7, 8, 9, 10], "key": [12, 13, 14, 15]},
    "444-1": {"agent": [3, 4, 5, 6], "goal": [7, 8, 9, 11], "key": [12, 13, 14, 16]},
    "444-2": {"agent": [2, 4, 5, 6], "goal": [7, 8, 10, 11], "key": [12, 13, 15, 16]},
    "444-3": {"agent": [2, 3, 5, 6], "goal": [8, 9, 10, 11], "key": [13, 14, 15, 16]}
}

def get_task_id_map(n_agents, n_goals, n_keys):
    map_dict = {
        "agent": [ENTITY_IDS["agent"][i] for i in range(n_agents)],
        "goal": [ENTITY_IDS["goal"][i] for i in range(n_goals)],
        "key": [ENTITY_IDS["key"][i] for i in range(n_keys)],
    }

    return map_dict 

def get_task_one_hot_map(map_dict):
    one_hot_map = {
        "agent": [ENTITY_ONE_HOT_IDS["agent"][i] for i in map_dict["agent"]],
        "goal": [ENTITY_ONE_HOT_IDS["goal"][i] for i in map_dict["goal"]],
        "key": [ENTITY_ONE_HOT_IDS["key"][i] for i in map_dict["key"]]
    }
    return one_hot_map 
 

def get_alicebob_params(map_name): 
    map_param_registry = {
        "222": {
            "n_agents": 2,
            "n_keys": 2,
            "n_goals": 2,
            "limit": 50,
        },
        "233": {
            "n_agents": 2,
            "n_keys": 3,
            "n_goals": 3,
            "limit": 50,
        },
        "244": {
            "n_agents": 2,
            "n_keys": 4,
            "n_goals": 4,
            "limit": 50,
        },
        "333": {
            "n_agents": 3,
            "n_keys": 3,
            "n_goals": 3,
            "limit": 50,
        },
        "344": {
            "n_agents": 3,
            "n_keys": 4,
            "n_goals": 4,
            "limit": 50,
        },
        "355": {
            "n_agents": 3,
            "n_keys": 5,
            "n_goals": 5,
            "limit": 50,
        },
        "433": {
            "n_agents": 4,
            "n_keys": 3,
            "n_goals": 3,
            "limit": 50,
        },
        "444": {
            "n_agents": 4,
            "n_keys": 4,
            "n_goals": 4,
            "limit": 50,
        },
        "455": {
            "n_agents": 4,
            "n_keys": 5,
            "n_goals": 5,
            "limit": 50,
        },
    }
    map_name = map_name.split("-")[0]
    return map_param_registry[map_name]


def get_map_id_list(agent_list, goal_list, key_list, id_mode="Fixed", seed=None, id_list=None):
    """
    id_mode: 
        Fixed: choose ids based on the default orders
        Shuffle: randomly choose a certain number of ids
        Given: use provided ids
    """
    map_id_list = []

    if seed is not None:
        random.seed(seed)

    # Shuffle if mode is not fixed
    if id_mode == "Provided":
        one_hot_map_list = []
        for ids in id_list:
            one_hot_map_list.append([to_one_hot(id_, ALL_IDS) for id_ in ids])
        return list(zip(id_list, one_hot_map_list))
    else:
        if id_mode == "Shuffle":
            # shuffle first
            random.shuffle(ENTITY_IDS["agent"])
            random.shuffle(ENTITY_IDS["goal"])
            random.shuffle(ENTITY_IDS["key"])
        for n_agent, n_goal, n_key in zip(agent_list, goal_list, key_list):
            map_dict = get_task_id_map(n_agent, n_goal, n_key)
            one_hot_map = get_task_one_hot_map(map_dict)
            map_id_list.append([map_dict, one_hot_map])
        return map_id_list

def get_map_id(n_agent, n_goal, n_key, id_mode="Fixed", seed=None, task_id=None):
    if seed is not None:
        random.seed(seed)
        
    if id_mode == "Shuffle":
        random.shuffle(ENTITY_IDS["agent"])
        random.shuffle(ENTITY_IDS["goal"])
        random.shuffle(ENTITY_IDS["key"])
        map_dict = get_task_id_map(n_agent, n_goal, n_key)
        one_hot_map = get_task_one_hot_map(map_dict)
    elif id_mode == "Provided" and task_id is not None and task_id in ENTITY_IDS_TASKS.keys():
        map_dict = ENTITY_IDS_TASKS[task_id]
        one_hot_map = get_task_one_hot_map(map_dict)
            
    return map_dict, one_hot_map
