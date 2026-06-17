

map_param_registry = {
    "academy_3_vs_1_with_keeper": {
        "n_agents": 3,
        "limit": 200,
        "n_entity": 4
    },
    "academy_counterattack_easy": {
        "n_agents": 4,
        "limit": 200,
        "n_entity": 4,
    },
    "academy_counterattack_hard": {
        "n_agents": 4,
        "limit": 1000,
    },
    "academy_corner": {
        "n_agents": 10,
        "limit": 1000,
        "n_entity": 4,
    },
    "academy_pass_and_shoot_with_keeper": {
        "n_agents": 2,
        "limit": 200,
        "n_entity": 4
    },
    "academy_run_pass_and_shoot_with_keeper": {
        "n_agents": 2,
        "limit": 200,
        "n_entity": 4
    }
}


def get_map_registry():
    return map_param_registry


def get_football_params(map_name):
    map_param_registry = get_map_registry()
    return map_param_registry[map_name]
