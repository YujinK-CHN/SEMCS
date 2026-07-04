
def get_common_config(parser):
    """
    The configuration parser for common hyper-parameters of all environments and algirothms.
    prepare parameters for settings, environments, algorithms.
    """
    parser.add_argument("--project_name", type=str, default="check", help="an identifier to distinguish different projects.")
    parser.add_argument("--algorithm_name", type=str, default='mcs', choices=["mcs", "dt2gs", "sft", "joint", "sesil", "hmasd", "ippo", "mappo", "mat", "tgcnet", "maic", "scvd"])
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='mcs',help="[for wandb usage], to specify user's name for simply collecting training data.")

    """
    env parameters
    """
    parser.add_argument("--env_name", type=str, default='StarCraft', choices=["AliceBob", "StarCraft", "Football"], help="specify the name of environment")
    parser.add_argument("--use_unified_env", type=int, default=False, help="by default True, whether use the unified smac env.")

    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--use_single_state", type=bool, default=False, help="Not copied states for all agents")
    parser.add_argument("--num_agents", type=int, default=0, help="set the number of agents which is useful for policy, buffer, and training")
    parser.add_argument("--num_enemies", type=int, default=0, help="set the number of agents which is useful for policy, buffer, and training only for SMAC")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", type=int, default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false', default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # for transformer, which can be shared across different methods
    parser.add_argument("--use_orth", type=int, default=True, help="by default True, whether use transformer orthogonal")
    parser.add_argument("--n_head", type=int, default=3, help="by default True, heads of multi-head attention in transformer")
    parser.add_argument("--n_block", type=int, default=1, help="by default 1, number of transformerblock")
    parser.add_argument("--n_embd", type=int, default=64, help="this is the parameters used in Transformers for generating skills")
    parser.add_argument("--shared_transformer", type=int, default=0, help="by default 0, whether use shared transformer for actor and critic")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a store_true recurrent policy')
    parser.add_argument("--use_recurrent_with_agents", type=int, default=False, help='Whether to sample agents together')
    parser.add_argument("--use_recurrent_policy", type=int, default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=100, help="time duration between contiunous twice models saving.")
    parser.add_argument('--run_dir', type=str, default='.', help="Which map to eval on")
    parser.add_argument('--replay_dir', type=str, default='.', help="Store replay to which folder, this need to be absolute path.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=20, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_false', default=True, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # parameters that can be extended
    parser.add_argument("--atrous_attention", type=int,
                        default=0, help="by default 0, whether use AtrousSelfAttention in transformer")
    parser.add_argument("--atrous_rate", type=int,
                        default=4, help="by default 2, atrous K for atrous attention")

    return parser


def get_alicebob_config(parser):
    """
    private hyper-parameters only used in alicebob
    """
    parser.add_argument("--use_unified_obs", type=int, default=1, help="Determine whether use entity-based observation or not")
    parser.add_argument("--id_mode", type=str, default='Provided', choices=["Fixed", "Shuffle", "Provided"], help="specify how ids are generated")
    parser.add_argument("--goal_key_locations", type=str, default='Shuffle', choices=["Fixed", "Shuffle"], help="specify how ids are generated")
    parser.add_argument("--vision", type=int, default=1, help="the observation range of agent.")
    parser.add_argument("--use_sparse_reward", type=int, default=1, help="decide if use bonus or not, 1: not use bonus, 0: use bonus")
    parser.add_argument("--use_grids_obs", type=int, default=1, help="decide if use grids surrounding to the agent")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=50, help="Max length for any episode")

    # communicaiton
    parser.add_argument("--single_comm", type=int, default=False, help="whether to communicate one another player in maximum")

    # during evaluation
    parser.add_argument("--eval_deterministic", type=int, default=False, help="by default True. If False, sample actions/messages according to probability")

    # parameters for entities input
    parser.add_argument("--actor_feat_dim", type=int, default=17, help="by default 17, input dim of actors")
    parser.add_argument("--critic_feat_dim_low", type=int, default=17, help="by default 17, input dim of critics")
    parser.add_argument("--critic_feat_dim_high", type=int, default=17, help="by default 17, input dim of critics")
    parser.add_argument("--input_feat_dim", type=int, default=17, help="by default 17, input dim of Q-functions")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video; only used for football.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    return parser



def get_smac_config(parser):
    """
    private hyper-parameters only used in SMAC
    """    
    # env parameters for unified environment
    parser.add_argument("--entity_feature", type=int, default=True, help="by default True, whether split obs/state as entity features")
    parser.add_argument("--use_map_ohid", type=int, 
                    default=False, help="by default False, whether use map one hot id or not")
    # parameters for entities input
    parser.add_argument("--actor_feat_dim", type=int, default=16, help="by default 16, input dim of actors")
    parser.add_argument("--critic_feat_dim_low", type=int, default=16, help="by default 16, input dim of critics")
    parser.add_argument("--critic_feat_dim_high", type=int, default=20, help="by default 20, input dim of critics")
    parser.add_argument("--input_feat_dim", type=int, default=16, help="by default 16, input dim of Q-functions")

    # communicaiton
    parser.add_argument("--single_comm", type=int, default=True, help="whether to communicate one another player in maximum")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=400, help="Max length for any episode")

    # during evaluation
    parser.add_argument("--eval_deterministic", type=int, default=True, help="by default True. If False, sample actions/messages according to probability")

    # common parameters for StarCraft
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--random_agent_order", action='store_true', default=False)
    parser.add_argument("--state_agent_specific", type=int, default=0)
    parser.add_argument("--use_sparse_reward", type=int, default=0)
    
    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video; only used for football.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    return parser



def get_football_config(parser):
    parser.add_argument("--representation", type=str, default="simple115v2", 
                        choices=["simple115v2", "extracted", "pixels_gray", 
                                 "pixels"],
                        help="representation used to build the observation.")
    parser.add_argument("--rewards", type=str, default="scoring", 
                        help="comma separated list of rewards to be added.")
    parser.add_argument("--smm_width", type=int, default=96,
                        help="width of super minimap.")
    parser.add_argument("--smm_height", type=int, default=72,
                        help="height of super minimap.")
    parser.add_argument("--remove_redundancy", action="store_true", 
                        default=False, 
                        help="by default False. If True, remove redundancy features")
    parser.add_argument("--zero_feature", action="store_true", 
                        default=False, 
                        help="by default False. If True, replace -1 by 0")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")
    parser.add_argument("--use_unified_obs", type=int, default=1, help="Determine whether use entity-based observation or not")
    parser.add_argument("--unifed_obs_dim", type=int, default=172, help="Determine the unified entity feature dimension for each environment")

    # during evaluation
    parser.add_argument("--eval_deterministic", type=int, default=True, help="by default True. If False, sample actions/messages according to probability")

    # save replay
    parser.add_argument("--save_videos", action="store_true", default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="", 
                        help="directory to save videos.")

    # communicaiton
    parser.add_argument("--single_comm", type=int, default=True, help="whether to communicate one another player in maximum")

    # params also used by others
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--episode_length", type=int, default=200, help="Max length for any episode")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--use_sparse_reward", type=int, default=0, help="whether use sparse reward or not")

    parser.add_argument("--actor_feat_dim", type=int, default=43, help="by default 16, input dim of actors")
    parser.add_argument("--critic_feat_dim_low", type=int, default=43, help="by default 17, input dim of critics")
    parser.add_argument("--critic_feat_dim_high", type=int, default=43, help="by default 17, input dim of critics")
    parser.add_argument("--input_feat_dim", type=int, default=43, help="by default 17, input dim of Q-functions")

    parser.add_argument("--gain", type=float, default=1, help="The gain # of last action layer")

    return parser
