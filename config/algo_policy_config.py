

"""
Parameters for policy-based algorithms
"""

def get_mcs_config(parser, env_name):
    # use single state
    parser.add_argument("--use_single_state", type=bool, default=False, help="Use one state for all agents")
    # for football and alicebob
    parser.add_argument("--use_unified_obs", type=int, default=1, help="Determine whether use entity-based observation or not")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # actor/critic obs format: 1 = entity obs (attention), 0 = flat obs (MLP)
    parser.add_argument("--use_entity_actor", type=int, default=1)

    # pearl parameters
    parser.add_argument("--use_pearl", type=int,
                        default=False,
                        help="by default False, Training Transformer agent on multi envs with pearl, Only effective when --multi_envs!=None")
    parser.add_argument('--kl_lambda', type=float,
                        default=1, help="by default False, coefficient of kl loss")
    parser.add_argument("--recent_context", type=int,
                        default=False, help="by default False, sample context from enc_buffer's recent transitions or not")
    parser.add_argument("--extra_rl_posterior", type=int,
                        default=False, help="by default False, sample transitions without insert them into enc_buffer or not after prior and posterior sample")
    parser.add_argument("--context_num_mini_batch", type=int,
                        default=10, help="by default 10, context's num_mini_batch for infer posterior of z")
    parser.add_argument("--context_map_ohid", type=int, 
                        default=False, 
                        help="by default False, whether use map-one-hot-id or not in replace of observation/reward/action as context when infers posterior")
    parser.add_argument("--context_enc_frozen", type=int,
                        default=False, 
                        help="by default False, whether froze context encoder or not. Note: This parameter can be set as True when transfer testing whether context encoder has learned hidden variables of not")
    parser.add_argument("--use_reconstruct_loss", type=int,
                        default=False, help="by default False, whether use reconstruct loss when optimize context_encoder")
    parser.add_argument("--use_actor_loss", type=int,
                        default=True, help="by defalut True, whether use actor loss when optimize context_encoder")
    # environment parameters
    parser.add_argument("--n_agents", type=int, default=0)
    parser.add_argument("--n_enemies", type=int, default=0)

    # vea 
    parser.add_argument("--use_vae", type=int,
                        default=False, help="by default False, use vae or not")
    parser.add_argument("--skill_kl_loss", type=int, default=False, help="whether use the skill consistent loss for training skills in VAE")
    parser.add_argument("--coef_kl_loss", type=float, default=0.001, help=" coefficience of huber loss.")

    return parser
    

def get_DT2GS_config(parser, env_name):
    # use single state
    parser.add_argument("--use_single_state", type=bool, default=False, help="Use one state for all agents")
    # for football and alicebob
    parser.add_argument("--use_unified_obs", type=int, default=1, help="Determine whether use entity-based observation or not")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # transformer parameters
    # 默认以下5项全部为False, 记得修改
    parser.add_argument("--transformer_actor", type=int,
                        default=False, help="by default True, whether use transformer actor")                        
    parser.add_argument("--transformer_critic", type=int,
                        default=False, help="by default True, whether use transformer critic")
    parser.add_argument("--transformer_dropout", type=float, 
                        default=0.0, help="by default 0.0, dropout of transformerblock")
    parser.add_argument("--latent_dim", type=int,
                        default=8, help="by default 8, latent dim of normal distribution's mean and var")
    
    # pearl parameters
    parser.add_argument("--use_pearl", type=int,
                        default=True,
                        help="by default True, Training Transformer agent on multi envs with pearl, Only effective when --multi_envs!=None")
    parser.add_argument('--kl_lambda', type=float,
                        default=1, help="by default False, coefficient of kl loss")
    parser.add_argument("--recent_context", type=int,
                        default=False, help="by default False, sample context from enc_buffer's recent transitions or not")
    parser.add_argument("--extra_rl_posterior", type=int,
                        default=False, help="by default False, sample transitions without insert them into enc_buffer or not after prior and posterior sample")
    parser.add_argument("--context_num_mini_batch", type=int,
                        default=10, help="by default 10, context's num_mini_batch for infer posterior of z")
    parser.add_argument("--context_map_ohid", type=int, 
                        default=False, 
                        help="by default False, whether use map-one-hot-id or not in replace of observation/reward/action as context when infers posterior")
    parser.add_argument("--context_enc_frozen", type=int,
                        default=False, 
                        help="by default False, whether froze context encoder or not. Note: This parameter can be set as True when transfer testing whether context encoder has learned hidden variables of not")
    parser.add_argument("--transfer_only_context_encoder", type=int,
                        default=False, help="by default False, whether just restore context encoder or not when transfer from pearl mappo")
    parser.add_argument("--use_reconstruct_loss", type=int,
                        default=True, help="by default True, whether use reconstruct loss when optimize context_encoder")
    parser.add_argument("--use_actor_loss", type=int,
                        default=False, help="by defalut False, whether use actor loss when optimize context_encoder")
    # environment parameters
    parser.add_argument("--n_agents", type=int, default=0)
    parser.add_argument("--n_enemies", type=int, default=0)

    # vea
    parser.add_argument("--use_vae", type=int,
                        default=False, help="by default False, use vae or not")
    parser.add_argument("--skill_kl_loss", type=int, default=False, help="whether use the skill consistent loss for training skills in VAE")
    parser.add_argument("--coef_kl_loss", type=float, default=0.001, help=" coefficience of huber loss.")

    # subtask
    parser.add_argument("--num_subtask", type=int,
                        default=4, help="the number of subtask")
    parser.add_argument("--subtask_to_rnn", type=int, 
                        default=True, help="whether use subtask-observation or not in GRU")
    parser.add_argument("--subtask_pi_choice", type=str, 
                        default="adaptive_semantics", help="the Permutation Invariant way of integrate entity cognition into env cognition")
    parser.add_argument("--subtask_kl_loss", type=int, 
                        default=False, help="whether use the subtasks consistent loss")
    
    return parser


def get_hmasd_config(parser, env_name):
    # use single state
    parser.add_argument("--use_single_state", type=bool, default=False, help="Use one state for all agents")
    # for football and alicebob
    parser.add_argument("--use_unified_obs", type=int, default=0, help="Determine whether use entity-based observation or not")
    
    ###### for skill learning
    parser.add_argument("--intri_rew_exp", type=int, default=0) # change this 1 -> 0
    parser.add_argument("--skill_last_layer", type=int, default=0)
    parser.add_argument("--skill_interval", type=int, default=5)
    parser.add_argument("--team_skill_dim", type=int, default=3)
    parser.add_argument("--indi_skill_dim", type=int, default=3)
    parser.add_argument("--use_recurrent_discri", type=int, default=0)
    parser.add_argument("--use_linear_lambda_decay", type=int, default=0)
    parser.add_argument("--d_epoch", type=int, default=15)
    parser.add_argument("--d_num_mini_batch", type=int, default=1)
    parser.add_argument("--d_max_grad_norm", type=float, default=10.0)
    parser.add_argument("--d_use_max_grad_norm", action='store_false', default=True)
    parser.add_argument("--policy_use_both_skill", type=int, default=0)
    parser.add_argument("--l_reward_mix_type", type=int, default=0)
    parser.add_argument("--h_entropy_coef_start", type=float, default=0.5)
    parser.add_argument("--h_entropy_coef_end", type=float, default=0.1)
    parser.add_argument("--h_entropy_coef_decay", type=int, default=1)

    parser.add_argument("--low_train_ratio", type=float, default=0.4)
    parser.add_argument("--low_skill_interval", type=int, default=50)
    parser.add_argument("--lambda_env", type=float, default=1) # change this: 1-> 10
    parser.add_argument("--lambda_team", type=float, default=0.1)  # change this: 1-> 0.1
    parser.add_argument("--lambda_indi", type=float, default=0.5)  # change this: 1-> 0.5
    parser.add_argument("--low_full_train", type=int, default=0)

    ###### for optimizer parameters
    # high-level policy
    parser.add_argument("--h_lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--h_critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--h_opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--h_weight_decay", type=float, default=0)
    # low-level policy
    parser.add_argument("--l_lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--l_critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--l_opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--l_weight_decay", type=float, default=0)
    # discriminator
    parser.add_argument("--d_team_lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--d_indi_lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--d_opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--d_weight_decay", type=float, default=0) 

    # ppo parameters
    # high-level policy
    parser.add_argument("--h_ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--h_use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--h_clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--h_num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--h_entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--h_value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--h_use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--h_max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--h_use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--h_gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--h_gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--h_use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--h_use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--h_use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--h_use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--h_huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    # low-level policy
    parser.add_argument("--l_ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--l_use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--l_clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--l_num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--l_entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--l_value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--l_use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--l_max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--l_use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--l_gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--l_gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--l_use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--l_use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--l_use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--l_use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--l_huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # for actors
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    return parser

def get_ppo_config(parser, env_name):
    # use single state
    parser.add_argument("--use_single_state", type=bool, default=False, help="Use one state for all agents")
    # for football and alicebob
    parser.add_argument("--use_unified_obs", type=int, default=0, help="Determine whether use entity-based observation or not")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-6,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-3,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--std_x_coef", type=float, default=1)
    parser.add_argument("--std_y_coef", type=float, default=0.5)
    parser.add_argument("--lr_decay", type=float, default=1)
    parser.add_argument("--ob_n_actions", type=int, default=1000)
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float, default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_true',
                        default=False, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.97,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    
    return parser


def get_SESiL_config(parser, env_name):
    parser.add_argument("--use_single_state", type=bool, default=False)
    parser.add_argument("--use_unified_obs", type=int, default=0)

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo
    parser.add_argument("--ppo_epoch", type=int, default=15)
    parser.add_argument("--use_clipped_value_loss", action='store_false', default=True)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=1)
    parser.add_argument("--use_max_grad_norm", action='store_false', default=True)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--use_gae", action='store_false', default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--use_proper_time_limits", action='store_true', default=False)
    parser.add_argument("--use_huber_loss", action='store_false', default=True)
    parser.add_argument("--use_value_active_masks", action='store_false', default=True)
    parser.add_argument("--use_policy_active_masks", action='store_false', default=True)
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # actor/critic obs format: 1 = entity obs (attention), 0 = flat obs (MLP)
    parser.add_argument("--use_entity_actor", type=int, default=1)

    # evolution parameters
    parser.add_argument("--evo_solver_algo", type=str, default="mappo", choices=["mappo", "mcs"], help="base algorithm for each solver: mappo (plain) or mcs (skills+comm+pred)")
    parser.add_argument("--evo_num_solvers", type=int, default=8, help="number of solvers in population")
    parser.add_argument("--evo_tasks_per_solver", type=int, default=3, help="number of tasks randomly assigned to each solver")
    parser.add_argument("--evo_num_generations", type=int, default=8, help="number of evolutionary generations")
    parser.add_argument("--evo_pretrain_budget", type=int, default=2000000, help="extra steps for pretraining the initial population before evolution (0=no pretraining)")
    parser.add_argument("--evo_gen_budget", type=int, default=1000000, help="fixed training budget per generation in env steps (0=auto: split remaining budget evenly)")
    parser.add_argument("--evo_keep_population", type=int, default=1, help="1: each pair produces 2 offspring (population size preserved), 0: shrink (1 offspring per pair)")
    parser.add_argument("--evo_eval_episodes", type=int, default=10, help="eval episodes per solver per task for fitness")
    parser.add_argument("--evo_threshold", type=float, default=1.0, help="min fitness (reward*win_rate) to consider a task known")
    parser.add_argument("--evo_weight_extra", type=float, default=0.9, help="weight for complementary skills in mating score")
    parser.add_argument("--evo_weight_common", type=float, default=0.1, help="weight for shared skills in mating score")

    # pearl (unused but needed for shared trainer compatibility)
    parser.add_argument("--use_pearl", type=int, default=False)
    parser.add_argument('--kl_lambda', type=float, default=1)
    parser.add_argument("--recent_context", type=int, default=False)
    parser.add_argument("--extra_rl_posterior", type=int, default=False)
    parser.add_argument("--context_num_mini_batch", type=int, default=10)
    parser.add_argument("--context_map_ohid", type=int, default=False)
    parser.add_argument("--context_enc_frozen", type=int, default=False)
    parser.add_argument("--use_reconstruct_loss", type=int, default=False)
    parser.add_argument("--use_actor_loss", type=int, default=True)
    parser.add_argument("--n_agents", type=int, default=0)
    parser.add_argument("--n_enemies", type=int, default=0)

    # vae (unused but needed for shared code paths)
    parser.add_argument("--use_vae", type=int, default=False)
    parser.add_argument("--skill_kl_loss", type=int, default=False)
    parser.add_argument("--coef_kl_loss", type=float, default=0.001)

    return parser


def get_mat_config(parser, env_name):
    # use single state
    parser.add_argument("--use_single_state", type=bool, default=False, help="Use one state for all agents")
    # for football and alicebob
    parser.add_argument("--use_unified_obs", type=int, default=0, help="Determine whether use entity-based observation or not")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 1e-3)')
    parser.add_argument("--comm_lr", type=float, default=5e-4,
                        help='learning rate for communication (default: 1e-3)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 1e-2)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-3)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--ob_n_actions", type=int, default=1000)
    parser.add_argument("--lr_decay", type=float, default=1)


    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=10,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    
    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    return parser