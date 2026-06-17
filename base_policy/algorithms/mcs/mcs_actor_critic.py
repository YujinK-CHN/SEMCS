import torch
import torch.nn as nn
from torch.nn import functional as F
from base_policy.utils.util import init, check, print_cuda
from base_policy.components.rnn_entity import RNNEntityLayer
from base_policy.components.popart import PopArt


from base_policy.algorithms.mcs.integation_critic import IntegrationCritic


"""
Actor and Critic for agents
"""

class R_Actor(nn.Module):
    """
    Actor network class for HMASD. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        # obs_space: [[100, [2, 14], [3, 5], [1, 4], [1, 17]], [180, [4, 17], [6, 5], [1, 4], [1, 22]]]
        # action_space: [[Discrete(9), Discrete(9), Discrete(9)], [Discrete(12), Discrete(12), Discrete(12), Discrete(12), Discrete(12)]]
        super(R_Actor, self).__init__()
        self.args = args
        self.use_actor_loss = args.use_actor_loss

        self.use_action_predictor = args.use_action_predictor

        # used for entities
        self.actor_feat_dim = args.actor_feat_dim
        
        # used for embeddings
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            print("use rnn for actor.....")
            self.rnn = RNNEntityLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        else:
            print("do not use rnn for actor.....")

        # used to integrate skills and observation inputs into a unified embedding
        if self.args.skill_to_obs == "entity":
            from base_policy.algorithms.mcs.mcs_entity.entity_integation_comm import EntityIntegrationComm
            self.integration = EntityIntegrationComm(args, self.actor_feat_dim, self.hidden_size, args.use_orth, device)
        else:   # merge and None
            from base_policy.algorithms.mcs.mcs_merge.merge_integation_comm import MergeIntegrationComm
            self.integration = MergeIntegrationComm(args, self.actor_feat_dim, self.hidden_size, args.use_orth, device)

        # predictor for future behaviors
        if self.use_action_predictor in [1, 2]:
            from base_policy.algorithms.mcs.skill_auto_predictor import SkillAutoPredictor
            self.act_predictor = SkillAutoPredictor(args, action_space, args.use_orth, device)
        elif self.use_action_predictor in [3]:
            from base_policy.algorithms.mcs.skill_all_predictor import SkillAllPredictor
            self.act_predictor = SkillAllPredictor(args, self.actor_feat_dim, self.hidden_size, action_space, args.use_orth, device)

        # action layers for domains
        if "StarCraft" in args.env_name:
            from base_policy.components.act_entity import EntityVAEACTLayer
            self.act_layer = EntityVAEACTLayer(args, action_space, self.hidden_size, self._use_orthogonal, self._gain)
        elif "AliceBob" in args.env_name:
            from base_policy.components.act_multi import MultiACTLayer
            self.act_layer = MultiACTLayer(args, action_space[0][0], self.hidden_size, self._use_orthogonal, self._gain)
        elif "Football" in args.env_name:
            from base_policy.components.act_multi import MultiACTLayer
            self.act_layer = MultiACTLayer(args, action_space[0][0], self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, rnn_states_comm, masks, active_masks, available_actions=None, deterministic=False, n_agents=None, n_enemies=None, n_entites=None, is_training=False):     
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        rnn_states_comm = check(rnn_states_comm, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)
        
        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)
        
        actor_feature_list, _, _, comm_skill_list, _, entity_ob_list, rnn_states_comm, train_info, record_info = self.integration(obs, rnn_states_comm, masks, active_masks,
                                                                                n_agents, n_enemies, n_entites, is_training, deterministic=deterministic)

        # use RNN to capture time-relevant features
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_feature_list, rnn_states = self.rnn(actor_feature_list, rnn_states, masks)

        actions, action_log_probs, pi_probs = self.act_layer.forward4skills(actor_feature_list, available_actions, deterministic, n_enemies)
        
        return actions, action_log_probs, pi_probs, rnn_states, rnn_states_comm, entity_ob_list, comm_skill_list, train_info, record_info


    def evaluate_actions(self, obs, rnn_states, rnn_states_comm, action, masks, active_masks, available_actions=None, n_agents=None, n_enemies=None, n_entites=None, is_training=True, future_available_actions=None):
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        rnn_states_comm = check(rnn_states_comm, self.tpdv)
        action = check(action, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)

        actor_feature_list, skill_list, _, comm_skill_list, _, _, rnn_states_comm, train_info, _ = self.integration(obs, rnn_states_comm, masks, active_masks, n_agents, n_enemies, n_entites, is_training)

        # use RNN to capture time-relevant features
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_feature_list, rnn_states = self.rnn(actor_feature_list, rnn_states, masks)

        action_log_probs, dist_entropy = self.act_layer.evaluate_actions(actor_feature_list, action, available_actions,
                                                                   active_masks=active_masks if self._use_policy_active_masks else None,
                                                                   n_enemies=n_enemies)

        """
        1. from sender's perspective, predict their own future actions based on their own generated skills
        2. from receiver's perspective, predict their own future actions based on communicated generated skills
            - this will be equal to point 1 when communication channel is not used
        3. from both sender and receiver's perspective, predict their own future actions based on their own generated skills or communicated generated skills
            - this will be repeated when communication channel is not used
        """
        if self.use_action_predictor == 1:          
            sender_pred_act_log = self.act_predictor(skill_list, future_available_actions)
            train_info["sender_pred_act_log"] = sender_pred_act_log
        elif self.use_action_predictor == 2:
            receiver_pred_act_log = self.act_predictor(comm_skill_list, future_available_actions)
            train_info["receiver_pred_act_log"] = receiver_pred_act_log
        elif self.use_action_predictor == 3:
            sender_pred_act_log = self.act_predictor(obs, skill_list, comm_skill_list, available_actions, active_masks, n_agents, n_enemies, n_entites)
            train_info["sender_pred_act_log"] = sender_pred_act_log

        return action_log_probs, dist_entropy, comm_skill_list, train_info


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        if args.use_obs_instead_of_state:
            self.critic_feat_dim = args.critic_feat_dim_low
        else:
            self.critic_feat_dim = args.critic_feat_dim_high

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNEntityLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.integration = IntegrationCritic(args, self.critic_feat_dim, self.hidden_size, args.use_orth, device)
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
            raise NotImplementedError
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)


    def forward(self, cent_obs, rnn_states, masks, active_masks, n_agents=None, n_enemies=None, n_entites=None, skill_actor=None):
        cent_obs = check(cent_obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)
        
        critic_feature_list = self.integration(cent_obs, n_agents, n_enemies, n_entites, skill_actor=skill_actor)

        # use RNN to capture time-relevant features
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_feature_list, rnn_states = self.rnn(critic_feature_list, rnn_states, masks)

        values = self.v_out(torch.cat([torch.mean(critic_feature, dim=1) for critic_feature in critic_feature_list]))

        train_info = {}
        
        return values, rnn_states, train_info

