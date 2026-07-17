"""
MAPPO actor and critic for baseline methods (MAPPO, SFT).

Controlled by --use_entity_actor / --no_entity_actor:
  Entity (default): entity obs → transformer encoder (IntegrationCritic)
  Flat:             flat obs → zero-pad to max dim → MLP

Actor and critic always use the same obs format.
"""
import torch
import torch.nn as nn
from base_policy.utils.util import init, init_, check
from base_policy.components.rnn_entity import RNNEntityLayer
from base_policy.algorithms.mcs.integation_critic import IntegrationCritic


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.actor_feat_dim = args.actor_feat_dim
        self.use_entity_actor = args.use_entity_actor

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.use_entity_actor:
            self.integration = IntegrationCritic(args, self.actor_feat_dim, self.hidden_size, args.use_orth, device)
        else:
            obs_dims = [sp[0][0] for sp in obs_space]
            input_dim = max(obs_dims)
            self._obs_dims = obs_dims
            self.input_dim = input_dim
            self.policy_head = nn.Sequential(
                nn.LayerNorm(input_dim),
                init_(args, nn.Linear(input_dim, self.hidden_size)),
                nn.ReLU(),
                init_(args, nn.Linear(self.hidden_size, self.hidden_size)),
                nn.ReLU(),
            )

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNEntityLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

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

    def _get_flat_obs(self, obs, n_agents):
        flat_obs_list = []
        for i, ob in enumerate(obs):
            flat_ob = ob[:, :self._obs_dims[i]] if ob.shape[-1] > self._obs_dims[i] else ob
            if flat_ob.shape[-1] < self.input_dim:
                pad = torch.zeros(flat_ob.shape[0], self.input_dim - flat_ob.shape[-1],
                                  dtype=flat_ob.dtype, device=flat_ob.device)
                flat_ob = torch.cat([flat_ob, pad], dim=-1)
            flat_obs_list.append(flat_ob)
        return flat_obs_list

    def _encode(self, obs, n_agents, n_enemies, n_entites):
        if self.use_entity_actor:
            fea_list = self.integration(obs, n_agents, n_enemies, n_entites, skill_actor=None)
        else:
            flat_obs_list = self._get_flat_obs(obs, n_agents)
            fea_list = [self.policy_head(ob) for ob in flat_obs_list]
            if "StarCraft" in self.args.env_name:
                fea_list = [fea.unsqueeze(1).expand(-1, ne, -1) for fea, ne in zip(fea_list, n_entites)]
        return fea_list

    def forward(self, obs, rnn_states, rnn_states_comm, masks, active_masks, available_actions=None,
                deterministic=False, n_agents=None, n_enemies=None, n_entites=None, is_training=False):
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        rnn_states_comm = check(rnn_states_comm, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)

        fea_list = self._encode(obs, n_agents, n_enemies, n_entites)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            fea_list, rnn_states = self.rnn(fea_list, rnn_states, masks)

        actions, action_log_probs, pi_probs = self.act_layer.forward4skills(fea_list, available_actions, deterministic, n_enemies)

        train_info = {}
        record_info = {"skill_dot": None, "comm_skill_dot": None, "comm_weights": [torch.zeros(0) for _ in fea_list]}

        return actions, action_log_probs, pi_probs, rnn_states, rnn_states_comm, fea_list, None, train_info, record_info

    def evaluate_actions(self, obs, rnn_states, rnn_states_comm, action, masks, active_masks, available_actions=None,
                         n_agents=None, n_enemies=None, n_entites=None, is_training=True, future_available_actions=None):
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        rnn_states_comm = check(rnn_states_comm, self.tpdv)
        action = check(action, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)

        fea_list = self._encode(obs, n_agents, n_enemies, n_entites)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            fea_list, rnn_states = self.rnn(fea_list, rnn_states, masks)

        action_log_probs, dist_entropy = self.act_layer.evaluate_actions(
            fea_list, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
            n_enemies=n_enemies)

        train_info = {}
        return action_log_probs, dist_entropy, None, train_info


class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.use_entity_critic = args.use_entity_actor
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

        if self.use_entity_critic:
            self.integration = IntegrationCritic(args, self.critic_feat_dim, self.hidden_size, args.use_orth, device)
        else:
            obs_dims = [sp[0][0] for sp in cent_obs_space]
            input_dim = max(obs_dims)
            self._obs_dims = obs_dims
            self.input_dim = input_dim
            self.policy_head = nn.Sequential(
                nn.LayerNorm(input_dim),
                init_(args, nn.Linear(input_dim, self.hidden_size)),
                nn.ReLU(),
                init_(args, nn.Linear(self.hidden_size, self.hidden_size)),
                nn.ReLU(),
            )

        def init_v(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            raise NotImplementedError
        else:
            self.v_out = init_v(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, active_masks, n_agents=None, n_enemies=None, n_entites=None, skill_actor=None):
        cent_obs = check(cent_obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)

        if self.use_entity_critic:
            critic_feature_list = self.integration(cent_obs, n_agents, n_enemies, n_entites, skill_actor=skill_actor)
        else:
            critic_feature_list = []
            for i, ob in enumerate(cent_obs):
                if ob.shape[-1] < self.input_dim:
                    pad = torch.zeros(ob.shape[0], self.input_dim - ob.shape[-1],
                                      dtype=ob.dtype, device=ob.device)
                    ob = torch.cat([ob, pad], dim=-1)
                elif ob.shape[-1] > self.input_dim:
                    ob = ob[:, :self.input_dim]
                critic_feature_list.append(self.policy_head(ob).unsqueeze(1))

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_feature_list, rnn_states = self.rnn(critic_feature_list, rnn_states, masks)

        values = self.v_out(torch.cat([torch.mean(critic_feature, dim=1) for critic_feature in critic_feature_list]))

        train_info = {}
        return values, rnn_states, train_info
