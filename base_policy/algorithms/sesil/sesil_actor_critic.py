import torch
import torch.nn as nn
from torch.nn import functional as F
from base_policy.utils.util import init, init_, check
from base_policy.components.rnn_entity import RNNEntityLayer
from base_policy.algorithms.sesil.encoder_population import EncoderPopulation
from base_policy.algorithms.mcs.integation_critic import IntegrationCritic
from base_policy.utils.entity_util import encode_entity


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_skills = args.num_skills
        self.actor_feat_dim = args.actor_feat_dim
        self.use_entity_obs = bool(args.sesil_use_entity_obs)

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.use_entity_obs:
            # Ablation: encode_entity → mean-pool → constant actor_feat_dim
            encoder_input_dim = self.actor_feat_dim
        else:
            # Default: flat individual obs, padded to max across tasks
            obs_dims = [sp[0][0] for sp in obs_space]
            encoder_input_dim = max(obs_dims)
            self._obs_dims = obs_dims

        self.encoder_input_dim = encoder_input_dim
        self.encoder_population = EncoderPopulation(args, encoder_input_dim, self.num_skills, device)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(encoder_input_dim + self.num_skills),
            init_(args, nn.Linear(encoder_input_dim + self.num_skills, self.hidden_size)),
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

    def _get_encoder_input(self, obs, n_agents, n_entites):
        """Returns encoder input list depending on obs mode.
        Default (use_entity_obs=0): flat obs with zero-padding to max dim.
        Ablation (use_entity_obs=1): encode_entity → mean-pool.
        """
        if self.use_entity_obs:
            entity_ob_list, _, _ = encode_entity(self.args, obs, n_agents, n_entites, self.actor_feat_dim)
            return [entity_ob.mean(dim=-2) for entity_ob in entity_ob_list]
        else:
            flat_obs_list = []
            for ob, na, ne in zip(obs, n_agents, n_entites):
                if self.args.skill_to_obs == "merge":
                    flat_ob = ob[:, :-self.args.num_skills]
                elif self.args.skill_to_obs == "entity":
                    flat_ob = ob[:, :-self.args.num_skills * ne]
                else:
                    flat_ob = ob
                # Zero-pad to encoder_input_dim if needed
                if flat_ob.shape[-1] < self.encoder_input_dim:
                    pad = torch.zeros(flat_ob.shape[0], self.encoder_input_dim - flat_ob.shape[-1],
                                      dtype=flat_ob.dtype, device=flat_ob.device)
                    flat_ob = torch.cat([flat_ob, pad], dim=-1)
                flat_obs_list.append(flat_ob)
            return flat_obs_list

    def forward(self, obs, rnn_states, rnn_states_comm, masks, active_masks, available_actions=None,
                deterministic=False, n_agents=None, n_enemies=None, n_entites=None, is_training=False):
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        rnn_states_comm = check(rnn_states_comm, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)

        enc_input_list = self._get_encoder_input(obs, n_agents, n_entites)

        skill_list, train_info = self.encoder_population(enc_input_list, n_agents, is_training)

        fea_list = [self.policy_head(torch.cat([inp, sk], dim=-1)) for inp, sk in zip(enc_input_list, skill_list)]

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            fea_list, rnn_states = self.rnn(fea_list, rnn_states, masks)

        fea_list = self._expand_for_act_layer(fea_list, n_entites)

        actions, action_log_probs, pi_probs = self.act_layer.forward4skills(fea_list, available_actions, deterministic, n_enemies)

        record_info = {"skill_dot": None, "comm_skill_dot": None, "comm_weights": [torch.zeros(0) for _ in skill_list]}

        return actions, action_log_probs, pi_probs, rnn_states, rnn_states_comm, enc_input_list, skill_list, train_info, record_info

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

        enc_input_list = self._get_encoder_input(obs, n_agents, n_entites)

        skill_list, train_info = self.encoder_population(enc_input_list, n_agents, is_training)

        fea_list = [self.policy_head(torch.cat([inp, sk], dim=-1)) for inp, sk in zip(enc_input_list, skill_list)]

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            fea_list, rnn_states = self.rnn(fea_list, rnn_states, masks)

        fea_list = self._expand_for_act_layer(fea_list, n_entites)

        action_log_probs, dist_entropy = self.act_layer.evaluate_actions(
            fea_list, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
            n_enemies=n_enemies)

        return action_log_probs, dist_entropy, skill_list, train_info

    def _expand_for_act_layer(self, fea_list, n_entites):
        """For StarCraft's EntityVAEACTLayer: expand (bs, hidden) → (bs, n_entity, hidden).
        For AliceBob/Football's MultiACTLayer: no-op (already flat)."""
        if "StarCraft" in self.args.env_name:
            return [fea.unsqueeze(1).expand(-1, ne, -1) for fea, ne in zip(fea_list, n_entites)]
        return fea_list


class R_Critic(nn.Module):
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

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_feature_list, rnn_states = self.rnn(critic_feature_list, rnn_states, masks)

        values = self.v_out(torch.cat([torch.mean(critic_feature, dim=1) for critic_feature in critic_feature_list]))

        train_info = {}
        return values, rnn_states, train_info
