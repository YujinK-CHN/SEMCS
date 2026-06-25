import torch
import torch.nn as nn
from torch.nn import functional as F
from base_policy.utils.util import init, init_, check
from base_policy.components.rnn_entity import RNNEntityLayer
from base_policy.components.transformers import EncoderBlock, Mod_Sequential
from base_policy.utils.entity_util import encode_entity
from base_policy.algorithms.sesil.encoder_population import EncoderPopulation
from base_policy.algorithms.mcs.integation_critic import IntegrationCritic


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.args = args
        self.use_actor_loss = args.use_actor_loss
        self.actor_feat_dim = args.actor_feat_dim
        self.hidden_size = args.hidden_size
        self.num_skills = args.num_skills
        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.n_block = args.n_block

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNEntityLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.encoder_population = EncoderPopulation(args, self.actor_feat_dim, self.num_skills, device)

        tblocks = []
        for _ in range(self.n_block):
            tblocks.append(EncoderBlock(args, self.n_embd, self.n_head, args.use_orth))
        self.tblocks = Mod_Sequential(*tblocks)

        if args.use_norm_init:
            self.input_embedding = nn.Sequential(
                nn.LayerNorm(self.actor_feat_dim),
                init_(args, nn.Linear(self.actor_feat_dim, self.n_embd * self.n_head)),
                nn.GELU()
            )
            self.skill_embedding = nn.Sequential(
                nn.LayerNorm(self.num_skills),
                init_(args, nn.Linear(self.num_skills, self.n_embd)),
                nn.GELU(),
                init_(args, nn.Linear(self.n_embd, self.n_embd * self.n_head)),
                nn.Tanh()
            )
        else:
            self.input_embedding = init_(args, nn.Linear(self.actor_feat_dim, self.n_embd * self.n_head))
            self.skill_embedding = nn.Sequential(
                init_(args, nn.Linear(self.num_skills, self.n_embd)),
                init_(args, nn.Linear(self.n_embd, self.n_embd * self.n_head)),
                nn.Tanh()
            )

        self.toprobs = nn.Sequential(
            nn.LayerNorm(2 * self.n_embd * self.n_head),
            init_(args, nn.Linear(2 * self.n_embd * self.n_head, self.hidden_size))
        )

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

    def forward(self, obs, rnn_states, rnn_states_comm, masks, active_masks, available_actions=None,
                deterministic=False, n_agents=None, n_enemies=None, n_entites=None, is_training=False):
        obs = check(obs, self.tpdv)
        rnn_states = check(rnn_states, self.tpdv)
        rnn_states_comm = check(rnn_states_comm, self.tpdv)
        masks = check(masks, self.tpdv)
        active_masks = check(active_masks, self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions, self.tpdv)

        entity_ob_list, past_skill_list, _ = encode_entity(self.args, obs, n_agents, n_entites, self.actor_feat_dim)

        skill_list, train_info = self.encoder_population(entity_ob_list, n_agents, n_entites, is_training)

        x_emb = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
        skill_emb = [self.skill_embedding(sk) for sk in skill_list]
        skill_emb = [s_emb.unsqueeze(-2).repeat(1, n_en, 1) for s_emb, n_en in zip(skill_emb, n_entites)]

        x_emb, _, skill_emb, _ = self.tblocks.forward_4_skills(x_emb, skill_emb)
        fea_list = [self.toprobs(torch.cat([i_x, i_s], dim=-1)) for i_x, i_s in zip(x_emb, skill_emb)]

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            fea_list, rnn_states = self.rnn(fea_list, rnn_states, masks)

        actions, action_log_probs, pi_probs = self.act_layer.forward4skills(fea_list, available_actions, deterministic, n_enemies)

        record_info = {"skill_dot": None, "comm_skill_dot": None, "comm_weights": [torch.zeros(0) for _ in skill_list]}

        return actions, action_log_probs, pi_probs, rnn_states, rnn_states_comm, entity_ob_list, skill_list, train_info, record_info

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

        entity_ob_list, past_skill_list, _ = encode_entity(self.args, obs, n_agents, n_entites, self.actor_feat_dim)

        skill_list, train_info = self.encoder_population(entity_ob_list, n_agents, n_entites, is_training)

        x_emb = [self.input_embedding(entity_ob) for entity_ob in entity_ob_list]
        skill_emb = [self.skill_embedding(sk) for sk in skill_list]
        skill_emb = [s_emb.unsqueeze(-2).repeat(1, n_en, 1) for s_emb, n_en in zip(skill_emb, n_entites)]

        x_emb, _, skill_emb, _ = self.tblocks.forward_4_skills(x_emb, skill_emb)
        fea_list = [self.toprobs(torch.cat([i_x, i_s], dim=-1)) for i_x, i_s in zip(x_emb, skill_emb)]

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            fea_list, rnn_states = self.rnn(fea_list, rnn_states, masks)

        action_log_probs, dist_entropy = self.act_layer.evaluate_actions(
            fea_list, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
            n_enemies=n_enemies)

        return action_log_probs, dist_entropy, skill_list, train_info


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
