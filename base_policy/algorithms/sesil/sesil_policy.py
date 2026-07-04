import torch
from base_policy.utils.util import update_linear_schedule
from base_policy.algorithms.sesil.sesil_actor_critic import R_Actor, R_Critic


class sesilPolicy():
    def __init__(self, args, multi_envs, num_thread_per_env, obs_space, cent_obs_space, act_space, device=torch.device('cpu')):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

        self.tpdv = dict(dtype=torch.float32, device=device)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_comm, rnn_states_critic, masks, active_masks,
                    available_actions=None, deterministic=False, n_agents=None, n_enemies=None, n_entities=None, is_training=False):
        actions, action_log_probs, pi_probs, rnn_states_actor, rnn_states_comm, _, skill_actor, train_info, record_info = self.actor(
            obs, rnn_states_actor, rnn_states_comm, masks, active_masks, available_actions,
            deterministic, n_agents, n_enemies, n_entities, is_training=False)

        values, rnn_states_critic, _ = self.critic(cent_obs, rnn_states_critic, masks, active_masks,
                                                    n_agents, n_enemies, n_entities, skill_actor=skill_actor)

        return values, actions, action_log_probs, pi_probs, rnn_states_actor, rnn_states_comm, rnn_states_critic, skill_actor, skill_actor, train_info, record_info

    def get_values(self, cent_obs, rnn_states_critic, masks, active_masks, n_agents=None, n_enemies=None, n_entities=None, is_training=False):
        values, _, _ = self.critic(cent_obs, rnn_states_critic, masks, active_masks, n_agents, n_enemies, n_entities, skill_actor=None)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_comm, rnn_states_critic, action, masks, active_masks,
                         available_actions=None, n_agents=None, n_enemies=None, n_entities=None, is_training=True, future_available_actions=None):
        action_log_probs, dist_entropy, skill_actor, actor_train_info = self.actor.evaluate_actions(
            obs, rnn_states_actor, rnn_states_comm, action, masks, active_masks,
            available_actions, n_agents, n_enemies, n_entities, is_training=True, future_available_actions=future_available_actions)

        values, _, critic_train_info = self.critic(cent_obs, rnn_states_critic, masks, active_masks,
                                                    n_agents, n_enemies, n_entities, skill_actor=skill_actor)

        return values, action_log_probs, dist_entropy, actor_train_info, critic_train_info

    def act(self, obs, rnn_states_actor, rnn_states_comm, masks, active_masks, available_actions=None,
            deterministic=False, n_agents=None, n_enemies=None, n_entities=None, is_training=False):
        actions, _, _, rnn_states_actor, rnn_states_comm, entity_obs, skill_actor, _, record_info = self.actor(
            obs, rnn_states_actor, rnn_states_comm, masks, active_masks, available_actions,
            deterministic, n_agents, n_enemies, n_entities, is_training=False)
        return actions, rnn_states_actor, rnn_states_comm, entity_obs, skill_actor, record_info
