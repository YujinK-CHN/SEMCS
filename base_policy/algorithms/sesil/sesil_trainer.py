import numpy as np
import torch
import torch.nn as nn
from base_policy.utils.util import get_gard_norm, huber_loss, mse_loss, check
from base_policy.components.valuenorm import ValueNorm


def _cast(x):
    return [x_i.reshape(-1, x_i.shape[-1]) for x_i in x]

def _cast_rnn(x):
    return [x_i.reshape(-1, x_i.shape[-2], x_i.shape[-1]) for x_i in x]


class sesilTrainer():
    def __init__(self, args, policy, n_agents_list, n_enemies_list, n_entities_list, device=torch.device("cpu")):
        self.args = args
        self.multi_envs = args.train_tasks.split('|')
        self.num_multi_envs = len(self.multi_envs)
        self.n_agents_list = n_agents_list
        self.n_enemies_list = n_enemies_list
        self.n_entities_list = n_entities_list
        self.run_dir = args.run_dir

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_recurrent_with_agents = args.use_recurrent_with_agents
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert not (self._use_popart and self._use_valuenorm)

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cat_sample(self, sample):
        for item in zip(*sample):
            yield item

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def get_actor_loss(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_comm_batch, rnn_states_critic_batch, \
        actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = tuple(self.cat_sample(sample))

        if self._use_recurrent_with_agents:
            share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = _cast(share_obs_batch), _cast(obs_batch), \
            _cast(actions_batch), _cast(value_preds_batch), _cast(return_batch), _cast(masks_batch), _cast(active_masks_batch), _cast(old_action_log_probs_batch), \
            _cast(adv_targ), _cast(available_actions_batch)
            rnn_states_batch, rnn_states_comm_batch, rnn_states_critic_batch = _cast_rnn(rnn_states_batch), _cast_rnn(rnn_states_comm_batch), _cast_rnn(rnn_states_critic_batch)

        old_action_log_probs_batch = check(old_action_log_probs_batch, self.tpdv)
        bs_na_list = [oalpb.size(0) for oalpb in old_action_log_probs_batch]

        old_action_log_probs_batch = torch.cat(old_action_log_probs_batch)
        adv_targ = torch.cat(check(adv_targ, self.tpdv))

        value_preds_batch = check(value_preds_batch, self.tpdv)
        return_batch = check(return_batch, self.tpdv)
        active_masks_batch = check(active_masks_batch, self.tpdv)

        values, action_log_probs, dist_entropy, actor_train_info, critic_train_info = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_comm_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, active_masks_batch, available_actions_batch,
            self.n_agents_list, self.n_enemies_list, self.n_entities_list, is_training=True)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        surr1_list = torch.split(surr1, bs_na_list, dim=0)
        surr2_list = torch.split(surr2, bs_na_list, dim=0)
        value_list = torch.split(values, bs_na_list, dim=0)

        value_loss_list = []
        actor_loss_list = []

        for surr1_i, surr2_i, active_masks_batch_i, dist_entropy_i, \
            value_i, value_preds_batch_i, return_batch_i in zip(
            surr1_list, surr2_list, active_masks_batch, dist_entropy, value_list,
            value_preds_batch, return_batch):

            if self._use_policy_active_masks:
                policy_action_loss = (-torch.sum(torch.min(surr1_i, surr2_i), dim=-1, keepdim=True) * active_masks_batch_i).sum() / active_masks_batch_i.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1_i, surr2_i), dim=-1, keepdim=True).mean()

            actor_loss = policy_action_loss - dist_entropy_i * self.entropy_coef
            value_loss = self.cal_value_loss(value_i, value_preds_batch_i, return_batch_i, active_masks_batch_i) * self.value_loss_coef

            value_loss_list.append(value_loss)
            actor_loss_list.append(actor_loss)

        return value_loss_list, actor_loss_list, dist_entropy, imp_weights, actor_train_info, critic_train_info

    def ppo_update(self, sample, num_episodes, update_actor=True):
        value_loss_list, actor_loss_list, dist_entropy, imp_weights, actor_train_info, critic_train_info = self.get_actor_loss(sample)

        actor_loss = torch.stack(actor_loss_list).mean()
        critic_loss = torch.stack(value_loss_list).mean()

        vae_loss = torch.tensor(0.0, dtype=actor_loss.dtype, device=self.device)
        if self.args.skill_choice == "UseVAE":
            vae_loss_kl = torch.stack(actor_train_info["vae_loss_kl"]).mean()
            vae_loss_re = torch.stack(actor_train_info["vae_loss_re"]).mean()
            vae_loss = vae_loss_kl + vae_loss_re

        self.policy.actor_optimizer.zero_grad()
        self.policy.critic_optimizer.zero_grad()
        total_loss = actor_loss + critic_loss + vae_loss
        total_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.actor_optimizer.step()
        self.policy.critic_optimizer.step()

        pred_train_info = {}
        return value_loss_list, critic_grad_norm, actor_loss_list, dist_entropy, actor_grad_norm, imp_weights, actor_train_info, critic_train_info, pred_train_info

    def train(self, buffer, num_episodes, update_actor=True):
        self.multi_actor_loss, self.multi_critic_loss = None, None
        multi_advantages = []
        for idx in range(self.num_multi_envs):
            if self._use_popart or self._use_valuenorm:
                advantages = buffer.buffer_lists[idx].returns[:-1] - self.value_normalizer.denormalize(buffer.buffer_lists[idx].value_preds[:-1])
            else:
                advantages = buffer.buffer_lists[idx].returns[:-1] - buffer.buffer_lists[idx].value_preds[:-1]
            advantages_copy = advantages.copy()
            advantages_copy[buffer.buffer_lists[idx].active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            multi_advantages.append(advantages)

        train_info = {}
        train_info['Ptrain/value_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['Ptrain/policy_loss'] = dict(zip(self.multi_envs, [0 for _ in self.multi_envs]))
        train_info['Ptrain/dist_entropy'] = 0
        train_info['grad/actor_grad_norm'] = 0
        train_info['grad/critic_grad_norm'] = 0
        train_info['Ptrain/ratio'] = 0

        if self.args.skill_choice == "UseVAE":
            train_info['Extra/actor_kl_loss'] = 0
            train_info['Extra/actor_re_loss'] = 0

        for n_epoch in range(self.ppo_epoch):
            if self._use_recurrent_policy or "skill" in self.args.experiment_name:
                if self._use_recurrent_with_agents:
                    data_generator = buffer.recurrent_generator_with_agents(multi_advantages, self.num_mini_batch, self.data_chunk_length)
                else:
                    data_generator = buffer.recurrent_generator(multi_advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(multi_advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(multi_advantages, self.num_mini_batch)

            for sample in zip(*data_generator):
                value_loss_list, critic_grad_norm, actor_loss_list, dist_entropy, actor_grad_norm, \
                    imp_weights, actor_train_info, critic_train_info, pred_train_info = self.ppo_update(sample, num_episodes, update_actor)

                for idx, value_loss in enumerate(value_loss_list):
                    train_info['Ptrain/value_loss'][self.multi_envs[idx]] += value_loss.item()
                for idx, actor_loss in enumerate(actor_loss_list):
                    train_info['Ptrain/policy_loss'][self.multi_envs[idx]] += actor_loss.item()
                train_info['Ptrain/dist_entropy'] += torch.stack(dist_entropy).mean().item()
                train_info['grad/actor_grad_norm'] += actor_grad_norm.item()
                train_info['grad/critic_grad_norm'] += critic_grad_norm.item()
                train_info['Ptrain/ratio'] += imp_weights.mean().item()

                if self.args.skill_choice == "UseVAE":
                    train_info['Extra/actor_kl_loss'] += torch.stack(actor_train_info["vae_loss_kl"]).mean().item()
                    train_info['Extra/actor_re_loss'] += torch.stack(actor_train_info["vae_loss_re"]).mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            if isinstance(train_info[k], dict):
                for key in train_info[k].keys():
                    train_info[k][key] /= num_updates
            else:
                train_info[k] /= num_updates

        return train_info

    def evolutionary_step(self, obs_batch):
        evo_info = self.policy.actor.encoder_population.evolve(obs_batch)

        encoder_params = list(self.policy.actor.encoder_population.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}

        for group in self.policy.actor_optimizer.param_groups:
            for p in group['params']:
                if id(p) in encoder_param_ids and p in self.policy.actor_optimizer.state:
                    self.policy.actor_optimizer.state[p] = {}

        return evo_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
