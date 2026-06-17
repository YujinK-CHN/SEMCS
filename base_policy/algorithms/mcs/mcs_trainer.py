import numpy as np
import torch
import torch.nn as nn
from base_policy.utils.util import get_gard_norm, huber_loss, mse_loss, cross_entropy_loss, kl_loss
from base_policy.components.valuenorm import ValueNorm
from base_policy.utils.util import check, print_numpy_array_sizes, print_cuda, report_model_memory
from torch.amp import autocast  # for flexible device_type support

def _cast(x):
    return [x_i.reshape(-1, x_i.shape[-1]) for x_i in x]

def _cast_future(x):
    return [x_i.reshape(-1, x_i.shape[-2], x_i.shape[-1]) for x_i in x]

def _cast_rnn(x):
    return [x_i.reshape(-1, x_i.shape[-2], x_i.shape[-1]) for x_i in x]


class mcsTrainer():
    """
    Trainer class for HMASD to update policies, which can be based on IPPO or MAPPO.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy, 
                 n_agents_list,
                 n_enemies_list,
                 n_entities_list,
                 device=torch.device("cpu")):

        # private parameters
        self.args = args
        self.multi_envs = args.train_tasks.split('|')
        self.num_multi_envs = len(self.multi_envs)
        self.n_agents_list = n_agents_list
        self.n_enemies_list = n_enemies_list
        self.n_entities_list = n_entities_list
        self.sim_coeffi = args.sim_coeffi
        self.skill_loss_coeffi = args.skill_loss_coeffi
        self.run_dir = args.run_dir

        # common paraemters
        self.device = device
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.use_mixed_percision = args.use_mixed_percision
        
        if self.use_mixed_percision:
            print("Use mixed percision..")
            from torch.amp import GradScaler  # GradScaler still lives here
            self.scaler = GradScaler(self.device_type)  # initialize once in __init__ of your trainer
        else:
            print("Do not use mixed percision..")

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
        
        self._use_action_predictor = args.use_action_predictor
        self.padded_values = args.padded_values
        self.pred_type = args.pred_type
        self.kl_gamma = args.kl_gamma
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None


    def cat_sample(self, sample):
        '''
        sample: [[obs1, act1, ...], [obs2, act2], ...],
        in which elem: [obsi, acti] is corresponds to ith env in multi_envs
        '''
        for item in zip(*sample):
            # every item is a tuple of obs/actions/..., in which every elem in this item 
            # corresponding to every multi_env respectively
            yield item
            
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        # normalized target
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            # why normalized for target values
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


    def cal_pred_loss(self, predict_action_log, future_action_probs_batch, future_action_batch):
        skill_loss_list = []
        skill_acc_list = []

        for log_probs_pred, target_probs, target_act in zip(predict_action_log, future_action_probs_batch, future_action_batch):
            # log_probs_pred: [B, T, C] - log-probabilities (F.log_softmax output)
            # target_probs:   [B, T, C] - probabilities (sum=1), unnormalized
            B, T, C = log_probs_pred.shape
            device = log_probs_pred.device
            
            # --- Step 1: Mask invalid target entries (e.g. padding) ---
            row_sums = target_probs.sum(dim=-1)  # [B, T]
            min_vals = target_probs.min(dim=-1).values
            valid_mask = (row_sums > 1e-6) & (min_vals >= 0)  # [B, T]
            valid_mask_act = (target_act >= 0) & (target_act < C)  # shape: [B, T, 1]
            valid_mask_act = valid_mask_act.squeeze(-1)            # shape: [B, T]
            total_mask = valid_mask & valid_mask_act

            if T==1:
                discounts = self.kl_gamma
            else:
                discounts = self.kl_gamma ** torch.arange(T, device=device, dtype=log_probs_pred.dtype)
                
            # --- Step 2: Prediction Loss ---
            if self.pred_type == "KL":
                # Replace invalid rows in target_probs with zeros (safe neutral value)
                target_probs = target_probs.clone()
                target_probs[~total_mask] = 0.0
                kl_per_class = kl_loss(log_probs_pred, target_probs.detach())  # [B, T, C]
                kl_per_step = kl_per_class.sum(dim=-1)                                   # [B, T]
                kl_per_step = kl_per_step * total_mask.float()                           # zero out invalid
                discounted_loss = (kl_per_step * discounts).mean()    # [B]
            elif self.pred_type == "CrossEntropy":
                # mask out invalid actions
                target_act_clamped = target_act.clone().long()
                target_act_clamped[~total_mask] = 0  # Prevent out-of-bounds
                
                gathered = torch.gather(log_probs_pred, dim=2, index=target_act_clamped).squeeze(2)  # [B, T]
                if T==1:
                    gathered = gathered * discounts  # [B, T]
                else:
                    discounts = discounts.view(1, T)  # [1, T] → broadcast over batch
                    gathered = gathered * discounts  # [B, T]
                masked = gathered.masked_fill(~total_mask, 0.0)  # [B, T], fill invalid with 0

                num_valid = total_mask.sum().clamp(min=1)
                discounted_loss = -masked.sum() / num_valid

            skill_loss_list.append(discounted_loss)

            # --- Step 3: Accuracy ---
            pred_labels = log_probs_pred.argmax(dim=-1)       # [B*T]
            true_labels = target_probs.argmax(dim=-1)         # [B*T]
            correct = (pred_labels == true_labels) & total_mask
            correct = correct.reshape(-1)
            total_mask = total_mask.reshape(-1)
            accuracy = correct.float().sum() / total_mask.sum().clamp(min=1)
            skill_acc_list.append(accuracy)

        return skill_loss_list, skill_acc_list


    def get_actor_loss(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """        
        if self._use_action_predictor:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_comm_batch, rnn_states_critic_batch, \
            actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, future_action_batch, future_pi_probs_batch, future_available_actions_batch = tuple(self.cat_sample(sample))
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_comm_batch, rnn_states_critic_batch, \
            actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = tuple(self.cat_sample(sample))
            future_available_actions_batch = None
        
        # _cast data for _use_recurrent_with_agents
        if self._use_recurrent_with_agents: 
            share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = _cast(share_obs_batch), _cast(obs_batch), \
            _cast(actions_batch), _cast(value_preds_batch), _cast(return_batch), _cast(masks_batch), _cast(active_masks_batch), _cast(old_action_log_probs_batch), \
            _cast(adv_targ), _cast(available_actions_batch)
            rnn_states_batch, rnn_states_comm_batch, rnn_states_critic_batch = _cast_rnn(rnn_states_batch), _cast_rnn(rnn_states_comm_batch), _cast_rnn(rnn_states_critic_batch)

            if self._use_action_predictor:
                future_action_batch, future_pi_probs_batch, future_available_actions_batch = _cast_future(future_action_batch), _cast_future(future_pi_probs_batch), _cast_future(future_available_actions_batch)
        
        old_action_log_probs_batch = check(old_action_log_probs_batch, self.tpdv)
        bs_na_list = [oalpb.size(0) for oalpb in old_action_log_probs_batch]

        old_action_log_probs_batch = torch.cat(old_action_log_probs_batch)
        adv_targ = torch.cat(check(adv_targ, self.tpdv))

        value_preds_batch = check(value_preds_batch, self.tpdv)
        return_batch = check(return_batch, self.tpdv)
        active_masks_batch = check(active_masks_batch, self.tpdv)
        if self._use_action_predictor:
            future_action_batch = check(future_action_batch, self.tpdv)
            future_pi_probs_batch = check(future_pi_probs_batch, self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, actor_train_info, critic_train_info = self.policy.evaluate_actions(share_obs_batch,
                                                                            obs_batch, 
                                                                            rnn_states_batch, 
                                                                            rnn_states_comm_batch,
                                                                            rnn_states_critic_batch, 
                                                                            actions_batch, 
                                                                            masks_batch, 
                                                                            active_masks_batch,
                                                                            available_actions_batch,
                                                                            self.n_agents_list,
                                                                            self.n_enemies_list,
                                                                            self.n_entities_list,
                                                                            is_training=True,
                                                                            future_available_actions=future_available_actions_batch)

        # skill updates
        if self._use_action_predictor == 1:
            # from sender's perspective
            sender_pred_act_log = actor_train_info["sender_pred_act_log"]
            skill_loss_list, skill_acc_list = self.cal_pred_loss(sender_pred_act_log, future_pi_probs_batch, future_action_batch)
            actor_train_info["prediction_loss"] = skill_loss_list
            actor_train_info["sender_accuracy"] = skill_acc_list
        elif self._use_action_predictor == 2:
            # from receiver's perspective
            receiver_pred_act_log = actor_train_info["receiver_pred_act_log"]
            skill_loss_list, skill_acc_list = self.cal_pred_loss(receiver_pred_act_log, future_pi_probs_batch, future_action_batch)
            actor_train_info["prediction_loss"] = skill_loss_list
            actor_train_info["receiver_accuracy"] = skill_acc_list
        elif self._use_action_predictor == 3:
            # from sender's perspective
            sender_pred_act_log = actor_train_info["sender_pred_act_log"]
            sender_skill_loss_list, sender_skill_acc_list = self.cal_pred_loss(sender_pred_act_log, future_pi_probs_batch, future_action_batch)
            # from receiver's perspective
            actor_train_info["prediction_loss"] = sender_skill_loss_list
            actor_train_info["sender_accuracy"] = sender_skill_acc_list

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # for each task?
        surr1_list = torch.split(surr1, bs_na_list, dim=0)
        surr2_list = torch.split(surr2, bs_na_list, dim=0)
        value_list = torch.split(values, bs_na_list, dim=0)

        value_loss_list = []
        actor_loss_list = []

        for surr1_i, surr2_i, active_masks_batch_i, dist_entropy_i, \
            value_i, value_preds_batch_i, return_batch_i in zip(
            surr1_list, surr2_list, active_masks_batch, dist_entropy, value_list,
            value_preds_batch, return_batch
        ):
            if self._use_policy_active_masks:
                policy_action_loss = (-torch.sum(torch.min(surr1_i, surr2_i),
                                                dim=-1,
                                                keepdim=True) * active_masks_batch_i).sum() / active_masks_batch_i.sum()
            else:
                policy_action_loss = -torch.sum(torch.min(surr1_i, surr2_i), dim=-1, keepdim=True).mean()

            policy_loss = policy_action_loss
            actor_loss = policy_loss - dist_entropy_i * self.entropy_coef

            # critic update
            value_loss = self.cal_value_loss(value_i, value_preds_batch_i, return_batch_i, active_masks_batch_i) * self.value_loss_coef            
            value_loss_list.append(value_loss)
            actor_loss_list.append(actor_loss)

        return value_loss_list, actor_loss_list, dist_entropy, imp_weights, actor_train_info, critic_train_info


    def ppo_update(self, sample, num_episodes, update_actor=True):
        value_loss_list, actor_loss_list, dist_entropy, imp_weights, actor_train_info, critic_train_info = self.get_actor_loss(sample)

        actor_loss = torch.stack(actor_loss_list).mean()
        critic_loss = torch.stack(value_loss_list).mean()

        vae_loss = torch.tensor(0.0, dtype=actor_loss.dtype, device=self.device)
        sim_loss = torch.tensor(0.0, dtype=actor_loss.dtype, device=self.device)
        pred_loss = torch.tensor(0.0, dtype=actor_loss.dtype, device=self.device)

        if self.args.skill_choice == "UseVAE":
            vae_loss_kl = torch.stack(actor_train_info["vae_loss_kl"]).mean()
            vae_loss_re = torch.stack(actor_train_info["vae_loss_re"]).mean()
            vae_loss = vae_loss_kl + vae_loss_re

        if self.args.use_similarity:
            sim_loss = torch.stack(actor_train_info["simm_loss"]).mean()

        if self._use_action_predictor:
            pred_loss = torch.stack(actor_train_info["prediction_loss"]).mean()
        
        # if self.args.comm_channel != "None":
        #     comm_loss = actor_train_info["comm_loss"]

        # diagnostic step
        pred_train_info = {}
        if num_episodes % self.args.record_grad_interval == 0 and self._use_action_predictor and self.args.comm_channel != "None":
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            param = self.policy.actor.integration.skill_generator.toprobs.weight
            
            # Helper: compute isolated gradient and clone
            def get_grad_from(loss):
                self.policy.actor_optimizer.zero_grad()
                self.policy.critic_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                grad = param.grad.clone()
                return grad

            # Compute individual gradients
            actor_grad  = get_grad_from(actor_loss)
            pred_grad   = get_grad_from(pred_loss * self.skill_loss_coeffi)
            critic_grad = get_grad_from(critic_loss + vae_loss + self.sim_coeffi * sim_loss)

            # Compute norms and relative influence
            actor_norm  = actor_grad.norm().item()
            pred_norm   = pred_grad.norm().item()
            critic_norm = critic_grad.norm().item()

            total_grad = actor_grad + pred_grad + critic_grad
            total_norm = total_grad.norm().item()
            pred_train_info["grad/actor_contrib"] = actor_norm / total_norm
            pred_train_info["grad/pred_contrib"] = pred_norm / total_norm
            pred_train_info["grad/critic_contrib"] = critic_norm / total_norm
            pred_train_info["grad/total_norm"] = total_norm
            torch.cuda.empty_cache()

        # real updates here
        self.policy.actor_optimizer.zero_grad()
        self.policy.critic_optimizer.zero_grad()
        total_loss = (
            actor_loss +
            critic_loss +
            vae_loss +
            self.sim_coeffi * sim_loss +
            self.skill_loss_coeffi * pred_loss)
            # 0.0001 * comm_loss)
        
        total_loss.backward()
        
        # report_model_memory(self.policy.actor, "Actor")
        # report_model_memory(self.policy.critic, "Critic")
        # Check if gradients are updated
        # for name, param in self.policy.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm().item()}, grad mean: {param.grad.mean().item()}")
        #     else:
        #         print(f"{name} has no gradient")

        # Clip gradients if needed
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.actor_optimizer.step()
        self.policy.critic_optimizer.step()
        
        return value_loss_list, critic_grad_norm, actor_loss_list, dist_entropy, actor_grad_norm, imp_weights, actor_train_info, critic_train_info, pred_train_info


    def ppo_update_mixed_percision(self, sample, num_episodes, update_actor=True):
        with autocast(device_type=self.device_type, dtype=torch.bfloat16, enabled=True):  # Enables mixed precision
            value_loss_list, actor_loss_list, dist_entropy, imp_weights, actor_train_info, critic_train_info = self.get_actor_loss(sample)

            actor_loss = torch.stack(actor_loss_list).mean()
            critic_loss = torch.stack(value_loss_list).mean()

            vae_loss = torch.tensor(0.0, dtype=actor_loss.dtype, device=self.device)
            sim_loss = torch.tensor(0.0, dtype=actor_loss.dtype, device=self.device)
            pred_loss = torch.tensor(0.0, dtype=actor_loss.dtype, device=self.device)

            if self.args.skill_choice == "UseVAE":
                vae_loss_kl = torch.stack(actor_train_info["vae_loss_kl"]).mean()
                vae_loss_re = torch.stack(actor_train_info["vae_loss_re"]).mean()
                vae_loss = vae_loss_kl + vae_loss_re

            if self.args.use_similarity:
                sim_loss = torch.stack(actor_train_info["simm_loss"]).mean()

            if self._use_action_predictor:
                pred_loss = torch.stack(actor_train_info["prediction_loss"]).mean()

            # diagnostic step
            pred_train_info = {}
            if self._use_action_predictor and num_episodes % self.args.record_grad_interval == 0:
                self.policy.actor_optimizer.zero_grad()
                self.policy.critic_optimizer.zero_grad()
                param = self.policy.actor.integration.skill_generator.toprobs.weight

                # Helper: compute isolated gradient and clone
                def get_grad_from(loss):
                    self.policy.actor_optimizer.zero_grad()
                    self.policy.critic_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    grad = param.grad.clone()
                    return grad

                # Compute individual gradients
                actor_grad  = get_grad_from(actor_loss)
                pred_grad   = get_grad_from(pred_loss * self.skill_loss_coeffi)
                critic_grad = get_grad_from(critic_loss + vae_loss + self.sim_coeffi * sim_loss)

                # Compute norms and relative influence
                actor_norm  = actor_grad.norm().item()
                pred_norm   = pred_grad.norm().item()
                critic_norm = critic_grad.norm().item()

                total_grad = actor_grad + pred_grad + critic_grad
                total_norm = total_grad.norm().item()
                pred_train_info["grad/actor_contrib"] = actor_norm / total_norm
                pred_train_info["grad/pred_contrib"] = pred_norm / total_norm
                pred_train_info["grad/critic_contrib"] = critic_norm / total_norm
                pred_train_info["grad/total_norm"] = total_norm
                torch.cuda.empty_cache()

            # real updates here
            self.policy.actor_optimizer.zero_grad()
            self.policy.critic_optimizer.zero_grad()
            total_loss = (
                actor_loss +
                critic_loss +
                vae_loss +
                self.sim_coeffi * sim_loss +
                self.skill_loss_coeffi * pred_loss)
        
        # total_loss.backward()

        # Backward pass with scaling
        self.scaler.scale(total_loss).backward()
    
        # report_model_memory(self.policy.actor, "Actor")
        # report_model_memory(self.policy.critic, "Critic")
        # # Check if gradients are updated
        # for name, param in self.policy.actor.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm().item()}, grad mean: {param.grad.mean().item()}")
        #     else:
        #         print(f"{name} has no gradient")

        # Clip gradients if needed
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        # self.policy.actor_optimizer.step()
        # self.policy.critic_optimizer.step()
        
        # Optimizer step
        self.scaler.step(self.policy.actor_optimizer)
        self.scaler.step(self.policy.critic_optimizer)
        self.scaler.update()  # updates the scale for next iteration
    
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
            # normalize advantages
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
        if self.args.use_similarity:
            train_info['Extra/simm_loss'] = 0
            train_info['Extra/obs_sim'] = 0
            train_info['Extra/skill_sim'] = 0
        if self._use_action_predictor:
            train_info['comm/prediction_loss'] = 0
            train_info['comm/sender_accuracy'] = 0
            if self._use_action_predictor == 2:
                train_info['comm/receiver_accuracy'] = 0
            if num_episodes % self.args.record_grad_interval == 0 and self.args.comm_channel != "None":
                train_info['comm/comm_links'] = 0
                train_info['grad/actor_contrib'] = 0
                train_info['grad/pred_contrib'] = 0
                train_info['grad/critic_contrib'] = 0
                train_info['grad/total_norm'] = 0
        
        for n_epoch in range(self.ppo_epoch):                        
            if (self._use_recurrent_policy or "skill" in self.args.experiment_name) and not self._use_recurrent_with_agents and not self._use_action_predictor:
                # print("1, recurrent_generator")
                data_generator = buffer.recurrent_generator(multi_advantages, self.num_mini_batch, self.data_chunk_length)
            elif (self._use_recurrent_policy or "skill" in self.args.experiment_name) and self._use_recurrent_with_agents and not self._use_action_predictor:
                # print("2, recurrent_generator with agents")
                data_generator = buffer.recurrent_generator_with_agents(multi_advantages, self.num_mini_batch, self.data_chunk_length)
            elif (self._use_recurrent_policy or "skill" in self.args.experiment_name) and not self._use_recurrent_with_agents and self._use_action_predictor:
                # print("3, recurrent_generator with future actions")
                data_generator = buffer.recurrent_generator_with_future_actions(multi_advantages, self.num_mini_batch, self.data_chunk_length)
            elif (self._use_recurrent_policy or "skill" in self.args.experiment_name) and self._use_recurrent_with_agents and self._use_action_predictor:
                # print("4, recurrent_generator with agents")
                data_generator = buffer.recurrent_generator_with_agents_with_future_actions(multi_advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                # print("5, naive_recurrent_generator")
                data_generator = buffer.naive_recurrent_generator(multi_advantages, self.num_mini_batch)
            else:
                # print("6, feed_forward_generator")
                data_generator = buffer.feed_forward_generator(multi_advantages, self.num_mini_batch)

            for sample in zip(*data_generator):
                if self.use_mixed_percision:
                    value_loss_list, critic_grad_norm, actor_loss_list, dist_entropy, actor_grad_norm, \
                        imp_weights, actor_train_info, critic_train_info, pred_train_info = self.ppo_update_mixed_percision(sample, num_episodes, update_actor)    
                else:  
                    value_loss_list, critic_grad_norm, actor_loss_list, dist_entropy, actor_grad_norm, \
                        imp_weights, actor_train_info, critic_train_info, pred_train_info = self.ppo_update(sample, num_episodes, update_actor)                
                for idx, value_loss in enumerate(value_loss_list):
                    train_info['Ptrain/value_loss'][self.multi_envs[idx]] \
                        += value_loss.item()
                for idx, actor_loss in enumerate(actor_loss_list):
                    train_info['Ptrain/policy_loss'][self.multi_envs[idx]] \
                        += actor_loss.item()
                train_info['Ptrain/dist_entropy'] += torch.stack(dist_entropy).mean().item()
                train_info['grad/actor_grad_norm'] += actor_grad_norm.item()
                train_info['grad/critic_grad_norm'] += critic_grad_norm.item()
                train_info['Ptrain/ratio'] += imp_weights.mean().item()
                
                if self.args.skill_choice == "UseVAE":
                    train_info['Extra/actor_kl_loss'] += torch.stack(actor_train_info["vae_loss_kl"]).mean().item()
                    train_info['Extra/actor_re_loss'] += torch.stack(actor_train_info["vae_loss_re"]).mean().item()
                if self.args.use_similarity:
                    train_info['Extra/simm_loss'] += torch.stack(actor_train_info["simm_loss"]).mean().item()
                    train_info['Extra/obs_sim'] += torch.stack(actor_train_info["obs_sim"]).mean().item()
                    train_info['Extra/skill_sim'] += torch.stack(actor_train_info["skill_sim"]).mean().item()
                if self._use_action_predictor:
                    train_info['comm/prediction_loss'] = torch.stack(actor_train_info["prediction_loss"]).mean().item()
                    if self._use_action_predictor == 1:
                        train_info['comm/sender_accuracy'] += torch.stack(actor_train_info["sender_accuracy"]).mean().item()
                    elif self._use_action_predictor == 2:
                        train_info['comm/receiver_accuracy'] += torch.stack(actor_train_info["receiver_accuracy"]).mean().item()
                    elif self._use_action_predictor == 3:
                        train_info['comm/sender_accuracy'] += torch.stack(actor_train_info["sender_accuracy"]).mean().item()
                    if num_episodes % self.args.record_grad_interval == 0 and self.args.comm_channel != "None":
                        train_info['comm/comm_links'] += actor_train_info["comm_links"]
                        train_info['grad/actor_contrib'] += pred_train_info['grad/actor_contrib']
                        train_info['grad/pred_contrib'] += pred_train_info['grad/pred_contrib']
                        train_info['grad/critic_contrib'] += pred_train_info['grad/critic_contrib']
                        train_info['grad/total_norm'] += pred_train_info['grad/total_norm']
        
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if isinstance(train_info[k], dict):
                for key in train_info[k].keys():
                    train_info[k][key] /= num_updates
            else:
                train_info[k] /= num_updates

        return train_info
    

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

