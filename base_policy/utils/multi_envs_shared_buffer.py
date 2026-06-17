import numpy as np
from copy import deepcopy

from base_policy.utils.shared_buffer import SharedReplayBuffer, RNN_Cognition_Buffer, SkillSharedReplayBuffer, subtask_Buffer

from base_policy.utils.shared_buffer_comm import SharedReplayBufferComm, SkillSharedReplayBufferComm

class MultiEnvSharedReplayBufferComm(SharedReplayBufferComm):
    def __init__(self, args, 
                 num_agents_list, 
                 num_entities_list, 
                 obs_space_list, 
                 cent_obs_space_list, 
                 act_space_list, 
                 num_thread_per_env):
        # a list of buffer for each environment
        self.buffer_lists = []
        for n_agents, n_entities, ob_space, cent_ob_space, ac_space in zip(num_agents_list, num_entities_list, obs_space_list, cent_obs_space_list, act_space_list):
            args_copy = deepcopy(args)
            args_copy.n_rollout_threads = num_thread_per_env
            self.buffer_lists.append(SkillSharedReplayBufferComm(args_copy, n_agents, n_entities, ob_space[0], cent_ob_space[0], ac_space[0]))

        self.num_thread_per_env = num_thread_per_env
        self.n_agents_list = num_agents_list
        self.n_entities_list = num_entities_list
        self.num_multi_envs = len(self.n_agents_list)

    def split_data(self, *data):
        data_lists = []
        for data_item in data:
            data_lists.append(np.split(data_item, self.num_multi_envs, axis=0))
        return data_lists

    def insert(self, idx, share_obs, obs, rnn_states_actor, rnn_states_comm, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, 
               bad_masks=None, active_masks=None, available_actions=None):
        self.buffer_lists[idx].insert(
            share_obs, obs, rnn_states_actor, rnn_states_comm, rnn_states_critic, actions, action_log_probs, 
            value_preds, rewards, masks, bad_masks, active_masks, available_actions)

    def insert_futures(self, idx, step, future_actions, future_pi_probs, future_available_actions):
        self.buffer_lists[idx].insert_futures(step, future_actions, future_pi_probs, future_available_actions)

    def after_update(self):
        for idx in range(len(self.n_agents_list)):
            self.buffer_lists[idx].after_update()

    def compute_returns(self, idx, next_value, value_normalizer=None):
        self.buffer_lists[idx].compute_returns(next_value, value_normalizer)
    
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].feed_forward_generator(env_advantages, num_mini_batch, mini_batch_size)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].naive_recurrent_generator(env_advantages, num_mini_batch)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_future_actions(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_agents(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_agents_with_future_actions(env_advantages, num_mini_batch, data_chunk_length)



class MultiEnvSharedReplayBuffer(SharedReplayBuffer):
    def __init__(self, args, 
                 num_agents_list, 
                 num_enemies_list, 
                 obs_space_list, 
                 cent_obs_space_list, 
                 act_space_list, 
                 num_thread_per_env):
        # a list of buffer for each environment
        self.buffer_lists = []
        for n_agents, n_enemies, ob_space, cent_ob_space, ac_space in zip(num_agents_list, num_enemies_list, obs_space_list, cent_obs_space_list, act_space_list):
            args_copy = deepcopy(args)
            args_copy.n_rollout_threads = num_thread_per_env
            
            if "RNN" in args_copy.experiment_name:
                self.buffer_lists.append(
                    RNN_Cognition_Buffer(args_copy, n_agents, n_enemies, ob_space[0], cent_ob_space[0], ac_space[0])
                )
            elif "subtask" in args_copy.experiment_name:
                self.buffer_lists.append(
                    subtask_Buffer(args_copy, n_agents, ob_space[0], cent_ob_space[0], ac_space[0])
                )
            else:
                self.buffer_lists.append(
                    SharedReplayBuffer(
                        args_copy, n_agents, ob_space[0], cent_ob_space[0], ac_space[0]
                    ) 
                ) 
        self.num_thread_per_env = num_thread_per_env
        self.n_agents_list = num_agents_list
        self.n_enemies_list = num_enemies_list
        self.num_multi_envs = len(self.n_agents_list)

    def split_data(self, *data):
        data_lists = []
        for data_item in data:
            data_lists.append(np.split(data_item, self.num_multi_envs, axis=0))
        return data_lists

    def insert(self, idx, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs, value_preds, rewards, masks, 
               bad_masks=None, active_masks=None, available_actions=None):
        self.buffer_lists[idx].insert(
            share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs, 
            value_preds, rewards, masks, bad_masks, active_masks, available_actions)

    def insert_futures(self, idx, step, future_actions, future_pi_probs, future_available_actions):
        self.buffer_lists[idx].insert_futures(step, future_actions, future_pi_probs, future_available_actions)

    def after_update(self):
        for idx in range(len(self.n_agents_list)):
            self.buffer_lists[idx].after_update()

    def compute_returns(self, idx, next_value, value_normalizer=None):
        self.buffer_lists[idx].compute_returns(next_value, value_normalizer)
    
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].feed_forward_generator(env_advantages, num_mini_batch, mini_batch_size)

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].naive_recurrent_generator(env_advantages, num_mini_batch)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_future_actions(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_agents(env_advantages, num_mini_batch, data_chunk_length)

    def recurrent_generator_with_agents_with_future_actions(self, advantages, num_mini_batch, data_chunk_length):
        for idx, env_advantages in enumerate(advantages):
            yield self.buffer_lists[idx].recurrent_generator_with_agents_with_future_actions(env_advantages, num_mini_batch, data_chunk_length)




