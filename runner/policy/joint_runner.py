"""
Joint MAPPO runner — multi-task simultaneous training with flat-obs MAPPO.

Reuses mcsETERunner's training loop but swaps in the flat-obs MAPPO policy
(no skills, no transformers in actor, no communication).
"""
from runner.policy.mcs_runner import mcsETERunner
from base_policy.algorithms.mappo.mappo_policy import mappoPolicy as Policy
from base_policy.algorithms.mcs.mcs_trainer import mcsTrainer as Trainer
from base_policy.utils.multi_envs_shared_buffer import MultiEnvSharedReplayBufferComm


class jointETERunner(mcsETERunner):
    def __init__(self, config):
        # Skip mcsETERunner.__init__ and call its parent directly,
        # then set up with MAPPO policy instead of MCS policy
        from runner.policy.base_runner import Runner
        Runner.__init__(self, config)

        self.use_sparse_reward = self.all_args.use_sparse_reward
        self.multi_envs = self.envs.multi_envs
        self.eval_multi_envs = self.eval_envs.multi_envs
        self.num_multi_envs = len(self.multi_envs)
        self.num_thread_per_env = self.envs.num_thread_per_env
        self.num_eval_thread_per_env = self.eval_envs.num_thread_per_env
        self.obs_space_list = self.envs.observation_space
        self.cent_obs_space_list = self.envs.share_observation_space
        self.act_space_list = self.envs.action_space
        self.quo = self.all_args.eval_episodes // (self.num_multi_envs * self.num_eval_thread_per_env)

        self.n_future_steps = self.all_args.n_future_steps
        self.use_action_predictor = self.all_args.use_action_predictor
        self.padded_values = self.all_args.padded_values

        self.eval_deterministic = self.all_args.eval_deterministic

        # MAPPO policy (flat obs, no skills)
        self.policy = Policy(self.all_args,
                             self.multi_envs,
                             self.num_thread_per_env,
                             self.envs.observation_space,
                             self.share_observation_space,
                             self.envs.action_space,
                             device=self.device)

        self.trainer = Trainer(self.all_args, self.policy, self.num_agents, self.num_enemies, self.num_entities, device=self.device)

        if self.model_dir != "None":
            self.restore()

        self.buffer = MultiEnvSharedReplayBufferComm(
            self.all_args, self.num_agents, self.num_entities,
            self.obs_space_list, self.cent_obs_space_list, self.act_space_list,
            self.num_thread_per_env
        )
