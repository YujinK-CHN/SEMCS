import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
import json


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # common parameters
        self.num_agents = self.envs.num_agents  # from environments
        self.num_enemies = self.envs.num_enemies  # from environments
        self.num_entities = self.envs.num_entities  # from environments
        self.eval_num_agents = self.eval_envs.num_agents  # from environments
        self.eval_num_enemies = self.eval_envs.num_enemies  # from environments
        self.eval_num_entities= self.eval_envs.num_entities  # from environments
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.render_episodes = self.all_args.render_episodes
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = self.all_args.run_dir
            self.log_dir = os.path.join(self.run_dir, 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = os.path.join(self.run_dir, 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            # save hyperparameters
            with open(os.path.join(str(self.run_dir), "args.json"), "wt") as f:
                json.dump(vars(self.all_args), f, indent=4) # Indent 4 spaces in JSON format 
        
        # save intermidiate data  
        self.trajectory_dir = os.path.join(self.run_dir, 'trajectory')
        if not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)

        self.share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V \
            else self.envs.observation_space[0]

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError


    def warmup(self):
        # reset env
        obs_tuple, share_obs_tuple, available_actions_tuple, idxs_tuple = self.envs.reset()

        for obs, share_obs, available_actions, idx in zip(obs_tuple, share_obs_tuple, available_actions_tuple, idxs_tuple):

            # replay buffer
            if not self.use_centralized_V:
                share_obs = obs

            self.buffer.buffer_lists[idx].share_obs[0] = share_obs.copy()
            self.buffer.buffer_lists[idx].obs[0] = obs.copy()
            self.buffer.buffer_lists[idx].available_actions[0] = available_actions.copy()


    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data, idx):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        raise NotImplementedError


    def train(self, num_episodes):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, num_episodes)     
        self.buffer.after_update()
        return train_infos


    def save(self):
        """Save policy's actor and critic networks."""
        if "mat" in self.algorithm_name:
            self.policy.save(self.save_dir)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
            if self.trainer._use_valuenorm:
                policy_vnorm = self.trainer.value_normalizer
                torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def evaluate4replay(self):
        if "StarCraft" in self.env_name:
            self.eval(total_num_steps=0, episode=0)
            if self.all_args.save_replay:
                print("Save replay for SMAC.....")
                self.eval_envs.save_replay()
        elif "AliceBob" in self.env_name:
            print("Render for AliceBob.....")
            self.eval(total_num_steps=0, episode=0)

    def restore(self):
        """Restore policy's networks from a saved model."""
        if "mat" in self.algorithm_name:
            self.policy.restore(self.model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict, strict=False)

    def log_train(self, train_infos, total_num_steps):
        """
        Log testing info.
        :param train_infos: (dict) information about training update.
            {
                {
                    info: {per_metric: value},
                }
            }
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k:v}, total_num_steps)

    def log_eval(self, eval_infos, total_num_steps):
        """
        Log eval info.
        :param eval_infos: (dict) information about eval envs.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in eval_infos.items():
            if self.use_wandb:
                wandb.log({k: np.mean(v)}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
