import random

import gfootball.env as football_env
from gym import spaces
import numpy as np

from .football_maps import get_football_params

class FootballUnifiedEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args):
        self.args = args
        self.map_name = args.map_name
        self.map_params = get_football_params(args.map_name)
        self.episode_limit = self.map_params["limit"]
        self.num_agents = self.map_params["n_agents"]
        self.use_unified_obs = args.use_unified_obs
        self.use_single_state = args.use_single_state
        self.unifed_obs_dim = args.unifed_obs_dim

        # make env
        if not (args.use_render and args.save_videos):
            self.env = football_env.create_environment(
                env_name=args.map_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=self.num_agents,
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                render=(args.use_render and args.save_gifs)
            )
        else:
            # render env and save videos
            self.env = football_env.create_environment(
                env_name=args.map_name,
                stacked=args.use_stacked_frames,
                representation=args.representation,
                rewards=args.rewards,
                number_of_left_players_agent_controls=self.num_agents,   # control left players
                number_of_right_players_agent_controls=0,
                channel_dimensions=(args.smm_width, args.smm_height),
                # video related params
                write_full_episode_dumps=True,
                render=True,
                write_video=True,
                dump_frequency=1,
                logdir=args.video_dir
            )

        self.max_steps = self.env.unwrapped.observation()[0]["steps_left"]
        self.remove_redundancy = args.remove_redundancy
        self.zero_feature = args.zero_feature
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        if self.num_agents == 1:
            self.action_space.append(self.env.action_space)
            self.observation_space.append(self.env.observation_space)
            self.share_observation_space.append(self.env.observation_space)
            self.n_actions = self.action_space[0].n
        else:
            for idx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(
                    n=self.env.action_space[idx].n
                ))
                if self.use_unified_obs:
                    self.observation_space.append([self.unifed_obs_dim])
                    if self.use_single_state:
                        self.share_observation_space.append([self.unifed_obs_dim*self.num_agents])
                    else:
                        self.share_observation_space.append([self.unifed_obs_dim])
                else:
                    obs_space = spaces.Box(low=self.env.observation_space.low[idx],
                            high=self.env.observation_space.high[idx],
                            shape=self.env.observation_space.shape[1:],
                            dtype=self.env.observation_space.dtype)
                    if self.use_single_state:
                        obs_shape = np.shape(obs_space)[0]
                        self.observation_space.append([obs_shape])
                        self.share_observation_space.append([obs_shape * self.num_agents])
                    else:
                        self.observation_space.append(obs_space)
                        self.share_observation_space.append(obs_space) 
                                
        self.n_actions = self.action_space[0].n
                
        # record stats
        self.win_rate = 0
        self.step_taken = 0
        self.goal_record = 0

    def get_unified_obs(self, obs):
        """
        unify the observations as:
            Row 1: left positions (22); ball pos (3) if owned by left (ball_ownership[1]==1), else zeros; active player (11); game mode (7)
            Row 2: left directions (22); ball dir (3) if owned by left (ball_ownership[1]==1), else zeros; active player (11); game mode (7)
            Row 3: right positions (22); ball pos (3) if owned by right (ball_ownership[2]==1), else zeros 11 zeros for active player; game mode (7)
            Row 4: right directions (22); ball dir (3) if owned by right (ball_ownership[2]==1), else zeros 11 zeros for active player; game mode (7)
        """
        unified_obs = []
        for per_o in obs:
            left_pos = per_o[0:22]
            left_dir = per_o[22:44]
            right_pos = per_o[44:66]
            right_dir = per_o[66:88]
            ball_pos = per_o[88:91]
            ball_dir = per_o[91:94]
            ball_ownership = per_o[94:97]
            active = per_o[97:108]
            game_mode = per_o[108:115]
            
            left_owns_ball = ball_ownership[1] == 1
            right_owns_ball = ball_ownership[2] == 1
            
            ball_pos_left = ball_pos if left_owns_ball else np.zeros(3)
            ball_dir_left = ball_dir if left_owns_ball else np.zeros(3)
            
            ball_pos_right = ball_pos if right_owns_ball else np.zeros(3)
            ball_dir_right = ball_dir if right_owns_ball else np.zeros(3)
            
            type1 = np.concatenate([left_pos, ball_pos_left, active, game_mode])
            type2 = np.concatenate([left_dir, ball_dir_left, active, game_mode])
            type3 = np.concatenate([right_pos, ball_pos_right, np.zeros(11), game_mode])
            type4 = np.concatenate([right_dir, ball_dir_right, np.zeros(11), game_mode])
            
            stack_obs = np.concatenate([type1, type2, type3, type4])
            unified_obs.append(stack_obs)
        return unified_obs        

    def reset(self):
        obs = self.env.reset()
        obs = self._obs_wrapper(obs)
        
        # record for each epsiode
        self.win_rate = 0
        self.goal_record = 0
        
        avail_actions = self.get_avail_actions()
        
        if self.use_unified_obs:
            obs = self.get_unified_obs(obs)
        
        shared_obs = self.get_state(obs)
        
        return obs, shared_obs, avail_actions

    def get_state(self, obs):
        if self.use_single_state:
            return np.array(obs).flatten()
        else:
            return obs

    def step(self, action):
        action_int = [int(act) for act in action]
        obs, reward, done, info = self.env.step(action_int)
        obs = self._obs_wrapper(obs)
        
        if self.use_unified_obs:
            obs = self.get_unified_obs(obs)
        
        shared_obs = self.get_state(obs)

        reward = reward.reshape(self.num_agents, 1)
        if self.share_reward:
            global_reward = np.sum(reward)
            reward = [[global_reward]] * self.num_agents

        done = np.array([done] * self.num_agents)
        
        info = self._info_wrapper(info)
        
        info['bad_transition'] =  False
        if info["max_steps"] - info["steps_left"] >= self.episode_limit:
            if not np.all(done):
                info['bad_transition'] =  True  # reach limit
                
        # record stats
        self.goal_record += info["score_reward"]
        if info["score_reward"] > 0:
            self.win_rate += 1
        step_taken = info["max_steps"] - info["steps_left"]

        info = {
            "score_reward": info["score_reward"],
            "goal": self.goal_record,
            "step": step_taken,
            "win_rate": self.win_rate,
            "bad_transition": info["bad_transition"]
        }
        info_list = [info for _ in range(self.num_agents)]

        availabel_actions = self.get_avail_actions()
        
        return obs, shared_obs, reward, done, info_list, availabel_actions
                    
                    
    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

    def _info_wrapper(self, info):
        state = self.env.unwrapped.observation()
        info.update(state[0])
        info["max_steps"] = self.max_steps
        info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info

    def get_stats(self):
        return {}

    def get_state_size(self):
        return self.get_obs_size() * self.num_agents

    def get_obs_size(self):
        return np.shape(self.observation_space[0])[0]

    def get_avail_actions(self):
        return [np.ones(self.n_actions, dtype=np.float32) for i in range(self.num_agents)]
        
    def get_env_info(self):
        return {
            "episode_limit": self.map_params["limit"],
            "n_agents": self.num_agents,
            "n_actions": self.n_actions,
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size()
        }
