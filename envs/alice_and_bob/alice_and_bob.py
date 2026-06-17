import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from gym.spaces import Discrete
import os
from pathlib import Path
import datetime

from .alicebob_unified_maps import get_alicebob_params, get_map_id


class AliceBob(object):
    def __init__(self, all_args, 
                 map_ids=None,
                 one_hot_map=None,
                 map_size=(10, 10),
                 n_actions=5):
        self.all_args = all_args
        self.map_name = all_args.map_name
        self.use_grids_state = all_args.use_grids_state
        self.replay_dir = os.path.join(all_args.replay_dir, "videos", all_args.algorithm_name + "_" + all_args.experiment_name)
        self.id_mode = all_args.id_mode
        self.map_ids = map_ids
        self.one_hot_map = one_hot_map
        self._seed = all_args.seed
        
        map_params = get_alicebob_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_goals = map_params["n_goals"]
        self.n_keys = map_params["n_keys"]

        assert (self.n_keys == self.n_goals) and self.n_agents==2, "Number of keys and goals must be the same and only 2 agents are supported now."        

        self.n_actions = n_actions
        self.n_entities = self.n_agents + self.n_goals + self.n_keys
        self.episode_limit = map_params["limit"]
        self.length, self.width = map_size

        # generate the map
        self.generate_map()
        
        # internal record information
        self.has_rewarded = [False, False]  # record whether a goal is rewarded or not
        self.battles_won = 0
        self.out_video = []
        
        # action, observation, and state spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for _ in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append([self.get_obs_size()])
            self.share_observation_space.append([self.get_state_size()])
    
    def generate_map_obs(self):
        occupancy_obs = np.zeros((self.length, self.width, 3))
        occupancy_obs[self.occupancy==0] = [1,1,1]
        occupancy_obs[self.occupancy==2] = [1,0,0]
        occupancy_obs[self.occupancy==3] = [0,0,1]
        occupancy_obs[self.occupancy==4] = [0,1,0]
        occupancy_obs[self.occupancy==5] = [1,1,0]
        occupancy_obs[self.occupancy==6] = [0,1,1]
        occupancy_obs[self.occupancy==7] = [1,0,1]
        return occupancy_obs

    def generate_map(self):
        self.occupancy = np.zeros((self.length, self.width))
        
        # enclose the surroundings
        for i in range(self.length):
            self.occupancy[i, 0] = 1
            self.occupancy[i, self.width - 1] = 1
        for i in range(self.width):
            self.occupancy[0, i] = 1
            self.occupancy[self.length -1, i] = 1

        # generate keys and goals
        self.keys_pos = [[1, self.width-2], [1, 1]]
        self.goals_pos = [[self.length-2, 1], [self.length-2, self.width-2]]
        self.keys_in_use = [False, False]
        self.goals_reach = [False, False]
        self.occupancy[self.keys_pos[0][0]][self.keys_pos[0][1]] = 4  # key 1
        self.occupancy[self.keys_pos[1][0]][self.keys_pos[1][1]] = 5  # key 2
        self.occupancy[self.goals_pos[0][0]][self.goals_pos[0][1]] = 6  # goal 1
        self.occupancy[self.goals_pos[1][0]][self.goals_pos[1][1]] = 7  # goal 2

        # initialize agents
        self.agt_pos = []
        ll, ww = self.length - 2, self.width - 2
        init_pos = np.random.choice(ll * ww, 2, replace=False)
        self.agt_pos.append([init_pos[0] // ww + 1, init_pos[0] % ww + 1])
        self.agt_pos.append([init_pos[1] // ww + 1, init_pos[1] % ww + 1])
        self.occupancy[self.agt_pos[0][0]][self.agt_pos[0][1]] = 2
        self.occupancy[self.agt_pos[1][0]][self.agt_pos[1][1]] = 3

        self.occupancy_obs = self.generate_map_obs()

    def reset(self):
        self.has_rewarded = [False, False]  # reset rewarded goals
        self._episode_steps = 0
        self.generate_map()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def step(self, action_list):
        action_list = [int(a) for a in action_list]
        self._episode_steps += 1
        # agent move
        for i in range(self.agent_num):
            if action_list[i] == 0:  # move up
                if self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]+1] != 1:  # if can move
                    self.agt_pos[i][1] = self.agt_pos[i][1] + 1
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]-1] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]
            elif action_list[i] == 1:  # move down
                if self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]-1] != 1:  # if can move
                    self.agt_pos[i][1] = self.agt_pos[i][1] - 1
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]+1] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]
            elif action_list[i] == 2:  # move left
                if self.occupancy[self.agt_pos[i][0]-1][self.agt_pos[i][1]] != 1:  # if can move
                    self.agt_pos[i][0] = self.agt_pos[i][0] - 1
                    self.occupancy[self.agt_pos[i][0]+1][self.agt_pos[i][1]] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]
            elif action_list[i] == 3:  # move right
                if self.occupancy[self.agt_pos[i][0]+1][self.agt_pos[i][1]] != 1:  # if can move
                    self.agt_pos[i][0] = self.agt_pos[i][0] + 1
                    self.occupancy[self.agt_pos[i][0]-1][self.agt_pos[i][1]] = 0
                    self.occupancy[self.agt_pos[i][0]][self.agt_pos[i][1]] = self.map['agent'][i]

        # check keys
        for i in range(len(self.keys_pos)):
            if self.keys_pos[i] in self.agt_pos:
                self.keys_in_use[i] = True
            else:
                self.keys_in_use[i] = False
        
        # check goals
        for i in range(len(self.keys_pos)):
            if self.keys_in_use[i] and self.goals_pos[i] in self.agt_pos:  # the same order of keys and goals in use
                self.goals_reach[i] = True
        
        for i in range(len(self.goals_pos)):
            if self.goals_pos[i] not in self.agt_pos:
                # set the goal as 0 only after the agent leave the goal position, otherwise the number will be the agent itself
                #   --> the goal is possessed by the agent, so the number will be changed as agent number
                if self.goals_reach[i]:
                    self.occupancy[self.goals_pos[i][0]][self.goals_pos[i][1]] = 0
                else:
                    self.occupancy[self.goals_pos[i][0]][self.goals_pos[i][1]] = self.map['goal'][i]
        
        for i in range(len(self.keys_pos)):
            if not self.keys_in_use[i]:
                self.occupancy[self.keys_pos[i][0]][self.keys_pos[i][1]] = self.map['key'][i]

        done = False
        info_ = {}
        info_['won'] = False
        info_['bad_transition'] =  False
        reward = 0.0
        # check treasure
        if np.all(self.goals_reach):
            reward = 2.0
            done = True
            info_['won'] = True
            self.battles_won += 1
            info_['battles_won'] = self.battles_won
        elif sum(self.goals_reach) > 0:
            # reward only one time when reaching a goal
            reward = 0.0
            for i, goal_re in enumerate(self.goals_reach):
                if goal_re:
                    if not self.has_rewarded[i]:
                        reward += 0.5
                        self.has_rewarded[i] = True
        else:
            reward = 0.0

        if self._episode_steps >= self.episode_limit:
            done = True
            if not np.all(self.goals_reach):  # exceed time limit but goals not reached
                info_['bad_transition'] =  True  # reach limit

        info_['goal1'] = int(self.goals_reach[0])
        info_['goal2'] = int(self.goals_reach[1])

        rewards = [[reward]]*self.agent_num
        dones = np.array([done] * self.agent_num)
        infos = [info_ for _ in range(self.agent_num)]

        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    def get_global_obs(self):
        return self.generate_map_obs()

    def get_agt_obs(self, i):
        """
        instead of using numbers as used in get_partial_obs, get_agt_obs uses one-hot vector for each entity. So the data shape becomes (width, height, 3)
        """
        obs = self.generate_map_obs()[self.agt_pos[i][0]-1:self.agt_pos[i][0]+2,self.agt_pos[i][1]-1:self.agt_pos[i][1]+2]
        return obs

    def get_all_agt_obs(self):
        return [self.get_agt_obs(i) for i in range(self.agent_num)]

    def get_partial_obs(self, i):
        """
        instead of using numbers for each entity. So the data shape becomes (width, height, 1)
        """
        # 3x3 surrounding env
        partial_obs = self.occupancy[self.agt_pos[i][0]-1:self.agt_pos[i][0]+2,self.agt_pos[i][1]-1:self.agt_pos[i][1]+2].reshape(1,-1)

        # dis from agent to keys
        rel_agt_land_dis = np.zeros((2,2))
        for j in range(len(self.keys_pos)):
            rel_agt_land_dis[j] = np.array(self.agt_pos[i]) - np.array(self.keys_pos[j])
        
        # invisible to goal that has been reached
        rel_dis = np.zeros((2,2))
        for j in range(len(self.goals_pos)):
            if not self.goals_reach[j]:
                rel_dis[j] = np.array(self.agt_pos[i]) - np.array(self.goals_pos[j])

        # relative distance from agent to the ohter
        other_pos = np.array(self.agt_pos[i])-np.array(self.agt_pos[1-i])

        # use richer obsrevations which includes agent 1's partial observation (range 1), it's own position, other_pos, distance to keys and goals
        # return np.concatenate([np.squeeze(partial_obs), self.agt_pos[i], other_pos, rel_agt_land_dis.flatten(), rel_dis.flatten()])
        # return np.concatenate([np.squeeze(partial_obs), self.agt_pos[i]])
        return np.concatenate([np.squeeze(partial_obs), self.agt_pos[i], rel_agt_land_dis.flatten(), rel_dis.flatten()])

    def get_obs(self):
        return [self.get_partial_obs(i) for i in range(self.agent_num)]
    
    def get_obs_size(self):
        return self.get_partial_obs(0).shape[0]

    def get_state_agent(self, agent_id):
        if self.use_grids_state: 
            return np.concatenate([line_state for line_state in self.occupancy], axis = 0)
        else:
            # use concatenate observations of each agent instead of the full grids
            return np.concatenate(self.get_obs())

    def get_state(self):
        return [self.get_state_agent(i) for i in range(self.agent_num)]

    def get_state_size(self):
        # use concatenation of observations
        return self.get_state_agent(0).shape[0]

    def plot_scene(self, idx):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        plt.xticks([])
        plt.yticks([])
        ax2 = fig.add_subplot(gs[2, 0:1])
        plt.xticks([])
        plt.yticks([])
        ax3 = fig.add_subplot(gs[2, 1:2])
        plt.xticks([])
        plt.yticks([])

        ax1.imshow(self.get_global_obs())
        ax2.imshow(self.get_agt_obs(0))
        ax3.imshow(self.get_agt_obs(1))
        plt.savefig('./alice_and_bob/images/step_{}'.format(idx))
        plt.clf()

    def color(self, loc):
        loc = loc.astype('int32')
        ## keys
        if (loc == [0,1,0]).all():  # key 1
            color = [0, 255, 0]  # green
        elif (loc == [1,1,0]).all():  # key 2
            color = [255, 255, 0]  # cyan
        ## goals
        elif (loc == [0,1,1]).all():  # goal 1
            color = [0, 255, 0]  # yellow 0, 255, 225 --> green 0, 255, 0
        elif (loc == [1,0,1]).all():  # goal 2
            color = [255, 255, 0]  # magenta 255, 0, 255 --> cyan 255, 255, 0
        ## agents
        elif (loc == [1,0,0]).all(): # agent 1
            color = [255, 0, 0]  # blue
        elif (loc == [0,0,1]).all(): # agnent 2
            color = [0, 0, 255]  # red
        ## others
        else:
            color = [255, 255, 255]  # white
        color = np.array(color, dtype=float)
        return color
    
    def render(self, mode="human"):
        obs = self.get_global_obs()
        enlarge = 30
        new_obs = np.zeros((self.length*enlarge, self.width*enlarge, 3))
        for i in range(self.length):
            for j in range(self.width):
                if np.sum(obs[i][j]) > 0:
                    new_obs = cv2.rectangle(new_obs, (j * enlarge, i * enlarge), (j * enlarge + enlarge, i * enlarge + enlarge), self.color(obs[i][j])[::-1], -1)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(new_obs, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        self.out_video.append(new_obs)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.agent_num)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.n_actions

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            _seed = self._seed
        else:
            _seed = seed
        np.random.seed(_seed)
        if self.map_ids is None or self.one_hot_map is None:
            self.map_ids, self.one_hot_map = get_map_id(self.n_agents, self.n_goals, self.n_keys, self.id_mode, _seed)

    def render_video(self, key):
        alg_setting_name = self.all_args.algorithm_name + "_" + self.all_args.experiment_name
        img_dir = Path(self.video_dir) / "videos" / alg_setting_name
        if not img_dir.exists():
            os.makedirs(str(img_dir))
        enlarge = 30
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = str(img_dir) + "/" + key + '.mp4'
        video = cv2.VideoWriter(video_path, codec, 30, (self.length*enlarge, self.width*enlarge), True)
        for img in self.out_video:
            img = np.array(img).astype('uint8')
            video.write(img)  
        video.release()
        cv2.destroyAllWindows() 
        
        # reset video array
        self.out_video = []
