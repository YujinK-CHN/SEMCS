import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from gym.spaces import Discrete
import os
import cv2
import datetime
import argparse
from pathlib import Path
from PIL import Image

from .alicebob_unified_maps import get_alicebob_params, get_map_id, ID_TO_ONE_HOT, ALL_IDS, MAX_AGENTS, STARTID

ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
    4: (0, 0),   # stay (always allowed)
}

class AliceBob_Unified(object):
    def __init__(self, all_args, 
                 map_ids=None,
                 one_hot_map=None,
                 map_size=(10, 10),
                 n_actions=5,
                 seed=None):
        self.all_args = all_args
        self.use_single_state = all_args.use_single_state # return a single state for mixing network or not
        self.map_name = all_args.map_name
        self.task_id = all_args.task_id
        self.id_mode = all_args.id_mode
        map_params = get_alicebob_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_goals = map_params["n_goals"]
        self.n_keys = map_params["n_keys"]
        assert (self.n_keys == self.n_goals), "Number of keys and goals must be the same."        
        self.n_pairs = self.n_keys

        self.use_unified_obs = all_args.use_unified_obs  # use entity based NN or not 
        self.feat_dim = all_args.input_feat_dim

        ######### set seed and map #########
        if seed is None:
            self._seed = all_args.seed
        else:
            self._seed = seed
        np.random.seed(self._seed)
        if map_ids is None or one_hot_map is None:
            self.map_ids, self.one_hot_map = get_map_id(self.n_agents, self.n_goals, self.n_keys, self.id_mode, self._seed, self.task_id)
        else:
            self.map_ids = map_ids
            self.one_hot_map = one_hot_map
        # set color to ids
        self.id_to_color = self.build_id_color_map(self.map_ids)
        ######### set seed and map #########

        self.n_actions = n_actions
        self.n_entities = self.n_agents + self.n_goals + self.n_keys
        self.episode_limit = all_args.episode_length     # map_params["limit"]
        self.current_episode = 0
        self.length, self.width = map_size

        # map parameters
        self.use_render = all_args.use_render
        self.save_replay = all_args.save_replay
        self.goal_key_locations = all_args.goal_key_locations
        self.goal_reached_reward = 5
        self.per_goal_reached_reward = 1
        self.collision_penalty = -0.5
        self.step_penalty = -0.1
        self.vision = all_args.vision
        self.use_sparse_reward = all_args.use_sparse_reward  # decide if use bonus or not; 1: not use bonus, and 0: use bonus
        self.use_grids_obs = all_args.use_grids_obs
        self.goals_pos = None
        self.keys_pos = None
        if self.save_replay:
            self.replay_dir = Path(os.path.join(all_args.replay_dir, "replay", all_args.algorithm_name + "_" + all_args.experiment_name))
        else:
            self.replay_dir = Path(os.path.join(all_args.replay_dir, "videos", all_args.algorithm_name + "_" + all_args.experiment_name))
        # generate the map
        self.generate_map()

        # internal record information
        self.has_rewarded = [False] * self.n_pairs  # record whether a goal is rewarded or not
        self.out_video = []
        
        # action, observation, and state spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for _ in range(self.n_agents):
            self.action_space.append(Discrete(self.n_actions))
            self.observation_space.append([self.get_obs_size()])
            self.share_observation_space.append([self.get_state_size()])

    def generate_map(self):
        self.occupancy = np.zeros((self.length, self.width), dtype=int)
        # Enclose with walls (value = 1)
        self.occupancy[0, :] = 1
        self.occupancy[-1, :] = 1
        self.occupancy[:, 0] = 1
        self.occupancy[:, -1] = 1

        # goals reached
        self.goals_reach = [False] * self.n_pairs
        
        # ---- Place GOALS randomly along row 1 ----
        if self.goal_key_locations == "Shuffle":
            possible_goals = [[1, col] for col in range(1, self.width-1)]
            np.random.shuffle(possible_goals)
            self.goals_pos = possible_goals[: self.n_goals]
        elif self.goal_key_locations == "Fixed":
            if self.goals_pos is None:
                cols = np.linspace(1, self.width-2, self.n_goals, dtype=int)
                self.goals_pos = [[1, c] for c in cols]
        # assign to occupancy
        for i, (x, y) in enumerate(self.goals_pos):
            self.occupancy[x, y] = self.map_ids["goal"][i]

        # ---- Place KEYS randomly along row length-2 ----
        if self.goal_key_locations == "Shuffle":
            possible_keys = [[self.length-2, col] for col in range(1, self.width-1)]
            np.random.shuffle(possible_keys)
            self.keys_pos = possible_keys[: self.n_keys]
        elif self.goal_key_locations == "Fixed":
            if self.keys_pos is None:
                cols = np.linspace(1, self.width-2, self.n_keys, dtype=int)
                self.keys_pos = [[self.length-2, c] for c in cols]
        # assign to occupancy
        for i, (x, y) in enumerate(self.keys_pos):
            self.occupancy[x, y] = self.map_ids["key"][i]

        # ---- Place AGENTS randomly ----
        free_positions = [(i, j) for i in range(1, self.length-1) for j in range(1, self.width-1) if self.occupancy[i, j] == 0]
        np.random.shuffle(free_positions)
        self.agt_pos = []
        for i in range(self.n_agents):
            x, y = free_positions.pop()
            self.occupancy[x, y] = self.map_ids["agent"][i]
            self.agt_pos.append([x, y])

        # print("Occupancy map:\n", self.occupancy)


    def reset(self):
        # save replay
        if self.use_render and len(self.out_video)>0:
            self.render()  # render the last step
            gif_path = os.path.join(self.replay_dir, f"{self.task_id}_e{self.current_episode}.gif")
            self.save_gif(self.out_video, gif_path, fps=10)
            self.out_video.clear()

        self.has_rewarded = [False] * self.n_pairs  # reset rewarded goals
        self.out_video = []
        self._episode_steps = 0
        self.generate_map()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def step(self, action_list):
        action_list = [int(a) for a in action_list]
        self._episode_steps += 1
        # apply action to the occupancy
        for i in range(self.n_agents):
            action = action_list[i]
            avail_actions = self.get_avail_agent_actions(i)
            if avail_actions[action] == 1:
                dx, dy = ACTIONS[action]
                x, y = self.agt_pos[i]
                new_x = x + dx
                new_y = y + dy
                # Clear old position
                if self.occupancy[new_x][new_y] >= MAX_AGENTS+STARTID:
                    raise ValueError("Should override keys and goals.", self.occupancy)
                self.occupancy[x][y] = 0
                # Move agent
                self.agt_pos[i] = [new_x, new_y]
                self.occupancy[new_x][new_y] = self.map_ids['agent'][i]
            # else: invalid move, agent stays (do nothing)
        
        # --- Key / Goal checks ---
        self.goals_reach = self.check_keys_goals_reach()

        # assign goals and keys as 0 when goal reached (will this be a problem in observation)?
        for i, goal_re in enumerate(self.goals_reach):
            self.occupancy[self.keys_pos[i][0]][self.keys_pos[i][1]] = 0 if goal_re else self.map_ids['key'][i]
            self.occupancy[self.goals_pos[i][0]][self.goals_pos[i][1]] = 0 if goal_re else self.map_ids['goal'][i]

        done = False
        info_ = {}
        info_['won'] = False
        info_['bad_transition'] =  False
        info_['battle_won'] = 0
        reward = 0.0
        # check treasure
        if not done:
            if np.all(self.goals_reach):
                reward += self.goal_reached_reward
                done = True
                info_['won'] = True
                info_['battle_won'] += 1
            elif sum(self.goals_reach) > 0 and not self.use_sparse_reward:
                # reward only one time when reaching a goal
                for i, goal_re in enumerate(self.goals_reach):
                    if goal_re and not self.has_rewarded[i]:
                        reward += self.per_goal_reached_reward
                        self.has_rewarded[i] = True
            else:
                reward += self.step_penalty
        else:
            reward += 0
        # check collision
        positions = np.array(self.agt_pos)  # shape: (n_agents, 2)
        _, counts = np.unique(positions, axis=0, return_counts=True)
        collision_count = sum(c - 1 for c in counts if c > 1)  # accumulate the number of agents involved collision
        reward += self.collision_penalty * collision_count

        if self._episode_steps >= self.episode_limit:
            done = True
            self.current_episode += 1
            if not np.all(self.goals_reach):  # exceed time limit but goals not reached
                info_['bad_transition'] =  True  # reach limit

        info_['goals'] = sum(self.goals_reach)
    
        rewards = [[reward]]*self.n_agents
        dones = np.array([done] * self.n_agents)
        infos = [info_ for _ in range(self.n_agents)]

        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()


    def check_keys_goals_reach(self):
        agent_array = np.array(self.agt_pos)
        n_keys = len(self.keys_pos)
        n_goals = len(self.goals_pos)

        goals_reach = [False] * self.n_pairs  # one goal-key pair per index

        key_agents = [[] for _ in range(n_keys)]
        goal_agents = [[] for _ in range(n_goals)]

        for k, key_pos in enumerate(self.keys_pos):
            if self.goals_reach[k]:  # already reached — skip
                continue
            key = np.array(key_pos)
            for agent_idx, agent_pos in enumerate(agent_array):
                if np.abs(agent_pos - key).sum() <= 1:
                    key_agents[k].append(agent_idx)

        for g, goal_pos in enumerate(self.goals_pos):
            if self.goals_reach[g]:  # already reached — skip
                continue
            goal = np.array(goal_pos)
            for agent_idx, agent_pos in enumerate(agent_array):
                if np.abs(agent_pos - goal).sum() <= 1:
                    goal_agents[g].append(agent_idx)
                
        for i in range(self.n_pairs):
            if self.goals_reach[i]:  # already completed
                goals_reach[i] = True
                continue
            for ka in key_agents[i]:
                for ga in goal_agents[i]:
                    if ka != ga:
                        goals_reach[i] = True
                        break
                if goals_reach[i]:
                    break
        return goals_reach

    def get_unified_partial_obs(self, i):
        """
        observation as (entity, features)
        """
        x, y = self.agt_pos[i]
        h, w = self.occupancy.shape
        agent_pos = np.array([x, y])
        
        # agent own features
        # TODO: check if we can know the absolute position of the agent
        # TODO: check if we need partial observation with observation grids of agents
        agent_features = [0, 0] + self.one_hot_map["agent"][i]
        
        # Relative position to all the other agents
        other_features = []
        for j, pos in enumerate(self.agt_pos):
            if j != i:
                rel = agent_pos - np.array(pos)
                other_features.append(list(rel) + self.one_hot_map["agent"][j])
        
        # Relative distances to keys or goals (mask if reached)
        key_features = []
        goal_features = []
        for j in range(self.n_pairs):
            if not self.goals_reach[j]:
                rel = agent_pos - np.array(self.keys_pos[j])
                key_features.append(list(rel) + self.one_hot_map["key"][j])
                rel = agent_pos - np.array(self.goals_pos[j])
                goal_features.append(list(rel) + self.one_hot_map["goal"][j])
            else:
                key_features.append([0, 0] + list(np.zeros_like(self.one_hot_map["key"][j])))
                goal_features.append([0, 0] + list(np.zeros_like(self.one_hot_map["goal"][j])))

        if self.use_grids_obs:
            grids_features = []
            for dx in range(-self.vision, self.vision + 1):
                for dy in range(-self.vision, self.vision + 1):
                    if dx == 0 and dy == 0:
                        continue  # skip agent's own position
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        ent_id = self.occupancy[nx, ny]
                        one_hot = ID_TO_ONE_HOT.get(ent_id, [0] * len(ALL_IDS))
                    else:
                        one_hot = [0] * len(ALL_IDS)  # out of bounds
                    grids_features.append([dx, dy] + one_hot)
            # Combine everything
            return np.concatenate([
                np.array(agent_features).flatten(),
                np.array(other_features).flatten(),
                np.array(key_features).flatten(),
                np.array(goal_features).flatten(),
                np.array(grids_features).flatten()
            ])
        else:
            # Combine everything
            return np.concatenate([
                np.array(agent_features).flatten(),
                np.array(other_features).flatten(),
                np.array(key_features).flatten(),
                np.array(goal_features).flatten()
            ])

    def get_partial_obs(self, i):
        """
        instead of using numbers for each entity. So the data shape becomes (width, height, 1)
        """
        x, y = self.agt_pos[i]
        agent_pos = np.array([x, y])
        
        # 3x3 surrounding env
        partial_obs = self.occupancy[self.agt_pos[i][0]-1:self.agt_pos[i][0]+2,self.agt_pos[i][1]-1:self.agt_pos[i][1]+2].reshape(1,-1)

        # Relative position to all the other agents
        other_features = []
        for j, pos in enumerate(self.agt_pos):
            if j != i:
                rel = agent_pos - np.array(pos)
                other_features.append(rel)

        # Relative distances to keys or goals (mask if reached)
        key_features = []
        goal_features = []
        for j in range(self.n_pairs):
            if not self.goals_reach[j]:
                rel = agent_pos - np.array(self.keys_pos[j])
                key_features.append(list(rel))
                rel = agent_pos - np.array(self.goals_pos[j])
                goal_features.append(list(rel))
            else:
                key_features.append([0, 0])
                goal_features.append([0, 0])

        return np.concatenate([np.squeeze(partial_obs), self.agt_pos[i], np.array(other_features).flatten(), np.array(key_features).flatten(), np.array(goal_features).flatten()])
        

    def get_unified_state(self, i):
        """
        observation as (entity, features)
        """
        x, y = self.agt_pos[i]
        h, w = self.occupancy.shape
        agent_pos = np.array([x, y])
        
        # agent own features
        # check if we can know the absolute position of the agent
        agent_features = list(agent_pos) + self.one_hot_map["agent"][i]

        # Relative position to all the other agents
        other_features = []
        for j, pos in enumerate(self.agt_pos):
            if j != i:
                other_features.append(list(pos) + self.one_hot_map["agent"][j])

        # positions of keys or goals (mask if reached)
        key_features = []
        goal_features = []
        for j in range(self.n_pairs):
            if not self.goals_reach[j]:
                key_features.append(self.keys_pos[j] + self.one_hot_map["key"][j])
                goal_features.append(self.goals_pos[j] + self.one_hot_map["goal"][j])
            else:
                key_features.append([0, 0] + list(np.zeros_like(self.one_hot_map["key"][j]))) 
                goal_features.append([0, 0] + list(np.zeros_like(self.one_hot_map["goal"][j])))

        if self.use_grids_obs:
            grids_features = []
            for dx in range(-self.vision, self.vision + 1):
                for dy in range(-self.vision, self.vision + 1):
                    if dx == 0 and dy == 0:
                        continue  # skip agent's own position
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w:
                        ent_id = self.occupancy[nx, ny]
                        one_hot = ID_TO_ONE_HOT.get(ent_id, [0] * len(ALL_IDS))
                    else:
                        one_hot = [0] * len(ALL_IDS)  # out of bounds
                    grids_features.append([nx, ny] + one_hot)
                        # Combine everything
            return np.concatenate([
                np.array(agent_features).flatten(),
                np.array(other_features).flatten(),
                np.array(key_features).flatten(),
                np.array(goal_features).flatten(),
                np.array(grids_features).flatten()
            ])
        else:
            # Combine everything
            return np.concatenate([
                np.array(agent_features).flatten(),
                np.array(other_features).flatten(),
                np.array(key_features).flatten(),
                np.array(goal_features).flatten()
            ])

    def get_obs(self):
        if self.use_unified_obs:
            return [self.get_unified_partial_obs(i) for i in range(self.n_agents)]
        else:
            return [self.get_partial_obs(i) for i in range(self.n_agents)]
    

    def get_state(self):
        if self.use_single_state:  # used for mix networks
            if self.use_unified_obs:
                return np.array([self.get_unified_state(i) for i in range(self.n_agents)]).flatten()
            else:
                return np.array([self.get_partial_obs(i) for i in range(self.n_agents)]).flatten()
        else:
            if self.use_unified_obs:
                return [self.get_unified_state(i) for i in range(self.n_agents)]
            else:
                return [self.get_partial_obs(i) for i in range(self.n_agents)]


    def get_obs_size(self):
        if self.use_unified_obs: 
            return self.get_unified_partial_obs(0).shape[0]
        else:
            return self.get_partial_obs(0).shape[0]


    def get_state_size(self):
        if self.use_single_state:
            return self.get_state().shape[0]
        else:
            return self.get_state()[0].shape[0]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """
        Returns a list of available actions (0–4) for the agent,
        excluding directions that lead into walls (ID == 1).
        """
        x, y = self.agt_pos[agent_id]
        avail_actions = [0]*len(ACTIONS)

        for action, (dx, dy) in ACTIONS.items():
            nx, ny = x + dx, y + dy
            # Always allow "stay"
            if action == 4:
                if self.occupancy[x, y] < MAX_AGENTS + STARTID:
                    avail_actions[action] = 1
            # Check within bounds and avoid walls
            elif 0 <= nx < self.length and 0 <= ny < self.width:
                if self.occupancy[nx, ny] != 1 and self.occupancy[nx, ny] < MAX_AGENTS+STARTID:  # 1 = wall, >= MAX_AGENTS+STARTID means goals or keys
                    avail_actions[action] = 1
        return avail_actions


    def close(self):
        self.out_video = None
        self.occupancy = None


    def render(self, mode="human"):
        enlarge = 30
        h, w = self.occupancy.shape
        canvas = np.full((h*enlarge, w*enlarge, 3), 255, dtype=np.uint8)
        # print("goal:", self.goals_reach, self.occupancy)
        for i in range(h):
            for j in range(w):
                ent_id = self.occupancy[i, j]
                if ent_id == 0:
                    continue
                color_bgr = tuple(self.id_to_color.get(ent_id, [0,0,0])[::-1])
                tl, br = (j*enlarge, i*enlarge), ((j+1)*enlarge, (i+1)*enlarge)
                cv2.rectangle(canvas, tl, br, color_bgr, -1)

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, f"{ts} ep{self.current_episode} st{self._episode_steps}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        self.out_video.append(canvas)


    def save_gif(self, frames, path, fps=30):
        """
        frames: list of H×W×3 uint8 NumPy arrays
        path:   where to write .gif
        fps:    frames per second
        """
        # 1) Convert each RGB array to a paletted PIL Image
        pil_frames = []
        for arr in frames:
            im = Image.fromarray(arr)                   # RGB
            im_p = im.convert('P', palette=Image.ADAPTIVE, colors=256)
            pil_frames.append(im_p)

        # 2) Ensure output dir exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 3) Save with full-frame disposal (disposal=2) so nothing is left transparent
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,                                # infinite loop
            duration=int(1000 / fps),             # ms per frame
            disposal=2
        )
        # print(f"Wrote GIF → {path}")

    def build_id_color_map(self, entity_ids):
        color_map = {}
        n_agents = len(entity_ids["agent"])
        
        # 1) agents: red → magenta
        for i, a_id in enumerate(entity_ids["agent"]):
            t = i / max(n_agents - 1, 1)
            color_map[a_id] = [255, 0, int(255 * t)]
        
        # 2) key–goal pairs: green→cyan (just as an example)
        #    both key and its matching goal get the *same* color
        for i, (k_id, g_id) in enumerate(zip(entity_ids["key"], entity_ids["goal"])):
            t = i / max(self.n_pairs - 1, 1)
            # e.g. vary blue channel from 0→255, keep green=255
            shared_color = [0, 255, int(255 * t)]
            color_map[k_id] = shared_color
            color_map[g_id] = shared_color
        
        return color_map


    def pick_spread_columns(self, n, min_col, max_col):
        """
        Pick `n` integer columns in [min_col…max_col] so that they're as far apart
        as possible (greedy farthest‐point sampling).
        Returns a sorted list of length n.
        """
        possible = list(range(min_col, max_col+1))
        assert n <= len(possible)
        
        # 1) seed with one extreme (leftmost)
        picks = [possible[0]]
        
        # 2) for each further pick, choose the column whose
        #    min distance to existing picks is largest
        for _ in range(1, n):
            # compute for each candidate its min distance to picks
            dists = [min(abs(c - p) for p in picks) for c in possible]
            # find the candidate with max of those dists
            idx = int(np.argmax(dists))
            picks.append(possible[idx])
        
        return sorted(picks)
