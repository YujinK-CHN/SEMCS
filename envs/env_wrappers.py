"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
import multiprocessing
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from utils.util import tile_images
import pickle
import logging
import os
import traceback

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class DillWrapper(object):
    """
    Uses DillWrapper to serialize contents (otherwise multiprocessing tries to use pickle) when there is weakref
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import dill
        return dill.dumps(self.x)

    def __setstate__(self, ob):
        import dill
        self.x = dill.loads(ob)



class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
            imgs = self.get_images()
            bigimg = tile_images(imgs)
            if mode == 'human':
                self.get_viewer().imshow(bigimg)
                return self.get_viewer().isopen
            elif mode == 'rgb_array':
                return bigimg
            else:
                raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class GuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # could cause zombie process
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)


    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            try:
                received_data = remote.recv()        
                if isinstance(received_data, tuple):
                    cmd, data = received_data
                else:
                    # print("Received pickle data!")
                    deserialized_data = pickle.loads(received_data)
                    cmd, data = deserialized_data
                if cmd == 'step':
                    ob, s_ob, reward, done, info, available_actions = env.step(data)
                    if 'bool' in done.__class__.__name__:
                        if done:
                            ob, s_ob, available_actions = env.reset()
                    else:
                        if np.all(done):
                            ob, s_ob, available_actions = env.reset()
                    remote.send((ob, s_ob, reward, done, info, available_actions))
                elif cmd == 'reset':
                    ob, s_ob, available_actions = env.reset()
                    remote.send((ob, s_ob, available_actions))
                elif cmd == 'reset_task':
                    ob = env.reset_task()
                    remote.send(ob)
                elif cmd == 'render':
                    if data == "rgb_array":
                        fr = env.render(mode=data)
                        remote.send(fr)
                    elif data == "human":
                        env.render(mode=data)
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
                elif cmd == 'get_spaces':
                    remote.send(
                        (env.observation_space, env.share_observation_space, env.action_space))
                elif cmd == 'render_vulnerability':
                    fr = env.render_vulnerability(data)
                    remote.send((fr))
                elif cmd == 'get_num_agents':
                    remote.send((env.n_agents))
                elif cmd == 'get_avail_actions':
                    remote.send(env.get_avail_actions())
                else:
                    raise NotImplementedError
            except Exception as e:
                tb = traceback.format_exc()
                print("exception:", tb)
                remote.send(("exception", (str(e), tb)))
                # remote.close()
                break
    except Exception as e:
        tb = traceback.format_exc()
        print("exception:", tb)
        remote.send(("exception", (str(e), tb)))
        # remote.close()


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, num_agents=None, num_enemies=None, num_entities=None, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
            
        # logger = multiprocessing.log_to_stderr(logging.INFO)

        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.num_entities = num_entities

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
                
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def get_avail_actions(self):
        for remote in self.remotes:
            remote.send(('get_avail_actions', None))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return np.stack(results)

    def step_async(self, actions):
        if self.closed:
            raise RuntimeError("Cannot call step_async() on a closed environment.")

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True


    def step_wait(self):
        if not self.waiting:
            raise RuntimeError("step_wait called without calling step_async first.")

        results = []
        for i, remote in enumerate(self.remotes):
            res = remote.recv()
            results.append(res)
        
        self.waiting = False
                
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    # def step_wait(self):
    #     results = [remote.recv() for remote in self.remotes]
    #     self.waiting = False
    #     obs, share_obs, rews, dones, infos, available_actions = zip(*results)
    #     return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = []
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res[0], str) and res[0] == "exception":
                msg, tb = res[1]
                raise RuntimeError(f"[Worker Exception during reset] {msg}\nTraceback:\n{tb}")
            results.append(res)
        
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
            
        # remove this can get rid of the error: assert self._parent_pid == os.getpid(), 'can only join a child process'
        # for p in self.ps:
        #     p.join()
        self.closed = True


def choosesimpleworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseSimpleSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=choosesimpleworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset(data)
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def chooseguardworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset(data)
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ChooseGuardSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=chooseguardworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, reset_choose):
        for remote, choose in zip(self.remotes, reset_choose):
            remote.send(('reset', choose))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError



class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns, num_agents=None, num_enemies=None, num_entities=None):
        self.envs = [fn() for fn in list(env_fns)]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.num_entities = num_entities
        
    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def render_video(self, key=None):
        for id, env in enumerate(self.envs):
            env.render_video(key + "_" + str(id)) 

    def save_replay(self):
        for env in self.envs:
            env.save_replay()
            

class ChooseDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))
        self.actions = None
        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self, reset_choose):
        results = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class ChooseSimpleDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.actions = None
        return obs, rews, dones, infos

    def reset(self, reset_choose):
        obs = [env.reset(choose)
                   for (env, choose) in zip(self.envs, reset_choose)]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


"""
shard worker for multiple environments
"""

def envshareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    vecenv = env_fn_wrapper.x
    try:
        while True:
            try:
                cmd, data, env_idx = remote.recv()
                if cmd == 'step':
                    # print(f"step: {env_idx} | action: {data}")
                    ob, s_ob, reward, done, info, available_actions = vecenv.step(data)
                    # 貌似不需要下边这个if...else...用来reset
                    # 因为这里vecenv实际上是类ShareSubprocVecEnv, 其本身有实现reset
                    # 但是加了好像也影响不大
                    # if 'bool' in done.__class__.__name__:
                    #     if done:
                    #         ob, s_ob, available_actions = vecenv.reset()
                    # else:
                    #     if np.all(done):
                    #         ob, s_ob, available_actions = vecenv.reset()
                    remote.send((ob, s_ob, reward, done, info, available_actions, env_idx))
                elif cmd == 'reset':
                    ob, s_ob, available_actions = vecenv.reset()                
                    remote.send((ob, s_ob, available_actions, env_idx))
                elif cmd == 'reset_task':
                    ob = vecenv.reset_task()
                    remote.send(ob)
                elif cmd == 'render':
                    if data == "rgb_array":
                        fr = vecenv.render(mode=data)
                        remote.send(fr)
                    elif data == "human":
                        vecenv.render(mode=data)
                elif cmd == 'close':
                    vecenv.close()
                    remote.close()
                    break
                elif cmd == 'save_replay':
                    print("TEST: save replay")
                    vecenv.save_replay()
                elif cmd == 'get_spaces':
                    remote.send((vecenv.observation_space, vecenv.share_observation_space, vecenv.action_space))
                else:
                    raise NotImplementedError
            except Exception as e:
                tb = traceback.format_exc()
                remote.send(("exception", (str(e), tb)))
                # remote.close()
                break
    except Exception as e:
        tb = traceback.format_exc()
        remote.send(("exception", (str(e), tb)))
        # remote.close()


class MultiEnvShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns_list, num_threads_per_env, multi_envs, num_agent_list, num_enemy_list, num_entity_list, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        if num_threads_per_env == 1:
            self.env_list = [ShareDummyVecEnv(env_fns, n_agents, n_enemy, n_entity) for env_fns, n_agents, n_enemy, n_entity in zip(env_fns_list, num_agent_list, num_enemy_list, num_entity_list)]
        else:
            self.env_list = [ShareSubprocVecEnv(env_fns, n_agents, n_enemy, n_entity) for env_fns, n_agents, n_enemy, n_entity in zip(env_fns_list, num_agent_list, num_enemy_list, num_entity_list)]

        self.waiting = False
        self.closed = False
        n_multi_envs = len(env_fns_list)
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_multi_envs)])
        # ! in MacOS, use fork to create process to make sure the child process (ShareSubprocVecEnv or ShareDummyVecEnv) will inherit (rather than open a fresh process) all resources (e.g., file descriptors and handles) of the parent. Otherwise it might have a conflict between child processes and parent process
        # ! by doing this, there won't be weakref problem so we can use CloudpickleWrapper again!
        # ! in Linux this won't be a problem
        
        self.ps = [Process(target=envshareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, self.env_list)]
        
        # logger = multiprocessing.log_to_stderr(logging.INFO)

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
    
        for remote in self.work_remotes:
            remote.close()        

        # logger = multiprocessing.log_to_stderr(logging.INFO)
        
        self.num_thread_per_env = num_threads_per_env
        self.multi_envs = multi_envs
        self.num_agents = num_agent_list
        self.num_enemies = num_enemy_list
        self.num_entities = num_entity_list
        self.observation_space, self.share_observation_space, self.action_space = [], [], []
        current_threads = 0
        
        for idx, _ in enumerate(multi_envs):
            ob_sp, sh_ob_sp, ac_sp = self.env_list[idx].observation_space, \
                self.env_list[idx].share_observation_space, \
                    self.env_list[idx].action_space
            self.observation_space.append(ob_sp)
            self.share_observation_space.append(sh_ob_sp)
            self.action_space.append(ac_sp)
            current_threads += num_threads_per_env
            # print(f"MultiTrainEnvShareSubprocVecEnv init {idx}")

        ShareVecEnv.__init__(self, len(self.env_list), self.observation_space, self.share_observation_space, self.action_space)
        
    # def step(self, actions, idx):
    #     return self.env_list[idx].step(actions)

    def save_replay(self):
        for idx, remote in enumerate(self.remotes):
            remote.send(('save_replay', None, idx))

    def step_async(self, actions_list):
        if self.closed:
            raise RuntimeError("step_async called on a closed MultiEnvShareSubprocVecEnv.")

        for idx, (remote, action) in enumerate(zip(self.remotes, actions_list)):
            remote.send(('step', action, idx))
        self.waiting = True

    def step_wait(self):
        if not self.waiting:
            raise RuntimeError("step_wait called without calling step_async first.")

        results = []
        for idx, remote in enumerate(self.remotes):
            res = remote.recv()
            results.append(res)

        self.waiting = False
        
        obs, share_obs, rews, dones, infos, available_actions, idxs = zip(*results)
        return obs, share_obs, rews, dones, infos, available_actions, idxs


    # def step_wait(self):
    #     results = [remote.recv() for remote in self.remotes]
    #     self.waiting = False
    #     obs, share_obs, rews, dones, infos, available_actions, idxs = zip(*results)
    #     # obs is a tuple consisting of obs corronspanding to different env in multi envs
    #     return obs, share_obs, rews, dones, infos, available_actions, idxs

    def reset(self):
        for idx, remote in enumerate(self.remotes):
            remote.send(('reset', None, idx))
            
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        for idx, res in enumerate(results):
            if isinstance(res[0], str) and res[0] == "exception":
                msg, tb = res[1]
                raise RuntimeError(f"Exception in ShareSubprocVecEnv {idx} during reset:\n{tb}")        

        obs, share_obs, available_actions, idxs = zip(*results)
        # return np.stack(obs), np.stack(share_obs), np.stack(available_actions)
        return obs, share_obs, available_actions, idxs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None, None))
        for p in self.ps:
            p.join()
        self.closed = True
        
    def render(self, idx):
        self.remotes[idx].send(('render', "human", None))
