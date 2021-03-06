import gym
import gym.wrappers
import gym.envs
import gym.spaces
import traceback
import logging
import random
import roboschool

from osim.env import RunEnv
from runenv.env import RunEnvVanilla, RunEnvFeatures

try:
    from gym.wrappers.monitoring import logger as monitor_logger

    monitor_logger.setLevel(logging.WARNING)
except Exception as e:
    traceback.print_exc()

import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


class FixedIntervalVideoSchedule(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, record_video=True, video_schedule=None,
            log_dir=None, record_log=True, force_reset=False, visualize=False,
            runenv_seed=None, difficulty=2, max_obstacles=3, history_len=4,
            filter_type=''):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        if env_name == 'RunEnv':
            env = RunEnv(visualize=visualize)
        elif env_name == 'RunEnvVanilla':
            env = RunEnvVanilla(visualize=visualize, difficulty=difficulty,
                                max_obstacles=max_obstacles)
        elif env_name == 'RunEnvFeatures':
            env = RunEnvFeatures(visualize=visualize, difficulty=difficulty,
                                 max_obstacles=max_obstacles, history_len=history_len,
                                 filter_type=filter_type)
        else:
            env = gym.envs.make(env_name)
        self.env = env
        if env_name.startswith('RunEnv'):
            self.env_id = env_name
        else:
            self.env_id = env.spec.id

        self.visualize = visualize
        self.runenv_seed = runenv_seed
        self.difficulty = difficulty
        self.max_obstacles = max_obstacles
        self.env_name = env_name
        self.history_len = history_len
        self.filter_type = filter_type

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            if not env_name.startswith('RunEnv') and not env_name.startswith('Roboschool'):
                self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=lambda x: True, force=True, uid="1")
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        if env_name.startswith('RunEnv'):
            self._horizon = env.horizon
        else:
             self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self, seed=None):
        if self._force_reset and self.monitoring and not self.env_name.startswith('RunEnv'):
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        if self.env_name == 'RunEnv':
            runenv_seed = random.randint(0, 100000000)
            logger.log("********** runenv reset seed: {}".format(runenv_seed))
            return self.env.reset(difficulty=self.difficulty, seed=runenv_seed)
        else:
            # This includes the envs defined in runenv/env.py
            return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        if self.monitoring:
            self.env._close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)

