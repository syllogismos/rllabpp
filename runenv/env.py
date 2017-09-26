from osim.env import RunEnv
from rllab.envs.base import Env
from rllab.envs.base import Step
from gym.spaces import Box
import numpy as np
import random
from rllab.misc import logger
from collections import deque


SEEDMAX = 100000000
HORIZON = 1000

class RunEnvVanilla(Env):
    def __init__(self, difficulty=2, visualize=False, max_obstacles=3):
        self.env = RunEnv(visualize=visualize)
        self.difficulty = difficulty
        self.visualize = visualize
        self.max_obstacles = max_obstacles
    
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=self.env.observation_space.shape)

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=self.env.action_space.shape)

    @property
    def horizon(self):
        return HORIZON

    def reset(self):
        seed = random.randint(0, SEEDMAX)
        logger.log('reset seed is {}'.format(seed))
        self._state = self.env.reset(difficulty=self.difficulty, seed=seed)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._old_state = np.copy(self._state)
        self._state, reward, done, info = self.env.step(action)
        observation = np.copy(self._state)
        return Step(observation=observation, reward=reward, done=done)


class RunEnvFeatures(Env):
    def __init__(self, difficulty=2, visualize=False, max_obstacles=3,
                 history_len=4, filter_type=''):
        self.env = RunEnv(visualize=visualize)
        self.difficulty = difficulty
        self.visualize = visualize
        self.max_obstacles = max_obstacles
        self.obs_len = len(get_features_from_history([[0.0]*41], filter_type=filter_type))
        self.history = deque(maxlen=history_len)
        self.filter_type = filter_type

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.obs_len,))

    @property
    def action_space(self):
        return Box(low=0.0, high=1.0, shape=self.env.action_space.shape)

    @property
    def horizon(self):
        return HORIZON

    def reset(self):
        seed = random.randint(0, SEEDMAX)
        logger.log('reset seed is {}'.format(seed))
        self.runenv_state = self.env.reset(difficulty=self.difficulty, seed=seed)
        self.history.append(np.copy(self.runenv_state))
        observation = get_features_from_history(self.history, filter_type=self.filter_type)
        return observation

    def step(self, action):
        self.runenv_state, reward, done, info = self.env.step(action)
        self.history.append(np.copy(self.runenv_state))
        observation = get_features_from_history(self.history, filter_type=self.filter_type)
        return Step(observation=observation, reward=reward, done=done)


def get_features_from_history(history, filter_type=''):
    curr_obs = np.copy(history[-1])
    past_obs = np.copy(history[0])
    if filter_type == '':
        return np.hstack((curr_obs, past_obs))
    else:
        raise NotImplementedError
