from osim.env import RunEnv
from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np
import random
from rllab.misc import logger



class RunEnvVanilla(Env):
    def __init__(self, difficulty=2, visualize=False, max_obstacles=3):
        self.env = RunEnv(visualize=visualize)
        self.difficulty = difficulty
        self.visualize = visualize
        self.max_obstacles = max_obstacles
    
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.env.observation_space.shape[0]))

    @property
    def action_space(self):
        return Box(low=0, high=1, shape(self.env.action_space.shape[0]))

    @property
    def horizon(self):
        return 1000

    def reset(self):
        seed = random.randint(0, 100000000)
        logger.log('reset seed is {}'.format(seed))
        self._state = self.env.reset(difficulty=self.difficulty, seed=seed)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._old_state = np.copy(self._state)
        self._state, reward, done, info = self.env.step(action)
        observation = np.copy(self._state)
        return Step(observation=observation, reward=reward, done=done)
        pass

    