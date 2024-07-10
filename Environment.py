from gym import Env
from gym.spaces import Discrete, MultiBinary
import numpy as np
import random 

class draftEnv(Env):
    def __init__(self):
        self.action_space = Discrete(6)
        self.observation_space = MultiBinary(19)
        self.state = [0] * 19;
    def step (self):
        pass
    def render(self):
        pass
    def reset(self):
        pass