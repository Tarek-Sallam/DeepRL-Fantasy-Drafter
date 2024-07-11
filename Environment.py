from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import pandas as pd
import random

class DraftBoard():
    def __init__(self, teams: int, agentPick: int, data_path: str):
        self.agentPick = agentPick;
        self.teams = teams;
        players = pd.read_csv(data_path)
        self.players = np.array()
        self.agentPoints = 0;

class DraftEnv(Env):
    def __init__(self, teams):
        self.action_space = Discrete(6)
        self.observation_space = Tuple(Discrete(19), Box(low=np.zeros(19), high=np.array(np.ones(19) * np.inf)))
        self.draftBoard = DraftBoard(teams, random.randint(1, teams))
        self.state = (np.zeros(19), )
        #randomize the agents place in the draft
        #draft all players until the agents pick
    def step (self):
        #draft x players to finish the round, and then reward based on current positional ranking
        #draft next y players 
        pass
    def render(self):
        pass
    def reset(self):
        pass