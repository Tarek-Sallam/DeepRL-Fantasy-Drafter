from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import pandas as pd
import random

class DraftBoard():
    def __init__(self, teams: int, agentPick: int, data_path: str):
        
        # set the agent pick and the amount of teams attributes
        self.agentPick = agentPick;
        self.teams = teams;
        players = pd.read_csv(data_path)
        
        # split the data up into position and make sure they are sorted by projected points
        rb = players[players["position"] == 'RB'].sort_values('proj', ascending=False)
        qb = players[players["position"] == 'QB'].sort_values('proj', ascending=False)
        wr = players[players["position"] == 'WR'].sort_values('proj', ascending=False)
        k = players[players["position"] == 'K'].sort_values('proj', ascending=False)
        defs = players[players["position"] == 'DEF'].sort_values('proj', ascending=False)
        te = players[players["position"] == 'TE'].sort_values('proj', ascending=False)

        # set the players array with the data
        self.players = [qb, rb, wr, te, k, defs]

    def getTopProjections(self) -> list[float]:
        l = []
        for pos in self.players:
            l.append(pos['proj'].max)
        return l

    def toNextPick():
        pass

        

class DraftEnv(Env):
    def __init__(self, teams):
        self.action_space = Discrete(6)
        self.observation_space = Tuple(Discrete(19), Box(low=np.zeros(19), high=np.array(np.ones(19) * np.inf)))
        self.draftBoard = DraftBoard(teams, random.randint(1, teams))
        self.state = (np.zeros(19), np.array(self.draftBoard.getTopProjections()))
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