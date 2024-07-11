from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import pandas as pd
import random
from DraftBoard import DraftBoard

class DraftEnv(Env):
    def __init__(self, teams):
        self.action_space = Discrete(6)
        self.observation_space = Tuple(Discrete(19), Box(low=np.zeros(19), high=np.array(np.ones(19) * np.inf)))
        self.draftBoard = DraftBoard(teams, random.randint(1, teams))
        self.totalPts = 0.0
        self.round = 1
        self.max_rounds = 19
        self.state = (np.zeros(19), np.array(self.draftBoard.getTopProjections()))

    def step (self, action):
        topPoints = self.state[1]
        roster, sub = self.addToRoster(action)
        if sub == 0:
            self.totalPts += topPoints[action]
        self.draftBoard.remove_player(action, 0) # remove the player from the available players
        self.round +=1 # increase the round of the draft
        if self.round > self.max_rounds:
            done = True
            reward = self.totalPts
        else:
            done = False
            reward = 0
            # go to next pick
            
        self.state = (roster, np.array(self.draftBoard.getTopProjections())) # set the state with the new roster, along with the new top projections
        info = {}
        return self.state, reward, done, info

    def render(self):
        pass
    def reset(self):
        self.draftBoard = DraftBoard(self.draftBoard.teams, random.randint(1, self.draftBoard.teams)) # reset draft board
        self.totalPts = 0.0 # reset total points
        self.round = 1 # reset the round
        self.state = (np.zeros(19), np.array(self.draftBoard.getTopProjections())) # reset the state
        return self.state

    def addToRoster(self, position) -> tuple[list, int]:
        roster = self.state[0].copy()
        if position == 1 or position ==2:
            if roster[position] == 0:
                roster[position] = 1
                return (roster, 0)
            elif roster[position + 5] == 0:
                roster[position + 5] = 1
                return (roster, 0)
            elif roster[8] == 0:
                roster[8] = 1
                return (roster, 0)
        else:
            if roster[position] == 0:
                roster[position] = 1
                return (roster, 0)
        
        i = 9
        while roster[i] == 1:
            i+=1
        roster[i] = 1
        return (roster, 1)

