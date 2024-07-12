from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary
import numpy as np
import pandas as pd
import random
import os
from DraftBoard import DraftBoard

class DraftEnv(Env):
    def __init__(self, teams, data_path):
        self.action_space = Discrete(6)
        self.observation_space = Dict({"roster": MultiBinary(26), "top_projections": Box(low=np.zeros(19), high=np.array(np.ones(19) * np.inf))})
        self.draftBoard = DraftBoard(teams, random.randint(1, teams), data_path)
        self.agentRoster = pd.DataFrame(columns=['display', 'position', 'proj', 'slot'])
        self.data_path = data_path
        self.totalPts = 0.0
        self.round = 1
        self.max_rounds = 19
        self.state = {"roster": np.zeros(26), "top_projections": np.array(self.draftBoard.getTopProjections())}

    def step (self, action):
        topPoints = self.state["top_projections"]
        roster, player = self.addToRoster(action)
        if player[3] == 'SUB':
            self.totalPts += topPoints[action]
        playerFrame = pd.DataFrame({"display": [player[0]], "position": [player[1]], "proj": [player[2]], 'slot': [player[3]]})
        self.agentRoster = pd.concat([self.agentRoster, playerFrame])
        self.draftBoard.removePlayer(action, 0) # remove the player from the available players
        self.round +=1 # increase the round of the draft
        if self.round > self.max_rounds:
            done = True
            reward = self.totalPts
        else:
            done = False
            reward = 0
            self.draftBoard.goToNext()
            
        self.state = {"roster": roster, "top_projections": np.array(self.draftBoard.getTopProjections())} # set the state with the new roster, along with the new top projections
        info = {"selected": player}
        return self.state, reward, done, info

    def render(self):
        pass
    def reset(self):
        self.draftBoard = DraftBoard(self.draftBoard.teams, random.randint(1, self.draftBoard.teams), self.data_path) # reset draft board
        self.agentRoster = pd.DataFrame(columns=['display', 'position', 'proj', 'slot'])
        self.totalPts = 0.0 # reset total points
        self.round = 1 # reset the round
        self.state = {"roster": np.zeros(26), "top_projections": np.array(self.draftBoard.getTopProjections())} # reset the state
        return self.state

    def addToRoster(self, position) -> tuple[list, list]:
        roster = self.state["roster"].copy()
        player = self.draftBoard.getPlayer(position, 0).to_list()
        if position == 1 or position ==2:
            if roster[position] == 0:
                roster[position] = 1
                player.append('RB1' if position == 1 else 'WR1')
                return (roster, player)
            elif roster[position + 5] == 0:
                roster[position + 5] = 1
                player.append('RB2' if position == 1 else 'WR2')
                return (roster, player)
            elif roster[8] == 0:
                roster[8] = 1
                player.append('W/R')
                return (roster, player)
        else:
            if roster[position] == 0:
                roster[position] = 1
                if position == 0:
                    player.append('QB')
                elif position == 3:
                    player.append('TE')
                elif position == 4:
                    player.append('K')
                elif position == 5:
                    player.append('DEF')
                return (roster, player)
        i = 9
        while roster[i] == 1:
            i+=1
        roster[i] = 1
        player.append('SUB')
        return (roster, player)

