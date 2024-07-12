from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary
import numpy as np
import pandas as pd
import random
import os
from collections import OrderedDict
from DraftBoard import DraftBoard

class DraftEnv(Env):
    def __init__(self, teams, data_path):
        self.action_space = Discrete(6)
        self.observation_space = Dict(roster = MultiBinary(27), top_projections= Box(low=np.zeros(6), high=np.array(np.ones(6) * np.inf)))
        self.draftBoard = DraftBoard(teams, random.randint(1, teams), data_path)
        self.agentRoster = pd.DataFrame(columns=['display', 'position', 'proj', 'slot'])
        self.data_path = data_path
        self.totalPts = 0.0
        self.round = 1
        self.max_rounds = 19
        self.observation = OrderedDict(roster = np.zeros(27, dtype='int8'), top_projections = np.array(self.draftBoard.getTopProjections(), dtype='float32'))

    def step (self, action):
        topPoints = self.observation["top_projections"] ### get the top projections
        roster, player = self.addToRoster(action) ### add the player to the current roster given the action (return the modified roster state and player)
        if player[3] != 'SUB':
            reward = topPoints[action] ## if player is not a sub, make the reward the points picked
        else:
            reward = 0 ## if player is sub then no reward
        playerFrame = pd.DataFrame({"display": [player[0]], "position": [player[1]], "proj": [player[2]], 'slot': [player[3]]}) ## create the dataframe to append to the roster
        if self.round == 1:
            self.agentRoster = playerFrame ## make the roster just the player if first round
        else:
            self.agentRoster = pd.concat([self.agentRoster, playerFrame], ignore_index=True) ## otherwise append the player to roster
        self.draftBoard.removePlayer(action, 0) # remove the player from the available players
        self.round +=1 # increase the round of the draft
        if self.round > self.max_rounds: 
            done = True ## if we have reached the end of the draft, done is true
        else:
            done = False
            self.draftBoard.goToNext() ## go to the next round
            
        self.observation = OrderedDict(roster = roster, top_projections = np.array(self.draftBoard.getTopProjections(), dtype='float32')) # set the state with the new roster, along with the new top projections
        info = {"selected": player}
        return self.observation, reward, False, done, info

    def render(self):
        pass
    def reset(self):
        self.draftBoard = DraftBoard(self.draftBoard.teams, random.randint(1, self.draftBoard.teams), self.data_path) # reset draft board
        self.agentRoster = pd.DataFrame(columns=['display', 'position', 'proj', 'slot'])
        self.totalPts = 0.0 # reset total points
        self.round = 1 # reset the round
        self.observation = OrderedDict(roster = np.zeros(27, dtype='int8'), top_projections = np.array(self.draftBoard.getTopProjections(), dtype='float32')) # reset the state
        return self.observation, {}

    def addToRoster(self, position) -> tuple[list, list]:
        roster = self.observation["roster"].copy()
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

