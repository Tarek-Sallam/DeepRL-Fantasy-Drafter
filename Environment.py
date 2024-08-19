from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary
import numpy as np
import pandas as pd
import random
import os
from collections import OrderedDict
from DraftBoard import DraftBoard

class DraftEnv(Env):

    ## Constructor
    def __init__(self, teams, rounds, projection_data_path, adp_data_path):

        ## init the vars
        self.projection_data_path = projection_data_path
        self.adp_data_path = adp_data_path
        self.rounds = rounds
        self.totalPts = 0.0
        self.round = 1

        ## action space and observation space:
            # Actions: SELECTING POSITION: (0) -> QB, (1) -> RB, (2) -> WR, (3) -> TE, (4) -> K, (5) -> DEF
            # Observation: CURRENT ROSTER -> array of binary values of size 9 for each starting roster,
                # TOP PLAYERS POINTS IN EACH POSITION (Normalised) -> array of floats between 0 and 1 of size 6
    
        self.action_space = Discrete(6)
        self.observation_space = Dict(roster = MultiBinary(9), top_projections= Box(low=np.zeros(6), high=np.array(np.ones(6))))

        ## init the draft board
        self.draftBoard = DraftBoard(teams, random.randint(1, teams), self.rounds, True, projection_data_path, adp_data_path)

        ## init the first observation, Roster: all zeros, Get Top projections
        self.observation = OrderedDict(roster = np.zeros(9, dtype='int8'), top_projections = np.array(self.draftBoard.get_top_projections(), dtype='float32'))

    ## Function for stepping
    def step (self, action):
        ## Get the top projected points of each position
        topPoints = self.observation["top_projections"]

        ## Add the player to the roster and return the modified roster as well as the list of player info
        roster, playerInfo = self.addToRoster(action)

        ## if the player's position is kicker or defense and we are in the first 16 rounds, then reward is 0
        if playerInfo[3] == 'K' or playerInfo[3] == 'DEF' and self.round <= 16:
            reward = 0

        ## if the player wasn't selected as a sub, then the reward is the points of that player
        elif playerInfo[3] != 'SUB':
            reward = topPoints[action]

        ## if the player is a sub the reward is 0
        else:
            reward = 0

        ## add the reward to the total points
        self.totalPts += reward

        ## create a dataframe of the player, and concatenate it to the roster list
        playerFrame = pd.DataFrame({"display": [playerInfo[0]], "position": [playerInfo[1]], "proj": [playerInfo[2]], 'slot': [playerInfo[3]]})
        if self.round == 1:
            self.agentRoster = playerFrame ## make the roster just the player if first round
        else:
            self.agentRoster = pd.concat([self.agentRoster, playerFrame], ignore_index=True)
        
        ## remove the player from the available players
        ## increase the round
        ## increase the pick in the draftboard
        self.draftBoard.removePlayer(action, 0)
        self.round +=1
        self.draftBoard.current_pick+=1;

        ## if we are in the last round then set done to true
        if self.round > self.rounds: 
            done = True ## if we have reached the end of the draft, done is true

        ## otherwise done is false and go to next pick in the draft board
        else:
            done = False
            self.draftBoard.goToNext() ## go to the next round
            
        ## update the observation with new data
        self.observation = OrderedDict(roster = roster, top_projections = np.array(self.draftBoard.get_top_projections(), dtype='float32')) # set the state with the new roster, along with the new top projections

        ## set the info to the selected player
        info = {"selected": playerInfo}

        ## return the info
        return self.observation, reward, False, done, info

    def render(self):
        pass

    ## Function for resetting the environment
    def reset(self):

        ## create a new draft board, with the specified info
        self.draftBoard = DraftBoard(self.draftBoard.teams, random.randint(1, self.draftBoard.teams), self.rounds, True, self.projection_data_path, self.adp_data_path) # reset draft board

        ## reset the points and round
        self.totalPts = 0.0
        self.round = 1

        ## set the observation
        self.observation = OrderedDict(roster = np.zeros(9, dtype='int8'), top_projections = np.array(self.draftBoard.get_top_projections(), dtype='float32'))

        ## return the observation
        return self.observation, {}

    ## function that adds a player to the roster list
    def addToRoster(self, position) -> tuple[list, list]:

        ## get a copy of the roster
        roster = self.observation["roster"].copy()

        ## get a list of the player information
        player = self.draftBoard.getPlayer(position, 0).to_list()

        ## if it is a RB or WR
        if position == 1 or position == 2:

            ## if the RB1 or WR1 is open then set it to 1 to fill it, and append the position to the list
            if roster[position] == 0:
                roster[position] = 1
                player.append('RB1' if position == 1 else 'WR1')
                return (roster, player)
            
            ## if the RB2 or WR2 is open then set it to 1 to fill it, and append the position to the list
            elif roster[position + 5] == 0:
                roster[position + 5] = 1
                player.append('RB2' if position == 1 else 'WR2')
                return (roster, player)
            
            ## if the WR/RB slot it open then set it to 1 to fill it, and append the position to the list
            elif roster[8] == 0:
                roster[8] = 1
                player.append('W/R')
                return (roster, player)
        
        ## otherwise for QB, TE, K, DEF, check if the spot has been filled, if it has fill it, and append the position to the list
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
            
        ## otherwise append the sub and do not update the roster
        player.append('SUB')
        return (roster, player)