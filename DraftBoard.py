import pandas as pd
import random
import numpy as np

class DraftBoard():
    def __init__(self, teams: int, agent_pick: int, data_path: str):
        self.teams = teams
        self.agent_pick = agent_pick
        self.isReverse = False;
        players = pd.read_csv(data_path)
        qb = players[players['position'] == 'QB'].reset_index(drop=True)
        rb = players[players['position'] == 'RB'].reset_index(drop=True)
        wr = players[players['position'] == 'WR'].reset_index(drop=True)
        te = players[players['position'] == 'TE'].reset_index(drop=True)
        k = players[players['position'] == 'K'].reset_index(drop=True)
        defs = players[players['position'] == 'DEF'].reset_index(drop=True)
        self.players = [qb, rb, wr, te, k, defs]
        for i in range(agent_pick-1):
            self.removePlayer(random.randint(0, 5), random.randint(0, 1))


    ## removes a player at a given position and index from the draft board
    def removePlayer(self, position: int, index: int) -> None:
        if not self.players[position].empty:
            self.players[position] = self.players[position].drop(index=index).reset_index(drop=True)

    ## returns a player's info at a given position and index from the draft board
    def getPlayer(self, position: int, index: int) -> pd.Series:
        if self.players[position].empty:
            positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
            item = {'display': 'NULL', 'position': positions[position], 'proj': 0.0}
            return pd.Series(data=item, index=item.keys())
        else:
            return self.players[position].iloc[index].drop(['name', 'last_name', 'first_name'])
    
    # go to the next agent draft pick by simulating the picking of (self.teams - 1) picks
    def goToNext(self) -> None:
        if not self.isReverse:
            for i in range(2(self.teams-self.agent_pick)):
                self.removePlayer(random.randint(0, 3), random.randint(0, 3))
            self.isReverse = True;
        else:
            for i in range(2(self.agent_pick - 1)):
                self.removePlayer(random.randint(0, 3), random.randint(0, 3))
    
        

    # returns a list of the projections of the top players from each position
    def get_top_projections_normalized(self) -> list[float]:
        l = []
        for i in self.players:
            if i.empty:
                l.append(0)
            else:
                l.append(i.iloc[i['proj'].idxmax()]['proj'])
        
        l_norm = (l - np.min(l)) / (np.max(l) - np.min(l))
        
        return l_norm
    
    def get_agent_pick(self):
        if not self.isReverse:
            return self.agent_pick
        else:
            return (self.teams - self.agent_pick) + 1
    