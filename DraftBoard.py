import pandas as pd
import random

class DraftBoard():
    def __init__(self, teams: int, agent_pick: int, data_path: str):
        self.teams = teams
        self.agent_pick = agent_pick
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
        self.players[position] = self.players[position].drop(index=index).reset_index(drop=True)

    ## returns a player's info at a given position and index from the draft board
    def getPlayer(self, position: int, index: int) -> pd.Series:
        return self.players[position].iloc[index].drop(['name', 'last_name', 'first_name'])
    
    # go to the next agent draft pick by simulating the picking of (self.teams - 1) picks
    def goToNext(self) -> None:
        for i in range(self.teams-1):
            self.removePlayer(random.randint(0, 3), random.randint(0, 3))

    # returns a list of the projections of the top players from each position
    def getTopProjections(self) -> list[float]:
        l = []
        for i in self.players:
            l.append(i.iloc[i['proj'].idxmax()]['proj'])
        return l
    