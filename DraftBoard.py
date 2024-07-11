import pandas as pd

class DraftBoard():
    def __init__(self, teams: int, agent_pick: int, data_path: str):
        self.teams = teams
        self.agent_pick = agent_pick
        players = pd.read_csv(data_path)
        qb = players[players['position'] == 'QB']
        rb = players[players['position'] == 'RB']
        wr = players[players['position'] == 'WR']
        te = players[players['position'] == 'TE']
        k = players[players['position'] == 'K']
        defs = players[players['position'] == 'DEF']
        self.players = [qb, rb, wr, te, k, defs]

    def remove_player(self, position: int, index: int) -> None:
        self.players[position].drop(index=index).reset_index(drop=True)

    def getTopProjections(self) -> list[float]:
        l = []
        for i in self.players:
            l.append(i.iloc[i['proj'].idxmax()]['proj'])
        return l