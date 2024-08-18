import pandas as pd
import numpy as np
import scipy.stats as scpy

## Class that manages all things related to the draft board
class DraftBoard():

    ## constructer
    def __init__(self, teams: int, agent_pick: int, rounds: int, is_training: bool, projection_data_path: str, adp_data_path: str):
        # set all vars
        self.teams = teams
        self.rounds = rounds
        self.agent_pick = agent_pick

        # set the current pick and reversed order vars to defaults
        self.current_pick = 1;
        self.isReverse = False;

        ## get the distribution from the adp data
        self.draft_dist = self.get_distribution(adp_data_path)

        ## get the projection data
        players = pd.read_csv(projection_data_path)

        ## get the projections and min-max normalize them before adding them back into the dataframe
        proj = players['proj'].to_numpy();
        proj = (proj - np.min(proj)) / (np.max(proj) - np.min(proj))
        players['proj'] = proj

        # split the projections by player
        qb = players[players['position'] == 'QB'].reset_index(drop=True)
        rb = players[players['position'] == 'RB'].reset_index(drop=True)
        wr = players[players['position'] == 'WR'].reset_index(drop=True)
        te = players[players['position'] == 'TE'].reset_index(drop=True)
        k = players[players['position'] == 'K'].reset_index(drop=True)
        defs = players[players['position'] == 'DEF'].reset_index(drop=True)

        # create a players array with each position
        self.players = [qb, rb, wr, te, k, defs]

        if (is_training):
            # loop until the agents pick and make picks
            for i in range(agent_pick-1):
                self.makePick()
        

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
    
    # go to the next agent draft pick by simulating the picking whether in reversed or not reversed order
    def goToNext(self) -> None:
        if not self.isReverse:
            for i in range(2*(self.teams-self.agent_pick)):
                self.makePick()
            self.isReverse = True
        else:
            for i in range(2*(self.agent_pick - 1)):
                self.makePick()
            self.isReverse = False
    
    # get the probabilites and make a choice based on those probabilites. Then increment the pick
    def makePick(self):
        probs = self.calculate_probs(self.current_pick)
        self.current_pick+=1;
        choice = np.random.choice([0, 1, 2, 3, 4, 5], p=probs);
        self.removePlayer(choice, 0)

    # returns a list of the projections of the top players from each position
    def get_top_projections(self) -> list[float]:
        l = []
        for i in self.players:
            if i.empty:
                l.append(0)
            else:
                l.append(i.iloc[i['proj'].idxmax()]['proj'])
        return l

    # returns the pick that the agent has in the current round
    def get_agent_pick(self):
        if not self.isReverse:
            return self.agent_pick
        else:
            return (self.teams - self.agent_pick) + 1
        
    # calculates the beta distributions based on the adp data
    def get_distribution(self, path):
        adp_data = np.load(path, allow_pickle=True).item()
        dist = {}
        for pos, data in adp_data.items():
            if pos == 'K' or pos == 'DEF':
                alpha, beta, loc, scale = scpy.beta.fit(data)
                loc += 7/self.rounds
            else:
                alpha, beta, loc, scale = scpy.beta.fit(data)
            dist[pos] = scpy.beta(alpha, beta, loc=loc, scale=scale)
        return dist
    
    # calculates the probabilites of selecting each position given a pick number based on the adp data distribution
    def calculate_probs(self, pick):
        probs = []
        print(pick)
        scaled_pick = (pick - 1) / (self.teams * self.rounds)
        for dist in self.draft_dist.values():
            prob = dist.pdf(scaled_pick)
            probs.append(prob)
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        return probs