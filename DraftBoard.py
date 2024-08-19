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
        players['proj_norm'] = proj

        ## save the players dataframe and save a list of positions
        self.all_players = players
        self.all_players.sort_values('proj', ascending=False).reset_index(drop=True)
        self.positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']

        if (is_training):
            # loop until the agents pick and make picks
            for i in range(agent_pick-1):
                self.makePick()
        

    ## removes a player at a given position and index from the draft board
    def removePlayer(self, position: int, index: int) -> None:
        pos_df = self.all_players[self.all_players['position'] == self.positions[position]]
        pos_df = pos_df.sort_values('proj', ascending=False).reset_index()
        if not pos_df.empty: 
            drop_index = pos_df.iloc[index]['index']
            self.all_players = self.all_players.drop(index=drop_index).reset_index(drop=True)

    def removePlayerByInfo(self, player):
        player = self.all_players.index[(self.all_players['display'] == player['display']) & (self.all_players['position'] == player['position']) & (self.all_players['proj'] == float(player['proj']))]
        self.all_players = self.all_players.drop(index=player)

    ## returns a player's info at a given position and index from the draft board
    def getPlayer(self, position: int, index: int) -> pd.Series:
        pos_df = self.all_players[self.all_players['position'] == self.positions[position]]
        pos_df = pos_df.reset_index(drop=True)
        if pos_df.empty:
            item = {'display': 'NULL', 'position': self.positions[position], 'proj': 0.0}
            return pd.Series(data=item, index=item.keys())
        else:
            return pos_df.iloc[index].drop(['name', 'last_name', 'first_name'])
    
    def getFullPlayer(self, player):
        player = self.all_players[(self.all_players['display'] == player['display']) & (self.all_players['position'] == player['position']) & (self.all_players['proj'] == float(player['proj']))]
        if player.empty:
            print("Player does not exist in database")
            item = {'name': 'NULL', 'last_name': 'NULL', 'first_name': 'NULL', 'display': 'NULL', 'position': 'NULL', 'proj': 0.0}
            return pd.Series(data=item, index=item.keys())
        else:
            return player
        
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
        for pos in self.positions:
            players_pos = self.all_players[self.all_players['position'] == pos].reset_index(drop=True)
            if players_pos.empty:
                l.append(0)
            else:
                l.append(players_pos.iloc[players_pos['proj_norm'].idxmax()]['proj_norm'])
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
        scaled_pick = (pick - 1) / (self.teams * self.rounds)
        for dist in self.draft_dist.values():
            prob = dist.pdf(scaled_pick)
            probs.append(prob)
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        return probs
    
    def get_players_df(self, query: str):
        if query == '':
            return self.all_players.drop(['name', 'first_name', 'last_name', 'proj_norm'], axis=1)
        else:
            return self.all_players.drop(['name', 'first_name', 'last_name', 'proj_norm'], axis=1)