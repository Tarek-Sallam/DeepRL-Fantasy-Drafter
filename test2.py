import pandas as pd
import os

players = pd.read_csv(os.path.join(os.getcwd(), 'data', 'projection_data.csv'))
qb = players[players['position'] == 'QB'].reset_index(drop=True)
rb = players[players['position'] == 'RB'].reset_index(drop=True)
wr = players[players['position'] == 'WR'].reset_index(drop=True)
te = players[players['position'] == 'TE'].reset_index(drop=True)
k = players[players['position'] == 'K'].reset_index(drop=True)
defs = players[players['position'] == 'DEF'].reset_index(drop=True)
q = [qb, rb, wr, te, k, defs]

print(q[0].iloc[0].drop(['name', 'last_name', 'first_name']))