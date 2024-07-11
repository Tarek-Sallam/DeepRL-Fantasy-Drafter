import pandas as pd
import os

players = pd.read_csv(os.path.join(os.getcwd(), 'data', 'projection_data.csv'))
print(players)
print(players.iloc[players['proj'].idxmax()]['proj'])