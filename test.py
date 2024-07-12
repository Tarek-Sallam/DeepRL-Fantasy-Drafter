from Environment import DraftEnv
import os

env = DraftEnv(10, os.path.join(os.getcwd(), 'data', 'projection_data.csv'))
episodes = 10

for episode in range(1, episodes+1):
    state = env.reset()
    done=False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
        print('Action: {}'.format(action))

    print('Roster: ')
    print(env.agentRoster)
    print('Episode: {}, Score: {}'.format(episode, score))
 
    
