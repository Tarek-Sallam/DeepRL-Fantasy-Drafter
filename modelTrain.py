import gym
from gym.wrappers import FlattenObservation
import numpy
import os
import tensorflow as tf
import numpy as np
from Agent import PolicyGradientAgent
import matplotlib as mp
from Environment import DraftEnv

env = DraftEnv(12, 19, os.path.join(os.getcwd(), 'data', 'projection_data.csv'), os.path.join(os.getcwd(), 'data', 'adp_data.npy'))
env = FlattenObservation(env)
n_inputs = env.observation_space.shape[0] ## get the shape
n_actions = env.action_space.n
init_epsilon = 0.8
final_epsilon = 0.2

agent = PolicyGradientAgent(learning_rate=0.001, discount_factor=1, n_actions = n_actions, epsilon=init_epsilon, n_inputs=n_inputs, n_layers=4, layer_size=[10] * 4)
episodes = 100

for episode in range(1, episodes + 1):
        state = env.reset()[0]
        done = False
        while not done:
                print(state)
                action = agent.choose_action(np.array([state]))
                next_state, reward, _ , done , info = env.step(action) # make the step in the environment based on the action
                agent.store_transition(state, action, reward)
                state = next_state # move to next state
        print('Episode {} finished.'.format(episode))
        agent.learn() 
        agent.set_epsilon(init_epsilon - (episode / episodes) * (init_epsilon - final_epsilon))
        agent.save_model(os.path.join(os.getcwd(), 'keras', 'fantasyDrafter.keras'))
        env.agentRoster.to_csv(os.path.join(os.getcwd(), 'trainingRosters', 'iteration_' + str(episode) + '_roster.csv'), index=False)