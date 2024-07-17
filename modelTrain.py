import gym
from gym.wrappers import FlattenObservation
import numpy
import os
import tensorflow as tf
import numpy as np
from Agent import PolicyGradientAgent
from Environment import DraftEnv

env = DraftEnv(10, os.path.join(os.getcwd(), 'data', 'projection_data.csv'))
env = FlattenObservation(env)
n_inputs = env.observation_space.shape[0] ## get the shape
n_actions = env.action_space.n

agent = PolicyGradientAgent(learning_rate=0.01, n_actions = n_actions, n_inputs=n_inputs, n_layers=2, layer_size=[64] * 2)

episodes = 100

for episode in range(1, episodes + 1):
    state = env.reset()[0]
    done = False
    while not done:
            action = agent.choose_action(np.array([state]))
            print(action)
            next_state, reward, _ , done , info = env.step(action) # make the step in the environment based on the action
            agent.store_transition(state, action, reward)
            state = next_state # move to next state
    agent.learn()
    agent.save_model(os.path.join(os.getcwd(), 'keras', 'fantasyDrafter.keras'))
    env.agentRoster.to_csv(os.path.join(os.getcwd(), 'trainingRosters', 'iteration_' + str(episode) + '_roster.csv'), index=False)