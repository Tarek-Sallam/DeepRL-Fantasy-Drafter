from Environment import DraftEnv
from gym.wrappers import FlattenObservation
import tensorflow as tf
import numpy as np
import os

env = DraftEnv(10, os.path.join(os.getcwd(), 'data', 'projection_data.csv'))
env = FlattenObservation(env)
input_shape = env.observation_space.shape[0]
num_actions = env.action_space.n
print(input_shape)

policyNetwork = tf.keras.models.Sequential([
   tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
   tf.keras.layers.Dense(32, activation='relu'),
   tf.keras.layers.Dense(num_actions, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

episodes = 1

for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    states = []
    actions = []
    rewards = []

    while True:
        action_probs = policyNetwork.predict(np.array([state]))[0]
        action = np.random.choice(num_actions, p=action_probs)
        next_state, reward, _ , done , info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            print('Episode {} finished.'.format(episode))
            break
        
    