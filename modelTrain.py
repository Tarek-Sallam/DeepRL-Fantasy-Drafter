import gym
from gym.wrappers import FlattenObservation
import numpy
import os
import tensorflow as tf
import numpy as np
from Agent import PolicyGradientAgent
import matplotlib.pyplot as plt
from Environment import DraftEnv

env = DraftEnv(
        12, 
        19, 
        os.path.join(os.getcwd(), 'data', 'projection_data.csv'), 
        os.path.join(os.getcwd(), 'data', 'adp_data.npy')
)

env = FlattenObservation(env)
n_inputs = env.observation_space.shape[0] ## get the shape
print(env.observation_space.shape)
n_actions = env.action_space.n
init_epsilon = 0.6
final_epsilon = 0.05

agent = PolicyGradientAgent(
        learning_rate=0.001, 
        discount_factor=1, n_actions = 
        n_actions, epsilon=init_epsilon, 
        n_inputs=n_inputs, 
        n_layers=2, 
        layer_size=[10] * 4
)

episodes = 1000
rewards_per_episode = []
epsilon_values = []

for episode in range(1, episodes + 1):
        state = env.reset()[0]
        done = False
        total_reward = 0
        agent.set_epsilon(init_epsilon - ((episode - 1) / (episodes - 1)) * (init_epsilon - final_epsilon))
        while not done:
                print(state)
                action = agent.choose_action(np.array([state]))
                next_state, reward, _ , done , info = env.step(action) # make the step in the environment based on the action
                agent.store_transition(state, action, reward)
                total_reward+=reward
                state = next_state # move to next state

        print('Episode {} finished.'.format(episode))
        agent.learn()
        rewards_per_episode.append(total_reward)
        epsilon_values.append(agent.epsilon)
        agent.save_model(os.path.join(os.getcwd(), 'keras', 'fantasyDrafter.keras'))
        env.agentRoster.to_csv(os.path.join(os.getcwd(), 'trainingRosters', 'iteration_' + str(episode) + '_roster.csv'), index=False)
        
# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot rewards on the primary y-axis
ax1.plot(rewards_per_episode, label='Total Reward', color='blue')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis to plot epsilon
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
ax2.plot(range(episodes), epsilon_values, label='Epsilon', color='orange', linestyle='--')
ax2.set_ylabel('Epsilon', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add a combined legend
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Add title and grid
plt.title('Total Reward and Epsilon Decay Over Episodes')
plt.grid(True)

# Save the plot to a file
plt.savefig(os.path.join(os.getcwd(), 'plots', '2_10_l_06_005_e_1000_ep.png'))

# Show the plot
plt.show()