import gym
from gym.wrappers import FlattenObservation
import numpy
import os
import tensorflow as tf
import numpy as np
from Agent import PolicyGradientAgent
import matplotlib.pyplot as plt
from Environment import DraftEnv
from scipy.signal import savgol_filter

teams = 12
rounds = 15

env = DraftEnv(
        teams,
        rounds, 
        os.path.join(os.getcwd(), 'data', 'projection_data.csv'), 
        os.path.join(os.getcwd(), 'data', 'adp_data.npy')
)

env = FlattenObservation(env)
n_inputs = env.observation_space.shape[0] ## get the shape
print(env.observation_space.shape)
n_actions = env.action_space.n
count = 0

settings = [
        [2, 12, 0.001, 0.001, 0.99, 0.01],
        # [2, 10, 0.005, 1],
        # [6, 10, 0.005, 1]
]

for setting in settings:
        init_epsilon = 0.0
        final_epsilon = 0.0
        layers_num = setting[0]
        layer_size = setting[1]
        learning_rate = setting[2]
        final_learning_rate = setting[3]
        discount_factor = setting[4]
        baseline_rate = setting[5]

        agent = PolicyGradientAgent(
                learning_rate=learning_rate, 
                discount_factor=discount_factor, n_actions = 
                n_actions, epsilon=init_epsilon, 
                n_inputs=n_inputs, 
                n_layers=layers_num, 
                layer_size=[layer_size] * layers_num
        )                                       

        episodes = 1000
        rewards_per_episode = []
        points_per_episode = []
        epsilon_values = []
        roster_per_episode = []
        loss_per_episode = []
        actions_per_episode = []

        baseline = 0

        for episode in range(1, episodes + 1):
                state = env.reset()[0]
                done = False
                total_reward = 0
                total_pts = 0
                roster_slots = 0
                actions_in_episode = []
                agent.set_epsilon(init_epsilon - ((episode - 1) / (episodes - 1)) * (init_epsilon - final_epsilon))
                agent.learning_rate = (learning_rate - ((episode - 1) / (episodes - 1)) * (learning_rate - final_learning_rate))

                while not done:
                        action = agent.choose_action(np.array([state]))
                        actions_in_episode.append(action)
                        next_state, reward, _ , done , info = env.step(action) # make the step in the environment based on the action
                        agent.store_transition(state, action, reward)
                        total_reward+=reward
                        total_pts+= info["points"]
                        if info["points"] != 0:
                                roster_slots += 1
                        state = next_state # move to next state

                print('Episode {} finished.'.format(episode))

                loss_per_episode.append(agent.learn(baseline)) ## append the loss per episode (and train the agent)
                avg_reward = total_reward / rounds ## calculate the average reward
                rewards_per_episode.append(total_reward-baseline) ## subtract the baseline and append the reward 
                points_per_episode.append(total_pts) ## append the points
                epsilon_values.append(agent.epsilon) ## append the epsilon values
                roster_per_episode.append(roster_slots) ## append the roster slots
                actions_per_episode.append(actions_in_episode) ## append the actions taken

                ## calculate the new baseline
                baseline = (1 - baseline_rate) * baseline + baseline_rate * avg_reward

                ## save the model and export the roster
                agent.save_model(os.path.join(os.getcwd(), 'keras', f'fantasyDrafter_{count}.keras'))
                env.agentRoster.to_csv(os.path.join(os.getcwd(), 'trainingRosters', 'iteration_' + str(episode) + '_roster.csv'), index=False)
                
        percentage = 0.2
        window_length = max(3, 2 * (int(episodes * percentage) // 2) + 1)  # Ensure integer division and odd window length
        poly_order = 2
        smoothed_rewards = savgol_filter(rewards_per_episode, window_length, poly_order)
        smoothed_points = savgol_filter(points_per_episode, window_length, poly_order)
        smoothed_loss = savgol_filter(loss_per_episode, window_length, poly_order)


        # Add a text box with the hyperparameters
        hyperparameters_text = (
                "Hyperparameters:\n"
                f"Episodes: {episodes}\n"
                f"Hidden Layers: {layers_num}\n"
                f"Layer Height: {layer_size}\n"
                f"Learning Rate: {learning_rate}\n"
                f"Discount Factor: {discount_factor}\n"
                f"Initial Epsilon: {init_epsilon}\n"
                f"Final Epsilon: {final_epsilon}\n"
        )

        # Points and Epsilon Decay Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(smoothed_points, label='Total Points', color='blue')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Points', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(range(episodes), epsilon_values, label='Epsilon', color='orange', linestyle='--')
        ax2.set_ylabel('Epsilon', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title('Total Points and Epsilon Decay Over Episodes')
        plt.grid(True)
        plt.subplots_adjust(right=0.75)

        plt.text(1.02, 0.5, hyperparameters_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))

        plt.savefig(os.path.join(os.getcwd(), 'plots', f"{count}_pts.png"))
        plt.close()  # Close the plot after saving to free memory

        # Rewards and Epsilon Decay Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(smoothed_rewards, label='Total Rewards', color='blue')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Rewards', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(range(episodes), epsilon_values, label='Epsilon', color='orange', linestyle='--')
        ax2.set_ylabel('Epsilon', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        plt.title('Total Rewards and Epsilon Decay Over Episodes')
        plt.grid(True)
        plt.subplots_adjust(right=0.75)

        plt.text(1.02, 0.5, hyperparameters_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))

        plt.savefig(os.path.join(os.getcwd(), 'plots', f"{count}_rewards.png"))
        plt.close()  # Close the plot after saving to free memory

        # Create the figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart for roster slots filled
        ax1.bar(range(1, episodes + 1), roster_per_episode, color='skyblue', label='Slots Filled')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Slots Filled')
        ax1.set_title('Roster Slots Filled Over Episodes with Epsilon Decay')
        ax1.grid(axis='y')

        # Secondary y-axis for epsilon decay
        ax2 = ax1.twinx()
        ax2.plot(range(1, episodes + 1), epsilon_values, color='orange', linestyle='--', label='Epsilon Decay')
        ax2.set_ylabel('Epsilon')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Adding the hyperparameters text box
        plt.text(1.02, 0.5, hyperparameters_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgrey'))

        # Add a legend
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        # Save the plot to a file
        plt.savefig(os.path.join(os.getcwd(), 'plots', f'{count}_roster.png'), bbox_inches='tight')
        plt.close()  # Close the plot after saving to free memory

        # Plot Average Loss per Episode
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(smoothed_loss, label='Average Loss', color='red')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Loss')
        ax.set_title('Average Loss per Episode')
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'plots', f"{count}_avg_loss.png"))
        plt.close()

        # Flatten the list of actions
        flat_actions = [action for episode_actions in actions_per_episode for action in episode_actions]

        # Plot a histogram of the actions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(flat_actions, bins=n_actions, color='blue', edgecolor='black')
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Distribution Over All Episodes')

        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), 'plots', f"{count}_actions.png"))
        plt.close()

        count+=1