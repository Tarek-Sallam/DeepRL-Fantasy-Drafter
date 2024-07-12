from Environment import DraftEnv
from gym.wrappers import FlattenObservation
import tensorflow as tf
import numpy as np
import os

#create the draft env and flatten the observation (since it is dict)
env = DraftEnv(10, os.path.join(os.getcwd(), 'data', 'projection_data.csv'))
env = FlattenObservation(env)
input_shape = env.observation_space.shape[0] ## get the shape
num_actions = env.action_space.n ## get the number of actions

## our policy network, with 3 layers, 
policyNetwork = tf.keras.models.Sequential()
policyNetwork.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(input_shape,)))
policyNetwork.add(tf.keras.layers.Dense(10, activation='relu'))
policyNetwork.add(tf.keras.layers.Dense(num_actions, activation='softmax'))
learning_rate = 0.001

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

episodes = 1
discount_factor = 0.99

episode_rewards = []
episode_lengths = []

### training loop
for episode in range(episodes):
    state = env.reset() # reset the environment (and get initial state)
    episode_reward = 0 # total reward in the episode 
    states = [] # all the states
    actions = [] # all the actions
    rewards = [] # all the rewards

    ### training iteration loop
    while True:
        action_probs = policyNetwork.predict(np.array([state]))[0] #make a prediction and record the probabilities
        action = np.random.choice(num_actions, p=action_probs) # random pick based on the probabilities
        next_state, reward, _ , done , info = env.step(action) # make the step in the environment based on the action
        states.append(state) # append the state
        actions.append(action) # append the action
        rewards.append(reward) # append the reward

        state = next_state # move to next state
        episode_reward += reward # increase the total reward

        ## if complete episode output to terminal and break the episode loop
        if done:
            print('Episode {} finished.'.format(episode))
            break
    
    discounted_rewards = np.zeros_like(rewards) ## create an empty array for discounted reward
    ## loop through each reward and apply the discount factor. Record the discounted rewards in new array
    for i in reversed(range(len(rewards))):
        discounted_rewards[i] = (discounted_rewards[i+1] if i != len(rewards) - 1 else 0) * discount_factor + rewards[i]
    
    ## normalize the discounted rewards
    discounted_rewards -= np.mean(discounted_rewards) 
    discounted_rewards /= np.std(discounted_rewards)

    ## convert the states actions and rewards to tensors
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards)

    ## using tensorflow gradient tape to record calculations made (for back-propogation)
    with tf.GradientTape() as tape:
        action_probs = policyNetwork(states) # calculate the probabilities again for all of the states used

        #calculate the loss by the REINFORCE loss function
        #tf.gather to gather all of the action probabilities that were taken, then take the log of that
        #then multiply by the discounted rewards, (i.e each log probability * each reward)
        #then finally sum it up and take the negative (to perform gradient ascent (to maximize rewards))
        loss = -tf.reduce_sum(tf.cast(tf.math.log(tf.gather(action_probs,actions,axis=1,batch_dims=1)),tf.float64) * discounted_rewards)
    
    # calculate the gradients for the parameters using the loss
    gradients = tape.gradient(loss, policyNetwork.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policyNetwork.trainable_variables)) # apply the gradients

    episode_rewards.append(episode_reward) # append the reward of the episode

    policyNetwork.save('keras/') # save the network

policyNetwork.summary()