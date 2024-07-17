import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

class PolicyGradientAgent():
    def __init__(self, learning_rate = 0.001, discount_factor = 0.99, epsilon = 0.5, n_actions = 10, n_inputs = 10, n_layers = 2, layer_size = [16, 16]):
        self.policy = self.buildModel(n_actions = n_actions, n_inputs = n_inputs, n_layers = n_layers, layer_size=layer_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.policy.compile(optimizer=self.optimizer)
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def buildModel(self, n_actions=10, n_inputs=10, n_layers=2, layer_size=[16, 16]):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(n_inputs,))
        ])
        for i in range(n_layers):
            model.add(tf.keras.layers.Dense(layer_size[i], activation='relu'))
        model.add(tf.keras.layers.Dense(n_actions, activation='softmax'))
        return model
        
    def choose_action(self, observation):
        probs = self.policy.predict(observation)[0]
        return self.epsilon_greedy_choice(probs)
    
    def store_transition(self, observation, action, reward):
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.state_memory.append(observation)

    def epsilon_greedy_choice(self, probs):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(len(probs))
        else:
            return np.random.choice(len(probs), p=probs)

    def reset_epsilon(self, epsilon):
        self.epsilon = epsilon

    def learn(self):
        rewards = self.reward_memory
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - np.mean(returns))/(np.std(returns) + 1e-8) ## whiten the returns
        for state, action, return_t in zip(self.state_memory, self.action_memory, returns):
            state = tf.convert_to_tensor([np.array(state)], tf.float32)
            return_t = tf.convert_to_tensor(return_t, tf.float32)
            
            with tf.GradientTape() as tape:
                probs = self.policy(state)
                probs = tf.clip_by_value(probs, 1e-10, 1.0)
                log_prob = tf.math.log(probs[0, action]) ### take the log of the probabilities
                loss = -log_prob * return_t # 

            grads = tape.gradient(loss, self.policy.trainable_variables)
            grads = [tf.clip_by_norm(grad, 1.0) for grad in grads]
            self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []

    def save_model(self, path):
        self.policy.save(path)
    
    def load_model(self, path):
        self.policy.load_model(path)