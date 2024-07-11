from Environment import DraftEnv

env = DraftEnv(10)

print(env.observation_space.sample())