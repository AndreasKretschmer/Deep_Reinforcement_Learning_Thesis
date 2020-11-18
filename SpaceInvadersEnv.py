import gym

env = gym.make('SpaceInvaders-v0')
env.reset()

print(env.action_space)

for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())

env.close()