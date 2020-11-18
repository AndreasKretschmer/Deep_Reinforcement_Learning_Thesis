import gym

env = gym.make('SpaceInvaders-v0')
# env.reset()

print(env.action_space) #actionspace of 6 actions

for episode in range(2):
    observation = env.reset()
    for _ in range(1000):
        #render game
        env.render()

        #print observation
        print(observation)

        #do step and get new observation, reward, state and terminationstate
        observation, reward, done, info   = env.step(env.action_space.sample())
        print(reward)
        #print values after each step
        print('reward: {:f}'.format(reward))
        print('done: {}'.format(done))
        print('info: {}'.format(info))

        if info['ale.lives'] == 0:
            break

env.close()