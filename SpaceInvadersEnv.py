import gym

env = gym.make('SpaceInvaders-v0')
# env.reset()

print(env.action_space) #actionspace of 6 actions
print(env.observation_space) #Box(210, 160, 3) = the givenpicture 210 x 160 pixels 3 colours

for episode in range(2):
    break
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

        #the game is lost, if the player has 0 lives left
        if info['ale.lives'] == 0:
            break

env.close()