import threading
import numpy as np
import gym
import random
from utility.hyperparameters import hyperparameters
from utility.UtilityFunctions import Utility
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from collections import deque
from tqdm import tqdm

global Episode
Episode = 0

class Worker(threading.Thread):
    def __init__(self, action_size, state_size, model, sess, optimizer, discount_factor, summary_ops, threadNo):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.env = gym.make(hyperparameters.ENV_NAME) #make local environment for this thread
        self.max_lives = self.env.unwrapped.ale.lives()

        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        self.states, self.actions, self.rewards = [],[],[]

        self.local_actor, self.local_critic = self.build_localmodel()
        self.utilizer = Utility()

        self.avg_q_max = 0
        self.avg_loss = 0
        self.total_reward = 0
        self.Steps = 0
        self.threadNo = threadNo

        # StepMax -> max batch size for training
        self.StepMax = hyperparameters.STEP_MAX

    def run(self):
        global Episode
        self.Steps = 0

        # for Episode in tqdm(range(1, hyperparameters.MAX_EPISODES_A3C + 1), ascii=True, unit='episode'):
        while Episode < hyperparameters.MAX_EPISODES_A3C:
            #initialize variables
            epsiodeStep, self.total_reward = 0, 0
            dead, done = False, False
            LivesAtStart = self.max_lives
            StackedState = deque([np.zeros((hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1]), dtype=np.uint8) for i in range(hyperparameters.STACKED_FRAME_SIZE)], maxlen=4)
            
            obs = self.env.reset() #get first state

            #do nothing at the beginning of an epsiode - idea of Deepmind
            for _ in range(random.randint(1, hyperparameters.NO_ACTION_STEPS)):
                obs, _, _, _ = self.env.step(0)

            obs, StackedState = self.utilizer.GetInitialStackedState(StackedState, obs)

            while not done:
                self.Steps += 1
                epsiodeStep += 1

                # get action for the current history and go one step in environment
                action, policy, qMax = self.get_action(obs)
                newObs, reward, done, info = self.env.step(action)

                #preprocess and stack new observation
                newObs, StackedState = self.utilizer.StackFrames(StackedState, newObs)

                self.avg_q_max += qMax

                if LivesAtStart > info['ale.lives']:
                    dead = True
                    LivesAtStart = info['ale.lives']

                self.total_reward += reward
                reward = np.clip(reward, -1., 1.)

                # save the sample <s, a, r> to the replay memory
                self.memory(obs, action, reward)

                #if agent loses the ball and still has lives left => get new initial state
                if dead:
                    dead = False
                    obs, StackedState = self.utilizer.GetInitialStackedState(StackedState, obs)
                else:
                    obs = newObs

                #update model if episode ends or max stepswere performed
                if self.Steps >= self.StepMax or done:
                    self.train_model(done)
                    self.update_localmodel()
                    self.Steps = 0

                # if done
                if done:
                    #update values for TensorBoard
                    stats = [self.total_reward, self.avg_q_max / float(epsiodeStep),
                             epsiodeStep]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, Episode + 1)

                    #print stats for debugging
                    print("episode:", Episode, 
                         "  score:", self.total_reward, 
                         "  step:", epsiodeStep, 
                         "  average_q:", self.avg_q_max / float(epsiodeStep),
                         "  Excecuted by Thread:", self.threadNo)

                    #reset/update variables
                    self.avg_q_max = 0
                    self.avg_loss = 0
                    epsiodeStep = 0
                    self.total_reward = 0
                    Episode += 1

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.array([np.float32(self.states[-1] / 255.)]))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_model(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        states = np.zeros((len(self.states), 84, 84, 4))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def build_localmodel(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.set_weights(self.actor.get_weights())
        critic.set_weights(self.critic.get_weights())

        actor.summary()
        critic.summary()

        return actor, critic

    def update_localmodel(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    def get_action(self, State):
        State = np.array([np.float32(State / 255.0)])
        policy = self.local_actor.predict(State)[0]
        qMax = np.amax(self.actor.predict(State))
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]

        return action_index, policy, qMax

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)
