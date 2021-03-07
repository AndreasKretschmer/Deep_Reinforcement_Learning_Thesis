import threading
import numpy as np
import gym
import random
from utility.hyperparameters import hyperparameters
from utility.PreProcessing import PreProcessing
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

def PreProcess(newObs, observe):
    observe = observe[25:201:]
    newObs = newObs[25:201:]
    PreProcessedObs = np.maximum(newObs, observe)
    PreProcessedObs = np.uint8(resize(rgb2gray(PreProcessedObs), (84, 84), mode='constant') * 255)
    return PreProcessedObs

global Episode
Episode = 0

class Worker(threading.Thread):
    def __init__(self, actionSpace, stateSpace, model, sess, optimizer, discFactor, summary_ops, threadNo):
        threading.Thread.__init__(self)

        #init training variables
        self.actionSpace = actionSpace
        self.stateSpace = stateSpace
        self.discountFactor = discFactor
        self.states, self.actions, self.rewards = [],[],[] #buffer to store last transitions
        self.env = gym.make(hyperparameters.ENV_NAME) #make local environment for this thread
        self.maxLives = self.env.unwrapped.ale.lives()


        self.actor, self.critic = model # assign global model
        self.local_actor, self.local_critic = self.BuildLocalModel() #init local model for this thread

        #variables to setup summary
        self.sess = sess
        self.optimizer = optimizer
        self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer = summary_ops

        self.PropActMaxAvg = 0
        self.LossAvg = 0
        self.total_reward = 0
        self.Steps = 0
        self.threadNo = threadNo

        # StepMax => number of steps for update of the model
        self.StepMax = hyperparameters.STEP_MAX

    def BuildLocalModel(self):
        input = Input(shape=self.stateSpace)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.actionSpace, activation='softmax')(fc)
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

    def run(self):
        global Episode
        self.Steps = 0

        while Episode < hyperparameters.MAX_EPISODES_A3C:
            #initialize variables
            epsiodeStep, self.total_reward = 0, 0
            dead, done = False, False
            LivesAtStart = self.maxLives

            obs = self.env.reset() #get first state
            newObs = obs

            #do nothing at the beginning of an epsiode - idea of Deepmind
            for _ in range(random.randint(1, hyperparameters.NO_ACTION_STEPS)):
                obs = newObs
                newObs, _, _, _ = self.env.step(1)


            PreProcessedObs = PreProcess(newObs, obs)
            FrameBuffer = np.stack((PreProcessedObs, PreProcessedObs, PreProcessedObs, PreProcessedObs), axis=2)
            FrameBuffer = np.reshape([FrameBuffer], (1, 84, 84, 4))            

            while not done:
                self.Steps += 1
                epsiodeStep += 1
                obs = newObs

                # get action for the current FrameBuffer and go one step in environment
                action = self.GetAction(FrameBuffer)

                if hyperparameters.ACTION_SIZE < 4:
                    realAction = action + 1 #play without action 0
                else:
                    realAction = action #play with action 0

                #start a new episode if agent died last step
                if dead:
                    dead = False
                    realAction = 1
                    action = 0

                newObs, reward, done, info = self.env.step(realAction) #perform Action

                newPreProcessedObs = PreProcess(newObs, obs)
                newPreProcessedObs = np.reshape([newPreProcessedObs], (1, 84, 84, 1))
                newFrameBuffer = np.append(newPreProcessedObs, FrameBuffer[:, :, :, :3], axis=3)

                self.PropActMaxAvg += np.amax(self.actor.predict(np.float32(FrameBuffer / 255.)))

                if LivesAtStart > info['ale.lives']:
                    dead = True
                    LivesAtStart = info['ale.lives']

                self.total_reward += reward
                reward = np.clip(reward, -1., 1.)

                # save Experience to calculate discounted reward
                self.SaveExperience(FrameBuffer, action, reward)

                #if agent loses the ball and still has lives left => get new initial state
                if dead:
                    FrameBuffer = np.stack((newPreProcessedObs, newPreProcessedObs, newPreProcessedObs, newPreProcessedObs), axis=2)
                    FrameBuffer = np.reshape([FrameBuffer], (1, 84, 84, 4))
                else:
                    FrameBuffer = newFrameBuffer

                #update model if episode ends or max steps were performed
                if self.Steps >= self.StepMax or done:
                    self.TrainGlobalModel(done)
                    self.UpdateLocalModel()
                    self.Steps = 0

                # if done
                if done:
                    #update values for TensorBoard
                    stats = [self.total_reward, self.PropActMaxAvg / float(epsiodeStep),
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
                         "  average_q:", self.PropActMaxAvg / float(epsiodeStep),
                         "  Excecuted by Thread:", self.threadNo)

                    #reset/update variables
                    self.PropActMaxAvg = 0
                    self.LossAvg = 0
                    epsiodeStep = 0
                    self.total_reward = 0
                    Episode += 1

    def CalcDiscRewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1] / 255.))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discountFactor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def TrainGlobalModel(self, done):
        discounted_rewards = self.CalcDiscRewards(self.rewards, done)

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


    def GetAction(self, State):
        State = np.float32(State / 255.0)
        policy = self.local_actor.predict(State)[0]
        ActionIdx = np.random.choice(self.actionSpace, 1, p=policy)[0]

        return ActionIdx

    # this is used for calculating discounted rewards
    def SaveExperience(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.actionSpace)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def UpdateLocalModel(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())
    