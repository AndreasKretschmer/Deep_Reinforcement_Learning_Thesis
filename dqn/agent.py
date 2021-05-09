import gym
import numpy as np
from collections import deque
from utility.hyperparameters import hyperparameters
from dqn.dqn import DQNModel as DQNModel
from tqdm import tqdm
import random
import os
from skimage.color import rgb2gray
from skimage.transform import resize

def PreProcess(newObs, observe):
    observe = observe[25:201:]
    newObs = newObs[25:201:]
    PreProcessedObs = np.maximum(newObs, observe)
    PreProcessedObs = np.uint8(resize(rgb2gray(PreProcessedObs), (84, 84), mode='constant') * 255)
    return PreProcessedObs
    
class DQNAgent:
    def __init__(self):
        #creates the environment
        self.env = gym.make(hyperparameters.ENV_NAME)
        self.maxLives = self.env.unwrapped.ale.lives()
        self.resizeShape = (hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1], hyperparameters.STACKED_FRAME_SIZE)

        #creates the nn model
        self.model = DQNModel(hyperparameters.ACTION_SIZE, self.resizeShape)

        #values to track the performance of the agent
        self.total_reward = 0
        self.Steps = 0

        # self.PreProcessing = PreProcessing();

    def train(self):  
        
        for episode in tqdm(range(1, hyperparameters.MAX_EPISODES_DQN + 1), ascii=True, unit='episode'): 
            #initialize variables
            epsiodeStep, realAction = 0, 0
            dead, done = False, False
            LivesAtStart = self.maxLives

            obs = self.env.reset() #get first state
            newObs = obs

            #do nothing at the beginning of an epsiode 
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

                action = self.model.GetAction(FrameBuffer) #get an action with a decay epsilon greedy policy
                if hyperparameters.ACTION_SIZE < 4:
                    realAction = action + 1 #play without action 0
                else:
                    realAction = action #play with action 0

                #start a new episode
                if dead:
                    dead = False
                    realAction = 1 # start new episode directly
                    action = 0

                #perform action predicted by the nn
                newObs, reward, done, info = self.env.step(realAction)

                #preprocess and stack new observation
                newPreProcessedObs = PreProcess(newObs, obs)
                newPreProcessedObs = np.reshape([newPreProcessedObs], (1, 84, 84, 1))
                newFrameBuffer = np.append(newPreProcessedObs, FrameBuffer[:, :, :, :3], axis=3)

                if info['ale.lives'] < LivesAtStart: #check if agent lost a live
                    dead = True
                    LivesAtStart = info['ale.lives']

                self.total_reward += reward
                reward = np.clip(reward, -1., 1.)
                 
                self.model.UpdateExperienceBuffer(FrameBuffer, action, reward, dead, newFrameBuffer) #save the experience in the expierience buffer

                # if self.Steps % hyperparameters.SKIP_FRAMES == 0:
                self.model.UpdateNetworkFromExperience() #trains the network with samples from the expirience buffer

                #update the target network after given steps
                if self.Steps % hyperparameters.UPDATE_TARGET_EVERY == 0:
                    self.model.UpdateTargetNetwork()
                    self.model.SaveModel()

                #if agent loses the ball and still has lives left => get new initial state
                if dead:
                    FrameBuffer = np.stack((newPreProcessedObs, newPreProcessedObs, newPreProcessedObs, newPreProcessedObs), axis=2)
                    FrameBuffer = np.reshape([FrameBuffer], (1, 84, 84, 4))
                else:
                    FrameBuffer = newFrameBuffer

                if done:
                    #update values for TensorBoard
                    stats = [self.total_reward, self.model.avgQ / epsiodeStep, epsiodeStep,
                                self.model.avgLoss / epsiodeStep]
                    for i in range(len(stats)):
                        self.model.sess.run(self.model.update_ops[i], feed_dict={
                                self.model.summaryPlaceholders[i]: float(stats[i])
                            })
                    summary_str = self.model.sess.run(self.model.summary_op)
                    self.model.summary_writer.add_summary(summary_str, episode + 1)

                    #print stats for debugging
                    print("episode:", episode, 
                      "  score:", self.total_reward, 
                      "  memory length:", len(self.model.ExperienceBuffer),
                      "  epsilon:", self.model.epsilon,
                      "  global_step:", self.Steps, 
                      "  average_q:", (self.model.avgQ / epsiodeStep),
                      "  average loss:", (self.model.avgLoss / epsiodeStep))
                    self.model.avgQ, self.model.avgLoss, self.total_reward = 0, 0, 0

    def Evaluate(self):
        self.model.LoadModel()
        # self.env = gym.wrappers.Monitor(self.env, "./logs/breakout_dqn/recordings/eval_A3C", force=True)

        for episode in tqdm(range(1, 1000 + 1), ascii=True, unit='episode'): 
            #initialize variables
            epsiodeStep, realAction = 0, 0
            dead, done = False, False
            LivesAtStart = self.maxLives

            obs = self.env.reset() #get first state
            newObs = obs

            #do nothing at the beginning of an epsiode 
            for _ in range(random.randint(1, hyperparameters.NO_ACTION_STEPS)):
                obs = newObs
                newObs, _, _, _ = self.env.step(1)

            PreProcessedObs = PreProcess(newObs, obs)
            FrameBuffer = np.stack((PreProcessedObs, PreProcessedObs, PreProcessedObs, PreProcessedObs), axis=2)
            FrameBuffer = np.reshape([FrameBuffer], (1, 84, 84, 4))    

            while not done:
                self.env.render()
                self.Steps += 1
                epsiodeStep += 1
                obs = newObs

                action = self.model.GetActionEval(FrameBuffer) #get an action with a decay epsilon greedy policy
                if hyperparameters.ACTION_SIZE < 4:
                    realAction = action + 1 #play without action 0
                else:
                    realAction = action #play with action 0

                #start a new episode
                if dead:
                    dead = False
                    realAction = 1 # start new episode directly
                    action = 0

                #perform action predicted by the nn
                newObs, reward, done, info = self.env.step(realAction)

                #preprocess and stack new observation
                newPreProcessedObs = PreProcess(newObs, obs)
                newPreProcessedObs = np.reshape([newPreProcessedObs], (1, 84, 84, 1))
                newFrameBuffer = np.append(newPreProcessedObs, FrameBuffer[:, :, :, :3], axis=3)

                if info['ale.lives'] < LivesAtStart: #check if agent lost a live
                    dead = True
                    LivesAtStart = info['ale.lives']

                self.total_reward += reward

                #if agent loses the ball and still has lives left => get new initial state
                if dead:
                    FrameBuffer = np.stack((newPreProcessedObs, newPreProcessedObs, newPreProcessedObs, newPreProcessedObs), axis=2)
                    FrameBuffer = np.reshape([FrameBuffer], (1, 84, 84, 4))
                else:
                    FrameBuffer = newFrameBuffer

                if done:
                    #update values for TensorBoard
                    stats = [self.total_reward, self.model.avgQ / epsiodeStep, epsiodeStep,
                                self.model.avgLoss / epsiodeStep]
                    for i in range(len(stats)):
                        self.model.sess.run(self.model.update_ops[i], feed_dict={
                                self.model.summaryPlaceholders[i]: float(stats[i])
                            })
                    summary_str = self.model.sess.run(self.model.summary_op)
                    self.model.summary_writer.add_summary(summary_str, episode + 1)

                    #print stats for debugging
                    print("episode:", episode, 
                      "  score:", self.total_reward, 
                      "  memory length:", len(self.model.ExperienceBuffer),
                      "  epsilon:", self.model.epsilon,
                      "  global_step:", self.Steps, 
                      "  average_q:", (self.model.avgQ / epsiodeStep),
                      "  average loss:", (self.model.avgLoss / epsiodeStep))
                    self.model.avgQ, self.model.avgLoss, self.total_reward = 0, 0, 0
