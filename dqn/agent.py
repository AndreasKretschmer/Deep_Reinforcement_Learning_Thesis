import gym
import numpy as np
from collections import deque
from utility.hyperparameters import hyperparameters
from utility.UtilityFunctions import Utility
from dqn.dqn import DQN as DQNModel
from tqdm import tqdm
import random
import os

class agent:
    def __init__(self):
        #creates the environment
        self.env = gym.make(hyperparameters.ENV_NAME)
        self.max_lives = self.env.unwrapped.ale.lives()

        #creates the nn model
        self.model = DQNModel(self.env.action_space.n, (hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1], hyperparameters.STACKED_FRAME_SIZE))

        #values to track the performance of the agent
        self.total_reward = 0
        self.Steps = 0

        self.utilizer = Utility();

    def train(self):  
        
        for episode in tqdm(range(1, hyperparameters.MAX_EPISODES_DQN + 1), ascii=True, unit='episode'): 
            epsiodeStep = 0
            dead = False
            done = False
            LivesAtStart = self.max_lives
            StackedState = deque([np.zeros((hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1]), dtype=np.uint8) for i in range(hyperparameters.STACKED_FRAME_SIZE)], maxlen=4)

            obs = self.env.reset() #get first state

            #do nothing at the beginning of an epsiode 
            for _ in range(random.randint(1, hyperparameters.NO_ACTION_STEPS)):
                lastObs = obs
                obs, _, _, _ = self.env.step(0)

            obs, StackedState = self.utilizer.GetInitialStackedState(StackedState, obs)
            # StackedState = self.utilizer.GetInitialStateForEpisode(obs, lastObs)
            # StackedState = self.preprocessImage(obs, True)

            while not done:
                self.Steps += 1
                epsiodeStep += 1
                lastObs = obs

                action = self.model.GetAction(obs) #get an action with a decay epsilon greedy policy
                
                # obs, reward, done, info = self.env.step(action) #perform action predicted by the nn
                newObs, reward, done, info = self.env.step(action)
                newObs, StackedState = self.utilizer.StackFrames(StackedState, newObs)
                # processedObs = self.utilizer.Preprocess_Image(obs, lastObs)
                # nextStackedState = np.append(StackedState[:, :, 1:], processedObs, axis=2)
                # obs = self.preprocessImage(obs, False) #preprocess the new State

                if info['ale.lives'] < LivesAtStart: #check if agent lost a live
                    dead = True
                    LivesAtStart = info['ale.lives']

                self.total_reward += reward
                reward = np.clip(reward, -1., 1.)
                 
                self.model.UpdateExperienceBuffer(obs, action, reward, dead, newObs) #save the experience in the expierience buffer

                if self.Steps % hyperparameters.SKIP_FRAMES == 0:
                    self.model.UpdateNetworkFromExperience() #trains the network with samples from the expirience buffer


                if self.Steps % hyperparameters.UPDATE_TARGET_EVERY == 0:
                    self.model.UpdateTargetNetwork()
                    self.model.SaveNetwork()

                if dead:
                    dead = False
                    obs, StackedState = self.utilizer.GetInitialStackedState(StackedState, obs)
                else:
                    obs = newObs

                if done:
                    #update values for TensorBoard
                    if self.Steps > hyperparameters.MIN_EXP:
                        stats = [self.total_reward, self.model.avg_q / epsiodeStep, epsiodeStep,
                                self.model.avg_loss / epsiodeStep]
                        for i in range(len(stats)):
                            self.model.sess.run(self.model.update_ops[i], feed_dict={
                                self.model.summaryPlaceholders[i]: float(stats[i])
                            })
                        summary_str = self.model.sess.run(self.model.summary_op)
                        self.model.summary_writer.add_summary(summary_str, episode + 1)

                    print("episode:", episode, 
                      "  score:", self.total_reward, 
                      "  memory length:", len(self.model.ExperienceBuffer),
                      "  epsilon:", self.model.epsilon,
                      "  global_step:", self.Steps, 
                      "  average_q:", (self.model.avg_q / epsiodeStep),
                      "  average loss:", (self.model.avg_loss / epsiodeStep))
                    self.model.avg_q, self.model.avg_loss, self.total_reward = 0, 0, 0
