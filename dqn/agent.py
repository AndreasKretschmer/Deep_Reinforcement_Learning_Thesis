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

        # if  os.path.isdir(hyperparameters.SAVE_MODEL_PATH):
        #     os.makedirs(hyperparameters.SAVE_MODEL_PATH)

    def skipFrames(self, action, LivesAtStart):
        totalReward = 0

        for _ in range(hyperparameters.STACKED_FRAME_SIZE):
            obs, reward, done, info = self.env.step(action)
            totalReward += reward

            if info['ale.lives'] < LivesAtStart: #check if agent lost a live
                done = True
                totalReward -= 1

            if done:
                break
        
        return obs, totalReward, done, info


    def train(self):
        
        for episode in tqdm(range(1, hyperparameters.MAX_EPISODES + 1), ascii=True, unit='episode'): 
            epsiodeStep = 0
            dead = False
            done = False
            LivesAtStart = self.max_lives

            obs = self.env.reset() #get first state

            #do nothing at the beginning of an epsiode 
            for _ in range(random.randint(1, hyperparameters.NO_ACTION_STEPS)):
                lastObs = obs
                obs, _, _, _ = self.env.step(0)

            StackedState = self.utilizer.GetInitialStateForEpisode(obs, lastObs)
            # StackedState = self.preprocessImage(obs, True)

            while not done:
                self.Steps += 1
                epsiodeStep += 1
                lastObs = obs

                action = self.model.GetAction(StackedState) #get an action with a decay epsilon greedy policy
                
                # obs, reward, done, info = self.env.step(action) #perform action predicted by the nn
                obs, reward, done, info = self.skipFrames(action, LivesAtStart)

                processedObs = self.utilizer.Preprocess_Image(obs, lastObs)
                nextStackedState = np.append(StackedState[1:, :, :], processedObs, axis=0)
                # obs = self.preprocessImage(obs, False) #preprocess the new State

                if info['ale.lives'] < LivesAtStart: #check if agent lost a live
                    dead = True
                    LivesAtStart = info['ale.lives']

                self.model.UpdateExperienceBuffer(StackedState, action, reward, dead, nextStackedState) #save the experience in the expierience buffer

                self.model.UpdateNetworkFromExperience() #trains the network with samples from the expirience buffer

                self.total_reward += reward

                if self.Steps % hyperparameters.UPDATE_TARGET_EVERY == 0:
                    self.model.UpdateTargetNetwork()
                    self.model.SaveNetwork()

                if dead:
                    dead = False
                else:
                    StackedState = nextStackedState

                if done:
                    #update values for TensorBoard
                    if self.Steps > hyperparameters.MIN_EXP:
                        stats = [self.total_reward, self.model.avg_q / epsiodeStep, epsiodeStep,
                                self.model.avg_loss / epsiodeStep]
                        for i in range(len(stats)):
                            self.model.sess.run(self.model.update_ops[i], feed_dict={
                                self.model.summary_placeholders[i]: float(stats[i])
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
