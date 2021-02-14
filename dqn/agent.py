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

        #Buffer that keeps the last 4 images of the states
        self.utilizer = Utility();
        self.stacked_frames = deque([np.zeros((hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1]), dtype=np.int) for i in range(hyperparameters.STACKED_FRAME_SIZE)], maxlen=4)

        if  os.path.isdir(hyperparameters.SAVE_MODEL_PATH):
            os.makedirs(hyperparameters.SAVE_MODEL_PATH)

    def preprocessImage(self, state, is_new_episode):  
        state = self.utilizer.Preprocess_Image(state)

        if is_new_episode:
            stacked_state = np.stack((state, state, state, state), axis=2)
            stacked_state = np.reshape([stacked_state], (1, 84, 84, 4))
        else:
            #Since deque append adds t right, we can fetch rightmost element
            stacked_state = np.reshape([state], (1, 84, 84, 1))
            stacked_state = np.append(stacked_state, self.stacked_frames[:, :, :, :3], axis=3)

        return stacked_state

    def train(self):
        
        for episode in tqdm(range(1, hyperparameters.MAX_EPISODES + 1), ascii=True, unit='episode'): 
            epsiodeStep = 0
            dead = False
            done = False
            LivesAtStart = self.max_lives

            obs = self.env.reset() #get first state

            for _ in range(random.randint(1, hyperparameters.NO_ACTION_STEPS)):
                obs, _, _, _ = self.env.step(1)

            self.stacked_frames = self.preprocessImage(obs, True)

            while not done:
                self.Steps += 1
                epsiodeStep += 1

                action = self.model.GetAction(self.stacked_frames) #get an action with an epsilon decay policy

                new_state, reward, done, info = self.env.step(action) #perform action predicted by the nn

                new_state = self.preprocessImage(new_state, False)

                if info['ale.lives'] < LivesAtStart:
                    dead = True
                    LivesAtStart = info['ale.lives']

                self.model.UpdateExperienceBuffer(self.stacked_frames, action, reward, dead, new_state) #save the experience in the expierience buffer

                self.model.UpdateNetworkFromExperience() #trains the network with samples from the expirience buffer
                self.total_reward += reward

                if self.Steps % hyperparameters.UPDATE_TARGET_EVERY == 0:
                    self.model.UpdateTargetNetwork()
                    self.model.SaveNetwork()

                if dead:
                    dead = False
                else:
                    self.stacked_frames = new_state

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
