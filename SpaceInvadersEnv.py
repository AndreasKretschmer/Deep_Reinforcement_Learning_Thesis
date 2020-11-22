import gym
from cv2 import cv2
import numpy as np
from DQN import DQN
from keras.callbacks import TensorBoard
from tqdm import tqdm
from collections import deque
import os
import tensorflow as tf
import numpy as np
import time

# List of hyper-parameters and constants
EPISODES = 100000
EXP_SIZE = 100000
MIN_REWARD = 250
# Number of frames to throw into network
NUM_FRAMES = 3

#epsilon values for decay
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


class SpaceInvaders:
    def __init__(self):

        np.random.seed(1)
        tf.random.set_seed(1)

        # Memory fraction, used mostly when trai8ning multiple agents
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')

        #init environment
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()

        #create the model
        self.model = DQN(self.env.action_space.n , 4)
        
        #init an image buffer
        self.stack_size = 4
        self.color = color = np.array([210, 164, 74]).mean()
        self.stacked_frames = deque([np.zeros((88,80), dtype=np.int) for i in range(self.stack_size)], maxlen=4)

    def preprocess_observation(self, obs):

        # Crop and resize the image
        img = obs[25:201:2, ::2]

        # Convert the image to greyscale
        img = img.mean(axis=2)

        # Improve image contrast
        img[img==self.color] = 0

        # Next we normalize the image from -1 to +1
        img = (img - 128) / 128 - 1

        return img.reshape(88,80)

    def stack_frames(self, state, is_new_episode):
        # Preprocess frame
        frame = self.preprocess_observation(state)
        
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((88,80), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
            
            # Because we're in a new episode, copy the same frame 4x, apply elementwise maxima
            maxframe = np.maximum(frame,frame)
            self.stacked_frames.append(maxframe)
            self.stacked_frames.append(maxframe)
            self.stacked_frames.append(maxframe)
            self.stacked_frames.append(maxframe)
            
            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)
        else:
            #Since deque append adds t right, we can fetch rightmost element
            maxframe=np.maximum(self.stacked_frames[-1],frame)
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(maxframe)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=2) 
        
        return stacked_state

    def train(self, num_frames):
        epsilon = 1 #is going to be decay while training
        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            # Update tensorboard step every episode
            self.model.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            ep_rewards = []
            step = 1
            done = False
            current_state = self.env.reset()

            # Reset environment and get initial state
            current_state = self.stack_frames(current_state, True)

            while not done:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random_sample() > epsilon:
                    # Get action from Q table
                    action = np.argmax(self.model.predictAction(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, done, info = self.env.step(action)

                new_state = self.stack_frames(new_state, False)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                #if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                #    self.env.render()

                # Every step we update replay memory and train main network
                self.model.update_replay_memory((current_state, action, reward, new_state, done))
                self.model.train(done, step)

                current_state = new_state
                step += 1

            
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                # self.model.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    self.model.model.save(f'models/{self.model.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)


    #def load_network(self, path):
    #   self.model.load_network(path)