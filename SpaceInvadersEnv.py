import gym
from cv2 import cv2
import numpy as np
from DQN import DQN
from expReplay import ExpReplay


# List of hyper-parameters and constants
EXP_SIZE = 100000
MINIBATCH_SIZE = 32
TOT_FRAME = 1000000
EPSILON_DECAY = 300000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0
# Number of frames to throw into network
NUM_FRAMES = 3


class SpaceInvaders:
    def __init__(self):
        #init environment
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()

        #init Expierience Replay Buffer
        self.replay_memory = ExpReplay(EXP_SIZE)

        #create the model
        self.model = DQN(self.env.action_space.n , NUM_FRAMES)

        #init an image buffer
        self.imageBuffer = []

        #fill image buffer with first 3 states
        s1, _, _, _ = self.env.step(0)
        s2, _, _, _ = self.env.step(0)
        s3, _, _, _ = self.env.step(0)
        self.imageBuffer = [s1, s2, s3]

    def preprocessImgaeBuffer(self):
        black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)) for x in self.imageBuffer]
        black_buffer = [x[1:85, :, np.newaxis] for x in black_buffer]
        return np.concatenate(black_buffer, axis=2)

    def train(self, num_frames):
        observation_num = 0
        curr_state = self.preprocessImgaeBuffer()
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        while observation_num < num_frames:
            if observation_num % 1000 == 999:
                print(("Executing loop %d" %observation_num))

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            initial_state = self.preprocessImgaeBuffer()
            self.process_buffer = []

            predict_movement, predict_q_value = self.model.predictAction(curr_state, epsilon)

            reward, done = 0, False
            for i in range(NUM_FRAMES):
                temp_observation, temp_reward, temp_done, _ = self.env.step(predict_movement)
                reward += temp_reward
                self.process_buffer.append(temp_observation)
                done = done | temp_done

            if observation_num % 10 == 0:
                print("We predicted a q value of ", predict_q_value)

            if done:
                print("Lived with maximum time ", alive_frame)
                print("Earned a total of reward equal to ", total_reward)
                self.env.reset()
                alive_frame = 0
                total_reward = 0

            new_state = self.preprocessImgaeBuffer()
            self.replay_memory.addExpierience((initial_state, predict_movement, reward, done, new_state))
            total_reward += reward

            if self.replay_memory.GetSize() > MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_memory.GetSampleExpierences(MINIBATCH_SIZE)
                self.model.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.model.target_train()

            # Save the network every 100000 iterations
            if observation_num % 10000 == 9999:
                print("Saving Network")
                self.model.save_network("saved.h5")

            alive_frame += 1
            observation_num += 1

    def simulate(self, path = "", save = False):
        """Simulates game"""
        done = False
        tot_award = 0
        if save:
            self.env.monitor.start(path, force=True)
        self.env.reset()
        self.env.render()
        while not done:
            state = self.preprocessImgaeBuffer()
            predict_movement = self.model.predictAction(state, 0)[0]
            self.env.render()
            observation, reward, done, _ = self.env.step(predict_movement)
            tot_award += reward
            self.process_buffer.append(observation)
            self.process_buffer = self.process_buffer[1:]
        if save:
            self.env.monitor.close()

    def calculate_mean(self, num_samples = 100):
        reward_list = []
        print("Printing scores of each trial")
        for i in range(num_samples):
            done = False
            tot_award = 0
            self.env.reset()
            while not done:
                state = self.preprocessImgaeBuffer()
                predict_movement = self.model.predictAction(state, 0.0)[0]
                observation, reward, done, _ = self.env.step(predict_movement)
                tot_award += reward
                self.process_buffer.append(observation)
                self.process_buffer = self.process_buffer[1:]
            print(tot_award)
            reward_list.append(tot_award)
        return np.mean(reward_list), np.std(reward_list)

    def load_network(self, path):
        self.model.load_network(path)