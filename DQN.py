import numpy as np
import random
import keras
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from TensorBoard import ModifiedTensorBoard
from keras.callbacks import TensorBoard
from collections import deque
import time



class DQN:
    TAU = 0.01
    DECAY_RATE = 0.99
    MODEL_NAME = 'DQN'
    MIN_REPLAY_MEMORY_SIZE = 5000
    REPLAY_MEMORY_SIZE = 100000
    SAMPLE_SIZE = 64
    DISCOUNT = 0.99
    UPDATE_TARGET_EVERY = 5

    def __init__(self, actionSpace, frameNumber):
        #set parameters
        self.actionSpace = actionSpace
        self.frameNumber = frameNumber

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        #Init tensorboard for statistics
        self.tensorboard = TensorBoard(log_dir="logs/{}-{}".format(self.MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        #Init Replay_memory
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)


    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(88, 80, self.frameNumber)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.actionSpace, activation='linear')) # actionSpace = how many choices in spaceInvaders(6)
        model.compile(loss='mse', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def predictAction(self, data):
        return self.model.predict(data.reshape(1, 88, 80, self.frameNumber), batch_size = 1)

    def train(self, terminal_state, step):
        
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.SAMPLE_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=self.SAMPLE_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_train()
            self.target_update_counter = 0

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = self.TAU * model_weights[i] + (1 - self.TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]