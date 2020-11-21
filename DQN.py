import numpy as np
import random
import keras
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

TAU = 0.01
DECAY_RATE = 0.99

class DQN:
    def __init__(self, actionSpace, frameNumber):
        #set parameters
        self.actionSpace = actionSpace
        self.frameNumber = frameNumber

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())


    def create_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, self.frameNumber)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.actionSpace)) # actionSpace = how many choices (6)
        model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        return model

    def predictAction(self, data, epsilon):
        q_actions = self.model.predict(data.reshape(1, 84, 84, self.frameNumber), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.actionSpace)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.actionSpace))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, self.actionSpace), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, self.actionSpace), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print("We had a loss equal to ", loss)

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)
        
    def save_network(self, path):
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")