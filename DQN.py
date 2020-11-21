import numpy as np
import random
import keras
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

TAU = 0.01

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
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]

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