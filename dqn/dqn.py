import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
# from keras.callbacks import callbacks
import tensorflow.keras.backend as backend
import random
from utility.hyperparameters import hyperparameters
from utility.expReplay import ExpReplay
from collections import deque
import datetime

class DQN:
    def __init__(self, ActionSpace, StateSpace):
        self.actionSpace = ActionSpace #number of possible actions
        self.StateSpace = StateSpace
        self.FrameStacks = hyperparameters.STACKED_FRAME_SIZE
        self.inputShape = (self.StateSpace[0], self.StateSpace[1], self.FrameStacks)

        self.QNetwork = self.CreateNetwork() #create action-value network
        self.TargetNetwork = self.CreateNetwork(); #create target-network

        self.UpdateTargetNetwork() #Initialize target-network with initial q-network parameters

        self.ExperienceBuffer = deque(maxlen=hyperparameters.EXP_SIZE) #init expierence replay buffer

        # init parameters for Tensorboard
        logDir = hyperparameters.SAVE_LOGS_PATH
        self.avg_q , self.avg_loss = 0, 0
        self.sess = tf.InteractiveSession()
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(logDir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        #init epsilon values
        self.epsilon = hyperparameters.START_EPSILON
        self.min_epsilon = hyperparameters.MIN_EPSILON
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / hyperparameters.EPSILON_DECAY_STEPS

        np.random.seed(1)
        tf.set_random_seed(1)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    def CreateNetwork(self):
        #creates the network for the model
        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=self.inputShape)) #input layer of shape (80,80,4) filter = 8x8
        model.add(Conv2D(64, 4, 4, subsample=(2, 2), activation='relu')) #first conv layer with filter 4x4
        model.add(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu')) #second conv layer with filter 3x3
        model.add(Flatten()) 
        model.add(Dense(512, activation='relu')) #fully connected layer
        model.add(Dense(self.actionSpace)) #outputlayer
        model.compile(loss='mse', optimizer=Adam(lr=0.00001) )
        model.summary()

        return model

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_steps = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total_Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average_Max_Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Steps/Episode', episode_steps)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_steps, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
    
    def UpdateTargetNetwork(self):
        self.TargetNetwork.set_weights(self.QNetwork.get_weights())

    def UpdateExperienceBuffer(self, state, action, reward, done, nextState):
        self.ExperienceBuffer.append((state, action, reward, done, nextState))

    def GetAction(self, State):
        State = np.float32(State / 255.0)
        if np.random.random() <= self.epsilon: #e-greedy decay policy
            return np.random.randint(self.actionSpace) #take random action 
        else:
            q_value = self.QNetwork.predict(State) #greedy policy
            self.avg_q += np.amax(q_value)
            return np.argmax(q_value[0])
        

    def UpdateEpsilon(self):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def SaveNetwork(self):
        self.QNetwork.save(hyperparameters.SAVE_MODEL_PATH + hyperparameters.DQN_MODEL_NAME)
        print("Successfully saved network")

    def UpdateNetworkFromExperience(self):
        # first gather some experience
        if len(self.ExperienceBuffer) < hyperparameters.MIN_EXP: 
            return

        self.UpdateEpsilon()

        expSample = random.sample(self.ExperienceBuffer, hyperparameters.EXP_SAMPLE_SIZE) #get samples from the experience buffer

        # SampleSize = s_Sample.shape[0] 
        Target = np.zeros((hyperparameters.EXP_SAMPLE_SIZE,self.actionSpace))#init traget q array
        #expStates = np.zeros((hyperparameters.EXP_SAMPLE_SIZE, self.StateSpace[0], self.StateSpace[1], self.FrameStacks))
        expStates = np.zeros((hyperparameters.EXP_SAMPLE_SIZE, self.StateSpace[0], self.StateSpace[1], self.FrameStacks))
        expNextStates = np.zeros((hyperparameters.EXP_SAMPLE_SIZE, self.StateSpace[0], self.StateSpace[1], self.FrameStacks))
        expAction, expReward, expDone = [],[],[]

        for i in range(hyperparameters.EXP_SAMPLE_SIZE):
            expStates[i] = expSample[i][0]
            expNextStates[i] = expSample[i][4]
            expAction.append(expSample[i][1])
            expReward.append(expSample[i][2])
            expDone.append(expSample[i][3])

        TargetQs = self.TargetNetwork.predict(expNextStates) #get qs of next state like in q learning

        # for i in range(SampleSize):
        for i in range(hyperparameters.EXP_SAMPLE_SIZE):
            if expDone[i]: #Terminal State
                Target[i][expAction[i]] = expReward[i]
            else:
                Target[i][expAction[i]] = expReward[i] + hyperparameters.LEARNINGRATE * np.amax(TargetQs[i])

        loss = self.QNetwork.train_on_batch(expStates, Target) #performs the gradient update
        self.avg_loss += loss
