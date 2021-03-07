import time
import threading
import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras import backend as K
from A3C.Worker import Worker
from utility.hyperparameters import hyperparameters

class agent:
    def __init__(self):
        # environment settings
        self.StateSpace = (hyperparameters.RESIZE_IMAGE_SIZE[0], hyperparameters.RESIZE_IMAGE_SIZE[1], hyperparameters.STACKED_FRAME_SIZE)
        self.actionSpace = hyperparameters.ACTION_SIZE

        # optimizer parameters
        self.actorLearningrate = hyperparameters.ACTOR_LEARNINGRATE
        self.criticLearningrate = hyperparameters.CRITIC_LEARNINGRATE
        self.threads = hyperparameters.NUMBER_WORKERS
        self.LearningRate = hyperparameters.LEARNINGRATE

        # create model for actor and critic network
        self.actor, self.critic = self.CreateNetwork()

        # method for training actor and critic network
        self.optimizer = [self.ActorOptimizer(), self.CriticOptimizer()]

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = self.SetupSummary()
        self.summary_writer = tf.summary.FileWriter('logs/breakout_a3c', self.sess.graph)

    def train(self):
        agents = [Worker(self.actionSpace, self.StateSpace, [self.actor, self.critic], self.sess, self.optimizer,
                        self.LearningRate, [self.summary_op, self.summary_placeholders,
                        self.update_ops, self.summary_writer], i) for i in range(self.threads)]

        for agent in agents:
            time.sleep(1)
            agent.start()

        while True:
            time.sleep(60*10)
            self.SaveNetwork("./models/breakout_a3c")

    def CreateNetwork(self):
        input = Input(shape=self.StateSpace)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fullyConnected = Dense(256, activation='relu')(conv)
        policy = Dense(self.actionSpace, activation='softmax')(fullyConnected)
        value = Dense(1, activation='linear')(fullyConnected)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def ActorOptimizer(self):
        action = K.placeholder(shape=[None, self.actionSpace])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * advantages
        actor_loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = actor_loss + 0.01*entropy
        optimizer = RMSprop(lr=self.actorLearningrate, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

        return train

    def CriticOptimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = RMSprop(lr=self.criticLearningrate, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def SaveNetwork(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + '_critic.h5')
        print("Successfully saved network")

    def SetupSummary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def LoadModel(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")