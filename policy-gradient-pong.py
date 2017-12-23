import tensorflow as tf
import gym
import numpy as np
import os.path
import time

#Hyper parameters
env = gym.make('Pong-v0')
actionSpace = 2
observationSpace = 6400
discountFactor = 0.99
learningRate = 3e-4
numberOfGames = 100000

UP_ACTION = 2
DOWN_ACTION = 3
# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

def imageProcessor(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2, ::2, 0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

class NeuralNetwork(object):

    def __init__(self):
        self.sess = tf.InteractiveSession()

        self.inputImage = tf.placeholder(tf.float32, shape=[None, observationSpace])

        self.firstLayer = tf.contrib.layers.fully_connected(
            inputs=self.inputImage,
            num_outputs=200,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )

        self.secondLayer = tf.contrib.layers.fully_connected(
            inputs=self.firstLayer,
            num_outputs=1,
            activation_fn=tf.sigmoid
        )

        self.rewards = tf.placeholder(tf.float32, shape=[None, 1])
        self.actions = tf.placeholder(tf.float32, shape=[None, 1])

        self.loss = tf.losses.log_loss(
            labels=self.actions,
            predictions=self.secondLayer,
            weights=self.rewards
        )

        self.sess.run(tf.global_variables_initializer())

        self.train = tf.train.AdamOptimizer(learningRate).minimize(self.loss)

        self.checkpoint = os.path.join('checkpoint', 'policyNetwork.ckpt')

        self.saver = tf.train.Saver()

    def getAction(self, observation):
        return self.sess.run(self.secondLayer, feed_dict={self.inputImage: observation.reshape([1, -1])})

    def backProp(self, stateActionRewardTuples):
        obs, acts, rews = zip(*stateActionRewardTuples)

        obs = np.vstack(obs)
        acts = np.vstack(acts)
        rews = np.vstack(rews)

        self.sess.run(self.train, feed_dict={
            self.inputImage: obs,
            self.rewards: rews,
            self.actions: acts})

    def SaveNet(self):
        self.saver.save(self.sess, self.checkpoint)

    def LoadNet(self):
        self.saver.restore(self.sess, self.checkpoint)


def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards


nn = NeuralNetwork()

nn.LoadNet()

episodeNumber = 0
totalRewardSum = 0

while episodeNumber < numberOfGames:
    batchStateActionRewardTuples = []
    observation, terminal, reward = env.reset(), False, 0.0
    rewardSum = 0
    previousObservation = None
    while not terminal:
        # env.render()
        # time.sleep(0.01)
        currentObservation = imageProcessor(observation)
        deltaObservation = currentObservation - previousObservation if previousObservation is not None else np.zeros_like(currentObservation)
        previousObservation = currentObservation

        upProb = nn.getAction(deltaObservation)
        action = UP_ACTION if np.random.uniform() < upProb else DOWN_ACTION

        batchStateActionRewardTuples.append((deltaObservation, action_dict[action], reward))

        observation, reward, terminal, info = env.step(action)
        rewardSum += reward

    states, actions, rewardList = zip(*batchStateActionRewardTuples)

    rewards = discount_rewards(rewardList, discountFactor)
    rewards -= np.mean(rewards)
    rewards /= np.std(rewards)

    batchStateActionRewardTuples = list(zip(states, actions, rewards))

    nn.backProp(batchStateActionRewardTuples)

    episodeNumber += 1
    totalRewardSum += rewardSum
    rewardMean = totalRewardSum / episodeNumber
    print("Episode " + str(episodeNumber) + " - Reward: " + str(rewardSum) + " - Reward mean: " + str(rewardMean))

    if episodeNumber % 500 == 0:
        nn.SaveNet()
