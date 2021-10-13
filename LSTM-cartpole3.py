# -*- coding: utf-8 -*-
import os
import gym
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Activation
#from keras.optimizers import Adam

import numpy as np

####Hyperparameters
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
      print(e)

class New_DQN:
    def __init__(self):



        #        if os.path.exists('dqn.h5'):
        #            self.model.load_weights('dqn.h5')
        # ------------------------Initialize replay memory with capacity maximum #2000 experiences--------------------


        # ------------------------Initialize action-value function Q with random weights theta------------------------
        # ------------------------Initialize target action-value function Q^ with target weight theta^ = theta------
        self.model = self.build_model()

        self.target_model = self.build_model()
        self.update_target_model()
        if os.path.exists('dqn.h5'):
            self.model.load_weights('dqn.h5')
        self.memory_buffer = []
        self.buffer_x = []
        self.buffer_nx = []

        self.max_seq = 64
        self.memory_size = 10000
        self.learning_start = 2000
        self.next_idx = 0
        # define parameters
        # discount rate
        self.gamma = 0.95
        # epsilon value, start with 1, decay during iterations
        self.epsilon = 1.0
        self.epsilon_decay = 0.997

        self.epsilon_min = 0.01
        self.env = gym.make('CartPole-v0')
    def build_model(self):

        network_inputs = Input(shape=(1, 4))
        x = LSTM(256,input_shape=(1024,32,4), return_sequences=True)(network_inputs)
        x = LSTM(256,return_sequences=True)(x)
        x = LSTM(256,return_sequences=True)(x)
        x = LSTM(256, return_sequences=True)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=network_inputs, outputs=x)

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):
        # epsilon greedy policy to select action
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 1)
        else:
            state = [state]
            state = np.array(state)
            q_value = self.model.predict(state)
            return np.argmax(q_value)

    def store_experience(self, state, action, reward, next_state, done):
        # add experience in memory buffer
        i = (state, action, reward, next_state, done)
        if len(self.memory_buffer) <= self.memory_size:
            self.memory_buffer.append(i)
            self.buffer_x.append(i[0])
            self.buffer_nx.append(i[3])
        else:
            self.memory_buffer[self.next_idx] = i
            self.buffer_x[self.next_idx] = i[0]
            self.buffer_nx[self.next_idx] = i[3]
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def clear_buffer(self):
        self.memory_buffer = []
        self.buffer_x = []
        self.buffer_nx = []

    def process_batch(self, batch_size):
        # process batch data from memory buffer
        X = []
        NX = []
        #for i in range(batch_size):
        #    finish = random.randint(self.max_seq, len(self.memory_buffer)-1)
        #    begin = finish - self.max_seq
        #    data = self.memory_buffer[begin:finish]
        #    X.append(self.buffer_x[begin:finish])
        #    NX.append(self.buffer_nx[begin:finish])
        finish = random.randint(batch_size,len(self.memory_buffer)-1)
        begin = finish - batch_size
        X.append(self.buffer_x[begin:finish])
        NX.append(self.buffer_nx[begin:finish])
        data = self.memory_buffer[begin:finish]

        X = np.array(X)
        X = np.concatenate(X)
        NX = np.array(NX)
        NX = np.concatenate(NX)
        y = self.model.predict(X)
        q = self.target_model.predict(NX)
        for i, (_, action, reward, _, done) in enumerate(data):
            # Pseudocode line 13: update y value, which is the target value
            target = reward
            if done:
                y[i][0][action] = target
            else:
                target += self.gamma * np.amax(q[i])
                y[i][0][action] = target

        # states are the states array of every data in the batch
        # y is the array of updated q_value for every state
        return X, y

    def train(self, episode, batch_size):
        # ---------------------------------------------------note
        self.model.compile(loss='mse', optimizer='adam')
        #self.target_model.compile(loss='mse', optimizer=Adam(1e-3))

        history = {'episode': [], 'episode_reward': [], 'loss': []}
        count = 0
        for i in range(episode):
            # -----------Pseudocode line 5: Initialize environment-------------------------------------------------
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                # reshape observation[0.1,0.1,0.1,0.1] to state[[0.1,0.1,0.1,0.1]]
                # ---------------Pseudocode line 5: Preprocess feature states for input to network---------------------------------
                state = observation.reshape(-1, 4)

                #self.buffer_x.append(state)
                #state = observation
                # ---------------Pseudocode line 7&8: Use probability epsilon to select a random action or select action with max q_value-------
                action = self.egreedy_action(state)
                # ---------------Pseudocode line 9: Execute emulator and observe reward rt and next state s(t+1)---------------------
                observation, reward, done, info = self.env.step(action)

                state_n = observation.reshape(-1, 4)
                #self.buffer_nx.append(state_n)
                # ---------------Pseudocode line 10: preprocess next state s(t+1)-------------------------
                #next_state = next_state.reshape(-1,4)
                reward_sum += reward
                # ---------------Pseudocode line 11: Store transition(state x, action, reward, next_state) to memory buffer-------------
                self.store_experience(state, action, reward, state_n, done)
                if len(self.memory_buffer) > self.learning_start:
                    #print("++++++++++++++++++++++++++++start learning from experience+++++++++++++++++++++++++++")
                    X, Y = self.process_batch(batch_size)
                    # Pseudocode line 14: perform a gradient descent step with respect to main network model
                    #loss = self.model.train_on_batch(X, Y)
                    td_loss = self.model.fit(X,Y,verbose=0)
                    loss = td_loss.history['loss'][0]
                    count += 1
                    self.update_epsilon()

                    # update target model every 10 steps
                    if count != 0 and count % 10 == 0:
                        # self.model_weights = self.model.get_weights()
                        self.update_target_model()
                # ---------------Pseudocode line 12: Sample random minibatch of transitions from memory buffer-----------

                #print("buffer size is: ", len(self.memory_buffer))
            count += 1

            if i % 1 == 0:
                history['episode'].append(i)
                history['episode_reward'].append(reward_sum)
                history['loss'].append(loss)
                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.4f}'.format(i, reward_sum, loss,
                                                                                          self.epsilon))
        self.model.save_weights('dqn.h5')
        return history

    def play(self):
        observation = self.env.reset()
        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 10:
            self.env.render()

            x = observation.reshape(-1, 4)
            x = [x]
            x = np.array(x)
            q_values = self.model.predict(x)
            action = np.argmax(q_values)
            observation, reward, done, _ = self.env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()

        self.env.close()


if __name__ == '__main__':
    model = New_DQN()
    history = model.train(30000,1024)
    model.play()
