import gym, random, pickle, os.path, math, glob
import argparse
import rlscope.api as rlscope
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import tensorflow as tf


# from tensorboardX import SummaryWriter

# USE_CUDA = torch.cuda.is_available()
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0],enable=True)
class DRQN:
    def __init__(self, num_actions=2, state=None):  # device=torch.device("cpu")):
        """
        Initialize a deep Q-learning network as described in
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
            device: cpu or gpu (cuda:0)
        """
        super(DRQN, self).__init__()
        # self.device = device
        self.num_actions = num_actions

        self.input = Input(shape=(1, 4))
        self.lstm1 = LSTM(128, input_shape=(256, 32, 4), return_sequences=True)(self.input)
        self.lstm2 = LSTM(128, return_sequences=True)(self.lstm1)
        self.lstm3 = LSTM(128, return_sequences=True)(self.lstm2)
        self.dense1 = Dense(128, activation='relu')(self.lstm3)
        self.output = Dense(2, activation='linear')(self.dense1)
        self.state = state
        self.model = Model(inputs=self.input, outputs=self.output)
        # self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc4 = nn.Linear(7 * 7 * 64, 512)
        # self.gru = nn.GRU(512, num_actions, batch_first=True)  # input shape (batch, seq, feature)

    def forward(self):  # , hidden=None, max_seq=1, batch_size=1):
        # DQN input B*C*feature (32 4 84 84)
        # DRQN input B*C*feature (32*seq_len 4 84 84)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        # hidden = self.init_hidden(batch_size) if hidden is None else hidden
        # before go to RNN, reshape the input to (barch, seq, feature)
        # x = x.reshape(batch_size, max_seq, 512)
        # return self.gru(x, hidden)

        predict = self.model.predict(self.state)
        return predict

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

    def train(self, x=None, y=None):
        loss = self.model.fit(x, y, steps_per_epoch=32, verbose=0)
        return loss

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    # def init_hidden(self, batch_size):
    # initialize hidden state to 0
    # return torch.zeros(1, batch_size, self.num_actions, device=self.device, dtype=torch.float)


class Recurrent_Memory_Buffer(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=1000, max_seq=10):
        self.buffer = []
        self.memory_size = memory_size
        self.max_seq = max_seq
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size:  # buffer not full
            self.buffer.append(data)
        else:  # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        # sample episodic memory
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            finish = random.randint(self.max_seq, self.size() - 1)
            begin = finish - self.max_seq

            data = self.buffer[begin:finish]
            state, action, reward, next_state, done = zip(*data)
            states.append(np.concatenate([self.observe(state_i) for state_i in state]))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.concatenate([self.observe(state_i) for state_i in next_state]))
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        # state = torch.from_numpy(lazyframe._force().transpose(2, 0, 1)[None] / 255).float()
        # if self.USE_CUDA:
        # state = state.cuda()
        state = lazyframe.reshape(1, 4)
        state = [state]
        state = np.array(state)
        return state


class DRQNAgent:
    # DRQN agent
    def __init__(self, action_space=None, USE_CUDA=False, memory_size=10000, epsilon=1, lr=1e-4,
                 max_seq=8, batch_size=32):
        self.USE_CUDA = USE_CUDA
        # self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.action_space = action_space
        self.rec_memory_buffer = Recurrent_Memory_Buffer(memory_size)
        self.DRQN = DRQN(num_actions=action_space.n)
        self.DRQN_target = DRQN(num_actions=action_space.n)
        # self.DRQN_target.load_state_dict(self.DRQN.state_dict())

        # if USE_CUDA:
        # self.DRQN = self.DRQN.cuda()
        # self.DRQN_target = self.DRQN_target.cuda()
        self.optimizer = RMSprop(learning_rate=lr)
        self.DRQN.model.compile(optimizer=self.optimizer, loss='mse')

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        # state = torch.from_numpy(lazyframe._force().transpose(2, 0, 1)[None] / 255).float()
        # if self.USE_CUDA:
        # state = state.cuda()
        state = lazyframe.reshape(1, 4)
        state = [state]
        state = np.array(state)
        return state

    def value(self, state):
        # get q_values of a given state
        q_values = self.DRQN.model.predict(state)
        return q_values

    def act(self, state, epsilon=None):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        if epsilon is None: epsilon = self.epsilon
        q_values = self.value(state)
        q_values = q_values.squeeze(1)
        if random.random() < epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        """ Compute td loss using torch operations only."""
        actions = tf.convert_to_tensor(actions)  # shape: [batch_size * seq_len]
        rewards = tf.convert_to_tensor(rewards)  # shape: [batch_size * seq_len]
        is_done = tf.convert_to_tensor(is_done)  # shape: [batch_size * seq_len]

        actions = tf.reshape(actions, [-1])
        rewards = tf.reshape(rewards, [-1])
        is_done = tf.reshape(is_done, [-1])
        states = tf.reshape(states, [batch_size * max_seq, 1, 4])
        next_states = tf.reshape(next_states, [batch_size * max_seq, 1, 4])
        # if self.USE_CUDA:
        # actions = actions.cuda()
        # rewards = rewards.cuda()
        # is_done = is_done.cuda()

        # get q-values for all actions in current states
        predicted_qvalues = self.DRQN.model.predict(states, steps=1)
        # predicted_qvalues = predicted_qvalues.reshape(-1, self.action_space.n)
        # predicted_qvalues = predicted_qvalues.squeeze(0)

        # select q-values for chosen actions
        # a = np.concatenate(actions)

        # predicted_qvalues_for_actions = predicted_qvalues[
        #    range(states.shape[0]), actions
        # ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.DRQN_target.model.predict(next_states, steps=1)  # YOUR CODE
        # predicted_next_qvalues = predicted_next_qvalues.squeeze(0)
        predicted_next_qvalues = predicted_next_qvalues.reshape(-1, self.action_space.n)

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(-1)
        next_state_values_arg = predicted_next_qvalues.argmax(-1)
        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma * next_state_values

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = tf.where(
            is_done, rewards, target_qvalues_for_actions)
        # if is_done:
        #    target_qvalues_for_actions = rewards
        # else:
        #    target_qvalues_for_actions = target_qvalues_for_actions
        for i in range(len(target_qvalues_for_actions)):
            j = next_state_values_arg[i]
            predicted_qvalues[i][0][j] = target_qvalues_for_actions[i]
        # mean squared error loss to minimize
        loss = self.DRQN.train(states, predicted_qvalues)

        return loss

    def sample_from_buffer(self, batch_size):
        # rewriten sample() in buffer with pytorch operations
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in range(batch_size):
            finish = random.randint(self.max_seq, self.rec_memory_buffer.size() - 1)
            begin = finish - self.max_seq

            data = self.rec_memory_buffer.buffer[begin:finish]
            state, action, reward, next_state, done = zip(*data)
            states.append(np.concatenate([self.observe(state_i) for state_i in state]))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.concatenate([self.observe(state_i) for state_i in next_state]))
            dones.append(done)

        states = tf.convert_to_tensor(states)
        next_states = tf.convert_to_tensor(next_states)

        # return np.concatenate(states), np.concatenate(actions), np.concatenate(rewards), np.concatenate(next_states), np.concatenate(dones)
        return states, actions, rewards, next_states, dones

    def learn_from_experience(self, batch_size):
        # learn from experience
        if self.rec_memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)

            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            # self.optimizer.zero_grad()
            # td_loss.backward()
            # for param in self.DRQN.model.get_weights():
            # clip the gradient
            #    param.grad.data.clamp_(-1, 1)
            # self.optimizer.step()
            return td_loss.history['loss'][0]
        else:
            return (0)


if __name__ == '__main__':
    #################################
    parser = argparse.ArgumentParser(description="Evaluate an RL policy")
    # rlscope will add custom arguments to the argparse argument parser
    # that allow you to customize (e.g., "--rlscope-directory <dir>"
    # for where to store results).
    rlscope.add_rlscope_arguments(parser)
    args = parser.parse_args()

    # Using the parsed arguments, rlscope will instantiate a singleton
    # profiler instance (rlscope.prof).
    rlscope.handle_rlscope_args(
        parser=parser,
        args=args,
    )
    # Provide a name for the algorithm and simulator (env) used so we can
    # generate meaningful plot labels.
    # The "process_name" and "phase_name" are useful identifiers for
    # multi-process workloads.
    rlscope.prof.set_metadata({
        'algo': 'LSTM',
        'env': 'CartPole-v1',
    })
    process_name = 'Real_LSTM_CartPole'
    phase_name = process_name
    #####################################
    env = gym.make('CartPole-v1')
    # env = wrap_deepmind(env, scale = False, frame_stack=True)

    gamma = 0.99  # discount factor
    epsilon_max = 1  # epsilon greedy parameter max
    epsilon_min = 0.01  # epsilon greedy parameter min
    eps_decay = 3000  # epsilon greedy parameter decay
    frames = 3000  # total training frames
    USE_CUDA = True  # training with gpu
    learning_rate = 2e-4  # learning rate
    max_buff = 10000  # maximum buffer size
    update_tar_interval = 1000  # frames for updating target network
    batch_size = 32
    max_seq = 8

    print_interval = 100
    log_interval = 1000
    learning_start = 1000  # 10000

    action_space = env.action_space
    print("action space is:", action_space)
    # action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    # state_channel = env.observation_space.shape[2]
    agent = DRQNAgent(action_space=action_space, USE_CUDA=USE_CUDA, lr=learning_rate, max_seq=max_seq,
                      batch_size=batch_size)
    frame = env.reset()

    episode_reward = 0
    print("episode_reward is:", episode_reward)
    all_rewards = []
    print("all_rewards is:", all_rewards)
    losses = []
    print("losses is:", losses)
    episode_num = 0
    print("episode_num is:", episode_num)
    # tensorboard
    # summary_writer = SummaryWriter(log_dir="DRQN", comment="good_makeatari")

    # e-greedy decay
    epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
        -1. * frame_idx / eps_decay)
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])

    for i in range(frames):
        # print("i is:",i)
        epsilon = epsilon_by_frame(i)
        state_tensor = agent.observe(frame)
        # print("state_tensor is", state_tensor)
        action = agent.act(state_tensor, epsilon)
        # print("action is: ",action)


        next_frame, reward, done, _ = env.step(action)

        episode_reward += reward
        agent.rec_memory_buffer.push(frame, action, reward, next_frame, done)
        frame = next_frame

        loss = 0
        if agent.rec_memory_buffer.size() >= learning_start:
            # print("learn from experience+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            loss = agent.learn_from_experience(batch_size)
            losses.append(loss)

        if i % print_interval == 0:
            mean = np.mean(all_rewards[-10:]).item()
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (
                i, mean, loss, epsilon, episode_num))
            # summary_writer.add_scalar("Temporal Difference Loss", loss, i)
            # summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
            # summary_writer.add_scalar("Epsilon", epsilon, i)

        if i % update_tar_interval == 0:
            print("TARGET GET UPDATED")
            print("****************************************************************************************")
            agent.DRQN_target.set_weights(agent.DRQN.get_weights())
            # agent.DRQN_target.load_state_dict(agent.DRQN.state_dict())

        if done:
            print("IT IS DONE")
            print('i is', i)
            print("--------------------------------------------------------------------------------------")
            frame = env.reset()
            # reset hidden to None
            all_rewards.append(episode_reward)
            print("all reward now is: ", all_rewards[-10:])
            episode_reward = 0
            episode_num += 1
            avg_reward = float(np.mean(all_rewards[:]))

    # summary_writer.close()

    observation = env.reset()
    count = 0
    reward_sum = 0
    random_episodes = 0

    while random_episodes < 10:
        # env.render()

        x = observation.reshape(-1, 4)
        x = [x]
        x = np.array(x)
        q_values = agent.DRQN.model.predict(x)
        # print(q_values)
        action = np.argmax(q_values)
        # print(action)

        observation, reward, done, _ = env.step(action)

        count += 1
        reward_sum += reward

        if done:
            print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
            random_episodes += 1
            reward_sum = 0
            count = 0
            observation = env.reset()

    env.close()
