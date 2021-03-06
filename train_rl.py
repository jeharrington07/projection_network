#https://github.com/jorditorresBCN/Deep-Reinforcement-Learning-Explained/blob/master/DRL_15_16_17_DQN_Pong.ipynb

from dqn import DQN
from env import make_env

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import collections
import datetime


#experience replay buffer
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

#agent
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):

        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def train(env_name):
    MEAN_REWARD_BOUND = 19           

    #hyperparameters
    gamma = 0.99                   
    batch_size = 32                
    replay_size = 10000            
    learning_rate = 1e-4           
    sync_target_frames = 1000      
    replay_start_size = 10000      

    eps_start=1.0
    eps_decay=.999985
    eps_min=0.02
    env = make_env(env_name)

    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + env_name)
    
    buffer = ExperienceReplay(replay_size)
    agent = Agent(env, buffer)

    epsilon = eps_start

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    frame_idx = 0  

    best_mean_reward = None

    print("Training. Current time: ", datetime.datetime.now())

    while True:
            frame_idx += 1
            epsilon = max(epsilon*eps_decay, eps_min)

            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)

                mean_reward = np.mean(total_rewards[-100:])

                print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
                    frame_idx, len(total_rewards), mean_reward, epsilon))
                
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)

                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), env_name + "-best.dat")
                    best_mean_reward = mean_reward
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f" % (best_mean_reward))

                if mean_reward > MEAN_REWARD_BOUND:
                    print("Solved in %d frames!" % frame_idx)
                    break

            if len(buffer) < replay_start_size:
                continue

            batch = buffer.sample(batch_size)
            states, actions, rewards, dones, next_states = batch

            states_v = torch.tensor(states).to(device)
            next_states_v = torch.tensor(next_states).to(device)
            actions_v = torch.tensor(actions).to(device)
            rewards_v = torch.tensor(rewards).to(device)
            done_mask = torch.ByteTensor(dones).to(device)

            state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

            next_state_values = target_net(next_states_v).max(1)[0]

            next_state_values[done_mask] = 0.0

            next_state_values = next_state_values.detach()

            expected_state_action_values = next_state_values * gamma + rewards_v

            loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

            if frame_idx % sync_target_frames == 0:
                target_net.load_state_dict(net.state_dict())
        
    writer.close()
    
    print("Training finished after %d frames" %frame_idx)
    print("Current time: ", datetime.datetime.now())


-



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-game", help="for now you have to choose pong", default="pong", required=False)
    args= parser.parse_args()

    env_dict = {"pong": "PongDeterministic-v4", "breakout": "BreakoutDeterministic-v4"}
    env_name = env_dict[args.game]

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    train(env_name)    
