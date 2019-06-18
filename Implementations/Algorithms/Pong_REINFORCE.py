import sys
import torch
import gym
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from MountainCarEnv import MountainCarEnv
from PIL import Image as img

# Constants
GAMMA = 0.9

class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def SaveImage(self, I, imgName):
        # data = np.zeros((h, w, 3), dtype=np.uint8)
        # data[256, 256] = [255, 0, 0]
        pic = img.fromarray(I)
        pic.save(imgName+".png")
        # pic.show()

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return torch.from_numpy(I.astype(np.float).ravel()).float().unsqueeze(0)

    def forward(self, state):
        state = self.prepro(state)
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

class Agent:

    def __init__(self, env, learning_rate=3e-4):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.policy_network = PolicyNetwork(6400, self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def train(self, max_episode=3000, max_step=200):
        numsteps = []
        avg_numsteps = []
        episode_rewards = []
        for episode in range(max_episode):
            state = env.reset()
            log_probs = []
            rewards = []
            episode_reward = 0
            new_state = state

            for steps in range(max_step):
                # env.render()
                action, log_prob = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                state = new_state

                if done:
                    self.update_policy(rewards, log_probs)
                    print("episode " + str(episode) + ": " + str(episode_reward))
                    numsteps.append(steps)
                    avg_numsteps.append(np.mean(numsteps[-10:]))
                    episode_rewards.append(episode_reward)
                    break

        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

if __name__ == '__main__':
    env = gym.make("Pong-v0")
    agent = Agent(env)
    agent.train(100,sys.maxsize)