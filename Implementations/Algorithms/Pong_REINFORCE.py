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
from PIL import Image as img
from datetime import datetime
import random

from torch.distributions import Categorical

# Constants
GAMMA = 0.9

class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def save_image(self, I, imgName):
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
        state = self.prepro(state).to(self.device)
        x = F.relu(self.linear1(state))
        # TODO: when refactor get_action remember to
        # take this softmax function away
        x = F.softmax(self.linear2(x), dim=1)
        return x

class Agent:

    def __init__(self, env, learning_rate=1e-3, save_plot=True, show_plot=False, torch_rand=True,
                 baseline=None):
        self.env = env
        self.show_plot = show_plot
        self.save_plot = save_plot

        self.num_actions = self.env.action_space.n
        self.policy_network = PolicyNetwork(6400, self.env.action_space.n)

        self.torch_rand = torch_rand

        print("Looking for GPU support...")
        using_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if using_cuda else "cpu")
        print("using cuda:",using_cuda)
        self.policy_network.set_device(self.device)

#         if cuda:
#             self.policy_network.cuda()
#             self.device = torch.device("cuda:0") # Uncomment this to run on GPU
#         else:
#             self.device = torch.device("cpu")

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        if self.torch_rand:
            logits = self.policy_network(state)
            dist = Categorical(logits=logits)
            highest_prob_action = dist.sample()
            log_prob = dist.log_prob(highest_prob_action)
        else:
            probs = self.policy_network(state).cpu()
            highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
            log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        return highest_prob_action, log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            for pw, r in enumerate(rewards[t:]):
                Gt = Gt + GAMMA**pw * r
            discounted_rewards.append(Gt)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum().to(self.device)
        policy_gradient.backward()
        self.optimizer.step()

    def train(self, max_episode=3000, max_step=200):
        print("Trainning agent for {} episodes".format(episodes))
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
        print("Avg reward:",sum(episode_rewards)/max_episode)
        plt.plot(episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        if self.show_plot:
            plt.show()
        if self.save_plot:
            plt.savefig('pong_{}_episodes_{}.png'.format(max_episode,
                int(datetime.timestamp(datetime.now()))))


        return episode_rewards

    def set_seeds(self, s):
        np.random.seed(s)
        torch.manual_seed(s)
        random.seed(s)

if __name__ == '__main__':
    env = gym.make("Pong-v0")
    agent = Agent(env)
    trand = False if len(sys.argv) > 2 and sys.argv[2] == 0 else True
    print("Using Gym random?", trand)
    base = None if len(sys.argv) <= 3 else sys.argv[3]
    print("Using baseline?", base is not None)
    agent = Agent(env, torch_rand=trand, baseline=base)
    episodes = int(sys.argv[1])
    agent.set_seeds(42)
    rewards = agent.train(episodes,sys.maxsize)
