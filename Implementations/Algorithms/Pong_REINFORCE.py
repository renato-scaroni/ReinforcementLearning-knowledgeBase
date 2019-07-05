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
import atari_wrappers

from torch.distributions import Categorical

# Constants
GAMMA = 0.9
WIDTH = 98
HEIGHT = 80
NUM_EPISODES = 7500

def save_image(I, imgName):
    # data = np.zeros((h, w, 3), dtype=np.uint8)
    # data[256, 256] = [255, 0, 0]
    pic = img.fromarray(I)
    pic.save(imgName+".png")
    # pic.show()

def to_torch(u):
    return torch.from_numpy(u).float().unsqueeze(0)

class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, state):
        v = to_torch(state).to(self.device)
        x = F.relu(self.linear1(v))
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
        self.policy_network = PolicyNetwork(WIDTH*HEIGHT, self.env.action_space.n)

        self.torch_rand = torch_rand
        self.baseline = baseline

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

    # Adaptive Reinforcement Baseline.
    # R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning
    @staticmethod
    def adaptive_baseline(last_return, last_baseline):
        return GAMMA*last_return + (1-GAMMA)*last_baseline

    # Optimal Constant Baseline (with two-sample approximation)
    # J. Peters, S. Schaal. Policy gradient methods for robotics
    @staticmethod
    def optimal_baseline(h, g, p, q):
        if q is None or p is None:
            return 0
        s, t = np.array(p), np.array(q)
        s = np.inner(s, s).sum()
        t = np.inner(t, t).sum()
        return (g*s + h*t)/(s+t)

    # Parameters:
    #  - rewards: array of episode rewards
    #  - log_probs: array of episode logprobs
    #  - last_g: last return
    #  - last_b: last baseline
    #  - last_log_probs: last episode's logprobs
    # Returns:
    #  - last return
    #  - last baseline
    #  - last log_probs
    def update_policy(self, rewards, log_probs, last_g, last_b, last_log_probs):
        discounted_rewards = []
        L = []
        g0 = 0

        for t in range(len(rewards)):
            gamma = GAMMA
            Gt = 0
            for r in rewards[t:]:
                Gt = Gt + gamma * r
                gamma *= GAMMA
            discounted_rewards.append(Gt)
            if self.baseline is not None:
                break
        g0 = discounted_rewards[0]

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            v = -log_prob
            L.append(v.cpu().detach().numpy())
            if self.baseline is None:
                v += Gt
            policy_gradient.append(v)
        policy_gradient = torch.stack(policy_gradient).sum()
        b = 0
        if self.baseline is not None:
            if self.baseline == 'adaptive':
                b = Agent.adaptive_baseline(last_g, last_b)
            elif self.baseline == 'optimal':
                b = Agent.optimal_baseline(g0, last_g, L, last_log_probs)
            policy_gradient = policy_gradient * (g0 - b)

        self.optimizer.zero_grad()
        policy_gradient = policy_gradient.sum().to(self.device)
        policy_gradient.backward()
        self.optimizer.step()

        return g0, b, L

    def train(self, max_episode=3000, max_step=200):
        print("Trainning agent for {} episodes".format(episodes))
        numsteps = []
        avg_numsteps = []
        episode_rewards = []
        g, b, L = 0, 0, None
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
                    g, b, L = self.update_policy(rewards, log_probs, g, b, L)
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

class DiffFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def prepro(I):
        """ prepro 210x160x3 uint8 frame into 7840 (98x80) 1D float vector """
        I = I[:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I

    def observation(self, obs):
        obs = np.array(obs)
        I, J = DiffFrame.prepro(obs[:,:,0:3]), DiffFrame.prepro(obs[:,:,3:6])
        return ((J-I).astype(np.float).ravel()+1)/2.0

def when_record(episode):
    return episode == 10 or episode == 100 or episode == 500 or episode % 1000 == 0 or \
            episode == NUM_EPISODES-1

def wrap_diff(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True, monitor=True):
    """Configure environment for DeepMind-style Atari.
    """
    # if episode_life:
        # env = atari_wrappers.EpisodicLifeEnv(env)
    # if scale:
        # env = atari_wrappers.ScaledFloatFrame(env)
    # if clip_rewards:
        # env = atari_wrappers.ClipRewardEnv(env)
    if frame_stack:
        env = atari_wrappers.FrameStack(env, 2)
    env = DiffFrame(env)
    if monitor:
        env = gym.wrappers.Monitor(env, '/tmp/videos', video_callable=when_record, resume=True)
    return env

if __name__ == '__main__':
    episodes = int(sys.argv[1])
    NUM_EPISODES = episodes
    env = wrap_diff(gym.make("Pong-v0"))
    trand = False if len(sys.argv) > 2 and sys.argv[2] == 0 else True
    print("Using Gym random?", trand)
    base = None if len(sys.argv) <= 3 else sys.argv[3]
    print("Using baseline?", base is not None)
    agent = Agent(env, torch_rand=trand, baseline=base)
    agent.set_seeds(42)
    rewards = agent.train(episodes,sys.maxsize)
