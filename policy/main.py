import gym
import torch
import numpy as np
from collections import deque
from agent_vpg import VPG

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0').unwrapped
# env = gym.make('MountainCar-v0').unwrapped
obs = env.reset()
n_obs = len(obs)
# Get number of actions from gym action space
n_actions = env.action_space.n
print('n_obs=%d, n_act=%d' % (n_obs, n_actions))

agent = VPG(n_obs, n_actions, device)

for iter in range(15000):
    memory = deque()
    
    scores = []
    for epi in range(128):
        obs = env.reset()
        state = torch.tensor([obs], dtype=torch.float, device=device)

        score = 0
        for step in range(10000):
            action = agent.select_action(state)
            
            obs, reward, done, _ = env.step(action)
            if done:
                mask = 0
            else:
                mask = 1
            score += reward

            # action_tensor = torch.zeros((1,n_actions), dtype=torch.long, device=device)
            # action_tensor[0,action] = 1  # one-hot
            action = torch.tensor([action], dtype=torch.long, device=device)
            reward = torch.tensor([reward], device=device)
            mask = torch.tensor([mask], device=device)
            memory.append([state, action, reward, mask])
            state = torch.tensor([obs], dtype=torch.float, device=device)
            
            if done:
                break
        scores.append(score)
    score_avg = np.mean(scores)
    print('{} episode score is {:.2f}'.format(128, score_avg))

    agent.train_model(memory)
