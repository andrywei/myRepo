import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

n_hidden = 64
LR_policy = 0.005
LR_base = 0.005
GAMMA = 0.99
BATCH_SIZE = 64

class PolicyNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = nn.Linear(n_hidden, n_outputs)
        self.fc2.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class BaselineNet(nn.Module):
    def __init__(self, n_inputs):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc2 = nn.Linear(n_hidden, 1)
        self.fc2.weight.data.normal_(0, 0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'mask'))

class VPG():
    def __init__(self, n_obs, n_action, device):
        self.device = device
        self.n_action = n_action
        self.policy = PolicyNet(n_obs, n_action).to(device)
        self.baseline = BaselineNet(n_obs).to(device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=LR_policy)
        self.base_optim = torch.optim.Adam(self.baseline.parameters(), lr=LR_base)

    def select_action(self, state):
        with torch.no_grad():
            actions_prob = F.softmax(self.policy(state), 1).squeeze(0)
        action = np.random.choice(range(self.n_action), p=actions_prob.cpu().data.numpy()) # select action w.r.t the actions prob
        return action

    def _calc_Q(self, rewards, masks):
        Q = torch.zeros_like(rewards)
        q = 0
        for t in reversed(range(0, len(rewards))):
            q = rewards[t] + GAMMA * q * masks[t]
            Q[t] = q
        # Q = (Q - Q.mean()) / Q.std()
        return Q

    def _train_policy(self, states, actions, A):
        neg_log_p = F.cross_entropy(self.policy(states), actions, reduction='none')
        loss = torch.mean(neg_log_p * A)
        print('loss: %f' % loss.item())

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

    def _train_baseline(self, states, Q):
        criterion = torch.nn.MSELoss()
        n = states.shape[0]
        arr = np.arange(n)
        for epoch in range(5):
            np.random.shuffle(arr)

            for i in range(n // BATCH_SIZE):
                batch_index = arr[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
                batch_index = torch.tensor(batch_index, dtype=torch.long, device=self.device)
                inputs = states[batch_index]
                target = Q[batch_index]

                values = self.baseline(inputs).squeeze(1)
                loss = criterion(values, target)
                self.base_optim.zero_grad()
                loss.backward()
                self.base_optim.step()

    def train_model(self, memory):
        batch = Transition(*zip(*memory))
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        masks = torch.cat(batch.mask)

        Q = self._calc_Q(rewards, masks)
        V = self.baseline(states).squeeze(1).detach()
        A = Q - V
        self._train_policy(states, actions, A)
        self._train_baseline(states, Q)
