import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


class Q(nn.Module):
    '''
    IQL-Q网络
    '''

    def __init__(self, dim_observation, dim_action):
        super(Q, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.obs_FC = nn.Linear(self.dim_observation, 64)
        self.action_FC = nn.Linear(dim_action, 64)
        self.FC1 = nn.Linear(128, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs_embedding = self.obs_FC(obs)
        action_embedding = self.action_FC(acts)
        embedding = torch.cat([obs_embedding, action_embedding], dim=-1)
        q = self.FC3(F.relu(self.FC2(F.relu(self.FC1(embedding)))))
        return q


class V(nn.Module):
    '''
        IQL-V网络
        '''

    def __init__(self, dim_observation):
        super(V, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        return self.FC4(result)


class Actor(nn.Module):
    '''
    IQL-动作网络
    '''

    def __init__(self, dim_observation, dim_action, log_std_min=-10, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.FC1 = nn.Linear(dim_observation, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC_mu = nn.Linear(64, dim_action)
        self.FC_std = nn.Linear(64, dim_action)

    def forward(self, obs):
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mu = self.FC_mu(x)
        log_std = self.FC_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, obs, epsilon=1e-6):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action, dist

    def get_action(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        return action.detach().cpu()

    def get_det_action(self, obs):
        mu, _ = self.forward(obs)
        return mu.detach().cpu()


class IQL:
    '''
    IQL网络
    '''

    def __init__(self, dim_obs=3, dim_actions=1, gamma=0.99, tau=0.01,
                 V_lr=1e-4, critic_lr=1e-4, actor_lr=1e-4,
                 network_random_seed=1, expectile=0.7, temperature=3.0):
        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions
        self.V_lr = V_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.network_random_seed = network_random_seed
        self.expectile = expectile
        self.temperature = temperature
        torch.random.manual_seed(self.network_random_seed)
        self.value_net = V(self.num_of_states)
        self.critic1 = Q(self.num_of_states, self.num_of_actions)
        self.critic2 = Q(self.num_of_states, self.num_of_actions)
        self.critic1_target = Q(self.num_of_states, self.num_of_actions)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Q(self.num_of_states, self.num_of_actions)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actors = Actor(self.num_of_states, self.num_of_actions)
        self.GAMMA = gamma
        self.tau = tau
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)
        self.deterministic_action = True
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.critic1.cuda()
            self.critic2.cuda()
            self.critic1_target.cuda()
            self.critic2_target.cuda()
            self.value_net.cuda()
            self.actors.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # self.bid_linear = torch.nn.Linear(3, 1)
        # self.marketPrice_linear = torch.nn.Linear(3, 1)
        # self.pvValue_linear = torch.nn.Linear(3, 1)
        # self.reward_linear = torch.nn.Linear(3, 1)
        # self.status_linear = torch.nn.Linear(3, 1)
        #
        # self.bid_optimizer = Adam(self.bid_linear.parameters(), lr=1e-4)
        # self.marketPrice_optimizer = Adam(self.bid_linear.parameters(), lr=1e-4)
        # self.pvValue_optimizer = Adam(self.bid_linear.parameters(), lr=1e-4)
        # self.reward_optimizer = Adam(self.bid_linear.parameters(), lr=1e-4)
        # self.status_optimizer = Adam(self.bid_linear.parameters(), lr=1e-4)

    def step(self, states, actions, rewards, next_states, dones):
        '''
        训练网络
        '''
        # bid = states[:, [13, 14, 15]]
        # marketPrice = states[:, [16, 17, 18]]
        # pvValue = states[:, [19, 20, 21]]
        # reward = states[:, [22, 23, 24]]
        # status = states[:, [25, 26, 27]]
        #
        # next_bid = next_states[:, [13, 14, 15]]
        # next_marketPrice = next_states[:, [16, 17, 18]]
        # next_pvValue = next_states[:, [19, 20, 21]]
        # next_reward = next_states[:, [22, 23, 24]]
        # next_status = next_states[:, [25, 26, 27]]
        #
        # weighted_bid = self.bid_linear(bid)
        # weighted_marketPrice = self.marketPrice_linear(marketPrice)
        # weighted_pvValue = self.pvValue_linear(pvValue)
        # weighted_reward = self.reward_linear(reward)
        # weighted_status = self.status_linear(status)
        #
        # weighted_next_bid = self.bid_linear(next_bid)
        # weighted_next_marketPrice = self.marketPrice_linear(next_marketPrice)
        # weighted_next_pvValue = self.pvValue_linear(next_pvValue)
        # weighted_next_reward = self.reward_linear(next_reward)
        # weighted_next_status = self.status_linear(next_status)
        #
        # front = states[:, 0:13]
        # back = states[:, 28:]
        #
        # next_front = next_states[:, 0:13]
        # next_back = next_states[:, 28:]
        #
        # states = torch.cat(
        #     (front, back, weighted_bid, weighted_marketPrice, weighted_pvValue, weighted_reward, weighted_status),
        #     dim=-1
        # )
        #
        # next_states = torch.cat(
        #     (next_front, next_back, weighted_next_bid, weighted_next_marketPrice, weighted_next_pvValue, weighted_next_reward, weighted_next_status),
        #     dim=-1
        # )

        self.value_optimizer.zero_grad()
        value_loss = self.calc_value_loss(states, actions)
        value_loss.backward()
        self.value_optimizer.step()
        # self.linear_optim()

        actor_loss = self.calc_policy_loss(states, actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # self.linear_optim()

        critic1_loss, critic2_loss = self.calc_q_loss(states, actions, rewards, dones, next_states)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # self.linear_optim()

        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)

        return critic1_loss.cpu().data.numpy(), value_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    # def linear_optim(self):
    #     self.bid_optimizer.zero_grad()
    #     self.bid_optimizer.step()
    #
    #     self.marketPrice_optimizer.zero_grad()
    #     self.marketPrice_optimizer.step()
    #
    #     self.pvValue_optimizer.zero_grad()
    #     self.pvValue_optimizer.step()
    #
    #     self.reward_optimizer.zero_grad()
    #     self.reward_optimizer.step()
    #
    #     self.status_optimizer.zero_grad()
    #     self.status_optimizer.step()

    def take_actions(self, states):
        '''
        输出动作
        '''
        states = torch.Tensor(states).type(self.FloatTensor)#.reshape(1, -1)

        # bid1, bid2, bid3 = states[:, [13]], states[:, [14]], states[:, [15]]
        # marketPrice1, marketPrice2, marketPrice3 = states[:, [16]], states[:, [17]], states[:, [18]]
        # pvValue1, pvValue2, pvValue3 = states[:, [19]], states[:, [20]], states[:, [21]]
        # reward1, reward2, reward3 = states[:, [22]], states[:, [23]], states[:, [24]]
        # status1, status2, status3 = states[:, [25]], states[:, [26]], states[:, [27]]
        #
        # weighted_bid = -0.3620 * bid1 + 0.5067 * bid2 + -0.3209 * bid3 + 0.5298
        # weighted_marketPrice = 0.0234 * marketPrice1 + -0.0828 * marketPrice2 + -0.1812 * marketPrice3 + 0.4492
        # weighted_pvValue = 0.3865 * pvValue1 + -0.3388 * pvValue2 + -0.2582 * pvValue3 + 0.3032
        # weighted_reward = -0.0813 * reward1 + -0.0797 * reward2 + 0.2889 * reward3 + 0.1936
        # weighted_status = -0.0735 * status1 + 0.5748 * status2 + -0.3763 * status3 + -0.3998
        #
        # front = states[:, 0:13]
        # back = states[:, 28:]

        # states = torch.cat(
        #     (front, back, weighted_bid, weighted_marketPrice, weighted_pvValue, weighted_reward, weighted_status),
        #     dim=-1
        # )

        if self.deterministic_action:
            actions = self.actors.get_det_action(states)
        else:
            actions = self.actors.get_action(states)
        actions = torch.clamp(actions, 0)
        actions = actions.cpu().data.numpy()
        return actions

    def calc_policy_loss(self, states, actions):
        with torch.no_grad():
            v = self.value_net(states)
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2)

        exp_a = torch.exp(min_Q - v) * self.temperature
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]))

        _, dist = self.actors.evaluate(states)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss

    def calc_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)
            min_Q = torch.min(q1, q2)

        value = self.value_net(states)
        value_loss = self.l2_loss(min_Q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.GAMMA * (1 - dones) * next_v)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        return critic1_loss, critic2_loss

    def update_target(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1. - self.tau) * target_param.data + self.tau * local_param.data)

    def save_net(self, save_path):
        '''
        存储模型
        '''
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.critic1, save_path + "/critic1" + ".pkl")
        torch.save(self.critic2, save_path + "/critic2" + ".pkl")
        torch.save(self.value_net, save_path + "/value_net" + ".pkl")
        torch.save(self.actors, save_path + "/actor" + ".pkl")
        # torch.save(self.bid_linear, save_path + "/bid_linear" + ".pkl")
        # torch.save(self.marketPrice_linear, save_path + "/marketPrice_linear" + ".pkl")
        # torch.save(self.pvValue_linear, save_path + "/pvValue_linear" + ".pkl")
        # torch.save(self.reward_linear, save_path + "/reward_linear" + ".pkl")
        # torch.save(self.status_linear, save_path + "/status_linear" + ".pkl")

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cuda:0'):
        '''
        加载模型
        '''
        if os.path.isfile(load_path + "/critic.pt"):
            self.critic1.load_state_dict(torch.load(load_path + "/critic1.pt", map_location='cpu'))
            self.critic2.load_state_dict(torch.load(load_path + "/critic2.pt", map_location='cpu'))
            self.actors.load_state_dict(torch.load(load_path + "/actor.pt", map_location='cpu'))
        else:
            self.critic1 = torch.load(load_path + "/critic1.pkl", map_location='cpu')
            self.critic2 = torch.load(load_path + "/critic2.pkl", map_location='cpu')
            self.actors = torch.load(load_path + "/actor.pkl", map_location='cpu')

            # self.bid_linear = torch.load(load_path + "/bid_linear.pkl", map_location='cpu')
            # self.marketPrice_linear = torch.load(load_path + "/marketPrice_linear.pkl", map_location='cpu')
            # self.pvValue_linear = torch.load(load_path + "/pvValue_linear.pkl", map_location='cpu')
            # self.reward_linear = torch.load(load_path + "/reward_linear.pkl", map_location='cpu')
            # self.status_linear = torch.load(load_path + "/status_linear.pkl", map_location='cpu')

        self.value_net = torch.load(load_path + "/value_net.pkl", map_location='cpu')
        # print("model stored path " + next(self.critic1.parameters()).device.type)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=self.V_lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)

        # cuda usage
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.critic1.cuda()
            self.critic2.cuda()
            self.value_net.cuda()
            self.actors.cuda()
            self.critic1_target.cuda()
            self.critic2_target.cuda()
        # print("model stored path " + next(self.critic1.parameters()).device.type)

    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)


if __name__ == '__main__':
    model = IQL()
    step_num = 100
    batch_size = 1000
    for i in range(step_num):
        states = np.random.uniform(2, 5, size=(batch_size, 3))
        next_states = np.random.uniform(2, 5, size=(batch_size, 3))
        actions = np.random.uniform(-1, 1, size=(batch_size, 1))
        rewards = np.random.uniform(0, 1, size=(batch_size, 1))
        terminals = np.zeros((batch_size, 1))
        states, next_states, actions, rewards, terminals = torch.tensor(states, dtype=torch.float), torch.tensor(
            next_states, dtype=torch.float), torch.tensor(actions, dtype=torch.float), torch.tensor(rewards,
                                                                                                    dtype=torch.float), torch.tensor(
            terminals, dtype=torch.float)
        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)
        print(f'step:{i} q_loss:{q_loss} v_loss:{v_loss} a_loss:{a_loss}')
