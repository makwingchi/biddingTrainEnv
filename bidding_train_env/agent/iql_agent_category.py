import numpy as np
import torch
import pickle

from bidding_train_env.agent.base_agent import BaseAgent
from bidding_train_env.baseline.iql.iql import IQL


class IqlAgent(BaseAgent):
    """
    IQL方法训练的出价智能体
    """

    def __init__(self, budget=100, name="Iql-PlayerAgent", cpa=2,category=0):
        super().__init__(budget, name, cpa,category)

        # 模型加载
        self.model1 = IQL(dim_obs=16)
        self.model2 = IQL(dim_obs=16)
        self.model3 = IQL(dim_obs=16)
        self.model4 = IQL(dim_obs=16)
        self.model5 = IQL(dim_obs=16)
        self.model6 = IQL(dim_obs=16)

        self.model1.load_net("./saved_model/IQLtest1")
        self.model2.load_net("./saved_model/IQLtest2")
        self.model3.load_net("./saved_model/IQLtest3")
        self.model4.load_net("./saved_model/IQLtest4")
        self.model5.load_net("./saved_model/IQLtest5")
        self.model6.load_net("./saved_model/IQLtest6")

        # Load and apply normalization to test_state
        with open('./saved_model/IQLtest1/normalize_dict.pkl', 'rb') as file:
            self.normalize_dict1 = pickle.load(file)

        with open('./saved_model/IQLtest2/normalize_dict.pkl', 'rb') as file:
            self.normalize_dict2 = pickle.load(file)

        with open('./saved_model/IQLtest3/normalize_dict.pkl', 'rb') as file:
            self.normalize_dict3 = pickle.load(file)

        with open('./saved_model/IQLtest4/normalize_dict.pkl', 'rb') as file:
            self.normalize_dict4 = pickle.load(file)

        with open('./saved_model/IQLtest5/normalize_dict.pkl', 'rb') as file:
            self.normalize_dict5 = pickle.load(file)

        with open('./saved_model/IQLtest6/normalize_dict.pkl', 'rb') as file:
            self.normalize_dict6 = pickle.load(file)

    def reset(self):
        self.remaining_budget = self.budget

    def action(self, tick_index, budget, remaining_budget, pv_values, history_pv_values, history_bid,
               history_status, history_reward, history_market_price):
        """
        根据当前状态生成出价。

        :param tick_index: 当前处于第几个tick
        :param budget: 出价智能体总预算
        :param remaining_budget: 出价智能体剩余预算
        :param pv_values: 该tick的流量价值
        :param history_pv_values: 历史tick的流量价值
        :param history_bid: 该出价智能体历史tick的流量出价
        :param history_status: 该出价智能体历史tick的流量竞得状态(1代表竞得，0代表未竞得)
        :param history_reward: 该出价智能体历史tick的流量竞得奖励（竞得流量的reward为该流量价值，未竞得流量的reward为0）
        :param history_market_price: 该出价智能体历史tick的流量市场价格
        :return: numpy.ndarray of bid values
        """
        model_selection = {
            1500: self.model1,
            1800: self.model2,
            2100: self.model3,
            2400: self.model4,
            2700: self.model5,
            3000: self.model6
        }

        normalize_dict_selection = {
            1500: self.normalize_dict1,
            1800: self.normalize_dict2,
            2100: self.normalize_dict3,
            2400: self.normalize_dict4,
            2700: self.normalize_dict5,
            3000: self.normalize_dict6
        }

        curr_model = model_selection[int(budget)]
        curr_normalize_dict = normalize_dict_selection[int(budget)]

        # agent category one hot encoding
        agent_category = [0, 0, 0, 0, 0]
        agent_category[int(self.category)] = 1

        time_left = (24 - tick_index) / 24
        budget_left = remaining_budget / budget if budget > 0 else 0

        # 计算历史状态的均值
        historical_status_mean = np.mean([np.mean(status) for status in history_status]) if history_status else 0
        # 计算历史回报的均值
        historical_reward_mean = np.mean([np.mean(reward) for reward in history_reward]) if history_reward else 0
        # 计算历史市场价格的均值
        historical_market_price_mean = np.mean(
            [np.mean(price) for price in history_market_price]) if history_market_price else 0
        # 计算历史pvValue的均值
        historical_pv_values_mean = np.mean([np.mean(value) for value in history_pv_values]) if history_pv_values else 0
        # 历史调控单元的出价均值
        historical_bid_mean = np.mean([np.mean(bid) for bid in history_bid]) if history_bid else 0

        # Calculate mean of the last three ticks for different history data
        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_status_mean = mean_of_last_n_elements(history_status, tick_index)
        last_three_reward_mean = mean_of_last_n_elements(history_reward, tick_index)
        last_three_market_price_mean = mean_of_last_n_elements(history_market_price, tick_index)
        last_three_pv_values_mean = mean_of_last_n_elements(history_pv_values, tick_index)
        last_three_bid_mean = mean_of_last_n_elements(history_bid, tick_index)

        current_pv_values_mean = np.mean(pv_values)
        current_pv_num = len(pv_values)

        historical_pv_num_total = sum(len(bids) for bids in history_bid) if history_bid else 0
        last_three_pv_num_total = sum(
            len(history_bid[i]) for i in range(max(0, tick_index - 3), tick_index)) if history_bid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_three_market_price_mean, last_three_pv_values_mean,
            last_three_reward_mean, last_three_status_mean, current_pv_values_mean,
            current_pv_num, last_three_pv_num_total, historical_pv_num_total
        ])

        test_state = np.concatenate((agent_category, test_state))

        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        for key, value in curr_normalize_dict.items():
            test_state[key] = normalize(test_state[key], value["min"], value["max"])

        test_state = torch.tensor(test_state, dtype=torch.float)
        alpha = curr_model.take_actions(test_state)
        bids = alpha * pv_values

        return bids
