import numpy as np
import torch
import pickle

from bidding_train_env.agent.base_agent import BaseAgent
from bidding_train_env.baseline.iql.iql import IQL

torch.manual_seed(1)
np.random.seed(1)


class IqlAgent(BaseAgent):
    """
    IQL方法训练的出价智能体
    """

    def __init__(self, budget=100, name="Iql-PlayerAgent", cpa=2, category=0):
        super().__init__(budget, name, cpa, category)

        # 模型加载
        self.model = IQL(dim_obs=16)
        self.model.load_net("./saved_model/IQLtest")

        # Load and apply normalization to test_state
        with open('./saved_model/IQLtest/normalize_dict.pkl', 'rb') as file:
            self.normalize_dict = pickle.load(file)

        self.remaining_budgets = []

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
        # budget consumption rate
        if len(self.remaining_budgets) == 0:
            budget_consumption_rate = 0
        else:
            budget_consumption_rate = (self.remaining_budgets[-1] - remaining_budget) / self.remaining_budgets[-1]

        # cost per mille & total value of winning impressions
        if len(history_pv_values) == 0:
            cost_per_mille = 0
            prev_total_value = 0
        else:
            prev_cost = self.remaining_budgets[-1] - remaining_budget
            prev_pv_values = np.array(history_pv_values[-1])
            prev_status = np.array(history_status[-1])
            prev_total_value = np.sum(prev_pv_values * prev_status)

            cost_per_mille = prev_cost / prev_total_value if prev_total_value > 0 else 0

        # auction win rate
        if len(history_status) == 0:
            prev_win_rate = 0
        else:
            prev_win_rate = np.mean(np.array(history_status[-1]))

        # print(f"budget_consumption_rate={budget_consumption_rate}")
        # print(f"cost_per_mille={cost_per_mille}")
        # print(f"prev_total_value={prev_total_value}")
        # print(f"prev_win_rate={prev_win_rate}")

        self.remaining_budgets.append(remaining_budget)

        # budget category one hot encoding
        budget_category = [0, 0, 0, 0, 0, 0]
        budget_category_idx = int(budget // 300 - 5)
        budget_category[budget_category_idx] = 1

        # agent category one hot encoding
        # agent_category = [0, 0, 0, 0, 0]
        # agent_category[int(self.category)] = 1

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
            time_left, budget_left,
            budget_consumption_rate, cost_per_mille, prev_win_rate,
            historical_bid_mean, last_three_bid_mean,
            historical_market_price_mean, historical_pv_values_mean, historical_reward_mean,
            historical_status_mean, last_three_market_price_mean, last_three_pv_values_mean,
            last_three_reward_mean, last_three_status_mean, current_pv_values_mean,
            current_pv_num, last_three_pv_num_total, historical_pv_num_total, prev_total_value
        ])

        test_state = np.concatenate((budget_category, test_state))

        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        for key, value in self.normalize_dict.items():
            test_state[key] = normalize(test_state[key], value["min"], value["max"])

        test_state = torch.tensor(test_state, dtype=torch.float)
        alpha = self.model.take_actions(test_state)
        bids = alpha * pv_values

        return bids
