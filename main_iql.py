import random
import numpy as np
import torch

from run.run_iql import run_iql
# from run.run_iql_agent_category import run_iql
# from run.run_iql_agent_category_separate import run_iql
# from run.run_iql_budget_category_separate import run_iql

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


if __name__ == "__main__":
    """程序主入口，运行IQL算法"""
    run_iql()
