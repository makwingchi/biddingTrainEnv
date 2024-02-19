import pickle

import numpy as np

from run.run_iql import run_iql
from run.run_evaluate import run_test


if __name__ == "__main__":
    rewards = []

    for episode in range(7):
        training_data, test_data = run_iql(episode=episode, val_mode=True)

        with open('./data/raw_data.pickle', 'wb') as file:
            pickle.dump(test_data, file)

        episodic_reward = run_test()

        rewards.append(episodic_reward)

    print(f"all episodic rewards: {rewards}")
    print(f"mean of rewards: {np.mean(rewards)}")
