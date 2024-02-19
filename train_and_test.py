import pickle

import numpy as np

from run.run_iql import run_iql
from run.run_evaluate import run_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="whether it is test mode or not", default="1")

    args = parser.parse_args()
    curr_mode = int(args.test)

    a = f"# You are in {'test' if curr_mode == 1 else 'train'} mode. #"
    b = "#" * len(a)

    print(b)
    print(a)
    print(b)

    if curr_mode == 1:
        rewards = []

        for episode in range(7):
            training_data, test_data = run_iql(episode=episode, val_mode=True)

            with open('./data/raw_data.pickle', 'wb') as file:
                pickle.dump(test_data, file)

            episodic_reward = run_test()

            rewards.append(episodic_reward)

        print(f"all episodic rewards: {rewards}")
        print(f"mean of rewards: {np.mean(rewards)}")
    else:
        run_iql(0, val_mode=False)
