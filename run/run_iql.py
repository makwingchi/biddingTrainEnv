import random
import numpy as np
import torch
import logging
from bidding_train_env.dataloader.iql_agent_onehot_dataloader import IqlDataLoader
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.iql.iql import IQL

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16 + 4 + 6

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def train_iql_model(episode=0, val_mode=False):
    """
    Train the IQL model.
    """
    # Normalize training data
    data_loader = IqlDataLoader(file_path='./data/log.csv', read_optimization=False)

    if not val_mode:
        training_data = data_loader.training_data
        test_data = None
    else:
        training_data = data_loader.training_data[~data_loader.training_data["episode"].eq(episode)].reset_index(drop=True)
        test_data = data_loader.raw_data[data_loader.raw_data["episode"].eq(episode)].reset_index(drop=True)

    is_normalize = True
    normalize_dic = None
    if is_normalize:
        normalize_dic = normalize_state(
            training_data,
            STATE_DIM,
            normalize_indices=[STATE_DIM-4, STATE_DIM-3, STATE_DIM-2, STATE_DIM-1]
        )
        training_data['reward'] = normalize_reward(training_data)
        save_normalize_dict(normalize_dic, "saved_model/IQLtest")

    print(training_data.loc[2000].to_dict())

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)

    # Train model
    model = IQL(dim_obs=STATE_DIM)
    train_model_steps(model, replay_buffer)

    # Save model
    model.save_net("saved_model/IQLtest")

    # Test trained model
    # test_state = np.ones(STATE_DIM)
    # test_trained_model(model, test_state)

    return training_data, test_data


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def train_model_steps(model, replay_buffer, step_num=20000, batch_size=100):
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)
        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)

        if i % 1000 == 0:
            logger.info(f'Step: {i} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')


def test_trained_model(model, test_state):
    test_tensor = torch.tensor(test_state, dtype=torch.float)
    action = model.take_actions(test_tensor)
    logger.info(f"Test action: {action}")


def run_iql(episode, val_mode):
    """
    Run IQL model training and evaluation.
    """
    return train_iql_model(episode, val_mode)


if __name__ == '__main__':
    curr_episode = 0
    curr_val_mode = True

    run_iql(episode=curr_episode, val_mode=curr_val_mode)
