import numpy as np
import torch
import logging
from bidding_train_env.dataloader.iql_dataloader import IqlDataLoader
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.iql.iql import IQL

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16


def train_iql_model():
    """
    Train the IQL model.
    """
    # Normalize training data
    data_loader = IqlDataLoader(file_path='./data/log.csv', read_optimization=False)
    training_data = data_loader.training_data

    training_data_dict = {}

    for category_index in range(5):
        for budget_index in range(6):
            curr_training_data = (
                training_data
                .loc[(training_data["agentIndex"] % 6 == budget_index) & (training_data["agentIndex"] // 6 == category_index)]
                .reset_index(drop=True)
            )

            curr_key = f"{category_index}_{budget_index}"

            training_data_dict[curr_key] = curr_training_data

    is_normalize = True
    normalize_dic = None
    if is_normalize:
        for _key, _value in training_data_dict.items():
            curr_normalize_dict = normalize_state(_value, STATE_DIM, normalize_indices=[13, 14, 15])
            _value["reward"] = normalize_reward(_value)
            save_normalize_dict(curr_normalize_dict, f"saved_model/IQLtest{_key}")

    # Build replay buffer
    replay_buffer_dict = {}

    for _key, _value in training_data_dict.items():
        curr_replay_buffer = ReplayBuffer()
        add_to_replay_buffer(curr_replay_buffer, _value, is_normalize)
        replay_buffer_dict[_key] = curr_replay_buffer

    # Train model
    for _key, _value in training_data_dict.items():
        curr_model = IQL(dim_obs=STATE_DIM)
        train_model_steps(curr_model, replay_buffer_dict[_key])
        curr_model.save_net(f"saved_model/IQLtest{_key}")

    # Test trained model
    # test_state = np.ones(STATE_DIM)
    # test_trained_model(model, test_state)


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
        logger.info(f'Step: {i} Q_loss: {q_loss} V_loss: {v_loss} A_loss: {a_loss}')


def test_trained_model(model, test_state):
    test_tensor = torch.tensor(test_state, dtype=torch.float)
    action = model.take_actions(test_tensor)
    logger.info(f"Test action: {action}")


def run_iql():
    """
    Run IQL model training and evaluation.
    """
    train_iql_model()


if __name__ == '__main__':
    run_iql()
