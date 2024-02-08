import random
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

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def train_iql_model():
    """
    Train the IQL model.
    """
    # Normalize training data
    data_loader = IqlDataLoader(file_path='./data/log.csv', read_optimization=False)
    training_data = data_loader.training_data

    training_data1 = training_data[training_data["agentIndex"] // 6 == 0].reset_index(drop=True)
    training_data2 = training_data[training_data["agentIndex"] // 6 == 1].reset_index(drop=True)
    training_data3 = training_data[training_data["agentIndex"] // 6 == 2].reset_index(drop=True)
    training_data4 = training_data[training_data["agentIndex"] // 6 == 3].reset_index(drop=True)
    training_data5 = training_data[training_data["agentIndex"] // 6 == 4].reset_index(drop=True)

    is_normalize = True
    normalize_dic = None
    if is_normalize:
        normalize_dic1 = normalize_state(
            training_data1,
            STATE_DIM,
            normalize_indices=[STATE_DIM-3, STATE_DIM-2, STATE_DIM-1]
        )
        normalize_dic2 = normalize_state(
            training_data2,
            STATE_DIM,
            normalize_indices=[STATE_DIM-3, STATE_DIM-2, STATE_DIM-1]
        )
        normalize_dic3 = normalize_state(
            training_data3,
            STATE_DIM,
            normalize_indices=[STATE_DIM-3, STATE_DIM-2, STATE_DIM-1]
        )
        normalize_dic4 = normalize_state(
            training_data4,
            STATE_DIM,
            normalize_indices=[STATE_DIM-3, STATE_DIM-2, STATE_DIM-1]
        )
        normalize_dic5 = normalize_state(
            training_data5,
            STATE_DIM,
            normalize_indices=[STATE_DIM-3, STATE_DIM-2, STATE_DIM-1]
        )

        training_data1['reward'] = normalize_reward(training_data1)
        training_data2['reward'] = normalize_reward(training_data2)
        training_data3['reward'] = normalize_reward(training_data3)
        training_data4['reward'] = normalize_reward(training_data4)
        training_data5['reward'] = normalize_reward(training_data5)

        save_normalize_dict(normalize_dic1, "saved_model/IQLtest1")
        save_normalize_dict(normalize_dic2, "saved_model/IQLtest2")
        save_normalize_dict(normalize_dic3, "saved_model/IQLtest3")
        save_normalize_dict(normalize_dic4, "saved_model/IQLtest4")
        save_normalize_dict(normalize_dic5, "saved_model/IQLtest5")

    # Build replay buffer
    replay_buffer1 = ReplayBuffer()
    replay_buffer2 = ReplayBuffer()
    replay_buffer3 = ReplayBuffer()
    replay_buffer4 = ReplayBuffer()
    replay_buffer5 = ReplayBuffer()

    add_to_replay_buffer(replay_buffer1, training_data1, is_normalize)
    add_to_replay_buffer(replay_buffer2, training_data2, is_normalize)
    add_to_replay_buffer(replay_buffer3, training_data3, is_normalize)
    add_to_replay_buffer(replay_buffer4, training_data4, is_normalize)
    add_to_replay_buffer(replay_buffer5, training_data5, is_normalize)

    # print(len(replay_buffer.memory))

    # Train model
    model1 = IQL(dim_obs=STATE_DIM)
    model2 = IQL(dim_obs=STATE_DIM)
    model3 = IQL(dim_obs=STATE_DIM)
    model4 = IQL(dim_obs=STATE_DIM)
    model5 = IQL(dim_obs=STATE_DIM)

    train_model_steps(model1, replay_buffer1)
    train_model_steps(model2, replay_buffer2)
    train_model_steps(model3, replay_buffer3)
    train_model_steps(model4, replay_buffer4)
    train_model_steps(model5, replay_buffer5)

    # Save model
    model1.save_net("saved_model/IQLtest1")
    model2.save_net("saved_model/IQLtest2")
    model3.save_net("saved_model/IQLtest3")
    model4.save_net("saved_model/IQLtest4")
    model5.save_net("saved_model/IQLtest5")

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

        if i % 1000 == 0:
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
