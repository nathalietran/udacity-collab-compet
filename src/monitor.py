from collections import deque
import torch
import numpy as np


def maddpg_train(env, agent, n_episodes=3000, print_every=500):
    """Run the agents inside the environment.

    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    scores_deque = deque(maxlen=100)
    all_scores = []
    rolling_mean = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        scores = np.zeros(len(agent.agents))
        while True:
            action = agent.act(state, i_episode, add_noise=True)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(i_episode, state, action, reward, next_state, done)
            scores += reward
            state = next_state
            if np.any(done):
                break

        score_max = np.max(scores)
        scores_deque.append(score_max)
        score_mean = np.mean(scores_deque)

        all_scores.append(score_max)
        rolling_mean.append(score_mean)

        print('\r{} episode\tavg score {:.5f}\tmax score {:.5f}'
              .format(i_episode, score_mean, score_max), end='')
        if score_mean >= 0.5:
            print('\nEnvironment solved after {} episodes with the '
                  'average score {}\n'.format(i_episode, score_mean))
            agent.save()
            break

        if i_episode % print_every == 0:
            print()

    return all_scores, rolling_mean


def test_agent(env, agents, max_t=1000):
    """Test the trained agent
    Params
    ======
        max_t (int): maximum number of timesteps per episode
    """
    raise NotImplementedError
