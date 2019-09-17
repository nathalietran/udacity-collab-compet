from collections import deque
import torch
import numpy as np


def run_agent(env, agents, n_episodes=1000, print_every=100, max_t=1000):
    """Run the agents inside the environment.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = np.reshape(env_info.vector_observations,
                            (1, agents.nb_agents * state_size))
        agents.reset()
        score = np.zeros(agents.nb_agents)
        for t in range(max_t):
            actions = agents.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = np.reshape(env_info.vector_observations,
                                     (1, agents.nb_agents * state_size))
            rewards = env_info.rewards
            dones = env_info.local_done
            agents.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break

        scores_deque.append(np.max(score))
        scores.append(np.max(score))

        print('\rEpisode {}\tAverage Score: {:.4f}'
              .format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'
                  .format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!'
                  '\tAverage Score: {:.2f}'
                  .format(i_episode - print_every, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(),
                       'weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'weights/checkpoint_critic.pth')
            break

    return scores


def test_agent(env, agents, max_t=1000):
    """Test the trained agent
    Params
    ======
        max_t (int): maximum number of timesteps per episode
    """
    raise NotImplementedError
