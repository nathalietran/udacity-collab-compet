import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import click
import torch
from src.monitor import run_agent, test_agent
from src.maddpg import MADDPG

# path to Tennis UnityEnvironment
# i.e. 'Tennis.app'
TENNIS_APP = ''


def plot_scores(scores, rolling_mean=None,
                rolling_window=100, save_fig=True):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('scores')
    if rolling_mean is None:
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);

    if save_fig:
        plt.savefig(f'figures/scores.png',
                    bbox_inches='tight', pad_inches=0)

    return rolling_mean


@click.command()
@click.option('--test', help='test or train agent', is_flag=True)
def main(test):
    # init the environment
    env = UnityEnvironment(file_name=TENNIS_APP)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # dimenison of the state space
    state_size = env_info.vector_observations.shape[1]
    # number of agents
    n_agents = len(env_info.agents)

    # create an MADDPG agent
    ma_agent = MADDPG(state_size, action_size, n_agents, 0)

    if not test:
        # train the agent
        scores, rolling_mean = run_agent(env, ma_agent, n_episodes=300)
        _ = plot_scores(scores, rolling_mean)
    else:
        # test the trained agent
        # load the weights from file
        for idx, agent in enumerate(ma_agent.agents):
            agent.actor_local.load_state_dict(
                torch.load(f'weights/checkpoint_actor_{idx}.pth'))
            agent.critic_local.load_state_dict(
                torch.load(f'weights/checkpoint_critic_{idx}.pth'))
            test_agent(env, ma_agent)

    env.close()


if __name__ == '__main__':
    main()
