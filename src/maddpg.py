import torch

import random
from collections import namedtuple, deque
import numpy as np

from .ddpg_agent import Agent

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
LEARN_NUM = 3
GAMMA = 0.99

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG():
    """Multi Agent DDPG"""
    def __init__(self, state_size, action_size, nb_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)

        self.agents = [Agent(state_size, action_size, nb_agents, random_seed)
                       for _ in range(nb_agents)]

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def act(self, state, i_episode, add_noise=True):
        """Return an action for each agent"""
        actions = []
        for state, agent in zip(state, self.agents):
            action = agent.act(state, i_episode, add_noise)
            actions.append(np.reshape(action, -1))
        actions = np.stack(actions)
        return actions

    def step(self, i_episode, state, action, reward, next_state, done):
        # flatten state and next_state to get
        # a single array for all the observations
        full_state = np.reshape(state, -1)
        next_full_state = np.reshape(next_state, -1)

        self.memory.add(state, full_state,
                        action, reward,
                        next_state, next_full_state, done)

        if len(self.memory) > BATCH_SIZE and i_episode > 500:
            for _ in range(LEARN_NUM):
                for idx, agent in enumerate(self.agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, idx, GAMMA)

    def learn(self, experiences, agent, agent_id, GAMMA):
        # states x^j (o^j_1, o^j_2) and next_states x'^j
        # actions (a^j) and rewards (r^j)
        (states, full_states,
            actions, rewards,
            next_states, next_full_states,
            dones) = experiences

        # compute the next actions (a'^j_i = actor_target(o^j_i))
        actions_next = torch.zeros(actions.shape,
                                     dtype=torch.float, device=device)
        for idx, agent_i in enumerate(self.agents):
            actions_next[:, idx, :] = agent_i.actor_target(states[:, idx])
        actions_next = actions_next.view(BATCH_SIZE, -1)

        # state, reward, done of the corresponding agent
        # and reshape the reward and done objects to allow the computation
        agent_state = states[:, agent_id, :]
        agent_reward = rewards[:, agent_id].view(-1, 1)
        agent_done = dones[:, agent_id].view(-1, 1)

        # compute the actions vector to pass to the local critic in order
        # to update the actor. With N = 2:
        # (a^j_1, ..., a_i, ..., a^j_N) with a_i = actor_local(o^j_i)
        actions_pred = actions.clone()
        actions_pred[:, agent_id, :] = agent.actor_local(agent_state)
        actions_pred = actions_pred.view(BATCH_SIZE, -1)

        actions = actions.view(BATCH_SIZE, -1)

        agent_experience = (full_states,
                            actions, actions_pred, actions_next,
                            agent_reward, agent_done,
                            next_states, next_full_states)

        agent.learn(agent_experience, GAMMA)

    def save(self):
        for idx, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),
                       f'weights/checkpoint_agent{idx}_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       f'weights/checkpoint_critic{idx}_critic.pth')


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=['state', 'full_state',
                                                  'action', 'reward',
                                                  'next_state',
                                                  'next_full_state',
                                                  'done'])
        self.seed = random.seed(seed)

    def add(self, state, full_state, action, reward,
            next_state, next_full_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, full_state,
                            action, reward,
                            next_state, next_full_state,
                            done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.array([e.state for e in experiences
                      if e is not None])).float().to(device)
        full_states = torch.from_numpy(
            np.array([e.full_state for e in experiences
                      if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.array([e.action for e in experiences
                      if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.array([e.reward for e in experiences
                      if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.array([e.next_state for e in experiences
                      if e is not None])).float().to(device)
        next_full_states = torch.from_numpy(
            np.array([e.next_full_state for e in experiences
                      if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.array([e.done for e in experiences
                      if e is not None]).astype(np.uint8)).float().to(device)

        return (states, full_states,
                actions, rewards,
                next_states, next_full_states,
                dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
