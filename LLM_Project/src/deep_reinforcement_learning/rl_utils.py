import numpy as np
import torch
import gym
from collections import deque


def compute_discounted_rewards(rewards, gamma=0.99):
    """
    Compute the discounted rewards for a single episode.
    
    Args:
        rewards (list): A list of rewards obtained in an episode.
        gamma (float): The discount factor.
    
    Returns:
        list: A list of discounted rewards.
    """
    discounted_rewards = []
    cumulative_reward = 0.0
    for r in reversed(rewards):
        cumulative_reward = r + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return discounted_rewards


def normalize_rewards(rewards):
    """
    Normalize the rewards to have zero mean and unit variance.
    
    Args:
        rewards (list or np.array): A list of rewards.
    
    Returns:
        np.array: Normalized rewards.
    """
    rewards = np.array(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards) + 1e-8  # Avoid division by zero
    return (rewards - mean) / std


class ReplayBuffer:
    """
    A simple Replay Buffer to store experiences for off-policy RL algorithms.
    """

    def __init__(self, capacity, state_dim, action_dim):
        """
        Args:
            capacity (int): Maximum number of experiences to store.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state (np.array): The state.
            action (np.array or int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state.
            done (bool): Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample.
        
        Returns:
            tuple: Batches of states, actions, rewards, next_states, and dones.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class LinearSchedule:
    """
    A linear schedule for exploration rates or other hyperparameters.
    """

    def __init__(self, start_value, end_value, max_steps):
        """
        Args:
            start_value (float): Initial value.
            end_value (float): Final value.
            max_steps (int): Number of steps over which to decay the value.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.max_steps = max_steps
        self.delta = (end_value - start_value) / max_steps

    def get_value(self, step):
        """
        Get the scheduled value at a specific step.
        
        Args:
            step (int): Current step.
        
        Returns:
            float: Scheduled value.
        """
        return max(self.end_value, self.start_value + step * self.delta)


def make_env(env_name, seed=None):
    """
    Create and configure an OpenAI Gym environment.
    
    Args:
        env_name (str): Name of the Gym environment.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        gym.Env: The configured environment.
    """
    env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return env


def epsilon_greedy_policy(q_values, epsilon):
    """
    Epsilon-greedy policy for action selection.
    
    Args:
        q_values (np.array): Q-values for all actions.
        epsilon (float): Probability of choosing a random action.
    
    Returns:
        int: Selected action.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)


class RunningMeanStd:
    """
    A class to compute a running mean and standard deviation, useful for reward normalization.
    """

    def __init__(self, epsilon=1e-8):
        self.mean = 0
        self.var = 0
        self.count = epsilon

    def update(self, x):
        """
        Update the running mean and variance with new data.
        
        Args:
            x (np.array): New data to include in the calculation.
        """
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        self.mean, self.var, self.count = self._update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def _update_mean_var_count(self, mean, var, count, batch_mean, batch_var, batch_count):
        """
        Internal helper to update mean, variance, and count.
        """
        delta = batch_mean - mean
        total_count = count + batch_count

        new_mean = mean + delta * batch_count / total_count
        m_a = var * count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * count * batch_count / total_count
        new_var = m_2 / total_count
        new_count = total_count

        return new_mean, new_var, new_count

    def normalize(self, x):
        """
        Normalize input data using the running mean and variance.
        
        Args:
            x (np.array): Input data.
        
        Returns:
            np.array: Normalized data.
        """
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
