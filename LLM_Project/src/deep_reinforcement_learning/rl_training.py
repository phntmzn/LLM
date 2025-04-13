import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from policy_network import PolicyNetwork


class ReinforceTrainer:
    """
    Trainer for policy gradient-based reinforcement learning (REINFORCE).
    """

    def __init__(self, env, policy_net, optimizer, gamma=0.99):
        """
        Args:
            env: The environment following OpenAI Gym API.
            policy_net (nn.Module): The policy network.
            optimizer (torch.optim.Optimizer): Optimizer for training the policy network.
            gamma (float): Discount factor for rewards.
        """
        self.env = env
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.gamma = gamma

    def compute_discounted_rewards(self, rewards):
        """
        Compute discounted rewards for an episode.
        
        Args:
            rewards (list): List of rewards obtained in an episode.
        
        Returns:
            list: Discounted rewards for the episode.
        """
        discounted_rewards = []
        cumulative_reward = 0.0
        for r in reversed(rewards):
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards

    def train_episode(self):
        """
        Train the policy network for a single episode.
        
        Returns:
            float: Total reward for the episode.
        """
        state = self.env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = self.policy_net(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            log_probs.append(action_distribution.log_prob(action))

            next_state, reward, done, _ = self.env.step(action.item())
            rewards.append(reward)
            total_reward += reward

            if done:
                break

            state = next_state

        # Compute discounted rewards
        discounted_rewards = self.compute_discounted_rewards(rewards)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards for numerical stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # Compute policy loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return total_reward

    def train(self, num_episodes, render=False):
        """
        Train the agent for a given number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to train the agent.
            render (bool): Whether to render the environment during training.
        """
        rewards = []

        for episode in range(num_episodes):
            if render:
                self.env.render()

            episode_reward = self.train_episode()
            rewards.append(episode_reward)

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward:.2f}")

        return rewards


if __name__ == "__main__":
    import gym

    # Environment setup
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]  # State space dimensions
    output_dim = env.action_space.n            # Action space dimensions

    # Policy network setup
    hidden_dim = 128
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)

    # Optimizer setup
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # Trainer setup
    trainer = ReinforceTrainer(env, policy_net, optimizer, gamma=0.99)

    # Train the policy network
    num_episodes = 1000
    rewards = trainer.train(num_episodes)

    # Close the environment
    env.close()

    # Save the trained policy
    torch.save(policy_net.state_dict(), "policy_network.pth")
    print("Trained policy network saved as policy_network.pth")
