import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    A simple policy network for reinforcement learning.
    It maps state observations to a probability distribution over actions.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): Dimension of input state space
            hidden_dim (int): Number of hidden units in the hidden layer
            output_dim (int): Dimension of action space (number of possible actions)
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): The input state tensor (batch_size x input_dim)
        
        Returns:
            torch.Tensor: Action probabilities (batch_size x output_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        action_probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        return action_probs


if __name__ == "__main__":
    # Example usage of the PolicyNetwork
    input_dim = 4   # Example: 4 state features
    hidden_dim = 128  # Example: Hidden layer with 128 units
    output_dim = 2  # Example: 2 possible actions (e.g., left or right)

    # Instantiate the policy network
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)

    # Create a dummy state tensor (batch_size=1, input_dim=4)
    dummy_state = torch.rand(1, input_dim)

    # Forward pass through the network
    action_probs = policy_net(dummy_state)

    print("Input state:", dummy_state)
    print("Action probabilities:", action_probs)
