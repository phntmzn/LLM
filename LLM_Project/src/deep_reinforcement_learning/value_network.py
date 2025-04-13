import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """
    A simple value network for reinforcement learning.
    It maps state observations to a scalar value (state value).
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Args:
            input_dim (int): Dimension of input state space
            hidden_dim (int): Number of hidden units in the hidden layer
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor (batch_size x input_dim)

        Returns:
            torch.Tensor: Predicted state value (batch_size x 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_out(x)
        return value


if __name__ == "__main__":
    # Example usage of the ValueNetwork
    input_dim = 4   # Example: State space has 4 features
    hidden_dim = 128  # Hidden layer with 128 units

    # Instantiate the value network
    value_net = ValueNetwork(input_dim, hidden_dim)

    # Create a dummy state tensor (batch_size=1, input_dim=4)
    dummy_state = torch.rand(1, input_dim)

    # Forward pass through the network
    value = value_net(dummy_state)

    print("Input state:", dummy_state)
    print("Predicted value:", value)
