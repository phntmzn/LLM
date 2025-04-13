import torch
import torch.nn as nn


def cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Computes the cross-entropy loss between logits and target indices.

    Args:
        logits (torch.Tensor): Logits from the model of shape (batch_size, seq_len, vocab_size).
        targets (torch.Tensor): Target indices of shape (batch_size, seq_len).
        ignore_index (int): Index to ignore in the targets, typically for padding tokens.

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    # Reshape logits and targets to match dimensions for cross-entropy calculation
    logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
    targets = targets.view(-1)  # (batch_size * seq_len)

    # Compute the cross-entropy loss
    loss = nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)
    return loss


class LabelSmoothingLoss(nn.Module):
    """
    Implements label smoothing for cross-entropy loss to improve generalization.
    """

    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing (float): Smoothing factor (0.0 means no smoothing).
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        """
        Forward pass for label smoothing loss.

        Args:
            logits (torch.Tensor): Logits from the model of shape (batch_size, seq_len, vocab_size).
            targets (torch.Tensor): Target indices of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Smoothed cross-entropy loss.
        """
        vocab_size = logits.size(-1)

        # One-hot encode the targets
        target_dist = torch.zeros_like(logits).scatter_(-1, targets.unsqueeze(-1), 1.0)

        # Apply label smoothing
        target_dist = (1 - self.smoothing) * target_dist + self.smoothing / vocab_size

        # Compute log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Compute the loss
        loss = -(target_dist * log_probs).sum(dim=-1).mean()
        return loss


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    # Dummy logits and targets
    logits = torch.rand(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Cross-entropy loss
    ce_loss = cross_entropy_loss(logits, targets)
    print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")

    # Label smoothing loss
    label_smoothing_loss_fn = LabelSmoothingLoss(smoothing=0.1)
    smoothed_loss = label_smoothing_loss_fn(logits, targets)
    print(f"Label Smoothing Loss: {smoothed_loss.item():.4f}")
