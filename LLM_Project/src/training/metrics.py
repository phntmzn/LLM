import torch
import torch.nn.functional as F


def calculate_perplexity(logits, targets, ignore_index=-100):
    """
    Calculate perplexity, a measure of how well a model predicts a sequence.
    
    Args:
        logits (torch.Tensor): Logits from the model of shape (batch_size, seq_len, vocab_size).
        targets (torch.Tensor): Target indices of shape (batch_size, seq_len).
        ignore_index (int): Index to ignore in the targets, typically for padding tokens.
    
    Returns:
        float: Perplexity score.
    """
    logits = logits.view(-1, logits.size(-1))  # Flatten logits to (batch_size * seq_len, vocab_size)
    targets = targets.view(-1)  # Flatten targets to (batch_size * seq_len)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    
    # Convert loss to perplexity
    perplexity = torch.exp(loss).item()
    return perplexity


def calculate_accuracy(logits, targets, ignore_index=-100):
    """
    Calculate accuracy of predictions.
    
    Args:
        logits (torch.Tensor): Logits from the model of shape (batch_size, seq_len, vocab_size).
        targets (torch.Tensor): Target indices of shape (batch_size, seq_len).
        ignore_index (int): Index to ignore in the targets, typically for padding tokens.
    
    Returns:
        float: Accuracy score.
    """
    logits = logits.view(-1, logits.size(-1))  # Flatten logits
    targets = targets.view(-1)  # Flatten targets

    # Ignore padding tokens
    mask = targets != ignore_index
    logits = logits[mask]
    targets = targets[mask]

    # Get predictions
    predictions = torch.argmax(logits, dim=-1)

    # Calculate accuracy
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


def calculate_loss(logits, targets, ignore_index=-100):
    """
    Calculate cross-entropy loss between logits and targets.

    Args:
        logits (torch.Tensor): Logits from the model of shape (batch_size, seq_len, vocab_size).
        targets (torch.Tensor): Target indices of shape (batch_size, seq_len).
        ignore_index (int): Index to ignore in the targets, typically for padding tokens.

    Returns:
        float: Cross-entropy loss.
    """
    logits = logits.view(-1, logits.size(-1))  # Flatten logits
    targets = targets.view(-1)  # Flatten targets

    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    return loss.item()


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    # Dummy logits and targets
    logits = torch.rand(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Calculate metrics
    perplexity = calculate_perplexity(logits, targets)
    accuracy = calculate_accuracy(logits, targets)
    loss = calculate_loss(logits, targets)

    # Print results
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")
