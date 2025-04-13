import torch
import os
from transformers import GPT2Tokenizer

def load_weights_into_gpt(model, weights):
    """
    Load pretrained weights into a GPT model.

    Args:
        model (nn.Module): The GPT model instance.
        weights (dict): Dictionary of pretrained weights.

    Returns:
        nn.Module: The model with loaded weights.
    """
    model.load_state_dict(weights)
    return model


def save_model_checkpoint(model, path):
    """
    Save the model checkpoint to a specified path.

    Args:
        model (nn.Module): The model instance.
        path (str): Path to save the model checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model checkpoint saved at: {path}")


def load_model_checkpoint(model, path, device="cpu"):
    """
    Load the model checkpoint from a specified path.

    Args:
        model (nn.Module): The model instance.
        path (str): Path to the model checkpoint.
        device (str): Device to map the model to.

    Returns:
        nn.Module: The model with loaded weights.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    print(f"Model checkpoint loaded from: {path}")
    return model


def text_to_token_ids(text, tokenizer):
    """
    Convert a string of text into token IDs.

    Args:
        text (str): Input text.
        tokenizer: Tokenizer instance (e.g., GPT2Tokenizer).

    Returns:
        torch.Tensor: Token IDs as a tensor.
    """
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back into a human-readable string.

    Args:
        token_ids (torch.Tensor): Token IDs.
        tokenizer: Tokenizer instance (e.g., GPT2Tokenizer).

    Returns:
        str: Decoded text.
    """
    return tokenizer.decode(token_ids.tolist())


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    eos_id=None,
    temperature=1.0,
    top_k=50
):
    """
    Generate text using the model.

    Args:
        model (nn.Module): GPT model for generation.
        idx (torch.Tensor): Current input indices (batch_size, seq_len).
        max_new_tokens (int): Maximum number of tokens to generate.
        context_size (int): Maximum size of the context window.
        eos_id (int, optional): ID of the end-of-sequence token.
        temperature (float): Sampling temperature.
        top_k (int): Number of top tokens to sample from.

    Returns:
        torch.Tensor: Generated token indices.
    """
    model.eval()
    generated = idx.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context to the supported size
            context = generated[:, -context_size:]

            # Get logits from the model
            logits = model(context)[:, -1, :]

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k)
                logits[logits < top_k_values[:, -1, None]] = -float("Inf")

            probabilities = torch.softmax(logits, dim=-1)

            # Sample from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Append the sampled token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Stop generation if end-of-sequence token is found
            if eos_id is not None and (next_token == eos_id).any():
                break

    return generated


if __name__ == "__main__":
    # Example usage
    from gpt_model import GPTModel
    from transformers import GPT2Tokenizer

    # Set up tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Example GPT model config
    vocab_size = 50257
    emb_dim = 768
    num_heads = 12
    num_layers = 12
    ffn_dim = 3072
    context_length = 1024
    dropout = 0.1

    # Initialize the model
    model = GPTModel(
        vocab_size, emb_dim, num_heads, num_layers, ffn_dim, context_length, dropout
    )

    # Load a checkpoint if available
    checkpoint_path = "checkpoints/gpt_model.pth"
    if os.path.exists(checkpoint_path):
        model = load_model_checkpoint(model, checkpoint_path)

    # Example input
    prompt = "Once upon a time"
    token_ids = text_to_token_ids(prompt, tokenizer).unsqueeze(0)

    # Generate text
    generated_ids = generate(
        model=model,
        idx=token_ids,
        max_new_tokens=50,
        context_size=context_length,
        eos_id=tokenizer.eos_token_id
    )
    generated_text = token_ids_to_text(generated_ids[0], tokenizer)
    print(f"Generated text:\n{generated_text}")
