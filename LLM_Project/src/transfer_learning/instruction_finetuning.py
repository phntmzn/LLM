import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from gpt_model import GPTModel
from utils import load_model_checkpoint, save_model_checkpoint
from metrics import calculate_loss, calculate_perplexity, calculate_accuracy
from data_utils import create_instruction_dataset


def format_instruction_data(entry):
    """
    Format an instruction entry into a complete prompt-response string.

    Args:
        entry (dict): A dictionary containing "instruction", "input", and "output".

    Returns:
        str: Formatted string for fine-tuning.
    """
    prompt = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{entry['instruction']}\n\n"
    )
    if entry["input"]:
        prompt += f"### Input:\n{entry['input']}\n\n"
    prompt += "### Response:"
    return prompt + f"\n{entry['output']}"


def instruction_finetune_one_epoch(model, data_loader, optimizer, device, ignore_index=-100):
    """
    Fine-tune the model for one epoch on instruction-based data.

    Args:
        model (nn.Module): The GPT model.
        data_loader (DataLoader): DataLoader for the instruction dataset.
        optimizer (Optimizer): Optimizer for training.
        device (str): Device to run the model on.
        ignore_index (int): Index to ignore in the target sequence.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate_instruction_model(model, data_loader, device, ignore_index=-100):
    """
    Evaluate the fine-tuned model on instruction-based data.

    Args:
        model (nn.Module): The GPT model.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        device (str): Device to run the model on.
        ignore_index (int): Index to ignore in the target sequence.

    Returns:
        dict: Evaluation metrics including loss, perplexity, and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = calculate_loss(logits, targets, ignore_index)
            accuracy = calculate_accuracy(logits, targets, ignore_index)

            total_loss += loss
            total_accuracy += accuracy

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"loss": avg_loss, "accuracy": avg_accuracy, "perplexity": perplexity}


def main():
    # Configuration
    config = {
        "vocab_size": 50257,
        "emb_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "ffn_dim": 3072,
        "context_length": 1024,
        "dropout": 0.1,
        "batch_size": 8,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "pretrained_checkpoint": "checkpoints/gpt2_pretrained.pth",
        "fine_tuned_checkpoint": "checkpoints/instruction_finetuned.pth"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load pre-trained model
    model = GPTModel(
        config["vocab_size"],
        config["emb_dim"],
        config["num_heads"],
        config["num_layers"],
        config["ffn_dim"],
        config["context_length"],
        config["dropout"]
    )
    model = load_model_checkpoint(model, config["pretrained_checkpoint"], device)
    model.to(device)

    # Load instruction-based dataset
    train_loader = create_instruction_dataset("data/processed/instruction_train.json", tokenizer, config["batch_size"])
    val_loader = create_instruction_dataset("data/processed/instruction_val.json", tokenizer, config["batch_size"])

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Fine-tuning loop
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}:")

        # Fine-tuning
        train_loss = instruction_finetune_one_epoch(model, train_loader, optimizer, device)
        print(f"  Training Loss: {train_loss:.4f}")

        # Evaluation
        metrics = evaluate_instruction_model(model, val_loader, device)
        print(f"  Validation Loss: {metrics['loss']:.4f}")
        print(f"  Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Validation Perplexity: {metrics['perplexity']:.4f}")

    # Save fine-tuned model
    save_model_checkpoint(model, config["fine_tuned_checkpoint"])
    print("Instruction fine-tuning complete. Model saved.")


if __name__ == "__main__":
    main()
