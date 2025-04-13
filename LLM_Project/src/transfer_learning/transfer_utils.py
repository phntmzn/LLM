import torch
import torch.nn as nn
import json
from transformers import GPT2Tokenizer


def freeze_model_layers(model, freeze_embeddings=True, num_unfrozen_layers=None):
    """
    Freeze layers in the model for transfer learning.

    Args:
        model (nn.Module): The GPT model instance.
        freeze_embeddings (bool): Whether to freeze the token and position embeddings.
        num_unfrozen_layers (int): Number of layers to keep unfrozen.
            If None, all layers are frozen except the final layer.
    """
    if freeze_embeddings:
        model.token_emb.weight.requires_grad = False
        model.pos_emb.weight.requires_grad = False

    if num_unfrozen_layers is not None:
        for layer in model.blocks[:-num_unfrozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False


class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for efficient parameter tuning.

    This module implements a lightweight adaptation layer for injecting task-specific
    knowledge into pre-trained models without modifying the original weights.
    """

    def __init__(self, base_layer, rank=4):
        """
        Args:
            base_layer (nn.Module): The layer to which LoRA is applied (e.g., a Linear layer).
            rank (int): The rank of the LoRA adaptation.
        """
        super(LoRA, self).__init__()
        self.base_layer = base_layer
        self.rank = rank

        # Add LoRA-specific parameters
        self.lora_A = nn.Parameter(torch.zeros(base_layer.weight.size(0), rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, base_layer.weight.size(1)))

        # Initialize LoRA parameters
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = x @ self.lora_B.T @ self.lora_A.T
        return base_output + lora_output


def apply_lora_to_gpt(model, rank=4):
    """
    Apply LoRA to all linear layers in the GPT model.

    Args:
        model (nn.Module): The GPT model instance.
        rank (int): The rank of the LoRA adaptation.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                LoRA(module, rank=rank)
            )


def load_instruction_dataset(json_file, tokenizer, max_length, stride, pad_token_id=50256):
    """
    Load instruction-following dataset from a JSON file.

    Args:
        json_file (str): Path to the JSON file containing the dataset.
        tokenizer: Pre-trained tokenizer instance.
        max_length (int): Maximum sequence length for tokenization.
        stride (int): Stride length for overlapping sequences.
        pad_token_id (int): Token ID used for padding.

    Returns:
        list: A list of tokenized input-target pairs.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    tokenized_data = []
    for entry in data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output_text = entry.get("output", "")

        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
        full_text = prompt + f"\n{output_text}"

        tokens = tokenizer.encode(full_text)
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i:i + max_length]
            target_chunk = tokens[i + 1:i + max_length + 1]
            tokenized_data.append((input_chunk, target_chunk))

    return tokenized_data


class InstructionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for instruction-based fine-tuning.

    Args:
        data (list): A list of tokenized input-target pairs.
        pad_token_id (int): Token ID used for padding.
        context_length (int): Maximum context length for sequences.
    """

    def __init__(self, data, pad_token_id=50256, context_length=1024):
        super(InstructionDataset, self).__init__()
        self.data = data
        self.pad_token_id = pad_token_id
        self.context_length = context_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, target_ids = self.data[idx]

        # Ensure input and target lengths are equal and fit within context length
        input_ids = input_ids[:self.context_length]
        target_ids = target_ids[:self.context_length]

        # Pad sequences to context length
        input_ids += [self.pad_token_id] * (self.context_length - len(input_ids))
        target_ids += [self.pad_token_id] * (self.context_length - len(target_ids))

        return torch.tensor(input_ids), torch.tensor(target_ids)


def create_instruction_dataloader(json_file, tokenizer, batch_size, max_length, stride, pad_token_id=50256):
    """
    Create a DataLoader for instruction-based datasets.

    Args:
        json_file (str): Path to the JSON file containing the dataset.
        tokenizer: Pre-trained tokenizer instance.
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum sequence length for tokenization.
        stride (int): Stride length for overlapping sequences.
        pad_token_id (int): Token ID used for padding.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    tokenized_data = load_instruction_dataset(json_file, tokenizer, max_length, stride, pad_token_id)
    dataset = InstructionDataset(tokenized_data, pad_token_id, max_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # Example usage
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    json_file = "data/processed/instruction_train.json"
    dataloader = create_instruction_dataloader(json_file, tokenizer, batch_size=8, max_length=512, stride=256)

    for inputs, targets in dataloader:
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        break
