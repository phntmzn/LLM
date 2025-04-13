import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    """

    def __init__(self, emb_dim, num_heads, context_length, dropout=0.1):
        """
        Args:
            emb_dim (int): Dimension of embeddings.
            num_heads (int): Number of attention heads.
            context_length (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super(MultiHeadAttention, self).__init__()
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.W_query = nn.Linear(emb_dim, emb_dim)
        self.W_key = nn.Linear(emb_dim, emb_dim)
        self.W_value = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        B, T, C = x.size()
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_scores = attn_scores.masked_fill(self.mask[:T, :T], float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ V
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) module.
    """

    def __init__(self, emb_dim, ffn_dim, dropout=0.1):
        """
        Args:
            emb_dim (int): Dimension of embeddings.
            ffn_dim (int): Dimension of the hidden layer in the FFN.
            dropout (float): Dropout rate.
        """
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer block.
    """

    def __init__(self, emb_dim, num_heads, ffn_dim, context_length, dropout=0.1):
        """
        Args:
            emb_dim (int): Dimension of embeddings.
            num_heads (int): Number of attention heads.
            ffn_dim (int): Dimension of the hidden layer in the FFN.
            context_length (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads, context_length, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = FeedForward(emb_dim, ffn_dim, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class GPTModel(nn.Module):
    """
    GPT model for language modeling tasks.
    """

    def __init__(self, vocab_size, emb_dim, num_heads, num_layers, ffn_dim, context_length, dropout=0.1):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            emb_dim (int): Dimension of embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer layers.
            ffn_dim (int): Dimension of the hidden layer in the FFN.
            context_length (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super(GPTModel, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, ffn_dim, context_length, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.size()
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(torch.arange(T, device=x.device))
        x = self.dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    # Example usage of GPTModel
    vocab_size = 50257
    emb_dim = 768
    num_heads = 12
    num_layers = 12
    ffn_dim = 3072
    context_length = 1024
    dropout = 0.1

    # Instantiate the GPT model
    model = GPTModel(vocab_size, emb_dim, num_heads, num_layers, ffn_dim, context_length, dropout)

    # Create a dummy input tensor (batch_size=2, seq_len=10)
    dummy_input = torch.randint(0, vocab_size, (2, 10))

    # Forward pass through the model
    logits = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Logits shape:", logits.shape)  # (batch_size, seq_len, vocab_size)
