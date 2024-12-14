import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Tuple

# Transformer Block remains the same
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: [seq_len, batch_size, hidden_size]
        attn_output, _ = self.attention(x, x, x)   # Multi-head Attention
        x = self.ln_1(x + attn_output)
        ff_output = self.mlp(x)
        x = self.ln_2(x + ff_output)
        return x

class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, seq_length):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(seq_length, hidden_size)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_length]
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # [1, seq_len]
        token_emb = self.token_embedding(input_ids)           # [batch_size, seq_len, hidden_size]
        position_emb = self.position_embedding(positions)     # [1, seq_len, hidden_size]
        x = token_emb + position_emb
        # TransformerBlock期望输入维度为 [seq_len, batch_size, hidden_size]
        x = x.transpose(0, 1)  # 变换为 [seq_len, batch_size, hidden_size]
        return x


# GPT3Model now returns an nn.Sequential
def build_gpt3_sequential(vocab_size, hidden_size, num_layers, num_heads, seq_length, dropout):
    layers = []
    
    # Embedding layer
    layers.append(EmbeddingModule(vocab_size, hidden_size, seq_length))
    
    # Transformer Blocks
    for _ in range(num_layers):
        layers.append(TransformerBlock(hidden_size, num_heads, dropout))
    
    # Final Layer Norm
    layers.append(nn.LayerNorm(hidden_size))
    
    # Output Head
    layers.append(nn.Linear(hidden_size, vocab_size, bias=False))
    
    # Convert to nn.Sequential
    return nn.Sequential(*layers)

def split_module(module: nn.Sequential,
                 balance: Iterable[int],
                 devices: List[torch.device],
                 ) -> Tuple[List[nn.Sequential], List[int], List[torch.device]]:
    """Splits a module into multiple partitions."""
    balance = list(balance)

    if len(module) != sum(balance):
        raise ValueError('module and sum of balance have different length '
                         f'(module: {len(module)}, sum of balance: {sum(balance)})')

    if any(x <= 0 for x in balance):
        raise ValueError(f'all balance numbers must be positive integer (balance: {balance})')

    if len(balance) > len(devices):
        raise IndexError('too few devices to hold given partitions '
                         f'(devices: {len(devices)}, partitions: {len(balance)})')

    # Splitting module into partitions
    partitions = []
    current_idx = 0
    for num_layers in balance:
        sub_module = nn.Sequential(*list(module.children())[current_idx:current_idx + num_layers])
        sub_module.to(devices[len(partitions)])  # Move to corresponding device
        partitions.append(sub_module)
        current_idx += num_layers

    return partitions, balance, devices


# 使用示例
if __name__ == "__main__":
    # Model parameters
    vocab_size = 50257
    hidden_size = 2304
    num_layers = 2
    num_heads = 24
    seq_length = 1024
    dropout = 0.1

    # Build GPT-3 as nn.Sequential
    gpt3_sequential = build_gpt3_sequential(vocab_size, hidden_size, num_layers, num_heads, seq_length, dropout)
    # print(gpt3_sequential)

    # Devices and balance
    devices = [torch.device(f'cuda:{i}') for i in range(4)]  # Example with 4 GPUs
    balance = [2, 1, 1, 1]  # Example partition sizes summing to 22 layers

    # Split model
    partitions, balance, devices = split_module(gpt3_sequential, balance, devices)
    # Print partitions and their assigned devices
    for i, partition in enumerate(partitions):
        print(f"Partition {i} assigned to {devices[i]}:")
        print(partition)