import argparse
import torch
import torch.multiprocessing as mp
from torch import nn
from typing import List

from gpipe import Pipe
from mlora.utils import setup_seed
from utils.generate_data import generate_test_data,Batch
from model.gpt3 import build_gpt3_sequential, split_module

def pipe_process(rank: int, world_size: int, model: List[nn.Sequential], batches: List[Batch] = None):
    setup_seed(42)
    model = Pipe(rank, world_size, model, batches)
    model.run()
    return model.batches_

def test_by_pipe(args):
    world_size = args.world_size

    # Parameters from command line arguments
    vocab_size = args.vocab_size
    seq_length = args.seq_length
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    num_batches = args.num_batches
    num_layers = args.num_layers
    num_heads = args.num_heads
    dropout = args.dropout

    # Generate data
    batches = generate_test_data(vocab_size, seq_length, batch_size, num_batches, device='cuda:0')
    
    # Build GPT-3 as nn.Sequential
    gpt3_sequential = build_gpt3_sequential(vocab_size, hidden_size, num_layers, num_heads, seq_length, dropout)

    # Devices and balance
    devices = [torch.device(f'cuda:{i}') for i in [0,1,2]]  # Devices based on user input
    balance = list(map(int, args.balance.split(',')))  # Balance as a comma-separated string

    # Split model
    partitions, balance, devices = split_module(gpt3_sequential, balance, devices)

    # Create processes to run Pipe
    ctx = mp.get_context("spawn")
    args = ((rank, world_size, partitions[rank], batches if (rank == 0 or rank == world_size - 1) else None) for rank in range(world_size))
    
    with ctx.Pool(world_size) as pool:
        res = pool.starmap(pipe_process, args)

def parse_args():
    parser = argparse.ArgumentParser(description="Run GPT-3 with Pipe on multiple GPUs")
    
    # General parameters with default values based on original code
    parser.add_argument('--world_size', type=int, default=3, help="Number of processes (default: 3)")
    parser.add_argument('--vocab_size', type=int, default=50257, help="Vocabulary size (default: 50257)")
    parser.add_argument('--seq_length', type=int, default=1024, help="Sequence length (default: 1024)")
    parser.add_argument('--hidden_size', type=int, default=2304, help="Hidden size of GPT-3 (default: 2304)")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size (default: 2)")
    parser.add_argument('--num_batches', type=int, default=3, help="Number of batches (default: 3)")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in GPT-3 (default: 2)")
    parser.add_argument('--num_heads', type=int, default=24, help="Number of attention heads (default: 24)")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument('--num_devices', type=int, default=3, help="Number of GPUs (default: 3)")
    parser.add_argument('--balance', type=str, default='2,2,1', help="Model balance across devices (default: '3,1,1')")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    test_by_pipe(args)