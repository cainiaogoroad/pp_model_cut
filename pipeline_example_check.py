from gpipe import Pipe, TestModel
from mlora.utils import setup_seed

from torch.nn import Sequential
from typing import List

import torch.multiprocessing as mp
import torch

from utils.generate_data import *
from model.gpt3 import *


def pipe_process(rank: int, world_size: int, model: List[nn.Sequential],batches: List[Batch] = None):
    setup_seed(42)
    model = Pipe(rank, world_size, model,batches, device=torch.device("cuda:0"))
    model.run()
    return model.test_grads_, model.model_.weight_, model.datas_


def test_by_pipe():
    world_size = 2

    vocab_size = 50257  # 词汇表大小
    seq_length = 1024   # 序列长度
    hidden_size = 2304
    batch_size = 8      # 每批次样本数量
    num_batches = 5     # 总批次数量
    num_layers = 2
    num_heads = 24
    seq_length = 1024
    dropout = 0.1
    # 生成数据
    batches = generate_test_data(vocab_size, seq_length, batch_size, num_batches,device='cuda:0')
    # Build GPT-3 as nn.Sequential
    gpt3_sequential = build_gpt3_sequential(vocab_size, hidden_size, num_layers, num_heads, seq_length, dropout)
    # print(gpt3_sequential)

    # Devices and balance
    devices = [torch.device(f'cuda:{i}') for i in range(2)]  # Example with 4 GPUs
    balance = [3, 2]  # Example partition sizes summing to 22 layers

    # Split model
    partitions, balance, devices = split_module(gpt3_sequential, balance, devices)

    # create 2 processes to run Pipe
    ctx = mp.get_context("spawn")
    args = ((rank, world_size,partitions[rank],batches if rank == 0 else None) for rank in range(world_size))
    with ctx.Pool(world_size) as pool:
        res = pool.starmap(
            pipe_process,
            args)

    res_1, res_2 = res

    grads_1, weight, datas = res_1
    grads_2, _, _ = res_2

    return grads_1, grads_2, weight, datas


def test_by_sequential(weight: torch.Tensor, datas: List[torch.Tensor]):
    model_1 = TestModel(torch.device("cuda:0"))
    model_2 = TestModel(torch.device("cuda:0"))

    # set the weight of the models to the provided weight
    model_1.weight_ = weight.clone().detach()
    model_2.weight_ = weight.clone().detach()

    # enable gradient calculation for the weights
    model_1.weight_.requires_grad = True
    model_2.weight_.requires_grad = True

    grads_1 = []
    grads_2 = []

    seq_model = Sequential(model_1, model_2).cuda()

    # calculate gradients for each data in the datas list
    for data in datas:
        res = seq_model(data)
        res.sum().backward()
        grads_1.append(model_1.weight_.grad.sum())
        grads_2.append(model_2.weight_.grad.sum())

    return grads_1, grads_2


def test_pipe():
    # calculate gradients using Pipe
    grads_1, grads_2, weight, datas = test_by_pipe()

    # calculate expected gradients using Sequential with same model weight and data
    expected_grads_1, expected_grads_2 = test_by_sequential(weight, datas)

    # check if the gradients are the same
    for i in range(len(grads_1)):
        assert torch.allclose(expected_grads_1[i], grads_1[i])
        assert torch.allclose(expected_grads_2[i], grads_2[i])


if __name__ == '__main__':
     test_pipe()
