import torch
from typing import List
from model.gpt3 import *
# 定义Batch类
class Batch:
    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        初始化一个Batch对象
        Args:
            input_ids (torch.Tensor): 输入序列的ID张量, shape为 (batch_size, seq_length)
            labels (torch.Tensor): 目标序列的ID张量, shape为 (batch_size, seq_length)
        """
        self.input_ids = input_ids
        self.labels = labels

    def __repr__(self):
        return f"Batch(input_ids={self.input_ids.shape}, labels={self.labels.shape})"
        
    def to(self, device: str):
        """
        将Batch对象移动到指定设备
        Args:
            device (str): 要移动到的设备 ('cpu' 或 'cuda' 或 'cuda:<device_id>')
        Returns:
            Batch: 返回移动到目标设备的Batch对象
        """
        self.input_ids = self.input_ids.to(device)
        self.labels = self.labels.to(device)
        return self


# 生成测试数据的函数
def generate_test_data(vocab_size: int, seq_length: int, batch_size: int, num_batches: int, device: str = 'cpu') -> List[Batch]:
    """
    生成模拟的测试数据，并将数据放置在指定的设备上。

    Args:
        vocab_size (int): 词汇表大小
        seq_length (int): 输入序列的长度
        batch_size (int): 每个批次中的样本数
        num_batches (int): 批次数量
        device (str, optional): 设备类型（'cpu' 或 'cuda'）。默认为 'cpu'。

    Returns:
        List[Batch]: 包含多个 Batch 对象的列表，数据位于指定设备上
    """
    batches = []
    for _ in range(num_batches):
        # 随机生成input_ids和labels (简单模拟，将input_ids右移一位作为labels)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.roll(-1, dims=1)  # 假设目标是将输入右移一位
        batches.append(Batch(input_ids, labels))
    return batches


# 测试代码
if __name__ == "__main__":
    # 参数
    vocab_size = 50257  # 词汇表大小
    seq_length = 1024   # 序列长度
    batch_size = 8      # 每批次样本数量
    num_batches = 5     # 总批次数量

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





    # 生成数据
    batches = generate_test_data(vocab_size, seq_length, batch_size, num_batches)

    # 打印生成的批次
    for i, batch in enumerate(batches):
        for batch in batches:
            partition = partitions[i]
            batch = batch.to('cuda')

            output = partition(batch.input_ids)
            print(output.shape)
            import ipdb
            ipdb.set_trace()
        # print(f"Batch {i}:")
        # print("Input IDs:", batch.input_ids.shape)
        # print("Labels:", batch.labels.shape)
        # print()