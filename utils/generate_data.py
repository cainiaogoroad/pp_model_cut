import torch
from typing import List

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


# 生成测试数据的函数
def generate_test_data(vocab_size: int, seq_length: int, batch_size: int, num_batches: int) -> List[Batch]:
    """
    生成模拟的测试数据

    Args:
        vocab_size (int): 词汇表大小
        seq_length (int): 输入序列的长度
        batch_size (int): 每个批次中的样本数
        num_batches (int): 批次数量

    Returns:
        List[Batch]: 包含多个Batch对象的列表
    """
    batches = []
    for _ in range(num_batches):
        # 随机生成input_ids和labels (简单模拟，将input_ids右移一位作为labels)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
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

    # 生成数据
    batches = generate_test_data(vocab_size, seq_length, batch_size, num_batches)

    # 打印生成的批次
    for i, batch in enumerate(batches):
        print(f"Batch {i}:")
        print("Input IDs:", batch.input_ids.shape)
        print("Labels:", batch.labels.shape)
        print()