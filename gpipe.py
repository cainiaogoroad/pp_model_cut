from pipeline.rpc_transport import RpcTransport
from pipeline.function import SendOperator, RecvOperator
from pipeline.messages import PipeMessageType
from pipeline.stream import CudaStream
from utils.setup_seed import setup_seed
from utils.generate_data import generate_test_data
from torch import nn
import os
import torch
import uuid
import logging
from utils.generate_data import *
from model.gpt3 import *
from enum import Enum, auto
from typing import Dict, List


logging.basicConfig(format="[%(asctime)s] [%(threadName)s] pp_model_cut: %(message)s",
                    level="DEBUG",
                    handlers=[logging.StreamHandler()],
                    force=True)

G_CALC_TOTAL_CNT = 10
G_TEST_TOTAL_CNT = 10
G_TEST_CNT = 0


class WorkerRole(Enum):
    HEAD = auto()
    MID = auto()
    TAIL = auto()


class Pipe():
    world_size_: int = -1
    rank_: int = -1
    device_: torch.device = None
    role_: WorkerRole = None
    model_: List[nn.Sequential] = None
    batches_: List[Batch] = None

    forward_stop_: bool = False
    input_stop_: bool = False
    backward_cache_: Dict[int, torch.Tensor] = {}

    forward_cnt_: int = 0
    stop_signal_: torch.Tensor = None

    def is_stop_signal(self, data: torch.tensor) -> bool:
        return data.dtype == torch.long and torch.numel(data) == 1

    def __init__(self, rank: int, world_size: int, model: List[nn.Sequential],batches: List[Batch] = None, device: torch.device = None) -> None:
        self.world_size_ = world_size
        self.rank_ = rank
        self.device_ = device if device else torch.device(f"cuda:{self.rank_}")

        if rank == 0:
            self.role_ = WorkerRole.HEAD
        elif rank == self.world_size_ - 1:
            self.role_ = WorkerRole.TAIL
        else:
            self.role_ = WorkerRole.MID
        self.transport_ = RpcTransport(
            self.rank_, self.world_size_, self.device_)

        self.model_ = model
        if self.role_== WorkerRole.HEAD:
            global G_TEST_TOTAL_CNT 
            G_TEST_TOTAL_CNT =len(batches)
            assert batches is not None
            self.data_ = batches
        
        if self.role_ == WorkerRole.TAIL:
            self.loss_fn_ = nn.CrossEntropyLoss()
            assert batches is not None
            self.data_ = batches

        self.forward_cnt_ = 0
        self.forward_stop_ = False
        self.input_stop_ = False

        self.default_stream_ = CudaStream(
            torch.cuda.default_stream(self.device_))

        self.test_grads_: List[torch.Tensor] = []

    def run(self):
        if self.role_ == WorkerRole.HEAD:
            self.forward_stop_ = True
        if self.role_ != WorkerRole.HEAD:
            self.input_stop_ = True

        while True:
            if self.role_ != WorkerRole.TAIL:
                self.process_backward()

            if not self.input_stop_:
                self.process_input()

            if not self.forward_stop_:
                self.process_forward()
    
            if len(self.backward_cache_) == 0 and self.forward_stop_ and self.input_stop_:
                # no froward and backward request
                break

        logging.info(f"Rank:{self.rank_} pipe done and to stop.")
        # clear the pipeline resource
        self.stop()

    def stop(self):
        transport = self.transport_
        if isinstance(transport, RpcTransport):
            transport.stop()
        logging.info("Transport stop.") 

    def process_backward(self):
        assert self.role_ != WorkerRole.TAIL

        message = self.transport_.recv_message(
            PipeMessageType.GRADIENTS, block=False)
        if message is None:
            return
        logging.info(
            f"device {self.device_} Recv the gradients - {str(message.msg_id_)[:8]} from {message.src_}")

        msg_id = message.msg_id_

        assert msg_id in self.backward_cache_
        phony: torch.Tensor = self.backward_cache_[msg_id]
        phony.grad_fn.grad_from_next_worker = message.tensor_data_
        phony.backward()
        # self.test_grads_.append(self.model_.weight_.grad.sum())
        del self.backward_cache_[msg_id]
        logging.info(f"backward_cache {msg_id} have been deleted.still left: {len(self.backward_cache_)}")

    def process_forward(self):
        assert self.role_ != WorkerRole.HEAD
        assert not self.forward_stop_

        # recv the tensors from prev-worker
        message = self.transport_.recv_message(
            PipeMessageType.ACTIVATIONS, block=False)
        if message is None:
            return
        logging.info(
            f"Recv the activations - {str(message.msg_id_)[:8]} from {message.src_}")
        logging.info(f"data is in: {message.tensor_data_.device}")

        # use RecvOperator get the real data
        #   the operator also auto send the backward grad to prev worker
        if self.is_stop_signal(message.tensor_data_):
            self.stop_signal_ = message.tensor_data_
            data = message.tensor_data_
            logging.info("Forward done be signaled.")
        else:
            logging.info(f"device {self.device_} foward count {self.forward_cnt_} Forward start.")
            data = RecvOperator.apply(
                torch.tensor(1.0, requires_grad=True), self.transport_, message)
            data.grad_fn.pre_stage_fn = self.default_stream_.poll
            self.forward_cnt_ += 1
            logging.debug("start")
            try:
                data = self.model_(data)
            except Exception as e:
                print(f"Error in model execution: {e}")
                # Optionally, log the stack trace
                import traceback
                traceback.print_exc()
            logging.info(f"device {self.device_} foward count {self.forward_cnt_} Forward done.")
        if self.stop_signal_ is not None:
            logging.debug(f"self.stop_signal_:{self.stop_signal_}")
            logging.debug(f"self.stop_signal_.item():{self.stop_signal_.item()}")
            logging.debug(f"self.forward_cnt_:{self.forward_cnt_}")
        if self.stop_signal_ is not None and self.stop_signal_.item() == self.forward_cnt_:
            self.forward_stop_ = True

        # mid worker need to send the result to next worker
        if self.role_ != WorkerRole.TAIL:
            self.default_stream_.poll()
            return self.send_next_worker(data)

        # tail worker need to calc the backward
        if not self.forward_stop_:
            data = data.permute(1, 0, 2).contiguous().view(-1, 50257)  # 先转置为 (batch_size, seq_length, vocab_size)，然后重塑
            target_batch = self.data_[self.forward_cnt_-1].to(data.device)
            labels = target_batch.labels
            labels = labels.contiguous().view(-1)
            loss = self.loss_fn_(data, labels)
            logging.info(f"Calc the grad {loss}.")
            loss.backward()
            # self.test_grads_.append(self.model_.weight_.grad.sum())


    def process_input(self):
        assert self.role_ == WorkerRole.HEAD
        assert not self.input_stop_

        global G_TEST_CNT


        if G_TEST_CNT >= G_TEST_TOTAL_CNT:
            self.input_stop_ = True
            data = torch.tensor(
                [self.forward_cnt_], dtype=torch.long, device="cpu", requires_grad=False)
            assert self.is_stop_signal(data)
            logging.info("Forward done be signaled.")
        else:
            logging.info(f"Train input data {G_TEST_CNT},all data is {G_TEST_TOTAL_CNT}.")
            self.forward_cnt_ += 1
            data = self.data_[G_TEST_CNT].input_ids
            data = self.model_(data)

        G_TEST_CNT += 1

        self.default_stream_.poll()
        self.send_next_worker(data)

    def send_next_worker(self, tensor_data: torch.Tensor) -> None:
        assert isinstance(tensor_data, torch.Tensor)

        msg_id = uuid.uuid4().int
        assert msg_id not in self.backward_cache_

        if self.is_stop_signal(tensor_data):
            msg_id = -1

        phony: torch.Tensor = SendOperator.apply(torch.tensor(
            1.0, requires_grad=True), tensor_data, self.transport_, msg_id, None)

        if self.is_stop_signal(tensor_data):
            return

        self.backward_cache_[msg_id] = phony


if __name__ == "__main__":
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
    batches = generate_test_data(vocab_size, seq_length, batch_size, num_batches)
    # Build GPT-3 as nn.Sequential
    gpt3_sequential = build_gpt3_sequential(vocab_size, hidden_size, num_layers, num_heads, seq_length, dropout)
    # print(gpt3_sequential)

    # Devices and balance
    devices = [torch.device(f'cuda:{i}') for i in range(4)]  # Example with 4 GPUs
    balance = [2, 1, 1, 1]  # Example partition sizes summing to 22 layers

    # Split model
    partitions, balance, devices = split_module(gpt3_sequential, balance, devices)



    
    rank = 0
    setup_seed(42)
    pipe = Pipe(rank, torch.cuda.device_count(),batches,partitions[rank])
    pipe.run()
