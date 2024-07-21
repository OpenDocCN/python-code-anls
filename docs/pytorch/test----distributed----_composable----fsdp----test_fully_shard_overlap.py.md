# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_overlap.py`

```
# Owner(s): ["oncall: distributed"]
# 导入 functools 模块，用于高阶函数操作
import functools
# 导入 Callable 类型提示，用于声明可调用对象的类型
from typing import Callable

# 导入 PyTorch 库
import torch
# 导入 torch 分布式模块
import torch.distributed as dist
# 导入 torch.nn 神经网络模块
import torch.nn as nn

# 导入分布式 FSDP 相关模块
from torch.distributed._composable.fsdp import fully_shard
# 导入张量的隐式复制模块
from torch.distributed._tensor.experimental import implicit_replication
# 导入用于分布式测试的公共函数
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 导入用于 FSDP 测试的公共函数和类
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    patch_all_gather,
    patch_reduce_scatter,
)
# 导入用于通用测试的工具函数
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests

# 定义测试类 TestFullyShardOverlap，继承自 FSDPTest 类
class TestFullyShardOverlap(FSDPTest):
    
    # 定义属性 world_size，返回值为当前 GPU 数量和 2 的最小值
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    # 装饰器，如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_fully_shard_training_overlap(self):
        # 设置随机种子，确保可重复性
        torch.manual_seed(42)

        # 定义参数：维度、线性层数量、计算时间和通信时间（毫秒）
        dim, num_linears, compute_sleep_ms, comm_sleep_ms = (4, 3, 25, 10)
        
        # 创建包含多个具有计算延迟的线性层的模型
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(num_linears)]
        )
        
        # 对模型中的每个线性层进行分片
        for lin in model:
            fully_shard(lin, reshard_after_forward=True)
        
        # 对整个模型进行分片
        fully_shard(model, reshard_after_forward=True)

        # 保存原始的全收集和减少分散张量函数
        orig_all_gather_into_tensor = dist.all_gather_into_tensor
        orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
        
        # 创建 CUDA 流对象
        comm_stream = torch.cuda.Stream()

        def delay_collective():
            # 使用共享流使得全收集和减少分散互相阻塞，类似于 `ProcessGroupNCCL`
            comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm_stream):
                # 延迟通信操作，模拟通信时间
                torch.cuda._sleep(int(comm_sleep_ms * get_cycles_per_ms()))
            torch.cuda.current_stream().wait_stream(comm_stream)

        # 延迟执行的全收集函数
        def delayed_all_gather(*args, **kwargs):
            delay_collective()
            return orig_all_gather_into_tensor(*args, **kwargs)

        # 延迟执行的减少分散函数
        def delayed_reduce_scatter(*args, **kwargs):
            delay_collective()
            return orig_reduce_scatter_tensor(*args, **kwargs)

        # 创建输入张量并将其移动到 CUDA 设备上
        inp = torch.randn((2, dim), device="cuda")
        
        # 计算模型的损失，用于 CUDA 和分配器的预热
        loss = model(inp).sum()
        
        # 计算损失的反向传播
        loss.backward()

        # 定义前向传播函数
        def fwd():
            # 使用 patch_all_gather 替换 delayed_all_gather 函数
            with patch_all_gather(delayed_all_gather):
                model(inp)

        # 测量前向传播的执行时间
        fwd_time = self._time_fn(fwd)
        
        # 预期的前向传播时间，考虑通信时间、计算时间和缓冲时间
        buffer_ms = 2  # CPU 延迟和复制时间
        expected_fwd_time = comm_sleep_ms + num_linears * compute_sleep_ms + buffer_ms
        
        # 断言前向传播时间不超过预期时间
        self.assertLessEqual(fwd_time, expected_fwd_time)

        # 定义前向传播和反向传播函数
        def fwd_bwd():
            # 使用 patch_all_gather 和 patch_reduce_scatter 替换相应的函数
            with patch_all_gather(delayed_all_gather), patch_reduce_scatter(
                delayed_reduce_scatter
            ):
                loss = model(inp).sum()
                loss.backward()

        # 测量前向传播和反向传播的总执行时间
        fwd_bwd_time = self._time_fn(fwd_bwd)
        
        # 预期的反向传播时间，考虑通信时间、计算时间和缓冲时间的影响
        expected_bwd_time = (
            comm_sleep_ms * 2 + num_linears * 2 * compute_sleep_ms + buffer_ms * 2
        )
        
        # 断言前向传播和反向传播的总时间不超过预期时间之和
        self.assertLessEqual(fwd_bwd_time, expected_fwd_time + expected_bwd_time)

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_post_optim_event_overlap(self):
        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, compute_sleep_ms, comm_sleep_ms = (4, 25, 10)
        # 定义模型，包含一个高计算量的线性层，后接一个低计算量的线性层，
        # 只有低计算量的线性层使用了FSDP（Fully Sharded DataParallel）
        model = nn.Sequential(
            LinearWithSleep(dim, compute_sleep_ms), nn.Linear(dim, dim)
        ).cuda()
        # 对模型的第二个线性层进行完全分片（Fully Shard），前向传播后不重新分片
        fully_shard(model[1], reshard_after_forward=False)
        # 使用AdamW优化器优化模型的参数
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        orig_all_gather_into_tensor = dist.all_gather_into_tensor

        def delayed_all_gather(*args, **kwargs):
            # 模拟延迟的全局搜集操作，通过CUDA睡眠来模拟通信时间
            torch.cuda._sleep(int(comm_sleep_ms * get_cycles_per_ms()))
            return orig_all_gather_into_tensor(*args, **kwargs)

        inp = torch.randn((2, dim), device="cuda")

        def run_train_steps(num_iters: int, use_post_optim_event: bool):
            for _ in range(num_iters):
                optim.zero_grad()
                # 使用patch_all_gather装饰器来替换全局搜集操作函数
                with patch_all_gather(delayed_all_gather):
                    # 计算模型输出并求和，作为损失函数
                    loss = model(inp).sum()
                # 反向传播求梯度
                loss.backward()
                # 使用隐式复制策略
                with implicit_replication():
                    optim.step()
                if use_post_optim_event:
                    # 记录当前CUDA流的后优化事件
                    post_optim_event = torch.cuda.current_stream().record_event()
                    model[1].set_post_optim_event(post_optim_event)

        run_train_steps(1, False)  # 预热CUDA和内存分配器
        num_iters = 5
        # 计算基准时间，用于测试时间
        baseline_time = self._time_fn(
            functools.partial(run_train_steps, num_iters, False)
        )
        # 测试时间
        test_time = self._time_fn(functools.partial(run_train_steps, num_iters, True))

        buffer_ms = 4  # CPU延迟和拷贝时间
        # 基准时间：FSDP全局搜集由于FSDP模块等待当前流，因此高计算量线性层会暴露
        self.assertLessEqual(
            baseline_time,
            num_iters * (3 * compute_sleep_ms + comm_sleep_ms + buffer_ms),
        )
        # 测试时间：FSDP全局搜集与高计算量线性层重叠，因为FSDP模块只等待后优化事件
        expected_test_time = (
            num_iters * (3 * compute_sleep_ms + buffer_ms) + comm_sleep_ms
        )
        self.assertLessEqual(test_time, expected_test_time)
        self.assertGreater(baseline_time, expected_test_time)

    def _time_fn(self, fn: Callable):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        torch.cuda.synchronize()
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        return elapsed_time
class Matmul(torch.autograd.Function):
    # 定义一个自定义的 autograd 函数 Matmul，用于实现矩阵乘法并模拟计算时间
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, sleep_ms: int):
        # 在前向传播中，保存需要用到的张量和睡眠时间
        ctx.save_for_backward(input, weight)
        ctx.sleep_ms = sleep_ms
        # 使用 CUDA API 模拟计算时间，根据睡眠时间乘以每毫秒的周期数
        torch.cuda._sleep(int(sleep_ms * get_cycles_per_ms()))
        # 返回输入张量与权重张量的矩阵乘法结果
        return input @ weight

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 在反向传播中，获取保存的输入张量和权重张量
        (input, weight) = ctx.saved_tensors
        # 延长模拟的计算时间，乘以睡眠时间的两倍乘以每毫秒的周期数
        torch.cuda._sleep(int(2 * ctx.sleep_ms * get_cycles_per_ms()))
        # 计算输入梯度
        grad_input = grad_output @ weight.T
        # 计算权重梯度
        grad_weight = input.T @ grad_output
        # 返回输入梯度、权重梯度和 None（无额外梯度项）
        return grad_input, grad_weight, None


class LinearWithSleep(nn.Module):
    def __init__(self, dim: int, sleep_ms: int):
        # 初始化函数，定义一个具有指定维度和睡眠时间的线性层
        super().__init__()
        # 将权重定义为可学习参数，并初始化为随机张量
        self.weight = nn.Parameter(torch.randn((dim, dim)))
        # 设置层的睡眠时间
        self.sleep_ms = sleep_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，应用自定义的 Matmul 函数计算输入和权重的矩阵乘法，并应用 ReLU 激活函数
        return nn.functional.relu(Matmul.apply(x, self.weight, self.sleep_ms))


if __name__ == "__main__":
    run_tests()
```