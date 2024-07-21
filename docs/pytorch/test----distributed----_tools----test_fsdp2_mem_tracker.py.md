# `.\pytorch\test\distributed\_tools\test_fsdp2_mem_tracker.py`

```
# Owner(s): ["module: unknown"]
# 导入 functools 模块，用于高阶函数支持
import functools
# 导入 Python 的垃圾回收模块，用于显式控制内存的释放
import gc
# 导入 Union 类型，用于指定多种类型的变量
from typing import Union

# 导入 PyTorch 主模块
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 分布式计算的检查点模块
from torch.distributed._composable import checkpoint
# 导入 PyTorch 分布式计算的 FSDP（Fully Sharded Data Parallel）模块
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
# 导入 PyTorch 分布式计算的设备网格初始化模块
from torch.distributed._tensor import init_device_mesh
# 导入 PyTorch 分布式计算的 FSDP 内存追踪模块
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
# 导入 PyTorch 分布式算法的检查点封装模块
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointWrapper,
)

# 导入 PyTorch 内部的分布式计算测试模块，用于 GPU 数量检测
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 导入 PyTorch 内部的 FSDP 测试模块和 MLP 模块
from torch.testing._internal.common_fsdp import FSDPTest, MLP
# 导入 PyTorch 内部的通用测试工具函数模块
from torch.testing._internal.common_utils import run_tests
# 导入 PyTorch 内部的分布式计算张量的通用数据类型模块
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


def _init_cublas_workspace(dev: torch.device):
    # 创建线性层，输入输出维度均为 768，使用指定设备
    lin = torch.nn.Linear(768, 768, device=dev)
    # 创建输入张量，形状为 [1, 768]，使用指定设备
    inp = torch.randn(1, 768, device=dev)
    # 对线性层的输出进行求和并反向传播
    lin(inp).sum().backward()
    # 释放线性层和输入张量的内存
    del lin
    del inp


def _reset_mem_stats(dev: torch.device):
    # 清空当前 CUDA 设备的缓存
    torch.cuda.empty_cache()
    # 重置当前 CUDA 设备的累积内存统计信息
    torch.cuda.reset_accumulated_memory_stats(dev)
    # 重置当前 CUDA 设备的内存峰值统计信息
    torch.cuda.reset_peak_memory_stats(dev)


class TestTrackerFullyShard1DTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        # 返回 CUDA 设备数和 4 的较小值作为当前测试的 world_size
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_tracker_multi_group_eager(self):
        """
        Tests tracker accuracy when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction) and different mixed precision policies.
        """
        # 运行多个子测试，测试多参数组情况下的追踪器准确性
        self.run_subtests(
            {
                # 是否在前向传播后进行重新分片的标志
                "reshard_after_forward": [True, False],
                # CPU 离载策略，不使用固定内存
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=False),
                    # 默认离载策略
                    OffloadPolicy(),
                ],
                # 混合精度策略，参数精度为 float16，减少时使用 float32
                "mp_policy": [
                    MixedPrecisionPolicy(
                        param_dtype=torch.float16, reduce_dtype=torch.float32
                    ),
                ],
            },
            self._test_tracker_multi_group,
        )

    def _test_tracker_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        mp_policy: MixedPrecisionPolicy,
        ):
        # 调试模式标志位，默认关闭
        debug = False
        # 获取当前 CUDA 设备并设置为开发设备
        dev = torch.device(torch.cuda.current_device())
        # 初始化 cuBLAS 工作空间
        _init_cublas_workspace(dev)
        # 手动触发垃圾回收
        gc.collect()
        # 重置 CUDA 内存统计
        _reset_mem_stats(dev)
        # 获取当前 CUDA 设备的内存统计信息
        mem_stats = torch.cuda.memory_stats(dev)
        # 记录 CUDA 操作前的活跃内存大小
        pre_cuda_active = mem_stats["active_bytes.all.current"]
        # 设置随机种子为 42
        torch.manual_seed(42)
        # 定义线性层维度和批大小
        lin_dim, bsz = 2048, 8192
        # 在指定的 CUDA 设备上创建包含 4 个 MLP 层的神经网络模型
        with torch.device(dev):
            model = nn.Sequential(*[MLP(dim=lin_dim, device=dev) for _ in range(4)])
        # 初始化 CUDA 设备的网格
        mesh = init_device_mesh("cuda", (self.world_size,))
        # 部分函数柯里化，生成部分分片函数
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
            mp_policy=mp_policy,
        )
        # 对模型中的每个 MLP 层应用分片函数
        for mlp in model:
            fully_shard_fn(mlp)
        # 对整个模型应用分片函数
        fully_shard_fn(model)
        # 使用 Adam 优化器优化模型参数
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        # 生成随机输入数据并发送到 CUDA 设备
        inp = torch.randn((bsz, lin_dim), device=dev)
        # 创建用于跟踪内存和性能统计的对象
        fmt = FSDPMemTracker(model, optim)
        # 跟踪输入数据
        fmt.track_inputs((inp,))
        # 进入性能跟踪上下文
        with fmt:
            # 迭代两次
            for iter_idx in range(2):
                # 前向传播计算损失
                loss = model(inp).sum()
                # 反向传播计算梯度
                loss.backward()
                # 执行优化步骤
                optim.step()
                # 清空梯度
                optim.zero_grad()
                # 在第一次迭代时，重置模型统计信息
                if iter_idx == 0:
                    fmt.reset_mod_stats()
        # 获取 CUDA 设备的内存统计信息
        mem_stats = torch.cuda.memory_stats()
        # 获取跟踪器的峰值内存使用量
        tracker_max = fmt.get_tracker_snapshot("peak")[dev]["Total"]
        # 计算 CUDA 设备的峰值内存使用量
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        # 计算准确率
        accuracy = tracker_max / cuda_max
        # 如果是主进程且开启了调试模式，则打印准确率和相关信息
        if self.rank == 0 and debug:
            print(f"Accuracy: {accuracy} Tracker Max:{tracker_max} CUDA Max:{cuda_max}")
        # 断言准确率接近 1.0
        self.assertAlmostEqual(
            accuracy,
            1.0,
            delta=0.1,
            msg=f"Tracker Max:{tracker_max} CUDA Max:{cuda_max}",
        )
        # 清理释放模型和输入数据的内存
        del model
        del inp
        del optim

    @skip_if_lt_x_gpu(2)
    def test_tracker_non_root_forward_backward(self):
        """
        Tests tracker accuracy when running forward/backward through a non-root.
        测试在非根节点上进行前向/后向传播时的追踪器精度。
        """
        debug = False
        dev = torch.device(torch.cuda.current_device())
        # 获取当前设备并初始化 CuBLAS 工作空间
        _init_cublas_workspace(dev)
        # 手动执行垃圾回收
        gc.collect()
        # 重置当前设备的内存统计
        _reset_mem_stats(dev)
        # 获取当前设备的内存统计信息
        mem_stats = torch.cuda.memory_stats(dev)
        # 记录 CUDA 激活内存的初始值
        pre_cuda_active = mem_stats["active_bytes.all.current"]
        # 设置随机种子
        torch.manual_seed(42)
        # 设置线性层的维度和批量大小
        lin_dim, bsz = 2048, 8
        # 创建一个包含三个 MLP 模块的顺序模型
        model = nn.Sequential(*[MLP(lin_dim, dev) for _ in range(3)])
        # 对每个 MLP 模块进行完全分片
        for mlp in model:
            fully_shard(mlp)
        # 对整个模型进行完全分片
        fully_shard(model)
        # 使用 Adam 优化器对模型参数进行优化
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        # 设置随机种子，考虑当前进程的 rank
        torch.manual_seed(42 + self.rank)
        # 在指定设备上生成随机输入数据
        inp = torch.randn((bsz, lin_dim), device=dev)
        # 创建 FSDP 内存追踪器对象
        fmt = FSDPMemTracker(model, optim)
        # 追踪输入数据
        fmt.track_inputs((inp,))
        # 进入追踪器上下文管理器
        with fmt:
            for iter_idx in range(2):
                # 计算非根模块的损失值
                nonroot_loss = model[0](inp).sum()
                # 反向传播非根模块的损失
                nonroot_loss.backward()
                # 执行优化步骤
                optim.step()
                # 清空优化器的梯度
                optim.zero_grad()
                # 在第一次迭代后重置模块的统计信息
                if iter_idx == 0:
                    fmt.reset_mod_stats()
        # 获取当前设备的内存统计信息
        mem_stats = torch.cuda.memory_stats()
        # 获取追踪器的峰值内存使用量
        tracker_max = fmt.get_tracker_snapshot("peak")[dev]["Total"]
        # 计算 CUDA 的峰值内存使用量
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        # 计算追踪器的精度
        accuracy = tracker_max / cuda_max
        # 如果当前进程的 rank 是 0 并且 debug 标志为真，则打印精度信息
        if self.rank == 0 and debug:
            print(f"Accuracy: {accuracy} Tracker Max:{tracker_max} CUDA Max:{cuda_max}")
        # 断言追踪器的精度接近 1.0
        self.assertAlmostEqual(
            accuracy,
            1.0,
            delta=0.1,
            msg=f"Tracker Max:{tracker_max} CUDA Max:{cuda_max}",
        )
        # 清理变量 inp, model, optim
        del inp
        del model
        del optim
# 定义一个测试类 `TestTrackerFullyShard1DTrainingCompose`，继承自 `FSDPTest` 类
class TestTrackerFullyShard1DTrainingCompose(FSDPTest):

    # 定义一个属性方法 `world_size`，返回当前 CUDA 设备数量和 4 中的较小值作为整数
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 4)

    # 定义一个测试方法 `test_tracker_with_activation_checkpointing`，在 GPU 数量小于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    def test_tracker_with_activation_checkpointing(self):
        """
        Tests tracker accuracy when composing with activation checkpointing.
        """
        # 运行子测试，测试不同参数组合下的激活检查点跟踪
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],  # 测试参数：是否在前向传播后重新分片
                "checkpoint_impl": ["composable", "wrapper"],  # 测试参数：检查点实现方式
            },
            self._test_tracker_with_activation_checkpointing,  # 测试方法的回调函数
        )

    # 定义一个测试辅助方法 `_test_tracker_with_activation_checkpointing`
    def _test_tracker_with_activation_checkpointing(
        self, reshard_after_forward: Union[bool, int], checkpoint_impl: str
    ):
        ):
            # 断言检查 checkpoint_impl 只能是 "composable" 或者 "wrapper"
            assert checkpoint_impl in ("composable", "wrapper")
            # 调试标志设为 False
            debug = False
            # 获取当前 CUDA 设备
            dev = torch.device(torch.cuda.current_device())
            # 初始化 CuBLAS 工作空间
            _init_cublas_workspace(dev)
            # 手动进行垃圾回收
            gc.collect()
            # 重置 CUDA 内存统计信息
            _reset_mem_stats(dev)
            # 获取当前 CUDA 设备的内存统计信息
            mem_stats = torch.cuda.memory_stats(dev)
            # 记录 CUDA 激活状态前的内存占用
            pre_cuda_active = mem_stats["active_bytes.all.current"]
            # 设置随机种子为 42
            torch.manual_seed(42)
            # 定义词汇表大小和批处理大小、序列长度
            vocab_size = 8192
            bsz, seq_len = 16, 512
            # 使用当前 CUDA 设备进行操作
            with torch.device(dev):
                # 设置模型参数
                model_args = ModelArgs(
                    n_layers=4,
                    n_heads=4,
                    vocab_size=vocab_size,
                    max_seq_len=seq_len,
                    dropout_p=0.1,
                )
                # 创建 Transformer 模型
                model = Transformer(model_args)
            # 是否对每个模块进行操作标志设为 False
            foreach = False
            # 定义一个部分应用了 reshard_after_forward 参数的函数
            fully_shard_fn = functools.partial(
                fully_shard,
                reshard_after_forward=reshard_after_forward,
            )
            # 根据 checkpoint_impl 的值执行不同的操作
            if checkpoint_impl == "wrapper":
                # 应用激活检查点到模型中的 TransformerBlock 模块
                apply_activation_checkpointing(
                    model, check_fn=lambda m: isinstance(m, TransformerBlock)
                )
                # 遍历模型的所有模块
                for module in model.modules():
                    # 对包装了 TransformerBlock 的 CheckpointWrapper 应用 fully_shard_fn 函数
                    if isinstance(module, CheckpointWrapper):
                        # 应用 fully_shard_fn 到 CheckpointWrapper
                        fully_shard_fn(module)
            else:
                # 对模型的所有模块进行遍历
                for module in model.modules():
                    # 如果是 TransformerBlock 模块
                    if isinstance(module, TransformerBlock):
                        # 根据 checkpoint_impl 的值执行不同的检查点操作
                        if checkpoint_impl == "composable":
                            # 对 TransformerBlock 应用检查点
                            checkpoint(module)
                        # 对 TransformerBlock 应用 fully_shard_fn 函数
                        fully_shard_fn(module)
            # 对整个模型应用 fully_shard_fn 函数
            fully_shard_fn(model)
            # 使用 Adam 优化器优化模型参数
            optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=foreach)

            # 设置随机种子为 42 加上当前进程的排名
            torch.manual_seed(42 + self.rank)
            # 生成一个随机输入张量
            inp = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
            # 创建 FSDPMemTracker 实例，用于跟踪模型和优化器的状态
            fmt = FSDPMemTracker(model, optim)
            # 跟踪输入数据
            fmt.track_inputs((inp,))
            # 进入 FSDPMemTracker 上下文管理器
            with fmt:
                # 执行两次迭代
                for iter_idx in range(2):
                    # 计算模型的损失值
                    loss = model(inp).sum()
                    # 反向传播
                    loss.backward()
                    # 执行优化器的一步参数更新
                    optim.step()
                    # 清空优化器的梯度缓存
                    optim.zero_grad()
                    # 如果是第一次迭代，重置 FSDPMemTracker 的模块统计信息
                    if iter_idx == 0:
                        fmt.reset_mod_stats()
            # 获取 CUDA 内存统计信息
            mem_stats = torch.cuda.memory_stats()
            # 获取跟踪器的峰值内存使用量
            tracker_max = fmt.get_tracker_snapshot("peak")[dev]["Total"]
            # 计算 CUDA 的峰值内存使用量
            cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
            # 计算准确率
            accuracy = tracker_max / cuda_max
            # 如果进程的排名是 0 并且调试标志为 True，则打印准确率和相关信息
            if self.rank == 0 and debug:
                print(f"Accuracy: {accuracy} Tracker Max:{tracker_max} CUDA Max:{cuda_max}")
            # 断言准确率接近 1.0，允许的误差为 0.1
            self.assertAlmostEqual(
                accuracy,
                1.0,
                delta=0.1,
                msg=f"Tracker Max:{tracker_max} CUDA Max:{cuda_max}",
            )
            # 清理变量
            del inp
            del model
            del optim
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```