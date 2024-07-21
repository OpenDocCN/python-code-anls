# `.\pytorch\test\distributed\_tools\test_mem_tracker.py`

```
# Owner(s): ["module: unknown"]

# 导入必要的库和模块
import gc  # 导入垃圾回收模块
import unittest  # 导入单元测试模块
from typing import Tuple  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._tools.mem_tracker import MemTracker  # 导入内存追踪器
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入CUDA测试工具
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase  # 导入测试工具和测试用例基类
from torch.utils.checkpoint import checkpoint  # 导入模型检查点功能


class TestMemTracker(TestCase):
    def _init_cublas_workspace(self, dev: torch.device):
        # 在指定设备上初始化CuBLAS工作空间
        lin = torch.nn.Linear(768, 768, device=dev)
        inp = torch.randn(1, 768, device=dev)
        lin(inp).sum().backward()
        del lin
        del inp

    def _reset_mem_stats(self, dev: torch.device):
        # 重置指定设备的CUDA内存统计信息
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.reset_accumulated_memory_stats(dev)  # 重置累积内存统计
        torch.cuda.reset_peak_memory_stats(dev)  # 重置内存峰值统计

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_cuda_tracker_equivalence(
        self,
    ):
        """
        Tests that the tracker correctly calculates the peak memory.
        """
        dev = torch.device(torch.cuda.current_device())  # 获取当前CUDA设备
        self._init_cublas_workspace(dev)  # 初始化CuBLAS工作空间
        gc.collect(1)  # 执行一次垃圾回收
        self._reset_mem_stats(dev)  # 重置CUDA内存统计信息
        mem_stats = torch.cuda.memory_stats(dev)  # 获取CUDA内存统计信息
        pre_cuda_active = mem_stats["active_bytes.all.current"]  # 获取当前活跃内存大小

        bsz, n_layers, dim, dtype = 16, 4, 512, torch.bfloat16

        class DummyModel(nn.Module):
            def __init__(self, n_layers: int, dim: int, dtype: torch.dtype):
                super().__init__()
                self.linears = nn.ModuleList()
                for _ in range(n_layers):
                    self.linears.append(nn.Linear(dim, dim, dtype=dtype))
                    self.linears.append(nn.ReLU())

            def forward(self, x):
                for layer in self.linears:
                    x = layer(x)
                return x

        with torch.device(dev):
            model = DummyModel(n_layers, dim, dtype=dtype)  # 创建DummyModel实例
        optim = torch.optim.Adam(model.parameters(), foreach=True)  # 创建Adam优化器
        input_batch = torch.randn(bsz, dim, device=dev, dtype=dtype)  # 创建输入数据批次
        mem_tracker = MemTracker()  # 创建内存追踪器实例
        mem_tracker.track_external(model, optim, input_batch)  # 使用追踪器追踪模型、优化器和输入数据

        with mem_tracker as mt:
            for iter_idx in range(2):
                model(input_batch).sum().backward()  # 模型前向、反向传播
                optim.step()  # 执行优化步骤
                optim.zero_grad()  # 清除梯度
                if iter_idx == 0:
                    mt.reset_mod_stats()  # 重置模块统计信息
        # 检查峰值内存的准确性
        tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]  # 获取追踪器记录的峰值内存
        mem_stats = torch.cuda.memory_stats(dev)  # 获取CUDA内存统计信息
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active  # 计算CUDA记录的峰值内存
        accuracy = tracker_max / cuda_max  # 计算精确度
        self.assertAlmostEqual(accuracy, 1.0, delta=0.1)  # 断言精确度在允许的误差范围内

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_tracker_with_activation_checkpointing(
        self,
    ):
        """
        Tests that the tracker correctly computes the peak memory during activation checkpointing.
        """
        # 获取当前 CUDA 设备
        dev = torch.device(torch.cuda.current_device())
        # 初始化 cuBLAS 工作空间
        self._init_cublas_workspace(dev)
        # 手动触发垃圾回收
        gc.collect(1)
        # 重置 CUDA 设备的内存统计
        self._reset_mem_stats(dev)
        # 获取 CUDA 设备的内存统计信息
        mem_stats = torch.cuda.memory_stats(dev)
        # 记录 CUDA 设备当前活跃内存
        pre_cuda_active = mem_stats["active_bytes.all.current"]

        # 定义 MLP 模块
        bsz, n_layers, dim, dtype = 128, 4, 1024, torch.float16

        class MLPBlock(nn.Module):
            def __init__(self, dim: int, dtype: torch.dtype):
                super().__init__()
                # 定义 MLP 块，包含两个线性层和 ReLU 激活函数
                self.mlp_block = nn.Sequential(
                    nn.Linear(dim, 2 * dim, dtype=dtype),
                    nn.ReLU(),
                    nn.Linear(2 * dim, dim, dtype=dtype),
                )

            def forward(self, x):
                return self.mlp_block(x)

        # 定义自定义模块 MyModule
        class MyModule(nn.Module):
            def __init__(
                self, n_layers: int, dim: int, dtype: torch.dtype, use_ac: bool = False
            ):
                super().__init__()
                self.mlp_blocks = nn.ModuleList()
                self.use_ac = use_ac
                # 根据层数循环添加 MLPBlock 到模块列表
                for _ in range(n_layers):
                    self.mlp_blocks.append(MLPBlock(dim, dtype=dtype))

            def forward(self, x):
                for i, block in enumerate(self.mlp_blocks):
                    if i >= 1 and self.use_ac:
                        # 使用 activation checkpointing 来减少内存占用
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        x = block(x)
                return x

        # 在指定 CUDA 设备上构建模型
        with torch.device(dev):
            model = MyModule(n_layers, dim, dtype, True)
        # 使用 Adam 优化器优化模型参数
        optim = torch.optim.Adam(model.parameters(), foreach=True)
        # 创建内存追踪器对象
        mem_tracker = MemTracker()
        # 追踪模型和优化器的外部内存使用情况
        mem_tracker.track_external(model, optim)
        # 使用内存追踪器对象 mt 进行内存追踪
        with mem_tracker as mt:
            # 生成指定形状和类型的随机输入张量
            input_batch = torch.randn(bsz, dim, dim, device=dev, dtype=dtype)
            # 进行两次迭代
            for iter_idx in range(2):
                # 模型前向传播，计算损失并反向传播
                model(input_batch).sum().backward()
                # 执行一步优化
                optim.step()
                # 清空梯度
                optim.zero_grad()
                # 在第一次迭代时重置模块的内存统计信息
                if iter_idx == 0:
                    mt.reset_mod_stats()

        # 检查峰值内存的准确性
        # 获取内存追踪器的峰值内存快照
        tracker_max = mt.get_tracker_snapshot("peak")[dev]["Total"]
        # 获取 CUDA 设备的内存统计信息
        mem_stats = torch.cuda.memory_stats(dev)
        # 计算 CUDA 设备的峰值活跃内存
        cuda_max = mem_stats["active_bytes.all.peak"] - pre_cuda_active
        # 计算内存追踪器和 CUDA 设备内存峰值的精度
        accuracy = tracker_max / cuda_max
        # 断言精度接近 1.0，允许的误差为 0.1
        self.assertAlmostEqual(accuracy, 1.0, delta=0.1)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
# 如果当前脚本被直接执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```