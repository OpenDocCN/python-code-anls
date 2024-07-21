# `.\pytorch\test\distributed\fsdp\test_fsdp_clip_grad_norm.py`

```py
# Owner(s): ["oncall: distributed"]

import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数
import sys  # 导入 sys 模块，提供了访问与 Python 解释器交互的功能
from typing import Union  # 导入 Union 类型提示，用于支持多种类型的注解

import torch  # 导入 PyTorch 模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch import distributed as dist  # 导入 PyTorch 分布式包中的 distributed 模块
from torch.distributed.fsdp import ShardingStrategy  # 导入 FSDP 中的 ShardingStrategy 类
from torch.distributed.fsdp.fully_sharded_data_parallel import (  # 导入 FSDP 中的主要组件
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入 FSDP 中的 ModuleWrapPolicy 类
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer  # 导入 Transformer 模块中的编码器和解码器层
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入 PyTorch 分布式数据并行模块作为 DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试模块中的 GPU 数量检查装饰器
from torch.testing._internal.common_fsdp import (  # 导入 FSDP 测试中的公共函数和类
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (  # 导入通用测试工具函数
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if not dist.is_available():  # 检查当前环境是否支持分布式运行，如果不支持则退出并打印提示信息
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:  # 如果当前测试配置中指定使用开发者模式的地址安全性分析，则打印相关问题并退出
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestClipGradNorm(FSDPTest):
    """Tests :meth:`FullyShardedDataParallel.clip_grad_norm_`."""

    @skip_if_lt_x_gpu(2)  # 装饰器，如果 GPU 数量少于 2 则跳过测试
    def test_non_root(self):
        """
        Tests that calling ``clip_grad_norm_()`` on a non-root FSDP instance
        raises an error.
        """

        class Model(nn.Module):  # 定义一个简单的神经网络模型类
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(5, 5)  # 添加线性层
                self.lin2 = nn.Linear(5, 5)  # 添加线性层

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin2(self.lin1(x))  # 前向传播

        model = Model().cuda()  # 创建模型实例并移动到 GPU 上
        model.lin2 = FSDP(model.lin2)  # 对模型的第二个线性层应用 FSDP 封装
        fsdp_model = FSDP(model)  # 对整个模型应用 FSDP 封装
        fsdp_model(torch.randn((2, 5), device=torch.device("cuda"))).sum().backward()  # 对模型进行前向传播和反向传播
        error_regex = "should only be called on the root FSDP instance"  # 期待的错误信息正则表达式
        with self.assertRaisesRegex(RuntimeError, error_regex):  # 断言捕获 RuntimeError 异常并检查错误信息是否符合预期
            fsdp_model.lin2.clip_grad_norm_(max_norm=2)  # 在非根 FSDP 实例上调用 clip_grad_norm_

    @skip_if_lt_x_gpu(2)  # 装饰器，如果 GPU 数量少于 2 则跳过测试
    def test_ddp_parity(self):
        """
        Tests FSDP with ``FullyShardedDataParallel.clip_grad_norm_()` against
        DDP with ``torch.nn.utils.clip_grad_norm_()` when using full precision.
        """
        self.run_subtests(
            {
                "max_norm": [1, 2.5],  # 不同的 max_norm 参数值列表
                "norm_type": [1, 2, float("inf")],  # 不同的 norm_type 参数值列表
                "sharding_strategy": [  # 不同的分片策略参数值列表
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                    "mixed_strategy",
                ],
                "use_orig_params": [False, True],  # 是否使用原始参数的布尔值列表
                "offload_params": [False, True],  # 是否卸载参数的布尔值列表
            },
            self._test_ddp_parity,  # 在子测试中运行 _test_ddp_parity 方法
        )
    @skip_if_lt_x_gpu(2)
    # 如果 GPU 数量小于 2，则跳过这个测试函数
    def test_low_precision_grads(self):
        """Tests ``clip_grad_norm_()`` when using low precision gradients."""
        # 运行子测试，测试不同参数组合下的低精度梯度情况
        self.run_subtests(
            {
                "max_norm": [1, 2.5],
                "norm_type": [1, 2, float("inf")],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_low_precision_grads,
        )

    def _test_low_precision_grads(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int],
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        # 设置 FSDP 参数的关键字参数
        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
            "use_orig_params": use_orig_params,
            "mixed_precision": MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                keep_low_precision_grads=True,
            ),
        }
        # 初始化 FSDP 模块
        fsdp_model = FSDP(
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                deterministic=True,
                fsdp_kwargs=fsdp_kwargs,
            ),
            **fsdp_kwargs,
        )
        # 获取输入数据并在 CUDA 设备上运行模型
        inp = fsdp_model.module.get_input(torch.device("cuda"))
        out = fsdp_model(*inp)
        # 计算梯度
        out.sum().backward()
        # 检查每个参数的梯度是否为 torch.float16 类型
        for param in fsdp_model.parameters():
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float16)
        # 对梯度进行裁剪，并返回裁剪后的总体范数
        total_norm = fsdp_model.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)
        # 检查总体范数的数据类型是否为 torch.float16
        self.assertEqual(total_norm.dtype, torch.float16)
        # 尽最大努力检查每个梯度的范数是否最大不超过设定的最大范数
        # 因为 DDP 不原生支持混合精度，所以无法直接比较实现完全一致性
        for param in fsdp_model.parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.linalg.vector_norm(param.grad, norm_type).item() <= max_norm,
                )

    @skip_if_lt_x_gpu(2)
    # 如果 GPU 数量小于 2，则跳过这个测试函数
    def test_no_gradients(self):
        """
        Tests that calling ``clip_grad_norm_()`` when the FDSP module has no
        gradients simply returns a scalar zero tensor in FP32 without erroring.
        """
        # 运行子测试，测试在没有梯度时调用 ``clip_grad_norm_()`` 的情况
        self.run_subtests(
            {"use_orig_params": [False, True]},
            self._test_no_gradients,
        )
    # 定义一个测试方法，用于验证在没有梯度的情况下的行为
    def _test_no_gradients(self, use_orig_params: bool):
        # 创建一个具有输入和输出大小均为 24 的线性模块
        lin_module = nn.Linear(24, 24)
        # 设置混合精度配置，使用 float16 参数，减少精度使用 float32，缓冲区数据类型也为 float32
        mixed_precision_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        # 使用 FSDP 模块封装线性模块，指定梯度分片策略为 SHARD_GRAD_OP，使用指定的混合精度配置
        fsdp_module = FSDP(
            lin_module,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mixed_precision_config,
            device_id=self.rank,  # 设置设备 ID
            use_orig_params=use_orig_params,  # 指定是否使用原始参数
        )
        # 创建一个在 CUDA 设备上的随机输入张量
        inp = torch.randn(32, 24, device="cuda")
        # 将输入张量传递给 FSDP 模块进行处理
        fsdp_module(inp)
        # 断言捕获到预期的警告类型和正则表达式内容
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="on rank "
            rf"{self.rank} with no gradients -- returning the total "
            "norm in the default dtype torch.float32",
        ):
            # 调用 FSDP 模块的梯度裁剪方法，设置最大梯度范数为 1
            total_norm = fsdp_module.clip_grad_norm_(1)
        # 断言返回的总梯度范数的数据类型为 torch.float32
        self.assertEqual(total_norm.dtype, torch.float32)
        # 断言返回的总梯度范数值为 0.0，在 CUDA 设备上
        self.assertEqual(total_norm, torch.tensor(0.0, device="cuda"))
# 调用函数 instantiate_parametrized_tests，传入 TestClipGradNorm 作为参数，实例化参数化测试
instantiate_parametrized_tests(TestClipGradNorm)

# 检查当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 如果作为主程序运行，则调用函数 run_tests()
    run_tests()
```