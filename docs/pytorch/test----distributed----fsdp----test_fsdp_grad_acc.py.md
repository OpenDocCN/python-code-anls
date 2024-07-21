# `.\pytorch\test\distributed\fsdp\test_fsdp_grad_acc.py`

```
# Owner(s): ["oncall: distributed"]

import contextlib  # 引入上下文管理器相关的模块
import itertools  # 引入迭代工具模块
import sys  # 引入系统相关的模块
from dataclasses import dataclass  # 引入用于创建数据类的装饰器
from typing import Any, Dict, List, Optional, Tuple  # 引入类型提示相关的模块

import torch  # 引入PyTorch深度学习框架
from torch import distributed as dist  # 引入PyTorch分布式通信模块
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP  # 引入FSDP模块及其组件
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,  # 引入梯度预取模块
    ShardingStrategy,  # 引入分片策略模块
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 引入用于跳过测试的分布式GPU检查
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,  # 引入CUDA初始化模式
    FSDPInitMode,  # 引入FSDP初始化模式
    FSDPTest,  # 引入用于FSDP测试的基类
    TransformerWithSharedParams,  # 引入带共享参数的Transformer
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 引入实例化参数化测试的工具函数
    parametrize,  # 引入参数化装饰器
    run_tests,  # 引入运行测试的函数
    TEST_WITH_DEV_DBG_ASAN,  # 引入用于开发调试的ASAN测试开关
)

if not dist.is_available():  # 如果分布式通信不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印跳过测试的信息到标准错误流
    sys.exit(0)  # 终止程序

if TEST_WITH_DEV_DBG_ASAN:  # 如果开启了开发调试的ASAN测试
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,  # 打印提示信息到标准错误流
    )
    sys.exit(0)  # 终止程序


@dataclass
class _GradAccConfig:
    """
    This configures how gradients are accumulated in :meth:`_test_grad_acc`.
    Each instance of this class represents ``num_iters``-many consecutive
    iterations, where the ``no_sync()`` context manager is used or not as given
    by ``use_no_sync``.

    Attributes:
        use_no_sync (bool): Indicates whether to use the ``no_sync()`` context
            manager as the way to accumulate gradients.
        num_iters (int): Number of iterations to accumulate gradients.
    """

    use_no_sync: bool  # 指示是否使用 `no_sync()` 上下文管理器来累积梯度
    num_iters: int  # 累积梯度的迭代次数

    def __repr__(self) -> str:
        # Override to remove any spaces in the string to appease the internal
        # build's test name parser
        return f"(use_no_sync={self.use_no_sync}," f"num_iters={self.num_iters})"  # 返回用于描述对象的字符串表示，用于测试名称解析


@dataclass
class _GradAccConfigs:
    """
    This wraps a :class:`list` of :class:`_GradAccConfig` instances with the
    sole purpose of overriding :meth:`__repr__` to remove spaces.
    """

    configs: List[_GradAccConfig]  # 包含多个 `_GradAccConfig` 实例的列表

    def __repr__(self) -> str:
        # Override to remove any spaces in the string to appease the internal
        # build's test name parser
        return "[" + ",".join(config.__repr__() for config in self.configs) + "]"  # 返回用于描述对象的字符串表示，用于测试名称解析


class TestGradAcc(FSDPTest):
    """Tests ``FullyShardedDataParallel``'s gradient accumulation via both its
    ``no_sync()`` context manager and without the context manager."""

    @property
    def world_size(self) -> int:
        return 2  # 返回测试所需的全局并行度大小

    def _test_grad_acc(
        self,
        batch_dim: int,
        configs: List[_GradAccConfig],
        cpu_offload: CPUOffload,
        backward_prefetch: Optional[BackwardPrefetch],
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    def _get_subtest_config(self) -> Dict[str, List[Any]]:
        """Returns a subtest configuration that subtests prefetching."""
        # 返回一个包含子测试配置的字典，用于测试预取
        return {
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
            "sharding_strategy": [
                ShardingStrategy.FULL_SHARD,
                ShardingStrategy.SHARD_GRAD_OP,
                ShardingStrategy.NO_SHARD,
            ],
        }

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "configs",
        [
            _GradAccConfigs(
                [
                    _GradAccConfig(use_no_sync=True, num_iters=3),
                    _GradAccConfig(use_no_sync=False, num_iters=3),
                    _GradAccConfig(use_no_sync=True, num_iters=3),
                ]
            ),
            _GradAccConfigs(
                [
                    _GradAccConfig(use_no_sync=False, num_iters=3),
                    _GradAccConfig(use_no_sync=True, num_iters=3),
                    _GradAccConfig(use_no_sync=False, num_iters=3),
                ]
            ),
        ],
    )
    @parametrize("use_orig_params", [False, True])
    def test_grad_acc(
        self,
        configs: _GradAccConfigs,
        use_orig_params: bool,
    ):
        """
        Tests gradient accumulation without parameter CPU offloading.

        This exercises gradient accumulation inside and outside the
        ``no_sync()`` context manager, in particular by interleaving the two.
        It tests both interleaving starting with (and ending with, resp.)
        inside versus outside ``no_sync()`` to ensure that initial conditions
        (and final conditions, resp.) do not affect the correctness.
        """
        # 获取子测试配置
        subtest_config = self._get_subtest_config()
        subtest_config["cpu_offload"] = [CPUOffload(offload_params=False)]
        # 运行子测试，测试梯度累积功能
        self.run_subtests(
            subtest_config,
            self._test_grad_acc,
            batch_dim=1,
            configs=configs.configs,
            use_orig_params=use_orig_params,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_orig_params", [False, True])
    def test_grad_acc_cpu_offload(
        self,
        use_orig_params: bool,
    ):
        """
        Tests gradient accumulation with parameter CPU offloading.

        This tests gradient accumulation while using CPU offloading of parameters.
        """
        """
        Tests gradient accumulation with parameter CPU offloading.

        NOTE: Gradient accumulation without using the ``no_sync()`` context
        manager is not currently compatible with CPU offloading.
        """
        # 仅测试使用 `no_sync` 的情况，因为在没有使用 `no_sync()` 上下文管理器的情况下，
        # 不支持参数 CPU 离载。
        configs = _GradAccConfigs([_GradAccConfig(use_no_sync=True, num_iters=3)])
        # 获取子测试配置
        subtest_config = self._get_subtest_config()
        # 设置 CPU 离载参数为开启状态
        subtest_config["cpu_offload"] = [CPUOffload(offload_params=True)]
        # 运行子测试，测试梯度积累
        self.run_subtests(
            subtest_config,            # 子测试配置
            self._test_grad_acc,       # 梯度积累测试函数
            batch_dim=1,               # 批处理维度设为1
            configs=configs.configs,   # 使用的梯度积累配置
            use_orig_params=use_orig_params,  # 是否使用原始参数
        )
# 使用给定的参数化测试类实例化参数化测试
instantiate_parametrized_tests(TestGradAcc)

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```