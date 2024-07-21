# `.\pytorch\test\distributed\_composable\fully_shard\test_fully_shard_runtime.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import contextlib
import copy
import functools
import sys
from enum import auto, Enum
from typing import Callable, List, Tuple

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式模块
import torch.distributed.fsdp._traversal_utils as traversal_utils  # 导入 FSDP 的遍历工具模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed._composable import fully_shard  # 导入 fully_shard 函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy  # 导入 FSDP 和分片策略
from torch.distributed.fsdp._common_utils import _FSDPState  # 导入 FSDP 的通用工具
from torch.distributed.fsdp._flat_param import FlatParamHandle  # 导入 FSDP 的平坦参数处理模块
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入 FSDP 模块包装策略
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,  # 导入测试用的组合参数模型
    UnitModule,  # 导入测试用的单元模型
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入分布式测试的 GPU 数量检查函数
from torch.testing._internal.common_fsdp import FSDPTest  # 导入 FSDP 测试基类
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试运行相关工具

# 如果分布式不可用，打印消息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果在进行 dev-asan 测试，则跳过，因为 torch + multiprocessing spawn 存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class FSDPWrapMode(Enum):
    """枚举类定义 FSDP 包装模式"""
    AUTO_WRAP = auto()  # 自动包装模式
    MANUAL_WRAP = auto()  # 手动包装模式


class TestRuntime(FSDPTest):
    """``fully_shard`` 运行时的测试（前向/反向传播/优化器）。"""

    @property
    def world_size(self) -> int:
        """返回当前 CUDA 设备的数量作为全局大小"""
        return torch.cuda.device_count()

    def _init_models_and_optims(
        self,
        device: torch.device,
        fsdp_wrap_mode: FSDPWrapMode,
        sharding_strategy: ShardingStrategy,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module, torch.optim.Optimizer]:
        # 创建一个本地模型对象，使用 CompositeParamModel 类，指定设备
        local_model = CompositeParamModel(device=device)

        # 深度复制本地模型，以便后续操作不影响原始模型
        composable_module = copy.deepcopy(local_model)

        # 根据 FSDPWrapMode 的设置进行条件判断
        if fsdp_wrap_mode == FSDPWrapMode.AUTO_WRAP:
            # 对本地模型进行深度复制，并使用 FSDP 进行自动包装
            fsdp_wrapped_model = FSDP(
                copy.deepcopy(local_model),
                auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                use_orig_params=True,
                sharding_strategy=sharding_strategy,
            )
            # 对 composable_module 进行完全分片，使用指定的模块包装策略和分片策略
            fully_shard(
                composable_module,
                policy=ModuleWrapPolicy({UnitModule}),
                strategy=sharding_strategy,
            )

        elif fsdp_wrap_mode == FSDPWrapMode.MANUAL_WRAP:
            # 对本地模型进行深度复制，准备进行手动包装
            fsdp_wrapped_model = copy.deepcopy(local_model)
            # 使用 FSDP 包装模型的 u2 属性，保留原始参数，并指定分片策略
            fsdp_wrapped_model.u2 = FSDP(
                fsdp_wrapped_model.u2,
                use_orig_params=True,
                sharding_strategy=sharding_strategy,
            )
            # 再次使用 FSDP 对整体模型进行包装，保留原始参数，并指定分片策略
            fsdp_wrapped_model = FSDP(
                fsdp_wrapped_model,
                use_orig_params=True,
                sharding_strategy=sharding_strategy,
            )
            # 对 composable_module 的 u2 属性进行完全分片，使用指定的分片策略
            fully_shard(composable_module.u2, strategy=sharding_strategy)
            # 对整体 composable_module 进行完全分片，使用指定的分片策略
            fully_shard(composable_module, strategy=sharding_strategy)

        else:
            # 抛出值错误，如果 fsdp_wrap_mode 的值未知
            raise ValueError(f"Unknown `fsdp_wrap_mode`: {fsdp_wrap_mode}")

        # 设置学习率 LR
        LR = 1e-2
        # 使用 Adam 优化器对 fsdp_wrapped_model 的参数进行优化
        fsdp_wrapped_optim = torch.optim.Adam(fsdp_wrapped_model.parameters(), lr=LR)
        # 使用 Adam 优化器对 composable_module 的参数进行优化
        composable_optim = torch.optim.Adam(composable_module.parameters(), lr=LR)

        # 返回四个对象作为元组，分别是 composable_module, composable_optim, fsdp_wrapped_model, fsdp_wrapped_optim
        return (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        )

    @skip_if_lt_x_gpu(2)
    def test_training(self):
        """Tests training (forward, backward, optimizer)."""
        # 运行测试用例的子测试，根据不同的 fsdp_wrap_mode 和 sharding_strategy 组合
        self.run_subtests(
            {
                "fsdp_wrap_mode": [
                    FSDPWrapMode.AUTO_WRAP,
                    FSDPWrapMode.MANUAL_WRAP,
                ],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                    ShardingStrategy.HYBRID_SHARD,
                ],
            },
            self._test_training,
        )

    def _test_training(
        self, fsdp_wrap_mode: FSDPWrapMode, sharding_strategy: ShardingStrategy
    ):
        # 实际的测试函数，接受 fsdp_wrap_mode 和 sharding_strategy 作为参数
    ):
        if (
            sharding_strategy
            in [ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2]
            and fsdp_wrap_mode == FSDPWrapMode.MANUAL_WRAP
        ):
            # 如果分片策略是混合分片或者特定的混合分片类型，并且包装模式是手动包装
            # TODO: 手动包装 + HSDP 需要显式指定 pg
            return

        # 设置设备为 CUDA
        device = torch.device("cuda")
        # 初始化模型和优化器，获取包装后的模型和优化器
        (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        ) = self._init_models_and_optims(device, fsdp_wrap_mode, sharding_strategy)
        # 设置随机种子
        torch.manual_seed(self.rank + 1)
        # 执行5次循环
        for _ in range(5):
            # 创建一个形状为 (2, 100) 的张量，放置在 CUDA 设备上
            inp = torch.randn(2, 100, device="cuda")
            # 初始化损失列表
            losses: List[torch.Tensor] = []
            # 遍历模型和优化器的组合
            for model, optim in (
                (fsdp_wrapped_model, fsdp_wrapped_optim),
                (composable_module, composable_optim),
            ):
                # 将优化器的梯度置零，设置为 None
                optim.zero_grad(set_to_none=True)
                # 将输入数据传递给模型，获取输出
                out = model(inp)
                # 计算输出的总和作为损失
                loss = out.sum()
                # 将损失添加到损失列表中
                losses.append(loss)
                # 反向传播计算梯度
                loss.backward()
                # 执行优化步骤
                optim.step()
            # 断言两个损失值相等
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_unshard_reshard_order(self):
        """
        Tests that the unshard/reshard order matches between ``fully_shard``
        and ``FullyShardedDataParallel`` for the same policy.

        NOTE: We use FQNs as the proxy for checking the order across the two
        versions. See ``_check_same_param_handles()`` for details.
        """
        # 运行子测试，测试非分片/重分片顺序是否匹配
        self.run_subtests(
            {"fsdp_wrap_mode": [FSDPWrapMode.AUTO_WRAP, FSDPWrapMode.MANUAL_WRAP]},
            self._test_unshard_reshard_order,
        )

    def _check_same_param_handles(
        self,
        composable_handle: FlatParamHandle,
        wrapped_handle: FlatParamHandle,
    ) -> None:
        """
        Checks that ``composable_handles`` matches ``wrapped_handles`` by
        checking FQNs.

        For ``fully_shard``, each ``FlatParamHandle`` 's saved FQNs are
        prefixed from the local FSDP root, while for wrapper FSDP, they are
        prefixed from its owning FSDP instance, which may not be the local FSDP
        root. Thus, we relax the check to only that the wrapper FQN is a suffix
        of the composable FQN.

        If this check passes for the entire model and we separately unit-test
        parity for wrapping policies, then we can be sure that the handles
        actually match.
        """
        # 获取 composable_handle 和 wrapped_handle 的 FQN 列表
        composable_fqns = composable_handle.flat_param._fqns
        wrapped_fqns = wrapped_handle.flat_param._fqns
        # 断言两个 FQN 列表的长度相等
        self.assertEqual(len(composable_fqns), len(wrapped_fqns))
        # 遍历 composable_fqns 和 wrapped_fqns 进行检查
        for composable_fqn, wrapped_fqn in zip(composable_fqns, wrapped_fqns):
            # 断言 wrapped_fqn 是 composable_fqn 的后缀
            self.assertTrue(composable_fqn.endswith(wrapped_fqn))
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```