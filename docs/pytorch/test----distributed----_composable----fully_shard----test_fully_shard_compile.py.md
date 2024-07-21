# `.\pytorch\test\distributed\_composable\fully_shard\test_fully_shard_compile.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块
import copy
import sys
import unittest

import torch
import torch.distributed as dist  # 导入分布式训练相关模块
import torch.nn as nn
from torch.distributed._composable import checkpoint, fully_shard  # 导入分布式相关功能
from torch.distributed.fsdp import ShardingStrategy  # 导入分片策略
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块包装策略
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入测试相关函数
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)  # 导入分布式测试相关模块和类
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试工具函数和标记
from torch.utils._triton import has_triton  # 导入 Triton 相关模块

# 如果分布式训练不可用，打印消息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果使用了 dev-asan 标记，打印消息并退出，因为 torch 和 multiprocessing spawn 存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestCompile(FSDPTest):  # 定义测试类 TestCompile，继承自 FSDPTest
    @property
    def world_size(self) -> int:  # 定义 world_size 属性，返回当前 CUDA 设备数量
        return torch.cuda.device_count()

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_compile(self):  # 定义测试方法 test_compile
        self.run_subtests(  # 运行子测试方法，参数是一个字典和方法引用
            {
                "sharding_strategy": [  # 分片策略选项列表
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "skip_fsdp_guards": [True, False],  # 跳过 FSDP 保护的选项列表
                "act_checkpoint": [True, False],  # 执行检查点的选项列表
            },
            self._test_compile,  # 调用内部测试方法 _test_compile
        )

    def _test_compile(  # 定义内部测试方法 _test_compile，接受多个参数
        self,
        sharding_strategy: ShardingStrategy,  # 分片策略参数
        skip_fsdp_guards: bool,  # 跳过 FSDP 保护的布尔参数
        act_checkpoint: bool,  # 执行检查点的布尔参数
        ):
        ):
            # 设置是否跳过 FSDP 保护
            torch._dynamo.config.skip_fsdp_guards = skip_fsdp_guards
            # 构建 FSDP 参数字典
            fsdp_kwargs = {
                "policy": ModuleWrapPolicy(
                    {
                        nn.TransformerEncoderLayer,
                        nn.TransformerDecoderLayer,
                    }
                ),
                "strategy": sharding_strategy,
            }
            # 初始化基础模型，不使用 FSDP，CUDA 初始化在 CUDA 操作前，确定性设置为 True
            base_model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                deterministic=True,
            )
            # 对基础模型进行完全分片，并复制为参考模型
            ref_model = fully_shard(copy.deepcopy(base_model), **fsdp_kwargs)
            # 创建参考模型的优化器，使用 Adam 优化器，学习率为 0.01
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            # 对基础模型进行完全分片，并复制为当前模型
            model = fully_shard(copy.deepcopy(base_model), **fsdp_kwargs)
            # 如果启用了激活检查点
            if act_checkpoint:
                # 遍历模型的每个模块
                for module in model.modules():
                    # 如果模块是 nn.TransformerEncoderLayer 或 nn.TransformerDecoderLayer 类型
                    if isinstance(
                        module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
                    ):
                        # 对该模块执行检查点操作
                        checkpoint(module)
            # 对模型进行编译
            model = torch.compile(model)
            # 创建当前模型的优化器，使用 Adam 优化器，学习率为 0.01
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            # 执行 10 轮训练
            for i in range(10):
                losses = []
                # 获取参考模型在 CUDA 设备上的输入
                inp = ref_model.get_input(torch.device("cuda"))
                # 遍历参考模型和当前模型以及它们各自的优化器
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    # 梯度清零
                    _optim.zero_grad()
                    # 计算模型输出的损失并求和
                    loss = _model(*inp).sum()
                    losses.append(loss)
                    # 反向传播损失
                    loss.backward()
                    # 执行优化步骤
                    _optim.step()
                # 断言两个模型的损失相等
                self.assertEqual(losses[0], losses[1])
# 如果当前脚本被直接执行而非被导入为模块，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```