# `.\pytorch\test\distributed\fsdp\test_fsdp_comm.py`

```py
# Owner(s): ["oncall: distributed"]

import sys  # 导入系统模块sys，用于与系统交互
from contextlib import nullcontext  # 导入nullcontext，用于创建一个不执行任何操作的上下文管理器
from enum import auto, Enum  # 导入auto和Enum，用于创建枚举类型
from typing import List, Optional  # 引入类型提示List和Optional
from unittest.mock import patch  # 导入patch，用于在测试中模拟对象

import torch  # 导入PyTorch深度学习库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch中的函数库
from torch import distributed as dist  # 导入PyTorch分布式模块的别名dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入全分片数据并行模块，并起别名为FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy  # 导入分片策略类
from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # 导入模块包装策略类
from torch.nn.parallel.distributed import DistributedDataParallel as DDP  # 导入分布式数据并行模块，并起别名为DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入跳过条件：如果GPU数量小于指定数量
from torch.testing._internal.common_fsdp import (  # 导入FSDP相关测试工具
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    MLP,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (  # 导入通用测试工具
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if not dist.is_available():  # 如果分布式不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出信息到标准错误流，表明跳过测试
    sys.exit(0)  # 退出程序，返回状态码0表示成功

if TEST_WITH_DEV_DBG_ASAN:  # 如果在开发调试模式下使用地址安全性开启
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )  # 输出信息到标准错误流，说明开发调试模式与多进程生成存在已知问题
    sys.exit(0)  # 退出程序，返回状态码0表示成功


class PassType(Enum):  # 定义PassType枚举类
    __order__ = "FWD BWD"  # 枚举成员的顺序为FWD和BWD
    FWD = auto()  # 自动分配FWD的值
    BWD = auto()  # 自动分配BWD的值


class TestCommunication(FSDPTest):  # 定义测试类TestCommunication，继承自FSDPTest基类
    """Tests ``FullyShardedDataParallel``'s collective communication usage."""

    def _init_model(
        self,
        nested_model: bool,
        sharding_strategy: ShardingStrategy,
        device: torch.device,
    ):
        fsdp_kwargs = {"sharding_strategy": sharding_strategy}  # 创建FSDP模型的关键字参数字典
        if nested_model:  # 如果使用嵌套模型
            model = NestedWrappedModule.init(  # 初始化嵌套封装模块
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_AFTER,
                fsdp_kwargs,
            )
            fsdp_model: FSDP = FSDP(  # 使用FSDP对模型进行全分片数据并行化处理
                model,
                self.process_group,
                **fsdp_kwargs,
            ).to(device)  # 将处理后的模型移动到指定设备
        else:  # 如果不使用嵌套模型
            fsdp_model: FSDP = TransformerWithSharedParams.init(  # 使用具有共享参数的Transformer初始化模型
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                fsdp_kwargs,
            )
        return fsdp_model  # 返回处理后的FSDP模型实例

    def _run_iter(self, fsdp_model, batch, use_no_sync: bool):
        """Runs an iteration inside or outside the ``no_sync()`` context."""
        context = fsdp_model.no_sync() if use_no_sync else nullcontext()  # 根据use_no_sync选择上下文管理器
        with context:  # 使用选定的上下文管理器
            output = fsdp_model(*batch)  # 对批次进行模型计算
            loss = fsdp_model.module.get_loss(batch, output)  # 获取损失值
            loss.backward()  # 反向传播计算梯度

    def _get_ref_num_reduce_scatters(
        self,
        num_fsdp: int,
        in_no_sync: bool,
    ) -> int:
        """Returns the reference number of reduce-scatters for an iteration
        in the ``no_sync()`` context."""
        return num_fsdp if not in_no_sync else 0  # 如果不在no_sync()上下文中，返回num_fsdp，否则返回0
    def _get_ref_num_all_gathers(
        self,
        num_fsdp: int,
        sharding_strategy: Optional[ShardingStrategy],
        is_first_iter: bool,
        is_last_iter_no_sync: bool,
    ) -> int:
        """Returns the reference number of all-gathers in an iteration, summing
        over the forward and backward passes."""
        # 计算所有传播中所有聚合操作的参考数量
        return sum(
            self._get_ref_num_all_gathers_in_pass(
                num_fsdp,
                sharding_strategy,
                pass_type,
                is_first_iter,
                is_last_iter_no_sync,
            )
            for pass_type in PassType
        )

    def _get_ref_num_all_gathers_in_pass(
        self,
        num_fsdp: int,
        sharding_strategy: Optional[ShardingStrategy],
        pass_type: PassType,
        is_first_iter: bool,
        is_last_iter_no_sync: bool,
    ):
        """Returns the reference number of all-gathers for a given setting."""
        if sharding_strategy is None:
            sharding_strategy = ShardingStrategy.FULL_SHARD  # 默认为完整分片策略

        # 前向传播：
        if (
            pass_type == PassType.FWD
            and sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
            and is_last_iter_no_sync
        ):
            # 如果是最后一个迭代的前向传播且采用分片梯度操作策略，
            # 模块在 `no_sync()` 中不会完全释放参数
            num_all_gathers = 0
        elif pass_type == PassType.FWD:
            # 否则，在前向传播中所有模块都会聚合完整参数
            num_all_gathers = num_fsdp

        # 后向传播：
        elif (
            pass_type == PassType.BWD
            and sharding_strategy == ShardingStrategy.FULL_SHARD
        ):
            # 在完整分片策略中，根节点不会在前向传播结束时完全释放参数
            num_all_gathers = num_fsdp - 1
        elif (
            pass_type == PassType.BWD
            and sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
        ):
            # 在分片梯度操作策略中，模块在前向传播结束时不会完全释放参数
            num_all_gathers = 0
        else:
            # 不支持的情况：需要为特定的 pass_type、is_first_iter、is_last_iter_no_sync、sharding_strategy 添加分支
            assert 0, (
                f"Unsupported: add a branch for pass_type={pass_type} "
                f"is_first_iter={is_first_iter} "
                f"is_last_iter_no_sync={is_last_iter_no_sync} "
                f"sharding_strategy={sharding_strategy}"
            )

        # 在第一个迭代且为前向传播时，考虑执行顺序验证，每个实际的前向传播中会有额外两个聚合操作
        if is_first_iter and pass_type == PassType.FWD:
            num_all_gathers *= 3

        return num_all_gathers

    def _print_ref_num_all_gathers_in_pass(
        self,
        num_fsdp: int,
        sharding_strategy: ShardingStrategy,
        pass_type: PassType,
        is_first_iter: bool,
        is_last_iter_no_sync: bool,
    ):
        # 此函数用于打印特定传播类型中的所有聚合操作的参考数量，但没有返回值
        pass
    ):
        """Helper method for printing the number of all-gathers for a specific
        setting. This may be helpful since the branching is complex."""
        # 如果当前进程的 rank 不是 0，直接返回，只在一个进程上打印
        if self.rank != 0:
            return
        # 获取在指定设置中所有收集操作的数量，这对于复杂的分支结构可能很有帮助
        num_all_gathers = self._get_ref_num_all_gathers_in_pass(
            num_fsdp,
            sharding_strategy,
            pass_type,
            is_first_iter,
            is_last_iter_no_sync,
        )
        # 打印包含多个信息的字符串，显示当前测试的一些参数和数量的信息
        print(
            f"Pass: {pass_type}\n"
            f"Is First Iteration: {is_first_iter}\n"
            f"Sharding Strategy: {sharding_strategy}\n"
            f"Last iteration in `no_sync()`: {is_last_iter_no_sync}\n"
            f"Number of all-gathers: {num_all_gathers}"
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("nested_model", [False, True])
    @parametrize("use_no_sync", [False, True])
    @parametrize("sharding_strategy", [ShardingStrategy.SHARD_GRAD_OP, None])
    # 测试通信功能，用于不同参数组合的场景
    def test_communication(
        self,
        nested_model: bool,
        use_no_sync: bool,
        sharding_strategy: Optional[ShardingStrategy],
# 定义一个名为 TestExplicitUnshard 的类，继承自 FSDPTest 类
class TestExplicitUnshard(FSDPTest):
    
    # 定义一个属性方法 world_size，返回当前机器上 CUDA 设备数量与 2 的较小值
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    # 使用装饰器 @skip_if_lt_x_gpu(2)，表示如果 CUDA 设备数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    
    # 使用 parametrize 装饰器，为 use_orig_params 参数生成两个测试实例：False 和 True
    @parametrize("use_orig_params", [False, True])
    
# 实例化参数化测试 TestCommunication，这将为 TestCommunication 类中带有 @parametrize 装饰器的测试方法生成两次测试
instantiate_parametrized_tests(TestCommunication)

# 实例化参数化测试 TestExplicitUnshard，这将为 TestExplicitUnshard 类中带有 @parametrize 装饰器的测试方法生成两次测试
instantiate_parametrized_tests(TestExplicitUnshard)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```