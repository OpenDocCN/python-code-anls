# `.\pytorch\test\distributed\pipelining\test_composability.py`

```py
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# 导入必要的库
import copy  # 导入 copy 模块
import os    # 导入 os 模块
import sys   # 导入 sys 模块
import tempfile  # 导入 tempfile 模块

# 导入模型注册模块
from model_registry import MLPModule

# 导入 PyTorch 库
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import (
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleGPipe,
)
from torch.nn.parallel import DistributedDataParallel as DDP

# 导入测试相关的辅助模块
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)

# 定义一个测试类，继承自 MultiProcContinousTest
class ComposabilityTest(MultiProcContinousTest):

    @classmethod
    def backend_str(cls) -> str:
        # 返回当前测试所使用的后端类型字符串
        return "nccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.cuda.device_count()  # 计算设备 ID
        cls.device = torch.device(f"cuda:{dev_id}")    # 根据设备 ID 设置 CUDA 设备
        # TODO: investigate why this is needed to prevent multiple NCCL ranks from hitting the same device
        torch.cuda.set_device(cls.device)  # 设置当前 CUDA 设备

    @requires_nccl()  # 需要 NCCL 支持的装饰器
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize("dp_type", ["DDP", "FSDP"])  # 参数化测试类型
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])  # 参数化调度类
    def instantiate_parametrized_tests(cls):  # 实例化参数化测试方法

# 如果当前脚本是主程序
if __name__ == "__main__":
    # 检查是否有足够的 GPU 和 NCCL 可用
    if not (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.device_count() >= 4
    ):
        # 输出错误信息并退出
        print(
            "Composability test requires at least 4 GPUs, but not enough found, skipping",
            file=sys.stderr,
        )
        sys.exit(0)

    rank = int(os.getenv("RANK", -1))       # 获取当前进程的排名
    world_size = int(os.getenv("WORLD_SIZE", 4))  # 获取世界大小

    if rank != -1:
        # 如果是使用 torchrun 或其他多进程启动器启动的，直接运行测试
        ComposabilityTest.run_rank(rank, world_size)
    else:
        # 如果是单进程启动，生成子进程来运行测试
        # 还需要一个用于 `init_process_group` 目的的会合文件
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name  # 创建临时文件
        torch.multiprocessing.spawn(
            ComposabilityTest.run_rank,  # 运行测试方法
            nprocs=world_size,           # 进程数
            args=(world_size, rdvz_file),  # 参数
        )
```