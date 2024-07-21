# `.\pytorch\test\distributed\fsdp\test_fsdp_overlap.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和类
import sys  # 系统操作相关
import time  # 时间相关操作
from statistics import mean  # 统计模块中的均值计算函数
from unittest.mock import patch  # 单元测试模块中的模拟功能

import torch  # PyTorch 主模块
import torch.nn as nn  # PyTorch 神经网络模块
from torch import distributed as dist  # PyTorch 分布式模块
from torch.cuda import Event  # PyTorch CUDA 事件对象
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # PyTorch FSDP 分布式模块
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 测试中的 GPU 数量检查
from torch.testing._internal.common_fsdp import FSDPTest  # 测试中的 FSDP 测试基类
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,  # 获取每毫秒 CPU 周期数
    run_tests,  # 运行测试函数
    TEST_WITH_DEV_DBG_ASAN,  # 是否使用 dev-asan 测试标志
)

# 如果分布式不可用，则跳过测试并输出信息
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果使用 dev-asan 测试，因为 torch + multiprocessing spawn 存在已知问题，跳过测试并输出信息
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class Layer(nn.Module):
    def __init__(self, compute_cycles, has_params: bool):
        super().__init__()
        self.sleep_cycles = compute_cycles  # 记录需要模拟的计算时长
        self.optional_param = None
        if has_params:
            self.optional_param = nn.Parameter(torch.rand(1))  # 如果有参数，则初始化一个随机参数

    def forward(self, x):
        # 创建两个 CUDA 事件对象，用于记录计时信息
        self.e1 = Event(enable_timing=True)
        self.e2 = Event(enable_timing=True)

        # 记录假的前向计算时间
        self.e1.record()
        if self.sleep_cycles > 0:
            torch.cuda._sleep(self.sleep_cycles)  # 模拟计算延迟
        if self.optional_param is not None:
            x = x + self.optional_param  # 强制参数成为计算图的一部分
        self.e2.record()
        return x

    def get_time(self):
        # 返回记录的持续时间
        return self.e1.elapsed_time(self.e2)


def _create_model(compute_cycles, has_params: bool):
    # 创建 FSDP 模型，包含多层 FSDP 封装的 Layer 模块
    # 使用 `limit_all_gathers=False` 以确保测试中 CPU 运行超前于 GPU
    model = FSDP(
        nn.Sequential(
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
            FSDP(Layer(compute_cycles, has_params), limit_all_gathers=False),
        ),
        limit_all_gathers=False,
    ).cuda()
    return model


class Min10:
    def __init__(self):
        self.data = []  # 初始化数据列表

    def add(self, new_data):
        if len(self.data) < 10:
            self.data.append(new_data)  # 如果数据列表长度小于 10，则添加新数据
        else:
            self.data = sorted(self.data)  # 如果长度已达到 10，则排序数据列表
            if new_data < self.data[-1]:
                self.data[-1] = new_data  # 如果新数据小于最大值，则替换最大值

    def avg(self):
        return mean(self.data)  # 返回数据列表的平均值


class TestForwardOverlapWorldSizeOne(FSDPTest):
    @property
    def world_size(self):
        return 1  # 设置世界大小为 1

    @skip_if_lt_x_gpu(2)
    def test_forward_overlap(self):
        self._dist_train()  # 执行分布式训练测试


class TestForwardOverlapWorldSizeTwo(TestForwardOverlapWorldSizeOne):
    @property
    def world_size(self):
        return 2  # 设置世界大小为 2


if __name__ == "__main__":
    run_tests()  # 运行所有测试
```