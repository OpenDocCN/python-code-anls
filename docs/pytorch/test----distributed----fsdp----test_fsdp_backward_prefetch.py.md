# `.\pytorch\test\distributed\fsdp\test_fsdp_backward_prefetch.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入系统库
import sys
# 导入类型提示相关的模块
from typing import List
# 导入单元测试中的模拟模块
from unittest.mock import patch

# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的神经网络模块
import torch.nn as nn
# 导入 PyTorch 分布式模块
from torch import distributed as dist
# 导入 PyTorch 的分布式自动混合精度模块
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP
# 导入 PyTorch 分布式 FSDP 内部通用工具函数
from torch.distributed.fsdp._common_utils import _get_handle_fqns_from_root
# 导入 PyTorch 分布式 FSDP 内部平坦参数处理模块
from torch.distributed.fsdp._flat_param import HandleTrainingState
# 导入 PyTorch 分布式 FSDP 运行时工具函数
from torch.distributed.fsdp._runtime_utils import (
    _get_handle_to_prefetch,
    _get_training_state,
)
# 导入 PyTorch 分布式 FSDP 模块的模块封装策略
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
# 导入 PyTorch 的分布式单元测试工具
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
# 导入 PyTorch FSDP 的通用单元测试
from torch.testing._internal.common_fsdp import FSDPTest
# 导入 PyTorch 的通用测试工具
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

# 设置迭代次数常量
NUM_ITERS = 2

# 解码器参数的全限定名称列表
DECODER_PARAM_FQNS = [
    "decoder.layers.{index}.self_attn.in_proj_weight",
    "decoder.layers.{index}.self_attn.in_proj_bias",
    "decoder.layers.{index}.self_attn.out_proj.weight",
    "decoder.layers.{index}.self_attn.out_proj.bias",
    "decoder.layers.{index}.multihead_attn.in_proj_weight",
    "decoder.layers.{index}.multihead_attn.in_proj_bias",
    "decoder.layers.{index}.multihead_attn.out_proj.weight",
    "decoder.layers.{index}.multihead_attn.out_proj.bias",
    "decoder.layers.{index}.linear1.weight",
    "decoder.layers.{index}.linear1.bias",
    "decoder.layers.{index}.linear2.weight",
    "decoder.layers.{index}.linear2.bias",
    "decoder.layers.{index}.norm1.weight",
    "decoder.layers.{index}.norm1.bias",
    "decoder.layers.{index}.norm2.weight",
    "decoder.layers.{index}.norm2.bias",
    "decoder.layers.{index}.norm3.weight",
    "decoder.layers.{index}.norm3.bias",
]

# 编码器参数的全限定名称列表
ENCODER_PARAM_FQNS = [
    "encoder.layers.{index}.self_attn.in_proj_weight",
    "encoder.layers.{index}.self_attn.in_proj_bias",
    "encoder.layers.{index}.self_attn.out_proj.weight",
    "encoder.layers.{index}.self_attn.out_proj.bias",
    "encoder.layers.{index}.linear1.weight",
    "encoder.layers.{index}.linear1.bias",
    "encoder.layers.{index}.linear2.weight",
    "encoder.layers.{index}.linear2.bias",
    "encoder.layers.{index}.norm1.weight",
    "encoder.layers.{index}.norm1.bias",
    "encoder.layers.{index}.norm2.weight",
    "encoder.layers.{index}.norm2.bias",
]

# 预取前的总数常量
TOTAL_NUM_PREFETCH_FOR_PRE = 12
# 预取后的总数常量
TOTAL_NUM_PREFETCH_FOR_POST = 11
# 编码器预取前的起始索引
ENCODER_BEGIN_INDEX_FOR_PRE = 6
# 编码器预取后的起始索引
ENCODER_BEGIN_INDEX_FOR_POST = 5
# 编码器预取数目
ENCODER_PREFETCH_NUM = 5

# 如果分布式不可用，则输出信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果设置了测试调试 ASAN 标志，则输出信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义测试类 TestBackwardPrefetch，继承自 FSDPTest 类
class TestBackwardPrefetch(FSDPTest):
    # 定义 world_size 属性，返回值为 2
    @property
    def world_size(self):
        return 2

    # 根据 GPU 数目跳过测试装饰器，至少需要两个 GPU
    @skip_if_lt_x_gpu(2)
    def test_backward_prefetch(self):
        # 定义一个测试方法，用于测试反向预取功能
        # 使用 run_subtests 方法运行子测试，以重用进程组以缩短测试时间
        self.run_subtests(
            {
                "backward_prefetch": [
                    None,
                    BackwardPrefetch.BACKWARD_PRE,
                    BackwardPrefetch.BACKWARD_POST,
                ],
            },
            self._test_backward_prefetch,  # 将 _test_backward_prefetch 方法作为回调函数传递给 run_subtests
        )

    def _test_backward_prefetch(self, backward_prefetch: BackwardPrefetch):
        # 定义一个私有方法，用于测试反向预取功能，接收一个 backward_prefetch 参数
        self._dist_train(backward_prefetch)  # 调用 _dist_train 方法，传递 backward_prefetch 参数
# 如果当前模块被直接执行（而不是被导入到另一个模块中执行），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试代码或测试套件
    run_tests()
```