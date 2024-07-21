# `.\pytorch\test\distributed\algorithms\quantization\test_quantization.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库
import os
import sys

# 导入 PyTorch 相关库
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms._quantization.quantization as quant
from torch.distributed.algorithms._quantization.quantization import DQuantType
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.testing._internal.common_utils import (
    NO_MULTIPROCESSING_SPAWN,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)

# 禁用 CUDA 中 TF32 计算
torch.backends.cuda.matmul.allow_tf32 = False

# 如果分布式不可用，则跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


# 创建一个填充了指定值的张量的辅助函数
def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, dtype=dtype).fill_(value).cuda(device_id)


# 如果使用开发调试工具 ASAN，则跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 如果不支持多进程 spawn，则跳过测试
if NO_MULTIPROCESSING_SPAWN:
    print("Spawn not available, skipping tests.", file=sys.stderr)
    sys.exit(0)

# 从环境变量获取后端信息
BACKEND = os.environ["BACKEND"]
# 如果后端是 "gloo" 或 "nccl"，则执行下面的测试
if BACKEND == "gloo" or BACKEND == "nccl":

# 如果作为主程序运行，则执行测试运行函数
if __name__ == "__main__":
    run_tests()
```