# `.\pytorch\test\distributed\test_distributed_spawn.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入标准库模块
import os
import sys

# 导入PyTorch及其分布式组件
import torch
import torch.distributed as dist

# 设置PyTorch后端的CUDA矩阵乘法选项
torch.backends.cuda.matmul.allow_tf32 = False

# 检查是否支持分布式运行环境，如果不支持则退出并输出错误信息
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试相关的内部工具和类
from torch.testing._internal.common_utils import (
    NO_MULTIPROCESSING_SPAWN,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.testing._internal.distributed.distributed_test import (
    DistributedTest,
    TestDistBackend,
)

# 如果测试使用了开发者调试ASAN，跳过测试并输出相应信息
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 如果不支持多进程SPAWN模式，跳过测试并输出相应信息
if NO_MULTIPROCESSING_SPAWN:
    print("Spawn not available, skipping tests.", file=sys.stderr)
    sys.exit(0)

# 允许的分布式后端选项
_allowed_backends = ("gloo", "nccl", "ucc")

# 检查是否设置了必要的环境变量，如果没有则抛出运行时错误并提供详细的环境变量要求说明
if (
    "BACKEND" not in os.environ
    or "WORLD_SIZE" not in os.environ
    or "TEMP_DIR" not in os.environ
):
    raise RuntimeError(
        "Missing expected env vars for `test_distributed_spawn.py`.  Please ensure to specify the following:\n"
        f"'BACKEND' = one of {_allowed_backends}\n"
        f"'WORLD_SIZE' = int >= 2\n"
        "'TEMP_DIR' specifying a directory containing a barrier file named 'barrier'.\n\n"
        f"e.g.\ntouch /tmp/barrier && TEMP_DIR=/tmp BACKEND='nccl' WORLD_SIZE=2 python {__file__}",
    )

# 从环境变量中获取后端选项
BACKEND = os.environ["BACKEND"]

# 如果后端选项在允许的后端列表中
if BACKEND in _allowed_backends:

    # 定义一个测试类，继承自TestDistBackend和_DistTestBase
    class TestDistBackendWithSpawn(TestDistBackend, DistributedTest._DistTestBase):
        def setUp(self):
            super().setUp()
            # 启动多进程SPAWN模式
            self._spawn_processes()
            # 设置CUDNN相关选项
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False).__enter__()

# 如果后端选项不在允许的后端列表中，则输出无效后端信息
else:
    print(f"Invalid backend {BACKEND}. Tests will not be run!")

# 如果作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```