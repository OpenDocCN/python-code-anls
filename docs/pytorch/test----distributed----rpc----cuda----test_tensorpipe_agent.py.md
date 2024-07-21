# `.\pytorch\test\distributed\rpc\cuda\test_tensorpipe_agent.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# 导入 sys 模块
import sys

# 导入 torch 分布式模块
import torch.distributed as dist

# 如果分布式不可用，输出消息并退出程序
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入 torch 库
import torch

# 导入测试工具函数
from torch.testing._internal.common_utils import run_tests
# 导入 TensorPipeRpcAgentTestFixture 类
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
# 导入生成测试函数和测试用例
from torch.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_CUDA_TESTS,
    TENSORPIPE_CUDA_TESTS,
)

# 如果 CUDA 可用，则设置内存分配策略为不可扩展
if torch.cuda.is_available():
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

# 动态更新全局命名空间，生成 TensorPipe 相关的测试
globals().update(
    generate_tests(
        "TensorPipe",
        TensorPipeRpcAgentTestFixture,
        GENERIC_CUDA_TESTS + TENSORPIPE_CUDA_TESTS,
        __name__,
    )
)

# 如果当前脚本为主程序，运行测试
if __name__ == "__main__":
    run_tests()
```