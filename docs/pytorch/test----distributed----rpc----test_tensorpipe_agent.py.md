# `.\pytorch\test\distributed\rpc\test_tensorpipe_agent.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# 导入系统模块
import sys

# 导入PyTorch相关模块
import torch
import torch.distributed as dist

# 如果分布式不可用，输出信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试相关模块和函数
from torch.testing._internal.common_utils import IS_CI, run_tests
from torch.testing._internal.distributed.rpc.tensorpipe_rpc_agent_test_fixture import (
    TensorPipeRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    generate_tests,
    GENERIC_TESTS,
    TENSORPIPE_TESTS,
)

# 在CircleCI上，如果是GPU作业，则不运行这些测试以节省资源
if not (IS_CI and torch.cuda.is_available()):
    # 动态生成TensorPipe相关测试，并更新全局变量
    globals().update(
        generate_tests(
            "TensorPipe",
            TensorPipeRpcAgentTestFixture,
            GENERIC_TESTS + TENSORPIPE_TESTS,
            __name__,
        )
    )

# 如果作为主程序执行，则运行所有测试
if __name__ == "__main__":
    run_tests()
```