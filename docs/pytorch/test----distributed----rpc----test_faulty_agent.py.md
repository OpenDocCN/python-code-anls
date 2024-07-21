# `.\pytorch\test\distributed\rpc\test_faulty_agent.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# 导入系统相关模块
import sys

# 导入PyTorch相关模块
import torch
import torch.distributed as dist

# 如果分布式环境不可用，则在标准错误输出提示并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试相关模块和函数
from torch.testing._internal.common_utils import IS_CI, run_tests
from torch.testing._internal.distributed.rpc.faulty_rpc_agent_test_fixture import (
    FaultyRpcAgentTestFixture,
)
from torch.testing._internal.distributed.rpc_utils import (
    FAULTY_AGENT_TESTS,
    generate_tests,
)


# 在CircleCI上，这些测试已在CPU作业上运行过了，因此在GPU作业上不再运行它们，以节省资源
if not (IS_CI and torch.cuda.is_available()):
    # 动态生成与故障代理相关的测试并更新到全局变量中
    globals().update(
        generate_tests(
            "Faulty",
            FaultyRpcAgentTestFixture,
            FAULTY_AGENT_TESTS,
            __name__,
        )
    )


# 如果脚本作为主程序运行，则执行测试运行函数
if __name__ == "__main__":
    run_tests()
```