# `.\pytorch\test\distributed\test_launcher.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库
import os
import sys
from contextlib import closing

# 导入分布式相关的库
import torch.distributed as dist
import torch.distributed.launch as launch
from torch.distributed.elastic.utils import get_socket_with_port

# 检查是否支持分布式，如果不支持则退出测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试相关的工具函数和类
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

# 定义一个辅助函数，用于获取脚本的绝对路径
def path(script):
    return os.path.join(os.path.dirname(__file__), script)

# 如果测试要求使用 ASAN 调试，则跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

# 定义一个测试类
class TestDistributedLaunch(TestCase):
    # 测试启动用户脚本的方法
    def test_launch_user_script(self):
        nnodes = 1
        nproc_per_node = 4
        world_size = nnodes * nproc_per_node
        
        # 获取一个可用的端口
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        
        # 定义启动参数
        args = [
            f"--nnodes={nnodes}",
            f"--nproc-per-node={nproc_per_node}",
            "--monitor-interval=1",
            "--start-method=spawn",
            "--master-addr=localhost",
            f"--master-port={master_port}",
            "--node-rank=0",
            "--use-env",
            path("bin/test_script.py"),  # 用户脚本的路径
        ]
        
        # 调用分布式启动的主函数
        launch.main(args)

# 如果作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```