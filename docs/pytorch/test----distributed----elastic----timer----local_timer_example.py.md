# `.\pytorch\test\distributed\elastic\timer\local_timer_example.py`

```py
# 指定 Python 解释器的位置，使脚本可执行
#!/usr/bin/env python3
# 所有权：["oncall: r2p"]

# 版权声明，保留所有权利
# 此源代码采用 BSD 风格许可证，许可证文件位于源树根目录下的 LICENSE 文件中
import logging  # 导入日志模块
import multiprocessing as mp  # 导入多进程模块
import signal  # 导入信号处理模块
import time  # 导入时间模块

import torch.distributed.elastic.timer as timer  # 导入 Torch 分布式弹性训练计时器模块
import torch.multiprocessing as torch_mp  # 导入 Torch 多进程模块
from torch.testing._internal.common_utils import (
    IS_MACOS,  # 导入系统平台判断变量：是否为 macOS
    IS_WINDOWS,  # 导入系统平台判断变量：是否为 Windows
    run_tests,  # 导入测试运行函数
    skip_but_pass_in_sandcastle_if,  # 导入测试跳过函数（在沙堡环境下会跳过但是会通过）
    TEST_WITH_DEV_DBG_ASAN,  # 导入测试调试标志
    TestCase,  # 导入测试用例基类
)

# 配置日志记录器，设置日志级别为 INFO，格式为 [级别] 时间 模块名: 消息
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(asctime)s %(module)s: %(message)s"
)


def _happy_function(rank, mp_queue):
    # 配置计时器使用本地计时器客户端，传入多进程队列
    timer.configure(timer.LocalTimerClient(mp_queue))
    # 设置超时时间为 1 秒，执行时睡眠 0.5 秒
    with timer.expires(after=1):
        time.sleep(0.5)


def _stuck_function(rank, mp_queue):
    # 配置计时器使用本地计时器客户端，传入多进程队列
    timer.configure(timer.LocalTimerClient(mp_queue))
    # 设置超时时间为 1 秒，执行时睡眠 5 秒
    with timer.expires(after=1):
        time.sleep(5)


# 如果不在 Windows 或 macOS 平台下执行以下代码块
if not (IS_WINDOWS or IS_MACOS):
    # 如果当前脚本作为主程序运行，则执行测试
    if __name__ == "__main__":
        run_tests()
```