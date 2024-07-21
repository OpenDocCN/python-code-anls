# `.\pytorch\test\test_openmp.py`

```
# Owner(s): ["module: unknown"]

# 导入必要的模块和类
import collections
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ASAN, TestCase

# 尝试导入 psutil 模块，标记是否成功导入
try:
    import psutil
    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None

# 设置默认设备为 CPU
device = torch.device("cpu")

# 定义一个简单的神经网络类，包含最大池化层
class Network(torch.nn.Module):
    maxp1 = torch.nn.MaxPool2d(1, 1)

    def forward(self, x):
        return self.maxp1(x)

# 测试类，跳过测试条件为没有安装 psutil 或者使用 ASAN
@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(TEST_WITH_ASAN, "Cannot test with ASAN")
class TestOpenMP_ParallelFor(TestCase):
    batch = 20
    channels = 1
    side_dim = 80
    x = torch.randn([batch, channels, side_dim, side_dim], device=device)
    model = Network()

    # 测试方法：评估内存使用是否稳定
    def func(self, runs):
        p = psutil.Process()
        # 创建一个长度为 5 的双向队列，用于存储最近 5 次运行的内存使用情况
        last_rss = collections.deque(maxlen=5)
        for n in range(10):
            for i in range(runs):
                self.model(self.x)
            last_rss.append(p.memory_info().rss)
        return last_rss

    # 测试方法：检查内存使用是否递增
    def func_rss(self, runs):
        last_rss = list(self.func(runs))
        # 检查队列中的内存使用是否递增
        is_increasing = True
        for idx in range(len(last_rss)):
            if idx == 0:
                continue
            is_increasing = is_increasing and (last_rss[idx] > last_rss[idx - 1])
        self.assertTrue(
            not is_increasing, msg=f"memory usage is increasing, {str(last_rss)}"
        )

    # 测试方法：使用单线程执行 func_rss 测试
    def test_one_thread(self):
        """Make sure there is no memory leak with one thread: issue gh-32284"""
        torch.set_num_threads(1)
        self.func_rss(300)

    # 测试方法：使用多线程执行 func_rss 测试
    def test_n_threads(self):
        """Make sure there is no memory leak with many threads"""
        ncores = min(5, psutil.cpu_count(logical=False))
        torch.set_num_threads(ncores)
        self.func_rss(300)

# 如果作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```