# `.\pytorch\test\test_itt.py`

```py
# Owner(s): ["module: intel"]

# 导入 PyTorch 库
import torch
# 导入单元测试模块
import unittest
# 从 torch.testing._internal.common_utils 中导入 TestCase, run_tests, load_tests 函数
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests 函数用于在 sandcastle 上自动过滤测试以进行分片。这行代码用于消除 flake 警告
load_tests = load_tests

# 如果 ITT 不可用，则跳过测试
@unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
class TestItt(TestCase):
    def test_itt(self):
        # 确保能看到符号
        # 开始一个 ITT 范围名为 "foo"
        torch.profiler.itt.range_push("foo")
        # 在 ITT 中打一个标记名为 "bar"
        torch.profiler.itt.mark("bar")
        # 结束当前 ITT 范围
        torch.profiler.itt.range_pop()

# 如果运行作为主程序，则执行所有测试
if __name__ == '__main__':
    run_tests()
```