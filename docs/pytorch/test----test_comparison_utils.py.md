# `.\pytorch\test\test_comparison_utils.py`

```
#!/usr/bin/env python3
# Owner(s): ["module: internals"]

# 导入PyTorch库
import torch
# 导入测试工具类和测试用例基类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义测试类TestComparisonUtils，继承自TestCase
class TestComparisonUtils(TestCase):
    
    # 测试函数：验证在不进行断言时所有元数据是否相等
    def test_all_equal_no_assert(self):
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, [1], [1], torch.float)

    # 测试函数：验证在不进行断言时所有元数据是否为None
    def test_all_equal_no_assert_nones(self):
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, None, None, None)

    # 测试函数：验证断言数据类型时是否引发运行时错误
    def test_assert_dtype(self):
        t = torch.tensor([0.5])

        # 使用断言验证是否引发RuntimeError异常
        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, None, torch.int32)

    # 测试函数：验证断言步长时是否引发运行时错误
    def test_assert_strides(self):
        t = torch.tensor([0.5])

        # 使用断言验证是否引发RuntimeError异常
        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, [3], torch.float)

    # 测试函数：验证断言大小时是否引发运行时错误
    def test_assert_sizes(self):
        t = torch.tensor([0.5])

        # 使用断言验证是否引发RuntimeError异常
        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, [3], [1], torch.float)

# 如果作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```