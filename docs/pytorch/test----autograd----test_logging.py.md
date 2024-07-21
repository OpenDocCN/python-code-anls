# `.\pytorch\test\autograd\test_logging.py`

```py
# Owner(s): ["module: autograd"]
# 导入 logging 模块，用于记录日志
import logging

# 导入 torch 库
import torch
# 导入 LoggingTestCase 和 make_logging_test 函数，用于测试日志记录
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class TestAutogradLogging(LoggingTestCase):
    # 使用 make_logging_test 装饰器，设置 autograd 模块的日志级别为 DEBUG
    @make_logging_test(autograd=logging.DEBUG)
    def test_logging(self, records):
        # 创建一个形状为 (10,) 的张量 a，需要计算梯度
        a = torch.rand(10, requires_grad=True)
        # 对张量 a 进行乘以 2、除以 3、求和的操作
        b = a.mul(2).div(3).sum()
        # 克隆张量 b
        c = b.clone()
        # 执行自动求导，计算梯度
        torch.autograd.backward((b, c))

        # 断言记录的日志条数为 5 条
        self.assertEqual(len(records), 5)
        # 预期的日志消息列表
        expected = [
            "CloneBackward0",
            "SumBackward0",
            "DivBackward0",
            "MulBackward0",
            "AccumulateGrad",
        ]

        # 遍历记录的日志条目
        for i, record in enumerate(records):
            # 断言每条记录的消息在预期的消息列表中
            self.assertIn(expected[i], record.getMessage())


if __name__ == "__main__":
    # 导入 run_tests 函数，运行测试用例
    from torch._dynamo.test_case import run_tests

    run_tests()
```