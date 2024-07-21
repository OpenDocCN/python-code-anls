# `.\pytorch\test\jit\test_decorator.py`

```py
# Owner(s): ["oncall: jit"]
# flake8: noqa

# 导入系统、单元测试、枚举类型和类型提示模块
import sys
import unittest
from enum import Enum
from typing import List, Optional

# 导入 PyTorch 库
import torch

# 导入自定义函数 my_function_a 和测试工具类 JitTestCase
from jit.myfunction_a import my_function_a
from torch.testing._internal.jit_utils import JitTestCase

# 定义一个继承自 JitTestCase 的测试类 TestDecorator
class TestDecorator(JitTestCase):
    
    # 定义测试装饰器的测试方法 test_decorator
    def test_decorator(self):
        # 注意: JitTestCase.checkScript() 不支持装饰器
        # self.checkScript(my_function_a, (1.0,))
        # 错误:
        #   RuntimeError: expected def but found '@' here:
        #   @my_decorator
        #   ~ <--- HERE
        #   def my_function_a(x: float) -> float:
        # 使用简单的 torch.jit.script() 进行测试替代
        fn = my_function_a
        fx = torch.jit.script(fn)
        # 断言 my_function_a 的执行结果与 torch.jit.script 后的结果相等
        self.assertEqual(fn(1.0), fx(1.0))
```