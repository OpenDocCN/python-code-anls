# `D:\src\scipysrc\pandas\pandas\tests\extension\test_extension.py`

```
"""
Tests for behavior if an author does *not* implement EA methods.
"""

# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

# 导入 pandas 扩展数组相关的类
from pandas.core.arrays import ExtensionArray


# 创建自定义的扩展数组类 MyEA
class MyEA(ExtensionArray):
    def __init__(self, values) -> None:
        self._values = values  # 初始化方法，接收并存储传入的值


# 定义 pytest 的 fixture，返回一个 MyEA 实例
@pytest.fixture
def data():
    arr = np.arange(10)  # 创建一个 NumPy 数组
    return MyEA(arr)  # 返回 MyEA 类的实例化对象


# 定义测试类 TestExtensionArray
class TestExtensionArray:
    # 定义测试方法 test_errors，接收 data 和 all_arithmetic_operators 参数
    def test_errors(self, data, all_arithmetic_operators):
        # 获取测试中使用的运算符名称
        op_name = all_arithmetic_operators
        # 使用 pytest 的断言检查是否会引发 AttributeError 异常
        with pytest.raises(AttributeError):
            getattr(data, op_name)  # 尝试调用 data 对象上的特定运算符方法
```