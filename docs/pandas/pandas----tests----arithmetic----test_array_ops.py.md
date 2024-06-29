# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_array_ops.py`

```
# 导入标准库中的 operator 模块，用于操作符函数
import operator

# 导入第三方库 numpy，并将其重命名为 np
import numpy as np

# 导入第三方库 pytest，用于编写和运行测试
import pytest

# 导入 pandas 内部的测试模块 pandas._testing，并重命名为 tm
import pandas._testing as tm

# 从 pandas 核心操作数组的模块中导入指定函数
from pandas.core.ops.array_ops import (
    comparison_op,     # 导入比较运算函数
    na_logical_op,     # 导入处理缺失值逻辑运算函数
)


# 定义测试函数 test_na_logical_op_2d
def test_na_logical_op_2d():
    # 创建一个 4x2 的二维数组 left，元素为 0 到 7
    left = np.arange(8).reshape(4, 2)
    
    # 将 left 数组转换为对象类型 ndarray，并赋值给 right
    right = left.astype(object)
    
    # 将 right 数组的第一个元素设置为 NaN（缺失值）
    right[0, 0] = np.nan

    # 使用 pytest 检查以下操作是否引发 TypeError 异常，且异常消息匹配特定字符串
    with pytest.raises(TypeError, match="unsupported operand type"):
        # 调用 operator.or_ 函数尝试执行 left 和 right 的按位或运算
        operator.or_(left, right)

    # 调用 na_logical_op 函数处理 left 和 right 的逻辑运算，使用 operator.or_ 函数
    result = na_logical_op(left, right, operator.or_)
    
    # 将预期结果设置为 right 数组
    expected = right
    
    # 使用 pandas 提供的 assert_numpy_array_equal 函数比较 result 和 expected 数组
    tm.assert_numpy_array_equal(result, expected)


# 定义测试函数 test_object_comparison_2d
def test_object_comparison_2d():
    # 创建一个 3x3 的二维数组 left，元素为 0 到 8，并将其类型转换为对象类型 ndarray
    left = np.arange(9).reshape(3, 3).astype(object)
    
    # 将 left 数组的转置作为 right 数组
    right = left.T

    # 调用 comparison_op 函数进行 left 和 right 的比较运算，使用 operator.eq 函数
    result = comparison_op(left, right, operator.eq)
    
    # 生成一个预期结果数组，对角线元素为 True，其余元素为 False
    expected = np.eye(3).astype(bool)
    
    # 使用 pandas 提供的 assert_numpy_array_equal 函数比较 result 和 expected 数组
    tm.assert_numpy_array_equal(result, expected)

    # 设置 right 数组的可写属性为 False，以确保 Cython 不会因为不可写参数而引发异常
    right.flags.writeable = False
    
    # 再次调用 comparison_op 函数进行 left 和 right 的比较运算，使用 operator.ne 函数
    result = comparison_op(left, right, operator.ne)
    
    # 使用 pandas 提供的 assert_numpy_array_equal 函数比较 result 和 ~expected 数组
    tm.assert_numpy_array_equal(result, ~expected)
```