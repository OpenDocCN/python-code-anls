# `D:\src\scipysrc\pandas\pandas\tests\io\sas\test_byteswap.py`

```
# 导入必要的模块和函数
from hypothesis import (
    assume,     # 假设函数，用于设置前提条件
    example,    # 示例装饰器，指定测试函数的例子
    given,      # 给定装饰器，指定测试函数的输入生成策略
    strategies as st,   # 策略模块的别名
)
import numpy as np   # 导入 NumPy 库，用于数值计算
import pytest    # 导入 PyTest 测试框架

from pandas._libs.byteswap import (
    read_double_with_byteswap,    # 字节交换读取双精度浮点数
    read_float_with_byteswap,     # 字节交换读取单精度浮点数
    read_uint16_with_byteswap,    # 字节交换读取无符号 16 位整数
    read_uint32_with_byteswap,    # 字节交换读取无符号 32 位整数
    read_uint64_with_byteswap,    # 字节交换读取无符号 64 位整数
)

import pandas._testing as tm    # 导入 Pandas 测试模块，简称为 tm


@given(read_offset=st.integers(0, 11), number=st.integers(min_value=0))   # 给定整数生成策略的假设
@example(number=2**16, read_offset=0)    # 测试函数的示例，指定特定参数值
@example(number=2**32, read_offset=0)    # 测试函数的示例，指定特定参数值
@example(number=2**64, read_offset=0)    # 测试函数的示例，指定特定参数值
@pytest.mark.parametrize("int_type", [np.uint16, np.uint32, np.uint64])    # 参数化测试整数类型
@pytest.mark.parametrize("should_byteswap", [True, False])    # 参数化测试字节交换标志
def test_int_byteswap(read_offset, number, int_type, should_byteswap):
    assume(number < 2 ** (8 * int_type(0).itemsize))   # 假设整数小于特定类型的最大表示范围
    _test(number, int_type, read_offset, should_byteswap)   # 调用测试函数


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")    # 忽略运行时警告
@given(read_offset=st.integers(0, 11), number=st.floats())    # 浮点数生成策略的假设
@pytest.mark.parametrize("float_type", [np.float32, np.float64])    # 参数化测试浮点数类型
@pytest.mark.parametrize("should_byteswap", [True, False])    # 参数化测试字节交换标志
def test_float_byteswap(read_offset, number, float_type, should_byteswap):
    _test(number, float_type, read_offset, should_byteswap)    # 调用测试函数


def _test(number, number_type, read_offset, should_byteswap):
    number = number_type(number)    # 将输入数字转换为指定类型
    data = np.random.default_rng(2).integers(0, 256, size=20, dtype="uint8")    # 生成随机数据数组
    data[read_offset : read_offset + number.itemsize] = number[None].view("uint8")    # 将数字插入到随机数据中
    swap_func = {    # 根据数据类型选择相应的字节交换函数
        np.float32: read_float_with_byteswap,
        np.float64: read_double_with_byteswap,
        np.uint16: read_uint16_with_byteswap,
        np.uint32: read_uint32_with_byteswap,
        np.uint64: read_uint64_with_byteswap,
    }[type(number)]
    output_number = number_type(swap_func(bytes(data), read_offset, should_byteswap))    # 调用选择的字节交换函数
    if should_byteswap:
        tm.assert_equal(output_number, number.byteswap())    # 如果需要字节交换，则比较交换后的结果
    else:
        tm.assert_equal(output_number, number)    # 否则比较原始结果
```