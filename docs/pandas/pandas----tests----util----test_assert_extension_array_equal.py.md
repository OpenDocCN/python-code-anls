# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_extension_array_equal.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    Timestamp,  # 时间戳相关功能
    array,  # 数组处理相关功能
)
import pandas._testing as tm  # 导入 pandas 内部测试模块作为 tm 别名
from pandas.core.arrays.sparse import SparseArray  # 从 pandas 稀疏数组模块导入 SparseArray 类


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    "kwargs",  # 参数名称为 kwargs
    [  # 参数列表包含以下内容：
        {},  # 默认情况下 check_exact=False
        {"check_exact": False},  # 设置 check_exact=False
        {"check_exact": True},   # 设置 check_exact=True
    ],
)
def test_assert_extension_array_equal_not_exact(kwargs):
    # see gh-23709
    arr1 = SparseArray([-0.17387645482451206, 0.3414148016424936])  # 创建 SparseArray 对象 arr1
    arr2 = SparseArray([-0.17387645482451206, 0.3414148016424937])  # 创建 SparseArray 对象 arr2

    if kwargs.get("check_exact", False):
        msg = """\
ExtensionArray are different

ExtensionArray values are different \\(50\\.0 %\\)
\\[left\\]:  \\[-0\\.17387645482.*, 0\\.341414801642.*\\]
\\[right\\]: \\[-0\\.17387645482.*, 0\\.341414801642.*\\]"""
        # 如果 kwargs 中 check_exact 为 True，定义匹配错误信息 msg

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
            # 使用 pytest 的 raises 函数检测是否抛出 AssertionError，且匹配错误信息 msg
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)
        # 否则直接调用 tm.assert_extension_array_equal 比较 arr1 和 arr2


@pytest.mark.parametrize("decimals", range(10))  # 参数化测试，decimals 取值范围为 0 到 9
def test_assert_extension_array_equal_less_precise(decimals):
    rtol = 0.5 * 10**-decimals  # 计算相对容差 rtol
    arr1 = SparseArray([0.5, 0.123456])  # 创建 SparseArray 对象 arr1
    arr2 = SparseArray([0.5, 0.123457])  # 创建 SparseArray 对象 arr2

    if decimals >= 5:
        msg = """\
ExtensionArray are different

ExtensionArray values are different \\(50\\.0 %\\)
\\[left\\]:  \\[0\\.5, 0\\.123456\\]
\\[right\\]: \\[0\\.5, 0\\.123457\\]"""
        # 如果小数位数 decimals 大于等于 5，定义匹配错误信息 msg

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)
            # 使用 pytest 的 raises 函数检测是否抛出 AssertionError，且匹配错误信息 msg
    else:
        tm.assert_extension_array_equal(arr1, arr2, rtol=rtol)
        # 否则直接调用 tm.assert_extension_array_equal 比较 arr1 和 arr2


def test_assert_extension_array_equal_dtype_mismatch(check_dtype):
    end = 5  # 设置 end 变量为 5
    kwargs = {"check_dtype": check_dtype}  # 创建 kwargs 字典，包含 check_dtype 键

    arr1 = SparseArray(np.arange(end, dtype="int64"))  # 创建 SparseArray 对象 arr1，数据类型为 int64
    arr2 = SparseArray(np.arange(end, dtype="int32"))  # 创建 SparseArray 对象 arr2，数据类型为 int32

    if check_dtype:
        msg = """\
ExtensionArray are different

Attribute "dtype" are different
\\[left\\]:  Sparse\\[int64, 0\\]
\\[right\\]: Sparse\\[int32, 0\\]"""
        # 如果 check_dtype 为 True，定义匹配错误信息 msg

        with pytest.raises(AssertionError, match=msg):
            tm.assert_extension_array_equal(arr1, arr2, **kwargs)
            # 使用 pytest 的 raises 函数检测是否抛出 AssertionError，且匹配错误信息 msg
    else:
        tm.assert_extension_array_equal(arr1, arr2, **kwargs)
        # 否则直接调用 tm.assert_extension_array_equal 比较 arr1 和 arr2


def test_assert_extension_array_equal_missing_values():
    arr1 = SparseArray([np.nan, 1, 2, np.nan])  # 创建 SparseArray 对象 arr1，包含 NaN 值
    arr2 = SparseArray([np.nan, 1, 2, 3])       # 创建 SparseArray 对象 arr2，不包含 NaN 值

    msg = """\
ExtensionArray NA mask are different

ExtensionArray NA mask values are different \\(25\\.0 %\\)
\\[left\\]:  \\[True, False, False, True\\]
\\[right\\]: \\[True, False, False, False\\]"""
    # 定义匹配错误信息 msg

    with pytest.raises(AssertionError, match=msg):
        tm.assert_extension_array_equal(arr1, arr2)
        # 使用 pytest 的 raises 函数检测是否抛出 AssertionError，且匹配错误信息 msg


@pytest.mark.parametrize("side", ["left", "right"])  # 参数化测试，side 取值为 "left" 和 "right"
def test_assert_extension_array_equal_non_extension_array(side):
    numpy_array = np.arange(5)  # 创建 NumPy 数组 numpy_array，包含 0 到 4
    extension_array = SparseArray(numpy_array)  # 使用 SparseArray 将 numpy_array 转换为 SparseArray 对象

    msg = f"{side} is not an ExtensionArray"
    # 定义非扩展数组错误信息 msg
    # 根据条件选择不同的参数元组：
    # 如果 side 等于 "left"，则 args 是 (numpy_array, extension_array)
    # 否则 args 是 (extension_array, numpy_array)
    args = (
        (numpy_array, extension_array)
        if side == "left"
        else (extension_array, numpy_array)
    )
    
    # 使用 pytest 的 assert_raises 函数验证是否会抛出 AssertionError，并匹配特定的错误消息
    with pytest.raises(AssertionError, match=msg):
        # 调用 tm 模块中的 assert_extension_array_equal 函数，传入 args 中的参数
        tm.assert_extension_array_equal(*args)
def test_assert_extension_array_equal_ignore_dtype_mismatch(any_int_dtype):
    # 根据该 GitHub 问题链接，创建包含不同整数类型的数组 left 和 right
    left = array([1, 2, 3], dtype="Int64")
    right = array([1, 2, 3], dtype=any_int_dtype)
    # 使用测试工具函数，比较两个扩展数组 left 和 right 是否相等，忽略数据类型的不匹配
    tm.assert_extension_array_equal(left, right, check_dtype=False)


def test_assert_extension_array_equal_time_units():
    # 根据该 GitHub 问题链接，创建两个具有不同时间单位的时间戳数组
    timestamp = Timestamp("2023-11-04T12")
    naive = array([timestamp], dtype="datetime64[ns]")
    utc = array([timestamp], dtype="datetime64[ns, UTC]")

    # 使用测试工具函数，比较两个扩展数组 naive 和 utc 是否相等，忽略数据类型的不匹配
    tm.assert_extension_array_equal(naive, utc, check_dtype=False)
    # 使用测试工具函数，再次比较两个扩展数组 utc 和 naive 是否相等，忽略数据类型的不匹配
    tm.assert_extension_array_equal(utc, naive, check_dtype=False)
```