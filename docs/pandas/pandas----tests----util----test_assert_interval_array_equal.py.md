# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_interval_array_equal.py`

```
import pytest  # 导入 pytest 测试框架

from pandas import interval_range  # 从 pandas 库中导入 interval_range 函数
import pandas._testing as tm  # 导入 pandas 内部测试模块 tm

# 使用 pytest 的 parametrize 装饰器，定义多组参数来执行同一个测试函数
@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 0, "periods": 4},  # 参数组1：起始为0，周期为4
        {"start": 1, "periods": 5},  # 参数组2：起始为1，周期为5
        {"start": 5, "end": 10, "closed": "left"},  # 参数组3：起始为5，结束为10，左闭合
    ],
)
def test_interval_array_equal(kwargs):
    arr = interval_range(**kwargs).values  # 使用 interval_range 函数创建区间数组，并获取其值
    tm.assert_interval_array_equal(arr, arr)  # 使用 tm.assert_interval_array_equal 断言数组 arr 与自身相等


# 定义测试函数：测试当闭合方式不同时的区间数组是否相等
def test_interval_array_equal_closed_mismatch():
    kwargs = {"start": 0, "periods": 5}  # 定义起始为0，周期为5的参数
    arr1 = interval_range(closed="left", **kwargs).values  # 创建左闭合区间数组
    arr2 = interval_range(closed="right", **kwargs).values  # 创建右闭合区间数组

    # 定义期望的错误信息
    msg = """\
IntervalArray are different

Attribute "closed" are different
\\[left\\]:  left
\\[right\\]: right"""

    # 使用 pytest 的 raises 断言抛出 AssertionError，并匹配错误信息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


# 定义测试函数：测试当周期数不同时的区间数组是否相等
def test_interval_array_equal_periods_mismatch():
    kwargs = {"start": 0}  # 定义起始为0的参数
    arr1 = interval_range(periods=5, **kwargs).values  # 创建周期为5的区间数组
    arr2 = interval_range(periods=6, **kwargs).values  # 创建周期为6的区间数组

    # 定义期望的错误信息
    msg = """\
IntervalArray.left are different

IntervalArray.left shapes are different
\\[left\\]:  \\(5,\\)
\\[right\\]: \\(6,\\)"""

    # 使用 pytest 的 raises 断言抛出 AssertionError，并匹配错误信息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


# 定义测试函数：测试当结束值不同时的区间数组是否相等
def test_interval_array_equal_end_mismatch():
    kwargs = {"start": 0, "periods": 5}  # 定义起始为0，周期为5的参数
    arr1 = interval_range(end=10, **kwargs).values  # 创建结束为10的区间数组
    arr2 = interval_range(end=20, **kwargs).values  # 创建结束为20的区间数组

    # 定义期望的错误信息
    msg = """\
IntervalArray.left are different

IntervalArray.left values are different \\(80.0 %\\)
\\[left\\]:  \\[0, 2, 4, 6, 8\\]
\\[right\\]: \\[0, 4, 8, 12, 16\\]"""

    # 使用 pytest 的 raises 断言抛出 AssertionError，并匹配错误信息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)


# 定义测试函数：测试当起始值不同时的区间数组是否相等
def test_interval_array_equal_start_mismatch():
    kwargs = {"periods": 4}  # 定义周期为4的参数
    arr1 = interval_range(start=0, **kwargs).values  # 创建起始为0的区间数组
    arr2 = interval_range(start=1, **kwargs).values  # 创建起始为1的区间数组

    # 定义期望的错误信息
    msg = """\
IntervalArray.left are different

IntervalArray.left values are different \\(100.0 %\\)
\\[left\\]:  \\[0, 1, 2, 3\\]
\\[right\\]: \\[1, 2, 3, 4\\]"""

    # 使用 pytest 的 raises 断言抛出 AssertionError，并匹配错误信息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_interval_array_equal(arr1, arr2)
```