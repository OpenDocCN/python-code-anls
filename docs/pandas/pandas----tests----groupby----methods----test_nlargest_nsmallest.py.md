# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_nlargest_nsmallest.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 pandas 库中导入需要使用的类和函数
from pandas import (
    MultiIndex,
    Series,
    date_range,
)

# 导入 pandas 测试工具模块
import pandas._testing as tm


# 定义一个测试函数 test_nlargest，用于测试 Series 的 nlargest 方法
def test_nlargest():
    # 创建一个 Series 对象 a，包含一组整数
    a = Series([1, 3, 5, 7, 2, 9, 0, 4, 6, 10])
    # 创建一个 Series 对象 b，包含一组字符串 'a' 和 'b'
    b = Series(list("a" * 5 + "b" * 5))
    # 根据 b 对 a 进行分组
    gb = a.groupby(b)
    # 对每个分组中的元素进行 nlargest 操作，保留前 3 个最大的值
    r = gb.nlargest(3)
    # 创建预期结果 Series 对象 e，包含指定的值和索引
    e = Series(
        [7, 5, 3, 10, 9, 6],
        index=MultiIndex.from_arrays([list("aaabbb"), [3, 2, 1, 9, 5, 8]]),
    )
    # 使用测试工具模块中的方法验证 r 和 e 是否相等
    tm.assert_series_equal(r, e)

    # 创建另一个 Series 对象 a，包含一组整数
    a = Series([1, 1, 3, 2, 0, 3, 3, 2, 1, 0])
    # 根据 b 对 a 进行分组
    gb = a.groupby(b)
    # 创建预期结果 Series 对象 e，包含指定的值和索引
    e = Series(
        [3, 2, 1, 3, 3, 2],
        index=MultiIndex.from_arrays([list("aaabbb"), [2, 3, 1, 6, 5, 7]]),
    )
    # 使用测试工具模块中的方法验证 nlargest(3, keep="last") 结果和预期结果 e 是否相等
    tm.assert_series_equal(gb.nlargest(3, keep="last"), e)


# 定义一个测试函数 test_nlargest_mi_grouper，用于测试带有 MultiIndex 分组的 nlargest 方法
def test_nlargest_mi_grouper():
    # 生成随机数生成器对象 npr
    npr = np.random.default_rng(2)

    # 创建一个日期范围对象 dts，包含从 "20180101" 开始的 10 个日期
    dts = date_range("20180101", periods=10)
    # 创建一个包含日期范围和字符串 "one"、"two" 的列表
    iterables = [dts, ["one", "two"]]

    # 使用 MultiIndex.from_product 创建一个 MultiIndex 对象 idx
    idx = MultiIndex.from_product(iterables, names=["first", "second"])
    # 创建一个 Series 对象 s，包含指定的标准正态分布随机数和 MultiIndex 索引
    s = Series(npr.standard_normal(20), index=idx)

    # 对 s 根据 "first" 列进行分组，然后对每组使用 nlargest 方法选择最大的 1 个值
    result = s.groupby("first").nlargest(1)

    # 创建预期的 MultiIndex 索引对象 exp_idx 和对应的值 exp_values
    exp_idx = MultiIndex.from_tuples(
        [
            (dts[0], dts[0], "one"),
            (dts[1], dts[1], "one"),
            (dts[2], dts[2], "one"),
            (dts[3], dts[3], "two"),
            (dts[4], dts[4], "one"),
            (dts[5], dts[5], "one"),
            (dts[6], dts[6], "one"),
            (dts[7], dts[7], "one"),
            (dts[8], dts[8], "one"),
            (dts[9], dts[9], "one"),
        ],
        names=["first", "first", "second"],
    )

    exp_values = [
        0.18905338179353307,
        -0.41306354339189344,
        1.799707382720902,
        0.7738065867276614,
        0.28121066979764925,
        0.9775674511260357,
        -0.3288239040579627,
        0.45495807124085547,
        0.5452887139646817,
        0.12682784711186987,
    ]

    # 创建预期的 Series 对象 expected，包含预期的值和 MultiIndex 索引
    expected = Series(exp_values, index=exp_idx)
    # 使用测试工具模块中的方法验证 result 和 expected 是否相等，允许数值误差在 1e-3 以内
    tm.assert_series_equal(result, expected, check_exact=False, rtol=1e-3)


# 定义一个测试函数 test_nsmallest，用于测试 Series 的 nsmallest 方法
def test_nsmallest():
    # 创建一个 Series 对象 a，包含一组整数
    a = Series([1, 3, 5, 7, 2, 9, 0, 4, 6, 10])
    # 创建一个 Series 对象 b，包含一组字符串 'a' 和 'b'
    b = Series(list("a" * 5 + "b" * 5))
    # 根据 b 对 a 进行分组
    gb = a.groupby(b)
    # 对每个分组中的元素进行 nsmallest 操作，保留前 3 个最小的值
    r = gb.nsmallest(3)
    # 创建预期结果 Series 对象 e，包含指定的值和索引
    e = Series(
        [1, 2, 3, 0, 4, 6],
        index=MultiIndex.from_arrays([list("aaabbb"), [0, 4, 1, 6, 7, 8]]),
    )
    # 使用测试工具模块中的方法验证 r 和 e 是否相等
    tm.assert_series_equal(r, e)

    # 创建另一个 Series 对象 a，包含一组整数
    a = Series([1, 1, 3, 2, 0, 3, 3, 2, 1, 0])
    # 根据 b 对 a 进行分组
    gb = a.groupby(b)
    # 创建预期结果 Series 对象 e，包含指定的值和索引
    e = Series(
        [0, 1, 1, 0, 1, 2],
        index=MultiIndex.from_arrays([list("aaabbb"), [4, 1, 0, 9, 8, 7]]),
    )
    # 使用测试工具模块中的方法验证 nsmallest(3, keep="last") 结果和预期结果 e 是否相等
    tm.assert_series_equal(gb.nsmallest(3, keep="last"), e)


# 使用 pytest 的 parametrize 装饰器定义多个参数化测试
@pytest.mark.parametrize(
    "data, groups",
    [([0, 1, 2, 3], [0, 0, 1, 1]), ([0], [0])],
)
@pytest.mark.parametrize("dtype", [None, *tm.ALL_INT_NUMPY_DTYPES])
def test_nlargest_and_smallest_noop(data, groups, dtype, nselect_method):
    # GH 15272, GH 16345, GH 29129
    # 测试 nlargest 和 nsmallest 方法当结果是空操作时的情况，
    # 即输入已排序且组大小 <= n 的情况
    if dtype is not None:
        data = np.array(data, dtype=dtype)
    # 如果选择的方法是 "nlargest"，则将数据列表反转
    if nselect_method == "nlargest":
        data = list(reversed(data))
    
    # 创建一个名为 "a" 的 Series 对象，使用给定的数据
    ser = Series(data, name="a")
    
    # 使用 getattr 函数动态调用 ser.groupby(groups) 后返回的对象的 nselect_method 方法，参数为 n=2
    result = getattr(ser.groupby(groups), nselect_method)(n=2)
    
    # 如果 groups 是列表，则将其转换为整数类型的 NumPy 数组 expidx，否则直接使用原始的 groups
    expidx = np.array(groups, dtype=int) if isinstance(groups, list) else groups
    
    # 创建一个名为 "a" 的 Series 对象 expected，使用原始的数据和一个多级索引，索引由 expidx 和 ser.index 组成
    expected = Series(data, index=MultiIndex.from_arrays([expidx, ser.index]), name="a")
    
    # 使用 tm.assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```