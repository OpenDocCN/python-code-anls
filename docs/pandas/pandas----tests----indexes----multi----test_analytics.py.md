# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_analytics.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库及其子模块和函数
import pandas as pd
from pandas import (
    Index,
    MultiIndex,
    date_range,
    period_range,
)
import pandas._testing as tm

# 定义一个测试函数，测试 infer_objects 方法
def test_infer_objects(idx):
    # 断言调用 infer_objects 方法时会抛出 NotImplementedError 异常，并匹配特定消息
    with pytest.raises(NotImplementedError, match="to_frame"):
        idx.infer_objects()

# 定义一个测试函数，测试 shift 方法
def test_shift(idx):
    # 准备错误消息内容
    msg = (
        "This method is only implemented for DatetimeIndex, PeriodIndex and "
        "TimedeltaIndex; Got type MultiIndex"
    )
    # 断言调用 shift 方法时会抛出 NotImplementedError 异常，并匹配特定消息
    with pytest.raises(NotImplementedError, match=msg):
        idx.shift(1)
    with pytest.raises(NotImplementedError, match=msg):
        idx.shift(1, 2)

# 定义一个测试函数，测试 groupby 方法
def test_groupby(idx):
    # 使用 np.array 创建分组，然后按组分组
    groups = idx.groupby(np.array([1, 1, 1, 2, 2, 2]))
    # 将索引转换为列表
    labels = idx.tolist()
    # 准备预期结果字典
    exp = {1: labels[:3], 2: labels[3:]}
    # 断言 groupby 方法的结果与预期结果字典相等
    tm.assert_dict_equal(groups, exp)

    # GH5620 检查索引自身分组
    groups = idx.groupby(idx)
    # 准备预期结果字典，每个键对应一个列表，包含该键自身
    exp = {key: [key] for key in idx}
    # 断言 groupby 方法的结果与预期结果字典相等
    tm.assert_dict_equal(groups, exp)

# 定义一个测试函数，测试 truncate 方法对 MultiIndex 的使用
def test_truncate_multiindex():
    # 创建主要轴和次要轴索引
    major_axis = Index(list(range(4)))
    minor_axis = Index(list(range(2)))

    # 创建主要轴和次要轴的编码
    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])

    # 创建 MultiIndex 对象
    index = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=["L1", "L2"],
    )

    # 测试 truncate 方法的 before 参数
    result = index.truncate(before=1)
    assert "foo" not in result.levels[0]
    assert 1 in result.levels[0]
    assert index.names == result.names

    # 测试 truncate 方法的 after 参数
    result = index.truncate(after=1)
    assert 2 not in result.levels[0]
    assert 1 in result.levels[0]
    assert index.names == result.names

    # 测试 truncate 方法的 before 和 after 参数结合使用
    result = index.truncate(before=1, after=2)
    assert len(result.levels[0]) == 2
    assert index.names == result.names

    # 测试 truncate 方法异常情况，after 参数小于 before 参数
    msg = "after < before"
    with pytest.raises(ValueError, match=msg):
        index.truncate(3, 1)

# TODO: reshape

# 定义一个测试函数，测试 reorder_levels 方法
def test_reorder_levels(idx):
    # 断言调用 reorder_levels 方法时会抛出 IndexError 异常，并匹配特定消息
    with pytest.raises(IndexError, match="^Too many levels"):
        idx.reorder_levels([2, 1, 0])

# 定义一个测试函数，测试 numpy 中的 repeat 函数
def test_numpy_repeat():
    reps = 2
    numbers = [1, 2, 3]
    names = np.array(["foo", "bar"])

    # 创建 MultiIndex 对象
    m = MultiIndex.from_product([numbers, names], names=names)
    # 准备预期结果的 MultiIndex 对象
    expected = MultiIndex.from_product([numbers, names.repeat(reps)], names=names)
    # 断言 np.repeat 函数的结果与预期结果的 MultiIndex 对象相等
    tm.assert_index_equal(np.repeat(m, reps), expected)

    # 测试异常情况，验证 np.repeat 函数不支持 axis 参数
    msg = "the 'axis' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        np.repeat(m, reps, axis=1)

# 定义一个测试函数，测试 append 方法在混合数据类型时的行为
def test_append_mixed_dtypes():
    # 创建一个日期时间索引和带时区的日期时间索引，以及一个周期索引
    dti = date_range("2011-01-01", freq="ME", periods=3)
    dti_tz = date_range("2011-01-01", freq="ME", periods=3, tz="US/Eastern")
    pi = period_range("2011-01", freq="M", periods=3)

    # 创建一个混合数据类型的 MultiIndex 对象
    mi = MultiIndex.from_arrays(
        [[1, 2, 3], [1.1, np.nan, 3.3], ["a", "b", "c"], dti, dti_tz, pi]
    )
    # 断言 MultiIndex 对象的层级数为 6
    assert mi.nlevels == 6

    # 测试 append 方法的行为
    res = mi.append(mi)
    # 创建一个新的 MultiIndex 对象，通过传入数组作为参数
    exp = MultiIndex.from_arrays(
        [
            [1, 2, 3, 1, 2, 3],                     # 第一层数组：整数
            [1.1, np.nan, 3.3, 1.1, np.nan, 3.3],   # 第二层数组：浮点数和NaN
            ["a", "b", "c", "a", "b", "c"],         # 第三层数组：字符串
            dti.append(dti),                       # 第四层数组：dti 扩展自身
            dti_tz.append(dti_tz),                 # 第五层数组：dti_tz 扩展自身
            pi.append(pi),                         # 第六层数组：pi 扩展自身
        ]
    )
    # 使用测试框架中的 assert_index_equal 函数验证 res 与 exp 相等
    tm.assert_index_equal(res, exp)

    # 创建另一个新的 MultiIndex 对象，通过传入数组作为参数
    other = MultiIndex.from_arrays(
        [
            ["x", "y", "z"],                       # 第一层数组：字符串 'x', 'y', 'z'
            ["x", "y", "z"],                       # 第二层数组：字符串 'x', 'y', 'z'
            ["x", "y", "z"],                       # 第三层数组：字符串 'x', 'y', 'z'
            ["x", "y", "z"],                       # 第四层数组：字符串 'x', 'y', 'z'
            ["x", "y", "z"],                       # 第五层数组：字符串 'x', 'y', 'z'
            ["x", "y", "z"],                       # 第六层数组：字符串 'x', 'y', 'z'
        ]
    )

    # 将 other 添加到 mi 中，返回结果到 res
    res = mi.append(other)

    # 创建预期的 MultiIndex 对象，通过传入数组作为参数
    exp = MultiIndex.from_arrays(
        [
            [1, 2, 3, "x", "y", "z"],               # 第一层数组：整数和字符串的组合
            [1.1, np.nan, 3.3, "x", "y", "z"],     # 第二层数组：浮点数、NaN 和字符串的组合
            ["a", "b", "c", "x", "y", "z"],        # 第三层数组：字符串的组合
            dti.append(Index(["x", "y", "z"])),    # 第四层数组：dti 扩展自身并添加字符串 'x', 'y', 'z'
            dti_tz.append(Index(["x", "y", "z"])), # 第五层数组：dti_tz 扩展自身并添加字符串 'x', 'y', 'z'
            pi.append(Index(["x", "y", "z"])),     # 第六层数组：pi 扩展自身并添加字符串 'x', 'y', 'z'
        ]
    )

    # 使用测试框架中的 assert_index_equal 函数验证 res 与 exp 相等
    tm.assert_index_equal(res, exp)
# 定义一个测试函数，用于测试索引对象的迭代功能
def test_iter(idx):
    # 将索引对象转换为列表，存储在 result 变量中
    result = list(idx)
    # 期望的结果列表，包含多个元组作为元素
    expected = [
        ("foo", "one"),
        ("foo", "two"),
        ("bar", "one"),
        ("baz", "two"),
        ("qux", "one"),
        ("qux", "two"),
    ]
    # 断言测试结果与期望结果相等
    assert result == expected


# 定义一个测试函数，用于测试索引对象的减法操作
def test_sub(idx):
    # 将索引对象赋值给变量 first
    first = idx

    # 检查减法操作是否引发预期的 TypeError 异常，匹配特定的错误信息
    msg = "cannot perform __sub__ with this index type: MultiIndex"
    with pytest.raises(TypeError, match=msg):
        first - idx[-3:]
    with pytest.raises(TypeError, match=msg):
        idx[-3:] - first
    with pytest.raises(TypeError, match=msg):
        idx[-3:] - first.tolist()
    msg = "cannot perform __rsub__ with this index type: MultiIndex"
    with pytest.raises(TypeError, match=msg):
        first.tolist() - idx[-3:]


# 定义一个测试函数，用于测试索引对象的映射功能
def test_map(idx):
    # 将索引对象赋值给变量 index
    index = idx

    # 对索引对象进行映射操作，使用 lambda 函数作为映射函数
    result = index.map(lambda x: x)
    # 断言映射后的结果与原索引对象相等
    tm.assert_index_equal(result, index)


# 使用 pytest 的参数化功能，定义多个测试用例，测试索引对象的映射功能
@pytest.mark.parametrize(
    "mapper",
    [
        lambda values, idx: {i: e for e, i in zip(values, idx)},  # 创建字典映射
        lambda values, idx: pd.Series(values, idx),  # 创建 Series 对象映射
    ],
)
def test_map_dictlike(idx, mapper):
    # 根据 mapper 函数创建的映射对象，赋值给 identity 变量
    identity = mapper(idx.values, idx)

    # 如果索引对象的数据类型为 np.uint64，并且映射结果是字典，则预期结果转换为 int64 类型的索引
    if idx.dtype == np.uint64 and isinstance(identity, dict):
        expected = idx.astype("int64")
    else:
        expected = idx

    # 对索引对象进行映射操作，将结果与预期结果进行断言
    result = idx.map(identity)
    tm.assert_index_equal(result, expected)

    # 创建一个空的映射对象，预期结果为包含 NaN 值的索引对象
    expected = Index([np.nan] * len(idx))
    result = idx.map(mapper(expected, idx))
    tm.assert_index_equal(result, expected)


# 使用 pytest 的参数化功能，定义多个测试用例，测试 numpy 的通用函数（ufuncs）
@pytest.mark.parametrize(
    "func",
    [
        np.exp, np.exp2, np.expm1, np.log, np.log2, np.log10, np.log1p,
        np.sqrt, np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan,
        np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh,
        np.deg2rad, np.rad2deg,
    ],
    ids=lambda func: func.__name__,  # 使用函数名作为测试用例的标识符
)
def test_numpy_ufuncs(idx, func):
    # 测试 numpy 的通用函数（ufuncs），验证是否会引发预期的 TypeError 异常
    # 参考 numpy 官方文档：https://numpy.org/doc/stable/reference/ufuncs.html
    expected_exception = TypeError
    msg = (
        "loop of ufunc does not support argument 0 of type tuple which "
        f"has no callable {func.__name__} method"
    )
    with pytest.raises(expected_exception, match=msg):
        func(idx)


# 使用 pytest 的参数化功能，定义多个测试用例，测试 numpy 的类型相关函数
@pytest.mark.parametrize(
    "func",
    [np.isfinite, np.isinf, np.isnan, np.signbit],
    ids=lambda func: func.__name__,  # 使用函数名作为测试用例的标识符
)
def test_numpy_type_funcs(idx, func):
    # 测试 numpy 的类型相关函数，验证是否会引发预期的 TypeError 异常
    msg = (
        f"ufunc '{func.__name__}' not supported for the input types, and the inputs "
        "could not be safely coerced to any supported types according to "
        "the casting rule ''safe''"
    )
    with pytest.raises(TypeError, match=msg):
        func(idx)
```