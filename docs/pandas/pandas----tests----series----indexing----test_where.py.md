# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_where.py`

```
import numpy as np
import pytest

from pandas._config import using_pyarrow_string_dtype

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Series,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm


def test_where_unsafe_int(any_signed_int_numpy_dtype):
    # 创建一个整数类型的 Series，值为 0 到 9
    s = Series(np.arange(10), dtype=any_signed_int_numpy_dtype)
    # 创建一个布尔掩码，表示 Series 中小于 5 的位置
    mask = s < 5

    # 将 mask 指示的位置上的值替换为从 2 到 6 的整数序列
    s[mask] = range(2, 7)
    # 创建预期的 Series，将序列 [2, 3, 4, 5, 6] 与剩余部分 [5, 6, 7, 8, 9] 组合起来
    expected = Series(
        list(range(2, 7)) + list(range(5, 10)),
        dtype=any_signed_int_numpy_dtype,
    )

    # 使用测试框架验证实际结果与预期结果是否相等
    tm.assert_series_equal(s, expected)


def test_where_unsafe_float(float_numpy_dtype):
    # 创建一个浮点数类型的 Series，值为 0.0 到 9.0
    s = Series(np.arange(10), dtype=float_numpy_dtype)
    # 创建一个布尔掩码，表示 Series 中小于 5 的位置
    mask = s < 5

    # 将 mask 指示的位置上的值替换为从 2 到 6 的整数序列
    s[mask] = range(2, 7)
    # 创建预期的 Series，将序列 [2, 3, 4, 5, 6] 与剩余部分 [5.0, 6.0, 7.0, 8.0, 9.0] 组合起来
    data = list(range(2, 7)) + list(range(5, 10))
    expected = Series(data, dtype=float_numpy_dtype)

    # 使用测试框架验证实际结果与预期结果是否相等
    tm.assert_series_equal(s, expected)


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [
        (np.int8, np.float64),
        (np.int16, np.float64),
        (np.int32, np.float64),
        (np.int64, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_where_unsafe_upcast(dtype, expected_dtype):
    # 见 GitHub 问题 gh-9743
    # 创建一个指定 dtype 的整数类型 Series，值为 0 到 9
    s = Series(np.arange(10), dtype=dtype)
    # 创建一个浮点数序列作为将要赋值的值
    values = [2.5, 3.5, 4.5, 5.5, 6.5]
    # 创建一个布尔掩码，表示 Series 中小于 5 的位置
    mask = s < 5
    # 创建预期的 Series，将浮点数序列与剩余部分 [5, 6, 7, 8, 9] 组合起来，类型为 expected_dtype
    expected = Series(values + list(range(5, 10)), dtype=expected_dtype)
    # 如果 dtype 和 expected_dtype 都是浮点数类型，则不会产生警告，否则产生 FutureWarning
    warn = (
        None
        if np.dtype(dtype).kind == np.dtype(expected_dtype).kind == "f"
        else FutureWarning
    )
    # 使用测试框架验证实际结果与预期结果是否相等，并验证是否产生了警告
    with tm.assert_produces_warning(warn, match="incompatible dtype"):
        s[mask] = values
    tm.assert_series_equal(s, expected)


def test_where_unsafe():
    # 见 GitHub 问题 gh-9731
    # 创建一个整数类型的 Series，值为 0 到 9
    s = Series(np.arange(10), dtype="int64")
    # 创建一个浮点数序列作为将要赋值的值
    values = [2.5, 3.5, 4.5, 5.5]

    # 创建一个布尔掩码，表示 Series 中大于 5 的位置
    mask = s > 5
    # 创建预期的 Series，将 [0, 1, 2, 3, 4, 5] 与浮点数序列 [2.5, 3.5, 4.5, 5.5] 组合起来，类型为 float64
    expected = Series(list(range(6)) + values, dtype="float64")

    # 使用测试框架验证实际结果与预期结果是否相等，并验证是否产生了警告
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        s[mask] = values
    tm.assert_series_equal(s, expected)

    # 见 GitHub 问题 gh-3235
    # 创建一个整数类型的 Series，值为 0 到 9
    s = Series(np.arange(10), dtype="int64")
    # 创建一个布尔掩码，表示 Series 中小于 5 的位置
    mask = s < 5
    # 将 mask 指示的位置上的值替换为从 2 到 6 的整数序列
    s[mask] = range(2, 7)
    # 创建预期的 Series，将序列 [2, 3, 4, 5, 6] 与剩余部分 [5, 6, 7, 8, 9] 组合起来，类型为 int64
    expected = Series(list(range(2, 7)) + list(range(5, 10)), dtype="int64")
    tm.assert_series_equal(s, expected)
    assert s.dtype == expected.dtype

    # 创建一个整数类型的 Series，值为 0 到 9
    s = Series(np.arange(10), dtype="int64")
    # 创建一个布尔掩码，表示 Series 中大于 5 的位置
    mask = s > 5
    # 将 mask 指示的位置上的值替换为 [0, 0, 0, 0]
    s[mask] = [0] * 4
    # 创建预期的 Series，将 [0, 1, 2, 3, 4, 5] 与 [0, 0, 0, 0] 组合起来，类型为 int64
    expected = Series([0, 1, 2, 3, 4, 5] + [0] * 4, dtype="int64")
    tm.assert_series_equal(s, expected)

    # 创建一个整数类型的 Series，值为 0 到 9
    s = Series(np.arange(10))
    # 创建一个布尔掩码，表示 Series 中大于 5 的位置
    mask = s > 5

    # 使用 pytest 验证设置长度不匹配的列表索引器会引发 ValueError
    msg = "cannot set using a list-like indexer with a different length than the value"
    with pytest.raises(ValueError, match=msg):
        s[mask] = [5, 4, 3, 2, 1]

    with pytest.raises(ValueError, match=msg):
        s[mask] = [0] * 5

    # 创建一个包含整数的 Series
    s = Series([1, 2, 3, 4])
    # 使用 where 方法，将小于 2 的值替换为 NaN
    result = s.where(s > 2, np.nan)
    # 创建预期的 Series，将小于 2 的值替换为 NaN
    expected = Series([np.nan, np.nan, 3, 4])
    tm.assert_series_equal(result, expected)

    # 见 GitHub 问题 GH 4667
    # 设置 None 改变 dtype
    # 创建一个 Pandas Series 对象，包含从 0 到 9 的浮点数
    s = Series(range(10)).astype(float)
    # 将索引为 8 的元素设置为 None（NaN）
    s[8] = None
    # 从 Series 中取出索引为 8 的元素
    result = s[8]
    # 使用断言检查 result 是否为 NaN（空值）
    assert isna(result)
    
    # 创建一个新的 Pandas Series 对象，包含从 0 到 9 的浮点数
    s = Series(range(10)).astype(float)
    # 将所有大于 8 的元素设置为 None（NaN）
    s[s > 8] = None
    # 从 Series 中选择所有值为 NaN 的元素
    result = s[isna(s)]
    # 创建一个预期的 Series 对象，所有值为 NaN，索引为 [9]
    expected = Series(np.nan, index=[9])
    # 使用 Pandas 测试工具比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
def test_where():
    # 创建一个包含5个标准正态分布随机数的序列
    s = Series(np.random.default_rng(2).standard_normal(5))
    # 使用条件判断序列中大于0的元素，生成布尔型序列
    cond = s > 0

    # 对满足条件的元素进行筛选并删除缺失值
    rs = s.where(cond).dropna()
    # 直接使用布尔索引获取满足条件的元素
    rs2 = s[cond]
    # 断言两个结果序列相等
    tm.assert_series_equal(rs, rs2)

    # 对序列进行条件填充，当不满足条件时使用-s替代
    rs = s.where(cond, -s)
    # 断言条件填充后的序列绝对值与原序列绝对值相等
    tm.assert_series_equal(rs, s.abs())

    # 对序列仅进行条件筛选，不进行填充
    rs = s.where(cond)
    # 断言条件筛选后的序列形状与原序列相同
    assert s.shape == rs.shape
    # 断言条件筛选后的序列与原序列不是同一个对象
    assert rs is not s

    # 测试序列对齐
    # 创建一个指定索引的布尔型序列条件
    cond = Series([True, False, False, True, False], index=s.index)
    # 对序列的绝对值取负数
    s2 = -(s.abs())

    # 期望结果是对s2中满足条件的元素重新索引，并填充为NaN
    expected = s2[cond].reindex(s2.index[:3]).reindex(s2.index)
    # 使用条件筛选填充为NaN的操作
    rs = s2.where(cond[:3])
    # 断言条件填充后的序列与期望结果相等
    tm.assert_series_equal(rs, expected)

    # 期望结果是对s2取绝对值后的序列
    expected = s2.abs()
    # 将第一个元素的值替换为s2中第一个元素的值
    expected.iloc[0] = s2[0]
    # 使用条件筛选进行填充
    rs = s2.where(cond[:3], -s2)
    # 断言条件填充后的序列与期望结果相等
    tm.assert_series_equal(rs, expected)


def test_where_error():
    # 创建一个包含5个标准正态分布随机数的序列
    s = Series(np.random.default_rng(2).standard_normal(5))
    # 创建一个大于0的条件序列
    cond = s > 0

    # 错误消息定义
    msg = "Array conditional must be same shape as self"
    # 断言抛出错误并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        s.where(1)
    with pytest.raises(ValueError, match=msg):
        s.where(cond[:3].values, -s)

    # GH 2745
    # 创建一个包含整数1和2的序列
    s = Series([1, 2])
    # 将序列中的第一个元素设置为0
    s[[True, False]] = [0, 1]
    # 期望结果是包含整数0和2的序列
    expected = Series([0, 2])
    # 断言序列与期望结果相等
    tm.assert_series_equal(s, expected)

    # 失败情况
    # 错误消息定义
    msg = "cannot set using a list-like indexer with a different length than the value"
    # 断言抛出错误并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = [0, 2, 3]

    with pytest.raises(ValueError, match=msg):
        s[[True, False]] = []


@pytest.mark.parametrize("klass", [list, tuple, np.array, Series])
def test_where_array_like(klass):
    # see gh-15414
    # 创建一个包含整数1、2、3的序列
    s = Series([1, 2, 3])
    # 创建一个布尔型条件序列
    cond = [False, True, True]
    # 期望结果是将第一个元素替换为NaN的序列
    expected = Series([np.nan, 2, 3])

    # 使用不同类型的数组类作为条件进行填充
    result = s.where(klass(cond))
    # 断言填充后的序列与期望结果相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "cond",
    [
        [1, 0, 1],
        Series([2, 5, 7]),
        ["True", "False", "True"],
        [Timestamp("2017-01-01"), pd.NaT, Timestamp("2017-01-02")],
    ],
)
def test_where_invalid_input(cond):
    # see gh-15414: only boolean arrays accepted
    # 创建一个包含整数1、2、3的序列
    s = Series([1, 2, 3])
    # 错误消息定义
    msg = "Boolean array expected for the condition"

    # 断言抛出类型错误并匹配错误消息
    with pytest.raises(TypeError, match=msg):
        s.where(cond)

    # 错误消息定义
    msg = "Array conditional must be same shape as self"
    # 断言抛出值错误并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        s.where([True])


def test_where_ndframe_align():
    # 错误消息定义
    msg = "Array conditional must be same shape as self"
    # 创建一个包含整数1、2、3的序列
    s = Series([1, 2, 3])

    # 创建一个长度为1的布尔型条件序列
    cond = [True]
    # 断言抛出值错误并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        s.where(cond)

    # 期望结果是对序列中的第一个元素填充为NaN
    expected = Series([1, np.nan, np.nan])

    # 使用Series对象作为条件进行填充
    out = s.where(Series(cond))
    # 断言填充后的序列与期望结果相等
    tm.assert_series_equal(out, expected)

    # 创建一个包含4个元素的布尔型条件序列
    cond = np.array([False, True, False, True])
    # 断言抛出值错误并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        s.where(cond)

    # 期望结果是对序列中第二个元素进行填充为2
    expected = Series([np.nan, 2, np.nan])

    # 使用Series对象作为条件进行填充
    out = s.where(Series(cond))
    # 断言填充后的序列与期望结果相等
    tm.assert_series_equal(out, expected)


@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set ints into string")
def test_where_setitem_invalid():
    # GH 2702
    pass
    # 确保在无效的列表赋值时引发正确的异常

    # 创建一个 lambda 函数，用于生成错误消息，指示使用了与值长度不同的切片索引器
    msg = (
        lambda x: f"cannot set using a {x} indexer with a "
        "different length than the value"
    )

    # 创建一个包含字符 "abc" 的 Series 对象
    s = Series(list("abc"))

    # 使用 pytest 来验证在赋值操作中使用切片时是否会引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg("slice")):
        s[0:3] = list(range(27))

    # 对切片进行赋值操作
    s[0:3] = list(range(3))

    # 创建期望的 Series 对象，包含整数序列 [0, 1, 2]
    expected = Series([0, 1, 2])

    # 使用 assert_series_equal 来验证 s 是否等于 expected，要求转换为 np.int64 类型
    tm.assert_series_equal(s.astype(np.int64), expected)

    # 使用切片和步长创建一个包含字符 "abcdef" 的 Series 对象
    s = Series(list("abcdef"))

    # 验证在带有步长的切片赋值操作中是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg("slice")):
        s[0:4:2] = list(range(27))

    # 对带有步长的切片赋值操作
    s[0:4:2] = list(range(2))

    # 创建期望的 Series 对象，包含序列 [0, "b", 1, "d", "e", "f"]
    expected = Series([0, "b", 1, "d", "e", "f"])

    # 使用 assert_series_equal 来验证 s 是否等于 expected
    tm.assert_series_equal(s, expected)

    # 使用负数索引创建一个包含字符 "abcdef" 的 Series 对象
    s = Series(list("abcdef"))

    # 验证在使用负数索引的切片赋值操作中是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg("slice")):
        s[:-1] = list(range(27))

    # 对负数索引的切片赋值操作
    s[-3:-1] = list(range(2))

    # 创建期望的 Series 对象，包含序列 ["a", "b", "c", 0, 1, "f"]
    expected = Series(["a", "b", "c", 0, 1, "f"])

    # 使用 assert_series_equal 来验证 s 是否等于 expected
    tm.assert_series_equal(s, expected)

    # 使用列表创建一个包含字符 "abc" 的 Series 对象
    s = Series(list("abc"))

    # 验证在使用列表赋值操作时是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg("list-like")):
        s[[0, 1, 2]] = list(range(27))

    # 使用列表赋值操作
    s[[0, 1, 2]] = list(range(2))

    # 使用标量赋值操作创建一个包含字符 "abc" 的 Series 对象
    s = Series(list("abc"))

    # 对标量赋值操作
    s[0] = list(range(10))

    # 创建期望的 Series 对象，包含序列 [list(range(10)), "b", "c"]
    expected = Series([list(range(10)), "b", "c"])

    # 使用 assert_series_equal 来验证 s 是否等于 expected
    tm.assert_series_equal(s, expected)
@pytest.mark.parametrize("size", range(2, 6))
@pytest.mark.parametrize(
    "mask", [[True, False, False, False, False], [True, False], [False]]
)
@pytest.mark.parametrize(
    "item", [2.0, np.nan, np.finfo(float).max, np.finfo(float).min]
)
@pytest.mark.parametrize("box", [np.array, list, tuple])
def test_broadcast(size, mask, item, box):
    # GH#8801, GH#4195
    # 创建一个大小为 'size' 的选择数组，将 'mask' 调整为该大小
    selection = np.resize(mask, size)

    # 创建一个从 0 到 size-1 的浮点数数组 'data'
    data = np.arange(size, dtype=float)

    # 根据选择 'selection' 构建预期的 Series 对象
    expected = Series(
        [item if use_item else data[i] for i, use_item in enumerate(selection)]
    )

    # 创建一个包含 'data' 的 Series 对象 's'
    s = Series(data)

    # 在 's' 中根据 'selection' 设置 'item'
    s[selection] = item
    # 断言 's' 与 'expected' 是否相等
    tm.assert_series_equal(s, expected)

    # 创建一个包含 'data' 的 Series 对象 's'
    s = Series(data)
    # 使用 'where' 方法将 'item' 插入到 's' 中，根据 'selection' 的取反
    result = s.where(~selection, box([item]))
    # 断言 'result' 与 'expected' 是否相等
    tm.assert_series_equal(result, expected)

    # 创建一个包含 'data' 的 Series 对象 's'
    s = Series(data)
    # 使用 'mask' 方法将 'item' 插入到 's' 中，根据 'selection'
    result = s.mask(selection, box([item]))
    # 断言 'result' 与 'expected' 是否相等
    tm.assert_series_equal(result, expected)


def test_where_inplace():
    # 创建一个具有随机标准正态分布的 Series 对象 's'
    s = Series(np.random.default_rng(2).standard_normal(5))
    # 根据条件 'cond' 创建一个新的 Series 对象 'rs'
    cond = s > 0
    rs = s.copy()

    # 在 'rs' 中应用 'where' 方法，根据条件 'cond' 进行就地更新
    rs.where(cond, inplace=True)
    # 断言删除NaN后的 'rs' 与 's[cond]' 是否相等
    tm.assert_series_equal(rs.dropna(), s[cond])
    # 断言 'rs' 与 's' 使用 'where' 方法根据条件 'cond' 是否相等
    tm.assert_series_equal(rs, s.where(cond))

    rs = s.copy()
    # 在 'rs' 中应用 'where' 方法，使用 '-s' 替换不满足条件 'cond' 的元素，进行就地更新
    rs.where(cond, -s, inplace=True)
    # 断言 'rs' 与 's' 使用 'where' 方法根据条件 'cond' 是否相等
    tm.assert_series_equal(rs, s.where(cond, -s))


def test_where_dups():
    # GH 4550
    # 在索引中存在重复值时，'where' 方法可能导致崩溃
    s1 = Series(list(range(3)))
    s2 = Series(list(range(3)))
    # 连接两个 Series 对象 's1' 和 's2' 成为 'comb'
    comb = pd.concat([s1, s2])
    # 使用 'where' 方法根据条件 'comb < 2'，创建新的 Series 对象 'result'
    result = comb.where(comb < 2)
    # 预期的结果 Series 对象
    expected = Series([0, 1, np.nan, 0, 1, np.nan], index=[0, 1, 2, 0, 1, 2])
    # 断言 'result' 与 'expected' 是否相等
    tm.assert_series_equal(result, expected)

    # GH 4548
    # 当索引中存在重复值时，'where' 方法的就地更新可能不起作用
    comb[comb < 1] = 5
    expected = Series([5, 1, 2, 5, 1, 2], index=[0, 1, 2, 0, 1, 2])
    # 断言 'comb' 与 'expected' 是否相等
    tm.assert_series_equal(comb, expected)

    comb[comb < 2] += 10
    expected = Series([5, 11, 2, 5, 11, 2], index=[0, 1, 2, 0, 1, 2])
    # 断言 'comb' 与 'expected' 是否相等
    tm.assert_series_equal(comb, expected)


def test_where_numeric_with_string():
    # GH 9280
    # 当使用字符串替换数值时，'where' 方法的行为测试
    s = Series([1, 2, 3])
    # 使用 'where' 方法，将数值小于1的元素替换为字符串 'X'
    w = s.where(s > 1, "X")

    # 断言第一个元素 'w[0]' 不是整数
    assert not is_integer(w[0])
    # 断言第二个元素 'w[1]' 是整数
    assert is_integer(w[1])
    # 断言第三个元素 'w[2]' 是整数
    assert is_integer(w[2])
    # 断言第一个元素 'w[0]' 的类型是字符串
    assert isinstance(w[0], str)
    # 断言 'w' 的数据类型是对象类型
    assert w.dtype == "object"

    # 使用列表 ['X', 'Y', 'Z'] 替换数值小于1的元素
    w = s.where(s > 1, ["X", "Y", "Z"])
    # 断言第一个元素 'w[0]' 不是整数
    assert not is_integer(w[0])
    # 断言第二个元素 'w[1]' 是整数
    assert is_integer(w[1])
    # 断言第三个元素 'w[2]' 是整数
    assert is_integer(w[2])
    # 断言第一个元素 'w[0]' 的类型是字符串
    assert isinstance(w[0], str)
    # 断言 'w' 的数据类型是对象类型
    assert w.dtype == "object"

    # 使用 NumPy 数组 ['X', 'Y', 'Z'] 替换数值小于1的元素
    w = s.where(s > 1, np.array(["X", "Y", "Z"]))
    # 断言第一个元素 'w[0]' 不是整数
    assert not is_integer(w[0])
    # 断言第二个元素 'w[1]' 是整数
    assert is_integer(w[1])
    # 断言第三个元素 'w[2]' 是整数
    assert is_integer(w[2])
    # 断言第一个元素 'w[0]' 的类型是字符串
    assert isinstance(w[0], str)
    # 断言 'w' 的数据类型是对象类型
    assert w.dtype == "object"


def test_where_datetimetz():
    # GH 15701
    # 测试带有时区的日期时间数据的 'where' 方法
    timestamps = ["2016-12-31 12:00:04+00:00", "2016-12-31 12:00:04.010000+00:00"]
    # 创建带有时区信息的日期时间 Series 对象 'ser'
    ser = Series([Timestamp(t) for t in timestamps], dtype="datetime64[ns, UTC]")
    # 使用 'where' 方法，根据条件 Series 创建新的日期时间 Series 对象 'rs'
    rs = ser.where(Series([False, True]))
    # 预期的结果 Series 对象
    expected = Series([pd.NaT, ser[1]], dtype="datetime64[ns, UTC]")
    # 使用测试框架中的函数来断言 Series 对象 `rs` 是否与期望的 Series 对象 `expected` 相等。
    tm.assert_series_equal(rs, expected)
# 定义一个测试函数，用于验证稀疏系列的 `where` 方法
def test_where_sparse():
    # GH#17198 确保我们不会因为 sp_index 而得到 AttributeError
    ser = Series(pd.arrays.SparseArray([1, 2]))
    # 使用 `where` 方法筛选出满足条件的元素，不满足的用 0 替代
    result = ser.where(ser >= 2, 0)
    # 期望的结果是一个稀疏系列，其中小于 2 的元素被替换为 0
    expected = Series(pd.arrays.SparseArray([0, 2]))
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，验证空系列和非布尔类型条件的 `where` 方法
def test_where_empty_series_and_empty_cond_having_non_bool_dtypes():
    # https://github.com/pandas-dev/pandas/issues/34592
    # 创建一个空的浮点型系列
    ser = Series([], dtype=float)
    # 对空系列应用空条件，预期结果应与原系列相同
    result = ser.where([])
    tm.assert_series_equal(result, ser)


# 定义一个测试函数，验证分类数据的 `where` 方法
def test_where_categorical(frame_or_series):
    # https://github.com/pandas-dev/pandas/issues/18888
    # 创建一个包含分类数据的系列，并指定分类的类别
    exp = frame_or_series(
        pd.Categorical(["A", "A", "B", "B", np.nan], categories=["A", "B", "C"]),
        dtype="category",
    )
    # 创建一个包含分类数据的系列
    df = frame_or_series(["A", "A", "B", "B", "C"], dtype="category")
    # 对系列应用条件，筛选出不等于 "C" 的元素
    res = df.where(df != "C")
    tm.assert_equal(exp, res)


# 定义一个测试函数，验证日期时间类别数据的 `where` 方法
def test_where_datetimelike_categorical(tz_naive_fixture):
    # GH#37682
    tz = tz_naive_fixture

    # 创建一个日期范围，带有时区信息
    dr = date_range("2001-01-01", periods=3, tz=tz)._with_freq(None)
    # 创建一个日期时间索引对象
    lvals = pd.DatetimeIndex([dr[0], dr[1], pd.NaT])
    # 创建一个分类数据对象，其中包含日期时间和缺失值
    rvals = pd.Categorical([dr[0], pd.NaT, dr[2]])

    # 创建一个布尔掩码数组
    mask = np.array([True, True, False])

    # 使用 `where` 方法处理日期时间索引
    # 对于日期时间索引，`where` 方法返回一个新的日期时间索引，根据条件选择左值或右值
    res = lvals.where(mask, rvals)
    tm.assert_index_equal(res, dr)

    # 对于 `DatetimeArray`，使用 `_where` 方法
    res = lvals._data._where(mask, rvals)
    tm.assert_datetime_array_equal(res, dr._data)

    # 对于系列，使用 `where` 方法
    res = Series(lvals).where(mask, rvals)
    tm.assert_series_equal(res, Series(dr))

    # 对于数据框，使用 `where` 方法
    res = pd.DataFrame(lvals).where(mask[:, None], pd.DataFrame(rvals))
    tm.assert_frame_equal(res, pd.DataFrame(dr))
```