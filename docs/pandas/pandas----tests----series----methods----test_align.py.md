# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_align.py`

```
# 导入所需的模块和库
from datetime import timezone  # 从datetime模块中导入timezone类

import numpy as np  # 导入numpy库并使用np作为别名
import pytest  # 导入pytest库

import pandas as pd  # 导入pandas库并使用pd作为别名
from pandas import (  # 从pandas库中导入Series、date_range、period_range等
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm  # 导入pandas._testing模块并使用tm作为别名


@pytest.mark.parametrize(  # 使用pytest的参数化装饰器，定义测试参数化
    "first_slice,second_slice",
    [
        [[2, None], [None, -5]],  # 第一组参数
        [[None, 0], [None, -5]],  # 第二组参数
        [[None, -5], [None, 0]],  # 第三组参数
        [[None, 0], [None, 0]],   # 第四组参数
    ],
)
@pytest.mark.parametrize("fill", [None, -1])  # 测试参数化，填充值为None或-1
def test_align(datetime_series, first_slice, second_slice, join_type, fill):
    a = datetime_series[slice(*first_slice)]  # 对日期时间序列进行切片操作，得到a
    b = datetime_series[slice(*second_slice)]  # 对日期时间序列进行切片操作，得到b

    aa, ab = a.align(b, join=join_type, fill_value=fill)  # 使用align方法对a和b进行对齐操作

    join_index = a.index.join(b.index, how=join_type)  # 获取a和b的索引合并结果
    if fill is not None:
        diff_a = aa.index.difference(join_index)  # 找到aa中不在join_index中的索引
        diff_b = ab.index.difference(join_index)  # 找到ab中不在join_index中的索引
        if len(diff_a) > 0:
            assert (aa.reindex(diff_a) == fill).all()  # 如果有差异，检查填充值是否符合预期
        if len(diff_b) > 0:
            assert (ab.reindex(diff_b) == fill).all()  # 如果有差异，检查填充值是否符合预期

    ea = a.reindex(join_index)  # 使用join_index重新索引a
    eb = b.reindex(join_index)  # 使用join_index重新索引b

    if fill is not None:
        ea = ea.fillna(fill)  # 如果fill不为None，用填充值填充ea中的缺失值
        eb = eb.fillna(fill)  # 如果fill不为None，用填充值填充eb中的缺失值

    tm.assert_series_equal(aa, ea)  # 使用pandas._testing中的方法检查aa和ea是否相等
    tm.assert_series_equal(ab, eb)  # 使用pandas._testing中的方法检查ab和eb是否相等
    assert aa.name == "ts"  # 检查aa的名称是否为"ts"
    assert ea.name == "ts"  # 检查ea的名称是否为"ts"
    assert ab.name == "ts"  # 检查ab的名称是否为"ts"
    assert eb.name == "ts"  # 检查eb的名称是否为"ts"


def test_align_nocopy(datetime_series):
    b = datetime_series[:5].copy()  # 复制日期时间序列的前5个元素到b

    # do copy
    a = datetime_series.copy()  # 复制整个日期时间序列到a
    ra, _ = a.align(b, join="left")  # 使用align方法左对齐a和b
    ra[:5] = 5  # 将ra的前5个元素设置为5
    assert not (a[:5] == 5).any()  # 检查a的前5个元素是否都不等于5

    # do not copy
    a = datetime_series.copy()  # 再次复制整个日期时间序列到a
    ra, _ = a.align(b, join="left")  # 使用align方法左对齐a和b
    ra[:5] = 5  # 将ra的前5个元素设置为5
    assert not (a[:5] == 5).any()  # 检查a的前5个元素是否都不等于5

    # do copy
    a = datetime_series.copy()  # 再次复制整个日期时间序列到a
    b = datetime_series[:5].copy()  # 复制日期时间序列的前5个元素到b的副本
    _, rb = a.align(b, join="right")  # 使用align方法右对齐a和b
    rb[:3] = 5  # 将rb的前3个元素设置为5
    assert not (b[:3] == 5).any()  # 检查b的前3个元素是否都不等于5

    # do not copy
    a = datetime_series.copy()  # 再次复制整个日期时间序列到a
    b = datetime_series[:5].copy()  # 复制日期时间序列的前5个元素到b的副本
    _, rb = a.align(b, join="right")  # 使用align方法右对齐a和b
    rb[:2] = 5  # 将rb的前2个元素设置为5
    assert not (b[:2] == 5).any()  # 检查b的前2个元素是否都不等于5


def test_align_same_index(datetime_series):
    a, b = datetime_series.align(datetime_series)  # 对相同的日期时间序列进行对齐操作
    assert a.index.is_(datetime_series.index)  # 检查a的索引是否是datetime_series的索引
    assert b.index.is_(datetime_series.index)  # 检查b的索引是否是datetime_series的索引

    a, b = datetime_series.align(datetime_series)  # 再次对相同的日期时间序列进行对齐操作
    assert a.index is not datetime_series.index  # 检查a的索引是否不是datetime_series的索引
    assert b.index is not datetime_series.index  # 检查b的索引是否不是datetime_series的索引
    assert a.index.is_(datetime_series.index)  # 检查a的索引是否是datetime_series的索引
    assert b.index.is_(datetime_series.index)  # 检查b的索引是否是datetime_series的索引


def test_align_multiindex():
    # GH 10665

    midx = pd.MultiIndex.from_product(  # 创建一个多级索引对象midx
        [range(2), range(3), range(2)], names=("a", "b", "c")
    )
    idx = pd.Index(range(2), name="b")  # 创建一个单级索引对象idx
    s1 = Series(np.arange(12, dtype="int64"), index=midx)  # 创建一个多级索引的Series对象s1
    s2 = Series(np.arange(2, dtype="int64"), index=idx)  # 创建一个单级索引的Series对象s2

    # these must be the same results (but flipped)
    res1l, res1r = s1.align(s2, join="left")  # 使用左对齐将s1和s2对齐
    res2l, res2r = s2.align(s1, join="right")  # 使用右对齐将s2和s1对齐

    expl = s1  # 期望的结果是s1
    tm.assert_series_equal(expl, res1l)  # 使用pandas._testing中的方法检查res1l和expl是否相等
    tm.assert_series_equal(expl, res2r)  # 使用pandas._testing中的方法检查res2r和expl是否相等
    # 创建一个 Series 对象，包含重复的数值和 NaN 值，索引使用 midx
    expr = Series([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
    # 使用测试模块 tm 检查 expr 和 res1r 是否相等
    tm.assert_series_equal(expr, res1r)
    # 使用测试模块 tm 检查 expr 和 res2l 是否相等

    tm.assert_series_equal(expr, res2l)

    # 调用 align 方法，将 s1 和 s2 对齐，使用右连接方式，分别返回对齐后的结果 res1l 和 res1r
    res1l, res1r = s1.align(s2, join="right")
    # 调用 align 方法，将 s2 和 s1 对齐，使用左连接方式，分别返回对齐后的结果 res2l 和 res2r
    res2l, res2r = s2.align(s1, join="left")

    # 创建一个包含多级索引的 Series 对象 exp_idx
    exp_idx = pd.MultiIndex.from_product(
        [range(2), range(2), range(2)], names=("a", "b", "c")
    )
    # 使用 exp_idx 作为索引创建 Series 对象 expl
    expl = Series([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
    # 使用测试模块 tm 检查 expl 和 res1l 是否相等
    tm.assert_series_equal(expl, res1l)
    # 使用测试模块 tm 检查 expl 和 res2r 是否相等
    tm.assert_series_equal(expl, res2r)

    # 创建一个包含重复数值的 Series 对象 expr，使用 exp_idx 作为索引
    expr = Series([0, 0, 1, 1] * 2, index=exp_idx)
    # 使用测试模块 tm 检查 expr 和 res1r 是否相等
    tm.assert_series_equal(expr, res1r)
    # 使用测试模块 tm 检查 expr 和 res2l 是否相等
    tm.assert_series_equal(expr, res2l)
# 定义一个测试函数，用于验证在时区不匹配的情况下，两个时间序列的对齐操作
def test_align_dt64tzindex_mismatched_tzs():
    # 创建一个日期范围，频率为每小时，时区为US/Eastern
    idx1 = date_range("2001", periods=5, freq="h", tz="US/Eastern")
    # 生成一个随机数序列，长度与日期范围相同，作为数据，创建时间序列
    ser = Series(np.random.default_rng(2).standard_normal(len(idx1)), index=idx1)
    # 将时间序列的时区转换为US/Central
    ser_central = ser.tz_convert("US/Central")
    # 不同的时区转换到UTC

    # 对两个时间序列进行对齐操作，返回对齐后的两个新序列
    new1, new2 = ser.align(ser_central)
    # 断言第一个新序列的索引时区为UTC
    assert new1.index.tz is timezone.utc
    # 断言第二个新序列的索引时区为UTC


# 定义一个测试函数，用于验证在PeriodIndex情况下的对齐操作
def test_align_periodindex(join_type):
    # 创建一个时间段范围，从2000年1月1日到2010年1月1日，频率为每年
    rng = period_range("1/1/2000", "1/1/2010", freq="Y")
    # 生成一个随机数序列，长度与时间段范围相同，作为数据，创建时间序列
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    # TODO: 断言某些内容？
    # 对时间序列与其每两个元素对齐，指定连接方式为join_type


# 定义一个测试函数，用于验证在MultiIndex情况下左侧序列少于右侧序列级别的对齐操作
def test_align_left_fewer_levels():
    # 创建一个左侧序列，包含值为2的单元素Series，索引为MultiIndex，其中包含元组(1, 3)，级别名为"a", "c"
    left = Series([2], index=pd.MultiIndex.from_tuples([(1, 3)], names=["a", "c"]))
    # 创建一个右侧序列，包含值为1的单元素Series，索引为MultiIndex，其中包含元组(1, 2, 3)，级别名为"a", "b", "c"
    right = Series([1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"]))
    # 对左右序列进行对齐操作，返回对齐后的左右两个结果序列
    result_left, result_right = left.align(right)

    # 创建预期的右侧结果序列，包含值为1的单元素Series，索引为MultiIndex，元组为(1, 3, 2)，级别名为"a", "c", "b"
    expected_right = Series([1], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=["a", "c", "b"]))
    # 创建预期的左侧结果序列，包含值为2的单元素Series，索引为MultiIndex，元组为(1, 3, 2)，级别名为"a", "c", "b"
    expected_left = Series([2], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=["a", "c", "b"]))
    # 使用测试工具(assert_series_equal)断言对齐后的左侧结果与预期左侧结果相等
    tm.assert_series_equal(result_left, expected_left)
    # 使用测试工具(assert_series_equal)断言对齐后的右侧结果与预期右侧结果相等


# 定义一个测试函数，用于验证在MultiIndex情况下左侧序列级别名称不完全匹配右侧序列的对齐操作
def test_align_left_different_named_levels():
    # 创建一个左侧序列，包含值为2的单元素Series，索引为MultiIndex，元组为(1, 4, 3)，级别名为"a", "d", "c"
    left = Series([2], index=pd.MultiIndex.from_tuples([(1, 4, 3)], names=["a", "d", "c"]))
    # 创建一个右侧序列，包含值为1的单元素Series，索引为MultiIndex，元组为(1, 2, 3)，级别名为"a", "b", "c"
    right = Series([1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"]))
    # 对左右序列进行对齐操作，返回对齐后的左右两个结果序列
    result_left, result_right = left.align(right)

    # 创建预期的左侧结果序列，包含值为2的单元素Series，索引为MultiIndex，元组为(1, 4, 3, 2)，级别名为"a", "d", "c", "b"
    expected_left = Series([2], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=["a", "d", "c", "b"]))
    # 创建预期的右侧结果序列，包含值为1的单元素Series，索引为MultiIndex，元组为(1, 4, 3, 2)，级别名为"a", "d", "c", "b"
    expected_right = Series([1], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=["a", "d", "c", "b"]))
    # 使用测试工具(assert_series_equal)断言对齐后的左侧结果与预期左侧结果相等
    tm.assert_series_equal(result_left, expected_left)
    # 使用测试工具(assert_series_equal)断言对齐后的右侧结果与预期右侧结果相等
```