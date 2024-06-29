# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_integrity.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from pandas._libs import index as libindex  # 导入pandas库中的索引模块

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike  # 从pandas核心数据类型转换模块导入函数

import pandas as pd  # 导入pandas库并使用pd作为别名
from pandas import (  # 从pandas库中导入多个子模块
    Index,
    IntervalIndex,
    MultiIndex,
    RangeIndex,
)
import pandas._testing as tm  # 导入pandas测试模块


def test_labels_dtypes():
    # GH 8456
    # 创建一个多级索引i，包含元组("A", 1)和("A", 2)
    i = MultiIndex.from_tuples([("A", 1), ("A", 2)])
    # 断言第一个代码数组的数据类型为"int8"
    assert i.codes[0].dtype == "int8"
    # 断言第二个代码数组的数据类型为"int8"
    assert i.codes[1].dtype == "int8"

    # 创建一个多级索引i，包含["a"]和range(40)
    i = MultiIndex.from_product([["a"], range(40)])
    # 断言第二个代码数组的数据类型为"int8"
    assert i.codes[1].dtype == "int8"

    # 创建一个多级索引i，包含["a"]和range(400)
    i = MultiIndex.from_product([["a"], range(400)])
    # 断言第二个代码数组的数据类型为"int16"
    assert i.codes[1].dtype == "int16"

    # 创建一个多级索引i，包含["a"]和range(40000)
    i = MultiIndex.from_product([["a"], range(40000)])
    # 断言第二个代码数组的数据类型为"int32"
    assert i.codes[1].dtype == "int32"

    # 创建一个多级索引i，包含["a"]和range(1000)
    i = MultiIndex.from_product([["a"], range(1000)])
    # 断言所有代码数组的值都大于等于0
    assert (i.codes[0] >= 0).all()
    assert (i.codes[1] >= 0).all()


def test_values_boxed():
    # 创建元组列表tuples
    tuples = [
        (1, pd.Timestamp("2000-01-01")),
        (2, pd.NaT),
        (3, pd.Timestamp("2000-01-03")),
        (1, pd.Timestamp("2000-01-04")),
        (2, pd.Timestamp("2000-01-02")),
        (3, pd.Timestamp("2000-01-03")),
    ]
    # 从元组列表创建多级索引result
    result = MultiIndex.from_tuples(tuples)
    # 从列表类似对象构造预期结果expected
    expected = construct_1d_object_array_from_listlike(tuples)
    # 断言result的值与expected的numpy数组相等
    tm.assert_numpy_array_equal(result.values, expected)
    # 检查对于包装值的代码分支产生相同的结果
    tm.assert_numpy_array_equal(result.values[:4], result[:4].values)


def test_values_multiindex_datetimeindex():
    # 测试确保我们触及MI.values的包装/未包装部分
    ints = np.arange(10**18, 10**18 + 5)
    naive = pd.DatetimeIndex(ints)

    aware = pd.DatetimeIndex(ints, tz="US/Central")

    # 从数组创建多级索引idx
    idx = MultiIndex.from_arrays([naive, aware])
    # 获取idx的值并赋给result
    result = idx.values

    # 从result的第一列创建DatetimeIndex对象outer
    outer = pd.DatetimeIndex([x[0] for x in result])
    # 断言outer与naive相等
    tm.assert_index_equal(outer, naive)

    # 从result的第二列创建DatetimeIndex对象inner
    inner = pd.DatetimeIndex([x[1] for x in result])
    # 断言inner与aware相等
    tm.assert_index_equal(inner, aware)

    # n_lev > n_lab
    # 对idx的前两行获取值并赋给result
    result = idx[:2].values

    # 从result的第一列创建DatetimeIndex对象outer
    outer = pd.DatetimeIndex([x[0] for x in result])
    # 断言outer与naive的前两行相等
    tm.assert_index_equal(outer, naive[:2])

    # 从result的第二列创建DatetimeIndex对象inner
    inner = pd.DatetimeIndex([x[1] for x in result])
    # 断言inner与aware的前两行相等
    tm.assert_index_equal(inner, aware[:2])


def test_values_multiindex_periodindex():
    # 测试确保我们触及MI.values的包装/未包装部分
    ints = np.arange(2007, 2012)
    pidx = pd.PeriodIndex(ints, freq="D")

    # 从数组创建多级索引idx
    idx = MultiIndex.from_arrays([ints, pidx])
    # 获取idx的值并赋给result
    result = idx.values

    # 从result的第一列创建Index对象outer
    outer = Index([x[0] for x in result])
    # 断言outer与整数数组ints相等
    tm.assert_index_equal(outer, Index(ints, dtype=np.int64))

    # 从result的第二列创建PeriodIndex对象inner
    inner = pd.PeriodIndex([x[1] for x in result])
    # 断言inner与pidx相等
    tm.assert_index_equal(inner, pidx)

    # n_lev > n_lab
    # 对idx的前两行获取值并赋给result
    result = idx[:2].values

    # 从result的第一列创建Index对象outer
    outer = Index([x[0] for x in result])
    # 断言outer与整数数组ints的前两行相等
    tm.assert_index_equal(outer, Index(ints[:2], dtype=np.int64))

    # 从result的第二列创建PeriodIndex对象inner
    inner = pd.PeriodIndex([x[1] for x in result])
    # 断言inner与pidx的前两行相等
    tm.assert_index_equal(inner, pidx[:2])


def test_consistency():
    # 需要构建一个溢出
    pass
    # 创建一个包含整数 0 到 69999 的列表，作为 MultiIndex 的主轴
    major_axis = list(range(70000))
    
    # 创建一个包含整数 0 到 9 的列表，作为 MultiIndex 的次轴
    minor_axis = list(range(10))
    
    # 使用 numpy 的 arange 函数生成一个包含 0 到 69999 的数组，作为主轴的编码
    major_codes = np.arange(70000)
    
    # 使用 numpy 的 repeat 函数生成一个重复的 0 到 9 的数组，总长度为 70000，作为次轴的编码
    minor_codes = np.repeat(range(10), 7000)
    
    # 创建 MultiIndex 对象，指定主轴和次轴的 levels 和 codes
    index = MultiIndex(
        levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
    )
    
    # 使用不一致的编码重新创建 MultiIndex 对象
    major_codes = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1])
    index = MultiIndex(
        levels=[major_axis, minor_axis], codes=[major_codes, minor_codes]
    )
    
    # 断言 MultiIndex 对象的唯一性属性 is_unique 应为 False
    assert index.is_unique is False
@pytest.mark.slow
# 标记此测试为慢速测试
def test_hash_collisions(monkeypatch):
    # 使用 monkeypatch 来修改 libindex 模块中的 _SIZE_CUTOFF 值
    size_cutoff = 50
    with monkeypatch.context() as m:
        m.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
        # 创建一个 MultiIndex 对象，包含由两个 np.arange(8) 生成的笛卡尔积
        index = MultiIndex.from_product(
            [np.arange(8), np.arange(8)], names=["one", "two"]
        )
        # 调用 MultiIndex 对象的 get_indexer 方法，返回索引数组 result
        result = index.get_indexer(index.values)
        # 使用 tm.assert_numpy_array_equal 检查 result 是否等于 np.arange(len(index), dtype="intp")
        tm.assert_numpy_array_equal(result, np.arange(len(index), dtype="intp"))

        # 对 MultiIndex 对象进行索引测试
        for i in [0, 1, len(index) - 2, len(index) - 1]:
            result = index.get_loc(index[i])
            # 使用 assert 断言 result 是否等于 i
            assert result == i


def test_dims():
    pass


def test_take_invalid_kwargs():
    # 创建一个 MultiIndex 对象
    vals = [["A", "B"], [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]]
    idx = MultiIndex.from_product(vals, names=["str", "dt"])
    indices = [1, 2]

    # 测试 take 方法传入未预期的关键字参数 'foo' 是否会引发 TypeError 异常
    msg = r"take\(\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        idx.take(indices, foo=2)

    # 测试 take 方法传入不支持的 'out' 参数是否会引发 ValueError 异常
    msg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        idx.take(indices, out=indices)

    # 测试 take 方法传入不支持的 'mode' 参数是否会引发 ValueError 异常
    msg = "the 'mode' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        idx.take(indices, mode="clip")


def test_isna_behavior(idx):
    # 测试 isna 方法对于 MultiIndex 对象是否抛出 NotImplementedError 异常
    # 如果 MultiIndex 的表示方式发生变化，允许 isna(MI) 则应考虑更新此处注释
    msg = "isna is not defined for MultiIndex"
    with pytest.raises(NotImplementedError, match=msg):
        pd.isna(idx)


def test_large_multiindex_error(monkeypatch):
    # GH12527 测试用例
    size_cutoff = 50
    with monkeypatch.context() as m:
        m.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
        # 创建一个小于 size_cutoff - 1 的 DataFrame，使用 MultiIndex 对象作为索引
        df_below_cutoff = pd.DataFrame(
            1,
            index=MultiIndex.from_product([[1, 2], range(size_cutoff - 1)]),
            columns=["dest"],
        )
        # 测试对小于 size_cutoff - 1 的索引范围进行 loc 操作是否会引发 KeyError 异常
        with pytest.raises(KeyError, match=r"^\(-1, 0\)$"):
            df_below_cutoff.loc[(-1, 0), "dest"]
        with pytest.raises(KeyError, match=r"^\(3, 0\)$"):
            df_below_cutoff.loc[(3, 0), "dest"]
        
        # 创建一个大于 size_cutoff + 1 的 DataFrame，使用 MultiIndex 对象作为索引
        df_above_cutoff = pd.DataFrame(
            1,
            index=MultiIndex.from_product([[1, 2], range(size_cutoff + 1)]),
            columns=["dest"],
        )
        # 测试对大于 size_cutoff + 1 的索引范围进行 loc 操作是否会引发 KeyError 异常
        with pytest.raises(KeyError, match=r"^\(-1, 0\)$"):
            df_above_cutoff.loc[(-1, 0), "dest"]
        with pytest.raises(KeyError, match=r"^\(3, 0\)$"):
            df_above_cutoff.loc[(3, 0), "dest"]


def test_mi_hashtable_populated_attribute_error(monkeypatch):
    # GH 18165 测试用例
    monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 50)
    r = range(50)
    # 创建一个 DataFrame 对象，包含两个列 'a' 和 'b'，使用 MultiIndex 对象作为索引
    df = pd.DataFrame({"a": r, "b": r}, index=MultiIndex.from_arrays([r, r]))

    # 测试对 DataFrame 的某列使用未定义的属性 'foo' 是否会引发 AttributeError 异常
    msg = "'Series' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        df["a"].foo()


def test_can_hold_identifiers(idx):
    # 测试 MultiIndex 对象的 _can_hold_identifiers_and_holds_name 方法
    key = idx[0]
    # 使用 assert 断言 _can_hold_identifiers_and_holds_name 方法返回 True
    assert idx._can_hold_identifiers_and_holds_name(key) is True


def test_metadata_immutable(idx):
    # 这个测试函数尚未实现，因此暂时留空
    pass
    # 获取索引对象中的 levels 和 codes 属性
    levels, codes = idx.levels, idx.codes
    
    # 创建一个正则表达式对象，用于匹配不支持可变操作的错误信息
    mutable_regex = re.compile("does not support mutable operations")
    
    # 使用 pytest 检查是否能够修改顶层索引 levels[0]，预期会抛出 TypeError 异常并匹配 mutable_regex 正则表达式
    with pytest.raises(TypeError, match=mutable_regex):
        levels[0] = levels[0]
    
    # 同样地，检查是否能够修改基础层索引 levels[0][0]，预期会抛出 TypeError 异常并匹配 mutable_regex 正则表达式
    with pytest.raises(TypeError, match=mutable_regex):
        levels[0][0] = levels[0][0]
    
    # 检查是否能够修改 codes[0]，预期会抛出 TypeError 异常并匹配 mutable_regex 正则表达式
    with pytest.raises(TypeError, match=mutable_regex):
        codes[0] = codes[0]
    
    # 检查是否能够修改 codes[0][0]，预期会抛出 ValueError 异常并匹配 "assignment destination is read-only" 字符串
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        codes[0][0] = codes[0][0]
    
    # 获取索引对象中的 names 属性，并检查是否能够修改 names[0]，预期会抛出 TypeError 异常并匹配 mutable_regex 正则表达式
    names = idx.names
    with pytest.raises(TypeError, match=mutable_regex):
        names[0] = names[0]
# 测试函数，验证在设定级别后重置属性
def test_level_setting_resets_attributes():
    # 创建一个多级索引对象，包含两个级别
    ind = MultiIndex.from_arrays([["A", "A", "B", "B", "B"], [1, 2, 1, 2, 3]])
    # 断言索引对象是否单调递增
    assert ind.is_monotonic_increasing
    # 设定索引级别，如果重置缓存不正确，则断言不再单调递增
    ind = ind.set_levels([["A", "B"], [1, 3, 2]])
    assert not ind.is_monotonic_increasing


# 测试特定情况下的索引错误回退问题
def test_rangeindex_fallback_coercion_bug():
    # 创建两个包含0到99的10x10数据帧
    df1 = pd.DataFrame(np.arange(100).reshape((10, 10)))
    df2 = pd.DataFrame(np.arange(100).reshape((10, 10)))
    # 沿列轴拼接数据帧，创建包含多级索引的数据帧
    df = pd.concat(
        {"df1": df1.stack(), "df2": df2.stack()},
        axis=1,
    )
    # 设定索引的名称
    df.index.names = ["fizz", "buzz"]

    # 创建期望的数据帧，包含相同的数据和索引结构
    expected = pd.DataFrame(
        {"df2": np.arange(100), "df1": np.arange(100)},
        index=MultiIndex.from_product([range(10), range(10)], names=["fizz", "buzz"]),
    )
    # 使用测试工具断言两个数据帧是否相似
    tm.assert_frame_equal(df, expected, check_like=True)

    # 获取索引中级别为"fizz"的值
    result = df.index.get_level_values("fizz")
    # 创建预期的索引对象，重复10次0到9的整数
    expected = Index(np.arange(10, dtype=np.int64), name="fizz").repeat(10)
    # 使用测试工具断言结果索引是否与预期相同
    tm.assert_index_equal(result, expected)

    # 获取索引中级别为"buzz"的值
    result = df.index.get_level_values("buzz")
    # 创建预期的索引对象，将0到9的整数重复10次
    expected = Index(np.tile(np.arange(10, dtype=np.int64), 10), name="buzz")
    # 使用测试工具断言结果索引是否与预期相同
    tm.assert_index_equal(result, expected)


# 测试索引对象的内存使用情况计算
def test_memory_usage(idx):
    # 计算索引对象的内存使用情况
    result = idx.memory_usage()
    # 如果索引长度不为0，执行以下逻辑
    if len(idx):
        # 获取索引第一个元素的位置
        idx.get_loc(idx[0])
        # 再次计算索引对象的内存使用情况
        result2 = idx.memory_usage()
        # 使用深度计算模式再次计算索引对象的内存使用情况
        result3 = idx.memory_usage(deep=True)

        # 对于不是RangeIndex或IntervalIndex类型的索引，断言第二次计算的内存使用情况大于第一次
        if not isinstance(idx, (RangeIndex, IntervalIndex)):
            assert result2 > result

        # 如果推断类型为"object"，断言深度计算模式的内存使用情况大于第二次计算的结果
        if idx.inferred_type == "object":
            assert result3 > result2

    else:
        # 对于长度为0的索引，断言内存使用情况为0
        assert result == 0


# 测试索引对象的级别数
def test_nlevels(idx):
    # 断言索引对象的级别数为2
    assert idx.nlevels == 2
```