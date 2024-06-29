# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_info.py`

```
# 从 io 模块导入 StringIO 类，用于字符串缓冲
from io import StringIO
# 从 string 模块导入 ascii_uppercase 字符串，包含所有大写字母
from string import ascii_uppercase
# 导入 textwrap 模块，用于格式化文本输出
import textwrap

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 从 pandas.compat 模块导入 PYPY 常量
from pandas.compat import PYPY
# 从 pandas 库导入以下对象
from pandas import (
    CategoricalIndex,   # 导入 CategoricalIndex 对象
    MultiIndex,          # 导入 MultiIndex 对象
    Series,              # 导入 Series 对象
    date_range,          # 导入 date_range 函数
)


# 测试函数，验证 Series 对象转换为分类数据的基本功能
def test_info_categorical_column_just_works():
    # 设定数据长度
    n = 2500
    # 生成随机整数数组作为数据源
    data = np.array(list("abcdefghij")).take(
        np.random.default_rng(2).integers(0, 10, size=n, dtype=int)
    )
    # 创建 Series 对象并转换为分类数据类型
    s = Series(data).astype("category")
    # 检查是否有缺失值
    s.isna()
    # 创建字符串缓冲区
    buf = StringIO()
    # 获取 Series 的信息并写入缓冲区
    s.info(buf=buf)

    # 通过条件筛选出符合条件的子集
    s2 = s[s == "d"]
    # 重新创建字符串缓冲区
    buf = StringIO()
    # 获取子集的信息并写入缓冲区
    s2.info(buf=buf)


# 测试函数，验证使用 CategoricalIndex 的 Series 对象的 info 方法
def test_info_categorical():
    # 创建 CategoricalIndex 对象
    idx = CategoricalIndex(["a", "b"])
    # 创建包含零元素的 Series 对象，使用 CategoricalIndex 作为索引
    s = Series(np.zeros(2), index=idx)
    # 创建字符串缓冲区
    buf = StringIO()
    # 获取 Series 的信息并写入缓冲区
    s.info(buf=buf)


# 参数化测试函数，验证 Series 对象的信息输出
@pytest.mark.parametrize("verbose", [True, False])
def test_info_series(lexsorted_two_level_string_multiindex, verbose):
    # 获取排序后的两级字符串 MultiIndex 对象
    index = lexsorted_two_level_string_multiindex
    # 创建 Series 对象，使用 MultiIndex 作为索引
    ser = Series(range(len(index)), index=index, name="sth")
    # 创建字符串缓冲区
    buf = StringIO()
    # 获取 Series 的信息并写入缓冲区
    ser.info(verbose=verbose, buf=buf)
    # 从缓冲区获取结果字符串
    result = buf.getvalue()

    # 期望的输出结果
    expected = textwrap.dedent(
        """\
        <class 'pandas.core.series.Series'>
        MultiIndex: 10 entries, ('foo', 'one') to ('qux', 'three')
        """
    )
    # 如果 verbose 参数为 True，则追加详细信息
    if verbose:
        expected += textwrap.dedent(
            """\
            Series name: sth
            Non-Null Count  Dtype
            --------------  -----
            10 non-null     int64
            """
        )
    # 追加数据类型、内存使用信息
    expected += textwrap.dedent(
        f"""\
        dtypes: int64(1)
        memory usage: {ser.memory_usage()}.0+ bytes
        """
    )
    # 断言实际输出与期望输出一致
    assert result == expected


# 测试函数，验证 Series 对象的内存使用信息输出
def test_info_memory():
    # 创建包含整数元素的 Series 对象
    s = Series([1, 2], dtype="i8")
    # 创建字符串缓冲区
    buf = StringIO()
    # 获取 Series 的信息并写入缓冲区
    s.info(buf=buf)
    # 从缓冲区获取结果字符串
    result = buf.getvalue()
    # 获取 Series 的内存使用量
    memory_bytes = float(s.memory_usage())
    # 期望的输出结果
    expected = textwrap.dedent(
        f"""\
    <class 'pandas.core.series.Series'>
    RangeIndex: 2 entries, 0 to 1
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    2 non-null      int64
    dtypes: int64(1)
    memory usage: {memory_bytes} bytes
    """
    )
    # 断言实际输出与期望输出一致
    assert result == expected


# 测试函数，验证 Series 对象的信息输出在数据宽度较大时的异常情况
def test_info_wide():
    # 创建包含标准正态分布随机数的 Series 对象
    s = Series(np.random.default_rng(2).standard_normal(101))
    # 期望的异常信息
    msg = "Argument `max_cols` can only be passed in DataFrame.info, not Series.info"
    # 断言在设置 max_cols 参数时会引发 ValueError 异常，并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        s.info(max_cols=1)


# 测试函数，验证 Series 对象的信息输出包含数据类型信息
def test_info_shows_dtypes():
    # 不同数据类型的列表
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    # 数据集大小
    n = 10
    # 遍历不同数据类型
    for dtype in dtypes:
        # 创建具有指定数据类型的 Series 对象
        s = Series(np.random.default_rng(2).integers(2, size=n).astype(dtype))
        # 创建字符串缓冲区
        buf = StringIO()
        # 获取 Series 的信息并写入缓冲区
        s.info(buf=buf)
        # 从缓冲区获取结果字符串
        res = buf.getvalue()
        # 期望的输出结果
        name = f"{n:d} non-null     {dtype}"
        # 断言数据类型信息在输出结果中存在
        assert name in res


# 参数化测试函数，验证在 PyPy 环境下 deep=True 参数对结果的影响
@pytest.mark.xfail(PYPY, reason="on PyPy deep=True doesn't change result")
def test_info_memory_usage_deep_not_pypy():
    # 此测试在 PyPy 环境下会失败，因为 deep=True 参数不会改变结果
    # 创建一个 Pandas Series 对象，该对象包含一个字典 {"a": [1]}，并指定索引为 ["foo"]
    s_with_object_index = Series({"a": [1]}, index=["foo"])
    # 使用 assert 语句检查带有对象索引的 Series 对象的内存使用量是否大于没有对象索引的 Series 对象的内存使用量，
    # 使用 index=True 表示要考虑索引的内存占用，deep=True 表示要深度遍历对象内部的内存占用情况
    assert s_with_object_index.memory_usage(index=True, deep=True) > s_with_object_index.memory_usage(index=True)
    
    # 创建一个 Pandas Series 对象，该对象包含一个字典 {"a": ["a"]}
    s_object = Series({"a": ["a"]})
    # 使用 assert 语句检查带有对象的 Series 对象的深度内存使用量是否大于没有对象的 Series 对象的内存使用量，
    # deep=True 表示要深度遍历对象内部的内存占用情况
    assert s_object.memory_usage(deep=True) > s_object.memory_usage()
# 使用 pytest.mark.xfail 装饰器标记测试函数，表示在满足条件时测试预期会失败
@pytest.mark.xfail(not PYPY, reason="on PyPy deep=True does not change result")
# 定义测试函数 test_info_memory_usage_deep_pypy，用于测试内存使用情况的方法在特定条件下的行为
def test_info_memory_usage_deep_pypy():
    # 创建一个包含对象索引的 Series 对象
    s_with_object_index = Series({"a": [1]}, index=["foo"])
    # 断言使用 deep=True 时的内存使用情况与不使用 deep=True 时相同
    assert s_with_object_index.memory_usage(
        index=True, deep=True
    ) == s_with_object_index.memory_usage(index=True)

    # 创建一个包含对象类型数据的 Series 对象
    s_object = Series({"a": ["a"]})
    # 断言使用 deep=True 时的内存使用情况与不使用 deep=True 时相同
    assert s_object.memory_usage(deep=True) == s_object.memory_usage()


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数
@pytest.mark.parametrize(
    "index, plus",
    [
        ([1, 2, 3], False),  # 索引为整数列表，不含特殊符号
        (list("ABC"), True),  # 索引为字符列表，含有特殊符号'+'
        (MultiIndex.from_product([range(3), range(3)]), False),  # 多级索引，不含特殊符号
        (MultiIndex.from_product([range(3), ["foo", "bar"]]), True),  # 多级索引，含有特殊符号'+'
    ],
)
# 定义参数化测试函数 test_info_memory_usage_qualified，用于测试 Series 对象的信息输出
def test_info_memory_usage_qualified(index, plus):
    # 创建一个 Series 对象，指定索引
    buf = StringIO()
    series = Series(1, index=index)
    # 将 Series 对象的信息输出到缓冲区
    series.info(buf=buf)
    # 根据参数 plus 断言是否含有特殊符号'+'，验证信息输出是否符合预期
    if plus:
        assert "+" in buf.getvalue()
    else:
        assert "+" not in buf.getvalue()


# 定义测试函数 test_info_memory_usage_bug_on_multiindex，用于验证多级索引情况下的内存使用 bug
def test_info_memory_usage_bug_on_multiindex():
    # GH 14308
    # 创建一个多级索引 Series 对象
    # 多级索引应该不会在内存使用分析时实例化 .values
    N = 100
    M = len(ascii_uppercase)
    index = MultiIndex.from_product(
        [list(ascii_uppercase), date_range("20160101", periods=N)],
        names=["id", "date"],
    )
    # 创建具有随机数据的 Series 对象
    s = Series(np.random.default_rng(2).standard_normal(N * M), index=index)

    # 对 Series 对象进行 unstack 操作
    unstacked = s.unstack("id")
    # 断言原始 Series 对象和 unstacked 后的对象在使用 deep=True 时的内存使用情况
    assert s.values.nbytes == unstacked.values.nbytes
    assert s.memory_usage(deep=True) > unstacked.memory_usage(deep=True).sum()

    # 断言 unstacked 对象的深度内存使用总和应该比原始 Series 对象少于2000字节
    diff = unstacked.memory_usage(deep=True).sum() - s.memory_usage(deep=True)
    assert diff < 2000
```