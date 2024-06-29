# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_put.py`

```
# 导入所需的库和模块
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    concat,
    date_range,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)
from pandas.util import _test_decorators as td

# 标记当前模块为单CPU环境测试
pytestmark = pytest.mark.single_cpu


def test_format_type(tmp_path, setup_path):
    # 创建一个简单的DataFrame对象
    df = DataFrame({"A": [1, 2]})
    
    # 使用HDFStore打开指定路径的存储
    with HDFStore(tmp_path / setup_path) as store:
        # 将DataFrame以"fixed"格式存储在HDFStore中的键"a"下
        store.put("a", df, format="fixed")
        # 将DataFrame以"table"格式存储在HDFStore中的键"b"下
        store.put("b", df, format="table")
        
        # 断言存储在HDFStore中的键"a"的格式为"fixed"
        assert store.get_storer("a").format_type == "fixed"
        # 断言存储在HDFStore中的键"b"的格式为"table"
        assert store.get_storer("b").format_type == "table"


def test_format_kwarg_in_constructor(tmp_path, setup_path):
    # GH 13291
    
    # 设置错误消息内容
    msg = "format is not a defined argument for HDFStore"
    
    # 使用pytest的raises断言捕获期望的ValueError异常，并验证异常消息匹配msg
    with pytest.raises(ValueError, match=msg):
        # 尝试在实例化HDFStore时传递不支持的参数"format"
        HDFStore(tmp_path / setup_path, format="table")


def test_api_default_format(tmp_path, setup_path):
    # default_format选项的测试
    
    # 使用ensure_clean_store辅助函数打开HDFStore
    with ensure_clean_store(setup_path) as store:
        # 创建一个DataFrame对象
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        # 使用pd.option_context设置上下文管理器，设置默认的IO操作格式为"fixed"
        with pd.option_context("io.hdf.default_format", "fixed"):
            # 在存储中存储DataFrame对象"df"
            _maybe_remove(store, "df")
            store.put("df", df)
            # 断言存储的对象"df"不是表格格式
            assert not store.get_storer("df").is_table
            
            # 设置期望的错误消息
            msg = "Can only append to Tables"
            # 断言在尝试向非表格格式的对象"df2"追加数据时，会抛出ValueError异常
            with pytest.raises(ValueError, match=msg):
                store.append("df2", df)

        # 使用pd.option_context设置上下文管理器，设置默认的IO操作格式为"table"
        with pd.option_context("io.hdf.default_format", "table"):
            # 重新存储DataFrame对象"df"
            _maybe_remove(store, "df")
            store.put("df", df)
            # 断言存储的对象"df"是表格格式
            assert store.get_storer("df").is_table
            
            # 清理存储中的对象"df2"
            _maybe_remove(store, "df2")
            # 向对象"df2"追加DataFrame数据
            store.append("df2", df)
            # 断言存储的对象"df2"是表格格式

    # 使用临时路径设置文件路径
    path = tmp_path / setup_path
    # 创建一个DataFrame对象
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # 使用pd.option_context设置上下文管理器，设置默认的IO操作格式为"fixed"
    with pd.option_context("io.hdf.default_format", "fixed"):
        # 将DataFrame对象"df"以HDF格式存储在指定路径下的键"df"
        df.to_hdf(path, key="df")
        # 使用HDFStore打开指定路径的存储
        with HDFStore(path) as store:
            # 断言存储的对象"df"不是表格格式
            assert not store.get_storer("df").is_table
        # 重新尝试在对象"df2"上追加数据时，验证是否抛出预期的ValueError异常
        with pytest.raises(ValueError, match=msg):
            df.to_hdf(path, key="df2", append=True)

    # 使用pd.option_context设置上下文管理器，设置默认的IO操作格式为"table"
    with pd.option_context("io.hdf.default_format", "table"):
        # 将DataFrame对象"df3"以HDF格式存储在指定路径下的键"df3"
        df.to_hdf(path, key="df3")
        # 使用HDFStore打开指定路径的存储
        with HDFStore(path) as store:
            # 断言存储的对象"df3"是表格格式
            assert store.get_storer("df3").is_table
        # 将DataFrame对象"df4"以HDF格式存储在指定路径下的键"df4"，追加数据
        df.to_hdf(path, key="df4", append=True)
        # 使用HDFStore打开指定路径的存储
        with HDFStore(path) as store:
            # 断言存储的对象"df4"是表格格式
            assert store.get_storer("df4").is_table


def test_put(setup_path):
    # 使用 ensure_clean_store 上下文管理器，确保 setup_path 的存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含 0 到 9 的浮点数 Series 对象 ts，索引从 "2020-01-01" 开始的 10 个日期
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 创建一个 20 行 4 列的随机标准正态分布 DataFrame 对象 df
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=20, freq="B"),
        )
        # 在存储中存储 Series 对象 ts，键为 "a"
        store["a"] = ts
        # 在存储中存储 DataFrame 对象 df 的前 10 行，键为 "b"
        store["b"] = df[:10]
        # 在存储中存储 DataFrame 对象 df 的前 10 行，键为 "foo/bar/bah"
        store["foo/bar/bah"] = df[:10]
        # 在存储中存储 DataFrame 对象 df 的前 10 行，键为 "foo"
        store["foo"] = df[:10]
        # 在存储中存储 DataFrame 对象 df 的前 10 行，键为 "/foo"
        store["/foo"] = df[:10]
        # 在存储中存储 DataFrame 对象 df 的前 10 行，键为 "c"，使用表格格式存储
        store.put("c", df[:10], format="table")

        # 不允许覆盖表格，抛出 ValueError 异常，异常信息为 msg
        msg = "Can only append to Tables"
        with pytest.raises(ValueError, match=msg):
            # 尝试将 df 的后 10 行追加到键为 "b" 的存储中，由于不是表格类型，应抛出异常
            store.put("b", df[10:], append=True)

        # 移除存储中键为 "f" 的内容，如果不存在则忽略
        _maybe_remove(store, "f")
        with pytest.raises(ValueError, match=msg):
            # 尝试将 df 的后 10 行追加到键为 "f" 的存储中，由于不是表格类型，应抛出异常
            store.put("f", df[10:], append=True)

        # 不能向表格类型存储键为 "c" 的内容追加数据，应抛出异常
        with pytest.raises(ValueError, match=msg):
            store.put("c", df[10:], append=True)

        # 覆盖存储中键为 "c" 的内容为 df 的前 10 行，使用表格格式存储
        store.put("c", df[:10], format="table", append=False)
        # 使用 tm.assert_frame_equal 检查存储中键为 "c" 的内容与 df 的前 10 行是否相等
        tm.assert_frame_equal(df[:10], store["c"])
def test_put_string_index(setup_path):
    # 使用 ensure_clean_store 函数确保测试前存储区域的清理工作
    with ensure_clean_store(setup_path) as store:
        # 创建一个 Index 对象，包含 20 个长字符串索引
        index = Index([f"I am a very long string index: {i}" for i in range(20)])
        # 根据索引创建 Series 对象
        s = Series(np.arange(20), index=index)
        # 根据 Series 对象创建 DataFrame 对象
        df = DataFrame({"A": s, "B": s})

        # 将 Series 对象存储到 store 中，验证存储的内容与原始的 Series 对象相等
        store["a"] = s
        tm.assert_series_equal(store["a"], s)

        # 将 DataFrame 对象存储到 store 中，验证存储的内容与原始的 DataFrame 对象相等
        store["b"] = df
        tm.assert_frame_equal(store["b"], df)

        # 创建一个混合长度的 Index 对象，包含一个长字符串和 20 个长字符串索引
        index = Index(
            ["abcdefghijklmnopqrstuvwxyz1234567890"]
            + [f"I am a very long string index: {i}" for i in range(20)]
        )
        # 根据混合长度的 Index 对象创建 Series 对象
        s = Series(np.arange(21), index=index)
        # 根据 Series 对象创建 DataFrame 对象
        df = DataFrame({"A": s, "B": s})
        # 将 Series 对象存储到 store 中，验证存储的内容与原始的 Series 对象相等
        store["a"] = s
        tm.assert_series_equal(store["a"], s)

        # 将 DataFrame 对象存储到 store 中，验证存储的内容与原始的 DataFrame 对象相等
        store["b"] = df
        tm.assert_frame_equal(store["b"], df)


def test_put_compression(setup_path):
    # 使用 ensure_clean_store 函数确保测试前存储区域的清理工作
    with ensure_clean_store(setup_path) as store:
        # 创建一个 DataFrame 对象，包含随机数据，指定列名和日期索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )

        # 将 DataFrame 对象以 table 格式和 zlib 压缩方式存储到 store 中
        store.put("c", df, format="table", complib="zlib")
        # 验证存储的内容与原始的 DataFrame 对象相等
        tm.assert_frame_equal(store["c"], df)

        # 当 format='fixed' 时，尝试使用 zlib 压缩会抛出 ValueError 异常
        msg = "Compression not supported on Fixed format stores"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df, format="fixed", complib="zlib")


@td.skip_if_windows
def test_put_compression_blosc(setup_path):
    # 创建一个 DataFrame 对象，包含随机数据，指定列名和日期索引
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    with ensure_clean_store(setup_path) as store:
        # 当 format='fixed' 时，尝试使用 blosc 压缩会抛出 ValueError 异常
        msg = "Compression not supported on Fixed format stores"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df, format="fixed", complib="blosc")

        # 将 DataFrame 对象以 table 格式和 blosc 压缩方式存储到 store 中
        store.put("c", df, format="table", complib="blosc")
        # 验证存储的内容与原始的 DataFrame 对象相等
        tm.assert_frame_equal(store["c"], df)


def test_put_mixed_type(setup_path, performance_warning):
    # 创建一个 DataFrame 对象，包含随机数据，指定列名和日期索引，以及多种数据类型的列
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 添加额外的列，包含不同的数据类型
    df["obj1"] = "foo"
    df["obj2"] = "bar"
    df["bool1"] = df["A"] > 0
    df["bool2"] = df["B"] > 0
    df["bool3"] = True
    df["int1"] = 1
    df["int2"] = 2
    df["timestamp1"] = Timestamp("20010102").as_unit("ns")
    df["timestamp2"] = Timestamp("20010103").as_unit("ns")
    df["datetime1"] = Timestamp("20010102").as_unit("ns")
    df["datetime2"] = Timestamp("20010103").as_unit("ns")
    df.loc[df.index[3:6], ["obj1"]] = np.nan
    # 对 DataFrame 对象进行内存优化
    df = df._consolidate()
    # 使用 ensure_clean_store 函数创建一个上下文管理器，并指定 setup_path 作为参数
    with ensure_clean_store(setup_path) as store:
        # 调用 _maybe_remove 函数，尝试删除 store 中名为 "df" 的内容（如果存在）
        _maybe_remove(store, "df")
        
        # 使用 tm.assert_produces_warning 函数确保执行下面的代码块会产生 performance_warning 警告
        with tm.assert_produces_warning(performance_warning):
            # 将 DataFrame df 存储到 store 中，键名为 "df"
            store.put("df", df)
        
        # 从 store 中获取键名为 "df" 的数据
        expected = store.get("df")
        
        # 使用 tm.assert_frame_equal 函数比较 expected 和 df 是否相等
        tm.assert_frame_equal(expected, df)
@pytest.mark.parametrize("format", ["table", "fixed"])
@pytest.mark.parametrize(
    "index",
    [
        Index([str(i) for i in range(10)]),  # 创建一个字符串索引对象，包含从 '0' 到 '9' 的索引
        Index(np.arange(10, dtype=float)),  # 创建一个浮点数索引对象，包含从 0.0 到 9.0 的索引
        Index(np.arange(10)),  # 创建一个整数索引对象，包含从 0 到 9 的索引
        date_range("2020-01-01", periods=10),  # 创建一个日期范围索引对象，从 '2020-01-01' 开始，包含 10 个日期
        pd.period_range("2020-01-01", periods=10),  # 创建一个周期范围索引对象，从 '2020-01-01' 开始，包含 10 个周期
    ],
)
def test_store_index_types(setup_path, format, index):
    # GH5386
    # 测试存储不同类型的索引

    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),  # 生成一个10行2列的随机数据帧
            columns=list("AB"),  # 指定列名为 'A' 和 'B'
            index=index,  # 使用给定的索引对象作为数据帧的索引
        )
        _maybe_remove(store, "df")  # 如果存在名为 "df" 的项，尝试删除它
        store.put("df", df, format=format)  # 将数据帧存储到指定格式的数据存储中
        tm.assert_frame_equal(df, store["df"])  # 断言存储的数据帧与原始数据帧相等


def test_column_multiindex(setup_path):
    # GH 4710
    # 正确地重新创建多重索引

    index = MultiIndex.from_tuples(
        [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")], names=["first", "second"]
    )  # 创建一个具有指定名称的多重索引对象

    df = DataFrame(np.arange(12).reshape(3, 4), columns=index)  # 生成一个3行4列的数据帧，列使用多重索引对象作为列名
    expected = df.set_axis(df.index.to_numpy())  # 生成一个期望的数据帧，以确保索引正确

    with ensure_clean_store(setup_path) as store:
        store.put("df", df)  # 将数据帧存储到数据存储中
        tm.assert_frame_equal(
            store["df"], expected, check_index_type=True, check_column_type=True
        )  # 断言存储的数据帧与期望的数据帧相等，同时检查索引类型和列类型

        store.put("df1", df, format="table")  # 使用表格格式将数据帧存储到数据存储中
        tm.assert_frame_equal(
            store["df1"], expected, check_index_type=True, check_column_type=True
        )  # 断言存储的数据帧与期望的数据帧相等，同时检查索引类型和列类型

        # 测试在指定数据列的情况下，尝试存储多重索引的限制
        msg = re.escape("cannot use a multi-index on axis [1] with data_columns ['A']")
        with pytest.raises(ValueError, match=msg):
            store.put("df2", df, format="table", data_columns=["A"])
        msg = re.escape("cannot use a multi-index on axis [1] with data_columns True")
        with pytest.raises(ValueError, match=msg):
            store.put("df3", df, format="table", data_columns=True)

    # 在现有表格上追加多列（参见 GH 6167）
    with ensure_clean_store(setup_path) as store:
        store.append("df2", df)  # 将数据帧追加到现有表格中
        store.append("df2", df)  # 再次将数据帧追加到现有表格中

        tm.assert_frame_equal(store["df2"], concat((df, df)))  # 断言存储的数据帧与预期的数据帧连接后相等

    # 非索引轴的名称
    df = DataFrame(np.arange(12).reshape(3, 4), columns=Index(list("ABCD"), name="foo"))  # 生成一个有名称为 "foo" 的索引列的数据帧
    expected = df.set_axis(df.index.to_numpy())  # 生成一个期望的数据帧，以确保索引正确

    with ensure_clean_store(setup_path) as store:
        store.put("df1", df, format="table")  # 使用表格格式将数据帧存储到数据存储中
        tm.assert_frame_equal(
            store["df1"], expected, check_index_type=True, check_column_type=True
        )  # 断言存储的数据帧与期望的数据帧相等，同时检查索引类型和列类型


def test_store_multiindex(setup_path):
    # 验证多重索引名称
    # GH 5527
    # 使用 ensure_clean_store 上下文管理器来确保 setup_path 中的存储被清理
    with ensure_clean_store(setup_path) as store:
        
        # 定义一个生成索引的函数 make_index，如果没有指定 names 参数，则使用默认日期范围
        def make_index(names=None):
            dti = date_range("2013-12-01", "2013-12-02")
            mi = MultiIndex.from_product([dti, range(2), range(3)], names=names)
            return mi

        # 不使用 names 参数生成 DataFrame，并存储到 store 中的 "df" 键
        _maybe_remove(store, "df")
        df = DataFrame(np.zeros((12, 2)), columns=["a", "b"], index=make_index())
        store.append("df", df)
        # 使用 assert_frame_equal 检查 store 中的 "df" 与预期的 df 是否相等
        tm.assert_frame_equal(store.select("df"), df)

        # 使用部分 names 参数生成 DataFrame，并存储到 store 中的 "df" 键
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", None, None]),
        )
        store.append("df", df)
        # 使用 assert_frame_equal 检查 store 中的 "df" 与预期的 df 是否相等
        tm.assert_frame_equal(store.select("df"), df)

        # 生成 Series，并存储到 store 中的 "ser" 键
        _maybe_remove(store, "ser")
        ser = Series(np.zeros(12), index=make_index(["date", None, None]))
        store.append("ser", ser)
        # 使用 assert_series_equal 检查 store 中的 "ser" 与预期的 xp Series 是否相等
        xp = Series(np.zeros(12), index=make_index(["date", "level_1", "level_2"]))
        tm.assert_series_equal(store.select("ser"), xp)

        # 生成带有重复列名的 DataFrame，并预期引发 ValueError 异常
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "a", "t"]),
        )
        msg = "duplicate names/columns in the multi-index when storing as a table"
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # 生成在同一层级内有重复索引的 DataFrame，并预期引发 ValueError 异常
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "date", "date"]),
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # 使用完整 names 参数生成 DataFrame，并存储到 store 中的 "df" 键
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "s", "t"]),
        )
        store.append("df", df)
        # 使用 assert_frame_equal 检查 store 中的 "df" 与预期的 df 是否相等
        tm.assert_frame_equal(store.select("df"), df)
# 使用 pytest.mark.parametrize 装饰器定义一个参数化测试，测试两种格式："fixed" 和 "table"
@pytest.mark.parametrize("format", ["fixed", "table"])
# 定义一个测试函数，用于测试 PeriodIndex 在 HDFStore 中的行为
def test_store_periodindex(tmp_path, setup_path, format):
    # GH 7796
    # 说明这是针对 GitHub 问题编号 7796 的测试
    # 创建一个 DataFrame，其中包含从标准正态分布中生成的随机数据，形状为 (5, 1)
    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 1)),
        # 使用 pd.period_range 创建一个 PeriodIndex，从 "20220101" 开始，频率为每月，共 5 个周期
        index=pd.period_range("20220101", freq="M", periods=5),
    )

    # 将 DataFrame 写入 HDF 文件
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="w", format=format)
    
    # 从 HDF 文件读取预期的 DataFrame 数据
    expected = pd.read_hdf(path, "df")
    
    # 使用 assert_frame_equal 检查 df 和 expected 是否相等
    tm.assert_frame_equal(df, expected)
```