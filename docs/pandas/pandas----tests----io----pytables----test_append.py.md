# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_append.py`

```
# 导入标准库中的日期时间模块
import datetime
# 从日期时间模块中导入时间间隔类
from datetime import timedelta
# 导入正则表达式模块
import re

# 导入第三方库 numpy，并用 np 别名引用它
import numpy as np
# 导入 pytest 测试框架
import pytest

# 从 pandas 库的 _libs.tslibs 中导入 Timestamp 类
from pandas._libs.tslibs import Timestamp
# 从 pandas 库的 compat 模块中导入 PY312 常量
from pandas.compat import PY312

# 导入 pandas 库，并用 pd 别名引用它
import pandas as pd
# 从 pandas 库中导入 DataFrame、Index、Series、_testing 模块、concat 函数、date_range 函数和 read_hdf 函数
from pandas import (
    DataFrame,
    Index,
    Series,
    _testing as tm,
    concat,
    date_range,
    read_hdf,
)
# 从 pandas 测试模块的 io.pytables.common 中导入 _maybe_remove 函数和 ensure_clean_store 函数
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

# 将当前测试标记为仅在单CPU上运行的测试
pytestmark = pytest.mark.single_cpu

# 导入并检查是否存在 tables 模块，如果不存在则跳过测试
tables = pytest.importorskip("tables")

# 使用 pytest 的标记，忽略 tables.NaturalNameWarning 类型的警告
@pytest.mark.filterwarnings("ignore::tables.NaturalNameWarning")
# 定义一个测试函数，接收 setup_path 参数
def test_append(setup_path):
    with ensure_clean_store(setup_path) as store:
        # 使用 ensure_clean_store 函数创建一个干净的数据存储空间，并将其作为 store 变量使用

        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=20, freq="B"),
        )
        # 创建一个包含随机标准正态分布数据的 DataFrame，列为 'A', 'B', 'C', 'D'，行索引为工作日日期从 '2000-01-01' 开始的 20 天
        # 将其赋值给 df 变量

        _maybe_remove(store, "df1")
        # 如果存储空间中存在名为 "df1" 的数据，尝试移除它

        store.append("df1", df[:10])
        # 向名为 "df1" 的存储空间追加 df 的前 10 行数据

        store.append("df1", df[10:])
        # 继续向名为 "df1" 的存储空间追加 df 的后 10 行数据

        tm.assert_frame_equal(store["df1"], df)
        # 使用断言检查存储空间中的 "df1" 数据与 df 是否相等

        _maybe_remove(store, "df2")
        # 如果存储空间中存在名为 "df2" 的数据，尝试移除它

        store.put("df2", df[:10], format="table")
        # 向名为 "df2" 的存储空间放置 df 的前 10 行数据，以表格格式存储

        store.append("df2", df[10:])
        # 继续向名为 "df2" 的存储空间追加 df 的后 10 行数据

        tm.assert_frame_equal(store["df2"], df)
        # 使用断言检查存储空间中的 "df2" 数据与 df 是否相等

        _maybe_remove(store, "df3")
        # 如果存储空间中存在名为 "df3" 的数据，尝试移除它

        store.append("/df3", df[:10])
        # 向名为 "/df3" 的存储空间追加 df 的前 10 行数据

        store.append("/df3", df[10:])
        # 继续向名为 "/df3" 的存储空间追加 df 的后 10 行数据

        tm.assert_frame_equal(store["df3"], df)
        # 使用断言检查存储空间中的 "/df3" 数据与 df 是否相等

        _maybe_remove(store, "/df3 foo")
        # 如果存储空间中存在名为 "/df3 foo" 的数据，尝试移除它

        store.append("/df3 foo", df[:10])
        # 向名为 "/df3 foo" 的存储空间追加 df 的前 10 行数据

        store.append("/df3 foo", df[10:])
        # 继续向名为 "/df3 foo" 的存储空间追加 df 的后 10 行数据

        tm.assert_frame_equal(store["df3 foo"], df)
        # 使用断言检查存储空间中的 "/df3 foo" 数据与 df 是否相等

        df = DataFrame(data=[[1, 2], [0, 1], [1, 2], [0, 0]])
        df["mixed_column"] = "testing"
        df.loc[2, "mixed_column"] = np.nan
        # 创建一个包含列表数据的 DataFrame，添加名为 "mixed_column" 的列，其中包含混合数据类型，并在索引为 2 的行的 "mixed_column" 列中设置为 NaN

        _maybe_remove(store, "df")
        # 如果存储空间中存在名为 "df" 的数据，尝试移除它

        store.append("df", df)
        # 向名为 "df" 的存储空间追加 df 数据

        tm.assert_frame_equal(store["df"], df)
        # 使用断言检查存储空间中的 "df" 数据与 df 是否相等

        uint_data = DataFrame(
            {
                "u08": Series(
                    np.random.default_rng(2).integers(0, high=255, size=5),
                    dtype=np.uint8,
                ),
                "u16": Series(
                    np.random.default_rng(2).integers(0, high=65535, size=5),
                    dtype=np.uint16,
                ),
                "u32": Series(
                    np.random.default_rng(2).integers(0, high=2**30, size=5),
                    dtype=np.uint32,
                ),
                "u64": Series(
                    [2**58, 2**59, 2**60, 2**61, 2**62],
                    dtype=np.uint64,
                ),
            },
            index=np.arange(5),
        )
        # 创建一个包含不同类型无符号整数数据的 DataFrame，指定了数据类型，并设置了行索引

        _maybe_remove(store, "uints")
        # 如果存储空间中存在名为 "uints" 的数据，尝试移除它

        store.append("uints", uint_data)
        # 向名为 "uints" 的存储空间追加 uint_data 数据

        tm.assert_frame_equal(store["uints"], uint_data, check_index_type=True)
        # 使用断言检查存储空间中的 "uints" 数据与 uint_data 是否相等，并且检查索引类型是否相等

        _maybe_remove(store, "uints")
        # 如果存储空间中存在名为 "uints" 的数据，尝试移除它

        store.append("uints", uint_data, data_columns=["u08", "u16", "u32"])
        # 向名为 "uints" 的存储空间追加 uint_data 数据，并指定 "u08", "u16", "u32" 为数据列

        tm.assert_frame_equal(store["uints"], uint_data, check_index_type=True)
        # 使用断言检查存储空间中的 "uints" 数据与 uint_data 是否相等，并且检查索引类型是否相等
def test_append_series(setup_path):
    # 使用带清理的存储路径创建上下文环境
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含20个浮点数的Series，指定数据类型和索引
        ss = Series(range(20), dtype=np.float64, index=[f"i_{i}" for i in range(20)])
        
        # 创建一个包含10个浮点数的Series，指定数据和日期索引
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        
        # 创建一个包含100个整数的Series
        ns = Series(np.arange(100))

        # 将Series ss 追加到存储中，并从存储中获取结果
        store.append("ss", ss)
        result = store["ss"]
        # 断言存储中的结果与原始ss Series相等
        tm.assert_series_equal(result, ss)
        # 断言结果的名称为None
        assert result.name is None

        # 将Series ts 追加到存储中，并从存储中获取结果
        store.append("ts", ts)
        result = store["ts"]
        # 断言存储中的结果与原始ts Series相等
        tm.assert_series_equal(result, ts)
        # 断言结果的名称为None
        assert result.name is None

        # 将Series ns 追加到存储中，并从存储中获取结果
        ns.name = "foo"
        store.append("ns", ns)
        result = store["ns"]
        # 断言存储中的结果与原始ns Series相等
        tm.assert_series_equal(result, ns)
        # 断言结果的名称与ns的名称相同
        assert result.name == ns.name

        # 选择大于60的值
        expected = ns[ns > 60]
        result = store.select("ns", "foo>60")
        # 断言存储中的结果与预期的Series相等
        tm.assert_series_equal(result, expected)

        # 在索引和值上进行选择
        expected = ns[(ns > 70) & (ns.index < 90)]
        # 读取/写入RangeIndex信息尚不支持
        expected.index = Index(expected.index._data)
        result = store.select("ns", "foo>70 and index<90")
        # 断言存储中的结果与预期的Series相等，并检查索引类型
        tm.assert_series_equal(result, expected, check_index_type=True)

        # 多级索引
        mi = DataFrame(np.random.default_rng(2).standard_normal((5, 1)), columns=["A"])
        mi["B"] = np.arange(len(mi))
        mi["C"] = "foo"
        mi.loc[3:5, "C"] = "bar"
        mi.set_index(["C", "B"], inplace=True)
        s = mi.stack()
        s.index = s.index.droplevel(2)
        # 将多级索引的Series s 追加到存储中
        store.append("mi", s)
        # 断言存储中的结果与原始s Series相等，并检查索引类型
        tm.assert_series_equal(store["mi"], s, check_index_type=True)
    # 使用 ensure_clean_store 函数确保 setup_path 的存储环境干净，返回一个上下文管理器对象 store
    with ensure_clean_store(setup_path) as store:
        # 创建一个 DataFrame 对象 df，包含多列数据：
        # 列 'A' 是一个长度为 20 的整数类型 Series，数据由标准正态分布生成
        # 列 'A1' 和 'A2' 是长度为 20 的浮点数数组，由相同的随机种子生成
        # 列 'B' 是一个字符串 "foo"
        # 列 'C' 是一个字符串 "bar"
        # 列 'D' 和 'E' 是两个 Timestamp 类型的时间戳，分别表示 "2001-01-01" 和 "2001-01-02"
        df = DataFrame(
            {
                "A": Series(np.random.default_rng(2).standard_normal(20)).astype(
                    "int32"
                ),
                "A1": np.random.default_rng(2).standard_normal(20),
                "A2": np.random.default_rng(2).standard_normal(20),
                "B": "foo",
                "C": "bar",
                "D": Timestamp("2001-01-01").as_unit("ns"),
                "E": Timestamp("2001-01-02").as_unit("ns"),
            },
            index=np.arange(20),  # 使用 np.arange(20) 作为索引
        )
        # 在 DataFrame 中的某些位置设置 NaN 值
        _maybe_remove(store, "df1")  # 如果存在名为 "df1" 的数据，先尝试移除它
        df.loc[0:15, ["A1", "B", "D", "E"]] = np.nan  # 将索引为 0 到 15 的行，列为 'A1', 'B', 'D', 'E' 的值设置为 NaN
        # 将 df 的前 10 行数据追加到 store 的 "df1" 中
        store.append("df1", df[:10])
        # 再将 df 的后 10 行数据追加到 store 的 "df1" 中，形成完整的数据集
        store.append("df1", df[10:])
        # 使用 tm.assert_frame_equal 检查 store 中的 "df1" 是否与 df 相等，包括索引类型的检查
        tm.assert_frame_equal(store["df1"], df, check_index_type=True)

        # 处理第一列 'A1'
        df1 = df.copy()  # 复制 df 到 df1
        df1["A1"] = np.nan  # 将 df1 中的 'A1' 列所有值设置为 NaN
        _maybe_remove(store, "df1")  # 尝试移除 store 中的 "df1"
        store.append("df1", df1[:10])  # 将 df1 的前 10 行数据追加到 store 的 "df1" 中
        store.append("df1", df1[10:])  # 将 df1 的后 10 行数据追加到 store 的 "df1" 中
        # 使用 tm.assert_frame_equal 检查 store 中的 "df1" 是否与 df1 相等，包括索引类型的检查
        tm.assert_frame_equal(store["df1"], df1, check_index_type=True)

        # 处理第二列 'A2'
        df2 = df.copy()  # 复制 df 到 df2
        df2["A2"] = np.nan  # 将 df2 中的 'A2' 列所有值设置为 NaN
        _maybe_remove(store, "df2")  # 尝试移除 store 中的 "df2"
        store.append("df2", df2[:10])  # 将 df2 的前 10 行数据追加到 store 的 "df2" 中
        store.append("df2", df2[10:])  # 将 df2 的后 10 行数据追加到 store 的 "df2" 中
        # 使用 tm.assert_frame_equal 检查 store 中的 "df2" 是否与 df2 相等，包括索引类型的检查
        tm.assert_frame_equal(store["df2"], df2, check_index_type=True)

        # 处理日期时间列 'E'
        df3 = df.copy()  # 复制 df 到 df3
        df3["E"] = np.nan  # 将 df3 中的 'E' 列所有值设置为 NaN
        _maybe_remove(store, "df3")  # 尝试移除 store 中的 "df3"
        store.append("df3", df3[:10])  # 将 df3 的前 10 行数据追加到 store 的 "df3" 中
        store.append("df3", df3[10:])  # 将 df3 的后 10 行数据追加到 store 的 "df3" 中
        # 使用 tm.assert_frame_equal 检查 store 中的 "df3" 是否与 df3 相等，包括索引类型的检查
        tm.assert_frame_equal(store["df3"], df3, check_index_type=True)
def test_append_all_nans(setup_path):
    # 测试在指定路径下追加 NaN 值的处理
    with ensure_clean_store(setup_path) as store:
        # column oriented
        # 创建一个包含随机标准正态分布数据的 DataFrame，4列10行
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 清除索引频率信息
        df.index = df.index._with_freq(None)  # freq doesn't round-trip

        # 确保存储中没有名称为 "df1" 的数据表
        _maybe_remove(store, "df1")
        # 将 DataFrame 的前两列追加到存储中，指定追加方向为 "columns"
        store.append("df1", df.iloc[:, :2], axes=["columns"])
        # 将 DataFrame 的后两列追加到存储中，默认追加方式为 "rows"
        store.append("df1", df.iloc[:, 2:])
        # 断言存储中的 "df1" 数据与原始 DataFrame 相等
        tm.assert_frame_equal(store["df1"], df)

        # 从存储中选择列名为 "A" 的数据表，并与预期结果进行比较
        result = store.select("df1", "columns=A")
        expected = df.reindex(columns=["A"])
        tm.assert_frame_equal(expected, result)

        # 在不可索引对象上进行选择
        request.applymarker(
            pytest.mark.xfail(
                PY312,
                reason="AST change in PY312",
                raises=ValueError,
            )
        )
        # 选择存储中 "df1" 数据，其中列名为 "A"，并且索引在 df.index 的前四个元素范围内
        result = store.select("df1", ("columns=A", "index=df.index[0:4]"))
        expected = df.reindex(columns=["A"], index=df.index[0:4])
        tm.assert_frame_equal(expected, result)

        # 这种选择方式不支持
        msg = re.escape(
            "passing a filterable condition to a non-table indexer "
            "[Filter: Not Initialized]"
        )
        # 断言在选择过程中抛出 TypeError 异常，并匹配特定的错误信息
        with pytest.raises(TypeError, match=msg):
            store.select("df1", "columns=A and index>df.index[4]")


def test_append_with_different_block_ordering(setup_path):
    # GH 4096; 使用相同的 DataFrame，但是不同的块排序
    with ensure_clean_store(setup_path) as store:
        for i in range(10):
            # 创建一个包含随机标准正态分布数据的 DataFrame，包括列 'A' 和 'B'
            df = DataFrame(
                np.random.default_rng(2).standard_normal((10, 2)), columns=list("AB")
            )
            # 向 DataFrame 添加 'index' 列，赋值为 0 到 90 的数列
            df["index"] = range(10)
            df["index"] += i * 10
            # 添加 'int64' 列，值为全部为 1 的 int64 类型 Series
            df["int64"] = Series([1] * len(df), dtype="int64")
            # 添加 'int16' 列，值为全部为 1 的 int16 类型 Series
            df["int16"] = Series([1] * len(df), dtype="int16")

            # 根据条件删除 'int64' 列，并重新添加 'int64' 列
            if i % 2 == 0:
                del df["int64"]
                df["int64"] = Series([1] * len(df), dtype="int64")
            # 根据条件删除 'A' 列，并重新添加 'A' 列
            if i % 3 == 0:
                a = df.pop("A")
                df["A"] = a

            # 将 'index' 列设置为索引
            df.set_index("index", inplace=True)

            # 将 DataFrame 追加到存储中，使用名称 "df"
            store.append("df", df)

    # 测试不同的排序方式，以及包含更多字段的情况（例如无效的组合）
    # 使用 ensure_clean_store 函数确保 setup_path 的存储空间是干净的，使用 with 语句管理上下文
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含随机标准正态分布数据的 DataFrame，10行2列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            columns=list("AB"),
            dtype="float64",
        )
        # 向 DataFrame 中添加名为 'int64' 的整数列，值为 1
        df["int64"] = Series([1] * len(df), dtype="int64")
        # 向 DataFrame 中添加名为 'int16' 的短整型列，值为 1
        df["int16"] = Series([1] * len(df), dtype="int16")
        # 将 DataFrame 存储到 store 中，命名为 'df'

        store.append("df", df)

        # 在不同的数据块中存储额外字段
        df["int16_2"] = Series([1] * len(df), dtype="int16")
        # 设置错误消息的正则表达式，用于捕获 ValueError 异常，检查表结构匹配性
        msg = re.escape(
            "cannot match existing table structure for [int16] on appending data"
        )
        # 断言在存储时会引发 ValueError，并且错误消息符合预期的正则表达式
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # 在不同的数据块中存储多个额外字段
        df["float_3"] = Series([1.0] * len(df), dtype="float64")
        # 设置错误消息的正则表达式，用于捕获 ValueError 异常，检查表结构匹配性
        msg = re.escape(
            "cannot match existing table structure for [A,B] on appending data"
        )
        # 断言在存储时会引发 ValueError，并且错误消息符合预期的正则表达式
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)
# 使用字符串进行测试附加操作，并在给定路径设置的环境中执行
def test_append_with_strings(setup_path):
    # 确保存储空间干净，并将数据帧 df 创建为包含字符串的 DataFrame
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({"A": "foo", "B": "bar"}, index=range(10))

        # 使用指定的 min_itemsize 创建数据列 "A"
        _maybe_remove(store, "df")
        store.append("df", df, min_itemsize={"A": 200})
        # 检查数据列 "A" 的最小大小是否为 200
        check_col("df", "A", 200)
        # 断言存储器中的数据列为 ["A"]
        assert store.get_storer("df").data_columns == ["A"]

        # 使用指定的 data_columns 和 min_itemsize 创建数据列 "B"
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["B"], min_itemsize={"A": 200})
        # 检查数据列 "A" 的最小大小是否为 200
        check_col("df", "A", 200)
        # 断言存储器中的数据列为 ["B", "A"]
        assert store.get_storer("df").data_columns == ["B", "A"]

        # 使用指定的 data_columns 和 min_itemsize 创建数据列 "B"
        _maybe_remove(store, "df")
        store.append("df", df, data_columns=["B"], min_itemsize={"values": 200})
        # 检查数据列 "B" 的最小大小是否为 200
        check_col("df", "B", 200)
        # 检查数据块 "values_block_0" 的最小大小是否为 200
        check_col("df", "values_block_0", 200)
        # 断言存储器中的数据列为 ["B"]
        assert store.get_storer("df").data_columns == ["B"]

        # 在后续附加操作中推断 .typ
        _maybe_remove(store, "df")
        store.append("df", df[:5], min_itemsize=200)
        store.append("df", df[5:], min_itemsize=200)
        # 断言存储器中的数据帧与 df 相等
        tm.assert_frame_equal(store["df"], df)

        # 对于无效的 min_itemsize 键进行测试
        df = DataFrame(["foo", "foo", "foo", "barh", "barh", "barh"], columns=["A"])
        _maybe_remove(store, "df")
        # 预期引发 ValueError，并检查消息是否匹配预期的正则表达式消息
        msg = re.escape(
            "min_itemsize has the key [foo] which is not an axis or data_column"
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df, min_itemsize={"foo": 20, "foobar": 20})


# 使用空字符串进行测试附加操作
def test_append_with_empty_string(setup_path):
    # 确保存储空间干净，并将数据帧 df 创建为包含空字符串的 DataFrame
    with ensure_clean_store(setup_path) as store:
        # 包含全部空字符串的 DataFrame df
        df = DataFrame({"x": ["a", "b", "c", "d", "e", "f", ""]})
        # 将 df 的前 n-1 行附加到 "df" 中，并设置 min_itemsize
        store.append("df", df[:-1], min_itemsize={"x": 1})
        # 将 df 的最后一行附加到 "df" 中，并设置 min_itemsize
        store.append("df", df[-1:], min_itemsize={"x": 1})
        # 断言存储器中选择的 "df" 与 df 相等
        tm.assert_frame_equal(store.select("df"), df)
    # 使用 ensure_clean_store 函数确保存储在 setup_path 中的环境干净，并且自动清理
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含随机标准正态分布数据的 DataFrame，指定列名和日期索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 将第一行第二列的元素设为 1.0
        df.iloc[0, df.columns.get_loc("B")] = 1.0
        # 如果存在名为 "df" 的项，则尝试移除它
        _maybe_remove(store, "df")
        # 将 DataFrame 的前两行追加到存储中，并指定 "B" 列作为数据列
        store.append("df", df[:2], data_columns=["B"])
        # 将 DataFrame 的剩余部分追加到存储中
        store.append("df", df[2:])
        # 断言存储中的 "df" 与原始 DataFrame 相等
        tm.assert_frame_equal(store["df"], df)

        # 检查是否已创建索引
        assert store._handle.root.df.table.cols.index.is_indexed is True
        assert store._handle.root.df.table.cols.B.is_indexed is True

        # 使用数据列查询数据
        result = store.select("df", "B>0")
        expected = df[df.B > 0]
        tm.assert_frame_equal(result, expected)

        # 使用带有索引条件的数据列查询数据
        result = store.select("df", "B>0 and index>df.index[3]")
        df_new = df.reindex(index=df.index[4:])
        expected = df_new[df_new.B > 0]
        tm.assert_frame_equal(result, expected)

        # 使用字符串数据列进行数据选择
        df_new = df.copy()
        df_new["string"] = "foo"
        df_new.loc[df_new.index[1:4], "string"] = np.nan
        df_new.loc[df_new.index[5:6], "string"] = "bar"
        _maybe_remove(store, "df")
        store.append("df", df_new, data_columns=["string"])
        result = store.select("df", "string='foo'")
        expected = df_new[df_new.string == "foo"]
        tm.assert_frame_equal(result, expected)

        # 使用 min_itemsize 和数据列进行检查
        def check_col(key, name, size):
            assert (
                getattr(store.get_storer(key).table.description, name).itemsize == size
            )

    # 使用 ensure_clean_store 函数确保存储在 setup_path 中的环境干净，并且自动清理
    with ensure_clean_store(setup_path) as store:
        # 如果存在名为 "df" 的项，则尝试移除它
        _maybe_remove(store, "df")
        # 向存储中追加 DataFrame，指定数据列 "string" 和最小项大小为 30
        store.append("df", df_new, data_columns=["string"], min_itemsize={"string": 30})
        # 检查存储中 "df" 的 "string" 列的项大小是否为 30
        check_col("df", "string", 30)
        # 如果存在名为 "df" 的项，则尝试移除它
        _maybe_remove(store, "df")
        # 向存储中追加 DataFrame，指定数据列 "string" 和最小项大小为 30
        store.append("df", df_new, data_columns=["string"], min_itemsize=30)
        # 检查存储中 "df" 的 "string" 列的项大小是否为 30
        check_col("df", "string", 30)
        # 如果存在名为 "df" 的项，则尝试移除它
        _maybe_remove(store, "df")
        # 向存储中追加 DataFrame，指定数据列 "string" 和最小项大小为 30
        store.append("df", df_new, data_columns=["string"], min_itemsize={"values": 30})
        # 检查存储中 "df" 的 "string" 列的项大小是否为 30
        check_col("df", "string", 30)

    # 使用 ensure_clean_store 函数确保存储在 setup_path 中的环境干净，并且自动清理
    with ensure_clean_store(setup_path) as store:
        # 向 DataFrame 中添加新的字符串列
        df_new["string2"] = "foobarbah"
        df_new["string_block1"] = "foobarbah1"
        df_new["string_block2"] = "foobarbah2"
        # 如果存在名为 "df" 的项，则尝试移除它
        _maybe_remove(store, "df")
        # 向存储中追加 DataFrame，指定数据列 "string" 和 "string2"，并指定最小项大小
        store.append(
            "df",
            df_new,
            data_columns=["string", "string2"],
            min_itemsize={"string": 30, "string2": 40, "values": 50},
        )
        # 检查存储中 "df" 的 "string" 列的项大小是否为 30
        check_col("df", "string", 30)
        # 检查存储中 "df" 的 "string2" 列的项大小是否为 40
        check_col("df", "string2", 40)
        # 检查存储中 "df" 的 "values_block_1" 列的项大小是否为 50
        check_col("df", "values_block_1", 50)
    with ensure_clean_store(setup_path) as store:
        # 使用 ensure_clean_store 函数确保 store 在使用前是干净的
        # 复制 DataFrame df 到 df_new
        df_new = df.copy()
        # 将 df_new 中第一行 "A" 列的值设为 1.0
        df_new.iloc[0, df_new.columns.get_loc("A")] = 1.0
        # 将 df_new 中第一行 "B" 列的值设为 -1.0
        df_new.iloc[0, df_new.columns.get_loc("B")] = -1.0
        # 在 df_new 中新增一个名为 "string" 的列，并将其值设为 "foo"
        df_new["string"] = "foo"

        # 获取列名为 "string" 的列索引
        sl = df_new.columns.get_loc("string")
        # 将 df_new 中第 1 到 3 行（不包括第 4 行）的 "string" 列设为 NaN
        df_new.iloc[1:4, sl] = np.nan
        # 将 df_new 中第 5 行的 "string" 列的值设为 "bar"
        df_new.iloc[5:6, sl] = "bar"

        # 在 df_new 中新增一个名为 "string2" 的列，并将其值设为 "foo"
        df_new["string2"] = "foo"
        # 获取列名为 "string2" 的列索引
        sl = df_new.columns.get_loc("string2")
        # 将 df_new 中第 2 到 4 行（不包括第 5 行）的 "string2" 列设为 NaN
        df_new.iloc[2:5, sl] = np.nan
        # 将 df_new 中第 7 行的 "string2" 列的值设为 "bar"
        df_new.iloc[7:8, sl] = "bar"
        
        # 在 store 中或许移除名为 "df" 的对象
        _maybe_remove(store, "df")
        # 将 df_new 附加到 store 中，使用 "A", "B", "string", "string2" 列作为数据列
        store.append("df", df_new, data_columns=["A", "B", "string", "string2"])
        # 从 store 中选择符合条件 "string='foo' and string2='foo' and A>0 and B<0" 的数据
        result = store.select("df", "string='foo' and string2='foo' and A>0 and B<0")
        # 从 df_new 中选择符合条件的期望数据
        expected = df_new[
            (df_new.string == "foo")
            & (df_new.string2 == "foo")
            & (df_new.A > 0)
            & (df_new.B < 0)
        ]
        # 断言结果与期望是否相等，不检查频率
        tm.assert_frame_equal(result, expected, check_freq=False)
        # FIXME: 2020-05-07 频率检查在 CI 中随机失败

        # 生成一个空的 DataFrame
        result = store.select("df", "string='foo' and string2='cool'")
        # 从 df_new 中选择符合条件的期望数据
        expected = df_new[(df_new.string == "foo") & (df_new.string2 == "cool")]
        # 断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    with ensure_clean_store(setup_path) as store:
        # 使用 ensure_clean_store 函数确保 store 在使用前是干净的
        # 复制 DataFrame df 到 df_dc
        df_dc = df.copy()
        # 在 df_dc 中新增一个名为 "string" 的列，并将其值设为 "foo"
        df_dc["string"] = "foo"
        # 将 df_dc 中索引为 4 到 6 的行的 "string" 列设为 NaN
        df_dc.loc[df_dc.index[4:6], "string"] = np.nan
        # 将 df_dc 中索引为 7 到 9 的行的 "string" 列的值设为 "bar"
        df_dc.loc[df_dc.index[7:9], "string"] = "bar"
        # 在 df_dc 中新增一个名为 "string2" 的列，并将其值设为 "cool"
        df_dc["string2"] = "cool"
        # 在 df_dc 中新增一个名为 "datetime" 的列，并将其值设为指定的时间戳
        df_dc["datetime"] = Timestamp("20010102").as_unit("ns")
        # 将 df_dc 中索引为 3 到 5 的行的 "A", "B", "datetime" 列设为 NaN
        df_dc.loc[df_dc.index[3:5], ["A", "B", "datetime"]] = np.nan

        # 在 store 中或许移除名为 "df_dc" 的对象
        _maybe_remove(store, "df_dc")
        # 将 df_dc 附加到 store 中，使用 "B", "C", "string", "string2", "datetime" 列作为数据列
        store.append(
            "df_dc", df_dc, data_columns=["B", "C", "string", "string2", "datetime"]
        )
        # 从 store 中选择符合条件 "B>0" 的数据
        result = store.select("df_dc", "B>0")
        # 从 df_dc 中选择符合条件的期望数据
        expected = df_dc[df_dc.B > 0]
        # 断言结果与期望是否相等

        tm.assert_frame_equal(result, expected)

        # 从 store 中选择符合条件 ["B > 0", "C > 0", "string == foo"] 的数据
        result = store.select("df_dc", ["B > 0", "C > 0", "string == foo"])
        # 从 df_dc 中选择符合条件的期望数据，不检查频率
        expected = df_dc[(df_dc.B > 0) & (df_dc.C > 0) & (df_dc.string == "foo")]
        # 断言结果与期望是否相等，不检查频率
        tm.assert_frame_equal(result, expected, check_freq=False)
        # FIXME: 2020-12-07 这里的 intermittent build failures 与 freq 为 None 而不是 BDay(4) 相关
    # 使用 ensure_clean_store 函数创建一个上下文管理器，并指定 setup_path 作为参数
    with ensure_clean_store(setup_path) as store:
        # 创建一个日期范围索引，从 "1/1/2000" 开始，包含 8 个时间点
        index = date_range("1/1/2000", periods=8)
        # 创建一个 DataFrame 对象 df_dc，包含 8 行和 3 列的随机标准正态分布数据，索引为 index，列名为 ["A", "B", "C"]
        df_dc = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)),
            index=index,
            columns=["A", "B", "C"],
        )
        # 在 df_dc 中新增一列 "string"，所有行的值为 "foo"
        df_dc["string"] = "foo"
        # 将 df_dc 中索引为 4 到 5 的行的 "string" 列值设为 NaN
        df_dc.loc[df_dc.index[4:6], "string"] = np.nan
        # 将 df_dc 中索引为 7 到 8 的行的 "string" 列值设为 "bar"
        df_dc.loc[df_dc.index[7:9], "string"] = "bar"
        # 将 df_dc 中 "B" 列和 "C" 列的所有值取绝对值
        df_dc[["B", "C"]] = df_dc[["B", "C"]].abs()
        # 在 df_dc 中新增一列 "string2"，所有行的值为 "cool"
        df_dc["string2"] = "cool"

        # 将 df_dc 存储到 store 中，使用 "df_dc" 作为键，并指定数据列为 ["B", "C", "string", "string2"]
        store.append("df_dc", df_dc, data_columns=["B", "C", "string", "string2"])

        # 从 store 中选择键为 "df_dc" 的数据，过滤条件为 "B > 0"
        result = store.select("df_dc", "B>0")
        # 从 df_dc 中选择符合条件 "B > 0" 的数据，作为预期结果
        expected = df_dc[df_dc.B > 0]
        # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 从 store 中选择键为 "df_dc" 的数据，过滤条件为 ["B > 0", "C > 0", 'string == "foo"']
        result = store.select("df_dc", ["B > 0", "C > 0", 'string == "foo"'])
        # 从 df_dc 中选择符合条件 "B > 0"、"C > 0" 和 'string == "foo"' 的数据，作为预期结果
        expected = df_dc[(df_dc.B > 0) & (df_dc.C > 0) & (df_dc.string == "foo")]
        # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
# 定义测试函数，用于测试在给定路径上追加数据到层次化存储中
def test_append_hierarchical(tmp_path, setup_path, multiindex_dataframe_random_data):
    # 获取随机生成的多级索引数据帧
    df = multiindex_dataframe_random_data
    # 清除列名
    df.columns.name = None

    # 在清理后的存储路径上创建上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 将数据帧追加到存储中，键为"mi"
        store.append("mi", df)
        # 从存储中选择键为"mi"的数据
        result = store.select("mi")
        # 断言选取的数据帧与原始数据帧相等
        tm.assert_frame_equal(result, df)

        # GH 3748
        # 选择存储中键为"mi"的数据，限定列为["A", "B"]
        result = store.select("mi", columns=["A", "B"])
        # 从原始数据帧中重新索引选定的列
        expected = df.reindex(columns=["A", "B"])
        # 断言选取的数据帧与重新索引后的期望数据帧相等
        tm.assert_frame_equal(result, expected)

    # 创建临时文件路径
    path = tmp_path / "test.hdf"
    # 将数据帧写入 HDF 文件，键为"df"，格式为表格格式
    df.to_hdf(path, key="df", format="table")
    # 从 HDF 文件中读取键为"df"的数据，限定列为["A", "B"]
    result = read_hdf(path, "df", columns=["A", "B"])
    # 从原始数据帧中重新索引选定的列
    expected = df.reindex(columns=["A", "B"])
    # 断言读取的数据帧与重新索引后的期望数据帧相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，用于测试在给定路径上追加各种数据到存储中
def test_append_misc(setup_path):
    # 在清理后的存储路径上创建上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 创建包含随机数据的数据帧，索引和列均为对象类型
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 将数据帧追加到存储中，键为"df"，每次追加块大小为1
        store.append("df", df, chunksize=1)
        # 从存储中选择键为"df"的数据
        result = store.select("df")
        # 断言选取的数据帧与原始数据帧相等
        tm.assert_frame_equal(result, df)

        # 将数据帧追加到存储中，键为"df1"，期望的行数为10
        store.append("df1", df, expectedrows=10)
        # 从存储中选择键为"df1"的数据
        result = store.select("df1")
        # 断言选取的数据帧与原始数据帧相等
        tm.assert_frame_equal(result, df)


# 使用参数化装饰器，定义测试函数，测试在给定路径上追加不同块大小的数据到存储中
@pytest.mark.parametrize("chunksize", [10, 200, 1000])
def test_append_misc_chunksize(setup_path, chunksize):
    # 创建包含随机数据的数据帧，索引和列均为对象类型
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 为数据帧添加额外的列和属性
    df["string"] = "foo"
    df["float322"] = 1.0
    df["float322"] = df["float322"].astype("float32")
    df["bool"] = df["float322"] > 0
    df["time1"] = Timestamp("20130101").as_unit("ns")
    df["time2"] = Timestamp("20130102").as_unit("ns")

    # 在清理后的存储路径上创建上下文管理器，以写入模式
    with ensure_clean_store(setup_path, mode="w") as store:
        # 将数据帧追加到存储中，键为"obj"，块大小由参数化提供
        store.append("obj", df, chunksize=chunksize)
        # 从存储中选择键为"obj"的数据
        result = store.select("obj")
        # 断言选取的数据帧与原始数据帧相等
        tm.assert_frame_equal(result, df)


# 定义测试函数，测试在给定路径上追加空数据帧到存储中
def test_append_misc_empty_frame(setup_path):
    # 空数据帧，GH4273
    with ensure_clean_store(setup_path) as store:
        # 创建空的数据帧，列为["A", "B", "C"]
        df_empty = DataFrame(columns=list("ABC"))
        # 将空数据帧追加到存储中，键为"df"
        store.append("df", df_empty)
        # 断言从存储中选择键为"df"时会引发 KeyError 异常，提示'No object named df in the file'
        with pytest.raises(KeyError, match="'No object named df in the file'"):
            store.select("df")

        # 创建包含随机数据的数据帧，列为["A", "B", "C"]
        df = DataFrame(np.random.default_rng(2).random((10, 3)), columns=list("ABC"))
        # 将数据帧追加到存储中，键为"df"
        store.append("df", df)
        # 断言从存储中选择键为"df"时返回的数据帧与原始数据帧相等
        tm.assert_frame_equal(store.select("df"), df)

        # 再次将空数据帧追加到存储中，键为"df"
        store.append("df", df_empty)
        # 断言从存储中选择键为"df"时返回的数据帧与第一次追加的数据帧相等
        tm.assert_frame_equal(store.select("df"), df)

        # 将空数据帧存入存储中，键为"df2"
        df = DataFrame(columns=list("ABC"))
        store.put("df2", df)
        # 断言从存储中选择键为"df2"时返回的数据帧与原始数据帧相等
        tm.assert_frame_equal(store.select("df2"), df)


# 定义测试函数，测试在给定路径上追加时引发异常的情况
def test_append_raise(setup_path):
    # 使用 ensure_clean_store 函数确保在 setup_path 路径下创建的临时存储，使用 with 上下文管理器，确保资源的正确释放

        # 测试在出现无效输入时追加内容，以获得良好的错误消息

        # 在 DataFrame 中创建一个列为列表的列
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),  # 创建一个 30 行 4 列的 DataFrame，数据为按照特定规则生成的浮点数
            columns=Index(list("ABCD"), dtype=object),  # 列名为 A, B, C, D，数据类型为对象
            index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 行索引为形如 i-0, i-1, ..., i-29 的对象类型索引
        )
        df["invalid"] = [["a"]] * len(df)  # 向 DataFrame 中的 invalid 列插入列表 [["a"]] * 30 的数据
        assert df.dtypes["invalid"] == np.object_  # 断言 invalid 列的数据类型为 np.object_

        msg = re.escape(
            """Cannot serialize the column [invalid]
        msg = re.escape(
            "cannot properly create the storer for: "
            "[group->df,value-><class 'pandas.core.series.Series'>]"
        )
        # 使用 pytest 来测试异常情况，确保抛出特定的 TypeError，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            # 将 Series 对象作为值尝试追加到存储中
            store.append("df", Series(np.arange(10)))

        # appending an incompatible table
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 将 DataFrame 对象追加到存储中
        store.append("df", df)

        # 在 DataFrame 中添加一个额外的列
        df["foo"] = "foo"
        msg = re.escape(
            "invalid combination of [non_index_axes] on appending data "
            "[(1, ['A', 'B', 'C', 'D', 'foo'])] vs current table "
            "[(1, ['A', 'B', 'C', 'D'])]"
        )
        # 使用 pytest 来测试异常情况，确保抛出特定的 ValueError，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 尝试将修改后的 DataFrame 对象再次追加到存储中
            store.append("df", df)

        # incompatible type (GH 41897)
        # 移除之前存储中的 "df" 表
        _maybe_remove(store, "df")
        # 向 DataFrame 中添加一个时间戳类型的列
        df["foo"] = Timestamp("20130101")
        # 将修改后的 DataFrame 对象追加到存储中
        store.append("df", df)
        # 再次修改 DataFrame 中的 "foo" 列
        df["foo"] = "bar"
        msg = re.escape(
            "invalid combination of [values_axes] on appending data "
            "[name->values_block_1,cname->values_block_1,"
            "dtype->bytes24,kind->string,shape->(1, 30)] "
            "vs current table "
            "[name->values_block_1,cname->values_block_1,"
            "dtype->datetime64[s],kind->datetime64[s],shape->None]"
        )
        # 使用 pytest 来测试异常情况，确保抛出特定的 ValueError，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 尝试将再次修改后的 DataFrame 对象追加到存储中
            store.append("df", df)
    # 在 DataFrame 中计算新列 "C"，其值为列 "A" 减去列 "B"
    df["C"] = df["A"] - df["B"]

    # 将 DataFrame 中索引为 3 到 5 的行，列 "C" 的数值设为 NaN（缺失值）
    df.loc[3:5, "C"] = np.nan

    # 使用 ensure_clean_store 上下文管理器，确保存储环境 setup_path 的干净状态
    with ensure_clean_store(setup_path) as store:
        # 在存储器中删除可能存在的名为 "df" 的表
        _maybe_remove(store, "df")
        # 将 DataFrame df 存储到 store 中，表名为 "df"，同时创建数据列索引
        store.append("df", df, data_columns=True)
        # 从 store 中选择名为 "df" 的表，将结果存储到 result
        result = store.select("df")
        # 断言选取的结果与原始 DataFrame df 相等
        tm.assert_frame_equal(result, df)

        # 从存储器中选取表 "df"，其中列 "C" 小于 100000 的行
        result = store.select("df", where="C<100000")
        # 断言选取的结果与原始 DataFrame df 相等
        tm.assert_frame_equal(result, df)

        # 从存储器中选取表 "df"，其中列 "C" 小于 -3 天的行
        result = store.select("df", where="C<pd.Timedelta('-3D')")
        # 断言选取的结果与原始 DataFrame df 的第 3 行及之后的行相等
        tm.assert_frame_equal(result, df.iloc[3:])

        # 从存储器中选取表 "df"，其中列 "C" 小于 -3 天的行
        result = store.select("df", "C<'-3D'")
        # 断言选取的结果与原始 DataFrame df 的第 3 行及之后的行相等
        tm.assert_frame_equal(result, df.iloc[3:])

        # 从存储器中选取表 "df"，其中列 "C" 小于 -500000 秒的行
        result = store.select("df", "C<'-500000s'")
        # 删除结果中列 "C" 中的缺失值行
        result = result.dropna(subset=["C"])
        # 断言选取的结果与原始 DataFrame df 的第 6 行及之后的行相等
        tm.assert_frame_equal(result, df.iloc[6:])

        # 从存储器中选取表 "df"，其中列 "C" 小于 -3.5 天的行
        result = store.select("df", "C<'-3.5D'")
        # 从结果中的第二行开始，选取所有行
        result = result.iloc[1:]
        # 断言选取的结果与原始 DataFrame df 的第 4 行及之后的行相等
        tm.assert_frame_equal(result, df.iloc[4:])

        # 在存储器中删除可能存在的名为 "df2" 的表
        _maybe_remove(store, "df2")
        # 将 DataFrame df 存储到 store 中，表名为 "df2"
        store.put("df2", df)
        # 从 store 中选择名为 "df2" 的表，将结果存储到 result
        result = store.select("df2")
        # 断言选取的结果与原始 DataFrame df 相等
        tm.assert_frame_equal(result, df)
def test_append_to_multiple(setup_path):
    # 创建一个包含随机数据的 DataFrame df1，列为 'A', 'B', 'C', 'D'，索引为工作日频率
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 复制 df1 并重命名列名，添加新列 'foo' 并赋值 'bar'，得到 DataFrame df2
    df2 = df1.copy().rename(columns="{}_2".format)
    df2["foo"] = "bar"
    # 将 df1 和 df2 沿列方向拼接成新的 DataFrame df
    df = concat([df1, df2], axis=1)

    # 使用 setup_path 确保存储环境干净，store 是一个上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 测试异常情况，验证是否抛出 ValueError 异常，并匹配指定的消息
        msg = "append_to_multiple requires a selector that is in passed dict"
        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple(
                {"df1": ["A", "B"], "df2": None}, df, selector="df3"
            )

        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple({"df1": None, "df2": None}, df, selector="df3")

        # 再次测试异常情况，验证是否抛出 ValueError 异常，并匹配指定的消息
        msg = (
            "append_to_multiple must have a dictionary specified as the way to "
            "split the value"
        )
        with pytest.raises(ValueError, match=msg):
            store.append_to_multiple("df1", df, "df1")

        # 正常操作，使用 append_to_multiple 方法向存储中的多个数据集添加数据
        store.append_to_multiple({"df1": ["A", "B"], "df2": None}, df, selector="df1")
        # 从存储中选择多个数据集并返回结果，筛选条件为 A > 0 和 B > 0
        result = store.select_as_multiple(
            ["df1", "df2"], where=["A>0", "B>0"], selector="df1"
        )
        # 期望的结果是 DataFrame df 中满足条件 A > 0 和 B > 0 的部分
        expected = df[(df.A > 0) & (df.B > 0)]
        # 使用 assert_frame_equal 检查结果与期望是否相等
        tm.assert_frame_equal(result, expected)


def test_append_to_multiple_dropna(setup_path):
    # 创建包含随机数据的 DataFrame df1，列为 'A', 'B', 'C', 'D'，索引为工作日频率
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 创建包含随机数据的 DataFrame df2，列与 df1 相同，索引为工作日频率，列名重命名
    df2 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    ).rename(columns="{}_2".format)
    # 在 df1 中的第二行，列 'A' 和 'B' 的位置设置为 NaN
    df1.iloc[1, df1.columns.get_indexer(["A", "B"])] = np.nan
    # 将 df1 和 df2 沿列方向拼接成新的 DataFrame df
    df = concat([df1, df2], axis=1)

    # 使用 setup_path 确保存储环境干净，store 是一个上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 使用 append_to_multiple 方法向存储中的多个数据集添加数据，dropna=True 确保同步行
        store.append_to_multiple(
            {"df1": ["A", "B"], "df2": None}, df, selector="df1", dropna=True
        )
        # 从存储中选择多个数据集并返回结果
        result = store.select_as_multiple(["df1", "df2"])
        # 期望的结果是 DataFrame df 中去除 NaN 后的部分
        expected = df.dropna()
        # 使用 assert_frame_equal 检查结果与期望是否相等，同时检查索引类型是否一致
        tm.assert_frame_equal(result, expected, check_index_type=True)
        # 检查存储中 'df1' 和 'df2' 的索引是否相等
        tm.assert_index_equal(store.select("df1").index, store.select("df2").index)


def test_append_to_multiple_dropna_false(setup_path):
    # 创建包含随机数据的 DataFrame df1，列为 'A', 'B', 'C', 'D'，索引为工作日频率
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 复制 df1 并重命名列名，得到 DataFrame df2
    df2 = df1.copy().rename(columns="{}_2".format)
    # 在 df1 中的第二行，列 'A' 和 'B' 的位置设置为 NaN
    df1.iloc[1, df1.columns.get_indexer(["A", "B"])] = np.nan
    # 将 df1 和 df2 沿列方向拼接成新的 DataFrame df
    df = concat([df1, df2], axis=1)

    # 使用 setup_path 确保存储环境干净，store 是一个上下文管理器
    with (
        ensure_clean_store(setup_path) as store,
        pd.option_context("io.hdf.dropna_table", True),
        # 设置上下文，io.hdf.dropna_table 选项为 True
    ):
    ):
        # 在存储中追加数据框 df 到多个表中，selector="df1a" 表示仅将数据追加到 "df1a" 表中，dropna=False 保证不同步行索引
        store.append_to_multiple(
            {"df1a": ["A", "B"], "df2a": None}, df, selector="df1a", dropna=False
        )

        # 出现 ValueError 异常，并检查异常消息是否为 "all tables must have exactly the same nrows!"
        msg = "all tables must have exactly the same nrows!"
        with pytest.raises(ValueError, match=msg):
            # 尝试从存储中选择多个表，验证它们是否具有相同的行数
            store.select_as_multiple(["df1a", "df2a"])

        # 断言：确保 "df1a" 和 "df2a" 表的索引不相等
        assert not store.select("df1a").index.equals(store.select("df2a").index)
def test_append_to_multiple_min_itemsize(setup_path):
    # GH 11238: 标识GitHub问题编号
    df = DataFrame(
        {
            "IX": np.arange(1, 21),  # 创建包含整数序列的"IX"列
            "Num": np.arange(1, 21),  # 创建包含整数序列的"Num"列
            "BigNum": np.arange(1, 21) * 88,  # 创建包含整数序列乘以88的"BigNum"列
            "Str": ["a" for _ in range(20)],  # 创建包含20个'a'的"Str"列
            "LongStr": ["abcde" for _ in range(20)],  # 创建包含20个'abcde'的"LongStr"列
        }
    )
    expected = df.iloc[[0]]  # 获取df的第一行作为预期结果
    # Reading/writing RangeIndex info is not supported yet
    expected.index = Index(list(range(len(expected.index))))  # 将预期结果的索引设为RangeIndex

    with ensure_clean_store(setup_path) as store:  # 打开一个干净的存储空间
        store.append_to_multiple(
            {
                "index": ["IX"],  # 将"IX"列映射到名为"index"的存储
                "nums": ["Num", "BigNum"],  # 将"Num"和"BigNum"列映射到名为"nums"的存储
                "strs": ["Str", "LongStr"],  # 将"Str"和"LongStr"列映射到名为"strs"的存储
            },
            df.iloc[[0]],  # 仅使用df的第一行数据进行追加操作
            "index",  # 使用"index"存储进行追加
            min_itemsize={"Str": 10, "LongStr": 100, "Num": 2},  # 指定最小项目大小限制
        )
        result = store.select_as_multiple(["index", "nums", "strs"])  # 从存储中选择"index", "nums", "strs"三个存储
        tm.assert_frame_equal(result, expected, check_index_type=True)  # 断言选择的结果与预期结果相等，检查索引类型是否一致
```