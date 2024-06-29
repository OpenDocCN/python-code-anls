# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_store.py`

```
# 导入需要的模块和库
import contextlib  # 上下文管理模块
import datetime as dt  # 时间日期处理模块
import hashlib  # 哈希函数库
import tempfile  # 创建临时文件和目录的库
import time  # 时间处理模块

import numpy as np  # 数值计算库
import pytest  # 测试框架

from pandas.compat import PY312  # 兼容性处理

import pandas as pd  # 数据分析库
from pandas import (  # 导入多个类和函数
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm  # 测试工具库
from pandas.tests.io.pytables.common import (  # 导入通用函数
    _maybe_remove,
    ensure_clean_store,
)

from pandas.io.pytables import (  # 导入 HDFStore 相关功能
    HDFStore,
    read_hdf,
)

pytestmark = pytest.mark.single_cpu  # 标记为单核 CPU 环境

tables = pytest.importorskip("tables")  # 导入并检查是否有 tables 模块


def test_context(setup_path):
    # 测试上下文管理器函数

    # 确保路径 setup_path 下的环境是干净的
    with tm.ensure_clean(setup_path) as path:
        try:
            # 尝试在 HDFStore 中执行操作，故意引发 ValueError 异常
            with HDFStore(path) as tbl:
                raise ValueError("blah")
        except ValueError:
            pass  # 捕获异常后不做处理

    # 确保路径 setup_path 下的环境是干净的
    with tm.ensure_clean(setup_path) as path:
        with HDFStore(path) as tbl:
            # 在 HDFStore 中创建一个名为 "a" 的 DataFrame 对象
            tbl["a"] = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),  # 创建数据
                columns=Index(list("ABCD"), dtype=object),  # 设置列索引
                index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 设置行索引
            )
            # 断言 HDFStore 中的对象数量为 1
            assert len(tbl) == 1
            # 断言 HDFStore 中的 "a" 对象为 DataFrame 类型
            assert type(tbl["a"]) == DataFrame


def test_no_track_times(tmp_path, setup_path):
    # 测试禁用 track_times 功能的情况（GH 32682）

    # 计算文件的校验和
    def checksum(filename, hash_factory=hashlib.md5, chunk_num_blocks=128):
        h = hash_factory()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
                h.update(chunk)
        return h.digest()

    # 创建 HDF5 文件并返回校验和
    def create_h5_and_return_checksum(tmp_path, track_times):
        path = tmp_path / setup_path
        df = DataFrame({"a": [1]})

        with HDFStore(path, mode="w") as hdf:
            hdf.put(
                "table",
                df,
                format="table",
                data_columns=True,
                index=None,
                track_times=track_times,  # 设置 track_times 参数
            )

        return checksum(path)

    # 对 track_times=False 的情况计算校验和
    checksum_0_tt_false = create_h5_and_return_checksum(tmp_path, track_times=False)
    # 对 track_times=True 的情况计算校验和
    checksum_0_tt_true = create_h5_and_return_checksum(tmp_path, track_times=True)

    # 等待一段时间确保创建了有不同创建时间的 HDF5 文件
    time.sleep(1)

    # 再次对 track_times=False 的情况计算校验和
    checksum_1_tt_false = create_h5_and_return_checksum(tmp_path, track_times=False)
    # 再次对 track_times=True 的情况计算校验和
    checksum_1_tt_true = create_h5_and_return_checksum(tmp_path, track_times=True)

    # 断言当 track_times=False 时，校验和相同
    assert checksum_0_tt_false == checksum_1_tt_false

    # 断言当 track_times=True 时，校验和不同
    assert checksum_0_tt_true != checksum_1_tt_true


def test_iter_empty(setup_path):
    # 测试空情况下的迭代行为（GH 12221）

    with ensure_clean_store(setup_path) as store:
        # 断言 store 中的迭代结果为空列表
        assert list(store) == []


def test_repr(setup_path, performance_warning):
    # 测试 HDFStore 对象的字符串表示形式
    # 这里没有具体的代码示例，因此无需添加额外的注释
    # 使用 ensure_clean_store 函数创建一个干净的存储环境，并在退出时清理
    with ensure_clean_store(setup_path) as store:
        # 打印存储对象的字符串表示形式，通常是对象的描述信息
        repr(store)
        # 打印存储对象的详细信息，如存储的内容等
        store.info()
        # 将 Series 对象存储到 store 中，键名为 "a"
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 将 Series 对象存储到 store 中，键名为 "b"
        store["b"] = Series(
            range(10), dtype="float64", index=[f"i_{i}" for i in range(10)]
        )
        # 将 DataFrame 对象存储到 store 中，键名为 "c"
        store["c"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        # 创建一个新的 DataFrame 对象 df
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 向 df 中添加列 "obj1" 和 "obj2"
        df["obj1"] = "foo"
        df["obj2"] = "bar"
        # 根据条件生成布尔列 "bool1", "bool2", "bool3"
        df["bool1"] = df["A"] > 0
        df["bool2"] = df["B"] > 0
        df["bool3"] = True
        # 向 df 中添加整数列 "int1" 和 "int2"
        df["int1"] = 1
        df["int2"] = 2
        # 向 df 中添加时间戳列 "timestamp1", "timestamp2"
        df["timestamp1"] = Timestamp("20010102")
        df["timestamp2"] = Timestamp("20010103")
        # 向 df 中添加日期时间列 "datetime1", "datetime2"
        df["datetime1"] = dt.datetime(2001, 1, 2, 0, 0)
        df["datetime2"] = dt.datetime(2001, 1, 3, 0, 0)
        # 将 df 中索引为 3 到 5 的行的 "obj1" 列设置为 NaN
        df.loc[df.index[3:6], ["obj1"]] = np.nan
        # 优化 df 的内存使用
        df = df._consolidate()

        # 在性能警告上下文中，将 df 存储到 store 中的键名为 "df"
        with tm.assert_produces_warning(performance_warning):
            store["df"] = df

        # 在 HDF 空间中创建一个名为 "bah" 的随机组
        store._handle.create_group(store._handle.root, "bah")

        # 断言存储对象的文件名出现在其字符串表示形式中
        assert store.filename in repr(store)
        # 断言存储对象的文件名出现在其字符串中
        assert store.filename in str(store)
        # 打印存储对象的详细信息
        store.info()

    # storers
    # 使用 ensure_clean_store 函数创建一个干净的存储环境，并在退出时清理
    with ensure_clean_store(setup_path) as store:
        # 创建一个新的 DataFrame 对象 df
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 将 df 追加到 store 中的键名为 "df" 的数据集中
        store.append("df", df)

        # 获取存储中 "df" 数据集的存储器对象 s
        s = store.get_storer("df")
        # 打印存储器对象 s 的字符串表示形式
        repr(s)
        # 打印存储器对象 s 的字符串形式
        str(s)
def test_contains(setup_path):
    # 使用提供的 setup_path 确保存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 向存储中添加名为 "a" 的 Series 对象
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 向存储中添加名为 "b" 的 DataFrame 对象
        store["b"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 向存储中添加名为 "foo/bar" 的 DataFrame 对象
        store["foo/bar"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 检查存储中是否包含特定键的断言
        assert "a" in store
        assert "b" in store
        assert "c" not in store
        assert "foo/bar" in store
        assert "/foo/bar" in store
        assert "/foo/b" not in store
        assert "bar" not in store

        # gh-2694: tables.NaturalNameWarning
        # 测试特定情况下的警告信息
        with tm.assert_produces_warning(
            tables.NaturalNameWarning, check_stacklevel=False
        ):
            # 向存储中添加名为 "node())" 的 DataFrame 对象
            store["node())"] = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )
        # 检查存储中是否包含特定键的断言
        assert "node())" in store


def test_versioning(setup_path):
    # 使用提供的 setup_path 确保存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 向存储中添加名为 "a" 的 Series 对象
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 向存储中添加名为 "b" 的 DataFrame 对象
        store["b"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 创建一个新的 DataFrame 对象
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=20, freq="B"),
        )
        # 尝试从存储中移除可能存在的 "df1" 键
        _maybe_remove(store, "df1")
        # 向存储中的 "df1" 添加前10行的 df 数据
        store.append("df1", df[:10])
        # 继续向存储中的 "df1" 添加后10行的 df 数据
        store.append("df1", df[10:])
        # 检查存储中特定节点的 pandas 版本属性是否正确设置
        assert store.root.a._v_attrs.pandas_version == "0.15.2"
        assert store.root.b._v_attrs.pandas_version == "0.15.2"
        assert store.root.df1._v_attrs.pandas_version == "0.15.2"

        # 尝试从存储中移除可能存在的 "df2" 键
        _maybe_remove(store, "df2")
        # 向存储中的 "df2" 添加完整的 df 数据
        store.append("df2", df)

        # 这是一个错误，因为其 table_type 是可追加的，但没有版本信息
        store.get_node("df2")._v_attrs.pandas_version = None

        # 使用 pytest 来断言特定异常消息是否匹配
        msg = "'NoneType' object has no attribute 'startswith'"
        with pytest.raises(Exception, match=msg):
            # 尝试从存储中选择 "df2"，这里应该触发异常
            store.select("df2")
    [
        (
            "/",  # 根路径
            {
                "": ({"first_group", "second_group"}, set()),  # 空路径，包含两个组，无文件
                "/first_group": (set(), {"df1", "df2"}),  # /first_group 路径，无组，包含 df1 和 df2 文件
                "/second_group": ({"third_group"}, {"df3", "s1"}),  # /second_group 路径，包含 third_group 组和 df3、s1 文件
                "/second_group/third_group": (set(), {"df4"}),  # /second_group/third_group 路径，无组，包含 df4 文件
            },
        ),
        (
            "/second_group",  # /second_group 路径
            {
                "/second_group": ({"third_group"}, {"df3", "s1"}),  # /second_group 路径，包含 third_group 组和 df3、s1 文件
                "/second_group/third_group": (set(), {"df4"}),  # /second_group/third_group 路径，无组，包含 df4 文件
            },
        ),
    ],
)
def test_walk(where, expected):
    # GH10143
    # 定义包含多种数据对象的字典
    objs = {
        "df1": DataFrame([1, 2, 3]),
        "df2": DataFrame([4, 5, 6]),
        "df3": DataFrame([6, 7, 8]),
        "df4": DataFrame([9, 10, 11]),
        "s1": Series([10, 9, 8]),
        # 下面三个对象不是 pandas 对象，应忽略
        "a1": np.array([[1, 2, 3], [4, 5, 6]]),
        "tb1": np.array([(1, 2, 3), (4, 5, 6)], dtype="i,i,i"),
        "tb2": np.array([(7, 8, 9), (10, 11, 12)], dtype="i,i,i"),
    }

    # 使用 ensure_clean_store 上下文管理器创建 HDFStore 对象，确保存储空间干净，以写入模式打开
    with ensure_clean_store("walk_groups.hdf", mode="w") as store:
        # 将 pandas 对象写入 HDFStore
        store.put("/first_group/df1", objs["df1"])
        store.put("/first_group/df2", objs["df2"])
        store.put("/second_group/df3", objs["df3"])
        store.put("/second_group/s1", objs["s1"])
        store.put("/second_group/third_group/df4", objs["df4"])
        
        # 创建非 pandas 对象
        store._handle.create_array("/first_group", "a1", objs["a1"])
        store._handle.create_table("/first_group", "tb1", obj=objs["tb1"])
        store._handle.create_table("/second_group", "tb2", obj=objs["tb2"])

        # 断言在指定路径下遍历的结果与期望长度相等
        assert len(list(store.walk(where=where))) == len(expected)
        # 遍历 HDFStore，检查每个路径、组和叶子节点
        for path, groups, leaves in store.walk(where=where):
            # 断言路径存在于期望结果中
            assert path in expected
            expected_groups, expected_frames = expected[path]
            # 断言组和叶子节点与期望的一致
            assert expected_groups == set(groups)
            assert expected_frames == set(leaves)
            # 对于每个叶子节点，根据名称从 store 中获取对象并与预期对象比较
            for leaf in leaves:
                frame_path = "/".join([path, leaf])
                obj = store.get(frame_path)
                if "df" in leaf:
                    tm.assert_frame_equal(obj, objs[leaf])
                else:
                    tm.assert_series_equal(obj, objs[leaf])


def test_getattr(setup_path):
    # 使用 ensure_clean_store 上下文管理器创建 HDFStore 对象
    with ensure_clean_store(setup_path) as store:
        # 创建 Series 对象并存储到 HDFStore
        s = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        store["a"] = s

        # 测试通过属性访问对象
        result = store.a
        tm.assert_series_equal(result, s)
        result = getattr(store, "a")
        tm.assert_series_equal(result, s)

        # 创建 DataFrame 对象并存储到 HDFStore
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        store["df"] = df
        result = store.df
        tm.assert_frame_equal(result, df)

        # 错误情况测试
        for x in ["d", "mode", "path", "handle", "complib"]:
            msg = f"'HDFStore' object has no attribute '{x}'"
            with pytest.raises(AttributeError, match=msg):
                getattr(store, x)

        # 测试非存储对象的属性访问
        for x in ["mode", "path", "handle", "complib"]:
            getattr(store, f"_{x}")


def test_store_dropna(tmp_path, setup_path):
    # 创建包含缺失值的 DataFrame 对象
    df_with_missing = DataFrame(
        {"col1": [0.0, np.nan, 2.0], "col2": [1.0, np.nan, np.nan]},
        index=list("abc"),
    )
    df_without_missing = DataFrame(
        {"col1": [0.0, 2.0], "col2": [1.0, np.nan]}, index=list("ac")
    )

    # 创建一个 DataFrame，包含两列：col1 和 col2，其中 col2 包含缺失值 NaN，指定索引为 'a' 和 'c'
    # 这是一个没有缺失值的 DataFrame 示例
    # 在测试中，确保默认情况下不会丢弃缺失值
    # 对应问题号 9382

    path = tmp_path / setup_path
    # 将 df_with_missing 对象保存为 HDF5 文件，使用表格式存储
    df_with_missing.to_hdf(path, key="df", format="table")
    # 从 HDF5 文件中重新加载数据，存入 reloaded 变量
    reloaded = read_hdf(path, "df")
    # 使用测试工具验证 df_with_missing 和 reloaded 是否相等
    tm.assert_frame_equal(df_with_missing, reloaded)

    path = tmp_path / setup_path
    # 将 df_with_missing 对象保存为 HDF5 文件，使用表格式存储，设置 dropna=False 不丢弃缺失值
    df_with_missing.to_hdf(path, key="df", format="table", dropna=False)
    # 从 HDF5 文件中重新加载数据，存入 reloaded 变量
    reloaded = read_hdf(path, "df")
    # 使用测试工具验证 df_with_missing 和 reloaded 是否相等
    tm.assert_frame_equal(df_with_missing, reloaded)

    path = tmp_path / setup_path
    # 将 df_with_missing 对象保存为 HDF5 文件，使用表格式存储，设置 dropna=True 丢弃缺失值
    df_with_missing.to_hdf(path, key="df", format="table", dropna=True)
    # 从 HDF5 文件中重新加载数据，存入 reloaded 变量
    reloaded = read_hdf(path, "df")
    # 使用测试工具验证 df_without_missing 和 reloaded 是否相等
    tm.assert_frame_equal(df_without_missing, reloaded)
# 定义测试函数，测试 DataFrame 对象写入 HDF 文件的功能，设置最小项目大小
def test_to_hdf_with_min_itemsize(tmp_path, setup_path):
    # 在临时路径下创建文件路径
    path = tmp_path / setup_path

    # 创建 DataFrame 对象
    df = DataFrame(
        {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype=object),
            "D": date_range("20130101", periods=5),
        }
    ).set_index("C")

    # 将 DataFrame 写入 HDF 文件，指定表格格式和最小项目大小
    df.to_hdf(path, key="ss3", format="table", min_itemsize={"index": 6})

    # 创建另一个 DataFrame 的副本，并在重置索引后插入更长的字符串，再写入 HDF 文件
    df2 = df.copy().reset_index().assign(C="longer").set_index("C")
    df2.to_hdf(path, key="ss3", append=True, format="table")

    # 断言读取的 HDF 文件内容与预期的合并结果一致
    tm.assert_frame_equal(read_hdf(path, "ss3"), concat([df, df2]))

    # 使用 Series 对象进行类似的操作
    df["B"].to_hdf(path, key="ss4", format="table", min_itemsize={"index": 6})
    df2["B"].to_hdf(path, key="ss4", append=True, format="table")

    # 断言读取的 HDF 文件内容与预期的合并结果一致
    tm.assert_series_equal(read_hdf(path, "ss4"), concat([df["B"], df2["B"]]))


# 测试在特定格式下使用 Series 对象写入 HDF 文件的错误处理
def test_to_hdf_errors(tmp_path, format, setup_path):
    # 创建包含特殊字符的 Series 对象
    data = ["\ud800foo"]
    ser = Series(data, index=Index(data))
    path = tmp_path / setup_path

    # 将 Series 对象写入 HDF 文件，设置格式和错误处理方式
    ser.to_hdf(path, key="table", format=format, errors="surrogatepass")

    # 读取 HDF 文件内容，确保与原始 Series 对象相同
    result = read_hdf(path, "table", errors="surrogatepass")
    tm.assert_series_equal(result, ser)


# 测试创建数据表索引的功能
def test_create_table_index(setup_path):
    # 使用 ensure_clean_store 确保测试环境清洁
    with ensure_clean_store(setup_path) as store:
        
        # 定义函数用于访问存储器中的列
        def col(t, column):
            return getattr(store.get_storer(t).table.cols, column)

        # 创建包含随机数据的 DataFrame 对象，并添加两列字符串数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df["string"] = "foo"
        df["string2"] = "bar"

        # 将 DataFrame 对象附加到存储器中，并指定数据列以进行索引
        store.append("f", df, data_columns=["string", "string2"])

        # 断言索引列是否已经创建
        assert col("f", "index").is_indexed is True
        assert col("f", "string").is_indexed is True
        assert col("f", "string2").is_indexed is True

        # 在指定列作为索引的情况下再次附加 DataFrame 对象
        store.append("f2", df, index=["string"], data_columns=["string", "string2"])

        # 断言索引列是否未被创建
        assert col("f2", "index").is_indexed is False
        assert col("f2", "string").is_indexed is True
        assert col("f2", "string2").is_indexed is False

        # 尝试在非表格格式的存储器上创建表索引，预期会引发 TypeError 异常
        _maybe_remove(store, "f2")
        store.put("f2", df)
        msg = "cannot create table index on a Fixed format store"
        with pytest.raises(TypeError, match=msg):
            store.create_table_index("f2")


# 测试使用 data_columns 参数创建数据表索引的功能
def test_create_table_index_data_columns_argument(setup_path):
    # GH 28156
    
def test_mi_data_columns(setup_path):
    # GH 14435
    # 创建一个多级索引对象，包含日期范围和整数范围，设定索引名称为"date"和"id"
    idx = MultiIndex.from_arrays(
        [date_range("2000-01-01", periods=5), range(5)], names=["date", "id"]
    )
    # 创建一个数据框，包含列"a"，使用上述多级索引对象作为索引
    df = DataFrame({"a": [1.1, 1.2, 1.3, 1.4, 1.5]}, index=idx)

    # 使用 ensure_clean_store 函数创建一个干净的数据存储环境，并将数据框 df 附加到存储中，设定数据列为 True
    with ensure_clean_store(setup_path) as store:
        store.append("df", df, data_columns=True)

        # 从存储中选择 id 等于 1 的行，期望结果与 df 中的相同，进行断言比较
        actual = store.select("df", where="id == 1")
        expected = df.iloc[[1], :]
        tm.assert_frame_equal(actual, expected)


def test_table_mixed_dtypes(setup_path):
    # frame
    # 创建一个数据框 df，包含特定数据和列的设置
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
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
    df = df._consolidate()

    # 使用 ensure_clean_store 函数创建一个干净的数据存储环境，并将数据框 df 附加到存储中，进行断言比较
    with ensure_clean_store(setup_path) as store:
        store.append("df1_mixed", df)
        tm.assert_frame_equal(store.select("df1_mixed"), df)


def test_calendar_roundtrip_issue(setup_path):
    # 8591
    # tseries 节假日部分的文档示例
    weekmask_egypt = "Sun Mon Tue Wed Thu"
    holidays = [
        "2012-05-01",
        dt.datetime(2013, 5, 1),
        np.datetime64("2014-05-01"),
    ]
    # 创建一个自定义工作日偏移量对象，使用 weekmask_egypt 和 holidays 参数
    bday_egypt = pd.offsets.CustomBusinessDay(
        holidays=holidays, weekmask=weekmask_egypt
    )
    mydt = dt.datetime(2013, 4, 30)
    # 创建一个日期范围对象 dts，从 mydt 开始，频率为 bday_egypt，包含 5 个时间点
    dts = date_range(mydt, periods=5, freq=bday_egypt)

    # 创建一个系列对象 s，将 dts 中的日期作为索引，周几作为值，并映射到对应的字符串
    s = Series(dts.weekday, dts).map(Series("Mon Tue Wed Thu Fri Sat Sun".split()))

    # 使用 ensure_clean_store 函数创建一个干净的数据存储环境，并将系列 s 存储为 "fixed" 键的数据
    with ensure_clean_store(setup_path) as store:
        store.put("fixed", s)
        # 从存储中选择 "fixed" 键的数据，进行断言比较
        result = store.select("fixed")
        tm.assert_series_equal(result, s)

        # 将系列 s 附加到存储中的 "table" 键
        store.append("table", s)
        # 从存储中选择 "table" 键的数据，进行断言比较
        result = store.select("table")
        tm.assert_series_equal(result, s)


def test_remove(setup_path):
    # 使用 ensure_clean_store 函数确保存储路径 setup_path 是干净的，并创建一个 store 上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 创建一个 Series 对象 ts，包含浮点数的序列，以日期范围作为索引
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 创建一个 DataFrame 对象 df，包含由 1.1 倍递增数组成的数据，指定列索引和行索引
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 将 ts 存储在 store 中的键 "a" 下
        store["a"] = ts
        # 将 df 存储在 store 中的键 "b" 下
        store["b"] = df
        # 调用 _maybe_remove 函数，尝试移除 store 中的键 "a"
        _maybe_remove(store, "a")
        # 断言 store 的长度为 1
        assert len(store) == 1
        # 使用 tm.assert_frame_equal 比较 df 和 store 中键 "b" 对应的值，保证它们相等
        tm.assert_frame_equal(df, store["b"])

        # 再次调用 _maybe_remove 函数，尝试移除 store 中的键 "b"
        _maybe_remove(store, "b")
        # 断言 store 的长度为 0
        assert len(store) == 0

        # 测试移除不存在的键时，期望引发 KeyError 异常，异常信息应包含指定字符串
        with pytest.raises(
            KeyError, match="'No object named a_nonexistent_store in the file'"
        ):
            store.remove("a_nonexistent_store")

        # 将 ts 存储在 store 中的键 "a" 下
        store["a"] = ts
        # 将 df 存储在 store 中的键 "b/foo" 下
        store["b/foo"] = df
        # 尝试移除 store 中的键 "foo"，调用 _maybe_remove 函数
        _maybe_remove(store, "foo")
        # 再次尝试移除 store 中的键 "b/foo"，调用 _maybe_remove 函数
        _maybe_remove(store, "b/foo")
        # 断言 store 的长度为 1
        assert len(store) == 1

        # 将 ts 存储在 store 中的键 "a" 下
        store["a"] = ts
        # 将 df 存储在 store 中的键 "b/foo" 下
        store["b/foo"] = df
        # 尝试移除 store 中的键 "b"，调用 _maybe_remove 函数
        _maybe_remove(store, "b")
        # 断言 store 的长度为 1，因为只有键 "a" 被保留了下来
        assert len(store) == 1

        # 测试使用 del 操作符删除 store 中的键 "a" 和 "b"
        store["a"] = ts
        store["b"] = df
        del store["a"]
        del store["b"]
        # 断言 store 的长度为 0，所有键都被删除了
        assert len(store) == 0
# 定义一个测试函数，用于测试相同名称的作用域
def test_same_name_scoping(setup_path):
    # 在确保存储环境清洁的情况下，创建存储对象
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含随机数据的DataFrame对象，使用日期范围作为索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 2)),
            index=date_range("20130101", periods=20),
        )
        # 将DataFrame存储到存储对象中，格式为表格
        store.put("df", df, format="table")
        # 根据条件选取DataFrame中符合条件的部分作为预期结果
        expected = df[df.index > Timestamp("20130105")]

        # 使用存储对象的select方法根据条件进行查询，将结果存储在result中
        result = store.select("df", "index>datetime.datetime(2013,1,5)")
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 改变命名空间中 'datetime' 指向的对象，影响 'select' 方法的查找行为

        # 技术上是一个错误，但允许这样使用
        result = store.select("df", "index>datetime.datetime(2013,1,5)")
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 使用 'datetime' 对象进行查询，而不是完整的 'datetime.datetime'
        result = store.select("df", "index>datetime(2013,1,5)")
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试存储索引名称的设置
def test_store_index_name(setup_path):
    # 创建一个DataFrame对象，包含一组特定数据和索引名称为 'foo'
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    df.index.name = "foo"

    # 在确保存储环境清洁的情况下，创建存储对象
    with ensure_clean_store(setup_path) as store:
        # 将DataFrame存储到存储对象中，键名为 'frame'
        store["frame"] = df
        # 从存储对象中重新获取 'frame' 键的值
        recons = store["frame"]
        # 断言重新获取的DataFrame与原始DataFrame相等
        tm.assert_frame_equal(recons, df)


# 使用pytest的参数化装饰器定义多个测试用例，测试存储索引名称为numpy字符串时的行为
@pytest.mark.parametrize("tz", [None, "US/Pacific"])
@pytest.mark.parametrize("table_format", ["table", "fixed"])
def test_store_index_name_numpy_str(tmp_path, table_format, setup_path, unit, tz):
    # 创建两个DatetimeIndex对象，分别具有 'cols\u05d2' 和 'rows\u05d0' 作为名称
    idx = (
        DatetimeIndex(
            [dt.date(2000, 1, 1), dt.date(2000, 1, 2)],
            name="cols\u05d2",
        )
        .tz_localize(tz)
        .as_unit(unit)
    )
    idx1 = (
        DatetimeIndex(
            [dt.date(2010, 1, 1), dt.date(2010, 1, 2)],
            name="rows\u05d0",
        )
        .as_unit(unit)
        .tz_localize(tz)
    )
    # 创建一个DataFrame对象，使用特定索引和列
    df = DataFrame(np.arange(4).reshape(2, 2), columns=idx, index=idx1)

    # 将DataFrame对象存储为HDF格式文件
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", format=table_format)
    # 从HDF文件中读取名为 'df' 的DataFrame对象
    df2 = read_hdf(path, "df")

    # 断言读取的DataFrame与原始DataFrame相等，检查索引和列的名称
    tm.assert_frame_equal(df, df2, check_names=True)

    # 断言索引名称和列名称为字符串类型
    assert isinstance(df2.index.name, str)
    assert isinstance(df2.columns.name, str)


# 定义一个测试函数，用于测试存储Series对象时的行为
def test_store_series_name(setup_path):
    # 创建一个DataFrame对象
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 从DataFrame中获取 'A' 列作为Series对象
    series = df["A"]

    # 在确保存储环境清洁的情况下，创建存储对象
    with ensure_clean_store(setup_path) as store:
        # 将Series对象存储到存储对象中，键名为 'series'
        store["series"] = series
        # 从存储对象中重新获取 'series' 键的值
        recons = store["series"]
        # 断言重新获取的Series与原始Series相等
        tm.assert_series_equal(recons, series)


# 定义一个测试函数，用于测试覆盖存储节点时的行为
def test_overwrite_node(setup_path):
    # 使用 ensure_clean_store 函数创建一个临时存储空间，并将其赋值给 store 变量
    with ensure_clean_store(setup_path) as store:
        # 在 store 中创建一个名为 "a" 的 DataFrame，数据为 10 行 4 列的随机标准正态分布数据
        # 列名为 A、B、C、D，索引为从 "2000-01-01" 开始的 10 个工作日频率的日期
        store["a"] = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 创建一个名为 ts 的 Series，数据为从 0 到 9 的浮点数，索引为从 "2020-01-01" 开始的 10 个日期
        ts = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        # 将 ts 赋值给 store 中的 "a"，覆盖之前的 DataFrame
        store["a"] = ts
    
        # 使用 tm.assert_series_equal 函数比较 store 中的 "a" 和 ts 是否相等
        tm.assert_series_equal(store["a"], ts)
# 定义一个测试函数，用于测试坐标操作
def test_coordinates(setup_path):
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    # 使用 ensure_clean_store 函数创建一个存储对象
    with ensure_clean_store(setup_path) as store:
        # 如果存在名为 "df" 的表，则删除
        _maybe_remove(store, "df")
        # 将 DataFrame 添加到存储对象中
        store.append("df", df)

        # all
        # 选择 "df" 表的所有坐标
        c = store.select_as_coordinates("df")
        # 断言坐标值等于索引的范围
        assert (c.values == np.arange(len(df.index))).all()

        # get coordinates back & test vs frame
        # 再次删除名为 "df" 的表
        _maybe_remove(store, "df")

        # 创建一个新的 DataFrame
        df = DataFrame({"A": range(5), "B": range(5)})
        # 将新的 DataFrame 添加到存储对象中
        store.append("df", df)
        # 选择 "df" 表中满足条件 "index<3" 的坐标
        c = store.select_as_coordinates("df", ["index<3"])
        # 断言坐标值等于前3个索引
        assert (c.values == np.arange(3)).all()
        # 选择存储对象中 "df" 表中符合坐标条件的数据
        result = store.select("df", where=c)
        # 期望的结果是 DataFrame 中前3行的数据
        expected = df.loc[0:2, :]
        # 检查结果是否与期望相符
        tm.assert_frame_equal(result, expected)

        # 选择 "df" 表中满足条件 "index>=3" 和 "index<=4" 的坐标
        c = store.select_as_coordinates("df", ["index>=3", "index<=4"])
        # 断言坐标值等于索引为3和4的值
        assert (c.values == np.arange(2) + 3).all()
        # 选择存储对象中 "df" 表中符合坐标条件的数据
        result = store.select("df", where=c)
        # 期望的结果是 DataFrame 中第4和第5行的数据
        expected = df.loc[3:4, :]
        # 检查结果是否与期望相符
        tm.assert_frame_equal(result, expected)
        # 断言坐标对象的类型为 Index

        # multiple tables
        # 再次删除名为 "df1" 和 "df2" 的表
        _maybe_remove(store, "df1")
        _maybe_remove(store, "df2")
        # 创建两个包含随机数据的 DataFrame
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 复制 df1 并重命名列名
        df2 = df1.copy().rename(columns="{}_2".format)
        # 将两个 DataFrame 添加到存储对象中
        store.append("df1", df1, data_columns=["A", "B"])
        store.append("df2", df2)

        # 选择 "df1" 表中满足条件 "A>0" 和 "B>0" 的坐标
        c = store.select_as_coordinates("df1", ["A>0", "B>0"])
        # 选择存储对象中 "df1" 表中符合坐标条件的数据
        df1_result = store.select("df1", c)
        df2_result = store.select("df2", c)
        # 将两个结果合并成一个 DataFrame
        result = concat([df1_result, df2_result], axis=1)

        # 期望的结果是将 df1 和 df2 合并后，且满足条件 A>0 和 B>0 的数据
        expected = concat([df1, df2], axis=1)
        expected = expected[(expected.A > 0) & (expected.B > 0)]
        # 检查结果是否与期望相符
        tm.assert_frame_equal(result, expected, check_freq=False)
        # FIXME: 2021-01-18 on some (mostly windows) builds we get freq=None
        #  but expect freq="18B"

    # pass array/mask as the coordinates
    # 使用 ensure_clean_store 上下文管理器来确保 setup_path 的存储被清理
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含随机标准正态分布数据的 DataFrame，形状为 (1000, 2)，索引为从 "20000101" 开始的 1000 个日期
        df = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 2)),
            index=date_range("20000101", periods=1000),
        )
        # 将 DataFrame df 追加到存储中，键为 "df"
        store.append("df", df)
        # 从存储中选择 "df" 的 "index" 列
        c = store.select_column("df", "index")
        # 获取索引为 5 月份的位置
        where = c[DatetimeIndex(c).month == 5].index
        # 从 DataFrame df 中提取预期的数据，这些数据的索引是 where 所指示的位置
        expected = df.iloc[where]

        # 测试用例：按照 where 所指示的位置从存储中选择数据，期望结果与 expected 相同
        result = store.select("df", where=where)
        tm.assert_frame_equal(result, expected)

        # 测试用例：按照 where 所指示的位置从存储中选择数据，期望结果与 expected 相同
        result = store.select("df", where=where)
        tm.assert_frame_equal(result, expected)

        # 测试用例：尝试使用不支持的 where 类型进行选择，应该抛出 TypeError 异常，并匹配特定的错误信息
        msg = (
            "where must be passed as a string, PyTablesExpr, "
            "or list-like of PyTablesExpr"
        )
        with pytest.raises(TypeError, match=msg):
            store.select("df", where=np.arange(len(df), dtype="float64"))

        with pytest.raises(TypeError, match=msg):
            store.select("df", where=np.arange(len(df) + 1))

        with pytest.raises(TypeError, match=msg):
            store.select("df", where=np.arange(len(df)), start=5)

        with pytest.raises(TypeError, match=msg):
            store.select("df", where=np.arange(len(df)), start=5, stop=10)

        # 测试用例：按照特定的筛选条件从存储中选择数据，期望结果与筛选后的 DataFrame 相同
        selection = date_range("20000101", periods=500)
        result = store.select("df", where="index in selection")
        expected = df[df.index.isin(selection)]
        tm.assert_frame_equal(result, expected)

        # 测试用例：按照给定的列表位置从存储中选择数据，期望结果与指定位置的 DataFrame 相同
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        store.append("df2", df)
        result = store.select("df2", where=[0, 3, 5])
        expected = df.iloc[[0, 3, 5]]
        tm.assert_frame_equal(result, expected)

        # 测试用例：按照布尔列表从存储中选择数据，期望结果与布尔条件过滤后的 DataFrame 相同
        where = [True] * 10
        where[-2] = False
        result = store.select("df2", where=where)
        expected = df.loc[where]
        tm.assert_frame_equal(result, expected)

        # 测试用例：按照给定的起始和结束位置从存储中选择数据，期望结果与切片后的 DataFrame 相同
        result = store.select("df2", start=5, stop=10)
        expected = df[5:10]
        tm.assert_frame_equal(result, expected)
def test_start_stop_table(setup_path):
    # 使用 ensure_clean_store 上下文管理器来确保 store 清空并关闭
    with ensure_clean_store(setup_path) as store:
        # 创建一个 DataFrame 对象 df，包含两列 A 和 B，每列包含 20 个随机数
        df = DataFrame(
            {
                "A": np.random.default_rng(2).random(20),
                "B": np.random.default_rng(2).random(20),
            }
        )
        # 将 DataFrame 对象 df 添加到 store 中，键名为 "df"
        store.append("df", df)

        # 从 store 中选择 "df" 表的 "A" 列，并指定起始行和结束行（0到4），返回结果为 DataFrame
        result = store.select("df", "columns=['A']", start=0, stop=5)
        # 创建预期的 DataFrame，包含 df 中前 5 行的 "A" 列数据
        expected = df.loc[0:4, ["A"]]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 选择超出范围的行（30到40），应该返回空结果
        result = store.select("df", "columns=['A']", start=30, stop=40)
        # 断言结果长度为 0
        assert len(result) == 0
        # 创建预期的 DataFrame，包含 df 中第 30 到 40 行的 "A" 列数据
        expected = df.loc[30:40, ["A"]]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)


def test_start_stop_multiple(setup_path):
    # GH 16209
    with ensure_clean_store(setup_path) as store:
        # 创建一个 DataFrame 对象 df，包含两列 "foo" 和 "bar"，每列各包含两个值
        df = DataFrame({"foo": [1, 2], "bar": [1, 2]})

        # 将 df 对象按照选择器 "selector" 添加到 store 中
        store.append_to_multiple(
            {"selector": ["foo"], "data": None}, df, selector="selector"
        )

        # 从 store 中选择多个数据项 ["selector", "data"]，并指定起始行和结束行（0到1），返回结果为 DataFrame
        result = store.select_as_multiple(
            ["selector", "data"], selector="selector", start=0, stop=1
        )
        # 创建预期的 DataFrame，包含 df 中第 0 行的 "foo" 和 "bar" 列数据
        expected = df.loc[[0], ["foo", "bar"]]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)


def test_start_stop_fixed(setup_path):
    with ensure_clean_store(setup_path) as store:
        # fixed, GH 8287
        # 创建一个 DataFrame 对象 df，包含两列 "A" 和 "B"，每列包含 20 个随机数，
        # 并设置日期索引为 2013 年 1 月 1 日开始的 20 个日期
        df = DataFrame(
            {
                "A": np.random.default_rng(2).random(20),
                "B": np.random.default_rng(2).random(20),
            },
            index=date_range("20130101", periods=20),
        )
        # 将 df 对象添加到 store 中，键名为 "df"
        store.put("df", df)

        # 从 store 中选择 "df" 表的所有列，并指定起始行和结束行（0到5），返回结果为 DataFrame
        result = store.select("df", start=0, stop=5)
        # 创建预期的 DataFrame，包含 df 中第 0 到 5 行的所有列数据
        expected = df.iloc[0:5, :]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 从 store 中选择 "df" 表的所有列，并指定起始行和结束行（5到10），返回结果为 DataFrame
        result = store.select("df", start=5, stop=10)
        # 创建预期的 DataFrame，包含 df 中第 5 到 10 行的所有列数据
        expected = df.iloc[5:10, :]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 选择超出范围的行（30到40），应该返回空结果
        result = store.select("df", start=30, stop=40)
        # 创建预期的 DataFrame，包含 df 中第 30 到 40 行的所有列数据
        expected = df.iloc[30:40, :]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建一个 Series 对象 s，包含 df 中的 "A" 列数据
        s = df.A
        # 将 Series 对象 s 添加到 store 中，键名为 "s"
        store.put("s", s)

        # 从 store 中选择 "s" 表，指定起始行和结束行（0到5），返回结果为 Series
        result = store.select("s", start=0, stop=5)
        # 创建预期的 Series，包含 s 中第 0 到 5 行的数据
        expected = s.iloc[0:5]
        # 断言 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 从 store 中选择 "s" 表，指定起始行和结束行（5到10），返回结果为 Series
        result = store.select("s", start=5, stop=10)
        # 创建预期的 Series，包含 s 中第 5 到 10 行的数据
        expected = s.iloc[5:10]
        # 断言 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个稀疏 DataFrame 对象 df，暂时没有实现相关功能
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df.iloc[3:5, 1:3] = np.nan
        df.iloc[8:10, -2] = np.nan


def test_select_filter_corner(setup_path, request):
    # 创建一个具有 50 行 100 列标准正态分布随机数的 DataFrame 对象 df
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 100)))
    # 设置 df 的行索引和列索引为三位数格式的字符串
    df.index = [f"{c:3d}" for c in df.index]
    df.columns = [f"{c:3d}" for c in df.columns]
    # 使用 ensure_clean_store 上下文管理器，确保 setup_path 中的存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame df 以表格格式存储在 store 中，键名为 "frame"
        store.put("frame", df, format="table")

        # 应用 pytest.mark.xfail 标记，用于标识 PY312 中的 AST 变化会导致失败
        request.applymarker(
            pytest.mark.xfail(
                PY312,
                reason="AST change in PY312",
                raises=ValueError,
            )
        )

        # 设定查询条件 crit，选择 df 的前 75 列，并从 store 中选取符合条件的数据
        crit = "columns=df.columns[:75]"
        result = store.select("frame", [crit])
        # 断言选取的结果与 df 中相应的数据相等
        tm.assert_frame_equal(result, df.loc[:, df.columns[:75]])

        # 更新查询条件 crit，选择 df 的前 75 列中每隔一列的数据，并从 store 中选取符合条件的数据
        crit = "columns=df.columns[:75:2]"
        result = store.select("frame", [crit])
        # 断言选取的结果与 df 中相应的数据相等
        tm.assert_frame_equal(result, df.loc[:, df.columns[:75:2]])
# 测试使用路径操作 pathlib 的功能
def test_path_pathlib():
    # 创建一个包含浮点数的 DataFrame，形状为 (30, 4)，列为 ['A', 'B', 'C', 'D']，行索引为 ['i-0', 'i-1', ..., 'i-29']
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # 使用 tm.round_trip_pathlib 方法，测试将 DataFrame 写入 HDF 文件并读回的完整路径操作
    result = tm.round_trip_pathlib(
        lambda p: df.to_hdf(p, key="df"),  # 将 DataFrame 写入 HDF 文件的函数
        lambda p: read_hdf(p, "df")  # 从 HDF 文件中读取 DataFrame 的函数
    )
    
    # 断言写入并读取后的 DataFrame 是否与原始 DataFrame 相等
    tm.assert_frame_equal(df, result)


# 使用不同参数组合来测试包含混合数据表的连续数据选择
@pytest.mark.parametrize("start, stop", [(0, 2), (1, 2), (None, None)])
def test_contiguous_mixed_data_table(start, stop, setup_path):
    # 创建一个包含 'a' 和 'b' 两列的 DataFrame，'a' 列为日期格式，'b' 列为字符串
    df = DataFrame(
        {
            "a": Series([20111010, 20111011, 20111012]),
            "b": Series(["ab", "cd", "ab"]),
        }
    )

    # 使用 ensure_clean_store(setup_path) 来确保存储空间的干净状态
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame 追加到名为 'test_dataset' 的数据集中
        store.append("test_dataset", df)

        # 从存储中选择起始到结束位置之间的数据，并将结果与 DataFrame 的切片进行断言比较
        result = store.select("test_dataset", start=start, stop=stop)
        tm.assert_frame_equal(df[start:stop], result)


# 测试使用 pathlib 操作 HDFStore 的写入和读取功能
def test_path_pathlib_hdfstore():
    # 创建一个包含浮点数的 DataFrame，形状为 (30, 4)，列为 ['A', 'B', 'C', 'D']，行索引为 ['i-0', 'i-1', ..., 'i-29']
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )

    # 定义写入函数，使用 HDFStore 将 DataFrame 写入指定路径
    def writer(path):
        with HDFStore(path) as store:
            df.to_hdf(store, key="df")

    # 定义读取函数，使用 HDFStore 从指定路径读取 DataFrame
    def reader(path):
        with HDFStore(path) as store:
            return read_hdf(store, "df")

    # 使用 tm.round_trip_pathlib 方法测试将 DataFrame 写入 HDF 文件并读回的完整路径操作
    result = tm.round_trip_pathlib(writer, reader)
    
    # 断言写入并读取后的 DataFrame 是否与原始 DataFrame 相等
    tm.assert_frame_equal(df, result)


# 测试使用路径和本地路径操作的 pickle 功能
def test_pickle_path_localpath():
    # 创建一个包含浮点数的 DataFrame，形状为 (30, 4)，列为 ['A', 'B', 'C', 'D']，行索引为 ['i-0', 'i-1', ..., 'i-29']
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    
    # 使用 tm.round_trip_pathlib 方法测试将 DataFrame 写入 HDF 文件并读回的完整路径操作
    result = tm.round_trip_pathlib(
        lambda p: df.to_hdf(p, key="df"),  # 将 DataFrame 写入 HDF 文件的函数
        lambda p: read_hdf(p, "df")  # 从 HDF 文件中读取 DataFrame 的函数
    )
    
    # 断言写入并读取后的 DataFrame 是否与原始 DataFrame 相等
    tm.assert_frame_equal(df, result)


# 使用不同参数 propindexes 来测试复制操作的功能
@pytest.mark.parametrize("propindexes", [True, False])
def test_copy(propindexes):
    # 创建一个包含浮点数的 DataFrame，形状为 (30, 4)，列为 ['A', 'B', 'C', 'D']，行索引为 ['i-0', 'i-1', ..., 'i-29']
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 使用 tm.ensure_clean() 确保在上下文管理器中路径被清理
    with tm.ensure_clean() as path:
        # 使用 HDFStore 打开路径为 path 的 HDF 文件，准备写入数据框 df，并指定列 "A" 作为数据列
        with HDFStore(path) as st:
            st.append("df", df, data_columns=["A"])
        # 使用 tempfile.NamedTemporaryFile() 创建一个临时文件对象 new_f
        with tempfile.NamedTemporaryFile() as new_f:
            # 再次打开 HDFStore，读取路径为 path 的 HDF 文件
            with HDFStore(path) as store:
                # 使用 store.copy() 方法将 HDF 文件中的数据复制到临时文件 new_f 中，并指定复制参数 propindexes
                with contextlib.closing(
                    store.copy(new_f.name, keys=None, propindexes=propindexes)
                ) as tstore:
                    # 检查 HDFStore 对象 store 的键列表
                    keys = store.keys()
                    # 断言 store 和 tstore 的键集合应该相同
                    assert set(keys) == set(tstore.keys())
                    # 遍历 tstore 的键列表
                    for k in tstore.keys():
                        # 如果 tstore 中对应键的存储对象是表格（is_table 为 True）
                        if tstore.get_storer(k).is_table:
                            # 获取新表格存储对象 new_t 和原始表格存储对象 orig_t
                            new_t = tstore.get_storer(k)
                            orig_t = store.get_storer(k)

                            # 断言新表格和原始表格的行数 nrows 应该相等
                            assert orig_t.nrows == new_t.nrows

                            # 检查属性索引 propindexes
                            if propindexes:
                                # 遍历原始表格的轴对象
                                for a in orig_t.axes:
                                    # 如果轴对象 a 被索引
                                    if a.is_indexed:
                                        # 断言新表格的列 a.name 应该被索引
                                        assert new_t[a.name].is_indexed
# 测试处理重复的列名问题
def test_duplicate_column_name(tmp_path, setup_path):
    # 创建包含重复列名的 DataFrame
    df = DataFrame(columns=["a", "a"], data=[[0, 0]])

    # 设置文件路径
    path = tmp_path / setup_path
    # 定义异常消息
    msg = "Columns index has to be unique for fixed format"
    # 检查写入 HDF 文件时是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", format="fixed")

    # 将 DataFrame 以表格格式写入 HDF 文件
    df.to_hdf(path, key="df", format="table")
    # 从 HDF 文件中读取数据
    other = read_hdf(path, "df")

    # 使用测试工具检查 DataFrame 是否相等
    tm.assert_frame_equal(df, other)
    # 使用 assert 检查 DataFrame 是否相等
    assert df.equals(other)
    # 使用 assert 检查 DataFrame 是否相等
    assert other.equals(df)


# 测试保留时间索引的类型
def test_preserve_timedeltaindex_type(setup_path):
    # 创建随机数据的 DataFrame
    df = DataFrame(np.random.default_rng(2).normal(size=(10, 5)))
    # 设置时间增量索引
    df.index = timedelta_range(start="0s", periods=10, freq="1s", name="example")

    # 使用确保清洁的存储环境进行测试
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame 存储到 store 中
        store["df"] = df
        # 使用测试工具检查存储的 DataFrame 是否与原始的 df 相等
        tm.assert_frame_equal(store["df"], df)


# 测试修改多级索引的列
def test_columns_multiindex_modified(tmp_path, setup_path):
    # BUG: 7212

    # 创建具有随机数据的 DataFrame，设置行索引和列索引
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    # 设置行索引的名称
    df.index.name = "letters"
    # 将列 'E' 添加到索引中，形成多级索引
    df = df.set_index(keys="E", append=True)

    # 创建要写入的数据列
    data_columns = df.index.names + df.columns.tolist()
    # 设置文件路径
    path = tmp_path / setup_path
    # 将 DataFrame 以附加模式写入 HDF 文件，指定数据列和禁用索引
    df.to_hdf(
        path,
        key="df",
        mode="a",
        append=True,
        data_columns=data_columns,
        index=False,
    )
    # 定义要加载的列
    cols2load = list("BCD")
    # 复制要加载的列
    cols2load_original = list(cols2load)
    # 读取 HDF 文件，确保读取操作不会在原地修改 cols2load
    read_hdf(path, "df", columns=cols2load)
    # 使用 assert 检查原始的 cols2load 是否等于修改后的 cols2load
    assert cols2load_original == cols2load


# 使用参数化测试验证具有对象列名的 HDF 写入操作失败
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize(
    "columns",
    [
        Index([0, 1], dtype=np.int64),
        Index([0.0, 1.0], dtype=np.float64),
        date_range("2020-01-01", periods=2),
        timedelta_range("1 day", periods=2),
        period_range("2020-01-01", periods=2, freq="D"),
    ],
)
def test_to_hdf_with_object_column_names_should_fail(tmp_path, setup_path, columns):
    # GH9057
    # 创建包含随机数据的 DataFrame，使用指定的列名
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=columns)
    # 设置文件路径
    path = tmp_path / setup_path
    # 定义异常消息
    msg = "cannot have non-object label DataIndexableCol"
    # 检查写入 HDF 文件时是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.to_hdf(path, key="df", format="table", data_columns=True)


# 使用参数化测试验证具有对象列名的 HDF 写入操作成功
@pytest.mark.parametrize("dtype", [None, "category"])
def test_to_hdf_with_object_column_names_should_run(tmp_path, setup_path, dtype):
    # GH9057
    # 创建包含随机数据的 DataFrame，使用对象类型的列名
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 2)),
        columns=Index(["a", "b"], dtype=dtype),
    )
    # 设置文件路径
    path = tmp_path / setup_path
    # 将 DataFrame 以表格格式写入 HDF 文件，指定数据列
    df.to_hdf(path, key="df", format="table", data_columns=True)
    # 从 HDF 文件中读取特定索引的结果
    result = read_hdf(path, "df", where=f"index = [{df.index[0]}]")
    # 使用 assert 检查结果长度是否大于 0
    assert len(result)


# 测试 HDF 存储的步长设置
def test_hdfstore_strides(setup_path):
    # GH22073
    # 创建包含两列的 DataFrame，并指定列名
    df = DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    # 使用 ensure_clean_store 函数确保在 setup_path 中的存储处于干净状态，并创建一个存储对象 store
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame df 存储到 store 中，键名为 "df"
        store.put("df", df)
        # 断言 DataFrame df 的列 "a" 的内存步长与 store 中键名为 "df" 的 DataFrame 的列 "a" 的内存步长相同
        assert df["a"].values.strides == store["df"]["a"].values.strides
# 定义一个函数用于测试存储布尔值索引的情况，使用临时路径和设置路径作为参数
def test_store_bool_index(tmp_path, setup_path):
    # GH#48667
    # 创建一个DataFrame对象，包含一个值为1的单元格，列名为True，索引为False的布尔类型
    df = DataFrame([[1]], columns=[True], index=Index([False], dtype="bool"))
    # 复制DataFrame以备后用
    expected = df.copy()

    # # Test to make sure defaults are to not drop.
    # # Corresponding to Issue 9382
    # 设置文件路径，结合临时路径和设置路径
    path = tmp_path / setup_path
    # 将DataFrame写入HDF5文件，使用键"a"
    df.to_hdf(path, key="a")
    # 从HDF5文件中读取数据，键为"a"，返回结果作为DataFrame对象
    result = read_hdf(path, "a")
    # 断言预期结果与读取结果相等
    tm.assert_frame_equal(expected, result)
```