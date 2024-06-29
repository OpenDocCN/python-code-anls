# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_read.py`

```
# 导入所需模块和函数
from contextlib import closing
from pathlib import Path
import re

import numpy as np
import pytest

# 导入 pandas 库及其组件
import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Series,
    _testing as tm,
    date_range,
    read_hdf,
)

# 导入测试相关的辅助函数和类
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

# 导入 PyTables 相关类
from pandas.io.pytables import TableIterator

# 定义 pytest 标记
pytestmark = pytest.mark.single_cpu


def test_read_missing_key_close_store(tmp_path, setup_path):
    # GH 25766
    # 设置临时文件路径
    path = tmp_path / setup_path
    # 创建 DataFrame 对象并写入 HDF 文件
    df = DataFrame({"a": range(2), "b": range(2)})
    df.to_hdf(path, key="k1")

    # 使用 pytest 检查读取不存在的键时是否引发 KeyError 异常
    with pytest.raises(KeyError, match="'No object named k2 in the file'"):
        read_hdf(path, "k2")

    # 确保在出现 KeyError 后文件能正常关闭，进行写操作
    df.to_hdf(path, key="k2")


def test_read_index_error_close_store(tmp_path, setup_path):
    # GH 25766
    # 设置临时文件路径
    path = tmp_path / setup_path
    # 创建空索引的 DataFrame 并写入 HDF 文件
    df = DataFrame({"A": [], "B": []}, index=[])
    df.to_hdf(path, key="k1")

    # 使用 pytest 检查读取时索引错误是否引发 IndexError 异常
    with pytest.raises(IndexError, match=r"list index out of range"):
        read_hdf(path, "k1", stop=0)

    # 确保在出现 IndexError 后文件能正常关闭，进行写操作
    df.to_hdf(path, key="k1")


def test_read_missing_key_opened_store(tmp_path, setup_path):
    # GH 28699
    # 设置临时文件路径
    path = tmp_path / setup_path
    # 创建 DataFrame 对象并写入 HDF 文件
    df = DataFrame({"a": range(2), "b": range(2)})
    df.to_hdf(path, key="k1")

    # 使用 HDFStore 打开文件，并使用 pytest 检查读取不存在的键时是否引发 KeyError 异常
    with HDFStore(path, "r") as store:
        with pytest.raises(KeyError, match="'No object named k2 in the file'"):
            read_hdf(store, "k2")

        # 确保在出现 KeyError 后仍能从文件中读取数据
        read_hdf(store, "k1")


def test_read_column(setup_path):
    # 创建具有随机数据的 DataFrame，包含日期时间索引和对象类型的列
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )


def test_pytables_native_read(datapath):
    # 使用 ensure_clean_store 打开 PyTables 文件，并检查返回的数据类型是否为 DataFrame
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf/pytables_native.h5"), mode="r"
    ) as store:
        d2 = store["detector/readout"]
    assert isinstance(d2, DataFrame)


@pytest.mark.skipif(is_platform_windows(), reason="native2 read fails oddly on windows")
def test_pytables_native2_read(datapath):
    # 使用 ensure_clean_store 打开另一个 PyTables 文件，并检查返回的数据类型是否为 DataFrame
    with ensure_clean_store(
        datapath("io", "data", "legacy_hdf", "pytables_native2.h5"), mode="r"
    ) as store:
        str(store)
        d1 = store["detector"]
    assert isinstance(d1, DataFrame)


def test_read_hdf_open_store(tmp_path, setup_path):
    # GH10330
    # 创建具有随机数据的 DataFrame，并设置索引名称和复合索引
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    df.index.name = "letters"
    df = df.set_index(keys="E", append=True)
    # 构建完整的文件路径，使用 tmp_path 和 setup_path 变量
    path = tmp_path / setup_path
    # 将 DataFrame df 写入 HDF5 文件中，使用 "df" 作为键名，写入模式为覆盖写入 ("w")
    df.to_hdf(path, key="df", mode="w")
    # 使用 read_hdf 函数直接从 HDF5 文件中读取数据到 direct 变量
    direct = read_hdf(path, "df")
    # 使用 HDFStore 打开 HDF5 文件，只读模式 ("r")，并将其作为 store 上下文管理器
    with HDFStore(path, mode="r") as store:
        # 使用 read_hdf 函数从 store 中读取数据到 indirect 变量
        indirect = read_hdf(store, "df")
        # 使用 tm.assert_frame_equal 检查 direct 和 indirect 是否相等
        tm.assert_frame_equal(direct, indirect)
        # 断言 HDFStore 是否处于打开状态
        assert store.is_open
# 测试函数：验证从 HDF5 文件读取的 DataFrame 的索引不是原始 recarray 的视图
def test_read_hdf_index_not_view(tmp_path, setup_path):
    # 创建一个随机数据的 DataFrame，指定索引和列名
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=[0, 1, 2, 3],
        columns=list("ABCDE"),
    )

    # 将 DataFrame 写入 HDF5 文件
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="w", format="table")

    # 从 HDF5 文件中读取 DataFrame
    df2 = read_hdf(path, "df")
    
    # 断言：验证 DataFrame 的索引不是原始 recarray 的视图
    assert df2.index._data.base is None
    # 断言：验证两个 DataFrame 在内容上是否相等
    tm.assert_frame_equal(df, df2)


# 测试函数：验证从 HDF5 文件迭代读取的正确性
def test_read_hdf_iterator(tmp_path, setup_path):
    # 创建一个随机数据的 DataFrame，指定索引和列名
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    df.index.name = "letters"
    df = df.set_index(keys="E", append=True)

    # 将 DataFrame 写入 HDF5 文件
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="w", format="t")

    # 从 HDF5 文件中直接读取 DataFrame
    direct = read_hdf(path, "df")
    # 使用迭代器从 HDF5 文件中读取 DataFrame
    iterator = read_hdf(path, "df", iterator=True)
    with closing(iterator.store):
        # 断言：验证迭代器类型是否为 TableIterator
        assert isinstance(iterator, TableIterator)
        # 通过迭代器获取的 DataFrame
        indirect = next(iterator.__iter__())
    # 断言：验证直接读取和迭代读取得到的 DataFrame 在内容上是否相等
    tm.assert_frame_equal(direct, indirect)


# 测试函数：验证从 HDF5 文件读取时没有指定 key 的情况
def test_read_nokey(tmp_path, setup_path):
    # 创建一个随机数据的 DataFrame，指定索引和列名
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )

    # 将 DataFrame 写入 HDF5 文件，指定 key 为 "df"
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="a")

    # 重新读取 HDF5 文件中的 DataFrame，不指定 key
    reread = read_hdf(path)
    # 断言：验证重新读取的 DataFrame 在内容上是否与原始 DataFrame 相等
    tm.assert_frame_equal(df, reread)

    # 再次将 DataFrame 写入 HDF5 文件，指定 key 为 "df2"
    df.to_hdf(path, key="df2", mode="a")

    # 断言：验证在 HDF5 文件包含多个数据集时未指定 key 会抛出 ValueError 异常
    msg = "key must be provided when HDF5 file contains multiple datasets."
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)


# 测试函数：验证从 HDF5 文件读取时没有指定 key 的情况（使用 format="table"）
def test_read_nokey_table(tmp_path, setup_path):
    # 创建一个包含分类数据的 DataFrame
    df = DataFrame({"i": range(5), "c": Series(list("abacd"), dtype="category")})

    # 将 DataFrame 写入 HDF5 文件，指定 key 为 "df"，格式为 "table"
    path = tmp_path / setup_path
    df.to_hdf(path, key="df", mode="a", format="table")

    # 重新读取 HDF5 文件中的 DataFrame，不指定 key
    reread = read_hdf(path)
    # 断言：验证重新读取的 DataFrame 在内容上是否与原始 DataFrame 相等
    tm.assert_frame_equal(df, reread)

    # 再次将 DataFrame 写入 HDF5 文件，指定 key 为 "df2"
    df.to_hdf(path, key="df2", mode="a", format="table")

    # 断言：验证在 HDF5 文件包含多个数据集时未指定 key 会抛出 ValueError 异常
    msg = "key must be provided when HDF5 file contains multiple datasets."
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)


# 测试函数：验证从空 HDF5 文件读取时的异常情况
def test_read_nokey_empty(tmp_path, setup_path):
    # 创建一个空的 HDF5 文件
    path = tmp_path / setup_path
    store = HDFStore(path)
    store.close()

    # 断言：验证从空 HDF5 文件读取时会抛出 ValueError 异常
    msg = re.escape(
        "Dataset(s) incompatible with Pandas data types, not table, or no "
        "datasets found in HDF5 file."
    )
    with pytest.raises(ValueError, match=msg):
        read_hdf(path)


# 测试函数：验证使用 pathlib.Path 作为路径参数时的读取功能
def test_read_from_pathlib_path(tmp_path, setup_path):
    # 创建一个随机数据的 DataFrame，指定索引和列名
    expected = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    filename = tmp_path / setup_path
    path_obj = Path(filename)

    # 将 DataFrame 写入 HDF5 文件，指定 key 为 "df"
    expected.to_hdf(path_obj, key="df", mode="a")

    # 从 HDF5 文件中读取 DataFrame，指定 key 为 "df"
    actual = read_hdf(path_obj, key="df")
    # 使用测试框架中的方法来比较两个数据框架（DataFrame）
    tm.assert_frame_equal(expected, actual)
# 使用 pytest 的 parametrize 装饰器，为单元测试 test_read_hdf_series_mode_r 提供多个参数化测试
@pytest.mark.parametrize("format", ["fixed", "table"])
def test_read_hdf_series_mode_r(tmp_path, format, setup_path):
    # GH 16583
    # 测试当提供 mode='r' 参数时，读取保存到 HDF 文件中的 Series 是否正常工作
    series = Series(range(10), dtype=np.float64)
    # 创建临时文件路径，使用 setup_path 配置路径
    path = tmp_path / setup_path
    # 将 Series 保存为 HDF 文件
    series.to_hdf(path, key="data", format=format)
    # 从 HDF 文件中读取数据
    result = read_hdf(path, key="data", mode="r")
    # 断言读取的结果与原始 Series 是否相等
    tm.assert_series_equal(result, series)


# 单元测试函数 test_read_infer_string
def test_read_infer_string(tmp_path, setup_path):
    # GH#54431
    # 导入 pyarrow 库，如果导入失败则跳过这个测试
    pytest.importorskip("pyarrow")
    # 创建包含字符串和 None 值的 DataFrame
    df = DataFrame({"a": ["a", "b", None]})
    # 创建临时文件路径，使用 setup_path 配置路径
    path = tmp_path / setup_path
    # 将 DataFrame 保存为格式为 table 的 HDF 文件
    df.to_hdf(path, key="data", format="table")
    # 在 future.infer_string 上下文中，从 HDF 文件中读取数据
    with pd.option_context("future.infer_string", True):
        result = read_hdf(path, key="data", mode="r")
    # 创建期望的 DataFrame，指定列的数据类型为字符串
    expected = DataFrame(
        {"a": ["a", "b", None]},
        dtype="string[pyarrow_numpy]",
        columns=Index(["a"], dtype="string[pyarrow_numpy]"),
    )
    # 断言读取的结果与期望的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# 单元测试函数 test_hdfstore_read_datetime64_unit_s
def test_hdfstore_read_datetime64_unit_s(tmp_path, setup_path):
    # GH 59004
    # 创建包含 datetime64[s] 类型数据的 DataFrame
    df_s = DataFrame(["2001-01-01", "2002-02-02"], dtype="datetime64[s]")
    # 创建临时文件路径，使用 setup_path 配置路径
    path = tmp_path / setup_path
    # 使用 HDFStore 写入 DataFrame 到 HDF 文件
    with HDFStore(path, mode="w") as store:
        store.put("df_s", df_s)
    # 使用 HDFStore 读取 HDF 文件中的数据
    with HDFStore(path, mode="r") as store:
        df_fromstore = store.get("df_s")
    # 断言读取的 DataFrame 与原始的 DataFrame 是否相等
    tm.assert_frame_equal(df_s, df_fromstore)
```