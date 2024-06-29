# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_errors.py`

```
import datetime  # 导入日期时间模块
from io import BytesIO  # 从 io 模块导入 BytesIO 类
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 数学库并重命名为 np
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库导入以下对象：
    CategoricalIndex,  # 分类索引对象
    DataFrame,  # 数据帧对象
    HDFStore,  # HDF 存储对象
    Index,  # 索引对象
    MultiIndex,  # 多级索引对象
    _testing as tm,  # _testing 模块重命名为 tm
    date_range,  # 日期范围生成函数
    read_hdf,  # 从 HDF 文件中读取函数
)

from pandas.tests.io.pytables.common import ensure_clean_store  # 从 pandas 测试模块导入确保清理存储的函数

from pandas.io.pytables import (  # 从 pandas IO 模块导入以下对象：
    Term,  # Term 对象
    _maybe_adjust_name,  # _maybe_adjust_name 函数
)

pytestmark = pytest.mark.single_cpu  # 在 pytest 中标记为单 CPU 模式的测试


def test_pass_spec_to_storer(setup_path):
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),  # 创建一个 30x4 的 DataFrame，值为 1.1 的倍数
        columns=Index(list("ABCD"), dtype=object),  # 指定列索引为 A, B, C, D
        index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 指定行索引为 i-0 到 i-29
    )

    with ensure_clean_store(setup_path) as store:
        store.put("df", df)  # 将 DataFrame 存储到 HDF5 文件中的 "df" 键下
        msg = (
            "cannot pass a column specification when reading a Fixed format "
            "store. this store must be selected in its entirety"
        )
        with pytest.raises(TypeError, match=msg):  # 断言在选择列时抛出 TypeError 异常，并包含指定消息
            store.select("df", columns=["A"])
        msg = (
            "cannot pass a where specification when reading from a Fixed "
            "format store. this store must be selected in its entirety"
        )
        with pytest.raises(TypeError, match=msg):  # 断言在选择行时抛出 TypeError 异常，并包含指定消息
            store.select("df", where=[("columns=A")])


def test_table_index_incompatible_dtypes(setup_path):
    df1 = DataFrame({"a": [1, 2, 3]})  # 创建包含列 "a" 的 DataFrame
    df2 = DataFrame({"a": [4, 5, 6]}, index=date_range("1/1/2000", periods=3))  # 创建带日期索引的 DataFrame

    with ensure_clean_store(setup_path) as store:
        store.put("frame", df1, format="table")  # 将 DataFrame 存储为 HDF5 表格格式
        msg = re.escape("incompatible kind in col [integer - datetime64[ns]]")
        with pytest.raises(TypeError, match=msg):  # 断言在追加 DataFrame 时抛出 TypeError 异常，并包含指定消息
            store.put("frame", df2, format="table", append=True)


def test_unimplemented_dtypes_table_columns(setup_path):
    with ensure_clean_store(setup_path) as store:
        dtypes = [("date", datetime.date(2001, 1, 2))]  # 创建包含日期对象的类型元组列表

        # currently not supported dtypes ####
        for n, f in dtypes:
            df = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),  # 创建 30x4 的 DataFrame，值为 1.1 的倍数
                columns=Index(list("ABCD"), dtype=object),  # 指定列索引为 A, B, C, D
                index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 指定行索引为 i-0 到 i-29
            )
            df[n] = f  # 将日期对象添加为 DataFrame 的列
            msg = re.escape(f"[{n}] is not implemented as a table column")
            with pytest.raises(TypeError, match=msg):  # 断言在追加 DataFrame 时抛出 TypeError 异常，并包含指定消息
                store.append(f"df1_{n}", df)

    # frame
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),  # 创建 30x4 的 DataFrame，值为 1.1 的倍数
        columns=Index(list("ABCD"), dtype=object),  # 指定列索引为 A, B, C, D
        index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 指定行索引为 i-0 到 i-29
    )
    df["obj1"] = "foo"  # 添加名为 "obj1" 的对象列，值为 "foo"
    df["obj2"] = "bar"  # 添加名为 "obj2" 的对象列，值为 "bar"
    df["datetime1"] = datetime.date(2001, 1, 2)  # 添加名为 "datetime1" 的日期列，值为指定日期
    df = df._consolidate()  # 将 DataFrame 合并为块

    with ensure_clean_store(setup_path) as store:
        # this fails because we have a date in the object block......
        msg = re.escape(
            """Cannot serialize the column [datetime1]
            # 无法序列化列 [datetime1]
        """
# 在测试中使用临时路径和设置路径
def test_invalid_terms(tmp_path, setup_path):
    # 在确保存储环境干净的情况下，创建存储对象
    with ensure_clean_store(setup_path) as store:
        # 创建一个 DataFrame 对象 df，包含 10 行 4 列的随机标准正态分布数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),  # 指定列索引为对象类型
            index=date_range("2000-01-01", periods=10, freq="B"),  # 设置日期索引
        )
        # 在 df 中添加名为 'string' 的列，赋值为 'foo'
        df["string"] = "foo"
        # 将 df 索引前四行的 'string' 列值设置为 'bar'
        df.loc[df.index[0:4], "string"] = "bar"

        # 将 df 存储到存储对象中，格式为 'table'
        store.put("df", df, format="table")

        # 断言期望抛出 TypeError 异常，并匹配指定的错误消息
        msg = re.escape("__init__() missing 1 required positional argument: 'where'")
        with pytest.raises(TypeError, match=msg):
            Term()

        # 断言期望抛出 ValueError 异常，并匹配指定的错误消息
        msg = re.escape(
            "cannot process expression [df.index[3]], "
            "[2000-01-06 00:00:00] is not a valid condition"
        )
        with pytest.raises(ValueError, match=msg):
            store.select("df", "df.index[3]")

        # 断言期望抛出 SyntaxError 异常，并匹配指定的错误消息
        msg = "invalid syntax"
        with pytest.raises(SyntaxError, match=msg):
            store.select("df", "index>")

    # 创建临时路径和设置路径的组合
    path = tmp_path / setup_path
    # 创建一个 DataFrame 对象 dfq，包含 10 行 4 列的随机标准正态分布数据
    dfq = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=list("ABCD"),  # 设置列索引为默认类型
        index=date_range("20130101", periods=10),  # 设置日期索引
    )
    # 将 dfq 存储到 HDF 文件中，key 为 'dfq'，格式为 'table'，数据列为 True
    dfq.to_hdf(path, key="dfq", format="table", data_columns=True)

    # 调用 read_hdf 函数，读取指定条件的数据
    read_hdf(path, "dfq", where="index>Timestamp('20130104') & columns=['A', 'B']")
    read_hdf(path, "dfq", where="A>0 or C>0")

    # 创建临时路径和设置路径的组合
    path = tmp_path / setup_path
    # 创建一个 DataFrame 对象 dfq，包含 10 行 4 列的随机标准正态分布数据
    dfq = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=list("ABCD"),  # 设置列索引为默认类型
        index=date_range("20130101", periods=10),  # 设置日期索引
    )
    # 将 dfq 存储到 HDF 文件中，key 为 'dfq'，格式为 'table'

    # 定义消息字符串，描述 where 表达式中的错误引用
    msg = (
        r"The passed where expression: A>0 or C>0\n\s*"
        r"contains an invalid variable reference\n\s*"
        r"all of the variable references must be a reference to\n\s*"
        r"an axis \(e.g. 'index' or 'columns'\), or a data_column\n\s*"
        r"The currently defined references are: index,columns\n"
    )
    # 断言期望抛出 ValueError 异常，并匹配指定的错误消息
    with pytest.raises(ValueError, match=msg):
        read_hdf(path, "dfq", where="A>0 or C>0")


# 测试在不同列名和类型时，追加操作引发 ValueError 异常
def test_append_with_diff_col_name_types_raises_value_error(setup_path):
    # 创建一个 DataFrame 对象 df，包含 10 行 1 列的随机标准正态分布数据
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
    # 创建一个 DataFrame 对象 df2，包含具有 'a' 列名的 10 行随机标准正态分布数据
    df2 = DataFrame({"a": np.random.default_rng(2).standard_normal(10)})
    # 创建一个 DataFrame 对象 df3，包含具有元组列名 (1, 2) 的 10 行随机标准正态分布数据
    df3 = DataFrame({(1, 2): np.random.default_rng(2).standard_normal(10)})
    # 创建一个 DataFrame 对象 df4，包含具有元组列名 ('1', 2) 的 10 行随机标准正态分布数据
    df4 = DataFrame({("1", 2): np.random.default_rng(2).standard_normal(10)})
    # 创建一个 DataFrame 对象 df5，包含具有元组列名 ('1', 2, object) 的 10 行随机标准正态分布数据
    df5 = DataFrame({("1", 2, object): np.random.default_rng(2).standard_normal(10)})
    # 使用 ensure_clean_store 上下文管理器，并指定 setup_path 参数，确保存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 定义变量 name，并赋值为 "df_diff_valerror"
        name = "df_diff_valerror"
        # 将 DataFrame df 附加到存储中，使用变量 name 作为键名
        store.append(name, df)

        # 遍历 DataFrame df2, df3, df4, df5
        for d in (df2, df3, df4, df5):
            # 定义变量 msg，其值为正则表达式转义后的字符串
            msg = re.escape(
                "cannot match existing table structure for [0] on appending data"
            )
            # 使用 pytest 的 raises 断言，检测是否会抛出 ValueError 异常，并匹配消息为 msg
            with pytest.raises(ValueError, match=msg):
                # 将当前 DataFrame d 附加到存储中，使用变量 name 作为键名
                store.append(name, d)
# 测试无效的压缩库设置函数
def test_invalid_complib(setup_path):
    # 创建一个 DataFrame，包含随机生成的数据，行索引为 'abcd'，列索引为 'ABCDE'
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    # 使用 pytest 提供的临时路径，并确保路径在使用后被清理
    with tm.ensure_clean(setup_path) as path:
        # 定义错误消息的正则表达式，验证是否引发 ValueError 异常
        msg = r"complib only supports \[.*\] compression."
        with pytest.raises(ValueError, match=msg):
            # 将 DataFrame 写入 HDF5 文件，使用不存在的压缩库 'foolib'
            df.to_hdf(path, key="df", complib="foolib")


@pytest.mark.parametrize(
    "idx",
    [
        # 创建一个参数化测试，包含两个不同的索引类型
        date_range("2019", freq="D", periods=3, tz="UTC"),
        CategoricalIndex(list("abc")),
    ],
)
def test_to_hdf_multiindex_extension_dtype(idx, tmp_path, setup_path):
    # GH 7775
    # 创建一个 MultiIndex，包含两个相同的索引数组 idx
    mi = MultiIndex.from_arrays([idx, idx])
    # 创建一个 DataFrame，索引为 mi，列为 ['a']
    df = DataFrame(0, index=mi, columns=["a"])
    # 使用临时路径设置测试文件路径
    path = tmp_path / setup_path
    # 验证是否引发 NotImplementedError 异常，用于保存 MultiIndex 到 HDF5 文件
    with pytest.raises(NotImplementedError, match="Saving a MultiIndex"):
        df.to_hdf(path, key="df")


def test_unsuppored_hdf_file_error(datapath):
    # GH 9539
    # 指定测试数据文件路径
    data_path = datapath("io", "data", "legacy_hdf/incompatible_dataset.h5")
    # 定义错误消息的正则表达式，验证是否引发 ValueError 异常
    message = (
        r"Dataset\(s\) incompatible with Pandas data types, "
        "not table, or no datasets found in HDF5 file."
    )
    # 验证是否引发 ValueError 异常，读取不兼容的 HDF5 文件
    with pytest.raises(ValueError, match=message):
        read_hdf(data_path)


def test_read_hdf_errors(setup_path, tmp_path):
    # 创建一个 DataFrame，包含随机生成的数据，行索引为 'abcd'，列索引为 'ABCDE'
    df = DataFrame(
        np.random.default_rng(2).random((4, 5)),
        index=list("abcd"),
        columns=list("ABCDE"),
    )
    # 使用临时路径设置测试文件路径
    path = tmp_path / setup_path
    # 定义错误消息的正则表达式，验证是否引发 OSError 异常
    msg = r"File [\S]* does not exist"
    with pytest.raises(OSError, match=msg):
        # 验证是否引发 OSError 异常，读取不存在的 HDF5 文件
        read_hdf(path, "key")

    # 将 DataFrame 写入 HDF5 文件
    df.to_hdf(path, key="df")
    # 打开 HDF5 存储对象，并立即关闭
    store = HDFStore(path, mode="r")
    store.close()

    # 定义错误消息，验证是否引发 OSError 异常
    msg = "The HDFStore must be open for reading."
    with pytest.raises(OSError, match=msg):
        # 验证是否引发 OSError 异常，使用已关闭的 HDFStore 对象读取 HDF5 文件
        read_hdf(store, "df")


def test_read_hdf_generic_buffer_errors():
    # 定义错误消息，验证是否引发 NotImplementedError 异常
    msg = "Support for generic buffers has not been implemented."
    with pytest.raises(NotImplementedError, match=msg):
        # 验证是否引发 NotImplementedError 异常，读取 HDF5 泛型缓冲区
        read_hdf(BytesIO(b""), "df")


@pytest.mark.parametrize("bad_version", [(1, 2), (1,), [], "12", "123"])
def test_maybe_adjust_name_bad_version_raises(bad_version):
    # 定义错误消息，验证是否引发 ValueError 异常
    msg = "Version is incorrect, expected sequence of 3 integers"
    with pytest.raises(ValueError, match=msg):
        # 验证是否引发 ValueError 异常，传递错误版本号到 _maybe_adjust_name 函数
        _maybe_adjust_name("values_block_0", version=bad_version)
```