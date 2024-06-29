# `D:\src\scipysrc\pandas\pandas\tests\io\test_feather.py`

```
"""test feather-format compat"""

# 导入需要的库和模块
import zoneinfo

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)

from pandas.io.feather_format import read_feather, to_feather  # isort:skip

# 设定 pytest 的标记，忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 导入 pyarrow 库，如果不存在则跳过测试
pa = pytest.importorskip("pyarrow")

# 定义 TestFeather 类，用于测试 Feather 格式的功能
@pytest.mark.single_cpu
class TestFeather:
    def check_error_on_write(self, df, exc, err_msg):
        # 检查写操作时是否引发了预期的异常

        with pytest.raises(exc, match=err_msg):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_external_error_on_write(self, df):
        # 检查写操作时是否引发了外部异常

        with tm.external_error_raised(Exception):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_round_trip(self, df, expected=None, write_kwargs=None, **read_kwargs):
        # 检查数据的完整读写往复过程
        if write_kwargs is None:
            write_kwargs = {}
        if expected is None:
            expected = df.copy()

        with tm.ensure_clean() as path:
            # 将数据写入 Feather 格式文件
            to_feather(df, path, **write_kwargs)

            # 从 Feather 格式文件中读取数据
            result = read_feather(path, **read_kwargs)

            # 断言读取结果与期望结果相等
            tm.assert_frame_equal(result, expected)

    def test_error(self):
        # 测试在非 DataFrame 对象上写入时是否会引发异常
        msg = "feather only support IO with DataFrames"
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            self.check_error_on_write(obj, ValueError, msg)
    def test_basic(self):
        # 创建一个表示美国东部时区的 ZoneInfo 对象
        tz = zoneinfo.ZoneInfo("US/Eastern")
        # 创建一个包含多种数据类型的 DataFrame
        df = pd.DataFrame(
            {
                "string": list("abc"),                           # 字符串列
                "int": list(range(1, 4)),                        # 整数列
                "uint": np.arange(3, 6).astype("u1"),            # 无符号整数列
                "float": np.arange(4.0, 7.0, dtype="float64"),   # 浮点数列
                "float_with_null": [1.0, np.nan, 3],             # 带有空值的浮点数列
                "bool": [True, False, True],                     # 布尔值列
                "bool_with_null": [True, np.nan, False],         # 带有空值的布尔值列
                "cat": pd.Categorical(list("abc")),              # 分类数据列
                "dt": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3)), freq=None
                ),                                              # 日期时间索引列
                "dttz": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, tz=tz)),
                    freq=None,
                ),                                              # 含有时区信息的日期时间索引列
                "dt_with_null": [
                    pd.Timestamp("20130101"),
                    pd.NaT,
                    pd.Timestamp("20130103"),
                ],                                              # 带有空值的日期时间列
                "dtns": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, freq="ns")), freq=None
                ),                                              # 纳秒精度的日期时间索引列
            }
        )
        # 添加一个周期范围列
        df["periods"] = pd.period_range("2013", freq="M", periods=3)
        # 添加一个时间间隔范围列
        df["timedeltas"] = pd.timedelta_range("1 day", periods=3)
        # 添加一个区间范围列
        df["intervals"] = pd.interval_range(0, 3, 3)

        # 断言 DataFrame 中带有时区信息的列的时区键值为 "US/Eastern"
        assert df.dttz.dtype.tz.key == "US/Eastern"

        # 复制 DataFrame 以备预期结果
        expected = df.copy()
        # 将预期结果中第二行的 bool_with_null 列值设置为 None
        expected.loc[1, "bool_with_null"] = None
        # 使用自定义函数检查往返操作的结果是否符合预期
        self.check_round_trip(df, expected=expected)

    def test_duplicate_columns(self):
        # 问题链接：https://github.com/wesm/feather/issues/53
        # 目前无法处理重复列
        # 创建一个包含重复列的 DataFrame
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
        # 使用自定义函数检查写入时是否会出现外部错误
        self.check_external_error_on_write(df)

    def test_read_columns(self):
        # 问题链接：GH 24025
        # 创建一个包含多列的 DataFrame
        df = pd.DataFrame(
            {
                "col1": list("abc"),
                "col2": list(range(1, 4)),
                "col3": list("xyz"),
                "col4": list(range(4, 7)),
            }
        )
        columns = ["col1", "col3"]
        # 使用自定义函数检查读取指定列的结果是否符合预期
        self.check_round_trip(df, expected=df[columns], columns=columns)

    def test_read_columns_different_order(self):
        # 问题链接：GH 33878
        # 创建一个包含多列的 DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [True, False]})
        expected = df[["B", "A"]]
        # 使用自定义函数检查读取指定列并以不同顺序排列的结果是否符合预期
        self.check_round_trip(df, expected, columns=["B", "A"])

    def test_unsupported_other(self):
        # 混合 Python 对象
        # 创建一个包含混合 Python 对象的 DataFrame
        df = pd.DataFrame({"a": ["a", 1, 2.0]})
        # 使用自定义函数检查写入时是否会出现外部错误
        self.check_external_error_on_write(df)

    def test_rw_use_threads(self):
        # 创建一个包含一列从 0 到 99999 的整数的 DataFrame
        df = pd.DataFrame({"A": np.arange(100000)})
        # 使用自定义函数检查使用线程时的往返操作结果是否符合预期
        self.check_round_trip(df, use_threads=True)
        # 使用自定义函数检查不使用线程时的往返操作结果是否符合预期
        self.check_round_trip(df, use_threads=False)
    def test_path_pathlib(self):
        # 创建一个包含随机数据的 DataFrame，用于测试
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),  # 生成一个 30 行 4 列的 DataFrame，数据为等差数列乘以 1.1
            columns=pd.Index(list("ABCD"), dtype=object),  # 指定列索引为 ["A", "B", "C", "D"]，数据类型为 object
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),  # 指定行索引为 ["i-0", "i-1", ..., "i-29"]，数据类型为 object
        ).reset_index()  # 重置索引，将原来的索引列变成普通列，并添加默认的整数索引

        # 调用 tm.round_trip_pathlib 方法，进行 DataFrame 到 feather 文件再到 DataFrame 的往返测试
        result = tm.round_trip_pathlib(df.to_feather, read_feather)
        # 使用测试工具（tm.assert_frame_equal）断言 df 和 result 是否相等
        tm.assert_frame_equal(df, result)

    def test_passthrough_keywords(self):
        # 创建一个包含随机数据的 DataFrame，用于测试
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),  # 生成一个 30 行 4 列的 DataFrame，数据为等差数列乘以 1.1
            columns=pd.Index(list("ABCD"), dtype=object),  # 指定列索引为 ["A", "B", "C", "D"]，数据类型为 object
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),  # 指定行索引为 ["i-0", "i-1", ..., "i-29"]，数据类型为 object
        ).reset_index()  # 重置索引，将原来的索引列变成普通列，并添加默认的整数索引

        # 调用 self.check_round_trip 方法，测试带有特定写入关键字参数的往返过程
        self.check_round_trip(df, write_kwargs={"version": 1})

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_http_path(self, feather_file, httpserver):
        # GH 29055
        # 使用 read_feather 方法读取预期的 feather 文件内容
        expected = read_feather(feather_file)
        # 打开 feather 文件，将其内容作为 HTTP 服务器的响应内容
        with open(feather_file, "rb") as f:
            httpserver.serve_content(content=f.read())
            # 通过 HTTP 请求读取 feather 文件的内容
            res = read_feather(httpserver.url)
        # 使用测试工具（tm.assert_frame_equal）断言 expected 和 res 是否相等
        tm.assert_frame_equal(expected, res)
    # 定义一个测试函数，用于测试读取 feather 文件时的数据类型和后端存储
    def test_read_feather_dtype_backend(self, string_storage, dtype_backend):
        # 创建一个包含多种数据类型的 DataFrame
        df = pd.DataFrame(
            {
                "a": pd.Series([1, np.nan, 3], dtype="Int64"),
                "b": pd.Series([1, 2, 3], dtype="Int64"),
                "c": pd.Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": pd.Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

        # 根据条件设定 string_array 和 string_array_na 变量的值
        if string_storage == "python":
            string_array = StringArray(np.array(["a", "b", "c"], dtype=np.object_))
            string_array_na = StringArray(np.array(["a", "b", pd.NA], dtype=np.object_))
        elif dtype_backend == "pyarrow":
            # 当 dtype_backend 为 'pyarrow' 时，使用 ArrowExtensionArray
            from pandas.arrays import ArrowExtensionArray
            string_array = ArrowExtensionArray(pa.array(["a", "b", "c"]))
            string_array_na = ArrowExtensionArray(pa.array(["a", "b", None]))
        else:
            # 其他情况使用 ArrowStringArray
            string_array = ArrowStringArray(pa.array(["a", "b", "c"]))
            string_array_na = ArrowStringArray(pa.array(["a", "b", None]))

        # 确保测试环境干净，使用临时路径 path
        with tm.ensure_clean() as path:
            # 将 DataFrame 写入 feather 格式文件
            to_feather(df, path)
            # 设定上下文选项，设置 mode.string_storage 的值为 string_storage
            with pd.option_context("mode.string_storage", string_storage):
                # 读取 feather 文件的内容，根据 dtype_backend 参数选择后端
                result = read_feather(path, dtype_backend=dtype_backend)

        # 创建期望的 DataFrame
        expected = pd.DataFrame(
            {
                "a": pd.Series([1, np.nan, 3], dtype="Int64"),
                "b": pd.Series([1, 2, 3], dtype="Int64"),
                "c": pd.Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": pd.Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": pd.Series([True, False, pd.NA], dtype="boolean"),
                "f": pd.Series([True, False, True], dtype="boolean"),
                "g": string_array,
                "h": string_array_na,
            }
        )

        # 当 dtype_backend 为 'pyarrow' 时，转换 expected DataFrame 的列为 ArrowExtensionArray
        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray
            expected = pd.DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试整数列和索引的 round trip
    def test_int_columns_and_index(self):
        # 创建一个包含整数列和自定义索引的 DataFrame
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([3, 4, 5], name="test"))
        # 调用 check_round_trip 函数，检查 DataFrame 的 round trip
        self.check_round_trip(df)

    # 测试无效的 dtype_backend 参数时是否会引发 ValueError 异常
    def test_invalid_dtype_backend(self):
        # 错误消息
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        # 创建一个包含整数列的 DataFrame
        df = pd.DataFrame({"int": list(range(1, 4))})
        # 使用临时路径 "tmp.feather"
        with tm.ensure_clean("tmp.feather") as path:
            # 将 DataFrame 写入 feather 格式文件
            df.to_feather(path)
            # 断言读取 feather 文件时是否引发 ValueError 异常，并匹配错误消息
            with pytest.raises(ValueError, match=msg):
                read_feather(path, dtype_backend="numpy")
    def test_string_inference(self, tmp_path):
        # 定义测试方法 test_string_inference，接受参数 self 和 tmp_path
        # GH#54431，关联GitHub问题编号
        # 设置文件路径为临时路径下的 test_string_inference.p 文件
        path = tmp_path / "test_string_inference.p"
        # 创建一个包含数据的 DataFrame，列名为 'a'，包含字符串 'x' 和 'y'
        df = pd.DataFrame(data={"a": ["x", "y"]})
        # 将 DataFrame 保存为 Feather 格式到指定路径
        df.to_feather(path)
        # 在上下文中设置 Pandas 的选项，启用未来推断字符串类型
        with pd.option_context("future.infer_string", True):
            # 调用 read_feather 函数读取文件内容
            result = read_feather(path)
        # 创建预期的 DataFrame，包含列 'a'，数据类型为 'string[pyarrow_numpy]'
        expected = pd.DataFrame(data={"a": ["x", "y"]}, dtype="string[pyarrow_numpy]")
        # 使用测试框架的 assert 方法比较 result 和 expected 的内容是否相等
        tm.assert_frame_equal(result, expected)
```