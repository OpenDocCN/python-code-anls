# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_pandas.py`

```
# 导入所需的标准库和第三方库
import datetime  # 导入处理日期时间的模块
from datetime import timedelta  # 导入 timedelta 类用于处理时间差
from decimal import Decimal  # 导入 Decimal 类处理精确的十进制浮点数运算
from io import StringIO  # 导入 StringIO 用于在内存中操作字符串数据
import json  # 导入 JSON 序列化和反序列化的支持
import os  # 导入与操作系统交互的模块
import sys  # 导入与 Python 解释器交互的模块
import time  # 导入时间相关的模块

# 导入第三方库和模块
import numpy as np  # 导入数值计算库 NumPy
import pytest  # 导入用于编写和运行测试的 pytest

# 导入 pandas 库及其相关模块
from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 配置相关模块
from pandas.compat import IS64  # 导入 pandas 兼容性相关模块
import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器
import pandas as pd  # 导入 pandas 库并简化命名为 pd
from pandas import (  # 从 pandas 中导入多个常用对象和函数
    NA,
    DataFrame,
    DatetimeIndex,
    Index,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    read_json,
)
import pandas._testing as tm  # 导入 pandas 测试相关模块
from pandas.core.arrays import (  # 导入 pandas 核心数组相关模块
    ArrowStringArray,
    StringArray,
)
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics  # 导入 pandas ArrowStringArray 的 NumPy 语义支持
from pandas.io.json import ujson_dumps  # 导入 pandas JSON 序列化支持函数

# 定义测试函数，测试 JSON 解析错误的情况
def test_literal_json_raises():
    # 测试 JSON 字符串中缺少文件的情况
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    msg = r".* does not exist"  # 错误消息的正则表达式模式

    # 使用 pytest 检测 read_json 函数在不同情况下的文件未找到异常
    with pytest.raises(FileNotFoundError, match=msg):
        read_json(jsonl, lines=False)

    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=True)

    with pytest.raises(FileNotFoundError, match=msg):
        read_json(
            '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n',
            lines=False,
        )

    with pytest.raises(FileNotFoundError, match=msg):
        read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=False)


# 定义辅助函数，用于比较 JSON 解析后的结果是否与预期相等
def assert_json_roundtrip_equal(result, expected, orient):
    if orient in ("records", "values"):
        expected = expected.reset_index(drop=True)
    if orient == "values":
        expected.columns = range(len(expected.columns))
    tm.assert_frame_equal(result, expected)


# 定义测试类 TestPandasContainer，用于测试 pandas 的 JSON 序列化和反序列化功能
class TestPandasContainer:
    # 使用 pytest 的 fixture 装饰器定义 datetime_series
    @pytest.fixture
    def datetime_series(self):
        # 创建一个 Series 对象，包含浮点数数据和日期时间索引，索引频率设置为 None
        ser = Series(
            1.1 * np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        ser.index = ser.index._with_freq(None)  # 设置索引频率为 None
        return ser

    # 使用 pytest 的 fixture 装饰器定义 datetime_frame
    @pytest.fixture
    def datetime_frame(self):
        # 创建一个 DataFrame 对象，包含随机数数据，日期时间索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=30, freq="B"),
        )
        df.index = df.index._with_freq(None)  # 设置索引频率为 None
        return df

    # 定义测试方法，测试 DataFrame 对象的双重编码标签的 JSON 序列化和反序列化
    def test_frame_double_encoded_labels(self, orient):
        # 创建一个 DataFrame 对象，包含字符串数据，索引和列名包含特殊字符
        df = DataFrame(
            [["a", "b"], ["c", "d"]],
            index=['index " 1', "index / 2"],
            columns=["a \\ b", "y / z"],
        )

        # 使用 StringIO 将 DataFrame 对象转换为 JSON 字符串数据
        data = StringIO(df.to_json(orient=orient))
        result = read_json(data, orient=orient)  # 使用 read_json 函数解析 JSON 数据
        expected = df.copy()  # 复制原始的 DataFrame 作为预期结果
        assert_json_roundtrip_equal(result, expected, orient)  # 比较解析后的结果与预期结果是否相等
    # 使用 pytest 的参数化功能，依次运行以下测试用例，每次使用不同的 orient 参数值
    @pytest.mark.parametrize("orient", ["split", "records", "values"])
    # 测试处理具有非唯一索引的 DataFrame 情况
    def test_frame_non_unique_index(self, orient):
        # 创建一个包含非唯一索引的 DataFrame 对象
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        # 将 DataFrame 转换为 JSON 格式并存储到 StringIO 对象中
        data = StringIO(df.to_json(orient=orient))
        # 调用 read_json 函数读取 JSON 数据
        result = read_json(data, orient=orient)
        # 创建预期结果，复制原始 DataFrame
        expected = df.copy()

        # 断言结果与预期结果相等，测试 JSON 数据的往返转换是否正确
        assert_json_roundtrip_equal(result, expected, orient)

    # 使用 pytest 的参数化功能，依次运行以下测试用例，每次使用不同的 orient 参数值
    @pytest.mark.parametrize("orient", ["index", "columns"])
    # 测试处理具有非唯一索引的 DataFrame 情况，并断言是否引发 ValueError 异常
    def test_frame_non_unique_index_raises(self, orient):
        # 创建一个包含非唯一索引的 DataFrame 对象
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 1], columns=["x", "y"])
        # 根据 orient 参数值生成相应的错误消息
        msg = f"DataFrame index must be unique for orient='{orient}'"
        # 使用 pytest 的上下文管理器检查是否引发指定异常和消息
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)

    # 使用 pytest 的参数化功能，依次运行以下测试用例，每次使用不同的 orient 和 data 参数值
    @pytest.mark.parametrize("orient", ["split", "values"])
    @pytest.mark.parametrize(
        "data",
        [
            [["a", "b"], ["c", "d"]],
            [[1.5, 2.5], [3.5, 4.5]],
            [[1, 2.5], [3, 4.5]],
            [[Timestamp("20130101"), 3.5], [Timestamp("20130102"), 4.5]],
        ],
    )
    # 测试处理具有非唯一列名的 DataFrame 情况，并检查在特定情况下是否标记为预期失败
    def test_frame_non_unique_columns(self, orient, data, request):
        # 如果数据包含 Timestamp 类型且 orient 为 "split"，则标记为预期失败
        if isinstance(data[0][0], Timestamp) and orient == "split":
            mark = pytest.mark.xfail(
                reason="GH#55827 non-nanosecond dt64 fails to round-trip"
            )
            request.applymarker(mark)

        # 创建一个包含非唯一列名的 DataFrame 对象
        df = DataFrame(data, index=[1, 2], columns=["x", "x"])

        expected_warning = None
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 如果 DataFrame 的第一列数据类型为 datetime64[s]，则设置预期警告为 FutureWarning
        if df.iloc[:, 0].dtype == "datetime64[s]":
            expected_warning = FutureWarning

        # 使用 pytest 的上下文管理器检查是否产生预期的警告和消息匹配
        with tm.assert_produces_warning(expected_warning, match=msg):
            # 调用 read_json 函数读取 JSON 数据
            result = read_json(
                StringIO(df.to_json(orient=orient)), orient=orient, convert_dates=["x"]
            )
        # 根据 orient 参数值设置预期的 DataFrame 结果
        if orient == "values":
            expected = DataFrame(data)
            # 如果预期 DataFrame 的第一列数据类型为 datetime64[s]，则进行单位转换
            if expected.iloc[:, 0].dtype == "datetime64[s]":
                # 对于 orient == "values"，默认会以毫秒写入 Timestamp 对象，
                # 而这些对象在内部以纳秒存储，因此需要除以 1000000 转换
                expected.isetitem(0, expected.iloc[:, 0].astype(np.int64) // 1000000)
        elif orient == "split":
            expected = df
            expected.columns = ["x", "x.1"]

        # 使用 pandas.testing 模块断言结果 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化功能，依次运行以下测试用例，每次使用不同的 orient 参数值
    @pytest.mark.parametrize("orient", ["index", "columns", "records"])
    # 测试处理具有非唯一列名的 DataFrame 情况，并断言是否引发 ValueError 异常
    def test_frame_non_unique_columns_raises(self, orient):
        # 创建一个包含非唯一列名的 DataFrame 对象
        df = DataFrame([["a", "b"], ["c", "d"]], index=[1, 2], columns=["x", "x"])

        # 根据 orient 参数值生成相应的错误消息
        msg = f"DataFrame columns must be unique for orient='{orient}'"
        # 使用 pytest 的上下文管理器检查是否引发指定异常和消息
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient)
    # 测试默认方向的数据帧转换为 JSON 格式后与以 "columns" 方向转换后的 JSON 相等
    def test_frame_default_orient(self, float_frame):
        assert float_frame.to_json() == float_frame.to_json(orient="columns")

    # 参数化测试：使用指定的 orient、convert_axes、dtype 参数进行简单的 JSON 读写往返测试
    @pytest.mark.parametrize("dtype", [False, float])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_simple(self, orient, convert_axes, dtype, float_frame):
        # 将浮点数数据帧转换为 JSON 字符串并创建 StringIO 对象
        data = StringIO(float_frame.to_json(orient=orient))
        # 使用 read_json 函数读取 JSON 数据
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = float_frame  # 预期的数据帧为原始的浮点数数据帧

        # 断言读写往返后的结果与预期结果相等
        assert_json_roundtrip_equal(result, expected, orient)

    # 参数化测试：使用指定的 orient、convert_axes、dtype 参数进行整数数据帧的 JSON 读写往返测试
    @pytest.mark.parametrize("dtype", [False, np.int64])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_intframe(self, orient, convert_axes, dtype, int_frame):
        # 将整数数据帧转换为 JSON 字符串并创建 StringIO 对象
        data = StringIO(int_frame.to_json(orient=orient))
        # 使用 read_json 函数读取 JSON 数据
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)
        expected = int_frame  # 预期的数据帧为原始的整数数据帧
        # 断言读写往返后的结果与预期结果相等
        assert_json_roundtrip_equal(result, expected, orient)

    # 参数化测试：使用指定的 orient、convert_axes、dtype 参数进行字符串索引和列名的数据帧的 JSON 读写往返测试
    @pytest.mark.parametrize("dtype", [None, np.float64, int, "U3"])
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_str_axes(self, orient, convert_axes, dtype):
        # 创建具有指定维度和 dtype 的数据帧 df
        df = DataFrame(
            np.zeros((200, 4)),
            columns=[str(i) for i in range(4)],
            index=[str(i) for i in range(200)],
            dtype=dtype,
        )

        # 将数据帧 df 转换为 JSON 字符串并创建 StringIO 对象
        data = StringIO(df.to_json(orient=orient))
        # 使用 read_json 函数读取 JSON 数据
        result = read_json(data, orient=orient, convert_axes=convert_axes, dtype=dtype)

        expected = df.copy()  # 预期的数据帧为原始的数据帧 df 的副本
        if not dtype:
            expected = expected.astype(np.int64)

        # 对于 "index" 和 "columns" 方向，由于 JSON 键必须是字符串，因此无法完全保留字符串类型的索引和列名
        if convert_axes and (orient in ("index", "columns")):
            expected.columns = expected.columns.astype(np.int64)
            expected.index = expected.index.astype(np.int64)
        # 对于 "records" 方向且需要转换轴时，列名会被转换为整数类型
        elif orient == "records" and convert_axes:
            expected.columns = expected.columns.astype(np.int64)
        # 对于 "split" 方向且需要转换轴时，列名会被转换为整数类型
        elif convert_axes and orient == "split":
            expected.columns = expected.columns.astype(np.int64)

        # 断言读写往返后的结果与预期结果相等
        assert_json_roundtrip_equal(result, expected, orient)

    # 参数化测试：使用指定的 convert_axes 参数进行分类数据的 JSON 读写往返测试
    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_categorical(
        self, request, orient, convert_axes, using_infer_string
    ):
        # 当 orient 参数为 "index" 或 "columns" 时，应用 pytest 标记来标记预期会失败的测试
        if orient in ("index", "columns"):
            # 将 xfail 标记应用于测试用例，原因是在指定的 orient 下不能有重复的索引值
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"Can't have duplicate index values for orient '{orient}')"
                )
            )

        # 创建一个包含随机数据的字典，使用 numpy 的随机数生成器
        data = {
            c: np.random.default_rng(i).standard_normal(30)
            for i, c in enumerate(list("ABCD"))
        }

        # 创建一个分类数据列
        cat = ["bah"] * 5 + ["bar"] * 5 + ["baz"] * 5 + ["foo"] * 15
        data["E"] = list(reversed(cat))
        data["sort"] = np.arange(30, dtype="int64")
        
        # 创建一个 DataFrame，使用 pd.CategoricalIndex 作为索引
        categorical_frame = DataFrame(data, index=pd.CategoricalIndex(cat, name="E"))
        
        # 将 DataFrame 转换为 JSON 字符串，并用 StringIO 封装
        data = StringIO(categorical_frame.to_json(orient=orient))
        
        # 调用 read_json 函数读取 JSON 数据，并返回结果
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        # 创建预期结果的副本
        expected = categorical_frame.copy()

        # 如果 convert_axes 参数为 False，则进行特定处理
        expected.index = expected.index.astype(
            str if not using_infer_string else "string[pyarrow_numpy]"
        )  # Categorical not preserved
        expected.index.name = None  # 索引名称在 JSON 中不会被保留

        # 断言结果与预期结果相等
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_empty(self, orient, convert_axes):
        # 创建一个空的 DataFrame
        empty_frame = DataFrame()
        
        # 将空的 DataFrame 转换为 JSON 字符串，并用 StringIO 封装
        data = StringIO(empty_frame.to_json(orient=orient))
        
        # 调用 read_json 函数读取 JSON 数据，并返回结果
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        
        # 根据 orient 参数设置预期结果
        if orient == "split":
            idx = Index([], dtype=(float if convert_axes else object))
            expected = DataFrame(index=idx, columns=idx)
        elif orient in ["index", "columns"]:
            expected = DataFrame()
        else:
            expected = empty_frame.copy()

        # 使用 pytest 的测试工具 tm 来断言结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("convert_axes", [True, False])
    def test_roundtrip_timestamp(self, orient, convert_axes, datetime_frame):
        # TODO: improve coverage with date_format parameter
        # 将日期时间的 DataFrame 转换为 JSON 字符串，并用 StringIO 封装
        data = StringIO(datetime_frame.to_json(orient=orient))
        
        # 调用 read_json 函数读取 JSON 数据，并返回结果
        result = read_json(data, orient=orient, convert_axes=convert_axes)
        
        # 创建预期结果的副本
        expected = datetime_frame.copy()

        # 如果 convert_axes 参数为 False，则进行特定处理
        if not convert_axes:  # one off for ts handling
            # 将日期时间索引转换为 epoch 值
            idx = expected.index.view(np.int64) // 1000000
            if orient != "split":  # TODO: handle consistently across orients
                idx = idx.astype(str)

            expected.index = idx

        # 断言结果与预期结果相等
        assert_json_roundtrip_equal(result, expected, orient)

    @pytest.mark.parametrize("convert_axes", [True, False])


这些注释将每行代码进行了详细的解释，说明了其作用和意图。
    # 定义一个测试方法，用于测试混合数据结构的序列化和反序列化
    def test_roundtrip_mixed(self, orient, convert_axes):
        # 创建一个包含五个元素的索引对象
        index = Index(["a", "b", "c", "d", "e"])
        # 创建一个包含四列数据的字典，每列数据类型不同
        values = {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
            "D": [True, False, True, False, True],
        }

        # 使用数据和索引创建一个 DataFrame 对象
        df = DataFrame(data=values, index=index)

        # 将 DataFrame 对象转换为 JSON 字符串并存储在内存中
        data = StringIO(df.to_json(orient=orient))
        # 调用 read_json 函数读取 JSON 数据并返回结果
        result = read_json(data, orient=orient, convert_axes=convert_axes)

        # 创建一个期望的 DataFrame 对象，并将数值列的数据类型转换为 int64
        expected = df.copy()
        expected = expected.assign(**expected.select_dtypes("number").astype(np.int64))

        # 使用自定义的断言函数验证序列化和反序列化后的结果是否相等
        assert_json_roundtrip_equal(result, expected, orient)

    # 标记该测试为预期失败，并提供失败的原因和期望的异常类型
    @pytest.mark.xfail(
        reason="#50456 Column multiindex is stored and loaded differently",
        raises=AssertionError,
    )
    # 参数化测试，对不同的列组合进行测试
    @pytest.mark.parametrize(
        "columns",
        [
            [["2022", "2022"], ["JAN", "FEB"]],
            [["2022", "2023"], ["JAN", "JAN"]],
            [["2022", "2022"], ["JAN", "JAN"]],
        ],
    )
    # 定义一个测试方法，用于测试多级索引的序列化和反序列化
    def test_roundtrip_multiindex(self, columns):
        # 使用指定的列组合创建一个包含数据的 DataFrame 对象
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=pd.MultiIndex.from_arrays(columns),
        )
        # 将 DataFrame 对象转换为 JSON 字符串并存储在内存中
        data = StringIO(df.to_json(orient="split"))
        # 调用 read_json 函数读取 JSON 数据并返回结果
        result = read_json(data, orient="split")
        # 使用 assert_frame_equal 函数验证结果与期望是否一致
        tm.assert_frame_equal(result, df)

    # 参数化测试，对不同的 JSON 数据和期望的错误消息进行测试
    @pytest.mark.parametrize(
        "data,msg,orient",
        [
            ('{"key":b:a:d}', "Expected object or value", "columns"),
            # 数据中索引数量不匹配的情况
            (
                '{"columns":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "|".join(
                    [
                        r"Length of values \(3\) does not match length of index \(2\)",
                    ]
                ),
                "split",
            ),
            # 数据中列数量不匹配的情况
            (
                '{"columns":["A","B","C"],'
                '"index":["1","2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                "3 columns passed, passed data had 2 columns",
                "split",
            ),
            # 数据中存在不正确的键
            (
                '{"badkey":["A","B"],'
                '"index":["2","3"],'
                '"data":[[1.0,"1"],[2.0,"2"],[null,"3"]]}',
                r"unexpected key\(s\): badkey",
                "split",
            ),
        ],
    )
    # 定义一个测试方法，用于测试从 JSON 数据中读取不良数据时是否能正确引发异常
    def test_frame_from_json_bad_data_raises(self, data, msg, orient):
        # 使用 pytest 的断言检查 read_json 是否引发了预期的 ValueError 异常并包含指定的错误消息
        with pytest.raises(ValueError, match=msg):
            read_json(StringIO(data), orient=orient)

    # 参数化测试，对 convert_axes 参数进行测试
    @pytest.mark.parametrize("dtype", [True, False])
    @pytest.mark.parametrize("convert_axes", [True, False])
    # 定义一个测试方法，测试从 JSON 数据中读取 DataFrame 时，处理缺失数据的情况
    def test_frame_from_json_missing_data(self, orient, convert_axes, dtype):
        # 创建一个数值类型的 DataFrame，包含不同长度的子列表，用于测试
        num_df = DataFrame([[1, 2], [4, 5, 6]])

        # 将 DataFrame 转换为 JSON 字符串，然后从 StringIO 中读取并解析为 DataFrame
        result = read_json(
            StringIO(num_df.to_json(orient=orient)),  # 读取 DataFrame 转换的 JSON 数据
            orient=orient,  # 设置读取的方向参数
            convert_axes=convert_axes,  # 是否转换轴标签
            dtype=dtype,  # 指定数据类型
        )
        # 断言结果 DataFrame 中指定位置的值为 NaN
        assert np.isnan(result.iloc[0, 2])

        # 创建一个对象类型的 DataFrame，包含不同长度的子列表，用于测试
        obj_df = DataFrame([["1", "2"], ["4", "5", "6"]])

        # 将 DataFrame 转换为 JSON 字符串，然后从 StringIO 中读取并解析为 DataFrame
        result = read_json(
            StringIO(obj_df.to_json(orient=orient)),  # 读取 DataFrame 转换的 JSON 数据
            orient=orient,  # 设置读取的方向参数
            convert_axes=convert_axes,  # 是否转换轴标签
            dtype=dtype,  # 指定数据类型
        )
        # 断言结果 DataFrame 中指定位置的值为 NaN
        assert np.isnan(result.iloc[0, 2])

    # 使用 pytest 的参数化装饰器，测试从 JSON 数据中读取 DataFrame 时，处理数据类型和缺失值的情况
    @pytest.mark.parametrize("dtype", [True, False])
    def test_frame_read_json_dtype_missing_value(self, dtype):
        # 使用 read_json 解析包含 null 值的 JSON 数据
        result = read_json(StringIO("[null]"), dtype=dtype)
        # 创建预期的 DataFrame，期望的数据类型为对象或者 None（根据 dtype 参数决定）
        expected = DataFrame([np.nan], dtype=object if not dtype else None)

        # 使用 pytest 的断言方法，比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化装饰器，测试包含无穷大值时的处理情况
    @pytest.mark.parametrize("inf", [np.inf, -np.inf])
    @pytest.mark.parametrize("dtype", [True, False])
    def test_frame_infinity(self, inf, dtype):
        # 创建一个包含无穷大值的 DataFrame
        df = DataFrame([[1, 2], [4, 5, 6]])
        df.loc[0, 2] = inf  # 将第一行第三列的值设置为无穷大

        data = StringIO(df.to_json())  # 将 DataFrame 转换为 JSON 字符串
        result = read_json(data, dtype=dtype)  # 使用 read_json 解析 JSON 数据
        # 断言结果 DataFrame 中指定位置的值为 NaN
        assert np.isnan(result.iloc[0, 2])

    # 使用 pytest 的条件跳过装饰器，仅在不符合条件时跳过测试
    @pytest.mark.skipif(not IS64, reason="not compliant on 32-bit, xref #15865")
    @pytest.mark.parametrize(
        "value,precision,expected_val",
        [
            (0.95, 1, 1.0),
            (1.95, 1, 2.0),
            (-1.95, 1, -2.0),
            (0.995, 2, 1.0),
            (0.9995, 3, 1.0),
            (0.99999999999999944, 15, 1.0),
        ],
    )
    # 测试将 DataFrame 转换为 JSON 时，浮点数的精度设置
    def test_frame_to_json_float_precision(self, value, precision, expected_val):
        df = DataFrame([{"a_float": value}])
        encoded = df.to_json(double_precision=precision)  # 转换 DataFrame 为 JSON 字符串
        # 断言编码后的 JSON 字符串与预期的字符串相等
        assert encoded == f'{{"a_float":{{"0":{expected_val}}}}}'

    # 测试在 DataFrame 转换为 JSON 时，指定无效的参数值时是否会引发异常
    def test_frame_to_json_except(self):
        df = DataFrame([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        # 使用 pytest 的断言方法，检查是否会引发 ValueError 异常，并且异常消息与预期相符
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient="garbage")

    # 测试空 DataFrame 转换为 JSON 再解析的情况
    def test_frame_empty(self):
        df = DataFrame(columns=["jim", "joe"])  # 创建一个空的 DataFrame
        assert not df._is_mixed_type  # 断言 DataFrame 不包含混合类型数据

        data = StringIO(df.to_json())  # 将 DataFrame 转换为 JSON 字符串
        result = read_json(data, dtype=dict(df.dtypes))  # 使用 read_json 解析 JSON 数据
        # 使用 pytest 的 DataFrame 相等断言方法，比较解析后的结果与原 DataFrame 是否相等（不检查索引类型）
        tm.assert_frame_equal(result, df, check_index_type=False)

    # 测试空 DataFrame 转换为 JSON 的情况
    def test_frame_empty_to_json(self):
        # 创建一个空的 DataFrame，并将其转换为 JSON 字符串
        df = DataFrame({"test": []}, index=[])
        result = df.to_json(orient="columns")
        expected = '{"test":{}}'  # 预期的 JSON 字符串
        assert result == expected  # 断言转换后的 JSON 字符串与预期相等
    # 测试处理空的混合类型 DataFrame
    def test_frame_empty_mixedtype(self):
        # 创建一个空的 DataFrame，列为["jim", "joe"]
        df = DataFrame(columns=["jim", "joe"])
        # 将"joe"列的数据类型转换为整型（int64）
        df["joe"] = df["joe"].astype("i8")
        # 断言 DataFrame 含有混合类型数据
        assert df._is_mixed_type
        # 将 DataFrame 转换为 JSON 字符串
        data = df.to_json()
        # 读取 JSON 数据，将其转换回 DataFrame，使用字典作为 dtype 参数
        tm.assert_frame_equal(
            read_json(StringIO(data), dtype=dict(df.dtypes)),
            df,
            check_index_type=False,
        )

    # 测试处理混合类型 DataFrame 的不同 orient
    def test_frame_mixedtype_orient(self):  # GH10289
        # 创建包含不同数据类型的 DataFrame
        vals = [
            [10, 1, "foo", 0.1, 0.01],
            [20, 2, "bar", 0.2, 0.02],
            [30, 3, "baz", 0.3, 0.03],
            [40, 4, "qux", 0.4, 0.04],
        ]
        df = DataFrame(
            vals, index=list("abcd"), columns=["1st", "2nd", "3rd", "4th", "5th"]
        )
        # 断言 DataFrame 含有混合类型数据
        assert df._is_mixed_type
        # 复制 DataFrame
        right = df.copy()

        # 遍历不同的 orient 参数
        for orient in ["split", "index", "columns"]:
            # 将 DataFrame 转换为 JSON 字符串，并将其转换回 DataFrame
            inp = StringIO(df.to_json(orient=orient))
            left = read_json(inp, orient=orient, convert_axes=False)
            # 断言转换后的 DataFrame 与原始 DataFrame 相等
            tm.assert_frame_equal(left, right)

        # 将 right DataFrame 的索引改为 RangeIndex
        right.index = RangeIndex(len(df))
        inp = StringIO(df.to_json(orient="records"))
        left = read_json(inp, orient="records", convert_axes=False)
        tm.assert_frame_equal(left, right)

        # 将 right DataFrame 的列索引改为 RangeIndex
        right.columns = RangeIndex(df.shape[1])
        inp = StringIO(df.to_json(orient="values"))
        left = read_json(inp, orient="values", convert_axes=False)
        tm.assert_frame_equal(left, right)

    # 测试与版本 12 兼容性
    def test_v12_compat(self, datapath):
        # 创建一个 DateTimeIndex 对象
        dti = date_range("2000-01-03", "2000-01-07")
        # 将 DateTimeIndex 对象的频率设为 None
        dti = DatetimeIndex(np.asarray(dti), freq=None)
        # 创建包含不同数据类型的 DataFrame
        df = DataFrame(
            [
                [1.56808523, 0.65727391, 1.81021139, -0.17251653],
                [-0.2550111, -0.08072427, -0.03202878, -0.17581665],
                [1.51493992, 0.11805825, 1.629455, -1.31506612],
                [-0.02765498, 0.44679743, 0.33192641, -0.27885413],
                [0.05951614, -2.69652057, 1.28163262, 0.34703478],
            ],
            columns=["A", "B", "C", "D"],
            index=dti,
        )
        # 向 DataFrame 添加一个日期列
        df["date"] = Timestamp("19920106 18:21:32.12").as_unit("ns")
        # 修改 DataFrame 中某个日期值
        df.iloc[3, df.columns.get_loc("date")] = Timestamp("20130101")
        # 将 DataFrame 中的 date 列复制到 modified 列
        df["modified"] = df["date"]
        # 将 modified 列中的某个值设为 NaT
        df.iloc[1, df.columns.get_loc("modified")] = pd.NaT

        # 获取数据路径
        dirpath = datapath("io", "json", "data")
        # 读取版本 12 的 JSON 数据文件，并将其转换为 DataFrame
        v12_json = os.path.join(dirpath, "tsframe_v012.json")
        df_unser = read_json(v12_json)
        # 断言读取的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(df, df_unser)

        # 从 df 中删除 modified 列，并将其写入版本 12 ISO 格式的 JSON 文件
        df_iso = df.drop(["modified"], axis=1)
        v12_iso_json = os.path.join(dirpath, "tsframe_iso_v012.json")
        df_unser_iso = read_json(v12_iso_json)
        # 断言读取的 DataFrame 与 df_iso 相等，忽略列类型检查
        tm.assert_frame_equal(df_iso, df_unser_iso, check_column_type=False)
    def test_frame_nonprintable_bytes(self):
        # GH14256: failing column caused segfaults, if it is not the last one

        class BinaryThing:
            def __init__(self, hexed) -> None:
                self.hexed = hexed
                self.binary = bytes.fromhex(hexed)

            def __str__(self) -> str:
                return self.hexed

        hexed = "574b4454ba8c5eb4f98a8f45"
        binthing = BinaryThing(hexed)

        # verify the proper conversion of printable content
        # 创建一个包含可打印内容的 DataFrame，验证其正确转换为 JSON 格式
        df_printable = DataFrame({"A": [binthing.hexed]})
        assert df_printable.to_json() == f'{{"A":{{"0":"{hexed}"}}}}'

        # check if non-printable content throws appropriate Exception
        # 创建包含不可打印内容的 DataFrame，验证是否抛出正确的异常
        df_nonprintable = DataFrame({"A": [binthing]})
        msg = "Unsupported UTF-8 sequence length when encoding string"
        with pytest.raises(OverflowError, match=msg):
            df_nonprintable.to_json()

        # the same with multiple columns threw segfaults
        # 同样适用于包含多列时是否抛出异常
        df_mixed = DataFrame({"A": [binthing], "B": [1]}, columns=["A", "B"])
        with pytest.raises(OverflowError, match=msg):
            df_mixed.to_json()

        # default_handler should resolve exceptions for non-string types
        # 使用 default_handler 处理非字符串类型的异常情况
        result = df_nonprintable.to_json(default_handler=str)
        expected = f'{{"A":{{"0":"{hexed}"}}}}'
        assert result == expected
        assert (
            df_mixed.to_json(default_handler=str)
            == f'{{"A":{{"0":"{hexed}"}},"B":{{"0":1}}}}'
        )

    def test_label_overflow(self):
        # GH14256: buffer length not checked when writing label
        # 测试在写入标签时未检查缓冲区长度的情况
        result = DataFrame({"bar" * 100000: [1], "foo": [1337]}).to_json()
        expected = f'{{"{"bar" * 100000}":{{"0":1}},"foo":{{"0":1337}}}}'
        assert result == expected

    def test_series_non_unique_index(self):
        s = Series(["a", "b"], index=[1, 1])

        msg = "Series index must be unique for orient='index'"
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient="index")

        tm.assert_series_equal(
            s,
            read_json(
                StringIO(s.to_json(orient="split")), orient="split", typ="series"
            ),
        )
        unserialized = read_json(
            StringIO(s.to_json(orient="records")), orient="records", typ="series"
        )
        tm.assert_equal(s.values, unserialized.values)

    def test_series_default_orient(self, string_series):
        assert string_series.to_json() == string_series.to_json(orient="index")
    # 定义测试方法，测试简单系列数据的往返转换
    def test_series_roundtrip_simple(self, orient, string_series, using_infer_string):
        # 将字符串系列转换为 JSON 格式并封装为 StringIO 对象
        data = StringIO(string_series.to_json(orient=orient))
        # 调用 read_json 函数读取数据，期望返回系列数据
        result = read_json(data, typ="series", orient=orient)

        expected = string_series
        # 如果使用推断字符串并且方向是 "split", "index", "columns" 中的一种
        if using_infer_string and orient in ("split", "index", "columns"):
            # 这些模式不包含数据类型信息，因此推断为字符串类型
            expected.index = expected.index.astype("string[pyarrow_numpy]")
        # 如果方向是 "values", "records"，则重置期望的索引
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        # 如果方向不是 "split"，则将期望的名称设为 None
        if orient != "split":
            expected.name = None

        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的系列数据
        tm.assert_series_equal(result, expected)

    # 使用参数化标记定义测试方法，测试对象系列的往返转换
    @pytest.mark.parametrize("dtype", [False, None])
    def test_series_roundtrip_object(self, orient, dtype, object_series):
        # 将对象系列转换为 JSON 格式并封装为 StringIO 对象
        data = StringIO(object_series.to_json(orient=orient))
        # 调用 read_json 函数读取数据，期望返回系列数据，可以指定数据类型
        result = read_json(data, typ="series", orient=orient, dtype=dtype)

        expected = object_series
        # 如果方向是 "values", "records"，则重置期望的索引
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        # 如果方向不是 "split"，则将期望的名称设为 None
        if orient != "split":
            expected.name = None

        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的系列数据
        tm.assert_series_equal(result, expected)

    # 定义测试方法，测试空系列的往返转换
    def test_series_roundtrip_empty(self, orient):
        # 创建空的浮点数类型系列
        empty_series = Series([], index=[], dtype=np.float64)
        # 将空系列转换为 JSON 格式并封装为 StringIO 对象
        data = StringIO(empty_series.to_json(orient=orient))
        # 调用 read_json 函数读取数据，期望返回重置索引后的空系列数据
        result = read_json(data, typ="series", orient=orient)

        expected = empty_series.reset_index(drop=True)
        # 如果方向是 "split"，则重置期望的索引为浮点数类型
        if orient in ("split"):
            expected.index = expected.index.astype(np.float64)

        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的系列数据
        tm.assert_series_equal(result, expected)

    # 定义测试方法，测试时间序列系列的往返转换
    def test_series_roundtrip_timeseries(self, orient, datetime_series):
        # 将时间序列系列转换为 JSON 格式并封装为 StringIO 对象
        data = StringIO(datetime_series.to_json(orient=orient))
        # 调用 read_json 函数读取数据，期望返回时间序列系列数据
        result = read_json(data, typ="series", orient=orient)

        expected = datetime_series
        # 如果方向是 "values", "records"，则重置期望的索引
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)
        # 如果方向不是 "split"，则将期望的名称设为 None
        if orient != "split":
            expected.name = None

        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的系列数据
        tm.assert_series_equal(result, expected)

    # 使用参数化标记定义测试方法，测试数值型系列的往返转换
    @pytest.mark.parametrize("dtype", [np.float64, int])
    def test_series_roundtrip_numeric(self, orient, dtype):
        # 创建数值型系列
        s = Series(range(6), index=["a", "b", "c", "d", "e", "f"])
        # 将数值型系列转换为 JSON 格式并封装为 StringIO 对象
        data = StringIO(s.to_json(orient=orient))
        # 调用 read_json 函数读取数据，期望返回数值型系列数据
        result = read_json(data, typ="series", orient=orient)

        expected = s.copy()
        # 如果方向是 "values", "records"，则重置期望的索引
        if orient in ("values", "records"):
            expected = expected.reset_index(drop=True)

        # 使用 pytest 的 assert_series_equal 函数比较结果和期望的系列数据
        tm.assert_series_equal(result, expected)

    # 定义测试方法，测试系列转换为 JSON 时出现异常的情况
    def test_series_to_json_except(self):
        # 创建系列对象
        s = Series([1, 2, 3])
        msg = "Invalid value 'garbage' for option 'orient'"
        # 使用 pytest 的 raises 函数检查是否抛出 ValueError 异常，并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            s.to_json(orient="garbage")
    # 测试从精确浮点数 JSON 中创建 Series 对象
    def test_series_from_json_precise_float(self):
        # 创建包含浮点数的 Series 对象
        s = Series([4.56, 4.56, 4.56])
        # 将 Series 对象转换为 JSON 字符串，然后读取并转换为 Series 对象
        result = read_json(StringIO(s.to_json()), typ="series", precise_float=True)
        # 断言读取的结果与原始 Series 对象相等，不检查索引类型
        tm.assert_series_equal(result, s, check_index_type=False)

    # 测试带数据类型的 Series 对象
    def test_series_with_dtype(self):
        # 创建包含浮点数的 Series 对象
        s = Series([4.56, 4.56, 4.56])
        # 将 Series 对象转换为 JSON 字符串，然后读取并转换为 Series 对象，指定数据类型为 np.int64
        result = read_json(StringIO(s.to_json()), typ="series", dtype=np.int64)
        # 创建预期的 Series 对象，元素为整数 4
        expected = Series([4] * 3)
        # 断言读取的结果与预期的 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 使用 pytest 参数化装饰器测试带数据类型的 Series 对象（日期时间类型）
    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (True, Series(["2000-01-01"], dtype="datetime64[ns]")),  # 期望结果为日期时间 Series 对象
            (False, Series([946684800000])),  # 期望结果为整数 Series 对象
        ],
    )
    def test_series_with_dtype_datetime(self, dtype, expected):
        # 创建日期时间类型的 Series 对象
        s = Series(["2000-01-01"], dtype="datetime64[ns]")
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 断言在运行中产生 FutureWarning 警告，并匹配特定的警告信息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            data = StringIO(s.to_json())
        # 读取 JSON 数据并转换为 Series 对象，指定数据类型由参数传入
        result = read_json(data, typ="series", dtype=dtype)
        # 断言读取的结果与预期的 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 测试从精确浮点数 JSON 中创建 DataFrame 对象
    def test_frame_from_json_precise_float(self):
        # 创建包含浮点数的 DataFrame 对象
        df = DataFrame([[4.56, 4.56, 4.56], [4.56, 4.56, 4.56]])
        # 将 DataFrame 对象转换为 JSON 字符串，然后读取并转换为 DataFrame 对象
        result = read_json(StringIO(df.to_json()), precise_float=True)
        # 断言读取的结果与原始 DataFrame 对象相等
        tm.assert_frame_equal(result, df)

    # 测试不指定 typ 参数的情况下从 JSON 中创建 Series 对象
    def test_typ(self):
        # 创建整数类型的 Series 对象，指定索引
        s = Series(range(6), index=["a", "b", "c", "d", "e", "f"], dtype="int64")
        # 将 Series 对象转换为 JSON 字符串，然后读取并转换为 Series 对象，不指定 typ 参数
        result = read_json(StringIO(s.to_json()), typ=None)
        # 断言读取的结果与原始 Series 对象相等
        tm.assert_series_equal(result, s)

    # 测试从 JSON 中重新构建 DataFrame 对象并保持索引的一致性
    def test_reconstruction_index(self):
        # 创建包含整数的 DataFrame 对象
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        # 将 DataFrame 对象转换为 JSON 字符串，然后读取并转换为 DataFrame 对象
        result = read_json(StringIO(df.to_json()))
        # 断言读取的结果与原始 DataFrame 对象相等
        tm.assert_frame_equal(result, df)

        # 创建包含字典数据的 DataFrame 对象，指定索引
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["A", "B", "C"])
        # 将 DataFrame 对象转换为 JSON 字符串，然后读取并转换为 DataFrame 对象
        result = read_json(StringIO(df.to_json()))
        # 断言读取的结果与原始 DataFrame 对象相等
        tm.assert_frame_equal(result, df)

    # 测试从文件路径中读取 JSON 数据并处理
    def test_path(self, float_frame, int_frame, datetime_frame):
        # 确保测试时清理生成的文件 "test.json"
        with tm.ensure_clean("test.json") as path:
            # 对包含浮点数、整数、日期时间的 DataFrame 对象分别进行 JSON 格式化并保存到文件中，然后读取
            for df in [float_frame, int_frame, datetime_frame]:
                df.to_json(path)
                read_json(path)

    # 测试处理带有日期时间索引的 Series 和 DataFrame 对象
    def test_axis_dates(self, datetime_series, datetime_frame):
        # frame
        # 将日期时间类型的 DataFrame 对象转换为 JSON 字符串，然后读取并转换为 DataFrame 对象
        json = StringIO(datetime_frame.to_json())
        result = read_json(json)
        # 断言读取的结果与原始 DataFrame 对象相等
        tm.assert_frame_equal(result, datetime_frame)

        # series
        # 将日期时间类型的 Series 对象转换为 JSON 字符串，然后读取并转换为 Series 对象，指定 typ 参数为 "series"
        json = StringIO(datetime_series.to_json())
        result = read_json(json, typ="series")
        # 断言读取的结果与原始 Series 对象相等，不检查名称
        tm.assert_series_equal(result, datetime_series, check_names=False)
        # 断言结果的名称为 None
        assert result.name is None
    # 测试函数，用于测试日期转换功能，接受两个参数：datetime_series 和 datetime_frame
    def test_convert_dates(self, datetime_series, datetime_frame):
        # 将 datetime_frame 赋值给 df 变量
        df = datetime_frame
        # 给 df 添加一个名为 "date" 的列，其值为 Timestamp("20130101") 的纳秒单位形式
        df["date"] = Timestamp("20130101").as_unit("ns")

        # 警告信息内容
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 使用 tm.assert_produces_warning 上下文管理器确保产生 FutureWarning 警告，并匹配 msg 字符串
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 将 df 转换为 JSON 格式，并封装为 StringIO 对象
            json = StringIO(df.to_json())
        # 调用 read_json 函数读取 json 数据，将结果与 df 进行帧数据比较
        result = read_json(json)
        tm.assert_frame_equal(result, df)

        # 给 df 添加一个名为 "foo" 的列，其值为浮点数 1.0
        df["foo"] = 1.0
        # 再次使用 tm.assert_produces_warning 上下文管理器确保产生 FutureWarning 警告，并匹配 msg 字符串
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 将 df 转换为 JSON 格式，指定日期单位为纳秒，并封装为 StringIO 对象
            json = StringIO(df.to_json(date_unit="ns"))

        # 调用 read_json 函数读取 json 数据，设置 convert_dates=False，将结果与期望值 expected 进行帧数据比较
        result = read_json(json, convert_dates=False)
        # 创建一个 expected 变量，其值为 df 的副本，修改 "date" 列的值为整型视图 "i8" 类型，"foo" 列的值转换为 int64 类型
        expected = df.copy()
        expected["date"] = expected["date"].values.view("i8")
        expected["foo"] = expected["foo"].astype("int64")
        tm.assert_frame_equal(result, expected)

        # series
        # 创建一个 Series 对象 ts，其值为 Timestamp("20130101") 的纳秒单位形式，索引与 datetime_series 相同
        ts = Series(Timestamp("20130101").as_unit("ns"), index=datetime_series.index)
        # 再次使用 tm.assert_produces_warning 上下文管理器确保产生 FutureWarning 警告，并匹配 msg 字符串
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 将 ts 转换为 JSON 格式，并封装为 StringIO 对象
            json = StringIO(ts.to_json())
        # 调用 read_json 函数读取 json 数据，设置 typ="series"，将结果与 ts 进行序列数据比较
        result = read_json(json, typ="series")
        tm.assert_series_equal(result, ts)

    # 参数化测试函数，测试不同日期格式、对象类型和日期类型的索引和值
    @pytest.mark.parametrize("date_format", ["epoch", "iso"])
    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("date_typ", [datetime.date, datetime.datetime, Timestamp])
    def test_date_index_and_values(self, date_format, as_object, date_typ):
        # 创建一个包含日期和 NaT 值的列表 data
        data = [date_typ(year=2020, month=1, day=1), pd.NaT]
        # 如果 as_object 为 True，则向 data 列表中添加字符串 "a"
        if as_object:
            data.append("a")

        # 创建一个 Series 对象 ser，其值为 data，索引也为 data
        ser = Series(data, index=data)
        # 如果 as_object 为 False，则将 ser 转换为 "M8[ns]" 类型，并将其索引转换为纳秒单位
        if not as_object:
            ser = ser.astype("M8[ns]")
            if isinstance(ser.index, DatetimeIndex):
                ser.index = ser.index.as_unit("ns")

        # 预期的 JSON 字符串和警告类型
        expected_warning = None
        if date_format == "epoch":
            expected = '{"1577836800000":1577836800000,"null":null}'
            expected_warning = FutureWarning
        else:
            expected = (
                '{"2020-01-01T00:00:00.000":"2020-01-01T00:00:00.000","null":null}'
            )

        # 警告信息内容
        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        # 使用 tm.assert_produces_warning 上下文管理器确保产生 expected_warning 类型的警告，并匹配 msg 字符串
        with tm.assert_produces_warning(expected_warning, match=msg):
            # 调用 ser 的 to_json 方法，指定日期格式为 date_format，将结果保存到 result
            result = ser.to_json(date_format=date_format)

        # 如果 as_object 为 True，则在 expected 字符串末尾添加 ',"a":"a"}' 字符串
        if as_object:
            expected = expected.replace("}", ',"a":"a"}')

        # 断言 result 等于 expected
        assert result == expected

    # 参数化测试函数，测试推断词汇的处理
    @pytest.mark.parametrize(
        "infer_word",
        [
            "trade_time",
            "date",
            "datetime",
            "sold_at",
            "modified",
            "timestamp",
            "timestamps",
        ],
    )
    # 定义一个测试方法，用于测试日期推断功能
    def test_convert_dates_infer(self, infer_word):
        # GH10747

        # 准备测试数据
        data = [{"id": 1, infer_word: 1036713600000}, {"id": 2}]
        # 期望的数据框架
        expected = DataFrame(
            [[1, Timestamp("2002-11-08")], [2, pd.NaT]], columns=["id", infer_word]
        )
        # 将期望数据中的日期列转换为指定的时间单位（纳秒）
        expected[infer_word] = expected[infer_word].astype("M8[ns]")

        # 调用读取 JSON 数据的函数，提取指定的列
        result = read_json(StringIO(ujson_dumps(data)))[["id", infer_word]]
        # 使用测试框架验证结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化装饰器标记的测试方法，用于测试日期格式化框架
    @pytest.mark.parametrize(
        "date,date_unit",
        [
            ("20130101 20:43:42.123", None),
            ("20130101 20:43:42", "s"),
            ("20130101 20:43:42.123", "ms"),
            ("20130101 20:43:42.123456", "us"),
            ("20130101 20:43:42.123456789", "ns"),
        ],
    )
    def test_date_format_frame(self, date, date_unit, datetime_frame):
        # 获取日期时间框架
        df = datetime_frame

        # 将指定日期字符串转换为时间戳，并设置单位为纳秒
        df["date"] = Timestamp(date).as_unit("ns")
        # 将指定行的日期设置为 NaT（Not a Time）
        df.iloc[1, df.columns.get_loc("date")] = pd.NaT
        df.iloc[5, df.columns.get_loc("date")] = pd.NaT
        # 根据指定的日期单位生成 JSON 数据
        if date_unit:
            json = df.to_json(date_format="iso", date_unit=date_unit)
        else:
            json = df.to_json(date_format="iso")

        # 调用读取 JSON 数据的函数，返回结果与期望进行比较
        result = read_json(StringIO(json))
        expected = df.copy()
        tm.assert_frame_equal(result, expected)

    # 测试方法，用于验证日期格式化框架是否会引发异常
    def test_date_format_frame_raises(self, datetime_frame):
        # 获取日期时间框架
        df = datetime_frame
        # 准备错误消息
        msg = "Invalid value 'foo' for option 'date_unit'"
        # 使用 pytest 检查是否会引发预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.to_json(date_format="iso", date_unit="foo")

    # 使用参数化装饰器标记的测试方法，用于测试日期格式化系列
    @pytest.mark.parametrize(
        "date,date_unit",
        [
            ("20130101 20:43:42.123", None),
            ("20130101 20:43:42", "s"),
            ("20130101 20:43:42.123", "ms"),
            ("20130101 20:43:42.123456", "us"),
            ("20130101 20:43:42.123456789", "ns"),
        ],
    )
    def test_date_format_series(self, date, date_unit, datetime_series):
        # 获取日期时间系列
        ts = Series(Timestamp(date).as_unit("ns"), index=datetime_series.index)
        # 将指定行的日期设置为 NaT（Not a Time）
        ts.iloc[1] = pd.NaT
        ts.iloc[5] = pd.NaT
        # 根据指定的日期单位生成 JSON 数据
        if date_unit:
            json = ts.to_json(date_format="iso", date_unit=date_unit)
        else:
            json = ts.to_json(date_format="iso")

        # 调用读取 JSON 数据的函数，返回结果与期望进行比较
        result = read_json(StringIO(json), typ="series")
        expected = ts.copy()
        tm.assert_series_equal(result, expected)

    # 测试方法，用于验证日期格式化系列是否会引发异常
    def test_date_format_series_raises(self, datetime_series):
        # 获取日期时间系列
        ts = Series(Timestamp("20130101 20:43:42.123"), index=datetime_series.index)
        # 准备错误消息
        msg = "Invalid value 'foo' for option 'date_unit'"
        # 使用 pytest 检查是否会引发预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            ts.to_json(date_format="iso", date_unit="foo")
    def test_date_unit(self, unit, datetime_frame):
        # 将 datetime_frame 赋值给 df 变量
        df = datetime_frame
        # 将特定时间戳转换为纳秒单位并赋给"date"列
        df["date"] = Timestamp("20130101 20:43:42").as_unit("ns")
        # 获取"date"列在 DataFrame 中的位置索引
        dl = df.columns.get_loc("date")
        # 修改第一行的"date"列为指定时间戳
        df.iloc[1, dl] = Timestamp("19710101 20:43:42")
        # 修改第二行的"date"列为指定时间戳
        df.iloc[2, dl] = Timestamp("21460101 20:43:42")
        # 将第四行的"date"列设置为 NaT (Not a Time)
        df.iloc[4, dl] = pd.NaT

        # 设置警告消息内容
        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        # 断言产生 FutureWarning 警告并匹配消息内容
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 将 DataFrame 转换为 JSON 格式，指定日期格式为 'epoch' 并单位为 unit
            json = df.to_json(date_format="epoch", date_unit=unit)

        # 强制使用指定的日期单位调用 read_json 函数
        result = read_json(StringIO(json), date_unit=unit)
        # 断言 result 与 df 相等
        tm.assert_frame_equal(result, df)

        # 检测日期单位并调用 read_json 函数
        result = read_json(StringIO(json), date_unit=None)
        # 断言 result 与 df 相等
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize(
        "df, warn",
        [
            # 参数化测试数据集合，DataFrame 包含字符串和整数列
            (DataFrame({"A": ["a", "b", "c"], "B": np.arange(3)}), None),
            # 参数化测试数据集合，DataFrame 包含布尔列
            (DataFrame({"A": [True, False, False]}), None),
            # 参数化测试数据集合，DataFrame 包含字符串列和时间增量单位为天的整数列
            (
                DataFrame(
                    {"A": ["a", "b", "c"], "B": pd.to_timedelta(np.arange(3), unit="D")}
                ),
                FutureWarning,
            ),
            # 参数化测试数据集合，DataFrame 包含日期时间格式列
            (
                DataFrame(
                    {"A": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])}
                ),
                FutureWarning,
            ),
        ],
    )
    def test_default_epoch_date_format_deprecated(self, df, warn):
        # GH 57063
        # 设置警告消息内容
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 断言产生特定警告并匹配消息内容
        with tm.assert_produces_warning(warn, match=msg):
            # 将 DataFrame 转换为 JSON 格式，默认使用 'epoch' 日期格式
            df.to_json()

    @pytest.mark.parametrize("unit", ["s", "ms", "us"])
    # 测试处理非纳秒分辨率的 numpy 日期时间序列在 Index 或列中的序列化能力
    def test_iso_non_nano_datetimes(self, unit):
        # 创建一个 DateTimeIndex 对象，包含一个特定日期时间的 numpy 数组，单位为给定的 unit
        index = DatetimeIndex(
            [np.datetime64("2023-01-01T11:22:33.123456", unit)],
            dtype=f"datetime64[{unit}]",
        )
        # 创建一个 DataFrame 对象
        df = DataFrame(
            {
                "date": Series(
                    [np.datetime64("2022-01-01T11:22:33.123456", unit)],
                    dtype=f"datetime64[{unit}]",
                    index=index,
                ),
                "date_obj": Series(
                    [np.datetime64("2023-01-01T11:22:33.123456", unit)],
                    dtype=object,
                    index=index,
                ),
            },
        )

        # 创建一个 StringIO 对象 buf，并将 DataFrame 对象 df 以 ISO 格式写入 buf
        buf = StringIO()
        df.to_json(buf, date_format="iso", date_unit=unit)
        buf.seek(0)

        # 使用 read_json 读取 buf 中的 JSON 数据，并比较读取结果与 df 是否相等
        # check_dtype 和 check_index_type 参数设为 False，因为 read_json 尚不支持非纳秒分辨率
        tm.assert_frame_equal(
            read_json(buf, convert_dates=["date", "date_obj"]),
            df,
            check_index_type=False,
            check_dtype=False,
        )

    # 测试处理奇怪的嵌套 JSON 数据
    def test_weird_nested_json(self):
        # 定义一个包含奇怪嵌套结构的 JSON 字符串 s
        s = r"""{
        "status": "success",
        "data": {
        "posts": [
            {
            "id": 1,
            "title": "A blog post",
            "body": "Some useful content"
            },
            {
            "id": 2,
            "title": "Another blog post",
            "body": "More content"
            }
           ]
          }
        }"""
        # 使用 read_json 读取 JSON 字符串 s，并执行相关操作
        read_json(StringIO(s))

    # 测试文档示例
    def test_doc_example(self):
        # 创建一个 DataFrame 对象 dfj2，包含随机生成的数据和特定的列和索引
        dfj2 = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=list("AB")
        )
        dfj2["date"] = Timestamp("20130101")
        dfj2["ints"] = range(5)
        dfj2["bools"] = True
        dfj2.index = date_range("20130101", periods=5)

        # 创建一个警告消息字符串 msg
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 使用 tm.assert_produces_warning 检查 dfj2.to_json() 生成的 JSON 是否产生 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match=msg):
            json = StringIO(dfj2.to_json())
        # 使用 read_json 读取 json 对象，并比较结果是否等于自身
        result = read_json(json, dtype={"ints": np.int64, "bools": np.bool_})
        tm.assert_frame_equal(result, result)

    # 测试 JSON 数据的往返操作是否正确
    def test_round_trip_exception(self, datapath):
        # 从 datapath 中读取一个 CSV 文件并创建 DataFrame 对象 df
        path = datapath("io", "json", "data", "teams.csv")
        df = pd.read_csv(path)
        # 将 DataFrame 对象 df 转换为 JSON 字符串 s
        s = df.to_json()

        # 使用 read_json 读取 JSON 字符串 s，并与原始 DataFrame 对象 df 进行比较
        result = read_json(StringIO(s))
        # 重新索引结果 DataFrame 对象 res，并与原始 DataFrame 对象 df 进行比较
        res = result.reindex(index=df.index, columns=df.columns)
        res = res.fillna(np.nan)
        tm.assert_frame_equal(res, df)

    @pytest.mark.network
    @pytest.mark.single_cpu
    @pytest.mark.parametrize(
        "field,dtype",
        [
            ["created_at", pd.DatetimeTZDtype(tz="UTC")],  # 参数化测试用例，定义字段名和期望的数据类型
            ["closed_at", "datetime64[ns]"],  # 参数化测试用例，定义字段名和期望的数据类型
            ["updated_at", pd.DatetimeTZDtype(tz="UTC")],  # 参数化测试用例，定义字段名和期望的数据类型
        ],
    )
    def test_url(self, field, dtype, httpserver):
        data = '{"created_at": ["2023-06-23T18:21:36Z"], "closed_at": ["2023-06-23T18:21:36"], "updated_at": ["2023-06-23T18:21:36Z"]}\n'  # 定义测试数据，模拟 JSON 格式的数据
        httpserver.serve_content(content=data)  # 启动 HTTP 服务器，提供测试数据
        result = read_json(httpserver.url, convert_dates=True)  # 调用函数 read_json 读取 JSON 数据，并进行日期转换
        assert result[field].dtype == dtype  # 断言检查读取结果中指定字段的数据类型是否符合预期

    def test_timedelta(self):
        converter = lambda x: pd.to_timedelta(x, unit="ms")  # 定义转换器函数，将输入转换为时间增量格式

        ser = Series([timedelta(23), timedelta(seconds=5)])  # 创建时间增量的序列
        assert ser.dtype == "timedelta64[ns]"  # 断言检查序列的数据类型是否为时间增量格式

        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):  # 检查警告信息是否符合预期
            result = read_json(StringIO(ser.to_json()), typ="series").apply(converter)  # 读取并转换 JSON 格式的序列数据
        tm.assert_series_equal(result, ser)  # 断言检查转换后的序列数据是否与原始序列相等

        ser = Series([timedelta(23), timedelta(seconds=5)], index=Index([0, 1]))  # 创建带索引的时间增量序列
        assert ser.dtype == "timedelta64[ns]"  # 断言检查序列的数据类型是否为时间增量格式
        with tm.assert_produces_warning(FutureWarning, match=msg):  # 检查警告信息是否符合预期
            result = read_json(StringIO(ser.to_json()), typ="series").apply(converter)  # 读取并转换 JSON 格式的序列数据
        tm.assert_series_equal(result, ser)  # 断言检查转换后的序列数据是否与原始序列相等

        frame = DataFrame([timedelta(23), timedelta(seconds=5)])  # 创建时间增量的数据框架
        assert frame[0].dtype == "timedelta64[ns]"  # 断言检查数据框架的第一列数据类型是否为时间增量格式

        with tm.assert_produces_warning(FutureWarning, match=msg):  # 检查警告信息是否符合预期
            json = frame.to_json()  # 将数据框架转换为 JSON 格式
        tm.assert_frame_equal(frame, read_json(StringIO(json)).apply(converter))  # 断言检查转换后的数据框架是否与原始数据框架相等

    def test_timedelta2(self):
        frame = DataFrame(
            {
                "a": [timedelta(days=23), timedelta(seconds=5)],  # 创建包含时间增量的数据框架的列
                "b": [1, 2],  # 第二列，整数数据
                "c": date_range(start="20130101", periods=2),  # 第三列，日期范围数据
            }
        )
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):  # 检查警告信息是否符合预期
            data = StringIO(frame.to_json(date_unit="ns"))  # 将数据框架转换为 JSON 格式
        result = read_json(data)  # 读取 JSON 格式的数据
        result["a"] = pd.to_timedelta(result.a, unit="ns")  # 将结果中的列 'a' 转换为时间增量格式
        result["c"] = pd.to_datetime(result.c)  # 将结果中的列 'c' 转换为日期时间格式
        tm.assert_frame_equal(frame, result)  # 断言检查转换后的数据框架是否与原始数据框架相等
    # 定义一个测试方法，用于测试混合类型的时间差和日期时间转换
    def test_mixed_timedelta_datetime(self):
        # 创建一个时间差对象，表示23天
        td = timedelta(23)
        # 创建一个时间戳对象，表示"20130101"
        ts = Timestamp("20130101")
        # 创建一个数据帧，包含两列，其中一列是时间差对象，另一列是时间戳对象
        frame = DataFrame({"a": [td, ts]}, dtype=object)

        # 创建预期结果的数据帧，将时间差和时间戳转换为纳秒单位的整数值
        expected = DataFrame(
            {"a": [pd.Timedelta(td).as_unit("ns")._value, ts.as_unit("ns")._value]}
        )
        # 将数据帧转换为 JSON 格式的字符串，指定日期单位为纳秒
        data = StringIO(frame.to_json(date_unit="ns"))
        # 调用 read_json 函数解析 JSON 数据，指定数据类型为整数
        result = read_json(data, dtype={"a": "int64"})
        # 使用断言比较结果数据帧与预期数据帧是否相等，忽略索引类型检查
        tm.assert_frame_equal(result, expected, check_index_type=False)

    # 使用参数化装饰器定义测试方法，测试时间差对象转换为 JSON 格式字符串
    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("date_format", ["iso", "epoch"])
    @pytest.mark.parametrize("timedelta_typ", [pd.Timedelta, timedelta])
    def test_timedelta_to_json(self, as_object, date_format, timedelta_typ):
        # GH28156: to_json not correctly formatting Timedelta
        # 创建包含时间差对象、NaT（Not a Time）值的数据列表
        data = [timedelta_typ(days=1), timedelta_typ(days=2), pd.NaT]
        # 如果 as_object 为 True，则在数据列表末尾添加一个字符串 "a"
        if as_object:
            data.append("a")

        # 创建系列对象，索引与数据相同
        ser = Series(data, index=data)
        expected_warning = None
        # 根据 date_format 参数设置预期的 JSON 字符串和相应的警告类型
        if date_format == "iso":
            expected = (
                '{"P1DT0H0M0S":"P1DT0H0M0S","P2DT0H0M0S":"P2DT0H0M0S","null":null}'
            )
        else:
            expected_warning = FutureWarning
            expected = '{"86400000":86400000,"172800000":172800000,"null":null}'

        # 如果 as_object 为 True，则在预期结果字符串末尾添加 ',"a":"a"}'
        if as_object:
            expected = expected.replace("}", ',"a":"a"}')

        # 设置警告消息
        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        # 使用断言检查是否产生了预期的警告消息
        with tm.assert_produces_warning(expected_warning, match=msg):
            # 调用 Series 对象的 to_json 方法，将数据转换为 JSON 格式字符串
            result = ser.to_json(date_format=date_format)
        # 使用断言比较实际输出的 JSON 字符串与预期的 JSON 字符串是否相等
        assert result == expected

    # 使用参数化装饰器定义测试方法，测试时间差对象转换为 JSON 格式字符串的小数精度
    @pytest.mark.parametrize("as_object", [True, False])
    @pytest.mark.parametrize("timedelta_typ", [pd.Timedelta, timedelta])
    def test_timedelta_to_json_fractional_precision(self, as_object, timedelta_typ):
        # 创建包含毫秒时间差对象的数据列表
        data = [timedelta_typ(milliseconds=42)]
        # 创建系列对象，索引与数据相同
        ser = Series(data, index=data)
        warn = FutureWarning
        # 如果 as_object 为 True，则将系列对象转换为对象类型，同时取消警告
        if as_object:
            ser = ser.astype(object)
            warn = None

        # 设置警告消息
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 使用断言检查是否产生了预期的警告消息
        with tm.assert_produces_warning(warn, match=msg):
            # 调用 Series 对象的 to_json 方法，将数据转换为 JSON 格式字符串
            result = ser.to_json()
        # 设置预期的 JSON 格式字符串
        expected = '{"42":42}'
        # 使用断言比较实际输出的 JSON 字符串与预期的 JSON 字符串是否相等
        assert result == expected

    # 定义一个测试方法，测试读取 JSON 数据并应用默认处理程序的情况
    def test_default_handler(self):
        # 创建一个 Python 对象
        value = object()
        # 创建一个数据帧，包含一个整数和一个 Python 对象
        frame = DataFrame({"a": [7, value]})
        # 创建预期结果的数据帧，将 Python 对象转换为字符串
        expected = DataFrame({"a": [7, str(value)]})
        # 将数据帧转换为 JSON 格式的字符串，使用默认处理程序将 Python 对象转换为字符串
        result = read_json(StringIO(frame.to_json(default_handler=str)))
        # 使用断言比较结果数据帧与预期数据帧是否相等，忽略索引类型检查
        tm.assert_frame_equal(expected, result, check_index_type=False)
    def test_default_handler_indirect(self):
        # 定义一个内部函数 default，用于处理复杂对象的转换，返回特定格式的字符串表示
        def default(obj):
            # 如果对象是复数类型，则返回一个包含数学库标识和实部、虚部的列表
            if isinstance(obj, complex):
                return [("mathjs", "Complex"), ("re", obj.real), ("im", obj.imag)]
            # 否则返回对象的字符串表示
            return str(obj)

        # 创建一个包含整数和 DataFrame 对象的列表 df_list
        df_list = [
            9,
            DataFrame(
                {"a": [1, "STR", complex(4, -5)], "b": [float("nan"), None, "N/A"]},
                columns=["a", "b"],
            ),
        ]
        # 期望的输出字符串
        expected = (
            '[9,[[1,null],["STR",null],[[["mathjs","Complex"],'
            '["re",4.0],["im",-5.0]],"N\\/A"]]]'
        )
        # 断言序列化 df_list 后的输出与期望的输出字符串相等
        assert (
            ujson_dumps(df_list, default_handler=default, orient="values") == expected
        )

    def test_default_handler_numpy_unsupported_dtype(self):
        # 创建一个包含不支持的 numpy 数据类型的 DataFrame 对象 df
        df = DataFrame(
            {"a": [1, 2.3, complex(4, -5)], "b": [float("nan"), None, complex(1.2, 0)]},
            columns=["a", "b"],
        )
        # 期望的输出字符串
        expected = (
            '[["(1+0j)","(nan+0j)"],'
            '["(2.3+0j)","(nan+0j)"],'
            '["(4-5j)","(1.2+0j)"]]'
        )
        # 断言序列化 df 后的输出与期望的输出字符串相等
        assert df.to_json(default_handler=str, orient="values") == expected

    def test_default_handler_raises(self):
        # 定义一个会抛出异常的内部处理函数 my_handler_raises
        msg = "raisin"

        def my_handler_raises(obj):
            raise TypeError(msg)

        # 测试 DataFrame 对象的 to_json 方法在使用自定义处理函数时是否抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": [1, 2, object()]}).to_json(
                default_handler=my_handler_raises
            )
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": [1, 2, complex(4, -5)]}).to_json(
                default_handler=my_handler_raises
            )

    def test_categorical(self):
        # 创建一个包含分类数据的 DataFrame 对象 df
        df = DataFrame({"A": ["a", "b", "c", "a", "b", "b", "a"]})
        df["B"] = df["A"]
        # 获取 df 默认的 JSON 序列化结果
        expected = df.to_json()

        # 将 df["B"] 列转换为分类数据类型
        df["B"] = df["A"].astype("category")
        # 断言转换后的 JSON 序列化结果与原始结果相等
        assert expected == df.to_json()

        # 获取 df["A"] 和 df["B"] 列的序列化 JSON 结果，并进行比较
        s = df["A"]
        sc = df["B"]
        assert s.to_json() == sc.to_json()

    def test_datetime_tz(self):
        # 创建一个包含时区信息的日期范围 tz_range 和对应的无时区信息 tz_naive
        tz_range = date_range("20130101", periods=3, tz="US/Eastern")
        tz_naive = tz_range.tz_convert("utc").tz_localize(None)

        # 创建一个包含日期列的 DataFrame 对象 df
        df = DataFrame({"A": tz_range, "B": date_range("20130101", periods=3)})

        # 复制 df，并将 "A" 列替换为无时区的 tz_naive
        df_naive = df.copy()
        df_naive["A"] = tz_naive
        # 生成警告消息
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 断言 df_naive 和 df 的 JSON 序列化结果相等，并检查是否产生了 FutureWarning 警告
        with tm.assert_produces_warning(FutureWarning, match=msg):
            expected = df_naive.to_json()
            assert expected == df.to_json()

        # 创建一个包含日期列的 Series 对象 stz 和 s_naive，并进行序列化后的比较
        stz = Series(tz_range)
        s_naive = Series(tz_naive)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            assert stz.to_json() == s_naive.to_json()
    def test_sparse(self):
        # 测试稀疏数据处理
        # 创建一个 10 行 4 列的随机数 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 将前 9 行置为 NaN（缺失值）
        df.loc[:8] = np.nan

        # 将 DataFrame 转换为稀疏数据类型
        sdf = df.astype("Sparse")
        # 获取转换后的 DataFrame 的 JSON 表示
        expected = df.to_json()
        # 断言转换后的 JSON 是否与期望值相同
        assert expected == sdf.to_json()

        # 创建一个含有 10 个元素的随机数 Series
        s = Series(np.random.default_rng(2).standard_normal(10))
        # 将前 9 个元素置为 NaN
        s.loc[:8] = np.nan
        # 将 Series 转换为稀疏数据类型
        ss = s.astype("Sparse")

        # 获取转换后的 Series 的 JSON 表示
        expected = s.to_json()
        # 断言转换后的 JSON 是否与期望值相同
        assert expected == ss.to_json()

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2013-01-10 05:00:00Z"),
            Timestamp("2013-01-10 00:00:00", tz="US/Eastern"),
            Timestamp("2013-01-10 00:00:00-0500"),
        ],
    )
    def test_tz_is_utc(self, ts):
        # 测试时区为 UTC 的时间戳序列化
        exp = '"2013-01-10T05:00:00.000Z"'

        # 使用 ujson 序列化时间戳为 ISO 格式日期字符串，断言结果与期望值相同
        assert ujson_dumps(ts, iso_dates=True) == exp
        # 将时间戳转换为 Python datetime 对象
        dt = ts.to_pydatetime()
        # 使用 ujson 序列化 datetime 对象为 ISO 格式日期字符串，断言结果与期望值相同
        assert ujson_dumps(dt, iso_dates=True) == exp

    def test_tz_is_naive(self):
        # 测试时区为本地时区（无时区信息）的时间戳序列化
        ts = Timestamp("2013-01-10 05:00:00")
        exp = '"2013-01-10T05:00:00.000"'

        # 使用 ujson 序列化时间戳为 ISO 格式日期字符串，断言结果与期望值相同
        assert ujson_dumps(ts, iso_dates=True) == exp
        # 将时间戳转换为 Python datetime 对象
        dt = ts.to_pydatetime()
        # 使用 ujson 序列化 datetime 对象为 ISO 格式日期字符串，断言结果与期望值相同
        assert ujson_dumps(dt, iso_dates=True) == exp

    @pytest.mark.parametrize(
        "tz_range",
        [
            date_range("2013-01-01 05:00:00Z", periods=2),
            date_range("2013-01-01 00:00:00", periods=2, tz="US/Eastern"),
            date_range("2013-01-01 00:00:00-0500", periods=2),
        ],
    )
    def test_tz_range_is_utc(self, tz_range):
        # 测试时区为 UTC 的日期范围序列化
        exp = '["2013-01-01T05:00:00.000Z","2013-01-02T05:00:00.000Z"]'
        dfexp = (
            '{"DT":{'
            '"0":"2013-01-01T05:00:00.000Z",'
            '"1":"2013-01-02T05:00:00.000Z"}}'
        )

        # 使用 ujson 序列化日期范围对象为 ISO 格式日期字符串，断言结果与期望值相同
        assert ujson_dumps(tz_range, iso_dates=True) == exp
        # 创建一个 DatetimeIndex 对象
        dti = DatetimeIndex(tz_range)
        # 确保对象数组中的日期时间正确序列化
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        # 创建一个 DataFrame 对象，包含 DatetimeIndex
        df = DataFrame({"DT": dti})
        # 使用 ujson 序列化 DataFrame，断言结果与期望值相同
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({"DT": object}), iso_dates=True)

    def test_tz_range_is_naive(self):
        # 测试时区为本地时区（无时区信息）的日期范围序列化
        dti = date_range("2013-01-01 05:00:00", periods=2)

        exp = '["2013-01-01T05:00:00.000","2013-01-02T05:00:00.000"]'
        dfexp = '{"DT":{"0":"2013-01-01T05:00:00.000","1":"2013-01-02T05:00:00.000"}}'

        # 使用 ujson 序列化日期范围对象为 ISO 格式日期字符串，断言结果与期望值相同
        assert ujson_dumps(dti, iso_dates=True) == exp
        assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
        # 创建一个 DataFrame 对象，包含 DatetimeIndex
        df = DataFrame({"DT": dti})
        # 使用 ujson 序列化 DataFrame，断言结果与期望值相同
        result = ujson_dumps(df, iso_dates=True)
        assert result == dfexp
        assert ujson_dumps(df.astype({"DT": object}), iso_dates=True)
    # 定义测试函数，用于测试读取内联 JSON Lines 格式数据
    def test_read_inline_jsonl(self):
        # GH9180

        # 调用 read_json 函数，解析内联 JSON Lines 字符串，生成 DataFrame 对象
        result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
        # 期望的 DataFrame 结果
        expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 标记此测试函数为单 CPU 执行，并且要求环境使用 US 本地化设置
    @pytest.mark.single_cpu
    @td.skip_if_not_us_locale
    def test_read_s3_jsonl(self, s3_public_bucket_with_data, s3so):
        # GH17200

        # 调用 read_json 函数，从 S3 中读取 JSON Lines 文件，使用给定的 storage_options
        result = read_json(
            f"s3n://{s3_public_bucket_with_data.name}/items.jsonl",
            lines=True,
            storage_options=s3so,
        )
        # 期望的 DataFrame 结果
        expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，用于测试读取本地 JSON Lines 格式数据
    def test_read_local_jsonl(self):
        # GH17200
        # 使用 tm.ensure_clean 函数创建临时文件 "tmp_items.json"，写入测试数据
        with tm.ensure_clean("tmp_items.json") as path:
            with open(path, "w", encoding="utf-8") as infile:
                infile.write('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n')
            # 调用 read_json 函数，读取临时文件中的 JSON Lines 数据
            result = read_json(path, lines=True)
            # 期望的 DataFrame 结果
            expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
            # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等
            tm.assert_frame_equal(result, expected)

    # 定义测试函数，用于测试读取包含非 ASCII Unicode 字符的 JSON Lines 格式数据
    def test_read_jsonl_unicode_chars(self):
        # GH15132: non-ascii unicode characters
        # \u201d == RIGHT DOUBLE QUOTATION MARK

        # 模拟文件句柄，创建 StringIO 对象，并调用 read_json 函数读取 JSON Lines 数据
        json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
        json = StringIO(json)
        result = read_json(json, lines=True)
        # 期望的 DataFrame 结果，包含特殊字符 "foo”"
        expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
        # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 模拟字符串，创建 StringIO 对象，并调用 read_json 函数读取 JSON Lines 数据
        json = StringIO('{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n')
        result = read_json(json, lines=True)
        # 期望的 DataFrame 结果，包含特殊字符 "foo”"
        expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
        # 使用测试框架中的 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器标记测试函数，测试处理大整数转换为 JSON 的情况
    @pytest.mark.parametrize("bigNum", [sys.maxsize + 1, -(sys.maxsize + 2)])
    def test_to_json_large_numbers(self, bigNum):
        # GH34473
        # 创建 Series 对象，包含大整数，转换为 JSON 字符串
        series = Series(bigNum, dtype=object, index=["articleId"])
        json = series.to_json()
        expected = '{"articleId":' + str(bigNum) + "}"
        # 使用 assert 断言判断 json 字符串是否与 expected 一致
        assert json == expected

        # 创建 DataFrame 对象，包含大整数，转换为 JSON 字符串
        df = DataFrame(bigNum, dtype=object, index=["articleId"], columns=[0])
        json = df.to_json()
        expected = '{"0":{"articleId":' + str(bigNum) + "}}"
        # 使用 assert 断言判断 json 字符串是否与 expected 一致
        assert json == expected

    # 使用 pytest.mark.parametrize 装饰器标记测试函数，测试读取包含大整数的 JSON 数据的情况
    @pytest.mark.parametrize("bigNum", [-(2**63) - 1, 2**64])
    def test_read_json_large_numbers(self, bigNum):
        # GH20599, 26068
        # 创建 StringIO 对象，包含大整数的 JSON 字符串
        json = StringIO('{"articleId":' + str(bigNum) + "}")
        # 期望捕获 ValueError 异常，提示数值过小或过大
        msg = r"Value is too small|Value is too big"
        with pytest.raises(ValueError, match=msg):
            # 调用 read_json 函数，尝试读取 JSON 数据
            read_json(json)

        # 创建 StringIO 对象，包含嵌套大整数的 JSON 字符串
        json = StringIO('{"0":{"articleId":' + str(bigNum) + "}}")
        with pytest.raises(ValueError, match=msg):
            # 调用 read_json 函数，尝试读取 JSON 数据
            read_json(json)
    # 定义一个测试方法：test_read_json_large_numbers2
    def test_read_json_large_numbers2(self):
        # GH18842: 标识GitHub上的问题编号
        # 创建一个包含大数值的 JSON 字符串
        json = '{"articleId": "1404366058080022500245"}'
        # 将 JSON 字符串封装成 StringIO 对象
        json = StringIO(json)
        # 调用 read_json 函数解析 JSON 数据为 Series 类型
        result = read_json(json, typ="series")
        # 创建预期的 Series 对象，包含指定的大数值
        expected = Series(1.404366e21, index=["articleId"])
        # 使用测试工具比较结果与预期的 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 创建另一个包含嵌套结构的 JSON 字符串
        json = '{"0": {"articleId": "1404366058080022500245"}}'
        # 将 JSON 字符串封装成 StringIO 对象
        json = StringIO(json)
        # 调用 read_json 函数解析 JSON 数据为 DataFrame 类型
        result = read_json(json)
        # 创建预期的 DataFrame 对象，包含指定的大数值
        expected = DataFrame(1.404366e21, index=["articleId"], columns=[0])
        # 使用测试工具比较结果与预期的 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法：test_to_jsonl
    def test_to_jsonl(self):
        # GH9180: 标识GitHub上的问题编号
        # 创建一个包含整数的 DataFrame
        df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
        # 将 DataFrame 转换为 JSON Lines 格式的字符串
        result = df.to_json(orient="records", lines=True)
        # 创建预期的 JSON Lines 格式字符串
        expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
        # 使用断言验证结果与预期是否一致
        assert result == expected

        # 创建一个包含特殊字符的 DataFrame
        df = DataFrame([["foo}", "bar"], ['foo"', "bar"]], columns=["a", "b"])
        # 将 DataFrame 转换为 JSON Lines 格式的字符串
        result = df.to_json(orient="records", lines=True)
        # 创建预期的 JSON Lines 格式字符串，包含转义字符
        expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
        # 使用断言验证结果与预期是否一致
        assert result == expected
        # 使用测试工具验证反向操作：将 JSON Lines 字符串转换为 DataFrame
        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

        # 创建一个包含转义字符的 DataFrame
        df = DataFrame([["foo\\", "bar"], ['foo"', "bar"]], columns=["a\\", "b"])
        # 将 DataFrame 转换为 JSON Lines 格式的字符串
        result = df.to_json(orient="records", lines=True)
        # 创建预期的 JSON Lines 格式字符串，包含多个转义字符
        expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
        # 使用断言验证结果与预期是否一致
        assert result == expected
        # 使用测试工具验证反向操作：将 JSON Lines 字符串转换为 DataFrame
        tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

    # 使用 pytest.mark.xfail 标记的测试方法，预期会失败
    @pytest.mark.xfail(reason="GH#13774 encoding kwarg not supported", raises=TypeError)
    # 使用 @pytest.mark.parametrize 进行参数化测试
    @pytest.mark.parametrize(
        "val",
        [
            [b"E\xc9, 17", b"", b"a", b"b", b"c"],
            [b"E\xc9, 17", b"a", b"b", b"c"],
            [b"EE, 17", b"", b"a", b"b", b"c"],
            [b"E\xc9, 17", b"\xf8\xfc", b"a", b"b", b"c"],
            [b"", b"a", b"b", b"c"],
            [b"\xf8\xfc", b"a", b"b", b"c"],
            [b"A\xf8\xfc", b"", b"a", b"b", b"c"],
            [np.nan, b"", b"b", b"c"],
            [b"A\xf8\xfc", np.nan, b"", b"b", b"c"],
        ],
    )
    # 使用 @pytest.mark.parametrize 进行参数化测试
    @pytest.mark.parametrize("dtype", ["category", object])
    # 定义一个测试方法：test_latin_encoding
    def test_latin_encoding(self, dtype, val):
        # GH 13774: 标识GitHub上的问题编号
        # 创建一个包含特定编码和数据的 Series 对象
        ser = Series(
            [x.decode("latin-1") if isinstance(x, bytes) else x for x in val],
            dtype=dtype,
        )
        # 指定编码格式为 latin-1
        encoding = "latin-1"
        # 使用测试工具确保测试过程中环境的干净性，并获取临时文件路径
        with tm.ensure_clean("test.json") as path:
            # 将 Series 对象以指定编码格式写入到 JSON 文件中
            ser.to_json(path, encoding=encoding)
            # 从 JSON 文件中读取数据，并转换为 Series 对象
            retr = read_json(StringIO(path), encoding=encoding)
            # 使用测试工具比较结果与原始的 Series 对象是否相等，忽略分类数据的检查
            tm.assert_series_equal(ser, retr, check_categorical=False)
    # 定义一个测试方法，用于测试转换为 JSON 后数据框的大小不变
    def test_data_frame_size_after_to_json(self):
        # GH15344
        # 创建一个包含单列字符串的数据框
        df = DataFrame({"a": [str(1)]})

        # 计算转换为 JSON 前数据框的内存使用量总和
        size_before = df.memory_usage(index=True, deep=True).sum()
        
        # 将数据框转换为 JSON 字符串（此处调用了不带保存参数的 to_json 方法）
        df.to_json()

        # 计算转换为 JSON 后数据框的内存使用量总和
        size_after = df.memory_usage(index=True, deep=True).sum()

        # 断言转换前后的内存使用量总和相等
        assert size_before == size_after

    # 使用参数化装饰器标记多个测试参数
    @pytest.mark.parametrize(
        "index", [None, [1, 2], [1.0, 2.0], ["a", "b"], ["1", "2"], ["1.", "2."]]
    )
    @pytest.mark.parametrize("columns", [["a", "b"], ["1", "2"], ["1.", "2."]])
    def test_from_json_to_json_table_index_and_columns(self, index, columns):
        # GH25433 GH25435
        # 创建一个预期的数据框，包含指定的索引和列
        expected = DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
        
        # 将预期的数据框转换为 JSON 表格格式
        dfjson = expected.to_json(orient="table")

        # 使用 read_json 函数从 JSON 中读取数据，期望结果与预期的数据框相等
        result = read_json(StringIO(dfjson), orient="table")
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，测试从 JSON 到 JSON 表格的转换并检查数据类型
    def test_from_json_to_json_table_dtypes(self):
        # GH21345
        # 创建一个预期的数据框，包含不同的数据类型列
        expected = DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["5", "6"]})
        
        # 将预期的数据框转换为 JSON 表格格式
        dfjson = expected.to_json(orient="table")

        # 使用 read_json 函数从 JSON 中读取数据，期望结果与预期的数据框相等
        result = read_json(StringIO(dfjson), orient="table")
        tm.assert_frame_equal(result, expected)

    # 标记测试用例为预期失败，因为 pyarrow 的字符串数据类型转换存在问题
    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="incorrect na conversion")
    @pytest.mark.parametrize("orient", ["split", "records", "index", "columns"])
    def test_to_json_from_json_columns_dtypes(self, orient):
        # GH21892 GH33205
        # 创建一个包含不同数据类型列的数据框
        expected = DataFrame.from_dict(
            {
                "Integer": Series([1, 2, 3], dtype="int64"),
                "Float": Series([None, 2.0, 3.0], dtype="float64"),
                "Object": Series([None, "", "c"], dtype="object"),
                "Bool": Series([True, False, True], dtype="bool"),
                "Category": Series(["a", "b", None], dtype="category"),
                "Datetime": Series(
                    ["2020-01-01", None, "2020-01-03"], dtype="datetime64[ns]"
                ),
            }
        )
        
        # 创建一个警告消息，用于检测未来版本中的警告
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )

        # 使用 assert_produces_warning 上下文管理器检测警告消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 将数据框按指定方向转换为 JSON 字符串
            dfjson = expected.to_json(orient=orient)

        # 使用 read_json 函数从 JSON 中读取数据，指定每列的数据类型进行比较
        result = read_json(
            StringIO(dfjson),
            orient=orient,
            dtype={
                "Integer": "int64",
                "Float": "float64",
                "Object": "object",
                "Bool": "bool",
                "Category": "category",
                "Datetime": "datetime64[ns]",
            },
        )
        # 断言读取结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化装饰器标记不同的测试参数
    @pytest.mark.parametrize("dtype", [True, {"b": int, "c": int}])
    def test_read_json_table_dtype_raises(self, dtype):
        # GH21345
        # 创建一个包含整型、浮点型和字符串的DataFrame
        df = DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["5", "6"]})
        # 将DataFrame转换为JSON格式，使用表格（table）方向
        dfjson = df.to_json(orient="table")
        # 准备错误消息，表明不能同时传递dtype和orient='table'
        msg = "cannot pass both dtype and orient='table'"
        # 使用pytest检查是否会抛出ValueError异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用read_json函数尝试读取JSON，期望抛出异常
            read_json(dfjson, orient="table", dtype=dtype)

    @pytest.mark.parametrize("orient", ["index", "columns", "records", "values"])
    def test_read_json_table_empty_axes_dtype(self, orient):
        # GH28558
        # 准备一个预期的空DataFrame
        expected = DataFrame()
        # 调用read_json函数，从空的StringIO对象中读取JSON，指定orient参数和convert_axes为True
        result = read_json(StringIO("{}"), orient=orient, convert_axes=True)
        # 使用tm模块检查结果的索引是否与预期相同
        tm.assert_index_equal(result.index, expected.index)
        # 使用tm模块检查结果的列是否与预期相同
        tm.assert_index_equal(result.columns, expected.columns)

    def test_read_json_table_convert_axes_raises(self):
        # GH25433 GH25435
        # 创建一个包含整型数据和自定义索引的DataFrame
        df = DataFrame([[1, 2], [3, 4]], index=[1.0, 2.0], columns=["1.", "2."])
        # 将DataFrame转换为JSON格式，使用表格（table）方向
        dfjson = df.to_json(orient="table")
        # 准备错误消息，表明不能同时传递convert_axes和orient='table'
        msg = "cannot pass both convert_axes and orient='table'"
        # 使用pytest检查是否会抛出ValueError异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用read_json函数尝试读取JSON，期望抛出异常
            read_json(dfjson, orient="table", convert_axes=True)

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                DataFrame([[1, 2], [4, 5]], columns=["a", "b"]),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (
                DataFrame([[1, 2], [4, 5]], columns=["a", "b"]).rename_axis("foo"),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (
                DataFrame(
                    [[1, 2], [4, 5]], columns=["a", "b"], index=[["a", "b"], ["c", "d"]]
                ),
                {"columns": ["a", "b"], "data": [[1, 2], [4, 5]]},
            ),
            (Series([1, 2, 3], name="A"), {"name": "A", "data": [1, 2, 3]}),
            (
                Series([1, 2, 3], name="A").rename_axis("foo"),
                {"name": "A", "data": [1, 2, 3]},
            ),
            (
                Series([1, 2], name="A", index=[["a", "b"], ["c", "d"]]),
                {"name": "A", "data": [1, 2]},
            ),
        ],
    )
    def test_index_false_to_json_split(self, data, expected):
        # GH 17394
        # 测试在orient='split'时使用index=False的情况

        # 将数据对象转换为JSON格式，使用orient='split'且index=False
        result = data.to_json(orient="split", index=False)
        # 解析JSON格式的结果
        result = json.loads(result)

        # 断言结果与预期是否相等
        assert result == expected

    @pytest.mark.parametrize(
        "data",
        [
            (DataFrame([[1, 2], [4, 5]], columns=["a", "b"])),
            (DataFrame([[1, 2], [4, 5]], columns=["a", "b"]).rename_axis("foo")),
            (
                DataFrame(
                    [[1, 2], [4, 5]], columns=["a", "b"], index=[["a", "b"], ["c", "d"]]
                )
            ),
            (Series([1, 2, 3], name="A")),
            (Series([1, 2, 3], name="A").rename_axis("foo")),
            (Series([1, 2], name="A", index=[["a", "b"], ["c", "d"]])),
        ],
    )
    def test_index_false_to_json_table(self, data):
        # GH 17394
        # Testing index=False in to_json with orient='table'

        # 将数据转换为 JSON 格式，orient 设置为 'table'，并且不包含索引
        result = data.to_json(orient="table", index=False)
        # 将 JSON 字符串解析为 Python 对象
        result = json.loads(result)

        # 构建预期的结果字典，包括表结构和数据
        expected = {
            "schema": pd.io.json.build_table_schema(data, index=False),
            "data": DataFrame(data).to_dict(orient="records"),
        }

        # 断言实际结果与预期结果是否相等
        assert result == expected

    @pytest.mark.parametrize("orient", ["index", "columns"])
    def test_index_false_error_to_json(self, orient):
        # GH 17394, 25513
        # Testing error message from to_json with index=False

        # 创建一个包含数据的 DataFrame
        df = DataFrame([[1, 2], [4, 5]], columns=["a", "b"])

        # 定义预期的错误消息
        msg = (
            "'index=False' is only valid when 'orient' is 'split', "
            "'table', 'records', or 'values'"
        )
        # 使用 pytest 检查是否抛出预期的 ValueError 异常，并且消息匹配预期的错误信息
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=False)

    @pytest.mark.parametrize("orient", ["records", "values"])
    def test_index_true_error_to_json(self, orient):
        # GH 25513
        # Testing error message from to_json with index=True

        # 创建一个包含数据的 DataFrame
        df = DataFrame([[1, 2], [4, 5]], columns=["a", "b"])

        # 定义预期的错误消息
        msg = (
            "'index=True' is only valid when 'orient' is 'split', "
            "'table', 'index', or 'columns'"
        )
        # 使用 pytest 检查是否抛出预期的 ValueError 异常，并且消息匹配预期的错误信息
        with pytest.raises(ValueError, match=msg):
            df.to_json(orient=orient, index=True)

    @pytest.mark.parametrize("orient", ["split", "table"])
    @pytest.mark.parametrize("index", [True, False])
    def test_index_false_from_json_to_json(self, orient, index):
        # GH25170
        # Test index=False in from_json to_json

        # 创建一个预期的 DataFrame 对象
        expected = DataFrame({"a": [1, 2], "b": [3, 4]})
        # 将 DataFrame 转换为 JSON 字符串，根据 orient 和 index 参数进行设置
        dfjson = expected.to_json(orient=orient, index=index)
        # 从 JSON 字符串中读取数据，返回的结果应与预期的 DataFrame 相等
        result = read_json(StringIO(dfjson), orient=orient)
        # 使用测试工具库检查实际结果是否与预期结果相等
        tm.assert_frame_equal(result, expected)

    def test_read_timezone_information(self):
        # GH 25546
        # 从 JSON 数据中读取时间信息

        # 从给定的 JSON 字符串中读取数据，typ 参数设置为 "series"，orient 设置为 "index"
        result = read_json(
            StringIO('{"2019-01-01T11:00:00.000Z":88}'), typ="series", orient="index"
        )
        # 构建预期的时间索引对象
        exp_dti = DatetimeIndex(["2019-01-01 11:00:00"], dtype="M8[ns, UTC]")
        # 创建预期的 Series 对象
        expected = Series([88], index=exp_dti)
        # 使用测试工具库检查实际结果是否与预期结果相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "url",
        [
            "s3://example-fsspec/",
            "gcs://another-fsspec/file.json",
            "https://example-site.com/data",
            "some-protocol://data.txt",
        ],
    )
    def test_read_json_with_url_value(self, url):
        # GH 36271
        # 从包含 URL 值的 JSON 数据中读取

        # 从 JSON 字符串中读取数据，并构建包含 url 列的 DataFrame
        result = read_json(StringIO(f'{{"url":{{"0":"{url}"}}}}'))
        # 构建预期的 DataFrame 对象
        expected = DataFrame({"url": [url]})
        # 使用测试工具库检查实际结果是否与预期结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "compression",
        ["", ".gz", ".bz2", ".tar"],
    )
    def test_read_json_with_very_long_file_path(self, compression):
        # 定义一个非常长的 JSON 文件路径，包含压缩后缀
        long_json_path = f'{"a" * 1000}.json{compression}'
        # 使用 pytest 来断言文件不存在，并匹配特定的错误信息
        with pytest.raises(
            FileNotFoundError, match=f"File {long_json_path} does not exist"
        ):
            # 调用读取 JSON 文件的函数，期望抛出文件不存在的异常
            read_json(long_json_path)

    @pytest.mark.parametrize(
        "date_format,key", [("epoch", 86400000), ("iso", "P1DT0H0M0S")]
    )
    def test_timedelta_as_label(self, date_format, key):
        # 创建一个包含 timedelta 作为标签的 DataFrame
        df = DataFrame([[1]], columns=[pd.Timedelta("1D")])
        # 根据给定的格式和键值生成预期的 JSON 字符串
        expected = f'{{"{key}":{{"0":1}}}}'

        expected_warning = None
        if date_format == "epoch":
            # 如果日期格式为 'epoch'，设置预期的警告类型为 FutureWarning
            expected_warning = FutureWarning

        # 设置警告信息内容
        msg = (
            "'epoch' date format is deprecated and will be removed in a future "
            "version, please use 'iso' date format instead."
        )
        # 使用 pytest 的 assert_produces_warning 上下文来检查是否产生了预期的警告
        with tm.assert_produces_warning(expected_warning, match=msg):
            # 将 DataFrame 转换为 JSON 字符串，使用给定的日期格式
            result = df.to_json(date_format=date_format)

        # 断言生成的 JSON 结果与预期值相等
        assert result == expected

    @pytest.mark.parametrize(
        "orient,expected",
        [
            ("index", "{\"('a', 'b')\":{\"('c', 'd')\":1}}"),
            ("columns", "{\"('c', 'd')\":{\"('a', 'b')\":1}}"),
            # 下面的情况需要特殊的编码过程
            pytest.param(
                "split",
                "",
                marks=pytest.mark.xfail(
                    reason="Produces JSON but not in a consistent manner"
                ),
            ),
            pytest.param(
                "table",
                "",
                marks=pytest.mark.xfail(
                    reason="Produces JSON but not in a consistent manner"
                ),
            ),
        ],
    )
    def test_tuple_labels(self, orient, expected):
        # 创建一个包含元组作为标签的 DataFrame
        df = DataFrame([[1]], index=[("a", "b")], columns=[("c", "d")])
        # 将 DataFrame 按照给定的 orient 参数转换为 JSON 字符串
        result = df.to_json(orient=orient)
        # 断言生成的 JSON 结果与预期值相等
        assert result == expected

    @pytest.mark.parametrize("indent", [1, 2, 4])
    def test_to_json_indent(self, indent):
        # 创建一个简单的 DataFrame
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"])

        # 将 DataFrame 转换为 JSON 字符串，指定缩进量
        result = df.to_json(indent=indent)
        spaces = " " * indent
        # 生成预期的 JSON 字符串，带有指定缩进量的格式
        expected = f"""{{
{spaces}"a":{{"{spaces}0":"foo","{spaces}1":"baz"}},
{spaces}"b":{{"{spaces}0":"bar","{spaces}1":"qux"}}
}}"""
    @pytest.mark.skipif(
        using_pyarrow_string_dtype(),
        reason="Adjust expected when infer_string is default, no bug here, "
        "just a complicated parametrization",
    )
    @pytest.mark.parametrize(
        "orient,expected",
        [
            (
                "split",
                """{
    "columns":[
        "a",
        "b"
    ],
    "index":[
        0,
        1
    ],
    "data":[
        [
            "foo",
            "bar"
        ],
        [
            "baz",
            "qux"
        ]
    ]
}""",
            ),
            (
                "records",
                """[
    {
        "a":"foo",
        "b":"bar"
    },
    {
        "a":"baz",
        "b":"qux"
    }
]""",
            ),
            (
                "index",
                """{
    "0":{
        "a":"foo",
        "b":"bar"
    },
    "1":{
        "a":"baz",
        "b":"qux"
    }
}""",
            ),
            (
                "columns",
                """{
    "a":{
        "0":"foo",
        "1":"baz"
    },
    "b":{
        "0":"bar",
        "1":"qux"
    }
}""",
            ),
            (
                "values",
                """[
    [
        "foo",
        "bar"
    ],
    [
        "baz",
        "qux"
    ]
]""",
            ),
            (
                "table",
                """{
    "schema":{
        "fields":[
            {
                "name":"index",
                "type":"integer"
            },
            {
                "name":"a",
                "type":"string"
            },
            {
                "name":"b",
                "type":"string"
            }
        ],
        "primaryKey":[
            "index"
        ],
        "pandas_version":"1.4.0"
    },
    "data":[
        {
            "index":0,
            "a":"foo",
            "b":"bar"
        },
        {
            "index":1,
            "a":"baz",
            "b":"qux"
        }
    ]
}""",
            ),
        ],
    )
    # 定义测试方法，验证 DataFrame 对象以指定方向序列化为 JSON 时的输出是否符合预期
    def test_json_indent_all_orients(self, orient, expected):
        # 创建包含数据的 DataFrame 对象
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"])
        # 将 DataFrame 对象序列化为 JSON 字符串，指定缩进为 4
        result = df.to_json(orient=orient, indent=4)
        # 断言序列化后的 JSON 字符串与预期输出是否一致
        assert result == expected

    # 测试方法，验证当传入负数缩进时是否会引发 ValueError 异常
    def test_json_negative_indent_raises(self):
        # 使用 pytest 的上下文管理器，验证调用 DataFrame.to_json() 传入负数缩进是否引发 ValueError 异常
        with pytest.raises(ValueError, match="must be a nonnegative integer"):
            DataFrame().to_json(indent=-1)

    # 测试方法，验证读取 JSON 数据时对 ECMAScript 262 中 NaN 和 Infinity 的支持
    def test_emca_262_nan_inf_support(self):
        # 创建包含 JSON 数据的 StringIO 对象
        data = StringIO(
            '["a", NaN, "NaN", Infinity, "Infinity", -Infinity, "-Infinity"]'
        )
        # 调用 read_json() 函数读取 JSON 数据并转换为 DataFrame 对象
        result = read_json(data)
        # 创建预期的 DataFrame 对象，替换 ECMAScript 262 中的 NaN 和 Infinity
        expected = DataFrame(
            ["a", None, "NaN", np.inf, "Infinity", -np.inf, "-Infinity"]
        )
        # 使用 pandas 的 tm.assert_frame_equal() 函数断言结果与预期是否一致
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，测试处理整数溢出问题
    def test_frame_int_overflow(self):
        # GH 30320：GitHub issue编号，指出问题所在
        # 创建一个包含特定内容的 JSON 字符串
        encoded_json = json.dumps([{"col": "31900441201190696999"}, {"col": "Text"}])
        # 期望的 DataFrame 对象，包含指定列的数据
        expected = DataFrame({"col": ["31900441201190696999", "Text"]})
        # 调用 read_json 函数读取编码后的 JSON 字符串，返回结果
        result = read_json(StringIO(encoded_json))
        # 使用测试工具比较两个 DataFrame 对象，确认其相等性
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，测试处理带有多级索引的 JSON 转换
    def test_json_multiindex(self):
        # 创建一个简单的 DataFrame 对象，包含两列数据
        dataframe = DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        # 期望的 JSON 字符串，包含多级索引的键值对
        expected = (
            '{"(0, \'x\')":1,"(0, \'y\')":"a","(1, \'x\')":2,'
            '"(1, \'y\')":"b","(2, \'x\')":3,"(2, \'y\')":"c"}'
        )
        # 将 DataFrame 对象堆叠为 Series 对象
        series = dataframe.stack()
        # 调用 to_json 方法将 Series 对象转换为 JSON 格式的字符串
        result = series.to_json(orient="index")
        # 断言结果与期望值相等
        assert result == expected

    # 使用 pytest 标记单 CPU 测试，测试将 DataFrame 对象写入 S3
    @pytest.mark.single_cpu
    def test_to_s3(self, s3_public_bucket, s3so):
        # GH 28375：GitHub issue编号，指出问题所在
        # 获取 S3 存储桶的名称和目标文件名
        mock_bucket_name, target_file = s3_public_bucket.name, "test.json"
        # 创建一个简单的 DataFrame 对象
        df = DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        # 调用 to_json 方法将 DataFrame 对象写入指定的 S3 路径
        df.to_json(f"s3://{mock_bucket_name}/{target_file}", storage_options=s3so)
        # 等待文件在模拟环境中出现，最长等待时间为5秒
        timeout = 5
        while True:
            # 检查目标文件是否出现在 S3 存储桶中的对象列表中
            if target_file in (obj.key for obj in s3_public_bucket.objects.all()):
                break
            # 等待0.1秒
            time.sleep(0.1)
            # 超时时间递减
            timeout -= 0.1
            # 如果超时时间小于等于0，则抛出超时异常
            assert timeout > 0, "Timed out waiting for file to appear on moto"

    # 定义一个测试方法，测试处理 Pandas 中的空值转换为 JSON
    def test_json_pandas_nulls(self, nulls_fixture, request):
        # GH 31615：GitHub issue编号，指出问题所在
        # 如果 nulls_fixture 是 Decimal 类型，则标记为预期失败，原因是未实现
        if isinstance(nulls_fixture, Decimal):
            mark = pytest.mark.xfail(reason="not implemented")
            request.applymarker(mark)

        # 预期的警告信息为 None
        expected_warning = None
        # 消息内容，警告默认 'epoch' 日期格式即将移除
        msg = (
            "The default 'epoch' date format is deprecated and will be removed "
            "in a future version, please use 'iso' date format instead."
        )
        # 如果 nulls_fixture 是 pd.NaT 类型，则预期的警告为 FutureWarning
        if nulls_fixture is pd.NaT:
            expected_warning = FutureWarning

        # 使用测试工具确保在执行代码块期间产生预期的警告信息
        with tm.assert_produces_warning(expected_warning, match=msg):
            # 将包含 nulls_fixture 的列表转换为 DataFrame，再转换为 JSON 字符串
            result = DataFrame([[nulls_fixture]]).to_json()
        # 断言生成的 JSON 字符串与预期结果相等
        assert result == '{"0":{"0":null}}'

    # 定义一个测试方法，测试读取 JSON 中的布尔类型 Series
    def test_readjson_bool_series(self):
        # GH31464：GitHub issue编号，指出问题所在
        # 调用 read_json 函数读取 JSON 字符串中的布尔类型 Series 数据
        result = read_json(StringIO("[true, true, false]"), typ="series")
        # 期望的 Series 对象，包含布尔值 True 和 False
        expected = Series([True, True, False])
        # 使用测试工具比较两个 Series 对象，确认其相等性
        tm.assert_series_equal(result, expected)
    # 测试函数，验证 DataFrame 转换为 JSON 格式时处理多索引转义的情况
    def test_to_json_multiindex_escape(self):
        # 创建一个包含布尔值的 DataFrame，将其堆叠成 Series，日期索引从 "2017-01-20" 到 "2017-01-23"
        df = DataFrame(
            True,
            index=date_range("2017-01-20", "2017-01-23"),
            columns=["foo", "bar"],
        ).stack()
        # 将 Series 转换为 JSON 格式字符串
        result = df.to_json()
        # 预期的 JSON 字符串，包含多个日期和列的布尔值
        expected = (
            "{\"(Timestamp('2017-01-20 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-20 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-21 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-21 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-22 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-22 00:00:00'), 'bar')\":true,"
            "\"(Timestamp('2017-01-23 00:00:00'), 'foo')\":true,"
            "\"(Timestamp('2017-01-23 00:00:00'), 'bar')\":true}"
        )
        # 断言实际输出与预期输出相符
        assert result == expected

    # 测试函数，验证 Series 包含自定义对象时转换为 JSON 格式的行为
    def test_to_json_series_of_objects(self):
        # 定义一个测试用的对象类 _TestObject
        class _TestObject:
            def __init__(self, a, b, _c, d) -> None:
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                return 5

        # 创建一个包含 _TestObject 实例的 Series
        series = Series([_TestObject(a=1, b=2, _c=3, d=4)])
        # 将 Series 转换为 JSON 格式字符串，并验证其转换后的内容
        assert json.loads(series.to_json()) == {"0": {"a": 1, "b": 2, "d": 4}}

    # 使用参数化测试装饰器标记的测试函数，验证复数数据结构转换为 JSON 格式字符串的正确性
    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                Series({0: -6 + 8j, 1: 0 + 1j, 2: 9 - 5j}),
                '{"0":{"imag":8.0,"real":-6.0},'   # 复数 0
                '"1":{"imag":1.0,"real":0.0},'     # 复数 1
                '"2":{"imag":-5.0,"real":9.0}}',   # 复数 2
            ),
            (
                Series({0: -9.39 + 0.66j, 1: 3.95 + 9.32j, 2: 4.03 - 0.17j}),
                '{"0":{"imag":0.66,"real":-9.39},'   # 复数 0
                '"1":{"imag":9.32,"real":3.95},'     # 复数 1
                '"2":{"imag":-0.17,"real":4.03}}',   # 复数 2
            ),
            (
                DataFrame([[-2 + 3j, -1 - 0j], [4 - 3j, -0 - 10j]]),
                '{"0":{"0":{"imag":3.0,"real":-2.0},'   # 复数 0，第一行
                '"1":{"imag":-3.0,"real":4.0}},'         # 复数 0，第二行
                '"1":{"0":{"imag":0.0,"real":-1.0},'    # 复数 1，第一行
                '"1":{"imag":-10.0,"real":0.0}}}',       # 复数 1，第二行
            ),
            (
                DataFrame(
                    [[-0.28 + 0.34j, -1.08 - 0.39j], [0.41 - 0.34j, -0.78 - 1.35j]]
                ),
                '{"0":{"0":{"imag":0.34,"real":-0.28},'   # 复数 0，第一行
                '"1":{"imag":-0.34,"real":0.41}},'         # 复数 0，第二行
                '"1":{"0":{"imag":-0.39,"real":-1.08},'   # 复数 1，第一行
                '"1":{"imag":-1.35,"real":-0.78}}}',       # 复数 1，第二行
            ),
        ],
    )
    # 测试函数，验证复数数据结构转换为 JSON 格式时的正确性
    def test_complex_data_tojson(self, data, expected):
        # 执行数据转换为 JSON 格式字符串
        result = data.to_json()
        # 断言实际输出与预期输出相符
        assert result == expected
    # 定义一个测试方法，用于测试处理 JSON 中的无符号 64 位整数
    def test_json_uint64(self):
        # 注释: 测试用例标识为 GH21073
        expected = (
            '{"columns":["col1"],"index":[0,1],'
            '"data":[[13342205958987758245],[12388075603347835679]]}'
        )
        # 创建一个 DataFrame 对象，包含一列无符号 64 位整数
        df = DataFrame(data={"col1": [13342205958987758245, 12388075603347835679]})
        # 将 DataFrame 对象转换为 JSON 格式，按照 split 方式
        result = df.to_json(orient="split")
        # 断言转换结果与预期结果相同
        assert result == expected

    # 定义另一个测试方法，用于测试读取 JSON 数据时的数据类型和后端存储方式
    def test_read_json_dtype_backend(
        self, string_storage, dtype_backend, orient, using_infer_string
        # 对以下测试用例进行参数化，orient 参数分别使用 "split", "records", "index"
        @pytest.mark.parametrize("orient", ["split", "records", "index"])
        # GH#50750
        # 如果 pyarrow 模块不可用，则跳过当前测试用例
        pa = pytest.importorskip("pyarrow")
        # 创建一个 DataFrame 对象 df，包含不同数据类型的列
        df = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": [True, False, None],
                "f": [True, False, True],
                "g": ["a", "b", "c"],
                "h": ["a", "b", None],
            }
        )

        # 根据条件选择合适的字符串数组类型
        if using_infer_string:
            # 使用 ArrowStringArrayNumpySemantics 类型的字符串数组
            string_array = ArrowStringArrayNumpySemantics(pa.array(["a", "b", "c"]))
            string_array_na = ArrowStringArrayNumpySemantics(pa.array(["a", "b", None]))
        elif string_storage == "python":
            # 使用 StringArray 类型的字符串数组（Python 对象）
            string_array = StringArray(np.array(["a", "b", "c"], dtype=np.object_))
            string_array_na = StringArray(np.array(["a", "b", NA], dtype=np.object_))
        elif dtype_backend == "pyarrow":
            # 使用 ArrowExtensionArray 类型的字符串数组
            string_array = ArrowExtensionArray(pa.array(["a", "b", "c"]))
            string_array_na = ArrowExtensionArray(pa.array(["a", "b", None]))
        else:
            # 默认使用 ArrowStringArray 类型的字符串数组
            string_array = ArrowStringArray(pa.array(["a", "b", "c"]))
            string_array_na = ArrowStringArray(pa.array(["a", "b", None]))

        # 将 DataFrame df 转换为 JSON 字符串，指定 orient 参数
        out = df.to_json(orient=orient)
        # 在特定的上下文中，设置 pandas 的选项 mode.string_storage 为 string_storage 的值
        with pd.option_context("mode.string_storage", string_storage):
            # 调用 read_json 函数读取 JSON 字符串，指定 dtype_backend 和 orient 参数
            result = read_json(
                StringIO(out), dtype_backend=dtype_backend, orient=orient
            )

        # 创建预期的 DataFrame 对象 expected，包含不同数据类型的列
        expected = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),
                "b": Series([1, 2, 3], dtype="Int64"),
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
                "e": Series([True, False, NA], dtype="boolean"),
                "f": Series([True, False, True], dtype="boolean"),
                "g": string_array,
                "h": string_array_na,
            }
        )

        # 如果 dtype_backend 为 "pyarrow"，则进一步处理 expected 的列类型
        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            # 使用 ArrowExtensionArray 类型处理 expected 的每一列
            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        # 如果 orient 参数为 "values"，则将 expected 的列名改为整数索引
        if orient == "values":
            expected.columns = list(range(8))

        # 使用 tm.assert_frame_equal 函数比较 result 和 expected，确认其内容是否一致
        tm.assert_frame_equal(result, expected)
    # 测试读取 JSON 的空值序列情况
    def test_read_json_nullable_series(self, string_storage, dtype_backend, orient):
        # 导入 pytest 库，如果未安装则跳过当前测试
        pa = pytest.importorskip("pyarrow")
        
        # 创建包含空值的整数序列
        ser = Series([1, np.nan, 3], dtype="Int64")

        # 将序列转换为 JSON 格式的字符串
        out = ser.to_json(orient=orient)
        
        # 在特定上下文中设置 Pandas 的选项，控制字符串存储的方式
        with pd.option_context("mode.string_storage", string_storage):
            # 调用 read_json 函数读取 JSON 字符串，返回 Pandas 的 Series 对象
            result = read_json(
                StringIO(out), dtype_backend=dtype_backend, orient=orient, typ="series"
            )

        # 创建预期的 Series 对象，包含相同的数据和类型
        expected = Series([1, np.nan, 3], dtype="Int64")

        # 如果 dtype_backend 选择为 'pyarrow'，则转换预期结果为 ArrowExtensionArray
        if dtype_backend == "pyarrow":
            from pandas.arrays import ArrowExtensionArray

            expected = Series(ArrowExtensionArray(pa.array(expected, from_pandas=True)))

        # 使用测试工具函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 测试非法的 dtype_backend 参数情况
    def test_invalid_dtype_backend(self):
        # 定义预期的错误消息
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        
        # 使用 pytest 来检查调用 read_json 函数时是否会引发 ValueError 异常，并匹配预期错误消息
        with pytest.raises(ValueError, match=msg):
            read_json("test", dtype_backend="numpy")
# 测试无效引擎的函数
def test_invalid_engine():
    # GH 48893：GitHub issue编号
    ser = Series(range(1))  # 创建一个包含1个元素的Series对象
    out = ser.to_json()  # 将Series对象转换为JSON格式的字符串
    # 使用pytest断言语句检测是否会引发值错误，并匹配特定错误消息
    with pytest.raises(ValueError, match="The engine type foo"):
        read_json(out, engine="foo")


# 测试pyarrow引擎且lines参数为False的函数
def test_pyarrow_engine_lines_false():
    # GH 48893：GitHub issue编号
    ser = Series(range(1))  # 创建一个包含1个元素的Series对象
    out = ser.to_json()  # 将Series对象转换为JSON格式的字符串
    # 使用pytest断言语句检测是否会引发值错误，并匹配特定错误消息
    with pytest.raises(ValueError, match="currently pyarrow engine only supports"):
        read_json(out, engine="pyarrow", lines=False)


# 测试JSON往返字符串推断的函数（使用给定的orient参数）
def test_json_roundtrip_string_inference(orient):
    pytest.importorskip("pyarrow")  # 如果pyarrow库不可用，则跳过测试
    # 创建一个DataFrame对象
    df = DataFrame(
        [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
    )
    out = df.to_json()  # 将DataFrame对象转换为JSON格式的字符串
    # 使用pd.option_context设置上下文来推断字符串类型
    with pd.option_context("future.infer_string", True):
        result = read_json(StringIO(out))  # 读取JSON格式的输入数据并返回DataFrame对象
    # 创建预期的DataFrame对象
    expected = DataFrame(
        [["a", "b"], ["c", "d"]],
        dtype="string[pyarrow_numpy]",
        index=Index(["row 1", "row 2"], dtype="string[pyarrow_numpy]"),
        columns=Index(["col 1", "col 2"], dtype="string[pyarrow_numpy]"),
    )
    tm.assert_frame_equal(result, expected)  # 断言结果与预期相等


# 如果有pyarrow库，则执行测试
@td.skip_if_no("pyarrow")
def test_to_json_ea_null():
    # GH#57224：GitHub issue编号
    # 创建一个包含两列的DataFrame对象，每列包含一个值和一个空值
    df = DataFrame(
        {
            "a": Series([1, NA], dtype="int64[pyarrow]"),
            "b": Series([2, NA], dtype="Int64"),
        }
    )
    # 将DataFrame对象转换为JSON格式的字符串（每行一个记录）
    result = df.to_json(orient="records", lines=True)
    # 预期的JSON格式字符串
    expected = """{"a":1,"b":2}
{"a":null,"b":null}
"""
    assert result == expected  # 断言结果与预期相等


# 测试读取JSON格式数据，并返回索引对象的函数
def test_read_json_lines_rangeindex():
    # GH 57429：GitHub issue编号
    data = """
{"a": 1, "b": 2}
{"a": 3, "b": 4}
"""
    # 读取JSON格式的输入数据，并获取其索引对象
    result = read_json(StringIO(data), lines=True).index
    # 预期的索引对象为RangeIndex类型，包含两个元素
    expected = RangeIndex(2)
    tm.assert_index_equal(result, expected, exact=True)  # 断言索引对象与预期相等
```