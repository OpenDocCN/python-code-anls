# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_convert_dtypes.py`

```
# 导入 datetime 模块，用于处理日期和时间
import datetime

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 测试框架
import pytest

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 导入 pandas 测试模块
import pandas._testing as tm

# 定义 TestConvertDtypes 类，用于测试数据类型转换功能
class TestConvertDtypes:

    # 使用 pytest 的 parametrize 装饰器，定义参数化测试
    @pytest.mark.parametrize(
        # 参数列表，包括 convert_integer 和 expected 的不同组合
        "convert_integer, expected", [(False, np.dtype("int32")), (True, "Int32")]
    )
    def test_convert_dtypes(
        self, convert_integer, expected, string_storage, using_infer_string
    ):
        # 在 tests/series/test_dtypes.py 中会对特定类型进行详细测试
        # 这里仅检查它在 DataFrame 中的工作情况

        # 如果使用推断字符串存储
        if using_infer_string:
            string_storage = "pyarrow_numpy"
        
        # 创建一个包含两列的 DataFrame
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
                "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
            }
        )
        
        # 使用指定的字符串存储选项上下文
        with pd.option_context("string_storage", string_storage):
            # 调用 convert_dtypes 方法进行数据类型转换
            result = df.convert_dtypes(True, True, convert_integer, False)
        
        # 创建预期的 DataFrame 结果
        expected = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=expected),
                "b": pd.Series(["x", "y", "z"], dtype=f"string[{string_storage}]"),
            }
        )
        
        # 使用测试模块中的 assert_frame_equal 方法检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    # 测试空 DataFrame 的 convert_dtypes 方法
    def test_convert_empty(self):
        # 创建一个空的 DataFrame
        empty_df = pd.DataFrame()
        
        # 使用 convert_dtypes 方法进行转换，并验证其与自身相等
        tm.assert_frame_equal(empty_df, empty_df.convert_dtypes())

    # 测试 convert_dtypes 方法保留列名的功能
    def test_convert_dtypes_retain_column_names(self):
        # 创建一个包含两列的 DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        # 设置列名
        df.columns.name = "cols"

        # 调用 convert_dtypes 方法进行数据类型转换
        result = df.convert_dtypes()
        
        # 使用测试模块中的 assert_index_equal 方法检查列名是否保留
        tm.assert_index_equal(result.columns, df.columns)
        
        # 使用 assert 语句检查结果的列名是否为 "cols"
        assert result.columns.name == "cols"
    # 测试使用 pyarrow 库进行数据类型转换的函数
    def test_pyarrow_dtype_backend(self):
        # 导入 pytest，如果导入失败则跳过测试
        pa = pytest.importorskip("pyarrow")
        # 创建一个包含多种数据类型的 Pandas DataFrame
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
                "b": pd.Series(["x", "y", None], dtype=np.dtype("O")),
                "c": pd.Series([True, False, None], dtype=np.dtype("O")),
                "d": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
                "e": pd.Series(pd.date_range("2022", periods=3)),
                "f": pd.Series(pd.date_range("2022", periods=3, tz="UTC").as_unit("s")),
                "g": pd.Series(pd.timedelta_range("1D", periods=3)),
            }
        )
        # 使用 pyarrow 库将 DataFrame 的数据类型转换为 Arrow 扩展数组
        result = df.convert_dtypes(dtype_backend="pyarrow")
        # 创建预期的 DataFrame，包含了转换后的 Arrow 扩展数组
        expected = pd.DataFrame(
            {
                "a": pd.arrays.ArrowExtensionArray(
                    pa.array([1, 2, 3], type=pa.int32())
                ),
                "b": pd.arrays.ArrowExtensionArray(pa.array(["x", "y", None])),
                "c": pd.arrays.ArrowExtensionArray(pa.array([True, False, None])),
                "d": pd.arrays.ArrowExtensionArray(pa.array([None, 100.5, 200.0])),
                "e": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.datetime(2022, 1, 1),
                            datetime.datetime(2022, 1, 2),
                            datetime.datetime(2022, 1, 3),
                        ],
                        type=pa.timestamp(unit="ns"),
                    )
                ),
                "f": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.datetime(2022, 1, 1),
                            datetime.datetime(2022, 1, 2),
                            datetime.datetime(2022, 1, 3),
                        ],
                        type=pa.timestamp(unit="s", tz="UTC"),
                    )
                ),
                "g": pd.arrays.ArrowExtensionArray(
                    pa.array(
                        [
                            datetime.timedelta(1),
                            datetime.timedelta(2),
                            datetime.timedelta(3),
                        ],
                        type=pa.duration("ns"),
                    )
                ),
            }
        )
        # 使用测试工具比较结果和预期 DataFrame
        tm.assert_frame_equal(result, expected)
    
    # 测试已经使用 pyarrow 数据类型的情况下的函数
    def test_pyarrow_dtype_backend_already_pyarrow(self):
        # 导入 pytest，如果导入失败则跳过测试
        pytest.importorskip("pyarrow")
        # 创建预期的 DataFrame，其中包含使用 pyarrow 数据类型声明的一维数组
        expected = pd.DataFrame([1, 2, 3], dtype="int64[pyarrow]")
        # 使用 pyarrow 库将 DataFrame 的数据类型转换为 Arrow 扩展数组
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        # 使用测试工具比较结果和预期 DataFrame
        tm.assert_frame_equal(result, expected)
    def test_pyarrow_dtype_backend_from_pandas_nullable(self):
        # 导入 pyarrow 库，如果导入失败则跳过此测试
        pa = pytest.importorskip("pyarrow")
        # 创建一个包含不同数据类型的 Pandas DataFrame
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, None], dtype="Int32"),
                "b": pd.Series(["x", "y", None], dtype="string[python]"),
                "c": pd.Series([True, False, None], dtype="boolean"),
                "d": pd.Series([None, 100.5, 200], dtype="Float64"),
            }
        )
        # 使用 pyarrow 进行数据类型的转换
        result = df.convert_dtypes(dtype_backend="pyarrow")
        # 创建一个期望的结果 DataFrame，其中使用了 ArrowExtensionArray
        expected = pd.DataFrame(
            {
                "a": pd.arrays.ArrowExtensionArray(
                    pa.array([1, 2, None], type=pa.int32())
                ),
                "b": pd.arrays.ArrowExtensionArray(pa.array(["x", "y", None])),
                "c": pd.arrays.ArrowExtensionArray(pa.array([True, False, None])),
                "d": pd.arrays.ArrowExtensionArray(pa.array([None, 100.5, 200.0])),
            }
        )
        # 使用 pytest 的 assert_frame_equal 函数比较结果和期望是否一致
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_dtype_empty_object(self):
        # GH 50970
        # 导入 pyarrow 库，如果导入失败则跳过此测试
        pytest.importorskip("pyarrow")
        # 创建一个空的 DataFrame，列名为 [0]
        expected = pd.DataFrame(columns=[0])
        # 使用 pyarrow 进行数据类型的转换
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        # 使用 pytest 的 assert_frame_equal 函数比较结果和期望是否一致
        tm.assert_frame_equal(result, expected)

    def test_pyarrow_engine_lines_false(self):
        # GH 48893
        # 创建一个简单的 DataFrame
        df = pd.DataFrame({"a": [1, 2, 3]})
        # 准备一个错误消息，用于测试异常情况
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        # 使用 pytest 的 raises 函数测试是否抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            df.convert_dtypes(dtype_backend="numpy")

    def test_pyarrow_backend_no_conversion(self):
        # GH#52872
        # 导入 pyarrow 库，如果导入失败则跳过此测试
        pytest.importorskip("pyarrow")
        # 创建一个包含不同数据类型的 DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": 1.5, "c": True, "d": "x"})
        # 复制原始 DataFrame 作为期望的结果
        expected = df.copy()
        # 使用 pyarrow 进行数据类型转换，但禁用所有类型转换选项
        result = df.convert_dtypes(
            convert_floating=False,
            convert_integer=False,
            convert_boolean=False,
            convert_string=False,
            dtype_backend="pyarrow",
        )
        # 使用 pytest 的 assert_frame_equal 函数比较结果和期望是否一致
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_pyarrow_to_np_nullable(self):
        # GH 53648
        # 导入 pyarrow 库，如果导入失败则跳过此测试
        pytest.importorskip("pyarrow")
        # 创建一个包含整数的 DataFrame，使用 pyarrow 进行数据类型转换
        ser = pd.DataFrame(range(2), dtype="int32[pyarrow]")
        # 使用 numpy_nullable 进行数据类型的再转换
        result = ser.convert_dtypes(dtype_backend="numpy_nullable")
        # 创建一个期望的结果 DataFrame，数据类型为 Int32
        expected = pd.DataFrame(range(2), dtype="Int32")
        # 使用 pytest 的 assert_frame_equal 函数比较结果和期望是否一致
        tm.assert_frame_equal(result, expected)

    def test_convert_dtypes_pyarrow_timestamp(self):
        # GH 54191
        # 导入 pyarrow 库，如果导入失败则跳过此测试
        pytest.importorskip("pyarrow")
        # 创建一个时间序列 Series，从 "2020-01-01" 到 "2020-01-02"，频率为每分钟
        ser = pd.Series(pd.date_range("2020-01-01", "2020-01-02", freq="1min"))
        # 将时间序列转换为指定类型的时间戳，使用 pyarrow 进行数据类型转换
        expected = ser.astype("timestamp[ms][pyarrow]")
        # 使用 pyarrow 进行数据类型的转换
        result = expected.convert_dtypes(dtype_backend="pyarrow")
        # 使用 pytest 的 assert_series_equal 函数比较结果和期望是否一致
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于验证避免块分割的数据类型转换
    def test_convert_dtypes_avoid_block_splitting(self):
        # GH#55341: GitHub issue reference
        # 创建一个包含三列的 Pandas 数据框，其中一列是整数，一列是列表，一列是字符串
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": "a"})
        # 使用 convert_dtypes 方法将数据框中的整数列保持为 Python 对象，而不转换为 Pandas 的整数类型
        result = df.convert_dtypes(convert_integer=False)
        # 创建预期结果的数据框，其中整数列仍为 Python 对象，字符串列类型为 string[python]
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": pd.Series(["a"] * 3, dtype="string[python]"),
            }
        )
        # 使用 assert_frame_equal 检查结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
        # 检查结果数据框内部的块数是否为 2
        assert result._mgr.nblocks == 2

    # 定义一个测试方法，用于验证从 Arrow 格式进行数据类型转换
    def test_convert_dtypes_from_arrow(self):
        # GH#56581: GitHub issue reference
        # 创建一个包含单行数据的 Pandas 数据框，其中一列是字符串，一列是时间对象
        df = pd.DataFrame([["a", datetime.time(18, 12)]], columns=["a", "b"])
        # 使用 convert_dtypes 方法将数据框中的列按照 Pandas 的推荐方式进行类型转换
        result = df.convert_dtypes()
        # 创建预期结果的数据框，将指定列的类型转换为 string[python]
        expected = df.astype({"a": "string[python]"})
        # 使用 assert_frame_equal 检查结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
```