# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_convert_dtypes.py`

```
# 从 itertools 模块导入 product 函数，用于生成参数的笛卡尔积
from itertools import product

# 导入 numpy 库，并简写为 np
import numpy as np

# 导入 pytest 库
import pytest

# 从 pandas._libs 中导入 lib
from pandas._libs import lib

# 导入 pandas 库，并简写为 pd
import pandas as pd

# 导入 pandas._testing 库，并简写为 tm
import pandas._testing as tm

# 定义一个测试类 TestSeriesConvertDtypes
class TestSeriesConvertDtypes:

    # 使用 pytest 的 parametrize 装饰器，生成参数化测试用例
    @pytest.mark.parametrize("params", product(*[(True, False)] * 5))
    def test_convert_dtypes(
        self,
        data,
        maindtype,
        expected_default,
        expected_other,
        params,
        using_infer_string,
    ):
        # 如果 data 具有 dtype 属性，并且是 numpy 的日期时间类型，并且 maindtype 是 pd.DatetimeTZDtype 类型的实例
        if (
            hasattr(data, "dtype")
            and lib.is_np_dtype(data.dtype, "M")
            and isinstance(maindtype, pd.DatetimeTZDtype)
        ):
            # 提示使用 tz_localize 替代此处的 astype
            msg = "Cannot use .astype to convert from timezone-naive dtype"
            # 使用 pytest 的 raises 断言，验证是否抛出 TypeError 异常，并匹配特定的错误消息
            with pytest.raises(TypeError, match=msg):
                pd.Series(data, dtype=maindtype)
            return

        # 根据 maindtype 是否为 None，创建 Series 对象
        if maindtype is not None:
            series = pd.Series(data, dtype=maindtype)
        else:
            series = pd.Series(data)

        # 调用 convert_dtypes 方法，并使用 params 参数
        result = series.convert_dtypes(*params)

        # 定义参数名称列表
        param_names = [
            "infer_objects",
            "convert_string",
            "convert_integer",
            "convert_boolean",
            "convert_floating",
        ]
        # 创建参数名称与值的字典
        params_dict = dict(zip(param_names, params))

        # 初始化预期的数据类型为 expected_default
        expected_dtype = expected_default

        # 遍历 expected_other 字典，根据 params 的值确定预期的数据类型
        for spec, dtype in expected_other.items():
            if all(params_dict[key] is val for key, val in zip(spec[::2], spec[1::2])):
                expected_dtype = dtype

        # 如果 using_infer_string 为 True，且 expected_default 为 "string"，且 expected_dtype 为 object
        # 且 params 的第一个值为 True，第二个值为 False
        if (
            using_infer_string
            and expected_default == "string"
            and expected_dtype == object
            and params[0]
            and not params[1]
        ):
            # 如果我们将字符串转换为对象，则使用 infer_objects 选项进行转换
            expected_dtype = "string[pyarrow_numpy]"

        # 创建预期的 Series 对象
        expected = pd.Series(data, dtype=expected_dtype)

        # 使用 pandas._testing 的 assert_series_equal 函数，验证 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 测试结果是否为副本
        copy = series.copy(deep=True)

        # 如果 result 中非空值的数量大于 0，且数据类型为 "interval[int64, right]"
        if result.notna().sum() > 0 and result.dtype in ["interval[int64, right]"]:
            # 使用 assert_produces_warning 断言，验证是否产生 FutureWarning 警告，并匹配特定的警告消息
            with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
                result[result.notna()] = np.nan
        else:
            # 将 result 中非空值的位置赋值为 NaN
            result[result.notna()] = np.nan

        # 确保原始 series 没有改变
        tm.assert_series_equal(series, copy)

    # 定义测试方法 test_convert_string_dtype，接受参数 nullable_string_dtype
    def test_convert_string_dtype(self, nullable_string_dtype):
        # 创建一个 DataFrame 对象 df，包含两列 "A" 和 "B"，使用 nullable_string_dtype 指定数据类型
        df = pd.DataFrame(
            {"A": ["a", "b", pd.NA], "B": ["ä", "ö", "ü"]}, dtype=nullable_string_dtype
        )
        # 调用 convert_dtypes 方法，返回转换后的 DataFrame 对象 result
        result = df.convert_dtypes()
        # 使用 pandas._testing 的 assert_frame_equal 函数，验证 df 和 result 是否相等
        tm.assert_frame_equal(df, result)

    # 定义测试方法 test_convert_bool_dtype
    def test_convert_bool_dtype(self):
        # 创建一个 DataFrame 对象 df，包含一列 "A"，数据类型为布尔类型
        df = pd.DataFrame({"A": pd.array([True])})
        # 使用 pandas._testing 的 assert_frame_equal 函数，验证 df 和 df.convert_dtypes() 是否相等
        tm.assert_frame_equal(df, df.convert_dtypes())
    # 定义一个测试方法，用于测试将字节字符串转换为特定数据类型的功能
    def test_convert_byte_string_dtype(self):
        # GH-43183：标识 GitHub issue 编号
        byte_str = b"binary-string"

        # 创建一个包含字节字符串的数据框，列名为"A"，索引为[0]
        df = pd.DataFrame(data={"A": byte_str}, index=[0])
        # 调用数据框的convert_dtypes方法进行数据类型转换
        result = df.convert_dtypes()
        # 预期结果与原数据框一致
        expected = df
        # 使用pytest提供的方法比较测试结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)

    # 使用pytest的参数化装饰器，定义一个测试方法，测试在包含NA值的情况下将对象列转换为特定数据类型的功能
    @pytest.mark.parametrize(
        "infer_objects, dtype", [(True, "Int64"), (False, "object")]
    )
    def test_convert_dtype_object_with_na(self, infer_objects, dtype):
        # GH#48791：标识 GitHub issue 编号
        ser = pd.Series([1, pd.NA])
        # 调用序列的convert_dtypes方法进行数据类型转换，根据参数infer_objects确定是否推断数据类型
        result = ser.convert_dtypes(infer_objects=infer_objects)
        # 根据参数dtype创建预期结果序列
        expected = pd.Series([1, pd.NA], dtype=dtype)
        # 使用pytest提供的方法比较测试结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 使用pytest的参数化装饰器，定义一个测试方法，测试在包含NA值的情况下将浮点数列转换为特定数据类型的功能
    @pytest.mark.parametrize(
        "infer_objects, dtype", [(True, "Float64"), (False, "object")]
    )
    def test_convert_dtype_object_with_na_float(self, infer_objects, dtype):
        # GH#48791：标识 GitHub issue 编号
        ser = pd.Series([1.5, pd.NA])
        # 调用序列的convert_dtypes方法进行数据类型转换，根据参数infer_objects确定是否推断数据类型
        result = ser.convert_dtypes(infer_objects=infer_objects)
        # 根据参数dtype创建预期结果序列
        expected = pd.Series([1.5, pd.NA], dtype=dtype)
        # 使用pytest提供的方法比较测试结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试将PyArrow数据类型转换为NumPy可空类型的功能
    def test_convert_dtypes_pyarrow_to_np_nullable(self):
        # GH 53648：标识 GitHub issue 编号
        # 如果未安装pyarrow，则跳过当前测试
        pytest.importorskip("pyarrow")
        # 创建一个包含整数范围的序列，数据类型为"int32[pyarrow]"
        ser = pd.Series(range(2), dtype="int32[pyarrow]")
        # 调用序列的convert_dtypes方法进行数据类型转换，指定dtype_backend为"numpy_nullable"
        result = ser.convert_dtypes(dtype_backend="numpy_nullable")
        # 创建预期结果序列，数据类型为"Int32"
        expected = pd.Series(range(2), dtype="Int32")
        # 使用pytest提供的方法比较测试结果和预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试将PyArrow数据类型转换为Pandas的pyarrow后端数据类型的功能
    def test_convert_dtypes_pyarrow_null(self):
        # GH#55346：标识 GitHub issue 编号
        # 导入pyarrow库，如果未安装pyarrow，则跳过当前测试
        pa = pytest.importorskip("pyarrow")
        # 创建一个包含两个None值的序列
        ser = pd.Series([None, None])
        # 调用序列的convert_dtypes方法进行数据类型转换，指定dtype_backend为"pyarrow"
        result = ser.convert_dtypes(dtype_backend="pyarrow")
        # 创建预期结果序列，数据类型为pd.ArrowDtype类型，使用pyarrow.null()创建null类型
        expected = pd.Series([None, None], dtype=pd.ArrowDtype(pa.null()))
        # 使用pytest提供的方法比较测试结果和预期结果是否相等
        tm.assert_series_equal(result, expected)
```