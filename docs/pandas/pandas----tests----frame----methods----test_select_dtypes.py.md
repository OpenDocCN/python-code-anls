# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_select_dtypes.py`

```
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import ExtensionDtype

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray

# 定义一个自定义的数据类型，继承自ExtensionDtype类
class DummyDtype(ExtensionDtype):
    type = int

    def __init__(self, numeric) -> None:
        self._numeric = numeric

    @property
    def name(self):
        return "Dummy"

    @property
    def _is_numeric(self):
        return self._numeric

# 定义一个自定义的数组类型，继承自ExtensionArray类
class DummyArray(ExtensionArray):
    def __init__(self, data, dtype) -> None:
        self.data = data
        self._dtype = dtype

    def __array__(self, dtype=None, copy=None):
        return self.data

    @property
    def dtype(self):
        return self._dtype

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        pass

    def copy(self):
        return self

# 测试类 TestSelectDtypes
class TestSelectDtypes:
    # 测试方法，验证 select_dtypes 方法在包含 np.number 时的行为
    def test_select_dtypes_include_using_list_like(self):
        # 创建一个 DataFrame 对象 df，包含多种数据类型的列
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        # 选择数值类型列，与预期结果 ei 进行比较
        ri = df.select_dtypes(include=[np.number])
        ei = df[["b", "c", "d", "k"]]
        tm.assert_frame_equal(ri, ei)

        # 选择数值类型列，排除 timedelta 类型，与预期结果 ei 进行比较
        ri = df.select_dtypes(include=[np.number], exclude=["timedelta"])
        ei = df[["b", "c", "d"]]
        tm.assert_frame_equal(ri, ei)

        # 选择数值类型和分类类型列，排除 timedelta 类型，与预期结果 ei 进行比较
        ri = df.select_dtypes(include=[np.number, "category"], exclude=["timedelta"])
        ei = df[["b", "c", "d", "f"]]
        tm.assert_frame_equal(ri, ei)

        # 选择 datetime 类型列，与预期结果 ei 进行比较
        ri = df.select_dtypes(include=["datetime"])
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        # 选择 datetime64 类型列，与预期结果 ei 进行比较
        ri = df.select_dtypes(include=["datetime64"])
        ei = df[["g"]]
        tm.assert_frame_equal(ri, ei)

        # 选择 datetimetz 类型列，与预期结果 ei 进行比较
        ri = df.select_dtypes(include=["datetimetz"])
        ei = df[["h", "i"]]
        tm.assert_frame_equal(ri, ei)

        # 验证选择 period 类型列时是否引发 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=r"^$"):
            df.select_dtypes(include=["period"])
    # 定义一个测试方法，用于测试 DataFrame 的 select_dtypes 方法，通过排除特定数据类型选择列
    def test_select_dtypes_exclude_using_list_like(self):
        # 创建一个 DataFrame 对象，包含不同数据类型的列
        df = DataFrame(
            {
                "a": list("abc"),                           # 字符串列表类型的列
                "b": list(range(1, 4)),                      # 整数范围的列
                "c": np.arange(3, 6).astype("u1"),           # 无符号整数数组转换为 'u1' 类型的列
                "d": np.arange(4.0, 7.0, dtype="float64"),   # 浮点数数组转换为 'float64' 类型的列
                "e": [True, False, True],                    # 布尔值列表类型的列
            }
        )
        # 使用 select_dtypes 方法排除数值类型的列，返回筛选后的 DataFrame
        re = df.select_dtypes(exclude=[np.number])
        # 期望的结果 DataFrame，包含列 'a' 和 'e'
        ee = df[["a", "e"]]
        # 使用 tm.assert_frame_equal 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(re, ee)

    # 定义另一个测试方法，测试 select_dtypes 方法的 include 和 exclude 参数使用列表的情况
    def test_select_dtypes_exclude_include_using_list_like(self):
        # 创建一个 DataFrame 对象，包含不同数据类型的列
        df = DataFrame(
            {
                "a": list("abc"),                           # 字符串列表类型的列
                "b": list(range(1, 4)),                      # 整数范围的列
                "c": np.arange(3, 6, dtype="u1"),            # 无符号整数数组转换为 'u1' 类型的列
                "d": np.arange(4.0, 7.0, dtype="float64"),   # 浮点数数组转换为 'float64' 类型的列
                "e": [True, False, True],                    # 布尔值列表类型的列
                "f": pd.date_range("now", periods=3).values, # 日期时间范围的列
            }
        )
        # 设置要排除的数据类型为 np.datetime64 类型的元组
        exclude = (np.datetime64,)
        # 设置要包含的数据类型为 np.bool_ 和 "integer" 的列表
        include = np.bool_, "integer"
        # 使用 select_dtypes 方法根据 include 和 exclude 参数进行数据类型选择
        r = df.select_dtypes(include=include, exclude=exclude)
        # 期望的结果 DataFrame，包含列 'b'、'c' 和 'e'
        e = df[["b", "c", "e"]]
        # 使用 tm.assert_frame_equal 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(r, e)

        # 更新排除的数据类型为字符串 'datetime' 的元组
        exclude = ("datetime",)
        # 更新包含的数据类型为字符串 'bool'、'int64' 和 'int32' 的列表
        include = "bool", "int64", "int32"
        # 再次使用 select_dtypes 方法根据 include 和 exclude 参数进行数据类型选择
        r = df.select_dtypes(include=include, exclude=exclude)
        # 更新期望的结果 DataFrame，包含列 'b' 和 'e'
        e = df[["b", "e"]]
        # 再次使用 tm.assert_frame_equal 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(r, e)

    # 使用 pytest.mark.parametrize 装饰器定义一个参数化测试方法，测试 select_dtypes 方法的 include 参数包含整数的情况
    @pytest.mark.parametrize(
        "include", [(np.bool_, "int"), (np.bool_, "integer"), ("bool", int)]
    )
    def test_select_dtypes_exclude_include_int(self, include):
        # 修复 Windows 下 select_dtypes(include='int') 的问题，参见 #36596
        # 创建一个 DataFrame 对象，包含不同数据类型的列
        df = DataFrame(
            {
                "a": list("abc"),                           # 字符串列表类型的列
                "b": list(range(1, 4)),                      # 整数范围的列
                "c": np.arange(3, 6, dtype="int32"),         # 整数数组转换为 'int32' 类型的列
                "d": np.arange(4.0, 7.0, dtype="float64"),   # 浮点数数组转换为 'float64' 类型的列
                "e": [True, False, True],                    # 布尔值列表类型的列
                "f": pd.date_range("now", periods=3).values, # 日期时间范围的列
            }
        )
        # 设置要排除的数据类型为 np.datetime64 类型的元组
        exclude = (np.datetime64,)
        # 使用参数化的 include 参数调用 select_dtypes 方法进行数据类型选择
        result = df.select_dtypes(include=include, exclude=exclude)
        # 期望的结果 DataFrame，包含列 'b'、'c' 和 'e'
        expected = df[["b", "c", "e"]]
        # 使用 tm.assert_frame_equal 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    # 定义测试方法：使用标量作为参数选择包含特定数据类型的列
    def test_select_dtypes_include_using_scalars(self):
        # 创建包含不同数据类型的 DataFrame
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        # 选择包含数值类型的列
        ri = df.select_dtypes(include=np.number)
        # 预期的结果 DataFrame，包含列 'b', 'c', 'd', 'k'
        ei = df[["b", "c", "d", "k"]]
        # 断言结果 DataFrame 是否与预期一致
        tm.assert_frame_equal(ri, ei)

        # 选择包含 datetime 类型的列
        ri = df.select_dtypes(include="datetime")
        # 预期的结果 DataFrame，包含列 'g'
        ei = df[["g"]]
        # 断言结果 DataFrame 是否与预期一致
        tm.assert_frame_equal(ri, ei)

        # 选择包含 datetime64 类型的列
        ri = df.select_dtypes(include="datetime64")
        # 预期的结果 DataFrame，包含列 'g'
        ei = df[["g"]]
        # 断言结果 DataFrame 是否与预期一致
        tm.assert_frame_equal(ri, ei)

        # 选择包含 category 类型的列
        ri = df.select_dtypes(include="category")
        # 预期的结果 DataFrame，包含列 'f'
        ei = df[["f"]]
        # 断言结果 DataFrame 是否与预期一致
        tm.assert_frame_equal(ri, ei)

        # 使用 pytest 引发预期的 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=r"^$"):
            # 选择包含 period 类型的列，但此功能尚未实现
            df.select_dtypes(include="period")

    # 定义测试方法：使用标量作为参数选择排除特定数据类型的列
    def test_select_dtypes_exclude_using_scalars(self):
        # 创建包含不同数据类型的 DataFrame
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),
                "g": pd.date_range("20130101", periods=3),
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),
                "i": pd.date_range("20130101", periods=3, tz="CET"),
                "j": pd.period_range("2013-01", periods=3, freq="M"),
                "k": pd.timedelta_range("1 day", periods=3),
            }
        )

        # 排除数值类型的列
        ri = df.select_dtypes(exclude=np.number)
        # 预期的结果 DataFrame，包含列 'a', 'e', 'f', 'g', 'h', 'i', 'j'
        ei = df[["a", "e", "f", "g", "h", "i", "j"]]
        # 断言结果 DataFrame 是否与预期一致
        tm.assert_frame_equal(ri, ei)

        # 排除 category 类型的列
        ri = df.select_dtypes(exclude="category")
        # 预期的结果 DataFrame，包含列 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k'
        ei = df[["a", "b", "c", "d", "e", "g", "h", "i", "j", "k"]]
        # 断言结果 DataFrame 是否与预期一致
        tm.assert_frame_equal(ri, ei)

        # 使用 pytest 引发预期的 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=r"^$"):
            # 排除 period 类型的列，但此功能尚未实现
            df.select_dtypes(exclude="period")
    def test_select_dtypes_include_exclude_using_scalars(self):
        # 创建一个包含多种数据类型的 DataFrame
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),  # 创建一个无符号整数类型的数组
                "d": np.arange(4.0, 7.0, dtype="float64"),  # 创建一个浮点数类型的数组
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),  # 创建一个分类数据类型的列
                "g": pd.date_range("20130101", periods=3),  # 创建一个日期范围
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),  # 创建一个带时区的日期范围
                "i": pd.date_range("20130101", periods=3, tz="CET"),  # 创建一个带时区的日期范围
                "j": pd.period_range("2013-01", periods=3, freq="M"),  # 创建一个周期范围
                "k": pd.timedelta_range("1 day", periods=3),  # 创建一个时间增量范围
            }
        )

        # 选择包含数值类型但不包含浮点数类型的列
        ri = df.select_dtypes(include=np.number, exclude="floating")
        # 期望的结果 DataFrame，包含 'b', 'c', 'k' 列
        ei = df[["b", "c", "k"]]
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_include_exclude_mixed_scalars_lists(self):
        # 创建一个包含多种数据类型的 DataFrame
        df = DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),  # 创建一个无符号整数类型的数组
                "d": np.arange(4.0, 7.0, dtype="float64"),  # 创建一个浮点数类型的数组
                "e": [True, False, True],
                "f": pd.Categorical(list("abc")),  # 创建一个分类数据类型的列
                "g": pd.date_range("20130101", periods=3),  # 创建一个日期范围
                "h": pd.date_range("20130101", periods=3, tz="US/Eastern"),  # 创建一个带时区的日期范围
                "i": pd.date_range("20130101", periods=3, tz="CET"),  # 创建一个带时区的日期范围
                "j": pd.period_range("2013-01", periods=3, freq="M"),  # 创建一个周期范围
                "k": pd.timedelta_range("1 day", periods=3),  # 创建一个时间增量范围
            }
        )

        # 选择包含数值类型但不包含浮点数和时间增量类型的列
        ri = df.select_dtypes(include=np.number, exclude=["floating", "timedelta"])
        # 期望的结果 DataFrame，包含 'b', 'c' 列
        ei = df[["b", "c"]]
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(ri, ei)

        # 选择包含数值类型和分类数据类型但不包含浮点数类型的列
        ri = df.select_dtypes(include=[np.number, "category"], exclude="floating")
        # 期望的结果 DataFrame，包含 'b', 'c', 'f', 'k' 列
        ei = df[["b", "c", "f", "k"]]
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(ri, ei)

    def test_select_dtypes_duplicate_columns(self):
        # 创建一个包含多种数据类型的 DataFrame，包括重复的列名
        df = DataFrame(
            {
                "a": ["a", "b", "c"],
                "b": [1, 2, 3],
                "c": np.arange(3, 6).astype("u1"),  # 创建一个无符号整数类型的数组
                "d": np.arange(4.0, 7.0, dtype="float64"),  # 创建一个浮点数类型的数组
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,  # 创建一个日期范围
            }
        )
        # 修改列名使其包含重复值
        df.columns = ["a", "a", "b", "b", "b", "c"]

        # 期望的结果 DataFrame，包含 'a', 'b' 列
        expected = DataFrame(
            {"a": list(range(1, 4)), "b": np.arange(3, 6).astype("u1")}
        )

        # 选择包含数值类型但不包含浮点数类型的列
        result = df.select_dtypes(include=[np.number], exclude=["floating"])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    # 测试选择指定数据类型的方法，不是属性但仍然是有效的数据类型
    def test_select_dtypes_not_an_attr_but_still_valid_dtype(self, using_infer_string):
        # 创建一个包含多种数据类型的数据框架
        df = DataFrame(
            {
                "a": list("abc"),  # 字符串列
                "b": list(range(1, 4)),  # 整数列
                "c": np.arange(3, 6).astype("u1"),  # 无符号整数列
                "d": np.arange(4.0, 7.0, dtype="float64"),  # 浮点数列
                "e": [True, False, True],  # 布尔值列
                "f": pd.date_range("now", periods=3).values,  # 日期时间列
            }
        )
        # 计算日期时间列的差分并添加为新列
        df["g"] = df.f.diff()
        # 断言：numpy 库中不存在属性 "u8"
        assert not hasattr(np, "u8")
        # 选择数据框架中指定类型的列，排除一些特定类型
        r = df.select_dtypes(include=["i8", "O"], exclude=["timedelta"])
        # 根据条件选择预期的列
        if using_infer_string:
            e = df[["b"]]  # 如果 using_infer_string 为真，则选择列 "b"
        else:
            e = df[["a", "b"]]  # 否则选择列 "a" 和 "b"
        # 断言：选择的数据框架与预期的数据框架相等
        tm.assert_frame_equal(r, e)

        # 选择包含指定类型的列，包括 timedelta64[ns] 类型
        r = df.select_dtypes(include=["i8", "O", "timedelta64[ns]"])
        # 根据条件选择预期的列
        if using_infer_string:
            e = df[["b", "g"]]  # 如果 using_infer_string 为真，则选择列 "b" 和 "g"
        else:
            e = df[["a", "b", "g"]]  # 否则选择列 "a", "b", 和 "g"
        # 断言：选择的数据框架与预期的数据框架相等
        tm.assert_frame_equal(r, e)

    # 测试选择数据类型为空的情况
    def test_select_dtypes_empty(self):
        # 创建一个简单的数据框架
        df = DataFrame({"a": list("abc"), "b": list(range(1, 4))})
        # 错误消息
        msg = "at least one of include or exclude must be nonempty"
        # 使用 pytest 断言捕获 ValueError，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            df.select_dtypes()

    # 测试选择错误的 datetime64 类型的情况
    def test_select_dtypes_bad_datetime64(self):
        # 创建一个包含多种数据类型的数据框架
        df = DataFrame(
            {
                "a": list("abc"),  # 字符串列
                "b": list(range(1, 4)),  # 整数列
                "c": np.arange(3, 6).astype("u1"),  # 无符号整数列
                "d": np.arange(4.0, 7.0, dtype="float64"),  # 浮点数列
                "e": [True, False, True],  # 布尔值列
                "f": pd.date_range("now", periods=3).values,  # 日期时间列
            }
        )
        # 使用 pytest 断言捕获 ValueError，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=".+ is too specific"):
            df.select_dtypes(include=["datetime64[D]"])

        # 使用 pytest 断言捕获 ValueError，并检查错误消息是否匹配
        with pytest.raises(ValueError, match=".+ is too specific"):
            df.select_dtypes(exclude=["datetime64[as]"])

    # 测试带有时区的 datetime64 类型的情况
    def test_select_dtypes_datetime_with_tz(self):
        # 创建一个包含带时区的日期时间的数据框架
        df2 = DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),  # 美国东部时区的时间戳
                "B": Timestamp("20130603", tz="CET"),  # 中欧时区的时间戳
            },
            index=range(5),
        )
        # 按列合并两个时间戳列的数据框架
        df3 = pd.concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        # 选择包含 datetime64[ns] 类型的列
        result = df3.select_dtypes(include=["datetime64[ns]"])
        # 重新索引结果数据框架的列为空
        expected = df3.reindex(columns=[])
        # 断言：选择的数据框架与预期的数据框架相等
        tm.assert_frame_equal(result, expected)

    # 参数化测试，测试不同数据类型和参数的情况
    @pytest.mark.parametrize("dtype", [str, "str", np.bytes_, "S1", np.str_, "U1"])
    @pytest.mark.parametrize("arg", ["include", "exclude"])
    # 测试函数：测试在选择数据类型为字符串时是否会引发异常
    def test_select_dtypes_str_raises(self, dtype, arg):
        # 创建一个包含不同数据类型的 DataFrame 对象
        df = DataFrame(
            {
                "a": list("abc"),
                "g": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )
        # 定义错误消息字符串
        msg = "string dtypes are not allowed"
        # 设置函数参数字典
        kwargs = {arg: [dtype]}

        # 使用 pytest 的断言检查是否抛出预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            df.select_dtypes(**kwargs)

    # 测试函数：测试在选择无效参数时是否会引发异常
    def test_select_dtypes_bad_arg_raises(self):
        # 创建一个包含不同数据类型的 DataFrame 对象
        df = DataFrame(
            {
                "a": list("abc"),
                "g": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("u1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("now", periods=3).values,
            }
        )

        # 定义错误消息字符串
        msg = "data type.*not understood"
        # 使用 pytest 的断言检查是否抛出预期的 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            df.select_dtypes(["blargy, blarg, blarg"])

    # 测试函数：测试在选择特定类型代码时 DataFrame 是否保持不变
    def test_select_dtypes_typecodes(self):
        # GH 11990
        # 创建一个包含随机数的 DataFrame 对象
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        # 获取所有浮点数类型的类型码列表
        FLOAT_TYPES = list(np.typecodes["AllFloat"])
        # 使用 pandas 的 assert_frame_equal 函数检查选择特定类型代码后 DataFrame 是否保持不变
        tm.assert_frame_equal(df.select_dtypes(FLOAT_TYPES), df)

    # 使用 pytest.mark.parametrize 进行参数化测试
    @pytest.mark.parametrize(
        "arr,expected",
        (
            (np.array([1, 2], dtype=np.int32), True),
            (pd.array([1, 2], dtype="Int32"), True),
            (DummyArray([1, 2], dtype=DummyDtype(numeric=True)), True),
            (DummyArray([1, 2], dtype=DummyDtype(numeric=False)), False),
        ),
    )
    # 测试函数：测试在选择数值类型时 DataFrame 是否符合预期的选择结果
    def test_select_dtypes_numeric(self, arr, expected):
        # GH 35340

        # 创建一个包含特定数据的 DataFrame 对象
        df = DataFrame(arr)
        # 检查选择数值类型后的 DataFrame 是否符合预期
        is_selected = df.select_dtypes(np.number).shape == df.shape
        # 使用断言验证结果是否与预期相符
        assert is_selected == expected

    # 测试函数：测试在选择可空字符串类型时 DataFrame 是否符合预期的选择结果
    def test_select_dtypes_numeric_nullable_string(self, nullable_string_dtype):
        # 创建一个包含可空字符串的数组
        arr = pd.array(["a", "b"], dtype=nullable_string_dtype)
        # 创建 DataFrame 对象
        df = DataFrame(arr)
        # 检查选择数值类型后的 DataFrame 是否不符合预期（应该不被选择）
        is_selected = df.select_dtypes(np.number).shape == df.shape
        # 使用断言验证结果是否与预期相符
        assert not is_selected
    @pytest.mark.parametrize(
        "expected, float_dtypes",
        [  # 参数化测试，定义多组输入参数
            [
                DataFrame(
                    {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)}
                ).astype(dtype={"A": float, "B": np.float64, "C": np.float32}),
                float,
            ],  # 第一组参数化测试数据，预期输出DataFrame和float数据类型
            [
                DataFrame(
                    {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)}
                ).astype(dtype={"A": float, "B": np.float64, "C": np.float32}),
                "float",
            ],  # 第二组参数化测试数据，预期输出DataFrame和字符串"float"数据类型
            [DataFrame({"C": range(10, 7, -1)}, dtype=np.float32), np.float32],  # 第三组参数化测试数据，预期输出DataFrame和np.float32数据类型
            [
                DataFrame({"A": range(3), "B": range(5, 8)}).astype(
                    dtype={"A": float, "B": np.float64}
                ),
                np.float64,
            ],  # 第四组参数化测试数据，预期输出DataFrame和np.float64数据类型
        ],
    )
    def test_select_dtypes_float_dtype(self, expected, float_dtypes):
        # 测试方法：选择DataFrame中指定数据类型的列
        # GH#42452：GitHub问题编号，指明相关问题
        dtype_dict = {"A": float, "B": np.float64, "C": np.float32}  # 定义数据类型字典
        df = DataFrame(
            {"A": range(3), "B": range(5, 8), "C": range(10, 7, -1)},
        )
        df = df.astype(dtype_dict)  # 将DataFrame列转换为指定数据类型
        result = df.select_dtypes(include=float_dtypes)  # 选择指定数据类型的列
        tm.assert_frame_equal(result, expected)  # 断言DataFrame是否与预期相等

    def test_np_bool_ea_boolean_include_number(self):
        # GH 46870：GitHub问题编号，指明相关问题
        df = DataFrame(
            {
                "a": [1, 2, 3],
                "b": pd.Series([True, False, True], dtype="boolean"),  # 使用布尔类型的Series
                "c": np.array([True, False, True]),  # 使用布尔类型的NumPy数组
                "d": pd.Categorical([True, False, True]),  # 使用分类数据类型
                "e": pd.arrays.SparseArray([True, False, True]),  # 使用稀疏数组
            }
        )
        result = df.select_dtypes(include="number")  # 选择数值类型的列
        expected = DataFrame({"a": [1, 2, 3]})  # 预期输出DataFrame
        tm.assert_frame_equal(result, expected)  # 断言DataFrame是否与预期相等

    def test_select_dtypes_no_view(self):
        # https://github.com/pandas-dev/pandas/issues/48090：指向相关问题的链接
        # result of this method is not a view on the original dataframe
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()  # 复制原始DataFrame
        result = df.select_dtypes(include=["number"])  # 选择数值类型的列
        result.iloc[0, 0] = 0  # 修改结果DataFrame的元素
        tm.assert_frame_equal(df, df_orig)  # 断言原始DataFrame与复制的DataFrame是否相等
```