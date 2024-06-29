# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_dtypes.py`

```
# 导入时间间隔模块
from datetime import timedelta

# 导入numpy库，并重命名为np
import numpy as np

# 导入pytest库
import pytest

# 从pandas库的core.dtypes.dtypes模块中导入DatetimeTZDtype类
from pandas.core.dtypes.dtypes import DatetimeTZDtype

# 导入pandas库，并重命名为pd
import pandas as pd

# 从pandas库中导入DataFrame、Series、date_range方法
from pandas import (
    DataFrame,
    Series,
    date_range,
)

# 导入pandas库中的_testing模块，并重命名为tm
import pandas._testing as tm


# 定义一个名为TestDataFrameDataTypes的测试类
class TestDataFrameDataTypes:

    # 定义test_empty_frame_dtypes方法，测试空DataFrame的数据类型
    def test_empty_frame_dtypes(self):
        # 创建一个空的DataFrame对象
        empty_df = DataFrame()
        # 使用测试模块中的方法验证DataFrame的数据类型与预期的Series(dtype=object)相等
        tm.assert_series_equal(empty_df.dtypes, Series(dtype=object))

        # 创建一个没有列名的DataFrame对象
        nocols_df = DataFrame(index=[1, 2, 3])
        # 使用测试模块中的方法验证DataFrame的数据类型与预期的Series(dtype=object)相等
        tm.assert_series_equal(nocols_df.dtypes, Series(dtype=object))

        # 创建一个没有行的DataFrame对象，但有列名为"abc"
        norows_df = DataFrame(columns=list("abc"))
        # 使用测试模块中的方法验证DataFrame的数据类型与预期的Series(object, index=list("abc"))相等
        tm.assert_series_equal(norows_df.dtypes, Series(object, index=list("abc")))

        # 创建一个没有行的DataFrame对象，并将其转换为np.int32类型
        norows_int_df = DataFrame(columns=list("abc")).astype(np.int32)
        # 使用测试模块中的方法验证DataFrame的数据类型与预期的Series(np.dtype("int32"), index=list("abc"))相等
        tm.assert_series_equal(
            norows_int_df.dtypes, Series(np.dtype("int32"), index=list("abc"))
        )

        # 创建一个带有数据的DataFrame对象
        df = DataFrame({"a": 1, "b": True, "c": 1.0}, index=[1, 2, 3])
        # 创建预期的数据类型Series对象
        ex_dtypes = Series({"a": np.int64, "b": np.bool_, "c": np.float64})
        # 使用测试模块中的方法验证DataFrame的数据类型与预期的ex_dtypes相等
        tm.assert_series_equal(df.dtypes, ex_dtypes)

        # 对空切片的DataFrame进行数据类型测试
        tm.assert_series_equal(df[:0].dtypes, ex_dtypes)

    # 定义test_datetime_with_tz_dtypes方法，测试带有时区的日期时间数据类型
    def test_datetime_with_tz_dtypes(self):
        # 创建一个带有时区信息的DataFrame对象
        tzframe = DataFrame(
            {
                "A": date_range("20130101", periods=3),
                "B": date_range("20130101", periods=3, tz="US/Eastern"),
                "C": date_range("20130101", periods=3, tz="CET"),
            }
        )
        # 将指定位置的值设置为pd.NaT（缺失值）
        tzframe.iloc[1, 1] = pd.NaT
        tzframe.iloc[1, 2] = pd.NaT
        # 获取DataFrame的数据类型并排序
        result = tzframe.dtypes.sort_index()
        # 创建预期的数据类型Series对象
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype("ns", "US/Eastern"),
                DatetimeTZDtype("ns", "CET"),
            ],
            ["A", "B", "C"],
        )
        # 使用测试模块中的方法验证结果与预期的数据类型Series对象相等
        tm.assert_series_equal(result, expected)

    # 定义test_dtypes_are_correct_after_column_slice方法，测试列切片后的数据类型是否正确
    def test_dtypes_are_correct_after_column_slice(self):
        # 创建一个带有指定数据类型的DataFrame对象
        df = DataFrame(index=range(5), columns=list("abc"), dtype=np.float64)
        # 使用测试模块中的方法验证DataFrame的数据类型与预期的Series相等
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float64, "b": np.float64, "c": np.float64}),
        )
        # 对DataFrame进行列切片后，再次验证数据类型是否如预期
        tm.assert_series_equal(df.iloc[:, 2:].dtypes, Series({"c": np.float64}))
        # 再次验证整个DataFrame的数据类型是否未被改变
        tm.assert_series_equal(
            df.dtypes,
            Series({"a": np.float64, "b": np.float64, "c": np.float64}),
        )

    # 使用pytest.mark.parametrize装饰器，定义参数化测试方法
    @pytest.mark.parametrize(
        "data",
        [pd.NA, True],
    )
    # 定义test_dtypes_are_correct_after_groupby_last方法，测试分组操作后的数据类型是否正确
    def test_dtypes_are_correct_after_groupby_last(self, data):
        # 创建一个带有特定数据类型的DataFrame对象，并将其转换为适合的数据类型
        df = DataFrame(
            {"id": [1, 2, 3, 4], "test": [True, pd.NA, data, False]}
        ).convert_dtypes()
        # 对分组后的结果获取最后一行的'test'列数据
        result = df.groupby("id").last().test
        # 创建预期的数据类型Series对象
        expected = df.set_index("id").test
        # 验证结果的数据类型是否为pd.BooleanDtype
        assert result.dtype == pd.BooleanDtype()
        # 使用测试模块中的方法验证预期数据与结果数据是否相等
        tm.assert_series_equal(expected, result)
    # 定义测试方法，测试处理 DataFrame 对象时数据类型的一致性
    def test_dtypes_gh8722(self, float_string_frame):
        # 将 DataFrame 列"A"中大于0的元素转换为布尔类型，并存入新列"bool"
        float_string_frame["bool"] = float_string_frame["A"] > 0
        # 获取 DataFrame 各列的数据类型信息
        result = float_string_frame.dtypes
        # 生成预期的数据类型 Series，包含DataFrame每列的数据类型，索引与result相同
        expected = Series(
            {k: v.dtype for k, v in float_string_frame.items()}, index=result.index
        )
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    # 测试处理包含日期时间和时间差的 DataFrame 对象时数据类型的正确性
    def test_dtypes_timedeltas(self):
        # 创建包含日期时间和时间差的 DataFrame
        df = DataFrame(
            {
                "A": Series(date_range("2012-1-1", periods=3, freq="D")),
                "B": Series([timedelta(days=i) for i in range(3)]),
            }
        )
        # 获取 DataFrame 各列的数据类型信息
        result = df.dtypes
        # 生成预期的数据类型 Series，包含每列的预期数据类型，索引为列名"AB"
        expected = Series(
            [np.dtype("datetime64[ns]"), np.dtype("timedelta64[ns]")], index=list("AB")
        )
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 添加新列"C"，其值为列"A"与列"B"对应元素相加后的结果
        df["C"] = df["A"] + df["B"]
        # 再次获取 DataFrame 各列的数据类型信息
        result = df.dtypes
        # 更新预期的数据类型 Series，包含新增列"C"的数据类型，索引为列名"ABC"
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[ns]"),
                np.dtype("datetime64[ns]"),
            ],
            index=list("ABC"),
        )
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 添加新列"D"，其值为整数1
        df["D"] = 1
        # 再次获取 DataFrame 各列的数据类型信息
        result = df.dtypes
        # 更新预期的数据类型 Series，包含新增列"D"的数据类型，索引为列名"ABCD"
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[ns]"),
                np.dtype("datetime64[ns]"),
                np.dtype("int64"),
            ],
            index=list("ABCD"),
        )
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    # 测试应用函数时返回值为 NumPy 数组时的数据类型一致性
    def test_frame_apply_np_array_return_type(self, using_infer_string):
        # 创建包含单元素列表的 DataFrame
        df = DataFrame([["foo"]])
        # 对 DataFrame 应用 lambda 函数，使每列返回一个包含字符串"bar"的 NumPy 数组
        result = df.apply(lambda col: np.array("bar"))
        # 根据使用推断字符串标志位决定预期结果
        if using_infer_string:
            expected = Series([np.array(["bar"])])
        else:
            expected = Series(["bar"])
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
```