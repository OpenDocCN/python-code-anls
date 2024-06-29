# `D:\src\scipysrc\pandas\pandas\tests\frame\test_cumulative.py`

```
"""
Tests for DataFrame cumulative operations

See also
--------
tests.series.test_cumulative
"""

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入以下模块：
    DataFrame,  # 数据框对象
    Series,  # 系列对象
    Timestamp,  # 时间戳对象
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


class TestDataFrameCumulativeOps:
    # ---------------------------------------------------------------------
    # Cumulative Operations - cumsum, cummax, ...

    def test_cumulative_ops_smoke(self):
        # it works
        df = DataFrame({"A": np.arange(20)}, index=np.arange(20))
        df.cummax()  # 计算每列的累积最大值
        df.cummin()  # 计算每列的累积最小值
        df.cumsum()  # 计算每列的累积和

        dm = DataFrame(np.arange(20).reshape(4, 5), index=range(4), columns=range(5))
        # TODO(wesm): do something with this?
        dm.cumsum()  # 计算每列的累积和

    def test_cumprod_smoke(self, datetime_frame):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan

        # ints
        df = datetime_frame.fillna(0).astype(int)  # 填充缺失值为0并转换为整数类型
        df.cumprod(0)  # 按列计算累积乘积
        df.cumprod(1)  # 按行计算累积乘积

        # ints32
        df = datetime_frame.fillna(0).astype(np.int32)  # 填充缺失值为0并转换为32位整数类型
        df.cumprod(0)  # 按列计算累积乘积
        df.cumprod(1)  # 按行计算累积乘积

    def test_cumulative_ops_match_series_apply(
        self, datetime_frame, all_numeric_accumulations
    ):
        datetime_frame.iloc[5:10, 0] = np.nan
        datetime_frame.iloc[10:15, 1] = np.nan
        datetime_frame.iloc[15:, 2] = np.nan

        # axis = 0
        result = getattr(datetime_frame, all_numeric_accumulations)()  # 执行累积操作（如累积和、最大值等），默认按列
        expected = datetime_frame.apply(getattr(Series, all_numeric_accumulations))  # 预期结果，使用 apply 应用到每列或每行

        tm.assert_frame_equal(result, expected)  # 断言结果与预期相等

        # axis = 1
        result = getattr(datetime_frame, all_numeric_accumulations)(axis=1)  # 执行累积操作，按行
        expected = datetime_frame.apply(
            getattr(Series, all_numeric_accumulations), axis=1
        )  # 预期结果，应用到每行

        tm.assert_frame_equal(result, expected)  # 断言结果与预期相等

        # fix issue TODO: GH ref?
        assert np.shape(result) == np.shape(datetime_frame)  # 检查结果的形状与原始数据框是否相同

    def test_cumsum_preserve_dtypes(self):
        # GH#19296 dont incorrectly upcast to object
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3.0], "C": [True, False, False]})  # 创建包含不同数据类型的数据框

        result = df.cumsum()  # 计算每列的累积和

        expected = DataFrame(
            {
                "A": Series([1, 3, 6], dtype=np.int64),  # 预期结果：A 列累积和，保持 int64 类型
                "B": Series([1, 3, 6], dtype=np.float64),  # 预期结果：B 列累积和，保持 float64 类型
                "C": df["C"].cumsum(),  # 预期结果：C 列累积和
            }
        )
        tm.assert_frame_equal(result, expected)  # 断言结果与预期相等

    @pytest.mark.parametrize("method", ["cumsum", "cumprod", "cummin", "cummax"])
    @pytest.mark.parametrize("axis", [0, 1])
    # 定义测试方法，用于测试带有数字类型数据标志的函数
    def test_numeric_only_flag(self, method, axis):
        # 创建一个 DataFrame 包含不同类型的列：整数、布尔、字符串、浮点数和日期时间
        df = DataFrame(
            {
                "int": [1, 2, 3],
                "bool": [True, False, False],
                "string": ["a", "b", "c"],
                "float": [1.0, 3.5, 4.0],
                "datetime": [
                    Timestamp(2018, 1, 1),
                    Timestamp(2019, 1, 1),
                    Timestamp(2020, 1, 1),
                ],
            }
        )
        # 创建仅包含数字类型列的新 DataFrame，删除了字符串和日期时间列
        df_numeric_only = df.drop(["string", "datetime"], axis=1)

        # 调用 DataFrame 对象的特定方法，传入参数 axis 和 numeric_only=True
        result = getattr(df, method)(axis=axis, numeric_only=True)
        # 调用仅包含数字类型列的 DataFrame 的相同方法，传入参数 axis
        expected = getattr(df_numeric_only, method)(axis)
        # 使用测试工具检查结果是否符合预期
        tm.assert_frame_equal(result, expected)
```