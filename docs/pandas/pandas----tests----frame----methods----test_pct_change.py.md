# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_pct_change.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestDataFramePctChange:
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义多组参数化测试
        "periods, exp",
        [
            (1, [np.nan, np.nan, np.nan, 1, 1, 1.5, np.nan, np.nan]),
            (-1, [np.nan, np.nan, -0.5, -0.5, -0.6, np.nan, np.nan, np.nan]),
        ],
    )
    def test_pct_change_with_nas(self, periods, exp, frame_or_series):
        vals = [np.nan, np.nan, 1, 2, 4, 10, np.nan, np.nan]  # 创建测试数据值列表
        obj = frame_or_series(vals)  # 使用 frame_or_series 函数创建对象

        res = obj.pct_change(periods=periods)  # 计算对象的百分比变化
        tm.assert_equal(res, frame_or_series(exp))  # 使用测试模块断言函数检查结果

    def test_pct_change_numeric(self):
        # GH#11150
        pnl = DataFrame(  # 创建一个 DataFrame 对象
            [np.arange(0, 40, 10), np.arange(0, 40, 10), np.arange(0, 40, 10)]
        ).astype(np.float64)  # 将 DataFrame 转换为浮点数类型
        pnl.iat[1, 0] = np.nan  # 设置特定位置为 NaN
        pnl.iat[1, 1] = np.nan  # 设置特定位置为 NaN
        pnl.iat[2, 3] = 60  # 设置特定位置的值为 60

        for axis in range(2):  # 遍历两个轴
            expected = pnl / pnl.shift(axis=axis) - 1  # 计算百分比变化的期望值
            result = pnl.pct_change(axis=axis)  # 计算实际的百分比变化
            tm.assert_frame_equal(result, expected)  # 使用测试模块断言函数检查结果

    def test_pct_change(self, datetime_frame):
        rs = datetime_frame.pct_change()  # 计算 DataFrame 的百分比变化
        tm.assert_frame_equal(rs, datetime_frame / datetime_frame.shift(1) - 1)  # 使用测试模块断言函数检查结果

        rs = datetime_frame.pct_change(2)  # 计算 DataFrame 的两期百分比变化
        filled = datetime_frame.ffill()  # 对 DataFrame 进行向前填充缺失值
        tm.assert_frame_equal(rs, filled / filled.shift(2) - 1)  # 使用测试模块断言函数检查结果

        rs = datetime_frame.pct_change()  # 再次计算 DataFrame 的百分比变化
        tm.assert_frame_equal(rs, datetime_frame / datetime_frame.shift(1) - 1)  # 使用测试模块断言函数检查结果

        rs = datetime_frame.pct_change(freq="5D")  # 根据频率计算 DataFrame 的百分比变化
        tm.assert_frame_equal(
            rs,
            (datetime_frame / datetime_frame.shift(freq="5D") - 1).reindex_like(
                datetime_frame
            ),  # 使用测试模块断言函数检查结果
        )

    def test_pct_change_shift_over_nas(self):
        s = Series([1.0, 1.5, np.nan, 2.5, 3.0])  # 创建一个 Series 对象

        df = DataFrame({"a": s, "b": s})  # 创建一个包含 Series 的 DataFrame 对象

        chg = df.pct_change()  # 计算 DataFrame 的百分比变化
        expected = Series([np.nan, 0.5, np.nan, np.nan, 0.2])  # 创建预期的结果 Series
        edf = DataFrame({"a": expected, "b": expected})  # 创建包含预期结果的 DataFrame 对象
        tm.assert_frame_equal(chg, edf)  # 使用测试模块断言函数检查结果

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义多组参数化测试
        "freq, periods",
        [
            ("5B", 5),
            ("3B", 3),
            ("14B", 14),
        ],
    )
    def test_pct_change_periods_freq(
        self,
        datetime_frame,
        freq,
        periods,
    ):
        # GH#7292
        rs_freq = datetime_frame.pct_change(freq=freq)  # 根据频率计算 DataFrame 的百分比变化
        rs_periods = datetime_frame.pct_change(periods)  # 根据期数计算 DataFrame 的百分比变化
        tm.assert_frame_equal(rs_freq, rs_periods)  # 使用测试模块断言函数检查结果

        empty_ts = DataFrame(index=datetime_frame.index, columns=datetime_frame.columns)  # 创建一个空的 DataFrame 对象
        rs_freq = empty_ts.pct_change(freq=freq)  # 根据频率计算空 DataFrame 的百分比变化
        rs_periods = empty_ts.pct_change(periods)  # 根据期数计算空 DataFrame 的百分比变化
        tm.assert_frame_equal(rs_freq, rs_periods)  # 使用测试模块断言函数检查结果


def test_pct_change_with_duplicated_indices():
    # GH30463
    data = DataFrame(  # 创建一个 DataFrame 对象
        {0: [np.nan, 1, 2, 3, 9, 18], 1: [0, 1, np.nan, 3, 9, 18]}, index=["a", "b"] * 3
    )
    # 计算数据的百分比变化，即相邻元素之间的变化率
    result = data.pct_change()

    # 创建一个包含 NaN、无穷大和 NaN 值的第二列列表
    second_column = [np.nan, np.inf, np.nan, np.nan, 2.0, 1.0]
    # 创建预期的 DataFrame 对象，包含两列，第一列为特定值和 NaN 的组合，第二列为 second_column
    expected = DataFrame(
        {0: [np.nan, np.nan, 1.0, 0.5, 2.0, 1.0], 1: second_column},
        index=["a", "b"] * 3,
    )
    # 使用测试框架中的函数比较计算结果与预期结果的 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试数据框架在出现空值时的百分比变化计算
def test_pct_change_none_beginning():
    # GH#54481: 关联 GitHub 上的 issue 编号，用于跟踪和查看相关问题详情

    # 创建一个包含数据的数据框架，包括整数和空值
    df = DataFrame(
        [
            [1, None],  # 第一行：包含整数 1 和空值
            [2, 1],     # 第二行：包含整数 2 和整数 1
            [3, 2],     # 第三行：包含整数 3 和整数 2
            [4, 3],     # 第四行：包含整数 4 和整数 3
            [5, 4],     # 第五行：包含整数 5 和整数 4
        ]
    )

    # 对数据框架 df 进行百分比变化计算
    result = df.pct_change()

    # 创建预期的数据框架，包含百分比变化的期望结果
    expected = DataFrame(
        {0: [np.nan, 1, 0.5, 1 / 3, 0.25], 1: [np.nan, np.nan, 1, 0.5, 1 / 3]}
    )

    # 使用测试工具函数来断言计算结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)
```