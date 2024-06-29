# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_asof.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas._libs.tslibs import IncompatibleFrequency  # 导入异常类 IncompatibleFrequency，用于时间序列操作异常处理

from pandas import (  # 导入 Pandas 库中的多个模块和函数
    DataFrame,  # 导入 DataFrame 类，用于操作二维数据
    Period,  # 导入 Period 类，用于时间周期的表示
    Series,  # 导入 Series 类，用于操作一维数据
    Timestamp,  # 导入 Timestamp 类，用于时间戳表示
    date_range,  # 导入 date_range 函数，用于生成日期范围
    period_range,  # 导入 period_range 函数，用于生成周期范围
    to_datetime,  # 导入 to_datetime 函数，用于将输入转换为时间类型
)
import pandas._testing as tm  # 导入 Pandas 测试模块 pandas._testing as tm

@pytest.fixture  # 声明一个 pytest 的 fixture
def date_range_frame():
    """
    Fixture for DataFrame of ints with date_range index

    Columns are ['A', 'B'].
    """
    N = 50
    rng = date_range("1/1/1990", periods=N, freq="53s")  # 生成一个时间范围，频率为 53 秒
    return DataFrame({"A": np.arange(N), "B": np.arange(N)}, index=rng)  # 返回一个具有整数数据和日期索引的 DataFrame 对象


class TestFrameAsof:
    def test_basic(self, date_range_frame):
        # Explicitly cast to float to avoid implicit cast when setting np.nan
        df = date_range_frame.astype({"A": "float"})  # 将 DataFrame 中的 'A' 列显式转换为浮点数类型
        N = 50
        df.loc[df.index[15:30], "A"] = np.nan  # 将 DataFrame 中索引为 15 到 29 的 'A' 列设为 NaN
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")  # 生成一个长期日期范围，频率为 25 秒

        result = df.asof(dates)  # 使用给定的日期查询 DataFrame 的最近有效值
        assert result.notna().all(axis=1).all()  # 断言结果 DataFrame 中所有值都不是 NaN

        lb = df.index[14]  # 设置 lb 为 DataFrame 中的第 14 个索引
        ub = df.index[30]  # 设置 ub 为 DataFrame 中的第 30 个索引

        dates = list(dates)  # 将日期范围转换为列表

        result = df.asof(dates)  # 使用给定的日期列表查询 DataFrame 的最近有效值
        assert result.notna().all(axis=1).all()  # 断言结果 DataFrame 中所有值都不是 NaN

        mask = (result.index >= lb) & (result.index < ub)  # 创建一个布尔掩码，选择在 lb 和 ub 之间的索引
        rs = result[mask]  # 根据掩码选择结果 DataFrame 中的数据
        assert (rs == 14).all(axis=1).all()  # 断言结果 DataFrame 中所有选定的值都等于 14

    def test_subset(self, date_range_frame):
        N = 10
        # explicitly cast to float to avoid implicit upcast when setting to np.nan
        df = date_range_frame.iloc[:N].copy().astype({"A": "float"})  # 复制并选择 DataFrame 的前 N 行，并显式将 'A' 列转换为浮点数类型
        df.loc[df.index[4:8], "A"] = np.nan  # 将 DataFrame 中索引为 4 到 7 的 'A' 列设为 NaN
        dates = date_range("1/1/1990", periods=N * 3, freq="25s")  # 生成一个长期日期范围，频率为 25 秒

        # with a subset of A should be the same
        result = df.asof(dates, subset="A")  # 使用给定的日期和 'A' 列查询 DataFrame 的最近有效值
        expected = df.asof(dates)  # 使用给定的日期查询完整 DataFrame 的最近有效值
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试模块中的函数比较结果和预期 DataFrame 是否相等

        # same with A/B
        result = df.asof(dates, subset=["A", "B"])  # 使用给定的日期和 'A'、'B' 列查询 DataFrame 的最近有效值
        expected = df.asof(dates)  # 使用给定的日期查询完整 DataFrame 的最近有效值
        tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试模块中的函数比较结果和预期 DataFrame 是否相等

        # B gives df.asof
        result = df.asof(dates, subset="B")  # 使用给定的日期和 'B' 列查询 DataFrame 的最近有效值
        expected = df.resample("25s", closed="right").ffill().reindex(dates)  # 使用 25 秒频率重采样并填充前值，然后重新索引日期
        expected.iloc[20:] = 9  # 将索引大于等于 20 的行设为 9
        expected["B"] = expected["B"].astype(df["B"].dtype)  # 将 'B' 列转换为与 df['B'] 相同的数据类型（整数型）

        tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试模块中的函数比较结果和预期 DataFrame 是否相等
    # 测试处理缺失值情况的函数，使用指定的日期范围框架作为输入
    def test_missing(self, date_range_frame):
        # GH 15118
        # 未找到匹配项 - `where` 值位于索引中最早日期之前
        N = 10
        # 将数据切片并复制，转换为 'float64' 类型，以避免在 df.asof 中引入 nan 时的类型提升
        df = date_range_frame.iloc[:N].copy().astype("float64")

        # 使用指定日期进行 asof 操作，并返回结果
        result = df.asof("1989-12-31")

        # 生成预期的 Series 对象，其索引为 ["A", "B"]，名称为指定的 Timestamp，数据类型为 np.float64
        expected = Series(
            index=["A", "B"], name=Timestamp("1989-12-31"), dtype=np.float64
        )
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

        # 使用日期列表进行 asof 操作，并返回结果
        result = df.asof(to_datetime(["1989-12-31"]))
        # 生成预期的 DataFrame 对象，其索引为指定的日期列表，列为 ["A", "B"]，数据类型为 "float64"
        expected = DataFrame(
            index=to_datetime(["1989-12-31"]), columns=["A", "B"], dtype="float64"
        )
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

        # 确保正确处理 PeriodIndex，确保不会在系列名称中使用 period.ordinal
        df = df.to_period("D")
        # 使用指定日期进行 asof 操作，并返回结果
        result = df.asof("1989-12-31")
        # 断言结果的名称类型为 Period
        assert isinstance(result.name, Period)

    # 测试处理全部为 NaN 的情况，使用指定的 DataFrame 或 Series 作为输入
    def test_asof_all_nans(self, frame_or_series):
        # GH 15713
        # DataFrame 或 Series 全部为 NaN 的情况
        result = frame_or_series([np.nan]).asof([0])
        expected = frame_or_series([np.nan])
        # 断言两个对象是否相等
        tm.assert_equal(result, expected)

    # 测试处理全部为 NaN 的情况，使用指定的日期范围框架作为输入
    def test_all_nans(self, date_range_frame):
        # GH 15713
        # DataFrame 全部为 NaN 的情况

        # 测试非默认索引和多个输入情况
        N = 150
        rng = date_range_frame.index
        dates = date_range("1/1/1990", periods=N, freq="25s")
        # 使用指定的索引和列生成 DataFrame 对象，数据全部为 NaN，并进行 asof 操作
        result = DataFrame(np.nan, index=rng, columns=["A"]).asof(dates)
        # 生成预期的 DataFrame 对象，其索引为指定的日期列表，列为 ["A"]，数据类型为 "float64"
        expected = DataFrame(np.nan, index=dates, columns=["A"])
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

        # 测试多列情况
        dates = date_range("1/1/1990", periods=N, freq="25s")
        # 使用指定的索引和列生成 DataFrame 对象，数据全部为 NaN，并进行 asof 操作
        result = DataFrame(np.nan, index=rng, columns=["A", "B", "C"]).asof(dates)
        # 生成预期的 DataFrame 对象，其索引为指定的日期列表，列为 ["A", "B", "C"]，数据类型为 "float64"
        expected = DataFrame(np.nan, index=dates, columns=["A", "B", "C"])
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

        # 测试标量输入情况
        result = DataFrame(np.nan, index=[1, 2], columns=["A", "B"]).asof([3])
        # 生成预期的 DataFrame 对象，其索引为 [3]，列为 ["A", "B"]，数据全部为 NaN
        expected = DataFrame(np.nan, index=[3], columns=["A", "B"])
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

        result = DataFrame(np.nan, index=[1, 2], columns=["A", "B"]).asof(3)
        # 生成预期的 Series 对象，其索引为 ["A", "B"]，名称为 3，数据全部为 NaN
        expected = Series(np.nan, index=["A", "B"], name=3)
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "stamp,expected",
        [
            (
                Timestamp("2018-01-01 23:22:43.325+00:00"),
                Series(2, name=Timestamp("2018-01-01 23:22:43.325+00:00")),
            ),
            (
                Timestamp("2018-01-01 22:33:20.682+01:00"),
                Series(1, name=Timestamp("2018-01-01 22:33:20.682+01:00")),
            ),
        ],
    )
    # 测试时区感知的 DataFrame 索引功能
    def test_time_zone_aware_index(self, stamp, expected):
        # GH21194
        # 测试 DataFrame 索引的时区感知能力，考虑不同的 UTC 和时区
        df = DataFrame(
            data=[1, 2],
            index=[
                Timestamp("2018-01-01 21:00:05.001+00:00"),
                Timestamp("2018-01-01 22:35:10.550+00:00"),
            ],
        )

        # 使用给定时间戳获取最接近的数据行
        result = df.asof(stamp)
        # 检查结果是否符合预期
        tm.assert_series_equal(result, expected)

    # 测试 asof 方法在 PeriodIndex 频率不匹配时的行为
    def test_asof_periodindex_mismatched_freq(self):
        N = 50
        # 创建一个 PeriodIndex，频率为每小时
        rng = period_range("1/1/1990", periods=N, freq="h")
        # 创建一个随机数据的 DataFrame，索引为 rng
        df = DataFrame(np.random.default_rng(2).standard_normal(N), index=rng)

        # 检查不匹配频率时的异常处理
        msg = "Input has different freq"
        with pytest.raises(IncompatibleFrequency, match=msg):
            # 使用不匹配频率的 PeriodIndex 调用 asof 方法
            df.asof(rng.asfreq("D"))

    # 测试 asof 方法在保留布尔类型数据时的行为
    def test_asof_preserves_bool_dtype(self):
        # GH#16063 曾经会将布尔值转换为浮点数
        # 创建一个日期范围为每月开始的 Series，包含 True 和 False 值
        dti = date_range("2017-01-01", freq="MS", periods=4)
        ser = Series([True, False, True], index=dti[:-1])

        # 最后一个时间戳
        ts = dti[-1]
        # 获取该时间戳处的数据
        res = ser.asof([ts])

        # 预期结果是包含 True 的 Series
        expected = Series([True], index=[ts])
        # 检查结果是否符合预期
        tm.assert_series_equal(res, expected)
```