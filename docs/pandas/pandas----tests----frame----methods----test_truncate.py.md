# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_truncate.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库，并从中导入需要的模块和函数
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    date_range,
)
# 导入 pandas 内部测试工具模块
import pandas._testing as tm

# 定义 TestDataFrameTruncate 类，用于测试 DataFrame 的截断操作
class TestDataFrameTruncate:
    # 定义测试方法 test_truncate，接受 datetime_frame 和 frame_or_series 作为参数
    def test_truncate(self, datetime_frame, frame_or_series):
        # 从 datetime_frame 中每隔三个时间步长取一个时间戳
        ts = datetime_frame[::3]
        # 使用测试工具函数 get_obj 对 ts 进行进一步处理
        ts = tm.get_obj(ts, frame_or_series)

        # 获取索引第 4 和第 7 个位置的时间戳作为截断操作的起始和结束时间
        start, end = datetime_frame.index[3], datetime_frame.index[6]

        # 获取索引第 3 和第 8 个位置的时间戳，用于测试截断操作中缺失起始或结束时间的情况
        start_missing = datetime_frame.index[2]
        end_missing = datetime_frame.index[7]

        # 如果未指定起始和结束时间，则对 ts 进行不截断操作
        truncated = ts.truncate()
        tm.assert_equal(truncated, ts)

        # 如果同时指定了起始和结束时间，则截取指定时间段的数据作为预期结果
        expected = ts[1:3]
        truncated = ts.truncate(start, end)
        tm.assert_equal(truncated, expected)

        # 对于缺失起始或结束时间的情况，也应当返回与同时指定起始和结束时间相同的结果
        truncated = ts.truncate(start_missing, end_missing)
        tm.assert_equal(truncated, expected)

        # 如果只指定了起始时间，则返回从指定起始时间开始的数据
        expected = ts[1:]
        truncated = ts.truncate(before=start)
        tm.assert_equal(truncated, expected)

        # 对于缺失起始时间的情况，也应当返回从指定起始时间开始的数据
        truncated = ts.truncate(before=start_missing)
        tm.assert_equal(truncated, expected)

        # 如果只指定了结束时间，则返回从数据开始到指定结束时间的数据
        expected = ts[:3]
        truncated = ts.truncate(after=end)
        tm.assert_equal(truncated, expected)

        # 对于缺失结束时间的情况，也应当返回从数据开始到指定结束时间的数据
        truncated = ts.truncate(after=end_missing)
        tm.assert_equal(truncated, expected)

        # 处理极端情况，如果指定的起始或结束时间超出了数据范围，应当返回空的 Series 或 DataFrame
        truncated = ts.truncate(after=ts.index[0] - ts.index.freq)
        assert len(truncated) == 0

        truncated = ts.truncate(before=ts.index[-1] + ts.index.freq)
        assert len(truncated) == 0

        # 使用 pytest 的断言检查，确保对于不合理的时间截断操作会抛出 ValueError 异常，并检查异常消息
        msg = "Truncate: 2000-01-06 00:00:00 must be after 2000-01-11 00:00:00"
        with pytest.raises(ValueError, match=msg):
            ts.truncate(
                before=ts.index[-1] - ts.index.freq, after=ts.index[0] + ts.index.freq
            )

    # 定义测试方法 test_truncate_nonsortedindex，用于测试非排序索引的情况
    def test_truncate_nonsortedindex(self, frame_or_series):
        # 创建一个具有非排序索引的 DataFrame 对象 obj
        obj = DataFrame({"A": ["a", "b", "c", "d", "e"]}, index=[5, 3, 2, 9, 0])
        # 使用测试工具函数 get_obj 对 obj 进行处理
        obj = tm.get_obj(obj, frame_or_series)

        # 使用 pytest 的断言检查，确保对于非排序索引的 DataFrame 进行截断操作会抛出 ValueError 异常，并检查异常消息
        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            obj.truncate(before=3, after=9)

    # 定义测试方法 test_sort_values_nonsortedindex，用于测试排序操作中非排序索引的情况
    def test_sort_values_nonsortedindex(self):
        # 创建一个时间范围为一年的 DataFrame 对象 ts，包含两列 A 和 B，数据为随机生成的标准正态分布值
        rng = date_range("2011-01-01", "2012-01-01", freq="W")
        ts = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(len(rng)),
                "B": np.random.default_rng(2).standard_normal(len(rng)),
            },
            index=rng,
        )

        # 对 DataFrame 按列 A 的值降序排列
        decreasing = ts.sort_values("A", ascending=False)

        # 使用 pytest 的断言检查，确保对于包含非排序索引的 DataFrame 进行截断操作会抛出 ValueError 异常，并检查异常消息
        msg = "truncate requires a sorted index"
        with pytest.raises(ValueError, match=msg):
            decreasing.truncate(before="2011-11", after="2011-12")
    `
        # 定义测试方法：验证在 axis=1 轴上对非排序索引进行截断操作
        def test_truncate_nonsortedindex_axis1(self):
            # GH#17935
            # 创建一个 DataFrame 对象，其中包含四列，每列为由正态分布生成的随机数
            df = DataFrame(
                {
                    3: np.random.default_rng(2).standard_normal(5),
                    20: np.random.default_rng(2).standard_normal(5),
                    2: np.random.default_rng(2).standard_normal(5),
                    0: np.random.default_rng(2).standard_normal(5),
                },
                columns=[3, 20, 2, 0],
            )
            # 定义错误信息
            msg = "truncate requires a sorted index"
            # 使用 pytest 的断言检查 ValueError 是否被触发，且错误信息与预期一致
            with pytest.raises(ValueError, match=msg):
                # 在 axis=1 轴上进行截断操作，期望抛出 ValueError 异常
                df.truncate(before=2, after=20, axis=1)
    
        @pytest.mark.parametrize(
            "before, after, indices",
            [(1, 2, [2, 1]), (None, 2, [2, 1, 0]), (1, None, [3, 2, 1])],
        )
        @pytest.mark.parametrize("dtyp", [*tm.ALL_REAL_NUMPY_DTYPES, "datetime64[ns]"])
        # 定义测试方法：验证在递减索引条件下的截断操作
        def test_truncate_decreasing_index(
            self, before, after, indices, dtyp, frame_or_series
        ):
            # https://github.com/pandas-dev/pandas/issues/33756
            # 创建一个指定数据类型的索引对象
            idx = Index([3, 2, 1, 0], dtype=dtyp)
            # 如果索引是 DatetimeIndex 类型，则转换 before 和 after 到 Timestamp 类型
            if isinstance(idx, DatetimeIndex):
                before = pd.Timestamp(before) if before is not None else None
                after = pd.Timestamp(after) if after is not None else None
                indices = [pd.Timestamp(i) for i in indices]
            # 根据给定的索引创建 frame_or_series 对象
            values = frame_or_series(range(len(idx)), index=idx)
            # 执行截断操作，返回截断后的结果
            result = values.truncate(before=before, after=after)
            # 根据 indices 切片预期结果
            expected = values.loc[indices]
            # 使用 pandas 的测试工具函数验证结果是否相等
            tm.assert_equal(result, expected)
    
        # 定义测试方法：验证在多级索引下的截断操作
        def test_truncate_multiindex(self, frame_or_series):
            # GH 34564
            # 创建一个多级索引对象
            mi = pd.MultiIndex.from_product([[1, 2, 3, 4], ["A", "B"]], names=["L1", "L2"])
            # 创建一个 DataFrame 对象，设置索引和列
            s1 = DataFrame(range(mi.shape[0]), index=mi, columns=["col"])
            # 根据 frame_or_series 参数选择适当的对象类型
            s1 = tm.get_obj(s1, frame_or_series)
    
            # 执行截断操作，返回截断后的结果
            result = s1.truncate(before=2, after=3)
    
            # 创建预期的 DataFrame 对象，并设置索引
            df = DataFrame.from_dict(
                {"L1": [2, 2, 3, 3], "L2": ["A", "B", "A", "B"], "col": [2, 3, 4, 5]}
            )
            expected = df.set_index(["L1", "L2"])
            expected = tm.get_obj(expected, frame_or_series)
    
            # 使用 pandas 的测试工具函数验证结果是否相等
            tm.assert_equal(result, expected)
    
        # 定义测试方法：验证在只有一个唯一值索引的情况下的截断操作
        def test_truncate_index_only_one_unique_value(self, frame_or_series):
            # GH 42365
            # 创建一个 Series 对象，包含重复的时间索引
            obj = Series(0, index=date_range("2021-06-30", "2021-06-30")).repeat(5)
            # 如果 frame_or_series 是 DataFrame 类型，则将 obj 转换为 DataFrame 对象
            if frame_or_series is DataFrame:
                obj = obj.to_frame(name="a")
    
            # 执行截断操作，返回截断后的结果
            truncated = obj.truncate("2021-06-28", "2021-07-01")
    
            # 使用 pandas 的测试工具函数验证结果是否相等
            tm.assert_equal(truncated, obj)
```