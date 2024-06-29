# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_truncate.py`

```
    # 导入datetime模块中的datetime类
    from datetime import datetime

    # 导入pytest模块，用于测试框架
    import pytest

    # 导入pandas模块，并将其命名为pd
    import pandas as pd

    # 导入pandas中的Series和date_range类
    from pandas import (
        Series,
        date_range,
    )

    # 导入pandas内部测试模块
    import pandas._testing as tm

    # 定义一个测试类TestTruncate
    class TestTruncate:

        # 测试方法：测试在带有时区的DatetimeIndex上的truncate方法
        def test_truncate_datetimeindex_tz(self):
            # 创建一个带有时区的DatetimeIndex对象idx
            idx = date_range("4/1/2005", "4/30/2005", freq="D", tz="US/Pacific")
            # 创建一个Series对象s，其索引为idx
            s = Series(range(len(idx)), index=idx)
            # 使用pytest检查是否会抛出TypeError异常，匹配特定的错误消息字符串
            with pytest.raises(TypeError, match="Cannot compare tz-naive"):
                # 调用Series对象s的truncate方法，传入两个datetime对象作为参数
                # 该语句注释中提到了GH#36148，说明这是一个特定的GitHub issue相关的修复
                s.truncate(datetime(2005, 4, 2), datetime(2005, 4, 4))

            # 获取idx的第二个元素作为lb
            lb = idx[1]
            # 获取idx的第四个元素作为ub
            ub = idx[3]
            # 调用Series对象s的truncate方法，传入lb和ub的datetime表示作为参数
            result = s.truncate(lb.to_pydatetime(), ub.to_pydatetime())
            # 创建一个期望的Series对象expected，与result进行比较
            expected = Series([1, 2, 3], index=idx[1:4])
            # 使用pandas._testing模块中的assert_series_equal函数比较result和expected
            tm.assert_series_equal(result, expected)

        # 测试方法：测试在PeriodIndex上的truncate方法
        def test_truncate_periodindex(self):
            # 创建一个PeriodIndex对象idx1
            idx1 = pd.PeriodIndex(
                [pd.Period("2017-09-02"), pd.Period("2017-09-02"), pd.Period("2017-09-03")]
            )
            # 创建一个Series对象series1，其索引为idx1
            series1 = Series([1, 2, 3], index=idx1)
            # 调用Series对象series1的truncate方法，传入字符串形式的日期作为参数
            result1 = series1.truncate(after="2017-09-02")

            # 创建一个期望的PeriodIndex对象expected_idx1
            expected_idx1 = pd.PeriodIndex(
                [pd.Period("2017-09-02"), pd.Period("2017-09-02")]
            )
            # 使用pandas._testing模块中的assert_series_equal函数比较result1和期望的Series对象
            tm.assert_series_equal(result1, Series([1, 2], index=expected_idx1))

            # 创建一个PeriodIndex对象idx2
            idx2 = pd.PeriodIndex(
                [pd.Period("2017-09-03"), pd.Period("2017-09-02"), pd.Period("2017-09-03")]
            )
            # 创建一个Series对象series2，其索引为idx2
            series2 = Series([1, 2, 3], index=idx2)
            # 对series2按索引进行排序，并调用truncate方法，传入字符串形式的日期作为参数
            result2 = series2.sort_index().truncate(after="2017-09-02")

            # 创建一个期望的PeriodIndex对象expected_idx2
            expected_idx2 = pd.PeriodIndex([pd.Period("2017-09-02")])
            # 使用pandas._testing模块中的assert_series_equal函数比较result2和期望的Series对象
            tm.assert_series_equal(result2, Series([2], index=expected_idx2))

        # 测试方法：测试仅包含一个元素的Series对象的truncate方法
        def test_truncate_one_element_series(self):
            # 创建一个仅包含一个元素的Series对象series
            series = Series([0.1], index=pd.DatetimeIndex(["2020-08-04"]))
            # 创建两个pd.Timestamp对象before和after
            before = pd.Timestamp("2020-08-02")
            after = pd.Timestamp("2020-08-04")

            # 调用Series对象series的truncate方法，传入before和after作为参数
            result = series.truncate(before=before, after=after)

            # 使用pandas._testing模块中的assert_series_equal函数比较result和series
            # 以确保输入的Series对象和期望的Series对象相同
            tm.assert_series_equal(result, series)

        # 测试方法：测试仅包含一个唯一值的Series对象的truncate方法
        def test_truncate_index_only_one_unique_value(self):
            # 创建一个Series对象obj，其索引为指定日期范围内重复的日期
            obj = Series(0, index=date_range("2021-06-30", "2021-06-30")).repeat(5)

            # 调用Series对象obj的truncate方法，传入字符串形式的日期作为参数
            truncated = obj.truncate("2021-06-28", "2021-07-01")

            # 使用pandas._testing模块中的assert_series_equal函数比较truncated和obj
            tm.assert_series_equal(truncated, obj)
```