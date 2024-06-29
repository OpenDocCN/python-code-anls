# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_fillna.py`

```
import pytest  # 导入 pytest 模块

import pandas as pd  # 导入 pandas 库并使用别名 pd
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestDatetimeIndexFillNA:  # 定义测试类 TestDatetimeIndexFillNA
    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])  # 使用 pytest 的参数化装饰器定义参数化测试
    def test_fillna_datetime64(self, tz):  # 定义测试方法 test_fillna_datetime64，接受参数 tz
        # GH 11343
        idx = pd.DatetimeIndex(["2011-01-01 09:00", pd.NaT, "2011-01-01 11:00"])  # 创建一个 DatetimeIndex 对象 idx，包含日期时间和 NaT

        exp = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"]
        )  # 创建期望的 DatetimeIndex 对象 exp，填充 NaT 处的缺失值为指定时间点

        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00")), exp)  # 使用 pandas._testing.assert_index_equal 方法断言填充后的结果与期望一致

        # tz mismatch
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00"),
                pd.Timestamp("2011-01-01 10:00", tz=tz),
                pd.Timestamp("2011-01-01 11:00"),
            ],
            dtype=object,
        )  # 创建带有时区不匹配的 Index 对象 exp

        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00", tz=tz)), exp)  # 断言填充时指定时区后的结果与期望一致

        # object
        exp = pd.Index(
            [pd.Timestamp("2011-01-01 09:00"), "x", pd.Timestamp("2011-01-01 11:00")],
            dtype=object,
        )  # 创建带有对象类型的 Index 对象 exp

        tm.assert_index_equal(idx.fillna("x"), exp)  # 断言填充后的结果与对象类型的期望一致

        idx = pd.DatetimeIndex(["2011-01-01 09:00", pd.NaT, "2011-01-01 11:00"], tz=tz)  # 创建带有时区信息的 DatetimeIndex 对象 idx

        exp = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"], tz=tz
        )  # 创建带有时区信息的期望 DatetimeIndex 对象 exp

        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00", tz=tz)), exp)  # 断言填充时指定时区后的结果与期望一致

        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00", tz=tz),
                pd.Timestamp("2011-01-01 10:00"),
                pd.Timestamp("2011-01-01 11:00", tz=tz),
            ],
            dtype=object,
        )  # 创建带有时区信息的 Index 对象 exp

        tm.assert_index_equal(idx.fillna(pd.Timestamp("2011-01-01 10:00")), exp)  # 断言填充时不指定时区的结果与带有时区信息的期望一致

        # object
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01 09:00", tz=tz),
                "x",
                pd.Timestamp("2011-01-01 11:00", tz=tz),
            ],
            dtype=object,
        )  # 创建带有时区信息的对象类型 Index 对象 exp

        tm.assert_index_equal(idx.fillna("x"), exp)  # 断言填充后的结果与带有时区信息的对象类型期望一致
```