# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_pickle.py`

```
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入以下模块：
    NaT,  # NaT 表示 pandas 中的缺失日期时间值
    date_range,  # 创建一个日期范围
    to_datetime,  # 将输入转换为 datetime 对象
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestPickle:
    def test_pickle(self):
        # GH#4606
        # 创建一个包含日期时间和 NaT 的索引
        idx = to_datetime(["2013-01-01", NaT, "2014-01-06"])
        # 对索引进行往返序列化并返回
        idx_p = tm.round_trip_pickle(idx)
        # 断言序列化后的结果与原索引的第一个元素相同
        assert idx_p[0] == idx[0]
        # 断言序列化后的结果的第二个元素是 NaT
        assert idx_p[1] is NaT
        # 断言序列化后的结果与原索引的第三个元素相同
        assert idx_p[2] == idx[2]

    def test_pickle_dont_infer_freq(self):
        # GH#11002
        # 不推断频率
        # 创建一个日期范围，指定频率为每周7天
        idx = date_range("1750-1-1", "2050-1-1", freq="7D")
        # 对日期范围进行往返序列化并返回
        idx_p = tm.round_trip_pickle(idx)
        # 断言序列化后的结果与原日期范围相等
        tm.assert_index_equal(idx, idx_p)

    def test_pickle_after_set_freq(self):
        # 创建一个带有时区和名称的日期索引
        dti = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        # 设置频率为 None
        dti = dti._with_freq(None)

        # 对日期索引进行往返序列化并返回
        res = tm.round_trip_pickle(dti)
        # 断言序列化后的结果与原日期索引相等
        tm.assert_index_equal(res, dti)

    def test_roundtrip_pickle_with_tz(self):
        # GH#8367
        # 包含时区信息的往返序列化
        # 创建一个带有时区信息和名称的日期索引
        index = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        # 对日期索引进行往返序列化并返回
        unpickled = tm.round_trip_pickle(index)
        # 断言序列化后的结果与原日期索引相等
        tm.assert_index_equal(index, unpickled)

    @pytest.mark.parametrize("freq", ["B", "C"])
    def test_pickle_unpickle(self, freq):
        # 创建一个日期范围，指定频率为参数传入的值
        rng = date_range("2009-01-01", "2010-01-01", freq=freq)
        # 对日期范围进行往返序列化并返回
        unpickled = tm.round_trip_pickle(rng)
        # 断言序列化后的结果的频率与传入的频率参数相等
        assert unpickled.freq == freq
```