# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_freq_attr.py`

```
import pytest
# 导入 pytest 模块

from pandas import (
    DatetimeIndex,
    date_range,
)
# 从 pandas 模块中导入 DatetimeIndex 和 date_range 函数

from pandas.tseries.offsets import (
    BDay,
    DateOffset,
    Day,
    Hour,
)
# 从 pandas.tseries.offsets 模块中导入 BDay, DateOffset, Day, Hour 类


class TestFreq:
    def test_freq_setter_errors(self):
        # GH#20678
        # 创建一个 DatetimeIndex 对象
        idx = DatetimeIndex(["20180101", "20180103", "20180105"])

        # 设置不兼容频率的错误消息
        msg = (
            "Inferred frequency 2D from passed values does not conform to "
            "passed frequency 5D"
        )
        # 使用 pytest 检查是否会引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            idx._data.freq = "5D"

        # 设置非频率字符串的错误消息
        with pytest.raises(ValueError, match="Invalid frequency"):
            idx._data.freq = "foo"

    @pytest.mark.parametrize("values", [["20180101", "20180103", "20180105"], []])
    @pytest.mark.parametrize("freq", ["2D", Day(2), "2B", BDay(2), "48h", Hour(48)])
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_freq_setter(self, values, freq, tz):
        # GH#20678
        # 创建一个 DatetimeIndex 对象
        idx = DatetimeIndex(values, tz=tz)

        # 可以设置为一个偏移量，必要时从字符串转换
        idx._data.freq = freq
        # 断言 idx.freq 等于 freq
        assert idx.freq == freq
        # 断言 idx.freq 是 DateOffset 类的实例
        assert isinstance(idx.freq, DateOffset)

        # 可以重置为 None
        idx._data.freq = None
        assert idx.freq is None

    def test_freq_view_safe(self):
        # 设置一个 DatetimeIndex 的频率不应该改变另一个查看相同数据的 DatetimeIndex 的频率

        # 创建一个日期范围
        dti = date_range("2016-01-01", periods=5)
        dta = dti._data

        # 创建另一个 DatetimeIndex 对象，频率设置为 None
        dti2 = DatetimeIndex(dta)._with_freq(None)
        assert dti2.freq is None

        # 原始对象未被修改
        assert dti.freq == "D"
        assert dta.freq == "D"
```