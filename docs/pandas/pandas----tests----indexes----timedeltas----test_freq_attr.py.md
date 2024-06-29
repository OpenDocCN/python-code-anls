# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_freq_attr.py`

```
import pytest  # 导入 pytest 库

from pandas import TimedeltaIndex  # 从 pandas 库导入 TimedeltaIndex 类

from pandas.tseries.offsets import (  # 从 pandas 库的 tseries.offsets 模块导入以下偏移量类
    DateOffset,
    Day,
    Hour,
    MonthEnd,
)


class TestFreq:  # 定义测试类 TestFreq

    @pytest.mark.parametrize("values", [["0 days", "2 days", "4 days"], []])
    @pytest.mark.parametrize("freq", ["2D", Day(2), "48h", Hour(48)])
    def test_freq_setter(self, values, freq):
        # GH#20678
        idx = TimedeltaIndex(values)  # 创建 TimedeltaIndex 对象

        # 可以设置偏移量，必要时将字符串转换为偏移量对象
        idx._data.freq = freq
        assert idx.freq == freq  # 断言确保 freq 已设置成功
        assert isinstance(idx.freq, DateOffset)  # 断言确保 freq 是 DateOffset 类型的对象

        # 可以重置为 None
        idx._data.freq = None
        assert idx.freq is None  # 断言 freq 已重置为 None

    def test_with_freq_empty_requires_tick(self):
        idx = TimedeltaIndex([])  # 创建空的 TimedeltaIndex 对象

        off = MonthEnd(1)  # 创建一个 MonthEnd 偏移量对象
        msg = "TimedeltaArray/Index freq must be a Tick"
        with pytest.raises(TypeError, match=msg):  # 断言捕获 TypeError 异常，并匹配错误信息
            idx._with_freq(off)
        with pytest.raises(TypeError, match=msg):  # 断言捕获 TypeError 异常，并匹配错误信息
            idx._data._with_freq(off)

    def test_freq_setter_errors(self):
        # GH#20678
        idx = TimedeltaIndex(["0 days", "2 days", "4 days"])  # 创建 TimedeltaIndex 对象

        # 使用不兼容的频率设置
        msg = (
            "Inferred frequency 2D from passed values does not conform to "
            "passed frequency 5D"
        )
        with pytest.raises(ValueError, match=msg):  # 断言捕获 ValueError 异常，并匹配错误信息
            idx._data.freq = "5D"

        # 使用非固定频率设置
        msg = r"<2 \* BusinessDays> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):  # 断言捕获 ValueError 异常，并匹配错误信息
            idx._data.freq = "2B"

        # 使用非频率字符串设置
        with pytest.raises(ValueError, match="Invalid frequency"):  # 断言捕获 ValueError 异常，并匹配错误信息
            idx._data.freq = "foo"

    def test_freq_view_safe(self):
        # 设置一个 TimedeltaIndex 的频率不应该影响到另一个视图同样数据的对象的频率

        tdi = TimedeltaIndex(["0 days", "2 days", "4 days"], freq="2D")  # 创建 TimedeltaIndex 对象并设置频率为 "2D"
        tda = tdi._data  # 获取 tdi 的底层数据对象

        tdi2 = TimedeltaIndex(tda)._with_freq(None)  # 创建一个新的 TimedeltaIndex 对象，频率设置为 None
        assert tdi2.freq is None  # 断言新对象的频率为 None

        # 原始对象 tdi 的频率未受影响
        assert tdi.freq == "2D"  # 断言 tdi 对象的频率仍为 "2D"
        assert tda.freq == "2D"  # 断言 tda 对象的频率仍为 "2D"
```