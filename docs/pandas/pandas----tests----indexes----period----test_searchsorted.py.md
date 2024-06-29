# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_searchsorted.py`

```
import numpy as np
import pytest

from pandas._libs.tslibs import IncompatibleFrequency

from pandas import (
    NaT,
    Period,
    PeriodIndex,
)
import pandas._testing as tm

# 定义测试类 TestSearchsorted
class TestSearchsorted:
    # 使用参数化装饰器，参数为 ["D", "2D"]
    @pytest.mark.parametrize("freq", ["D", "2D"])
    # 定义测试方法 test_searchsorted
    def test_searchsorted(self, freq):
        # 创建 PeriodIndex 对象
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq=freq,
        )

        # 创建 Period 对象 p1
        p1 = Period("2014-01-01", freq=freq)
        # 断言 p1 在 pidx 中的位置为 0
        assert pidx.searchsorted(p1) == 0

        # 创建 Period 对象 p2
        p2 = Period("2014-01-04", freq=freq)
        # 断言 p2 在 pidx 中的位置为 3
        assert pidx.searchsorted(p2) == 3

        # 断言 NaT 在 pidx 中的位置为 5
        assert pidx.searchsorted(NaT) == 5

        # 使用 pytest 的异常断言，检查是否抛出 IncompatibleFrequency 异常
        msg = "Input has different freq=h from PeriodArray"
        with pytest.raises(IncompatibleFrequency, match=msg):
            pidx.searchsorted(Period("2014-01-01", freq="h"))

        msg = "Input has different freq=5D from PeriodArray"
        with pytest.raises(IncompatibleFrequency, match=msg):
            pidx.searchsorted(Period("2014-01-01", freq="5D"))

    # 定义测试方法 test_searchsorted_different_argument_classes，参数为 listlike_box
    def test_searchsorted_different_argument_classes(self, listlike_box):
        # 创建 PeriodIndex 对象
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq="D",
        )
        # 调用 searchsorted 方法，传入 listlike_box(pidx) 作为参数
        result = pidx.searchsorted(listlike_box(pidx))
        # 创建期望结果数组
        expected = np.arange(len(pidx), dtype=result.dtype)
        # 使用 pandas._testing 模块的方法进行数组相等性断言
        tm.assert_numpy_array_equal(result, expected)

        # 调用 _data 的 searchsorted 方法，传入 listlike_box(pidx) 作为参数
        result = pidx._data.searchsorted(listlike_box(pidx))
        # 使用 pandas._testing 模块的方法进行数组相等性断言
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法 test_searchsorted_invalid
    def test_searchsorted_invalid(self):
        # 创建 PeriodIndex 对象
        pidx = PeriodIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
            freq="D",
        )

        # 创建一个不兼容的数组 other
        other = np.array([0, 1], dtype=np.int64)

        # 构建异常消息
        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Period', 'NaT', or array of those. Got",
            ]
        )
        # 使用 pytest 的异常断言，检查是否抛出 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(other)

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(other.astype("timedelta64[ns]"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.timedelta64(4))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.timedelta64("NaT", "ms"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.datetime64(4, "ns"))

        with pytest.raises(TypeError, match=msg):
            pidx.searchsorted(np.datetime64("NaT", "ns"))
```