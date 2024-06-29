# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_searchsorted.py`

```
import numpy as np
import pytest

from pandas import (
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm

class TestSearchSorted:
    def test_searchsorted_different_argument_classes(self, listlike_box):
        # 创建一个 TimedeltaIndex 对象，包含三个 Timedelta 对象
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        # 对 TimedeltaIndex 调用 searchsorted 方法，寻找 listlike_box(idx) 在 idx 中的插入点
        result = idx.searchsorted(listlike_box(idx))
        # 创建一个预期结果数组，包含与 idx 相同长度的连续整数，与结果数组的数据类型相同
        expected = np.arange(len(idx), dtype=result.dtype)
        # 使用 pandas._testing 模块的方法，比较两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 调用 TimedeltaIndex 对象的 _data 属性的 searchsorted 方法
        result = idx._data.searchsorted(listlike_box(idx))
        # 使用 pandas._testing 模块的方法，比较两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
    )
    def test_searchsorted_invalid_argument_dtype(self, arg):
        # 创建一个 TimedeltaIndex 对象，包含三个 Timedelta 对象
        idx = TimedeltaIndex(["1 day", "2 days", "3 days"])
        # 定义错误消息的预期字符串
        msg = "value should be a 'Timedelta', 'NaT', or array of those. Got"
        # 使用 pytest 的断言上下文，检查调用 searchsorted 方法时传递的参数类型错误是否引发 TypeError 异常，并检查错误消息是否包含预期的字符串
        with pytest.raises(TypeError, match=msg):
            idx.searchsorted(arg)
```