# `D:\src\scipysrc\pandas\pandas\tests\tools\test_to_time.py`

```
# 导入必要的模块和类
from datetime import time  # 导入 time 类
import locale  # 导入 locale 模块

import numpy as np  # 导入 numpy 库，并简称为 np
import pytest  # 导入 pytest 测试框架

from pandas.compat import PY311  # 从 pandas 包中导入 PY311 兼容性常量

from pandas import Series  # 从 pandas 包中导入 Series 类
import pandas._testing as tm  # 导入 pandas._testing 模块，并简称为 tm
from pandas.core.tools.times import to_time  # 从 pandas 包中导入 to_time 函数

# 标记：只有在非英语环境下可能会失败
# 如果系统区域设置不是 zh_CN 或 it_IT，则测试会失败
fails_on_non_english = pytest.mark.xfail(
    locale.getlocale()[0] in ("zh_CN", "it_IT"),
    reason="fail on a CI build with LC_ALL=zh_CN.utf8/it_IT.utf8",
    strict=False,
)


class TestToTime:
    @pytest.mark.parametrize(
        "time_string",
        [
            "14:15",
            "1415",
            pytest.param("2:15pm", marks=fails_on_non_english),  # 根据 locale 设置标记为 xfail 的参数
            pytest.param("0215pm", marks=fails_on_non_english),  # 根据 locale 设置标记为 xfail 的参数
            "14:15:00",
            "141500",
            pytest.param("2:15:00pm", marks=fails_on_non_english),  # 根据 locale 设置标记为 xfail 的参数
            pytest.param("021500pm", marks=fails_on_non_english),  # 根据 locale 设置标记为 xfail 的参数
            time(14, 15),
        ],
    )
    def test_parsers_time(self, time_string):
        # GH#11818：测试特定的时间字符串转换为 time 对象的情况
        assert to_time(time_string) == time(14, 15)

    def test_odd_format(self):
        new_string = "14.15"
        msg = r"Cannot convert arg \['14\.15'\] to a time"
        if not PY311:
            # 当不是在 Python 3.11 下时，预期会抛出 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                to_time(new_string)
        # 使用指定的格式将字符串转换为 time 对象
        assert to_time(new_string, format="%H.%M") == time(14, 15)

    def test_arraylike(self):
        arg = ["14:15", "20:20"]
        expected_arr = [time(14, 15), time(20, 20)]
        assert to_time(arg) == expected_arr
        assert to_time(arg, format="%H:%M") == expected_arr
        assert to_time(arg, infer_time_format=True) == expected_arr
        assert to_time(arg, format="%I:%M%p", errors="coerce") == [None, None]

        # 预期会抛出 ValueError 异常，因为错误参数不正确
        with pytest.raises(ValueError, match="errors must be"):
            to_time(arg, format="%I:%M%p", errors="ignore")

        msg = "Cannot convert.+to a time with given format"
        # 预期会抛出 ValueError 异常，因为格式不匹配
        with pytest.raises(ValueError, match=msg):
            to_time(arg, format="%I:%M%p", errors="raise")

        # 检查 Series 转换的正确性
        tm.assert_series_equal(
            to_time(Series(arg, name="test")), Series(expected_arr, name="test")
        )

        # 将 numpy 数组转换为列表，并检查结果
        res = to_time(np.array(arg))
        assert isinstance(res, list)
        assert res == expected_arr
```