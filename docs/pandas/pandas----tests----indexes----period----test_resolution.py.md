# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_resolution.py`

```
# 导入 pytest 模块，用于测试框架
import pytest

# 导入 pandas 库并使用 pd 别名
import pandas as pd

# 定义测试类 TestResolution
class TestResolution:
    # 使用 pytest 的 parametrize 装饰器，对 test_resolution 方法进行参数化测试
    @pytest.mark.parametrize(
        "freq,expected",
        [
            ("Y", "year"),
            ("Q", "quarter"),
            ("M", "month"),
            ("D", "day"),
            ("h", "hour"),
            ("min", "minute"),
            ("s", "second"),
            ("ms", "millisecond"),
            ("us", "microsecond"),
        ],
    )
    # 定义测试方法 test_resolution，接受 freq 和 expected 参数
    def test_resolution(self, freq, expected):
        # 使用 pd.period_range 生成时间区间索引 idx，从 "2013-04-01" 开始，包含 30 个时间点，频率由 freq 参数指定
        idx = pd.period_range(start="2013-04-01", periods=30, freq=freq)
        # 断言 idx 的分辨率（resolution）等于预期的字符串 expected
        assert idx.resolution == expected
```