# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_freq_attr.py`

```
# 导入pytest模块，用于单元测试
import pytest

# 从pandas.compat模块导入PY311
from pandas.compat import PY311

# 从pandas模块中导入offsets和period_range函数
from pandas import (
    offsets,
    period_range,
)

# 导入pandas._testing模块并简称为tm
import pandas._testing as tm

# 定义一个测试类TestFreq
class TestFreq:
    # 定义测试方法test_freq_setter_deprecated
    def test_freq_setter_deprecated(self):
        # 创建一个PeriodIndex对象idx，包含从'2018Q1'开始的四个季度，频率为'Q'
        idx = period_range("2018Q1", periods=4, freq="Q")

        # 在获取频率属性时不应产生警告
        with tm.assert_produces_warning(None):
            idx.freq

        # 尝试设置频率属性时应产生警告
        # 根据PY311的值选择不同的警告消息
        msg = (
            "property 'freq' of 'PeriodArray' object has no setter"
            if PY311
            else "can't set attribute"
        )
        # 使用pytest.raises断言捕获AttributeError，并验证匹配的警告消息
        with pytest.raises(AttributeError, match=msg):
            # 尝试设置idx的freq属性为offsets.Day()，预期会引发AttributeError
            idx.freq = offsets.Day()
```