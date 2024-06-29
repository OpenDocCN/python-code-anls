# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_tzconversion.py`

```
# 导入所需的库
import numpy as np
import pytest  # 导入 pytest 库，用于编写和运行测试
import pytz  # 导入 pytz 库，用于处理时区

# 从 pandas 库中导入时区转换相关的函数
from pandas._libs.tslibs.tzconversion import tz_localize_to_utc


class TestTZLocalizeToUTC:
    def test_tz_localize_to_utc_ambiguous_infer(self):
        # val 是一个时间戳，当转换为 US/Eastern 时可能存在歧义
        val = 1_320_541_200_000_000_000
        # 创建包含多个时间戳的 numpy 数组 vals，用于测试
        vals = np.array([val, val - 1, val], dtype=np.int64)

        # 使用 pytest 框架检查是否会引发 AmbiguousTimeError 异常，并验证异常消息
        with pytest.raises(pytz.AmbiguousTimeError, match="2011-11-06 01:00:00"):
            # 调用 tz_localize_to_utc 函数进行时区本地化和转换，期望抛出 AmbiguousTimeError 异常
            tz_localize_to_utc(vals, pytz.timezone("US/Eastern"), ambiguous="infer")

        # 继续使用 pytest 检查处理边界情况下是否正确引发异常
        with pytest.raises(pytz.AmbiguousTimeError, match="are no repeated times"):
            tz_localize_to_utc(vals[:1], pytz.timezone("US/Eastern"), ambiguous="infer")

        # 修改 vals 数组以引发另一种 AmbiguousTimeError 异常，验证异常消息
        vals[1] += 1
        msg = "There are 2 dst switches when there should only be 1"
        with pytest.raises(pytz.AmbiguousTimeError, match=msg):
            tz_localize_to_utc(vals, pytz.timezone("US/Eastern"), ambiguous="infer")
```