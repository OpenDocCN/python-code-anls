# `D:\src\scipysrc\pandas\pandas\tests\tseries\frequencies\test_frequencies.py`

```
# 导入 pytest 模块，用于测试框架
import pytest

# 从 pandas._libs.tslibs 模块导入 offsets
from pandas._libs.tslibs import offsets

# 从 pandas.tseries.frequencies 模块导入 is_subperiod 和 is_superperiod 函数
from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)

# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "p1,p2,expected",
    [
        # 输入参数验证
        (offsets.MonthEnd(), None, False),        # p1 是 MonthEnd 对象，p2 为 None，期望返回 False
        (offsets.YearEnd(), None, False),         # p1 是 YearEnd 对象，p2 为 None，期望返回 False
        (None, offsets.YearEnd(), False),         # p1 为 None，p2 是 YearEnd 对象，期望返回 False
        (None, offsets.MonthEnd(), False),        # p1 为 None，p2 是 MonthEnd 对象，期望返回 False
        (None, None, False),                      # p1 和 p2 都为 None，期望返回 False
        (offsets.YearEnd(), offsets.MonthEnd(), True),   # p1 是 YearEnd 对象，p2 是 MonthEnd 对象，期望返回 True
        (offsets.Hour(), offsets.Minute(), True),        # p1 是 Hour 对象，p2 是 Minute 对象，期望返回 True
        (offsets.Second(), offsets.Milli(), True),       # p1 是 Second 对象，p2 是 Milli 对象，期望返回 True
        (offsets.Milli(), offsets.Micro(), True),        # p1 是 Milli 对象，p2 是 Micro 对象，期望返回 True
        (offsets.Micro(), offsets.Nano(), True),         # p1 是 Micro 对象，p2 是 Nano 对象，期望返回 True
    ],
)
# 定义测试函数 test_super_sub_symmetry，用于验证 is_superperiod 和 is_subperiod 函数的对称性
def test_super_sub_symmetry(p1, p2, expected):
    # 断言 is_superperiod(p1, p2) 的返回值与 expected 相同
    assert is_superperiod(p1, p2) is expected
    # 断言 is_subperiod(p2, p1) 的返回值与 expected 相同
    assert is_subperiod(p2, p1) is expected
```