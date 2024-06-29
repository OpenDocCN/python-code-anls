# `D:\src\scipysrc\pandas\pandas\tests\copy_view\index\test_periodindex.py`

```
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入以下模块
    Period,          # 时间段对象
    PeriodIndex,     # 时间段索引对象
    Series,          # 数据序列对象
    period_range,    # 创建时间段范围的函数
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块

# 使用 pytest 的标记功能，忽略特定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting a value on a view:FutureWarning"
)

# 使用 pytest 的参数化装饰器，定义测试函数 test_periodindex，参数为 box
@pytest.mark.parametrize("box", [lambda x: x, PeriodIndex])
def test_periodindex(box):
    # 创建一个日期范围，从 "2019-12-31" 开始，包含 3 个日期，频率为每日
    dt = period_range("2019-12-31", periods=3, freq="D")
    # 将日期范围转换为 Series 对象
    ser = Series(dt)
    # 使用 box 参数将 Series 转换为 PeriodIndex 对象
    idx = box(PeriodIndex(ser))
    # 复制 idx 对象，生成一个深拷贝 expected
    expected = idx.copy(deep=True)
    # 修改 ser 中的第一个元素为 "2020-12-31" 对应的 Period 对象
    ser.iloc[0] = Period("2020-12-31")
    # 断言 idx 与 expected 相等，即测试期望输出与实际输出是否一致
    tm.assert_index_equal(idx, expected)
```