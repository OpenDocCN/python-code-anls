# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_is_full.py`

```
# 导入 pytest 库，用于测试
import pytest

# 从 pandas 库中导入 PeriodIndex 类
from pandas import PeriodIndex


# 定义一个测试函数 test_is_full
def test_is_full():
    # 创建一个 PeriodIndex 对象，表示年度频率的索引
    index = PeriodIndex([2005, 2007, 2009], freq="Y")
    # 断言该索引不是完整的（不包含所有年份）
    assert not index.is_full

    # 创建另一个 PeriodIndex 对象，包含连续的年份
    index = PeriodIndex([2005, 2006, 2007], freq="Y")
    # 断言该索引是完整的（包含所有年份）
    assert index.is_full

    # 创建另一个 PeriodIndex 对象，包含重复的年份
    index = PeriodIndex([2005, 2005, 2007], freq="Y")
    # 断言该索引不是完整的（包含重复的年份）
    assert not index.is_full

    # 创建另一个 PeriodIndex 对象，包含连续但有重复的年份
    index = PeriodIndex([2005, 2005, 2006], freq="Y")
    # 断言该索引是完整的（包含所有年份，即使有重复）
    assert index.is_full

    # 创建另一个 PeriodIndex 对象，包含非单调递增的年份
    index = PeriodIndex([2006, 2005, 2005], freq="Y")
    # 断言如果索引非单调递增，则抛出 ValueError 异常
    with pytest.raises(ValueError, match="Index is not monotonic"):
        index.is_full

    # 断言空切片的索引对象是完整的
    assert index[:0].is_full
```