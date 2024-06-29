# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\methods\test_repeat.py`

```
import numpy as np  # 导入 numpy 库，用于处理数值数组
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入 PeriodIndex 和 period_range 函数
    PeriodIndex,
    period_range,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块 tm


class TestRepeat:
    @pytest.mark.parametrize("use_numpy", [True, False])  # 使用 pytest 参数化装饰器定义参数 use_numpy，取值为 True 和 False
    @pytest.mark.parametrize(  # 使用 pytest 参数化装饰器定义参数 index，分别是不同的 PeriodIndex 对象
        "index",
        [
            period_range("2000-01-01", periods=3, freq="D"),  # 生成一个每日频率的 PeriodIndex，从 '2000-01-01' 开始，共 3 个周期
            period_range("2001-01-01", periods=3, freq="2D"),  # 生成一个每两日频率的 PeriodIndex，从 '2001-01-01' 开始，共 3 个周期
            PeriodIndex(["2001-01", "NaT", "2003-01"], freq="M"),  # 创建一个月度频率的 PeriodIndex，包含指定的月份
        ],
    )
    def test_repeat_freqstr(self, index, use_numpy):
        # GH#10183
        # 创建期望的 PeriodIndex，通过迭代每个周期元素重复 3 次得到
        expected = PeriodIndex([per for per in index for _ in range(3)])
        # 根据 use_numpy 参数选择使用 numpy 的 repeat 函数或 pandas 的 repeat 方法进行重复操作
        result = np.repeat(index, 3) if use_numpy else index.repeat(3)
        # 使用 pandas._testing 模块的 assert_index_equal 函数断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)
        # 断言 result 的频率字符串与 index 的频率字符串相同
        assert result.freqstr == index.freqstr
```