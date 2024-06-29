# `D:\src\scipysrc\pandas\pandas\tests\arrays\timedeltas\test_cumulative.py`

```
import pytest  # 导入 pytest 模块

import pandas._testing as tm  # 导入 pandas 内部测试模块的别名 tm
from pandas.core.arrays import TimedeltaArray  # 导入 pandas 时间增量数组模块


class TestAccumulator:
    def test_accumulators_disallowed(self):
        # GH#50297: 标识 GitHub 问题编号
        arr = TimedeltaArray._from_sequence(["1D", "2D"], dtype="m8[ns]")  # 使用 TimedeltaArray 创建时间增量数组
        with pytest.raises(TypeError, match="cumprod not supported"):  # 使用 pytest 断言捕获 TypeError 异常，匹配异常信息字符串
            arr._accumulate("cumprod")  # 调用时间增量数组的累积方法 "_accumulate" 并传递 "cumprod" 参数

    def test_cumsum(self, unit):
        # GH#50297: 标识 GitHub 问题编号
        dtype = f"m8[{unit}]"  # 根据传入的单位参数生成时间增量数组的 dtype
        arr = TimedeltaArray._from_sequence(["1D", "2D"], dtype=dtype)  # 使用 TimedeltaArray 创建时间增量数组
        result = arr._accumulate("cumsum")  # 调用时间增量数组的累积方法 "_accumulate" 并传递 "cumsum" 参数
        expected = TimedeltaArray._from_sequence(["1D", "3D"], dtype=dtype)  # 生成预期的时间增量数组
        tm.assert_timedelta_array_equal(result, expected)  # 使用 pandas 测试模块 tm 的方法比较结果与预期是否相等
```