# `D:\src\scipysrc\pandas\pandas\tests\extension\base\accumulate.py`

```
# 导入 pytest 模块，用于测试框架
import pytest

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 导入 pandas 内部测试模块
import pandas._testing as tm

# 定义一个基础测试类 BaseAccumulateTests
class BaseAccumulateTests:
    """
    Accumulation specific tests. Generally these only
    make sense for numeric/boolean operations.
    """

    # 判断是否支持累积操作的方法
    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # 我们默认假设不支持该数据类型的累积操作；子类应在此处重写。
        return False

    # 检查累积操作的方法
    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool):
        try:
            # 尝试将序列转换为 float64 类型
            alt = ser.astype("float64")
        except TypeError:
            # 捕获可能的类型错误，如 Period 类型无法转换为 float64
            alt = ser.astype(object)

        # 使用 getattr 动态调用序列的指定操作 op_name，并传入 skipna 参数
        result = getattr(ser, op_name)(skipna=skipna)
        # 使用 getattr 动态调用转换后的 alt 序列的指定操作 op_name，并传入 skipna 参数
        expected = getattr(alt, op_name)(skipna=skipna)
        # 使用测试模块 tm 的方法 assert_series_equal 检查结果是否相等，不检查数据类型
        tm.assert_series_equal(result, expected, check_dtype=False)

    # 使用 pytest.mark.parametrize 注解进行参数化测试，参数为 skipna，分别为 True 和 False
    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna):
        # 从参数 all_numeric_accumulations 中获取操作名称
        op_name = all_numeric_accumulations
        # 创建 pd.Series 对象，使用参数 data 作为数据
        ser = pd.Series(data)

        # 如果支持当前序列的累积操作
        if self._supports_accumulation(ser, op_name):
            # 调用 check_accumulate 方法进行累积操作的检查
            self.check_accumulate(ser, op_name, skipna)
        else:
            # 如果不支持当前序列的累积操作，期望引发 NotImplementedError 或 TypeError 异常
            with pytest.raises((NotImplementedError, TypeError)):
                # TODO: 需要 TypeError 的要求，对于绝对不会成功的操作？
                # 使用 getattr 动态调用序列的指定操作 op_name，并传入 skipna 参数
                getattr(ser, op_name)(skipna=skipna)
```