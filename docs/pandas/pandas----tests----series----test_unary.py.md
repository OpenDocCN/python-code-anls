# `D:\src\scipysrc\pandas\pandas\tests\series\test_unary.py`

```
# 导入 pytest 库，用于测试和断言
import pytest

# 导入 Series 类，它是 pandas 库中的一种数据结构，用于存储一维数组的数据
from pandas import Series

# 导入 pandas 测试模块，包含一些用于测试的工具和函数
import pandas._testing as tm

# 定义一个测试类 TestSeriesUnaryOps，用于测试 Series 类的一元操作符
class TestSeriesUnaryOps:
    
    # 测试负号操作 __neg__
    def test_neg(self):
        # 创建一个浮点数 Series 对象，范围是 [0, 1, 2, 3, 4]，名称为 "series"
        ser = Series(range(5), dtype="float64", name="series")
        # 断言负号操作后的 Series 应与原 Series 乘以 -1 的结果相等
        tm.assert_series_equal(-ser, -1 * ser)

    # 测试按位取反操作 __invert__
    def test_invert(self):
        # 创建一个浮点数 Series 对象，范围是 [0, 1, 2, 3, 4]，名称为 "series"
        ser = Series(range(5), dtype="float64", name="series")
        # 断言按位取反操作后的 Series 应与逻辑取反操作后的结果相等
        tm.assert_series_equal(-(ser < 0), ~(ser < 0))

    # 参数化测试函数，测试所有数值型的一元操作符
    @pytest.mark.parametrize(
        "source, neg_target, abs_target",
        [
            ([1, 2, 3], [-1, -2, -3], [1, 2, 3]),  # 整数列表测试，负号操作和绝对值操作的目标值
            ([1, 2, None], [-1, -2, None], [1, 2, None]),  # 含有 None 的列表测试，负号操作的目标值
        ],
    )
    def test_all_numeric_unary_operators(
        self, any_numeric_ea_dtype, source, neg_target, abs_target
    ):
        # GH38794
        # 获取任意数值类型的 dtype
        dtype = any_numeric_ea_dtype
        # 创建一个 Series 对象，使用参数提供的数据源和 dtype
        ser = Series(source, dtype=dtype)
        # 执行负号操作、正号操作和绝对值操作
        neg_result, pos_result, abs_result = -ser, +ser, abs(ser)
        # 根据 dtype 的类型重新定义负号操作的目标值
        if dtype.startswith("U"):
            neg_target = -Series(source, dtype=dtype)
        else:
            neg_target = Series(neg_target, dtype=dtype)

        abs_target = Series(abs_target, dtype=dtype)

        # 断言负号操作、正号操作和绝对值操作的结果与目标值相等
        tm.assert_series_equal(neg_result, neg_target)
        tm.assert_series_equal(pos_result, ser)
        tm.assert_series_equal(abs_result, abs_target)

    # 参数化测试函数，测试浮点数的一元操作符对缺失值的处理
    @pytest.mark.parametrize("op", ["__neg__", "__abs__"])
    def test_unary_float_op_mask(self, float_ea_dtype, op):
        # 获取浮点数的 dtype
        dtype = float_ea_dtype
        # 创建一个浮点数 Series 对象，数据为 [1.1, 2.2, 3.3]，使用提供的 dtype
        ser = Series([1.1, 2.2, 3.3], dtype=dtype)
        # 执行指定的一元操作符
        result = getattr(ser, op)()
        # 深拷贝结果作为目标值
        target = result.copy(deep=True)
        # 将 Series 对象的第一个元素设置为 None（缺失值）
        ser[0] = None
        # 断言操作后的结果与目标值相等
        tm.assert_series_equal(result, target)
```