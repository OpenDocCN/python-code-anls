# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_arithmetic.py`

```
# 导入运算符模块
import operator

# 导入 numpy 库并用 np 别名表示
import numpy as np

# 导入 pytest 模块用于测试
import pytest

# 导入 pandas 库并用 pd 别名表示
import pandas as pd

# 导入 pandas 内部测试工具
import pandas._testing as tm

# 导入 pandas 的 FloatingArray 类
from pandas.core.arrays import FloatingArray

# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


# 定义测试函数 test_array_op，使用 pytest.mark.parametrize 装饰器进行参数化测试
@pytest.mark.parametrize(
    "opname, exp",
    [
        ("add", [1.1, 2.2, None, None, 5.5]),
        ("mul", [0.1, 0.4, None, None, 2.5]),
        ("sub", [0.9, 1.8, None, None, 4.5]),
        ("truediv", [10.0, 10.0, None, None, 10.0]),
        ("floordiv", [9.0, 9.0, None, None, 10.0]),
        ("mod", [0.1, 0.2, None, None, 0.0]),
    ],
    ids=["add", "mul", "sub", "div", "floordiv", "mod"],
)
def test_array_op(dtype, opname, exp):
    # 创建包含浮点数和 None 的 pandas 数组 a
    a = pd.array([1.0, 2.0, None, 4.0, 5.0], dtype=dtype)
    
    # 创建包含浮点数和 None 的 pandas 数组 b
    b = pd.array([0.1, 0.2, 0.3, None, 0.5], dtype=dtype)

    # 根据操作名获取对应的运算符函数
    op = getattr(operator, opname)

    # 执行数组操作
    result = op(a, b)
    
    # 创建期望结果的 pandas 数组
    expected = pd.array(exp, dtype=dtype)
    
    # 使用测试工具进行断言，验证结果是否符合期望
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_divide_by_zero，使用 pytest.mark.parametrize 装饰器进行参数化测试
@pytest.mark.parametrize("zero, negative", [(0, False), (0.0, False), (-0.0, True)])
def test_divide_by_zero(dtype, zero, negative):
    # TODO pending NA/NaN discussion
    # https://github.com/pandas-dev/pandas/issues/32265/
    
    # 创建包含整数和 None 的 pandas 数组 a
    a = pd.array([0, 1, -1, None], dtype=dtype)
    
    # 执行除以零操作
    result = a / zero
    
    # 创建期望结果的 FloatingArray 对象
    expected = FloatingArray(
        np.array([np.nan, np.inf, -np.inf, np.nan], dtype=dtype.numpy_dtype),
        np.array([False, False, False, True]),
    )
    
    # 如果是负数除法，需要乘以 -1
    if negative:
        expected *= -1
    
    # 使用测试工具进行断言，验证结果是否符合期望
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_pow_scalar，测试幂运算
def test_pow_scalar(dtype):
    # 创建包含整数和 None 的 pandas 数组 a
    a = pd.array([-1, 0, 1, None, 2], dtype=dtype)
    
    # 执行 a 的 0 次幂运算
    result = a ** 0
    expected = pd.array([1, 1, 1, 1, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行 a 的 1 次幂运算
    result = a ** 1
    expected = pd.array([-1, 0, 1, None, 2], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行 a 的 pd.NA 幂运算
    result = a ** pd.NA
    expected = pd.array([None, None, 1, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行 a 的 np.nan 幂运算
    expected = FloatingArray(
        np.array([np.nan, np.nan, 1, np.nan, np.nan], dtype=dtype.numpy_dtype),
        mask=a._mask,
    )
    tm.assert_extension_array_equal(result, expected)

    # 反向操作，从数组中排除第一个元素
    a = a[1:]  # Can't raise integers to negative powers.

    # 执行 0 的 a 次幂运算
    result = 0 ** a
    expected = pd.array([1, 0, None, 0], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行 1 的 a 次幂运算
    result = 1 ** a
    expected = pd.array([1, 1, 1, 1], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行 pd.NA 的 a 次幂运算
    result = pd.NA ** a
    expected = pd.array([1, None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)

    # 执行 np.nan 的 a 次幂运算
    expected = FloatingArray(
        np.array([1, np.nan, np.nan, np.nan], dtype=dtype.numpy_dtype), mask=a._mask
    )
    tm.assert_extension_array_equal(result, expected)
# 测试指定数据类型的幂运算
def test_pow_array(dtype):
    # 创建包含整数、空值和 None 的 Pandas 扩展数组 a
    a = pd.array([0, 0, 0, 1, 1, 1, None, None, None], dtype=dtype)
    # 创建包含整数、空值和 None 的 Pandas 扩展数组 b
    b = pd.array([0, 1, None, 0, 1, None, 0, 1, None], dtype=dtype)
    # 计算数组 a 的 b 次幂结果
    result = a**b
    # 预期的结果数组，执行相同的幂运算操作
    expected = pd.array([1, 0, None, 1, 1, 1, 1, None, None], dtype=dtype)
    # 使用 Pandas 测试工具比较结果和预期数组
    tm.assert_extension_array_equal(result, expected)


# 测试幂运算中的特定情况处理
def test_rpow_one_to_na():
    # 引用 GitHub 上的问题说明
    # https://github.com/pandas-dev/pandas/issues/22022
    # https://github.com/pandas-dev/pandas/issues/29997
    # 创建包含 NaN 值的 Pandas 扩展数组 arr
    arr = pd.array([np.nan, np.nan], dtype="Float64")
    # 使用 NumPy 数组计算 1.0 和 2.0 的 arr 次幂结果
    result = np.array([1.0, 2.0]) ** arr
    # 预期的结果数组，使用 Pandas 扩展数组执行相同的幂运算操作
    expected = pd.array([1.0, np.nan], dtype="Float64")
    # 使用 Pandas 测试工具比较结果和预期数组
    tm.assert_extension_array_equal(result, expected)


# 测试包含零维 NumPy 数组的算术运算
@pytest.mark.parametrize("other", [0, 0.5])
def test_arith_zero_dim_ndarray(other):
    # 创建包含浮点数和空值的 Pandas 扩展数组 arr
    arr = pd.array([1, None, 2], dtype="Float64")
    # 计算 Pandas 扩展数组 arr 与给定数值 other 的加法结果
    result = arr + np.array(other)
    # 预期的结果数组，执行相同的加法操作
    expected = arr + other
    # 使用 Pandas 测试工具比较结果和预期数组
    tm.assert_equal(result, expected)


# 测试错误值情况下的算术运算
def test_error_invalid_values(data, all_arithmetic_operators, using_infer_string):
    # 从给定数据创建 Pandas Series 对象 s
    s = pd.Series(data)
    # 选择要执行的算术操作
    op = all_arithmetic_operators
    # 获取对应的操作函数 ops
    ops = getattr(s, op)

    # 根据不同的条件设置错误类型的集合 errs
    if using_infer_string:
        import pyarrow as pa

        errs = (TypeError, pa.lib.ArrowNotImplementedError, NotImplementedError)
    else:
        errs = TypeError

    # 错误消息的正则表达式匹配模式
    msg = "|".join(
        [
            r"can only perform ops with numeric values",
            r"FloatingArray cannot perform the operation mod",
            "unsupported operand type",
            "not all arguments converted during string formatting",
            "can't multiply sequence by non-int of type 'float'",
            "ufunc 'subtract' cannot use operands with types dtype",
            r"can only concatenate str \(not \"float\"\) to str",
            "ufunc '.*' not supported for the input types, and the inputs could not",
            "ufunc '.*' did not contain a loop with signature matching types",
            "Concatenation operation is not implemented for NumPy arrays",
            "has no kernel",
            "not implemented",
        ]
    )
    # 使用 pytest 的断言，验证 ops 方法在特定错误情况下是否会引发预期的异常
    with pytest.raises(errs, match=msg):
        ops("foo")
    with pytest.raises(errs, match=msg):
        ops(pd.Timestamp("20180101"))

    # 使用 pytest 的断言，验证 ops 方法在传入无效的数组时是否会引发预期的异常
    with pytest.raises(errs, match=msg):
        ops(pd.Series("foo", index=s.index))
    # 定义消息变量，包含多个错误信息字符串，用 "|" 连接
    msg = "|".join(
        [
            "can only perform ops with numeric values",  # 只能对数值执行操作
            "cannot perform .* with this index type: DatetimeArray",  # 无法对 DatetimeArray 类型的索引执行 .* 操作
            "Addition/subtraction of integers and integer-arrays "
            "with DatetimeArray is no longer supported. *",  # 不再支持使用 DatetimeArray 执行整数和整数数组的加法/减法操作
            "unsupported operand type",  # 不支持的操作数类型
            "not all arguments converted during string formatting",  # 字符串格式化时未转换所有参数
            "can't multiply sequence by non-int of type 'float'",  # 不能将序列乘以非整数类型 'float'
            "ufunc 'subtract' cannot use operands with types dtype",  # ufunc 'subtract' 无法使用具有指定 dtype 的操作数
            (
                "ufunc 'add' cannot use operands with types "
                rf"dtype\('{tm.ENDIAN}M8\[ns\]'\)"
            ),  # ufunc 'add' 不能使用具有特定类型的操作数
            r"ufunc 'add' cannot use operands with types dtype\('float\d{2}'\)",  # ufunc 'add' 不能使用特定 float 类型的操作数
            "cannot subtract DatetimeArray from ndarray",  # 无法从 ndarray 中减去 DatetimeArray
            "has no kernel",  # 没有内核
            "not implemented",  # 未实现
        ]
    )
    # 使用 pytest 检查是否抛出了指定异常 errs，并匹配预定义的错误消息 msg
    with pytest.raises(errs, match=msg):
        # 调用 ops 函数，传入一个包含日期范围的 pandas Series 对象
        ops(pd.Series(pd.date_range("20180101", periods=len(s))))
# Various
# -----------------------------------------------------------------------------
# 定义一个测试函数，用于测试不同类型的算术运算
def test_cross_type_arithmetic():
    # 创建一个包含三列数据的DataFrame，每列数据类型不同
    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, np.nan], dtype="Float64"),
            "B": pd.array([1, np.nan, 3], dtype="Float32"),
            "C": np.array([1, 2, 3], dtype="float64"),
        }
    )

    # 执行列 A 和列 C 的加法运算
    result = df.A + df.C
    # 预期结果是一个包含三个元素的 Series，数据类型为 Float64
    expected = pd.Series([2, 4, np.nan], dtype="Float64")
    tm.assert_series_equal(result, expected)

    # 执行 (列 A + 列 C) * 3 == 12 的逻辑运算
    result = (df.A + df.C) * 3 == 12
    # 预期结果是一个包含三个元素的 Series，数据类型为 boolean
    expected = pd.Series([False, True, None], dtype="boolean")
    tm.assert_series_equal(result, expected)

    # 执行列 A 和列 B 的加法运算
    result = df.A + df.B
    # 预期结果是一个包含三个元素的 Series，数据类型为 Float64
    expected = pd.Series([2, np.nan, np.nan], dtype="Float64")
    tm.assert_series_equal(result, expected)


# 使用 pytest 的参数化功能定义一个测试函数，用于测试一元浮点运算符
@pytest.mark.parametrize(
    "source, neg_target, abs_target",
    [
        ([1.1, 2.2, 3.3], [-1.1, -2.2, -3.3], [1.1, 2.2, 3.3]),
        ([1.1, 2.2, None], [-1.1, -2.2, None], [1.1, 2.2, None]),
        ([-1.1, 0.0, 1.1], [1.1, 0.0, -1.1], [1.1, 0.0, 1.1]),
    ],
)
def test_unary_float_operators(float_ea_dtype, source, neg_target, abs_target):
    # GH38794
    # 获取参数化的浮点数据类型
    dtype = float_ea_dtype
    # 创建一个包含特定数据的 ExtensionArray
    arr = pd.array(source, dtype=dtype)
    # 执行一元负号运算、一元正号运算和绝对值运算
    neg_result, pos_result, abs_result = -arr, +arr, abs(arr)
    # 创建目标的负数版本和绝对值版本的 ExtensionArray
    neg_target = pd.array(neg_target, dtype=dtype)
    abs_target = pd.array(abs_target, dtype=dtype)

    # 断言负数结果与目标负数版本相等
    tm.assert_extension_array_equal(neg_result, neg_target)
    # 断言正数结果与原始数据相等
    tm.assert_extension_array_equal(pos_result, arr)
    # 断言正数结果不与原始数据共享内存
    assert not tm.shares_memory(pos_result, arr)
    # 断言绝对值结果与目标绝对值版本相等
    tm.assert_extension_array_equal(abs_result, abs_target)


# 定义一个测试函数，用于测试位运算
def test_bitwise(dtype):
    # 创建两个包含空值的 ExtensionArray
    left = pd.array([1, None, 3, 4], dtype=dtype)
    right = pd.array([None, 3, 5, 4], dtype=dtype)

    # 使用 pytest 的断言来测试位运算符的行为
    with pytest.raises(TypeError, match="unsupported operand type"):
        left | right
    with pytest.raises(TypeError, match="unsupported operand type"):
        left & right
    with pytest.raises(TypeError, match="unsupported operand type"):
        left ^ right
```