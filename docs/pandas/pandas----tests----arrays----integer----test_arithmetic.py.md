# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_arithmetic.py`

```
# 导入运算符模块，用于动态获取运算符函数
import operator

# 导入 numpy 库，并重命名为 np
import numpy as np

# 导入 pytest 库，用于测试
import pytest

# 导入 pandas 库，并重命名为 pd
import pandas as pd

# 导入 pandas 内部测试模块
import pandas._testing as tm

# 从 pandas 核心模块中导入操作模块 ops
from pandas.core import ops

# 从 pandas 核心数组模块中导入浮点数数组类型
from pandas.core.arrays import FloatingArray

# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


# 使用 pytest 的 parametrize 装饰器定义参数化测试
@pytest.mark.parametrize(
    "opname, exp",
    [("add", [1, 3, None, None, 9]), ("mul", [0, 2, None, None, 20])],
    ids=["add", "mul"],
)
# 定义测试函数 test_add_mul，接收 dtype、opname 和 exp 参数
def test_add_mul(dtype, opname, exp):
    # 创建包含 None 值的扩展数组 a
    a = pd.array([0, 1, None, 3, 4], dtype=dtype)
    # 创建包含 None 值的扩展数组 b
    b = pd.array([1, 2, 3, None, 5], dtype=dtype)

    # 创建预期结果的扩展数组 expected
    expected = pd.array(exp, dtype=dtype)

    # 使用 getattr 函数动态获取运算符函数，opname 可能是 'add' 或 'mul'
    op = getattr(operator, opname)
    # 对数组 a 和 b 执行运算
    result = op(a, b)
    # 使用扩展测试工具函数 assert_extension_array_equal 检查结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 使用 ops 模块中的 'r' + opname 动态获取反向运算函数
    op = getattr(ops, "r" + opname)
    # 对数组 a 和 b 执行反向运算
    result = op(a, b)
    # 使用扩展测试工具函数 assert_extension_array_equal 检查结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_sub，接收 dtype 参数
def test_sub(dtype):
    # 创建包含 None 值的扩展数组 a
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    # 创建包含 None 值的扩展数组 b
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    # 计算数组 a 和 b 的减法结果
    result = a - b
    # 创建预期结果的扩展数组 expected
    expected = pd.array([1, 1, None, None, 1], dtype=dtype)
    # 使用扩展测试工具函数 assert_extension_array_equal 检查结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_div，接收 dtype 参数
def test_div(dtype):
    # 创建包含 None 值的扩展数组 a
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    # 创建包含 None 值的扩展数组 b
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    # 计算数组 a 除以 b 的结果
    result = a / b
    # 创建预期结果的扩展数组 expected
    expected = pd.array([np.inf, 2, None, None, 1.25], dtype="Float64")
    # 使用扩展测试工具函数 assert_extension_array_equal 检查结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器定义参数化测试
@pytest.mark.parametrize("zero, negative", [(0, False), (0.0, False), (-0.0, True)])
# 定义测试函数 test_divide_by_zero，接收 zero 和 negative 参数
def test_divide_by_zero(zero, negative):
    # 创建包含 None 值的整数扩展数组 a
    a = pd.array([0, 1, -1, None], dtype="Int64")
    # 计算数组 a 除以 zero 的结果
    result = a / zero
    # 创建预期结果的浮点数数组 FloatingArray
    expected = FloatingArray(
        np.array([np.nan, np.inf, -np.inf, 1], dtype="float64"),
        np.array([False, False, False, True]),
    )
    # 如果 negative 为 True，则将 expected 元素取反
    if negative:
        expected *= -1
    # 使用扩展测试工具函数 assert_extension_array_equal 检查结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_floordiv，接收 dtype 参数
def test_floordiv(dtype):
    # 创建包含 None 值的扩展数组 a
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    # 创建包含 None 值的扩展数组 b
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    # 计算数组 a 除以 b 的整数部分结果
    result = a // b
    # 创建预期结果的扩展数组 expected
    expected = pd.array([0, 2, None, None, 1], dtype=dtype)
    # 使用扩展测试工具函数 assert_extension_array_equal 检查结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 定义测试函数 test_floordiv_by_int_zero_no_mask，接收 any_int_ea_dtype 参数
def test_floordiv_by_int_zero_no_mask(any_int_ea_dtype):
    # GH 48223: Aligns with non-masked floordiv
    # but differs from numpy
    # https://github.com/pandas-dev/pandas/issues/30188#issuecomment-564452740
    # 创建包含整数的 Series 对象 ser
    ser = pd.Series([0, 1], dtype=any_int_ea_dtype)
    # 计算 1 除以 ser 的结果
    result = 1 // ser
    # 创建预期结果的浮点数 Series 对象 expected
    expected = pd.Series([np.inf, 1.0], dtype="Float64")
    # 使用扩展测试工具函数 assert_series_equal 检查结果是否与预期相等
    tm.assert_series_equal(result, expected)

    # 将 ser 转换为非空的 Series 对象 ser_non_nullable
    ser_non_nullable = ser.astype(ser.dtype.numpy_dtype)
    # 将 expected 转换为 np.float64 类型的浮点数 Series 对象
    expected = expected.astype(np.float64)
    # 使用扩展测试工具函数 assert_series_equal 检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_mod，接收 dtype 参数
def test_mod(dtype):
    # 创建包含 None 值的扩展数组 a
    a = pd.array([1, 2, 3, None, 5], dtype=dtype)
    # 创建包含 None 值的扩展数组 b
    b = pd.array([0, 1, None, 3, 4], dtype=dtype)

    # 计算数组 a 对 b 的取模结果
    result = a % b
    # 创建一个 Pandas 的 ExtensionArray，其中包含整数和空值的组合
    expected = pd.array([0, 0, None, None, 1], dtype=dtype)
    # 使用测试工具函数来比较计算得到的结果和预期结果是否相等
    tm.assert_extension_array_equal(result, expected)
# 定义一个测试函数，用于测试在 Pandas 扩展数组上的指数运算
def test_pow_scalar():
    # 创建一个包含整数和缺失值的 Pandas 扩展数组
    a = pd.array([-1, 0, 1, None, 2], dtype="Int64")
    # 执行指数运算 a**0，并将结果赋给 result
    result = a**0
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([1, 1, 1, 1, 1], dtype="Int64")
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 执行指数运算 a**1，并将结果赋给 result
    result = a**1
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([-1, 0, 1, None, 2], dtype="Int64")
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 执行指数运算 a**pd.NA，并将结果赋给 result
    result = a**pd.NA
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([None, None, 1, None, None], dtype="Int64")
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 执行指数运算 a**np.nan，并将结果赋给 result
    result = a**np.nan
    # 创建预期结果的浮点数数组，包含 NaN 值
    expected = FloatingArray(
        np.array([np.nan, np.nan, 1, np.nan, np.nan], dtype="float64"),
        np.array([False, False, False, True, False]),
    )
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 反转数组 a，去掉第一个元素，因为整数不能作为负指数的底数
    a = a[1:]

    # 执行指数运算 0**a，并将结果赋给 result
    result = 0**a
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([1, 0, None, 0], dtype="Int64")
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 执行指数运算 1**a，并将结果赋给 result
    result = 1**a
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([1, 1, 1, 1], dtype="Int64")
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 执行指数运算 pd.NA**a，并将结果赋给 result
    result = pd.NA**a
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([1, None, None, None], dtype="Int64")
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)

    # 执行指数运算 np.nan**a，并将结果赋给 result
    result = np.nan**a
    # 创建预期结果的浮点数数组，包含 NaN 值
    expected = FloatingArray(
        np.array([1, np.nan, np.nan, np.nan], dtype="float64"),
        np.array([False, False, True, False]),
    )
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 定义一个测试函数，用于测试在 Pandas 扩展数组上的数组间指数运算
def test_pow_array():
    # 创建两个包含整数和缺失值的 Pandas 扩展数组
    a = pd.array([0, 0, 0, 1, 1, 1, None, None, None])
    b = pd.array([0, 1, None, 0, 1, None, 0, 1, None])
    # 执行指数运算 a**b，并将结果赋给 result
    result = a**b
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([1, 0, None, 1, 1, 1, 1, None, None])
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 定义一个测试函数，用于测试在 Pandas 扩展数组和 NumPy 数组之间的指数运算
def test_rpow_one_to_na():
    # 创建一个包含 NaN 值的 Pandas 扩展数组
    arr = pd.array([np.nan, np.nan], dtype="Int64")
    # 执行指数运算 np.array([1.0, 2.0]) ** arr，并将结果赋给 result
    result = np.array([1.0, 2.0]) ** arr
    # 创建预期结果的 Pandas 扩展数组
    expected = pd.array([1.0, np.nan], dtype="Float64")
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_extension_array_equal(result, expected)


# 测试 numpy 零维 ndarray 和 Pandas 扩展数组之间的加法操作
@pytest.mark.parametrize("other", [0, 0.5])
def test_numpy_zero_dim_ndarray(other):
    # 创建一个包含整数和缺失值的 Pandas 扩展数组
    arr = pd.array([1, None, 2])
    # 执行加法操作 arr + np.array(other)，并将结果赋给 result
    result = arr + np.array(other)
    # 创建预期结果的 Pandas 扩展数组
    expected = arr + other
    # 使用测试工具函数验证结果是否与预期相等
    tm.assert_equal(result, expected)


# 测试通用特性 / 错误情况
# -----------------------------------------------------------------------------
def test_error_invalid_values(data, all_arithmetic_operators, using_infer_string):
    # 从输入数据创建一个 Pandas Series
    s = pd.Series(data)
    # 获取 Pandas Series 上指定算术操作的函数
    ops = getattr(s, all_arithmetic_operators)

    if using_infer_string:
        # 如果使用 'infer' 字符串，导入 pyarrow 库
        import pyarrow as pa
        # 定义可能出现的错误类型列表
        errs = (TypeError, pa.lib.ArrowNotImplementedError, NotImplementedError)
    else:
        # 否则，错误类型为 TypeError
        errs = TypeError

    # 无效的标量值
    # 构建包含多条错误消息的字符串，用竖线分隔
    msg = "|".join(
        [
            r"can only perform ops with numeric values",  # 只能对数值进行操作
            r"IntegerArray cannot perform the operation mod",  # IntegerArray 不能执行模运算操作
            r"unsupported operand type",  # 不支持的操作数类型
            r"can only concatenate str \(not \"int\"\) to str",  # 只能将字符串连接（不能将整数连接到字符串）
            "not all arguments converted during string",  # 字符串转换时未转换所有参数
            "ufunc '.*' not supported for the input types, and the inputs could not",  # 输入类型不支持 ufunc '.*'，且无法处理输入
            "ufunc '.*' did not contain a loop with signature matching types",  # ufunc '.*' 中没有符合类型的循环签名
            "Addition/subtraction of integers and integer-arrays with Timestamp",  # 不能使用 Timestamp 进行整数和整数数组的加减操作
            "has no kernel",  # 没有核心
            "not implemented",  # 未实现
            "The 'out' kwarg is necessary. Use numpy.strings.multiply without it.",  # 'out' 参数是必需的。在没有它的情况下使用 numpy.strings.multiply
        ]
    )
    # 使用 pytest 来断言操作 'ops("foo")' 会引发 'errs' 异常，异常消息匹配 'msg' 中的任一字符串
    with pytest.raises(errs, match=msg):
        ops("foo")
    # 使用 pytest 来断言操作 'ops(pd.Timestamp("20180101"))' 会引发 'errs' 异常，异常消息匹配 'msg' 中的任一字符串
    with pytest.raises(errs, match=msg):
        ops(pd.Timestamp("20180101"))

    # 创建一个包含字符串 "foo" 的 Pandas Series，索引与 s 相同
    str_ser = pd.Series("foo", index=s.index)
    # 如果所有算术运算符在 "__mul__" 和 "__rmul__" 中，并且不使用 'using_infer_string'，则执行以下操作
    if (
        all_arithmetic_operators
        in [
            "__mul__",
            "__rmul__",
        ]
        and not using_infer_string
    ):
        # 对 str_ser 执行操作 'ops'
        res = ops(str_ser)
        # 创建一个预期结果的 Pandas Series，内容为 ["foo" * x for x in data]，索引与 s 相同
        expected = pd.Series(["foo" * x for x in data], index=s.index)
        # 将预期结果中的 NaN 值填充为 np.nan
        expected = expected.fillna(np.nan)
        # TODO: 为了使测试通过，使用 fillna，但使用 pd.NA 的预期似乎比 np.nan 更正确。
        #  保持测试的一致性。
        # 使用 Pandas 测试模块来比较结果 'res' 和预期值 'expected'
        tm.assert_series_equal(res, expected)
    else:
        # 否则，使用 pytest 来断言操作 'ops(str_ser)' 会引发 'errs' 异常，异常消息匹配 'msg' 中的任一字符串
        with pytest.raises(errs, match=msg):
            ops(str_ser)

    # 构建包含多条错误消息的字符串，用竖线分隔
    msg = "|".join(
        [
            "can only perform ops with numeric values",  # 只能对数值进行操作
            "cannot perform .* with this index type: DatetimeArray",  # 无法使用此索引类型 DatetimeArray 执行 .*
            "Addition/subtraction of integers and integer-arrays with DatetimeArray is no longer supported. *",  # 不再支持整数和整数数组与 DatetimeArray 的加减操作
            "unsupported operand type",  # 不支持的操作数类型
            r"can only concatenate str \(not \"int\"\) to str",  # 只能将字符串连接（不能将整数连接到字符串）
            "not all arguments converted during string",  # 字符串转换时未转换所有参数
            "cannot subtract DatetimeArray from ndarray",  # 不能从 ndarray 减去 DatetimeArray
            "has no kernel",  # 没有核心
            "not implemented",  # 未实现
        ]
    )
    # 使用 pytest 来断言操作 'ops(pd.Series(pd.date_range("20180101", periods=len(s))))' 会引发 'errs' 异常，异常消息匹配 'msg' 中的任一字符串
    with pytest.raises(errs, match=msg):
        ops(pd.Series(pd.date_range("20180101", periods=len(s))))
# Various
# -----------------------------------------------------------------------------

# 定义测试函数，用于测试单个数值与 Series 的算术运算
def test_arith_coerce_scalar(data, all_arithmetic_operators):
    # 从给定的算术运算名称获取操作函数
    op = tm.get_op_from_name(all_arithmetic_operators)
    # 将输入数据转换为 Series
    s = pd.Series(data)
    # 设置另一个操作数为浮点数 0.01
    other = 0.01

    # 执行算术运算
    result = op(s, other)
    # 预期结果是将 Series 转换为 float 类型后执行相同操作
    expected = op(s.astype(float), other)
    # 将预期结果转换为 Float64 类型
    expected = expected.astype("Float64")

    # 对于特定的算术运算 '__rmod__'，处理结果中 NaN 值的情况
    if all_arithmetic_operators == "__rmod__":
        # 找出原始 Series 中非 NA 且为零的位置，并解除掩码
        mask = (s == 0).fillna(False).to_numpy(bool)
        expected.array._mask[mask] = False

    # 断言结果的一致性
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("other", [1.0, np.array(1.0)])
# 测试不同的算术运算，验证结果是否为浮点数
def test_arithmetic_conversion(all_arithmetic_operators, other):
    # 根据算术运算名称获取操作函数
    op = tm.get_op_from_name(all_arithmetic_operators)

    # 创建一个整型 Series
    s = pd.Series([1, 2, 3], dtype="Int64")
    # 执行算术运算
    result = op(s, other)
    # 断言结果的数据类型为 Float64
    assert result.dtype == "Float64"


# 测试不同类型之间的算术运算
def test_cross_type_arithmetic():
    # 创建包含不同数据类型列的 DataFrame
    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, np.nan], dtype="Int64"),
            "B": pd.Series([1, np.nan, 3], dtype="UInt8"),
            "C": [1, 2, 3],
        }
    )

    # 测试整型列与整型列的加法
    result = df.A + df.C
    expected = pd.Series([2, 4, np.nan], dtype="Int64")
    tm.assert_series_equal(result, expected)

    # 测试加法后乘法是否正确计算
    result = (df.A + df.C) * 3 == 12
    expected = pd.Series([False, True, None], dtype="boolean")
    tm.assert_series_equal(result, expected)

    # 测试整型列与无符号整型列的加法
    result = df.A + df.B
    expected = pd.Series([2, np.nan, np.nan], dtype="Int64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("op", ["mean"])
# 测试数据缩减操作，验证是否总是返回浮点数
def test_reduce_to_float(op):
    # 创建包含不同数据类型的 DataFrame
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([1, None, 3], dtype="Int64"),
        }
    )

    # 测试操作 op 在列 C 上的结果类型
    result = getattr(df.C, op)()
    assert isinstance(result, float)

    # 测试 groupby 后操作 op 的结果
    result = getattr(df.groupby("A"), op)()

    # 预期的 DataFrame 结果
    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([1, 3], dtype="Float64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "source, neg_target, abs_target",
    [
        ([1, 2, 3], [-1, -2, -3], [1, 2, 3]),
        ([1, 2, None], [-1, -2, None], [1, 2, None]),
        ([-1, 0, 1], [1, 0, -1], [1, 0, 1]),
    ],
)
# 测试一元整数操作符的行为
def test_unary_int_operators(any_signed_int_ea_dtype, source, neg_target, abs_target):
    # 获取任意有符号整数类型
    dtype = any_signed_int_ea_dtype
    # 创建一个带有给定数据的扩展数组
    arr = pd.array(source, dtype=dtype)
    # 执行一元负数、正数和绝对值操作
    neg_result, pos_result, abs_result = -arr, +arr, abs(arr)
    # 创建预期的一元负数和绝对值的结果
    neg_target = pd.array(neg_target, dtype=dtype)
    abs_target = pd.array(abs_target, dtype=dtype)

    # 断言一元负数的结果是否与预期一致
    tm.assert_extension_array_equal(neg_result, neg_target)
    # 使用测试工具函数来比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(pos_result, arr)
    
    # 确认两个数组在内存中没有共享的部分，即它们是完全独立的
    assert not tm.shares_memory(pos_result, arr)
    
    # 使用测试工具函数来比较两个扩展数组的绝对值是否相等
    tm.assert_extension_array_equal(abs_result, abs_target)
# 测试函数：测试将pd.NA与包含大量元素的Series相乘的行为
def test_values_multiplying_large_series_by_NA():
    # 标识符 GH#33701，可能是指GitHub上某个问题或需求编号

    # 使用pd.NA乘以一个包含10001个零的NumPy数组构建Series
    result = pd.NA * pd.Series(np.zeros(10001))
    # 构建期望的Series，其中所有元素都是pd.NA
    expected = pd.Series([pd.NA] * 10001)

    # 使用测试工具比较result和expected两个Series是否相等
    tm.assert_series_equal(result, expected)


# 测试函数：测试位运算操作
def test_bitwise(dtype):
    # 创建左右两个带有None的pandas数组
    left = pd.array([1, None, 3, 4], dtype=dtype)
    right = pd.array([None, 3, 5, 4], dtype=dtype)

    # 进行位或运算
    result = left | right
    # 期望的结果数组，使用位或运算符
    expected = pd.array([None, None, 3 | 5, 4 | 4], dtype=dtype)
    # 使用测试工具比较result和expected两个扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 进行位与运算
    result = left & right
    # 期望的结果数组，使用位与运算符
    expected = pd.array([None, None, 3 & 5, 4 & 4], dtype=dtype)
    # 使用测试工具比较result和expected两个扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 进行位异或运算
    result = left ^ right
    # 期望的结果数组，使用位异或运算符
    expected = pd.array([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    # 使用测试工具比较result和expected两个扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # TODO: 当与布尔值操作时的期望行为？推迟处理？

    # 将right数组转换为Float64类型
    floats = right.astype("Float64")
    # 检查在与布尔类型操作时是否抛出TypeError异常
    with pytest.raises(TypeError, match="unsupported operand type"):
        left | floats
    with pytest.raises(TypeError, match="unsupported operand type"):
        left & floats
    with pytest.raises(TypeError, match="unsupported operand type"):
        left ^ floats
```