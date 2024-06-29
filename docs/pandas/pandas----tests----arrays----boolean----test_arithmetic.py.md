# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_arithmetic.py`

```
# 导入运算符模块，用于进行操作符操作
import operator

# 导入 numpy 库，并指定别名 np
import numpy as np

# 导入 pytest 库，用于编写和运行测试
import pytest

# 导入 pandas 库，并指定别名 pd
import pandas as pd

# 导入 pandas 内部测试工具模块
import pandas._testing as tm


# 定义一个 pytest fixture，返回一个包含有效值和缺失值的布尔数组
@pytest.fixture
def data():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


# 定义一个 pytest fixture，返回一个包含有效值和缺失值的布尔数组
@pytest.fixture
def left_array():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")


# 定义一个 pytest fixture，返回一个包含有效值和缺失值的布尔数组
@pytest.fixture
def right_array():
    """Fixture returning boolean array with valid and missing values."""
    return pd.array([True, False, None] * 3, dtype="boolean")


# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


# 参数化测试函数，测试加法和乘法操作
@pytest.mark.parametrize(
    "opname, exp",
    [
        ("add", [True, True, None, True, False, None, None, None, None]),
        ("mul", [True, False, None, False, False, None, None, None, None]),
    ],
    ids=["add", "mul"],
)
def test_add_mul(left_array, right_array, opname, exp):
    # 获取操作符函数
    op = getattr(operator, opname)
    # 执行操作符函数，对左右数组进行操作
    result = op(left_array, right_array)
    # 构造期望结果数组
    expected = pd.array(exp, dtype="boolean")
    # 使用测试工具函数验证结果
    tm.assert_extension_array_equal(result, expected)


# 测试减法操作
def test_sub(left_array, right_array):
    # 准备异常消息字符串
    msg = (
        r"numpy boolean subtract, the `-` operator, is (?:deprecated|not supported), "
        r"use the bitwise_xor, the `\^` operator, or the logical_xor function instead\."
    )
    # 检查是否引发预期的 TypeError 异常，并匹配异常消息
    with pytest.raises(TypeError, match=msg):
        left_array - right_array


# 测试除法操作
def test_div(left_array, right_array):
    # 准备异常消息字符串
    msg = "operator '.*' not implemented for bool dtypes"
    # 检查是否引发预期的 NotImplementedError 异常，并匹配异常消息
    with pytest.raises(NotImplementedError, match=msg):
        # 检查是否匹配未掩码 Series 的行为
        pd.Series(left_array._data) / pd.Series(right_array._data)

    # 检查是否引发预期的 NotImplementedError 异常，并匹配异常消息
    with pytest.raises(NotImplementedError, match=msg):
        left_array / right_array


# 参数化测试函数，测试整数除法、求模和乘方操作
@pytest.mark.parametrize(
    "opname",
    [
        "floordiv",
        "mod",
        "pow",
    ],
)
def test_op_int8(left_array, right_array, opname):
    # 获取操作符函数
    op = getattr(operator, opname)
    if opname != "mod":
        # 准备异常消息字符串
        msg = "operator '.*' not implemented for bool dtypes"
        # 检查是否引发预期的 NotImplementedError 异常，并匹配异常消息
        with pytest.raises(NotImplementedError, match=msg):
            result = op(left_array, right_array)
        return
    # 执行操作符函数，对左右数组进行操作
    result = op(left_array, right_array)
    # 准备期望结果数组
    expected = op(left_array.astype("Int8"), right_array.astype("Int8"))
    # 使用测试工具函数验证结果
    tm.assert_extension_array_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


# 测试无效值的错误情况
def test_error_invalid_values(data, all_arithmetic_operators, using_infer_string):
    # invalid ops

    if using_infer_string:
        # 导入 pyarrow 库，并指定别名 pa
        import pyarrow as pa

        # 准备异常类型元组
        err = (TypeError, pa.lib.ArrowNotImplementedError, NotImplementedError)
    else:
        # 准备异常类型
        err = TypeError

    # 获取所有算术运算符
    op = all_arithmetic_operators
    # 创建一个 Pandas Series 对象，使用给定的数据初始化
    s = pd.Series(data)
    
    # 获取 Pandas Series 对象的特定操作函数，例如 'sum'、'mean' 等
    ops = getattr(s, op)

    # 处理无效的标量输入时的错误情况
    msg = (
        "did not contain a loop with signature matching types|"
        "BooleanArray cannot perform the operation|"
        "not supported for the input types, and the inputs could not be safely coerced "
        "to any supported types according to the casting rule ''safe''"
    )
    # 使用 pytest 断言捕获 TypeError 异常，并验证其错误信息是否与 msg 变量匹配
    with pytest.raises(TypeError, match=msg):
        ops("foo")
    
    # 处理无效的数组类型输入时的错误情况
    msg = "|".join(
        [
            r"unsupported operand type\(s\) for",
            "Concatenation operation is not implemented for NumPy arrays",
            "has no kernel",
        ]
    )
    # 使用 pytest 断言捕获特定错误类型 (err)，并验证其错误信息是否与 msg 变量匹配
    with pytest.raises(err, match=msg):
        ops(pd.Timestamp("20180101"))

    # 处理不支持的类似数组输入时的错误情况，排除乘法操作的特殊情况
    if op not in ("__mul__", "__rmul__"):
        # TODO(extension) numpy's mul with object array sees booleans as numbers
        msg = "|".join(
            [
                r"unsupported operand type\(s\) for",
                "can only concatenate str",
                "not all arguments converted during string formatting",
                "has no kernel",
                "not implemented",
            ]
        )
        # 使用 pytest 断言捕获特定错误类型 (err)，并验证其错误信息是否与 msg 变量匹配
        with pytest.raises(err, match=msg):
            ops(pd.Series("foo", index=s.index))
```