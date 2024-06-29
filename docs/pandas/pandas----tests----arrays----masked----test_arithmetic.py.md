# `D:\src\scipysrc\pandas\pandas\tests\arrays\masked\test_arithmetic.py`

```
# 从未来模块导入注解支持
from __future__ import annotations

# 导入必要的类型提示
from typing import Any

# 导入 NumPy 库，并使用别名 np
import numpy as np

# 导入 pytest 库，用于测试
import pytest

# 导入 Pandas 库，并使用别名 pd
import pandas as pd

# 导入 Pandas 内部测试工具模块，并使用别名 tm
import pandas._testing as tm

# integer dtypes
# 创建包含各种整数数据类型的 Pandas 数组列表
arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_INT_EA_DTYPES]

# 创建与数组数量相同的标量列表，每个标量为整数 2
scalars: list[Any] = [2] * len(arrays)

# floating dtypes
# 将各种浮点数据类型的 Pandas 数组添加到数组列表中
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]

# 添加两个浮点数标量 0.2 到标量列表中
scalars += [0.2, 0.2]

# boolean
# 添加包含布尔值的 Pandas 数组到数组列表中
arrays += [pd.array([True, False, True, None], dtype="boolean")]

# 添加布尔值 False 到标量列表中
scalars += [False]

# Fixture 参数化，返回元组 (array, scalar)
@pytest.fixture(params=zip(arrays, scalars), ids=[a.dtype.name for a in arrays])
def data(request):
    """Fixture returning parametrized (array, scalar) tuple.

    Used to test equivalence of scalars, numpy arrays with array ops, and the
    equivalence of DataFrame and Series ops.
    """
    return request.param


# 检查是否需要跳过测试
def check_skip(data, op_name):
    if isinstance(data.dtype, pd.BooleanDtype) and "sub" in op_name:
        pytest.skip("subtract not implemented for boolean")


# 检查布尔操作是否未实现
def is_bool_not_implemented(data, op_name):
    # 匹配未屏蔽的行为
    return data.dtype.kind == "b" and op_name.strip("_").lstrip("r") in [
        "pow",
        "truediv",
        "floordiv",
    ]


# Test equivalence of scalars, numpy arrays with array ops
# -----------------------------------------------------------------------------


# 测试标量与数组操作的等效性
def test_array_scalar_like_equivalence(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    scalar_array = pd.array([scalar] * len(data), dtype=data.dtype)

    # 对于每个值进行测试
    for val in [scalar, data.dtype.type(scalar)]:
        if is_bool_not_implemented(data, all_arithmetic_operators):
            msg = "operator '.*' not implemented for bool dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op(data, val)
            with pytest.raises(NotImplementedError, match=msg):
                op(data, scalar_array)
        else:
            # 执行操作并验证结果与预期一致
            result = op(data, val)
            expected = op(data, scalar_array)
            tm.assert_extension_array_equal(result, expected)


# 测试数组中的 NA 值
def test_array_NA(data, all_arithmetic_operators):
    data, _ = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    scalar = pd.NA
    scalar_array = pd.array([pd.NA] * len(data), dtype=data.dtype)

    mask = data._mask.copy()

    if is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(data, scalar)
        # GH#45421 check op doesn't alter data._mask inplace
        tm.assert_numpy_array_equal(mask, data._mask)
        return

    # 执行操作并验证结果
    result = op(data, scalar)
    # GH#45421 check op doesn't alter data._mask inplace
    # 断言：验证numpy数组mask与data对象的_mask属性相等
    tm.assert_numpy_array_equal(mask, data._mask)
    
    # 计算预期结果：使用op函数将data与scalar_array进行操作，得到预期的结果
    expected = op(data, scalar_array)
    tm.assert_numpy_array_equal(mask, data._mask)
    
    # 断言：验证扩展数组result与预期结果expected相等
    tm.assert_extension_array_equal(result, expected)
# 测试 NumPy 数组等价性函数，用于比较数据和所有算术运算符
def test_numpy_array_equivalence(data, all_arithmetic_operators):
    # 解包数据元组，data 包含数据，scalar 是一个标量值
    data, scalar = data
    # 从给定的所有算术运算符名称获取运算符函数
    op = tm.get_op_from_name(all_arithmetic_operators)
    # 检查是否需要跳过测试
    check_skip(data, all_arithmetic_operators)

    # 创建 NumPy 数组，将标量值复制为数组，指定数据类型与原始数据相同
    numpy_array = np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype)
    # 创建 Pandas 数组，使用 NumPy 数组作为输入数据，指定数据类型与原始数据相同
    pd_array = pd.array(numpy_array, dtype=data.dtype)

    # 如果操作符对布尔类型未实现
    if is_bool_not_implemented(data, all_arithmetic_operators):
        # 抛出特定错误信息的 pytest 异常
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(data, numpy_array)
        with pytest.raises(NotImplementedError, match=msg):
            op(data, pd_array)
        return

    # 使用运算符函数对数据和 NumPy 数组执行操作
    result = op(data, numpy_array)
    # 使用运算符函数对数据和 Pandas 数组执行操作，作为预期结果
    expected = op(data, pd_array)
    # 断言两个扩展数组的相等性
    tm.assert_extension_array_equal(result, expected)


# 测试与 Series 和 DataFrame 的操作等价性
# -----------------------------------------------------------------------------


# 测试 DataFrame 的操作
def test_frame(data, all_arithmetic_operators):
    # 解包数据元组，data 包含数据，scalar 是一个标量值
    data, scalar = data
    # 从给定的所有算术运算符名称获取运算符函数
    op = tm.get_op_from_name(all_arithmetic_operators)
    # 检查是否需要跳过测试
    check_skip(data, all_arithmetic_operators)

    # 创建包含单列数据的 DataFrame
    df = pd.DataFrame({"A": data})

    # 如果操作符对布尔类型未实现
    if is_bool_not_implemented(data, all_arithmetic_operators):
        # 抛出特定错误信息的 pytest 异常
        msg = "operator '.*' not implemented for bool dtypes"
        with pytest.raises(NotImplementedError, match=msg):
            op(df, scalar)
        with pytest.raises(NotImplementedError, match=msg):
            op(data, scalar)
        return

    # 使用运算符函数对 DataFrame 和标量值执行操作
    result = op(df, scalar)
    # 使用运算符函数对数据和标量值执行操作，并将结果构造为预期的 DataFrame
    expected = pd.DataFrame({"A": op(data, scalar)})
    # 断言两个 DataFrame 的相等性
    tm.assert_frame_equal(result, expected)


# 测试 Series 的操作
def test_series(data, all_arithmetic_operators):
    # 解包数据元组，data 包含数据，scalar 是一个标量值
    data, scalar = data
    # 从给定的所有算术运算符名称获取运算符函数
    op = tm.get_op_from_name(all_arithmetic_operators)
    # 检查是否需要跳过测试
    check_skip(data, all_arithmetic_operators)

    # 创建包含数据的 Series
    ser = pd.Series(data)

    # 准备其他要测试的数据类型
    others = [
        scalar,
        np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype),
        pd.array([scalar] * len(data), dtype=data.dtype),
        pd.Series([scalar] * len(data), dtype=data.dtype),
    ]

    # 遍历测试每种数据类型
    for other in others:
        # 如果操作符对布尔类型未实现
        if is_bool_not_implemented(data, all_arithmetic_operators):
            # 抛出特定错误信息的 pytest 异常
            msg = "operator '.*' not implemented for bool dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op(ser, other)
        else:
            # 使用运算符函数对 Series 和其他数据类型执行操作
            result = op(ser, other)
            # 使用运算符函数对数据和其他数据类型执行操作，并将结果构造为预期的 Series
            expected = pd.Series(op(data, other))
            # 断言两个 Series 的相等性
            tm.assert_series_equal(result, expected)


# 测试一般特征和错误
# -----------------------------------------------------------------------------


# 测试无效对象时的错误情况
def test_error_invalid_object(data, all_arithmetic_operators):
    # 解包数据元组，data 包含数据
    data, _ = data

    # 使用给定的算术运算符名称
    op = all_arithmetic_operators
    # 从数据对象中获取运算符方法
    opa = getattr(data, op)

    # 对二维数据执行操作返回 NotImplemented
    result = opa(pd.DataFrame({"A": data}))
    assert result is NotImplemented

    # 准备错误信息的正则表达式
    msg = r"can only perform ops with 1-d structures"
    # 断言抛出特定错误信息的 pytest 异常
    with pytest.raises(NotImplementedError, match=msg):
        opa(np.arange(len(data)).reshape(-1, len(data)))
# 测试函数，用于检查在长度不匹配时操作列表是否会引发错误
def test_error_len_mismatch(data, all_arithmetic_operators):
    # 解包数据元组，将其中的第一个元素赋给data，第二个赋给scalar
    data, scalar = data
    # 从名称中获取操作符对象
    op = tm.get_op_from_name(all_arithmetic_operators)

    # 创建一个列表，长度比data少1，并用scalar填充
    other = [scalar] * (len(data) - 1)

    # 默认错误类型为ValueError
    err = ValueError
    # 默认错误信息为两个可能的广播错误
    msg = "|".join(
        [
            r"operands could not be broadcast together with shapes \(3,\) \(4,\)",
            r"operands could not be broadcast together with shapes \(4,\) \(3,\)",
        ]
    )

    # 如果data的dtype为布尔类型且操作符不在[sub, rsub]中，则错误类型为TypeError
    if data.dtype.kind == "b" and all_arithmetic_operators.strip("_") in [
        "sub",
        "rsub",
    ]:
        err = TypeError
        # 错误信息更新为布尔类型的减法不支持的提示信息
        msg = (
            r"numpy boolean subtract, the `\-` operator, is not supported, use "
            r"the bitwise_xor, the `\^` operator, or the logical_xor function instead"
        )
    # 如果data和操作符不支持的布尔运算，则错误信息为布尔类型未实现指定运算符的提示
    elif is_bool_not_implemented(data, all_arithmetic_operators):
        msg = "operator '.*' not implemented for bool dtypes"
        err = NotImplementedError

    # 遍历other列表和其对应的numpy数组
    for val in [other, np.array(other)]:
        # 使用pytest断言错误err，并匹配错误消息msg
        with pytest.raises(err, match=msg):
            op(data, val)

        # 将data转换为Series对象s，并使用pytest断言错误err，并匹配错误消息msg
        s = pd.Series(data)
        with pytest.raises(err, match=msg):
            op(s, val)


# 使用参数化测试op参数为['__neg__', '__abs__', '__invert__']来执行测试
@pytest.mark.parametrize("op", ["__neg__", "__abs__", "__invert__"])
def test_unary_op_does_not_propagate_mask(data, op):
    # 解包数据元组，将其中的第一个元素赋给data，第二个赋给_
    data, _ = data
    # 将data转换为Series对象ser
    ser = pd.Series(data)

    # 如果操作为'__invert__'且data的dtype为浮点类型，则按照numpy的行为抛出TypeError
    if op == "__invert__" and data.dtype.kind == "f":
        msg = "ufunc 'invert' not supported for the input types"
        with pytest.raises(TypeError, match=msg):
            getattr(ser, op)()
        with pytest.raises(TypeError, match=msg):
            getattr(data, op)()
        with pytest.raises(TypeError, match=msg):
            # 检查这是否仍然是numpy的行为
            getattr(data._data, op)()

        return

    # 执行一元操作op，并将结果赋给result
    result = getattr(ser, op)()
    # 深拷贝result，并赋给expected
    expected = result.copy(deep=True)
    # 将ser的第一个元素设为None
    ser[0] = None
    # 断言result与expected相等
    tm.assert_series_equal(result, expected)
```