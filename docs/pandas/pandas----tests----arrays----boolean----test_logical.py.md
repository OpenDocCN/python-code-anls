# `D:\src\scipysrc\pandas\pandas\tests\arrays\boolean\test_logical.py`

```
# 导入运算符模块
import operator

# 导入 NumPy 库并使用别名 np
import numpy as np

# 导入 pytest 库
import pytest

# 导入 pandas 库并使用别名 pd
import pandas as pd

# 导入 pandas 内部测试模块
import pandas._testing as tm

# 导入 pandas 的 BooleanArray 类
from pandas.arrays import BooleanArray

# 导入 pandas 内核操作模块中的特定函数
from pandas.core.ops.mask_ops import (
    kleene_and,
    kleene_or,
    kleene_xor,
)

# 导入 pandas 测试扩展基类
from pandas.tests.extension.base import BaseOpsUtil


class TestLogicalOps(BaseOpsUtil):
    def test_numpy_scalars_ok(self, all_logical_operators):
        # 创建一个包含 True、False 和 None 的布尔类型的 pandas 数组
        a = pd.array([True, False, None], dtype="boolean")

        # 从 a 中获取特定逻辑操作函数（and/or/xor）
        op = getattr(a, all_logical_operators)

        # 断言调用 op(True) 与 op(np.bool_(True)) 返回的扩展数组相等
        tm.assert_extension_array_equal(op(True), op(np.bool_(True)))

        # 断言调用 op(False) 与 op(np.bool_(False)) 返回的扩展数组相等
        tm.assert_extension_array_equal(op(False), op(np.bool_(False)))

    def get_op_from_name(self, op_name):
        # 去除操作符名称中的下划线
        short_opname = op_name.strip("_")

        # 如果操作符名称不包含 "xor"，则在其末尾添加下划线
        short_opname = short_opname if "xor" in short_opname else short_opname + "_"

        try:
            # 尝试从 operator 模块获取对应的操作函数
            op = getattr(operator, short_opname)
        except AttributeError:
            # 如果找不到对应的操作函数，假设是反向操作符
            rop = getattr(operator, short_opname[1:])
            op = lambda x, y: rop(y, x)  # 创建一个 lambda 函数来实现反向操作

        return op

    def test_empty_ok(self, all_logical_operators):
        # 创建一个空的布尔类型的 pandas 数组
        a = pd.array([], dtype="boolean")

        # 获取所有逻辑操作符中的一个
        op_name = all_logical_operators

        # 对空数组应用逻辑操作符，并断言结果与原数组相等
        result = getattr(a, op_name)(True)
        tm.assert_extension_array_equal(a, result)

        result = getattr(a, op_name)(False)
        tm.assert_extension_array_equal(a, result)

        result = getattr(a, op_name)(pd.NA)
        tm.assert_extension_array_equal(a, result)

    @pytest.mark.parametrize(
        "other", ["a", pd.Timestamp(2017, 1, 1, 12), np.timedelta64(4)]
    )
    def test_eq_mismatched_type(self, other):
        # 创建一个包含 True 和 False 的布尔类型的 pandas 数组
        arr = pd.array([True, False])

        # 使用不同类型的 other 对象进行等值比较，断言结果与预期相等
        result = arr == other
        expected = pd.array([False, False])
        tm.assert_extension_array_equal(result, expected)

        result = arr != other
        expected = pd.array([True, True])
        tm.assert_extension_array_equal(result, expected)

    def test_logical_length_mismatch_raises(self, all_logical_operators):
        # 获取所有逻辑操作符中的一个
        op_name = all_logical_operators

        # 创建一个包含 True、False 和 None 的布尔类型的 pandas 数组
        a = pd.array([True, False, None], dtype="boolean")

        msg = "Lengths must match"

        # 断言对不匹配长度的操作会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)([True, False])

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)(np.array([True, False]))

        with pytest.raises(ValueError, match=msg):
            getattr(a, op_name)(pd.array([True, False], dtype="boolean"))

    def test_logical_nan_raises(self, all_logical_operators):
        # 获取所有逻辑操作符中的一个
        op_name = all_logical_operators

        # 创建一个包含 True、False 和 None 的布尔类型的 pandas 数组
        a = pd.array([True, False, None], dtype="boolean")

        msg = "Got float instead"

        # 断言对包含 NaN 的操作会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            getattr(a, op_name)(np.nan)

    @pytest.mark.parametrize("other", ["a", 1])
    def test_non_bool_or_na_other_raises(self, other, all_logical_operators):
        # 创建一个包含 True 和 False 的 Pandas 数组，数据类型为 boolean
        a = pd.array([True, False], dtype="boolean")
        # 使用 pytest 来确保调用 getattr(a, all_logical_operators)(other) 时抛出 TypeError 异常，并匹配异常信息为 other 的类型名
        with pytest.raises(TypeError, match=str(type(other).__name__)):
            getattr(a, all_logical_operators)(other)

    def test_kleene_or(self):
        # A clear test of behavior.
        # 创建一个包含 True、False 和 None 的 Pandas 数组，数据类型为 boolean
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        # 进行逻辑或运算，生成结果数组 result
        result = a | b
        # 预期的结果数组，与 result 进行比较
        expected = pd.array(
            [True, True, True, True, False, None, True, None, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        # 再次进行逻辑或运算，验证 commutativity
        result = b | a
        tm.assert_extension_array_equal(result, expected)

        # 确保没有对原数组进行就地修改
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [True, None, None]),           # 测试 NA 和 boolean 数组的逻辑或运算
            (True, [True, True, True]),           # 测试 True 和 boolean 数组的逻辑或运算
            (np.bool_(True), [True, True, True]), # 测试 NumPy bool 和 boolean 数组的逻辑或运算
            (False, [True, False, None]),         # 测试 False 和 boolean 数组的逻辑或运算
            (np.bool_(False), [True, False, None]), # 测试 NumPy bool 和 boolean 数组的逻辑或运算
        ],
    )
    def test_kleene_or_scalar(self, other, expected):
        # TODO: test True & False
        # 创建一个包含 True、False 和 None 的 Pandas 数组，数据类型为 boolean
        a = pd.array([True, False, None], dtype="boolean")
        # 进行标量和数组的逻辑或运算，生成结果数组 result
        result = a | other
        # 与预期的结果数组进行比较
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        # 再次进行逻辑或运算，验证 commutativity
        result = other | a
        tm.assert_extension_array_equal(result, expected)

        # 确保没有对原数组进行就地修改
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    def test_kleene_and(self):
        # A clear test of behavior.
        # 创建一个包含 True、False 和 None 的 Pandas 数组，数据类型为 boolean
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        # 进行逻辑与运算，生成结果数组 result
        result = a & b
        # 预期的结果数组，与 result 进行比较
        expected = pd.array(
            [True, False, None, False, False, False, None, False, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)

        # 再次进行逻辑与运算，验证 commutativity
        result = b & a
        tm.assert_extension_array_equal(result, expected)

        # 确保没有对原数组进行就地修改
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )
    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [None, False, None]),  # 参数化测试：other为pd.NA时，期望结果为[None, False, None]
            (True, [True, False, None]),   # 参数化测试：other为True时，期望结果为[True, False, None]
            (False, [False, False, False]),  # 参数化测试：other为False时，期望结果为[False, False, False]
            (np.bool_(True), [True, False, None]),  # 参数化测试：other为np.bool_(True)时，期望结果为[True, False, None]
            (np.bool_(False), [False, False, False]),  # 参数化测试：other为np.bool_(False)时，期望结果为[False, False, False]
        ],
    )
    def test_kleene_and_scalar(self, other, expected):
        a = pd.array([True, False, None], dtype="boolean")
        result = a & other  # 对pd.array a和参数other进行按位与操作，计算结果存入result
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)  # 使用tm.assert_extension_array_equal断言result与expected相等

        result = other & a  # 对参数other和pd.array a进行按位与操作，计算结果存入result
        tm.assert_extension_array_equal(result, expected)  # 使用tm.assert_extension_array_equal断言result与expected相等

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )  # 使用tm.assert_extension_array_equal断言a没有就地修改

    def test_kleene_xor(self):
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")
        result = a ^ b  # 对pd.array a和b进行按位异或操作，计算结果存入result
        expected = pd.array(
            [False, True, None, True, False, None, None, None, None], dtype="boolean"
        )
        tm.assert_extension_array_equal(result, expected)  # 使用tm.assert_extension_array_equal断言result与expected相等

        result = b ^ a  # 对pd.array b和a进行按位异或操作，计算结果存入result
        tm.assert_extension_array_equal(result, expected)  # 使用tm.assert_extension_array_equal断言result与expected相等

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )  # 使用tm.assert_extension_array_equal断言a没有就地修改
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )  # 使用tm.assert_extension_array_equal断言b没有就地修改

    @pytest.mark.parametrize(
        "other, expected",
        [
            (pd.NA, [None, None, None]),  # 参数化测试：other为pd.NA时，期望结果为[None, None, None]
            (True, [False, True, None]),  # 参数化测试：other为True时，期望结果为[False, True, None]
            (np.bool_(True), [False, True, None]),  # 参数化测试：other为np.bool_(True)时，期望结果为[False, True, None]
            (np.bool_(False), [True, False, None]),  # 参数化测试：other为np.bool_(False)时，期望结果为[True, False, None]
        ],
    )
    def test_kleene_xor_scalar(self, other, expected):
        a = pd.array([True, False, None], dtype="boolean")
        result = a ^ other  # 对pd.array a和参数other进行按位异或操作，计算结果存入result
        expected = pd.array(expected, dtype="boolean")
        tm.assert_extension_array_equal(result, expected)  # 使用tm.assert_extension_array_equal断言result与expected相等

        result = other ^ a  # 对参数other和pd.array a进行按位异或操作，计算结果存入result
        tm.assert_extension_array_equal(result, expected)  # 使用tm.assert_extension_array_equal断言result与expected相等

        # ensure we haven't mutated anything inplace
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )  # 使用tm.assert_extension_array_equal断言a没有就地修改

    @pytest.mark.parametrize("other", [True, False, pd.NA, [True, False, None] * 3])
    # 定义一个测试方法，用于测试没有假设掩码值为 False 的逻辑操作
    def test_no_masked_assumptions(self, other, all_logical_operators):
        # 创建一个布尔数组 a，包含特定的值和掩码信息
        a = pd.arrays.BooleanArray(
            np.array([True, True, True, False, False, False, True, False, True]),
            np.array([False] * 6 + [True, True, True]),
        )
        # 创建一个布尔数组 b，包含特定的值和缺失信息
        b = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        
        # 如果 other 是 list 类型，则转换为布尔数组
        if isinstance(other, list):
            other = pd.array(other, dtype="boolean")

        # 对 a 应用给定的所有逻辑运算符操作
        result = getattr(a, all_logical_operators)(other)
        # 对 b 应用相同的所有逻辑运算符操作，作为预期结果
        expected = getattr(b, all_logical_operators)(other)
        # 检查结果与预期结果是否相等
        tm.assert_extension_array_equal(result, expected)

        # 如果 other 是 BooleanArray 类型
        if isinstance(other, BooleanArray):
            # 将 other 的掩码位置设为 True
            other._data[other._mask] = True
            # 将 a 的掩码位置设为 False
            a._data[a._mask] = False

            # 再次对 a 应用给定的所有逻辑运算符操作
            result = getattr(a, all_logical_operators)(other)
            # 再次对 b 应用相同的所有逻辑运算符操作，作为预期结果
            expected = getattr(b, all_logical_operators)(other)
            # 检查结果与预期结果是否相等
            tm.assert_extension_array_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_error_both_scalar，传入三种操作函数 kleene_or, kleene_xor, kleene_and
@pytest.mark.parametrize("operation", [kleene_or, kleene_xor, kleene_and])
# 定义测试函数 test_error_both_scalar，测试传入两个标量参数时的错误情况
def test_error_both_scalar(operation):
    # 定义错误信息的正则表达式
    msg = r"Either `left` or `right` need to be a np\.ndarray."
    # 使用 pytest.raises 检查是否抛出指定类型的异常，并匹配指定的错误信息
    with pytest.raises(TypeError, match=msg):
        # 调用操作函数，传入两个标量参数和两个 np.ndarray 参数
        # 注意：masks 需要是非 None 的，否则会导致无限递归
        operation(True, True, np.zeros(1), np.zeros(1))
```