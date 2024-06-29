# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_arithmetics.py`

```
import operator  # 导入 operator 模块，用于进行运算符操作

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 Pandas 库，并使用 pd 别名
from pandas import SparseDtype  # 从 Pandas 中导入 SparseDtype 类型
import pandas._testing as tm  # 导入 Pandas 内部测试模块，使用 tm 别名
from pandas.core.arrays.sparse import SparseArray  # 从 Pandas 的 sparse 模块导入 SparseArray 类


@pytest.fixture(params=["integer", "block"])
def kind(request):
    """kind kwarg to pass to SparseArray"""
    return request.param  # 返回请求的参数值，用于 SparseArray 的 kind 参数


@pytest.fixture(params=[True, False])
def mix(request):
    """
    Fixture returning True or False, determining whether to operate
    op(sparse, dense) instead of op(sparse, sparse)
    """
    return request.param  # 返回请求的参数值，决定是否进行 op(sparse, dense) 而不是 op(sparse, sparse)


class TestSparseArrayArithmetics:
    def _assert(self, a, b):
        # We have to use tm.assert_sp_array_equal. See GH #45126
        tm.assert_numpy_array_equal(a, b)  # 使用 Pandas 内部测试模块的函数断言两个 SparseArray 对象是否相等

    def _check_numeric_ops(self, a, b, a_dense, b_dense, mix: bool, op):
        # Check that arithmetic behavior matches non-Sparse Series arithmetic
        # 检查算术行为是否与非稀疏 Series 算术匹配

        if isinstance(a_dense, np.ndarray):
            expected = op(pd.Series(a_dense), b_dense).values  # 如果 a_dense 是 NumPy 数组，使用 Pandas Series 进行操作
        elif isinstance(b_dense, np.ndarray):
            expected = op(a_dense, pd.Series(b_dense)).values  # 如果 b_dense 是 NumPy 数组，使用 Pandas Series 进行操作
        else:
            raise NotImplementedError  # 抛出未实现错误，目前不支持的操作类型

        with np.errstate(invalid="ignore", divide="ignore"):
            if mix:
                result = op(a, b_dense).to_dense()  # 如果 mix 为 True，执行 op(sparse, dense)，将稀疏数组转换为稠密数组
            else:
                result = op(a, b).to_dense()  # 否则执行 op(sparse, sparse)，将稀疏数组转换为稠密数组

        self._assert(result, expected)  # 断言稠密数组的操作结果与预期结果相等

    def _check_bool_result(self, res):
        assert isinstance(res, SparseArray)  # 断言结果是 SparseArray 类型
        assert isinstance(res.dtype, SparseDtype)  # 断言结果的数据类型是 SparseDtype
        assert res.dtype.subtype == np.bool_  # 断言数据类型的子类型是布尔型
        assert isinstance(res.fill_value, bool)  # 断言填充值的类型是布尔型
    def _check_comparison_ops(self, a, b, a_dense, b_dense):
        # 使用 np.errstate(invalid="ignore") 来临时忽略无效操作的警告
        with np.errstate(invalid="ignore"):
            # Unfortunately, trying to wrap the computation of each expected
            # value is with np.errstate() is too tedious.
            #
            # sparse & sparse

            # 检查稀疏矩阵 a 是否等于稀疏矩阵 b，返回布尔结果并进行检查
            self._check_bool_result(a == b)
            # 断言稀疏矩阵 a 与 b 转换为密集形式后是否相等
            self._assert((a == b).to_dense(), a_dense == b_dense)

            # 检查稀疏矩阵 a 是否不等于稀疏矩阵 b，返回布尔结果并进行检查
            self._check_bool_result(a != b)
            # 断言稀疏矩阵 a 与 b 转换为密集形式后是否不相等
            self._assert((a != b).to_dense(), a_dense != b_dense)

            # 检查稀疏矩阵 a 是否大于等于稀疏矩阵 b，返回布尔结果并进行检查
            self._check_bool_result(a >= b)
            # 断言稀疏矩阵 a 与 b 转换为密集形式后是否大于等于
            self._assert((a >= b).to_dense(), a_dense >= b_dense)

            # 检查稀疏矩阵 a 是否小于等于稀疏矩阵 b，返回布尔结果并进行检查
            self._check_bool_result(a <= b)
            # 断言稀疏矩阵 a 与 b 转换为密集形式后是否小于等于
            self._assert((a <= b).to_dense(), a_dense <= b_dense)

            # 检查稀疏矩阵 a 是否大于稀疏矩阵 b，返回布尔结果并进行检查
            self._check_bool_result(a > b)
            # 断言稀疏矩阵 a 与 b 转换为密集形式后是否大于
            self._assert((a > b).to_dense(), a_dense > b_dense)

            # 检查稀疏矩阵 a 是否小于稀疏矩阵 b，返回布尔结果并进行检查
            self._check_bool_result(a < b)
            # 断言稀疏矩阵 a 与 b 转换为密集形式后是否小于
            self._assert((a < b).to_dense(), a_dense < b_dense)

            # sparse & dense

            # 检查稀疏矩阵 a 是否等于密集矩阵 b_dense，返回布尔结果并进行检查
            self._check_bool_result(a == b_dense)
            # 断言稀疏矩阵 a 与密集矩阵 b_dense 转换为密集形式后是否相等
            self._assert((a == b_dense).to_dense(), a_dense == b_dense)

            # 检查稀疏矩阵 a 是否不等于密集矩阵 b_dense，返回布尔结果并进行检查
            self._check_bool_result(a != b_dense)
            # 断言稀疏矩阵 a 与密集矩阵 b_dense 转换为密集形式后是否不相等
            self._assert((a != b_dense).to_dense(), a_dense != b_dense)

            # 检查稀疏矩阵 a 是否大于等于密集矩阵 b_dense，返回布尔结果并进行检查
            self._check_bool_result(a >= b_dense)
            # 断言稀疏矩阵 a 与密集矩阵 b_dense 转换为密集形式后是否大于等于
            self._assert((a >= b_dense).to_dense(), a_dense >= b_dense)

            # 检查稀疏矩阵 a 是否小于等于密集矩阵 b_dense，返回布尔结果并进行检查
            self._check_bool_result(a <= b_dense)
            # 断言稀疏矩阵 a 与密集矩阵 b_dense 转换为密集形式后是否小于等于
            self._assert((a <= b_dense).to_dense(), a_dense <= b_dense)

            # 检查稀疏矩阵 a 是否大于密集矩阵 b_dense，返回布尔结果并进行检查
            self._check_bool_result(a > b_dense)
            # 断言稀疏矩阵 a 与密集矩阵 b_dense 转换为密集形式后是否大于
            self._assert((a > b_dense).to_dense(), a_dense > b_dense)

            # 检查稀疏矩阵 a 是否小于密集矩阵 b_dense，返回布尔结果并进行检查
            self._check_bool_result(a < b_dense)
            # 断言稀疏矩阵 a 与密集矩阵 b_dense 转换为密集形式后是否小于
            self._assert((a < b_dense).to_dense(), a_dense < b_dense)

    def _check_logical_ops(self, a, b, a_dense, b_dense):
        # sparse & sparse

        # 检查稀疏矩阵 a 是否与稀疏矩阵 b 逻辑与，返回布尔结果并进行检查
        self._check_bool_result(a & b)
        # 断言稀疏矩阵 a 与 b 逻辑与的结果转换为密集形式是否相等
        self._assert((a & b).to_dense(), a_dense & b_dense)

        # 检查稀疏矩阵 a 是否与稀疏矩阵 b 逻辑或，返回布尔结果并进行检查
        self._check_bool_result(a | b)
        # 断言稀疏矩阵 a 与 b 逻辑或的结果转换为密集形式是否相等
        self._assert((a | b).to_dense(), a_dense | b_dense)

        # sparse & dense

        # 检查稀疏矩阵 a 是否与密集矩阵 b_dense 逻辑与，返回布尔结果并进行检查
        self._check_bool_result(a & b_dense)
        # 断言稀疏矩阵 a 与密集矩阵 b_dense 逻辑与的结果转换为密集形式是否相等
        self._assert((a & b_dense).to_dense(), a_dense & b_dense)

        # 检查稀疏矩阵 a 是否与密集矩阵 b_dense 逻辑或，返回布尔结果并进行检查
        self._check_bool_result(a | b_dense)
        # 断言稀疏矩阵 a 与密集矩阵 b_dense 逻辑或的结果转换为密集形式是否相等
        self._assert((a | b_dense).to_dense(), a_dense | b_dense)

    @pytest.mark.parametrize("scalar", [0, 1, 3])
    @pytest.mark.parametrize("fill_value", [None, 0, 2])
    def test_float_scalar(
        self, kind, mix, all_arithmetic_functions, fill_value, scalar, request
    ):
        op = all_arithmetic_functions
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        # 使用给定的值创建稀疏数组 a
        a = SparseArray(values, kind=kind, fill_value=fill_value)
        # 调用 _check_numeric_ops 方法检查稀疏数组 a 与标量 scalar 的各种数学
    # 测试稀疏数组与标量浮点数的比较操作
    def test_float_scalar_comparison(self, kind):
        # 创建包含 NaN 和整数的 NumPy 数组
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])

        # 使用给定的稀疏数组类型创建稀疏数组对象 a
        a = SparseArray(values, kind=kind)
        # 对稀疏数组 a 执行与标量 1 的比较操作，并检查结果
        self._check_comparison_ops(a, 1, values, 1)
        # 对稀疏数组 a 执行与标量 0 的比较操作，并检查结果
        self._check_comparison_ops(a, 0, values, 0)
        # 对稀疏数组 a 执行与标量 3 的比较操作，并检查结果
        self._check_comparison_ops(a, 3, values, 3)

        # 使用给定的稀疏数组类型和填充值 0 创建稀疏数组对象 a
        a = SparseArray(values, kind=kind, fill_value=0)
        # 对稀疏数组 a 执行与标量 1 的比较操作，并检查结果
        self._check_comparison_ops(a, 1, values, 1)
        # 对稀疏数组 a 执行与标量 0 的比较操作，并检查结果
        self._check_comparison_ops(a, 0, values, 0)
        # 对稀疏数组 a 执行与标量 3 的比较操作，并检查结果
        self._check_comparison_ops(a, 3, values, 3)

        # 使用给定的稀疏数组类型和填充值 2 创建稀疏数组对象 a
        a = SparseArray(values, kind=kind, fill_value=2)
        # 对稀疏数组 a 执行与标量 1 的比较操作，并检查结果
        self._check_comparison_ops(a, 1, values, 1)
        # 对稀疏数组 a 执行与标量 0 的比较操作，并检查结果
        self._check_comparison_ops(a, 0, values, 0)
        # 对稀疏数组 a 执行与标量 3 的比较操作，并检查结果
        self._check_comparison_ops(a, 3, values, 3)

    # 测试稀疏数组与非 NaN 值的相同索引的浮点数操作
    def test_float_same_index_without_nans(self, kind, mix, all_arithmetic_functions):
        # 设置操作函数为所有算术函数
        op = all_arithmetic_functions

        # 创建包含浮点数的 NumPy 数组 values 和 rvalues
        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        # 使用给定的稀疏数组类型和填充值 0 创建稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        # 执行稀疏数组 a 和 b 之间的数值操作，检查结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    # 测试稀疏数组与含有 NaN 值的相同索引的浮点数操作
    def test_float_same_index_with_nans(
        self, kind, mix, all_arithmetic_functions, request
    ):
        # 设置操作函数为所有算术函数
        op = all_arithmetic_functions
        # 创建包含 NaN 和整数的 NumPy 数组 values 和 rvalues
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        # 使用给定的稀疏数组类型创建稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        # 执行稀疏数组 a 和 b 之间的数值操作，检查结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    # 测试稀疏数组与另一个稀疏数组的比较操作
    def test_float_same_index_comparison(self, kind):
        # 设置操作函数为所有算术函数
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([np.nan, 2, 3, 4, np.nan, 0, 1, 3, 2, np.nan])

        # 使用给定的稀疏数组类型创建稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        # 执行稀疏数组 a 和 b 之间的比较操作，检查结果
        self._check_comparison_ops(a, b, values, rvalues)

        # 创建包含浮点数的 NumPy 数组 values 和 rvalues
        values = np.array([0.0, 1.0, 2.0, 6.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        rvalues = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0])

        # 使用给定的稀疏数组类型和填充值 0 创建稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        # 执行稀疏数组 a 和 b 之间的比较操作，检查结果
        self._check_comparison_ops(a, b, values, rvalues)
    # 测试稀疏数组的浮点数操作，包括不同的种类、混合、所有算术函数
    def test_float_array(self, kind, mix, all_arithmetic_functions):
        # 将所有算术函数赋值给变量 op
        op = all_arithmetic_functions

        # 创建包含 NaN 的 numpy 数组 values 和 rvalues
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        # 使用 SparseArray 类创建稀疏数组 a 和 b，指定种类为 kind
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        # 调用 _check_numeric_ops 方法，检查稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        # 调用 _check_numeric_ops 方法，检查稀疏数组与常数相乘后的数值操作结果
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        # 创建填充值为 0 的稀疏数组 a 和 b
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        # 调用 _check_numeric_ops 方法，检查填充值不同的稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 创建填充值均为 0 的稀疏数组 a 和 b
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        # 调用 _check_numeric_ops 方法，检查填充值相同的稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 创建填充值分别为 1 和 2 的稀疏数组 a 和 b
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        # 调用 _check_numeric_ops 方法，检查填充值不同的稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    # 测试不同种类稀疏数组的浮点数操作，包括混合、所有算术函数
    def test_float_array_different_kind(self, mix, all_arithmetic_functions):
        # 将所有算术函数赋值给变量 op
        op = all_arithmetic_functions

        # 创建包含 NaN 的 numpy 数组 values 和 rvalues
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        # 使用 SparseArray 类创建种类分别为 "integer" 和 "block" 的稀疏数组 a 和 b
        a = SparseArray(values, kind="integer")
        b = SparseArray(rvalues, kind="block")
        # 调用 _check_numeric_ops 方法，检查不同种类稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        # 调用 _check_numeric_ops 方法，检查稀疏数组与常数相乘后的数值操作结果
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        # 创建填充值为 0 的种类分别为 "integer" 和 "block" 的稀疏数组 a 和 b
        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block")
        # 调用 _check_numeric_ops 方法，检查填充值不同的稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 创建填充值均为 0 的种类分别为 "integer" 和 "block" 的稀疏数组 a 和 b
        a = SparseArray(values, kind="integer", fill_value=0)
        b = SparseArray(rvalues, kind="block", fill_value=0)
        # 调用 _check_numeric_ops 方法，检查填充值相同的稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 创建填充值分别为 1 和 2 的种类分别为 "integer" 和 "block" 的稀疏数组 a 和 b
        a = SparseArray(values, kind="integer", fill_value=1)
        b = SparseArray(rvalues, kind="block", fill_value=2)
        # 调用 _check_numeric_ops 方法，检查填充值不同的稀疏数组的数值操作结果
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    # 测试稀疏数组的浮点数比较操作，包括指定种类 kind
    def test_float_array_comparison(self, kind):
        # 创建包含 NaN 的 numpy 数组 values 和 rvalues
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, np.nan, 2, 3, np.nan, 0, 1, 5, 2, np.nan])

        # 使用 SparseArray 类创建种类为 kind 的稀疏数组 a 和 b
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)
        # 调用 _check_comparison_ops 方法，检查稀疏数组的比较操作结果
        self._check_comparison_ops(a, b, values, rvalues)
        # 调用 _check_comparison_ops 方法，检查稀疏数组与常数相乘后的比较操作结果
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        # 创建填充值为 0 的种类为 kind 的稀疏数组 a 和 b
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)
        # 调用 _check_comparison_ops 方法，检查填充值不同的稀疏数组的比较操作结果
        self._check_comparison_ops(a, b, values, rvalues)

        # 创建填充值均为 0 的种类为 kind 的稀疏数组 a 和 b
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)
        # 调用 _check_comparison_ops 方法，检查填充值相同的稀疏数组的比较操作结果
        self._check_comparison_ops(a, b, values, rvalues)

        # 创建填充值分别为 1 和 2 的种类为 kind 的稀疏数组 a 和 b
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)
        # 调用 _check_comparison_ops 方法，检查填充值不同的稀疏数组的比较操作结果
        self._check_comparison_ops(a, b, values, rvalues)
    def test_int_array(self, kind, mix, all_arithmetic_functions):
        # 将所有算术函数存储在变量 op 中，以便后续使用
        op = all_arithmetic_functions

        # 由于修复 GH 667 之前必须显式指定 dtype
        dtype = np.int64

        # 创建包含指定值的 NumPy 数组，使用指定的 dtype
        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)

        # 使用 SparseArray 类创建稀疏数组 a 和 b，指定 dtype 和 kind
        a = SparseArray(values, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        # 对稀疏数组执行数值运算的检查
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        # 创建另一组稀疏数组 a 和 b，指定 fill_value，并进行类型检查
        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        # 对新的稀疏数组执行数值运算的检查
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 创建另一组稀疏数组 a 和 b，均指定 fill_value，并进行类型检查
        a = SparseArray(values, fill_value=0, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype)
        b = SparseArray(rvalues, fill_value=0, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype)

        # 对新的稀疏数组执行数值运算的检查
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 创建另一组稀疏数组 a 和 b，分别指定不同的 fill_value，并进行类型检查
        a = SparseArray(values, fill_value=1, dtype=dtype, kind=kind)
        assert a.dtype == SparseDtype(dtype, fill_value=1)
        b = SparseArray(rvalues, fill_value=2, dtype=dtype, kind=kind)
        assert b.dtype == SparseDtype(dtype, fill_value=2)

        # 对新的稀疏数组执行数值运算的检查
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

    def test_int_array_comparison(self, kind):
        # 指定 dtype 为字符串 "int64"
        dtype = "int64"

        # 创建包含指定值的 NumPy 数组，使用指定的 dtype
        values = np.array([0, 1, 2, 0, 0, 0, 1, 2, 1, 0], dtype=dtype)
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=dtype)

        # 使用 SparseArray 类创建稀疏数组 a 和 b，指定 dtype 和 kind
        a = SparseArray(values, dtype=dtype, kind=kind)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)

        # 对稀疏数组执行比较运算的检查
        self._check_comparison_ops(a, b, values, rvalues)
        self._check_comparison_ops(a, b * 0, values, rvalues * 0)

        # 创建另一组稀疏数组 a 和 b，其中 a 指定了 fill_value，并进行比较运算的检查
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind)
        self._check_comparison_ops(a, b, values, rvalues)

        # 创建另一组稀疏数组 a 和 b，均指定了 fill_value，并进行比较运算的检查
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=0)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=0)
        self._check_comparison_ops(a, b, values, rvalues)

        # 创建另一组稀疏数组 a 和 b，分别指定不同的 fill_value，并进行比较运算的检查
        a = SparseArray(values, dtype=dtype, kind=kind, fill_value=1)
        b = SparseArray(rvalues, dtype=dtype, kind=kind, fill_value=2)
        self._check_comparison_ops(a, b, values, rvalues)

    @pytest.mark.parametrize("fill_value", [True, False, np.nan])
    def test_bool_same_index(self, kind, fill_value):
        # GH 14000
        # 当 sp_index 相同时

        # 创建包含布尔值数组的 Numpy 数组
        values = np.array([True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, True, True], dtype=np.bool_)

        # 使用 SparseArray 类创建稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)

        # 调用 _check_logical_ops 方法进行逻辑操作的验证
        self._check_logical_ops(a, b, values, rvalues)

    @pytest.mark.parametrize("fill_value", [True, False, np.nan])
    def test_bool_array_logical(self, kind, fill_value):
        # GH 14000
        # 当 sp_index 相同时

        # 创建包含布尔值数组的 Numpy 数组
        values = np.array([True, False, True, False, True, True], dtype=np.bool_)
        rvalues = np.array([True, False, False, True, False, True], dtype=np.bool_)

        # 使用 SparseArray 类创建稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind, dtype=np.bool_, fill_value=fill_value)
        b = SparseArray(rvalues, kind=kind, dtype=np.bool_, fill_value=fill_value)

        # 调用 _check_logical_ops 方法进行逻辑操作的验证
        self._check_logical_ops(a, b, values, rvalues)

    def test_mixed_array_float_int(self, kind, mix, all_arithmetic_functions, request):
        # 设置用于所有算术函数的操作对象
        op = all_arithmetic_functions
        # 定义结果数据类型为 int64
        rdtype = "int64"

        # 创建包含混合类型值的 Numpy 数组
        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)

        # 使用 SparseArray 类创建稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind)
        b = SparseArray(rvalues, kind=kind)

        # 断言 b 的数据类型为 SparseDtype 类型的 rdtype
        assert b.dtype == SparseDtype(rdtype)

        # 调用 _check_numeric_ops 方法进行数值操作的验证
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
        # 调用 _check_numeric_ops 方法进行数值操作的验证，使用 b 的缩放版本
        self._check_numeric_ops(a, b * 0, values, rvalues * 0, mix, op)

        # 使用 SparseArray 类创建填充值为 0 的稀疏数组对象 a
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind)

        # 断言 b 的数据类型为 SparseDtype 类型的 rdtype
        assert b.dtype == SparseDtype(rdtype)

        # 调用 _check_numeric_ops 方法进行数值操作的验证
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 使用 SparseArray 类创建填充值为 0 的稀疏数组对象 a 和 b
        a = SparseArray(values, kind=kind, fill_value=0)
        b = SparseArray(rvalues, kind=kind, fill_value=0)

        # 断言 b 的数据类型为 SparseDtype 类型的 rdtype
        assert b.dtype == SparseDtype(rdtype)

        # 调用 _check_numeric_ops 方法进行数值操作的验证
        self._check_numeric_ops(a, b, values, rvalues, mix, op)

        # 使用 SparseArray 类创建填充值为 1 的稀疏数组对象 a 和填充值为 2 的稀疏数组对象 b
        a = SparseArray(values, kind=kind, fill_value=1)
        b = SparseArray(rvalues, kind=kind, fill_value=2)

        # 断言 b 的数据类型为具有填充值 2 的 SparseDtype 类型的 rdtype
        assert b.dtype == SparseDtype(rdtype, fill_value=2)

        # 调用 _check_numeric_ops 方法进行数值操作的验证
        self._check_numeric_ops(a, b, values, rvalues, mix, op)
    def test_mixed_array_comparison(self, kind):
        rdtype = "int64"
        # 定义变量 rdtype 为字符串 "int64"

        values = np.array([np.nan, 1, 2, 0, np.nan, 0, 1, 2, 1, np.nan])
        # 创建一个 NumPy 数组 values，包含 NaN 和整数值

        rvalues = np.array([2, 0, 2, 3, 0, 0, 1, 5, 2, 0], dtype=rdtype)
        # 创建一个 NumPy 数组 rvalues，指定数据类型为 rdtype

        a = SparseArray(values, kind=kind)
        # 使用 SparseArray 类创建稀疏数组 a，传入 values 和 kind 参数

        b = SparseArray(rvalues, kind=kind)
        # 使用 SparseArray 类创建稀疏数组 b，传入 rvalues 和 kind 参数

        assert b.dtype == SparseDtype(rdtype)
        # 断言 b 的数据类型为 SparseDtype，并与 rdtype 匹配

        self._check_comparison_ops(a, b, values, rvalues)
        # 调用类中的 _check_comparison_ops 方法，比较稀疏数组 a 和 b 的操作结果

        self._check_comparison_ops(a, b * 0, values, rvalues * 0)
        # 再次调用 _check_comparison_ops 方法，比较稀疏数组 a 和 b 乘以 0 的操作结果

        a = SparseArray(values, kind=kind, fill_value=0)
        # 使用 SparseArray 类创建填充值为 0 的稀疏数组 a，传入 values 和 kind 参数

        b = SparseArray(rvalues, kind=kind)
        # 使用 SparseArray 类创建稀疏数组 b，传入 rvalues 和 kind 参数

        assert b.dtype == SparseDtype(rdtype)
        # 断言 b 的数据类型为 SparseDtype，并与 rdtype 匹配

        self._check_comparison_ops(a, b, values, rvalues)
        # 调用类中的 _check_comparison_ops 方法，比较稀疏数组 a 和 b 的操作结果

        a = SparseArray(values, kind=kind, fill_value=0)
        # 使用 SparseArray 类创建填充值为 0 的稀疏数组 a，传入 values 和 kind 参数

        b = SparseArray(rvalues, kind=kind, fill_value=0)
        # 使用 SparseArray 类创建填充值为 0 的稀疏数组 b，传入 rvalues 和 kind 参数

        assert b.dtype == SparseDtype(rdtype)
        # 断言 b 的数据类型为 SparseDtype，并与 rdtype 匹配

        self._check_comparison_ops(a, b, values, rvalues)
        # 调用类中的 _check_comparison_ops 方法，比较稀疏数组 a 和 b 的操作结果

        a = SparseArray(values, kind=kind, fill_value=1)
        # 使用 SparseArray 类创建填充值为 1 的稀疏数组 a，传入 values 和 kind 参数

        b = SparseArray(rvalues, kind=kind, fill_value=2)
        # 使用 SparseArray 类创建填充值为 2 的稀疏数组 b，传入 rvalues 和 kind 参数

        assert b.dtype == SparseDtype(rdtype, fill_value=2)
        # 断言 b 的数据类型为 SparseDtype，并与 rdtype 匹配，填充值为 2

        self._check_comparison_ops(a, b, values, rvalues)
        # 调用类中的 _check_comparison_ops 方法，比较稀疏数组 a 和 b 的操作结果

    def test_xor(self):
        s = SparseArray([True, True, False, False])
        # 使用 SparseArray 类创建布尔类型的稀疏数组 s，传入布尔值列表

        t = SparseArray([True, False, True, False])
        # 使用 SparseArray 类创建布尔类型的稀疏数组 t，传入布尔值列表

        result = s ^ t
        # 对稀疏数组 s 和 t 执行异或操作，得到结果稀疏数组 result

        sp_index = pd.core.arrays.sparse.IntIndex(4, np.array([0, 1, 2], dtype="int32"))
        # 创建一个 Pandas 稀疏整数索引 sp_index，包含 4 个元素，索引值为 [0, 1, 2]

        expected = SparseArray([False, True, True], sparse_index=sp_index)
        # 使用 SparseArray 类创建布尔类型的稀疏数组 expected，传入布尔值列表和稀疏索引 sp_index

        tm.assert_sp_array_equal(result, expected)
        # 使用 tm.assert_sp_array_equal 方法断言 result 和 expected 稀疏数组相等
# 使用 pytest 的标记，参数化测试函数，测试不同的操作符
@pytest.mark.parametrize("op", [operator.eq, operator.add])
def test_with_list(op):
    # 创建稀疏数组对象 arr，填充值为 0
    arr = SparseArray([0, 1], fill_value=0)
    # 对稀疏数组 arr 和普通列表 [0, 1] 执行操作符 op
    result = op(arr, [0, 1])
    # 对稀疏数组 arr 和另一个稀疏数组 SparseArray([0, 1]) 执行操作符 op
    expected = op(arr, SparseArray([0, 1]))
    # 断言 result 和 expected 相等
    tm.assert_sp_array_equal(result, expected)


def test_with_dataframe():
    # GH#27910
    # 创建稀疏数组对象 arr，填充值为 0
    arr = SparseArray([0, 1], fill_value=0)
    # 创建 DataFrame 对象 df
    df = pd.DataFrame([[1, 2], [3, 4]])
    # 尝试对稀疏数组 arr 和 DataFrame df 执行加法操作
    result = arr.__add__(df)
    # 断言结果为 NotImplemented
    assert result is NotImplemented


def test_with_zerodim_ndarray():
    # GH#27910
    # 创建稀疏数组对象 arr，填充值为 0
    arr = SparseArray([0, 1], fill_value=0)

    # 对 arr 乘以一个零维 ndarray np.array(2)
    result = arr * np.array(2)
    # 预期结果是 arr 乘以标量值 2
    expected = arr * 2
    # 断言 result 和 expected 相等
    tm.assert_sp_array_equal(result, expected)


# 使用 pytest 的标记，参数化测试函数，测试不同的 ufunc 函数
@pytest.mark.parametrize("ufunc", [np.abs, np.exp])
@pytest.mark.parametrize(
    "arr", [SparseArray([0, 0, -1, 1]), SparseArray([None, None, -1, 1])]
)
def test_ufuncs(ufunc, arr):
    # 对稀疏数组 arr 应用 ufunc 函数
    result = ufunc(arr)
    # 计算填充值 arr.fill_value 的 ufunc 函数值
    fill_value = ufunc(arr.fill_value)
    # 构建期望的稀疏数组对象，其数据为 arr 应用 ufunc 函数的结果，填充值为 fill_value
    expected = SparseArray(ufunc(np.asarray(arr)), fill_value=fill_value)
    # 断言 result 和 expected 相等
    tm.assert_sp_array_equal(result, expected)


# 使用 pytest 的标记，参数化测试函数，测试不同的二元 ufunc 函数
@pytest.mark.parametrize(
    "a, b",
    [
        (SparseArray([0, 0, 0]), np.array([0, 1, 2])),
        (SparseArray([0, 0, 0], fill_value=1), np.array([0, 1, 2])),
    ],
)
@pytest.mark.parametrize("ufunc", [np.add, np.greater])
def test_binary_ufuncs(ufunc, a, b):
    # 不能确定这里的填充值
    # 对稀疏数组 a 和数组 b 执行 ufunc 函数
    result = ufunc(a, b)
    # 计算期望结果，对稀疏数组 a 和数组 b 的元素逐个应用 ufunc 函数
    expected = ufunc(np.asarray(a), np.asarray(b))
    # 断言 result 是 SparseArray 类型
    assert isinstance(result, SparseArray)
    # 断言 result 的数值部分与 expected 相等
    tm.assert_numpy_array_equal(np.asarray(result), expected)


def test_ndarray_inplace():
    # 创建稀疏数组对象 sparray
    sparray = SparseArray([0, 2, 0, 0])
    # 创建普通 ndarray 对象 ndarray
    ndarray = np.array([0, 1, 2, 3])
    # 将 sparray 添加到 ndarray 中（原地操作）
    ndarray += sparray
    # 期望的结果 ndarray
    expected = np.array([0, 3, 2, 3])
    # 断言 ndarray 和 expected 相等
    tm.assert_numpy_array_equal(ndarray, expected)


def test_sparray_inplace():
    # 创建稀疏数组对象 sparray
    sparray = SparseArray([0, 2, 0, 0])
    # 创建普通 ndarray 对象 ndarray
    ndarray = np.array([0, 1, 2, 3])
    # 将 ndarray 添加到 sparray 中（原地操作）
    sparray += ndarray
    # 期望的结果 sparray
    expected = SparseArray([0, 3, 2, 3], fill_value=0)
    # 断言 sparray 和 expected 相等
    tm.assert_sp_array_equal(sparray, expected)


# 使用 pytest 的标记，参数化测试函数，测试不同的构造函数
@pytest.mark.parametrize("cons", [list, np.array, SparseArray])
def test_mismatched_length_cmp_op(cons):
    # 创建稀疏数组对象 left，元素为布尔值 True
    left = SparseArray([True, True])
    # 创建 cons 类型对象 right，元素个数为 3
    right = cons([True, True, True])
    # 使用 pytest 的断言检查是否抛出 ValueError 异常，且异常信息为 "operands have mismatched length"
    with pytest.raises(ValueError, match="operands have mismatched length"):
        left & right


# 使用 pytest 的标记，参数化测试函数，测试不同的二元操作符函数和填充值
@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv", "floordiv", "pow"])
@pytest.mark.parametrize("fill_value", [np.nan, 3])
def test_binary_operators(op, fill_value):
    # 获取操作符对应的函数对象 op
    op = getattr(operator, op)
    # 生成随机数据
    data1 = np.random.default_rng(2).standard_normal(20)
    data2 = np.random.default_rng(2).standard_normal(20)
    # 根据填充值填充 data1 和 data2
    data1[::2] = fill_value
    data2[::3] = fill_value
    # 创建稀疏数组对象 first 和 second
    first = SparseArray(data1, fill_value=fill_value)
    second = SparseArray(data2, fill_value=fill_value)
    # 忽略 NumPy 的错误和警告
    with np.errstate(all="ignore"):
        # 执行稀疏数组的操作，并存储结果
        res = op(first, second)
        # 使用稠密数组的结果创建稀疏数组作为期望结果
        exp = SparseArray(
            op(first.to_dense(), second.to_dense()), fill_value=first.fill_value
        )
        # 断言 res 是 SparseArray 类型
        assert isinstance(res, SparseArray)
        # 检查稀疏数组和期望稀疏数组的稠密表示是否几乎相等
        tm.assert_almost_equal(res.to_dense(), exp.to_dense())

        # 对第二个参数为稠密数组的情况进行操作，比较结果
        res2 = op(first, second.to_dense())
        assert isinstance(res2, SparseArray)
        tm.assert_sp_array_equal(res, res2)

        # 对第一个参数为稠密数组的情况进行操作，比较结果
        res3 = op(first.to_dense(), second)
        assert isinstance(res3, SparseArray)
        tm.assert_sp_array_equal(res, res3)

        # 对第二个参数为标量的情况进行操作，比较结果
        res4 = op(first, 4)
        assert isinstance(res4, SparseArray)

        # 忽略由于操作引发的错误（例如乘方操作）
        try:
            # 使用稠密数组的第一个参数和标量进行操作，并比较填充值
            exp = op(first.to_dense(), 4)
            exp_fv = op(first.fill_value, 4)
        except ValueError:
            # 如果操作引发 ValueError，则忽略该情况
            pass
        else:
            # 断言稀疏数组的填充值几乎等于期望的填充值
            tm.assert_almost_equal(res4.fill_value, exp_fv)
            # 断言稀疏数组的稠密表示几乎等于期望的稠密表示
            tm.assert_almost_equal(res4.to_dense(), exp)
```