# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_unary.py`

```
import operator  # 导入 Python 标准库中的 operator 模块，用于支持各种运算操作

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 内部的测试工具模块
from pandas.core.arrays import SparseArray  # 从 Pandas 库的 core.arrays 模块中导入 SparseArray 类


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.parametrize("fill_value", [0, np.nan])
@pytest.mark.parametrize("op", [operator.pos, operator.neg])
def test_unary_op(op, fill_value):
    arr = np.array([0, 1, np.nan, 2])  # 创建一个 NumPy 数组 arr
    sparray = SparseArray(arr, fill_value=fill_value)  # 使用 SparseArray 类创建稀疏数组 sparray
    result = op(sparray)  # 对稀疏数组 sparray 执行操作 op
    expected = SparseArray(op(arr), fill_value=op(fill_value))  # 创建预期的稀疏数组 expected
    tm.assert_sp_array_equal(result, expected)  # 使用测试工具模块 tm 检查 result 和 expected 是否相等


@pytest.mark.parametrize("fill_value", [True, False])
def test_invert(fill_value):
    arr = np.array([True, False, False, True])  # 创建一个布尔类型的 NumPy 数组 arr
    sparray = SparseArray(arr, fill_value=fill_value)  # 使用 SparseArray 类创建稀疏数组 sparray
    result = ~sparray  # 对稀疏数组 sparray 执行按位取反操作
    expected = SparseArray(~arr, fill_value=not fill_value)  # 创建预期的稀疏数组 expected
    tm.assert_sp_array_equal(result, expected)  # 使用测试工具模块 tm 检查 result 和 expected 是否相等

    result = ~pd.Series(sparray)  # 对 Pandas Series 中的稀疏数组 sparray 执行按位取反操作
    expected = pd.Series(expected)  # 创建预期的 Pandas Series
    tm.assert_series_equal(result, expected)  # 使用测试工具模块 tm 检查 result 和 expected 是否相等

    result = ~pd.DataFrame({"A": sparray})  # 对 Pandas DataFrame 中的稀疏数组 sparray 执行按位取反操作
    expected = pd.DataFrame({"A": expected})  # 创建预期的 Pandas DataFrame
    tm.assert_frame_equal(result, expected)  # 使用测试工具模块 tm 检查 result 和 expected 是否相等


class TestUnaryMethods:
    @pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
    def test_neg_operator(self):
        arr = SparseArray([-1, -2, np.nan, 3], fill_value=np.nan, dtype=np.int8)  # 创建指定填充值和数据类型的稀疏数组 arr
        res = -arr  # 对稀疏数组 arr 执行负号运算
        exp = SparseArray([1, 2, np.nan, -3], fill_value=np.nan, dtype=np.int8)  # 创建预期的稀疏数组 exp
        tm.assert_sp_array_equal(exp, res)  # 使用测试工具模块 tm 检查 exp 和 res 是否相等

        arr = SparseArray([-1, -2, 1, 3], fill_value=-1, dtype=np.int8)  # 创建另一个稀疏数组 arr
        res = -arr  # 对稀疏数组 arr 执行负号运算
        exp = SparseArray([1, 2, -1, -3], fill_value=1, dtype=np.int8)  # 创建预期的稀疏数组 exp
        tm.assert_sp_array_equal(exp, res)  # 使用测试工具模块 tm 检查 exp 和 res 是否相等

    @pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
    def test_abs_operator(self):
        arr = SparseArray([-1, -2, np.nan, 3], fill_value=np.nan, dtype=np.int8)  # 创建指定填充值和数据类型的稀疏数组 arr
        res = abs(arr)  # 对稀疏数组 arr 执行绝对值运算
        exp = SparseArray([1, 2, np.nan, 3], fill_value=np.nan, dtype=np.int8)  # 创建预期的稀疏数组 exp
        tm.assert_sp_array_equal(exp, res)  # 使用测试工具模块 tm 检查 exp 和 res 是否相等

        arr = SparseArray([-1, -2, 1, 3], fill_value=-1, dtype=np.int8)  # 创建另一个稀疏数组 arr
        res = abs(arr)  # 对稀疏数组 arr 执行绝对值运算
        exp = SparseArray([1, 2, 1, 3], fill_value=1, dtype=np.int8)  # 创建预期的稀疏数组 exp
        tm.assert_sp_array_equal(exp, res)  # 使用测试工具模块 tm 检查 exp 和 res 是否相等

    def test_invert_operator(self):
        arr = SparseArray([False, True, False, True], fill_value=False, dtype=np.bool_)  # 创建布尔类型的稀疏数组 arr
        exp = SparseArray(
            np.invert([False, True, False, True]), fill_value=True, dtype=np.bool_
        )  # 创建预期的稀疏数组 exp，使用 np.invert 对原始数组进行按位取反操作
        res = ~arr  # 对稀疏数组 arr 执行按位取反操作
        tm.assert_sp_array_equal(exp, res)  # 使用测试工具模块 tm 检查 exp 和 res 是否相等

        arr = SparseArray([0, 1, 0, 2, 3, 0], fill_value=0, dtype=np.int32)  # 创建整数类型的稀疏数组 arr
        res = ~arr  # 对稀疏数组 arr 执行按位取反操作
        exp = SparseArray([-1, -2, -1, -3, -4, -1], fill_value=-1, dtype=np.int32)  # 创建预期的稀疏数组 exp
        tm.assert_sp_array_equal(exp, res)  # 使用测试工具模块 tm 检查 exp 和 res 是否相等
```