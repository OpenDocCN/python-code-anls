# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_numpy_compat.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入以下模块：
    CategoricalIndex,  # 分类索引
    DatetimeIndex,  # 日期时间索引
    Index,  # 索引
    PeriodIndex,  # 时期索引
    TimedeltaIndex,  # 时间增量索引
    isna,  # 判断是否为缺失值的函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块
from pandas.api.types import (  # 从 Pandas API 中导入以下类型判断函数：
    is_complex_dtype,  # 判断是否为复数类型
    is_numeric_dtype,  # 判断是否为数值类型
)
from pandas.core.arrays import BooleanArray  # 导入 Pandas 的布尔数组类型
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin  # 导入日期时间操作的 mixin 类


def test_numpy_ufuncs_out(index):
    result = index == index  # 检查索引是否等于自身，返回布尔数组

    out = np.empty(index.shape, dtype=bool)  # 创建一个空的布尔类型数组，形状与索引相同
    np.equal(index, index, out=out)  # 使用 NumPy 的 equal 函数比较索引的元素，结果存储在 out 中
    tm.assert_numpy_array_equal(out, result)  # 断言 out 数组与 result 相等

    if not index._is_multi:
        # 如果索引不是多重索引
        out = np.empty(index.shape, dtype=bool)  # 创建一个空的布尔类型数组，形状与索引相同
        np.equal(index.array, index.array, out=out)  # 比较索引的 ExtensionArray 的元素，结果存储在 out 中
        tm.assert_numpy_array_equal(out, result)  # 断言 out 数组与 result 相等


@pytest.mark.parametrize(
    "func",
    [  # 参数化测试函数 func，包括以下 NumPy 的数学函数：
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        np.sqrt,
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.deg2rad,
        np.rad2deg,
    ],
    ids=lambda x: x.__name__,  # 将每个函数的名称作为其 id
)
def test_numpy_ufuncs_basic(index, func):
    # 测试 NumPy 的通用函数（ufuncs），参考：
    # https://numpy.org/doc/stable/reference/ufuncs.html

    if isinstance(index, DatetimeIndexOpsMixin):
        # 如果索引实现了 DatetimeIndexOpsMixin 接口
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all="ignore"):
                func(index)  # 调用函数 func 处理索引
    elif is_numeric_dtype(index) and not (
        is_complex_dtype(index) and func in [np.deg2rad, np.rad2deg]
    ):
        # 如果索引是数值类型且不是复数类型，并且 func 不是 np.deg2rad 或 np.rad2deg
        with np.errstate(all="ignore"):
            result = func(index)  # 对索引应用函数 func
            arr_result = func(index.values)  # 对索引值数组应用函数 func
            if arr_result.dtype == np.float16:
                arr_result = arr_result.astype(np.float32)  # 如果结果是 float16 类型，转换为 float32
            exp = Index(arr_result, name=index.name)  # 创建一个索引对象 exp，使用处理后的结果

        tm.assert_index_equal(result, exp)  # 断言处理后的索引与预期的索引对象 exp 相等
        if isinstance(index.dtype, np.dtype) and is_numeric_dtype(index):
            if is_complex_dtype(index):
                assert result.dtype == index.dtype  # 对于复数类型，断言结果与索引类型相同
            elif index.dtype in ["bool", "int8", "uint8"]:
                assert result.dtype in ["float16", "float32"]  # 对于 bool、int8、uint8 类型，断言结果为 float16 或 float32
            elif index.dtype in ["int16", "uint16", "float32"]:
                assert result.dtype == "float32"  # 对于 int16、uint16、float32 类型，断言结果为 float32
            else:
                assert result.dtype == "float64"  # 否则，断言结果为 float64
        else:
            # 例如，对于 Int64 类型的 np.exp -> Float64
            assert type(result) is Index  # 断言结果是索引对象
    elif len(index) == 0:
        pass  # 如果索引长度为 0，则什么也不做
    else:
        with tm.external_error_raised((TypeError, AttributeError)):
            with np.errstate(all="ignore"):
                func(index)  # 调用函数 func，可能引发 AttributeError 或 TypeError


@pytest.mark.parametrize(
    "func", [np.isfinite, np.isinf, np.isnan, np.signbit], ids=lambda x: x.__name__
)
# 测试 numpy 的通用函数（ufuncs），参考文档：
# https://numpy.org/doc/stable/reference/ufuncs.html
def test_numpy_ufuncs_other(index, func):
    # 如果 index 是 DatetimeIndex 或 TimedeltaIndex 类型
    if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
        # 如果 func 是 np.isfinite, np.isinf, np.isnan 中的一个
        if func in (np.isfinite, np.isinf, np.isnan):
            # 调用 func 函数计算结果
            result = func(index)
            # 确保返回结果是 np.ndarray 类型
            assert isinstance(result, np.ndarray)

            # 创建一个布尔类型的空数组 out，与 index 形状相同
            out = np.empty(index.shape, dtype=bool)
            # 将 func 应用于 index，并将结果存储到 out 中
            func(index, out=out)
            # 检查 out 是否与 result 相等
            tm.assert_numpy_array_equal(out, result)
        else:
            # 如果 func 不在上述列表中，期望抛出 TypeError 异常
            with tm.external_error_raised(TypeError):
                func(index)

    # 如果 index 是 PeriodIndex 类型
    elif isinstance(index, PeriodIndex):
        # 期望抛出 TypeError 异常
        with tm.external_error_raised(TypeError):
            func(index)

    # 如果 index 是数值类型且不是复数类型或 func 不是 np.signbit
    elif is_numeric_dtype(index) and not (
        is_complex_dtype(index) and func is np.signbit
    ):
        # 调用 func 函数计算结果，结果为布尔数组
        result = func(index)
        # 如果 index 不是 np.dtype 类型
        if not isinstance(index.dtype, np.dtype):
            # 期望返回结果是 BooleanArray 类型
            assert isinstance(result, BooleanArray)
        else:
            # 否则期望返回结果是 np.ndarray 类型
            assert isinstance(result, np.ndarray)

        # 创建一个布尔类型的空数组 out，与 index 形状相同
        out = np.empty(index.shape, dtype=bool)
        # 将 func 应用于 index，并将结果存储到 out 中
        func(index, out=out)

        # 如果 index 不是 np.dtype 类型
        if not isinstance(index.dtype, np.dtype):
            # 检查 out 是否与 result._data 相等
            tm.assert_numpy_array_equal(out, result._data)
        else:
            # 否则检查 out 是否与 result 相等
            tm.assert_numpy_array_equal(out, result)

    # 如果 index 长度为 0
    elif len(index) == 0:
        # 什么也不做，pass
        pass
    else:
        # 对于其他情况，期望抛出 TypeError 异常
        with tm.external_error_raised(TypeError):
            func(index)


@pytest.mark.parametrize("func", [np.maximum, np.minimum])
def test_numpy_ufuncs_reductions(index, func, request):
    # TODO: 与 tests.series.test_ufunc.test_reductions 有重叠
    if len(index) == 0:
        # 如果 index 长度为 0，跳过测试
        pytest.skip("Test doesn't make sense for empty index.")

    # 如果 index 是 CategoricalIndex 类型且未排序
    if isinstance(index, CategoricalIndex) and index.dtype.ordered is False:
        # 期望抛出 TypeError 异常，匹配字符串 "is not ordered for"
        with pytest.raises(TypeError, match="is not ordered for"):
            func.reduce(index)
        return
    else:
        # 否则调用 func.reduce 计算结果
        result = func.reduce(index)

    # 如果 func 是 np.maximum
    if func is np.maximum:
        # 期望结果与 index 的最大值相等，不跳过 NaN 值
        expected = index.max(skipna=False)
    else:
        # 否则期望结果与 index 的最小值相等，不跳过 NaN 值
        expected = index.min(skipna=False)
        # TODO: 是否有同时包含和不包含 NA 值的情况？

    # 检查 result 的类型与 expected 的类型是否相同
    assert type(result) is type(expected)
    # 如果 result 是 NaN
    if isna(result):
        # 期望 expected 也是 NaN
        assert isna(expected)
    else:
        # 否则检查 result 是否与 expected 相等
        assert result == expected


@pytest.mark.parametrize("func", [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
def test_numpy_ufuncs_bitwise(func):
    # https://github.com/pandas-dev/pandas/issues/46769
    # 创建两个 Int64 类型的 Index 对象
    idx1 = Index([1, 2, 3, 4], dtype="int64")
    idx2 = Index([3, 4, 5, 6], dtype="int64")

    # 忽略所有警告
    with tm.assert_produces_warning(None):
        # 调用 func 计算 idx1 和 idx2 的结果
        result = func(idx1, idx2)

    # 期望的结果是 func 应用于 idx1.values 和 idx2.values 后的 Index 对象
    expected = Index(func(idx1.values, idx2.values))
    # 检查 result 是否与 expected 相等
    tm.assert_index_equal(result, expected)
```