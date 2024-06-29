# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_function.py`

```
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray

# 使用 pytest 的 parametrize 标记来定义多组参数化测试
@pytest.mark.parametrize("ufunc", [np.abs, np.sign])
# 忽略 numpy 中 np.sign 对 NaN 值发出的 RuntimeWarning 警告
@pytest.mark.filterwarnings("ignore:invalid value encountered in sign:RuntimeWarning")
def test_ufuncs_single_int(ufunc):
    # 创建一个包含整数和 NaN 值的 Pandas 数组
    a = pd.array([1, 2, -3, np.nan])
    # 对 Pandas 数组应用 ufunc 函数
    result = ufunc(a)
    # 将 ufunc 函数应用于浮点表示的 Pandas 数组，并指定返回类型为 Int64
    expected = pd.array(ufunc(a.astype(float)), dtype="Int64")
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(result, expected)

    # 将 Pandas 数组转换为 Series 对象
    s = pd.Series(a)
    # 对 Series 应用 ufunc 函数
    result = ufunc(s)
    # 将 ufunc 函数应用于浮点表示的 Pandas 数组，并包装为 Int64 类型的 Series 对象
    expected = pd.Series(pd.array(ufunc(a.astype(float)), dtype="Int64"))
    # 使用 Pandas 测试工具比较两个 Series 对象的内容是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.log, np.exp, np.sin, np.cos, np.sqrt])
def test_ufuncs_single_float(ufunc):
    # 创建一个包含整数和 NaN 值的 Pandas 数组
    a = pd.array([1, 2, -3, np.nan])
    # 在忽略无效值的情况下，对 Pandas 数组应用 ufunc 函数
    with np.errstate(invalid="ignore"):
        result = ufunc(a)
        # 使用 FloatingArray 封装 ufunc 函数应用后的结果，保留掩码信息
        expected = FloatingArray(ufunc(a.astype(float)), mask=a._mask)
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(result, expected)

    # 将 Pandas 数组转换为 Series 对象
    s = pd.Series(a)
    # 在忽略无效值的情况下，对 Series 应用 ufunc 函数
    with np.errstate(invalid="ignore"):
        result = ufunc(s)
    # 将 ufunc 函数应用于浮点表示的 Pandas 数组，并转换为与 expected 类型相同的 Series 对象
    expected = pd.Series(expected)
    # 使用 Pandas 测试工具比较两个 Series 对象的内容是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.add, np.subtract])
def test_ufuncs_binary_int(ufunc):
    # 创建一个包含整数和 NaN 值的 Pandas 数组
    a = pd.array([1, 2, -3, np.nan])
    # 对两个 IntegerArrays 应用二元 ufunc 函数
    result = ufunc(a, a)
    # 使用浮点表示的 Pandas 数组作为参数，对 ufunc 函数应用二元操作，并指定返回类型为 Int64
    expected = pd.array(ufunc(a.astype(float), a.astype(float)), dtype="Int64")
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(result, expected)

    # 将一个 NumPy 数组与 IntegerArray 进行二元 ufunc 操作
    arr = np.array([1, 2, 3, 4])
    result = ufunc(a, arr)
    # 使用浮点表示的 Pandas 数组作为参数，对 ufunc 函数应用二元操作，并指定返回类型为 Int64
    expected = pd.array(ufunc(a.astype(float), arr), dtype="Int64")
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    # 使用浮点表示的 Pandas 数组作为参数，对 ufunc 函数应用二元操作，并指定返回类型为 Int64
    expected = pd.array(ufunc(arr, a.astype(float)), dtype="Int64")
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(result, expected)

    # 将 IntegerArray 与标量进行二元 ufunc 操作
    result = ufunc(a, 1)
    # 使用浮点表示的 Pandas 数组和标量作为参数，对 ufunc 函数应用二元操作，并指定返回类型为 Int64
    expected = pd.array(ufunc(a.astype(float), 1), dtype="Int64")
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(1, a)
    # 使用浮点表示的 Pandas 数组和标量作为参数，对 ufunc 函数应用二元操作，并指定返回类型为 Int64
    expected = pd.array(ufunc(1, a.astype(float)), dtype="Int64")
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    tm.assert_extension_array_equal(result, expected)


def test_ufunc_binary_output():
    # 创建一个包含整数和 NaN 值的 Pandas 数组
    a = pd.array([1, 2, np.nan])
    # 对 Pandas 数组应用 np.modf 函数
    result = np.modf(a)
    # 使用 to_numpy 方法将 Pandas 数组转换为 NumPy 数组，忽略 NaN 值，并指定返回类型为 float
    expected = np.modf(a.to_numpy(na_value=np.nan, dtype="float"))
    # 使用 Pandas 测试工具比较两个元组的内容是否相等
    expected = (pd.array(expected[0]), pd.array(expected[1]))
    assert isinstance(result, tuple)
    assert len(result) == 2
    # 使用 Pandas 测试工具比较两个扩展数组的内容是否相等
    for x, y in zip(result, expected):
        tm.assert_extension_array_equal(x, y)


@pytest.mark.parametrize("values", [[0, 1], [0, None]])
def test_ufunc_reduce_raises(values):
    # 创建一个包含整数和 None 值的 Pandas 数组
    arr = pd.array(values)
    # 对 Pandas 数组应用 np.add.reduce 函数
    res = np.add.reduce(arr)
    # 使用 Pandas 数组的 sum 方法计算其总和，不忽略 NaN 值
    expected = arr.sum(skipna=False)
    # 使用 Pandas 测试工具比较两个数值的近似性
    tm.assert_almost_equal(res, expected)


@pytest.mark.parametrize(
    "pandasmethname, kwargs",
    [
        # 计算方差（无偏估计）：参数ddof=0表示除以n
        ("var", {"ddof": 0}),
        # 计算方差（有偏估计）：参数ddof=1表示除以n-1
        ("var", {"ddof": 1}),
        # 计算标准差（无偏估计）：参数ddof=0表示除以n
        ("std", {"ddof": 0}),
        # 计算标准差（有偏估计）：参数ddof=1表示除以n-1
        ("std", {"ddof": 1}),
        # 计算峰度（样本数据）
        ("kurtosis", {}),
        # 计算偏度（样本数据）
        ("skew", {}),
        # 计算标准误差（样本数据）
        ("sem", {}),
    ],
# 定义测试统计方法的函数，接受 Pandas 方法名和关键字参数作为输入
def test_stat_method(pandasmethname, kwargs):
    # 创建一个包含整数和缺失值的 Pandas Series 对象
    s = pd.Series(data=[1, 2, 3, 4, 5, 6, np.nan, np.nan], dtype="Int64")
    # 获取指定名称的 Pandas 方法对象
    pandasmeth = getattr(s, pandasmethname)
    # 使用指定的关键字参数调用 Pandas 方法
    result = pandasmeth(**kwargs)
    # 创建另一个包含整数的 Pandas Series 对象
    s2 = pd.Series(data=[1, 2, 3, 4, 5, 6], dtype="Int64")
    # 获取相同名称的 Pandas 方法对象
    pandasmeth = getattr(s2, pandasmethname)
    # 使用相同的关键字参数调用 Pandas 方法，作为预期结果
    expected = pandasmeth(**kwargs)
    # 断言实际结果与预期结果相等
    assert expected == result


# 定义测试 value_counts 方法的函数，测试包含 NA 值的 Pandas Array
def test_value_counts_na():
    # 创建包含整数和 NA 值的 Pandas Array 对象
    arr = pd.array([1, 2, 1, pd.NA], dtype="Int64")
    # 调用 value_counts 方法，不丢弃 NA 值
    result = arr.value_counts(dropna=False)
    # 创建一个期望的索引对象，包含不丢弃 NA 值的所有可能值
    ex_index = pd.Index([1, 2, pd.NA], dtype="Int64")
    # 断言期望的索引对象的数据类型为 Int64
    assert ex_index.dtype == "Int64"
    # 创建一个期望的 Series 对象，包含每个值的计数，数据类型为 Int64
    expected = pd.Series([2, 1, 1], index=ex_index, dtype="Int64", name="count")
    # 使用测试工具 tm 检查实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)

    # 调用 value_counts 方法，丢弃 NA 值
    result = arr.value_counts(dropna=True)
    # 创建一个期望的 Series 对象，包含每个值的计数，索引与不含 NA 值的数组的前两个值相同，数据类型为 Int64
    expected = pd.Series([2, 1], index=arr[:2], dtype="Int64", name="count")
    # 断言期望的索引对象的数据类型与原数组的数据类型相同
    assert expected.index.dtype == arr.dtype
    # 使用测试工具 tm 检查实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)


# 定义测试空数组的 value_counts 方法的函数
def test_value_counts_empty():
    # 创建一个空的 Pandas Series 对象，数据类型为 Int64
    ser = pd.Series([], dtype="Int64")
    # 调用 value_counts 方法
    result = ser.value_counts()
    # 创建一个空的索引对象，数据类型与 ser 相同
    idx = pd.Index([], dtype=ser.dtype)
    # 断言空的索引对象的数据类型与 ser 的数据类型相同
    assert idx.dtype == ser.dtype
    # 创建一个期望的空 Series 对象，数据类型为 Int64
    expected = pd.Series([], index=idx, dtype="Int64", name="count")
    # 使用测试工具 tm 检查实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)


# 定义测试带有 normalize 参数的 value_counts 方法的函数
def test_value_counts_with_normalize():
    # 创建包含整数和 NA 值的 Pandas Series 对象
    ser = pd.Series([1, 2, 1, pd.NA], dtype="Int64")
    # 调用 value_counts 方法，计算标准化后的值
    result = ser.value_counts(normalize=True)
    # 创建一个期望的 Series 对象，包含每个值的比例，数据类型为 Float64，除以 3
    expected = pd.Series([2, 1], index=ser[:2], dtype="Float64", name="proportion") / 3
    # 断言期望的索引对象的数据类型与原 Series 的数据类型相同
    assert expected.index.dtype == ser.dtype
    # 使用测试工具 tm 检查实际结果与期望结果是否相等
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试整数数组的 sum 方法
@pytest.mark.parametrize("min_count", [0, 4])
def test_integer_array_sum(skipna, min_count, any_int_ea_dtype):
    # 获取任意整数数组的数据类型
    dtype = any_int_ea_dtype
    # 创建一个包含整数和 None 值的 Pandas Array 对象
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    # 调用 sum 方法，根据参数设置计算结果
    result = arr.sum(skipna=skipna, min_count=min_count)
    # 如果 skipna 为 True 并且 min_count 为 0，断言结果为 6
    if skipna and min_count == 0:
        assert result == 6
    # 否则，断言结果为 None
    else:
        assert result is pd.NA


# 使用参数化测试来测试整数数组的 min 和 max 方法
@pytest.mark.parametrize("method", ["min", "max"])
def test_integer_array_min_max(skipna, method, any_int_ea_dtype):
    # 获取任意整数数组的数据类型
    dtype = any_int_ea_dtype
    # 创建一个包含整数和 None 值的 Pandas Array 对象
    arr = pd.array([0, 1, None], dtype=dtype)
    # 获取相应的 min 或 max 方法
    func = getattr(arr, method)
    # 根据参数设置调用相应的方法
    result = func(skipna=skipna)
    # 如果 skipna 为 True，根据 method 的值判断结果
    if skipna:
        assert result == (0 if method == "min" else 1)
    # 否则，断言结果为 None
    else:
        assert result is pd.NA


# 使用参数化测试来测试整数数组的 prod 方法
@pytest.mark.parametrize("min_count", [0, 9])
def test_integer_array_prod(skipna, min_count, any_int_ea_dtype):
    # 获取任意整数数组的数据类型
    dtype = any_int_ea_dtype
    # 创建一个包含整数和 None 值的 Pandas Array 对象
    arr = pd.array([1, 2, None], dtype=dtype)
    # 调用 prod 方法，根据参数设置计算结果
    result = arr.prod(skipna=skipna, min_count=min_count)
    # 如果 skipna 为 True 并且 min_count 为 0，断言结果为 2
    if skipna and min_count == 0:
        assert result == 2
    # 否则，断言结果为 None
    else:
        assert result is pd.NA


# 使用参数化测试来测试整数数组的 numpy sum 方法
@pytest.mark.parametrize(
    "values, expected", [([1, 2, 3], 6), ([1, 2, 3, None], 6), ([None], 0)]
)
def test_integer_array_numpy_sum(values, expected):
    # 创建包含整数和 None 值的 Pandas Array 对象
    arr = pd.array(values, dtype="Int64")
    # 使用 numpy 的 sum 方法计算数组的和
    result = np.sum(arr)
    # 断言结果与预期值相等
    assert result == expected
# 定义一个测试函数，用于验证数据框架的降维操作
def test_dataframe_reductions(op):
    # 提供的 GitHub 链接指向的问题修复页面
    # 确保在降维过程中整数不被转换为浮点数
    # 创建一个包含整数数组的数据框架，指定整数类型为 Int64
    df = pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")})
    # 对数据框架进行最大值操作，返回结果
    result = df.max()
    # 断言结果中的 "a" 列数据类型是 np.int64 类型
    assert isinstance(result["a"], np.int64)


# TODO(jreback) - these need testing / are broken
# 待完成：这些需要进行测试 / 目前存在问题

# shift
# 数据框架的位移操作

# set_index (destroys type)
# 对数据框架执行 set_index 操作会破坏原有的数据类型
```