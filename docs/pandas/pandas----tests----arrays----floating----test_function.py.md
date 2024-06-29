# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\test_function.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 pandas 兼容模块中导入 IS64 变量
from pandas.compat import IS64

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 导入 pandas 内部测试模块
import pandas._testing as tm

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试 numpy 的绝对值和符号函数
@pytest.mark.parametrize("ufunc", [np.abs, np.sign])
# 忽略 np.sign 函数在处理 NaN 值时触发的 RuntimeWarning 警告
@pytest.mark.filterwarnings("ignore:invalid value encountered in sign:RuntimeWarning")
def test_ufuncs_single(ufunc):
    # 创建一个包含 NaN 的 pandas 浮点数数组
    a = pd.array([1, 2, -3, np.nan], dtype="Float64")
    # 对给定的 ufunc 函数应用到数组 a 上
    result = ufunc(a)
    # 生成期望的结果，将 ufunc 应用到 a 的浮点表示上
    expected = pd.array(ufunc(a.astype(float)), dtype="Float64")
    # 使用测试工具函数检查扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 创建一个 pandas Series 对象
    s = pd.Series(a)
    # 对 Series 对象应用 ufunc 函数
    result = ufunc(s)
    # 生成期望的 Series 对象，将 ufunc 应用到 s 的浮点表示上
    expected = pd.Series(expected)
    # 使用测试工具函数检查 Series 是否相等
    tm.assert_series_equal(result, expected)

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试 numpy 的对数、指数、正弦、余弦和平方根函数
@pytest.mark.parametrize("ufunc", [np.log, np.exp, np.sin, np.cos, np.sqrt])
def test_ufuncs_single_float(ufunc):
    # 创建一个包含 NaN 的 pandas 浮点数数组
    a = pd.array([1.0, 0.2, 3.0, np.nan], dtype="Float64")
    # 在忽略无效值的错误状态下，对给定的 ufunc 函数应用到数组 a 上
    with np.errstate(invalid="ignore"):
        result = ufunc(a)
        # 生成期望的结果，将 ufunc 应用到 a 的浮点表示上
        expected = pd.array(ufunc(a.astype(float)), dtype="Float64")
    # 使用测试工具函数检查扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 创建一个 pandas Series 对象
    s = pd.Series(a)
    # 在忽略无效值的错误状态下，对 Series 对象应用 ufunc 函数
    with np.errstate(invalid="ignore"):
        result = ufunc(s)
        # 生成期望的 Series 对象，将 ufunc 应用到 s 的浮点表示上
        expected = pd.Series(ufunc(s.astype(float)), dtype="Float64")
    # 使用测试工具函数检查 Series 是否相等
    tm.assert_series_equal(result, expected)

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试 numpy 的加法和减法函数
@pytest.mark.parametrize("ufunc", [np.add, np.subtract])
def test_ufuncs_binary_float(ufunc):
    # 创建一个包含 NaN 的 pandas 浮点数数组
    a = pd.array([1, 0.2, -3, np.nan], dtype="Float64")
    # 对给定的 ufunc 函数应用到数组 a 上
    result = ufunc(a, a)
    # 生成期望的结果，将 ufunc 应用到 a 和 a 的浮点表示上
    expected = pd.array(ufunc(a.astype(float), a.astype(float)), dtype="Float64")
    # 使用测试工具函数检查扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 创建一个 numpy 数组
    arr = np.array([1, 2, 3, 4])
    # 对给定的 ufunc 函数应用到数组 a 和 arr 上
    result = ufunc(a, arr)
    # 生成期望的结果，将 ufunc 应用到 a 的浮点表示和 arr 上
    expected = pd.array(ufunc(a.astype(float), arr), dtype="Float64")
    # 使用测试工具函数检查扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 对给定的 ufunc 函数应用到 arr 和 a 上
    result = ufunc(arr, a)
    # 生成期望的结果，将 ufunc 应用到 arr 和 a 的浮点表示上
    expected = pd.array(ufunc(arr, a.astype(float)), dtype="Float64")
    # 使用测试工具函数检查扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 对给定的 ufunc 函数应用到数组 a 和标量 1 上
    result = ufunc(a, 1)
    # 生成期望的结果，将 ufunc 应用到 a 的浮点表示和标量 1 上
    expected = pd.array(ufunc(a.astype(float), 1), dtype="Float64")
    # 使用测试工具函数检查扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

    # 对给定的 ufunc 函数应用到标量 1 和数组 a 上
    result = ufunc(1, a)
    # 生成期望的结果，将 ufunc 应用到标量 1 和 a 的浮点表示上
    expected = pd.array(ufunc(1, a.astype(float)), dtype="Float64")
    # 使用测试工具函数检查扩展数组是否相等
    tm.assert_extension_array_equal(result, expected)

# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试 pandas 浮点数数组的 reduce 操作引发异常的情况
@pytest.mark.parametrize("values", [[0, 1], [0, None]])
def test_ufunc_reduce_raises(values):
    # 创建一个包含指定值的 pandas 浮点数数组
    arr = pd.array(values, dtype="Float64")

    # 对数组 arr 应用 np.add.reduce 函数
    res = np.add.reduce(arr)
    # 生成期望的结果，对 arr 执行 skipna=False 的求和操作
    expected = arr.sum(skipna=False)
    # 使用测试工具函数检查结果是否近似相等
    tm.assert_almost_equal(res, expected)

# 使用 pytest.mark.skipif 装饰器定义条件测试，仅在 IS64 为 True 时执行
@pytest.mark.skipif(not IS64, reason="GH 36579: fail on 32-bit system")
@pytest.mark.parametrize(
    "pandasmethname, kwargs",
    [
        ("var", {"ddof": 0}),
        ("var", {"ddof": 1}),
        ("std", {"ddof": 0}),
        ("std", {"ddof": 1}),
        ("kurtosis", {}),
        ("skew", {}),
        ("sem", {}),
    ],
)
def test_stat_method(pandasmethname, kwargs):
    # 创建一个 Pandas Series 对象，包含浮点数和 NaN 值
    s = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan, np.nan], dtype="Float64")
    
    # 使用 getattr 函数获取 Pandas Series 对象 s 的方法 pandasmethname，并将其赋给变量 pandasmeth
    pandasmeth = getattr(s, pandasmethname)
    
    # 调用获取到的方法 pandasmeth，使用传入的关键字参数 kwargs 执行操作，将结果赋给变量 result
    result = pandasmeth(**kwargs)
    
    # 创建另一个 Pandas Series 对象 s2，包含浮点数数据
    s2 = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype="float64")
    
    # 使用 getattr 函数获取 Pandas Series 对象 s2 的方法 pandasmethname，并将其赋给变量 pandasmeth
    pandasmeth = getattr(s2, pandasmethname)
    
    # 调用获取到的方法 pandasmeth，使用传入的关键字参数 kwargs 执行操作，将结果赋给变量 expected
    expected = pandasmeth(**kwargs)
    
    # 断言语句，验证变量 expected 和 result 的值是否相等
    assert expected == result
# 测试处理缺失值的情况，使用 Pandas 的 array 结构创建数组 arr
def test_value_counts_na():
    arr = pd.array([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    # 对数组 arr 进行值计数，包括缺失值
    result = arr.value_counts(dropna=False)
    # 创建包含指定值的索引 idx，保持与 arr 相同的数据类型
    idx = pd.Index([0.1, 0.2, pd.NA], dtype=arr.dtype)
    # 断言索引 idx 的数据类型与数组 arr 的数据类型相同
    assert idx.dtype == arr.dtype
    # 创建预期的 Series 对象 expected，包含值的计数结果
    expected = pd.Series([2, 1, 1], index=idx, dtype="Int64", name="count")
    # 使用测试工具 tm 来比较 result 和 expected 的一致性
    tm.assert_series_equal(result, expected)

    # 再次对数组 arr 进行值计数，忽略缺失值
    result = arr.value_counts(dropna=True)
    # 创建预期的 Series 对象 expected，包含忽略缺失值后的计数结果
    expected = pd.Series([2, 1], index=idx[:-1], dtype="Int64", name="count")
    # 使用测试工具 tm 来比较 result 和 expected 的一致性
    tm.assert_series_equal(result, expected)


# 测试处理空数组的情况
def test_value_counts_empty():
    # 使用 Pandas 创建一个空的 Series 对象 ser
    ser = pd.Series([], dtype="Float64")
    # 对空的 Series 对象 ser 进行值计数
    result = ser.value_counts()
    # 创建空的索引 idx，保持数据类型为 Float64
    idx = pd.Index([], dtype="Float64")
    # 断言索引 idx 的数据类型为 Float64
    assert idx.dtype == "Float64"
    # 创建预期的空的 Series 对象 expected
    expected = pd.Series([], index=idx, dtype="Int64", name="count")
    # 使用测试工具 tm 来比较 result 和 expected 的一致性
    tm.assert_series_equal(result, expected)


# 测试带有 normalize 参数的值计数情况
def test_value_counts_with_normalize():
    # 使用 Pandas 创建一个带有缺失值的 Series 对象 ser
    ser = pd.Series([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    # 对带有 normalize 参数的 Series 对象 ser 进行值计数
    result = ser.value_counts(normalize=True)
    # 创建预期的 Series 对象 expected，包含计数结果的归一化比例
    expected = pd.Series([2, 1], index=ser[:2], dtype="Float64", name="proportion") / 3
    # 断言预期的 Series 对象 expected 的索引数据类型与 ser 相同
    assert expected.index.dtype == ser.dtype
    # 使用测试工具 tm 来比较 result 和 expected 的一致性
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试浮点数组的求和情况
@pytest.mark.parametrize("skipna, min_count, dtype", [(True, 0, "Float64"), (False, 0, "Float64")])
def test_floating_array_sum(skipna, min_count, dtype):
    # 使用 Pandas 的 array 结构创建浮点数组 arr，包含一个 None 值
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    # 对浮点数组 arr 进行求和操作，根据 skipna 和 min_count 参数决定是否跳过缺失值
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        # 如果 skipna=True 且 min_count=0，则断言结果等于 6.0
        assert result == 6.0
    else:
        # 否则断言结果为缺失值 pd.NA
        assert result is pd.NA


# 使用参数化测试来测试浮点数组的 NumPy 求和情况
@pytest.mark.parametrize("values, expected", [([1, 2, 3], 6.0), ([1, 2, 3, None], 6.0), ([None], 0.0)])
def test_floating_array_numpy_sum(values, expected):
    # 使用 Pandas 的 array 结构创建浮点数组 arr
    arr = pd.array(values, dtype="Float64")
    # 使用 NumPy 对浮点数组 arr 进行求和操作
    result = np.sum(arr)
    # 断言求和结果与预期值 expected 相等
    assert result == expected


# 使用参数化测试来测试浮点数组的汇总操作（如求和、最小值、最大值、乘积）
@pytest.mark.parametrize("op", ["sum", "min", "max", "prod"])
def test_preserve_dtypes(op):
    # 使用 Pandas 创建一个包含不同数据类型的 DataFrame 对象 df
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([0.1, None, 3.0], dtype="Float64"),
        }
    )

    # 对 DataFrame 的列 df.C 执行指定的汇总操作 op
    result = getattr(df.C, op)()
    # 断言汇总结果的数据类型为 np.float64
    assert isinstance(result, np.float64)

    # 对 DataFrame 按列 "A" 分组后执行指定的汇总操作 op
    result = getattr(df.groupby("A"), op)()

    # 创建预期的 DataFrame 对象 expected，包含分组后的汇总结果
    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([0.1, 3], dtype="Float64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    # 使用测试工具 tm 来比较 result 和 expected 的一致性
    tm.assert_frame_equal(result, expected)


# 使用参数化测试来测试浮点数组的最小值和最大值情况
@pytest.mark.parametrize("skipna, method, dtype", [(True, "min", "Float64"), (False, "min", "Float64"),
                                                   (True, "max", "Float64"), (False, "max", "Float64")])
def test_floating_array_min_max(skipna, method, dtype):
    # 使用 Pandas 的 array 结构创建浮点数组 arr，包含一个 None 值
    arr = pd.array([0.0, 1.0, None], dtype=dtype)
    # 获取 arr 对象的指定方法（min 或 max）
    func = getattr(arr, method)
    # 对浮点数组 arr 进行最小值或最大值计算，根据 skipna 参数决定是否跳过缺失值
    result = func(skipna=skipna)
    if skipna:
        # 如果 skipna=True，则断言结果为最小值或最大值（0 或 1）
        assert result == (0 if method == "min" else 1)
    else:
        # 否则断言结果为缺失值 pd.NA
        assert result is pd.NA


# 使用参数化测试来测试浮点数组的乘积情况
@pytest.mark.parametrize("skipna, min_count, dtype", [(True, 0, "Float64"), (False, 0, "Float64")])
def test_floating_array_prod(skipna, min_count, dtype):
    # 使用 Pandas 的 array 结构创建浮点数组 arr，包含一个 None 值
    arr = pd.array([1.0, 2.0, None], dtype=dtype)
    # 对浮点数组 arr 进行乘积计算，根据 skipna 和 min_count 参数决定是否跳过缺失值
    result = arr.prod(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        # 如果 skipna=True 且 min_count=0，则断言结果等于 2
        assert result == 2
    else:
        # 否则断言结果为缺失值 pd.NA
        assert result is pd.NA
```