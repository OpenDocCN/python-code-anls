# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_array.py`

```
# 导入正则表达式模块
import re

# 导入 NumPy 库，并使用别名 np
import numpy as np

# 导入 pytest 测试框架
import pytest

# 从 pandas._libs.sparse 中导入 IntIndex 类
from pandas._libs.sparse import IntIndex

# 导入 pandas 库，并使用别名 pd
import pandas as pd

# 从 pandas 中导入 SparseDtype、isna 函数
from pandas import (
    SparseDtype,
    isna,
)

# 导入 pandas._testing 模块，并使用别名 tm
import pandas._testing as tm

# 从 pandas.core.arrays.sparse 中导入 SparseArray 类
from pandas.core.arrays.sparse import SparseArray


# 定义一个 pytest 的 fixture，返回包含有效和缺失条目的 NumPy 数组
@pytest.fixture
def arr_data():
    """Fixture returning numpy array with valid and missing entries"""
    return np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 6])


# 定义一个 pytest 的 fixture，返回从 arr_data 创建的 SparseArray 对象
@pytest.fixture
def arr(arr_data):
    """Fixture returning SparseArray from 'arr_data'"""
    return SparseArray(arr_data)


# 定义一个 pytest 的 fixture，返回具有整数条目和 'fill_value=0' 的 SparseArray 对象
@pytest.fixture
def zarr():
    """Fixture returning SparseArray with integer entries and 'fill_value=0'"""
    return SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)


# 定义一个测试类 TestSparseArray
class TestSparseArray:
    
    # 使用 pytest.mark.parametrize 标记的参数化测试，测试 shift 方法的 fill_value 参数
    @pytest.mark.parametrize("fill_value", [0, None, np.nan])
    def test_shift_fill_value(self, fill_value):
        # GH #24128
        
        # 创建 SparseArray 对象 sparse，初始值为 [1, 0, 0, 3, 0]，填充值为 8.0
        sparse = SparseArray(np.array([1, 0, 0, 3, 0]), fill_value=8.0)
        
        # 调用 shift 方法，将 sparse 向右移动 1 个位置，使用 fill_value 参数
        res = sparse.shift(1, fill_value=fill_value)
        
        # 如果 fill_value 是 NaN，则将 fill_value 设置为 res 的 dtype 的 na_value
        if isna(fill_value):
            fill_value = res.dtype.na_value
        
        # 创建期望的 SparseArray 对象 exp，初始值为 [fill_value, 1, 0, 0, 3]，填充值为 8.0
        exp = SparseArray(np.array([fill_value, 1, 0, 0, 3]), fill_value=8.0)
        
        # 使用 pandas._testing 模块的 assert_sp_array_equal 方法比较 res 和 exp 是否相等
        tm.assert_sp_array_equal(res, exp)

    # 测试 set_fill_value 方法
    def test_set_fill_value(self):
        # 创建浮点数类型的 SparseArray 对象 arr，初始值为 [1.0, NaN, 2.0]，填充值为 NaN
        arr = SparseArray([1.0, np.nan, 2.0], fill_value=np.nan)
        
        # 设置 arr 的 fill_value 为 2
        arr.fill_value = 2
        # 断言 arr 的 fill_value 是否为 2
        assert arr.fill_value == 2

        # 创建整数类型的 SparseArray 对象 arr，初始值为 [1, 0, 2]，填充值为 0，dtype 为 np.int64
        arr = SparseArray([1, 0, 2], fill_value=0, dtype=np.int64)
        
        # 设置 arr 的 fill_value 为 2
        arr.fill_value = 2
        # 断言 arr 的 fill_value 是否为 2
        assert arr.fill_value == 2

        # 如果设置的 fill_value 不是 SparseDtype.subtype 的有效值，则抛出 ValueError 异常
        msg = "fill_value must be a valid value for the SparseDtype.subtype"
        with pytest.raises(ValueError, match=msg):
            # GH#53043
            arr.fill_value = 3.1
        # 再次断言 arr 的 fill_value 是否为 2
        assert arr.fill_value == 2

        # 设置 arr 的 fill_value 为 NaN
        arr.fill_value = np.nan
        # 断言 arr 的 fill_value 是否为 NaN
        assert np.isnan(arr.fill_value)

        # 创建布尔类型的 SparseArray 对象 arr，初始值为 [True, False, True]，填充值为 False，dtype 为 np.bool_
        arr = SparseArray([True, False, True], fill_value=False, dtype=np.bool_)
        
        # 设置 arr 的 fill_value 为 True
        arr.fill_value = True
        # 断言 arr 的 fill_value 是否为 True
        assert arr.fill_value is True

        # 如果设置的 fill_value 不是 SparseDtype.subtype 的有效值，则抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            arr.fill_value = 0
        # 断言 arr 的 fill_value 是否为 True
        assert arr.fill_value is True

        # 设置 arr 的 fill_value 为 NaN
        arr.fill_value = np.nan
        # 断言 arr 的 fill_value 是否为 NaN
        assert np.isnan(arr.fill_value)

    # 使用 pytest.mark.parametrize 标记的参数化测试，测试设置非标量 fill_value 的情况
    @pytest.mark.parametrize("val", [[1, 2, 3], np.array([1, 2]), (1, 2, 3)])
    def test_set_fill_invalid_non_scalar(self, val):
        # 创建布尔类型的 SparseArray 对象 arr，初始值为 [True, False, True]，填充值为 False，dtype 为 np.bool_
        arr = SparseArray([True, False, True], fill_value=False, dtype=np.bool_)
        
        # 设置非标量的 fill_value 值，预期会抛出 ValueError 异常
        msg = "fill_value must be a scalar"
        with pytest.raises(ValueError, match=msg):
            arr.fill_value = val

    # 测试 SparseArray 对象的复制方法 copy()
    def test_copy(self, arr):
        # 调用 arr 的 copy() 方法，复制出一个新的 SparseArray 对象 arr2
        arr2 = arr.copy()
        
        # 断言 arr2 的 sp_values 不是 arr 的 sp_values
        assert arr2.sp_values is not arr.sp_values
        
        # 断言 arr2 的 sp_index 是 arr 的 sp_index
        assert arr2.sp_index is arr.sp_index

    # 测试 SparseArray 对象的 to_dense() 方法
    def test_values_asarray(self, arr_data, arr):
        # 使用 pandas._testing 模块的 assert_almost_equal 方法，比较 arr 转为稠密数组后的值与 arr_data 是否近似相等
        tm.assert_almost_equal(arr.to_dense(), arr_data)

    # 使用 pytest.mark.parametrize 标记的参数化测试，测试不同数据、形状和 dtype 的 SparseArray 对象的创建
    @pytest.mark.parametrize(
        "data,shape,dtype",
        [
            ([0, 0, 0, 0, 0], (5,), None),
            ([], (0,), None),
            ([0], (1,), None),
            (["A", "A", np.nan, "B"], (4,), object),
        ],
    )
    # 定义测试方法，验证 SparseArray 对象的形状是否符合预期
    def test_shape(self, data, shape, dtype):
        # 创建 SparseArray 对象，使用给定的数据和数据类型
        out = SparseArray(data, dtype=dtype)
        # 断言 SparseArray 对象的形状与给定的形状相等
        assert out.shape == shape

    # 使用 pytest 的参数化装饰器，定义多组参数进行测试
    @pytest.mark.parametrize(
        "vals",
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [1, np.nan, np.nan, 3, np.nan],
            [1, np.nan, 0, 3, 0],
        ],
    )
    @pytest.mark.parametrize("fill_value", [None, 0])
    # 测试 SparseArray 对象的稠密表示是否正确
    def test_dense_repr(self, vals, fill_value):
        # 将参数转换为 NumPy 数组
        vals = np.array(vals)
        # 创建 SparseArray 对象，使用给定的值和填充值
        arr = SparseArray(vals, fill_value=fill_value)

        # 转换为稠密数组
        res = arr.to_dense()
        # 断言稠密数组与原始值数组相等
        tm.assert_numpy_array_equal(res, vals)

    # 使用 pytest 的参数化装饰器，定义多组参数进行测试
    @pytest.mark.parametrize("fix", ["arr", "zarr"])
    # 测试 SparseArray 对象的 pickle 序列化和反序列化
    def test_pickle(self, fix, request):
        # 获取指定名称的 fixture 对象
        obj = request.getfixturevalue(fix)
        # 对象进行 pickle 的往返测试
        unpickled = tm.round_trip_pickle(obj)
        # 断言反序列化后的对象与原始对象相等
        tm.assert_sp_array_equal(unpickled, obj)

    # 测试生成器警告是否正常工作
    def test_generator_warnings(self):
        # 创建 SparseArray 对象
        sp_arr = SparseArray([1, 2, 3])
        # 验证在特定条件下是否会产生警告
        with tm.assert_produces_warning(None):
            for _ in sp_arr:
                pass

    # 测试 _where 方法在保留填充值时的行为
    def test_where_retain_fill_value(self):
        # 创建带有填充值的 SparseArray 对象
        # GH#45691 表明在 _where 方法中不应丢失填充值
        arr = SparseArray([np.nan, 1.0], fill_value=0)

        # 创建布尔掩码
        mask = np.array([True, False])

        # 使用 _where 方法进行条件替换
        res = arr._where(~mask, 1)
        # 预期的 SparseArray 对象
        exp = SparseArray([1, 1.0], fill_value=0)
        # 断言结果与预期相等
        tm.assert_sp_array_equal(res, exp)

        # 创建 Pandas Series 对象
        ser = pd.Series(arr)
        # 使用 Pandas 的 where 方法进行条件替换
        res = ser.where(~mask, 1)
        # 断言 Series 结果与预期的 Series 对象相等
        tm.assert_series_equal(res, pd.Series(exp))
    # 测试 SparseArray 对象的 fillna 方法
    def test_fillna(self):
        # 创建一个包含 NaN 值的 SparseArray 对象
        s = SparseArray([1, np.nan, np.nan, 3, np.nan])
        # 调用 fillna 方法，将 NaN 值填充为 -1
        res = s.fillna(-1)
        # 创建预期的 SparseArray 对象，填充值为 -1，数据类型为 np.float64
        exp = SparseArray([1, -1, -1, 3, -1], fill_value=-1, dtype=np.float64)
        # 断言填充后的结果与预期结果相等
        tm.assert_sp_array_equal(res, exp)

        # 创建一个指定 fill_value 的 SparseArray 对象，并再次调用 fillna 方法
        s = SparseArray([1, np.nan, np.nan, 3, np.nan], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([1, -1, -1, 3, -1], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        # 创建包含 0 和 NaN 值的 SparseArray 对象，并调用 fillna 方法
        s = SparseArray([1, np.nan, 0, 3, 0])
        res = s.fillna(-1)
        exp = SparseArray([1, -1, 0, 3, 0], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        # 创建指定 fill_value 的 SparseArray 对象，并再次调用 fillna 方法
        s = SparseArray([1, np.nan, 0, 3, 0], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([1, -1, 0, 3, 0], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        # 创建只包含 NaN 值的 SparseArray 对象，并调用 fillna 方法
        s = SparseArray([np.nan, np.nan, np.nan, np.nan])
        res = s.fillna(-1)
        exp = SparseArray([-1, -1, -1, -1], fill_value=-1, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        # 创建指定 fill_value 的 SparseArray 对象，并再次调用 fillna 方法
        s = SparseArray([np.nan, np.nan, np.nan, np.nan], fill_value=0)
        res = s.fillna(-1)
        exp = SparseArray([-1, -1, -1, -1], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)

        # 创建只包含 0.0 值的 SparseArray 对象，并调用 fillna 方法
        s = SparseArray([0.0, 0.0, 0.0, 0.0])
        res = s.fillna(-1)
        exp = SparseArray([0.0, 0.0, 0.0, 0.0], fill_value=-1)
        tm.assert_sp_array_equal(res, exp)

        # 创建只包含整数值的 SparseArray 对象，并调用 fillna 方法
        s = SparseArray([0, 0, 0, 0])
        assert s.dtype == SparseDtype(np.int64)
        assert s.fill_value == 0
        res = s.fillna(-1)
        # 因为数据类型为整数，填充 NaN 不会改变 SparseArray 对象
        tm.assert_sp_array_equal(res, s)

        # 创建指定 fill_value 的整数类型 SparseArray 对象，并再次调用 fillna 方法
        s = SparseArray([0, 0, 0, 0], fill_value=0)
        assert s.dtype == SparseDtype(np.int64)
        assert s.fill_value == 0
        res = s.fillna(-1)
        exp = SparseArray([0, 0, 0, 0], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        # 创建指定 fill_value 为 NaN 的 SparseArray 对象，并调用 fillna 方法
        s = SparseArray([0, 0, 0, 0], fill_value=np.nan)
        assert s.dtype == SparseDtype(np.int64, fill_value=np.nan)
        assert np.isnan(s.fill_value)
        res = s.fillna(-1)
        exp = SparseArray([0, 0, 0, 0], fill_value=-1)
        tm.assert_sp_array_equal(res, exp)

    # 测试 SparseArray 对象的 fillna 方法，特别是填充值与现有值重叠的情况
    def test_fillna_overlap(self):
        # 创建包含 NaN 值的 SparseArray 对象，并调用 fillna 方法
        s = SparseArray([1, np.nan, np.nan, 3, np.nan])
        # 使用现有值 3 填充 NaN，不会用 fill_value 替换现有值
        res = s.fillna(3)
        # 创建预期的 numpy 数组，将 SparseArray 转为稠密数组后进行比较
        exp = np.array([1, 3, 3, 3, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(res.to_dense(), exp)

        # 创建指定 fill_value 的 SparseArray 对象，并调用 fillna 方法
        s = SparseArray([1, np.nan, np.nan, 3, np.nan], fill_value=0)
        res = s.fillna(3)
        exp = SparseArray([1, 3, 3, 3, 3], fill_value=0, dtype=np.float64)
        tm.assert_sp_array_equal(res, exp)
    def test_nonzero(self):
        # 测试回归问题 #21172
        # 创建稀疏数组，包含 NaN 值和整数值
        sa = SparseArray([float("nan"), float("nan"), 1, 0, 0, 2, 0, 0, 0, 3, 0, 0])
        # 期望的非零元素索引数组，数据类型为 np.int32
        expected = np.array([2, 5, 9], dtype=np.int32)
        # 调用 SparseArray 对象的 nonzero 方法，获取非零元素的索引
        (result,) = sa.nonzero()
        # 使用测试框架的函数验证 numpy 数组相等性
        tm.assert_numpy_array_equal(expected, result)

        # 创建另一个稀疏数组，用于进一步测试
        sa = SparseArray([0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0])
        # 再次调用 nonzero 方法，获取非零元素的索引
        (result,) = sa.nonzero()
        # 使用测试框架的函数验证 numpy 数组相等性
        tm.assert_numpy_array_equal(expected, result)
class TestSparseArrayAnalytics:
    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                np.array([1, 2, 3, 4, 5], dtype=float),  # non-null data
                SparseArray(np.array([1.0, 3.0, 6.0, 10.0, 15.0])),
            ),
            (
                np.array([1, 2, np.nan, 4, 5], dtype=float),  # null data
                SparseArray(np.array([1.0, 3.0, np.nan, 7.0, 12.0])),
            ),
        ],
    )
    @pytest.mark.parametrize("numpy", [True, False])
    def test_cumsum(self, data, expected, numpy):
        # 定义累积求和函数，根据是否使用 NumPy 决定使用 np.cumsum 还是 lambda 函数
        cumsum = np.cumsum if numpy else lambda s: s.cumsum()

        # 测试 SparseArray 对象的累积求和，并断言结果是否与期望值相等
        out = cumsum(SparseArray(data))
        tm.assert_sp_array_equal(out, expected)

        # 使用指定的填充值测试 SparseArray 对象的累积求和
        out = cumsum(SparseArray(data, fill_value=np.nan))
        tm.assert_sp_array_equal(out, expected)

        # 使用不同的填充值测试 SparseArray 对象的累积求和
        out = cumsum(SparseArray(data, fill_value=2))
        tm.assert_sp_array_equal(out, expected)

        if numpy:  # 对于 NumPy，进行额外的兼容性检查
            msg = "the 'dtype' parameter is not supported"
            # 使用 pytest 的断言检查是否抛出 ValueError 异常，并验证异常消息是否匹配
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), dtype=np.int64)

            msg = "the 'out' parameter is not supported"
            # 使用 pytest 的断言检查是否抛出 ValueError 异常，并验证异常消息是否匹配
            with pytest.raises(ValueError, match=msg):
                np.cumsum(SparseArray(data), out=out)
        else:
            axis = 1  # SparseArray 目前是一维的，因此只有 axis = 0 是有效的
            msg = re.escape(f"axis(={axis}) out of bounds")
            # 使用 pytest 的断言检查是否抛出 ValueError 异常，并验证异常消息是否匹配
            with pytest.raises(ValueError, match=msg):
                SparseArray(data).cumsum(axis=axis)

    def test_ufunc(self):
        # GH 13853 确保 ufunc 应用于 fill_value
        # 创建 SparseArray 对象并验证其绝对值是否与预期结果相等
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray([1, np.nan, 2, np.nan, 2])
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)

        # 使用指定的填充值创建 SparseArray 对象，并验证其绝对值是否与预期结果相等
        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray([1, 2, 2], sparse_index=sparse.sp_index, fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), result)
        tm.assert_sp_array_equal(np.abs(sparse), result)

        # 使用不同的填充值创建 SparseArray 对象，并验证其绝对值是否与预期结果相等
        sparse = SparseArray([1, -1, 2, -2], fill_value=-1)
        exp = SparseArray([1, 1, 2, 2], fill_value=1)
        tm.assert_sp_array_equal(abs(sparse), exp)
        tm.assert_sp_array_equal(np.abs(sparse), exp)

        # 创建 SparseArray 对象并验证其 sin 函数应用后的结果是否与预期结果相等
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        result = SparseArray(np.sin([1, np.nan, 2, np.nan, -2]))
        tm.assert_sp_array_equal(np.sin(sparse), result)

        # 使用指定的填充值创建 SparseArray 对象，并验证其 sin 函数应用后的结果是否与预期结果相等
        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        result = SparseArray(np.sin([1, -1, 2, -2]), fill_value=np.sin(1))
        tm.assert_sp_array_equal(np.sin(sparse), result)

        # 使用不同的填充值创建 SparseArray 对象，并验证其 sin 函数应用后的结果是否与预期结果相等
        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        result = SparseArray(np.sin([1, -1, 0, -2]), fill_value=np.sin(0))
        tm.assert_sp_array_equal(np.sin(sparse), result)
    # 定义一个测试方法，验证稀疏数组的ufunc应用于fill_value，包括其参数
    def test_ufunc_args(self):
        # 创建一个包含NaN值的稀疏数组
        sparse = SparseArray([1, np.nan, 2, np.nan, -2])
        # 预期的结果稀疏数组
        result = SparseArray([2, np.nan, 3, np.nan, -1])
        # 断言稀疏数组加1后是否与预期结果相等
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

        # 创建一个带有自定义fill_value的稀疏数组
        sparse = SparseArray([1, -1, 2, -2], fill_value=1)
        # 预期的结果稀疏数组
        result = SparseArray([2, 0, 3, -1], fill_value=2)
        # 断言稀疏数组加1后是否与预期结果相等
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

        # 创建一个带有不同fill_value的稀疏数组
        sparse = SparseArray([1, -1, 0, -2], fill_value=0)
        # 预期的结果稀疏数组
        result = SparseArray([2, 0, 1, -1], fill_value=1)
        # 断言稀疏数组加1后是否与预期结果相等
        tm.assert_sp_array_equal(np.add(sparse, 1), result)

    # 使用pytest的参数化标记定义一个测试方法，验证modf函数的稀疏数组行为
    @pytest.mark.parametrize("fill_value", [0.0, np.nan])
    def test_modf(self, fill_value):
        # 创建一个带有自定义fill_value的稀疏数组
        sparse = SparseArray([fill_value] * 10 + [1.1, 2.2], fill_value=fill_value)
        # 对稀疏数组应用np.modf函数，得到小数部分和整数部分
        r1, r2 = np.modf(sparse)
        # 使用np.asarray转换稀疏数组，再应用np.modf函数，得到预期的小数部分和整数部分
        e1, e2 = np.modf(np.asarray(sparse))
        # 断言计算得到的小数部分是否与预期结果相等
        tm.assert_sp_array_equal(r1, SparseArray(e1, fill_value=fill_value))
        # 断言计算得到的整数部分是否与预期结果相等
        tm.assert_sp_array_equal(r2, SparseArray(e2, fill_value=fill_value))

    # 定义一个测试方法，验证整数类型稀疏数组的nbytes计算
    def test_nbytes_integer(self):
        # 创建一个整数类型的稀疏数组
        arr = SparseArray([1, 0, 0, 0, 2], kind="integer")
        # 计算稀疏数组的字节大小
        result = arr.nbytes
        # 断言计算得到的字节大小是否等于预期结果
        assert result == 24

    # 定义一个测试方法，验证块类型稀疏数组的nbytes计算
    def test_nbytes_block(self):
        # 创建一个块类型的稀疏数组
        arr = SparseArray([1, 2, 0, 0, 0], kind="block")
        # 计算稀疏数组的字节大小
        result = arr.nbytes
        # 断言计算得到的字节大小是否等于预期结果
        assert result == 24

    # 定义一个测试方法，验证将稀疏数组转换为datetime64数组的行为
    def test_asarray_datetime64(self):
        # 创建一个包含日期时间对象的稀疏数组
        s = SparseArray(pd.to_datetime(["2012", None, None, "2013"]))
        # 将稀疏数组转换为numpy数组
        np.asarray(s)

    # 定义一个测试方法，验证稀疏数组的密度计算
    def test_density(self):
        # 创建一个稀疏数组
        arr = SparseArray([0, 1])
        # 断言稀疏数组的密度是否等于预期值
        assert arr.density == 0.5

    # 定义一个测试方法，验证稀疏数组的非零元素数量计算
    def test_npoints(self):
        # 创建一个稀疏数组
        arr = SparseArray([0, 1])
        # 断言稀疏数组的非零元素数量是否等于预期值
        assert arr.npoints == 1
# 定义一个测试函数，用于验证在修改 fill_value 后调用 fillna 仍然有效
def test_setting_fill_value_fillna_still_works():
    # 这就是为什么允许用户更新 fill_value / dtype 是不好的原因
    # astype 也有同样的问题
    arr = SparseArray([1.0, np.nan, 1.0], fill_value=0.0)
    # 修改 SparseArray 的 fill_value 属性为 np.nan
    arr.fill_value = np.nan
    # 调用 isna() 方法检查稀疏数组中的缺失值情况，结果赋给 result
    result = arr.isna()
    # 由于稀疏数组的索引可能不同，不能直接进行比较
    # 因此将结果转换为 ndarray 后再进行比较
    result = np.asarray(result)

    # 期望的结果是一个布尔 ndarray
    expected = np.array([False, True, False])
    # 使用 assert_numpy_array_equal 断言比较结果和期望值
    tm.assert_numpy_array_equal(result, expected)


# 定义一个测试函数，用于验证修改 fill_value 后的更新操作
def test_setting_fill_value_updates():
    arr = SparseArray([0.0, np.nan], fill_value=0)
    # 修改 SparseArray 的 fill_value 属性为 np.nan
    arr.fill_value = np.nan
    # 使用私有构造函数 _simple_new 来确保索引正确
    # 否则两个 NaN 值都不会被存储
    expected = SparseArray._simple_new(
        sparse_array=np.array([np.nan]),
        sparse_index=IntIndex(2, [1]),
        dtype=SparseDtype(float, np.nan),
    )
    # 使用 assert_sp_array_equal 断言比较 arr 和 expected 是否相等
    tm.assert_sp_array_equal(arr, expected)


# 使用 pytest.mark.parametrize 运行参数化测试
@pytest.mark.parametrize(
    "arr,fill_value,loc",
    [
        ([None, 1, 2], None, 0),
        ([0, None, 2], None, 1),
        ([0, 1, None], None, 2),
        ([0, 1, 1, None, None], None, 3),
        ([1, 1, 1, 2], None, -1),
        ([], None, -1),
        ([None, 1, 0, 0, None, 2], None, 0),
        ([None, 1, 0, 0, None, 2], 1, 1),
        ([None, 1, 0, 0, None, 2], 2, 5),
        ([None, 1, 0, 0, None, 2], 3, -1),
        ([None, 0, 0, 1, 2, 1], 0, 1),
        ([None, 0, 0, 1, 2, 1], 1, 3),
    ],
)
# 定义参数化测试函数，测试 SparseArray 的 _first_fill_value_loc 方法
def test_first_fill_value_loc(arr, fill_value, loc):
    # 调用 SparseArray 构造函数创建 arr，设置 fill_value
    # 调用 _first_fill_value_loc 方法获取结果
    result = SparseArray(arr, fill_value=fill_value)._first_fill_value_loc()
    # 使用 assert 断言检查 result 是否等于预期的 loc 值
    assert result == loc


# 使用 pytest.mark.parametrize 运行参数化测试
@pytest.mark.parametrize(
    "arr",
    [
        [1, 2, np.nan, np.nan],
        [1, np.nan, 2, np.nan],
        [1, 2, np.nan],
        [np.nan, 1, 0, 0, np.nan, 2],
        [np.nan, 0, 0, 1, 2, 1],
    ],
)
# 运行参数化测试函数，测试 SparseArray 的 unique 方法
@pytest.mark.parametrize("fill_value", [np.nan, 0, 1])
def test_unique_na_fill(arr, fill_value):
    # 创建 SparseArray 对象 a，并调用 unique 方法
    a = SparseArray(arr, fill_value=fill_value).unique()
    # 创建 Pandas Series 对象 b，并调用 unique 方法
    b = pd.Series(arr).unique()
    # 使用 assert 检查 a 是否为 SparseArray 类型
    assert isinstance(a, SparseArray)
    # 将 SparseArray 对象 a 转换为 ndarray，再和 b 比较
    a = np.asarray(a)
    tm.assert_numpy_array_equal(a, b)


# 定义测试函数，用于验证全是稀疏值的 unique 方法
def test_unique_all_sparse():
    # 创建只包含稀疏值的 SparseArray 对象 arr
    arr = SparseArray([0, 0])
    # 调用 unique 方法获取结果
    result = arr.unique()
    # 创建预期的 SparseArray 对象 expected
    expected = SparseArray([0])
    # 使用 assert_sp_array_equal 断言比较 result 和 expected 是否相等
    tm.assert_sp_array_equal(result, expected)


# 定义测试函数，用于验证 map 方法的不同用法
def test_map():
    arr = SparseArray([0, 1, 2])
    expected = SparseArray([10, 11, 12], fill_value=10)

    # 使用字典作为映射
    result = arr.map({0: 10, 1: 11, 2: 12})
    tm.assert_sp_array_equal(result, expected)

    # 使用 Pandas Series 对象作为映射
    result = arr.map(pd.Series({0: 10, 1: 11, 2: 12}))
    tm.assert_sp_array_equal(result, expected)

    # 使用函数作为映射
    result = arr.map(pd.Series({0: 10, 1: 11, 2: 12}))
    expected = SparseArray([10, 11, 12], fill_value=10)
    tm.assert_sp_array_equal(result, expected)


# 定义测试函数，用于验证 map 方法在缺失映射值时的行为
def test_map_missing():
    arr = SparseArray([0, 1, 2])
    expected = SparseArray([10, 11, None], fill_value=10)

    # 只提供部分映射值，缺失的映射值将使用 fill_value 填充
    result = arr.map({0: 10, 1: 11})
    # 使用测试工具模块 `tm` 中的函数 `assert_sp_array_equal` 来比较变量 `result` 和 `expected` 是否相等
    tm.assert_sp_array_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器为 test_dropna 函数添加参数化测试，测试 fill_value 参数为 np.nan 和 1 时的情况
@pytest.mark.parametrize("fill_value", [np.nan, 1])
def test_dropna(fill_value):
    # GH-28287
    # 创建 SparseArray 对象 arr，填充值为 fill_value，包含元素 np.nan 和 1
    arr = SparseArray([np.nan, 1], fill_value=fill_value)
    # 创建期望的 SparseArray 对象 exp，填充值与 arr 相同，但仅包含元素 1.0
    exp = SparseArray([1.0], fill_value=fill_value)
    # 使用 tm.assert_sp_array_equal 断言 arr.dropna() 返回的 SparseArray 与 exp 相等
    tm.assert_sp_array_equal(arr.dropna(), exp)

    # 创建 DataFrame df，包含两列 'a' 和 'b'，其中 'b' 列为 SparseArray 对象 arr
    df = pd.DataFrame({"a": [0, 1], "b": arr})
    # 创建期望的 DataFrame expected_df，与 df 结构相同，但 'b' 列的 SparseArray 对象为 exp，且索引为 1
    expected_df = pd.DataFrame({"a": [1], "b": exp}, index=pd.Index([1]))
    # 使用 tm.assert_equal 断言 df.dropna() 返回的 DataFrame 与 expected_df 相等
    tm.assert_equal(df.dropna(), expected_df)


# 定义测试函数 test_drop_duplicates_fill_value，用于测试 DataFrame 的 drop_duplicates 方法
def test_drop_duplicates_fill_value():
    # GH 11726
    # 创建一个 5x5 的 DataFrame，每列都转换为 SparseArray 对象，填充值为 0
    df = pd.DataFrame(np.zeros((5, 5))).apply(lambda x: SparseArray(x, fill_value=0))
    # 对 DataFrame 调用 drop_duplicates 方法，返回结果 DataFrame result
    result = df.drop_duplicates()
    # 创建期望的 DataFrame expected，每列都包含一个 SparseArray 对象，只包含元素 0.0
    expected = pd.DataFrame({i: SparseArray([0.0], fill_value=0) for i in range(5)})
    # 使用 tm.assert_frame_equal 断言 result 与 expected 相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_zero_sparse_column，用于测试包含 SparseArray 列的 DataFrame 操作
def test_zero_sparse_column():
    # GH 27781
    # 创建 DataFrame df1，包含两列 'A' 和 'B'，其中 'A' 列为全零的 SparseArray 对象，'B' 列为普通的数值列
    df1 = pd.DataFrame({"A": SparseArray([0, 0, 0]), "B": [1, 2, 3]})
    # 创建 DataFrame df2，结构与 df1 相同，但 'A' 列的 SparseArray 对象第二行值为 1
    df2 = pd.DataFrame({"A": SparseArray([0, 1, 0]), "B": [1, 2, 3]})
    # 使用 df1["B"] != 2 条件筛选 df1 的行，结果赋给 result
    result = df1.loc[df1["B"] != 2]
    # 使用 df2["B"] != 2 条件筛选 df2 的行，结果赋给 expected
    expected = df2.loc[df2["B"] != 2]
    # 使用 tm.assert_frame_equal 断言 result 与 expected 相等
    tm.assert_frame_equal(result, expected)

    # 创建期望的 DataFrame expected，仅包含 df1 中 'A' 列第一行和第三行的值，'B' 列为 1 和 3
    expected = pd.DataFrame({"A": SparseArray([0, 0]), "B": [1, 3]}, index=[0, 2])
    # 再次使用 tm.assert_frame_equal 断言 result 与 expected 相等
    tm.assert_frame_equal(result, expected)
```