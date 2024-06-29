# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_constructors.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas._libs.sparse import IntIndex  # 导入 Sparse 库中的 IntIndex 类

import pandas as pd  # 导入 Pandas 库，并使用 pd 别名
from pandas import (  # 导入 Pandas 中的 SparseDtype、isna 函数
    SparseDtype,
    isna,
)
import pandas._testing as tm  # 导入 Pandas 测试工具模块
from pandas.core.arrays.sparse import SparseArray  # 导入 Pandas 中的 SparseArray 类


class TestConstructors:
    def test_constructor_dtype(self):
        arr = SparseArray([np.nan, 1, 2, np.nan])  # 创建 SparseArray 对象，存储 NaN 和整数
        assert arr.dtype == SparseDtype(np.float64, np.nan)  # 断言 SparseArray 对象的数据类型
        assert arr.dtype.subtype == np.float64  # 断言 SparseArray 对象数据类型的子类型
        assert np.isnan(arr.fill_value)  # 断言 SparseArray 对象的填充值为 NaN

        arr = SparseArray([np.nan, 1, 2, np.nan], fill_value=0)  # 使用指定填充值创建 SparseArray 对象
        assert arr.dtype == SparseDtype(np.float64, 0)  # 断言 SparseArray 对象的数据类型
        assert arr.fill_value == 0  # 断言 SparseArray 对象的填充值为 0

        arr = SparseArray([0, 1, 2, 4], dtype=np.float64)  # 创建指定数据类型的 SparseArray 对象
        assert arr.dtype == SparseDtype(np.float64, np.nan)  # 断言 SparseArray 对象的数据类型
        assert np.isnan(arr.fill_value)  # 断言 SparseArray 对象的填充值为 NaN

        arr = SparseArray([0, 1, 2, 4], dtype=np.int64)  # 创建指定数据类型的 SparseArray 对象
        assert arr.dtype == SparseDtype(np.int64, 0)  # 断言 SparseArray 对象的数据类型
        assert arr.fill_value == 0  # 断言 SparseArray 对象的填充值为 0

        arr = SparseArray([0, 1, 2, 4], fill_value=0, dtype=np.int64)  # 使用指定填充值和数据类型创建 SparseArray 对象
        assert arr.dtype == SparseDtype(np.int64, 0)  # 断言 SparseArray 对象的数据类型
        assert arr.fill_value == 0  # 断言 SparseArray 对象的填充值为 0

        arr = SparseArray([0, 1, 2, 4], dtype=None)  # 创建未指定数据类型的 SparseArray 对象
        assert arr.dtype == SparseDtype(np.int64, 0)  # 断言 SparseArray 对象的数据类型
        assert arr.fill_value == 0  # 断言 SparseArray 对象的填充值为 0

        arr = SparseArray([0, 1, 2, 4], fill_value=0, dtype=None)  # 使用指定填充值创建未指定数据类型的 SparseArray 对象
        assert arr.dtype == SparseDtype(np.int64, 0)  # 断言 SparseArray 对象的数据类型
        assert arr.fill_value == 0  # 断言 SparseArray 对象的填充值为 0

    def test_constructor_dtype_str(self):
        result = SparseArray([1, 2, 3], dtype="int")  # 创建指定数据类型的 SparseArray 对象
        expected = SparseArray([1, 2, 3], dtype=int)  # 创建指定数据类型的 SparseArray 对象
        tm.assert_sp_array_equal(result, expected)  # 使用测试工具函数比较两个 SparseArray 对象是否相等

    def test_constructor_sparse_dtype(self):
        result = SparseArray([1, 0, 0, 1], dtype=SparseDtype("int64", -1))  # 创建指定数据类型和填充值的 SparseArray 对象
        expected = SparseArray([1, 0, 0, 1], fill_value=-1, dtype=np.int64)  # 创建指定数据类型和填充值的 SparseArray 对象
        tm.assert_sp_array_equal(result, expected)  # 使用测试工具函数比较两个 SparseArray 对象是否相等
        assert result.sp_values.dtype == np.dtype("int64")  # 断言 SparseArray 对象的值数组数据类型为 int64

    def test_constructor_sparse_dtype_str(self):
        result = SparseArray([1, 0, 0, 1], dtype="Sparse[int32]")  # 创建指定数据类型的 SparseArray 对象
        expected = SparseArray([1, 0, 0, 1], dtype=np.int32)  # 创建指定数据类型的 SparseArray 对象
        tm.assert_sp_array_equal(result, expected)  # 使用测试工具函数比较两个 SparseArray 对象是否相等
        assert result.sp_values.dtype == np.dtype("int32")  # 断言 SparseArray 对象的值数组数据类型为 int32

    def test_constructor_object_dtype(self):
        # GH#11856
        arr = SparseArray(["A", "A", np.nan, "B"], dtype=object)  # 创建指定数据类型的 SparseArray 对象
        assert arr.dtype == SparseDtype(object)  # 断言 SparseArray 对象的数据类型为 object
        assert np.isnan(arr.fill_value)  # 断言 SparseArray 对象的填充值为 NaN

        arr = SparseArray(["A", "A", np.nan, "B"], dtype=object, fill_value="A")  # 创建指定数据类型和填充值的 SparseArray 对象
        assert arr.dtype == SparseDtype(object, "A")  # 断言 SparseArray 对象的数据类型为 object，并且填充值为 "A"
        assert arr.fill_value == "A"  # 断言 SparseArray 对象的填充值为 "A"
    # 定义测试函数，用于测试 SparseArray 的构造函数，验证 object 类型的数据、填充值功能
    def test_constructor_object_dtype_bool_fill(self):
        # GH#17574
        # 准备测试数据
        data = [False, 0, 100.0, 0.0]
        # 创建 SparseArray 对象，指定数据类型为 object，填充值为 False
        arr = SparseArray(data, dtype=object, fill_value=False)
        # 断言 SparseArray 对象的数据类型为 SparseDtype(object, False)
        assert arr.dtype == SparseDtype(object, False)
        # 断言 SparseArray 对象的填充值为 False
        assert arr.fill_value is False
        # 准备预期结果的 NumPy 数组
        arr_expected = np.array(data, dtype=object)
        # 比较 SparseArray 对象与预期结果的每个元素类型和值是否相等
        it = (type(x) == type(y) and x == y for x, y in zip(arr, arr_expected))
        # 断言比较结果中所有元素为 True
        assert np.fromiter(it, dtype=np.bool_).all()

    @pytest.mark.parametrize("dtype", [SparseDtype(int, 0), int])
    # 参数化测试函数，测试 SparseArray 构造函数对不支持的 dtype 参数的处理
    def test_constructor_na_dtype(self, dtype):
        # 使用 pytest 断言，期望抛出 ValueError 异常，并匹配指定错误信息
        with pytest.raises(ValueError, match="Cannot convert"):
            SparseArray([0, 1, np.nan], dtype=dtype)

    # 测试 SparseArray 构造函数在丢失时区信息时是否能发出警告
    def test_constructor_warns_when_losing_timezone(self):
        # GH#32501 warn when losing timezone information
        # 创建带时区信息的日期时间索引
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        # 准备预期的 SparseArray 对象，将日期时间索引转换为 datetime64[ns] 类型
        expected = SparseArray(np.asarray(dti, dtype="datetime64[ns]"))
        # 设置警告消息内容
        msg = "loses timezone information"
        # 使用 pytest 断言，期望在下面的操作中发出 UserWarning 警告，并匹配指定警告信息
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = SparseArray(dti)
        # 断言 SparseArray 对象与预期结果相等
        tm.assert_sp_array_equal(result, expected)

        # 使用 pytest 断言，期望在下面的操作中发出 UserWarning 警告，并匹配指定警告信息
        with tm.assert_produces_warning(UserWarning, match=msg):
            result = SparseArray(pd.Series(dti))
        # 断言 SparseArray 对象与预期结果相等
        tm.assert_sp_array_equal(result, expected)

    # 测试 SparseArray 构造函数在指定 SparseIndex 时的行为
    def test_constructor_spindex_dtype(self):
        # 创建带有指定 SparseIndex 的 SparseArray 对象，测试是否影响填充值
        arr = SparseArray(data=[1, 2], sparse_index=IntIndex(4, [1, 2]))
        # TODO: actionable?
        # XXX: Behavior change: specifying SparseIndex no longer changes the
        # fill_value
        # 准备预期结果的 SparseArray 对象，指定 kind="integer"
        expected = SparseArray([0, 1, 2, 0], kind="integer")
        # 使用 tm.assert_sp_array_equal 函数断言 SparseArray 对象与预期结果相等
        tm.assert_sp_array_equal(arr, expected)
        # 断言 SparseArray 对象的数据类型为 SparseDtype(np.int64)
        assert arr.dtype == SparseDtype(np.int64)
        # 断言 SparseArray 对象的填充值为 0
        assert arr.fill_value == 0

        # 创建带有指定 SparseIndex、dtype 和 fill_value 的 SparseArray 对象
        arr = SparseArray(
            data=[1, 2, 3],
            sparse_index=IntIndex(4, [1, 2, 3]),
            dtype=np.int64,
            fill_value=0,
        )
        # 准备预期的 SparseArray 对象，指定 dtype=np.int64 和 fill_value=0
        exp = SparseArray([0, 1, 2, 3], dtype=np.int64, fill_value=0)
        # 使用 tm.assert_sp_array_equal 函数断言 SparseArray 对象与预期结果相等
        tm.assert_sp_array_equal(arr, exp)
        # 断言 SparseArray 对象的数据类型为 SparseDtype(np.int64)
        assert arr.dtype == SparseDtype(np.int64)
        # 断言 SparseArray 对象的填充值为 0
        assert arr.fill_value == 0

        # 创建带有指定 SparseIndex 和 fill_value 的 SparseArray 对象，指定 dtype=np.int64
        arr = SparseArray(
            data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=np.int64
        )
        # 准备预期的 SparseArray 对象，指定 fill_value=0 和 dtype=np.int64
        exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=np.int64)
        # 使用 tm.assert_sp_array_equal 函数断言 SparseArray 对象与预期结果相等
        tm.assert_sp_array_equal(arr, exp)
        # 断言 SparseArray 对象的数据类型为 SparseDtype(np.int64)
        assert arr.dtype == SparseDtype(np.int64)
        # 断言 SparseArray 对象的填充值为 0

        # 创建带有指定 SparseIndex、dtype 和 fill_value 的 SparseArray 对象，dtype=None
        arr = SparseArray(
            data=[1, 2, 3],
            sparse_index=IntIndex(4, [1, 2, 3]),
            dtype=None,
            fill_value=0,
        )
        # 准备预期的 SparseArray 对象，指定 dtype=None
        exp = SparseArray([0, 1, 2, 3], dtype=None)
        # 使用 tm.assert_sp_array_equal 函数断言 SparseArray 对象与预期结果相等
        tm.assert_sp_array_equal(arr, exp)
        # 断言 SparseArray 对象的数据类型为 SparseDtype(np.int64)
        assert arr.dtype == SparseDtype(np.int64)
        # 断言 SparseArray 对象的填充值为 0

    @pytest.mark.parametrize("sparse_index", [None, IntIndex(1, [0])])
    # 测试用例：测试稀疏数组构造函数处理标量输入的情况
    def test_constructor_spindex_dtype_scalar(self, sparse_index):
        # 定义错误消息，指出不能从标量数据构造稀疏数组，应传入一个序列
        msg = "Cannot construct SparseArray from scalar data. Pass a sequence instead"
        # 使用 pytest 断言异常 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            SparseArray(data=1, sparse_index=sparse_index, dtype=None)

        # 同样使用 pytest 断言异常 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            SparseArray(data=1, sparse_index=IntIndex(1, [0]), dtype=None)

    # 测试用例：测试稀疏数组构造函数处理标量输入并广播的情况
    def test_constructor_spindex_dtype_scalar_broadcasts(self):
        # 使用 SparseArray 构造稀疏数组，data 为 [1, 2]，sparse_index 为 IntIndex(4, [1, 2])
        arr = SparseArray(
            data=[1, 2], sparse_index=IntIndex(4, [1, 2]), fill_value=0, dtype=None
        )
        # 期望的稀疏数组，数据为 [0, 1, 2, 0]，fill_value 为 0，dtype 为 None
        exp = SparseArray([0, 1, 2, 0], fill_value=0, dtype=None)
        # 使用测试工具函数 tm.assert_sp_array_equal 检查 arr 与 exp 是否相等
        tm.assert_sp_array_equal(arr, exp)
        # 使用普通 assert 检查 arr 的 dtype 是否为 SparseDtype(np.int64)
        assert arr.dtype == SparseDtype(np.int64)
        # 使用普通 assert 检查 arr 的 fill_value 是否为 0
        assert arr.fill_value == 0

    # 参数化测试：测试不同数据和填充值的情况
    @pytest.mark.parametrize(
        "data, fill_value",
        [
            (np.array([1, 2]), 0),
            (np.array([1.0, 2.0]), np.nan),
            ([True, False], False),
            ([pd.Timestamp("2017-01-01")], pd.NaT),
        ],
    )
    def test_constructor_inferred_fill_value(self, data, fill_value):
        # 使用 SparseArray 构造稀疏数组，并获取其 fill_value 属性
        result = SparseArray(data).fill_value

        # 如果 fill_value 是 NaN（not a number），则使用 assert isna 检查 result 是否为 NaN
        if isna(fill_value):
            assert isna(result)
        # 否则，使用普通 assert 检查 result 是否等于 fill_value
        else:
            assert result == fill_value

    # 参数化测试：测试从 scipy 稀疏矩阵构造稀疏数组的情况
    @pytest.mark.parametrize("format", ["coo", "csc", "csr"])
    @pytest.mark.parametrize("size", [0, 10])
    def test_from_spmatrix(self, size, format):
        # 导入 scipy.sparse 模块并进行必要的检查
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 生成指定格式和大小的稀疏矩阵 mat
        mat = sp_sparse.random(size, 1, density=0.5, format=format)
        # 使用 SparseArray.from_spmatrix 从稀疏矩阵构造稀疏数组
        result = SparseArray.from_spmatrix(mat)

        # 将 result 转换为 numpy 数组，并与期望的数组进行比较
        result = np.asarray(result)
        expected = mat.toarray().ravel()
        tm.assert_numpy_array_equal(result, expected)

    # 参数化测试：测试包含显式零值的 scipy 稀疏矩阵构造稀疏数组的情况
    @pytest.mark.parametrize("format", ["coo", "csc", "csr"])
    def test_from_spmatrix_including_explicit_zero(self, format):
        # 导入 scipy.sparse 模块并进行必要的检查
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 生成指定格式的稀疏矩阵 mat，并将第一个数据元素设为零
        mat = sp_sparse.random(10, 1, density=0.5, format=format)
        mat.data[0] = 0
        # 使用 SparseArray.from_spmatrix 从稀疏矩阵构造稀疏数组
        result = SparseArray.from_spmatrix(mat)

        # 将 result 转换为 numpy 数组，并与期望的数组进行比较
        result = np.asarray(result)
        expected = mat.toarray().ravel()
        tm.assert_numpy_array_equal(result, expected)

    # 测试用例：测试 SparseArray.from_spmatrix 对不兼容的稀疏矩阵引发异常的情况
    def test_from_spmatrix_raises(self):
        # 导入 scipy.sparse 模块并进行必要的检查
        sp_sparse = pytest.importorskip("scipy.sparse")

        # 生成 5x4 的单位稀疏矩阵 mat（格式为 csc）
        mat = sp_sparse.eye(5, 4, format="csc")

        # 使用 pytest 断言异常 ValueError，并匹配指定的错误消息
        with pytest.raises(ValueError, match="not '4'"):
            SparseArray.from_spmatrix(mat)

    # 测试用例：测试构造函数对过大的数组引发异常的情况
    def test_constructor_from_too_large_array(self):
        # 使用 pytest 断言异常 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match="expected dimension <= 1 data"):
            SparseArray(np.arange(10).reshape((2, 5)))

    # 测试用例：测试从另一个稀疏数组构造稀疏数组的情况
    def test_constructor_from_sparse(self):
        # 使用 SparseArray 构造稀疏数组 zarr
        zarr = SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)
        # 使用 SparseArray 构造新的稀疏数组 res，以 zarr 为参数
        res = SparseArray(zarr)
        # 使用普通 assert 检查 res 的 fill_value 是否为 0
        assert res.fill_value == 0
        # 使用测试工具函数 tm.assert_almost_equal 检查 res 的 sp_values 是否与 zarr 的相等
        tm.assert_almost_equal(res.sp_values, zarr.sp_values)
    # 定义一个测试方法，用于测试 SparseArray 类的拷贝构造函数
    def test_constructor_copy(self):
        # 创建一个包含 NaN 值的 NumPy 数组
        arr_data = np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 6])
        # 使用 arr_data 创建 SparseArray 对象 arr
        arr = SparseArray(arr_data)

        # 使用拷贝构造函数创建 SparseArray 对象 cp，复制 arr 的内容
        cp = SparseArray(arr, copy=True)
        # 修改 cp 的前三个稀疏值为 0
        cp.sp_values[:3] = 0
        # 断言 arr 的前三个稀疏值不为 0
        assert not (arr.sp_values[:3] == 0).any()

        # 使用默认构造函数创建 SparseArray 对象 not_copy，共享 arr 的数据
        not_copy = SparseArray(arr)
        # 修改 not_copy 的前三个稀疏值为 0
        not_copy.sp_values[:3] = 0
        # 断言 arr 的前三个稀疏值都为 0
        assert (arr.sp_values[:3] == 0).all()

    # 定义一个测试方法，用于测试 SparseArray 类在布尔类型数据上的行为
    def test_constructor_bool(self):
        # 创建一个包含布尔值的 NumPy 数组
        data = np.array([False, False, True, True, False, False])
        # 使用 fill_value 和 dtype 参数创建 SparseArray 对象 arr
        arr = SparseArray(data, fill_value=False, dtype=bool)

        # 断言 arr 的 dtype 是 SparseDtype(bool)
        assert arr.dtype == SparseDtype(bool)
        # 断言 arr 的稀疏值与预期的稀疏值数组相等
        tm.assert_numpy_array_equal(arr.sp_values, np.array([True, True]))
        # 断言 arr 的稀疏索引 indices 与预期的索引数组相等
        tm.assert_numpy_array_equal(arr.sp_index.indices, np.array([2, 3], np.int32))

        # 将 arr 转换为密集形式的数组 dense
        dense = arr.to_dense()
        # 断言 dense 的 dtype 是 bool
        assert dense.dtype == bool
        # 断言 dense 与原始数据 data 相等
        tm.assert_numpy_array_equal(dense, data)

    # 定义一个测试方法，用于测试 SparseArray 类在布尔类型数据上的行为，包含填充值的情况
    def test_constructor_bool_fill_value(self):
        # 使用默认参数创建 SparseArray 对象 arr
        arr = SparseArray([True, False, True], dtype=None)
        # 断言 arr 的 dtype 是 SparseDtype(np.bool_)
        assert arr.dtype == SparseDtype(np.bool_)
        # 断言 arr 的 fill_value 是 False
        assert not arr.fill_value

        # 使用 dtype 参数创建 SparseArray 对象 arr
        arr = SparseArray([True, False, True], dtype=np.bool_)
        # 断言 arr 的 dtype 是 SparseDtype(np.bool_)
        assert arr.dtype == SparseDtype(np.bool_)
        # 断言 arr 的 fill_value 是 False
        assert not arr.fill_value

        # 使用 dtype 和 fill_value 参数创建 SparseArray 对象 arr
        arr = SparseArray([True, False, True], dtype=np.bool_, fill_value=True)
        # 断言 arr 的 dtype 是 SparseDtype(np.bool_, True)
        assert arr.dtype == SparseDtype(np.bool_, True)
        # 断言 arr 的 fill_value 是 True
        assert arr.fill_value

    # 定义一个测试方法，用于测试 SparseArray 类在 float32 类型数据上的行为
    def test_constructor_float32(self):
        # 创建一个包含 float32 类型数据的 NumPy 数组
        data = np.array([1.0, np.nan, 3], dtype=np.float32)
        # 使用 dtype 参数创建 SparseArray 对象 arr
        arr = SparseArray(data, dtype=np.float32)

        # 断言 arr 的 dtype 是 SparseDtype(np.float32)
        assert arr.dtype == SparseDtype(np.float32)
        # 断言 arr 的稀疏值与预期的稀疏值数组相等
        tm.assert_numpy_array_equal(arr.sp_values, np.array([1, 3], dtype=np.float32))
        # 断言 arr 的稀疏索引 indices 与预期的索引数组相等
        tm.assert_numpy_array_equal(arr.sp_index.indices, np.array([0, 2], dtype=np.int32))

        # 将 arr 转换为密集形式的数组 dense
        dense = arr.to_dense()
        # 断言 dense 的 dtype 是 np.float32
        assert dense.dtype == np.float32
        # 断言 dense 与原始数据 data 相等
        tm.assert_numpy_array_equal(dense, data)
```