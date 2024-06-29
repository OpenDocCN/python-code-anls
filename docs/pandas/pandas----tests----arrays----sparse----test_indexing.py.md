# `D:\src\scipysrc\pandas\pandas\tests\arrays\sparse\test_indexing.py`

```
# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 pytest 库
import pytest

# 导入 pandas 库，并从中导入 SparseDtype 类
import pandas as pd
from pandas import SparseDtype
# 导入 pandas 内部测试模块
import pandas._testing as tm
# 从 pandas.core.arrays.sparse 模块导入 SparseArray 类
from pandas.core.arrays.sparse import SparseArray


# 定义 arr_data fixture，返回一个包含 NaN 值的 numpy 数组
@pytest.fixture
def arr_data():
    return np.array([np.nan, np.nan, 1, 2, 3, np.nan, 4, 5, np.nan, 6])


# 定义 arr fixture，使用 arr_data fixture 创建 SparseArray 对象
@pytest.fixture
def arr(arr_data):
    return SparseArray(arr_data)


# 定义 TestGetitem 类，用于测试 SparseArray 的索引功能
class TestGetitem:
    # 测试索引取值功能
    def test_getitem(self, arr):
        # 将 SparseArray 转换为稠密数组
        dense = arr.to_dense()
        # 遍历 SparseArray，对比每个值是否与稠密数组中的值几乎相等
        for i, value in enumerate(arr):
            tm.assert_almost_equal(value, dense[i])
            # 对比负索引的值是否与稠密数组中的负索引值几乎相等
            tm.assert_almost_equal(arr[-i], dense[-i])

    # 测试索引取值功能（使用数组形式的布尔掩码）
    def test_getitem_arraylike_mask(self, arr):
        # 创建 SparseArray 对象
        arr = SparseArray([0, 1, 2])
        # 使用布尔掩码进行索引操作
        result = arr[[True, False, True]]
        # 期望的 SparseArray 结果
        expected = SparseArray([0, 2])
        tm.assert_sp_array_equal(result, expected)

    # 使用 pytest 的参数化装饰器定义多个测试用例，测试切片功能
    @pytest.mark.parametrize(
        "slc",
        [
            np.s_[:],
            np.s_[1:10],
            np.s_[1:100],
            np.s_[10:1],
            np.s_[:-3],
            np.s_[-5:-4],
            np.s_[:-12],
            np.s_[-12:],
            np.s_[2:],
            np.s_[2::3],
            np.s_[::2],
            np.s_[::-1],
            np.s_[::-2],
            np.s_[1:6:2],
            np.s_[:-6:-2],
        ],
    )
    # 使用 pytest 的参数化装饰器定义多个测试用例，测试 as_dense 参数
    @pytest.mark.parametrize(
        "as_dense", [[np.nan] * 10, [1] * 10, [np.nan] * 5 + [1] * 5, []]
    )
    # 测试切片功能
    def test_getslice(self, slc, as_dense):
        # 将 as_dense 转换为 numpy 数组
        as_dense = np.array(as_dense)
        # 创建 SparseArray 对象
        arr = SparseArray(as_dense)

        # 对 SparseArray 进行切片操作
        result = arr[slc]
        # 期望的 SparseArray 结果
        expected = SparseArray(as_dense[slc])

        tm.assert_sp_array_equal(result, expected)

    # 测试元组形式的切片功能
    def test_getslice_tuple(self):
        # 创建稠密数组
        dense = np.array([np.nan, 0, 3, 4, 0, 5, np.nan, np.nan, 0])

        # 创建 SparseArray 对象
        sparse = SparseArray(dense)
        # 执行切片操作
        res = sparse[(slice(4, None),)]
        # 期望的 SparseArray 结果
        exp = SparseArray(dense[4:])
        tm.assert_sp_array_equal(res, exp)

        # 创建 SparseArray 对象，指定填充值为 0
        sparse = SparseArray(dense, fill_value=0)
        # 执行切片操作
        res = sparse[(slice(4, None),)]
        # 期望的 SparseArray 结果
        exp = SparseArray(dense[4:], fill_value=0)
        tm.assert_sp_array_equal(res, exp)

        # 检查索引错误情况
        msg = "too many indices for array"
        with pytest.raises(IndexError, match=msg):
            sparse[4:, :]

        with pytest.raises(IndexError, match=msg):
            # 检查 numpy 兼容性
            dense[4:, :]

    # 测试布尔型切片为空的情况
    def test_boolean_slice_empty(self):
        # 创建 SparseArray 对象
        arr = SparseArray([0, 1, 2])
        # 使用布尔型切片
        res = arr[[False, False, False]]
        # 断言结果的数据类型与原 SparseArray 相同
        assert res.dtype == arr.dtype
    # 测试用例：使用布尔类型的稀疏数组索引
    def test_getitem_bool_sparse_array(self, arr):
        # 创建一个布尔类型的稀疏数组，交替包含 False 和 True，并设置数据类型为 np.bool_，填充值为 True
        spar_bool = SparseArray([False, True] * 5, dtype=np.bool_, fill_value=True)
        # 期望的结果，创建一个包含 np.nan、2、np.nan、5、6 的稀疏数组
        exp = SparseArray([np.nan, 2, np.nan, 5, 6])
        # 使用稀疏数组 spar_bool 对 arr 进行索引，比较结果是否与 exp 相等
        tm.assert_sp_array_equal(arr[spar_bool], exp)

        # 将 spar_bool 取反
        spar_bool = ~spar_bool
        # 使用取反后的 spar_bool 对 arr 进行索引，期望得到包含 np.nan、1、3、4、np.nan 的稀疏数组
        res = arr[spar_bool]
        exp = SparseArray([np.nan, 1, 3, 4, np.nan])
        # 检查取反索引后的结果是否与期望一致
        tm.assert_sp_array_equal(res, exp)

        # 创建另一个布尔类型的稀疏数组，包含 False、True、np.nan 交替出现三次，并设置数据类型为 np.bool_，填充值为 np.nan
        spar_bool = SparseArray(
            [False, True, np.nan] * 3, dtype=np.bool_, fill_value=np.nan
        )
        # 使用这个稀疏数组对 arr 进行索引，期望得到包含 np.nan、3、5 的稀疏数组
        res = arr[spar_bool]
        exp = SparseArray([np.nan, 3, 5])
        # 检查结果是否与期望一致
        tm.assert_sp_array_equal(res, exp)

    # 测试用例：使用布尔类型的稀疏数组作为比较条件索引
    def test_getitem_bool_sparse_array_as_comparison(self):
        # 创建一个填充值为 np.nan 的稀疏数组 arr，包含数字 1、2、3、4，以及两个 np.nan
        arr = SparseArray([1, 2, 3, 4, np.nan, np.nan], fill_value=np.nan)
        # 使用条件 arr > 2 对 arr 进行索引，期望得到包含 3.0、4.0 的稀疏数组
        res = arr[arr > 2]
        exp = SparseArray([3.0, 4.0], fill_value=np.nan)
        # 检查条件索引后的结果是否与期望一致
        tm.assert_sp_array_equal(res, exp)

    # 测试用例：常规索引测试
    def test_get_item(self, arr):
        # 创建一个填充值为 0 的稀疏数组 zarr，包含数字 0、0、1、2、3、0、4、5、0、6
        zarr = SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)

        # 检查索引 1 的结果是否为 np.nan
        assert np.isnan(arr[1])
        # 检查索引 2 的结果是否为 1
        assert arr[2] == 1
        # 检查索引 7 的结果是否为 5
        assert arr[7] == 5

        # 检查索引 0 的结果是否为 0
        assert zarr[0] == 0
        # 检查索引 2 的结果是否为 1
        assert zarr[2] == 1
        # 检查索引 7 的结果是否为 5
        assert zarr[7] == 5

        # 准备错误信息字符串
        errmsg = "must be an integer between -10 and 10"

        # 使用 pytest 检查索引 11 是否引发 IndexError 异常，且异常信息是否匹配 errmsg
        with pytest.raises(IndexError, match=errmsg):
            arr[11]

        # 使用 pytest 检查索引 -11 是否引发 IndexError 异常，且异常信息是否匹配 errmsg
        with pytest.raises(IndexError, match=errmsg):
            arr[-11]

        # 检查索引 -1 的结果是否与 arr[len(arr) - 1] 的结果相等
        assert arr[-1] == arr[len(arr) - 1]
class TestSetitem:
    # 测试设置元素操作的类
    def test_set_item(self, arr_data):
        # 创建稀疏数组，并进行复制操作
        arr = SparseArray(arr_data).copy()

        # 定义设置单个元素的函数
        def setitem():
            arr[5] = 3

        # 定义设置切片范围的函数
        def setslice():
            arr[1:5] = 2

        # 使用 pytest 检查设置单个元素操作是否抛出 TypeError 异常，并匹配特定错误信息
        with pytest.raises(TypeError, match="assignment via setitem"):
            setitem()

        # 使用 pytest 检查设置切片范围操作是否抛出 TypeError 异常，并匹配特定错误信息
        with pytest.raises(TypeError, match="assignment via setitem"):
            setslice()


class TestTake:
    # 测试取值操作的类
    def test_take_scalar_raises(self, arr):
        # 错误信息内容
        msg = "'indices' must be an array, not a scalar '2'."
        # 使用 pytest 检查当传入标量而非数组时是否抛出 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match=msg):
            arr.take(2)

    def test_take(self, arr_data, arr):
        # 创建预期稀疏数组，并比较取值操作的结果是否正确
        exp = SparseArray(np.take(arr_data, [2, 3]))
        tm.assert_sp_array_equal(arr.take([2, 3]), exp)

        # 创建预期稀疏数组，并比较取值操作的结果是否正确
        exp = SparseArray(np.take(arr_data, [0, 1, 2]))
        tm.assert_sp_array_equal(arr.take([0, 1, 2]), exp)

    def test_take_all_empty(self):
        # 创建空的稀疏数组，并测试 take 操作是否能正确处理填充值
        sparse = pd.array([0, 0], dtype=SparseDtype("int64"))
        result = sparse.take([0, 1], allow_fill=True, fill_value=np.nan)
        tm.assert_sp_array_equal(sparse, result)

    def test_take_different_fill_value(self):
        # 使用不同的填充值进行 take 操作，确保原始数组不会被覆盖
        sparse = pd.array([0.0], dtype=SparseDtype("float64", fill_value=0.0))
        result = sparse.take([0, -1], allow_fill=True, fill_value=np.nan)
        expected = pd.array([0, np.nan], dtype=sparse.dtype)
        tm.assert_sp_array_equal(expected, result)

    def test_take_fill_value(self):
        # 创建带有填充值的稀疏数组，并比较 take 操作的结果是否正确
        data = np.array([1, np.nan, 0, 3, 0])
        sparse = SparseArray(data, fill_value=0)

        exp = SparseArray(np.take(data, [0]), fill_value=0)
        tm.assert_sp_array_equal(sparse.take([0]), exp)

        exp = SparseArray(np.take(data, [1, 3, 4]), fill_value=0)
        tm.assert_sp_array_equal(sparse.take([1, 3, 4]), exp)

    def test_take_negative(self, arr_data, arr):
        # 创建预期稀疏数组，并比较取负索引的操作结果是否正确
        exp = SparseArray(np.take(arr_data, [-1]))
        tm.assert_sp_array_equal(arr.take([-1]), exp)

        # 创建预期稀疏数组，并比较取负索引的操作结果是否正确
        exp = SparseArray(np.take(arr_data, [-4, -3, -2]))
        tm.assert_sp_array_equal(arr.take([-4, -3, -2]), exp)

    def test_bad_take(self, arr):
        # 使用 pytest 检查当索引超出边界时是否抛出 IndexError 异常，并匹配特定错误信息
        with pytest.raises(IndexError, match="bounds"):
            arr.take([11])
    def test_take_filling(self):
        # similar tests as GH 12631
        # 创建一个稀疏数组，包含NaN值，用于测试
        sparse = SparseArray([np.nan, np.nan, 1, np.nan, 4])
        # 调用稀疏数组的take方法，传入索引数组，返回结果
        result = sparse.take(np.array([1, 0, -1]))
        # 创建预期的稀疏数组，用于测试结果
        expected = SparseArray([np.nan, np.nan, 4])
        # 断言结果与预期相等
        tm.assert_sp_array_equal(result, expected)

        # TODO: actionable?
        # XXX: test change: fill_value=True -> allow_fill=True
        # 调用稀疏数组的take方法，传入索引数组和allow_fill参数，返回结果
        result = sparse.take(np.array([1, 0, -1]), allow_fill=True)
        # 创建预期的稀疏数组，用于测试结果
        expected = SparseArray([np.nan, np.nan, np.nan])
        # 断言结果与预期相等
        tm.assert_sp_array_equal(result, expected)

        # allow_fill=False
        # 调用稀疏数组的take方法，传入索引数组和allow_fill、fill_value参数，返回结果
        result = sparse.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        # 创建预期的稀疏数组，用于测试结果
        expected = SparseArray([np.nan, np.nan, 4])
        # 断言结果与预期相等
        tm.assert_sp_array_equal(result, expected)

        # 在索引数组包含无效值时，抛出值错误异常
        msg = "Invalid value in 'indices'"
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -2]), allow_fill=True)

        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -5]), allow_fill=True)

        # 在索引数组包含超出边界值时，抛出索引错误异常
        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), allow_fill=True)

    def test_take_filling_fill_value(self):
        # same tests as GH#12631
        # 创建一个填充值为0的稀疏数组，用于测试
        sparse = SparseArray([np.nan, 0, 1, 0, 4], fill_value=0)
        # 调用稀疏数组的take方法，传入索引数组，返回结果
        result = sparse.take(np.array([1, 0, -1]))
        # 创建预期的稀疏数组，用于测试结果
        expected = SparseArray([0, np.nan, 4], fill_value=0)
        # 断言结果与预期相等
        tm.assert_sp_array_equal(result, expected)

        # fill_value
        # 调用稀疏数组的take方法，传入索引数组和allow_fill参数，返回结果
        result = sparse.take(np.array([1, 0, -1]), allow_fill=True)
        # TODO: actionable?
        # XXX: behavior change.
        # the old way of filling self.fill_value doesn't follow EA rules.
        # It's supposed to be self.dtype.na_value (nan in this case)
        # 创建预期的稀疏数组，用于测试结果
        expected = SparseArray([0, np.nan, np.nan], fill_value=0)
        # 断言结果与预期相等
        tm.assert_sp_array_equal(result, expected)

        # allow_fill=False
        # 调用稀疏数组的take方法，传入索引数组和allow_fill、fill_value参数，返回结果
        result = sparse.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        # 创建预期的稀疏数组，用于测试结果
        expected = SparseArray([0, np.nan, 4], fill_value=0)
        # 断言结果与预期相等
        tm.assert_sp_array_equal(result, expected)

        # 在索引数组包含无效值时，抛出值错误异常
        msg = "Invalid value in 'indices'."
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -2]), allow_fill=True)
        with pytest.raises(ValueError, match=msg):
            sparse.take(np.array([1, 0, -5]), allow_fill=True)

        # 在索引数组包含超出边界值时，抛出索引错误异常
        msg = "out of bounds value in 'indices'"
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            sparse.take(np.array([1, 5]), fill_value=True)
    # 使用 pytest 的 parametrize 装饰器标记该测试函数，指定参数 "kind" 可以为 "block" 或 "integer"
    @pytest.mark.parametrize("kind", ["block", "integer"])
    # 定义测试函数，测试 SparseArray 类的 take 方法处理全为 NaN 的情况
    def test_take_filling_all_nan(self, kind):
        # 创建一个稀疏数组对象 sparse，包含五个 NaN 值，根据参数 kind 指定类型
        sparse = SparseArray([np.nan, np.nan, np.nan, np.nan, np.nan], kind=kind)
        # 调用 sparse 的 take 方法，传入索引数组 [1, 0, -1]，返回结果存储在 result 中
        result = sparse.take(np.array([1, 0, -1]))
        # 创建期望的稀疏数组对象 expected，包含三个 NaN 值，类型与 sparse 相同
        expected = SparseArray([np.nan, np.nan, np.nan], kind=kind)
        # 使用测试工具 tm.assert_sp_array_equal 检查 result 是否等于 expected
        tm.assert_sp_array_equal(result, expected)
    
        # 再次调用 sparse 的 take 方法，传入索引数组 [1, 0, -1] 和 fill_value=True，返回结果存储在 result 中
        result = sparse.take(np.array([1, 0, -1]), fill_value=True)
        # 创建另一个期望的稀疏数组对象 expected，同样包含三个 NaN 值，类型与 sparse 相同
        expected = SparseArray([np.nan, np.nan, np.nan], kind=kind)
        # 使用测试工具 tm.assert_sp_array_equal 检查 result 是否等于 expected
        tm.assert_sp_array_equal(result, expected)
    
        # 设置错误消息字符串 "out of bounds value in 'indices'"
        msg = "out of bounds value in 'indices'"
        # 使用 pytest.raises 检查 sparse.take 方法是否能捕获 IndexError 异常，并匹配错误消息 msg
        with pytest.raises(IndexError, match=msg):
            # 调用 sparse 的 take 方法，传入索引数组 [1, -6]，期望抛出 IndexError 异常
            sparse.take(np.array([1, -6]))
        with pytest.raises(IndexError, match=msg):
            # 调用 sparse 的 take 方法，传入索引数组 [1, 5]，期望抛出 IndexError 异常
            sparse.take(np.array([1, 5]))
        with pytest.raises(IndexError, match=msg):
            # 再次调用 sparse 的 take 方法，传入索引数组 [1, 5] 和 fill_value=True，期望抛出 IndexError 异常
            sparse.take(np.array([1, 5]), fill_value=True)
class TestWhere:
    # 定义一个测试类 TestWhere
    def test_where_retain_fill_value(self):
        # 定义测试方法 test_where_retain_fill_value，测试 _where 方法是否保留 fill_value

        # 创建一个 SparseArray 对象 arr，包含两个元素 [NaN, 1.0]，填充值为 0
        arr = SparseArray([np.nan, 1.0], fill_value=0)

        # 创建一个布尔类型的数组 mask，包含两个元素 True 和 False
        mask = np.array([True, False])

        # 调用 arr 对象的 _where 方法，将 mask 取反后作为条件，替换符合条件的元素为 1
        res = arr._where(~mask, 1)

        # 创建一个期望的 SparseArray 对象 exp，包含两个元素 [1, 1.0]，填充值为 0
        exp = SparseArray([1, 1.0], fill_value=0)

        # 使用测试框架的方法验证 res 是否等于 exp
        tm.assert_sp_array_equal(res, exp)

        # 创建一个 Series 对象 ser，以 arr 作为其数据
        ser = pd.Series(arr)

        # 调用 Series 对象 ser 的 where 方法，将 mask 取反后作为条件，替换符合条件的元素为 1
        res = ser.where(~mask, 1)

        # 使用测试框架的方法验证 res 是否等于期望的 Series 对象 exp
        tm.assert_series_equal(res, pd.Series(exp))
```