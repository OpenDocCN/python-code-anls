# `D:\src\scipysrc\pandas\pandas\tests\test_take.py`

```
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入第三方库 numpy，并使用别名 np
import numpy as np

# 导入 pytest 测试框架
import pytest

# 导入 pandas 内部库 pandas._libs 中的 iNaT
from pandas._libs import iNaT

# 导入 pandas 测试模块 pandas._testing，并使用别名 tm
import pandas._testing as tm

# 导入 pandas 核心算法模块 pandas.core.algorithms，并使用别名 algos
import pandas.core.algorithms as algos


# 定义 pytest 的测试 fixture，参数为一系列元组，每个元组包含三个值
@pytest.fixture(
    params=[
        (np.int8, np.int16(127), np.int8),
        (np.int8, np.int16(128), np.int16),
        (np.int32, 1, np.int32),
        (np.int32, 2.0, np.float64),
        (np.int32, 3.0 + 4.0j, np.complex128),
        (np.int32, True, np.object_),
        (np.int32, "", np.object_),
        (np.float64, 1, np.float64),
        (np.float64, 2.0, np.float64),
        (np.float64, 3.0 + 4.0j, np.complex128),
        (np.float64, True, np.object_),
        (np.float64, "", np.object_),
        (np.complex128, 1, np.complex128),
        (np.complex128, 2.0, np.complex128),
        (np.complex128, 3.0 + 4.0j, np.complex128),
        (np.complex128, True, np.object_),
        (np.complex128, "", np.object_),
        (np.bool_, 1, np.object_),
        (np.bool_, 2.0, np.object_),
        (np.bool_, 3.0 + 4.0j, np.object_),
        (np.bool_, True, np.bool_),
        (np.bool_, "", np.object_),
    ]
)
def dtype_fill_out_dtype(request):
    # 返回 pytest fixture 的参数化元组
    return request.param


# 定义测试类 TestTake
class TestTake:
    # 测试方法，测试一维数据的填充与索引
    def test_1d_fill_nonna(self, dtype_fill_out_dtype):
        # 解包 pytest fixture 的参数元组
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        
        # 生成随机数据，类型为指定的 dtype
        data = np.random.default_rng(2).integers(0, 2, 4).astype(dtype)
        
        # 索引器，指定数据索引顺序
        indexer = [2, 1, 0, -1]

        # 使用 algos 模块中的 take_nd 函数进行索引操作
        result = algos.take_nd(data, indexer, fill_value=fill_value)
        
        # 断言：索引后的部分与预期相符
        assert (result[[0, 1, 2]] == data[[2, 1, 0]]).all()
        
        # 断言：指定位置的值应与填充值相符
        assert result[3] == fill_value
        
        # 断言：返回结果的数据类型应与预期的输出数据类型相符
        assert result.dtype == out_dtype
        
        # 更新索引器，进行第二轮索引操作
        indexer = [2, 1, 0, 1]

        # 再次使用 take_nd 函数进行索引操作
        result = algos.take_nd(data, indexer, fill_value=fill_value)
        
        # 断言：索引后的部分与预期相符
        assert (result[[0, 1, 2, 3]] == data[indexer]).all()
        
        # 断言：返回结果的数据类型应与原始数据的数据类型相符
        assert result.dtype == dtype

    # 测试方法，测试二维数据的填充与索引
    def test_2d_fill_nonna(self, dtype_fill_out_dtype):
        # 解包 pytest fixture 的参数元组
        dtype, fill_value, out_dtype = dtype_fill_out_dtype
        
        # 生成随机二维数据，类型为指定的 dtype
        data = np.random.default_rng(2).integers(0, 2, (5, 3)).astype(dtype)
        
        # 索引器，指定数据索引顺序
        indexer = [2, 1, 0, -1]

        # 使用 algos 模块中的 take_nd 函数进行轴向为 0 的索引操作
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        
        # 断言：索引后的部分与预期相符
        assert (result[[0, 1, 2], :] == data[[2, 1, 0], :]).all()
        
        # 断言：指定位置的值应与填充值相符
        assert (result[3, :] == fill_value).all()
        
        # 断言：返回结果的数据类型应与预期的输出数据类型相符
        assert result.dtype == out_dtype

        # 使用 algos 模块中的 take_nd 函数进行轴向为 1 的索引操作
        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        
        # 断言：索引后的部分与预期相符
        assert (result[:, [0, 1, 2]] == data[:, [2, 1, 0]]).all()
        
        # 断言：指定位置的值应与填充值相符
        assert (result[:, 3] == fill_value).all()
        
        # 断言：返回结果的数据类型应与预期的输出数据类型相符
        assert result.dtype == out_dtype
        
        # 更新索引器，进行第二轮索引操作
        indexer = [2, 1, 0, 1]
        
        # 再次使用 take_nd 函数进行轴向为 0 的索引操作
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        
        # 断言：索引后的部分与预期相符
        assert (result[[0, 1, 2, 3], :] == data[indexer, :]).all()
        
        # 断言：返回结果的数据类型应与原始数据的数据类型相符
        assert result.dtype == dtype

        # 再次使用 take_nd 函数进行轴向为 1 的索引操作
        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        
        # 断言：索引后的部分与预期相符
        assert (result[:, [0, 1, 2, 3]] == data[:, indexer]).all()
        
        # 断言：返回结果的数据类型应与原始数据的数据类型相符
        assert result.dtype == dtype
    # 测试函数：测试在填充非空值时的多维索引操作
    def test_3d_fill_nonna(self, dtype_fill_out_dtype):
        # 解包参数：数据类型、填充值、输出数据类型
        dtype, fill_value, out_dtype = dtype_fill_out_dtype

        # 创建随机数据矩阵，形状为 (5, 4, 3)，并转换为指定数据类型
        data = np.random.default_rng(2).integers(0, 2, (5, 4, 3)).astype(dtype)
        
        # 定义索引器
        indexer = [2, 1, 0, -1]

        # 在指定轴向（axis=0）使用 algos.take_nd 进行索引操作，并指定填充值
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        # 断言结果的部分与原数据的对应部分相等
        assert (result[[0, 1, 2], :, :] == data[[2, 1, 0], :, :]).all()
        # 断言结果的特定位置的值与填充值相等
        assert (result[3, :, :] == fill_value).all()
        # 断言结果的数据类型与预期输出数据类型相等
        assert result.dtype == out_dtype

        # 在指定轴向（axis=1）使用 algos.take_nd 进行索引操作
        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2], :] == data[:, [2, 1, 0], :]).all()
        assert (result[:, 3, :] == fill_value).all()
        assert result.dtype == out_dtype

        # 在指定轴向（axis=2）使用 algos.take_nd 进行索引操作
        result = algos.take_nd(data, indexer, axis=2, fill_value=fill_value)
        assert (result[:, :, [0, 1, 2]] == data[:, :, [2, 1, 0]]).all()
        assert (result[:, :, 3] == fill_value).all()
        assert result.dtype == out_dtype

        # 修改索引器并再次进行测试
        indexer = [2, 1, 0, 1]

        # 在指定轴向（axis=0）使用 algos.take_nd 进行索引操作
        result = algos.take_nd(data, indexer, axis=0, fill_value=fill_value)
        assert (result[[0, 1, 2, 3], :, :] == data[indexer, :, :]).all()
        assert result.dtype == dtype

        # 在指定轴向（axis=1）使用 algos.take_nd 进行索引操作
        result = algos.take_nd(data, indexer, axis=1, fill_value=fill_value)
        assert (result[:, [0, 1, 2, 3], :] == data[:, indexer, :]).all()
        assert result.dtype == dtype

        # 在指定轴向（axis=2）使用 algos.take_nd 进行索引操作
        result = algos.take_nd(data, indexer, axis=2, fill_value=fill_value)
        assert (result[:, :, [0, 1, 2, 3]] == data[:, :, indexer]).all()
        assert result.dtype == dtype

    # 测试函数：测试在不同数据类型下的一维索引操作
    def test_1d_other_dtypes(self):
        # 创建随机标准正态分布数据，长度为 10，并转换为指定数据类型 np.float32
        arr = np.random.default_rng(2).standard_normal(10).astype(np.float32)

        # 定义索引器
        indexer = [1, 2, 3, -1]

        # 使用 algos.take_nd 进行一维索引操作
        result = algos.take_nd(arr, indexer)
        # 生成预期结果
        expected = arr.take(indexer)
        expected[-1] = np.nan
        # 使用断言检查结果与预期是否几乎相等
        tm.assert_almost_equal(result, expected)

    # 测试函数：测试在不同数据类型下的二维索引操作
    def test_2d_other_dtypes(self):
        # 创建随机标准正态分布数据矩阵，形状为 (10, 5)，并转换为指定数据类型 np.float32
        arr = np.random.default_rng(2).standard_normal((10, 5)).astype(np.float32)

        # 定义索引器
        indexer = [1, 2, 3, -1]

        # 在 axis=0 上使用 algos.take_nd 进行二维索引操作
        result = algos.take_nd(arr, indexer, axis=0)
        # 生成预期结果
        expected = arr.take(indexer, axis=0)
        expected[-1] = np.nan
        # 使用断言检查结果与预期是否几乎相等
        tm.assert_almost_equal(result, expected)

        # 在 axis=1 上使用 algos.take_nd 进行二维索引操作
        result = algos.take_nd(arr, indexer, axis=1)
        # 生成预期结果
        expected = arr.take(indexer, axis=1)
        expected[:, -1] = np.nan
        # 使用断言检查结果与预期是否几乎相等
        tm.assert_almost_equal(result, expected)

    # 测试函数：测试在布尔类型数据下的一维索引操作
    def test_1d_bool(self):
        # 创建布尔类型数组
        arr = np.array([0, 1, 0], dtype=bool)

        # 使用 algos.take_nd 进行一维索引操作
        result = algos.take_nd(arr, [0, 2, 2, 1])
        # 生成预期结果
        expected = arr.take([0, 2, 2, 1])
        # 使用断言检查 numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 使用 algos.take_nd 进行一维索引操作
        result = algos.take_nd(arr, [0, 2, -1])
        # 使用断言检查结果的数据类型是否为 np.object_
        assert result.dtype == np.object_
    # 定义测试函数，测试 algos.take_nd 函数在二维布尔数组上的行为
    def test_2d_bool(self):
        # 创建一个二维布尔数组
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)

        # 测试 axis=0 的情况
        result = algos.take_nd(arr, [0, 2, 2, 1])
        expected = arr.take([0, 2, 2, 1], axis=0)
        tm.assert_numpy_array_equal(result, expected)

        # 测试 axis=1 的情况
        result = algos.take_nd(arr, [0, 2, 2, 1], axis=1)
        expected = arr.take([0, 2, 2, 1], axis=1)
        tm.assert_numpy_array_equal(result, expected)

        # 测试带有负索引的情况
        result = algos.take_nd(arr, [0, 2, -1])
        # 断言结果的数据类型为 np.object_
        assert result.dtype == np.object_

    # 定义测试函数，测试 algos.take_nd 函数在二维 float32 数组上的行为
    def test_2d_float32(self):
        # 创建一个二维 float32 随机数组
        arr = np.random.default_rng(2).standard_normal((4, 3)).astype(np.float32)
        indexer = [0, 2, -1, 1, -1]

        # 测试 axis=0 的情况
        result = algos.take_nd(arr, indexer, axis=0)
        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = np.nan
        tm.assert_almost_equal(result, expected)

        # 测试 axis=1 的情况
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = np.nan
        tm.assert_almost_equal(result, expected)

    # 定义测试函数，测试 algos.take_nd 函数在二维 datetime64 数组上的行为
    def test_2d_datetime64(self):
        # 创建一个二维 datetime64 数组
        # 2005/01/01 - 2006/01/01 之间的随机日期数组
        arr = (
            np.random.default_rng(2).integers(11_045_376, 11_360_736, (5, 3))
            * 100_000_000_000
        )
        arr = arr.view(dtype="datetime64[ns]")
        indexer = [0, 2, -1, 1, -1]

        # 测试 axis=0 的情况
        result = algos.take_nd(arr, indexer, axis=0)
        expected = arr.take(indexer, axis=0)
        expected.view(np.int64)[[2, 4], :] = iNaT
        tm.assert_almost_equal(result, expected)

        # 测试 axis=0 的情况，使用 fill_value=datetime(2007, 1, 1)
        result = algos.take_nd(arr, indexer, axis=0, fill_value=datetime(2007, 1, 1))
        expected = arr.take(indexer, axis=0)
        expected[[2, 4], :] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

        # 测试 axis=1 的情况
        result = algos.take_nd(arr, indexer, axis=1)
        expected = arr.take(indexer, axis=1)
        expected.view(np.int64)[:, [2, 4]] = iNaT
        tm.assert_almost_equal(result, expected)

        # 测试 axis=1 的情况，使用 fill_value=datetime(2007, 1, 1)
        result = algos.take_nd(arr, indexer, axis=1, fill_value=datetime(2007, 1, 1))
        expected = arr.take(indexer, axis=1)
        expected[:, [2, 4]] = datetime(2007, 1, 1)
        tm.assert_almost_equal(result, expected)

    # 定义测试函数，测试 algos.take 函数在二维数组上的行为
    def test_take_axis_0(self):
        # 创建一个二维数组
        arr = np.arange(12).reshape(4, 3)

        # 测试 axis=0 的情况
        result = algos.take(arr, [0, -1])
        expected = np.array([[0, 1, 2], [9, 10, 11]])
        tm.assert_numpy_array_equal(result, expected)

        # 测试 allow_fill=True 的情况，使用 fill_value=0
        result = algos.take(arr, [0, -1], allow_fill=True, fill_value=0)
        expected = np.array([[0, 1, 2], [0, 0, 0]])
        tm.assert_numpy_array_equal(result, expected)
    # 定义一个测试方法，测试在轴向1上的索引取值
    def test_take_axis_1(self):
        # 创建一个4x3的NumPy数组，值为0到11的整数
        arr = np.arange(12).reshape(4, 3)
        # 使用算法库中的take函数，按指定的索引列表[0, -1]在轴1上取值
        result = algos.take(arr, [0, -1], axis=1)
        # 预期结果是一个2列的数组，每行包含轴1上指定索引位置的元素
        expected = np.array([[0, 2], [3, 5], [6, 8], [9, 11]])
        # 使用测试工具函数检查结果数组是否与预期数组相等
        tm.assert_numpy_array_equal(result, expected)

        # 在允许填充的情况下，再次使用take函数在轴1上取值，并指定填充值为0
        result = algos.take(arr, [0, -1], axis=1, allow_fill=True, fill_value=0)
        # 预期结果是一个2列的数组，使用填充值0来填充索引-1超出范围的位置
        expected = np.array([[0, 0], [3, 0], [6, 0], [9, 0]])
        # 使用测试工具函数检查结果数组是否与预期数组相等
        tm.assert_numpy_array_equal(result, expected)

        # 使用pytest验证器，测试当索引超出边界时是否会引发IndexError异常，并包含指定的错误信息
        with pytest.raises(IndexError, match="indices are out-of-bounds"):
            # 调用take函数，在轴1上取索引[0, 3]的值，允许填充，填充值为0
            algos.take(arr, [0, 3], axis=1, allow_fill=True, fill_value=0)

    # 定义一个测试方法，测试使用非可哈希填充值的情况
    def test_take_non_hashable_fill_value(self):
        # 创建一个包含整数的NumPy数组
        arr = np.array([1, 2, 3])
        # 创建一个索引器数组，包含索引1和-1
        indexer = np.array([1, -1])
        # 使用pytest验证器，测试当填充值不是标量时是否会引发ValueError异常，并包含指定的错误信息
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            # 调用take函数，尝试使用非标量填充值[1]填充
            algos.take(arr, indexer, allow_fill=True, fill_value=[1])

        # 创建一个包含对象类型的NumPy数组
        arr = np.array([1, 2, 3], dtype=object)
        # 使用take函数，在允许填充的情况下，使用非标量填充值[1]填充
        result = algos.take(arr, indexer, allow_fill=True, fill_value=[1])
        # 预期结果是一个包含对象类型的数组，填充值为[1]的索引位置
        expected = np.array([2, [1]], dtype=object)
        # 使用测试工具函数检查结果数组是否与预期数组相等
        tm.assert_numpy_array_equal(result, expected)
class TestExtensionTake:
    # pd.api.extensions 中的 take 方法测试

    def test_bounds_check_large(self):
        arr = np.array([1, 2])

        msg = "indices are out-of-bounds"
        # 确保对超出边界的索引抛出 IndexError 异常
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=True)

        msg = "index 2 is out of bounds for( axis 0 with)? size 2"
        # 确保对超出边界的索引抛出带有特定消息的 IndexError 异常
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [2, 3], allow_fill=False)

    def test_bounds_check_small(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        indexer = [0, -1, -2]

        msg = r"'indices' contains values less than allowed \(-2 < -1\)"
        # 确保对不符合要求的索引值抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            algos.take(arr, indexer, allow_fill=True)

        # 确保 take 方法按照预期返回正确的结果
        result = algos.take(arr, indexer)
        expected = np.array([1, 3, 2], dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("allow_fill", [True, False])
    def test_take_empty(self, allow_fill):
        arr = np.array([], dtype=np.int64)
        # 空的 take 操作应该返回空数组
        result = algos.take(arr, [], allow_fill=allow_fill)
        tm.assert_numpy_array_equal(arr, result)

        msg = "|".join(
            [
                "cannot do a non-empty take from an empty axes.",
                "indices are out-of-bounds",
            ]
        )
        # 确保对从空数组中进行非空 take 操作抛出正确的 IndexError 异常
        with pytest.raises(IndexError, match=msg):
            algos.take(arr, [0], allow_fill=allow_fill)

    def test_take_na_empty(self):
        # 确保在空数组上使用 allow_fill=True 和 fill_value 参数进行 take 操作按预期返回
        result = algos.take(np.array([]), [-1, -1], allow_fill=True, fill_value=0.0)
        expected = np.array([0.0, 0.0])
        tm.assert_numpy_array_equal(result, expected)

    def test_take_coerces_list(self):
        # 确保当传入列表时，抛出预期的 TypeError 异常
        arr = [1, 2, 3]
        msg = (
            "pd.api.extensions.take requires a numpy.ndarray, ExtensionArray, "
            "Index, or Series, got list"
        )
        with pytest.raises(TypeError, match=msg):
            algos.take(arr, [0, 0])
```