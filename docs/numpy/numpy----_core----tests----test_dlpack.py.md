# `.\numpy\numpy\_core\tests\test_dlpack.py`

```py
# 导入系统相关模块和 pytest 模块
import sys
import pytest

# 导入 NumPy 库并从中导入所需的函数和变量
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY

# 定义一个生成器函数，生成包含新旧 DLPack 对象的迭代器
def new_and_old_dlpack():
    # 生成一个包含从 0 到 4 的整数数组的 NumPy 对象
    yield np.arange(5)

    # 定义一个旧版本的 DLPack 类，继承自 np.ndarray
    class OldDLPack(np.ndarray):
        # 只支持“旧”版本的 __dlpack__ 方法
        def __dlpack__(self, stream=None):
            return super().__dlpack__(stream=None)

    # 生成一个将整数数组视图转换为 OldDLPack 类型的对象
    yield np.arange(5).view(OldDLPack)

# 定义一个测试类 TestDLPack
class TestDLPack:
    # 标记测试为跳过状态，如果是在 PyPy 环境下运行则跳过
    @pytest.mark.skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    # 参数化测试，参数为 max_version，包括 (0, 0), None, (1, 0), (100, 3)
    @pytest.mark.parametrize("max_version", [(0, 0), None, (1, 0), (100, 3)])
    # 测试 __dlpack__ 方法的引用计数
    def test_dunder_dlpack_refcount(self, max_version):
        x = np.arange(5)
        # 调用 x 对象的 __dlpack__ 方法
        y = x.__dlpack__(max_version=max_version)
        # 断言 x 对象的引用计数为 3
        assert sys.getrefcount(x) == 3
        del y
        # 删除 y 对象后，再次断言 x 对象的引用计数为 2
        assert sys.getrefcount(x) == 2

    # 测试 __dlpack__ 方法的 stream 参数
    def test_dunder_dlpack_stream(self):
        x = np.arange(5)
        # 调用 x 对象的 __dlpack__ 方法，stream 参数为 None
        x.__dlpack__(stream=None)

        # 使用 pytest 断言捕获 RuntimeError 异常
        with pytest.raises(RuntimeError):
            # 再次调用 x 对象的 __dlpack__ 方法，此时 stream 参数为 1
            x.__dlpack__(stream=1)

    # 测试 __dlpack__ 方法的 copy 参数
    def test_dunder_dlpack_copy(self):
        # 显式检查 __dlpack__ 方法的参数解析
        x = np.arange(5)
        x.__dlpack__(copy=True)
        x.__dlpack__(copy=None)
        x.__dlpack__(copy=False)

        # 使用 pytest 断言捕获 ValueError 异常
        with pytest.raises(ValueError):
            # __dlpack__ 方法的 copy 参数传入一个 NumPy 数组
            x.__dlpack__(copy=np.array([1, 2, 3]))

    # 测试 strides 不是 itemsize 的倍数时的情况
    def test_strides_not_multiple_of_itemsize(self):
        # 创建一个复合数据类型的零数组
        dt = np.dtype([('int', np.int32), ('char', np.int8)])
        y = np.zeros((5,), dtype=dt)
        # 获取复合数组的 'int' 字段作为 z
        z = y['int']

        # 使用 pytest 断言捕获 BufferError 异常
        with pytest.raises(BufferError):
            # 尝试从 DLPack 对象 z 中导入数据
            np.from_dlpack(z)

    # 标记测试为跳过状态，如果是在 PyPy 环境下运行则跳过
    @pytest.mark.skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    # 参数化测试，参数为 new_and_old_dlpack() 生成器的迭代结果
    @pytest.mark.parametrize("arr", new_and_old_dlpack())
    # 测试从 DLPack 对象到 NumPy 对象的引用计数
    def test_from_dlpack_refcount(self, arr):
        arr = arr.copy()
        y = np.from_dlpack(arr)
        # 断言 arr 对象的引用计数为 3
        assert sys.getrefcount(arr) == 3
        del y
        # 删除 y 对象后，再次断言 arr 对象的引用计数为 2
        assert sys.getrefcount(arr) == 2

    # 参数化测试，参数为 dtype 和 new_and_old_dlpack() 生成器的迭代结果
    @pytest.mark.parametrize("dtype", [
        np.bool,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128
    ])
    @pytest.mark.parametrize("arr", new_and_old_dlpack())
    # 测试 dtype 在 from_dlpack 调用中的传递
    def test_dtype_passthrough(self, arr, dtype):
        x = arr.astype(dtype)
        y = np.from_dlpack(x)

        # 断言 y 的数据类型与 x 相同
        assert y.dtype == x.dtype
        # 断言 y 与 x 相等
        assert_array_equal(x, y)

    # 测试传入无效 dtype 参数时的情况
    def test_invalid_dtype(self):
        # 创建一个包含 np.datetime64 对象的 NumPy 数组
        x = np.asarray(np.datetime64('2021-05-27'))

        # 使用 pytest 断言捕获 BufferError 异常
        with pytest.raises(BufferError):
            # 尝试从 x 中导入数据
            np.from_dlpack(x)

    # 测试传入无效的字节交换方式时的情况
    def test_invalid_byte_swapping(self):
        # 创建一个使用 newbyteorder() 方法改变字节顺序的数组
        dt = np.dtype('=i8').newbyteorder()
        x = np.arange(5, dtype=dt)

        # 使用 pytest 断言捕获 BufferError 异常
        with pytest.raises(BufferError):
            # 尝试从 x 中导入数据
            np.from_dlpack(x)
    def test_non_contiguous(self):
        x = np.arange(25).reshape((5, 5))

        y1 = x[0]
        assert_array_equal(y1, np.from_dlpack(y1))
        # 获取数组 x 的第一行，转换为 DLPack 格式，与从 DLPack 转换回来的结果进行比较

        y2 = x[:, 0]
        assert_array_equal(y2, np.from_dlpack(y2))
        # 获取数组 x 的第一列，转换为 DLPack 格式，与从 DLPack 转换回来的结果进行比较

        y3 = x[1, :]
        assert_array_equal(y3, np.from_dlpack(y3))
        # 获取数组 x 的第二行，转换为 DLPack 格式，与从 DLPack 转换回来的结果进行比较

        y4 = x[1]
        assert_array_equal(y4, np.from_dlpack(y4))
        # 获取数组 x 的第二行，转换为 DLPack 格式，与从 DLPack 转换回来的结果进行比较

        y5 = np.diagonal(x).copy()
        assert_array_equal(y5, np.from_dlpack(y5))
        # 获取数组 x 的对角线元素，复制为新数组，转换为 DLPack 格式，与从 DLPack 转换回来的结果进行比较

    @pytest.mark.parametrize("ndim", range(33))
    def test_higher_dims(self, ndim):
        shape = (1,) * ndim
        x = np.zeros(shape, dtype=np.float64)

        assert shape == np.from_dlpack(x).shape
        # 创建指定维度的零数组 x，将其转换为 DLPack 格式后，比较其形状是否与原始形状一致

    def test_dlpack_device(self):
        x = np.arange(5)
        assert x.__dlpack_device__() == (1, 0)
        # 检查数组 x 的设备信息是否为 (1, 0)
        y = np.from_dlpack(x)
        assert y.__dlpack_device__() == (1, 0)
        # 将数组 x 转换为 DLPack 格式后，检查转换后数组 y 的设备信息是否为 (1, 0)
        z = y[::2]
        assert z.__dlpack_device__() == (1, 0)
        # 对从 DLPack 转换回来的数组 y 进行切片操作，检查切片后数组 z 的设备信息是否为 (1, 0)

    def dlpack_deleter_exception(self, max_version):
        x = np.arange(5)
        _ = x.__dlpack__(max_version=max_version)
        raise RuntimeError
        # 尝试使用指定的 max_version 将数组 x 转换为 DLPack 格式，并抛出 RuntimeError 异常

    @pytest.mark.parametrize("max_version", [None, (1, 0)])
    def test_dlpack_destructor_exception(self, max_version):
        with pytest.raises(RuntimeError):
            self.dlpack_deleter_exception(max_version=max_version)
        # 测试在使用不同的 max_version 参数时，调用 dlpack_deleter_exception 方法是否会抛出 RuntimeError 异常

    def test_readonly(self):
        x = np.arange(5)
        x.flags.writeable = False
        # 设置数组 x 为不可写
        # 没有指定 max_version 时应该引发异常
        with pytest.raises(BufferError):
            x.__dlpack__()

        # 但是如果我们尝试指定版本，应该正常工作
        y = np.from_dlpack(x)
        assert not y.flags.writeable
        # 将不可写的数组 x 转换为 DLPack 格式后，检查转换后数组 y 是否为不可写

    def test_ndim0(self):
        x = np.array(1.0)
        y = np.from_dlpack(x)
        assert_array_equal(x, y)
        # 将标量数组 x 转换为 DLPack 格式后，与从 DLPack 转换回来的结果进行比较

    def test_size1dims_arrays(self):
        x = np.ndarray(dtype='f8', shape=(10, 5, 1), strides=(8, 80, 4),
                       buffer=np.ones(1000, dtype=np.uint8), order='F')
        y = np.from_dlpack(x)
        assert_array_equal(x, y)
        # 创建一个特定属性的多维数组 x，将其转换为 DLPack 格式后，与从 DLPack 转换回来的结果进行比较

    def test_copy(self):
        x = np.arange(5)

        y = np.from_dlpack(x)
        assert np.may_share_memory(x, y)
        # 将数组 x 转换为 DLPack 格式后，检查转换后数组 y 是否与原数组共享内存
        y = np.from_dlpack(x, copy=False)
        assert np.may_share_memory(x, y)
        # 使用 copy=False 将数组 x 转换为 DLPack 格式后，检查转换后数组 y 是否与原数组共享内存
        y = np.from_dlpack(x, copy=True)
        assert not np.may_share_memory(x, y)
        # 使用 copy=True 将数组 x 转换为 DLPack 格式后，检查转换后数组 y 是否与原数组共享内存

    def test_device(self):
        x = np.arange(5)
        # 请求 (1, 0)，即 CPU 设备，在两次调用中均有效：
        x.__dlpack__(dl_device=(1, 0))
        np.from_dlpack(x, device="cpu")
        np.from_dlpack(x, device=None)
        # 将数组 x 转换为 DLPack 格式时，指定不同的设备参数，验证其是否符合预期

        with pytest.raises(ValueError):
            x.__dlpack__(dl_device=(10, 0))
        with pytest.raises(ValueError):
            np.from_dlpack(x, device="gpu")
        # 尝试使用无效的设备参数时，应该引发 ValueError 异常
```