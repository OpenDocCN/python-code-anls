# `.\pytorch\test\torch_np\numpy_tests\core\test_dlpack.py`

```
# Owner(s): ["module: dynamo"]  # 标明代码所有者，这里是动态的模块

import functools  # 导入 functools 模块，用于创建偏函数
import sys  # 导入 sys 模块，提供对解释器相关的系统调用访问
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

from unittest import skipIf as skipif  # 导入 unittest 中的 skipIf 别名为 skipif

import numpy  # 导入 numpy 库，用于数值计算

import pytest  # 导入 pytest 库，用于编写测试用例

import torch  # 导入 torch 库，用于深度学习框架 Torch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试的工具函数
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试的函数
    skipIfTorchDynamo,  # 导入如果使用 TorchDynamo 则跳过的装饰器
    TEST_WITH_TORCHDYNAMO,  # 导入 TorchDynamo 的测试标志
    TestCase,  # 导入测试用例基类
    xpassIfTorchDynamo,  # 导入如果使用 TorchDynamo 则通过的装饰器
)

if TEST_WITH_TORCHDYNAMO:  # 如果需要使用 TorchDynamo
    import numpy as np  # 导入 NumPy 并命名为 np
    from numpy.testing import assert_array_equal  # 导入 NumPy 的数组相等断言函数
else:  # 否则
    import torch._numpy as np  # 导入 Torch 的 NumPy 兼容模块并命名为 np
    from torch._numpy.testing import assert_array_equal  # 导入 Torch 的 NumPy 测试模块的数组相等断言函数

skip = functools.partial(skipif, True)  # 创建一个偏函数 skip，始终跳过测试

IS_PYPY = False  # 设置 PyPy 环境标志为 False


@skipif(numpy.__version__ < "1.24", reason="numpy.dlpack is new in numpy 1.23")
@instantiate_parametrized_tests  # 实例化参数化测试
class TestDLPack(TestCase):  # 定义测试类 TestDLPack，继承自 TestCase
    @xpassIfTorchDynamo  # 如果使用 TorchDynamo 则跳过此测试
    @skipif(IS_PYPY, reason="PyPy can't get refcounts.")  # 如果运行在 PyPy 上则跳过此测试
    def test_dunder_dlpack_refcount(self):
        x = np.arange(5)  # 创建一个长度为 5 的 NumPy 数组 x
        y = x.__dlpack__()  # 获取 x 的 dlpack 对象
        assert sys.getrefcount(x) == 3  # 断言 x 的引用计数为 3
        del y  # 删除 y 变量
        assert sys.getrefcount(x) == 2  # 断言 x 的引用计数为 2

    @unittest.expectedFailure  # 标记预期失败的测试
    @skipIfTorchDynamo("I can't figure out how to get __dlpack__ into trace_rules.py")  # 如果使用 TorchDynamo 则跳过此测试
    def test_dunder_dlpack_stream(self):
        x = np.arange(5)  # 创建一个长度为 5 的 NumPy 数组 x
        x.__dlpack__(stream=None)  # 使用指定流获取 x 的 dlpack 对象

        with pytest.raises(RuntimeError):  # 断言抛出 RuntimeError 异常
            x.__dlpack__(stream=1)  # 使用流值为 1 获取 x 的 dlpack 对象

    @xpassIfTorchDynamo  # 如果使用 TorchDynamo 则跳过此测试
    @skipif(IS_PYPY, reason="PyPy can't get refcounts.")  # 如果运行在 PyPy 上则跳过此测试
    def test_from_dlpack_refcount(self):
        x = np.arange(5)  # 创建一个长度为 5 的 NumPy 数组 x
        y = np.from_dlpack(x)  # 从 dlpack 对象创建 NumPy 数组 y
        assert sys.getrefcount(x) == 3  # 断言 x 的引用计数为 3
        del y  # 删除 y 变量
        assert sys.getrefcount(x) == 2  # 断言 x 的引用计数为 2

    @parametrize(
        "dtype",
        [
            np.int8,  # 测试不同的数据类型
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ],
    )
    def test_dtype_passthrough(self, dtype):
        x = np.arange(5, dtype=dtype)  # 创建指定数据类型的长度为 5 的 NumPy 数组 x
        y = np.from_dlpack(x)  # 从 dlpack 对象创建 NumPy 数组 y

        assert y.dtype == x.dtype  # 断言 y 的数据类型与 x 的数据类型相同
        assert_array_equal(x, y)  # 断言 x 和 y 的数组内容相等

    def test_non_contiguous(self):
        x = np.arange(25).reshape((5, 5))  # 创建一个形状为 (5, 5) 的 NumPy 数组 x

        y1 = x[0]  # 获取 x 的非连续切片 y1
        assert_array_equal(y1, np.from_dlpack(y1))  # 断言 y1 和其对应的 dlpack 对象转换后的数组内容相等

        y2 = x[:, 0]  # 获取 x 的非连续切片 y2
        assert_array_equal(y2, np.from_dlpack(y2))  # 断言 y2 和其对应的 dlpack 对象转换后的数组内容相等

        y3 = x[1, :]  # 获取 x 的非连续切片 y3
        assert_array_equal(y3, np.from_dlpack(y3))  # 断言 y3 和其对应的 dlpack 对象转换后的数组内容相等

        y4 = x[1]  # 获取 x 的非连续切片 y4
        assert_array_equal(y4, np.from_dlpack(y4))  # 断言 y4 和其对应的 dlpack 对象转换后的数组内容相等

        y5 = np.diagonal(x).copy()  # 获取 x 的对角线并复制为连续数组 y5
        assert_array_equal(y5, np.from_dlpack(y5))  # 断言 y5 和其对应的 dlpack 对象转换后的数组内容相等

    @parametrize("ndim", range(33))  # 参数化测试，测试不同的维度
    def test_higher_dims(self, ndim):
        shape = (1,) * ndim  # 创建一个维度为 ndim 的形状元组
        x = np.zeros(shape, dtype=np.float64)  # 创建指定形状和数据类型的 NumPy 数组 x

        assert shape == np.from_dlpack(x).shape  # 断言 x 经 dlpack 转换后的形状与原始形状相同
    # 测试 dlpack_device 方法是否正确返回设备信息
    def test_dlpack_device(self):
        # 创建一个 NumPy 数组 x，包含元素 [0, 1, 2, 3, 4]
        x = np.arange(5)
        # 断言调用 x 的 __dlpack_device__ 方法返回 (1, 0)，表示 CPU 设备
        assert x.__dlpack_device__() == (1, 0)
        # 使用 from_dlpack 方法将 NumPy 数组 x 转换为 y
        y = np.from_dlpack(x)
        # 断言调用 y 的 __dlpack_device__ 方法也返回 (1, 0)
        assert y.__dlpack_device__() == (1, 0)
        # 从 y 中取出索引为偶数的元素组成新数组 z
        z = y[::2]
        # 断言调用 z 的 __dlpack_device__ 方法仍然返回 (1, 0)
        assert z.__dlpack_device__() == (1, 0)

    # 测试 dlpack 方法在发生异常时是否正确抛出 RuntimeError
    def dlpack_deleter_exception(self):
        # 创建一个 NumPy 数组 x，包含元素 [0, 1, 2, 3, 4]
        x = np.arange(5)
        # 调用 x 的 __dlpack__ 方法
        _ = x.__dlpack__()
        # 抛出一个 RuntimeError 异常
        raise RuntimeError

    # 测试 dlpack_deleter_exception 方法在运行时是否抛出 RuntimeError 异常
    def test_dlpack_destructor_exception(self):
        # 使用 pytest 的 raises 方法捕获 RuntimeError 异常
        with pytest.raises(RuntimeError):
            self.dlpack_deleter_exception()

    # 标记为 skip 的测试用例，原因是 PyTorch 不支持只读数组
    @skip(reason="no readonly arrays in pytorch")
    def test_readonly(self):
        # 创建一个 NumPy 数组 x，包含元素 [0, 1, 2, 3, 4]
        x = np.arange(5)
        # 将数组 x 设置为不可写
        x.flags.writeable = False
        # 使用 pytest 的 raises 方法捕获 BufferError 异常
        with pytest.raises(BufferError):
            x.__dlpack__()

    # 测试 from_dlpack 方法是否正确处理维度为 0 的数组
    def test_ndim0(self):
        # 创建一个包含单个元素 1.0 的 NumPy 数组 x
        x = np.array(1.0)
        # 使用 from_dlpack 方法将 NumPy 数组 x 转换为 y
        y = np.from_dlpack(x)
        # 断言 x 和 y 数组相等
        assert_array_equal(x, y)

    # 测试从 PyTorch 数组转换为 NumPy 数组的正确性
    def test_from_torch(self):
        # 创建一个 PyTorch 数组 t，包含元素 [0, 1, 2, 3]
        t = torch.arange(4)
        # 使用 from_dlpack 方法将 PyTorch 数组 t 转换为 NumPy 数组 a
        a = np.from_dlpack(t)
        # 断言 NumPy 数组 a 和 PyTorch 数组 t 相等
        assert_array_equal(a, np.asarray(t))

    # 测试从 NumPy 数组转换为 PyTorch 数组的正确性
    def test_to_torch(self):
        # 创建一个 NumPy 数组 a，包含元素 [0, 1, 2, 3]
        a = np.arange(4)
        # 使用 torch.from_dlpack 方法将 NumPy 数组 a 转换为 PyTorch 数组 t
        t = torch.from_dlpack(a)
        # 断言 PyTorch 数组 t 和 NumPy 数组 a 相等
        assert_array_equal(np.asarray(t), a)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试
    run_tests()
```