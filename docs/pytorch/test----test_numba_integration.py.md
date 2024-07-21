# `.\pytorch\test\test_numba_integration.py`

```
# 导入单元测试模块
import unittest

# 导入PyTorch模块
import torch

# 导入PyTorch内部测试工具的公共函数
import torch.testing._internal.common_utils as common

# 导入PyTorch内部CUDA相关的测试工具
from torch.testing._internal.common_cuda import (
    TEST_CUDA,         # 是否测试CUDA
    TEST_MULTIGPU,     # 是否测试多GPU
    TEST_NUMBA_CUDA,   # 是否测试Numba CUDA
)

# 导入PyTorch内部测试工具的NumPy测试标志
from torch.testing._internal.common_utils import TEST_NUMPY

# 如果要进行NumPy测试，则导入NumPy模块
if TEST_NUMPY:
    import numpy

# 如果要进行Numba CUDA测试，则导入Numba CUDA模块
if TEST_NUMBA_CUDA:
    import numba.cuda


class TestNumbaIntegration(common.TestCase):
    # 定义测试类TestNumbaIntegration，继承自common.TestCase

    # 装饰器：如果不满足条件TEST_NUMPY，则跳过此测试
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    # 装饰器：如果不满足条件TEST_CUDA，则跳过此测试
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    # 定义一个测试方法，用于验证 CUDA 张量是否暴露了 __cuda_array_interface__

    types = [
        torch.DoubleTensor,     # 双精度浮点数张量
        torch.FloatTensor,      # 单精度浮点数张量
        torch.HalfTensor,       # 半精度浮点数张量
        torch.LongTensor,       # 长整型张量
        torch.IntTensor,        # 整型张量
        torch.ShortTensor,      # 短整型张量
        torch.CharTensor,       # 字符型张量
        torch.ByteTensor,       # 字节型张量
    ]
    dtypes = [
        numpy.float64,          # 双精度浮点数类型
        numpy.float32,          # 单精度浮点数类型
        numpy.float16,          # 半精度浮点数类型
        numpy.int64,            # 长整型类型
        numpy.int32,            # 整型类型
        numpy.int16,            # 短整型类型
        numpy.int8,             # 字节型整数类型
        numpy.uint8,            # 无符号字节型整数类型
    ]
    # 遍历每种类型的张量及其对应的 NumPy 数据类型
    for tp, npt in zip(types, dtypes):
        # 创建一个 CPU 张量对象
        cput = tp(10)

        # 验证 CPU 张量对象没有 __cuda_array_interface__ 属性
        self.assertFalse(hasattr(cput, "__cuda_array_interface__"))
        self.assertRaises(AttributeError, lambda: cput.__cuda_array_interface__)

        # 如果不是半精度张量，创建稀疏 CPU/CUDA 张量对象
        if tp not in (torch.HalfTensor,):
            indices_t = torch.empty(1, cput.size(0), dtype=torch.long).clamp_(min=0)
            sparse_t = torch.sparse_coo_tensor(indices_t, cput)

            # 验证稀疏 CPU 张量对象没有 __cuda_array_interface__ 属性
            self.assertFalse(hasattr(sparse_t, "__cuda_array_interface__"))
            self.assertRaises(AttributeError, lambda: sparse_t.__cuda_array_interface__)

            # 将稀疏 CPU 张量对象转换为 CUDA 张量对象
            sparse_cuda_t = torch.sparse_coo_tensor(indices_t, cput).cuda()

            # 验证稀疏 CUDA 张量对象没有 __cuda_array_interface__ 属性
            self.assertFalse(hasattr(sparse_cuda_t, "__cuda_array_interface__"))
            self.assertRaises(AttributeError, lambda: sparse_cuda_t.__cuda_array_interface__)

        # 创建 CUDA 张量对象
        cudat = tp(10).cuda()

        # 验证 CUDA 张量对象有 __cuda_array_interface__ 属性
        self.assertTrue(hasattr(cudat, "__cuda_array_interface__"))

        # 获取 CUDA 张量对象的 __cuda_array_interface__ 字典
        ar_dict = cudat.__cuda_array_interface__

        # 验证 __cuda_array_interface__ 字典包含指定的键
        self.assertEqual(
            set(ar_dict.keys()), {"shape", "strides", "typestr", "data", "version"}
        )

        # 验证 __cuda_array_interface__ 字典中的 shape 键值对应张量的形状
        self.assertEqual(ar_dict["shape"], (10,))
        # 验证 __cuda_array_interface__ 字典中的 strides 键为 None
        self.assertIs(ar_dict["strides"], None)
        # 验证 __cuda_array_interface__ 字典中的 typestr 键对应 NumPy 类型字符串
        # CUDA 本地端序为小端序
        self.assertEqual(ar_dict["typestr"], numpy.dtype(npt).newbyteorder("<").str)
        # 验证 __cuda_array_interface__ 字典中的 data 键对应张量的数据指针和只读属性
        self.assertEqual(ar_dict["data"], (cudat.data_ptr(), False))
        # 验证 __cuda_array_interface__ 字典中的 version 键为 2
        self.assertEqual(ar_dict["version"], 2)
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_array_adaptor(self):
        """测试函数，用于验证Tensor的CUDA适配器功能"""
    
        torch_dtypes = [
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
    
        for dt in torch_dtypes:
            # 创建CPU上的Tensor对象，各种类型的Tensor都不会注册为CUDA数组，尝试转换会引发类型错误。
            cput = torch.arange(10).to(dt)
            npt = cput.numpy()
    
            self.assertTrue(not numba.cuda.is_cuda_array(cput))
            with self.assertRaises(TypeError):
                numba.cuda.as_cuda_array(cput)
    
            # 任何CUDA上的Tensor都被视为CUDA数组。
            cudat = cput.to(device="cuda")
            self.assertTrue(numba.cuda.is_cuda_array(cudat))
    
            # 使用numba.cuda.as_cuda_array将CUDA Tensor转换为numba的CUDA数组对象。
            numba_view = numba.cuda.as_cuda_array(cudat)
            self.assertIsInstance(numba_view, numba.cuda.devicearray.DeviceNDArray)
    
            # CUDA数组的报告类型与CPU Tensor的NumPy类型匹配。
            self.assertEqual(numba_view.dtype, npt.dtype)
            self.assertEqual(numba_view.strides, npt.strides)
            self.assertEqual(numba_view.shape, cudat.shape)
    
            # 从主机（host）拷贝回CUDA，以便进行以下所有相等性检查，特别是需要用于float16比较，CPU端不支持的类型。
            # 视图中的数据是相同的。
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to("cuda"))
    
            # 对Torch.Tensor的写入会反映在numba数组中。
            cudat[:5] = 11
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to("cuda"))
    
            # 支持跨步（strided）Tensor。
            strided_cudat = cudat[::2]
            strided_npt = cput[::2].numpy()
            strided_numba_view = numba.cuda.as_cuda_array(strided_cudat)
    
            self.assertEqual(strided_numba_view.dtype, strided_npt.dtype)
            self.assertEqual(strided_numba_view.strides, strided_npt.strides)
            self.assertEqual(strided_numba_view.shape, strided_cudat.shape)
    
            # 截至numba 0.40.0，对跨步视图的支持是有限的，不能验证跨步视图操作的正确性。
    def test_conversion_errors(self):
        """Numba properly detects array interface for tensor.Tensor variants."""

        # 创建一个 CPU 张量（tensor），包含100个元素
        cput = torch.arange(100)

        # 断言：CPU 张量不是 CUDA 数组
        self.assertFalse(numba.cuda.is_cuda_array(cput))
        
        # 使用 numba 尝试将 CPU 张量转换为 CUDA 数组，预期抛出 TypeError
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cput)

        # 创建一个稀疏张量（sparse tensor），包含100个元素
        sparset = torch.sparse_coo_tensor(cput[None, :], cput)

        # 断言：稀疏张量不是 CUDA 数组
        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        
        # 使用 numba 尝试将稀疏张量转换为 CUDA 数组，预期抛出 TypeError
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)

        # 将稀疏张量转移到 CUDA 设备上
        sparse_cuda_t = sparset.cuda()

        # 断言：转移到 CUDA 设备的稀疏张量仍然不是 CUDA 数组
        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        
        # 使用 numba 尝试将稀疏张量转换为 CUDA 数组，预期抛出 TypeError
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)

        # 创建一个带梯度的 CPU 张量
        cpu_gradt = torch.zeros(100).requires_grad_(True)

        # 断言：带梯度的 CPU 张量不是 CUDA 数组
        self.assertFalse(numba.cuda.is_cuda_array(cpu_gradt))
        
        # 使用 numba 尝试将带梯度的 CPU 张量转换为 CUDA 数组，预期抛出 TypeError
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cpu_gradt)

        # 创建一个带梯度的 CUDA 张量
        cuda_gradt = torch.zeros(100).requires_grad_(True).cuda()

        # 断言：带梯度的 CUDA 张量在检查或转换时会引发 RuntimeError
        with self.assertRaises(RuntimeError):
            numba.cuda.is_cuda_array(cuda_gradt)
        with self.assertRaises(RuntimeError):
            numba.cuda.as_cuda_array(cuda_gradt)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    @unittest.skipIf(not TEST_MULTIGPU, "No multigpu")
    def test_active_device(self):
        """'as_cuda_array' tensor device must match active numba context."""

        # 创建一个 CUDA 张量，默认在设备 0 上
        cudat = torch.arange(10, device="cuda")
        
        # 断言：CUDA 张量在设备 0 上
        self.assertEqual(cudat.device.index, 0)
        
        # 使用 numba 将 CUDA 张量转换为 CUDA 数组，并断言返回的类型是 DeviceNDArray
        self.assertIsInstance(
            numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray
        )

        # 创建一个在非默认设备上的 CUDA 张量
        cudat = torch.arange(10, device=torch.device("cuda", 1))

        # 使用 numba 尝试将非默认设备上的 CUDA 张量转换为 CUDA 数组，预期抛出 CudaAPIError
        with self.assertRaises(numba.cuda.driver.CudaAPIError):
            numba.cuda.as_cuda_array(cudat)

        # 在切换到设备上下文后，可以成功将 CUDA 张量转换为 CUDA 数组
        with numba.cuda.devices.gpus[cudat.device.index]:
            self.assertIsInstance(
                numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray
            )
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_from_cuda_array_interface_inferred_strides(self):
        """测试函数：test_from_cuda_array_interface_inferred_strides
    
        这个测试函数验证了在使用torch.as_tensor(numba_ary)时，应正确推断（连续）的步长。
    
        """
        # 在理论上，这个测试可以与test_from_cuda_array_interface合并，但是那个测试
        # 过于严格：它检查导出的协议是否完全相同，这无法处理不同的导出协议版本。
        dtypes = [
            numpy.float64,
            numpy.float32,
            numpy.int64,
            numpy.int32,
            numpy.int16,
            numpy.int8,
            numpy.uint8,
        ]
        for dtype in dtypes:
            numpy_ary = numpy.arange(6).reshape(2, 3).astype(dtype)
            numba_ary = numba.cuda.to_device(numpy_ary)
            self.assertTrue(numba_ary.is_c_contiguous())
            torch_ary = torch.as_tensor(numba_ary, device="cuda")
            self.assertTrue(torch_ary.is_contiguous())
    
    @unittest.skip(
        "Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418"
    )
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_from_cuda_array_interface_lifetime(self):
        """测试函数：test_from_cuda_array_interface_lifetime
    
        这个测试函数验证了torch.as_tensor(obj)张量会获取对obj的引用，使得obj的生命周期超过张量。
    
        """
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device="cuda")
        self.assertEqual(
            torch_ary.__cuda_array_interface__, numba_ary.__cuda_array_interface__
        )  # No copy
        del numba_ary
        self.assertEqual(
            torch_ary.cpu().data.numpy(), numpy.arange(6)
        )  # `torch_ary` is still alive
    
    @unittest.skip(
        "Test is temporary disabled, see https://github.com/pytorch/pytorch/issues/54418"
    )
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    @unittest.skipIf(not TEST_MULTIGPU, "No multigpu")
    # 定义一个测试方法，用于验证 torch.as_tensor() 的行为
    def test_from_cuda_array_interface_active_device(self):
        """torch.as_tensor() tensor device must match active numba context."""

        # 零拷贝：当 torch 和 numba 都默认使用设备 0 时，它们可以自由互操作
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device="cuda")
        self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary))
        self.assertEqual(
            torch_ary.__cuda_array_interface__, numba_ary.__cuda_array_interface__
        )

        # 隐式拷贝：当 Numba 和 Torch 使用不同的设备时
        numba_ary = numba.cuda.to_device(numpy.arange(6))
        torch_ary = torch.as_tensor(numba_ary, device=torch.device("cuda", 1))
        self.assertEqual(torch_ary.get_device(), 1)
        self.assertEqual(torch_ary.cpu().data.numpy(), numpy.asarray(numba_ary))
        # 检查 Torch 和 Numba 的 CUDA 数组接口是否相同，除了 "data" 字段
        if1 = torch_ary.__cuda_array_interface__
        if2 = numba_ary.__cuda_array_interface__
        self.assertNotEqual(if1["data"], if2["data"])
        del if1["data"]
        del if2["data"]
        self.assertEqual(if1, if2)
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行下面的代码块
if __name__ == "__main__":
    # 调用 common 模块中的 run_tests 函数来执行测试
    common.run_tests()
```