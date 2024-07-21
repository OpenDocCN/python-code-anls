# `.\pytorch\test\test_tensor_creation_ops.py`

```py
# Owner(s): ["module: tensor creation"]

import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库

import sys  # 导入sys模块
import math  # 导入math模块
import warnings  # 导入warnings模块
import unittest  # 导入unittest模块
from itertools import product, combinations, combinations_with_replacement, permutations  # 导入itertools模块中的函数
import random  # 导入random模块
import tempfile  # 导入tempfile模块
from typing import Any, Dict, List, Tuple  # 导入类型提示相关模块

from torch.testing import make_tensor  # 导入测试相关模块
from torch.testing._internal.common_utils import (  # 导入测试工具相关模块
    TestCase, run_tests, do_test_empty_full, TEST_WITH_ROCM, suppress_warnings,
    torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict, slowTest,
    set_default_dtype, set_default_tensor_type,
    TEST_SCIPY, IS_MACOS, IS_PPC, IS_JETSON, IS_WINDOWS, parametrize, skipIfTorchDynamo,
    xfailIfTorchDynamo)
from torch.testing._internal.common_device_type import (  # 导入测试设备相关模块
    expectedFailureMeta, instantiate_device_type_tests, deviceCountAtLeast, onlyNativeDeviceTypes,
    onlyCPU, largeTensorTest, precisionOverride, dtypes,
    onlyCUDA, skipCPUIf, dtypesIfCUDA, dtypesIfCPU, skipMeta)
from torch.testing._internal.common_dtype import (  # 导入数据类型相关模块
    all_types_and_complex, all_types_and_complex_and, all_types_and, floating_and_complex_types, complex_types,
    floating_types, floating_and_complex_types_and, integral_types, integral_types_and, get_all_dtypes,
    float_to_corresponding_complex_type_map
)

from torch.utils.dlpack import to_dlpack  # 导入DLpack相关模块

# TODO: replace with make_tensor
# 生成输入数据的函数，根据指定的形状、数据类型和设备生成张量
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)  # 如果形状是空的，创建空张量
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # 解决 torch.randn 对 bfloat16 类型不支持的问题
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)  # 生成随机张量并转换为bfloat16类型
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)  # 生成随机张量
            x[torch.randn(*shape) > 0.5] = 0  # 随机部分置为0
            if with_extremal and dtype.is_floating_point:
                # 使用极端值
                x[torch.randn(*shape) > 0.5] = float('nan')  # 随机部分置为NaN
                x[torch.randn(*shape) > 0.5] = float('inf')  # 随机部分置为正无穷
                x[torch.randn(*shape) > 0.5] = float('-inf')  # 随机部分置为负无穷
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')  # 随机部分置为复数NaN
                x[torch.randn(*shape) > 0.5] = complex('inf')  # 随机部分置为复数正无穷
                x[torch.randn(*shape) > 0.5] = complex('-inf')  # 随机部分置为复数负无穷
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)  # 创建布尔类型的零张量
            x[torch.randn(*shape) > 0.5] = True  # 随机部分置为True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)  # 创建指定范围内的随机整数张量

    return x  # 返回生成的张量

# TODO: replace with make_tensor
# 随机生成指定维度、指定范围的形状
def _rand_shape(dim, min_size, max_size):
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))  # 在指定范围内生成随机整数
    return tuple(shape)  # 返回生成的形状元组

# Test suite for tensor creation ops
#
# 创建一个测试类 TestTensorCreation，用于测试张量的创建操作。
class TestTensorCreation(TestCase):
    # 精确比较数据类型标志，暂未使用。
    exact_dtype = True

    # 限定仅在 CPU 上运行，且数据类型为 torch.float 时执行该测试方法。
    @onlyCPU
    @dtypes(torch.float)
    def test_diag_embed(self, device, dtype):
        # 创建一个形状为 (3, 4) 的张量 x，其中元素为从 0 到 11 的整数，数据类型为指定的 dtype。
        x = torch.arange(3 * 4, dtype=dtype, device=device).view(3, 4)
        # 对 x 进行对角嵌入操作，返回一个新的张量 result。
        result = torch.diag_embed(x)
        # 创建期望结果 expected，其为 x 中每行的对角线形成的张量的堆叠。
        expected = torch.stack([torch.diag(r) for r in x], 0)
        # 断言 result 与 expected 相等。
        self.assertEqual(result, expected)

        # 对 x 进行带有指定参数的对角嵌入操作，返回一个新的张量 result。
        result = torch.diag_embed(x, offset=1, dim1=0, dim2=2)
        # 创建期望结果 expected，其为 x 中每行的偏移对角线形成的张量的堆叠。
        expected = torch.stack([torch.diag(r, 1) for r in x], 1)
        # 断言 result 与 expected 相等。
        self.assertEqual(result, expected)

    # 测试 torch.cat 函数在内存重叠情况下的行为。
    def test_cat_mem_overlap(self, device):
        # 创建形状为 (1, 3) 的随机张量 x，并将其扩展为形状为 (6, 3)。
        x = torch.rand((1, 3), device=device).expand((6, 3))
        # 创建形状为 (3, 3) 的随机张量 y。
        y = torch.rand((3, 3), device=device)
        # 使用 assertRaisesRegex 断言，在执行 torch.cat([y, y], out=x) 时抛出 RuntimeError 异常，并且异常信息包含 'unsupported operation'。
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.cat([y, y], out=x)

    # 限定仅在原生设备类型上执行该测试方法。
    @onlyNativeDeviceTypes
    def test_vander(self, device):
        # 创建一个设备为 device 的张量 x，其值为 [1, 2, 3, 5]。
        x = torch.tensor([1, 2, 3, 5], device=device)

        # 断言 torch.vander(torch.tensor([]), 0) 的形状为 (0, 0)。
        self.assertEqual((0, 0), torch.vander(torch.tensor([]), 0).shape)

        # 使用 assertRaisesRegex 断言，在执行 torch.vander(x, N=-1) 时抛出 RuntimeError 异常，并且异常信息包含 "N must be non-negative."。
        with self.assertRaisesRegex(RuntimeError, "N must be non-negative."):
            torch.vander(x, N=-1)

        # 使用 assertRaisesRegex 断言，在执行 torch.vander(torch.stack((x, x))) 时抛出 RuntimeError 异常，并且异常信息包含 "x must be a one-dimensional tensor."。
        with self.assertRaisesRegex(RuntimeError, "x must be a one-dimensional tensor."):
            torch.vander(torch.stack((x, x)))

    # 限定仅在原生设备类型上，且多种数据类型下执行该测试方法。
    @onlyNativeDeviceTypes
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.short, torch.int, torch.long,
            torch.float, torch.double,
            torch.cfloat, torch.cdouble)
    # 测试函数，用于验证不同数据类型和设备上的 Vandermonde 矩阵生成函数的行为
    def test_vander_types(self, device, dtype):
        if dtype is torch.uint8:
            # 注意：uint8 类型没有负值
            X = [[1, 2, 3, 5], [0, 1 / 3, 1, math.pi, 3 / 7]]
        elif dtype is torch.bool:
            # 注意：参考 https://github.com/pytorch/pytorch/issues/37398
            # 确认此处的必要性
            X = [[True, True, True, True], [False, True, True, True, True]]
        elif dtype in [torch.cfloat, torch.cdouble]:
            X = [[1 + 1j, 1 + 0j, 0 + 1j, 0 + 0j],
                 [2 + 2j, 3 + 2j, 4 + 3j, 5 + 4j]]
        else:
            X = [[1, 2, 3, 5], [-math.pi, 0, 1 / 3, 1, math.pi, 3 / 7]]

        N = [None, 0, 1, 3]
        increasing = [False, True]

        # 使用 product 函数迭代 X, N, increasing 的组合
        for x, n, inc in product(X, N, increasing):
            # 根据 dtype 获取对应的 numpy 数据类型
            numpy_dtype = torch_to_numpy_dtype_dict[dtype]
            # 创建 torch 张量 pt_x 和 numpy 数组 np_x
            pt_x = torch.tensor(x, device=device, dtype=dtype)
            np_x = np.array(x, dtype=numpy_dtype)

            # 使用 torch.vander() 和 np.vander() 生成 Vandermonde 矩阵
            pt_res = torch.vander(pt_x, increasing=inc) if n is None else torch.vander(pt_x, n, inc)
            np_res = np.vander(np_x, n, inc)

            # 断言两者的结果是否相等
            self.assertEqual(
                pt_res,
                torch.from_numpy(np_res),
                atol=1e-3,
                rtol=0,
                exact_dtype=False)

    # 测试函数，用于验证在所有数据类型和设备上进行 torch.cat() 操作的行为
    def test_cat_all_dtypes_and_devices(self, device):
        # 遍历所有数据类型，包括复杂类型和特定数据类型
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.chalf):
            # 创建一个 tensor x
            x = torch.tensor([[1, 2], [3, 4]], dtype=dt, device=device)

            # 预期的结果 tensor expected1 和 expected2
            expected1 = torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 0), expected1)

            expected2 = torch.tensor([[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 1), expected2)

    # 测试函数，用于验证在所有数据类型和设备上进行 tensor 填充操作的行为
    def test_fill_all_dtypes_and_devices(self, device):
        # 遍历所有数据类型，包括复杂类型和特定数据类型
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.chalf):
            for x in [torch.tensor((10, 10), dtype=dt, device=device),
                      torch.empty(10000, dtype=dt, device=device)]:  # 大型 tensor
                numel = x.numel()
                bound = 100 if dt in (torch.uint8, torch.int8) else 2000
                # 遍历不同的填充值 n
                for n in range(-bound, bound, bound // 10):
                    # 填充 tensor x
                    x.fill_(n)
                    # 断言填充后的结果是否与预期一致，并且数据类型是否正确
                    self.assertEqual(x, torch.tensor([n] * numel, dtype=dt, device=device))
                    self.assertEqual(dt, x.dtype)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_diagflat(self, device):
        # 测试 torch.diagflat 函数

        # 定义数据类型为 float32
        dtype = torch.float32
        
        # 基本的健全性测试，生成一个设备上的随机张量 x
        x = torch.randn((100,), dtype=dtype, device=device)
        
        # 使用 torch.diagflat 对 x 进行对角化操作
        result = torch.diagflat(x)
        
        # 使用 torch.diag 生成预期结果
        expected = torch.diag(x)
        
        # 断言结果与预期相等
        self.assertEqual(result, expected)

        # 测试偏移参数的情况
        x = torch.randn((100,), dtype=dtype, device=device)
        result = torch.diagflat(x, 17)
        expected = torch.diag(x, 17)
        self.assertEqual(result, expected)

        # 测试输入张量具有多个维度的情况
        x = torch.randn((2, 3, 4), dtype=dtype, device=device)
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # 非连续内存的输入张量测试
        x = torch.randn((2, 3, 4), dtype=dtype, device=device).transpose(2, 0)
        self.assertFalse(x.is_contiguous())
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # 复数支持测试
        result = torch.diagflat(torch.ones(4, dtype=torch.complex128))
        expected = torch.eye(4, dtype=torch.complex128)
        self.assertEqual(result, expected)

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    def test_block_diag_scipy(self, device):
        # 使用 Scipy 实现的 block_diag 函数进行测试

        # 导入 Scipy 的线性代数模块
        import scipy.linalg
        
        # 定义多个 Scipy 张量列表
        scipy_tensors_list = [
            [
                1,
                [2],
                [],
                [3, 4, 5],
                [[], []],
                [[6], [7.3]]
            ],
            [
                [[1, 2], [3, 4]],
                [1]
            ],
            [
                [[4, 9], [7, 10]],
                [4.6, 9.12],
                [1j + 3]
            ],
            []
        ]

        # 预期的 Torch 数据类型列表
        expected_torch_types = [
            torch.float32,
            torch.int64,
            torch.complex64,
            torch.float32
        ]

        # 预期的 Scipy 数据类型列表
        expected_scipy_types = [
            torch.float64,
            torch.int32 if IS_WINDOWS else torch.int64,
            torch.complex128,
            torch.float64
        ]

        # 遍历并比较每组张量
        for scipy_tensors, torch_type, scipy_type in zip(scipy_tensors_list, expected_torch_types, expected_scipy_types):
            # 将 Scipy 张量列表转换为 Torch 张量列表
            torch_tensors = [torch.tensor(t, device=device) for t in scipy_tensors]
            
            # 使用 torch.block_diag 对 Torch 张量进行 block_diag 操作
            torch_result = torch.block_diag(*torch_tensors)
            
            # 断言 Torch 结果的数据类型与预期相同
            self.assertEqual(torch_result.dtype, torch_type)

            # 使用 Scipy 实现的 block_diag 函数得到 Scipy 结果
            scipy_result = torch.tensor(
                scipy.linalg.block_diag(*scipy_tensors),
                device=device
            )
            
            # 断言 Scipy 结果的数据类型与预期相同，并转换为相同的 Torch 类型
            self.assertEqual(scipy_result.dtype, scipy_type)
            scipy_result = scipy_result.to(torch_type)

            # 断言 Torch 结果与 Scipy 结果相等
            self.assertEqual(torch_result, scipy_result)

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.float32, torch.float64)
    # 测试 torch.complex 函数
    def test_torch_complex(self, device, dtype):
        # 创建实部张量 [1, 2]，设备为指定设备，数据类型为指定数据类型
        real = torch.tensor([1, 2], device=device, dtype=dtype)
        # 创建虚部张量 [3, 4]，设备为指定设备，数据类型为指定数据类型
        imag = torch.tensor([3, 4], device=device, dtype=dtype)
        # 使用实部和虚部张量创建复数张量 z
        z = torch.complex(real, imag)
        # 确定复数类型与输入数据类型对应的映射关系
        complex_dtype = float_to_corresponding_complex_type_map[dtype]
        # 断言创建的复数张量 z 是否与预期值一致
        self.assertEqual(torch.tensor([1.0 + 3.0j, 2.0 + 4.0j], dtype=complex_dtype), z)

    # 标记仅适用于本地设备类型的测试函数装饰器
    @onlyNativeDeviceTypes
    # 标记仅适用于指定数据类型的测试函数装饰器
    @dtypes(torch.float32, torch.float64)
    # 测试 torch.polar 函数
    def test_torch_polar(self, device, dtype):
        # 创建绝对值张量 [1, 2, -3, -4.5, 1, 1]，设备为指定设备，数据类型为指定数据类型
        abs = torch.tensor([1, 2, -3, -4.5, 1, 1], device=device, dtype=dtype)
        # 创建角度张量 [π/2, 5π/4, 0, -11π/6, π, -π]，设备为指定设备，数据类型为指定数据类型
        angle = torch.tensor([math.pi / 2, 5 * math.pi / 4, 0, -11 * math.pi / 6, math.pi, -math.pi],
                             device=device, dtype=dtype)
        # 使用绝对值和角度张量创建极坐标形式的复数张量 z
        z = torch.polar(abs, angle)
        # 根据数据类型确定复数类型的精度
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        # 断言创建的复数张量 z 是否与预期值一致，设置绝对误差和相对误差容忍度
        self.assertEqual(torch.tensor([1j, -1.41421356237 - 1.41421356237j, -3,
                                       -3.89711431703 - 2.25j, -1, -1],
                                      dtype=complex_dtype),
                         z, atol=1e-5, rtol=1e-5)

    # 标记仅适用于本地设备类型的测试函数装饰器
    @onlyNativeDeviceTypes
    # 标记仅适用于指定数据类型的测试函数装饰器
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.complex64, torch.complex128, torch.bool)
    # 测试当数据类型为复数或浮点数时，torch.complex 和 torch.polar 函数是否抛出错误
    def test_torch_complex_floating_dtype_error(self, device, dtype):
        for op in (torch.complex, torch.polar):
            # 创建张量 a 和 b，设备为指定设备，数据类型为指定数据类型
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            # 设置预期的错误消息正则表达式
            error = r"Expected both inputs to be Half, Float or Double tensors but " \
                    r"got [A-Za-z]+ and [A-Za-z]+"
        # 断言调用 torch.complex 或 torch.polar 函数时是否抛出预期的 RuntimeError 错误
        with self.assertRaisesRegex(RuntimeError, error):
            op(a, b)

    # 标记仅适用于本地设备类型的测试函数装饰器
    @onlyNativeDeviceTypes
    # 标记仅适用于指定数据类型的测试函数装饰器
    @dtypes(torch.float32, torch.float64)
    # 测试当复数类型不一致时，torch.complex 和 torch.polar 函数是否抛出错误
    def test_torch_complex_same_dtype_error(self, device, dtype):

        # 定义函数，根据数据类型返回对应的名称
        def dtype_name(dtype):
            return 'Float' if dtype == torch.float32 else 'Double'

        for op in (torch.complex, torch.polar):
            # 根据当前数据类型确定另一种数据类型
            other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
            # 创建张量 a，设备为指定设备，数据类型为当前数据类型
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            # 创建张量 b，设备为指定设备，数据类型为另一种数据类型
            b = torch.tensor([3, 4], device=device, dtype=other_dtype)
            # 设置预期的错误消息
            error = f"Expected object of scalar type {dtype_name(dtype)} but got scalar type " \
                    f"{dtype_name(other_dtype)} for second argument"
            # 断言调用 torch.complex 或 torch.polar 函数时是否抛出预期的 RuntimeError 错误
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    # 标记仅适用于本地设备类型的测试函数装饰器
    @onlyNativeDeviceTypes
    # 标记仅适用于指定数据类型的测试函数装饰器
    @dtypes(torch.float32, torch.float64)
    # 在测试方法中，测试 torch.complex 和 torch.polar 函数在特定设备和数据类型下的输出类型错误情况

    def test_torch_complex_out_dtype_error(self, device, dtype):
        # 定义一个内部函数，根据数据类型返回对应的名称字符串，用于错误消息的格式化
        def dtype_name(dtype):
            return 'Float' if dtype == torch.float32 else 'Double'

        # 定义一个内部函数，根据复数数据类型返回对应的名称字符串，用于错误消息的格式化
        def complex_dtype_name(dtype):
            return 'ComplexFloat' if dtype == torch.complex64 else 'ComplexDouble'

        # 遍历 torch.complex 和 torch.polar 函数
        for op in (torch.complex, torch.polar):
            # 创建张量 a 和 b，指定设备和数据类型
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            # 创建输出张量 out，全零初始化，指定设备和数据类型
            out = torch.zeros(2, device=device, dtype=dtype)
            # 根据输入数据类型确定期望的输出数据类型
            expected_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            # 构造错误消息字符串，描述预期的数据类型和实际得到的数据类型不匹配
            error = f"Expected object of scalar type {complex_dtype_name(expected_dtype)} but got scalar type " \
                    f"{dtype_name(dtype)} for argument 'out'"
            # 使用断言确保在调用 op 函数时会抛出 RuntimeError，并且错误消息符合预期格式
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b, out=out)

    # 测试方法，用于测试 torch.cat 在处理空张量时的行为
    def test_cat_empty_legacy(self, device):
        # 提示：这是一个遗留行为，当我们支持具有任意大小的空张量时，应予以移除

        # 指定数据类型为 torch.float32
        dtype = torch.float32

        # 创建形状为 (4, 3, 32, 32) 的随机张量 x，指定设备和数据类型
        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        # 创建形状为 (0,) 的随机张量 empty，指定设备和数据类型
        empty = torch.randn((0,), dtype=dtype, device=device)

        # 在维度 1 上拼接张量 x 和 empty，得到 res1 和 res2
        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        # 使用断言确保 res1 和 res2 相等
        self.assertEqual(res1, res2)

        # 在维度 1 上拼接两个空张量 empty，得到 res1
        res1 = torch.cat([empty, empty], dim=1)
        # 使用断言确保 res1 等于空张量 empty
        self.assertEqual(res1, empty)

    # 测试方法，用于测试 torch.cat 在处理空张量时的行为
    def test_cat_empty(self, device):
        # 指定数据类型为 torch.float32
        dtype = torch.float32

        # 创建形状为 (4, 3, 32, 32) 的随机张量 x，指定设备和数据类型
        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        # 创建形状为 (4, 0, 32, 32) 的随机张量 empty，指定设备和数据类型
        empty = torch.randn((4, 0, 32, 32), dtype=dtype, device=device)

        # 在维度 1 上拼接张量 x 和 empty，得到 res1 和 res2
        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        # 使用断言确保 res1 和 res2 相等
        self.assertEqual(res1, res2)

        # 在维度 1 上拼接两个空张量 empty，得到 res1
        res1 = torch.cat([empty, empty], dim=1)
        # 使用断言确保 res1 等于空张量 empty
        self.assertEqual(res1, empty)
    # 定义一个测试方法，测试 torch.cat 方法的不同用法
    def test_cat_out(self, device):
        # 创建一个大小为 (0,) 的零张量 x，并将设备指定为 device
        x = torch.zeros((0), device=device)
        # 创建一个形状为 (4, 6) 的随机张量 y，并将设备指定为 device
        y = torch.randn((4, 6), device=device)

        # 将 y 展平成一维张量，并克隆为 w
        w = y.view(-1).clone()
        # 使用 torch.cat 将 w 的部分片段拼接起来形成张量 a
        a = torch.cat([w[:2], w[4:6]])
        # 使用 torch.cat 将 w 的部分片段拼接起来形成张量 b，并将结果写入 w 的指定位置
        b = torch.cat([w[:2], w[4:6]], out=w[6:10])
        # 断言张量 a 和 b 相等
        self.assertEqual(a, b)
        # 断言张量 a 和 w 的指定片段相等
        self.assertEqual(a, w[6:10])
        # 断言张量 w 和 y 展平后的前六个元素相等
        self.assertEqual(w[:6], y.view(-1)[:6])

        # Case:
        # 引用: https://github.com/pytorch/pytorch/issues/49878
        # 遍历维度列表 [0, 1]
        for dim in [0, 1]:
            # 创建形状为 (10, 5, 2) 的零张量 x，并将设备指定为 device
            x = torch.zeros((10, 5, 2), device=device)

            # 随机生成一个长度，使得 y 在指定维度上的长度减少
            random_length = random.randint(1, 4)
            # 使用 x.narrow 在指定维度上获取子张量 y
            y = x.narrow(dim, 0, x.shape[dim] - random_length)
            # 创建一个与 y[0] 形状相同的全值张量 val，并将设备指定为 device
            val = torch.full_like(y[0], 3., device=device)

            # 如果 dim 为 0，则断言 y 是连续的张量
            if dim == 0:
                self.assertTrue(y.is_contiguous())
            else:
                self.assertFalse(y.is_contiguous())

            # 使用 torch.cat 在指定维度上拼接张量 val 的复制，将结果写入 y
            torch.cat((val[None],) * y.shape[0], dim=0, out=y)

            # 创建预期的张量 expected_y，在指定维度上拼接张量 val 的复制
            expected_y = torch.cat((val[None],) * y.shape[0], dim=0)
            # 创建预期的张量 expected_x，初始化为零张量与 x 相同的形状
            expected_x = torch.zeros((10, 5, 2), device=device)
            # 根据不同的 dim 设置预期的张量 expected_x 的部分值
            if dim == 0:
                expected_x[:x.shape[dim] - random_length, :, :] = expected_y
            elif dim == 1:
                expected_x[:, :x.shape[dim] - random_length, :] = expected_y

            # 断言张量 y 与预期的张量 expected_y 相等
            self.assertEqual(y, expected_y)
            # 断言张量 x 与预期的张量 expected_x 相等
            self.assertEqual(x, expected_x)

    # 使用装饰器 dtypes 标记该方法适用于所有类型和复杂类型，以及指定的 uint16、uint32、uint64 类型
    @dtypes(*all_types_and_complex(), torch.uint16, torch.uint32, torch.uint64)
    # 定义测试方法，测试在 channels_last 内存格式下的 torch.cat 方法
    def test_cat_out_channels_last(self, device):
        # 创建形状为 (4, 3, 8, 8) 的随机张量 x
        x = torch.randn((4, 3, 8, 8))
        # 创建形状与 x 相同的随机张量 y
        y = torch.randn(x.shape)
        # 使用 torch.cat 在默认内存格式下拼接张量 x 和 y
        res1 = torch.cat((x, y))
        # 将 res1 克隆为 channels_last 内存格式的张量 z
        z = res1.clone().contiguous(memory_format=torch.channels_last)
        # 使用 torch.cat 在 channels_last 内存格式下拼接张量 x 和 y，并将结果写入 z
        res2 = torch.cat((x, y), out=z)
        # 断言拼接结果 res1 和 res2 相等
        self.assertEqual(res1, res2)

    # 使用装饰器 onlyNativeDeviceTypes 标记该方法仅适用于原生设备类型
    @onlyNativeDeviceTypes
    # 定义测试方法，测试在 channels_last 内存格式下的 torch.cat 方法
    def test_cat_in_channels_last(self, device):
        # 遍历维度范围 [0, 1, 2, 3]
        for dim in range(4):
            # 创建形状为 (4, 15, 8, 8) 的随机张量 x，并将设备指定为 device
            x = torch.randn((4, 15, 8, 8), device=device)
            # 创建形状与 x 相同的随机张量 y，并将设备指定为 device
            y = torch.randn(x.shape, device=device)
            # 使用 torch.cat 在指定维度上拼接张量 x 和 y，形成结果张量 res1
            res1 = torch.cat((x, y), dim=dim)
            # 将 x 和 y 克隆为 channels_last 内存格式的张量
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            # 使用 torch.cat 在 channels_last 内存格式下拼接张量 x 和 y，形成结果张量 res2
            res2 = torch.cat((x, y), dim=dim)
            # 断言结果张量 res2 在 channels_last 内存格式下是连续的
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            # 断言拼接结果 res1 和 res2 相等
            self.assertEqual(res1, res2)

            # 创建大于 grain size 的随机形状为 (4, 15, 256, 256) 的张量 x，并将设备指定为 device
            x = torch.randn((4, 15, 256, 256), device=device)
            # 创建形状与 x 相同的随机张量 y，并将设备指定为 device
            y = torch.randn(x.shape, device=device)
            # 使用 torch.cat 在指定维度上拼接张量 x 和 y，形成结果张量 res1
            res1 = torch.cat((x, y), dim=dim)
            # 将 x 和 y 克隆为 channels_last 内存格式的张量
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            # 使用 torch.cat 在 channels_last 内存格式下拼接张量 x 和 y，形成结果张量 res2
            res2 = torch.cat((x, y), dim=dim)
            # 断言结果张量 res2 在 channels_last 内存格式下是连续的
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            # 断言拼接结果 res1 和 res2 相等
            self.assertEqual(res1, res2)
    # 定义一个测试方法，用于测试在保留 channels_last 内存格式下的 torch.cat 操作
    def test_cat_preserve_channels_last(self, device):
        # 创建一个形状为 (4, 3, 8, 8) 的张量 x，其中 4 是批量大小，3 是通道数，8x8 是空间维度，使用指定设备
        x = torch.randn((4, 3, 8, 8), device=device)
        # 创建一个与 x 形状相同的随机张量 y，使用指定设备
        y = torch.randn(x.shape, device=device)
        # 对 x 和 y 进行 torch.cat 操作，沿第一维（默认）拼接
        res1 = torch.cat((x, y))
        # 将 x 和 y 转换为 channels_last 内存格式后再进行 torch.cat 操作
        res2 = torch.cat((x.contiguous(memory_format=torch.channels_last), y.contiguous(memory_format=torch.channels_last)))
        # 断言 res1 和 res2 在数值上相等
        self.assertEqual(res1, res2)
        # 断言 res2 在 channels_last 内存格式下是连续的
        self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
        
        # 创建一个在 channels_last 内存格式下的张量 x，形状为 (2, 2, 3, 2)，数据从 0 到 23，使用指定设备
        x = torch.arange(24, dtype=torch.float, device=device).reshape(2, 2, 3, 2).to(memory_format=torch.channels_last)
        # 从 x 中切片出 x1 和 x2，x1 包含前两个列，x2 包含后两个列
        x1 = x[:, :, :2]
        x2 = x[:, :, 1:]
        # 对 x1 和 x2 沿最后一个维度进行 torch.cat 操作
        res1 = torch.cat((x1, x2), dim=-1)
        # 分别对 x1 和 x2 进行连续化处理后再进行 torch.cat 操作
        res2 = torch.cat((x1.contiguous(), x2.contiguous()), dim=-1)
        # 断言 res1 和 res2 在数值上相等
        self.assertEqual(res1, res2)
        # 断言 res1 在 channels_last 内存格式下是连续的
        self.assertTrue(res1.is_contiguous(memory_format=torch.channels_last))
    # 定义一个测试函数，用于测试 torch.cat 操作在不同内存格式和设备上的行为
    def test_cat_out_memory_format(self, device):
        # 定义输入张量的大小
        inp_size = (4, 4, 4, 4)
        # 定义预期输出张量的大小
        expected_size = (8, 4, 4, 4)
        
        # 在 CUDA 设备上生成随机张量 a_cuda，并使用 channels_last 内存格式保证连续性
        a_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.channels_last)
        # 在 CPU 上生成随机张量 a_cpu，并使用 channels_last 内存格式保证连续性
        a_cpu = torch.randn(inp_size, device='cpu').contiguous(memory_format=torch.channels_last)
        
        # 在 CUDA 设备上生成随机张量 b_cuda，并使用默认的 contiguous_format 内存格式保证连续性
        b_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.contiguous_format)
        # 在 CPU 上生成随机张量 b_cpu，并使用默认的 contiguous_format 内存格式保证连续性
        b_cpu = torch.randn(inp_size, device='cpu').contiguous(memory_format=torch.contiguous_format)
        
        # 在 CUDA 设备上生成随机张量 c_cuda，并使用 channels_last 内存格式保证连续性
        c_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.channels_last)

        # Case 1: 如果指定了 out= 参数并且其形状正确，则 res1_cuda 和 res1_cpu 将保持给定的内存格式
        # 在 CUDA 设备上创建一个空的输出张量 out_cuda，并使用 contiguous_format 内存格式保证连续性
        out_cuda = torch.empty(expected_size, device=device).contiguous(memory_format=torch.contiguous_format)
        # 执行 torch.cat 操作，将 a_cuda 和 b_cuda 连接到 out_cuda 上
        res1_cuda = torch.cat((a_cuda, b_cuda), out=out_cuda)
        
        # 在 CPU 上创建一个空的输出张量 out_cpu，并使用 contiguous_format 内存格式保证连续性
        out_cpu = torch.empty(expected_size, device='cpu').contiguous(memory_format=torch.contiguous_format)
        # 执行 torch.cat 操作，将 a_cpu 和 b_cpu 连接到 out_cpu 上
        res1_cpu = torch.cat((a_cpu, b_cpu), out=out_cpu)

        # 断言 res1_cuda 和 res1_cpu 的内存格式是否符合预期
        self.assertTrue(res1_cuda.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(res1_cpu.is_contiguous(memory_format=torch.contiguous_format))

        # Case 2: 如果指定了 out= 参数但其形状不正确，则 res2_cuda 和 res2_cpu 将在内部重新调整尺寸
        # - 对于 CPU 和 CUDA 变体，只有当所有张量具有相同的内存格式时，才会传播内存格式，否则将使用 contiguous_format 作为默认值
        # 在 CUDA 设备上创建一个空的输出张量 out_cuda，并使用 contiguous_format 内存格式保证连续性
        out_cuda = torch.empty((0), device=device).contiguous(memory_format=torch.contiguous_format)
        # 由于 a_cuda 和 b_cuda 的内存格式不同，所以 res2_cuda 将使用默认的 contiguous_format 内存格式
        res2_cuda = torch.cat((a_cuda, b_cuda), out=out_cuda)

        # 在 CPU 上创建一个空的输出张量 out_cpu，并使用 contiguous_format 内存格式保证连续性
        out_cpu = torch.empty((0), device='cpu').contiguous(memory_format=torch.contiguous_format)
        # 由于 a_cpu 和 b_cpu 的内存格式不同，所以 res2_cpu 将使用默认的 contiguous_format 内存格式
        res2_cpu = torch.cat((a_cpu, b_cpu), out=out_cpu)

        # 断言 res2_cuda 和 res2_cpu 的内存格式是否符合预期
        self.assertTrue(res2_cuda.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(res2_cpu.is_contiguous(memory_format=torch.contiguous_format))

        # 在 CUDA 设备上创建一个空的输出张量 out_cuda，并使用 contiguous_format 内存格式保证连续性
        out_cuda = torch.empty((0), device=device).contiguous(memory_format=torch.contiguous_format)
        # 由于 a_cuda 和 c_cuda 的内存格式相同，所以 res3_cuda 将保持 channels_last 内存格式
        res3_cuda = torch.cat((a_cuda, c_cuda), out=out_cuda)

        # 断言 res3_cuda 的内存格式是否符合预期
        self.assertTrue(res3_cuda.is_contiguous(memory_format=torch.channels_last))

    # 用于测试在不同设备上进行 torch.cat 和 torch.stack 操作时的行为，仅限于 CUDA 设备
    @onlyCUDA
    def test_cat_stack_cross_devices(self, device):
        # 在 CUDA 设备上生成随机张量 cuda
        cuda = torch.randn((3, 3), device=device)
        # 在 CPU 上生成随机张量 cpu
        cpu = torch.randn((3, 3), device='cpu')

        # 对于 torch.stack 操作，期望所有张量在同一设备上，因此会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cuda, cpu))
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cpu, cuda))

    # TODO: reconcile with other cat tests
    # TODO: Compare with a NumPy reference instead of CPU
    # 定义一个只在 CUDA 环境下执行的测试函数，用于测试 torch.cat() 的功能
    @onlyCUDA
    def test_cat(self, device):
        SIZE = 10
        # 对于指定的维度范围进行迭代测试
        for dim in range(-3, 3):
            # 将负数维度转换为对应的非负索引
            pos_dim = dim if dim >= 0 else 3 + dim
            # 生成在指定设备上的随机张量，并进行维度转置操作
            x = torch.rand(13, SIZE, SIZE, device=device).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE, device=device).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE, device=device).transpose(0, pos_dim)

            # 对 x, y, z 张量沿指定维度 dim 进行拼接
            res1 = torch.cat((x, y, z), dim)
            # 使用断言验证拼接后张量的分段是否与预期一致
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        # 针对特定张量进行测试，验证 torch.split() 和 torch.chunk() 的拼接结果
        x = torch.randn(20, SIZE, SIZE, device=device)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        # 在指定设备上创建随机张量，并与另一个张量进行拼接，验证拼接后的形状
        y = torch.randn(1, SIZE, SIZE, device=device)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    # TODO: update this test to compare against NumPy instead of CPU
    # 定义一个在 CUDA 环境下根据指定数据类型执行的测试函数，用于测试 torch.tensor().round() 的功能
    @onlyCUDA
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_device_rounding(self, device, dtype):
        # 测试半精度舍入操作
        a = [-5.8, -3.5, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, 5.8]
        res = [-6., -4., -2., -2., 0., 0., 2., 2., 4., 6.]

        # 在指定设备上创建输入张量并进行舍入操作，与预期结果进行断言比较
        a_tensor = torch.tensor(a, device=device).round()
        res_tensor = torch.tensor(res, device='cpu')
        self.assertEqual(a_tensor, res_tensor)

    # Note: This test failed on XLA since its test cases are created by empty_strided which
    #       doesn't support overlapping sizes/strides in XLA impl
    # 根据指定条件跳过在 TorchDynamo 环境下执行的测试
    @skipIfTorchDynamo("TorchDynamo fails on this test for unknown reasons")
    @onlyNativeDeviceTypes
    def test_like_fn_stride_proparation_vs_tensoriterator_unary_op(self, device):
        # 测试类似函数与基于张量迭代器的一元操作符（exp）的行为，确保从类似函数返回的张量遵循与张量迭代器相同的步幅传播规则。
        # 这里的类似函数的输出步幅总是在 CPU 端计算，这里不需要在 GPU 上进行测试。

        def compare_helper_(like_fn, t):
            # 对张量 t 应用 exp 函数得到 te
            te = torch.exp(t)
            # 使用类似函数 like_fn 对张量 t 进行操作得到 tl
            tl = like_fn(t)
            # 断言 te 和 tl 的步幅相等
            self.assertEqual(te.stride(), tl.stride())
            # 断言 te 和 tl 的大小相等
            self.assertEqual(te.size(), tl.size())

        like_fns = [
            lambda t, **kwargs: torch.zeros_like(t, **kwargs),     # 返回与 t 相同大小的零张量
            lambda t, **kwargs: torch.ones_like(t, **kwargs),      # 返回与 t 相同大小的全一张量
            lambda t, **kwargs: torch.randint_like(t, 10, 100, **kwargs),   # 返回与 t 相同大小的随机整数张量（指定范围）
            lambda t, **kwargs: torch.randint_like(t, 100, **kwargs),      # 返回与 t 相同大小的随机整数张量（默认范围）
            lambda t, **kwargs: torch.randn_like(t, **kwargs),     # 返回与 t 相同大小的正态分布随机张量
            lambda t, **kwargs: torch.rand_like(t, **kwargs),      # 返回与 t 相同大小的均匀分布随机张量
            lambda t, **kwargs: torch.full_like(t, 7, **kwargs),    # 返回与 t 相同大小的常数张量（指定常数值）
            lambda t, **kwargs: torch.empty_like(t, **kwargs)]     # 返回与 t 相同大小的空张量

        # 不重叠的稠密张量、非重叠的切片张量、非重叠的间隔张量、非重叠的步幅为零的张量、重叠的一般张量、
        # 重叠的切片张量、重叠的间隔张量、重叠的步幅为零的张量、重叠的等步幅张量
        tset = (
            torch.randn(4, 3, 2, device=device),
            torch.randn(4, 3, 2, device=device)[:, :, ::2],
            torch.empty_strided((4, 3, 2), (10, 3, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 0, 3), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (4, 2, 1), device=device)[:, :, ::2].fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 1, 1, 2), (10, 0, 0, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 2, 3), (10, 3, 3), device=device).fill_(1.0))

        for like_fn in like_fns:
            for t in tset:
                for p in permutations(range(t.dim())):
                    tp = t.permute(p)
                    compare_helper_(like_fn, tp)
    # Helper function for splitting tensors along a specified dimension using both Torch and NumPy functions
    def _hvd_split_helper(self, torch_fn, np_fn, op_name, inputs, device, dtype, dim):
        # Error message for dimension requirements
        dimension_error_message = op_name + " requires a tensor with at least "
        # Error message for divisibility check
        divisibiliy_error_message = op_name + " attempted to split along dimension "

        # Iterate over each input tuple containing shape and argument(s)
        for shape, arg in inputs:
            # Calculate the direction based on the shape and dimension
            direction = dim - (len(shape) == 1 and dim == 1)
            # Calculate the boundary based on the dimension
            bound = dim + 2 * (dim == 0) + (dim == 2)
            # Determine if an error is expected based on shape and argument(s)
            error_expected = len(shape) < bound or (not isinstance(arg, list) and shape[direction] % arg != 0)

            # Create a Torch tensor with the specified shape, dtype, and device
            t = make_tensor(shape, dtype=dtype, device=device)
            # Convert the Torch tensor to a NumPy array
            t_np = t.cpu().numpy()

            # Check if an error is expected or not, and assert equality or raise exceptions accordingly
            if not error_expected:
                self.assertEqual(torch_fn(t, arg), np_fn(t_np, arg))
            else:
                self.assertRaises(RuntimeError, lambda: torch_fn(t, arg))
                self.assertRaises(ValueError, lambda: np_fn(t, arg))
                # Determine the expected error message based on the type of error
                expected_error_message = dimension_error_message if len(shape) < bound else divisibiliy_error_message
                # Assert that the raised RuntimeError matches the expected error message pattern
                self.assertRaisesRegex(RuntimeError, expected_error_message, lambda: torch_fn(t, arg))

    # Decorated test function for testing horizontal splitting (hsplit) across native device types and data types
    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_hsplit(self, device, dtype):
        # Define various inputs as tuples of shape and split argument(s)
        inputs = (
            ((), 3),
            ((), [2, 4, 6]),
            ((6,), 2),
            ((6,), 4),
            ((6,), [2, 5]),
            ((6,), [7, 9]),
            ((3, 8), 4),
            ((3, 8), 5),
            ((3, 8), [1, 5]),
            ((3, 8), [3, 8]),
            ((5, 5, 5), 2),
            ((5, 5, 5), [1, 4]),
            ((5, 0, 5), 3),
            ((5, 5, 0), [2, 6]),
        )
        # Invoke the helper function for horizontal splitting tests
        self._hvd_split_helper(torch.hsplit, np.hsplit, "torch.hsplit", inputs, device, dtype, 1)

    # Decorated test function for testing vertical splitting (vsplit) across native device types and data types
    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_vsplit(self, device, dtype):
        # Define various inputs as tuples of shape and split argument(s)
        inputs = (
            ((6,), 2),
            ((6,), 4),
            ((6, 5), 2),
            ((6, 5), 4),
            ((6, 5), [1, 2, 3]),
            ((6, 5), [1, 5, 9]),
            ((6, 5, 5), 2),
            ((6, 0, 5), 2),
            ((5, 0, 5), [1, 5]),
        )
        # Invoke the helper function for vertical splitting tests
        self._hvd_split_helper(torch.vsplit, np.vsplit, "torch.vsplit", inputs, device, dtype, 0)

    # Decorated test function for testing depthwise splitting (dsplit) across native device types and data types
    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_dsplit(self, device, dtype):
        # Define various inputs as tuples of shape and split argument(s)
        inputs = (
            ((6,), 4),
            ((6, 6), 3),
            ((5, 5, 6), 2),
            ((5, 5, 6), 4),
            ((5, 5, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
            ((5, 5, 0), 2),
            ((5, 0, 6), 4),
            ((5, 0, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
        )
        # Invoke the helper function for depthwise splitting tests
        self._hvd_split_helper(torch.dsplit, np.dsplit, "torch.dsplit", inputs, device, dtype, 2)
    # 定义一个测试特殊堆栈操作的方法，接受维度、至少维度、torch 函数、np 函数、设备和数据类型作为参数
    def _test_special_stacks(self, dim, at_least_dim, torch_fn, np_fn, device, dtype):
        # 测试非元组参数时的错误
        t = torch.randn(10)
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn(t)
        # 再次测试单个数组时的错误
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn(t)

        # 测试零维情况
        num_tensors = random.randint(1, 5)
        input_t = [torch.tensor(random.uniform(0, 10), device=device, dtype=dtype) for i in range(num_tensors)]
        actual = torch_fn(input_t)
        expected = np_fn([input.cpu().numpy() for input in input_t])
        self.assertEqual(actual, expected)

        # 循环测试从 1 到 4 维度的情况
        for ndims in range(1, 5):
            base_shape = list(_rand_shape(ndims, min_size=1, max_size=5))
            for i in range(ndims):
                shape = list(base_shape)
                num_tensors = random.randint(1, 5)
                torch_input = []
                # 仅在一个轴上的形状不同创建张量
                for param in range(num_tensors):
                    shape[i] = random.randint(1, 5)
                    torch_input.append(_generate_input(tuple(shape), dtype, device, with_extremal=False))

                # 检查输入张量的维度是否有效
                valid_dim = True
                for k in range(len(torch_input) - 1):
                    for tdim in range(ndims):
                        # 检查除拼接维度外的所有张量是否具有相同的形状
                        # 除非维度少于至少函数维度，因为原始拼接维度在应用至少后将移位，并且不再是拼接维度
                        if (ndims < at_least_dim or tdim != dim) and torch_input[k].size()[tdim] != torch_input[k + 1].size()[tdim]:
                            valid_dim = False

                # 对于 ndims 为 1 时的特殊情况，需要 hstack 的特殊处理，因为 hstack 在 ndims 为 1 时的工作方式不同
                if valid_dim or (torch_fn is torch.hstack and ndims == 1):
                    # 维度有效，与 numpy 进行比较测试
                    np_input = [input.cpu().numpy() for input in torch_input]
                    actual = torch_fn(torch_input)
                    expected = np_fn(np_input)
                    self.assertEqual(actual, expected)
                else:
                    # 维度无效，测试错误情况
                    with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match except in dimension"):
                        torch_fn(torch_input)
                    with self.assertRaises(ValueError):
                        np_input = [input.cpu().numpy() for input in torch_input]
                        np_fn(np_input)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_hstack_column_stack(self, device, dtype):
        # 定义操作符列表，包括torch.hstack和np.hstack，以及torch.column_stack和np.column_stack
        ops = ((torch.hstack, np.hstack), (torch.column_stack, np.column_stack))
        # 遍历每个操作符对
        for torch_op, np_op in ops:
            # 调用内部方法_test_special_stacks，测试特殊的堆叠操作
            self._test_special_stacks(1, 1, torch_op, np_op, device, dtype)

        # 测试torch.column_stack对1D和2D张量输入的组合
        one_dim_tensor = torch.arange(0, 10).to(dtype=dtype, device=device)
        two_dim_tensor = torch.arange(0, 100).to(dtype=dtype, device=device).reshape(10, 10)
        inputs = two_dim_tensor, one_dim_tensor, two_dim_tensor, one_dim_tensor
        # 使用torch.column_stack对输入进行堆叠
        torch_result = torch.column_stack(inputs)

        # 将输入转换为NumPy数组
        np_inputs = [input.cpu().numpy() for input in inputs]
        # 使用np.column_stack对NumPy输入进行堆叠
        np_result = np.column_stack(np_inputs)

        # 断言torch结果与NumPy结果相等
        self.assertEqual(np_result,
                         torch_result)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_vstack_row_stack(self, device, dtype):
        # 定义操作符列表，包括torch.vstack和np.vstack，以及torch.row_stack和np.vstack
        ops = ((torch.vstack, np.vstack), (torch.row_stack, np.vstack))
        # 遍历每个操作符对
        for torch_op, np_op in ops:
            # 调用内部方法_test_special_stacks，测试特殊的堆叠操作
            self._test_special_stacks(0, 2, torch_op, np_op, device, dtype)
            # 循环5次，测试1D大小为(N)和2D大小为(1, N)的张量的维度变化
            for i in range(5):
                n = random.randint(1, 10)
                # 生成大小为(n,)的输入A
                input_a = _generate_input((n,), dtype, device, with_extremal=False)
                # 生成大小为(1, n)的输入B
                input_b = _generate_input((1, n), dtype, device, with_extremal=False)
                # 将torch输入转换为列表
                torch_input = [input_a, input_b]
                # 将torch输入转换为NumPy数组
                np_input = [input.cpu().numpy() for input in torch_input]
                # 使用torch_op对torch_input进行堆叠
                actual = torch_op(torch_input)
                # 使用np_op对np_input进行堆叠，作为预期结果
                expected = np_op(np_input)
                # 断言实际结果与预期结果相等
                self.assertEqual(actual, expected)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    # 定义一个测试方法，用于测试 torch.dstack 的功能
    def test_dstack(self, device, dtype):
        # 调用 _test_special_stacks 方法，测试 torch.dstack 和 np.dstack 的特殊堆叠行为
        self._test_special_stacks(2, 3, torch.dstack, np.dstack, device, dtype)
        
        # 循环执行五次测试
        for i in range(5):
            # 生成一个随机整数 n，用作 tensor 尺寸的一部分
            n = random.randint(1, 10)
            # 生成一个尺寸为 (n,) 的一维 tensor 输入 input_a
            input_a = _generate_input((n,), dtype, device, with_extremal=False)
            # 生成一个尺寸为 (1, n) 的二维 tensor 输入 input_b
            input_b = _generate_input((1, n), dtype, device, with_extremal=False)
            # 生成一个尺寸为 (1, n, 1) 的三维 tensor 输入 input_c
            input_c = _generate_input((1, n, 1), dtype, device, with_extremal=False)
            
            # 将 torch tensor 组成列表
            torch_input = [input_a, input_b, input_c]
            # 将 torch tensor 转换成 numpy 数组，并组成列表
            np_input = [input.cpu().numpy() for input in torch_input]
            
            # 使用 torch.dstack 对 torch_input 进行堆叠
            actual = torch.dstack(torch_input)
            # 使用 np.dstack 对 np_input 进行堆叠
            expected = np.dstack(np_input)
            
            # 断言 actual 是否等于 expected
            self.assertEqual(actual, expected)

            # 生成两个随机整数 m 和 n，作为 tensor 尺寸的一部分
            m = random.randint(1, 10)
            n = random.randint(1, 10)
            # 生成尺寸为 (m, n) 的二维 tensor 输入 input_a
            input_a = _generate_input((m, n), dtype, device, with_extremal=False)
            # 生成尺寸为 (m, n, 1) 的三维 tensor 输入 input_b
            input_b = _generate_input((m, n, 1), dtype, device, with_extremal=False)
            
            # 将 torch tensor 组成列表
            torch_input = [input_a, input_b]
            # 将 torch tensor 转换成 numpy 数组，并组成列表
            np_input = [input.cpu().numpy() for input in torch_input]
            
            # 使用 torch.dstack 对 torch_input 进行堆叠
            actual = torch.dstack(torch_input)
            # 使用 np.dstack 对 np_input 进行堆叠
            expected = np.dstack(np_input)
            
            # 断言 actual 是否等于 expected
            self.assertEqual(actual, expected)

    # 用于跳过由于 TorchDynamo 导致的测试失败情况的装饰器
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    # 使用 torch.int32 和 torch.int64 两种数据类型进行测试的装饰器
    @dtypes(torch.int32, torch.int64)
    # 测试生成大范围 linspace 的方法
    def test_large_linspace(self, device, dtype):
        # 获取 dtype 的最小值作为 linspace 的起始点
        start = torch.iinfo(dtype).min
        # 获取 dtype 的最大值，并通过位运算清除低 12 位作为 linspace 的终点
        end = torch.iinfo(dtype).max & ~0xfff
        # 设置 linspace 的步数为 15
        steps = 15
        # 使用 torch.linspace 生成从 start 到 end 的等间距数列，返回 tensor x
        x = torch.linspace(start, end, steps, dtype=dtype, device=device)
        # 断言相邻两个元素之间的差值大于给定范围内的值
        self.assertGreater(x[1] - x[0], (end - start) / steps)

    # 使用 torch.float32 和 torch.float64 两种数据类型进行测试的装饰器
    @dtypes(torch.float32, torch.float64)
    # 测试将 double 解包时的方法
    def test_unpack_double(self, device, dtype):
        # 参考链接：https://github.com/pytorch/pytorch/issues/33111
        # 定义一组需要测试的数值
        vals = (2 ** 24 + 1, 2 ** 53 + 1,
                np.iinfo(np.int64).max, np.iinfo(np.uint64).max, np.iinfo(np.uint64).max + 1,
                -1e500, 1e500)
        
        # 遍历 vals 中的每一个数值
        for val in vals:
            # 使用给定的 dtype 和 device 创建 torch tensor t
            t = torch.tensor(val, dtype=dtype, device=device)
            # 将 val 转换成 torch_to_numpy_dtype_dict[dtype] 对应的 numpy 数组 a
            a = np.array(val, dtype=torch_to_numpy_dtype_dict[dtype])
            # 断言 tensor t 是否等于从 numpy 数组 a 转换而来的 tensor
            self.assertEqual(t, torch.from_numpy(a))

    # 测试辅助方法，用于检查 float 到 integer 转换时不会产生未定义行为的错误
    def _float_to_int_conversion_helper(self, vals, device, dtype, refs=None):
        # 如果未提供 refs，则根据 vals 生成 numpy 数组 a，并将其转换为 torch tensor 作为 refs
        if refs is None:
            a = np.array(vals, dtype=np.float32).astype(torch_to_numpy_dtype_dict[dtype])
            refs = torch.from_numpy(a)
        
        # 使用给定的 device 和 dtype 创建 torch tensor t，将 vals 转换为 torch.float 类型
        t = torch.tensor(vals, device=device, dtype=torch.float).to(dtype)
        # 断言生成的 tensor refs 是否等于 t 的 cpu() 版本
        self.assertEqual(refs, t.cpu())

    # 检查 float 到 integer 的转换是否会导致未定义行为的错误
    # 注意：在 C++ 中，如果从浮点值转换到整数 dtype 时，如果浮点值不在整数 dtype 的动态范围内，
    # 这可能（并且应该）导致未定义行为
    # 使用修饰器指定仅在原生设备类型上运行该测试
    @onlyNativeDeviceTypes
    # 使用 unittest 跳过 macOS 和 Jetson 上的测试，详细信息见指定的 GitHub 问题链接
    @unittest.skipIf(IS_MACOS or IS_JETSON, "Test is broken on MacOS and Jetson, \
        see https://github.com/pytorch/pytorch/issues/38752")
    # 使用 unittest 跳过在 PowerPC 上的测试，详细信息见指定的 GitHub 问题链接
    @unittest.skipIf(IS_PPC, "Test is broken on PowerPC, see https://github.com/pytorch/pytorch/issues/39671")
    # 使用 dtypes 定义测试使用的数据类型范围
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_finite(self, device, dtype):
        # 获取浮点数类型的最小值和最大值
        min = torch.finfo(torch.float).min
        max = torch.finfo(torch.float).max

        # 定义测试用例中的浮点数值
        vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2, max)
        refs = None
        # 根据设备类型和版本进行特定的值设定
        if self.device_type == 'cuda':
            if torch.version.hip:
                # 当运行在 HIP 环境下时，修改 vals 数组的值
                vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)
            else:
                # 当运行在 CUDA 环境下时，修改 vals 数组的值
                vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
        elif dtype == torch.uint8:
            # 当数据类型为 torch.uint8 时，修改 vals 数组的值
            vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
            # 设定 refs 数组作为对应的参考值
            # 注意：对于 numpy 中 -2.0 或 -1.5 到 uint8 的转换是未定义的
            # 参考详细信息请参见指定的 GitHub 问题链接
            refs = (0, 254, 255, 0, 0, 0, 1, 2)

        # 调用辅助函数来执行浮点数到整数的转换测试
        self._float_to_int_conversion_helper(vals, device, dtype, refs)

    # 注意：在大多数数据类型上，CUDA 将在此测试中失败，通常是显著的失败
    # 注意：torch.uint16, torch.uint32, torch.uint64 被排除在外，因为这些测试会出现不确定的失败，警告“在转换中遇到无效值”
    @onlyCPU
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_nonfinite(self, device, dtype):
        # 定义测试用例中的特殊浮点数值（负无穷、正无穷、NaN）
        vals = (float('-inf'), float('inf'), float('nan'))

        # 调用辅助函数来执行浮点数到整数的特殊情况转换测试
        self._float_to_int_conversion_helper(vals, device, dtype)

    # 使用修饰器指定仅在原生设备类型上运行该测试
    @onlyNativeDeviceTypes
    def test_complex_type_conversions(self, device):
        # 定义被测试的数据类型列表
        dtypes = [torch.float, torch.complex64, torch.complex128]
        # 遍历数据类型列表，进行从一个数据类型到另一个数据类型的转换测试
        for from_type in dtypes:
            for to_type in dtypes:
                # 创建一个符合指定数据类型的随机张量
                from_tensor = torch.randn(4, dtype=from_type, device=device)
                # 执行从一个数据类型到另一个数据类型的转换
                to_tensor = from_tensor.to(to_type)
                if from_type.is_complex and not to_type.is_complex:
                    # 如果源数据类型是复数，目标数据类型不是复数，则验证实部是否相等
                    self.assertEqual(torch.real(from_tensor), to_tensor, exact_dtype=False)
                elif not from_type.is_complex and to_type.is_complex:
                    # 如果源数据类型不是复数，目标数据类型是复数，则验证实部和虚部是否正确
                    self.assertEqual(from_tensor, torch.real(to_tensor), exact_dtype=False)
                    self.assertEqual(torch.zeros_like(torch.imag(to_tensor)), torch.imag(to_tensor), exact_dtype=False)
                else:
                    # 其他情况下，验证转换后的张量是否相等
                    self.assertEqual(from_tensor, to_tensor, exact_dtype=False)

    # 使用修饰器指定此测试为慢速测试，并仅在 CPU 上运行
    @slowTest
    @onlyCPU
    # 测试拼接两个大张量的功能
    def test_cat_big(self, device):
        # 定义两个大张量的大小
        SIZE1 = 6500
        SIZE2 = 4500
        # 创建一个空列表，用于存放张量
        concat_list = []
        # 向列表中添加一个由 1 组成的张量，大小为 SIZE1 x (1024 * 512)，数据类型为 uint8
        concat_list.append(torch.ones((SIZE1, 1024 * 512), dtype=torch.uint8, device=device))
        # 向列表中添加另一个由 1 组成的张量，大小为 SIZE2 x (1024 * 512)，数据类型为 uint8
        concat_list.append(torch.ones((SIZE2, 1024 * 512), dtype=torch.uint8, device=device))
        # 使用 torch.cat 进行张量拼接
        result = torch.cat(concat_list)
        # 断言拼接后的张量大小是否等于 SIZE1 + SIZE2
        self.assertEqual(result.size(0), SIZE1 + SIZE2)

    # 用不同的数据类型和维度进行张量拼接的测试
    @onlyCPU
    @dtypes(torch.half, torch.double, torch.int)
    def test_cat2(self, device, dtype):
        # 定义张量的大小
        SIZE = 10
        # 遍历不同的维度值
        for dim in range(-3, 3):
            # 将负数维度转换为正数维度
            pos_dim = dim if dim >= 0 else 3 + dim
            # 创建三个随机整数张量，并根据维度进行转置
            x = torch.randint(low=-100, high=100, size=(13, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            y = torch.randint(low=-100, high=100, size=(17, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            z = torch.randint(low=-100, high=100, size=(19, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)

            # 沿指定维度 dim 进行张量拼接
            res1 = torch.cat((x, y, z), dim)
            # 使用断言验证拼接后的张量切片是否等于原始张量 x，使用 atol 和 rtol 进行数值比较
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        # 创建一个随机整数张量
        x = torch.randint(low=-100, high=100, size=(20, SIZE, SIZE), device=device).to(dtype)
        # 使用 torch.split 拆分后再拼接，使用断言验证拼接后的张量是否等于原始张量 x
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        # 使用 torch.chunk 拆分后再拼接，使用断言验证拼接后的张量是否等于原始张量 x
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        # 创建一个随机整数张量 y 和之前的 x 进行拼接
        y = torch.randint(low=-100, high=100, size=(1, SIZE, SIZE), device=device).to(dtype)
        z = torch.cat([x, y])
        # 使用断言验证拼接后张量的大小是否为 (21, SIZE, SIZE)
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    # 测试创建零张量时，数据类型、布局和设备是否匹配
    # FIXME: 基于 OpInfo 的张量创建方法测试，验证所有的张量创建方法以及所有的数据类型和布局
    @dtypes(torch.bool, torch.uint8, torch.int16, torch.int64, torch.float16, torch.float32, torch.complex64)
    def test_zeros_dtype_layout_device_match(self, device, dtype):
        # 定义张量的布局为 strided
        layout = torch.strided
        # 创建一个零张量，指定设备和数据类型
        t = torch.zeros((2, 3), device=device, dtype=dtype, layout=layout)
        # 使用断言验证张量的数据类型是否正确
        self.assertIs(dtype, t.dtype)
        # 使用断言验证张量的布局是否正确
        self.assertIs(layout, t.layout)
        # 使用断言验证张量的设备是否正确
        self.assertEqual(torch.device(device), t.device)

    # TODO: 更新以在 CUDA 上运行
    @onlyCPU
    # 定义测试方法，用于测试 torch.stack 函数在不同数据类型下的行为
    def test_stack(self, device):
        # 遍历三种数据类型：半精度浮点数、双精度浮点数、整数
        for dtype in (torch.half, torch.double, torch.int):
            # 创建随机整数张量 x，y，z，形状为 (2, 3, 4)，并将其转换为指定数据类型
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            # 在维度范围内进行迭代，测试 torch.stack 函数
            for dim in range(4):
                # 使用 torch.stack 在指定维度上将 x、y、z 组合成一个新张量 res
                res = torch.stack((x, y, z), dim)
                # 使用负数索引测试 torch.stack 在负维度上的表现
                res_neg = torch.stack((x, y, z), dim - 4)
                # 预期的结果张量形状应为 x 在指定维度之前的形状 + (3,) + x 在指定维度之后的形状
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                # 使用断言检查 res 和 res_neg 是否相等
                self.assertEqual(res, res_neg)
                # 使用断言检查 res 的形状是否与预期相符
                self.assertEqual(res.size(), expected_size)
                # 使用断言检查 res 在指定维度上选择的第 0、1、2 个元素是否分别为 x、y、z
                self.assertEqual(res.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 2), z, atol=0, rtol=0)

    # TODO: update to work on CUDA, too
    # 使用 onlyCPU 装饰器修饰的测试方法，用于测试 torch.stack 函数在 CPU 上的行为
    @onlyCPU
    def test_stack_out(self, device):
        # 遍历三种数据类型：半精度浮点数、双精度浮点数、整数
        for dtype in (torch.half, torch.double, torch.int):
            # 创建随机整数张量 x，y，z，形状为 (2, 3, 4)，并将其转换为指定数据类型
            x = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4)).to(dtype)
            # 在维度范围内进行迭代，测试 torch.stack 函数
            for dim in range(4):
                # 预期的结果张量形状应为 x 在指定维度之前的形状 + (3,) + x 在指定维度之后的形状
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                # 创建与预期大小相同的新张量 res_out 和 res_neg_out
                res_out = x.new(expected_size)
                res_neg_out = x.new(expected_size)
                # 获取 res_out 和 res_neg_out 的数据指针
                res_out_dp = res_out.data_ptr()
                res_neg_out_dp = res_neg_out.data_ptr()
                # 使用 torch.stack 将 x、y、z 组合成一个新张量，并将结果存储在 res_out 中
                torch.stack((x, y, z), dim, out=res_out)
                # 使用负数索引测试 torch.stack 在负维度上的表现，并将结果存储在 res_neg_out 中
                torch.stack((x, y, z), dim - 4, out=res_neg_out)
                # 使用断言检查 res_out 和 res_neg_out 是否相等
                self.assertEqual(res_out, res_neg_out)
                # 使用断言检查 res_out 的形状是否与预期相符
                self.assertEqual(res_out.size(), expected_size)
                # 使用断言检查 res_out 的数据指针是否与之前获取的一致
                self.assertEqual(res_out_dp, res_out.data_ptr())
                # 使用断言检查 res_neg_out 的数据指针是否与之前获取的一致
                self.assertEqual(res_neg_out_dp, res_neg_out.data_ptr())
                # 使用断言检查 res_out 在指定维度上选择的第 0、1、2 个元素是否分别为 x、y、z
                self.assertEqual(res_out.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 2), z, atol=0, rtol=0)
    # 定义一个测试方法，用于测试 torch.repeat_interleave 函数的不同用例
    def test_repeat_interleave(self, device):
        # 创建一个张量 x，包含 [0, 1, 2, 3]，指定设备为 device
        x = torch.tensor([0, 1, 2, 3], device=device)
        # 预期的张量 expected，包含 [1, 2, 2, 3, 3, 3]，指定设备为 device
        expected = torch.tensor([1, 2, 2, 3, 3, 3], device=device)
        # 断言 torch.repeat_interleave(x) 的结果与 expected 相等
        self.assertEqual(torch.repeat_interleave(x), expected)

        # 测试异常情况：torch.repeat_interleave 不支持对形状为 (2, 2) 的张量进行操作时抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4, device=device).reshape(2, 2))

        # 测试异常情况：torch.repeat_interleave 不支持对浮点数张量进行操作时抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0, device=device))

        # 测试异常情况：torch.repeat_interleave 不支持包含负数的张量进行操作时抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4], device=device))

        # 创建一个二维张量 y，包含 [[1, 2], [3, 4]]，指定设备为 device
        y = torch.tensor([[1, 2], [3, 4]], device=device)

        # 测试 torch.repeat_interleave(y, 2) 的三种不同调用方式，验证结果是否与预期 y1_expect 相等
        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2, device=device))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2], device=device))
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        # 测试 torch.repeat_interleave(y, 3, dim=1)，验证结果是否与预期 y2_expect 相等
        y2 = torch.repeat_interleave(y, 3, dim=1)
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        self.assertEqual(y2, y2_expect)

        # 测试 torch.repeat_interleave(y, [1, 2], dim=0)，验证结果是否与预期 y3_expect 相等
        y3 = torch.repeat_interleave(y, torch.tensor([1, 2], device=device), dim=0)
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        self.assertEqual(y3, y3_expect)

        # 测试异常情况：torch.repeat_interleave(y, [1, 2, 3], dim=0) 时抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3], device=device), dim=0)

        # 测试异常情况：torch.repeat_interleave(y, torch.arange(9).reshape(3, 3), dim=0) 时抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9, device=device).reshape(3, 3), dim=0)

        # 测试在零大小维度上的情况
        # 创建一个零大小的张量 x，形状为 (5, 0)，指定设备为 device
        x = torch.zeros((5, 0), device=device)
        # 对 x 在维度 1 上重复 3 次，验证结果是否与预期的全零张量相等
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        # 创建一个空的整型张量 x，指定设备为 device
        x = torch.tensor([], dtype=torch.int64, device=device)
        # 对空张量 x 自身进行重复插值，验证结果是否与 x 相等
        y = torch.repeat_interleave(x, x)
        self.assertEqual(y, x)

    # TODO: udpate to work on CUDA, too
    @onlyCPU
    # 定义一个测试方法，用于验证新方法是否正确设置梯度要求
    def test_new_methods_requires_grad(self, device):
        # 定义测试用例，包括方法名称和参数
        size = (10,)
        test_cases = [
            # method name, args
            ('new_full', [size, 1]),
            ('new_empty', [size]),
            ('new_zeros', [size]),
            ('new_ones', [size]),
        ]
        # 遍历每个测试用例
        for method_name, args in test_cases:
            # 创建一个大小为 size 的随机张量 x
            x = torch.randn(size)
            # 对于每个 requires_grad 值，测试方法调用
            for requires_grad in [True, False]:
                # 调用指定名称的方法，传入参数 args 和 requires_grad
                x_new = x.__getattribute__(method_name)(*args, requires_grad=requires_grad)
                # 断言新张量的 requires_grad 属性是否符合预期值
                self.assertEqual(x_new.requires_grad, requires_grad)
            # 创建一个大小为 size 的整数张量 x
            x = torch.randint(10, size)
            # 使用断言捕获 RuntimeError 异常，验证特定条件下的方法调用
            with self.assertRaisesRegex(
                    RuntimeError,
                    r'Only Tensors of floating point and complex dtype can require gradients'):
                # 调用指定名称的方法，传入参数 args 和 requires_grad=True
                x_new = x.__getattribute__(method_name)(*args, requires_grad=True)

    # TODO: update to work on CUDA, too?
    # 定义一个测试方法，验证从序列创建张量的行为
    @onlyCPU
    def test_tensor_from_sequence(self, device):
        # 定义 MockSequence 类，模拟一个序列
        class MockSequence:
            def __init__(self, lst):
                self.lst = lst

            def __len__(self):
                return len(self.lst)

            def __getitem__(self, item):
                raise TypeError

        # 定义 GoodMockSequence 类，继承自 MockSequence，修复了 __getitem__ 方法
        class GoodMockSequence(MockSequence):
            def __getitem__(self, item):
                return self.lst[item]

        # 创建不良的 MockSequence 实例和良好的 GoodMockSequence 实例
        bad_mock_seq = MockSequence([1.0, 2.0, 3.0])
        good_mock_seq = GoodMockSequence([1.0, 2.0, 3.0])
        
        # 使用断言捕获 ValueError 异常，验证异常情况下的张量创建行为
        with self.assertRaisesRegex(ValueError, 'could not determine the shape'):
            torch.tensor(bad_mock_seq)
        
        # 使用断言验证从 GoodMockSequence 创建张量是否正常
        self.assertEqual(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(good_mock_seq))

    # TODO: update to work on CUDA, too?
    # 定义一个测试方法，验证简单标量类型转换的行为
    @onlyCPU
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_simple_scalar_cast(self, device):
        # 定义允许的张量列表和其对应的标量值列表
        ok = [torch.tensor([1.5]), torch.zeros(1, 1, 1, 1)]
        ok_values = [1.5, 0]

        # 定义不允许的张量序列
        not_ok = map(torch.Tensor, [[], [1, 2], [[1, 2], [3, 4]]])

        # 遍历每个允许的张量和其对应的标量值
        for tensor, value in zip(ok, ok_values):
            # 使用断言验证标量转换的正确性
            self.assertEqual(int(tensor), int(value))
            self.assertEqual(float(tensor), float(value))
            self.assertEqual(complex(tensor), complex(value))

        # 使用断言验证复数类型的转换
        self.assertEqual(complex(torch.tensor(1.5j)), 1.5j)

        # 遍历每个不允许的张量，使用断言捕获 ValueError 异常
        for tensor in not_ok:
            self.assertRaises(ValueError, lambda: int(tensor))
            self.assertRaises(ValueError, lambda: float(tensor))
            self.assertRaises(ValueError, lambda: complex(tensor))

        # 使用断言捕获 RuntimeError 异常，验证特定条件下的标量转换
        self.assertRaises(RuntimeError, lambda: float(torch.tensor(1.5j)))
        self.assertRaises(RuntimeError, lambda: int(torch.tensor(1.5j)))

    # TODO: update to work on CUDA, too?
    # 定义一个测试方法，验证张量切片后的标量转换行为
    @onlyCPU
    def test_offset_scalar_cast(self, device):
        # 创建张量 x 和其切片 y
        x = torch.tensor([1., 2., 3.])
        y = x[2:]
        # 使用断言验证切片后的标量转换
        self.assertEqual(int(y), 3)
    def test_meshgrid_empty(self):
        # 使用断言检查运行时错误，确保函数抛出指定异常并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError,
                                    'expects a non-empty TensorList'):
            # 调用 torch.meshgrid() 函数，期望抛出异常
            torch.meshgrid()

    def test_meshgrid_unsupported_indexing(self):
        # 使用断言检查运行时错误，确保函数抛出指定异常并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError,
                                    'indexing must be one of "xy" or "ij"'):
            # 调用 torch.meshgrid() 函数，并传入不支持的 indexing 参数
            torch.meshgrid(torch.tensor([1, 2]), indexing='')

    def test_meshgrid_non_1d_tensor(self):
        # 使用断言检查运行时错误，确保函数抛出指定异常并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError,
                                    'Expected 0D or 1D tensor'):
            # 调用 torch.meshgrid() 函数，并传入非一维张量
            torch.meshgrid(torch.tensor([[1, 2], [3, 4]]))

    def test_meshgrid_inconsistent_dtype(self):
        # 使用断言检查运行时错误，确保函数抛出指定异常并包含特定错误信息
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same dtype'):
            # 调用 torch.meshgrid() 函数，并传入类型不一致的张量
            torch.meshgrid(torch.tensor([1], dtype=torch.int),
                           torch.tensor([2], dtype=torch.float))

    def test_meshgrid_inconsistent_device(self):
        # 使用断言检查运行时错误，确保函数抛出指定异常并包含特定错误信息
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same device'):
            # 调用 torch.meshgrid() 函数，并传入设备不一致的张量
            torch.meshgrid(torch.tensor([1], device='cpu'),
                           torch.tensor([2], device='meta'))

    def test_meshgrid_warns_if_no_indexing(self):
        # 使用 assertWarnsOnceRegex 检查是否发出警告，确保警告信息符合预期
        with self.assertWarnsOnceRegex(
                UserWarning, '.*will be required to pass the indexing arg.*'):
            # 调用 torch.meshgrid() 函数，并传入参数触发警告
            torch.meshgrid(torch.tensor([1, 2]))

    def test_meshgrid_default_indexing(self, device):
        # 创建张量 a, b, c，使用 torch.meshgrid() 函数生成网格
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c])
        # 断言生成的网格形状与预期一致
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        # 断言生成的网格形状与预期一致
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        # 创建预期的张量网格
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        # 使用 assertTrue 检查生成的网格与预期是否相等
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))
    # 定义一个测试函数，用于测试 meshgrid 函数在不同索引方式下的行为
    def test_meshgrid_xy_indexing(self, device):
        # 创建一个包含单个整数 1 的张量，指定设备
        a = torch.tensor(1, device=device)
        # 创建一个包含整数 1, 2, 3 的张量，指定设备
        b = torch.tensor([1, 2, 3], device=device)
        # 创建一个包含整数 1, 2 的张量，指定设备
        c = torch.tensor([1, 2], device=device)
        
        # 使用 meshgrid 函数创建三个网格张量，采用 'xy' 索引方式
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c], indexing='xy')
        # 断言 grid_a 的形状为 [3, 1, 2]
        self.assertEqual(grid_a.shape, torch.Size([3, 1, 2]))
        # 断言 grid_b 的形状为 [3, 1, 2]
        self.assertEqual(grid_b.shape, torch.Size([3, 1, 2]))
        # 断言 grid_c 的形状为 [3, 1, 2]
        self.assertEqual(grid_c.shape, torch.Size([3, 1, 2]))
        
        # 再次使用 meshgrid 函数创建三个网格张量，采用 'xy' 索引方式
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c, indexing='xy')
        # 断言 grid_a2 的形状为 [3, 1, 2]
        self.assertEqual(grid_a2.shape, torch.Size([3, 1, 2]))
        # 断言 grid_b2 的形状为 [3, 1, 2]
        self.assertEqual(grid_b2.shape, torch.Size([3, 1, 2]))
        # 断言 grid_c2 的形状为 [3, 1, 2]
        self.assertEqual(grid_c2.shape, torch.Size([3, 1, 2]))
        
        # 创建预期的 grid_a 张量，元素全为 1，数据类型为 int64，指定设备
        expected_grid_a = torch.ones(3, 1, 2, dtype=torch.int64, device=device)
        # 创建预期的 grid_b 张量，包含特定值，指定设备
        expected_grid_b = torch.tensor([[[1, 1]],
                                        [[2, 2]],
                                        [[3, 3]]], device=device)
        # 创建预期的 grid_c 张量，包含特定值，指定设备
        expected_grid_c = torch.tensor([[[1, 2]],
                                        [[1, 2]],
                                        [[1, 2]]], device=device)
        
        # 断言 grid_a 与 expected_grid_a 张量相等
        self.assertTrue(grid_a.equal(expected_grid_a))
        # 断言 grid_b 与 expected_grid_b 张量相等
        self.assertTrue(grid_b.equal(expected_grid_b))
        # 断言 grid_c 与 expected_grid_c 张量相等
        self.assertTrue(grid_c.equal(expected_grid_c))
        
        # 断言 grid_a2 与 expected_grid_a 张量相等
        self.assertTrue(grid_a2.equal(expected_grid_a))
        # 断言 grid_b2 与 expected_grid_b 张量相等
        self.assertTrue(grid_b2.equal(expected_grid_b))
        # 断言 grid_c2 与 expected_grid_c 张量相等
        self.assertTrue(grid_c2.equal(expected_grid_c))

    # 定义一个测试函数，用于测试 meshgrid 函数在不同索引方式下的行为
    def test_meshgrid_ij_indexing(self, device):
        # 创建一个包含单个整数 1 的张量，指定设备
        a = torch.tensor(1, device=device)
        # 创建一个包含整数 1, 2, 3 的张量，指定设备
        b = torch.tensor([1, 2, 3], device=device)
        # 创建一个包含整数 1, 2 的张量，指定设备
        c = torch.tensor([1, 2], device=device)
        
        # 使用 meshgrid 函数创建三个网格张量，采用 'ij' 索引方式
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c], indexing='ij')
        # 断言 grid_a 的形状为 [1, 3, 2]
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        # 断言 grid_b 的形状为 [1, 3, 2]
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        # 断言 grid_c 的形状为 [1, 3, 2]
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        
        # 再次使用 meshgrid 函数创建三个网格张量，采用 'ij' 索引方式
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c, indexing='ij')
        # 断言 grid_a2 的形状为 [1, 3, 2]
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        # 断言 grid_b2 的形状为 [1, 3, 2]
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        # 断言 grid_c2 的形状为 [1, 3, 2]
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        
        # 创建预期的 grid_a 张量，元素全为 1，数据类型为 int64，指定设备
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        # 创建预期的 grid_b 张量，包含特定值，指定设备
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        # 创建预期的 grid_c 张量，包含特定值，指定设备
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        
        # 断言 grid_a 与 expected_grid_a 张量相等
        self.assertTrue(grid_a.equal(expected_grid_a))
        # 断言 grid_b 与 expected_grid_b 张量相等
        self.assertTrue(grid_b.equal(expected_grid_b))
        # 断言 grid_c 与 expected_grid_c 张量相等
        self.assertTrue(grid_c.equal(expected_grid_c))
        
        # 断言 grid_a2 与 expected_grid_a 张量相等
        self.assertTrue(grid_a2.equal(expected_grid_a))
        # 断言 grid_b2 与 expected_grid_b 张量相等
        self.assertTrue(grid_b2.equal(expected_grid_b))
        # 断言 grid_c2 与 expected_grid_c 张量相等
        self.assertTrue(grid_c2.equal(expected_grid_c))
    # 定义一个测试方法，用于验证默认情况下的 meshgrid 函数行为
    def test_meshgrid_ij_indexing_is_default(self, device):
        # 创建一个包含单个整数 1 的张量 a，指定设备为参数 device
        a = torch.tensor(1, device=device)
        # 创建一个包含三个整数 1, 2, 3 的张量 b，指定设备为参数 device
        b = torch.tensor([1, 2, 3], device=device)
        # 创建一个包含两个整数 1, 2 的张量 c，指定设备为参数 device
        c = torch.tensor([1, 2], device=device)
        # 使用 meshgrid 函数生成三个网格张量 grid_a, grid_b, grid_c，使用 'ij' 索引顺序
        grid_a, grid_b, grid_c = torch.meshgrid(a, b, c, indexing='ij')
        # 再次使用 meshgrid 函数生成三个网格张量 grid_a2, grid_b2, grid_c2，默认使用 'xy' 索引顺序
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        # 断言两次生成的 grid_a 相等
        self.assertTrue(grid_a.equal(grid_a2))
        # 断言两次生成的 grid_b 相等
        self.assertTrue(grid_b.equal(grid_b2))
        # 断言两次生成的 grid_c 相等
        self.assertTrue(grid_c.equal(grid_c2))

    # 跳过元数据的测试方法，测试 meshgrid 函数与 numpy 的对比
    @skipMeta
    def test_meshgrid_vs_numpy(self, device):
        # 定义不同形状的张量的测试用例
        cases = [
            [[]],
            [[1], [1], [1]],
            [[], [], []],
            [[3], [5], [7]],
            [[3], [], [7]],
            [[11], [13]],
            [[15]],
        ]

        # 定义索引模式的对应关系，包括 PyTorch 和 NumPy 的对应关系
        indexing_correspondence = [
            # PyTorch 中无索引对应于 NumPy 中的 'ij' 索引
            ({}, {'indexing': 'ij'}),

            # NumPy 中无索引对应于 PyTorch 中的 'xy' 索引
            ({'indexing': 'xy'}, {}),

            # 'ij' 和 'xy' 在两者中实现方式相同
            ({'indexing': 'ij'}, {'indexing': 'ij'}),
            ({'indexing': 'xy'}, {'indexing': 'xy'}),
        ]

        # 对于每个形状和索引模式的组合，执行测试
        for shapes, (torch_kwargs, numpy_kwargs) in product(cases, indexing_correspondence):
            # 使用子测试进行分组，显示当前形状、PyTorch 参数和 NumPy 参数
            with self.subTest(shapes=shapes, torch_kwargs=torch_kwargs, numpy_kwargs=numpy_kwargs):
                # 创建与 shapes 匹配的张量列表，并将其设备设置为参数 device，数据类型为 torch.int
                tensors = [make_tensor(shape, device=device, dtype=torch.int) for shape in shapes]
                # 使用 PyTorch 的 meshgrid 函数生成网格 torch_grids，根据传入的索引参数进行设置
                torch_grids = torch.meshgrid(*tensors, **torch_kwargs)
                # 使用 NumPy 的 meshgrid 函数生成网格 numpy_grids，将张量转换为 NumPy 数组并根据索引参数设置
                numpy_grids = np.meshgrid(*(tensor.cpu().numpy() for tensor in tensors), **numpy_kwargs)
                # 断言 PyTorch 生成的网格与 NumPy 生成的网格相等
                self.assertEqual(torch_grids, numpy_grids)
    # 测试函数：test_cartesian_prod，用于测试 torch.cartesian_prod 函数的各种输入情况
    def test_cartesian_prod(self, device):
        # 创建张量 a，包含一个元素，位于指定设备上
        a = torch.tensor([1], device=device)
        # 创建张量 b，包含多个元素，位于指定设备上
        b = torch.tensor([1, 2, 3], device=device)
        # 创建张量 c，包含多个元素，位于指定设备上
        c = torch.tensor([1, 2], device=device)
        # 使用 torch.cartesian_prod 生成 a, b, c 的笛卡尔积
        prod = torch.cartesian_prod(a, b, c)
        # 创建预期结果，通过 Python 的 itertools.product 生成
        expected = torch.tensor(list(product([a], b, c)), device=device)
        # 断言生成的笛卡尔积与预期结果相等
        self.assertEqual(expected, prod)

        # 测试空输入情况
        d = torch.empty(0, dtype=b.dtype, device=device)
        # 使用 torch.cartesian_prod 生成 a, b, c, d 的笛卡尔积
        prod = torch.cartesian_prod(a, b, c, d)
        # 创建预期结果为一个空张量
        expected = torch.empty(0, 4, dtype=b.dtype, device=device)
        # 断言生成的笛卡尔积与预期结果相等
        self.assertEqual(expected, prod)

        # 测试单个输入情况
        prod = torch.cartesian_prod(b)
        # 断言生成的笛卡尔积与输入张量 b 相等
        self.assertEqual(b, prod)

    # 测试函数：test_combinations，用于测试 torch.combinations 函数的各种输入情况
    def test_combinations(self, device):
        # 创建张量 a，包含多个元素，位于指定设备上
        a = torch.tensor([1, 2, 3], device=device)

        # 测试 r=0 的组合
        c = torch.combinations(a, r=0)
        # 创建预期结果为一个空张量
        expected = torch.empty(0, dtype=a.dtype, device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试 r=1 的组合
        c = torch.combinations(a, r=1)
        # 创建预期结果，通过 Python 的 itertools.combinations 生成
        expected = torch.tensor(list(combinations(a, r=1)), device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试带有替换的 r=1 的组合
        c = torch.combinations(a, r=1, with_replacement=True)
        # 创建预期结果，通过 Python 的 itertools.combinations_with_replacement 生成
        expected = torch.tensor(list(combinations_with_replacement(a, r=1)), device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试默认 r=2 的组合
        c = torch.combinations(a)
        # 创建预期结果，通过 Python 的 itertools.combinations 生成
        expected = torch.tensor(list(combinations(a, r=2)), device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试带有替换的 r=2 的组合
        c = torch.combinations(a, with_replacement=True)
        # 创建预期结果，通过 Python 的 itertools.combinations_with_replacement 生成
        expected = torch.tensor(list(combinations_with_replacement(a, r=2)), device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试 r=3 的组合
        c = torch.combinations(a, r=3)
        # 创建预期结果，通过 Python 的 itertools.combinations 生成
        expected = torch.tensor(list(combinations(a, r=3)), device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试 r=4 的组合
        c = torch.combinations(a, r=4)
        # 创建预期结果为一个空张量
        expected = torch.empty(0, 4, dtype=a.dtype, device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试 r=5 的组合
        c = torch.combinations(a, r=5)
        # 创建预期结果为一个空张量
        expected = torch.empty(0, 5, dtype=a.dtype, device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c, expected)

        # 测试空输入情况
        a = torch.empty(0, device=device)
        c1 = torch.combinations(a)
        c2 = torch.combinations(a, with_replacement=True)
        # 创建预期结果为一个空张量
        expected = torch.empty(0, 2, dtype=a.dtype, device=device)
        # 断言生成的组合结果与预期结果相等
        self.assertEqual(c1, expected)
        self.assertEqual(c2, expected)

    # 测试函数：test_linlogspace_mem_overlap，用于测试在不支持重叠内存情况下的 torch.linspace 和 torch.logspace 函数
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    @skipMeta
    def test_linlogspace_mem_overlap(self, device):
        # 创建张量 x，随机初始化并扩展为长度为 10，位于指定设备上
        x = torch.rand(1, device=device).expand(10)
        # 测试在不支持重叠内存情况下调用 torch.linspace 时是否抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.linspace(1, 10, 10, out=x)

        # 测试在不支持重叠内存情况下调用 torch.logspace 时是否抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.logspace(1, 10, 10, out=x)
    # 测试构造函数，使用 numpy 数组作为输入，检查不同数据类型的行为
    def test_ctor_with_numpy_array(self, device):
        # 正确的数据类型列表，包括双精度浮点数、普通浮点数、半精度浮点数、64位整数、32位整数、16位整数、8位整数、无符号8位整数、布尔值
        correct_dtypes = [
            np.double,
            float,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            bool,
        ]

        # 根据系统字节顺序创建错误的数据类型列表
        incorrect_byteorder = '>' if sys.byteorder == 'little' else '<'
        incorrect_dtypes = [incorrect_byteorder + t for t in ['d', 'f']]

        # 遍历所有正确的数据类型
        for dtype in correct_dtypes:
            # 创建 numpy 数组
            array = np.array([1, 2, 3, 4], dtype=dtype)

            # 转换为 Torch 张量并发送到指定设备，进行上溯转换测试
            tensor = torch.DoubleTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            # 下溯转换（有时）
            tensor = torch.FloatTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            tensor = torch.HalfTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    # 使用指定的数据类型参数进行随机数生成测试
    def test_random(self, device, dtype):
        # 此测试在概率 p <= (2/(ub-lb))^200 = 6e-36 时可能失败
        t = torch.empty(200, dtype=dtype, device=device)
        lb = 1
        ub = 4

        # 填充张量为 -1，并生成指定范围内的随机数
        t.fill_(-1)
        t.random_(lb, ub)
        self.assertEqual(t.min(), lb)
        self.assertEqual(t.max(), ub - 1)

        # 再次填充为 -1，并生成从 0 到 ub 范围内的随机数
        t.fill_(-1)
        t.random_(ub)
        self.assertEqual(t.min(), 0)
        self.assertEqual(t.max(), ub - 1)

    # 测试生成布尔类型随机数
    def test_random_bool(self, device):
        size = 2000
        t = torch.empty(size, dtype=torch.bool, device=device)

        # 填充张量为 False，并生成随机布尔数
        t.fill_(False)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        
        # 验证生成的 True 占比在 0.4 到 0.6 之间
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

        # 再次填充为 True，并生成随机布尔数
        t.fill_(True)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)

        # 验证生成的 True 占比在 0.4 到 0.6 之间
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

    # 由于问题，这个测试可能会在某些情况下失败，链接到 GitHub 问题页面
    @xfailIfTorchDynamo
    # 定义一个测试方法，用于测试在指定设备上生成随机布尔张量的行为
    def test_random_from_to_bool(self, device):
        # 设置随机张量的大小
        size = 2000

        # 获取 torch.int64 的最小值和最大值
        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        # 设置允许的最小和最大值范围
        min_val = 0
        max_val = 1

        # 定义测试用例中的 'from' 和 'to' 的取值范围
        froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
        tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]

        # 遍历所有 'from' 和 'to' 的组合
        for from_ in froms:
            for to_ in tos:
                # 创建一个空的 torch.bool 张量 t
                t = torch.empty(size, dtype=torch.bool, device=device)
                
                # 如果 to_ 大于 from_
                if to_ > from_:
                    # 检查 from_ 是否在 min_val 和 max_val 范围内，如果不是，引发异常
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    # 检查 to_ - 1 是否在 min_val 和 max_val 范围内，如果不是，引发异常
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        # 在指定范围内生成随机整数填充张量 t
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = 1
                        # 检查生成的随机整数范围是否符合预期
                        self.assertTrue(from_ <= t.to(torch.int).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.int).max() < to_)
                else:
                    # 如果 to_ 不大于 from_，引发异常
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    # 注意事项：uint64 是因为其最大值不能在 int64_t 中表示而出现问题，但这是 random 函数预期的行为
    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch.uint16, torch.uint32))
    # 定义一个测试函数，用于测试在给定设备和数据类型下，生成随机数的完整范围
    def test_random_full_range(self, device, dtype):
        # 设定生成随机数的数组大小
        size = 2000
        # 设定范围因子
        alpha = 0.1

        # 获取指定数据类型的最小和最大值
        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        # 根据数据类型选择浮点数的限制值
        if dtype == torch.double:
            fp_limit = 2**53
        elif dtype == torch.float:
            fp_limit = 2**24
        elif dtype == torch.half:
            fp_limit = 2**11
        elif dtype == torch.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        # 创建一个空的张量，指定数据类型和设备
        t = torch.empty(size, dtype=dtype, device=device)

        # 根据数据类型和平台限制，确定随机数的生成范围
        if dtype in [torch.float, torch.double, torch.half, torch.bfloat16]:
            from_ = int(max(-fp_limit, int64_min_val))
            to_inc_ = int(min(fp_limit, int64_max_val))
        else:
            from_ = int(max(torch.iinfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.iinfo(dtype).max, int64_max_val))
        range_ = to_inc_ - from_ + 1

        # 在指定范围内生成随机整数并填充张量 t
        t.random_(from_, None)
        
        # 计算最小值的预期范围
        delta = max(1, alpha * range_)
        
        # 断言：生成的最小值在指定范围内
        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
        # 断言：生成的最大值在指定范围内
        self.assertTrue((to_inc_ - delta) < t.to(torch.double).max() <= to_inc_)

    # NB: uint64 is broken because its max value is not representable in
    # int64_t, but this is what random expects
    # https://github.com/pytorch/pytorch/issues/126834
    # 标记：uint64 类型存在问题，因为其最大值无法在 int64_t 中表示，但是随机数生成函数期望的是这种行为
    # 参考：https://github.com/pytorch/pytorch/issues/126834
    @xfailIfTorchDynamo
    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch.uint16, torch.uint32))
    # https://github.com/pytorch/pytorch/issues/126834
    @xfailIfTorchDynamo
    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch.uint16, torch.uint32))
    # 定义一个测试函数，用于测试随机数生成范围
    def test_random_to(self, device, dtype):
        # 设置生成随机数的张量大小
        size = 2000
        # 设置alpha值，用于计算范围的一部分
        alpha = 0.1

        # 获取torch.int64类型的最小和最大值
        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        # 根据不同的dtype设置min_val和max_val以及生成测试范围tos
        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        # 初始化生成随机数的起始点from_
        from_ = 0
        # 遍历所有的to值进行测试
        for to_ in tos:
            # 创建一个指定dtype和device的空张量t
            t = torch.empty(size, dtype=dtype, device=device)
            # 如果to_大于from_
            if to_ > from_:
                # 检查to - 1是否在[min_val, max_val]范围内
                if not (min_val <= (to_ - 1) <= max_val):
                    # 如果不在范围内，断言出现RuntimeError，提示to - 1超出范围
                    self.assertRaisesRegex(
                        RuntimeError,
                        "to - 1 is out of bounds",
                        lambda: t.random_(from_, to_)
                    )
                else:
                    # 否则调用随机数生成函数random_，生成[from_, to_)范围内的随机数
                    t.random_(to_)
                    # 计算范围和delta值
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    # 根据dtype类型进行不同的断言检查
                    if dtype == torch.bfloat16:
                        # 对于bfloat16类型，由于舍入误差，检查宽松一些
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                    else:
                        # 其他类型的检查
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
            else:
                # 如果to_ <= from_，断言出现RuntimeError，提示from >= to的错误
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )
    # 测试默认随机数生成器行为的方法，接受设备和数据类型作为参数
    def test_random_default(self, device, dtype):
        # 定义生成张量的大小
        size = 2000
        # 定义用于控制生成随机数范围的参数
        alpha = 0.1

        # 根据数据类型设置增量值
        if dtype == torch.float:
            to_inc = 1 << 24
        elif dtype == torch.double:
            to_inc = 1 << 53
        elif dtype == torch.half:
            to_inc = 1 << 11
        elif dtype == torch.bfloat16:
            to_inc = 1 << 8
        else:
            to_inc = torch.iinfo(dtype).max

        # 使用指定的设备和数据类型创建一个空张量
        t = torch.empty(size, dtype=dtype, device=device)
        # 用随机数填充张量
        t.random_()
        # 断言：张量最小值应大于等于 0 且小于 alpha 乘以增量值
        self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        # 断言：张量最大值应大于 (1 - alpha) 乘以增量值 且小于等于增量值本身
        self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)

    # TODO: 需要更新此测试
    @onlyNativeDeviceTypes
    # 测试空张量和全张量的方法，接受设备作为参数
    def test_empty_full(self, device):
        # 转换设备为 Torch 设备对象
        torch_device = torch.device(device)
        # 获取设备类型
        device_type = torch_device.type

        # 获取所有数据类型列表，不包括半精度和 bfloat16
        dtypes = get_all_dtypes(include_half=False, include_bfloat16=False, include_complex32=True)

        # 根据设备类型选择测试方式
        if device_type == 'cpu':
            # 在 CPU 上执行测试空张量和全张量的方法
            do_test_empty_full(self, dtypes, torch.strided, torch_device)
        elif device_type == 'cuda':
            # 在 CUDA 上执行测试空张量和全张量的方法（两种方式）
            do_test_empty_full(self, dtypes, torch.strided, None)
            do_test_empty_full(self, dtypes, torch.strided, torch_device)

    # TODO: 需要更新此测试
    @suppress_warnings
    @onlyNativeDeviceTypes
    @deviceCountAtLeast(1)
    # 测试函数，用于验证张量在不同设备上的行为
    def test_tensor_device(self, devices):
        # 获取第一个设备的类型
        device_type = torch.device(devices[0]).type
        # 如果设备类型是 'cpu'
        if device_type == 'cpu':
            # 验证创建在 CPU 上的张量的设备类型为 'cpu'
            self.assertEqual('cpu', torch.tensor(5).device.type)
            # 验证在 CPU 上创建的指定设备为 'cpu' 的张量的设备类型为 'cpu'
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu').device.type)
            # 验证在 CPU 上创建的指定设备为 'cpu:0' 的张量的设备类型为 'cpu'
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu:0').device.type)
            # 验证在 CPU 上创建的指定设备为 'cpu:0' 的张量的设备类型为 'cpu'
            self.assertEqual('cpu',
                             torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cpu:0').device.type)
            # 验证在 CPU 上创建的指定设备为 'cpu' 的张量的设备类型为 'cpu'
            self.assertEqual('cpu', torch.tensor(np.random.randn(2, 3), device='cpu').device.type)
        # 如果设备类型是 'cuda'
        if device_type == 'cuda':
            # 验证在 CUDA 设备上创建的指定设备为 'cuda:0' 的张量的设备类型为 'cuda:0'
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda(0).device))
            # 验证在 CUDA 设备上创建的指定设备为 'cuda:0' 的张量的设备类型为 'cuda:0'
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda('cuda:0').device))
            # 验证在 CUDA 设备上创建的指定设备为 'cuda:0' 的张量的设备类型为 'cuda:0'
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device=0).device))
            # 验证在 CUDA 设备上创建的指定设备为 'cuda:0' 的张量的设备类型为 'cuda:0'
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device='cuda:0').device))
            # 验证在 CUDA 设备上创建的指定设备为 'cuda:0' 的张量的设备类型为 'cuda:0'
            self.assertEqual('cuda:0',
                             str(torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cuda:0').device))

            # 验证在 CUDA 设备上创建的指定设备为 'cuda:0' 的张量的设备类型为 'cuda:0'
            self.assertEqual('cuda:0', str(torch.tensor(np.random.randn(2, 3), device='cuda:0').device))

            # 遍历设备列表中的每个设备
            for device in devices:
                # 在当前 CUDA 设备上执行张量操作
                with torch.cuda.device(device):
                    # 获取当前 CUDA 设备的字符串表示
                    device_string = 'cuda:' + str(torch.cuda.current_device())
                    # 验证在指定设备为 'cuda' 的情况下创建的张量的设备类型
                    self.assertEqual(device_string,
                                     str(torch.tensor(5, dtype=torch.int64, device='cuda').device))

            # 验证尝试在 'cpu' 上使用 CUDA 方法会引发 RuntimeError
            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu')
            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu:0')

            # 如果设备列表中有多个设备
            if len(devices) > 1:
                # 验证在 CUDA 设备 1 上创建的张量的设备类型为 'cuda:1'
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda(1).device))
                # 验证在 CUDA 设备 1 上创建的张量的设备类型为 'cuda:1'
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda('cuda:1').device))
                # 验证在 CUDA 设备 1 上创建的张量的设备类型为 'cuda:1'
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device=1).device))
                # 验证在 CUDA 设备 1 上创建的张量的设备类型为 'cuda:1'
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device='cuda:1').device))
                # 验证在 CUDA 设备 1 上创建的指定设备为 'cuda:1' 的张量的设备类型为 'cuda:1'
                self.assertEqual('cuda:1',
                                 str(torch.tensor(torch.ones((2, 3), dtype=torch.float32),
                                     device='cuda:1').device))

                # 验证在 CUDA 设备 1 上创建的指定设备为 'cuda:1' 的张量的设备类型为 'cuda:1'
                self.assertEqual('cuda:1',
                                 str(torch.tensor(np.random.randn(2, 3), device='cuda:1').device))

    # TODO: this test should be updated
    @onlyNativeDeviceTypes
    # 负步长不支持的测试函数
    def test_as_strided_neg(self, device):
        # 定义错误信息正则表达式
        error = r'as_strided: Negative strides are not supported at the ' \
                r'moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]'
        # 断言运行时异常，并检查异常信息是否匹配预期错误消息
        with self.assertRaisesRegex(RuntimeError, error):
            # 调用 torch.as_strided 函数，传入包含负步长的参数，触发异常
            torch.as_strided(torch.ones(3, 3, device=device), (1, 1), (2, -1))
        with self.assertRaisesRegex(RuntimeError, error):
            # 再次调用 torch.as_strided 函数，传入负步长的参数，触发异常
            torch.as_strided(torch.ones(14, device=device), (2,), (-11,))

    # TODO: this test should be updated
    def test_zeros(self, device):
        # 创建全零张量，设备为指定设备
        res1 = torch.zeros(100, 100, device=device)
        # 创建空张量，设备为指定设备
        res2 = torch.tensor((), device=device)
        # 在 res2 上执行 torch.zeros 操作，结果传给 res2
        torch.zeros(100, 100, device=device, out=res2)

        # 断言两个张量相等
        self.assertEqual(res1, res2)

        # 创建布尔型全零张量，设备为指定设备
        boolTensor = torch.zeros(2, 2, device=device, dtype=torch.bool)
        # 创建期望的布尔型张量
        expected = torch.tensor([[False, False], [False, False]],
                                device=device, dtype=torch.bool)
        # 断言两个张量相等
        self.assertEqual(boolTensor, expected)

        # 创建半精度浮点型全零张量，设备为指定设备
        halfTensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        # 创建期望的半精度浮点型张量
        expected = torch.tensor([[0.]], device=device, dtype=torch.float16)
        # 断言两个张量相等
        self.assertEqual(halfTensor, expected)

        # 创建 bfloat16 类型全零张量，设备为指定设备
        bfloat16Tensor = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
        # 创建期望的 bfloat16 类型张量
        expected = torch.tensor([[0.]], device=device, dtype=torch.bfloat16)
        # 断言两个张量相等
        self.assertEqual(bfloat16Tensor, expected)

        # 创建复数类型全零张量，设备为指定设备
        complexTensor = torch.zeros(2, 2, device=device, dtype=torch.complex64)
        # 创建期望的复数类型张量
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex64)
        # 断言两个张量相等
        self.assertEqual(complexTensor, expected)

        # 创建半精度复数类型全零张量，设备为指定设备
        complexHalfTensor = torch.zeros(2, 2, device=device, dtype=torch.complex32)
        # 创建期望的半精度复数类型张量
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex32)
        # 断言两个张量相等
        self.assertEqual(complexHalfTensor, expected)

    # TODO: this test should be updated
    def test_zeros_out(self, device):
        # 定义张量形状
        shape = (3, 4)
        # 创建形状为 shape 的全零张量，设备为指定设备
        out = torch.zeros(shape, device=device)
        # 在指定设备上创建形状为 shape 的全零张量，将结果传给 out
        torch.zeros(shape, device=device, out=out)

        # 改变 dtype、布局、设备，预期抛出运行时异常
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, dtype=torch.int64, out=out)
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, layout=torch.sparse_coo, out=out)

        # 保持 dtype 不变，断言两个张量相等
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, dtype=out.dtype, out=out))
        # 保持布局不变，断言两个张量相等
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, layout=torch.strided, out=out))
        # 保持张量不变，断言两个张量相等
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, out=out))
    # 定义一个测试方法，用于测试在指定设备上创建全为1的张量，并进行断言比较
    def test_ones(self, device):
        # 创建一个在指定设备上全为1的张量
        res1 = torch.ones(100, 100, device=device)
        # 创建一个空张量，在指定设备上
        res2 = torch.tensor((), device=device)
        # 使用torch.ones创建全为1的张量，并将结果写入已有的res2张量中
        torch.ones(100, 100, device=device, out=res2)
        # 断言res1和res2的值相等
        self.assertEqual(res1, res2)

        # 测试布尔类型张量
        res1 = torch.ones(1, 2, device=device, dtype=torch.bool)
        # 创建一个预期的布尔类型张量，与res1进行比较
        expected = torch.tensor([[True, True]], device=device, dtype=torch.bool)
        self.assertEqual(res1, expected)

        # 测试chalf类型（复数的半精度浮点数）
        self.assertEqual(torch.ones(100, 100, device=device, dtype=torch.chalf),
                         torch.ones(100, 100, device=device, dtype=torch.cfloat), exact_dtype=False)

    # TODO: this test should be updated
    # 使用onlyCPU装饰器定义的测试方法，用于测试构造函数的数据类型设置
    @onlyCPU
    def test_constructor_dtypes(self, device):
        # 断言空张量的数据类型与默认数据类型相同
        self.assertIs(torch.tensor([]).dtype, torch.get_default_dtype())

        # 断言torch.uint8与torch.ByteTensor.dtype相同
        self.assertIs(torch.uint8, torch.ByteTensor.dtype)
        # 断言torch.float32与torch.FloatTensor.dtype相同
        self.assertIs(torch.float32, torch.FloatTensor.dtype)
        # 断言torch.float64与torch.DoubleTensor.dtype相同
        self.assertIs(torch.float64, torch.DoubleTensor.dtype)

        # 使用set_default_tensor_type上下文管理器设置默认的张量类型为torch.FloatTensor
        with set_default_tensor_type('torch.FloatTensor'):
            # 断言torch.float32与当前的默认数据类型相同
            self.assertIs(torch.float32, torch.get_default_dtype())
            # 断言torch.FloatStorage与torch.Storage相同
            self.assertIs(torch.FloatStorage, torch.Storage)

        # 只有浮点类型可以作为默认类型
        # 尝试将默认张量类型设置为非浮点类型，预期抛出TypeError异常
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type('torch.IntTensor'))

        # 使用set_default_dtype上下文管理器设置默认的数据类型为torch.float64
        with set_default_dtype(torch.float64):
            # 断言torch.float64与当前的默认数据类型相同
            self.assertIs(torch.float64, torch.get_default_dtype())
            # 断言torch.DoubleStorage与torch.Storage相同
            self.assertIs(torch.DoubleStorage, torch.Storage)

        # 使用set_default_tensor_type上下文管理器设置默认的张量类型为torch.FloatTensor
        with set_default_tensor_type(torch.FloatTensor):
            # 断言torch.float32与当前的默认数据类型相同
            self.assertIs(torch.float32, torch.get_default_dtype())
            # 断言torch.FloatStorage与torch.Storage相同
            self.assertIs(torch.FloatStorage, torch.Storage)

        # 如果CUDA可用，使用set_default_tensor_type上下文管理器设置默认的张量类型为torch.cuda.FloatTensor
        if torch.cuda.is_available():
            with set_default_tensor_type(torch.cuda.FloatTensor):
                # 断言torch.float32与当前的默认数据类型相同
                self.assertIs(torch.float32, torch.get_default_dtype())
                # 断言torch.float32与torch.cuda.FloatTensor.dtype相同
                self.assertIs(torch.float32, torch.cuda.FloatTensor.dtype)
                # 断言torch.cuda.FloatStorage与torch.Storage相同
                self.assertIs(torch.cuda.FloatStorage, torch.Storage)

                # 使用set_default_dtype上下文管理器设置默认的数据类型为torch.float64
                with set_default_dtype(torch.float64):
                    # 断言torch.float64与当前的默认数据类型相同
                    self.assertIs(torch.float64, torch.get_default_dtype())
                    # 断言torch.cuda.DoubleStorage与torch.Storage相同
                    self.assertIs(torch.cuda.DoubleStorage, torch.Storage)

        # 不允许向set_default_tensor_type传递数据类型参数
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type(torch.float32))

        # 不允许向set_default_dtype传递数据类型参数
        for t in all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.qint8):
            # 只有浮点类型可以作为默认类型
            if t in (
                    torch.half,
                    torch.float,
                    torch.double,
                    torch.bfloat16):
                with set_default_dtype(t):
                    pass
            else:
                self.assertRaises(TypeError, lambda: torch.set_default_dtype(t))
    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_device_legacy(self, device):
        # 检查在旧版构造函数中使用设备参数时是否引发了 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor((2.0, 3.0), device='cuda'))

        self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cuda'))

        # Tensor 构造函数和 new 方法在使用 Tensor 参数时指定设备时应该抛出 RuntimeError 异常
        i = torch.tensor([1], device='cpu')
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))

        x = torch.randn((3,), device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cuda'))

        # 如果 CUDA 可用，检查在 CUDA 设备上的类似情况是否引发了 RuntimeError 异常
        if torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor((2.0, 3.0), device='cpu'))

            # Tensor 构造函数和 new 方法在使用 Tensor 参数时指定设备时应该抛出 RuntimeError 异常
            i = torch.tensor([1], device='cuda')
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))

            # 使用 set_default_tensor_type 设置默认为 CUDA 的 FloatTensor 后，检查类似的异常情况
            with set_default_tensor_type(torch.cuda.FloatTensor):
                self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cpu'))
                self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cpu'))
                self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cpu'))
            x = torch.randn((3,), device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cpu'))
    # 定义一个测试方法，用于测试张量的创建和比较
    def test_tensor_factory(self, device):
        # TODO: This test probably doesn't make too much sense now that
        # torch.tensor has been established for a while; it makes more
        # sense to test the legacy behavior in terms of the new behavior

        # 期望的张量对象
        expected = torch.Tensor([1, 1])

        # 测试数据：创建一个张量对象
        res1 = torch.tensor([1, 1])
        self.assertEqual(res1, expected, exact_dtype=False)

        # 创建具有指定数据类型的张量对象
        res1 = torch.tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # 测试复制功能：通过复制期望的张量对象创建新的张量对象
        res2 = torch.tensor(expected)
        self.assertEqual(res2, expected)

        # 修改新创建的张量对象，并验证原始期望张量对象没有被修改
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))

        # 以指定数据类型复制张量对象
        res2 = torch.tensor(expected, dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # 使用numpy进行复制测试
        for dtype in [np.float64, np.int64, np.int8, np.uint8]:
            a = np.array([5.]).astype(dtype)
            res1 = torch.tensor(a)
            self.assertEqual(5., res1[0].item())
            a[0] = 7.
            self.assertEqual(5., res1[0].item())

        # 测试布尔类型张量
        a = torch.tensor([True, True, False, True, True], dtype=torch.bool)
        b = torch.tensor([-1, -1.1, 0, 1, 1.1], dtype=torch.bool)
        self.assertEqual(a, b)
        c = torch.tensor([-0.1, -1.1, 0, 1, 0.1], dtype=torch.bool)
        self.assertEqual(a, c)
        d = torch.tensor((-.3, 0, .3, 1, 3 / 7), dtype=torch.bool)
        e = torch.tensor((True, False, True, True, True), dtype=torch.bool)
        self.assertEqual(e, d)
        f = torch.tensor((-1, 0, -1.1, 1, 1.1), dtype=torch.bool)
        self.assertEqual(e, f)

        # 测试特殊值的布尔类型张量
        int64_max = torch.iinfo(torch.int64).max
        int64_min = torch.iinfo(torch.int64).min
        float64_max = torch.finfo(torch.float64).max
        float64_min = torch.finfo(torch.float64).min

        g_1 = torch.tensor((float('nan'), 0, int64_min, int64_max, int64_min - 1), dtype=torch.bool)
        self.assertEqual(e, g_1)

        g_2 = torch.tensor((int64_max + 1, 0, (int64_max + 1) * 2, (int64_max + 1) * 2 + 1, float64_min), dtype=torch.bool)
        self.assertEqual(e, g_2)

        g_3 = torch.tensor((float64_max, 0, float64_max + 1, float64_min - 1, float64_max + 1e291), dtype=torch.bool)
        self.assertEqual(e, g_3)

        h = torch.tensor([True, False, False, True, False, True, True], dtype=torch.bool)
        i = torch.tensor([1e-323, 1e-324, 0j, 1e-323j, 1e-324j, 1 + 2j, -1j], dtype=torch.bool)
        self.assertEqual(h, i)

        j = torch.tensor((True, True, True, True), dtype=torch.bool)
        k = torch.tensor((1e323, -1e323, float('inf'), -float('inf')), dtype=torch.bool)
        self.assertEqual(j, k)
    # 定义一个测试函数，用于验证张量工厂函数的复制行为，接受一个设备参数
    def test_tensor_factory_copy_var(self, device):
        # 定义内部函数，用于检查复制后的张量属性
        def check_copy(copy, is_leaf, requires_grad, data_ptr=None):
            # 如果未提供数据指针，则默认使用复制对象的数据指针
            if data_ptr is None:
                data_ptr = copy.data_ptr
            # 断言复制后的张量与原始张量在数值上相等，不要求精确的数据类型匹配
            self.assertEqual(copy, source, exact_dtype=False)
            # 断言复制后的张量的 is_leaf 属性与预期值相符
            self.assertTrue(copy.is_leaf == is_leaf)
            # 断言复制后的张量的 requires_grad 属性与预期值相符
            self.assertTrue(copy.requires_grad == requires_grad)
            # 断言复制后的张量的数据指针与预期值相符
            self.assertTrue(copy.data_ptr == data_ptr)

        # 创建一个随机张量作为源张量，数据类型为双精度浮点型，允许计算梯度
        source = torch.randn(5, 5, dtype=torch.double, requires_grad=True)

        # 测试 torch.tensor() 函数的复制行为
        check_copy(torch.tensor(source), True, False)
        check_copy(torch.tensor(source, requires_grad=False), True, False)
        check_copy(torch.tensor(source, requires_grad=True), True, True)

        # 测试 tensor.new_tensor() 函数的复制行为
        copy = torch.randn(1)
        check_copy(copy.new_tensor(source), True, False)
        check_copy(copy.new_tensor(source, requires_grad=False), True, False)
        check_copy(copy.new_tensor(source, requires_grad=True), True, True)

        # 测试 torch.as_tensor() 函数的复制行为
        # 对于 torch.as_tensor()，不会进行复制，而是返回原始张量的视图
        check_copy(torch.as_tensor(source), source.is_leaf, source.requires_grad, source.data_ptr)  # not copy
        check_copy(torch.as_tensor(source, dtype=torch.float), False, True)  # copy and keep the graph

    # TODO: this test should be updated
    @onlyCPU
    # 定义测试函数，用于检查张量工厂的类型推断，接受设备参数
    def test_tensor_factory_type_inference(self, device):
        # 定义内部函数，用于执行推断测试
        def test_inference(default_dtype):
            # 根据默认数据类型选择复数类型，若默认是 float32 则选择 complex64，否则选择 complex128
            default_complex_dtype = torch.complex64 if default_dtype == torch.float32 else torch.complex128
            # 断言默认数据类型与空张量的数据类型相同
            self.assertIs(default_dtype, torch.tensor(()).dtype)
            # 断言默认数据类型与包含单个浮点数的张量的数据类型相同
            self.assertIs(default_dtype, torch.tensor(5.).dtype)
            # 断言 int64 数据类型与包含单个整数的张量的数据类型相同
            self.assertIs(torch.int64, torch.tensor(5).dtype)
            # 断言 bool 数据类型与包含单个布尔值的张量的数据类型相同
            self.assertIs(torch.bool, torch.tensor(True).dtype)
            # 断言 int32 数据类型与包含单个整数的张量的数据类型相同（指定 dtype）
            self.assertIs(torch.int32, torch.tensor(5, dtype=torch.int32).dtype)
            # 断言默认数据类型与包含元组和数字的张量的数据类型相同
            self.assertIs(default_dtype, torch.tensor(((7, 5), (9, 5.))).dtype)
            # 断言默认数据类型与包含元组和混合数据类型的张量的数据类型相同
            self.assertIs(default_dtype, torch.tensor(((5., 5), (3, 5))).dtype)
            # 断言 int64 数据类型与包含元组和整数的张量的数据类型相同
            self.assertIs(torch.int64, torch.tensor(((5, 3), (3, 5))).dtype)
            # 断言默认复数数据类型与包含元组和复数的张量的数据类型相同
            self.assertIs(default_complex_dtype, torch.tensor(((5, 3 + 2j), (3, 5 + 4j))).dtype)

            # 断言 float64 数据类型与空 numpy 数组的张量的数据类型相同
            self.assertIs(torch.float64, torch.tensor(np.array(())).dtype)
            # 断言 float64 数据类型与包含单个 numpy 浮点数的张量的数据类型相同
            self.assertIs(torch.float64, torch.tensor(np.array(5.)).dtype)
            # 根据 numpy 数组的数据类型断言 int64 或 int32 数据类型与包含单个 numpy 数组元素的张量的数据类型相同
            if np.array(5).dtype == np.int64:  # 在 Windows 上可能是 4 字节的 np long
                self.assertIs(torch.int64, torch.tensor(np.array(5)).dtype)
            else:
                self.assertIs(torch.int32, torch.tensor(np.array(5)).dtype)
            # 断言 uint8 数据类型与包含单个 numpy 无符号整数的张量的数据类型相同
            self.assertIs(torch.uint8, torch.tensor(np.array(3, dtype=np.uint8)).dtype)
            # 断言默认数据类型与包含元组和 numpy 数组的张量的数据类型相同
            self.assertIs(default_dtype, torch.tensor(((7, np.array(5)), (np.array(9), 5.))).dtype)
            # 断言 float64 数据类型与包含元组和 numpy 数组的张量的数据类型相同
            self.assertIs(torch.float64, torch.tensor(((7, 5), (9, np.array(5.)))).dtype)
            # 断言 int64 数据类型与包含元组和 numpy 数组的张量的数据类型相同
            self.assertIs(torch.int64, torch.tensor(((5, np.array(3)), (np.array(3), 5))).dtype)

        # 遍历不同的数据类型，执行推断测试
        for dtype in [torch.float64, torch.float32]:
            with set_default_dtype(dtype):
                test_inference(dtype)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    # 定义一个测试方法，用于测试在指定设备上创建新的张量
    def test_new_tensor(self, device):
        # 创建一个预期的 Torch 变量，包含 ByteTensor 类型的数据 [1, 1]
        expected = torch.autograd.Variable(torch.ByteTensor([1, 1]))
        # 使用 new_tensor 方法创建一个新的张量 res1，数据为 [1, 1]
        res1 = expected.new_tensor([1, 1])
        # 断言新创建的张量 res1 与预期的 expected 相等
        self.assertEqual(res1, expected)
        # 使用 new_tensor 方法创建一个新的张量 res1，数据为 [1, 1]，指定数据类型为 torch.int
        res1 = expected.new_tensor([1, 1], dtype=torch.int)
        # 断言新创建的张量 res1 与预期的 expected 相等，不要求精确的数据类型匹配
        self.assertEqual(res1, expected, exact_dtype=False)
        # 断言 res1 的数据类型为 torch.int
        self.assertIs(torch.int, res1.dtype)

        # 测试复制操作
        # 使用 new_tensor 方法创建一个新的张量 res2，数据与 expected 相同
        res2 = expected.new_tensor(expected)
        # 断言新创建的张量 res2 与预期的 expected 相等
        self.assertEqual(res2, expected)
        # 修改 res2 的第二个元素为 2
        res2[1] = 2
        # 断言 expected 的值为全为 1 的张量
        self.assertEqual(expected, torch.ones_like(expected))
        # 使用 new_tensor 方法创建一个新的张量 res2，数据与 expected 相同，指定数据类型为 torch.int
        res2 = expected.new_tensor(expected, dtype=torch.int)
        # 断言新创建的张量 res2 与预期的 expected 相等，不要求精确的数据类型匹配
        self.assertEqual(res2, expected, exact_dtype=False)
        # 断言 res2 的数据类型为 torch.int
        self.assertIs(torch.int, res2.dtype)

        # 测试与 numpy 数组的复制
        # 创建一个 numpy 数组 a，包含一个浮点数 5.0
        a = np.array([5.])
        # 使用 torch.tensor 将 numpy 数组 a 转换为 Torch 张量 res1
        res1 = torch.tensor(a)
        # 使用 new_tensor 方法创建一个新的张量 res1，数据与 numpy 数组 a 相同
        res1 = res1.new_tensor(a)
        # 断言 res1 的第一个元素为 5.0
        self.assertEqual(5., res1[0].item())
        # 修改 numpy 数组 a 的第一个元素为 7.0
        a[0] = 7.
        # 断言 res1 的第一个元素仍为 5.0（new_tensor 方法复制时不会随着原始数据的变化而变化）

        # 如果存在至少两个 CUDA 设备
        if torch.cuda.device_count() >= 2:
            # 将 expected 移动到 CUDA 设备 1 上
            expected = expected.cuda(1)
            # 使用 new_tensor 方法创建一个新的张量 res1，数据为 [1, 1]
            res1 = expected.new_tensor([1, 1])
            # 断言 res1 的设备编号与 expected 的设备编号相同
            self.assertEqual(res1.get_device(), expected.get_device())
            # 使用 new_tensor 方法创建一个新的张量 res1，数据为 [1, 1]，指定数据类型为 torch.int
            res1 = expected.new_tensor([1, 1], dtype=torch.int)
            # 断言 res1 的数据类型为 torch.int
            self.assertIs(torch.int, res1.dtype)
            # 断言 res1 的设备编号与 expected 的设备编号相同
            self.assertEqual(res1.get_device(), expected.get_device())

            # 使用 new_tensor 方法创建一个新的张量 res2，数据与 expected 相同
            res2 = expected.new_tensor(expected)
            # 断言 res2 的设备编号与 expected 的设备编号相同
            self.assertEqual(res2.get_device(), expected.get_device())
            # 使用 new_tensor 方法创建一个新的张量 res2，数据与 expected 相同，指定数据类型为 torch.int
            res2 = expected.new_tensor(expected, dtype=torch.int)
            # 断言 res1 的数据类型为 torch.int
            self.assertIs(torch.int, res1.dtype)
            # 断言 res2 的设备编号与 expected 的设备编号相同
            self.assertEqual(res2.get_device(), expected.get_device())
            # 使用 new_tensor 方法创建一个新的张量 res2，数据与 expected 相同，指定数据类型为 torch.int，设备编号为 0
            res2 = expected.new_tensor(expected, dtype=torch.int, device=0)
            # 断言 res1 的数据类型为 torch.int
            self.assertIs(torch.int, res1.dtype)
            # 断言 res2 的设备编号为 0
            self.assertEqual(res2.get_device(), 0)

            # 使用 new_tensor 方法创建一个新的张量 res1，数据为 1
            res1 = expected.new_tensor(1)
            # 断言 res1 的设备编号与 expected 的设备编号相同
            self.assertEqual(res1.get_device(), expected.get_device())
            # 使用 new_tensor 方法创建一个新的张量 res1，数据为 1，指定数据类型为 torch.int
            res1 = expected.new_tensor(1, dtype=torch.int)
            # 断言 res1 的数据类型为 torch.int
            self.assertIs(torch.int, res1.dtype)
            # 断言 res1 的设备编号与 expected 的设备编号相同
            self.assertEqual(res1.get_device(), expected.get_device())
    # 定义一个测试方法，测试 torch.as_tensor 方法在不同情况下的行为，使用给定的设备进行测试

    # 从普通 Python 数据创建张量
    x = [[0, 1], [2, 3]]
    # 测试 torch.tensor 和 torch.as_tensor 方法返回的张量是否相等
    self.assertEqual(torch.tensor(x), torch.as_tensor(x))
    # 测试指定了数据类型后是否仍然相等
    self.assertEqual(torch.tensor(x, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32))

    # 包含不同类型数据的 Python 列表
    z = [0, 'torch']
    # 测试在包含非同一类型数据时，torch.tensor 和 torch.as_tensor 是否引发 TypeError 异常
    with self.assertRaisesRegex(TypeError, "invalid data type"):
        torch.tensor(z)
        torch.as_tensor(z)

    # 包含自引用列表的 Python 数据
    z = [0]
    z += [z]
    # 测试包含自引用列表时，torch.tensor 和 torch.as_tensor 是否引发 TypeError 异常
    with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
        torch.tensor(z)
        torch.as_tensor(z)

    # 包含自引用列表的嵌套列表
    z = [[1, 2], z]
    # 测试包含自引用列表时，torch.tensor 和 torch.as_tensor 是否引发 TypeError 异常
    with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
        torch.tensor(z)
        torch.as_tensor(z)

    # 从张量创建张量（除非数据类型不同，否则不复制）
    y = torch.tensor(x)
    # 测试不同情况下 torch.as_tensor 是否返回相同的张量对象
    self.assertIs(y, torch.as_tensor(y))
    # 测试指定不同数据类型时是否复制张量对象
    self.assertIsNot(y, torch.as_tensor(y, dtype=torch.float32))
    if torch.cuda.is_available():
        # 测试不同设备上的张量是否返回相同的张量对象
        self.assertIsNot(y, torch.as_tensor(y, device='cuda'))
        # 测试在 CUDA 设备上进行转换是否返回相同的张量对象
        y_cuda = y.to('cuda')
        self.assertIs(y_cuda, torch.as_tensor(y_cuda))
        self.assertIs(y_cuda, torch.as_tensor(y_cuda, device='cuda'))

    # 不复制的情况
    for dtype in [np.float64, np.int64, np.int8, np.uint8]:
        n = np.random.rand(5, 6).astype(dtype)
        n_astensor = torch.as_tensor(n)
        # 测试从 NumPy 数组到张量的转换是否正确
        self.assertEqual(torch.tensor(n), n_astensor)
        n_astensor[0][0] = 25.7
        # 测试修改张量是否影响原始的 NumPy 数组
        self.assertEqual(torch.tensor(n), n_astensor)

    # 修改数据类型导致复制
    n = np.random.rand(5, 6).astype(np.float32)
    n_astensor = torch.as_tensor(n, dtype=torch.float64)
    # 测试修改数据类型是否导致复制
    self.assertEqual(torch.tensor(n, dtype=torch.float64), n_astensor)
    n_astensor[0][1] = 250.8
    # 测试修改后的张量与原始张量是否相等
    self.assertNotEqual(torch.tensor(n, dtype=torch.float64), n_astensor)

    # 修改设备导致复制
    if torch.cuda.is_available():
        n = np.random.randn(5, 6)
        n_astensor = torch.as_tensor(n, device='cuda')
        # 测试从 NumPy 数组到 CUDA 设备上的张量的转换是否正确
        self.assertEqual(torch.tensor(n, device='cuda'), n_astensor)
        n_astensor[0][2] = 250.9
        # 测试修改后的张量与原始张量是否相等
        self.assertNotEqual(torch.tensor(n, device='cuda'), n_astensor)
    # 定义测试方法，测试 torch.range 函数的不同用法和参数组合
    def test_range(self, device, dtype):
        # 使用 torch.range 函数生成从 0 到 1 的张量 res1
        res1 = torch.range(0, 1, device=device, dtype=dtype)
        # 创建一个空张量 res2
        res2 = torch.tensor((), device=device, dtype=dtype)
        # 将 torch.range 的结果存储到 res2 中
        torch.range(0, 1, device=device, dtype=dtype, out=res2)
        # 断言 res1 和 res2 相等，允许绝对误差和相对误差为 0
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # 检查非连续张量的 range 函数
        # 创建一个大小为 (2, 3) 的零张量 x
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        # 将 torch.range 的结果存储到 x 的子张量中
        torch.range(0, 3, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        # 创建预期结果张量 res2
        res2 = torch.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=dtype)
        # 断言 x 和 res2 相等，允许绝对误差为 1e-16，相对误差为 0
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # 检查负数步长的 range 函数
        # 创建预期结果张量 res1
        res1 = torch.tensor((1, 0), device=device, dtype=dtype)
        # 创建一个空张量 res2
        res2 = torch.tensor((), device=device, dtype=dtype)
        # 将 torch.range 的结果存储到 res2 中
        torch.range(1, 0, -1, device=device, dtype=dtype, out=res2)
        # 断言 res1 和 res2 相等，允许绝对误差和相对误差为 0
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # 检查上下界相等的 range 函数
        # 创建预期结果张量 res1
        res1 = torch.ones(1, device=device, dtype=dtype)
        # 创建一个空张量 res2
        res2 = torch.tensor((), device=device, dtype=dtype)
        # 将 torch.range 的结果存储到 res2 中
        torch.range(1, 1, -1, device=device, dtype=dtype, out=res2)
        # 断言 res1 和 res2 相等，允许绝对误差和相对误差为 0
        self.assertEqual(res1, res2, atol=0, rtol=0)
        # 再次使用相同的上下界和步长，将 torch.range 的结果存储到 res2 中
        torch.range(1, 1, 1, device=device, dtype=dtype, out=res2)
        # 断言 res1 和 res2 相等，允许绝对误差和相对误差为 0
        self.assertEqual(res1, res2, atol=0, rtol=0)
    # 定义测试函数，用于验证 torch.arange 方法的推断行为
    def test_arange_inference(self, device):
        # 测试仅指定结束参数时的情况
        self.assertIs(torch.float32, torch.arange(1.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1.)).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1)).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1, dtype=torch.int16)).dtype)

        # 测试指定起始、结束参数的情况，以及可选的步长参数
        self.assertIs(torch.float32, torch.arange(1., 3).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64), 3).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1, dtype=torch.int16), torch.tensor(3.)).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3, 1.).dtype)
        self.assertIs(torch.float32,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3, dtype=torch.int16),
                                   torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1, 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), torch.tensor(3, dtype=torch.int16)).dtype)
        self.assertIs(torch.int64, torch.arange(1, 3, 1).dtype)
        self.assertIs(torch.int64,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3),
                                   torch.tensor(1, dtype=torch.int16)).dtype)

    # 标记为跳过 meta tensor 的测试用例
    @skipMeta
    def test_empty_strided(self, device):
        # 遍历不同形状的空 strided 张量和步幅组合
        for shape in [(2, 3, 4), (0, 2, 0)]:
            # 一些案例比较奇怪，验证 as_strided 允许的情况下，empty_strided 也可以
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                # 创建一个指定形状和步幅的空 strided 张量
                empty_strided = torch.empty_strided(shape, strides, device=device)
                # 使用 empty_strided 设置存储大小，创建一个与 as_strided 类似的张量
                as_strided = torch.empty(empty_strided.storage().size(),
                                         device=device).as_strided(shape, strides)
                # 验证 empty_strided 和 as_strided 张量的形状和步幅相等
                self.assertEqual(empty_strided.shape, as_strided.shape)
                self.assertEqual(empty_strided.stride(), as_strided.stride())
    # 测试新建空的分步张量
    def test_new_empty_strided(self, device):
        # 定义内部测试函数，接收大小、步幅和数据类型作为参数
        def _test(sizes, strides, dtype):
            # 创建指定大小和数据类型的全零张量
            x = torch.zeros(5, 5, dtype=dtype, device=device)
            # 使用 new_empty_strided 方法创建新的空分步张量
            result = x.new_empty_strided(sizes, strides)
            # 创建预期的空分步张量
            expected = torch.empty_strided(sizes, strides, dtype=x.dtype, device=x.device)
            # 断言结果张量的形状与预期相同
            self.assertEqual(result.shape, expected.shape)
            # 断言结果张量的步幅与预期相同
            self.assertEqual(result.stride(), expected.stride())
            # 断言结果张量的数据类型与预期相同
            self.assertEqual(result.dtype, expected.dtype)
            # 断言结果张量的设备与预期相同
            self.assertEqual(result.device, expected.device)

        # 测试不同的大小、步幅和数据类型组合
        _test([2, 3], [3, 1], torch.float)
        _test([5, 3], [0, 1], torch.int)
        _test([], [], torch.float)

        # 一些异常的测试案例
        for shape in [(2, 3, 4), (0, 2, 0)]:
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                _test(shape, strides, torch.float)

        # 确保大小和步幅长度相同
        # https://github.com/pytorch/pytorch/issues/82416
        with self.assertRaisesRegex(
                RuntimeError,
                r"dimensionality of sizes \(1\) must match dimensionality of strides \(0\)"):
            # 创建一个标量张量，并尝试使用不匹配的大小和步幅
            dtype = torch.float64
            x = torch.tensor(-4.8270, dtype=dtype, device=device)
            size = (2,)
            stride = ()
            x.new_empty_strided(size, stride, dtype=dtype, device=device)

    # 测试不匹配的步幅和形状的分步张量
    def test_strided_mismatched_stride_shape(self, device):
        # 对于每个形状和步幅不匹配的情况，确保引发运行时错误
        for shape, strides in [((1, ), ()), ((1, 2), (1, ))]:
            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                # 使用 as_strided 方法创建张量并检查异常
                torch.tensor(0.42, device=device).as_strided(shape, strides)

            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                # 使用 as_strided_ 方法创建张量并检查异常
                torch.tensor(0.42, device=device).as_strided_(shape, strides)

    # 测试空张量的属性
    def test_empty_tensor_props(self, device):
        # 不同大小的空张量的测试案例
        sizes = [(0,), (0, 3), (5, 0), (5, 0, 3, 0, 2), (0, 3, 0, 2), (0, 5, 0, 2, 0)]
        for size in sizes:
            # 创建指定大小的空张量
            x = torch.empty(tuple(size), device=device)
            # 断言张量的形状与预期相同
            self.assertEqual(size, x.shape)
            # 断言张量是连续的
            self.assertTrue(x.is_contiguous())
            # 将大小中的零替换为一，以创建新的张量
            size_ones_instead_of_zeros = (x if x != 0 else 1 for x in size)
            y = torch.empty(tuple(size_ones_instead_of_zeros), device=device)
            # 断言张量的步幅与新张量的步幅相同
            self.assertEqual(x.stride(), y.stride())

    @onlyNativeDeviceTypes
    # 定义一个测试方法，用于测试在不同情况下是否会引发 RuntimeError 异常
    def test_empty_overflow(self, device):
        # 测试创建超出内存大小限制的张量是否会引发 RuntimeError，并检查异常消息
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            torch.empty([2, 4, 2**29, 2**29], dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            torch.empty([8, 8, 2**29, 2**29], dtype=torch.float64)
        # 测试创建超出步长计算限制的张量是否会引发 RuntimeError，并检查异常消息
        with self.assertRaisesRegex(RuntimeError, 'Stride calculation overflowed'):
            torch.empty_strided([8, 8], [2**61, 1], dtype=torch.float64)
        # 测试创建指定大小的张量是否会引发 RuntimeError，并检查异常消息
        with self.assertRaisesRegex(RuntimeError, 'Storage size calculation overflowed'):
            torch.empty([0, 4, 2305843009213693952], dtype=torch.float32)

    # 定义一个测试方法，用于测试 torch.eye 方法的各种情况
    def test_eye(self, device):
        # 遍历所有数据类型，包括复杂数据类型，但不包括 torch.bfloat16
        for dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if dtype == torch.bfloat16:
                continue
            # 当 n 或 m 为负数时，测试是否会引发 RuntimeError，并检查异常消息
            for n, m in ((-1, 1), (1, -1), (-1, -1)):
                with self.assertRaisesRegex(RuntimeError, 'must be greater or equal to'):
                    torch.eye(n, m, device=device, dtype=dtype)

            # 当不提供 m 参数时，测试 torch.eye 方法的输出是否正确
            for n in (3, 5, 7):
                res1 = torch.eye(n, device=device, dtype=dtype)
                naive_eye = torch.zeros(n, n, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # 检查使用 eye_out 输出的结果是否与直接调用 torch.eye 一致
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch.eye(n, out=res2)
                self.assertEqual(res1, res2)

            # 当 n 和 m 取不同值时，测试 torch.eye 方法的输出是否正确
            for n, m in product([3, 5, 7], repeat=2):
                res1 = torch.eye(n, m, device=device, dtype=dtype)
                naive_eye = torch.zeros(n, m, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # 检查使用 eye_out 输出的结果是否与直接调用 torch.eye 一致
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch.eye(n, m, out=res2)
                self.assertEqual(res1, res2)
    # 测试 torch.linspace 与 np.linspace 的比较，针对不同的设备和数据类型
    def test_linspace_vs_numpy(self, device, dtype):
        # 设置起始值，若数据类型为复数则添加虚部
        start = -0.0316082797944545745849609375 + (0.8888888888j if dtype.is_complex else 0)
        # 设置结束值，若数据类型为复数则添加虚部
        end = .0315315723419189453125 + (0.444444444444j if dtype.is_complex else 0)

        # 遍历不同的步数进行测试
        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            # 使用 torch.linspace 生成张量 t
            t = torch.linspace(start, end, steps, device=device, dtype=dtype)
            # 使用 np.linspace 生成数组 a，并转换为对应的 numpy 数据类型
            a = np.linspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            # 将张量 t 转移到 CPU
            t = t.cpu()
            # 断言张量 t 等于从 numpy 数组 a 转换得到的 torch 张量
            self.assertEqual(t, torch.from_numpy(a))
            # 断言张量 t 的第一个元素与数组 a 的第一个元素相等
            self.assertTrue(t[0].item() == a[0])
            # 断言张量 t 的最后一个元素与数组 a 的最后一个元素相等
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    # 针对整数数据类型的测试，比较 torch.linspace 与 np.linspace 的结果
    @dtypes(*integral_types())
    def test_linspace_vs_numpy_integral(self, device, dtype):
        # 设置起始值和结束值
        start = 1
        end = 127

        # 遍历不同的步数进行测试
        for steps in [25, 50]:
            # 使用 torch.linspace 生成张量 t
            t = torch.linspace(start, end, steps, device=device, dtype=dtype)
            # 使用 np.linspace 生成数组 a，并转换为对应的 numpy 数据类型
            a = np.linspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            # 将张量 t 转移到 CPU
            t = t.cpu()
            # 断言张量 t 等于从 numpy 数组 a 转换得到的 torch 张量
            self.assertEqual(t, torch.from_numpy(a))
            # 断言张量 t 的第一个元素与数组 a 的第一个元素相等
            self.assertTrue(t[0].item() == a[0])
            # 断言张量 t 的最后一个元素与数组 a 的最后一个元素相等
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    # 辅助函数，用于复杂数数据类型下测试 linspace 与 logspace
    def _test_linspace_logspace_complex_helper(self, torch_fn, np_fn, device, dtype):
        # 随机生成起始值和结束值
        start = torch.randn(1, dtype=dtype).item()
        end = (start + torch.randn(1, dtype=dtype) + random.randint(5, 15)).item()

        # 辅助函数，用于测试指定函数 torch_fn 和 np_fn 在给定步数下的结果
        def test_fn(torch_fn, numpy_fn, steps):
            # 使用 torch_fn 生成张量 t
            t = torch_fn(start, end, steps, device=device)
            # 使用 numpy_fn 生成数组 a，并转换为对应的 numpy 数据类型
            a = numpy_fn(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            # 将张量 t 转移到 CPU
            t = t.cpu()
            # 断言张量 t 等于从 numpy 数组 a 转换得到的 torch 张量
            self.assertEqual(t, torch.from_numpy(a))

        # 遍历不同的步数进行测试
        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            # 调用辅助函数 test_fn 进行测试
            test_fn(torch_fn, np_fn, steps)

    # 针对复杂数数据类型的测试，比较 torch.linspace 与 np.linspace 的结果
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    @dtypes(torch.complex64)
    def test_linspace_vs_numpy_complex(self, device, dtype):
        # 调用辅助函数 _test_linspace_logspace_complex_helper 进行测试
        self._test_linspace_logspace_complex_helper(torch.linspace, np.linspace,
                                                    device, dtype)

    # 针对复杂数数据类型的测试，比较 torch.logspace 与 np.logspace 的结果
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    @dtypes(torch.complex64)
    def test_logspace_vs_numpy_complex(self, device, dtype):
        # 调用辅助函数 _test_linspace_logspace_complex_helper 进行测试
        self._test_linspace_logspace_complex_helper(torch.logspace, np.logspace,
                                                    device, dtype)

    # 重载精度设置，用于浮点数数据类型的测试
    @precisionOverride({torch.float: 1e-6, torch.double: 1e-10})
    @dtypes(*floating_types())
    # 测试 torch.logspace 与 numpy.logspace 的比较
    def test_logspace_vs_numpy(self, device, dtype):
        # 设置起始和结束值
        start = -0.0316082797944545745849609375
        end = .0315315723419189453125

        # 针对不同的步数进行测试
        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            # 使用 torch.logspace 生成对数间隔的张量 t
            t = torch.logspace(start, end, steps, device=device, dtype=dtype)
            # 使用 numpy.logspace 生成对数间隔的数组 a
            a = np.logspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            # 将张量 t 转移到 CPU 上
            t = t.cpu()
            # 断言 torch 生成的张量 t 与 numpy 生成的数组 a 相等
            self.assertEqual(t, torch.from_numpy(a))
            # 断言张量 t 的第一个元素与数组 a 的第一个元素相等
            self.assertEqual(t[0], a[0])
            # 断言张量 t 的最后一个元素与数组 a 的最后一个元素相等
            self.assertEqual(t[steps - 1], a[steps - 1])

    # 标注仅适用于 CUDA 设备的测试函数，并且要求使用大型张量
    @onlyCUDA
    @largeTensorTest('16GB')
    def test_range_factories_64bit_indexing(self, device):
        # 设置一个大整数
        bigint = 2 ** 31 + 1
        # 使用 torch.arange 生成从 0 到 bigint-1 的长整型张量 t
        t = torch.arange(bigint, dtype=torch.long, device=device)
        # 断言张量 t 的最后一个元素的值为 bigint-1
        self.assertEqual(t[-1].item(), bigint - 1)
        del t

        # 使用 torch.linspace 生成从 0 到 1 之间的均匀间隔的浮点数张量 t
        t = torch.linspace(0, 1, bigint, dtype=torch.float, device=device)
        # 断言张量 t 的最后一个元素的值为 1
        self.assertEqual(t[-1].item(), 1)
        del t

        # 使用 torch.logspace 生成从 0 到 1 之间的对数间隔的浮点数张量 t
        t = torch.logspace(0, 1, bigint, 2, dtype=torch.float, device=device)
        # 断言张量 t 的最后一个元素的值为 2
        self.assertEqual(t[-1].item(), 2)
        del t

    # 标注预期失败的测试函数，检查张量构造中的设备推断
    @expectedFailureMeta  # RuntimeError: The tensor has a non-zero number of elements
    @onlyNativeDeviceTypes
    def test_tensor_ctor_device_inference(self, device):
        # 将 device 转换为 torch.device 类型
        torch_device = torch.device(device)
        # 在指定设备上创建张量 values
        values = torch.tensor((1, 2, 3), device=device)

        # 测试 tensor 和 as_tensor 函数
        # 注意：忽略警告信息
        for op in (torch.tensor, torch.as_tensor):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 断言 op(values) 的设备与 torch_device 相同
                self.assertEqual(op(values).device, torch_device)
                # 断言 op(values, dtype=torch.float64) 的设备与 torch_device 相同
                self.assertEqual(op(values, dtype=torch.float64).device, torch_device)

                if self.device_type == 'cuda':
                    with torch.cuda.device(device):
                        # 断言 op(values.cpu()) 的设备为 CPU
                        self.assertEqual(op(values.cpu()).device, torch.device('cpu'))

        # 测试稀疏张量的构造
        indices = torch.tensor([[0, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0]], device=device)
        sparse_size = (3, 3, 3)

        # 创建默认数据类型的稀疏张量 sparse_default
        sparse_default = torch.sparse_coo_tensor(indices, values, sparse_size)
        # 断言稀疏张量 sparse_default 的设备与 torch_device 相同
        self.assertEqual(sparse_default.device, torch_device)

        # 创建指定数据类型的稀疏张量 sparse_with_dtype
        sparse_with_dtype = torch.sparse_coo_tensor(indices, values, sparse_size, dtype=torch.float64)
        # 断言稀疏张量 sparse_with_dtype 的设备与 torch_device 相同
        self.assertEqual(sparse_with_dtype.device, torch_device)

        if self.device_type == 'cuda':
            with torch.cuda.device(device):
                # 将 CPU 上的张量转移到 CUDA 上，并检查设备是否正确
                sparse_with_dtype = torch.sparse_coo_tensor(indices.cpu(), values.cpu(),
                                                            sparse_size, dtype=torch.float64)
                self.assertEqual(sparse_with_dtype.device, torch.device('cpu'))
    # 定义测试函数，用于测试信号窗口函数
    def _test_signal_window_functions(self, name, dtype, device, **kwargs):
        # 导入 scipy 的信号处理模块
        import scipy.signal as signal

        # 获取 torch 模块中对应窗口函数的方法
        torch_method = getattr(torch, name + '_window')
        
        # 如果数据类型不是浮点数，断言会抛出 RuntimeError 异常
        if not dtype.is_floating_point:
            with self.assertRaisesRegex(RuntimeError, r'floating point'):
                torch_method(3, dtype=dtype)
            return
        
        # 遍历不同窗口大小和周期性参数的组合
        for size in [0, 1, 2, 5, 10, 50, 100, 1024, 2048]:
            for periodic in [True, False]:
                # 调用 torch 中的窗口生成方法，返回结果保存在 res 中
                res = torch_method(
                    size,
                    periodic=periodic,
                    layout=torch.strided,
                    requires_grad=False,
                    **kwargs,
                    device=device,
                    dtype=dtype,
                )
                # 使用 scipy 的 get_window 函数获取参考数据，转换为 torch.Tensor 类型
                ref = torch.from_numpy(
                    signal.get_window(
                        (name, *(kwargs.values())), size, fftbins=periodic
                    )
                )
                # 断言 torch 生成的结果与参考数据 ref 相等
                self.assertEqual(res, ref.to(dtype))
        
        # 断言对于稀疏类型，会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r'not implemented for sparse types'):
            torch_method(3, layout=torch.sparse_coo)
        
        # 断言生成的 Tensor 是否支持梯度
        self.assertTrue(torch_method(3, requires_grad=True).requires_grad)
        self.assertFalse(torch_method(3).requires_grad)

    # 标记仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 针对不同数据类型设置精度调整
    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    # 如果未安装 Scipy，则跳过这个测试
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    # 如果不是 TorchDynamo 环境，则跳过这个测试
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    # 标记 CUDA 情况下的数据类型
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16, torch.half, torch.long)
    # 标记数据类型
    @dtypes(torch.float, torch.double, torch.long)
    # 使用参数化测试窗口函数
    @parametrize("window", ['hann', 'hamming', 'bartlett', 'blackman'])
    # 测试信号窗口函数
    def test_signal_window_functions(self, device, dtype, window):
        self._test_signal_window_functions(window, dtype, device)

    # 标记仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 针对不同数据类型设置精度调整
    @precisionOverride({torch.bfloat16: 5e-2, torch.half: 1e-3})
    # 如果未安装 Scipy，则跳过这个测试
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    # 如果 TorchDynamo 无法通过测试，则跳过这个测试
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    # 标记 CUDA 情况下的数据类型
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16, torch.half, torch.long)
    # 标记数据类型
    @dtypes(torch.float, torch.double, torch.long, torch.bfloat16, torch.float16)
    # 测试 Kaiser 窗口函数
    def test_kaiser_window(self, device, dtype):
        # 随机生成 beta 参数进行多次测试
        for num_test in range(50):
            self._test_signal_window_functions('kaiser', dtype, device, beta=random.random() * 30)
    # 定义一个测试函数，用于测试信号处理中的窗口函数在 Windows 平台下的行为
    def _test_signal_windows_functions(self, name, dtype, device, **kwargs):
        # 导入 scipy.signal 库中的 signal 模块
        import scipy.signal as signal

        # 获取 torch.signal.windows 模块中的指定窗口函数
        torch_method = getattr(torch.signal.windows, name)
        
        # 如果数据类型不是浮点型，则断言引发 RuntimeError 异常，提示需要浮点数作为输入
        if not dtype.is_floating_point:
            with self.assertRaisesRegex(RuntimeError, r'floating point'):
                torch_method(3, dtype=dtype)
            return
        
        # 遍历不同的窗口大小和周期性参数进行测试
        for size in [0, 1, 2, 5, 10, 50, 100, 1024, 2048]:
            for periodic in [True, False]:
                # 调用 torch 中的窗口函数，生成窗口数据
                res = torch_method(size, sym=not periodic, **kwargs, device=device, dtype=dtype)
                # 注意：scipy 中的窗口函数始终返回 float64 类型的结果
                ref = torch.from_numpy(signal.get_window((name, *(kwargs.values())), size, fftbins=periodic))
                # 断言 torch 生成的窗口数据与 scipy 生成的窗口数据一致，允许数据类型不完全匹配
                self.assertEqual(res, ref, exact_dtype=False)
        
        # 验证是否能正确处理需要梯度的情况
        self.assertTrue(torch_method(3, requires_grad=True).requires_grad)
        # 验证是否能正确处理不需要梯度的情况
        self.assertFalse(torch_method(3).requires_grad)

    # 测试所有 torch.signal.windows 中的窗口函数（除了具有额外参数的函数）
    @onlyNativeDeviceTypes
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @dtypes(torch.float, torch.double)
    @parametrize("window", ['bartlett', 'blackman', 'cosine', 'hamming', 'hann', 'nuttall'])
    def test_signal_windows_functions(self, device, dtype, window):
        # 调用 _test_signal_windows_functions 进行具体窗口函数的测试
        self._test_signal_windows_functions(window, dtype, device)

    # 测试 torch.signal.windows.kaiser 窗口函数
    @onlyNativeDeviceTypes
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    @dtypes(torch.float, torch.double)
    def test_kaiser(self, device, dtype):
        # 随机生成 beta 参数，进行多次测试
        for num_test in range(50):
            self._test_signal_windows_functions('kaiser', dtype, device, beta=random.random() * 30)

    # 测试在 GPU 上张量工厂函数的类型推断
    @onlyCUDA
    def test_tensor_factory_gpu_type_inference(self, device):
        # 设置默认的张量类型为 CUDA 的双精度张量
        with set_default_tensor_type(torch.cuda.DoubleTensor):
            with set_default_dtype(torch.float32):
                # 验证默认的数据类型是否为 torch.float32
                self.assertIs(torch.float32, torch.tensor(0.).dtype)
                # 验证张量所在的设备是否与给定的设备相符
                self.assertEqual(torch.device(device), torch.tensor(0.).device)
            with set_default_dtype(torch.float64):
                # 验证默认的数据类型是否为 torch.float64
                self.assertIs(torch.float64, torch.tensor(0.).dtype)
                # 验证张量所在的设备是否与给定的设备相符
                self.assertEqual(torch.device(device), torch.tensor(0.).device)

    # 测试在 GPU 上张量工厂函数的类型
    @onlyCUDA
    def test_tensor_factory_gpu_type(self, device):
        # 设置默认的张量类型为 CUDA 的单精度张量
        with set_default_tensor_type(torch.cuda.FloatTensor):
            x = torch.zeros((5, 5))
            # 验证张量的数据类型是否为 torch.float32
            self.assertIs(torch.float32, x.dtype)
            # 验证张量是否位于 GPU 上
            self.assertTrue(x.is_cuda)
        # 设置默认的张量类型为 CUDA 的双精度张量
        with set_default_tensor_type(torch.cuda.DoubleTensor):
            x = torch.zeros((5, 5))
            # 验证张量的数据类型是否为 torch.float64
            self.assertIs(torch.float64, x.dtype)
            # 验证张量是否位于 GPU 上
            self.assertTrue(x.is_cuda)

    # 如果是 CPU，跳过此测试，因为它与 CPU 设备比较
    @skipCPUIf(True, 'compares device with cpu')
    @dtypes(torch.int, torch.long, torch.float, torch.double)
    # 定义一个测试方法，用于比较在不同设备上生成的 arange 张量是否相等
    def test_arange_device_vs_cpu(self, device, dtype):
        # 在 CPU 上生成指定范围的张量
        cpu_tensor = torch.arange(0, 10, dtype=dtype, device='cpu')
        # 在指定设备上生成指定范围的张量
        device_tensor = torch.arange(0, 10, dtype=dtype, device=device)
        # 断言两个张量是否相等
        self.assertEqual(cpu_tensor, device_tensor)

    # 使用装饰器指定多种数据类型，测试在低精度条件下生成 arange 张量的正确性
    @dtypes(torch.bfloat16, torch.float16)
    def test_arange_lowp(self, device, dtype):
        # 创建参考张量，在指定设备和数据类型下生成
        ref_tensor = torch.tensor([0, 1, 2, 3], dtype=dtype, device=device)
        # 使用 arange 在指定设备和数据类型下生成张量
        f16_tensor = torch.arange(0, 4, dtype=dtype, device=device)
        # 断言两个张量是否相等
        self.assertEqual(ref_tensor, f16_tensor)

        # 以步长 2 生成张量
        ref_tensor = torch.tensor([0, 2, 4], dtype=dtype, device=device)
        f16_tensor = torch.arange(0, 6, step=2, dtype=dtype, device=device)
        # 断言两个张量是否相等
        self.assertEqual(ref_tensor, f16_tensor)

    # 辅助方法，用于测试 linspace 和 logspace 函数的推导能力
    @dtypes(*all_types_and_complex_and(torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.bfloat16))
    def _test_linspace_logspace_deduction_helper(self, fn, device):
        # 遍历不同的起始点和结束点组合
        for start, end in [(1, 2), (1., 2), (1., -2.), (1j, 2j), (0., 2j), (1j, 2)]:
            # 确定数据类型为 float32 或 cfloat，根据是否包含复数来决定
            dtype = torch.float32
            if isinstance(start, complex) or isinstance(end, complex):
                dtype = torch.cfloat

            # 断言生成的张量数据类型是否符合预期
            self.assertEqual(fn(start, end, steps=100, device=device).dtype, dtype)

    # 在不同设备上测试 linspace 函数的推导能力
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_linspace_deduction(self, device):
        # 测试从输入参数推导 linspace 函数的能力
        self._test_linspace_logspace_deduction_helper(torch.linspace, device)

    # 在不同设备上测试 logspace 函数的推导能力
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_logspace_deduction(self, device):
        # 测试从输入参数推导 logspace 函数的能力
        self._test_linspace_logspace_deduction_helper(torch.logspace, device)

    # 精度调整注释：Linspace 和 Logspace 的 torch.half CUDA 内核精度较低
    # 由于 linspace/logspace 是确定性的，我们可以计算预期误差的量（通过不使用精度调整进行测试），
    # 在此基础上增加一个小量（EPS），将该值作为精度调整的参数。
    LINSPACE_LOGSPACE_EXTRA_EPS = 1e-5

    # 注意 [Linspace+Logspace precision override]
    # 我们的 Linspace 和 Logspace 在 torch.half CUDA 内核下精度不高。
    # 由于 linspace/logspace 是确定性的，我们可以计算一个预期的误差量（通过不使用精度覆盖进行测试），
    # 并添加一个小量（EPS），使用该值作为精度覆盖的参数。
    LINSPACE_LOGSPACE_SPECIAL_STEPS = [0, 1]

    # 在不同设备上比较 linspace 函数生成的张量与 CPU 生成的张量
    def _test_linspace(self, device, dtype, steps):
        # 在指定设备上生成指定步长的 linspace 张量
        a = torch.linspace(0, 10, steps=steps, dtype=dtype, device=device)
        # 在 CPU 上生成指定步长的 linspace 张量
        b = torch.linspace(0, 10, steps=steps)
        # 断言两个张量是否相等，允许数据类型不完全匹配
        self.assertEqual(a, b, exact_dtype=False)

    # 查看注意 [Linspace+Logspace precision override]
    # 如果与 CPU 比较，跳过测试
    @skipCPUIf(True, "compares with CPU")
    # 精度覆盖装饰器，针对浮点数和复数类型的张量，特别是 torch.half 和 torch.bfloat16
    @precisionOverride({torch.half: 0.0039 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def test_linspace_device_vs_cpu(self, device, dtype):
        # 调用 _test_linspace 方法，在指定设备上进行测试
        self._test_linspace(device, dtype, steps=10)
    # 根据条件跳过CPU测试，用于比较与CPU的性能
    @skipCPUIf(True, "compares with CPU")
    # 标记测试函数支持的数据类型，包括浮点数和复数类型，以及指定的半精度和bfloat16类型
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def test_linspace_special_steps(self, device, dtype):
        # 遍历预定义的特殊步数列表，对每个步数进行 _test_linspace 测试
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_linspace(device, dtype, steps=steps)

    # 比较在设备和CPU上的 logspace 结果
    def _test_logspace(self, device, dtype, steps):
        # 使用 torch.logspace 在指定设备和数据类型下创建张量 a
        a = torch.logspace(1, 1.1, steps=steps, dtype=dtype, device=device)
        # 使用 torch.logspace 在 CPU 上创建张量 b
        b = torch.logspace(1, 1.1, steps=steps)
        # 断言张量 a 与 b 的值相等（忽略精确数据类型）
        self.assertEqual(a, b, exact_dtype=False)

    # 比较在设备和CPU上的 base=2 的 logspace 结果
    def _test_logspace_base2(self, device, dtype, steps):
        # 使用 torch.logspace 在指定设备和数据类型下创建张量 a，设置基数为2
        a = torch.logspace(1, 1.1, steps=steps, base=2, dtype=dtype, device=device)
        # 使用 torch.logspace 在 CPU 上创建张量 b，设置基数为2
        b = torch.logspace(1, 1.1, steps=steps, base=2)
        # 断言张量 a 与 b 的值相等（忽略精确数据类型）
        self.assertEqual(a, b, exact_dtype=False)

    # 查看 NOTE [Linspace+Logspace precision override] 注释内容
    @skipCPUIf(True, "compares with CPU")
    # 设置在 CUDA 环境下的数据类型精度覆盖，特别处理半精度浮点数
    @precisionOverride({torch.half: 0.025 + LINSPACE_LOGSPACE_EXTRA_EPS})
    # 标记测试函数支持的数据类型，如果在 CUDA 下，包括半精度、单精度和双精度类型，否则包括单精度和双精度类型
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_device_vs_cpu(self, device, dtype):
        # 调用 _test_logspace 函数进行测试，设置步数为 10
        self._test_logspace(device, dtype, steps=10)

    # 查看 NOTE [Linspace+Logspace precision override] 注释内容
    @skipCPUIf(True, "compares with CPU")
    # 设置在 CUDA 环境下的数据类型精度覆盖，特别处理半精度浮点数
    @precisionOverride({torch.half: 0.0201 + LINSPACE_LOGSPACE_EXTRA_EPS})
    # 标记测试函数支持的数据类型，如果在 CUDA 下，包括半精度、单精度和双精度类型，否则包括单精度和双精度类型
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_base2(self, device, dtype):
        # 调用 _test_logspace_base2 函数进行测试，设置步数为 10
        self._test_logspace_base2(device, dtype, steps=10)

    # 标记跳过CPU测试，用于比较与CPU的性能
    @skipCPUIf(True, "compares with CPU")
    # 标记测试函数支持的数据类型，如果在 CUDA 下，包括半精度、单精度和双精度类型，否则包括单精度和双精度类型
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_special_steps(self, device, dtype):
        # 遍历预定义的特殊步数列表，对每个步数进行 _test_logspace 和 _test_logspace_base2 测试
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_logspace(device, dtype, steps=steps)
            self._test_logspace_base2(device, dtype, steps=steps)

    # 标记测试函数支持的所有数据类型和指定的 bfloat16 类型
    @dtypes(*all_types_and(torch.bfloat16))
    # 如果在 CUDA 下，标记测试函数支持的整数类型和指定的半精度、bfloat16、单精度和双精度类型；否则支持所有类型和指定的半精度和 bfloat16 类型
    @dtypesIfCUDA(*integral_types_and(torch.half, torch.bfloat16, torch.float32, torch.float64) if TEST_WITH_ROCM else
                  all_types_and(torch.half, torch.bfloat16))
    # 定义一个测试函数 test_logspace，接受 device 和 dtype 作为参数
    def test_logspace(self, device, dtype):
        # 生成一个随机的起始点 _from
        _from = random.random()
        # 在 _from 的基础上再生成一个随机数，作为结束点 to
        to = _from + random.random()
        # 使用 torch.logspace 生成一个张量 res1，包含从 _from 到 to 的对数间隔值，共 137 个点
        res1 = torch.logspace(_from, to, 137, device=device, dtype=dtype)
        # 创建一个空的张量 res2，设备和数据类型与 res1 相同
        res2 = torch.tensor((), device=device, dtype=dtype)
        # 将生成的对数间隔值写入 res2 中
        torch.logspace(_from, to, 137, device=device, dtype=dtype, out=res2)
        # 断言 res1 和 res2 的值相等，允许误差范围为绝对误差和相对误差都为 0
        self.assertEqual(res1, res2, atol=0, rtol=0)
        # 预期引发 RuntimeError，因为 steps 参数不能为负数
        self.assertRaises(RuntimeError, lambda: torch.logspace(0, 1, -1, device=device, dtype=dtype))
        # 预期引发 TypeError，因为未提供 steps 参数
        self.assertRaises(TypeError, lambda: torch.logspace(0, 1, device=device, dtype=dtype))
        # 断言生成包含一个元素的张量，其值为 1，设备和数据类型与指定相同
        self.assertEqual(torch.logspace(0, 1, 1, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype), atol=0, rtol=0)

        # 如果 dtype 是 torch.float 类型
        if dtype == torch.float:
            # 预期引发 RuntimeError，因为无法安全地将传递的 dtype 强制转换为推断的 dtype
            with self.assertRaisesRegex(RuntimeError, r"torch.logspace\(\): inferred dtype"):
                torch.logspace(0, 1j, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"torch.logspace\(\): inferred dtype"):
                torch.logspace(0j, 1, 5, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, r"torch.logspace\(\): inferred dtype"):
                torch.logspace(0j, 1j, 5, device=device, dtype=dtype)

        # 检查精度 - 选择的 start、stop 和 base 避免溢出
        # 选择 steps 以避免由于舍入误差而导致步长大小不准确
        # 由于计算差异，GPU 测试需要设置公差
        atol = None
        rtol = None
        if self.device_type == 'cpu':
            atol = 0
            rtol = 0
        # 断言生成的张量与手动计算的张量相等，使用设备和数据类型进行计算
        self.assertEqual(torch.tensor([2. ** (i / 8.) for i in range(49)], device=device, dtype=dtype),
                         torch.logspace(0, 6, steps=49, base=2, device=device, dtype=dtype),
                         atol=atol, rtol=rtol)

        # 检查非默认 base=2 的情况
        # 断言生成的张量与手动计算的张量相等，使用设备和数据类型进行计算
        self.assertEqual(torch.logspace(1, 1, 1, 2, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype) * 2)
        # 断言生成的张量与手动计算的张量相等，使用设备和数据类型进行计算
        self.assertEqual(torch.logspace(0, 2, 3, 2, device=device, dtype=dtype),
                         torch.tensor((1, 2, 4), device=device, dtype=dtype))

        # 检查 logspace_ 函数用于生成起始点大于结束点的情况
        # 断言生成的张量与手动计算的张量相等，使用设备和数据类型进行计算
        self.assertEqual(torch.logspace(1, 0, 2, device=device, dtype=dtype),
                         torch.tensor((10, 1), device=device, dtype=dtype), atol=0, rtol=0)

        # 检查 logspace_ 函数用于非连续张量的情况
        # 创建一个形状为 (2, 3) 的全零张量 x
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        # 在 x 的第二列上使用 logspace 函数生成对数间隔值
        y = torch.logspace(0, 3, 4, base=2, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        # 断言 x 的值与手动计算的张量相等，使用设备和数据类型进行计算
        self.assertEqual(x, torch.tensor(((0, 1, 2), (0, 4, 8)), device=device, dtype=dtype), atol=0, rtol=0)
    # 定义一个测试函数，用于全面推断
    def test_full_inference(self, device, dtype):
        # 设置张量的大小为 (2, 2)
        size = (2, 2)

        # 使用指定的数据类型作为默认类型环境
        with set_default_dtype(dtype):
            # 测试布尔类型填充值的推断
            t = torch.full(size, True)
            self.assertEqual(t.dtype, torch.bool)

            # 测试整数类型填充值的推断
            t = torch.full(size, 1)
            self.assertEqual(t.dtype, torch.long)

            # 测试浮点数类型填充值的推断
            t = torch.full(size, 1.)
            self.assertEqual(t.dtype, dtype)

            # 测试复数类型的推断
            t = torch.full(size, (1 + 1j))
            # 根据指定的数据类型确定复数类型的精度
            ctype = torch.complex128 if dtype is torch.double else torch.complex64
            self.assertEqual(t.dtype, ctype)

    # 定义一个测试函数，用于测试填充操作的输出
    def test_full_out(self, device):
        # 设置张量的大小为 (5,)
        size = (5,)
        # 创建一个空的张量，指定设备和数据类型为 long
        o = torch.empty(size, device=device, dtype=torch.long)

        # 验证当输出张量的数据类型与填充值的数据类型不一致时是否会抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.full(o.shape, 1., dtype=torch.float, out=o)

        # 验证输出张量的数据类型是否覆盖了推断结果
        self.assertEqual(torch.full(o.shape, 1., out=o).dtype, o.dtype)
        self.assertEqual(torch.full(size, 1, out=o).dtype, o.dtype)

    # 检查当从不可写入的 NumPy 数组创建副本时，是否抑制了警告信息
    # 关于复制操作的警告信息。
    # 见问题 #47160
    def test_tensor_from_non_writable_numpy(self, device):
        with warnings.catch_warnings(record=True) as w:
            # 创建一个包含 5 个元素的 NumPy 浮点数数组
            a = np.arange(5.)
            # 将数组标记为不可写入
            a.flags.writeable = False
            # 将 NumPy 数组转换为 PyTorch 张量
            t = torch.tensor(a)
            # 验证是否没有产生警告信息
            self.assertEqual(len(w), 0)

    # 从文件创建张量的测试函数，用于测试特定平台和参数化设置
    @onlyCPU  # 仅在 CPU 环境下执行测试
    @parametrize('shared', [True, False])  # 参数化测试，包括共享和非共享模式
    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")  # 如果是 Windows 系统，则跳过测试
    def test_from_file(self, device, shared):
        # 指定数据类型为双精度浮点型
        dtype = torch.float64
        # 创建一个指定设备和数据类型的随机张量
        t = torch.randn(2, 5, dtype=dtype, device=device)
        # 使用临时文件
        with tempfile.NamedTemporaryFile() as f:
            # 如果共享模式为真，则设置预期的文件名，否则为 None
            expected_filename = f.name if shared else None
            # 将张量的数据写入到临时文件中
            t.numpy().tofile(f)
            # 从文件中读取张量，并验证文件名是否符合预期，数据是否正确
            t_mapped = torch.from_file(f.name, shared=shared, size=t.numel(), dtype=dtype)
            self.assertTrue(t_mapped.untyped_storage().filename == expected_filename)
            self.assertEqual(torch.flatten(t), t_mapped)

            # 使用未类型化存储对象从文件中读取数据，验证文件名是否符合预期
            s = torch.UntypedStorage.from_file(f.name, shared, t.numel() * dtype.itemsize)
            self.assertTrue(s.filename == expected_filename)

    # 测试存储对象的文件名是否为 None
    @onlyCPU  # 仅在 CPU 环境下执行测试
    def test_storage_filename(self, device):
        # 创建一个指定设备的随机张量
        t = torch.randn(2, 5, device=device)
        # 验证未类型化存储对象的文件名是否为 None
        self.assertIsNone(t.untyped_storage().filename)
# 用于测试随机张量创建操作的测试类，例如 torch.randint
class TestRandomTensorCreation(TestCase):
    # 精确的数据类型匹配标志，设置为 True
    exact_dtype = True

    # 添加 torch.complex64 和 torch.complex128 的待办事项
    # 此测试用例装饰器，指定测试使用的数据类型为 torch.float 和 torch.double
    @dtypes(torch.float, torch.double)
    # 当 `std` < 0 时，确保 normal 函数会引发适当的错误
    def test_normal_std_error(self, device):
        # 在指定设备上创建 torch.float32 类型的张量 a，并赋值为 0
        a = torch.tensor(0, dtype=torch.float32, device=device)
        # 在指定设备上创建 torch.float32 类型的张量 std，并赋值为 -1
        std = torch.tensor(-1, dtype=torch.float32, device=device)

        # 对输入列表中的每个输入进行测试
        for input in [0, a]:
            # 使用断言检查是否会抛出 RuntimeError 异常，且异常信息中包含 'normal expects std >= 0.0, but found std'
            with self.assertRaisesRegex(RuntimeError, r'normal expects std >= 0.0, but found std'):
                torch.normal(input, -1, (10,))

            # 使用断言检查是否会抛出 RuntimeError 异常，且异常信息中包含 'normal expects all elements of std >= 0.0'
            with self.assertRaisesRegex(RuntimeError, r'normal expects all elements of std >= 0.0'):
                torch.normal(input, std)

    # 添加链接到 GitHub 问题编号为 126834 的注释
    # 如果运行环境不是 Torch Dynamo，则跳过此测试
    @xfailIfTorchDynamo
    # 根据设备类型不同，指定不同的数据类型进行测试
    @dtypes(torch.float, torch.double, torch.half)
    @dtypesIfCUDA(torch.float, torch.double, torch.half, torch.bfloat16)
    # 测试均匀分布生成函数 uniform_ 的行为，针对不同的设备和数据类型
    def test_uniform_from_to(self, device, dtype):
        # 设置生成张量的大小
        size = 2000
        # 设置 alpha 值
        alpha = 0.1

        # 获取不同数据类型的最小和最大值
        float_min = torch.finfo(torch.float).min
        float_max = torch.finfo(torch.float).max
        double_min = torch.finfo(torch.double).min
        double_max = torch.finfo(torch.double).max

        # 根据数据类型设置不同的最小和最大值
        if dtype == torch.bfloat16:
            min_val = -3.389531389251535e+38
            max_val = 3.389531389251535e+38
        else:
            min_val = torch.finfo(dtype).min
            max_val = torch.finfo(dtype).max

        # 定义一组测试值
        values = [double_min, float_min, -42, 0, 42, float_max, double_max]

        # 循环遍历测试值
        for from_ in values:
            for to_ in values:
                # 创建一个空张量 t
                t = torch.empty(size, dtype=dtype, device=device)
                # 检查 from_ 和 to_ 是否在设定的范围内，如果不是则跳过
                if not (min_val <= from_ <= max_val) or not (min_val <= to_ <= max_val):
                    pass
                # 如果 to_ 小于 from_，则测试 uniform_ 是否会引发 RuntimeError
                elif to_ < from_:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to return",
                        lambda: t.uniform_(from_, to_)
                    )
                # 如果 to_ - from_ 大于 max_val，则测试 uniform_ 是否会引发 RuntimeError
                elif to_ - from_ > max_val:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "uniform_ expects to-from",
                        lambda: t.uniform_(from_, to_)
                    )
                else:
                    # 执行 uniform_ 函数生成张量 t
                    t.uniform_(from_, to_)
                    # 计算生成的数值范围
                    range_ = to_ - from_
                    # 进行额外的数值范围和精度检查
                    if not (dtype == torch.bfloat16) and not (
                            dtype == torch.half and device == 'cpu') and not torch.isnan(t).all():
                        delta = alpha * range_
                        double_t = t.to(torch.double)
                        if range_ == 0:
                            self.assertTrue(double_t.min() == from_)
                            self.assertTrue(double_t.max() == to_)
                        elif dtype == torch.half:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() <= to_)
                        else:
                            self.assertTrue(from_ <= double_t.min() <= (from_ + delta))
                            self.assertTrue((to_ - delta) <= double_t.max() < to_)

    # 测试随机生成负值的行为
    def test_random_neg_values(self, device):
        SIZE = 10
        # 定义有符号数据类型的列表
        signed_dtypes = [torch.double, torch.float, torch.long, torch.int, torch.short]
        # 循环遍历有符号数据类型
        for dtype in signed_dtypes:
            # 生成 SIZE x SIZE 的随机张量并转换到指定设备和数据类型
            res = torch.rand(SIZE, SIZE).to(device=device, dtype=dtype)
            # 在范围 [-10, -1] 内生成随机数
            res.random_(-10, -1)
            # 断言生成的随机数张量的最大值不超过 9
            self.assertLessEqual(res.max().item(), 9)
            # 断言生成的随机数张量的最小值不低于 -10
            self.assertGreaterEqual(res.min().item(), -10)

    # TODO: this test should be updated
    @onlyCPU
    # 在给定设备上测试 torch.randint 函数的推断
    def test_randint_inference(self, device):
        # 定义尺寸为 (2, 1)
        size = (2, 1)
        # 遍历不同参数组合：(3,) 和 (1, 3)，分别表示 (low,) 和 (low, high)
        for args in [(3,), (1, 3)]:  
            # 测试返回的数据类型是否为 torch.int64
            self.assertIs(torch.int64, torch.randint(*args, size=size).dtype)
            # 测试返回的数据类型是否为 torch.int64，同时指定 layout 为 torch.strided
            self.assertIs(torch.int64, torch.randint(*args, size=size, layout=torch.strided).dtype)
            # 测试返回的数据类型是否为 torch.int64，同时指定 generator 为默认生成器
            self.assertIs(torch.int64, torch.randint(*args, size=size, generator=torch.default_generator).dtype)
            # 测试返回的数据类型是否为 torch.float32，同时指定数据类型为 torch.float32
            self.assertIs(torch.float32, torch.randint(*args, size=size, dtype=torch.float32).dtype)
            # 创建一个空的张量 out，指定数据类型为 torch.float32
            out = torch.empty(size, dtype=torch.float32)
            # 测试返回的数据类型是否为 torch.float32，同时将结果写入预先创建的 out 张量中
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out).dtype)
            # 测试返回的数据类型是否为 torch.float32，同时将结果写入预先创建的 out 张量中，并指定数据类型为 torch.float32
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out, dtype=torch.float32).dtype)
            # 创建一个空的张量 out，指定数据类型为 torch.int64
            out = torch.empty(size, dtype=torch.int64)
            # 测试返回的数据类型是否为 torch.int64，同时将结果写入预先创建的 out 张量中
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out).dtype)
            # 测试返回的数据类型是否为 torch.int64，同时将结果写入预先创建的 out 张量中，并指定数据类型为 torch.int64
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out, dtype=torch.int64).dtype)

    # TODO: 需要更新此测试
    @onlyCPU
    def test_randint(self, device):
        # 定义尺寸大小为 100
        SIZE = 100

        # 定义种子函数，用于设置随机数生成器的种子
        def seed(generator):
            if generator is None:
                torch.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        # 遍历 generator 的两种可能取值：None 和 torch.Generator()
        for generator in (None, torch.Generator()):
            generator = seed(generator)
            # 使用指定生成器生成随机整数张量，范围为 [0, 6)，大小为 (SIZE, SIZE)
            res1 = torch.randint(0, 6, (SIZE, SIZE), generator=generator)
            # 创建一个空的张量 res2，数据类型为 torch.int64
            res2 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            # 使用指定生成器生成随机整数张量，范围为 [0, 6)，大小为 (SIZE, SIZE)，并将结果写入 res2 中
            torch.randint(0, 6, (SIZE, SIZE), generator=generator, out=res2)
            generator = seed(generator)
            # 使用指定生成器生成随机整数张量，范围为 [0, 6)，大小为 (SIZE, SIZE)，并指定 res3 为结果
            res3 = torch.randint(6, (SIZE, SIZE), generator=generator)
            # 创建一个空的张量 res4，数据类型为 torch.int64
            res4 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            # 使用指定生成器生成随机整数张量，范围为 [0, 6)，大小为 (SIZE, SIZE)，将结果写入 res4 中
            torch.randint(6, (SIZE, SIZE), out=res4, generator=generator)
            # 检查各对结果张量是否相等
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res3)
            self.assertEqual(res1, res4)
            self.assertEqual(res2, res3)
            self.assertEqual(res2, res4)
            self.assertEqual(res3, res4)
            # 检查 res1 张量中所有元素是否都小于 6
            self.assertTrue((res1 < 6).all().item())
            # 检查 res1 张量中所有元素是否都大于等于 0
            self.assertTrue((res1 >= 0).all().item())

    # 测试 torch.randn 函数，指定不同数据类型
    @dtypes(torch.half, torch.float, torch.bfloat16, torch.double,
            torch.complex32, torch.complex64, torch.complex128)
    def test_randn(self, device, dtype):
        # 定义尺寸大小为 100
        SIZE = 100
        # 遍历尺寸为 0 和 SIZE 的情况
        for size in [0, SIZE]:
            # 设置随机种子
            torch.manual_seed(123456)
            # 使用指定数据类型和设备生成正态分布张量
            res1 = torch.randn(size, size, dtype=dtype, device=device)
            # 创建一个空的张量 res2，指定数据类型和设备
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            # 使用指定数据类型和设备生成正态分布张量，并将结果写入 res2 中
            torch.randn(size, size, out=res2)
            # 检查 res1 和 res2 张量是否相等
            self.assertEqual(res1, res2)
    # 定义一个测试函数，用于测试 torch.rand 的随机性和设备兼容性
    def test_rand(self, device, dtype):
        SIZE = 100
        # 遍历测试用例，包括 size 为 0 和 SIZE 两种情况
        for size in [0, SIZE]:
            # 设置随机种子，保证结果可复现
            torch.manual_seed(123456)
            # 使用 torch.rand 生成指定大小的随机张量 res1，指定数据类型和设备
            res1 = torch.rand(size, size, dtype=dtype, device=device)
            # 使用 torch.tensor 创建一个空张量 res2，指定数据类型和设备
            res2 = torch.tensor([], dtype=dtype, device=device)
            # 重新设置随机种子，保证结果与 res1 相同
            torch.manual_seed(123456)
            # 使用 torch.rand 生成指定大小的随机张量 res2，输出到预先创建的空张量中
            torch.rand(size, size, out=res2)
            # 断言两个张量 res1 和 res2 在值上相等
            self.assertEqual(res1, res2)

    # 仅在 CUDA 环境下执行的测试函数，用于测试 randperm 在设备兼容性上的异常情况
    @onlyCUDA
    def test_randperm_device_compatibility(self, device):
        # 创建一个在 CUDA 设备上的生成器 cuda_gen 和一个在 CPU 设备上的生成器 cpu_gen
        cuda_gen = torch.Generator(device='cuda')
        cpu_gen = torch.Generator(device='cpu')

        # 对 n=0 的特殊情况进行测试，此时不需要使用生成器，因此不会因设备和生成器不匹配而产生错误
        torch.randperm(0, device='cuda:0', generator=torch.Generator(device='cuda:1'))
        # 如果存在多个 CUDA 设备，也不会因为设备和生成器不匹配而产生错误
        if torch.cuda.device_count() > 1:
            torch.randperm(0, device='cuda:1', generator=torch.Generator(device='cuda:0'))
        # 在 CUDA 设备上使用 CPU 生成器，会引发异常
        torch.randperm(0, device='cuda', generator=torch.Generator(device='cpu'))
        # 在 CPU 设备上使用 CUDA 生成器，同样会引发异常
        torch.randperm(0, device='cpu', generator=torch.Generator(device='cuda'))

        # 遍历不同的 n 值进行测试
        for n in (1, 3, 100, 30000):
            # 在 CUDA 设备上使用 CUDA 生成器，保证设备和生成器匹配
            torch.randperm(n, device='cuda', generator=torch.Generator(device='cuda:0'))
            torch.randperm(n, device='cuda:0', generator=torch.Generator(device='cuda'))
            
            # 对于 cuda:0 设备，需要确保生成器的设备类型匹配行为与 torch.randint 一致
            # generator 应忽略设备顺序，因为这不会影响生成器的使用
            torch.randint(low=0, high=n + 1, size=(1,), device="cuda:0", generator=torch.Generator(device='cuda:1'))
            torch.randperm(n, device='cuda:0', generator=torch.Generator(device='cuda:1'))
            
            # 如果存在多个 CUDA 设备，需要确保设备类型匹配行为与 torch.randint 一致
            torch.randint(low=0, high=n + 1, size=(1,), device="cuda:1", generator=torch.Generator(device='cuda:0'))
            torch.randperm(n, device='cuda:1', generator=torch.Generator(device='cuda:0'))

            # 使用 lambda 函数和 assertRaisesRegex 断言捕获 RuntimeError 异常，并检查异常消息
            regex = 'Expected a .* device type for generator but found .*'
            cuda_t = torch.tensor(n, device='cuda')
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cuda', generator=cpu_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cuda', generator=cpu_gen, out=cuda_t))
            cpu_t = torch.tensor(n, device='cpu')
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cpu', generator=cuda_gen))
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, device='cpu', generator=cuda_gen, out=cpu_t))
            # 隐式在 CPU 上执行，使用 cuda_gen 生成器，预期捕获 RuntimeError 异常
            self.assertRaisesRegex(RuntimeError, regex, lambda: torch.randperm(n, generator=cuda_gen))  # implicitly on CPU
# Class for testing *like ops, like torch.ones_like
class TestLikeTensorCreation(TestCase):
    # 是否精确匹配数据类型的标志
    exact_dtype = True

    # TODO: this test should be updated
    # 测试 torch.ones_like 函数
    def test_ones_like(self, device):
        # 期望的张量为全1张量，形状为 (100, 100)，在指定设备上生成
        expected = torch.ones(100, 100, device=device)

        # 使用 torch.ones_like 函数生成与 expected 相同形状的全1张量
        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

        # 测试布尔类型张量的情况
        expected = torch.tensor([True, True], device=device, dtype=torch.bool)
        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

    # TODO: this test should be updated
    # 仅在CPU上运行的测试函数
    @onlyCPU
    def test_empty_like(self, device):
        # 创建不同类型的 Variable 张量
        x = torch.autograd.Variable(torch.tensor([]))
        y = torch.autograd.Variable(torch.randn(4, 4))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))

        # 遍历每个张量，验证 torch.empty_like 函数生成的张量形状与原张量一致
        for a in (x, y, z):
            self.assertEqual(torch.empty_like(a).shape, a.shape)
            # 验证 torch.empty_like 函数生成的张量与原张量的类型和字符串表示一致
            self.assertEqualTypeString(torch.empty_like(a), a)

    # 测试 torch.zeros_like 函数
    def test_zeros_like(self, device):
        # 期望的张量为全0张量，形状为 (100, 100)，在指定设备上生成
        expected = torch.zeros((100, 100,), device=device)

        # 使用 torch.zeros_like 函数生成与 expected 相同形状的全0张量
        res1 = torch.zeros_like(expected)
        self.assertEqual(res1, expected)

    # 在至少包含两个设备的环境下测试 torch.zeros_like 函数
    @deviceCountAtLeast(2)
    def test_zeros_like_multiple_device(self, devices):
        # 期望的张量为全0张量，形状为 (100, 100)，在指定设备上生成
        expected = torch.zeros(100, 100, device=devices[0])
        # 在不同设备上生成随机张量 x
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        # 使用 torch.zeros_like 函数生成与 x 相同形状的全0张量
        output = torch.zeros_like(x)
        self.assertEqual(output, expected)

    # 在至少包含两个设备的环境下测试 torch.ones_like 函数
    @deviceCountAtLeast(2)
    def test_ones_like_multiple_device(self, devices):
        # 期望的张量为全1张量，形状为 (100, 100)，在指定设备上生成
        expected = torch.ones(100, 100, device=devices[0])
        # 在不同设备上生成随机张量 x
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        # 使用 torch.ones_like 函数生成与 x 相同形状的全1张量
        output = torch.ones_like(x)
        self.assertEqual(output, expected)

    # 测试 torch.full_like 函数的类型推断
    # 当 dtype 显式设置时，优先使用该 dtype；否则使用 "like" 张量的 dtype
    @onlyNativeDeviceTypes
    def test_full_like_inference(self, device):
        size = (2, 2)
        # 使用 torch.empty 函数创建形状为 (5,) 的空张量，指定设备和数据类型为 torch.long
        like = torch.empty((5,), device=device, dtype=torch.long)

        # 验证使用 torch.full_like 函数生成全1张量时的数据类型为 torch.long
        self.assertEqual(torch.full_like(like, 1.).dtype, torch.long)
        # 验证使用 torch.full_like 函数生成指定复数类型全1张量时的数据类型为 torch.complex64
        self.assertEqual(torch.full_like(like, 1., dtype=torch.complex64).dtype,
                         torch.complex64)


# Tests for the `frombuffer` function (only work on CPU):
#   Constructs tensors from Python objects that implement the buffer protocol,
#   without copying data.
SIZE = 5
SHAPE = (SIZE,)

# 根据数据类型判断是否可能需要梯度
def may_require_grad(dtype):
    return dtype.is_floating_point or dtype.is_complex

# 获取数据类型的元素大小
def get_dtype_size(dtype):
    return int(torch.empty((), dtype=dtype).element_size())
    # 定义一个测试函数，用于测试给定形状、数据类型的张量转换操作
    def _run_test(self, shape, dtype, count=-1, first=0, offset=None, **kwargs):
        # 将 Torch 的数据类型转换为对应的 NumPy 数据类型
        numpy_dtype = torch_to_numpy_dtype_dict[dtype]

        # 如果未指定偏移量，则根据首元素位置和数据类型大小计算偏移量
        if offset is None:
            offset = first * get_dtype_size(dtype)

        # 创建指定形状、数据类型的张量并转换为 NumPy 数组
        numpy_original = make_tensor(shape, dtype=dtype, device="cpu").numpy()
        # 使用内存视图创建原始数据的内存视图
        original = memoryview(numpy_original)
        
        # 调用 PyTorch 的 frombuffer 方法进行转换，检查是否成功
        torch_frombuffer = torch.frombuffer(original, dtype=dtype, count=count, offset=offset, **kwargs)
        # 使用 NumPy 的 frombuffer 方法进行相同的转换
        numpy_frombuffer = np.frombuffer(original, dtype=numpy_dtype, count=count, offset=offset)

        # 断言两种方法得到的数组相等
        self.assertEqual(numpy_frombuffer, torch_frombuffer)
        # 断言两种方法得到的数据指针相同
        self.assertEqual(numpy_frombuffer.__array_interface__["data"][0], torch_frombuffer.data_ptr())
        
        # 返回原始的 NumPy 数组和转换后的 Torch 张量
        return (numpy_original, torch_frombuffer)

    # 使用各种 NumPy 到 Torch 数据类型映射进行测试
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_same_type(self, device, dtype):
        # 分别对空张量、长度为 4 的张量、大小为 10x10 的张量进行测试
        self._run_test((), dtype)
        self._run_test((4,), dtype)
        self._run_test((10, 10), dtype)

    # 使用各种 NumPy 到 Torch 数据类型映射进行测试
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_requires_grad(self, device, dtype):
        # 定义一个内部函数，用于测试是否正确设置了 requires_grad
        def _run_test_and_check_grad(requires_grad, *args, **kwargs):
            kwargs["requires_grad"] = requires_grad
            _, tensor = self._run_test(*args, **kwargs)
            self.assertTrue(tensor.requires_grad == requires_grad)

        # 获取当前数据类型是否需要 requires_grad
        requires_grad = may_require_grad(dtype)
        
        # 对空张量、长度为 4 的张量、大小为 10x10 的张量进行测试，设置 requires_grad
        _run_test_and_check_grad(requires_grad, (), dtype)
        _run_test_and_check_grad(requires_grad, (4,), dtype)
        _run_test_and_check_grad(requires_grad, (10, 10), dtype)
        
        # 对不需要 requires_grad 的情况进行测试
        _run_test_and_check_grad(False, (), dtype)
        _run_test_and_check_grad(False, (4,), dtype)
        _run_test_and_check_grad(False, (10, 10), dtype)

    # 使用各种 NumPy 到 Torch 数据类型映射进行测试
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_with_offset(self, device, dtype):
        # 测试带有偏移量的情况，偏移量应当在至少还有一个元素的情况下有效
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, first=i)

    # 使用各种 NumPy 到 Torch 数据类型映射进行测试
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_with_count(self, device, dtype):
        # 测试带有计数参数的情况，计数应当在 [-1, 输入长度] 区间有效，但不能为 0
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)
    # 测试函数，用于测试带有 count 和 offset 参数的功能
    def test_with_count_and_offset(self, device, dtype):
        # 明确指定默认的 count 参数范围为 [-1, 1, 2, ..., SIZE]
        for i in range(-1, SIZE + 1):
            if i != 0:
                # 调用测试函数，并传入指定的 SHAPE、dtype 和 count 参数值 i
                self._run_test(SHAPE, dtype, count=i)
        
        # 明确指定默认的 offset 参数范围为 [0, 1, ..., SIZE-1]
        for i in range(SIZE):
            # 调用测试函数，并传入指定的 SHAPE、dtype 和 first 参数值 i
            self._run_test(SHAPE, dtype, first=i)
        
        # 遍历所有可能的 count 和 first 参数组合，要求它们对齐
        # 对于 'input' 的 offset
        # count:[1, 2, ..., SIZE-1] x first:[0, 1, ..., SIZE-count]
        for i in range(1, SIZE):
            for j in range(SIZE - i + 1):
                # 调用测试函数，并传入指定的 SHAPE、dtype、count 和 first 参数值 i, j
                self._run_test(SHAPE, dtype, count=i, first=j)

    # 使用装饰器 dtypes，设置参数为 numpy_to_torch_dtype_dict 的所有值
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    # 测试函数，用于测试非法的位置参数情况
    def test_invalid_positional_args(self, device, dtype):
        # 获取指定 dtype 的每个元素的字节大小
        bytes = get_dtype_size(dtype)
        # 计算输入数组的总字节数
        in_bytes = SIZE * bytes
        
        # 测试空数组情况
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length \(0\) and count"):
            # 创建空的 numpy 数组
            empty = np.array([])
            # 尝试使用空数组创建 torch 张量，期望抛出特定的 ValueError 异常
            torch.frombuffer(empty, dtype=dtype)
        
        # 测试 count 参数为 0 的情况
        with self.assertRaisesRegex(ValueError,
                                    r"both buffer length .* and count \(0\)"):
            # 调用测试函数，并传入指定的 SHAPE、dtype 和 count 参数值 0
            self._run_test(SHAPE, dtype, count=0)
        
        # 测试 offset 参数为负数且大于总长度的情况
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \(-{bytes} bytes\) must be"):
            # 调用测试函数，并传入指定的 SHAPE、dtype 和 first 参数值 -1
            self._run_test(SHAPE, dtype, first=-1)
        
        with self.assertRaisesRegex(ValueError,
                                    rf"offset \({in_bytes} bytes\) must be .* "
                                    rf"buffer length \({in_bytes} bytes\)"):
            # 调用测试函数，并传入指定的 SHAPE、dtype 和 first 参数值 SIZE
            self._run_test(SHAPE, dtype, first=SIZE)
        
        # 非倍数偏移量与所有元素的情况
        if bytes > 1:
            offset = bytes - 1
            with self.assertRaisesRegex(ValueError,
                                        rf"buffer length \({in_bytes - offset} bytes\) after "
                                        rf"offset \({offset} bytes\) must be"):
                # 调用测试函数，并传入指定的 SHAPE、dtype 和 offset 参数值 bytes - 1
                self._run_test(SHAPE, dtype, offset=bytes - 1)
        
        # count 参数对于每个良好的 first 元素太大的情况
        for first in range(SIZE):
            count = SIZE - first + 1
            with self.assertRaisesRegex(ValueError,
                                        rf"requested buffer length \({count} \* {bytes} bytes\) "
                                        rf"after offset \({first * bytes} bytes\) must .*"
                                        rf"buffer length \({in_bytes} bytes\)"):
                # 调用测试函数，并传入指定的 SHAPE、dtype、count 和 first 参数值 count, first
                self._run_test(SHAPE, dtype, count=count, first=first)
    # 在给定设备和数据类型下创建一个大小为 (1,) 的张量 x
    x = make_tensor((1,), dtype=dtype, device=device)
    # 调用 _run_test 方法，返回数组 arr 和张量 tensor
    arr, tensor = self._run_test(SHAPE, dtype)
    # 将 tensor 的所有元素替换为 x 的值
    tensor[:] = x
    # 断言数组 arr 等于 tensor
    self.assertEqual(arr, tensor)
    # 断言张量 tensor 的所有元素都等于 x 的所有元素
    self.assertTrue((tensor == x).all().item())

    # 根据给定的 count 值修改所有有效偏移量的整个张量
    for count in range(-1, SIZE + 1):
        if count == 0:
            continue

        # 计算实际有效的 count 值
        actual_count = count if count > 0 else SIZE
        # 遍历所有可能的起始偏移量 first
        for first in range(SIZE - actual_count):
            last = first + actual_count
            # 调用 _run_test 方法，返回数组 arr 和张量 tensor
            arr, tensor = self._run_test(SHAPE, dtype, first=first, count=count)
            # 将 tensor 的所有元素替换为 x 的值
            tensor[:] = x
            # 断言数组 arr 在范围 [first:last] 内与 tensor 相等
            self.assertEqual(arr[first:last], tensor)
            # 断言张量 tensor 的所有元素都等于 x 的所有元素
            self.assertTrue((tensor == x).all().item())

            # 修改数组 arr 中第一个值
            arr[first] = x.item() - 1
            # 断言数组 arr 在范围 [first:last] 内与 tensor 相等
            self.assertEqual(arr[first:last], tensor)

@dtypes(*set(numpy_to_torch_dtype_dict.values()))
def test_not_a_buffer(self, device, dtype):
    # 断言从非缓冲区创建张量时会引发 ValueError 异常，并且异常信息包含特定字符串
    with self.assertRaisesRegex(ValueError,
                                r"object does not implement Python buffer protocol."):
        torch.frombuffer([1, 2, 3, 4], dtype=dtype)

@dtypes(*set(numpy_to_torch_dtype_dict.values()))
def test_non_writable_buffer(self, device, dtype):
    # 创建一个 numpy 数组，并将其转换为字节序列
    numpy_arr = make_tensor((1,), dtype=dtype, device=device).numpy()
    byte_arr = numpy_arr.tobytes()
    # 断言从不可写缓冲区创建张量时会引发 UserWarning 警告，并且警告信息包含特定字符串
    with self.assertWarnsOnceRegex(UserWarning,
                                   r"The given buffer is not writable."):
        torch.frombuffer(byte_arr, dtype=dtype)

def test_byte_to_int(self):
    # 根据系统的字节序创建一个 np.byte 类型的 numpy 数组
    byte_array = np.array([-1, 0, 0, 0, -1, 0, 0, 0], dtype=np.byte) if sys.byteorder == 'little' \
        else np.array([0, 0, 0, -1, 0, 0, 0, -1], dtype=np.byte)
    # 将 byte_array 转换为 torch.int32 类型的张量 tensor
    tensor = torch.frombuffer(byte_array, dtype=torch.int32)
    # 断言张量 tensor 的元素个数为 2
    self.assertEqual(tensor.numel(), 2)
    # 断言张量 tensor 的值序列与给定序列 [255, 255] 相等
    self.assertSequenceEqual(tensor, [255, 255])
# Tests for the `asarray` function:
#   Constructs tensors from a Python object that has one of the following
#   characteristics:
#       1. is a Tensor
#       2. is a DLPack capsule
#       3. implements the Python Buffer protocol
#       4. is an arbitrary list
#   The implementation itself is based on the Python Array API:
#   https://data-apis.org/array-api/latest/API_specification/creation_functions.html
class TestAsArray(TestCase):
    # Method to perform checks on the 'asarray' function
    def _check(self, original, cvt=lambda t: t, is_alias=True, same_dtype=True, same_device=True, **kwargs):
        """Check the output of 'asarray', given its input and assertion information.

        Besides calling 'asarray' itself, this function does 4 different checks:
            1. Whether the result is aliased or not, depending on 'is_alias'
            2. Whether the result has the expected dtype and elements
            3. Whether the result lives in the expected device
            4. Whether the result has its 'requires_grad' set or not
        """
        # Convert 'original' to a tensor using 'cvt' function if provided,
        # then call 'asarray' with additional keyword arguments from 'kwargs'
        result = torch.asarray(cvt(original), **kwargs)
        # Assert that 'result' is indeed a torch.Tensor object
        self.assertTrue(isinstance(result, torch.Tensor))

        # 1. Check if the storage pointers are equal only if 'is_alias' is True
        if is_alias:
            self.assertEqual(result.data_ptr(), original.data_ptr())
        else:
            self.assertNotEqual(result.data_ptr(), original.data_ptr())

        # 2. Perform element-wise comparison if 'same_dtype' is True,
        #    otherwise compare shape and dtype
        if same_dtype:
            self.assertEqual(original, result)
        else:
            # Get dtype from 'kwargs' or default dtype and assert shape and dtype
            dtype = kwargs.get("dtype", torch.get_default_dtype())
            self.assertEqual(original.shape, result.shape)
            self.assertEqual(dtype, result.dtype)

        # 3. Determine the expected device ('device') based on 'kwargs' or 'original',
        #    then compare 'result.device' with 'device'
        if same_device:
            device = original.device
        else:
            device = torch.device(kwargs.get("device", "cpu"))

        # Compare 'device.type' and 'device.index' with 'result.device.type' and 'result.device.index'
        self.assertEqual(device.type, result.device.type)
        if device.index is not None:
            self.assertEqual(device.index, result.device.index)

        # 4. Check if 'requires_grad' attribute of 'result' matches 'kwargs.get("requires_grad", False)'
        self.assertEqual(result.requires_grad, kwargs.get("requires_grad", False))
    # 定义一个测试函数，用于测试带有类型转换器的别名处理
    def _test_alias_with_cvt(self, cvt, device, dtype, shape=(5, 5), only_with_dtype=False):
        # 创建原始张量
        original = make_tensor(shape, dtype=dtype, device=device)

        # 定义内部函数，用于检查别名处理后的结果
        def check(**kwargs):
            self._check(original, cvt=cvt, **kwargs)

        # 如果不仅仅测试特定数据类型，执行以下检查
        if not only_with_dtype:
            check(copy=False)  # 检查不复制时的别名处理
            check(device=device)  # 检查在指定设备上的别名处理
            check(device=device, copy=False)  # 检查在指定设备上且不复制时的别名处理

        check(dtype=dtype)  # 检查指定数据类型的别名处理
        check(dtype=dtype, copy=False)  # 检查指定数据类型且不复制时的别名处理
        check(requires_grad=False, dtype=dtype)  # 检查指定数据类型且不要求梯度时的别名处理
        check(requires_grad=may_require_grad(dtype), dtype=dtype)  # 检查可能需要梯度的数据类型的别名处理
        check(device=device, dtype=dtype)  # 检查在指定设备上和指定数据类型的别名处理
        check(device=device, dtype=dtype, copy=False)  # 检查在指定设备上、指定数据类型且不复制时的别名处理

    # 跳过 'meta' 设备，因为比较它们的数据指针（这基本上是重点）没有意义，
    # 因为它们都返回 0。
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_alias_from_tensor(self, device, dtype):
        self._test_alias_with_cvt(identity, device, dtype)

    # 仅在 CPU 上运行的测试用例，测试从 numpy 转换的别名处理
    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_alias_from_numpy(self, device, dtype):
        self._test_alias_with_cvt(to_numpy, device, dtype)

    # 跳过 'meta'，因为 'to_dlpack' 在这些设备上不起作用。
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_alias_from_dlpack(self, device, dtype):
        self._test_alias_with_cvt(to_dlpack, device, dtype)

    # 仅在 CPU 上运行的测试用例，测试从缓冲区转换的别名处理
    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_alias_from_buffer(self, device, dtype):
        self._test_alias_with_cvt(to_memview, device, dtype, shape=(5,), only_with_dtype=True)

    # 定义一个测试函数，用于测试带有类型转换器的复制处理
    def _test_copy_with_cvt(self, cvt, device, dtype, shape=(5, 5), only_with_dtype=False):
        # 创建原始张量
        original = make_tensor(shape, dtype=dtype, device=device)

        # 定义内部函数，用于检查复制处理后的结果
        def check(**kwargs):
            self._check(original, cvt=cvt, is_alias=False, **kwargs)

        # 如果不仅仅测试特定数据类型，执行以下检查
        if not only_with_dtype:
            check(copy=True)  # 检查复制时的别名处理
            check(device=device, copy=True)  # 检查在指定设备上且复制时的别名处理

        check(requires_grad=False, dtype=dtype, copy=True)  # 检查指定数据类型且不要求梯度时的复制处理
        check(requires_grad=may_require_grad(dtype), dtype=dtype, copy=True)  # 检查可能需要梯度的数据类型的复制处理
        check(dtype=dtype, copy=True)  # 检查指定数据类型的复制处理
        check(device=device, dtype=dtype, copy=True)  # 检查在指定设备上和指定数据类型的复制处理

        # 如果 CUDA 可用，因为设备不同，强制复制
        if torch.cuda.is_available():
            other = get_another_device(device)
            check(same_device=False, device=other, dtype=dtype)  # 检查在不同设备上的数据类型处理
            check(same_device=False, device=other, dtype=dtype, copy=True)  # 检查在不同设备上且复制时的数据类型处理

        # 如果不仅仅测试特定数据类型，因为数据类型不同，强制复制
        if not only_with_dtype:
            for other in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
                if dtype != other:
                    check(same_dtype=False, dtype=other)  # 检查不同数据类型的处理
                    check(same_dtype=False, dtype=other, copy=True)  # 检查不同数据类型且复制时的处理
    # 定义一个测试函数，用于测试在给定设备和数据类型下执行张量复制操作
    def test_copy_tensor(self, device, dtype):
        # 调用内部方法，使用给定的转换函数进行张量复制测试
        self._test_copy_with_cvt(identity, device, dtype)

    # 标记为仅在CPU上执行的测试，使用numpy转换函数进行测试张量复制
    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_copy_from_numpy(self, device, dtype):
        # 调用内部方法，使用给定的转换函数进行张量复制测试
        self._test_copy_with_cvt(to_numpy, device, dtype)

    # 跳过某些元数据的测试，使用DLPack转换函数进行测试张量复制
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_copy_from_dlpack(self, device, dtype):
        # 调用内部方法，使用给定的转换函数进行张量复制测试
        self._test_copy_with_cvt(to_dlpack, device, dtype)

    # 标记为仅在CPU上执行的测试，使用内存视图转换函数进行测试张量复制
    @onlyCPU
    @dtypes(*set(numpy_to_torch_dtype_dict.values()))
    def test_copy_from_buffer(self, device, dtype):
        # 调用内部方法，使用给定的转换函数进行张量复制测试，设置张量形状为(5,)
        self._test_copy_with_cvt(to_memview, device, dtype, shape=(5,), only_with_dtype=True)

    # 定义一个测试函数，用于在多个设备上测试张量复制操作
    def _test_copy_mult_devices(self, devices, dtype, cvt):
        # 分配两个CUDA设备
        cuda1 = devices[0]
        cuda2 = devices[1]
        # 创建原始张量，指定数据类型和第一个CUDA设备
        original = make_tensor((5, 5), dtype=dtype, device=cuda1)

        # 定义检查函数，用于验证张量复制操作的不同参数组合
        def check(**kwargs):
            self._check(original, cvt, is_alias=False, same_device=False, device=cuda2, **kwargs)

        # 进行不同参数组合的验证检查
        check()
        check(copy=True)
        check(dtype=dtype, copy=True)

    # 标记为仅在CUDA上执行的测试，至少需要两个设备，使用identity转换函数进行测试张量复制
    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_copy_from_tensor_mult_devices(self, devices, dtype):
        # 调用内部方法，使用给定的设备和数据类型在多设备上进行张量复制测试
        self._test_copy_mult_devices(devices, dtype, identity)

    # 标记为仅在CUDA上执行的测试，至少需要两个设备，使用to_dlpack转换函数进行测试张量复制
    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_copy_from_dlpack_mult_devices(self, devices, dtype):
        # 调用内部方法，使用给定的设备和数据类型在多设备上进行张量复制测试
        self._test_copy_mult_devices(devices, dtype, to_dlpack)

    # 根据指定的数据类型进行张量列表复制测试
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_copy_list(self, device, dtype):
        # 创建原始张量，指定形状为(5, 5)，数据类型为指定的dtype，设备为CPU
        original = make_tensor((5, 5), dtype=dtype, device=torch.device("cpu"))

        # 定义检查函数，用于验证张量列表复制操作的不同参数组合
        def check(**kwargs):
            # 调用内部方法，验证给定的转换函数对张量列表进行复制，设置为非别名操作
            self._check(original, torch.Tensor.tolist, is_alias=False, **kwargs)

        # 判断设备是否与CPU相同
        same_device = torch.device("cpu") == device
        # 进行不同参数组合的验证检查
        check(same_device=same_device, device=device, dtype=dtype)
        check(same_device=same_device, device=device, dtype=dtype, requires_grad=False)
        check(same_device=same_device, device=device, dtype=dtype, requires_grad=may_require_grad(dtype))
        check(same_device=same_device, device=device, dtype=dtype, copy=True)

    # 根据指定的数据类型为torch.float32进行测试
    @dtypes(torch.float32)
    def test_unsupported_alias(self, device, dtype):
        # 创建一个指定设备和数据类型的张量
        original = make_tensor((5, 5), dtype=dtype, device=device)

        # 如果CUDA可用，测试从当前设备到另一个设备的转换是否会抛出特定异常
        if torch.cuda.is_available():
            other_device = get_another_device(device)
            with self.assertRaisesRegex(ValueError,
                                        f"from device '{device}' to '{other_device}'"):
                torch.asarray(original, device=other_device, copy=False)

        # 测试在不同数据类型间进行转换时是否会抛出特定异常
        with self.assertRaisesRegex(ValueError,
                                    "with dtype '.*' into dtype '.*'"):
            torch.asarray(original, dtype=torch.float64, copy=False)

        # 测试不能将任意序列直接转换为张量时是否会抛出特定异常
        with self.assertRaisesRegex(ValueError,
                                    "can't alias arbitrary sequence"):
            torch.asarray(original.tolist(), copy=False)

    @onlyCUDA
    @deviceCountAtLeast(2)
    @dtypes(torch.float32)
    def test_unsupported_alias_mult_devices(self, devices, dtype):
        # 获取前两个设备
        dev1, dev2 = devices[:2]
        # 创建一个指定设备和数据类型的张量
        original = make_tensor((5, 5), dtype=dtype, device=dev1)
        # 测试从一个设备到另一个设备的转换是否会抛出特定异常
        with self.assertRaisesRegex(ValueError,
                                    f"from device '{dev1}' to '{dev2}'"):
            torch.asarray(original, device=dev2, copy=False)

    @dtypes(torch.float32, torch.complex64)
    def test_retain_autograd_history(self, device, dtype):
        # 创建一个指定设备和数据类型的张量，并要求保留自动求导历史
        original = make_tensor((5, 5), dtype=dtype, device=device, requires_grad=True)
        # 克隆张量，保留克隆操作的求导历史信息
        cloned = original.clone()

        def check(**kwargs):
            # 使用给定的参数重新创建张量
            a = torch.asarray(cloned, **kwargs)
            requires_grad = kwargs.get("requires_grad", False)
            # 断言新创建的张量的requires_grad属性与预期一致
            self.assertEqual(a.requires_grad, requires_grad)
            # 当requires_grad为False时，断言自动求导历史未被保留
            self.assertEqual(a.grad_fn is None, not requires_grad)

        # 检查不同参数下的张量属性
        check()
        check(requires_grad=True)
        check(copy=True)
        check(requires_grad=True, copy=True)
        check(requires_grad=False)
        check(requires_grad=False, copy=True)

    @onlyCPU
    def test_astensor_consistency(self, device):
        # 参考问题：https://github.com/pytorch/pytorch/pull/71757

        examples = [
            # 标量
            True,
            42,
            1.0,
            # 同类列表
            [True, True, False],
            [1, 2, 3, 42],
            [0.0, 1.0, 2.0, 3.0],
            # 混合列表
            [True, False, 0],
            [0.0, True, False],
            [0, 1.0, 42],
            [0.0, True, False, 42],
            # 包含复数的列表
            [0.0, True, False, 42, 5j],
            # 包含范围的对象
            range(5),
        ]

        # 对于每个示例，测试将其转换为张量后是否保持一致性
        for e in examples:
            original = torch.as_tensor(e)
            t = torch.asarray(e)
            self.assertEqual(t, original)

    @skipIfTorchDynamo()
    @onlyCPU
    # 测试处理 NumPy 标量的函数
    def test_numpy_scalars(self, device):
        # 创建一个 NumPy 浮点数标量
        scalar = np.float64(0.5)

        # 使用断言检查在不复制情况下将 NumPy 标量转换为 PyTorch 张量时是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "can't alias NumPy scalars."):
            torch.asarray(scalar, copy=False)

        # 将 NumPy 标量转换为 PyTorch 张量
        tensor = torch.asarray(scalar)
        # 检查张量的维度是否为0
        self.assertEqual(tensor.dim(), 0)
        # 检查张量的值是否与原始标量的值相等
        self.assertEqual(tensor.item(), scalar.item())
        # 检查张量的数据类型是否为 torch.float64
        self.assertEqual(tensor.dtype, torch.float64)

        # 回归测试：检查针对 https://github.com/pytorch/pytorch/issues/97021 的问题
        zerodim_arr = np.array(1.)
        # 将 NumPy 零维数组转换为指定数据类型的 PyTorch 张量
        tensor = torch.asarray(zerodim_arr, dtype=torch.int32)
        # 检查张量的维度是否为0
        self.assertEqual(tensor.dim(), 0)
        # 检查张量的值是否与原始数组的值相等
        self.assertEqual(tensor.item(), zerodim_arr.item())
        # 检查张量的数据类型是否为 torch.int32

    # 测试默认设备下的函数
    def test_default_device(self, device):
        # 创建原始张量
        original = torch.arange(5)

        # 定义示例数据及其关键字参数
        examples: List[Tuple[Any, Dict]] = [
            (3, {}),
            (original, {}),
            (to_numpy(original), {}),
            (to_memview(original), {"dtype": original.dtype}),
        ]

        # 遍历示例数据及其关键字参数
        for data, kwargs in examples:
            # 使用指定设备上下文管理器
            with torch.device(device):
                # 将数据转换为 PyTorch 张量
                tensor = torch.asarray(data, **kwargs)
                # 检查张量所在设备是否与指定设备相同
                self.assertEqual(tensor.device, torch.device(device))

                # 检查张量内容是否与原始数据一致
                if isinstance(data, int):
                    self.assertEqual(data, tensor.item())
                else:
                    self.assertEqual(data, tensor)

    # 仅在 CUDA 设备上运行的测试函数
    @onlyCUDA
    def test_device_without_index(self, device):
        # 在 CUDA 设备上创建原始张量
        original = torch.arange(5, device="cuda")

        # 将原始张量转换为指定 CUDA 设备上的 PyTorch 张量
        tensor = torch.asarray(original, device="cuda")
        # 检查原始张量和新张量的存储指针是否相等
        self.assertEqual(original.data_ptr(), tensor.data_ptr())

        # 将原始张量复制到指定 CUDA 设备上的 PyTorch 张量
        tensor = torch.asarray(original, copy=True, device="cuda")
        # 检查原始张量和新张量的存储指针是否不相等
        self.assertNotEqual(original.data_ptr(), tensor.data_ptr())
# 对 TestTensorCreation 类进行设备类型测试实例化，使用全局变量
instantiate_device_type_tests(TestTensorCreation, globals())

# 对 TestRandomTensorCreation 类进行设备类型测试实例化，使用全局变量
instantiate_device_type_tests(TestRandomTensorCreation, globals())

# 对 TestLikeTensorCreation 类进行设备类型测试实例化，使用全局变量
instantiate_device_type_tests(TestLikeTensorCreation, globals())

# 对 TestBufferProtocol 类进行设备类型测试实例化，仅针对 CPU，使用全局变量
instantiate_device_type_tests(TestBufferProtocol, globals(), only_for="cpu")

# 对 TestAsArray 类进行设备类型测试实例化，使用全局变量
instantiate_device_type_tests(TestAsArray, globals())

# 如果当前脚本作为主程序运行，则启用 TestCase 类的默认数据类型检查
if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    # 运行所有测试
    run_tests()
```