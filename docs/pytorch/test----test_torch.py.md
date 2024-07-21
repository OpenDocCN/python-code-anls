# `.\pytorch\test\test_torch.py`

```py
# mypy: allow-untyped-defs
# Owner(s): ["module: tests"]

# 导入所需的库和模块
import torch  # 导入PyTorch库
import torch.utils.data  # 导入PyTorch的数据工具模块
import numpy as np  # 导入NumPy库

import contextlib  # 提供上下文管理工具的模块
import gc  # Python的垃圾回收模块
import io  # 提供核心的Python I/O功能的模块
import inspect  # 提供检查活动对象的工具模块
import itertools  # 提供用于迭代的工具模块
import math  # 提供数学函数的模块
import random  # 提供生成伪随机数的模块
import re  # 提供正则表达式操作的模块
import copy  # 提供深浅复制操作的模块
import os  # 提供与操作系统交互的功能的模块
import tempfile  # 提供临时文件和目录的功能的模块
import unittest  # 提供单元测试框架的模块
import warnings  # 提供警告控制的模块
import types  # 提供Python类型和类的操作的模块
import pickle  # 提供序列化和反序列化Python对象的功能的模块
import textwrap  # 提供文本缩进和填充的功能的模块
import subprocess  # 提供生成子进程的功能的模块
import weakref  # 提供弱引用对象的模块
import sys  # 提供与Python解释器交互的变量和函数的功能的模块
import copyreg  # 提供注册可序列化对象处理程序的功能的模块
from torch import inf, nan  # 从torch模块中导入特定的常量

from itertools import product, combinations, permutations, chain  # 从itertools模块导入多个迭代工具函数
from functools import partial  # 提供创建偏函数的工具函数
from torch import multiprocessing as mp  # 导入PyTorch的多进程模块
from torch.testing import make_tensor  # 从PyTorch测试模块导入make_tensor函数
from torch.testing._internal.common_optimizers import (  # 导入PyTorch内部优化器相关模块和函数
    optim_db, optims, _get_optim_inputs_including_global_cliquey_kwargs)

from torch.testing._internal.common_utils import (  # 导入PyTorch内部通用工具函数
    TEST_WITH_TORCHINDUCTOR, TEST_WITH_ROCM, run_tests, IS_JETSON,
    IS_WINDOWS, IS_FILESYSTEM_UTF8_ENCODING, NO_MULTIPROCESSING_SPAWN,
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, skipIfTorchInductor, load_tests, slowTest, slowTestIf,
    TEST_WITH_CROSSREF, skipIfTorchDynamo, skipRocmIfTorchInductor, set_default_dtype,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, CudaSyncGuard,
    bytes_to_scalar, parametrize, skipIfMps, noncontiguous_like,
    AlwaysWarnTypedStorageRemoval, TEST_WITH_TORCHDYNAMO, xfailIfTorchDynamo)
from multiprocessing.reduction import ForkingPickler  # 导入多进程中用于对象传递的模块
from torch.testing._internal.common_device_type import (  # 导入PyTorch内部设备类型相关模块
    expectedFailureMeta,
    expectedFailureXLA,
    instantiate_device_type_tests,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta, PYTORCH_CUDA_MEMCHECK, largeTensorTest, onlyNativeDeviceTypes,
    get_all_device_types, skipXLA)
from typing import Tuple  # 导入Python的类型提示模块
import torch.backends.quantized  # 导入PyTorch量化后端相关模块
import torch.testing._internal.data  # 导入PyTorch内部测试数据模块
from torch.testing._internal.common_cuda import (  # 导入PyTorch CUDA相关模块
    tf32_on_and_off, tf32_is_not_fp32, TEST_CUDNN, TEST_MULTIGPU,
    _create_scaling_case, _create_scaling_models_optimizers)
from torch.testing._internal.common_mkldnn import bf32_on_and_off  # 导入PyTorch MKLDNN相关模块
from torch.testing._internal.common_dtype import (  # 导入PyTorch数据类型相关模块
    floating_types_and, get_all_math_dtypes, all_types_and_complex_and, complex_types,
    all_types_and, floating_types, floating_and_complex_types, integral_types_and,
    get_all_qint_dtypes,
)
from torch.testing._internal.two_tensor import TwoTensor  # 导入PyTorch两个张量比较相关模块

if TEST_WITH_TORCHINDUCTOR:
    from torch._inductor.test_case import TestCase  # 如果在Torch Inductor中测试，则导入Inductor的测试用例
else:
    from torch.testing._internal.common_utils import TestCase  # 否则导入PyTorch内部通用测试用例

# 防止导入设置默认数据类型
assert torch.get_default_dtype() is torch.float32

# load_tests函数用于在沙堡上分片测试，此行代码禁止flake警告
load_tests = load_tests
# 根据测试 ROCm 是否可用或 tf32 是否不等于 fp32，确定是否使用 AMPERE_OR_ROCM
AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# 上下文管理器，用于设置 TORCH_VITAL 环境变量的值
@contextlib.contextmanager
def torch_vital_set(value):
    # 存储当前 TORCH_VITAL 的值，如果存在的话
    stash = None
    if 'TORCH_VITAL' in os.environ:
        stash = os.environ['TORCH_VITAL']
    # 设置 TORCH_VITAL 环境变量的值为给定的 value
    os.environ['TORCH_VITAL'] = value
    try:
        yield  # 执行 with 语句块中的代码
    finally:
        if stash:
            os.environ['TORCH_VITAL'] = stash  # 恢复 TORCH_VITAL 的旧值
        else:
            del os.environ['TORCH_VITAL']  # 删除 TORCH_VITAL 环境变量

# 测试 Torch 的关键状态
# FIXME: 需要记录或废弃这段代码
class TestBasicVitalSigns(TestCase):
    def test_basic_vitals(self):
        with torch_vital_set(''):
            self.assertFalse(torch.vitals_enabled())  # 断言 Torch 的 vital 状态为 False
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())  # 断言 Torch 的 vital 状态为 True

    def test_basic_vitals_read_write(self):
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())  # 断言 Torch 的 vital 状态为 True
            # 测试设置 vital 的代码路径
            self.assertTrue(torch.set_vital('Dataloader', 'basic_unit_test', 'TEST_VALUE_STRING'))
            self.assertIn('TEST_VALUE_STRING', torch.read_vitals())  # 断言 vital 中包含 'TEST_VALUE_STRING'
            self.assertIn('CUDA.used', torch.read_vitals())  # 断言 vital 中包含 'CUDA.used'

    def test_dataloader_vitals(self):
        with torch_vital_set('ON'):
            inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            dataset = torch.utils.data.TensorDataset(inps, tgts)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            self.assertIn('Dataloader.enabled\t\t True', torch.read_vitals())  # 断言 vital 中包含 'Dataloader.enabled\t\t True'

# FIXME: 需要记录或废弃这段代码
class TestVitalSignsCuda(TestCase):
    @onlyCUDA  # 仅在 CUDA 可用时运行该测试
    def test_cuda_vitals_gpu_only(self, device):
        with torch_vital_set('ON'):
            self.assertIn('CUDA.used\t\t true', torch.read_vitals())  # 断言 vital 中包含 'CUDA.used\t\t true'


# 检查是否 CUDA 计算能力符合 8.6，并且 CUDA 可用
is_cuda_sm86 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 6)

# 测试 Torch 设备类型
class TestTorchDeviceType(TestCase):
    exact_dtype = True

    # TODO: 将所有张量创建移动到通用操作中
    def _rand_shape(self, dim, min_size, max_size):
        shape = []
        for i in range(dim):
            shape.append(random.randint(min_size, max_size))
        return tuple(shape)

    # 验证数学常量是否按照 Python 数组 API 的要求定义
    @onlyCPU  # 仅在 CPU 上运行该测试
    def test_constants(self, device):
        self.assertIsInstance(torch.e, float)  # 断言 torch.e 是 float 类型
        self.assertEqual(torch.e, math.e, atol=0, rtol=0)  # 断言 torch.e 等于 math.e，精度为绝对误差 0，相对误差 0

        self.assertIsInstance(torch.pi, float)  # 断言 torch.pi 是 float 类型
        self.assertEqual(torch.pi, math.pi, atol=0, rtol=0)  # 断言 torch.pi 等于 math.pi，精度为绝对误差 0，相对误差 0

        self.assertIsInstance(torch.nan, float)  # 断言 torch.nan 是 float 类型
        self.assertEqual(torch.nan, math.nan, equal_nan=True)  # 断言 torch.nan 等于 math.nan，允许 NaN 相等

        self.assertIsInstance(torch.inf, float)  # 断言 torch.inf 是 float 类型
        self.assertEqual(torch.inf, math.inf)  # 断言 torch.inf 等于 math.inf

    @onlyNativeDeviceTypes  # 仅在本地设备类型上运行该测试
    # 使用装饰器 @dtypes 指定多种数据类型，测试字节转标量的函数
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    # 定义测试函数 test_bytes_to_scalar，接受设备和数据类型作为参数
    def test_bytes_to_scalar(self, device, dtype):
        # 内部函数 rand_byte 生成随机字节，根据数据类型生成不同范围的值
        def rand_byte():
            if dtype == torch.bool:
                return torch.randint(0, 2, ()).item()
            else:
                return torch.randint(0, 256, ()).item()

        # 获取数据类型的元素大小
        element_size = torch._utils._element_size(dtype)

        # 迭代测试 10 次
        for i in range(10):
            # 生成指定大小的随机字节列表
            bytes_list = [rand_byte() for _ in range(element_size)]
            # 使用 bytes_to_scalar 函数将字节列表转换为标量
            scalar = bytes_to_scalar(bytes_list, dtype, device)
            # 断言转换后的标量与原始字节列表存储的值相等
            self.assertEqual(scalar.storage().untyped().tolist(), bytes_list)

    # 使用装饰器 @dtypes 指定多种数据类型，测试存储器相关操作
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    # 定义测试函数 test_storage，接受设备和数据类型作为参数
    def test_storage(self, device, dtype):
        # 使用 make_tensor 函数生成指定形状和数据类型的张量 v
        v = make_tensor((3, 5), dtype=dtype, device=device, low=-9, high=9)
        # 断言存储器的第一个元素与张量的第一个元素相等
        self.assertEqual(v.storage()[0], v[0][0])
        # 断言存储器的第 14 个元素与张量的最后一个元素相等
        self.assertEqual(v.storage()[14], v[2][4])
        # 获取张量的存储器对象
        v_s = v.storage()

        # 遍历张量的每个元素编号
        for el_num in range(v.numel()):
            # 计算当前元素在张量中的二维索引
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            # 断言存储器中的元素与张量中对应位置的元素相等
            self.assertEqual(
                v_s[el_num],
                v[dim0][dim1])

        # 获取张量存储器的字节流视图
        v_s_byte = v.storage().untyped()
        # 获取张量每个元素的字节大小
        el_size = v.element_size()

        # 遍历张量的每个元素编号
        for el_num in range(v.numel()):
            # 计算当前元素在字节流中的起始和结束位置
            start = el_num * el_size
            end = start + el_size
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            # 断言从字节流中读取的标量与张量中对应位置的元素相等
            self.assertEqual(
                bytes_to_scalar(v_s_byte[start:end], dtype, device),
                v[dim0][dim1])

    # 使用装饰器 @onlyNativeDeviceTypes 和 @dtypes 指定多种数据类型，测试存储器设置项操作
    @onlyNativeDeviceTypes
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.quint8, torch.qint8, torch.qint32,
            torch.quint4x2)
    # 定义测试函数 test_storage_setitem，接受设备和数据类型作为参数
    def test_storage_setitem(self, device, dtype):
        # 跳过 CUDA 下不支持的量化数据类型
        if torch.device(device).type == 'cuda':
            if dtype in [torch.quint8, torch.qint8, torch.qint32, torch.quint4x2]:
                return

        # 获取数据类型对应的存储器类型名称
        storage_type_name = torch.storage._dtype_to_storage_type_map()[dtype]
        # 根据设备类型选择存储器类型
        if torch.device(device).type == 'cuda':
            storage_type = eval('torch.cuda.' + storage_type_name)
        else:
            storage_type = eval('torch.' + storage_type_name)

        N = 10

        # 创建指定类型和大小的存储器对象 s，并初始化为 0
        s = storage_type(N)
        s[:] = 0
        l = [0] * N
        # 断言存储器对象 s 与列表 l 相等
        self.assertEqual(s, storage_type(l))

        # 遍历存储器对象 s 的索引
        for i in range(N):
            s[i] = i
            l[i] = i

        # 断言存储器对象 s 与更新后的列表 l 相等
        self.assertEqual(s, storage_type(l))

        # 更新列表 l 的部分元素为 1
        l[2:7] = [1] * 5
        # 使用切片操作更新存储器对象 s 的部分元素为 1
        s[2:7] = 1
        # 断言存储器对象 s 与更新后的列表 l 相等
        self.assertEqual(s, storage_type(l))
    # 如果条件满足，则跳过测试；条件为: https://github.com/pytorch/torchdynamo/issues/1991
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    # 仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 使用指定的数据类型进行测试，包括所有类型以及 torch.half, torch.bool, torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_tensor_storage_type(self, device, dtype):
        # 创建指定设备和数据类型的张量 a，形状为 (10,)
        a = make_tensor((10,), dtype=dtype, device=device, low=-9, high=9)
    
        # 根据设备类型选择 module 为 torch.cuda 或 torch
        module = torch.cuda if (torch.device(device).type == 'cuda') else torch
        # 获取预期的存储类型
        expected_storage_type = getattr(module, torch.storage._dtype_to_storage_type_map()[dtype])
    
        # 断言张量 a 的存储类型与预期的存储类型相等
        self.assertEqual(a.storage_type(), expected_storage_type)
    
    # 仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 使用指定的数据类型进行测试，包括所有类型以及 torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    def test_tensor_from_storage(self, device, dtype):
        # 创建指定设备和数据类型的张量 a，形状为 (4, 5, 3)
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        # 获取张量 a 的存储
        a_s = a.storage()
        # 从存储创建张量 b，并重新形状为 a 的形状，与 a 断言相等
        b = torch.tensor(a_s, device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, b)
        # 从未指定类型的存储创建张量 c，并重新形状为 a 的形状，与 a 断言相等
        c = torch.tensor(a_s.untyped(), device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, c)
    
        # 对于所有类型及 torch.half, torch.bool, torch.bfloat16 的错误数据类型
        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            # 如果错误数据类型与当前数据类型相同，则继续下一个循环
            if error_dtype == dtype:
                continue
            # 断言抛出 RuntimeError，并检查其消息包含 'Expected a Storage of type'
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                # 将 a 转换为错误数据类型的存储
                error_storage = a.to(error_dtype).storage()
                torch.tensor(error_storage, device=device, dtype=dtype)
    
    # 仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 使用指定的数据类型进行测试，包括所有类型以及 torch.half, torch.bool, torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_set_storage(self, device, dtype):
        # 创建指定设备和数据类型的张量 a，形状为 (4, 5, 3)
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        # 获取张量 a 的存储
        a_s = a.storage()
        # 使用 a_s 设置空张量 b 的存储，并重新形状为 a 的形状，与 a 断言相等
        b = torch.tensor([], device=device, dtype=dtype).set_(a_s).reshape(a.size())
        self.assertEqual(a, b)
        # 使用未指定类型的 a_s 设置空张量 c 的存储，并重新形状为 a 的形状，与 a 断言相等
        c = torch.tensor([], device=device, dtype=dtype).set_(a_s.untyped()).reshape(a.size())
        self.assertEqual(a, c)
    
        # 对于所有类型及 torch.half, torch.bool, torch.bfloat16 的错误数据类型
        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            # 如果错误数据类型与当前数据类型相同，则继续下一个循环
            if error_dtype == dtype:
                continue
            # 断言抛出 RuntimeError，并检查其消息包含 'Expected a Storage of type'
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                # 将 a 转换为错误数据类型的存储
                error_storage = a.to(error_dtype).storage()
                # 使用 error_storage 设置空张量 b 的存储
                b = torch.tensor([], device=device, dtype=dtype).set_(error_storage)
    # 检查存储元数据的私有方法，验证传入的存储对象和其检查对象的类型是否为 UntypedStorage 或 TypedStorage
    def _check_storage_meta(self, s, s_check):
        # 使用 assertTrue 断言，验证 s 和 s_check 是否为 UntypedStorage 或 TypedStorage 类型
        self.assertTrue(
            isinstance(s, (torch.UntypedStorage, torch.TypedStorage)) and
            isinstance(s_check, type(s)),
            (
                's and s_check must both be one of UntypedStorage or '
                'TypedStorage, but got'
                f' {type(s).__name__} and {type(s_check).__name__}'))

        # 验证 s 的设备类型是否为 'meta'
        self.assertEqual(s.device.type, 'meta')
        # 验证 s 和 s_check 的存储空间大小是否相等
        self.assertEqual(s.nbytes(), s_check.nbytes())
        # 验证 s 和 s_check 的大小（元素数）是否相等
        self.assertEqual(s.size(), s_check.size())
        # 验证 s 的数据指针是否为 0
        self.assertEqual(s.data_ptr(), 0)

        # 使用 assertRaisesRegex 断言，验证访问 s 的第一个元素是否会引发 NotImplementedError 异常并显示 'Not available'
        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s[0]

        # 如果 s 是 TypedStorage 类型，则继续验证其 untyped 存储是否符合预期
        if isinstance(s, torch.TypedStorage):
            self.assertEqual(s.dtype, s_check.dtype)
            self._check_storage_meta(s.untyped(), s_check.untyped())

    # 标记为仅适用于本地设备类型的测试方法修饰器，测试 TypedStorage 类型的 'meta' 存储
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_typed_storage_meta(self, device, dtype):
        # 参数列表，用于创建 TypedStorage 对象的不同参数组合
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        # 遍历参数列表，分别测试每种参数组合的 TypedStorage 对象
        for args in args_list:
            # 创建 TypedStorage 的检查对象
            s_check = torch.TypedStorage(*args, dtype=dtype, device=device)
            # 创建 TypedStorage 的 'meta' 存储对象
            s = torch.TypedStorage(*args, dtype=dtype, device='meta')
            # 调用 _check_storage_meta 方法验证 'meta' 存储对象与其检查对象的一致性
            self._check_storage_meta(s, s_check)

    # 标记为仅适用于本地设备类型的测试方法修饰器，测试 UntypedStorage 类型的 'meta' 存储
    @onlyNativeDeviceTypes
    def test_untyped_storage_meta(self, device):
        # 参数列表，用于创建 UntypedStorage 对象的不同参数组合
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        # 遍历参数列表，分别测试每种参数组合的 UntypedStorage 对象
        for args in args_list:
            # 创建 UntypedStorage 的检查对象
            s_check = torch.UntypedStorage(*args, device=device)
            # 创建 UntypedStorage 的 'meta' 存储对象
            s = torch.UntypedStorage(*args, device='meta')
            # 调用 _check_storage_meta 方法验证 'meta' 存储对象与其检查对象的一致性
            self._check_storage_meta(s, s_check)

    # 标记为仅适用于本地设备类型的测试方法修饰器，测试从 Tensor 对象获取的存储对象的 'meta' 存储
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_storage_meta_from_tensor(self, device, dtype):
        # 创建指定形状和类型的 Tensor 对象作为检查对象
        t_check = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        # 将 Tensor 对象转换为 'meta' 存储
        t = t_check.to('meta')

        # 从检查对象获取存储对象
        s_check = t_check.storage()
        # 从 Tensor 对象获取 'meta' 存储对象
        s = t.storage()
        # 调用 _check_storage_meta 方法验证 'meta' 存储对象与其检查对象的一致性
        self._check_storage_meta(s, s_check)

    # 测试方法修饰器，涵盖所有类型（除非指定排除的类型）和复杂类型（如 torch.half, torch.bool, torch.bfloat16）
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义一个测试方法，用于测试存储元数据相关的错误情况，接受设备和数据类型作为参数
    def test_storage_meta_errors(self, device, dtype):
        # 创建一个 TypedStorage 对象 s0，包含四个元素 [1, 2, 3, 4]，存储于 'meta' 设备上，指定数据类型为 dtype
        s0 = torch.TypedStorage([1, 2, 3, 4], device='meta', dtype=dtype)

        # 断言捕获 NotImplementedError 异常，并验证异常信息包含 'Cannot copy out'
        with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
            s0.cpu()

        # 断言捕获 RuntimeError 异常，并验证异常信息包含 'only available on CPU'
        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_fd_cpu_()

        # 断言捕获 RuntimeError 异常，并验证异常信息包含 'only available on CPU'
        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_filename_cpu_()

        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 断言捕获 NotImplementedError 异常，并验证异常信息包含 'Cannot copy out'
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s0.cuda()

            # 断言捕获 RuntimeError 异常，并验证异常信息包含 'only available on CUDA'
            with self.assertRaisesRegex(RuntimeError, r'only available on CUDA'):
                s0._share_cuda_()

            # 断言捕获 TypeError 异常，并验证异常信息包含 "cannot pin 'torch.storage.UntypedStorage' only CPU memory can be pinned"
            with self.assertRaisesRegex(TypeError, r"cannot pin 'torch.storage.UntypedStorage' only CPU memory can be pinned"):
                s0.pin_memory()

        # 断言捕获 RuntimeError 异常，并验证异常信息包含 'only available on CPU'
        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0.share_memory_()

        # 断言捕获 NotImplementedError 异常，并验证异常信息包含 'Not available'
        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s0.tolist()

        # 使用临时文件 f，断言捕获 NotImplementedError 异常，并验证异常信息包含 'Cannot copy out'
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                # 调用 s0._write_file 方法，向文件 f 写入数据，设置覆盖（True）、锁定（True）、元素大小为 s0 的元素大小
                s0._write_file(f, True, True, s0.element_size())

        # 遍历设备列表 ['cpu', 'cuda']，如果 CUDA 可用则包含 'cuda'，否则仅包含 'cpu'
        for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
            # 创建一个 TypedStorage 对象 s1，包含四个元素 [1, 2, 3, 4]，存储于指定设备上（'cpu' 或 'cuda'），数据类型为 dtype
            s1 = torch.TypedStorage([1, 2, 3, 4], device=device, dtype=dtype)

            # 断言捕获 NotImplementedError 异常，并验证异常信息包含 'Cannot copy out'
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                # 调用 s1.copy_ 方法，将 s0 的内容复制到 s1
                s1.copy_(s0)
    # 定义一个测试函数，用于测试深拷贝操作
    def test_deepcopy(self, device, dtype):
        # 导入深拷贝函数
        from copy import deepcopy
        # 创建两个随机张量 a 和 b，指定设备和数据类型
        a = torch.randn(5, 5, dtype=dtype, device=device)
        b = torch.randn(5, 5, dtype=dtype, device=device)
        # 将张量 a 重新视图为一维张量 c
        c = a.view(25)
        # 创建包含多种对象的列表 q，包括张量 a 和 b 的存储，以及张量 a、b、c 自身
        q = [a, [a.storage(), b.storage()], b, c]
        # 对列表 q 进行深拷贝操作，得到新列表 w
        w = deepcopy(q)
        # 断言深拷贝后的结果与原始列表相等
        self.assertEqual(w[0], q[0], atol=0, rtol=0)
        self.assertEqual(w[1][0], q[1][0], atol=0, rtol=0)
        self.assertEqual(w[1][1], q[1][1], atol=0, rtol=0)
        self.assertEqual(w[1], q[1], atol=0, rtol=0)
        self.assertEqual(w[2], q[2], atol=0, rtol=0)

        # 检查深拷贝是否保留了对象之间的共享关系
        # 修改 w[0] 对应的张量，应该反映在原始张量 a 上
        w[0].add_(1)
        # 遍历张量 a 的所有元素，检查存储共享是否正确
        for i in range(a.numel()):
            self.assertEqual(w[1][0][i], q[1][0][i] + 1)
        # 检查视图张量 c 的修改是否反映在 w[3] 上
        self.assertEqual(w[3], c + 1)
        # 修改 w[2] 对应的张量，应该反映在原始张量 b 上
        w[2].sub_(1)
        # 再次遍历张量 b 的所有元素，检查存储共享是否正确
        for i in range(a.numel()):
            self.assertEqual(w[1][1][i], q[1][1][i] - 1)

        # 检查深拷贝是否保留了对象的属性
        # 给张量 a 添加一个自定义属性 foo，并检查深拷贝后的对象是否保留了该属性
        a.foo = 3
        self.assertEqual(deepcopy(a).foo, 3)

    # 装饰器函数，用于指定测试函数的数据类型
    @dtypes(torch.float32, torch.complex64)
    def test_deepcopy_scalar(self, device, dtype):
        # 导入深拷贝函数
        from copy import deepcopy
        # 创建一个标量张量 a，指定设备和数据类型
        a = torch.tensor(5, dtype=dtype, device=device)
        # 断言深拷贝后的张量的尺寸与原始张量相同
        self.assertEqual(a.size(), deepcopy(a).size())
        # 断言深拷贝后的张量与原始张量相等
        self.assertEqual(a, deepcopy(a))

    # 检查内存重叠的测试函数
    def check_internal_mem_overlap(self, inplace_op, num_inputs,
                                   dtype, device,
                                   expected_failure=False):
        # 如果 inplace_op 是字符串，则将其转换为对应的张量方法
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        # 创建一个随机输入张量 input，然后根据 num_inputs 创建多个随机张量作为输入列表
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input)
                            for i in range(num_inputs - 1)]
        # 如果不是预期失败的情况，确保执行 inplace_op 时会抛出 RuntimeError 异常
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                inplace_op(*inputs)
        else:
            # 如果是预期失败的情况，检查是否会抛出 AssertionError 异常
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                    inplace_op(*inputs)
    # 定义一个方法，用于检查一元操作符对输入和输出内存重叠的情况
    def unary_check_input_output_mem_overlap(self, data, sz, op,
                                             expected_failure=False):

        # 定义内部测试函数，验证操作符对给定输入和输出的行为是否符合预期
        def _test(op, output, input):
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)  # 执行操作符，将结果存储在output_exp中
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)  # 检查操作的输出是否与预期的output_exp相等

        # output 与 input 完全相同的情况:
        _test(op, output=data[0:sz], input=data[0:sz])

        # output 和 input 是独立的情况:
        _test(op, output=data[0:sz], input=data[sz:2 * sz])

        # output 与 input 部分重叠的情况:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, data[0:sz], data[1:sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, data[0:sz], data[1:sz + 1])

        # output 是 input 的转置的情况:
        length = int(math.sqrt(sz))
        input = data[:length**2].view([length, length])  # 将data转换为length*length的矩阵
        out = input.t()  # 计算input的转置
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, out, input)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, out, input)

    # 定义一个方法，用于检查三元操作符对输入和输出内存重叠的情况
    def ternary_check_input_output_mem_overlap(self, op, device,
                                               expected_failure=False):
        sz = 9
        data = torch.randn(2 * sz, device=device)  # 生成一个大小为2*sz的随机张量
        other1 = torch.randn(sz, device=device)   # 生成一个大小为sz的随机张量
        other2 = torch.randn(sz, device=device)   # 生成一个大小为sz的随机张量

        # 调用一元检查函数，验证三元操作符在不同参数组合下的行为
        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(input, other1.view(input.shape), other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), input, other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), other2.view(input.shape), input, out=out),
            expected_failure=expected_failure)
    # 选择可广播的维度
    def _select_broadcastable_dims(self, dims_full=None):
        # 如果未提供完整维度信息，则随机生成
        if dims_full is None:
            dims_full = []
            # 随机生成维度数量
            ndims = random.randint(1, 4)
            # 随机生成每个维度的大小
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # 选择操作的实际维度:
        # 较大的情况下：保留完整的维度数，但各维度大小可能被减少
        # 较小的情况下：可能减少维度数，且各维度大小可能被减少
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        # 逆序遍历维度列表
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # 没有减少的单一维度
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # 较大的可能有减少的单一维度
                ds = dims_full[i]
                # 如果小维度的数量少于设定的数量，则将维度设为1，否则保留原大小
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # 较小的可能有减少的单一维度
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            # 如果小维度列表长度小于设定的数量，则添加当前维度大小到小维度列表
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        # 返回三元组，分别包含小维度、大维度和完整维度信息
        return (dims_small, dims_large, dims_full)
    # 定义一个测试方法，用于检查 `torch._check_tensor_all` 函数的行为
    def test_check_tensor_all(self, device):
        # 默认错误消息
        default_message = 'Expected cond to be True'
        # 获取检查函数的引用
        check_fn = torch._check_tensor_all
        # 预期的错误类型
        expected_error = RuntimeError

        # cond 必须是一个张量，否则抛出 TypeError 异常，错误消息中包含 'cond must be a tensor'
        with self.assertRaisesRegex(TypeError, 'cond must be a tensor'):
            check_fn(True)

        # cond 张量必须是布尔类型，否则抛出 TypeError 异常，错误消息中包含 'cond tensor must have dtype torch.bool'
        with self.assertRaisesRegex(TypeError, 'cond tensor must have dtype torch.bool'):
            check_fn(torch.ones(1, device=device))

        # 定义不同尺寸的测试数据
        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        # 遍历测试尺寸
        for size in test_sizes:
            # 创建全为 True 的布尔张量
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            # 创建全为 False 的布尔张量
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # 不应该抛出错误
            check_fn(t_all_true)

            # 当传入全为 False 的张量时，应该抛出预期的 RuntimeError 异常，消息为 default_message
            with self.assertRaisesRegex(expected_error, default_message):
                check_fn(t_all_false)

            # 如果张量中元素数量大于 1，则测试一个元素为 False 的情况
            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # 随机选择一个元素设置为 False
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                # 应该抛出预期的 RuntimeError 异常，消息为 default_message
                with self.assertRaisesRegex(expected_error, default_message):
                    check_fn(t_all_true_but_one)

            # 测试简单的失败消息
            message = 'message'
            with self.assertRaisesRegex(expected_error, message):
                check_fn(t_all_false, lambda: message)

            # 测试带有张量的消息
            def message():
                return torch.arange(4)

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)

            # 测试格式化字符串消息
            def message():
                return f"{'test'} {[1, 2, 'a', True]} {True} {100} {torch.arange(4)}"

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)
    # 定义一个测试方法，用于在指定设备上测试不同尺寸的张量
    def test_check_tensor_internal(self, device):
        # 定义不同的张量尺寸作为测试数据
        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        # 遍历测试数据，逐个生成全为 True 和全为 False 的张量进行测试
        for size in test_sizes:
            # 创建全为 True 的张量
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            # 创建全为 False 的张量
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # 应该不会抛出错误
            # 调用被测试函数，测试全为 True 的张量
            torch._test_check_tensor(t_all_true)

            # 应该抛出 RuntimeError，并且错误信息包含特定字符串
            with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                # 调用被测试函数，测试全为 False 的张量
                torch._test_check_tensor(t_all_false)

            # 如果张量元素数量大于 1，创建一个除一个元素外全为 True 的张量进行测试
            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # 随机选择一个元素设置为 False
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                # 应该抛出 RuntimeError，并且错误信息包含特定字符串
                with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                    # 调用被测试函数，测试除一个元素外全为 True 的张量
                    torch._test_check_tensor(t_all_true_but_one)

    # 使用不匹配的 arange 输出尺寸触发警告
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref perturbs line numbering")
    # 测试函数，验证 C++ 警告是否具有 Python 上下文
    def test_cpp_warnings_have_python_context(self, device):
        # 预先创建长字符串，以避免 Python 行过长
        s = ".+Triggered internally at.+RangeFactories.+"
        # 设置过滤器，忽略 nvfuser 的弃用警告
        warnings.filterwarnings("ignore", "torch::jit::fuser::cuda", UserWarning)

        # 定义一个函数，触发 C++ 警告
        def cpp_warn_fn():
            out = torch.empty((5,))
            torch.arange(0, 3, out=out)
            return out

        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            # 调用函数触发 C++ 警告
            cpp_warn_fn()
            # 获取当前栈帧信息
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            # 获取第一个警告
            warning = w[0]

            # 检查警告消息中是否包含 C++ 上下文
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # 检查警告的 Python 特征
            # 注意：eager 模式警告引用的是触发警告的函数内的行号
            self.assertEqual(frameinfo.lineno - 6, warning.lineno)
            self.assertEqual(len(w), 1)

        # 捕获脚本化后的 C++ 警告
        with warnings.catch_warnings(record=True) as w:
            # 脚本化函数
            scripted_cpp_warn_fn = torch.jit.script(cpp_warn_fn)
            # 调用脚本化函数触发警告
            scripted_cpp_warn_fn()
            # 获取第一个警告
            warning = w[0]

            # 检查警告消息中是否包含 C++ 上下文
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # 检查警告的 Python 特征
            # 注意：脚本化警告的行号指的是调用脚本化函数的行号，
            # 在我们的测试套件中有一层间接调用，导致 Python 行号的检查较为脆弱
            self.assertEqual(len(w), 1)

        # 捕获脚本化后的 Python 警告
        def warn_fn():
            warnings.warn("Warning!")

        # 脚本化函数模拟 eager 模式的 Python 警告
        with warnings.catch_warnings(record=True) as w:
            # 脚本化函数
            scripted_warn_fn = torch.jit.script(warn_fn)
            # 调用脚本化函数触发警告
            scripted_warn_fn()
            # 获取当前栈帧信息
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            # 获取第一个警告
            warning = w[0]

            # 检查警告消息中是否包含 "Warning!"
            self.assertTrue(re.search('Warning!', str(warning.message)) is not None)

            # 检查警告的 Python 特征
            self.assertEqual(frameinfo.lineno - 6, warning.lineno)
            self.assertEqual(len(w), 1)
    # 定义一个测试方法，用于验证能够捕获 TORCH_WARN_ONCE 警告两次
    def test_warn_always_caught(self, device):
        # 检查是否能够多次捕获 TORCH_WARN_ONCE 警告
        # assertWarnsOnceRegex 使用 set_warn_always(True)，将 TORCH_WARN_ONCE 改为 TORCH_WARN
        a = np.arange(10)
        a.flags.writeable = False
        # 使用 assertWarnsOnceRegex 上下文管理器捕获 UserWarning，并检查警告消息中是否包含 'non-writable'
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # 第二次尝试捕获相同的警告
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # 确保发出两次警告不会导致 assertWarnsOnceRegex 上下文管理器失败
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)
            torch.from_numpy(a)

    # 标记为只在原生设备类型上运行的测试方法
    @onlyNativeDeviceTypes
    def test_complex_half_experimental_warning(self, device):
        msg = 'ComplexHalf support is experimental'
        # 使用 assertWarnsOnceRegex 捕获 UserWarning，并检查警告消息中是否包含 'ComplexHalf support is experimental'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            t = torch.randn(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros(3, dtype=torch.chalf, device=device)

        t = torch.randn(3, dtype=torch.chalf, device=device)
        # 检查是否会发出警告消息
        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.randn_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros_like(t)

        # t + 1 分配一个新的张量来保存结果，使用 empty
        with self.assertWarnsOnceRegex(UserWarning, msg):
            t + 1

    # 仅在 CUDA 上运行的测试方法
    @onlyCUDA
    def test_dtypetensor_warnings(self, device):
        msg = 'The torch.cuda.*DtypeTensor constructors are no longer recommended'
        # 使用 assertWarnsOnceRegex 捕获 UserWarning，并检查警告消息中是否包含 'The torch.cuda.*DtypeTensor constructors are no longer recommended'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            t = torch.cuda.FloatTensor([0])

        with self.assertWarnsOnceRegex(UserWarning, msg):
            t = torch.cuda.DoubleTensor([0])
    # 设置一个正则表达式，用于匹配警告消息，提示用户使用 torch.set_default_dtype() 替代 torch.set_default_tensor_type()
    msg = '.*is deprecated as of PyTorch 2.1, please use torch.set_default_dtype().*'
    
    # 获取当前默认张量类型并保存
    default_type = torch.tensor([]).type()
    
    try:
        # 断言在上下文中会发出一次 UserWarning 警告，并且匹配给定的正则表达式消息
        with self.assertWarnsOnceRegex(UserWarning, msg):
            # 设置默认张量类型为 torch.FloatTensor
            torch.set_default_tensor_type(torch.FloatTensor)

        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 断言在上下文中会发出一次 UserWarning 警告，并且匹配给定的正则表达式消息
            with self.assertWarnsOnceRegex(UserWarning, msg):
                # 设置默认张量类型为 torch.cuda.FloatTensor
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
    finally:
        # 恢复默认的张量类型
        torch.set_default_tensor_type(default_type)

# TODO: this test should be in test_nn.py
# 测试转置卷积在不同设备上的反向传播和内存格式的不可知性
def test_conv_transposed_backward_agnostic_to_memory_format(self, device):
    # 定义输入通道数、输出通道数、缩放因子、批量大小和长度
    in_channels = 64
    out_channels = 128
    scale_factor = 8
    batch_size = 8
    length = 16

    # 创建 ConvTranspose1d 层和 LayerNorm 层，并移动到指定设备
    conv = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor).to(device)
    layer_norm = torch.nn.LayerNorm(out_channels).to(device)

    # 生成随机输入张量，并移动到指定设备，保证其内存布局连续
    input_ = torch.randn(batch_size, in_channels, length).to(device).contiguous()
    # 对输入应用转置卷积层并确保内存布局连续
    input_ = conv(input_).contiguous()
    # 对转置卷积层输出进行 LayerNorm，并确保内存布局连续
    input_ = layer_norm(input_.transpose(1, 2).contiguous()).contiguous()
    # 计算张量的所有元素之和，并执行反向传播
    input_.sum().backward()

    # 3D 情况下的转置卷积操作
    conv = torch.nn.ConvTranspose3d(3, 3, kernel_size=3).to(device)
    # 生成随机输入张量，并移动到指定设备
    input = torch.randn(batch_size, 3, length, length, length, device=device)
    # 对输入应用 3D 转置卷积操作
    out = conv(input)
    # 对输出执行反向传播，并对张量按特定维度进行转置
    out.backward(torch.ones_like(out).transpose(-2, -1))

# TODO: this test should be in test_nn.py
# 测试在 CUDA 设备上进行大尺寸输入张量的转置卷积操作
@onlyCUDA
@largeTensorTest('12GB')
def test_conv_transposed_large(self, device):
    # 定义输入通道数、输出通道数、卷积核大小
    in_channels = 64
    out_channels = 128
    kernel_size = 5

    # 创建 ConvTranspose3d 层，并移动到指定设备
    conv = torch.nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=kernel_size,
        stride=2, padding=2, output_padding=1).to(device)

    # 生成随机输入张量，并移动到指定设备
    x = torch.rand([1, 64, 8, 128, 172]).to(device)
    # 对输入进行 ConvTranspose3d 操作
    y = conv(x)
    # 定义测试函数，测试Tensor对象的is_set_to方法
    def test_is_set_to(self, device):
        # 创建两个空的Tensor对象t1和t2，形状为[3, 4, 9, 10]，使用指定设备
        t1 = torch.empty(3, 4, 9, 10, device=device)
        t2 = torch.empty(3, 4, 9, 10, device=device)
        # 创建一个空的Tensor对象t3，并使用t1的内容来设置它
        t3 = torch.tensor([], device=device).set_(t1)
        # 克隆t3，改变其形状为[12, 90]，得到t4
        t4 = t3.clone().resize_(12, 90)
        # 断言t1不是设置为t2
        self.assertFalse(t1.is_set_to(t2))
        # 断言t1被设置为t3
        self.assertTrue(t1.is_set_to(t3))
        # 断言t3被设置为t1，附带自定义消息"is_set_to should be symmetric"
        self.assertTrue(t3.is_set_to(t1), "is_set_to should be symmetric")
        # 断言t1不是设置为t4
        self.assertFalse(t1.is_set_to(t4))
        # 断言空的Tensor对象不会被设置为另一个空的Tensor对象，附带自定义消息
        self.assertFalse(torch.tensor([]).is_set_to(torch.tensor([])),
                         "Tensors with no storages should not appear to be set "
                         "to each other")

        # 创建一个包含布尔值的Tensor对象t1，使用指定设备
        t1 = torch.tensor([True, True], dtype=torch.bool, device=device)
        # 创建一个空的Tensor对象t2，并使用t1的内容来设置它
        t2 = torch.tensor([0], dtype=torch.bool, device=device).set_(t1)
        # 断言t1被设置为t2
        self.assertTrue(t1.is_set_to(t2))

        # 测试尺寸不匹配的情况
        # 创建一个空的Tensor对象t1，形状为[2, 3, 4]，使用指定设备
        t1 = torch.empty([2, 3, 4], device=device)
        # 将t1视图变形为[4, 3, 2]得到t2
        t2 = t1.view(4, 3, 2)
        # 断言t1不是设置为t2
        self.assertFalse(t1.is_set_to(t2))
        # 断言t2不是设置为t1
        self.assertFalse(t2.is_set_to(t1))

        # 测试旧版本空Tensor尺寸行为（即所有空Tensor逻辑上被折叠到尺寸[0]）
        # 创建一个空的Tensor对象t1，形状为[2, 5, 0]，使用指定设备
        t1 = torch.empty([2, 5, 0], device=device)
        # 将t1视图变形为[0]得到t2
        t2 = t1.view([0])
        # 断言t1不是设置为t2
        self.assertFalse(t1.is_set_to(t2))
        # 断言t2不是设置为t1
        self.assertFalse(t2.is_set_to(t1))
        def test_cublas_config_nondeterministic_alert(self, device):
            # 定义测试用例列表，每个元素包含函数名和对应的张量尺寸元组
            test_cases = [
                ('mm', ((2, 2), (2, 2),)),
                ('mv', ((2, 2), (2,),)),
                ('bmm', ((1, 2, 2), (1, 2, 2),))
            ]

            # 定义测试配置列表，每个元素包含CuBLAS工作空间配置和是否确定性的布尔值
            test_configs = [
                ('garbage', False),
                (None, False),
                (':4096:8', True),
                (':16:8', True)
            ]

            cublas_var_name = 'CUBLAS_WORKSPACE_CONFIG'
            is_cuda10_2_or_higher = (
                (torch.version.cuda is not None)
                and ([int(x) for x in torch.version.cuda.split(".")] >= [10, 2])
            )

            def test_case_info(fn_name, config):
                # 返回包含函数名和配置信息的字符串
                return f'function "{fn_name}" with config "{"" if config is None else config}"'

            # 创建进程以测试每个测试用例和配置组合的情况
            processes = []
            for fn_name, arg_sizes in test_cases:
                for config, is_config_deterministic in test_configs:
                    env = os.environ.copy()
                    if config is None:
                        # 如果配置为None且环境变量中存在CUBLAS_WORKSPACE_CONFIG，则删除该变量
                        if env.get(cublas_var_name) is not None:
                            del env[cublas_var_name]
                    else:
                        # 否则，将配置添加到环境变量中
                        env[cublas_var_name] = config
                    
                    # 确定是否应该抛出错误，如果CUDA版本大于等于10.2且配置是非确定性的
                    should_throw_error = is_cuda10_2_or_higher and not is_config_deterministic
                    
                    # 构建测试脚本字符串，这部分代码还未结束
                    script = f"""
# 导入 PyTorch 库
import torch
# 设置 PyTorch 使用确定性算法，确保结果的确定性
torch.use_deterministic_algorithms(True)
# 定义函数名，用于后续调用
fn = torch.{fn_name}
# 参数尺寸列表，用于生成随机张量的大小
arg_sizes = {arg_sizes}
# 指定设备，如 'cpu' 或 'cuda:0'
device = '{device}'
# 是否预期抛出错误，从输入中获取
should_throw_error = {should_throw_error}
# 初始化参数列表
args = []
# 根据每个参数大小生成对应的随机张量，并加入参数列表
for arg_size in arg_sizes:
    args.append(torch.randn(*arg_size, device=device))
# 尝试调用指定函数 fn，并传入生成的参数列表 args
try:
    fn(*args)
# 捕获 RuntimeError 异常
except RuntimeError as e:
    # 如果不期望抛出错误，则抛出自定义异常信息
    if not should_throw_error:
        raise RuntimeError('Did not expect any error to be raised')
    # 如果错误信息不包含 CuBLAS 的非确定性行为提示，则抛出自定义异常信息
    elif 'Deterministic behavior was enabled with either' not in str(e):
        raise RuntimeError('Expected a CuBLAS nondeterministic error, but got a different error')
# 如果没有抛出 RuntimeError 异常
else:
    # 如果预期抛出错误，则抛出自定义异常信息
    if should_throw_error:
        raise RuntimeError('Expected a CuBLAS nondeterministic error, but it was not raised')

"""
尝试运行指定的 Python 脚本，捕获其输出的异常信息。
在 Windows 平台下，默认的当前工作目录可能导致 `import torch` 失败，
因此显式设置当前工作目录为脚本所在的目录，同时使用指定的环境变量。
"""
try:
    subprocess.check_output(
        [sys.executable, '-c', script],
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.realpath(__file__)),
        env=env)
# 捕获 subprocess.CalledProcessError 异常
except subprocess.CalledProcessError as e:
    # 抛出自定义异常信息，包括测试用例名称和异常输出内容
    self.fail(msg=(
        f'Subprocess exception while attempting to run {test_case_info(fn_name, config)}:\n'
        + e.output.decode("utf-8")))

# 定义测试方法，用于检查非确定性的量化张量 resize 操作
@onlyCPU
@skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
@dtypes(*get_all_qint_dtypes())
def test_nondeterministic_resize_quantized(self, device, dtype):
    # 创建一个浮点类型的张量 a，并量化为指定的 dtype
    a = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.float, device=device)
    b = torch.quantize_per_tensor(a, 0.1, 10, dtype)
    # 调用测试函数，检查是否会触发非确定性警告
    self.check_nondeterministic_alert(
        lambda: b.resize_((10,)),
        'quantized_resize_cpu_')

# 跳过 XLA 平台的测试
@skipXLA
@skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
@dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    # 定义一个测试方法，用于测试张量的确定性调整大小功能
    def test_deterministic_resize(self, device, dtype):
        # 定义测试用例，每个测试用例包括初始大小、步长和调整后的大小
        test_cases = [
            # size, stride, resize_size
            ((10,), (1,), (5,)),                # 一维张量，步长为1，调整大小为5
            ((10,), (0,), (10,)),               # 一维张量，步长为0，保持大小为10不变
            ((10,), (1,), (20,)),               # 一维张量，步长为1，调整大小为20
            ((2, 3, 4), None, (2, 3, 4)),       # 三维张量，无步长，保持大小为(2, 3, 4)不变
            ((2, 3, 4), None, (6, 3, 4)),       # 三维张量，无步长，调整大小为(6, 3, 4)
            ((2, 3, 4), None, (2, 5, 4)),       # 三维张量，无步长，调整大小为(2, 5, 4)
            ((2, 3, 4), None, (2, 3, 6)),       # 三维张量，无步长，调整大小为(2, 3, 6)
            ((2, 3, 4), None, (3, 4, 5)),       # 三维张量，无步长，调整大小为(3, 4, 5)
            ((2, 3, 4), (1, 4, 12), (2, 3, 4)),  # 三维张量，指定步长，保持大小为(2, 3, 4)不变
            ((2, 3, 4), (1, 4, 12), (4, 3, 4)),  # 三维张量，指定步长，调整大小为(4, 3, 4)
            ((2, 3, 4), (1, 4, 12), (2, 4, 4)),  # 三维张量，指定步长，调整大小为(2, 4, 4)
            ((2, 3, 4), (1, 4, 12), (2, 3, 5)),  # 三维张量，指定步长，调整大小为(2, 3, 5)
            ((2, 3, 4), (1, 4, 12), (3, 4, 5)),  # 三维张量，指定步长，调整大小为(3, 4, 5)
            ((2, 3, 4), (1, 0, 1), (2, 4, 5)),   # 三维张量，指定步长，调整大小为(2, 4, 5)
        ]

        # 遍历每个测试用例
        for size, stride, resize_size in test_cases:
            # 根据是否有步长选择合适的张量创建方法
            if stride is None:
                a = torch.zeros(size, dtype=dtype, device=device)
            else:
                a = torch.empty_strided(size, stride, dtype=dtype, device=device).fill_(0)
            # 复制旧的存储内容
            old_storage = a.untyped_storage().clone()
            # 使用确定性保护和填充未初始化内存的选项来调整张量大小
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                a.resize_(resize_size)

            new_storage = a.untyped_storage()

            # 如果存储大小增加，检查新部分是否被填充为NaN/MAX_INT；否则检查存储内容是否相等
            old_tensor = torch.tensor(old_storage, dtype=dtype)
            old_numel = old_tensor.numel()
            new_tensor = torch.tensor(new_storage, dtype=dtype)
            new_numel = new_tensor.numel()

            if new_numel > old_numel:
                self.assertEqual(new_tensor[:old_numel], old_tensor)
                fill_section = new_tensor[old_numel:]

                # 如果是浮点数或复数类型，检查填充部分是否全为NaN
                if dtype.is_floating_point or dtype.is_complex:
                    self.assertTrue(fill_section.isnan().all())
                else:
                    # 否则检查填充部分是否全为最大值（根据dtype不同确定最大值）
                    if dtype == torch.bool:
                        max_val = True
                    else:
                        max_val = torch.iinfo(dtype).max
                    self.assertTrue(fill_section.eq(max_val).all())
            else:
                # 存储大小未变化时，直接比较旧存储和新存储
                self.assertEqual(old_tensor, new_tensor)

    # 在启用确定性算法时，`torch.empty` 应当用NaN填充浮点张量，用MAX_INT填充整数张量
    @skipXLA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    # 定义一个测试方法，用于测试生成确定性空张量的各种方式
    def test_deterministic_empty(self, device, dtype):
        # 定义生成张量的函数列表，每个函数使用不同的方法生成张量
        gen_fns = [
            lambda: torch.empty(10, 9, device=device, dtype=dtype),  # 使用 torch.empty 创建空张量
            lambda: torch.empty(10, 9, out=torch.zeros(1, device=device, dtype=dtype)),  # 使用 torch.empty 创建空张量并填充为零
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype)),  # 使用 torch.empty_like 根据现有张量形状创建空张量
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype), memory_format=torch.contiguous_format),  # 使用 torch.empty_like 创建连续内存格式的空张量
            lambda: torch.empty_strided((10, 9), (1, 5), device=device, dtype=dtype),  # 使用 torch.empty_strided 创建步长张量
            lambda: torch.empty_permuted((2, 3, 5), (1, 0, 2), device=device, dtype=dtype),  # 使用 torch.empty_permuted 创建排列后的张量
        ]

        # 遍历生成张量的函数列表
        for gen_fn in gen_fns:
            # 使用 DeterministicGuard 确保生成的张量是确定性的，并且填充未初始化的内存
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                # 调用生成张量的函数
                res = gen_fn()

            # 检查生成的张量是否包含 NaN（对于浮点数和复数类型）
            if dtype.is_floating_point or dtype.is_complex:
                self.assertTrue(res.isnan().all())
            else:
                # 对于非浮点数类型，检查是否所有元素均等于最大值
                if dtype == torch.bool:
                    max_val = True
                else:
                    max_val = torch.iinfo(dtype).max
                self.assertTrue(res.eq(max_val).all())

    # FIXME: update OpInfos to support "nondeterministic samples" and port these tests
    #   to that architecture
    # 使用 skipIfMps 装饰器，跳过测试用例，因为 OpInfos 不支持 "nondeterministic samples"，需要更新该功能
    # 同时使用 skipIfTorchInductor 装饰器，因为当前的 torch inductor 存在问题
    def test_nondeterministic_alert_AvgPool3d(self, device):
        # 创建 AvgPool3d 模块和随机输入张量
        module = torch.nn.AvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        # 对输入张量进行 AvgPool3d 操作
        res = module(input)
        grad = torch.ones_like(res)

        # 检查 AvgPool3d 操作是否引发了非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    # 使用 skipIfMps 装饰器，跳过测试用例，因为 OpInfos 不支持 "nondeterministic samples"，需要更新该功能
    # 同时使用 skipIfTorchInductor 装饰器，因为当前的 torch inductor 存在问题
    def test_nondeterministic_alert_AdaptiveAvgPool2d(self, device):
        # 创建 AdaptiveAvgPool2d 模块和随机输入张量
        module = torch.nn.AdaptiveAvgPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        # 对输入张量进行 AdaptiveAvgPool2d 操作
        res = module(input)
        grad = torch.ones_like(res)

        # 检查 AdaptiveAvgPool2d 操作是否引发了非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    # 使用 skipIfMps 装饰器，跳过测试用例，因为 OpInfos 不支持 "nondeterministic samples"，需要更新该功能
    # 同时使用 skipIfTorchInductor 装饰器，因为当前的 torch inductor 存在问题
    def test_nondeterministic_alert_AdaptiveAvgPool3d(self, device):
        # 创建 AdaptiveAvgPool3d 模块和随机输入张量
        module = torch.nn.AdaptiveAvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        # 对输入张量进行 AdaptiveAvgPool3d 操作
        res = module(input)
        grad = torch.ones_like(res)

        # 检查 AdaptiveAvgPool3d 操作是否引发了非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    # 使用 skipIfMps 装饰器，跳过测试用例，因为 OpInfos 不支持 "nondeterministic samples"，需要更新该功能
    # 同时使用 skipIfTorchInductor 装饰器，因为当前的 torch inductor 存在问题
    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes



    # 标记测试函数仅适用于浮点类型和半精度浮点数，且仅适用于本地设备类型



    def test_nondeterministic_alert_MaxUnpool1d(self, device, dtype):
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        # 创建 MaxUnpool1d 模块，指定参数 3 和 1
        module = torch.nn.MaxUnpool1d(3, 1)
        # 创建随机输入张量，形状为 [1, 1, 7]，指定数据类型和设备
        input = torch.randn(1, 1, 7, dtype=dtype, device=device)
        # 创建与输入张量相同形状的索引张量，数据类型为 long，设备同输入
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        # 调用检查非确定性警告函数，检查 MaxUnpool1d 模块的前向输出
        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')



        module = torch.nn.FractionalMaxPool3d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        # 调用检查非确定性警告函数，检查 FractionalMaxPool3d 模块的反向传播
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')



        module = torch.nn.FractionalMaxPool2d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        # 调用检查非确定性警告函数，检查 FractionalMaxPool2d 模块的反向传播
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')



        module = torch.nn.AdaptiveMaxPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        # 调用检查非确定性警告函数，检查 AdaptiveMaxPool2d 模块的反向传播
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')



        module = torch.nn.MaxPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        # 调用检查非确定性警告函数，检查 MaxPool3d 模块的反向传播
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'max_pool3d_with_indices_backward_cuda',
            torch.device(device).type == 'cuda')
    # 定义一个测试方法，用于测试 MaxUnpool2d 的非确定性警报功能
    def test_nondeterministic_alert_MaxUnpool2d(self, device, dtype):
        # 如果数据类型是半精度并且设备类型是 CPU，则跳过测试
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        # 创建 MaxUnpool2d 模块实例，指定核大小为 3
        module = torch.nn.MaxUnpool2d(3, 1)
        # 创建随机输入张量，形状为 [1, 1, 7, 7]，指定数据类型和设备
        input = torch.randn(1, 1, 7, 7, dtype=dtype, device=device)
        # 创建与输入张量相同形状的零张量，作为 MaxUnpool2d 操作的索引
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        # 调用自定义方法检查非确定性警报
        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')

    # 使用装饰器指定测试方法的数据类型为浮点数类型和半精度类型
    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    # 定义一个测试方法，用于测试 MaxUnpool3d 的非确定性警报功能
    def test_nondeterministic_alert_MaxUnpool3d(self, device, dtype):
        # 如果数据类型是半精度并且设备类型是 CPU，则跳过测试
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        # 创建 MaxUnpool3d 模块实例，指定核大小为 3
        module = torch.nn.MaxUnpool3d(3, 1)
        # 创建随机输入张量，形状为 [1, 1, 7, 7, 7]，指定数据类型和设备
        input = torch.randn(1, 1, 7, 7, 7, dtype=dtype, device=device)
        # 创建与输入张量相同形状的零张量，作为 MaxUnpool3d 操作的索引
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        # 调用自定义方法检查非确定性警报
        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling3d_forward_out')

    # 使用装饰器指定测试方法需要跳过 MPS 测试
    @skipIfMps
    # 使用装饰器指定测试方法需要跳过特定的 Torch Inductor 问题
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    # 定义一个测试方法，用于测试 interpolate 函数中 linear 模式的非确定性警报功能
    def test_nondeterministic_alert_interpolate_linear(self, device):
        # 创建随机输入张量，形状为 [1, 2, 4]，指定设备和需要梯度计算
        input = torch.randn(1, 2, 4, device=device, requires_grad=True)
        # 使用 interpolate 函数对输入张量进行线性插值，调整大小为 12，不对齐角点
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='linear',
            align_corners=False)
        # 创建与 res 张量形状相同的全为 1 的梯度张量
        grad = torch.ones_like(res)

        # 调用自定义方法检查非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_linear1d_backward_out_cuda',
            # 检查设备类型是否为 CUDA
            torch.device(device).type == 'cuda')

    # 使用装饰器指定测试方法需要跳过特定的 Torch Inductor 问题
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    # 定义一个测试方法，用于测试 interpolate 函数中 bilinear 模式的非确定性警报功能
    def test_nondeterministic_alert_interpolate_bilinear(self, device):
        # 创建随机输入张量，形状为 [1, 2, 4, 4]，指定设备和需要梯度计算
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        # 使用 interpolate 函数对输入张量进行双线性插值，调整大小为 12，不对齐角点
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bilinear',
            align_corners=False)
        # 创建与 res 张量形状相同的全为 1 的梯度张量
        grad = torch.ones_like(res)

        # 调用自定义方法检查非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bilinear2d_backward_out_cuda',
            # 检查设备类型是否为 CUDA
            torch.device(device).type == 'cuda')

    # 使用装饰器指定测试方法需要跳过特定的 Torch Inductor 问题
    @skipIfTorchInductor("aot-autograd issue")
    # 定义一个测试函数，用于测试 DeterministicGuard 在 Replication Pad 2D 操作中的行为
    def test_deterministic_replication_pad2d(self, device):
        # 定义测试用例列表，每个元素是一个元组，包含输入张量的尺寸和填充参数
        test_cases = [
            # size, padding
            [(1, 2, 4, 4), (0, 0, 0, 0)],    # 示例1: 尺寸为(1, 2, 4, 4)，无填充
            [(1, 2, 4, 4), (3, 4, 5, 6)],   # 示例2: 尺寸为(1, 2, 4, 4)，填充参数为(3, 4, 5, 6)
            [(3, 8, 7), (0, 0, 0, 0)],      # 示例3: 尺寸为(3, 8, 7)，无填充
            [(3, 8, 7), (4, 3, 2, 7)],      # 示例4: 尺寸为(3, 8, 7)，填充参数为(4, 3, 2, 7)
        ]

        # 如果设备不是 'xla' 类型，添加额外的测试用例
        if torch.device(device).type != 'xla':
            test_cases += [
                [(4, 3, 5, 10), (-9, 4, 5, 6)],    # 示例5: 尺寸为(4, 3, 5, 10)，填充参数为(-9, 4, 5, 6)
                [(3, 8, 7), (-4, -2, -2, -3)],    # 示例6: 尺寸为(3, 8, 7)，填充参数为(-4, -2, -2, -3)
            ]

        # 遍历所有测试用例
        for size, padding in test_cases:
            # 创建指定尺寸的随机输入张量，设置 requires_grad=True 以计算梯度
            input = torch.randn(*size, device=device, requires_grad=True)
            grad = None
            # 使用 DeterministicGuard 确保结果的确定性
            with DeterministicGuard(True):
                # 执行 replicate 模式的填充操作
                res = torch.nn.functional.pad(
                    input,
                    padding,
                    mode='replicate')
                # 计算反向传播并检查梯度
                res.backward(torch.ones_like(res))
                # 如果 grad 为空，则初始化为当前输入的梯度；否则断言当前梯度与之前的一致
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                # 清空输入张量的梯度
                input.grad = None

    # 标记为在 Torch Inductor 下跳过测试的函数
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    # 测试 DeterministicGuard 在 bilinear 插值操作中的行为
    def test_deterministic_interpolate_bilinear(self, device):
        # 创建指定尺寸的随机输入张量，设置 requires_grad=True 以计算梯度
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        grad = None
        # 使用 DeterministicGuard 确保结果的确定性
        with DeterministicGuard(True):
            # 多次进行 bilinear 插值操作，并检查梯度
            for _ in range(5):
                res = torch.nn.functional.interpolate(
                    input,
                    size=12,
                    mode='bilinear',
                    align_corners=False)
                res.backward(torch.ones_like(res))
                # 如果 grad 为空，则初始化为当前输入的梯度；否则断言当前梯度与之前的一致
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                # 清空输入张量的梯度
                input.grad = None

    # 标记为在 MPS 下跳过测试的函数，并在 Torch Inductor 下跳过特定问题
    @skipIfMps
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    # 测试 bicubic 插值操作的非确定性情况
    def test_nondeterministic_alert_interpolate_bicubic(self, device):
        # 创建指定尺寸的随机输入张量，设置 requires_grad=True 以计算梯度
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        # 执行 bicubic 插值操作，并初始化梯度为全 1
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bicubic',
            align_corners=False)
        grad = torch.ones_like(res)

        # 检查非确定性警报，并执行反向传播计算梯度
        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bicubic2d_backward_out_cuda',
            torch.device(device).type == 'cuda')
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    # 定义测试方法，用于验证 ReplicationPad2d 模块的非确定性警报
    def test_nondeterministic_alert_ReplicationPad2d(self, device):
        # 创建 ReplicationPad2d 模块，设置填充参数为 (1, 2, 3, 4)
        module = torch.nn.ReplicationPad2d((1, 2, 3, 4))
        # 生成随机输入张量，指定设备并需要计算梯度
        input = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
        # 将输入张量传入模块，获取输出结果
        res = module(input)
        # 创建与输出结果同形状的全一张量作为梯度
        grad = torch.ones_like(res)

        # 检查反向传播调用是否非确定性，并在条件满足时引发警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            torch.device(device).type == 'cuda')

        # 使用 DeterministicGuard 设置为 True，确保模块行为是确定性的
        with DeterministicGuard(True):
            res = module(input)

        # 重新生成全一张量作为梯度
        grad = torch.ones_like(res)

        # 如果前向传播是确定性的，不应引发非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            False)

    # 使用装饰器 skipIfMps，条件为 True 时跳过此测试
    # 使用装饰器 skipIfTorchInductor，条件为指定的 GitHub 问题链接时跳过此测试
    def test_nondeterministic_alert_ReplicationPad3d(self, device):
        # 创建 ReplicationPad3d 模块，设置填充参数为 (1, 2, 3, 4, 5, 6)
        module = torch.nn.ReplicationPad3d((1, 2, 3, 4, 5, 6))
        # 生成随机输入张量，指定设备并需要计算梯度
        input = torch.randn(2, 3, 4, 4, 4, device=device, requires_grad=True)
        # 将输入张量传入模块，获取输出结果
        res = module(input)
        # 创建与输出结果同形状的全一张量作为梯度
        grad = torch.ones_like(res)

        # 检查反向传播调用是否非确定性，并在条件满足时引发警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad3d_backward_cuda',
            torch.device(device).type == 'cuda')

    # 使用装饰器 skipIfTorchDynamo，条件为 "Warning is not raised." 时跳过此测试
    def test_nondeterministic_alert_NLLLoss(self, device):
        # 创建 NLLLoss 模块
        module = torch.nn.NLLLoss()
        # 生成随机输入张量，指定设备
        input = torch.randn(2, 3, 5, 5, device=device)
        # 生成随机目标张量，先乘以 3 再向下取整转为长整型，指定设备
        target = torch.rand(2, 5, 5, device=device).mul(3).floor().long()

        # 检查前向传播调用是否非确定性，并在条件满足时引发警报
        self.check_nondeterministic_alert(
            lambda: module(input, target),
            'nll_loss2d_forward_out_cuda_template',
            torch.device(device).type == 'cuda')

    # 使用装饰器 skipIfTorchInductor，条件为指定的 GitHub 问题链接时跳过此测试
    def test_nondeterministic_alert_CTCLoss(self, device):
        # 创建 CTCLoss 模块
        module = torch.nn.CTCLoss()
        # 生成随机输入张量，指定设备并需要计算梯度
        input = torch.randn(50, 3, 15, device=device, requires_grad=True)
        # 生成随机目标张量，指定设备
        target = torch.randint(0, 14, (3, 30), device=device)
        # 指定输入和目标的长度列表
        input_lengths = [50, 50, 50]
        target_lengths = [30, 25, 20]
        # 将输入、目标及长度传入模块，获取输出结果
        res = module(input, target, input_lengths, target_lengths)
        # 创建与输出结果同形状的全一张量作为梯度
        grad = torch.ones_like(res)

        # 检查反向传播调用是否非确定性，并在条件满足时引发警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'ctc_loss_backward_gpu',
            torch.device(device).type == 'cuda')
    # 定义一个测试方法，用于测试 EmbeddingBag 在 max 模式下的非确定性警报
    def test_nondeterministic_alert_EmbeddingBag_max(self, device):
        # 创建 EmbeddingBag 模块，包括指定的参数和随机权重，设备为给定的设备
        module = torch.nn.EmbeddingBag(
            4, 3, None, 2., False, 'max',
            _weight=torch.randn(4, 3, device=device, requires_grad=True))
        # 创建一个输入张量，形状为 (4, 3)，元素值为 0 到 2 之间的随机整数，设备为给定的设备
        input = torch.randint(0, 3, (4, 3), device=device)
        # 对输入数据进行 EmbeddingBag 操作，得到结果张量 res
        res = module(input)
        # 创建一个与 res 相同形状的全 1 张量 grad
        grad = torch.ones_like(res)

        # 调用检查非确定性警报的方法，传入一个函数，用于计算梯度反向传播，并指定警报名称和是否在 CUDA 设备上运行
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'embedding_bag_backward_cuda_max',
            torch.device(device).type == 'cuda')

    # 用于测试 cumsum 操作的非确定性警报
    @dtypes(*all_types_and_complex_and(torch.bool))
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_cumsum(self, device, dtype):
        # 创建一个指定设备和数据类型的张量 input，形状为 (10,)
        input = make_tensor((10,), dtype=dtype, device=device, low=-9, high=9)
        # 根据设备类型和数据类型是否为浮点型或复数型，确定是否需要警报
        should_alert = torch.device(device).type == 'cuda' and (dtype.is_floating_point or dtype.is_complex)

        # 遍历两种 cumsum 操作，检查其非确定性警报
        for op_call in [torch.Tensor.cumsum, torch.cumsum]:
            self.check_nondeterministic_alert(
                lambda: op_call(input, 0),
                'cumsum_cuda_kernel',
                should_alert)

    # 标记为预期失败的测试方法，用于测试 put 操作的非确定性警报
    @expectedFailureMeta  # 预期出现非确定性错误，但实际未引发
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_put(self, device):
        # 创建一个指定设备的随机张量 a，形状为 (10,)
        a = torch.randn(10, device=device)
        # 创建一个索引张量 indices，设备为给定设备
        indices = torch.tensor([0, 0], device=device)
        # 创建一个值张量 values，设备为给定设备
        values = torch.tensor([0., 1.], device=device)

        # 遍历两种 put 操作，检查其非确定性警报
        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=False),
                'put_')

    # 标记为预期失败的测试方法，用于测试 put 操作在累积模式下的非确定性警报
    @skipIfTorchInductor("warning is logged from the FallbackKernel: torch.ops.aten.put_.default when warn_only=True")
    def test_nondeterministic_alert_put_accumulate(self, device):
        # 创建一个指定设备的随机张量 a，形状为 (10,)
        a = torch.randn(10, device=device)
        # 创建一个索引张量 indices，设备为给定设备
        indices = torch.tensor([0, 0], device=device)
        # 创建一个值张量 values，设备为给定设备
        values = torch.tensor([0., 1.], device=device)

        # 遍历两种 put 操作，检查其在累积模式下的非确定性警报，并指定是否在 CUDA 设备上运行
        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=True),
                'put_',
                torch.device(device).type == 'cuda')

    # 标记为预期失败的测试方法，用于测试 histc 操作的非确定性警报
    @skipIfMps
    def test_nondeterministic_alert_histc(self, device):
        # 创建一个空的张量 a，设备为给定设备
        a = torch.tensor([], device=device)
        # 遍历两种 histc 操作，检查其非确定性警报，并指定是否在 CUDA 设备上运行
        for op_call in [torch.histc, torch.Tensor.histc]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, min=0, max=3),
                '_histc_cuda',
                torch.device(device).type == 'cuda')
    # 定义一个测试方法，用于测试在给定设备上执行 torch.bincount 或 torch.Tensor.bincount 时的非确定性警报
    def test_nondeterministic_alert_bincount(self, device):
        # 创建一个空的长整型张量 a，并指定设备
        a = torch.tensor([], device=device, dtype=torch.long)
        # 创建一个空的张量 weights，并指定设备
        weights = torch.tensor([], device=device)

        # 对于每个操作调用 torch.bincount 或 torch.Tensor.bincount
        for op_call in [torch.bincount, torch.Tensor.bincount]:
            # 当设备为 CUDA 且提供了权重时，应该引发错误
            self.check_nondeterministic_alert(
                lambda: op_call(a, weights),
                '_bincount_cuda',
                torch.device(device).type == 'cuda')

            # 当不满足 CUDA 设备条件时，不应引发错误
            self.check_nondeterministic_alert(
                lambda: op_call(a),
                '_bincount_cuda',
                False)

    # 确保在正确情况下 kthvalue 抛出非确定性警报
    @dtypes(torch.double)
    def test_nondeterministic_alert_kthvalue(self, device, dtype):
        # 定义一个测试函数，根据不同的调用类型调用 torch.kthvalue
        def test_func(call_type):
            S = 10
            k = 5
            # 创建一个在指定设备上的随机张量 a
            a = torch.randn(S, device=device)
            if call_type == 'function':
                # 调用 torch.kthvalue 函数
                torch.kthvalue(a, k)
            elif call_type == 'method':
                # 调用 torch.kthvalue 方法
                a.kthvalue(k)
            elif call_type == 'out':
                # 准备用于输出的 values 张量，并指定设备
                values = torch.empty_like(a)
                # 准备用于索引的 indices 张量，并指定设备和数据类型
                indices = torch.empty((), device=device, dtype=torch.long)
                # 调用 torch.kthvalue，并将结果存储在指定的 values 和 indices 中
                torch.kthvalue(a, k, out=(values, indices))
            else:
                # 如果调用类型无效，则标记为失败
                self.fail(f"'{call_type}' is not a valid call type")

        # 对于每种调用类型，检查在 CUDA 设备上是否会引发非确定性警报
        for call_type in ['function', 'method', 'out']:
            self.check_nondeterministic_alert(
                lambda: test_func('function'),
                'kthvalue CUDA',
                torch.device(device).type == 'cuda')

    # 在满足特定条件下确保 grid_sample 2D 后向传播时引发非确定性警报
    @skipIfMps
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_2d(self, device):
        # 创建一个空的具有梯度的输入张量 input，并指定设备
        input = torch.empty(1, 1, 2, 2, device=device, requires_grad=True)
        # 创建一个空的 grid 张量，并指定设备
        grid = torch.empty(1, 1, 1, 2, device=device)
        # 执行 grid_sample 函数，并指定是否需要保持角点对齐
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        # 创建一个与 res 张量相同形状的梯度张量 grad
        grad = torch.ones_like(res)

        # 检查在 CUDA 设备上执行后向传播时是否会引发非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_2d_backward_cuda',
            torch.device(device).type == 'cuda')

    # 在满足特定条件下确保 grid_sample 3D 后向传播时引发非确定性警报
    @skipIfMps
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_3d(self, device):
        # 创建一个空的具有梯度的 3D 输入张量 input，并指定设备
        input = torch.empty(1, 1, 2, 2, 2, device=device, requires_grad=True)
        # 创建一个空的 3D grid 张量，并指定设备
        grid = torch.empty(1, 1, 1, 2, 3, device=device)
        # 执行 grid_sample 函数，并指定是否需要保持角点对齐
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        # 创建一个与 res 张量相同形状的梯度张量 grad
        grad = torch.ones_like(res)

        # 检查在 CUDA 设备上执行后向传播时是否会引发非确定性警报
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_3d_backward_cuda',
            torch.device(device).type == 'cuda')
    # 测试无效形状的网格采样器函数，接受设备参数
    def test_invalid_shapes_grid_sampler(self, device):
        # 偏函数，用于创建张量，指定设备、数据类型为双精度浮点数，需要梯度
        make_arg = partial(
            make_tensor, device=device, dtype=torch.float64, requires_grad=True)

        # 定义输入参数组合
        inputs = (
            # input, grid
            ((5, 5, 5, 5, 5,), (1, 1, 1, 4, 4,)),  # 3维
            ((5, 5, 5, 5,), (1, 1, 4, 4,)),  # 2维
        )

        # 插值模式和填充模式设为0
        interpolation_mode = 0
        padding_mode = 0
        align_corners = True

        # 期望的错误信息
        err = "expected grid and input to have same batch size"

        # 遍历每组输入参数
        for input, grid in inputs:
            # 创建输入张量
            input = make_arg(input)
            # 创建网格张量，并限定值域在[-1, 1]之间
            grid = make_arg(grid, low=-1, high=1)

            # 使用断言检查运行时错误，期望抛出特定错误信息
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # 期望2维输入，检查是否抛出错误
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_2d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # 期望3维输入，检查是否抛出错误
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_3d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # 期望2维输入，CPU环境下使用的函数，检查是否抛出错误
            with self.assertRaisesRegex(RuntimeError, err):
                torch._grid_sampler_2d_cpu_fallback(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # 期望2维输入，CUDA环境下使用的函数，检查是否抛出错误
            # 不适用于CPU和ROCm
            if device != 'cpu' and TEST_CUDNN and not TEST_WITH_ROCM:
                with self.assertRaisesRegex(RuntimeError, err):
                    torch.cudnn_grid_sampler(input, grid)

    # 测试距离函数
    def test_dist(self, device):
        # 定义测试函数，计算两个张量之间的距离
        def run_test(x, y):
            # 遍历不同的p值
            for p in [0, 1, 2, 3, 4, inf, -inf]:
                # 计算张量x和y之间的距离
                dist_xy = torch.dist(x, y, p)
                # 计算张量x和y之间的范数距离
                dist_xy_norm = torch.norm(x - y, p)
                # 使用断言检查它们的距离是否相等
                self.assertEqual(dist_xy, dist_xy_norm)

        # 在指定设备上运行第一组测试
        run_test(torch.randn(5, device=device), torch.randn(5, device=device))

        # 创建零张量x和y，并使得y的第二个元素为1
        x = torch.zeros(3, device=device)
        y = torch.zeros(3, device=device)
        y[1] = 1.
        # 在指定设备上运行第二组测试
        run_test(x, y)

    # 确保中位数在正确情况下引发非确定性警报
    @dtypes(torch.double)
    # 测试不确定性警告的中位数计算函数
    def test_nondeterministic_alert_median(self, device, dtype):
        # 内部函数，根据调用类型测试中位数计算
        def test_func(call_type):
            # 设置张量长度为 10
            S = 10
            # 生成在设备上的随机张量
            a = torch.randn(S, device=device)
            # 根据调用类型选择计算中位数的方式
            if call_type == 'function':
                torch.median(a)
            elif call_type == 'function with indices':
                torch.median(a, 0)
            elif call_type == 'method':
                a.median()
            elif call_type == 'method with indices':
                a.median(0)
            elif call_type == 'out with indices':
                # 准备用于输出的张量和索引
                result = torch.empty_like(a)
                indices = torch.empty((), dtype=torch.long, device=device)
                torch.median(a, 0, out=(result, indices))
            else:
                self.fail(f"'{call_type}' is not a valid call type")

        # 内部函数，测试预期是否出现错误警告
        def test_func_expect_error(call_type, should_error):
            self.check_nondeterministic_alert(
                lambda: test_func(call_type),
                'median CUDA with indices output',
                should_error)

        # 检查设备是否为 CUDA
        is_cuda = torch.device(device).type == 'cuda'

        # 对各种调用类型进行测试并期望不出现错误
        test_func_expect_error('function', False)
        test_func_expect_error('function with indices', is_cuda)
        test_func_expect_error('method', False)
        test_func_expect_error('method with indices', is_cuda)
        test_func_expect_error('out with indices', is_cuda)

    # FIXME: move to test_scatter_gather_ops
    # 测试在指定设备上进行 gather 操作的反向传播
    def _test_gather_backward_one_dim(self, device, deterministic: bool = False) -> None:
        # 使用确定性保护上下文管理器
        with DeterministicGuard(deterministic):
            # 随机生成参数
            m = random.randint(2000, 3000)
            elems = random.randint(10 * m, 20 * m)
            dim = 0
            # 在指定设备上生成随机张量并启用梯度追踪
            src = torch.randn(m, device=device, requires_grad=True)
            idx = torch.randint(m, (elems,), device=device)
            # 执行 gather 操作
            res = torch.gather(src, dim, idx)
            # 准备权重张量
            weight = torch.rand_like(res, device=device) * 10 ** 6
            # 执行反向传播
            res.backward(weight)
            assert src.grad is not None
            grad = src.grad.detach().clone()

            # 如果设备为 CUDA
            if torch.device(device).type == 'cuda':
                # 对于 CUDA 设备，进行多次反向传播的验证
                for _ in range(2):
                    src.grad.data.zero_()
                    res = torch.gather(src, dim, idx)
                    res.backward(weight)
                    self.assertEqual(src.grad, grad, atol=0, rtol=0)
            else:
                # 对于其他设备，验证梯度计算的正确性
                expected = torch.zeros_like(src, device=device)
                for i in range(elems):
                    expected[idx[i]] += weight[i]
                self.assertEqual(grad, expected, atol=0, rtol=0)

    # FIXME: move to test_scatter_gather_ops
    # 在本地设备上测试 gather 操作的确定性路径
    @onlyNativeDeviceTypes
    def test_gather_backward_deterministic_path(self, device) -> None:
        self._test_gather_backward_one_dim(device, True)

    # FIXME: move to test_scatter_gather_ops
    # 在 CPU 上测试 gather 操作的单维反向传播
    @onlyCPU
    def test_gather_backward_one_dim(self, device) -> None:
        self._test_gather_backward_one_dim(device, False)

    # FIXME: move to test_scatter_gather_ops
    # 使用装饰器确保只在本机设备类型上运行测试
    @onlyNativeDeviceTypes
    # 定义测试函数，验证在一维张量上的确定性 scatter_add 操作
    def test_scatter_add_one_dim_deterministic(self, device) -> None:
        # 使用确定性保护，确保生成的随机数固定
        with DeterministicGuard(True):
            # 随机生成一个数，范围在 [20, 30] 之间
            m = random.randint(20, 30)
            # 计算元素数量，范围在 [2000*m, 3000*m] 之间
            elems = random.randint(2000 * m, 3000 * m)
            # 定义操作的维度
            dim = 0
            # 在指定设备上生成随机张量
            src = torch.randn(elems, device=device)
            # 在指定设备上生成随机索引张量
            idx = torch.randint(m, (elems,), device=device)

            # 创建一个全零张量 x
            x = torch.zeros(m, device=device)
            # 使用 scatter_add 函数执行操作，并返回结果
            res = x.scatter_add(dim, idx, src)

            # 检查 scatter_add 是否是确定性的
            for i in range(5):
                # 再次执行 scatter_add 操作
                res_next = x.scatter_add(dim, idx, src)
                # 检查两次结果是否相等
                self.assertEqual(res, res_next, atol=0, rtol=0)
                # 更新当前结果
                res = res_next

            # 生成期望结果张量，用于后续比较
            expected = torch.zeros(m, device=device)
            for i in range(elems):
                expected[idx[i]] += src[i]

            # 检查最终的结果是否与期望一致
            self.assertEqual(res, expected, atol=1e-4, rtol=1e-5)

    # FIXME: 将此测试移到 test_scatter_gather_ops 中
    @onlyNativeDeviceTypes
    # 定义测试函数，验证在零大小索引情况下的 scatter 操作
    def test_scatter_zero_size_index(self, device) -> None:
        # 创建一个零大小的索引张量和对应的零大小的数值张量
        null_index = torch.zeros((0, 4), dtype=torch.int64)
        null_arr = torch.zeros((0, 4))
        # 创建一个原始张量
        original = torch.arange(4, dtype=torch.float32)
        # 执行 scatter 操作，并获得结果
        result = original.scatter(0, null_index, null_arr)
        # 检查结果是否与原始张量相等
        self.assertEqual(result, original, atol=0, rtol=0)

    # 只在 CUDA 设备上运行的装饰器
    @onlyCUDA
    # 如果 Torch Inductor 存在 FIXME，跳过测试
    @skipIfTorchInductor("FIXME")
    # 定义一个测试方法，用于测试同步警告
    def test_sync_warning(self, device):
        
        # 定义内部辅助函数，测试同步操作是否会引发警告
        def _sync_raises_helper(f, level):
            # 使用 CUDA 同步保护器，根据级别执行不同的测试
            with CudaSyncGuard(level):
                if level == 1:
                    # 断言调用 f() 时会产生 UserWarning，并且警告信息包含特定字符串
                    with self.assertWarnsRegex(UserWarning, "called a synchronizing "):
                        f()
                elif level == 2:
                    # 断言调用 f() 时会引发 RuntimeError，并且异常信息包含特定字符串
                    with self.assertRaisesRegex(RuntimeError, "called a synchronizing "):
                        f()

        # 定义内部辅助函数，测试同步操作不会引发警告
        def _no_sync_helper(f, level):
            # 使用 CUDA 同步保护器，根据级别执行操作
            with CudaSyncGuard(level):
                f()

        # 定义操作函数：在张量的指定索引处放置值并返回
        def _ind_put_fn(x, ind, val):
            x[ind] = val
            return x

        # 定义操作函数：获取张量的指定索引处的值并返回
        def _ind_get_fn(x, ind):
            return x[ind]

        # 定义条件函数：根据输入值 x 的真假决定返回 x 或者 2 * x
        def _cond_fn(x):
            if x:  # 获取张量的布尔值会导致同步
                return x
            else:
                return 2 * x

        # 准备后续操作的输入数据
        size = 4
        x = torch.rand(size, device=device)
        y = torch.rand((), device=device)
        ind = torch.randint(size, (3,), device=device)
        ind_cpu = ind.cpu()
        repeats = torch.full((1,), 2, device=device)
        mask = torch.randint(2, (size,), device=device, dtype=bool)
        
        # 定义预期不会同步的操作列表，每个元素是一个 lambda 函数
        expect_no_sync = (lambda: _ind_put_fn(x, mask, 1.),
                          lambda: _ind_put_fn(x, ind, y),
                          lambda: _ind_get_fn(x, ind),
                          lambda: torch.nn.functional.one_hot(ind, num_classes=size),
                          lambda: torch.randperm(20000, device=device),
                          lambda: torch.repeat_interleave(x, 2, output_size=2 * size),
                          lambda: torch.repeat_interleave(x, repeats, output_size=2 * size),
                          lambda: torch.any(y))
        
        # 定义预期会同步的操作列表，每个元素是一个 lambda 函数
        expect_sync = (lambda: _ind_put_fn(x, mask, y),
                       lambda: _ind_put_fn(x, ind_cpu, y),
                       lambda: _ind_get_fn(x, mask),
                       lambda: _ind_get_fn(x, ind_cpu),
                       lambda: x.nonzero(),
                       lambda: _cond_fn(y),
                       lambda: torch.nn.functional.one_hot(ind),
                       lambda: torch.repeat_interleave(x, repeats))
        
        # 遍历预期不会同步的操作和级别，执行 _no_sync_helper 函数进行测试
        for f, level in product(expect_no_sync, (1, 2)):
            _no_sync_helper(f, level)
        
        # 遍历预期会同步的操作和级别，执行 _sync_raises_helper 函数进行测试
        for f, level in product(expect_sync, (1, 2)):
            _sync_raises_helper(f, level)
    # 定义一个测试方法，用于测试 repeat_interleave 函数的功能
    def test_repeat_interleave(self, device):
        # 创建一个张量 y，包含两行两列的数据，指定设备为参数 device
        y = torch.tensor([[1, 2], [3, 4]], device=device)
        # 调用 repeat_interleave 函数，重复张量 y 中的每个元素两次
        temp = y.repeat_interleave(2)
        # 断言 temp 的尺寸为 [8]
        self.assertEqual(torch.Size([8]), temp.size())

        # 遍历浮点数数据类型 [torch.int, torch.long]
        for dtype in [torch.int, torch.long]:
            # 创建一个长度为 [1, 2] 的张量 lengths，数据类型为 dtype，指定设备为参数 device
            lengths = torch.tensor([1, 2], dtype=dtype, device=device)
            # 计算 lengths 中所有元素的和
            output_size = torch.sum(lengths)
            # 使用 repeat_interleave 函数，根据 lengths 指定的重复次数在维度 dim=0 上重复张量 y
            a = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
            )
            # 断言 a 的数据类型与 y 相同
            self.assertEqual(a.dtype, y.dtype)
            # 断言 a 的尺寸为 [3, 2]
            self.assertEqual(a.size(), torch.Size([3, 2]))

            # 使用 repeat_interleave 函数，根据 lengths 指定的重复次数在维度 dim=0 上重复张量 y
            # 同时指定 output_size 参数为计算得到的总长度
            a_with_output = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
                output_size=output_size,
            )
            # 断言 a_with_output 的数据类型与 y 相同
            self.assertEqual(a_with_output.dtype, y.dtype)
            # 断言 a_with_output 的尺寸为 [3, 2]

    # 使用 dtypes 装饰器，测试 Bernoulli 分布函数的功能，对浮点数数据类型进行测试
    @dtypes(*floating_types())
    # 使用 dtypesIfCPU 装饰器，测试 Bernoulli 分布函数的功能，对浮点数数据类型和半精度数据类型进行测试
    @dtypesIfCPU(*floating_types_and(torch.bfloat16, torch.half))
    # 使用 dtypesIfCUDA 装饰器，测试 Bernoulli 分布函数的功能，对浮点数数据类型和半精度数据类型进行测试
    @dtypesIfCUDA(*floating_types_and(torch.half))
    # 定义测试 Bernoulli 分布函数的方法，传入设备和数据类型作为参数
    def test_bernoulli_p(self, device, dtype):
        # 遍历不同的 trivial_p 值，对每一个值进行测试
        for trivial_p in ([0, 1], [1, 0, 1, 1, 0, 1]):
            # 创建一个张量 x，数据类型为 dtype，数据值为 trivial_p，指定设备为参数 device
            x = torch.tensor(trivial_p, dtype=dtype, device=device)
            # 断言 x 使用 Bernoulli 函数后得到的结果与 trivial_p 相同
            self.assertEqual(x.bernoulli().tolist(), trivial_p)

        # 定义一个函数 isBinary，用于检查张量 t 中的元素是否只包含二进制值
        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        # 创建一个大小为 [5, 5] 的随机张量 p，数据类型为 dtype，指定设备为参数 device
        p = torch.rand(5, 5, dtype=dtype, device=device)
        # 断言 p 使用 Bernoulli 函数后得到的结果是二进制值
        self.assertTrue(isBinary(p.bernoulli()))

        # 创建一个大小为 [5] 的随机张量 p，数据类型为 dtype，指定设备为参数 device，并将其扩展为大小为 [5, 5] 的张量
        p = torch.rand(5, dtype=dtype, device=device).expand(5, 5)
        # 断言 p 使用 Bernoulli 函数后得到的结果是二进制值
        self.assertTrue(isBinary(p.bernoulli()))

        # 创建一个大小为 [5, 5] 的随机张量 p，数据类型为 dtype，指定设备为参数 device
        p = torch.rand(5, 5, dtype=dtype, device=device)
        # 使用 Bernoulli 函数，并传入一个与 p 相同大小的随机张量，将结果保存在 p 中
        torch.bernoulli(torch.rand_like(p), out=p)
        # 断言 p 中的元素是二进制值
        self.assertTrue(isBinary(p))

    # 在 XLA 测试中，对于整数类型，未实现 RngUniform
    # 使用 dtypes 装饰器，测试 Bernoulli 分布函数的功能，对浮点数数据类型进行测试
    @dtypes(*floating_types())
    # 使用 dtypesIfCPU 装饰器，测试 Bernoulli 分布函数的功能，对所有数据类型和半精度数据类型进行测试
    @dtypesIfCPU(*all_types_and(torch.bool, torch.half))
    # 使用 dtypesIfCUDA 装饰器，测试 Bernoulli 分布函数的功能，对所有数据类型和半精度数据类型进行测试
    @dtypesIfCUDA(*all_types_and(torch.bool, torch.half))
    # 定义测试 Bernoulli 分布函数的方法，传入设备和数据类型作为参数
    def test_bernoulli_self(self, device, dtype):
        # 定义一个函数 isBinary，用于检查张量 t 中的元素是否只包含二进制值
        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        # 创建一个大小为 [10, 10] 的空张量 t，数据类型为 dtype，指定设备为参数 device
        t = torch.empty(10, 10, dtype=dtype, device=device)

        # 将张量 t 填充为常数值 2
        t.fill_(2)
        # 使用 Bernoulli_ 函数，根据概率 0.5 在张量 t 中生成 Bernoulli 分布的随机数
        t.bernoulli_(0.5)
        # 断言 t 中的元素是二进制值
        self.assertTrue(isBinary(t))

        # 遍历浮点数数据类型中的每一种 p_dtype
        for p_dtype in floating_types_and(*[torch.half] if device.startswith('cuda') else []):
            # 创建一个大小为 [10] 的随机张量 p，数据类型为 p_dtype，指定设备为参数 device，并将其扩展为大小为 [10, 10] 的张量
            p = torch.rand(10, dtype=p_dtype, device=device).expand(10, 10)
            # 将张量 t 填充为常数值 2
            t.fill_(2)
            # 使用 Bernoulli_ 函数，根据张量 p 中的概率生成 Bernoulli 分布的随机数，并保存在张量 t 中
            t.bernoulli_(p)
            # 断言 t 中的元素是二进制值
            self.assertTrue(isBinary(t))

            # 将张量 t 填充为常数值 2
            t.fill_(2)
            # 使用 Bernoulli 函数，根据与张量 p 相同大小的随机张量生成 Bernoulli 分布的随机数，并保存在张量 t 中
            torch.bernoulli(torch.rand_like(t, dtype=p_dtype), out=t)
            # 断言 t 中的元素是二进制值
            self.assertTrue(isBinary(t))

            # 将张量 t 填充为常数值 2
            t.fill_(2)
            # 使用 Bernoulli_ 函数，根据与张量 p 相同大小的随机张量生成 Bernoulli 分布的随机数，并保存在张量 t 中
            t.bernoulli_(torch.rand_like(t, dtype=p_dtype))
            # 断言 t 中的元素是二进制值
            self.assertTrue(isBinary(t))

    # 标
    # 测试 Bernoulli 分布的边界情况
    def test_bernoulli_edge_cases(self, device, dtype):
        # 创建全零张量，概率为 0，不可能取到值为 "1"
        a = torch.zeros(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 0
        # 统计张量中值为 "1" 的数量
        num_ones = (torch.bernoulli(a) == 1).sum()
        # 断言张量中值为 "1" 的数量为 0
        self.assertEqual(num_ones, 0)

        # 创建全一张量，概率为 1，不可能取到值为 "0"
        b = torch.ones(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 1
        # 统计张量中值为 "0" 的数量
        num_zeros = (torch.bernoulli(b) == 0).sum()
        # 断言张量中值为 "0" 的数量为 0
        self.assertEqual(num_zeros, 0)

    # 测试指数分布的特殊情况
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    def test_exponential(self, device, dtype):
        # 创建指定参数的指数分布张量 a
        a = torch.tensor([10], dtype=dtype, device=device).exponential_(0.5)
        # 断言张量的数据类型与预期一致
        self.assertEqual(a.dtype, dtype)
        # 断言张量的形状为 [1]
        self.assertEqual(a.size(), torch.Size([1]))

        # 测试极端情况，创建无穷大参数的指数分布张量 t
        t = torch.empty((1,), device=device, dtype=dtype).exponential_(float('inf'))
        # 断言张量 t 的值为 0
        self.assertTrue(t.item() == 0)

        # 测试负参数会引发 RuntimeError 的情况
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device=device, dtype=dtype).exponential_(-0.5)

    # 在 CUDA 设备上测试指数分布，确保不生成值为 0 的样本
    @onlyCUDA
    @dtypes(torch.half, torch.float)
    def test_exponential_no_zero(self, device, dtype):
        # 生成大量样本以检查不会生成值为 0 的情况
        x = torch.empty(50000000, device=device, dtype=dtype).exponential_()
        # 断言生成的最小值大于 0
        self.assertTrue(x.min() > 0)

    # 生成用于相关系数计算的张量
    def _generate_correlation_tensors(self, device, dtype):
        # 生成不同形状的张量
        yield make_tensor((0, 0), dtype=dtype, device=device)
        yield make_tensor((1, 0), dtype=dtype, device=device)
        yield make_tensor((0, 1), dtype=dtype, device=device)
        yield make_tensor((2,), dtype=dtype, device=device)
        yield make_tensor((2, 1), dtype=dtype, device=device)
        yield make_tensor((2, 2), dtype=dtype, device=device)
        yield make_tensor((2, 3), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        # 如果数据类型不是整数，生成包含特殊值的张量
        if dtype != torch.int:
            yield torch.tensor([0, -2, nan, 10.2, inf], dtype=dtype, device=device)

    # 在本机设备类型上测试相关系数计算
    @onlyNativeDeviceTypes
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_corrcoef(self, device, dtype):
        # 遍历生成的张量进行相关系数计算，并与 NumPy 的结果进行比较
        for x in self._generate_correlation_tensors(device, dtype):
            res = torch.corrcoef(x)
            ref = np.corrcoef(x.cpu().numpy())
            # 断言 Torch 计算的相关系数与 NumPy 的计算结果一致（允许数据类型不同）
            self.assertEqual(res, ref, exact_dtype=False)

    # 在 ROCm 上，如果使用 Torch Inductor 库，则跳过测试相关系数计算
    @skipRocmIfTorchInductor
    @dtypes(torch.int, torch.float, torch.cfloat)
    # 定义测试函数 `test_cov`，用于测试 `torch.cov` 函数的正确性
    def test_cov(self, device, dtype):
        
        # 定义内部函数 `check`，用于比较 `torch.cov` 和 `np.cov` 的结果
        def check(t, correction=1, fweights=None, aweights=None):
            # 计算 `torch.cov` 的结果
            res = torch.cov(t, correction=correction, fweights=fweights, aweights=aweights)
            # 将输入张量转换为 numpy 数组
            t = t.cpu().numpy()
            # 如果存在频率权重，将其转换为 numpy 数组
            fweights = fweights.cpu().numpy() if fweights is not None else None
            # 如果存在观测权重，将其转换为 numpy 数组
            aweights = aweights.cpu().numpy() if aweights is not None else None
            # 计算 `np.cov` 的参考结果
            ref = np.cov(t, ddof=correction, fweights=fweights, aweights=aweights)
            # 使用断言检查 `torch.cov` 和 `np.cov` 的结果是否相等
            self.assertEqual(res, ref, atol=1e-05, rtol=1e-05, exact_dtype=False)

        # 对生成的相关张量进行测试
        for x in self._generate_correlation_tensors(device, dtype):
            # 对单个张量调用 `check` 函数
            check(x)
            # 计算张量中的观测数量
            num_observations = x.numel() if x.ndim < 2 else x.size(1)
            # 如果观测数量大于 0
            if num_observations > 0:
                # 生成随机的频率权重
                fweights = torch.randint(1, 10, (num_observations,), device=device)
                # 生成随机的观测权重
                aweights = make_tensor((num_observations,), dtype=torch.float, device=device, low=1)
                # 对不同的校正值、频率权重、观测权重组合进行测试
                for correction, fw, aw in product([0, 1, 2], [None, fweights], [None, aweights]):
                    # 调用 `check` 函数进行测试
                    check(x, correction, fweights, aweights)

    # 使用装饰器 `skipIfNoSciPy` 跳过没有安装 SciPy 的情况，并指定支持的浮点数类型
    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    # 定义测试函数 `test_uniform_kstest`，用于测试 `torch` 中的均匀分布检验
    def test_uniform_kstest(self, device, dtype):
        # 导入 SciPy 中的统计模块
        from scipy import stats
        # 设置数据的大小
        size = 1000
        # 对不同的起始值和结束值进行测试
        for from_ in [-42, 0, 4.2]:
            for to_ in [-4.2, 0, 42]:
                # 如果结束值大于起始值
                if to_ > from_:
                    # 生成在指定范围内均匀分布的张量
                    t = torch.empty(size, dtype=dtype, device=device).uniform_(from_, to_)
                    # 使用 Kolmogorov-Smirnov 检验对生成的张量进行测试
                    res = stats.kstest(t.cpu().to(torch.double), 'uniform', args=(from_, (to_ - from_)))
                    # 使用断言检查统计量是否小于 0.1
                    self.assertTrue(res.statistic < 0.1)

    # 使用装饰器 `skipIfNoSciPy` 跳过没有安装 SciPy 的情况，并指定支持的浮点数类型
    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half))
    # 使用装饰器 `dtypesIfCUDA` 对 CUDA 环境下的数据类型进行选择
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    # 定义测试函数 `test_normal_kstest`，用于测试 `torch` 中的正态分布检验
    def test_normal_kstest(self, device, dtype):
        # 导入 SciPy 中的统计模块
        from scipy import stats
        # 设置数据的大小
        size = 1000
        # 对不同的均值和标准差进行测试
        for mean in [-10, 0, 50]:
            for std in [1, 5, 10]:
                # 生成指定均值和标准差的正态分布张量
                t = torch.empty(size, dtype=dtype, device=device).normal_(mean=mean, std=std)
                # 使用 Kolmogorov-Smirnov 检验对生成的张量进行测试
                res = stats.kstest(t.cpu().to(torch.double), 'norm', args=(mean, std))
                # 使用断言检查统计量是否小于 0.1
                self.assertTrue(res.statistic < 0.1)

    # 使用装饰器 `skipIfMps` 跳过在 MPS 环境下的测试
    # 使用装饰器 `skipIfNoSciPy` 跳过没有安装 SciPy 的情况，并指定支持的浮点数类型
    @skipIfMps
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    # 定义测试函数 `test_lognormal_kstest`，用于测试 `torch` 中的对数正态分布检验
    def test_lognormal_kstest(self, device, dtype):
        # 导入 SciPy 中的统计模块
        from scipy import stats
        # 设置数据的大小
        size = 1000
        # 对不同的均值和标准差进行测试
        for mean in [-3, 0, 7]:
            for std in [1, 5, 7]:
                # 生成指定均值和标准差的对数正态分布张量
                t = torch.empty(size, dtype=dtype, device=device).log_normal_(mean=mean, std=std)
                # 使用 Kolmogorov-Smirnov 检验对生成的张量进行测试
                res = stats.kstest(t.cpu().to(torch.double), 'lognorm', args=(std, 0, math.exp(mean)))
                # 根据数据类型选择不同的断言条件
                if dtype == torch.half:
                    # 使用断言检查统计量是否小于 0.3
                    self.assertTrue(res.statistic < 0.3)
                else:
                    # 使用断言检查统计量是否小于 0.1
                    self.assertTrue(res.statistic < 0.1)
    # 对指数分布进行 Kolmogorov-Smirnov 检验的测试方法
    def test_exponential_kstest(self, device, dtype):
        # 导入 scipy 的统计模块
        from scipy import stats
        # 设置样本大小
        size = 1000
        # 针对不同的 lambda 值进行循环测试
        for lambd in [0.5, 1.0, 5.0]:
            # 生成符合指数分布的随机数，并转换为指定的数据类型和设备
            t = torch.empty(size, dtype=dtype, device=device).exponential_(lambd=lambd)
            # 进行 Kolmogorov-Smirnov 检验，检验随机数是否符合指数分布
            res = stats.kstest(t.cpu().to(torch.double), 'expon', args=(0, 1 / lambd,))
            # 断言 KS 统计量小于 0.1
            self.assertTrue(res.statistic < 0.1)

    # 对柯西分布进行 Kolmogorov-Smirnov 检验的测试方法
    @skipIfMps
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy_kstest(self, device, dtype):
        # 导入 scipy 的统计模块
        from scipy import stats
        # 设置样本大小
        size = 1000
        # 针对不同的中位数和标准差进行循环测试
        for median in [-10, 0, 50]:
            for sigma in [0.5, 1.0, 10.0]:
                # 生成符合柯西分布的随机数，并转换为指定的数据类型和设备
                t = torch.empty(size, dtype=dtype, device=device).cauchy_(median=median, sigma=sigma)
                # 进行 Kolmogorov-Smirnov 检验，检验随机数是否符合柯西分布
                res = stats.kstest(t.cpu().to(torch.double), 'cauchy', args=(median, sigma))
                # 断言 KS 统计量小于 0.1
                self.assertTrue(res.statistic < 0.1)

    # 测试柯西分布生成的随机数中不包含无穷大的情况
    @slowTest
    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float32)
    def test_cauchy_no_inf(self, device, dtype):
        # 对于每一次循环，生成指定大小的随机数，并检查是否包含无穷大
        for _ in range((2**16) * 2):
            x = torch.empty((2**16), dtype=dtype, device=device)
            x.cauchy_()
            # 断言生成的随机数不包含无穷大
            self.assertFalse(x.isinf().sum())

    # 测试柯西分布的基本功能
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy(self, device, dtype):
        # 生成符合柯西分布的张量，并检查其数据类型和大小
        a = torch.tensor([10], dtype=dtype, device=device).cauchy_(0.0, 0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

        # 测试极端行为，生成具有无穷大中位数的柯西分布的随机数
        t = torch.empty((1,), device=device, dtype=dtype).cauchy_(float('inf'), 0.5)
        self.assertTrue(t.item() == float('inf'))

        # 测试非正参数失败的情况，预期会引发 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device=device, dtype=dtype).cauchy_(0.0, 0.0)

    # 对几何分布进行 Kolmogorov-Smirnov 检验的测试方法
    @skipIfMps
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_geometric_kstest(self, device, dtype):
        # 导入 scipy 的统计模块
        from scipy import stats
        # 设置样本大小
        size = 1000
        # 针对不同的几何分布参数 p 进行循环测试
        for p in [0.2, 0.5, 0.8]:
            # 生成符合几何分布的随机数，并转换为指定的数据类型和设备
            t = torch.empty(size, dtype=dtype, device=device).geometric_(p=p)
            # 计算实际数据的直方图
            actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
            # 计算预期的几何分布的概率质量函数
            expected = stats.geom(p).pmf(np.arange(1, 99)) * size
            # 进行卡方检验，检验实际数据和预期数据的拟合度
            res = stats.chisquare(actual, expected)
            # 断言 p 值等于 1.0，允许的误差为 0.1
            self.assertEqual(res.pvalue, 1.0, atol=0.1, rtol=0)

    # FIXME: find test suite for pdist and cdist
    # 定义一个测试方法，用于测试在空数据条件下的 torch.pairwise_distance 函数
    def test_pairwise_distance_empty(self, device):
        # 设定张量的形状为 (2, 0)，创建在指定设备上的随机张量 x 和 y
        shape = (2, 0)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)

        # 断言计算得到的 pairwise_distance 结果为全零张量
        self.assertEqual(torch.zeros(2, device=device), torch.pairwise_distance(x, y))
        # 断言计算得到的 pairwise_distance 结果为形状为 (2, 1) 的全零张量，并保持维度
        self.assertEqual(torch.zeros((2, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

        # 重新设定张量的形状为 (0, 2)，创建在指定设备上的随机张量 x 和 y
        shape = (0, 2)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)
        # 断言计算得到的 pairwise_distance 结果为全零张量
        self.assertEqual(torch.zeros(0, device=device), torch.pairwise_distance(x, y))
        # 断言计算得到的 pairwise_distance 结果为形状为 (0, 1) 的全零张量，并保持维度
        self.assertEqual(torch.zeros((0, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

    # 定义一个测试方法，用于测试在空数据条件下的 torch.pdist 函数
    def test_pdist_empty(self, device):
        # 设定张量的形状为 (0, 2)，创建在指定设备上的随机张量 x
        shape = (0, 2)
        x = torch.randn(shape, device=device)
        # 断言计算得到的 pdist 结果为形状为 (0) 的空张量
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        # 重新设定张量的形状为 (1, 2)，创建在指定设备上的随机张量 x
        shape = (1, 2)
        x = torch.randn(shape, device=device)
        # 断言计算得到的 pdist 结果为形状为 (0) 的空张量
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        # 重新设定张量的形状为 (3, 0)，创建在指定设备上的随机张量 x
        shape = (3, 0)
        x = torch.randn(shape, device=device)
        # 断言计算得到的 pdist 结果为形状为 (3) 的全零张量
        self.assertEqual(torch.zeros(3, device=device), torch.pdist(x))

    # 定义一个测试方法，用于测试在空数据条件下的 torch.cdist 函数
    def test_cdist_empty(self, device):
        # 创建在指定设备上的空张量 x 和非空张量 y
        x = torch.randn((0, 5), device=device)
        y = torch.randn((4, 5), device=device)
        # 断言计算得到的 cdist 结果为形状为 (0, 4) 的空张量
        self.assertEqual(torch.empty(0, 4, device=device), torch.cdist(x, y))

        # 创建在指定设备上的非空张量 x 和空张量 y
        x = torch.randn((2, 5), device=device)
        y = torch.randn((0, 5), device=device)
        # 断言计算得到的 cdist 结果为形状为 (2, 0) 的空张量
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y))

        # 创建在指定设备上的空张量 x 和 y，它们的列数均为 0
        x = torch.randn((2, 0), device=device)
        y = torch.randn((3, 0), device=device)
        # 断言计算得到的 cdist 结果为形状为 (2, 3) 的全零张量
        self.assertEqual(torch.zeros(2, 3, device=device), torch.cdist(x, y))

        # 创建在指定设备上的空张量 x 和 y，它们的列数均为 0
        x = torch.randn((2, 0), device=device)
        y = torch.randn((0, 0), device=device)
        # 断言计算得到的 cdist 结果为形状为 (2, 0) 的空张量
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y))

    # 定义一个辅助方法，用于在空数据条件下计算两个张量之间的欧几里得距离
    def _brute_cdist(self, x, y, p=2):
        # 获取张量 x 和 y 的倒数第二维度的长度
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        # 若其中一个张量的长度为 0，则返回形状为 (r1, r2) 的空张量
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        # 否则，计算两个张量之间的欧几里得距离并返回结果
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    # 跳过条件为 MPS 的测试用例
    @skipIfMps
    # 定义一个测试方法，用于测试 torch.cdist 函数的不同参数组合下的行为
    def test_cdist_norm(self, device):
        # 外层循环，遍历 r1 取值为 [3, 4, 5, 6]
        for r1 in [3, 4, 5, 6]:
            # 外层循环，遍历 m 取值为 [2, 3, 4, 10]
            for m in [2, 3, 4, 10]:
                # 外层循环，遍历 r2 取值为 [4, 6, 7, 8]
                for r2 in [4, 6, 7, 8]:
                    # 外层循环，遍历 p 取值为 [0, 1, 2, 3, 1.5, 2.5, float('inf')]
                    for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                        # 生成随机张量 x 和 y，形状分别为 (r1, m) 和 (r2, m)，使用给定的设备
                        x = torch.randn(r1, m, device=device)
                        y = torch.randn(r2, m, device=device)
                        # 如果 p 的值为 2
                        if p == 2:
                            # 内层循环，遍历 cm 取值为 ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                # 调用 torch.cdist 函数，使用 p=2 和指定的 compute_mode 计算模式
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                # 调用自定义函数 _brute_cdist 计算预期结果
                                expected = self._brute_cdist(x, y, p=2)
                                # 使用断言检查 actual 和 expected 是否相等，允许的相对误差为 0，绝对误差为 0.02
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            # 否则，调用 torch.cdist 函数，使用给定的 p 计算距离
                            actual = torch.cdist(x, y, p=p)
                            # 调用自定义函数 _brute_cdist 计算预期结果
                            expected = self._brute_cdist(x, y, p=p)
                            # 使用断言检查 actual 和 expected 是否相等
                            self.assertEqual(expected, actual)
    
    @skipIfMps  # 标记为测试前提，跳过 Multiprocess 测试环境
    def test_cdist_norm_batch(self, device):
        # 外层循环，遍历 r1 取值为 [3, 4, 5, 6]
        for r1 in [3, 4, 5, 6]:
            # 外层循环，遍历 m 取值为 [2, 3, 4, 10]
            for m in [2, 3, 4, 10]:
                # 外层循环，遍历 r2 取值为 [4, 6, 7, 8]
                for r2 in [4, 6, 7, 8]:
                    # 外层循环，遍历 p 取值为 [0, 1, 2, 3, 1.5, 2.5, float('inf')]
                    for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                        # 生成随机张量 x 和 y，形状为 (2, 3, 6, r1, m) 和 (2, 3, 6, r2, m)，使用给定的设备
                        x = torch.randn(2, 3, 6, r1, m, device=device)
                        y = torch.randn(2, 3, 6, r2, m, device=device)
                        # 如果 p 的值为 2
                        if p == 2:
                            # 内层循环，遍历 cm 取值为 ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                # 调用 torch.cdist 函数，使用 p=2 和指定的 compute_mode 计算模式
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                # 调用自定义函数 _brute_cdist 计算预期结果
                                expected = self._brute_cdist(x, y, p=2)
                                # 使用断言检查 actual 和 expected 是否相等，允许的相对误差为 0，绝对误差为 0.02
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            # 否则，调用 torch.cdist 函数，使用给定的 p 计算距离
                            actual = torch.cdist(x, y, p=p)
                            # 调用自定义函数 _brute_cdist 计算预期结果
                            expected = self._brute_cdist(x, y, p=p)
                            # 使用断言检查 actual 和 expected 是否相等
                            self.assertEqual(expected, actual)
    
    @onlyCUDA  # 标记为测试前提，仅在 CUDA 可用时执行该测试
    # 定义一个测试函数，用于测试 torch.cdist 的反向传播
    def test_cdist_cuda_backward(self, device):
        # 针对不同的 l1 长度进行循环测试
        for l1 in [1, 511, 513]:
            # 针对不同的 l2 长度进行循环测试
            for l2 in [1, 511, 513]:
                # 针对不同的 p 范数进行循环测试
                for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                    # 生成随机张量 x1 和 y1，要求计算梯度
                    x1 = torch.randn(4, l1, 32, device=device, requires_grad=True)
                    x2 = x1.clone().detach_().requires_grad_()
                    y1 = torch.randn(4, l2, 32, device=device, requires_grad=True)
                    y2 = y1.clone().detach_().requires_grad_()
                    # 当 p 为 2 时，进行特定的计算模式下的测试
                    if p == 2:
                        # 遍历两种计算模式
                        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                            # 使用 torch.cdist 计算距离并求均值，然后反向传播
                            z1 = torch.cdist(x1, y1, p=2, compute_mode=cm).mean()
                            z2 = self._brute_cdist(x2, y2, p=2).mean()
                            z1.backward()  # 对 z1 进行反向传播
                            z2.backward()  # 对 z2 进行反向传播
                            # 断言 x1 和 x2 的梯度相等，精确度为绝对误差 0.001
                            self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                            # 断言 y1 和 y2 的梯度相等，精确度为绝对误差 0.001
                            self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)
                    else:
                        # 对于其他的 p 值，直接计算 torch.cdist 的距离并求均值
                        z1 = torch.cdist(x1, y1, p=p).mean()
                        z2 = self._brute_cdist(x2, y2, p=p).mean()
                        # 断言 x1 和 x2 的梯度相等，精确度为绝对误差 0.001
                        self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                        # 断言 y1 和 y2 的梯度相等，精确度为绝对误差 0.001
                        self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)
    
    # 使用 tf32_on_and_off 和 bf32_on_and_off 装饰器设置浮点数处理精度
    @tf32_on_and_off(0.005)
    @bf32_on_and_off(0.005)
    # 定义一个测试函数，用于测试 torch.cdist 处理大数据集时的正确性
    def test_cdist_large(self, device):
        # 针对不同的计算模式进行循环测试
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            # 生成随机张量 x 和 y，并计算其之间的欧氏距离
            x = torch.randn(1000, 10, device=device)
            y = torch.randn(1000, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)  # 实际使用 torch.cdist 计算距离
            expected = self._brute_cdist(x, y, p=2)  # 使用备选方法计算期望值
            # 断言期望值和实际值相等
            self.assertEqual(expected, actual)
    
    # 使用 @slowTest 装饰器将测试标记为慢速测试
    @slowTest
    @tf32_on_and_off(0.01)
    @bf32_on_and_off(0.01)
    # 定义一个测试函数，用于测试 torch.cdist 在大批量数据集上的正确性
    def test_cdist_large_batch(self, device):
        # 针对不同的计算模式进行循环测试
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            # 生成随机张量 x 和 y，并计算其之间的欧氏距离
            x = torch.randn(4, 3, 1000, 10, device=device)
            y = torch.randn(4, 3, 1000, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)  # 实际使用 torch.cdist 计算距离
            expected = self._brute_cdist(x, y, p=2)  # 使用备选方法计算期望值
            # 断言期望值和实际值相等
            self.assertEqual(expected, actual)
    
    # 使用 tf32_on_and_off 和 bf32_on_and_off 装饰器设置浮点数处理精度
    @tf32_on_and_off(0.005)
    @bf32_on_and_off(0.005)
    # 定义一个测试方法，用于测试 torch.cdist 函数在非连续内存情况下的表现，接受设备参数
    def test_cdist_non_contiguous(self, device):
        # 对于两种计算模式（使用/不使用内存映射），分别进行测试
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            # 创建随机张量 x 和 y，形状分别为 (5, 7) 和 (5, 3)，并将其转置后非连续存储
            x = torch.randn(5, 7, device=device).mT
            y = torch.randn(5, 3, device=device).mT
            # 调用 torch.cdist 计算欧氏距离，使用指定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用自定义函数 _brute_cdist 计算期望的欧氏距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 和 y 都不是连续存储
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            # 断言计算得到的实际值与期望值相等
            self.assertEqual(expected, actual)

            # 创建随机张量 x 和 y，形状分别为 (7, 5) 和 (5, 3)，并将 y 转置后非连续存储
            x = torch.randn(7, 5, device=device)
            y = torch.randn(5, 3, device=device).t()
            # 调用 torch.cdist 计算欧氏距离，使用指定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用自定义函数 _brute_cdist 计算期望的欧氏距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 是连续存储，y 不是连续存储
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            # 断言计算得到的实际值与期望值相等
            self.assertEqual(expected, actual)

            # 创建随机张量 x 和 y，形状分别为 (5, 7) 和 (3, 5)，并将 x 转置后非连续存储
            x = torch.randn(5, 7, device=device).t()
            y = torch.randn(3, 5, device=device)
            # 调用 torch.cdist 计算欧氏距离，使用指定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用自定义函数 _brute_cdist 计算期望的欧氏距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 不是连续存储，y 是连续存储
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            # 断言计算得到的实际值与期望值相等
            self.assertEqual(expected, actual)

    # Maybe merge into OpInfo?
    # 定义一个测试方法，用于测试大规模欧几里德距离计算
    def test_cdist_euclidean_large(self, device):
        # 内部函数，测试大规模欧几里德距离计算
        def _test_euclidean_large_cdist(sizex, sizey=None):
            # 如果未指定 sizey，则默认与 sizex 相同
            if sizey is None:
                sizey = sizex
            # 生成随机张量 x 和 y，设备为指定设备，数据类型为浮点型
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # 为避免极端情况，调整 x 的值
            x = x - (((x - y) < eps).float() * 2 * eps)
            # 设置 x 和 y 允许梯度计算
            x.requires_grad = True
            y.requires_grad = True
            # 计算 x 和 y 之间的欧几里德距离
            dist = torch.cdist(x, y, p=2)
            # 计算损失为距离的总和
            loss = dist.sum()
            # 执行反向传播，验证对大规模矩阵的梯度计算是否有效
            loss.backward()

        # 调用内部函数，测试指定大小的欧几里德距离计算
        _test_euclidean_large_cdist((2000, 5))

    # 确保 cdist 在 p<1 的情况下不会产生 NaN 值
    @skipIfMps
    def test_cdist_grad_p_lt_1_no_nan(self, device):
        # 对不同的 p 值进行迭代测试
        for p in [0.99, 0.7, 0.5, 0.1, 0.01]:
            # 生成随机张量 x，并生成一个稍微偏移的张量 y
            x = torch.randn(1, 2, device=device)
            y = x.clone().detach() + torch.tensor([[1., 0.]], device=device)
            # 允许 x 和 y 计算梯度
            x.requires_grad = True
            y.requires_grad = True
            # 计算 x 和 y 之间的距离，使用指定的 p 范数
            result = torch.cdist(x, y, p=p)
            # 执行反向传播，使用全 1 的梯度值
            result.backward(torch.ones_like(result))
            # 断言梯度中不包含任何 NaN 值
            self.assertFalse(torch.isnan(x.grad).any())
            self.assertFalse(torch.isnan(y.grad).any())

    def test_cdist_same_inputs(self, device):
        # 用于检测 cdist 梯度计算中的问题，当距离为 0 时
        sizex = (1, 27, 32)
        # 对不同的 p 值进行迭代测试
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            # 生成随机张量 x 和 dist_grad，设备为指定设备，数据类型为浮点型
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            # 创建 y 作为 x 的克隆
            y = x.clone()
            eps = 1e-6
            # 允许 x 计算梯度
            x.requires_grad = True
            # 计算 x 和 y 之间的距离
            d = torch.cdist(x, y)
            # 执行反向传播，使用 dist_grad 作为输入梯度
            d.backward(dist_grad)
            # 断言梯度中不包含任何非有限值，如 NaN 或 inf
            assert torch.isfinite(x.grad).all()

    @skipIfMps
    # 定义一个测试函数，测试 torch.cumsum 方法在不同情况下的行为
    def test_cumsum(self, device):
        # 创建一个随机张量 x，形状为 (100, 100)，在指定设备上
        x = torch.rand(100, 100, device=device)
        # 对 x 按行进行累积求和，结果保存在 res1 中
        res1 = torch.cumsum(x, 1)
        # 创建一个空张量 res2，将 x 按行进行累积求和，结果存储在 res2 中
        res2 = torch.tensor([]).to(device)
        torch.cumsum(x, 1, out=res2)
        # 断言 res1 和 res2 相等
        self.assertEqual(res1, res2)
        # 在原地修改 x 按行进行累积求和
        x.cumsum_(1)
        # 断言 res1 和 x 相等
        self.assertEqual(res1, x)

        # 创建一个布尔类型的张量 a
        a = torch.tensor([[True, False, True],
                          [False, False, False],
                          [True, True, True]], device=device)
        # 将布尔型张量 a 转换为字节型张量 b
        b = a.byte()
        # 对张量 a 按列进行累积求和，结果保存在 aRes 中
        aRes = torch.cumsum(a, 0)
        # 对张量 b 按列进行累积求和，结果保存在 bRes 中
        bRes = torch.cumsum(b, 0)
        # 断言 aRes 和 bRes 相等
        self.assertEqual(aRes, bRes)
        # 断言 aRes 和指定张量相等
        self.assertEqual(aRes, torch.tensor([[1, 0, 1],
                                             [1, 0, 1],
                                             [2, 1, 2]]))

        # 对张量 a 按行进行累积求和，结果保存在 aRes 中
        aRes = torch.cumsum(a, 1)
        # 对张量 b 按行进行累积求和，结果保存在 bRes 中
        bRes = torch.cumsum(b, 1)
        # 断言 aRes 和 bRes 相等
        self.assertEqual(aRes, bRes)
        # 断言 aRes 和指定张量相等
        self.assertEqual(aRes, torch.tensor([[1, 1, 2],
                                             [0, 0, 0],
                                             [1, 2, 3]]))

        # 检查在零长度维度上的累积和不会导致反向传播崩溃
        # 同时检查在具有零长度维度的张量上进行其他维度的累积和也能正常工作
        # 还包括其他基本情况的测试套件
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                # 创建一个具有梯度的零张量 raw_tensor
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                # 对 raw_tensor 按指定维度 dim 进行累积求和
                integrated = raw_tensor.cumsum(dim=dim)
                # 检查反向传播不会崩溃
                integrated.sum().backward()
                # 检查输出保持正确的形状
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        # 检查标量的例子
        raw_tensor = torch.tensor(3., requires_grad=True)
        # 对标量进行累积求和，dim=-1 表示沿着最后一个维度，结果保存在 integrated 中
        integrated = raw_tensor.cumsum(dim=-1)
        # 断言 raw_tensor 和 integrated 相等
        self.assertEqual(raw_tensor, integrated)
        # 检查反向传播不会崩溃
        integrated.sum().backward()
        # 检查输出保持正确的形状
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

    @skipIfMps
    # 定义一个测试函数 `test_cumprod`，用于测试 torch.cumprod 函数的各种用例
    def test_cumprod(self, device):
        # 创建一个形状为 (100, 100) 的随机张量 x，并将其发送到指定的设备
        x = torch.rand(100, 100, device=device)
        # 在第二个维度上计算 x 的累积乘积
        res1 = torch.cumprod(x, 1)
        # 创建一个空张量 res2，并将其发送到指定的设备
        res2 = torch.tensor([]).to(device)
        # 如果不是在 torchinductor 下测试，将 x 在第二个维度上的累积乘积存入 res2
        if not TEST_WITH_TORCHINDUCTOR:
            torch.cumprod(x, 1, out=res2)
            # 断言 res1 和 res2 相等
            self.assertEqual(res1, res2)
        # 在原地修改 x，在第二个维度上计算其累积乘积
        x.cumprod_(1)
        # 断言 res1 和修改后的 x 相等
        self.assertEqual(res1, x)
    
        # 创建一个布尔类型的张量 a，形状为 (3, 3)，并将其发送到指定的设备
        a = torch.tensor([[True, False, True],
                          [False, False, False],
                          [True, True, True]], dtype=torch.bool, device=device)
        # 将布尔类型张量 a 转换为字节类型张量 b
        b = a.byte()
        # 计算 a 在第一个维度上的累积乘积
        aRes = torch.cumprod(a, 0)
        # 计算 b 在第一个维度上的累积乘积
        bRes = torch.cumprod(b, 0)
        # 断言 aRes 和 bRes 相等
        self.assertEqual(aRes, bRes)
        # 断言 aRes 和预期的张量相等
        self.assertEqual(aRes, torch.tensor([[1, 0, 1],
                                             [0, 0, 0],
                                             [0, 0, 0]]))
    
        # 计算 a 在第二个维度上的累积乘积
        aRes = torch.cumprod(a, 1)
        # 计算 b 在第二个维度上的累积乘积
        bRes = torch.cumprod(b, 1)
        # 断言 aRes 和 bRes 相等
        self.assertEqual(aRes, bRes)
        # 断言 aRes 和预期的张量相等
        self.assertEqual(aRes, torch.tensor([[1, 0, 0],
                                             [0, 0, 0],
                                             [1, 1, 1]]))
    
        # 针对各种基本情况进行测试，确保在零长度维度上的累积乘积不会导致后向传播崩溃
        # 同时检查具有零长度维度的张量在其他维度上的累积乘积也能正常工作
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                # 创建一个具有指定形状的零张量，并启用梯度跟踪
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                # 在指定维度上计算累积乘积
                integrated = raw_tensor.cumprod(dim=dim)
                # 检查反向传播不会崩溃
                integrated.sum().backward()
                # 检查输出的梯度张量具有与原始张量相同的形状
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)
    
        # 检查一个标量的情况
        # 创建一个标量张量，并启用梯度跟踪
        raw_tensor = torch.tensor(3., requires_grad=True)
        # 在最后一个维度上计算累积乘积
        integrated = raw_tensor.cumprod(dim=-1)
        # 断言原始张量和累积乘积张量相等
        self.assertEqual(raw_tensor, integrated)
        # 检查反向传播不会崩溃
        integrated.sum().backward()
        # 检查输出的梯度张量具有与原始张量相同的形状
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)
    # 定义一个测试函数，测试 logcumsumexp 方法在指定设备上的行为
    def test_logcumsumexp(self, device):
        # 定义一个内部函数 logcumsumexp，用于对输入张量沿指定轴进行累积求和后取对数的操作
        def logcumsumexp(a, axis):
            return torch.cumsum(a.exp(), axis=axis).log_()

        # 指定操作的轴向为最后一个维度
        axis = -1
        # 生成一个随机张量 a，形状为 100x100，在指定设备上生成
        a = torch.randn(100, 100, device=device)

        # 调用实际的 logcumsumexp 方法，计算在指定轴上的对数累积求和操作
        actual = a.logcumsumexp(axis)
        # 使用定义的 logcumsumexp 函数计算预期结果
        expected = logcumsumexp(a, axis)
        # 断言张量 a 的数据类型与计算结果 actual 的数据类型相同
        self.assertEqual(a.dtype, actual.dtype)
        # 断言预期结果的形状与计算结果 actual 的形状相同
        self.assertEqual(expected.shape, actual.shape)
        # 断言预期结果与计算结果 actual 相等
        self.assertEqual(expected, actual)

        # 检查处理 -inf 和 nan 的情况
        x = torch.tensor([-float('inf'), -float('inf'), 1.0, 1.0, float('inf'),
                         float('inf'), float('nan'), 1.0, 1.0], device=device)
        x2d = x.unsqueeze(0).expand(2, -1)

        # 对每个输入进行 logcumsumexp 计算，并断言预期结果与计算结果相等
        for inp in (x, x2d):
            actual = inp.logcumsumexp(axis)
            expected = logcumsumexp(inp, axis)
            self.assertEqual(expected, actual)

        # 检查输出 inplace 是否正常工作
        b = torch.randn(5, 2, device=device)
        inplace_out = torch.zeros(5, 2, device=device)

        # 计算预期结果
        expected = logcumsumexp(b, axis)
        # 使用 torch.logcumsumexp 方法计算，将结果存储在 inplace_out 中
        torch.logcumsumexp(b, axis=axis, out=inplace_out)

        # 断言 inplace_out 与预期结果相等
        self.assertEqual(inplace_out, expected)

        # 检查输入张量与 inplace 输出类型不匹配的情况
        b = torch.randn(5, 2, device=device, dtype=torch.float64)
        inplace_out = torch.zeros(5, 2, device=device, dtype=torch.float32)
        # 断言当输入张量和 inplace 输出类型不匹配时，会抛出 RuntimeError 异常
        with self.assertRaisesRegex(
                RuntimeError,
                'expected scalar_type Double but found Float'):
            torch.logcumsumexp(b, axis, out=inplace_out)
    # 定义用于测试 diff 函数的辅助函数，与 NumPy 参考实现进行比较
    def _test_diff_numpy(self, t, dims=None):
        # 将 PyTorch 张量转换为 NumPy 数组
        def to_np(t):
            # 如果张量的数据类型是 torch.bfloat16，则转换为 torch.float 类型的 NumPy 数组
            if t.dtype == torch.bfloat16:
                return t.to(dtype=torch.float, device="cpu").numpy()
            else:
                # 否则，将张量移动到 CPU 并转换为 NumPy 数组
                return t.cpu().numpy()

        # 遍历指定维度或所有维度
        for dim in dims if dims else range(t.dim()):
            # 在指定维度上选择第一个元素，作为 prepend 张量
            prepend = t.narrow(dim, 0, 1)
            # 在指定维度上选择第一个元素，作为 append 张量
            append = t.narrow(dim, 0, 1)
            # 将输入张量 t 转换为 NumPy 数组
            np_t = to_np(t)

            # 测试当没有 prepend 和 append 的情况
            for n in range(t.size(dim)):
                # 计算 PyTorch 中的 diff 结果
                actual = torch.diff(t, dim=dim, n=n)
                # 从 NumPy 中计算期望结果
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n))
                # 断言 PyTorch 的结果与 NumPy 的结果相等
                self.assertEqual(actual, expected.to(t.dtype))

            # 测试当 prepend 和 append 在指定维度上的大小为 1 的情况
            for n in range(1, t.size(dim) + 4):
                # 计算 PyTorch 中的 diff 结果
                actual = torch.diff(t, dim=dim, n=n, prepend=prepend, append=append)
                # 从 NumPy 中计算期望结果，同时考虑 prepend 和 append
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n, prepend=to_np(prepend), append=to_np(append)))
                # 断言 PyTorch 的结果与 NumPy 的结果相等
                self.assertEqual(actual, expected.to(t.dtype))

            # 测试当 prepend 和 append 在指定维度上的大小不为 1 的情况
            for n in range(1, t.size(dim) * 3):
                # 计算 PyTorch 中的 diff 结果
                actual = torch.diff(t, dim=dim, n=n, prepend=t, append=t)
                # 从 NumPy 中计算期望结果，同时使用输入张量 t 作为 prepend 和 append
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n, prepend=np_t, append=np_t))
                # 断言 PyTorch 的结果与 NumPy 的结果相等
                self.assertEqual(actual, expected.to(t.dtype))

    # 在 XLA 上，所有张量都是连续的
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    def test_diff_noncontig(self, device, dtype):
        # 不同形状的张量
        shapes = (
            (1,),
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5))

        # 遍历每种形状的张量
        for shape in shapes:
            # 创建连续的张量
            contig = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)

            # 创建非连续的张量
            non_contig = torch.empty(shape + (2, 2), device=device, dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            # 断言非连续张量确实不是连续的，或者形状为 (1,)
            self.assertTrue(not non_contig.is_contiguous() or shape == (1,))

            # 使用 _test_diff_numpy 函数测试非连续张量的 diff 函数
            self._test_diff_numpy(non_contig)

    # 在 XLA 上，类型为 f16 的张量不支持 RngNormal
    @dtypes(*all_types_and_complex_and(torch.bool))
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool))
    # 定义一个测试函数，用于测试在指定设备和数据类型下的不同张量形状的情况
    def test_diff(self, device, dtype):
        # 不同形状的张量列表
        shapes = (
            (1,),        # 一维张量
            (1, 5),      # 二维张量
            (3, 5),      # 二维张量
            (1, 5, 1),   # 三维张量
            (2, 3, 5))   # 三维张量

        # 遍历不同形状的张量
        for shape in shapes:
            # 创建具有指定形状的张量，数据类型为dtype，存储于指定设备上，值在-9到9之间
            contig = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            # 使用 numpy 进行 diff 函数的测试
            self._test_diff_numpy(contig)

        # 创建一个全为1的2x3张量
        t = torch.ones(2, 3)

        # 检查在运行时是否抛出异常，异常信息包含'diff expects prepend or append to be the same dimension as input'
        with self.assertRaisesRegex(
                RuntimeError, 'diff expects prepend or append to be the same dimension as input'):
            # 创建一个无效的 prepend 张量，不同设备和数据类型
            invalid_prepend = torch.tensor([1, 2, 3], device=device, dtype=dtype)
            # 在指定维度上对张量进行 diff 操作
            t.diff(dim=0, prepend=invalid_prepend)

        # 检查在运行时是否抛出异常，异常信息包含'diff expects the shape of tensor to prepend or append to match that of input'
        with self.assertRaisesRegex(
                RuntimeError, 'diff expects the shape of tensor to prepend or append to match that of input'):
            # 创建一个无效的 prepend 张量，不同设备和数据类型
            invalid_prepend = torch.tensor([[0, 1]], device=device, dtype=dtype)
            # 在指定维度上对张量进行 diff 操作
            t.diff(dim=0, prepend=invalid_prepend)

        # 检查在运行时是否抛出异常，异常信息包含'diff expects input to be at least one-dimensional'
        with self.assertRaisesRegex(
                RuntimeError, 'diff expects input to be at least one-dimensional'):
            # 创建一个标量张量
            scalar = torch.tensor(2, device=device, dtype=dtype)
            # 对标量张量进行 diff 操作
            torch.diff(scalar)

    # 如果给定的输入参数不是列表，将其包装成只含有单个元素的列表: [arg]
    def _wrap_to_list(self, input_array):
        return input_array if isinstance(input_array, list) else [input_array]

    # 为确保在 Numpy 和 PyTorch 之间不因 inf、-inf 和 nan 值而引起差异
    # 存在两种可能的差异情况:
    # 1. 当我们计算 a 和 b 为实数且具有非常小的绝对值（接近0.0）时，a/b 的结果可能为 inf、-inf 或 nan，从而导致差异。
    # 2. 当我们将复数除以零时，例如当 a = torch.tensor(3+5j) 时，PyTorch 中结果为 nan + nan*j，而 Numpy 中为 inf + inf*j。
    def _inf_nan_preprocess(self, actual, expected):
        # 遍历预期值列表
        for i in range(len(expected)):
            # 使用 numpy 的 nan_to_num 函数将预期值中的 nan、inf 和 -inf 替换为指定值 nan
            expected[i] = np.nan_to_num(expected[i], nan=nan, posinf=nan, neginf=nan)
            # 如果实际值的数据类型为复数
            if actual[i].dtype == torch.complex64 :
                # 分别对实部和虚部使用 torch 的 nan_to_num 函数，将 nan、inf 和 -inf 替换为指定值 nan
                actual[i].real = torch.nan_to_num(actual[i].real, nan=nan, posinf=nan, neginf=nan)
                actual[i].imag = torch.nan_to_num(actual[i].imag, nan=nan, posinf=nan, neginf=nan)
            else:
                # 对实际值使用 torch 的 nan_to_num 函数，将 nan、inf 和 -inf 替换为指定值 nan
                actual[i] = torch.nan_to_num(actual[i], nan=nan, posinf=nan, neginf=nan)

        # 返回处理后的实际值和预期值
        return actual, expected

    # 标记该函数只能应用于原生设备类型的装饰器
    @onlyNativeDeviceTypes
    # 指定该函数适用的数据类型，即 long、float32 和 complex64
    @dtypes(torch.long, torch.float32, torch.complex64)
    # 定义一个测试函数，用于测试梯度计算函数的多种情况
    def test_gradient_all(self, device, dtype):
        # 定义创建标量张量的函数
        def create_scalar(shape):
            return make_tensor((1,), device='cpu', dtype=dtype, low=1.).item()

        # 定义创建列表张量的函数
        def create_list(shape):
            return make_tensor((len(shape),), device='cpu', dtype=dtype, low=1.).tolist()

        # 定义创建坐标张量的函数
        def create_coordinate_tensors(shape):
            tensor_list = []
            for i in range(len(shape)):
                tensor_list.append(make_tensor((shape[i],), device=device, dtype=dtype))
            return tensor_list

        # 定义根据给定维度过滤形状的函数
        def filter_shape(shape, dim):
            filtered_shape = []
            for i in range(len(dim)):
                filtered_shape.append(shape[dim[i]])
            return filtered_shape

        # 定义测试案例，每个案例包含形状和维度信息
        test_cases = (
            ((5,), (0,)),
            ((4, 4), (0, 1)),
            ((3, 3, 3), (-1, 0)),
            ((4, 4, 4), (2,)),
            ((4, 4, 4), (0, 1)),
            ((4, 4, 4, 3), (0, 2, 3)),
            ((4, 5, 3, 4, 3), (1, 2)),
            ((4, 3, 6, 5, 3), (2, 4)),
            ((4, 3, 3, 5, 3), (0, 1, 2, 3, 4)),
            ((1, 3, 3), (1, 2)),
            ((1, 5), (1,)),
        )

        # 对每个测试案例进行迭代
        for case, contig, edge_order, space_fn in product(test_cases, [True, False], [1, 2],
                                                          (create_scalar, create_list, create_coordinate_tensors)):
            shape, dims = case
            # 根据给定维度过滤形状
            filtered_shape = filter_shape(shape, dims)

            # 使用对应的空间函数创建间隔值
            spacing = space_fn(filtered_shape)
            # 创建张量并转换为 NumPy 数组
            t = make_tensor(shape, device=device, dtype=dtype, noncontiguous=not contig)
            t_np = t.cpu().numpy()

            # 计算张量的梯度
            actual = torch.gradient(t, spacing=spacing, dim=dims, edge_order=edge_order)
            # 如果使用坐标张量函数并且间隔值不在 CPU 上，则转换为 NumPy 数组
            if space_fn == create_coordinate_tensors and spacing[0].device != 'cpu':
                spacing = [space.cpu().detach().numpy() for space in spacing]
            # 使用 NumPy 计算预期的梯度
            expected = np.gradient(t_np, *self._wrap_to_list(spacing), axis=dims, edge_order=edge_order)
            # 预处理无穷大和 NaN 值
            actual, expected = self._inf_nan_preprocess(list(actual), self._wrap_to_list(expected))
            # 断言实际值与预期值相等
            self.assertEqual(actual, expected, equal_nan=True, atol=1e-4, rtol=0, exact_dtype=False)

    # 标记函数，仅在本地设备类型上运行测试
    @onlyNativeDeviceTypes
    # 如果在 Torch Inductor 上进行测试，则将测试标记为慢速测试
    @slowTestIf(TEST_WITH_TORCHINDUCTOR)
    # 指定测试使用的数据类型
    @dtypes(torch.long, torch.float32, torch.complex64)
    # 定义一个测试函数，用于测试梯度计算在极端情况下的行为
    def test_gradient_extreme_cases(self, device, dtype):
        # 测试处理无穷大和非数值的行为
        actual = torch.gradient(torch.tensor([2, -2, inf, inf, -inf, -inf, inf, 3, -inf, 2, nan, nan, 3, inf, nan]))
        expected = np.gradient(np.array([2, -2, inf, inf, -inf, -inf, inf, 3, -inf, 2, nan, nan, 3, inf, nan]))
        self.assertEqual(actual, self._wrap_to_list(expected), exact_dtype=False)

        # 测试大尺寸张量的情况
        large_size = 100000
        # 生成指定设备和数据类型的张量
        t = make_tensor((large_size,), dtype=dtype, device=device)
        t_np = t.cpu().numpy()
        # 使用标准正态分布生成随机坐标
        coordinates_np = np.random.randn(large_size)
        coordinates = [torch.tensor(coordinates_np, device=device)]
        # 计算张量在指定坐标轴上的一阶导数
        actual = torch.gradient(t, spacing=coordinates, dim=0, edge_order=1)
        expected = [np.gradient(t_np, coordinates_np, axis=0, edge_order=1)]
        self.assertEqual(actual, expected, exact_dtype=False)

        # 计算张量在指定坐标轴上的二阶导数
        actual = torch.gradient(t, spacing=coordinates, dim=0, edge_order=2)
        expected = [np.gradient(t_np, coordinates_np, axis=0, edge_order=2)]
        self.assertEqual(actual, expected, exact_dtype=False)
    # 定义一个测试函数，用于测试梯度计算类型提升的情况，使用特定设备进行测试

    inputs = (
        make_tensor((4, 4), device=device, dtype=torch.float32),  # 创建一个 4x4 的浮点型张量
        make_tensor((4, 4), device=device, dtype=torch.complex64),  # 创建一个 4x4 的复数张量
        make_tensor((4, 4), device=device, dtype=torch.int64),  # 创建一个 4x4 的整数型张量
    )

    # 定义不同类型的间距或坐标，用于梯度计算的参数
    spacing = (
        make_tensor((1,), device='cpu', dtype=torch.float32).item(),  # 创建一个CPU上的浮点型张量，转换为 Python 标量
        make_tensor((1,), device='cpu', dtype=torch.int64).item(),  # 创建一个CPU上的整数型张量，转换为 Python 标量
        make_tensor((1,), device='cpu', dtype=torch.complex64).item(),  # 创建一个CPU上的复数张量，转换为 Python 标量
        make_tensor((2,), device='cpu', dtype=torch.float32, low=0.1).tolist(),  # 创建一个CPU上的浮点型张量，并转换为 Python 列表
        make_tensor((2,), device='cpu', dtype=torch.int64, low=1).tolist(),  # 创建一个CPU上的整数型张量，并转换为 Python 列表
        make_tensor((2,), device='cpu', dtype=torch.complex64).tolist(),  # 创建一个CPU上的复数张量，并转换为 Python 列表
        [make_tensor((4,), device=device, dtype=torch.float32),  # 创建一个 4 维浮点型张量列表
         make_tensor((4,), device=device, dtype=torch.float32)],  # 创建一个 4 维浮点型张量列表
        [make_tensor((4,), device=device, dtype=torch.int64),  # 创建一个 4 维整数型张量列表
         make_tensor((4,), device=device, dtype=torch.int64)],  # 创建一个 4 维整数型张量列表
        [make_tensor((4,), device=device, dtype=torch.complex64),  # 创建一个 4 维复数型张量列表
         make_tensor((4,), device=device, dtype=torch.complex64)],  # 创建一个 4 维复数型张量列表
    )

    # 对于每一组输入、间距或坐标、边缘顺序的组合，执行梯度计算测试
    for input, spacing_or_coord, edge_order in product(inputs, spacing, [1, 2]):
        # 将输入张量移至 CPU 并转换为 NumPy 数组
        input_np = input.cpu().numpy()
        input_np = input.cpu().numpy()

        # 使用 PyTorch 的梯度计算函数计算实际的梯度
        actual = torch.gradient(input, spacing=spacing_or_coord, dim=(0, 1), edge_order=edge_order)

        # 包装间距或坐标为列表形式，以便后续处理
        spacing_or_coord_wrapped = self._wrap_to_list(spacing_or_coord)

        # 初始化空列表，用于存储间距或坐标的 NumPy 数组表示
        spacing_or_coord_np = []

        # 检查第一个包装的张量是否是张量，并且不在 CPU 上
        if torch.is_tensor(spacing_or_coord_wrapped[0]) and torch.device(spacing_or_coord_wrapped[0].device).type != 'cpu':
            # 将每个包装的张量转换为 NumPy 数组并附加到列表中
            for i in range(len(spacing_or_coord_wrapped)):
                spacing_or_coord_np.append(spacing_or_coord_wrapped[i].detach().clone().cpu().numpy())
        else:
            # 如果不是张量，则直接使用包装的列表
            spacing_or_coord_np = spacing_or_coord_wrapped

        # 使用 NumPy 的梯度计算函数计算期望的梯度
        expected = np.gradient(input_np, *spacing_or_coord_np, axis=(0, 1), edge_order=edge_order)

        # 如果实际计算结果的数据类型是复数而输入不是复数类型
        if actual[0].dtype == torch.complex64 and input.dtype != torch.complex64:
            # 对每个结果进行比较，检查实部是否相等
            for i in range(len(actual)):
                self.assertEqual(actual[i].real, expected[i].real, exact_dtype=False)
                # 输出类型提升在 Numpy 中失败，当间距给定为复数而输入为实数时。
                # 结果仅作为实数给出，所有的虚部应该等于零。
                self.assertEqual(expected[i].imag, torch.zeros(actual[i].shape), exact_dtype=False)
        else:
            # 对实际和期望结果进行处理，处理无穷和NaN值
            actual, expected = self._inf_nan_preprocess(list(actual), expected)
            self.assertEqual(actual, expected, equal_nan=True, exact_dtype=False)

# 应用装饰器，仅允许本地设备类型
@onlyNativeDeviceTypes

# 应用装饰器，指定数据类型为长整型、浮点型、复数型
@dtypes(torch.long, torch.float32, torch.complex64)
    # 测试函数，用于检查梯度计算时的间距列表长度错误
    def test_gradient_spacing_list_length_error(self, device, dtype):
        # 创建一个设备和数据类型为指定参数的2x2张量
        t = make_tensor((2, 2), device=device, dtype=dtype)

        # 第一次测试，间距为一个包含2个元素的元组，期望引发 RuntimeError 异常
        spacing = (make_tensor((2,), device=device, dtype=dtype),)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        # 第二次测试，间距为一个包含4个相同元素的元组
        torch.gradient(t, spacing=spacing)

        # 第三次测试，间距为一个包含6个相同元素的元组，期望引发 RuntimeError 异常
        spacing = (make_tensor((2,), device=device, dtype=dtype),) * 3
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        # 第四次测试，间距为一个长度为1的元组，期望引发 RuntimeError 异常
        spacing = (2,)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        # 第五次测试，间距为一个包含2个元素的元组
        torch.gradient(t, spacing=spacing)

        # 第六次测试，间距为一个包含3个元素的元组，期望引发 RuntimeError 异常
        spacing = (2, 2, 2)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

    # 辅助函数，用于测试大型累积函数
    def _test_large_cum_fn_helper(self, x, fn):
        # 在 CPU 上计算期望值
        expected = fn(x.cpu().float())
        # 在当前设备上计算实际值
        actual = fn(x).cpu().float()
        # 使用 torch.testing.assert_close 避免内存消耗，比较期望值和实际值
        torch.testing.assert_close(expected, actual)

    # 跳过条件下的大型累积和测试，仅在 CUDA 上执行
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @unittest.skipIf(IS_JETSON, "psutil issue for largeTensorTest. Too large for Jetson.")
    @onlyCUDA
    @dtypes(torch.half)  # 只使用小的数据类型以避免内存溢出
    @largeTensorTest('25GB', device='cpu')
    @largeTensorTest('4GB', device='cuda')
    def test_large_cumsum(self, device, dtype):
        # 初始化张量以避免溢出和半精度注意事项
        x = torch.empty(2**30 + 200, device=device, dtype=dtype)
        x[::3] = -3
        x[1::3] = 2
        x[2::3] = 1
        self._test_large_cum_fn_helper(x, lambda x: torch.cumsum(x, 0))

    # 跳过条件下的大型累积积测试，仅在 CUDA 上执行
    @onlyCUDA
    @dtypes(torch.half)  # 只使用小的数据类型以避免内存溢出
    @largeTensorTest('25GB', device='cpu')
    @largeTensorTest('4GB', device='cuda')
    @unittest.skipIf(IS_JETSON, "psutil issue for largeTensorTest. Too large for Jetson.")
    def test_large_cumprod(self, device, dtype):
        # 初始化张量以避免溢出和半精度注意事项
        x = torch.empty(2**30 + 200, device=device, dtype=dtype)
        x[::3] = 8
        x[1::3] = .25
        x[2::3] = .5
        self._test_large_cum_fn_helper(x, lambda x: torch.cumprod(x, 0))

    # 跳过条件下的累积和测试，用于不连续输出，仅在 CPU 上执行
    @skipIfTorchDynamo("Torchdynamo fails with unknown reason")
    @skipIfMps
    def test_discontiguous_out_cumsum(self, device):
        # 创建一个形状为(4, 8)的随机张量
        x = torch.randn(4, 8, device=device)
        # 创建一个形状为(4, 16)的空张量，选取部分列
        y = torch.empty(4, 16, device=device)[:, ::2]
        # 计算累积和，并指定输出到 y 上
        out = torch.cumsum(x, 0)
        torch.cumsum(x, 0, out=y)
        # 检查 y 是否为非连续张量
        self.assertFalse(y.is_contiguous())
        # 比较 out 和 y 的值，允许的误差为0
        self.assertEqual(out, y, atol=0., rtol=0.)
    # 定义一个辅助测试函数，用于测试累积最小或最大值函数的行为
    def _test_cumminmax_helper(self, x, fn, expected_val, expected_ind):
        # 调用累积最小或最大值函数，返回计算结果和索引
        val, ind = fn(x, -1)
        # 断言计算结果与预期值相等，误差容差为0
        self.assertEqual(val, expected_val, atol=0, rtol=0)
        # 断言索引结果与预期索引相等，误差容差为0
        self.assertEqual(ind, expected_ind, atol=0, rtol=0)
        # 创建与val和ind张量相同形状的空张量，转置后保证连续性，再转置回来
        out_val = torch.empty_like(val).t().contiguous().t()
        out_ind = torch.empty_like(ind).t().contiguous().t()
        # 再次调用累积最小或最大值函数，将结果保存到指定的输出张量中
        fn(x, -1, out=(out_val, out_ind))
        # 如果不是在使用TorchInductor测试，则验证输出张量是否为非连续的
        if not TEST_WITH_TORCHINDUCTOR:
            self.assertFalse(out_val.is_contiguous())
            self.assertFalse(out_ind.is_contiguous())
        # 断言使用指定输出张量计算的结果与预期值相等，误差容差为0
        self.assertEqual(out_val, expected_val, atol=0, rtol=0)
        # 断言使用指定输出张量计算的索引与预期索引相等，误差容差为0
        self.assertEqual(out_ind, expected_ind, atol=0, rtol=0)

    # 在不支持MPS的情况下跳过测试
    @skipIfMps
    def test_cummax_discontiguous(self, device):
        # 创建测试张量x，转置后保证连续性，再转置回来
        x = torch.tensor([[0, 1, 2, 3, 2, 1], [4, 5, 6, 5, 6, 7]], device=device, dtype=torch.float).t().contiguous().t()
        # 预期的最大累积值张量和索引张量
        expected_val = torch.tensor([[0, 1, 2, 3, 3, 3], [4, 5, 6, 6, 6, 7]], device=device, dtype=torch.float)
        expected_ind = torch.tensor([[0, 1, 2, 3, 3, 3], [0, 1, 2, 2, 4, 5]], device=device, dtype=torch.long)
        # 调用辅助测试函数，测试torch.cummax函数的行为
        self._test_cumminmax_helper(x, torch.cummax, expected_val, expected_ind)

    # 在不支持MPS的情况下跳过测试
    @skipIfMps
    def test_cummin_discontiguous(self, device):
        # 创建测试张量x，转置后保证连续性，再转置回来
        x = torch.tensor([[3, 2, 1, 0, 1, 2], [7, 6, 5, 4, 5, 2]], device=device, dtype=torch.float).t().contiguous().t()
        # 预期的最小累积值张量和索引张量
        expected_val = torch.tensor([[3, 2, 1, 0, 0, 0], [7, 6, 5, 4, 4, 2]], device=device, dtype=torch.float)
        expected_ind = torch.tensor([[0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 3, 5]], device=device, dtype=torch.long)
        # 调用辅助测试函数，测试torch.cummin函数的行为
        self._test_cumminmax_helper(x, torch.cummin, expected_val, expected_ind)

    # 测试布尔类型张量的值变化
    def test_bool_tensor_value_change(self, device):
        # 创建布尔类型的张量x，并修改其中的值
        x = torch.tensor([True, False], dtype=torch.bool, device=device)
        x[0] = False
        x[1] = True
        # 断言修改后的张量与预期值相等
        self.assertEqual(x, torch.tensor([False, True], dtype=torch.bool, device=device))

    # FIXME: 移动到形状操作的测试套件中
    def test_unfold_all_devices_and_dtypes(self, device):
        # 遍历所有数据类型及其复杂性，包括torch.half、torch.bool、torch.bfloat16等
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            # 根据数据类型dt创建空张量x，设备为指定device
            if dt == torch.bool:
                x = torch.empty((0, 1, 3, 0), dtype=dt, device=device)
                # 断言使用unfold函数后的形状与预期形状相等
                self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)
            else:
                x = torch.empty((0, 1, 3, 0), dtype=dt, device=device)
                # 断言使用unfold函数后的形状与预期形状相等
                self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)

    # FIXME: 移动到形状操作的测试套件中
    # 定义一个测试方法，用于测试 unfold 方法在标量上的行为
    def test_unfold_scalars(self, device):
        # 创建一个包含标量值 0.5 的张量 x，指定设备为参数 device
        x = torch.tensor(0.5, device=device)
        # 对一个0维张量执行 unfold 应始终返回一个1维张量，其形状为 [size]（即 unfold 的第二个参数）

        # 测试 unfold 方法对长度为 0 的情况，期望返回一个空张量
        self.assertEqual(torch.empty(0, device=device), x.unfold(0, 0, 1))
        self.assertEqual(torch.empty(0, device=device), x.unfold(0, 0, 2))
        # 测试 unfold 方法，展开成长度为 1 的张量，包含值为 0.5
        self.assertEqual(torch.tensor([0.5], device=device), x.unfold(0, 1, 1))

    # FIXME: 移动到数据移动测试套件中
    def test_copy_all_dtypes_and_devices(self, device):
        # 导入 copy 函数
        from copy import copy
        # 针对所有数据类型和复杂类型以及 torch.half、torch.bool、torch.bfloat16 进行迭代
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            # 创建一个张量 x，包含整数 1, 2, 3, 4，指定数据类型为 dt，设备为 device
            x = torch.tensor([1, 2, 3, 4], dtype=dt, device=device)
            # 克隆张量 x
            x_clone = x.clone()
            # 使用 copy 函数复制张量 x 到 y
            y = copy(x)
            # 对 y 执行 fill_ 方法，填充为值 1
            y.fill_(1)
            # 断言：copy 是浅拷贝，只复制张量视图，不复制数据本身
            self.assertEqual(x, y)

    # 仅在 CPU 上执行的测试方法
    @onlyCPU
    def test_bfloat16_neg_abs(self, device):
        # 创建一个形状为 (256,) 的随机张量 src
        src = torch.randn(256)
        # 设置 src 的一些特殊值
        src[0] = torch.nan
        src[1] = -torch.nan
        src[2] = torch.inf
        src[3] = -torch.inf
        # 将 src 转换为 bfloat16 类型
        src_bf16 = src.bfloat16()
        # 断言：src 取负后再转换为 bfloat16，与 src_bf16 取负后相等
        self.assertEqual(src.neg().bfloat16(), src_bf16.neg())
        # 断言：src 取绝对值后再转换为 bfloat16，与 src_bf16 取绝对值后相等
        self.assertEqual(src.abs().bfloat16(), src_bf16.abs())

    # FIXME: 移动到数据移动测试套件中
    @onlyNativeDeviceTypes
    @dtypes(torch.bfloat16, torch.half)
    def test_reduced_type_float_copy(self, device, dtype):
        # 对于给定的多个形状进行迭代
        for shape in [(20, 7), (249, 137), (1029, 917), (1, 7, 19, 17), (3, 77, 1091)]:
            # 创建一个指定形状和设备的随机张量 input，数据类型为 torch.float
            input = torch.randn(shape, dtype=torch.float, device=device)
            # 将 input 转换为指定的 dtype 类型，并赋给 out1
            out1 = input.to(dtype=dtype)
            # 断言：input 等于 out1，允许误差，不要求精确的数据类型匹配
            self.assertEqual(input, out1, atol=None, rtol=None, exact_dtype=False)
            # 将 out1 转换回 torch.float 类型，并赋给 out2
            out2 = out1.to(torch.float)
            # 断言：out2 等于 out1，要求绝对误差为 0，相对误差为 0，不要求精确的数据类型匹配

            # 对 input 进行切片操作，赋给 input_s
            input_s = input[..., ::2, :]
            # 将 input_s 转换为指定的 dtype 类型，并赋给 out1
            out1 = input_s.to(dtype=dtype)
            # 断言：input_s 等于 out1，允许误差，不要求精确的数据类型匹配
            self.assertEqual(input_s, out1, atol=None, rtol=None, exact_dtype=False)
            # 将 out1 转换回 torch.float 类型，并赋给 out2
            out2 = out1.to(torch.float)
            # 断言：out2 等于 out1，要求绝对误差为 0，相对误差为 0，不要求精确的数据类型匹配
    # 测试函数：test_copy_math_view，用于测试张量拷贝和视图操作
    def test_copy_math_view(self, device):
        # 循环遍历不同的目标和源数据类型组合
        for dst_dtype, src_dtype in [
                (torch.float32, torch.float32),      # 浮点数类型
                (torch.float64, torch.float32),      # 不同精度的浮点数类型
                (torch.int64, torch.int32),          # 整数类型
                (torch.complex128, torch.complex64), # 复数类型
        ]:
            # 创建指定设备上的源张量
            src = make_tensor((100,), dtype=src_dtype, device=device)
            # 创建指定设备上的目标张量
            dst = torch.empty(100, dtype=dst_dtype, device=device)

            # 使用 src 拷贝数据到 dst
            dst.copy_(src)
            # 断言目标张量与源张量相等（精确匹配数据类型）
            self.assertEqual(dst, src, exact_dtype=False)

            # 使用 src 的负视图拷贝数据到 dst
            dst.copy_(src._neg_view())
            # 断言目标张量与 src 取负后的结果相等（精确匹配数据类型）
            self.assertEqual(dst, src.neg(), exact_dtype=False)

            # 使用 dst 的负视图拷贝数据到 dst（应该是不变的操作）
            dst._neg_view().copy_(torch._neg_view(src))
            # 断言目标张量与源张量相等（精确匹配数据类型）
            self.assertEqual(dst, src, exact_dtype=False)

            # 使用 src 拷贝数据到 dst 的负视图
            dst._neg_view().copy_(src)
            # 断言目标张量与 src 取负后的结果相等（精确匹配数据类型）
            self.assertEqual(dst, src.neg(), exact_dtype=False)

            # issue: https://github.com/pytorch/pytorch/issues/106051
            # 使用 dst 的负视图拷贝自身到 dst（应该是不变的操作）
            dst._neg_view().copy_(dst)
            # 断言目标张量与 src 取负后的结果相等（精确匹配数据类型）
            self.assertEqual(dst, src, exact_dtype=False)

        # 循环遍历复数类型的目标和源数据类型组合
        for dst_dtype, src_dtype in [
                (torch.complex64, torch.complex64),   # 复数类型
                (torch.complex128, torch.complex64),  # 不同精度的复数类型
        ]:
            # 创建指定设备上的源张量
            src = make_tensor((100,), dtype=src_dtype, device=device)
            # 创建指定设备上的目标张量
            dst = torch.empty(100, dtype=dst_dtype, device=device)

            # 使用 src 拷贝数据到 dst 的共轭视图
            dst.conj().copy_(src)
            # 断言目标张量与 src 的物理共轭相等（精确匹配数据类型）
            self.assertEqual(dst, src.conj_physical(), exact_dtype=False)

            # 使用 src 的负视图拷贝数据到 dst 的共轭视图
            dst.conj().copy_(src._neg_view())
            # 断言目标张量与 src 取负后的物理共轭相等（精确匹配数据类型）
            self.assertEqual(dst, src.neg().conj_physical(), exact_dtype=False)
    def test_clone_not_memory_dense(self):
        # 解决 GitHub 上的问题：https://github.com/pytorch/pytorch/issues/64176
        # 创建一个 10x8 的随机张量，并转置后取部分元素，步长为2
        x = torch.randn(10, 8).t()[::2, ::2]
        # 对 x 进行克隆操作
        y = x.clone()
        # 验证克隆后张量的步长是否为 (1, 4)
        self.assertTrue(y.stride() == (1, 4))

    # FIXME: 移动到元素级三元操作的测试套件中
    @dtypesIfCUDA(*set(get_all_math_dtypes('cuda')))
    @dtypes(*set(get_all_math_dtypes('cpu')))
    def test_addcmul(self, device, dtype):
        # 返回与 dtype 对应的浮点数或整数标量
        def _number(floating, integer, dtype):
            if dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
                return floating
            elif dtype in [torch.cfloat, torch.cdouble]:
                return floating * (1 + 1j)
            else:
                return integer

        # 生成指定大小、dtype 和设备的随机张量
        def rand_tensor(size, dtype, device):
            if dtype.is_floating_point or dtype.is_complex:
                return torch.rand(size=size, dtype=dtype, device=device)
            if dtype == torch.uint8:
                return torch.randint(1, 5, size=size, dtype=dtype, device=device)
            else:
                return torch.randint(-5, 5, size=size, dtype=dtype, device=device)

        # 生成随机张量 a, b, c
        a = rand_tensor((2, 2), dtype=dtype, device=device)
        b = rand_tensor((2, 2), dtype=dtype, device=device)
        c = rand_tensor((2, 2), dtype=dtype, device=device)

        # 根据 dtype 选择相应的数值
        alpha = _number(0.5, 3, dtype)

        # 执行 torch.addcmul 操作，计算实际结果
        actual = torch.addcmul(a, b, c, value=alpha)
        # 计算预期结果
        expected = a + alpha * b * c

        # 断言实际结果与预期结果相等
        self.assertEqual(expected, actual)

        # 使用 assertWarnsOnceRegex 检查是否发出 UserWarning
        with self.assertWarnsOnceRegex(
                UserWarning, "This overload of addcmul is deprecated"):
            # 再次执行 torch.addcmul 操作，并断言其结果与实际结果相等
            self.assertEqual(actual, torch.addcmul(a, alpha, b, c))

        # 如果设备类型为 'cuda'，且 dtype 为 torch.half，则进行特定测试
        if self.device_type == 'cuda' and dtype == torch.half:
            a = torch.tensor([60000.0], device=device, dtype=dtype)
            b = torch.tensor([60000.0], device=device, dtype=dtype)
            c = torch.tensor([2.0], device=device, dtype=dtype)
            # 执行 torch.addcmul 操作，将 value 设为 -1
            out = torch.addcmul(a, b, c, value=-1)
            # 验证结果不包含 NaN 或无穷大值
            self.assertTrue(not (out.isnan() or out.isinf()))

    # FIXME: 移动到形状操作的测试套件中
    def test_narrow_empty(self, device):
        # 生成指定设备上的随机张量
        x = torch.randn(2, 3, 4, device=device)
        # 遍历张量的每个维度
        for d in range(x.dim()):
            # 对张量进行窄化操作，使得窄化后的维度为 0
            y = x.narrow(d, x.size(d), 0)
            # 创建与原始张量相同尺寸的列表，并将当前维度的尺寸设置为 0
            sz = list(x.size())
            sz[d] = 0
            # 断言窄化后的张量的尺寸与 sz 相等
            self.assertEqual(sz, y.size())

    def test_narrow_copy_non_contiguous(self, device):
        # 参见 https://github.com/pytorch/pytorch/issues/91690
        # 生成随机张量并进行维度转置
        inp = torch.randn(10, 2, device=device).movedim(-1, 0)
        # 使用 torch.narrow_copy 进行窄化复制操作
        expected = torch.narrow_copy(inp.contiguous(), 1, 0, 10)
        # 执行 torch.narrow_copy 操作
        actual = torch.narrow_copy(inp, 1, 0, 10)
        # 断言预期结果与实际结果相等
        self.assertEqual(expected, actual)

    # FIXME: 移动到索引操作的测试套件中
    @parametrize("reduce", ['prod', 'amin', 'amax', 'mean'])
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    # 定义测试方法，测试索引减少操作
    def test_index_reduce(self, device, dtype, reduce):
        # 定义张量的尺寸
        size = (3, 4, 5)
        # 索引的数据类型
        index_dtypes = [torch.int, torch.long]
        # 是否包含自身索引
        include_selfs = [True, False]
        # 如果数据类型是浮点型，设置最小和最大初始值为正负无穷；否则为数据类型的最大和最小值
        amin_init = float('inf') if dtype.is_floating_point else torch.iinfo(dtype).max
        amax_init = -float('inf') if dtype.is_floating_point else torch.iinfo(dtype).min
        # 初始化减少操作的起始值
        reduction_init = {'prod': 1, 'mean': 0, 'amin': amin_init, 'amax': amax_init}

        # 通过 product 方法生成所有可能的组合
        for dest_noncontig, src_noncontig, index_noncontig in product([True, False], repeat=3):
            for idx_dtype, include_self in product(index_dtypes, include_selfs):
                # 遍历张量尺寸的维度
                for dim in range(len(size)):
                    # 随机生成源张量的元素数量
                    num_src = np.random.randint(10)
                    # 目标张量的元素数量
                    num_dest = size[dim]
                    # 创建目标张量，设备为指定设备，数据类型为指定类型，是否连续由参数决定
                    dest = make_tensor(size, device=device, dtype=dtype, noncontiguous=dest_noncontig)
                    # 根据维度生成源张量的尺寸
                    src_size = size[:dim] + (num_src,) + size[dim + 1:]
                    # 创建源张量，设备为指定设备，数据类型为指定类型，是否连续由参数决定
                    src = make_tensor(src_size, device=device, dtype=dtype, noncontiguous=src_noncontig)
                    # 创建索引张量，元素数量为 num_src，数据类型为 idx_dtype，设备为指定设备，是否连续由参数决定
                    idx = torch.testing.make_tensor(
                        num_src, low=0, high=num_dest, dtype=idx_dtype, device=device, noncontiguous=index_noncontig
                    )
                    # 复制目标张量，用于与期望结果比较
                    expected = dest.clone()
                    # 在指定维度上对目标张量进行索引减少操作
                    dest.index_reduce_(dim, idx, src, reduce, include_self=include_self)
                    # 如果不包含自身索引，则使用 reduction_init 中对应减少操作的初始值填充 idx 的行
                    if (not include_self):
                        expected.index_fill_(dim, idx.long(), reduction_init[reduce])
                    # 转置预期结果和源张量，以便处理维度为 0 的情况
                    expected = expected.transpose(0, dim)
                    src = src.transpose(0, dim)
                    # 遍历每个索引，根据减少操作类型更新期望结果
                    for i in range(num_src):
                        if reduce == 'prod':
                            expected[idx[i]] *= src[i]
                        elif reduce == 'amin':
                            torch.minimum(expected[idx[i]], src[i], out=expected[idx[i]])
                        elif reduce == 'amax':
                            torch.maximum(expected[idx[i]], src[i], out=expected[idx[i]])
                        else:
                            expected[idx[i]] += src[i]
                    # 如果减少操作为 'mean'，计算计数并相应地调整期望结果
                    if reduce == 'mean':
                        counts = torch.ones_like(expected) if include_self else torch.zeros_like(expected)
                        counts.index_add_(0, idx, torch.ones_like(src))
                        counts.masked_fill_(counts == 0, 1)
                        if (dtype.is_floating_point):
                            expected.div_(counts)
                        else:
                            expected.div_(counts, rounding_mode="floor")
                    # 再次转置期望结果和源张量，恢复原始维度顺序
                    expected = expected.transpose(0, dim)

                    # 断言目标张量与期望结果相等
                    self.assertEqual(dest, expected)

    # FIXME: move to test indexing
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义一个测试函数，用于测试 index_copy 方法在不同情况下的行为
    def test_index_copy(self, device, dtype):
        # 我们只测试 num_copy <= num_dest，否则会有重复的索引，行为未定义
        num_copy, num_dest = 3, 5

        # 定义一个辅助函数，根据给定的参数生成一个张量
        def make_arg(batch_sizes, n, dim, contig):
            size_arg = batch_sizes[:dim] + (n,) + batch_sizes[dim:]
            return make_tensor(size_arg, dtype=dtype, device=device, low=None, high=None, noncontiguous=not contig)

        # 定义一个参考实现的辅助函数，用于在指定维度上执行 index_copy 操作
        def ref_index_copy(tgt, dim, idx, src):
            for i in range(idx.size(0)):
                idx_dest = dim * (slice(None),) + (idx[i],)
                idx_src = dim * (slice(None),) + (i,)
                tgt[idx_dest] = src[idx_src]

        # 更详细的测试，类似于 index_add 的测试方式
        for dest_contig, src_contig, index_contig in product([True, False], repeat=3):
            for other_sizes in ((), (4, 5)):
                for dim in range(len(other_sizes)):
                    # 创建目标张量、源张量和索引张量
                    dest = make_arg(other_sizes, num_dest, dim, dest_contig)
                    src = make_arg(other_sizes, num_copy, dim, src_contig)
                    idx = torch.randperm(num_dest, dtype=torch.int64, device=device)[:num_copy]
                    if not index_contig:
                        idx = torch.repeat_interleave(idx, 2, dim=-1)
                        idx = idx[..., ::2]
                    dest2 = dest.clone()
                    # 使用 index_copy 方法更新目标张量
                    dest.index_copy_(dim, idx, src)
                    # 使用参考实现函数更新目标张量
                    ref_index_copy(dest2, dim, idx, src)
                    # 断言两种方法得到的结果应该一致
                    self.assertEqual(dest, dest2)

    # FIXME: move to test indexing
    # 仅限于原生设备类型，因为存在 XLA 错误：https://github.com/pytorch/pytorch/issues/53256
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_index_copy_scalars(self, device, dtype):
        # 创建 8 种可能的标量大小组合，用于目标、索引和源张量
        scalars = ((make_tensor(size_t, dtype=dtype, device=device, low=None, high=None),
                    make_tensor(size_i, dtype=torch.int64, device=device, low=0, high=1),
                    make_tensor(size_s, dtype=dtype, device=device, low=None, high=None))
                   for size_t, size_i, size_s in product([(), (1,)], repeat=3))
        for target, idx, source in scalars:
            # 使用 index_copy 方法更新目标张量
            target.index_copy_(0, idx, source)
            # 断言目标张量的值应该等于源张量的值
            self.assertEqual(target.item(), source.item())

    # FIXME: move to test indexing
    @onlyCPU
    def test_errors_index_copy(self, device):
        # We do not test the GPU as the CUDA_ASSERT would break the CUDA context
        # 在此方法中不测试 GPU，因为 CUDA_ASSERT 会破坏 CUDA 上下文

        idx_dim = 8
        tgt_dim = 5
        batch_dim = 3

        # Too large of an index
        # 索引过大的情况
        a = torch.randn(batch_dim, tgt_dim, device=device)
        idx = torch.full((idx_dim,), tgt_dim, device=device)
        c = torch.zeros(batch_dim, idx_dim, device=device)
        with self.assertRaises(IndexError):
            a.index_copy_(1, idx, c)

        # Too small (negative indices)
        # 索引过小（负索引）
        idx = torch.full((idx_dim,), -1, device=device)
        with self.assertRaises(IndexError):
            a.index_copy_(1, idx, c)

        # Too small (very negative indices) - they should be unsupported even
        # when support for negative indices is implemented for index_copy_
        # 索引过小（非常负的索引）- 即使对 index_copy_ 实现了负索引支持，这些索引也应该是不支持的
        idx = torch.full((idx_dim,), -tgt_dim - 1, device=device)
        with self.assertRaises(IndexError):
            a.index_copy_(1, idx, c)

    def _prepare_data_for_index_copy_and_add_deterministic(
        self, dim: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (dim >= 0 and dim < 3)
        # 准备数据用于 index_copy_ 和确定性添加
        a = [5, 4, 3]
        a[dim] = 2000
        x = torch.zeros(a, device=device)
        b = a.copy()
        elems = a[dim] * 20
        b[dim] = elems
        src = torch.rand(b, device=device)
        index = torch.randint(a[dim], (elems,), device=device)
        return (x, index, src)

    # FIXME: move to test indexing
    @onlyNativeDeviceTypes
    def test_index_copy_deterministic(self, device: torch.device) -> None:
        # 测试确定性 index_copy_
        for dim in range(3):
            x, index, src = self._prepare_data_for_index_copy_and_add_deterministic(dim, device)
            with DeterministicGuard(True):
                y0 = torch.index_copy(x, dim, index, src)

            x0 = x.clone().detach()
            index_list = index.tolist()
            for i in range(len(index_list)):
                if dim == 0:
                    x0[index_list[i], :, :] = src[i, :, :]
                elif dim == 1:
                    x0[:, index_list[i], :] = src[:, i, :]
                elif dim == 2:
                    x0[:, :, index_list[i]] = src[:, :, i]

            self.assertEqual(x0, y0, atol=0, rtol=0)

    # FIXME: move to test indexing
    @onlyNativeDeviceTypes


注：每个函数和代码段都根据要求添加了注释，解释了其作用和一些特定的行为或条件。
    # 测试函数：验证在确定性模式下索引添加操作的行为
    def test_index_add_deterministic(self, device: torch.device) -> None:
        # 对每个维度进行测试
        for dim in range(3):
            # 准备索引复制和添加操作所需的数据
            x, index, src = self._prepare_data_for_index_copy_and_add_deterministic(dim, device)
            # 随机生成一个大于1的 alpha 值
            alpha = random.random() + 1
            # 在 CPU 上，无论是否启用确定性模式，结果应该是确定性的
            with DeterministicGuard(True):
                # 执行索引添加操作，并记录第一次的结果 y0
                y0 = torch.index_add(x, dim, index, src, alpha=alpha)
                for _ in range(3):
                    # 多次执行索引添加操作，结果应与第一次一致
                    y = torch.index_add(x, dim, index, src, alpha=alpha)
                    self.assertEqual(y, y0, atol=0, rtol=0)

            # 禁用确定性模式后进行测试
            with DeterministicGuard(False):
                for _ in range(3):
                    # 多次执行索引添加操作，结果应与第一次一致，但允许一定的数值误差
                    y_nd = torch.index_add(x, dim, index, src, alpha=alpha)
                    self.assertEqual(y_nd, y0, atol=1e-3, rtol=1e-5)

    # FIXME: 找到适合 put 操作的测试套件
    @onlyNativeDeviceTypes
    def test_index_put_non_accumulate_deterministic(self, device) -> None:
        # 在确定性模式下执行测试
        with DeterministicGuard(True):
            for i in range(3):
                # 随机生成数组和索引
                m = random.randint(10, 20)
                elems = random.randint(20000, 30000)
                values = torch.rand(elems, device=device)
                indices = torch.randint(m, (elems,), device=device)
                input = torch.rand(m, device=device)
                # 执行非累加索引 put 操作
                output = input.index_put((indices,), values, accumulate=False)

                # 手动计算期望结果
                input_list = input.tolist()
                indices_list = indices.tolist()
                values_list = values.tolist()
                for i, v in zip(indices_list, values_list):
                    input_list[i] = v

                # 验证输出结果与期望结果一致
                self.assertEqual(output, input_list)

    # FIXME: 移至索引测试中
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @skipIfMps
    def test_index_fill(self, device, dtype):
        # 创建一个张量，并在指定维度上进行填充
        x = torch.tensor([[1, 2], [4, 5]], dtype=dtype, device=device)
        index = torch.tensor([0], device=device)
        x.index_fill_(1, index, 0)
        self.assertEqual(x, torch.tensor([[0, 2], [0, 5]], dtype=dtype, device=device))
        # 如果张量不是复数且不是 "meta" 设备，则验证对非标量填充的异常处理
        if not x.is_complex() and not device == "meta":
            with self.assertRaisesRegex(RuntimeError, r"Scalar"):
                x.index_fill_(1, index, 1 + 1j)
        # 确保应用于零维输入时结果保持零维
        x = torch.tensor(1, dtype=dtype, device=device)
        self.assertEqual(0, x.index_fill(0, index, -1).dim())
        self.assertEqual(0, x.index_fill_(0, index, -1).dim())

    # FIXME: 移至索引测试中
    # 在 XLA 上，零维张量的测试会失败
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义一个测试函数，用于测试 torch.index_select 方法
    def test_index_select(self, device, dtype):
        # 定义源张量的大小和输出张量的大小
        num_src, num_out = 3, 5

        # 定义生成参数的函数
        def make_arg(batch_sizes, n, dim, contig):
            # 构造张量的大小参数
            size_arg = batch_sizes[:dim] + (n,) + batch_sizes[dim:]
            # 调用 make_tensor 函数生成指定设备和数据类型的张量
            return make_tensor(size_arg, dtype=dtype, device=device, low=None, high=None, noncontiguous=not contig)

        # 定义参考的 index_select 实现函数
        def ref_index_select(src, dim, idx):
            # 如果数据类型是 torch.bfloat16，需要先将 src 转换为 float 类型
            if dtype == torch.bfloat16:
                src = src.float()
            # 使用 numpy 的 take 函数实现 index_select 操作，然后将其转换为 torch.Tensor
            out = torch.from_numpy(np.take(src.cpu().numpy(), idx.cpu().numpy(), axis=dim))
            # 如果数据类型是 torch.bfloat16，将输出张量转换为指定设备和数据类型
            if dtype == torch.bfloat16:
                out = out.to(device=device, dtype=dtype)
            # 返回 index_select 的结果
            return out

        # 使用 product 生成所有可能的 contig 和 idx_contig 的组合
        for src_contig, idx_contig in product([True, False], repeat=2):
            # 使用 product 生成其他大小的组合
            for other_sizes in ((), (4, 5)):
                # 遍历维度 dim
                for dim in range(len(other_sizes)):
                    # 生成源张量 src 和索引 idx
                    src = make_arg(other_sizes, num_src, dim, src_contig)
                    idx = make_tensor(
                        (num_out,), dtype=torch.int64, device=device, low=0, high=num_src, noncontiguous=not idx_contig
                    )
                    # 调用 torch.index_select 方法
                    out = torch.index_select(src, dim, idx)
                    # 调用 ref_index_select 函数
                    out2 = ref_index_select(src, dim, idx)
                    # 断言两个输出张量是否相等
                    self.assertEqual(out, out2)

        # 针对 torch.int32 和 torch.int64 类型的索引，以及不同的其他大小组合进行测试
        for idx_type in (torch.int32, torch.int64):
            other_sizes = (3, 2)
            dim = 1
            # 生成源张量 src 和索引 idx
            src = make_arg(other_sizes, num_src, dim, True)
            idx = make_tensor((num_out,), dtype=idx_type, device=device, low=0, high=num_src, noncontiguous=False)
            # 调用 torch.index_select 方法
            out = torch.index_select(src, dim, idx)
            # 调用 ref_index_select 函数
            out2 = ref_index_select(src, dim, idx)
            # 断言两个输出张量是否相等
            self.assertEqual(out, out2)

        # 创建四种可能的标量大小组合进行测试
        scalars = ((make_tensor(size_s, dtype=dtype, device=device),
                    torch.zeros(size_i, dtype=torch.int64, device=device))
                   for size_s, size_i in product([(), (1,)], repeat=2))
        # 遍历所有标量的组合
        for source, idx in scalars:
            # 调用 index_select 方法
            out = source.index_select(0, idx)
            # 断言输出张量的标量值是否相等
            self.assertEqual(out.item(), source.item())

    # FIXME: 找到适合 take 运算符的测试套件
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义测试函数 test_take，接受设备和数据类型作为参数
    def test_take(self, device, dtype):
        # 索引大小设定为 (4,)
        idx_size = (4,)

        # 使用偏函数 make_arg 和 make_idx 来创建张量和索引
        make_arg = partial(make_tensor, device=device, dtype=dtype)
        make_idx = partial(make_tensor, low=0, device=device, dtype=torch.int64)

        # 定义参考函数 ref_take，用于在 numpy 中实现 take 操作的参考实现
        def ref_take(src, idx):
            # 如果数据类型为 torch.bfloat16，则将源张量转换为半精度
            if dtype == torch.bfloat16:
                src = src.half()
            # 将源张量和索引张量转换为 numpy 数组，并使用 np.take 执行 take 操作
            src = src.cpu().numpy()
            idx = idx.cpu().numpy()
            out = torch.from_numpy(np.take(src, idx)).to(device=device, dtype=dtype)
            return out

        # 遍历所有可能的连续性组合和索引重塑组合
        for src_contig, idx_contig, idx_reshape in product([True, False], repeat=3):
            # 遍历不同的源张量大小
            for src_size in ((5,), (4, 5)):
                # 创建源张量和索引张量
                src = make_arg(src_size, noncontiguous=not src_contig)
                idx = make_idx(idx_size, high=src.numel(), noncontiguous=not idx_contig)
                # 如果需要重塑索引张量
                if idx_reshape:
                    idx = idx.reshape(2, 2)
                # 使用 torch.take 执行 take 操作
                out = torch.take(src, idx)
                # 使用参考函数 ref_take 执行 take 操作，并进行断言比较
                out2 = ref_take(src, idx)
                self.assertEqual(out, out2)

        # 创建源张量和索引均为标量的四种可能组合
        for size_s, size_i in product([(), (1,)], repeat=2):
            source = make_arg(size_s)
            idx = make_idx(size_i, high=1)
            # 使用张量的 take 方法
            out = source.take(idx)
            # 断言结果与源张量的值相等
            self.assertEqual(out.item(), source.item())

    # FIXME: 找一个适合 put 操作的测试套件
    # 在 GPU 上布尔实例无法工作。参见 https://github.com/pytorch/pytorch/issues/54317
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_put_accumulate(self, device, dtype):
        # 测试 accumulate=True 的并行加法操作
        low_precision = dtype == torch.half or dtype == torch.bfloat16
        # 减少数字数量以避免低精度溢出问题
        # grainsize 为 3000 用于 CPU 上的并行化循环
        sizes = ((100,)) if low_precision else ((200,), (3002,))
        # bfloat16 在这里表现特别差
        # 由于该操作在 GPU 上是非确定性的，因此 rtol 设定较大
        rtol, atol = (1e-1, 1e-2) if low_precision else (1e-3, 1e-4)

        # 使用偏函数 make_arg 来创建张量，将所有内容放置到第 0 位置
        make_arg = partial(make_tensor, low=-2, high=3, device=device, dtype=dtype)
        make_idx = partial(torch.zeros, device=device, dtype=torch.int64)
        args = ((make_idx(size), make_arg(size)) for size in sizes)

        # 遍历所有参数组合
        for idx, source in args:
            orig = make_arg((1,))
            # 使用 put 方法执行放置和累加操作
            out = orig.put(idx, source, accumulate=True)
            # 断言结果与原始张量加上源张量求和相等
            self.assertEqual(out, orig + source.sum(), rtol=rtol, atol=atol)

    # FIXME: 找一个适合 take 操作的测试套件
    @skipIfMps
    # 定义一个测试方法，测试torch.take操作在空输入上的行为
    def test_take_empty(self, device):
        # 对不同的输入形状和索引形状进行迭代测试
        for input_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                # 创建一个空的张量作为输入，指定设备
                input = torch.empty(input_shape, device=device)
                # 创建一个空的索引张量，数据类型为int64，指定设备
                indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                # 使用torch.take方法从输入张量中获取数据，预期结果是索引张量本身
                self.assertEqual(indices, torch.take(input, indices), exact_dtype=False)

    # FIXME: 找一个适合put操作的测试套件
    def test_put_empty(self, device):
        # 对不同的目标形状和索引形状进行迭代测试
        for dst_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                # 对于累加参数为False和True分别进行迭代测试
                for accumulate in [False, True]:
                    # 创建一个具有随机值的目标张量，指定设备
                    dst = torch.randn(dst_shape, device=device)
                    # 创建一个空的索引张量，数据类型为int64，指定设备
                    indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                    # 创建一个具有随机值的源张量，指定设备
                    src = torch.randn(indices_shape, device=device)
                    # 使用dst.put_方法，在目标张量上放置源张量数据，预期结果是目标张量本身
                    self.assertEqual(dst, dst.put_(indices, src, accumulate=accumulate))

    # FIXME: 迁移到test_scatter_gather_ops.py进行测试
    def scatter_allow_reduce(self, device, dtype, reduceop):
        # 获取设备类型
        device_type = torch.device(device).type
        # 如果是在CPU上，或者在CUDA上且reduceop是'multiply'并且数据类型是浮点型，返回True
        return device_type != 'cuda' or (reduceop == 'multiply' and dtype.is_floating_point)

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_operations_to_large_input(self, device, dtype):
        # 创建一个索引张量，指定设备和数据类型
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        # 定义测试数据列表
        test_data = [
            (torch.zeros(4, 4, device=device, dtype=dtype),
             torch.ones(2, 2, device=device, dtype=dtype),
             torch.tensor([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 0]],
                          device=device, dtype=dtype), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(4, 4),
             torch.tensor([6], device=device, dtype=dtype).repeat(2, 2),
             torch.tensor([[2, 2, 2, 2],
                           [12, 2, 2, 2],
                           [12, 2, 2, 2],
                           [2, 2, 2, 2]], device=device, dtype=dtype), "multiply"),
        ]

        # 对测试数据进行迭代测试
        for input, src, result, operation in test_data:
            # 如果scatter_allow_reduce方法返回False，跳过当前循环
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            # 使用input.scatter_方法进行scatter操作，根据reduce参数进行不同的规约操作
            input.scatter_(0, index, src, reduce=operation)
            # 断言input是否等于预期的result
            self.assertEqual(input, result)

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义测试函数，用于测试 scatter 操作中的标量减少情况，接受设备和数据类型参数
    def test_scatter_reduce_scalar(self, device, dtype):
        # 创建包含索引的张量，表示 scatter 操作的目标位置
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        # 定义测试数据列表，每个元组包含输入张量、标量、预期结果张量和操作类型
        test_data = [
            (torch.zeros(4, 4, device=device, dtype=dtype), 1,
             torch.tensor([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 0]],
                          device=device, dtype=dtype), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(4, 4), 2,
             torch.tensor([[2, 2, 2, 2],
                           [4, 2, 2, 2],
                           [4, 2, 2, 2],
                           [2, 2, 2, 2]], device=device, dtype=dtype), "multiply"),
        ]

        # 遍历测试数据
        for input, src, result, operation in test_data:
            # 如果不允许当前操作类型的减少，则跳过当前循环
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            # 执行 scatter 操作，修改输入张量
            input.scatter_(0, index, src, reduce=operation)
            # 断言输入张量是否等于预期结果张量
            self.assertEqual(input, result)

    # FIXME: port to test_scatter_gather_ops.py
    # TODO: remove this after scatter_add_ is deprecated.
    # 定义测试函数，测试 scatter_add_ 操作在非唯一索引情况下的表现，接受设备参数
    def test_scatter_add_non_unique_index(self, device):
        # 定义张量的高度和宽度
        height = 2
        width = 65536
        # 创建全为1的输入张量，以及全为0的索引张量和源张量
        input = torch.ones(height, width, device=device)
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        src = torch.ones(height, width, device=device)
        # 执行 scatter_add_ 操作，修改输入张量
        input.scatter_add_(0, index, src)

        # 断言输入张量是否等于预期张量，预期张量中每行的值应为3或1
        self.assertEqual(input,
                         torch.tensor([[3], [1]], device=device,
                                      dtype=torch.float32).repeat(1, width))

    # 使用复杂类型的装饰器，定义测试函数，测试 scatter 操作中的非唯一索引情况下的减少情况，接受设备和数据类型参数
    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_non_unique_index(self, device, dtype):
        # 定义张量的高度和宽度
        height = 2
        width = 2
        # 创建全为0的索引张量
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        # 定义测试数据列表，每个元组包含输入张量、源张量、预期结果张量和操作类型
        test_data = [
            (torch.ones(height, width, device=device, dtype=dtype),
             torch.ones(height, width, device=device, dtype=dtype),
             torch.tensor([[3], [1]], device=device, dtype=dtype).repeat(1, width), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(height, width),
             torch.tensor([2], device=device, dtype=dtype).repeat(height, width),
             torch.tensor([[8], [2]], device=device,
                          dtype=dtype).repeat(1, width), "multiply"),
        ]

        # 遍历测试数据
        for input, src, result, operation in test_data:
            # 如果不允许当前操作类型的减少，则跳过当前循环
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            # 执行 scatter 操作，修改输入张量
            input.scatter_(0, index, src, reduce=operation)
            # 断言输入张量是否等于预期结果张量，并输出详细信息以便调试
            self.assertEqual(input, result, msg=f"result: {result} input: {input} method: {str(operation)}")

    # 仅在CUDA环境下运行的装饰器，使用复杂类型的装饰器
    @onlyCUDA
    @dtypes(*complex_types())
    # 检验 scatter_ 方法对不支持的数据类型的处理，应抛出 RuntimeError 异常
    def test_scatter_reduce_multiply_unsupported_dtypes(self, device, dtype):
        # 设置张量的高度和宽度
        height = 2
        width = 2
        # 创建索引张量，全零，长整型，指定设备
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        # 创建输入张量，全一，指定设备和数据类型
        input = torch.ones(height, width, device=device, dtype=dtype)
        # 创建源张量，全一，指定设备和数据类型
        src = torch.ones(height, width, device=device, dtype=dtype)
        # 使用断言检查以下操作是否引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 在输入张量上执行 scatter_ 操作，使用 reduce="multiply" 参数
            input.scatter_(0, index, src, reduce="multiply")

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_to_large_input(self, device):
        # 创建大小为 4x4 的零张量，指定设备
        input = torch.zeros(4, 4, device=device)
        # 创建大小为 2x2 的全一张量，指定设备
        src = torch.ones(2, 2, device=device)
        # 创建索引张量，二维，指定设备和数据类型为长整型
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        # 在输入张量上执行 scatter_ 操作
        input.scatter_(0, index, src)
        # 使用断言检查输入张量是否等于指定的张量
        self.assertEqual(input, torch.tensor([[0, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 0]], device=device, dtype=torch.float32))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_add_to_large_input(self, device):
        # 创建大小为 4x4 的零张量，指定设备
        input = torch.zeros(4, 4, device=device)
        # 创建大小为 2x2 的全一张量，指定设备
        src = torch.ones(2, 2, device=device)
        # 创建索引张量，二维，指定设备和数据类型为长整型
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        # 在输入张量上执行 scatter_add_ 操作
        input.scatter_add_(0, index, src)
        # 使用断言检查输入张量是否等于指定的张量
        self.assertEqual(input, torch.tensor([[0, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 0]], device=device, dtype=torch.float32))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_bool(self, device):
        # 创建布尔值张量，大小为 2x3，指定设备
        x = torch.tensor([[True, True, True], [True, True, True]], device=device)
        # 创建全零布尔值张量，大小为 3x3，指定设备
        res = torch.zeros(3, 3, dtype=torch.bool, device=device)
        # 在 res 上执行 scatter_ 操作
        res = res.scatter_(0, torch.tensor([[0, 1, 2], [0, 1, 2]], device=device), x)
        # 使用断言检查结果张量是否等于指定的张量
        self.assertEqual(res, torch.tensor([[True, False, False],
                                            [False, True, False],
                                            [False, False, True]], device=device))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_add_bool(self, device):
        # 创建布尔值张量，大小为 2x5，指定设备
        x = torch.tensor([[True, True, True, True, True], [True, True, True, True, True]], device=device)
        # 创建全零布尔值张量，大小为 3x5，指定设备
        res = torch.zeros(3, 5, dtype=torch.bool, device=device)
        # 在 res 上执行 scatter_add_ 操作
        res = res.scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=device), x)
        # 使用断言检查结果张量是否等于指定的张量
        self.assertEqual(res, torch.tensor([[True, True, True, True, True],
                                            [False, True, False, True, False],
                                            [True, False, True, False, True]], device=device))

    # FIXME: find a test suite for the masked scatter operator
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    # 定义一个测试函数，用于测试 masked_scatter 方法
    def test_masked_scatter(self, device, dtype):
        # 设定数据类型
        dt = dtype
        # 设定拷贝的数量和目标张量的长度
        num_copy, num_dest = 3, 10
        # 创建目标张量 dest，并指定设备和数据类型
        dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dt, device=device)
        # 克隆 dest 以备后续比较
        dest2 = dest.clone()
        # 克隆 dest 以备后续比较
        dest_ones = dest.clone()
        # 克隆 dest 以备后续比较
        dest_ones_expected = dest.clone()
        # 创建源张量 src，并指定设备和数据类型
        src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt, device=device)
        # 创建全 1 张量 src_ones，并指定设备和数据类型
        src_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dt, device=device)
        # 创建布尔类型的掩码 mask，并指定设备
        mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device=device)

        # 使用 masked_scatter_ 方法将 src 根据 mask 复制到 dest 中
        dest.masked_scatter_(mask, src)

        # 初始化索引 j
        j = 0
        # 遍历 num_dest 次
        for i in range(num_dest):
            # 如果 mask[i] 为 True，则将 src[j] 复制到 dest2[i] 和 dest_ones_expected[i] 中
            if mask[i]:
                dest2[i] = src[j]
                dest_ones_expected[i] = src_ones[j]
                j += 1
        # 断言 dest 和 dest2 相等，精度为 0
        self.assertEqual(dest, dest2, atol=0, rtol=0)

        # 使用 masked_scatter_ 方法将 src_ones 根据 mask 复制到 dest_ones 中
        dest_ones.masked_scatter_(mask, src_ones)
        # 断言 dest_ones 和 dest_ones_expected 相等，精度为 0
        self.assertEqual(dest_ones, dest_ones_expected, atol=0, rtol=0)

        # 如果设备类型不是 'cuda'，则执行以下代码块
        if self.device_type != 'cuda':
            # 缩小 src 的尺寸，预期此处会失败
            src = torch.zeros(num_copy - 1, dtype=dt, device=device)
            # 使用 assertRaises 断言运行时会抛出异常
            with self.assertRaises(RuntimeError):
                dest.masked_scatter_(mask, src)

        # 创建空张量 dest，尺寸为 (5, 0, 5)，指定设备和数据类型
        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        # 创建与 dest 尺寸相同的全 1 布尔类型的掩码 mask
        mask = torch.ones_like(dest, dtype=torch.bool, device=device)
        # 创建空张量 src，尺寸为 (0,)，指定设备和数据类型
        src = torch.empty((0,), dtype=dt, device=device)
        # 使用 masked_scatter_ 方法将 src 根据 mask 复制到 dest 中

        dest.masked_scatter_(mask, src)

        # 创建空张量 dest，尺寸为 (5, 0, 5)，指定设备和数据类型
        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        # 创建尺寸为 (5, 1, 5) 的全 1 布尔类型的掩码 mask
        mask = torch.ones((5, 1, 5), dtype=torch.bool, device=device)
        # 创建空张量 src，尺寸为 (0,)，指定设备和数据类型
        src = torch.empty((0,), dtype=dt, device=device)
        # 使用 masked_scatter_ 方法将 src 根据 mask 复制到 dest 中
        dest.masked_scatter_(mask, src)

    # FIXME: find a test suite for the masked scatter operator
    @skipIfMps
    def test_masked_scatter_bool_tensor(self, device):
        # 创建布尔张量 src，元素为 True，指定设备
        src = torch.tensor([True, True, True], device=device)
        # 创建布尔张量 dst，元素为 False，指定设备
        dst = torch.tensor([False, False, False], device=device)
        # 创建布尔类型的掩码 mask
        mask = torch.tensor([False, True, False], device=device)

        # 使用 masked_scatter_ 方法将 src 根据 mask 复制到 dst 中
        dst.masked_scatter_(mask, src)
        # 断言 dst 等于预期的张量 [False, True, False]
        self.assertEqual(dst, torch.tensor([False, True, False], device=device))

        # 修改 mask 为 [True, False, True]
        mask = torch.tensor([True, False, True], device=device)
        # 使用 masked_scatter 方法将 src 根据 mask 复制到 dst 中
        dst = dst.masked_scatter(mask, src)
        # 断言 dst 等于预期的张量 [True, True, True]
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

    # FIXME: find a test suite for the masked scatter operator
    #   test_scatter_gather_ops or test_masked_ops?
    @onlyCUDA
    @largeTensorTest('30GB')
    def test_masked_scatter_large_tensor(self, device):
        # 创建长度为 2^31 + 1 的布尔类型的空张量 t_cpu，并随机初始化
        t_cpu = torch.empty(2**31 + 1, dtype=torch.bool).random_()
        # 将 t_cpu 移动到指定设备上，并命名为 t
        t = t_cpu.to(device)
        # 使用 masked_scatter 方法将 t_cpu 根据 t_cpu 复制到 t 中
        result_cpu = t_cpu.masked_scatter(t_cpu, t_cpu)
        # 使用 masked_scatter 方法将 t 根据 t 复制到 t 中
        result = t.masked_scatter(t, t)
        # 断言 result 等于 result_cpu
        self.assertEqual(result, result_cpu)
    # FIXME: 找到适合遮罩选择操作的测试套件
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 使用装饰器指定测试函数支持的数据类型，包括所有标准数据类型和特殊类型（如 torch.half, torch.bool, torch.bfloat16）
    def test_masked_select(self, device, dtype):
        # 根据设备类型设定警告消息
        if device == 'cpu':
            warn = 'masked_select received a mask with dtype torch.uint8,'
        else:
            warn = 'indexing with dtype torch.uint8 is now deprecated, pl'
        # 遍历支持的遮罩类型
        for maskType in integral_types_and(torch.bool):
            num_src = 10
            # 创建源张量并指定数据类型和设备
            src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device)
            # 生成随机遮罩张量并指定数据类型和设备
            mask = torch.randint(2, (num_src,), device=device, dtype=maskType)

            # 如果遮罩类型不是 torch.bool，则预期引发 RuntimeError 异常
            if maskType is not torch.bool:
                with self.assertRaisesRegex(RuntimeError, r'expected BoolTensor for mask'):
                    dst = src.masked_select(mask)
                continue
            else:
                # 对源张量应用遮罩选择操作
                dst = src.masked_select(mask)

            # 手动计算遮罩选择的预期结果
            dst2 = []
            for i in range(num_src):
                if mask[i]:
                    dst2 += [src[i]]
            # 断言遮罩选择操作的输出与预期结果相等
            self.assertEqual(dst, torch.tensor(dst2), atol=0, rtol=0)

            # 使用指定的输出张量进行遮罩选择操作
            dst3 = torch.empty(0, device=device, dtype=dtype)
            torch.masked_select(src, mask, out=dst3)
            # 断言指定输出张量的结果与预期结果相等
            self.assertEqual(dst3, torch.tensor(dst2, dtype=dst3.dtype), atol=0, rtol=0)

        # 如果数据类型是 torch.half 并且设备是 CPU，则跳过剩余的测试用例
        if dtype == torch.half and torch.device(device).type == 'cpu':
            return

        # 确保遮罩能够正确扩展以匹配张量
        a = torch.rand(100, 100, device=device).mul(100).to(dtype)
        mask_first_el_each_row = torch.zeros(100, device=device, dtype=torch.bool)
        mask_first_el_each_row[0] = True
        a_masked = a.masked_select(mask_first_el_each_row)
        # 断言遮罩选择操作的结果与预期的列切片相等
        self.assertEqual(a_masked, a[:, 0])

        mask_first_row = torch.zeros(100, 1, device=device, dtype=torch.bool)
        mask_first_row[0][0] = True
        a_masked = a.masked_select(mask_first_row)
        # 断言遮罩选择操作的结果与预期的行切片相等
        self.assertEqual(a_masked, a[0, :])

        # 确保张量能够正确扩展以匹配遮罩
        a = torch.rand(100, device=device).mul(100).to(dtype)
        mask_copy_3_times = torch.tensor([[True], [True], [False], [True]], device=device)
        a_masked = a.masked_select(mask_copy_3_times)
        # 断言遮罩选择操作的结果与预期的扩展张量相等
        self.assertEqual(a_masked, a.unsqueeze(0).expand(3, 100).flatten())

    # FIXME: 找到适合遮罩选择操作的测试套件
    # 定义测试方法，用于测试不连续的掩码选择功能，接受设备参数
    def test_masked_select_discontiguous(self, device):
        # 遍历不同尺寸的测试用例：10 和 200
        for size in (10, 200):
            # 创建大小为 (size, size) 的随机张量 vals，并将其移动到指定设备
            vals = torch.rand(size, size, device=device)
            # 创建与 vals 大小相同的全 False 布尔张量 mask，然后将偶数列设为 True
            mask = torch.full((size, size), False, dtype=torch.bool, device=device)
            mask[:, ::2] = True
            # 创建包含 vals 和其转置的元组 vals_list，创建包含 mask 和其转置的元组 mask_list
            vals_list = (vals, vals.t())
            mask_list = (mask, mask.t())
            # 创建空张量 out_dc，大小为 size*size，仅包含偶数索引位置的元素
            out_dc = torch.empty(size * size, device=device)[::2]
            # 遍历 vals_list 和 mask_list 的笛卡尔积，即每个组合 (v, m)
            for v, m in product(vals_list, mask_list):
                # 如果 m 是连续的，则从 v 中选择偶数列并展平为一维数组 expected
                if m.is_contiguous():
                    expected = v[:, ::2].clone().reshape((-1, ))
                # 否则，直接从 v 中选择偶数行并展平为一维数组 expected
                else:
                    expected = v[::2].clone().reshape((-1, ))
                # 使用 torch.masked_select 函数根据掩码 m 选择张量 v 中的元素，预期结果为 expected
                out = torch.masked_select(v, m)
                # 断言 out 等于 expected，允许的绝对误差和相对误差都为 0
                self.assertEqual(out, expected, atol=0, rtol=0)
                # 将 torch.masked_select 的结果写入预先分配的 out_dc 中
                torch.masked_select(v, m, out=out_dc)
                # 再次断言 out_dc 等于 expected，允许的绝对误差和相对误差都为 0
                self.assertEqual(out_dc, expected, atol=0, rtol=0)

    # FIXME: 找到适合掩码填充操作的测试套件
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16), (torch.uint8, torch.bool)))
    # 定义测试方法，接受设备和数据类型组合参数
    def test_masked_fill(self, device, dtypes):
        # 获取数据类型组合中的第一个数据类型作为 dtype
        dtype = dtypes[0]
        # 获取数据类型组合中的第二个数据类型作为 mask_dtype
        mask_dtype = dtypes[1]

        # 设置目标张量的大小为 10
        num_dest = 10
        # 创建全零张量 dst，数据类型为 dtype
        dst = torch.zeros(num_dest, dtype=dtype)
        # 创建随机整数张量 mask，大小为 num_dest，数据类型为 mask_dtype
        mask = torch.randint(2, (num_dest,), dtype=mask_dtype)
        # 随机选择一个值作为填充值 val
        val = random.random()
        # 复制 dst 到 dst2
        dst2 = dst.clone()

        # 如果 mask_dtype 不是 torch.bool 类型，预期抛出 RuntimeError 异常
        if mask_dtype is not torch.bool:
            with self.assertRaisesRegex(RuntimeError, 'only supports boolean masks'):
                dst.masked_fill_(mask, val)
            return

        # 使用 torch.masked_fill_ 函数根据 mask 填充 dst 中的元素为 val
        dst.masked_fill_(mask, val)
        # 手动根据 mask 修改 dst2 中的元素为 val
        for i in range(num_dest):
            if mask[i]:
                dst2[i] = val
        # 断言修改后的 dst 等于 dst2，允许的绝对误差和相对误差都为 0
        self.assertEqual(dst, dst2, atol=0, rtol=0)

        # 测试非连续情况
        # 创建三维张量 dst，并使其不连续，然后将其转置
        dst = ((torch.randn(num_dest, num_dest, num_dest) * 10).to(dtype)).permute((2, 0, 1))
        # 复制 dst 到 dst2，并保证 dst2 是连续的
        dst2 = dst.contiguous()
        # 如果 dtype 是复数类型，使用绝对值来创建掩码 mask，否则使用正值来创建掩码 mask
        if dtype.is_complex:
            mask = dst.abs() > 0
        else:
            mask = dst > 0
        # 断言 dst 不是连续的
        self.assertTrue(not dst.is_contiguous())
        # 断言 dst2 是连续的
        self.assertTrue(dst2.is_contiguous())
        # 使用 torch.masked_fill_ 函数根据 mask 填充 dst 中的元素为 val
        dst.masked_fill_(mask.to(mask_dtype), val)
        # 使用 torch.masked_fill_ 函数根据 mask 填充 dst2 中的元素为 val
        dst2.masked_fill_(mask.to(mask_dtype), val)
        # 断言修改后的 dst 等于 dst2，允许的绝对误差和相对误差都为 0
        self.assertEqual(dst, dst2, atol=0, rtol=0)

    # FIXME: 找到适合布尔张量掩码填充操作的测试套件
    def test_masked_fill_bool_tensor(self, device):
        # 创建布尔张量 dst
        dst = torch.tensor([True, False, True], device=device)
        # 创建布尔张量 mask
        mask = torch.tensor([False, True, False], device=device)

        # 使用 torch.masked_fill_ 函数根据 mask 填充 dst 中的元素为 True
        dst.masked_fill_(mask, True)
        # 断言修改后的 dst 等于预期的张量 [True, True, True]
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

        # 使用 torch.masked_fill 函数根据 mask 填充 dst 中的元素为 False，并赋值给 dst
        dst = dst.masked_fill(mask, False)
        # 断言修改后的 dst 等于预期的张量 [True, False, True]
        self.assertEqual(dst, torch.tensor([True, False, True], device=device))

    # 操作在一个维度上执行但不会减少尺寸的函数
    # FIXME: 找到适合 pdist 操作的测试套件
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @skipIfRocm
    @onlyCUDA
    # 使用装饰器定义一个大张量测试函数，设定内存为 '32GB'，运行在 CPU 设备上
    @largeTensorTest('32GB', device='cpu')
    # 使用装饰器定义一个大张量测试函数，设定内存为 '5GB'，运行在 CUDA 设备上
    @largeTensorTest('5GB', device='cuda')
    # 定义测试函数 test_pdist_norm_large，参数为 device
    def test_pdist_norm_large(self, device):
        # 使用 dim0>=46342 进行前向计算，参考：
        # https://github.com/pytorch/pytorch/issues/30583
        # 比较在 GPU 上的输出与 CPU 实现的输出
        x = torch.randn(50000, 1, dtype=torch.float32)      # 50k * 4 bytes = 200 KB
        # 预期在 CPU 上进行 p=2 范数计算，大约需要 1249975000 个 float32 数
        expected_cpu = torch.pdist(x, p=2)                  # 大约 1250M * 4 bytes = 5 GB 在 CPU 上
        # 在 GPU 上计算 p=2 范数，并将结果转移到 CPU
        actual_cpu = torch.pdist(x.to(device), p=2).cpu()   # 在 GPU 上约 5 GB + 在 CPU 上约 5 GB
        # 为了解决 self.assertTrue 的大内存开销问题（参见 issue #84944）
        self.assertTrue(torch.allclose(expected_cpu, actual_cpu))  # allclose 大约需要 20GB 的内存

    # FIXME: 将其移到元素级三元测试套件中
    # 仅在本地设备类型上执行该测试
    @onlyNativeDeviceTypes
    # 如果是 CUDA，设置为所有数学数据类型的集合
    @dtypesIfCUDA(*set(get_all_math_dtypes('cuda')))
    # 如果是 CPU，设置为所有数学数据类型的集合
    @dtypes(*set(get_all_math_dtypes('cpu')))
    # 定义一个测试函数，用于测试 torch.addcdiv 方法
    def test_addcdiv(self, device, dtype):
        # 返回与给定 dtype 对应的浮点或整数标量
        def _number(floating, integer, dtype):
            if dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
                return floating
            elif dtype in [torch.cfloat, torch.cdouble]:
                return floating * (1 + 1j)
            else:
                return integer

        # 生成一个非零随机张量，根据 dtype 和 device 不同采用不同的方式生成
        def non_zero_rand(size, dtype, device):
            if dtype.is_floating_point or dtype.is_complex:
                a = torch.rand(size=size, dtype=dtype, device=device)
            elif dtype == torch.uint8:
                a = torch.randint(1, 5, size=size, dtype=dtype, device=device)
            else:
                a = torch.randint(-5, 5, size=size, dtype=dtype, device=device)
            return a + (a == 0).to(dtype)

        # 定义一个内部测试函数，用于测试 addcdiv 方法
        def _test_addcdiv():
            # 生成非零随机张量
            a = non_zero_rand((2, 2), dtype=dtype, device=device)
            b = non_zero_rand((2, 2), dtype=dtype, device=device)
            c = non_zero_rand((2, 2), dtype=dtype, device=device)
            # 根据 dtype 生成一个标量 alpha
            alpha = _number(0.5, 3, dtype)

            # 计算预期结果
            expected = a + (alpha * b) / c
            # 调用 torch.addcdiv 方法计算实际结果
            actual = torch.addcdiv(a, b, c, value=alpha)
            # 断言预期结果与实际结果相等
            self.assertEqual(expected, actual)

            # 使用 assertWarnsOnceRegex 断言一次性发出 UserWarning，警告信息为 "This overload of addcdiv is deprecated"
            with self.assertWarnsOnceRegex(
                    UserWarning, "This overload of addcdiv is deprecated"):
                # 再次调用 torch.addcdiv 方法并断言结果相等
                self.assertEqual(actual, torch.addcdiv(a, alpha, b, c))

        # 如果 dtype 不是浮点数或复数，则禁止使用 addcdiv 进行整数除法操作，应该抛出 RuntimeError
        if not (dtype.is_floating_point or dtype.is_complex):
            with self.assertRaises(RuntimeError):
                _test_addcdiv()
        else:
            _test_addcdiv()

        # 如果设备类型为 'cuda'，且 dtype 为 torch.half，则进行特定情况下的测试
        if self.device_type == 'cuda' and dtype == torch.half:
            # 在 GPU 上生成特定张量
            a = torch.tensor([60000.0], device=device, dtype=dtype)
            b = torch.tensor([60000.0], device=device, dtype=dtype)
            c = torch.tensor([1.0], device=device, dtype=dtype)
            # 调用 torch.addcmul 方法
            out = torch.addcmul(a, b, c, value=-2)
            # 断言结果不是 NaN 或无穷大
            self.assertTrue(not (out.isnan() or out.isinf()))

    # 测试空操作，期望触发 RuntimeError 异常，异常信息为 'unsupported operation'
    def test_nullary_op_mem_overlap(self, device):
        # 定义一组操作及其参数
        ops = (
            ("random_", ()),
            ("uniform_", ()),
            ("cauchy_", ()),
            ("log_normal_", ()),
            ("exponential_", ()),
            ("geometric_", (0.5,)),
            ("normal_", ()),
        )

        # 生成一个张量 x，并将其形状扩展为 (3, 3)
        x = torch.rand((1, 3)).expand((3, 3))
        # 遍历操作及参数，并逐个调用 x 上的操作，预期每次操作都会抛出 RuntimeError 异常
        for op, args in ops:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                getattr(x, op)(*args)

    # TODO: 将该测试移到元素级三元运算的测试套件中，并将其改为 OpInfo 测试
    # https://github.com/pytorch/pytorch/issues/126474
    @xfailIfTorchDynamo
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/126474")
    @dtypes(torch.double)
    # 定义一个测试函数，用于测试三元操作的内存重叠情况
    def test_ternary_op_mem_overlap(self, device, dtype):
        # 如果设备是 "cpu" 并且设置了测试标志 TEST_WITH_TORCHINDUCTOR，则跳过测试
        if device == "cpu" and TEST_WITH_TORCHINDUCTOR:
            self.skipTest("Failing on cpu")

        # 定义一组操作的元组列表
        ops = [
            ("addcmul", True, True, 'cpu'),
            ("addcmul", True, True, 'cuda'),
            ("addcdiv", True, True, 'cpu'),
            ("addcdiv", True, True, 'cuda'),
            ("lerp", True, True, 'cpu'),
            ("lerp", True, True, 'cuda')
        ]

        # 遍历操作列表
        for (fn, has_input_output_mem_overlap_check,
             has_internal_mem_overlap_check, dev) in ops:
            # 如果当前操作的设备不是目标设备，则继续下一个操作
            if dev != device:
                continue
            # 获取操作的函数对象
            out_op = getattr(torch, fn)
            inplace_op = getattr(torch.Tensor, fn + '_')
            # 检查是否存在内部内存重叠，期望失败条件由 has_internal_mem_overlap_check 决定
            self.check_internal_mem_overlap(
                inplace_op, 3, dtype, device,
                expected_failure=not has_internal_mem_overlap_check)
            # 检查输入和输出的内存重叠情况，期望失败条件由 has_input_output_mem_overlap_check 决定
            self.ternary_check_input_output_mem_overlap(out_op, dev,
                                                        expected_failure=not has_input_output_mem_overlap_check)

    @expectedFailureMeta  # RuntimeError not raised
    @dtypes(torch.double)
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试复制操作的内存重叠情况
    def test_copy_mem_overlap(self, device, dtype):
        # 检查是否存在内部内存重叠
        self.check_internal_mem_overlap(
            torch.Tensor.copy_, num_inputs=2, dtype=dtype, device=device)
        # 定义一个大小为 9 的张量，并初始化为随机双精度数值
        sz = 9
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        # 检查输入和输出的内存重叠情况，使用 out.copy_(input) 进行检查
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: out.copy_(input))

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试 index_add_ 操作的内存重叠情况
    def test_index_add_mem_overlap(self, device):
        # 创建一个随机张量 x，并扩展为长度为 6 的张量
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        # 创建一个索引张量 ind，并创建一个值张量 value
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        # 使用 with 语句检查运行时错误是否包含 "unsupported operation"，期望抛出 RuntimeError
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.index_add_(0, ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_add_(0, ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_add_(0, ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_add_(0, ind.clone(), ind)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    # 定义测试方法，测试索引复制时的内存重叠情况
    def test_index_copy_mem_overlap(self, device):
        # 创建长度为 6 的随机张量 x
        x = torch.rand((1,), device=device).expand((6,))
        # 创建长度为 6 的随机张量 y
        y = torch.rand((6,), device=device)
        # 创建索引张量 ind 包含 [2, 1, 0]
        ind = torch.tensor([2, 1, 0], device=device)
        # 创建值张量 value 包含 3 个随机数
        value = torch.rand((3,), device=device)
        
        # 测试在运行时是否抛出 RuntimeError 异常并包含 'unsupported operation' 字符串
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 x 进行索引复制，使用 ind 作为索引，value 作为值
            x.index_copy_(0, ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 y 进行索引复制，使用 ind 作为索引，y[:3] 作为值
            y.index_copy_(0, ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 ind 进行索引复制，使用 ind 作为索引，ind 的克隆作为值
            ind.index_copy_(0, ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 ind 进行索引复制，使用 ind 的克隆作为索引，ind 作为值
            ind.index_copy_(0, ind.clone(), ind)

    # FIXME: 转换为 ErrorInputs
    # （但必须扩展 ErrorInputs 以处理仅限就地操作的错误！）
    @expectedFailureMeta  # 警告未触发
    @onlyNativeDeviceTypes
    # 定义测试方法，测试索引填充时的内存重叠情况
    def test_index_fill_mem_overlap(self, device):
        # 创建长度为 6 的随机张量 x
        x = torch.rand((1,), device=device).expand((6,))
        # 创建长度为 6 的随机张量 y
        y = torch.rand((6,), device=device)
        # 创建索引张量 ind 包含 [2, 1, 0]
        ind = torch.tensor([2, 1, 0], device=device)
        # 创建值张量 value 包含 3 个随机数
        value = torch.rand((3,), device=device)

        # 测试是否发出 UserWarning 警告并包含 "index_fill_ on expanded tensors" 字符串
        with self.assertWarnsRegex(UserWarning, "index_fill_ on expanded tensors"):
            # 对张量 x 进行索引填充，使用 ind 作为索引，填充值为 1.0
            x.index_fill_(0, ind, 1.0)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 ind 进行索引填充，使用 ind 作为索引，填充值为 0
            ind.index_fill_(0, ind, 0)

    # FIXME: 转换为 ErrorInputs
    @expectedFailureMeta  # RuntimeError 未抛出
    @onlyNativeDeviceTypes
    # 定义测试方法，测试位移操作时的内存重叠情况
    def test_shift_mem_overlap(self, device):
        # 创建长度为 3 的随机张量 x
        x = torch.rand(3, device=device)
        # 测试是否抛出 RuntimeError 异常并包含 'unsupported operation' 字符串
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 x 进行位左移操作
            x[:-1] <<= x[1:]
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 x 进行位右移操作
            x[:-1] >>= x[1:]

    # FIXME: 转换为 ErrorInputs
    # （但必须扩展 ErrorInputs 以处理仅限就地操作的错误！）
    @expectedFailureMeta  # RuntimeError 未抛出
    @onlyNativeDeviceTypes
    # 定义测试方法，测试伯努利抽样操作时的内存重叠情况
    def test_bernoulli_mem_overlap(self, device):
        # 创建长度为 6 的随机张量 x
        x = torch.rand((1,), device=device).expand((6,))

        # 测试是否抛出 RuntimeError 异常并包含 'unsupported operation' 字符串
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 x 进行伯努利抽样操作
            x.bernoulli_()
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 x 进行伯努利抽样操作，指定概率 p=0.1
            x.bernoulli_(p=0.1)
        p = torch.rand(6, device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 对张量 x 进行伯努利抽样操作，概率由张量 p 决定
            x.bernoulli_(p=p)
    # 测试在内存重叠情况下的 put_ 方法
    def test_put_mem_overlap(self, device):
        # 创建一个形状为 (6,) 的张量 x，并使用 device 参数指定设备
        x = torch.rand((1,), device=device).expand((6,))
        # 创建一个形状为 (6,) 的张量 y，并使用 device 参数指定设备
        y = torch.rand((6,), device=device)
        # 创建一个形状为 (3,) 的张量 ind，并使用 device 参数指定设备
        ind = torch.tensor([2, 1, 0], device=device)
        # 创建一个形状为 (3,) 的张量 value，并使用 device 参数指定设备
        value = torch.rand((3,), device=device)
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 x 的 put_ 方法，向索引 ind 放置值 value
            x.put_(ind, value)
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 y 的 put_ 方法，向索引 ind[0] 放置值 y[0]
            y.put_(ind[0], y[0])
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 ind 的 put_ 方法，向索引 ind 放置值 ind
            ind.put_(ind, ind)
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 y 的 put_ 方法，向索引 ind 放置值 y[:3]
            y.put_(ind, y[:3])
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 ind 的 put_ 方法，向索引 ind 放置值 ind 的克隆
            ind.put_(ind, ind.clone())
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 ind 的 put_ 方法，向索引 ind 的克隆放置值 ind
            ind.put_(ind.clone(), ind)


```    
    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    # 测试在内存重叠情况下的 index_put_ 方法
    def test_index_put_mem_overlap(self, device):
        # 创建一个形状为 (6,) 的张量 x，并使用 device 参数指定设备
        x = torch.rand((1,), device=device).expand((6,))
        # 创建一个形状为 (6,) 的张量 y，并使用 device 参数指定设备
        y = torch.rand((6,), device=device)
        # 创建一个形状为 (3,) 的张量 ind，并使用 device 参数指定设备
        ind = torch.tensor([2, 1, 0], device=device)
        # 创建一个形状为 (3,) 的张量 value，并使用 device 参数指定设备
        value = torch.rand((3,), device=device)
        # 使用 assertWarnsRegex 断言在用户警告(UserWarning)下抛出异常，异常信息包含 'expanded tensors'
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            # 调用 x 的 index_put_ 方法，向索引 (ind,) 放置值 value
            x.index_put_((ind,), value)
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 y 的 index_put_ 方法，向索引 (ind,) 放置值 y[0]
            y.index_put_((ind,), y[0])
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 ind 的 index_put_ 方法，向索引 (ind,) 放置值 ind
            ind.index_put_((ind,), ind)
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 y 的 index_put_ 方法，向索引 (ind,) 放置值 y[:3]
            y.index_put_((ind,), y[:3])
        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 ind 的 index_put_ 方法，向索引 (ind.clone(),) 放置值 ind
            ind.index_put_((ind.clone(),), ind)


```py    
    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    # 测试在内存重叠情况下的 masked_fill_ 方法
    def test_masked_fill_mem_overlap(self, device):
        # 创建一个形状为 (6,) 的张量 x，并使用 device 参数指定设备
        x = torch.rand((1,), device=device).expand((6,))
        # 创建一个形状为 (6,) 的布尔张量 mask，并使用 device 参数指定设备
        mask = torch.tensor([True, False, True, True, False, False], device=device)
        # 使用 assertWarnsRegex 断言在用户警告(UserWarning)下抛出异常，异常信息包含 'expanded tensors'
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            # 调用 x 的 masked_fill_ 方法，使用 mask 进行填充，填充值为 0.0
            x.masked_fill_(mask, 0.)

        # 创建一个形状为 () 的浮点数张量 fill_val，并使用 device 参数指定设备
        fill_val = torch.tensor(0., device=device)
        # 使用 assertWarnsRegex 断言在用户警告(UserWarning)下抛出异常，异常信息包含 'expanded tensors'
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            # 调用 x 的 masked_fill_ 方法，使用 mask 进行填充，填充值为 fill_val
            x.masked_fill_(mask, fill_val)

        # 使用 assertRaisesRegex 断言在运行时错误(RuntimeError)下抛出异常，异常信息包含 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 调用 mask[1:] 的 masked_fill_ 方法，使用 mask[:-1] 进行填充，填充值为 False
            mask[1:].masked_fill_(mask[:-1], False)
    # 限制仅在本地设备类型上运行此测试
    @onlyNativeDeviceTypes
    # 测试在内存重叠情况下的 masked_scatter_ 方法
    def test_masked_scatter_mem_overlap(self, device):
        # 创建一个大小为 (6,) 的张量 x，填充随机数据，并复制到指定设备上
        x = torch.rand((1,), device=device).expand((6,))
        # 创建一个大小为 (3,) 的张量 src，填充随机数据，并复制到指定设备上
        src = torch.rand((3,), device=device)
        # 创建一个大小为 (6,) 的布尔掩码张量 mask，指定设备为 device
        mask = torch.tensor([True, False, True, True, False, False], device=device)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，异常消息为 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 在 x 上执行 masked_scatter_ 操作，使用 mask 和 src
            x.masked_scatter_(mask, src)

    # FIXME: 转换为 ErrorInputs
    # (但必须扩展 ErrorInputs 以处理仅限原地操作的错误！)
    # 限制仅在本地设备类型上运行此测试
    @onlyNativeDeviceTypes
    # 测试在内存重叠情况下的 scatter_ 方法
    def test_scatter_mem_overlap(self, device):
        # 创建一个大小为 (6,) 的张量 x，填充随机数据，并复制到指定设备上
        x = torch.rand((1,), device=device).expand((6,))
        # 创建一个大小为 (3,) 的张量 src，填充随机数据，并复制到指定设备上
        src = torch.rand((3,), device=device)
        # 创建一个大小为 (3,) 的索引张量 ind，填充指定索引，并复制到指定设备上，数据类型为 torch.int64
        ind = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，异常消息为 'unsupported operation'
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 在 x 上执行 scatter_ 操作，使用维度 0，ind 和 src
            x.scatter_(0, ind, src)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 在 src 上执行 scatter_ 操作，使用维度 0，ind 和 src
            src.scatter_(0, ind, src)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            # 在 ind 上执行 scatter_ 操作，使用维度 0，ind 和 ind 的克隆
            ind.scatter_(0, ind, ind.clone())

    # FIXME: 移动到测试分布函数
    # 限制仅在 CUDA 设备上运行此测试
    @onlyCUDA
    # 测试在多项式分布时的设备限制
    def test_multinomial_device_constrain(self, device):
        # 创建一个大小为 (3,) 的未初始化张量 x，设备为 "cpu"
        x = torch.empty(3, device="cpu")
        # 创建一个大小为 (3,) 的未初始化张量 y，设备为指定设备
        y = torch.empty(3, device=device)
        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，异常消息为 "Expected all tensors to be on the same device"
        self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device",
            # 调用 torch.multinomial 函数，从 x 中采样 2 个样本，输出到 y
            lambda: torch.multinomial(x, 2, out=y))

    # FIXME: 移动到测试分布函数
    # 要求至少有两个设备
    @deviceCountAtLeast(2)
    # 限制仅在 CUDA 设备上运行此测试
    @onlyCUDA
    # 测试在 GPU 设备上多项式分布的设备限制
    def test_multinomial_gpu_device_constrain(self, devices):
        # 创建一个大小为 (3,) 的未初始化张量 x，设备为 devices[0]
        x = torch.empty(3, device=devices[0])
        # 创建一个大小为 (3,) 的未初始化张量 y，设备为 devices[1]
        y = torch.empty(3, device=devices[1])
        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，异常消息为 "Expected all tensors to be on the same device"
        self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device",
            # 调用 torch.multinomial 函数，从 x 中采样 2 个样本，输出到 y
            lambda: torch.multinomial(x, 2, out=y))

    # FIXME: 转换此测试为自动化 OpInfo 测试
    # 要求至少有两个设备
    @deviceCountAtLeast(2)
    # 限制仅在 CUDA 设备上运行此测试
    @onlyCUDA
    def test_device_guard(self, devices):
        # 验证所有具有 `device_guard: False` 的运算符在多设备下的正确行为。
        # TODO: 如果我们有运算符内省，可以自动找出这组运算符...
        
        # 创建一个在指定设备上的随机张量
        x = torch.randn((1, 2, 3), device=devices[1])
        # 创建一个在指定设备上的全零张量
        y = torch.zeros((1, 3, 2), device=devices[1])
        # 创建一个在指定设备上的标量张量
        scalar = torch.tensor(5, device=devices[1])

        # 属性操作
        torch.cudnn_is_acceptable(x)  # 检查是否适合使用 cuDNN 的张量
        x.is_distributed()  # 检查张量是否分布式存储
        x.is_floating_point()  # 检查张量是否浮点型
        x.is_complex()  # 检查张量是否复数类型
        x.is_same_size(y)  # 检查张量是否与另一个张量大小相同
        x.is_signed()  # 检查张量是否有符号
        x.size(0)  # 返回张量第一维的大小
        x.stride(0)  # 返回张量第一维的步幅
        x.numel()  # 返回张量元素总数
        x.is_set_to(y)  # 检查张量是否设置为另一个张量的值
        x.data_ptr()  # 返回张量数据的指针地址
        scalar.is_nonzero()  # 检查标量张量是否非零

        # 稀疏属性操作
        y[0][1] = 5  # 修改稀疏张量的值
        y_sparse = y.to_sparse()  # 将张量转换为稀疏张量
        y_sparse.sparse_dim()  # 返回稀疏张量的稀疏维度
        y_sparse._dimI()  # 返回稀疏张量的索引维度大小
        y_sparse.dense_dim()  # 返回稀疏张量的稠密维度
        y_sparse._dimV()  # 返回稀疏张量的值维度大小
        y_sparse._nnz()  # 返回稀疏张量的非零元素数量
        y_sparse.is_coalesced()  # 检查稀疏张量是否已合并
        y_sparse._indices()  # 返回稀疏张量的索引
        y_sparse._values()  # 返回稀疏张量的值
        y_sparse.indices()  # 返回稀疏张量的索引
        y_sparse.values()  # 返回稀疏张量的值

        # 原地操作
        def inplace():
            return torch.randn((1, 2, 3), device=devices[1])
        inplace().as_strided_(y.size(), y.stride())  # 将张量视图重设为指定大小和步幅
        inplace().resize_(y.size())  # 重置张量大小
        inplace().squeeze_()  # 压缩张量维度为 1
        inplace().squeeze_(0)  # 压缩张量第一维度为 1
        inplace().unsqueeze_(2)  # 增加张量维度为 1
        inplace().transpose_(1, 2)  # 转置张量的两个维度
        inplace().squeeze_().t_()  # 压缩张量维度并转置
        inplace().set_(x.storage())  # 设置张量值为另一个张量的存储
        inplace().set_(x.storage(), x.storage_offset(), x.size(), x.stride())  # 设置张量值为指定参数的存储
        inplace().set_(x)  # 设置张量值为另一个张量
        inplace().set_()  # 清空张量值
        y_sparse._coalesced_(True)  # 合并稀疏张量的值

        # 形状修改
        x.as_strided(y.size(), y.stride())  # 返回一个新的张量视图，具有指定大小和步幅
        x.expand((5, 2, 3))  # 扩展张量的大小
        x.expand_as(x)  # 以另一个张量的大小来扩展张量
        x.sum_to_size((1,))  # 将张量求和到指定大小
        torch.broadcast_tensors(x, x)  # 广播两个张量使它们可以进行元素级运算
        x.reshape((1, 3, 2))  # 重塑张量的形状
        x.reshape_as(y)  # 将张量重塑为另一个张量的形状
        x.squeeze()  # 压缩张量中所有维度为 1 的维度
        x.squeeze(0)  # 压缩张量第一维度为 1
        x.squeeze().t()  # 压缩张量维度并转置
        x.transpose(1, 2)  # 转置张量的两个维度
        x.unsqueeze(2)  # 在指定维度增加张量的维度为 1
        x.view((1, 3, 2))  # 返回一个张量视图，具有指定形状
        x.view_as(y)  # 返回一个与另一个张量相同形状的张量视图

        # 分块、分割等操作
        x.chunk(2, dim=1)  # 在指定维度上分块张量
        x.split(1, dim=2)  # 在指定维度上分割张量
        x.split_with_sizes([1, 2], dim=2)  # 在指定维度上按指定大小分割张量
        x.unfold(dimension=2, size=1, step=1)  # 在指定维度上展开张量

        x.narrow(1, 1, 1)  # 返回一个在指定维度上选取部分数据的新张量
        x.select(1, 1)  # 在指定维度上选取索引为 1 的数据
        torch.isnan(x)  # 检查张量是否包含 NaN 值

        torch.empty((1, 3, 2), out=y)  # 返回一个未初始化的张量，可以指定输出位置
        torch.empty_like(x)  # 返回一个与输入张量形状相同的未初始化张量
        torch.empty_like(x, dtype=torch.int64)  # 返回一个与输入张量形状相同且指定数据类型的未初始化张量

        # to 操作
        x.to(x)  # 将张量转换为与自身相同类型和设备的张量
        x.to(y)  # 将张量转换为与另一个张量相同设备的张量
        x.to(x, copy=True)  # 将张量转换为与自身相同类型和设备的张量，可选择复制数据

    def test_is_signed(self, device):
        # 检查不同类型张量在指定设备上是否有符号属性
        self.assertEqual(torch.IntTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.ByteTensor(5).to(device).is_signed(), False)
        self.assertEqual(torch.CharTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.FloatTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.HalfTensor(10).to(device).is_signed(), True)
    # 测试所有注册的张量类的类型
    def test_tensor_type(self):
        for t in torch._tensor_classes:
            # 检查是否在 CUDA 模块中，验证是否是 CUDA 张量
            if 'cuda' in t.__module__:
                self.assertEqual(t.is_cuda, True)
            else:
                self.assertEqual(t.is_cuda, False)
            # 检查是否在 XPU 模块中，验证是否是 XPU 张量
            if 'xpu' in t.__module__:
                self.assertEqual(t.is_xpu, True)
            else:
                self.assertEqual(t.is_xpu, False)

    # Note - reports a leak of 512 bytes on CUDA device 1
    # 检查在多GPU情况下张量设置错误的情况
    @deviceCountAtLeast(2)
    @skipCUDAMemoryLeakCheckIf(True)
    @onlyCUDA
    def test_tensor_set_errors_multigpu(self, devices):
        # 创建两个在不同 CUDA 设备上的随机张量
        f_cuda0 = torch.randn((2, 3), dtype=torch.float32, device=devices[0])
        f_cuda1 = torch.randn((2, 3), dtype=torch.float32, device=devices[1])

        # 测试各种张量设置错误情况是否会引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: f_cuda0.set_(f_cuda1.storage()))
        self.assertRaises(RuntimeError,
                          lambda: f_cuda0.set_(f_cuda1.storage(), 0, f_cuda1.size(), f_cuda1.stride()))
        self.assertRaises(RuntimeError, lambda: f_cuda0.set_(f_cuda1))

    # FIXME: move to test_serialization
    # 测试张量在序列化过程中的行为
    @onlyCUDA
    @deviceCountAtLeast(1)  # Note: Tests works with one but prefers more devices
    def test_serialization(self, devices):
        # 内部函数，测试张量的序列化行为
        def _test_serialization(filecontext_lambda):
            # 创建和填充一个 CUDA 浮点数张量
            t0 = torch.cuda.FloatTensor(5).fill_(1)
            with torch.cuda.device(devices[-1]):
                # 在指定设备上创建和填充另一个 CUDA 浮点数张量
                tn = torch.cuda.FloatTensor(3).fill_(2)
            torch.cuda.set_device(devices[0])
            b = (t0, tn)
            # 使用给定的文件上下文 lambda 函数打开临时文件
            with filecontext_lambda() as f:
                torch.save(b, f)
                f.seek(0)
                c = torch.load(f)
                # 验证加载后的张量与保存前的张量是否相等
                self.assertEqual(b, c, atol=0, rtol=0)
                u0, un = c
                # 验证加载后的张量的设备是否与预期设备匹配
                self.assertEqual(str(u0.device), devices[0])
                self.assertEqual(str(un.device), devices[-1])

        # 使用两种不同的文件上下文测试张量序列化行为
        _test_serialization(tempfile.NamedTemporaryFile)
        _test_serialization(BytesIOContext)

    # FIXME: move memory format tests to their own test class/suite
    # 测试在 permute 操作后内存格式是否得到保留
    def test_memory_format_preserved_after_permute(self, device):
        # 创建一个随机张量，并指定内存格式为 channels_last
        x = torch.randn(4, 3, 8, 8, device=device)
        nhwc = x.contiguous(memory_format=torch.channels_last)
        # 进行两次 permute 操作后，验证是否仍然保持 channels_last 内存格式
        y = nhwc.permute(0, 1, 3, 2).permute(0, 1, 3, 2)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))

        # 创建一个随机张量，并指定内存格式为 channels_last_3d
        x = torch.randn(4, 3, 8, 8, 8, device=device)
        ndhwc = x.contiguous(memory_format=torch.channels_last_3d)
        # 进行两次 permute 操作后，验证是否仍然保持 channels_last_3d 内存格式
        y = ndhwc.permute(0, 1, 4, 3, 2).permute(0, 1, 4, 3, 2)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last_3d))
    # 定义一个测试方法，用于验证内存格式传播规则的正确性，接受一个设备参数
    def test_memory_format_propagation_rules(self, device):

        # 创建一个连续存储的张量
        contiguous = torch.rand(10, 3, 5, 5, device=device)
        # 创建一个按通道优先存储的张量，并保证其连续性
        cl = torch.rand(10, 3, 5, 5, device=device).contiguous(memory_format=torch.channels_last)
        # 创建一个模糊的张量，并保证其连续性，并设置为按通道优先存储
        ambiguous = torch.rand(10, 3, 1, 1, device=device).contiguous(memory_format=torch.channels_last)
        # 验证模糊张量是否按通道优先存储
        self.assertTrue(ambiguous.is_contiguous(memory_format=torch.channels_last))
        # 验证模糊张量是否连续
        self.assertTrue(ambiguous.is_contiguous(memory_format=torch.contiguous_format))
        # 创建一个按通道优先存储的偏差张量
        bias = torch.rand(1, 1, 1, 1, device=device).contiguous(memory_format=torch.channels_last)

        # 定义一个内部方法，用于测试内存格式传播规则
        def _test_propagation_rules(self, contiguous, cl, ambiguous, bias):
            # 定义不同的张量组合及其预期的内存格式
            options = ((ambiguous, contiguous, torch.contiguous_format),
                       (ambiguous, cl, torch.channels_last),
                       (contiguous, ambiguous, torch.contiguous_format),
                       (contiguous, cl, torch.contiguous_format),
                       (cl, ambiguous, torch.channels_last),
                       (cl, contiguous, torch.channels_last),
                       (bias, cl, torch.channels_last),
                       (cl, bias, torch.channels_last),)

            # 遍历不同的张量组合，验证其相加结果是否满足预期的内存格式要求
            for a, b, mf in options:
                result = a + b
                self.assertTrue(result.is_contiguous(memory_format=mf))

        # 调用内部方法，测试内存格式传播规则
        _test_propagation_rules(self, contiguous, cl, ambiguous, bias)

        # 将按通道优先存储的张量转换为另一种内存格式，并再次测试内存格式传播规则
        cl = cl.to(memory_format=torch.channels_last)
        ambiguous = ambiguous.to(memory_format=torch.channels_last)
        bias = bias.to(memory_format=torch.channels_last)

        # 再次调用内部方法，测试转换后的内存格式传播规则
        _test_propagation_rules(self, contiguous, cl, ambiguous, bias)

        # 在模糊张量的情况下，测试当步长对结果有影响的情况
        for mf in (torch.channels_last, torch.contiguous_format):
            ambiguous = torch.rand(10, 3, 1, 1, device=device).to(memory_format=mf)
            bias = torch.rand(3, 1, 1, device=device)
            result = ambiguous + bias
            # 验证结果张量的步长是否与模糊张量相同
            self.assertEqual(ambiguous.stride(), result.stride())
            result = bias + ambiguous
            # 验证结果张量的步长是否与模糊张量相同
            self.assertEqual(ambiguous.stride(), result.stride())
            result = ambiguous * 5
            # 验证结果张量的步长是否与模糊张量相同
            self.assertEqual(ambiguous.stride(), result.stride())
    # 定义一个测试函数，用于验证内存格式相关功能中空的占位符
    def test_memory_format_empty_like(self, device):
        # 内部辅助函数，用于测试不同的输入张量和内存格式
        def test_helper(x, memory_format):
            # 获得连续化后的张量 xc
            xc = x.contiguous(memory_format=memory_format)

            # 创建一个与 xc 形状相同的空张量 like，并指定保持格式不变
            like = torch.empty_like(xc, memory_format=torch.preserve_format)
            # 断言 like 不是连续的
            self.assertFalse(like.is_contiguous())
            # 断言 like 在指定的内存格式下是连续的
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            # 创建一个与 x 形状相同的空张量 like_x，并指定保持格式不变
            like_x = torch.empty_like(x, memory_format=torch.preserve_format)
            # 断言 like_x 是连续的
            self.assertTrue(like_x.is_contiguous())
            # 断言 like_x 在指定的内存格式下不是连续的
            self.assertFalse(like_x.is_contiguous(memory_format=memory_format))

            # 创建一个与 x 形状相同的空张量 like，并指定特定的内存格式
            like = torch.empty_like(x, memory_format=memory_format)
            # 断言 like 不是连续的
            self.assertFalse(like.is_contiguous())
            # 断言 like 在指定的内存格式下是连续的
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            # 创建一个与 xc 形状相同的空张量 like，并指定连续格式
            like = torch.empty_like(xc, memory_format=torch.contiguous_format)
            # 断言 like 是连续的
            self.assertTrue(like.is_contiguous())
            # 断言 like 在指定的内存格式下不是连续的
            self.assertFalse(like.is_contiguous(memory_format=memory_format))

            # 创建一个与 xc 形状相同的空张量 like，但未指定内存格式
            like = torch.empty_like(xc)
            # 断言 like 不是连续的
            self.assertFalse(like.is_contiguous())
            # 断言 like 在指定的内存格式下是连续的
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            # 将 x 转换为稀疏张量 sparse
            sparse = x.to_sparse()
            # 断言在保持格式不变的情况下无法创建稀疏张量的空张量
            with self.assertRaises(RuntimeError):
                z = torch.empty_like(sparse, memory_format=torch.preserve_format)

        # 使用 channels_last 内存格式测试辅助函数
        test_helper(torch.randn(4, 3, 8, 8, device=device), torch.channels_last)
        # 使用 channels_last_3d 内存格式测试辅助函数
        test_helper(torch.randn(4, 3, 8, 8, 8, device=device), torch.channels_last_3d)

    # 测试内存格式的一致性
    def test_memory_format_consistency(self, device):
        # 创建一个形状为 (10, 3, 1, 1) 的随机张量 x
        x = torch.randn(10, 3, 1, 1, device=device)
        # 通过 as_strided 创建 x 的一个视图 x_rep，形状和步幅与 x 相同
        x_rep = x.as_strided(x.size(), x.stride())
        # 断言 x 和 x_rep 的形状相同
        self.assertEqual(x.size(), x_rep.size())
        # 断言 x 和 x_rep 的步幅相同
        self.assertEqual(x.stride(), x_rep.stride())
        # 断言 x 是连续的
        self.assertEqual(x.is_contiguous(), x_rep.is_contiguous())
        # 断言在 channels_last 内存格式下，x 和 x_rep 都是连续的
        self.assertEqual(x.is_contiguous(memory_format=torch.channels_last), x_rep.is_contiguous(memory_format=torch.channels_last))
        # 断言在 channels_last_3d 内存格式下，x 和 x_rep 都是连续的
        self.assertEqual(x.is_contiguous(memory_format=torch.channels_last_3d), x_rep.is_contiguous(memory_format=torch.channels_last_3d))

    # FIXME: 将此部分作为逐元素一元和二元 OpInfo 测试
    # FIXME: 将此部分作为逐元素一元和二元 OpInfo 测试
    # 定义测试方法，用于测试张量操作的步长传播特性，在指定设备上进行测试
    def test_strides_propagation(self, device):
        # 内部辅助函数，用于执行具体的测试
        def _test_helper(x, op, unary=False):
            # 比较两个步长数组，验证它们在除法后是否相等
            def compare_strides(s1, s2, div):
                sdiv = [s // div for s in s1]
                self.assertEqual(sdiv, s2)

            # 获取张量的维度
            dim = x.dim()
            # 如果是单目运算，则直接对输入张量进行操作
            # 否则，对输入张量或其它随机张量进行操作
            div = x.stride(-1)  # 获取张量在最后一个维度上的步长

            # 对张量维度的所有排列进行遍历
            for p in permutations(range(dim)):
                xp = x.permute(p)  # 对输入张量按照排列p进行重新排序
                if not unary:
                    # 如果是二元操作，则生成一个与xp最后一个维度大小相同的随机张量y
                    y = torch.randn(xp.size(-1), device=x.device, dtype=x.dtype)
                    # 对于三种输入组合进行操作：(xp, xp), (xp, y), (y, xp)
                    for inputs in ((xp, xp), (xp, y), (y, xp)):
                        res = op(*inputs)  # 执行操作
                        compare_strides(xp.stride(), res.stride(), div)  # 比较步长
                        self.assertEqual(xp.size(), res.size())  # 验证输出大小
                        out = torch.empty(0, device=xp.device, dtype=res.dtype)  # 创建一个空张量out
                        res = op(*inputs, out=out)  # 将结果写入out张量
                        compare_strides(xp.stride(), res.stride(), div)  # 比较步长
                        self.assertEqual(xp.size(), res.size())  # 验证输出大小
                else:
                    # 如果是单目操作，则直接对xp进行操作
                    res = op(xp)  # 执行单目操作
                    compare_strides(xp.stride(), res.stride(), div)  # 比较步长
                    self.assertEqual(xp.size(), res.size())  # 验证输出大小
                    out = torch.empty(0, device=xp.device, dtype=res.dtype)  # 创建一个空张量out
                    res = op(xp, out=out)  # 将结果写入out张量
                    compare_strides(xp.stride(), res.stride(), div)  # 比较步长
                    self.assertEqual(xp.size(), res.size())  # 验证输出大小

        # 定义需要测试的二元和单目操作
        binary_ops = (torch.eq, torch.add)
        unary_ops = (torch.exp,)
        # 定义不同形状的输入张量，用于测试
        xs = (torch.randn(2, 3, 4, device=device), torch.randn(2, 3, 8, device=device)[:, :, ::2],
              torch.randn(1, 1, 4, 12, device=device)[:, :, :, ::2])
        
        # 对于每个二元操作，以及每种输入张量形状，执行测试
        for op in binary_ops:
            for x in xs:
                _test_helper(x, op)
        
        # 对于每个单目操作，以及每种输入张量形状，执行测试
        for op in unary_ops:
            for x in xs:
                _test_helper(x, op, unary=True)

    # 下面的装饰器指定只在CUDA上运行的测试，并跳过特定条件下的测试
    @onlyCUDA
    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    @skipIfTorchDynamo("NotImplementedError: PrimTorch does not support pinned memory")
    # 定义一个测试函数，用于测试 pin_memory 参数在构造张量时的效果
    def test_pin_memory_from_constructor(self, device):
        # 定义一个内部函数，用于生成与给定张量 t 相同类型的随机张量集合
        def _get_like(t, **kwargs):
            return [
                torch.rand_like(t, **kwargs),      # 生成与 t 相同形状和类型的随机张量
                torch.randn_like(t, **kwargs),     # 生成与 t 相同形状和类型的标准正态分布张量
                torch.empty_like(t, **kwargs),     # 生成与 t 相同形状和类型的空张量
                torch.full_like(t, 4, **kwargs),   # 生成与 t 相同形状和类型，填充为指定值的张量
                torch.zeros_like(t, **kwargs),     # 生成与 t 相同形状和类型的零张量
                torch.ones_like(t, **kwargs),      # 生成与 t 相同形状和类型的全一张量
            ]

        # 定义一个内部函数，用于生成多种类型的张量集合
        def _get_tensors(**kwargs):
            return [
                torch.tensor([10, 11], **kwargs),  # 生成指定形状和类型的张量
                torch.randn(3, 5, **kwargs),       # 生成指定形状的标准正态分布张量
                torch.rand(3, **kwargs),           # 生成指定形状的随机张量
                # torch.randint(3, 5, **kwargs), // 不支持的操作，已被注释掉
                torch.zeros(3, **kwargs),          # 生成指定形状的零张量
                torch.randperm(3, **kwargs),       # 生成指定大小的随机排列张量
                torch.empty(6, **kwargs),          # 生成指定形状的空张量
                torch.ones(6, **kwargs),           # 生成指定形状的全一张量
                torch.eye(6, **kwargs),            # 生成指定大小的单位矩阵张量
                torch.arange(3, 5, **kwargs)       # 生成指定范围的等差序列张量
            ]

        # 获取 pin_memory=True 的张量集合，并与 _get_like 生成的张量集合合并
        pinned_tensors = _get_tensors(pin_memory=True) + _get_like(torch.empty(5, dtype=torch.float64), pin_memory=True)
        # 遍历 pinned_tensors，断言每个张量都是被固定在内存中的
        for x in pinned_tensors:
            self.assertTrue(x.is_pinned())

        # 获取未使用 pin_memory=True 的张量集合，并与 _get_like 生成的固定内存张量合并
        tensors = _get_tensors() + _get_like(torch.empty(5, dtype=torch.float64, pin_memory=True))
        # 遍历 tensors，断言每个张量都不是被固定在内存中的
        for x in tensors:
            self.assertFalse(x.is_pinned())

    # 使用装饰器 @deviceCountAtLeast(1) 和 @onlyCUDA 标记此测试函数
    def test_storage_all_devices(self, devices):
        # 遍历所有设备
        for device in devices:
            # 在指定设备上创建一个空张量
            t = torch.tensor((), device=device)
            # 断言张量 t 的数据类型与其存储的数据类型相同
            self.assertEqual(t.dtype, t.storage().dtype)

    # 注释部分包含关于 lazy_clone_ 测试的详细说明，以及与动态计算图相关的注意事项
    # 这些测试用例设计成可以在即时模式和编译模式 (`PYTORCH_TEST_WITH_INDUCTOR=1`) 下都能通过
    # 在编译模式和即时模式下，存在 COW 张量材料化时间和方式不同的情况，需要避免这些情况
    # 两个主要注意点需要关注：
    # 第一点是测试必须检查张量的内部属性，以确保它们按预期材料化，这些检查会导致动态图断裂
    # 解决方法是将所有对 COW 张量的操作编译进同一个计算图中，不在操作之间做任何检查，只在测试结束时做检查
    # 如果确实需要在两个操作之间进行检查，解决方法是创建两个不同的测试：一个测试只执行 `op1` 并检查，另一个测试执行 `op1` 然后立即执行 `op2` 再检查
    #
    # 第二点是在即时模式下，如果我们在两个 COW
    # 在编译模式下，懒克隆操作可能导致张量在同一计算图中的材料化顺序与即时模式不同。
    # 为了保证期望的顺序，这里通过在两个写操作之间添加检查，故意引发计算图断裂。
    # 这样做可以确保它们按预期的顺序材料化。
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone(self, device, dtype):
        # 创建一个张量 t
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        # 记录原始的存储地址和数据指针地址
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        # 对张量 t 进行懒克隆操作，返回克隆张量 clone
        clone = t._lazy_clone()

        # 懒克隆一个张量应使其及其克隆变为 COW 张量。
        # 它们应该有不同的存储，但是相同的数据指针。
        self.assertTrue(torch._C._is_cow_tensor(clone))
        self.assertTrue(torch._C._is_cow_tensor(t))

        # 检查克隆张量和原张量的存储地址是否一致
        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        # 检查克隆张量和原张量的数据指针地址是否一致
        self.assertTrue(torch._C._data_address(t) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

    # 查看带有感应电流启用的 lazy_clone_ 测试，请参见备注 [lazy_clone_ tests with inductor enabled]
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_view(self, device, dtype):
        # 创建一个张量 t
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        # 记录原始的存储地址和数据指针地址
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        # 对张量 t 进行懒克隆操作，返回克隆张量 clone
        clone = t._lazy_clone()
        # 创建一个视图张量 view
        view = t.view([4])

        # 查看张量 t 不应导致复制（材料化）操作发生。
        # 所有张量仍应为 COW 张量，并具有相同的数据指针。
        # view 和 t 应该有相同的存储，clone 应该有不同的存储。
        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(view))
        self.assertTrue(torch._C._is_cow_tensor(clone))

        # 检查张量 t 的存储地址是否与原始地址相同
        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        # 检查张量 t、view 和 clone 的数据指针地址是否与原始地址相同
        self.assertTrue(torch._C._data_address(t) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(view) == orig_data_ptr)

    # 查看带有感应电流启用的 lazy_clone_ 测试，请参见备注 [lazy_clone_ tests with inductor enabled]
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_view_materialize(self, device, dtype):
        # 创建一个张量 `t`，指定设备和数据类型
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        # 记录张量 `t` 的原始存储地址
        t_orig_storage_addr = torch._C._storage_address(t)
        # 记录张量 `t` 的原始数据地址
        orig_data_ptr = torch._C._data_address(t)
        # 创建 `t` 的惰性克隆 `clone`
        clone = t._lazy_clone()
        # 创建 `t` 的视图 `view`，重新塑形为长度为 4 的向量，并加上一个值为 1 的张量
        view = t.view([4])
        view += torch.ones(1, device=device, dtype=dtype)

        # 写入 `t` 应导致 `t` 和 `view` 下的存储被复制（实质化），但不影响 `clone`。

        # 检查 `t` 是否为 COW 张量（写时复制）
        self.assertFalse(torch._C._is_cow_tensor(t))
        # 检查 `view` 是否为 COW 张量
        self.assertFalse(torch._C._is_cow_tensor(view))
        # 检查 `clone` 是否为 COW 张量
        self.assertTrue(torch._C._is_cow_tensor(clone))

        # 检查 `t` 的存储地址是否与原始相同
        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        # 检查 `view` 的存储地址是否与 `t` 相同
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        # 检查 `clone` 的存储地址是否与原始不同
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        # 记录 `t` 的新数据地址
        t_new_data_addr = torch._C._data_address(t)
        # 检查 `t` 的数据地址是否与原始不同
        self.assertTrue(t_new_data_addr != orig_data_ptr)
        # 检查 `view` 的数据地址是否与 `t` 的新数据地址相同
        self.assertTrue(torch._C._data_address(view) == t_new_data_addr)
        # 检查 `clone` 的数据地址是否与原始相同
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

        # 对 `clone` 加上一个值为 1 的张量

        # 写入 `clone` 应使其实质化，因此不再是 COW 张量。
        # 然而，由于 `clone` 的存储是仅剩下引用原始数据指针的 COW 存储，因此这种实质化实际上不会导致复制，而是会重用原始数据指针。

        # 检查 `t` 是否为 COW 张量
        self.assertFalse(torch._C._is_cow_tensor(t))
        # 检查 `view` 是否为 COW 张量
        self.assertFalse(torch._C._is_cow_tensor(view))
        # 检查 `clone` 是否为 COW 张量
        self.assertFalse(torch._C._is_cow_tensor(clone))

        # 检查 `t` 的存储地址是否与原始相同
        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        # 检查 `view` 的存储地址是否与 `t` 相同
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        # 检查 `clone` 的存储地址是否与原始不同
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        # 检查 `t` 的数据地址是否与新的相同
        self.assertTrue(torch._C._data_address(t) == t_new_data_addr)
        # 检查 `view` 的数据地址是否与新的相同
        self.assertTrue(torch._C._data_address(view) == t_new_data_addr)
        # 检查 `clone` 的数据地址是否与原始相同
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_binary_op_no_materialize(self, device, dtype):
        # 创建一个张量 `t` 的惰性克隆 `clone`
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        clone = t._lazy_clone()
        # 计算 `t` 和 `clone` 的加法结果 `res`
        res = t + clone
        # 检查 `t` 是否为 COW 张量
        self.assertTrue(torch._C._is_cow_tensor(t))
        # 检查 `clone` 是否为 COW 张量
        self.assertTrue(torch._C._is_cow_tensor(clone))

    # 此测试验证如果在 `at::parallel_for` 循环函数内尝试 COW 实质化，则会引发错误。此测试在 Python 中实现而不是 C++ 中，因为 C++ 测试在 `at::parallel_for` 中不支持多线程。
    @skipXLA
    # 使用装饰器跳过当 TorchDynamo 存在时的测试，因为测试与此不相关
    @skipIfTorchDynamo("Torchdynamo fails and we do not need to test it here anyway")
    # 使用 dtypes 装饰器标记测试的参数类型，包括 torch.half, torch.bool, torch.bfloat16 等所有类型和复杂类型
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 并行测试，验证懒复制时的错误情况
    def test_parallel_cow_materialize_error(self, device, dtype):

        # 定义内部函数 run，用于运行具体的测试用例
        def run(num_threads, num_parallel, skip_first, should_error):
            # 保存原始的线程数
            orig_num_threads = torch.get_num_threads()

            try:
                # 设置当前线程数为 num_threads
                torch.set_num_threads(num_threads)

                # 创建一个张量 a，设备为 device，数据类型为 dtype，并进行懒复制
                a = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)._lazy_clone()

                # 如果 should_error 为 True，则测试是否会抛出 RuntimeError 异常
                if should_error:
                    with self.assertRaisesRegex(RuntimeError, r'Materializing a storage'):
                        torch._test_parallel_materialize(
                            a, num_parallel, skip_first)
                else:
                    # 否则直接调用 _test_parallel_materialize 进行测试
                    torch._test_parallel_materialize(a, num_parallel, skip_first)

                # 不管是否应该出错，都创建一个张量 b，并调用 _test_parallel_materialize 进行测试
                b = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
                torch._test_parallel_materialize(b, num_parallel, skip_first)

            finally:
                # 恢复原始的线程数
                torch.set_num_threads(orig_num_threads)

        # 运行一系列测试用例，每个用例具有不同的参数组合
        run(1, 1, False, True)
        run(1, 1, True, False)
        run(1, 10, False, True)
        run(1, 10, True, True)
        run(10, 1, False, True)
        run(10, 1, True, False)
        run(10, 10, False, True)
        run(10, 10, True, True)
        run(10, 2, False, True)
        run(10, 2, True, True)

    # FIXME: move to test distributions
    # 使用装饰器跳过当 Mps 存在时的测试
    @skipIfMps
    # 对于 CUDA 设备，测试 float、double、half 数据类型的多项式分布
    @dtypesIfCUDA(torch.float, torch.double, torch.half)
    # 在所有设备上，测试 float、double、half 数据类型的多项式分布
    @dtypes(torch.float, torch.double, torch.half)
    # FIXME: move to test distributions
    # 仅在 CUDA 设备上进行测试
    @onlyCUDA
    # 在 CUDA 设备上，测试 float、double、half 数据类型的多项式分布
    @dtypes(torch.float, torch.double, torch.half)
    # 测试确定性的多项式分布
    def test_multinomial_deterministic(self, device, dtype):
        # 创建一个指定设备的随机数生成器
        gen = torch.Generator(device=device)

        # 设置试验次数和种子
        trials = 5
        seed = 0
        # 创建一个概率分布的张量
        prob_dist = torch.rand(10000, 1000, device=device, dtype=dtype)
        # 每次抽样的数量
        n_sample = 1

        # 执行多次试验
        for i in range(trials):
            # 使用给定种子设置生成器的种子值
            gen.manual_seed(seed)
            # 第一次抽样
            samples_1 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

            # 使用相同的种子再次设置生成器的种子值
            gen.manual_seed(seed)
            # 第二次抽样
            samples_2 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

            # 断言两次抽样结果应该相同
            self.assertEqual(samples_1, samples_2)
            # 断言抽样结果的维度应为 2
            self.assertEqual(samples_1.dim(), 2, msg="wrong number of dimensions")
            # 断言每次抽样结果的数量应为 n_sample
            self.assertEqual(samples_1.size(1), n_sample, msg="wrong number of samples")

    # FIXME: move to test distributions
    # 标记为慢速测试
    @slowTest
    # 仅测试 float 数据类型
    @dtypes(torch.float)
    # 定义一个测试方法，用于测试多项式分布的随机数生成状态推进功能
    def test_multinomial_rng_state_advance(self, device, dtype):
        # 设置语料库大小
        corpus_size = 100000
        # 创建一个全为1的张量作为频率信息，数据类型为浮点型，存储于指定设备上
        freqs = torch.ones(corpus_size, dtype=torch.float, device=device)
        # 指定每次抽样的数量
        n_sample = 100
        # 进行第一轮抽样，允许替换
        samples1 = torch.multinomial(freqs, n_sample, replacement=True)
        # 进行第二轮抽样，允许替换
        samples2 = torch.multinomial(freqs, n_sample, replacement=True)
        # 合并两轮抽样的结果
        samples = torch.cat([samples1, samples2])
        # 断言：期望在两次尝试中生成的抽样结果中，重复元素不超过1个
        # 由于至少有一个元素重复的概率令人惊讶地高，约为18%
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 2)
        # 进行第一轮抽样，不允许替换
        samples1 = torch.multinomial(freqs, n_sample, replacement=False)
        # 进行第二轮抽样，不允许替换
        samples2 = torch.multinomial(freqs, n_sample, replacement=False)
        # 合并两轮抽样的结果
        samples = torch.cat([samples1, samples2])
        # 断言：期望在两次尝试中生成的抽样结果中，重复元素不超过1个
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 1)
    # 测试内存格式转换函数的功能，接受设备、输入生成函数、转换函数、内存格式等参数
    def _test_memory_format_transformations(self, device, input_generator_fn, transformation_fn,
                                            memory_format, compare_data=True, default_is_preserve=False):

        # 断言内存格式为 channels_last 或 channels_last_3d
        assert memory_format == torch.channels_last or memory_format == torch.channels_last_3d

        # 生成一个 channels last 的张量 xc
        xc = input_generator_fn(device)
        
        # 如果不使用 Torchinductor 进行测试
        if not TEST_WITH_TORCHINDUCTOR:
            # 如果内存格式为 channels_last，则对 xc 进行空间下采样
            if memory_format == torch.channels_last:
                xc = xc[..., ::2, ::2]
            # 如果内存格式为 channels_last_3d，则对 xc 进行空间下采样
            else:
                xc = xc[..., ::2, ::2, ::2]

        # 对 xc 进行转换，并且保持原有的内存格式
        clone = transformation_fn(xc, memory_format=torch.preserve_format)

        # 断言克隆张量不是连续的
        self.assertFalse(clone.is_contiguous())
        # 断言克隆张量在指定的内存格式下是连续的
        self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        
        # 如果不使用 Torchinductor 进行测试
        if not TEST_WITH_TORCHINDUCTOR:
            # 断言原始张量 xc 不是连续的
            self.assertFalse(xc.is_contiguous())
            # 断言原始张量 xc 在指定的内存格式下不是连续的
            self.assertFalse(xc.is_contiguous(memory_format=memory_format))
        
        # 如果需要比较数据
        if compare_data:
            # 断言原始张量 xc 等于转换后的克隆张量（转换为原始张量的数据类型）
            self.assertEqual(xc, clone.to(xc))

        # 重新生成 channels last 张量 xc
        xc = input_generator_fn(device)
        
        # 对 xc 进行转换，并且保持连续的内存格式
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        
        # 断言克隆张量是连续的
        self.assertTrue(clone.is_contiguous())
        # 断言克隆张量在指定的内存格式下不是连续的
        self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        
        # 如果需要比较数据
        if compare_data:
            # 断言原始张量 xc 等于转换后的克隆张量（转换为原始张量的数据类型）
            self.assertEqual(xc, clone.to(xc))

        # 重新生成 channels last 张量 xc
        xc = input_generator_fn(device)
        
        # 对 xc 进行转换，不指定内存格式
        clone = transformation_fn(xc)

        # 如果默认保持内存格式为 True
        if default_is_preserve:
            # 断言克隆张量不是连续的
            self.assertFalse(clone.is_contiguous())
            # 断言克隆张量在指定的内存格式下是连续的
            self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        else:
            # 断言克隆张量是连续的
            self.assertTrue(clone.is_contiguous())
            # 断言克隆张量在指定的内存格式下不是连续的
            self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        
        # 如果需要比较数据
        if compare_data:
            # 断言原始张量 xc 等于转换后的克隆张量（转换为原始张量的数据类型）
            self.assertEqual(xc, clone.to(xc))

        # TODO 将 _like 构造函数复制到步幅置换，而不仅仅是布局
        # 如果不使用 Torchinductor 进行测试
        if not TEST_WITH_TORCHINDUCTOR:
            # 生成一个形状为 (3, 4, 5, 6, 7, 8, 9) 的随机张量 x
            x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=device)
            # 进行10次循环
            for i in range(10):
                # 生成一个随机的维度排列
                permutation = list(range(len(x.shape)))
                random.shuffle(permutation)
                # 对张量 x 进行维度重排
                x = x.permute(permutation)
                # 断言重排后张量的步幅与以保持格式进行转换后的张量的步幅相等
                self.assertEqual(x.stride(), transformation_fn(x, memory_format=torch.preserve_format).stride())
    # 测试不同内存格式的输入对应的函数
    def test_memory_format_to(self, device):
        # 定义一个生成器函数，用于生成具有特定内存格式和形状的输入数据
        def get_generator(memory_format, shape):
            # 定义内部函数，生成在指定设备上的随机张量，并且保证是连续的，并使用指定的内存格式
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        # 定义一个转换函数，将张量转换为指定的数据类型和其他关键字参数
        def transformation_fn(tensor, **kwargs):
            return tensor.to(dtype=torch.float64, **kwargs)

        # 定义内存格式和形状的元组列表
        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        # 遍历每个内存格式和形状的元组
        for mf, shape in formats_shapes:
            # 调用测试内存格式转换的函数，验证是否保留默认设置
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, default_is_preserve=True)

    # 测试不同内存格式的输入类型
    def test_memory_format_type(self, device):
        # 定义一个生成器函数，用于生成具有特定内存格式和形状的输入数据
        def get_generator(memory_format, shape):
            # 定义内部函数，生成在指定设备上的随机张量，并且保证是连续的，并使用指定的内存格式
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        # 定义一个转换函数，将张量转换为指定的数据类型和其他关键字参数
        def transformation_fn(tensor, **kwargs):
            return tensor.to(torch.float64, **kwargs)

        # 定义内存格式和形状的元组列表
        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        # 遍历每个内存格式和形状的元组
        for mf, shape in formats_shapes:
            # 调用测试内存格式转换的函数，验证是否保留默认设置
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, default_is_preserve=True)

    # 测试克隆张量的内存格式
    def test_memory_format_clone(self, device):
        # 定义一个生成器函数，用于生成具有特定内存格式和形状的输入数据
        def get_generator(memory_format, shape):
            # 定义内部函数，生成在指定设备上的随机张量，并且保证是连续的，并使用指定的内存格式
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        # 定义一个转换函数，克隆张量并应用其他关键字参数
        def transformation_fn(tensor, **kwargs):
            return tensor.clone(**kwargs)

        # 定义内存格式和形状的元组列表
        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        # 遍历每个内存格式和形状的元组
        for mf, shape in formats_shapes:
            # 调用测试内存格式转换的函数，验证是否保留默认设置
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, True, default_is_preserve=True)
    # 测试内存格式工厂函数的保留性质，使用给定的设备
    def test_memory_format_factory_like_functions_preserve(self, device):
        # 定义一个生成器函数，用于生成输入数据
        def get_generator(memory_format, shape):
            # 定义一个输入生成器函数，返回一个在指定设备上形状为shape的随机张量，并保持指定的内存格式
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        # 定义多个转换函数列表
        transformation_fns = [
            lambda t, **kwargs: torch.zeros_like(t, **kwargs),
            lambda t, **kwargs: torch.ones_like(t, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 10, 100, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 100, **kwargs),
            lambda t, **kwargs: torch.randn_like(t, **kwargs),
            lambda t, **kwargs: torch.rand_like(t, **kwargs),
            lambda t, **kwargs: torch.full_like(t, 7, **kwargs),
            lambda t, **kwargs: torch.empty_like(t, **kwargs)]

        # 定义不同内存格式和形状的元组
        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        # 遍历每个内存格式和形状
        for mf, shape, in formats_shapes:
            # 对于每个转换函数，测试内存格式的转换
            for transformation_fn in transformation_fns:
                self._test_memory_format_transformations(
                    device, get_generator(mf, shape), transformation_fn, mf, compare_data=False, default_is_preserve=True)

    # 测试内存格式类型的快捷方式，使用给定的设备
    def test_memory_format_type_shortcuts(self, device):
        # 定义一个生成器函数，用于生成输入数据
        def get_generator(memory_format, shape, dtype):
            # 定义一个输入生成器函数，返回一个在指定设备上形状为shape和数据类型为dtype的随机张量，
            # 并在进行四舍五入后保持指定的内存格式
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=dtype).clamp(0, 1) \
                    .round().contiguous(memory_format=memory_format)
            return input_generator_fn

        # 定义一个获取函数的函数，用于返回特定函数名的转换函数
        def get_fn(fn_name):
            # 定义一个转换函数，通过函数名获取对应的张量方法，并对张量应用该方法
            def transformation_fn(tensor, **kwargs):
                fn = getattr(tensor, fn_name)
                return fn(**kwargs)
            return transformation_fn

        # 定义类型的快捷方式列表
        shortcuts = ['byte', 'char', 'double', 'bool', 'half', 'int', 'long', 'short']
        if device == 'cpu':
            shortcuts += ['bfloat16']

        # 定义不同内存格式和形状的元组
        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        # 遍历每个内存格式和形状
        for mf, shape in formats_shapes:
            # 对于每个快捷方式，测试内存格式的转换
            for fn_name in shortcuts:
                self._test_memory_format_transformations(
                    device, get_generator(mf, shape, torch.float32), get_fn(fn_name), mf, default_is_preserve=True)

        # 单独测试 'float'，避免浮点类型自我转换
        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape, torch.float64), get_fn('float'), mf, default_is_preserve=True)

    # 仅在CUDA环境下执行的测试
    @onlyCUDA
    # 定义一个测试函数，用于测试内存格式和 CPU 及 CUDA 操作
    def test_memory_format_cpu_and_cuda_ops(self, device):
        # 定义一个生成器函数，根据指定的内存格式和形状生成输入数据
        def get_generator(memory_format, shape):
            # 定义一个内部函数，生成具有指定设备上的随机数据，并保证数据是连续的，使用指定的内存格式
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        # 定义一个函数，将张量转移到 CPU 上
        def transformation_cpu_fn(tensor, **kwargs):
            return tensor.cpu(**kwargs)

        # 定义一个函数，将张量转移到 CUDA 上
        def transformation_cuda_fn(tensor, **kwargs):
            return tensor.cuda(**kwargs)

        # 定义多种内存格式和对应的形状
        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),        # 二维通道最后内存格式
            (torch.channels_last_3d, (4, 3, 8, 8, 8))   # 三维通道最后内存格式
        )

        # 遍历每种内存格式和形状的组合，进行测试
        for mf, shape in formats_shapes:
            # 在 CUDA 设备上测试内存格式转换，使用生成器函数生成数据，转换函数是从 CUDA 到 CPU
            self._test_memory_format_transformations(
                'cuda', get_generator(mf, shape), transformation_cpu_fn, mf, default_is_preserve=True)
            # 在 CPU 设备上测试内存格式转换，使用生成器函数生成数据，转换函数是从 CPU 到 CUDA
            self._test_memory_format_transformations(
                'cpu', get_generator(mf, shape), transformation_cuda_fn, mf, default_is_preserve=True)

    # FIXME: move to test_serialization
    # 只在原生设备类型上执行测试
    @onlyNativeDeviceTypes
    # 测试 GradScaler 类中的 pickle 功能和属性
    def test_pickle_gradscaler(self, device):
        # 该测试针对 cuda 有三种情况：
        #  1. cuda 不可用。
        #  2. cuda 可用但设备不是 cuda。
        #  3. cuda 可用且设备是 cuda。
        # 在情况 1 中，a 和 b 在构造时会禁用自身，并且不应尝试 pickle 工作属性。
        # 在情况 2 中，a 和 b 是启用的。工作属性会参与 pickle，但不会有延迟初始化为 cuda 张量的情况，
        # 因为如果设备不是 cuda，我不希望进行 cuda 相关操作。
        # 在情况 3 中，a 和 b 是启用的，并且我们可能尝试将 _scale 延迟初始化为 cuda 张量。
        device = torch.device(device)
        try_lazy_inits = (True, False)
        GradScaler = partial(torch.GradScaler, device=device.type)
        for lazy_init_scale in try_lazy_inits:
            # 初始化 GradScaler 对象 a
            a = GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
            if device.type == "cuda":
                # 如果设备是 cuda，则验证 a 是否未启用，或者如果 cuda.amp 未明确不可用，则验证 a 是否已启用。
                self.assertTrue(not a.is_enabled() if torch.cuda.amp.common.amp_definitely_not_available() else a.is_enabled())
            else:
                # 如果设备不是 cuda，则验证 a 是否已启用。
                self.assertTrue(a.is_enabled())
            if lazy_init_scale:
                # 如果 lazy_init_scale 为 True，则调用 a.scale() 来延迟初始化 a._scale 张量。
                a.scale(torch.tensor([4.0], dtype=torch.float32, device=device))
                self.assertTrue(a._scale.device.type == device.type)
            # 以下三行代码应在 cuda 是否可用的情况下都能工作。
            # 将对象 a 序列化为字节流
            serialized = pickle.dumps(a)
            # 从序列化的字节流中反序列化出对象 b
            b = pickle.loads(serialized)
            # 验证反序列化后的对象 b 的属性与原对象 a 的属性是否一致。
            self.assertEqual(b.is_enabled(), a.is_enabled())
            if a.is_enabled():
                # 如果 a 已启用，则验证 b 的相关属性值与 a 的是否一致。
                self.assertEqual(b.get_scale(), 3.)
                self.assertEqual(b.get_growth_factor(), 4.)
                self.assertEqual(b.get_backoff_factor(), .5)
                self.assertEqual(b.get_growth_interval(), 2)
                self.assertEqual(b._init_growth_tracker, 0)
                # 测试 defaultdict 的默认工厂函数
                # 提供一个虚拟的键以测试 defaultdict 的 default_factory
                self.assertEqual(b._per_optimizer_states["fdsa"],
                                 torch.amp.grad_scaler._refresh_per_optimizer_state())
                if lazy_init_scale:
                    # 如果 lazy_init_scale 为 True，则验证 b 的 scale() 方法的输出值是否正确。
                    self.assertEqual(b.scale(torch.tensor([4.0], dtype=torch.float32, device=device)), 12.0)

    # FIXME: move to test distributions
    # 测试当 probs 张量为空时的 multinomial 函数行为
    def _test_multinomial_empty(self, device, replacement, num_samples):
        probs = torch.ones(0, 3, device=device)
        expected = torch.empty(0, num_samples, dtype=torch.int64)
        out = torch.multinomial(probs, num_samples=num_samples, replacement=replacement)
        self.assertEqual(out, expected)

    # FIXME: move to test distributions
    # 测试当 probs 张量为空时的 multinomial 函数行为（带替换）
    def test_multinomial_empty_w_replacement(self, device):
        # 调用 _test_multinomial_empty() 测试函数，测试替换为 True 和样本数为 1 的情况
        self._test_multinomial_empty(device, True, 1)
        # 调用 _test_multinomial_empty() 测试函数，测试替换为 True 和样本数为 2 的情况
        self._test_multinomial_empty(device, True, 2)

    # FIXME: move to test distributions
    # 定义一个测试方法，测试不带替换的多项式抽样空情况
    def test_multinomial_empty_wo_replacement(self, device):
        # 调用内部方法，测试在指定设备上，不带替换的多项式抽样空情况，抽样数量为1
        self._test_multinomial_empty(device, False, 1)
        # 再次调用内部方法，测试在指定设备上，不带替换的多项式抽样空情况，抽样数量为2
        self._test_multinomial_empty(device, False, 2)

    # 设置装饰器，只在原生设备类型上执行测试
    @onlyNativeDeviceTypes
    # 设置装饰器，指定数据类型为 float 和 double 进行测试
    @dtypes(torch.float, torch.double)
    # 设置装饰器，只在原生设备类型上执行测试
    @onlyNativeDeviceTypes
    # 设置装饰器，指定数据类型为 float 进行测试
    @dtypes(torch.float)
    # 定义测试方法，测试梯度缩放更新比例
    def test_grad_scaling_update_scale(self, device, dtype):
        # 设置生长率、回退率、生长间隔的初始值
        growth = 2.0
        backoff = 0.25
        growth_interval = 2
        # 创建张量 scale，填充值为 4.0，指定数据类型和设备
        scale = torch.full((1,), 4.0, dtype=dtype, device=device)
        # 创建张量 growth_tracker，填充值为 0.0，数据类型为 int32，指定设备
        growth_tracker = torch.full((1,), 0.0, dtype=torch.int32, device=device)
        # 创建张量 found_inf，填充值为 0.0，数据类型为 float，指定设备
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device=device)

        # 模拟两个连续的未跳过迭代
        # 调用内部函数 _amp_update_scale_，更新缩放比例，检查生长追踪器和缩放比例的期望值
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 1)
        self.assertEqual(scale, 4.0)
        # 再次调用内部函数 _amp_update_scale_，更新缩放比例，检查生长追踪器和缩放比例的期望值
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 8.0)

        # 模拟跳过的迭代
        # 将 found_inf 张量填充为 1.0，表示发现了无穷大值
        found_inf.fill_(1.0)
        # 再次调用内部函数 _amp_update_scale_，更新缩放比例，检查生长追踪器和缩放比例的期望值
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 2.0)

    # 设置装饰器，如果 Torch Dynamo 失败，则跳过该测试
    @skipIfTorchDynamo("Failed running call_function for sparse_coo_tensor. See https://github.com/pytorch/pytorch/issues/118856")
    # 设置装饰器，只在原生设备类型上执行测试
    @onlyNativeDeviceTypes
    # 设置装饰器，指定数据类型为 float 进行测试
    @dtypes(torch.float)
    # 定义测试函数，测试梯度缩放和反缩放操作的稀疏张量情况
    def test_grad_scaling_unscale_sparse(self, device, dtype):
        # 将设备参数转换为torch设备对象
        device = torch.device(device)
        # 创建梯度缩放器对象，指定设备类型
        scaler = torch.GradScaler(device=device.type)

        # 创建一个具有单一元素的张量，用于反缩放操作
        inv_scale = torch.full((1,), 0.25, dtype=dtype, device=device)
        # 创建一个空的张量，用于存储是否找到无穷大的信息
        found_inf = torch.empty((1,), dtype=dtype, device=device)
        # 获取当前设备
        cur = found_inf.device

        # 创建一个稀疏张量，指定索引和值
        i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]], device=device, dtype=torch.int64)
        v = torch.tensor([16., 32., 64.], device=device, dtype=torch.float)
        s = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=dtype)

        # 克隆稀疏张量p作为备份
        p = s.clone()
        # 断言p是稀疏张量
        assert p.is_sparse
        # 创建SGD优化器，用于更新p
        opt = torch.optim.SGD([p], lr=1.)

        # 将p的梯度设置为稀疏张量s的克隆
        p.grad = s.clone()
        # 将found_inf置零
        found_inf.zero_()
        # 执行梯度反缩放操作
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        # 断言是否找到无穷大的梯度为0
        self.assertEqual(found_inf, 0.0)
        # 断言p的稠密梯度与(s / 4)的稠密版本相等
        self.assertEqual(p.grad.to_dense(), (s / 4).to_dense())

        # 修改稀疏张量的值，包含无穷大
        v = torch.FloatTensor([16., 32., float('inf')])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=dtype)
        # 将found_inf置零
        found_inf.zero_()
        # 执行梯度反缩放操作
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        # 断言是否找到无穷大的梯度为1
        self.assertEqual(found_inf, 1.0)

        # 修改稀疏张量的值，包含NaN
        v = torch.FloatTensor([16., 32., float('nan')])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=dtype)
        # 将found_inf置零
        found_inf.zero_()
        # 执行梯度反缩放操作
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, False)[cur]
        # 断言是否找到无穷大的梯度为1
        self.assertEqual(found_inf, 1.0)

        # 克隆稀疏张量p作为备份，并将其类型转换为half精度
        p = s.clone().half()
        # 断言p是稀疏张量
        assert p.is_sparse
        # 创建SGD优化器，用于更新p
        opt = torch.optim.SGD([p], lr=1.)

        # 将p的梯度设置为s的half精度版本的克隆
        p.grad = s.clone().half()
        # 将found_inf置零
        found_inf.zero_()
        # 执行梯度反缩放操作，标志位设置为True
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        # 断言是否找到无穷大的梯度为0
        self.assertEqual(found_inf, 0.0)
        # 断言p的稠密梯度与(s.half() / 4)的稠密版本相等
        self.assertEqual(p.grad.to_dense(), (s.half() / 4).to_dense())

        # 创建包含重复索引的fp16稀疏张量。非压缩表示在fp16中不会溢出，但压缩表示会溢出，因为64000 + 64000 > fp16最大值。
        # 在这里，_amp_non_finite_check_and_unscale_应该报告溢出。
        i = torch.LongTensor([[0, 1, 0],
                              [2, 0, 2]])
        v = torch.FloatTensor([64000., 32., 64000.])
        p.grad = torch.sparse_coo_tensor(i, v, torch.Size([2, 3]), device=device, dtype=torch.float16)
        # 将found_inf置零
        found_inf.zero_()
        # 执行梯度反缩放操作，标志位设置为True
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf, True)[cur]
        # 断言是否找到无穷大的梯度为1
        self.assertEqual(found_inf, 1.0)
    # 定义测试方法，用于测试梯度缩放器状态字典加载功能，参数为设备类型
    def test_grad_scaling_state_dict(self, device):
        # 将设备类型转换为 torch.device 对象
        device = torch.device(device)
        # 使用偏函数 partial 创建 GradScaler 类的实例化函数，指定设备类型
        GradScaler = partial(torch.GradScaler, device=device.type)
        
        # 对每个 lazy_init_scale 值为 True 和 False 进行循环测试
        for lazy_init_scale in True, False:
            # 创建 GradScaler 实例 s0 和 s1，指定不同的初始化参数
            s0 = GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
            s1 = GradScaler(init_scale=6., growth_factor=7., backoff_factor=.8, growth_interval=1)

            # 设置 s1 对象的 _init_growth_tracker 属性为随机值 7
            s1._init_growth_tracker = 7

            # 如果 lazy_init_scale 为 True，则执行以下代码块
            if lazy_init_scale:
                # 调用 scale 方法以确保懒初始化 scale 张量
                s1.scale(torch.full((1,), 4.0, dtype=torch.float32, device=device))
                # 根据设备类型验证 _scale 属性类型是否为对应的 FloatTensor
                if "cuda" == device.type:
                    self.assertTrue(isinstance(s1._scale, torch.cuda.FloatTensor))
                else:
                    self.assertTrue(isinstance(s1._scale, torch.FloatTensor))

            # 加载 s0 的状态字典到 s1
            s1.load_state_dict(s0.state_dict())

            # 断言 s1 的各属性是否与 s0 一致
            self.assertEqual(s1.get_scale(), 3.)
            self.assertEqual(s1.get_growth_factor(), 4.)
            self.assertEqual(s1.get_backoff_factor(), .5)
            self.assertEqual(s1.get_growth_interval(), 2)
            # 断言 s1 的 _init_growth_tracker 属性是否被加载状态字典重置为 0
            self.assertEqual(s1._init_growth_tracker, 0)

    # _run_scaling_case 通用化了某些单优化器测试逻辑，避免在下面进行过多的复制粘贴。
    # 在指定设备上运行缩放测试用例，测试两种情况：启用和禁用缩放
    def _run_scaling_case(self, device, run, unskipped, skipped, atol=1e-7, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
        # 确保缩放可以在不改变用户控制流的情况下禁用
        for enabled in True, False:
            # 创建用于缩放测试的模型、优化器、数据、损失函数和跳过的迭代次数
            (
                mod_control, mod_scaling, opt_control, opt_scaling, data, loss_fn, skip_iter,
            ) = _create_scaling_case(device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs)

            # 为了功能性，使用适度的初始规模和一个不现实地大的增长因子，
            # 这样可以放大增长因子处理中的任何潜在错误。
            GradScaler = partial(torch.GradScaler, device=device)
            scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)

            # 运行指定的函数，使用不同的模型和优化器，测试缩放是否生效
            _ = run(device, data, mod_control, opt_control, scaler, loss_fn, skip_iter, False)
            ret = run(device, data, mod_scaling, opt_scaling, scaler, loss_fn, skip_iter, True)

            # 允许 run() 函数选择性地返回不同的 scaler 实例。
            scaler = ret if ret else scaler

            # 如果启用了缩放，则缩放因子应该被增长因子的 unskipped 次方乘以跳过次数 skipped 的 backoff 因子次方。
            if enabled:
                net_growth = scaler.get_growth_factor()**unskipped if unskipped > 0 else 1.0
                net_backoff = scaler.get_backoff_factor()**skipped if skipped > 0 else 1.0
                self.assertTrue(scaler.get_scale() == (128. * net_growth * net_backoff))
            else:
                self.assertTrue(scaler.get_scale() == 1.0)

            # 检查控制模型和缩放模型的梯度是否相等
            for c, s in zip(mod_control.parameters(), mod_scaling.parameters()):
                self.assertEqual(c.grad, s.grad, atol=atol, rtol=1e-05)

                # 检查优化器状态是否相等
                c_state, s_state = opt_control.state[c], opt_scaling.state[s]
                for k in c_state:
                    self.assertEqual(c_state[k], s_state[k], atol=atol, rtol=1e-05, msg=k)

                # 检查模型参数是否相等
                self.assertEqual(c, s, atol=atol, rtol=1e-05)

    # 仅在原生设备类型上执行该测试
    @onlyNativeDeviceTypes
    # 参数化装饰器，用于指定测试函数的参数化组合
    @parametrize("foreach, fused", [(None, None), (True, None), (None, True)])
    # 优化器选择装饰器，限定优化器类型为 AdamW、Adam 或 SGD，并指定数据类型为 torch.float32
    @optims(
        [optim for optim in optim_db if optim.optim_cls in [torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD]],
        dtypes=[torch.float32]
    )
    # 定义一个测试方法，用于测试梯度缩放与自动类型转换功能
    def test_grad_scaling_autocast(self, device, dtype, optim_info, foreach, fused):
        # 是否尝试使用 pickle 进行对象序列化
        try_pickle = False

        # 定义运行函数，执行模型训练过程
        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # 遍历数据集中的每个批次
            for i, (input, target) in enumerate(data):
                # 梯度清零
                optimizer.zero_grad()
                # 根据设置开启自动混合精度计算
                with torch.autocast(device_type=device, dtype=torch.half, enabled=try_scaling_api):
                    # 模型前向传播
                    output = model(input)
                    # 计算损失
                    loss = loss_fn(output, target)
                # 如果尝试使用梯度缩放 API
                if try_scaling_api:
                    # 使用梯度缩放器对损失进行反向传播
                    scaler.scale(loss).backward()
                    # 在特定迭代步骤，检查是否需要无限制地填充权重梯度
                    if i == skip_iter and scaler.is_enabled():
                        with torch.no_grad():
                            model[1].weight.grad.fill_(float('inf'))
                    # 执行优化器步骤
                    scaler.step(optimizer)
                    # 更新梯度缩放器的状态
                    scaler.update()
                    # 如果尝试使用 pickle 序列化梯度缩放器
                    if try_pickle:
                        scaler = pickle.loads(pickle.dumps(scaler))
                else:
                    # 普通的反向传播
                    loss.backward()
                    # 如果未启用梯度缩放或不在跳过迭代步骤，则执行优化器步骤
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        optimizer.step()
            # 返回最终的梯度缩放器状态
            return scaler

        # 获取优化器构造器
        optimizer_ctor = optim_info.optim_cls

        # 对比未进行梯度缩放和自动类型转换的情况与进行了缩放和转换的情况
        # 注意：目前测试方式下，`torch.optim.Adam` 尽管使用了 `foreach` 和 `fused`，仍存在问题。
        # 为此，对这个测试给予一定的灵活性可能会有帮助。
        context = contextlib.nullcontext
        if optimizer_ctor in (torch.optim.Adam, torch.optim.AdamW):
            from functools import partial
            context = partial(self.assertRaises, AssertionError)
        with context():
            # 设置 atol=1e-3，因为我们比较的是纯 fp32 算术和 fp16 与 fp32 混合的情况
            self._run_scaling_case(
                device, run, unskipped=3, skipped=1, atol=1e-3,
                optimizer_ctor=optimizer_ctor, optimizer_kwargs={"foreach": foreach, "fused": fused},
            )
            # 在 run() 函数内部的 try_pickle 将被捕获
            try_pickle = True
            self._run_scaling_case(
                device, run, unskipped=3, skipped=1, atol=1e-3,
                optimizer_ctor=optimizer_ctor, optimizer_kwargs={"foreach": foreach, "fused": fused},
            )

    # 确保在缩放梯度有限但在 `optimizer.step` 之后被无效化之前，参数变得无意义
    @onlyNativeDeviceTypes
    @optims(
        # 仅选择优化器类为 `torch.optim.AdamW`, `torch.optim.Adam`, `torch.optim.SGD` 的优化器
        [optim for optim in optim_db if optim.optim_cls in [torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD]],
        # 选择数据类型为 `torch.float32`
        dtypes=[torch.float32]
    )
    # 测试函数，用于验证在 unscale 和 step 之间使参数失效
    def test_params_invalidated_with_grads_invalidated_between_unscale_and_step(self, device, dtype, optim_info):
        # 获取优化器类构造函数
        optimizer_ctor = optim_info.optim_cls
        # 获取包含全局参数的优化器输入列表，但跳过 'differentiable' 参数
        all_optim_inputs = _get_optim_inputs_including_global_cliquey_kwargs(
            device, dtype, optim_info, skip=("differentiable",))

        # 遍历所有优化器输入
        for optim_input in all_optim_inputs:
            # 创建一个模型、优化器、数据、损失函数等，在一个缩放场景下
            model, _, optimizer, _, data, loss_fn, _ = _create_scaling_case(
                device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optim_input.kwargs,
            )
            # 创建一个梯度缩放器对象
            scaler = torch.GradScaler(device=device, init_scale=128.0)

            # 遍历数据集中的输入和目标对
            for input, target in data:
                # 清除优化器梯度
                optimizer.zero_grad()
                # 自动混合精度上下文
                with torch.autocast(device_type=device, dtype=torch.half):
                    output = model(input)
                    loss = loss_fn(output, target)
                # 缩放损失并进行反向传播
                scaler.scale(loss).backward()
                # 反缩放优化器
                scaler.unscale_(optimizer)

                # 故意破坏梯度
                for j, param in enumerate(model.parameters()):
                    param.grad.copy_(torch.inf if j % 2 else torch.nan)

                # 执行优化步骤
                scaler.step(optimizer)
                # 更新缩放器状态
                scaler.update()

            # 断言：模型参数中至少有一个包含 NaN 或者 Inf
            self.assertTrue(all((p.isnan().any() or p.isinf().any()) for p in model.parameters()))

    # 仅在原生设备类型上执行的测试函数
    @onlyNativeDeviceTypes
    def test_grad_scale_will_not_overflow(self, device):
        # 将设备转换为 torch 设备对象
        device = torch.device(device)
        # 创建一个线性模型并将其移动到指定设备上
        model = torch.nn.Linear(5, 1).to(device)
        # 创建一个 Adam 优化器
        optimizer = torch.optim.Adam(model.parameters())
        # 创建一个梯度缩放器对象，设置生长间隔、生长因子和初始缩放
        scaler = torch.GradScaler(device=device.type, growth_interval=1, growth_factor=2**4, init_scale=1e38)
        # 清除优化器梯度
        optimizer.zero_grad()
        # 创建输入数据并移动到指定设备上
        x = torch.randn(1, 5).to(device)
        y = 1e-30 * torch.randn(1, 1).to(device)
        # 计算损失
        l = ((model(x) - y) ** 2).mean()
        # 缩放损失并进行反向传播
        scaler.scale(l).backward()
        # 执行一步优化
        scaler.step(optimizer)
        # 更新缩放器状态
        scaler.update()
        # 断言：缩放器的缩放值不是无穷大或 NaN
        assert scaler._scale != float("inf") and scaler._scale != float("nan")

    # 仅在原生设备类型上执行
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试梯度缩放和裁剪操作，需要指定设备类型
    def test_grad_scaling_clipping(self, device):
        # 将设备转换为torch设备对象
        device = torch.device(device)

        # 定义内部运行函数，负责实际执行模型训练的细节
        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # 设定梯度的最大范数，限制梯度的大小，以防止梯度爆炸
            max_norm = 0.2  # 一个合理的值，根据梯度打印输出确定其效果

            # 遍历数据集中的每个(batch)样本
            for i, (input, target) in enumerate(data):
                # 清除优化器的梯度信息
                optimizer.zero_grad()

                # 将输入数据通过模型前向传播得到输出
                output = model(input)

                # 计算模型输出与目标之间的损失
                loss = loss_fn(output, target)

                # 如果尝试使用梯度缩放 API
                if try_scaling_api:
                    # 使用梯度缩放器对损失进行缩放并反向传播
                    scaler.scale(loss).backward()

                    # 对模型参数的梯度进行裁剪，以防止梯度超过指定的最大范数
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm * scaler.get_scale())

                    # 如果当前迭代次数等于跳过迭代的次数，并且梯度缩放器已启用
                    if i == skip_iter and scaler.is_enabled():
                        # 将模型中第二个层的权重梯度数据填充为无穷大
                        model[1].weight.grad.data.fill_(float('inf'))

                    # 使用梯度缩放器执行一步优化器的操作
                    scaler.step(optimizer)

                    # 更新梯度缩放器内部状态
                    scaler.update()
                else:
                    # 否则，直接对损失进行反向传播
                    loss.backward()

                    # 对模型参数的梯度进行裁剪，以防止梯度超过指定的最大范数
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    # 如果梯度缩放器未启用或当前迭代次数不等于跳过迭代的次数
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        # 执行优化器的一步操作
                        optimizer.step()

        # 调用内部运行函数，传入设备类型、运行函数、以及未跳过和跳过的迭代次数
        self._run_scaling_case(device.type, run, unskipped=3, skipped=1, atol=1e-5)

    # 仅在本地设备类型上运行的测试函数修饰器
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试梯度缩放和裁剪操作，并支持单独的解缩放过程
    def test_grad_scaling_clipping_separate_unscale(self, device):
        # 将设备转换为torch设备对象
        device = torch.device(device)

        # 定义内部运行函数，负责实际执行模型训练的细节
        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # 设定梯度的最大范数，限制梯度的大小，以防止梯度爆炸
            max_norm = 0.2  # 一个合理的值，根据梯度打印输出确定其效果

            # 遍历数据集中的每个(batch)样本
            for i, (input, target) in enumerate(data):
                # 清除优化器的梯度信息
                optimizer.zero_grad()

                # 将输入数据通过模型前向传播得到输出
                output = model(input)

                # 计算模型输出与目标之间的损失
                loss = loss_fn(output, target)

                # 如果尝试使用梯度缩放 API
                if try_scaling_api:
                    # 使用梯度缩放器对损失进行缩放并反向传播
                    scaler.scale(loss).backward()

                    # 如果当前迭代次数等于跳过迭代的次数，并且梯度缩放器已启用
                    if i == skip_iter and scaler.is_enabled():
                        # 将模型中第二个层的权重梯度数据填充为无穷大
                        model[1].weight.grad.data.fill_(float('inf'))

                    # 对优化器进行解缩放操作
                    scaler.unscale_(optimizer)

                    # 对模型参数的梯度进行裁剪，以防止梯度超过指定的最大范数，不抛出错误
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)

                    # 使用梯度缩放器执行一步优化器的操作
                    scaler.step(optimizer)

                    # 更新梯度缩放器内部状态
                    scaler.update()
                else:
                    # 否则，直接对损失进行反向传播
                    loss.backward()

                    # 对模型参数的梯度进行裁剪，以防止梯度超过指定的最大范数
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    # 如果梯度缩放器未启用或当前迭代次数不等于跳过迭代的次数
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        # 执行优化器的一步操作
                        optimizer.step()

        # 调用内部运行函数，传入设备类型、运行函数、以及未跳过和跳过的迭代次数
        self._run_scaling_case(device.type, run, unskipped=3, skipped=1)

    # 仅在本地设备类型上运行的测试函数修饰器，仅在非Windows平台上运行测试
    @onlyNativeDeviceTypes
    @unittest.skipIf(IS_WINDOWS, 'FIXME: fix this test for Windows')
    # 测试梯度缩放惩罚功能，接受设备参数
    def test_grad_scaling_penalty(self, device):
        # 将设备参数转换为 torch 设备对象
        device = torch.device(device)

        # 定义运行函数，接受设备、数据、模型、优化器、缩放器、损失函数、跳过迭代次数、尝试缩放 API 参数
        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # 遍历数据集
            for i, (input, target) in enumerate(data):
                # 每次迭代前将优化器梯度清零
                optimizer.zero_grad()
                # 将输入数据输入模型，得到输出
                output = model(input)
                # 计算损失值
                loss = loss_fn(output, target)

                # 如果尝试缩放 API
                if try_scaling_api:
                    # 计算缩放后的损失对模型参数的梯度
                    grad_params = torch.autograd.grad(scaler.scale(loss),
                                                      model.parameters(), create_graph=True)
                    # 计算反向传播梯度缩放因子的倒数
                    inv_scale = 1. / scaler.get_scale()
                    # 将梯度乘以缩放因子的倒数
                    grad_params = [p * inv_scale for p in grad_params]
                else:
                    # 否则直接计算损失对模型参数的梯度
                    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                # 计算梯度范数
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                # 将损失值加上梯度范数作为最终损失
                loss = loss + grad_norm

                # 如果尝试缩放 API
                if try_scaling_api:
                    # 执行反向传播，并进行缩放
                    scaler.scale(loss).backward()
                    # 若当前迭代次数等于跳过迭代次数，并且缩放器已启用
                    if i == skip_iter and scaler.is_enabled():
                        # 将模型的第二层权重梯度数据填充为正无穷
                        model[1].weight.grad.data.fill_(float('inf'))
                    # 执行缩放器的优化步骤
                    scaler.step(optimizer)
                    # 更新缩放器状态
                    scaler.update()
                else:
                    # 否则执行标准的反向传播
                    loss.backward()
                    # 如果缩放器未启用或者当前迭代不等于跳过迭代次数
                    if (not scaler.is_enabled()) or (i != skip_iter):
                        # 执行优化器的优化步骤
                        optimizer.step()

        # 调用内部方法 _run_scaling_case，传入设备类型、运行函数、未跳过迭代次数、跳过迭代次数
        self._run_scaling_case(device.type, run, unskipped=3, skipped=1)

    # 标记为仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 测试梯度缩放累积功能，接受设备参数
    def test_grad_scaling_accumulation(self, device):
        # 将设备参数转换为 torch 设备对象
        device = torch.device(device)

        # 定义运行函数，接受设备、数据、模型、优化器、缩放器、损失函数、跳过迭代次数、尝试缩放 API 参数
        def run(device, data, model, optimizer, scaler, loss_fn, skip_iter, try_scaling_api):
            # 累积迭代的次数设定为2
            iters_to_accumulate = 2
            # 遍历数据集
            for i, (input, target) in enumerate(data):
                # 将输入数据输入模型，得到输出
                output = model(input)
                # 计算损失值
                loss = loss_fn(output, target)
                # 将损失值除以累积迭代次数
                loss = loss / iters_to_accumulate
                # 如果尝试缩放 API
                if try_scaling_api:
                    # 执行缩放后的反向传播
                    scaler.scale(loss).backward()
                else:
                    # 否则执行标准的反向传播
                    loss.backward()
                # 如果达到累积迭代次数
                if (i + 1) % iters_to_accumulate == 0:
                    # 如果尝试缩放 API
                    if try_scaling_api:
                        # 执行缩放器的优化步骤
                        scaler.step(optimizer)
                        # 更新缩放器状态
                        scaler.update()
                        # 将优化器的梯度清零
                        optimizer.zero_grad()
                    else:
                        # 否则执行优化器的优化步骤
                        optimizer.step()
                        # 将优化器的梯度清零
                        optimizer.zero_grad()

        # 调用内部方法 _run_scaling_case，传入设备类型、运行函数、未跳过迭代次数、跳过迭代次数
        self._run_scaling_case(device.type, run, unskipped=2, skipped=0)

    # 标记为仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试在多个模型和多个优化器中进行梯度缩放
    def test_grad_scaling_multiple(self, device):
        # 将设备转换为 torch 设备对象
        device = torch.device(device)
        
        # 测试梯度缩放，使用两个模型和两个优化器，它们都接收来自两个损失函数的梯度
        # 这里的一些逻辑无法重用为单优化器情况创建的通用辅助函数
        for enabled in True, False:
            # 创建缩放案例的控制模型、缩放模型、控制优化器、缩放优化器以及数据、损失函数和跳过迭代数
            mod_control0, mod_scaling0, opt_control0, opt_scaling0, data, loss_fn, skip_iter = \
                _create_scaling_case(device.type)
            
            # 创建另一组缩放模型和优化器
            mod_control1, mod_scaling1, opt_control1, opt_scaling1 = \
                _create_scaling_models_optimizers(device.type)
            
            # 使用 partial 创建 GradScaler 的实例，设置设备类型
            GradScaler = partial(torch.GradScaler, device=device.type)
            # 创建 GradScaler 实例，设置初始缩放、增长因子、是否启用以及增长间隔
            scaler = GradScaler(init_scale=128., growth_factor=2.0, enabled=enabled, growth_interval=1)
            
            # 定义运行函数，用于在两组模型和优化器上执行前向传播、损失计算和反向传播
            def run(model0, model1, optimizer0, optimizer1, try_scaling_api):
                for i, (input, target) in enumerate(data):
                    # 梯度清零
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    
                    # 分别计算两个模型的输出
                    output0 = model0(input)
                    output1 = model1(input)
                    
                    # 计算两个损失函数
                    loss0 = loss_fn(0.3 * output0 + 0.7 * output1, target)
                    loss1 = loss_fn(0.6 * output0 - 0.4 * output1, target)

                    if try_scaling_api:
                        # 如果尝试使用缩放 API
                        # 使用缩放器对损失进行缩放并执行反向传播
                        scaler.scale(loss0).backward(retain_graph=True)
                        scaler.scale(loss1).backward()

                        # 在特定迭代次数跳过时，设置模型中某个权重梯度为无穷大
                        if i == skip_iter and scaler.is_enabled():
                            model1[1].weight.grad.data.fill_(float('inf'))

                        # 另外进行一个压力测试，分别对一个优化器取消缩放
                        scaler.unscale_(optimizer0)

                        # 执行优化器步骤
                        scaler.step(optimizer0)
                        scaler.step(optimizer1)

                        # 更新缩放器状态
                        scaler.update()
                    else:
                        # 如果不使用缩放 API，直接执行正常的反向传播和优化步骤
                        loss0.backward(retain_graph=True)
                        loss1.backward()
                        optimizer0.step()
                        if (not scaler.is_enabled()) or (i != skip_iter):
                            optimizer1.step()

            # 分别在控制模型和缩放模型上运行函数
            run(mod_control0, mod_control1, opt_control0, opt_control1, False)
            run(mod_scaling0, mod_scaling1, opt_scaling0, opt_scaling1, True)

            # 断言：如果启用了缩放器，则验证损失缩放是否已经乘以增长因子 3 次和回退因子 1 次
            self.assertTrue(scaler.get_scale() == (128. * scaler.get_growth_factor()**3 *
                                                   scaler.get_backoff_factor()**1) if enabled else 1.0)

            # 断言：验证模型参数是否近似相等
            for c, s in zip(chain(mod_control0.parameters(), mod_control1.parameters()),
                            chain(mod_scaling0.parameters(), mod_scaling1.parameters())):
                self.assertEqual(c, s, rtol=1e-5, atol=1e-7)

    # 仅针对本地设备类型的装饰器
    @onlyNativeDeviceTypes
    # 定义一个测试方法，用于验证 GradScaler 能够正确地传递给自身
    def test_grad_scaler_pass_itself(self, device):
        # 将输入的设备字符串转换为 torch 设备对象
        device = torch.device(device)
        # 使用偏函数定义 GradScaler，设备类型与输入的设备一致
        GradScaler = partial(torch.amp.GradScaler, device=device.type)

        # 定义一个占位符优化器类，继承自 torch.optim.Optimizer
        class _PlaceHolderOptimizer(torch.optim.Optimizer):
            # 在类定义内部保存测试实例的引用
            tester = self

            # 初始化方法
            def __init__(self, params, defaults=None):
                # 如果未提供默认参数字典，则设为空字典
                if defaults is None:
                    defaults = {}
                # 调用父类初始化方法
                super().__init__(params, defaults)
                # 标记该优化器支持自动混合精度缩放
                self._step_supports_amp_scaling = True

        # 定义一个继承自 _PlaceHolderOptimizer 的优化器类 Optimizer1
        class Optimizer1(_PlaceHolderOptimizer):
            # 优化步骤方法
            def step(self, closure=None, *, grad_scaler=None):
                # 断言 grad_scaler 是 torch.amp.GradScaler 的实例
                self.tester.assertTrue(isinstance(grad_scaler, torch.amp.GradScaler))
                # 断言当前对象不具有 grad_scale 属性
                self.tester.assertFalse(hasattr(self, "grad_scale"))
                # 断言当前对象不具有 found_inf 属性
                self.tester.assertFalse(hasattr(self, "found_inf"))

        # 定义另一个继承自 _PlaceHolderOptimizer 的优化器类 Optimizer2
        class Optimizer2(_PlaceHolderOptimizer):
            # 优化步骤方法
            def step(self, closure=None):
                # 断言当前对象的 grad_scale 属性是 torch.Tensor 的实例
                self.tester.assertTrue(isinstance(self.grad_scale, torch.Tensor))
                # 断言当前对象的 found_inf 属性是 torch.Tensor 的实例
                self.tester.assertTrue(isinstance(self.found_inf, torch.Tensor))

        # 创建一个 tensor x，并将其移到指定设备上
        x = torch.randn(4, 4).to(device)
        # 创建一个线性层 m，并将其移到指定设备上
        m = torch.nn.Linear(4, 1).to(device)
        # 创建 Optimizer1 实例 o1，并传入 m 的参数
        o1 = Optimizer1(m.parameters())
        # 创建 Optimizer2 实例 o2，并传入 m 的参数
        o2 = Optimizer2(m.parameters())
        # 创建 GradScaler 实例 scaler，初始缩放因子为 2.0
        scaler = GradScaler(init_scale=2.0)

        # 使用自动混合精度上下文，指定设备类型和数据类型为 torch.half
        with torch.autocast(device_type=device.type, dtype=torch.half):
            # 对输入 x 进行线性变换并计算输出 y
            y = m(x)
            # 计算输出 y 的均值作为损失值
            loss = y.mean()
        # 使用 scaler 对损失值进行缩放并执行反向传播
        scaler.scale(loss).backward()
        # 断言在未来会发出警告
        with self.assertWarns(FutureWarning):
            # 使用 scaler 执行优化器 o1 的一步优化
            scaler.step(o1)
        # 使用 scaler 执行优化器 o2 的一步优化
        scaler.step(o2)
        # 更新 scaler 状态
        scaler.update()

    # 仅在原生设备类型下运行的测试方法装饰器
    @onlyNativeDeviceTypes
    def test_grad_scaler_deprecated_warning(self, device):
        # 将输入的设备字符串转换为 torch 设备对象
        device = torch.device(device)
        # 根据设备类型选择不同的 GradScaler 类型
        GradScaler = torch.cuda.amp.GradScaler if "cuda" == device.type else torch.cpu.amp.GradScaler

        # 断言在未来会发出警告，并匹配特定的警告信息
        with self.assertWarnsRegex(
            FutureWarning,
            rf"`torch.{device.type}.amp.GradScaler\(args...\)` is deprecated.",
        ):
            # 创建 GradScaler 实例，初始缩放因子为 2.0，并将其赋值给 _
            _ = GradScaler(init_scale=2.0)

    # 根据 CUDA 设备选择数据类型的测试方法装饰器
    @dtypesIfCUDA(torch.float, torch.double, torch.half)
    # 根据 CPU 设备选择数据类型的测试方法装饰器
    @dtypesIfCPU(torch.float, torch.double, torch.bfloat16, torch.half)
    # 通用数据类型的测试方法装饰器
    @dtypes(torch.float, torch.double)
    # 定义一个测试方法，用于在 CPU 上测试多项式分布的功能
    def test_multinomial_cpu(self, device, dtype):
        # 定义内部函数，生成概率分布
        def make_prob_dist(shape, is_contiguous):
            # 如果 is_contiguous 为 True
            if is_contiguous:
                # 如果数据类型是半精度或者 bfloat16，生成一个形状为 shape 的全零张量，并在该张量上生成均匀分布的随机数，然后转换为指定的数据类型
                if dtype == torch.half or dtype == torch.bfloat16:
                    return torch.zeros(shape, device=device).uniform_().to(dtype=dtype)
                # 如果数据类型不是半精度或者 bfloat16，生成一个形状为 shape 的全零张量，并在该张量上生成指定数据类型的均匀分布的随机数
                return torch.zeros(shape, device=device, dtype=dtype).uniform_()
            # 如果 is_contiguous 不为 True 且 shape 的长度为 1
            elif len(shape) == 1:
                # 如果数据类型是半精度或者 bfloat16，生成一个形状为 (shape + [5]) 的全零张量，并在该张量上生成均匀分布的随机数，然后取第 3 列的数据
                if dtype == torch.half or dtype == torch.bfloat16:
                    return torch.zeros((shape + [5]), device=device).uniform_().to(dtype=dtype)[:, 2]
                # 如果数据类型不是半精度或者 bfloat16，生成一个形状为 (shape + [5]) 的全零张量，并在该张量上生成指定数据类型的均匀分布的随机数，然后取第 3 列的数据
                return torch.zeros((shape + [5]), device=device, dtype=dtype).uniform_()[:, 2]
            else:
                # 如果 shape 的长度大于 1
                # 新形状为 [2, shape[1], 7, 1, shape[0], 1, 10]
                new_shape = [2, shape[1], 7, 1, shape[0], 1, 10]
                # 如果数据类型是半精度或者 bfloat16，生成一个形状为 new_shape 的全零张量，并在该张量上生成均匀分布的随机数，然后转换为指定的数据类型
                if dtype == torch.half or dtype == torch.bfloat16:
                    prob_dist = torch.zeros(new_shape, device=device).uniform_().to(dtype=dtype)
                # 如果数据类型不是半精度或者 bfloat16，生成一个形状为 new_shape 的全零张量，并在该张量上生成指定数据类型的均匀分布的随机数
                else:
                    prob_dist = torch.zeros(new_shape, device=device, dtype=dtype).uniform_()
                # 调整张量的维度顺序，交换维度 1 和 4 的位置
                prob_dist = prob_dist.transpose(1, 4)
                # 选择张量的部分数据，获取 [1, :, 5, 0, :, 0, 4] 处的数据
                prob_dist = prob_dist[1, :, 5, 0, :, 0, 4]
                # 断言张量不是连续的，用于检查程序的正确性
                assert not prob_dist.is_contiguous()  # sanity check
                # 返回生成的概率分布张量
                return prob_dist

    # FIXME: move to elementwise ternary test suite
    # As the test fails with Runtime Error not raised on XLA
    # 将此部分移动到逐元素三元测试套件中
    # 由于在 XLA 上没有引发 Runtime Error，此测试失败
    @onlyNativeDeviceTypes
    def test_where_scalar_handcrafted_values(self, device):
        # Tests ScalarxScalar, ScalarxTensor and TensorxScalar
        # variant of `where` against NumPy version with
        # handcrafted values.

        # 定义条件的形状为 (5, 5)
        condition_shape = (5, 5)
        # 定义不同的数据类型
        dtypes = (
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int64,
            torch.float16, torch.float32, torch.float64,
            torch.complex64, torch.complex128,
        )
        # 定义不同的张量形状
        shapes = ((), (5,), (1, 5),)

        with torch.no_grad():
            # 生成器推导式，创建不同形状和数据类型的张量，并填充为 17
            tensors = (torch.empty(shape, dtype=dtype, device=device).fill_(17)
                       for shape, dtype in product(shapes, dtypes))

        # 不同的 `x` 和 `y` 值用于输出值的比较
        x_vals = (True, 3, 7.0, 1 + 0.5j)
        y_vals = itertools.chain((False, 4, 8.0, 2 + 0.5j), tensors)
        for x in x_vals:
            for y in y_vals:
                # 随机生成布尔类型的条件张量
                condition = torch.empty(*condition_shape, dtype=torch.bool, device=device).bernoulli_()
                # 确定 `x` 和 `y` 的共同数据类型
                common_dtype = torch.result_type(x, y)

                # 定义函数检查 torch.where 的输出是否与 numpy.where 的输出相等
                def check_equal(condition, x, y):
                    # 将条件张量和输入张量转换为 NumPy 数组
                    condition_np = condition.cpu().numpy()
                    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

                    # NumPy 会自动提升数据类型到 double，因此需要将预期结果转换为正确的数据类型
                    expected = torch.from_numpy(np.where(condition_np, x_np, y_np)).to(common_dtype)
                    # 使用 torch.where 计算结果
                    result = torch.where(condition, x, y)
                    # 断言预期结果和计算结果相等
                    self.assertEqual(expected, result)

                # 分别检查 x 和 y 作为条件的结果
                check_equal(condition, x, y)
                check_equal(condition, y, x)

                # 如果运行在 CUDA 上，还需要额外检查一些情况
                if self.device_type == "cuda":
                    check_equal(condition, torch.tensor(x), y)
                    check_equal(condition, y, torch.tensor(x))
                    if not isinstance(y, torch.Tensor):
                        check_equal(condition, torch.tensor(y), torch.tensor(x))
                    if isinstance(y, torch.Tensor) and y.ndim > 0:
                        check_equal(torch.tensor(True), x, y)
                        check_equal(torch.tensor(True), y, x)
    # 测试钩子的移除功能
    def test_hook_remove(self, device):
        # 引用：https://github.com/pytorch/pytorch/issues/58354
        def _test_helper(remove_hook):
            # 定义安装钩子的函数
            def install_hook(tensor):
                handle = None

                # 定义钩子函数
                def hook(tensor):
                    if remove_hook:
                        handle.remove()  # 如果需要移除钩子，则执行移除操作
                    return torch.zeros_like(tensor)  # 返回与输入tensor相同形状的全零tensor
                handle = tensor.register_hook(hook)  # 注册钩子函数到给定的tensor上

            t = torch.ones((1, 5), device=device, requires_grad=True)  # 创建一个全1的tensor，设备由参数device指定，需要梯度信息
            install_hook(t)  # 在tensor上安装钩子

            # 第一次反向传播调用
            t.mean().backward()  # 计算tensor的均值，并执行反向传播
            self.assertEqual(t.grad, torch.zeros_like(t))  # 断言tensor的梯度为全零

            # 第二次反向传播调用
            t.mean().backward()  # 再次调用均值和反向传播
            if remove_hook:
                # 如果移除了钩子，确保返回正常的梯度值
                self.assertEqual(t.grad, 0.2 * torch.ones_like(t))  # 断言tensor的梯度为0.2乘以全1 tensor
            else:
                self.assertEqual(t.grad, torch.zeros_like(t))  # 否则，断言tensor的梯度为全零

        _test_helper(remove_hook=True)  # 调用辅助函数，测试移除钩子的情况
        _test_helper(remove_hook=False)  # 调用辅助函数，测试不移除钩子的情况

    # FIXME: 让 PyTorch/XLA 运行 test_testing
    # 这个测试理想情况下应该在 test_testing.py 中，
    # 但由于 pytorch/xla 从 test_torch.py 运行测试，所以我们将其放在这里。
    @skipXLA
    def test_skip_xla(self, device):
        if self.device_type == 'xla':
            # 不应该运行到这里！
            self.assertTrue(False)  # 断言为真，如果运行到这里则测试失败

    # FIXME: 让 PyTorch/XLA 运行 test_testing
    # 这个测试理想情况下应该在 test_testing.py 中，
    # 但由于 pytorch/xla 从 test_torch.py 运行测试，所以我们将其放在这里。
    @expectedFailureXLA
    def test_expected_failure_xla(self, device):
        if self.device_type == 'xla':
            self.assertTrue(False)  # 断言为真，如果设备类型为'xla'则测试失败

    # FIXME: 让 PyTorch/XLA 运行 test_testing
    # 这个测试理想情况下应该在 test_testing.py 中，
    # 但由于 pytorch/xla 从 test_torch.py 运行测试，所以我们将其放在这里。
    def test_assertRaisesRegex_ignore_msg_non_native_device(self, device):
        # 验证 self.assertRaisesRegex 只检查错误，并忽略非本地设备上的消息。
        x = torch.randn((10, 3), device=device)  # 在指定设备上生成随机tensor
        t = torch.empty(10, dtype=torch.int64, device=device).random_(0, 3)  # 在指定设备上生成随机整数tensor
        invalid_weight = torch.randn(4, device=device)  # 在指定设备上生成无效权重tensor
        msg = "weight tensor should be defined either for all 3 classes or no classes"  # 错误消息字符串

        # XLA 在运行时会引发不同消息的 RuntimeError。
        with self.assertRaisesRegex(RuntimeError, msg):  # 断言在运行时引发特定消息的 RuntimeError
            torch.nn.functional.nll_loss(x, t, weight=invalid_weight)  # 调用损失函数，预期引发异常

    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.complex32))
    # 定义测试方法 test_copy_
    def test_copy_(self, device, dtype):
        
        # 定义一个内部函数 can_cast，用于判断是否可以进行数据类型转换
        def can_cast(src_dtype, dst_dtype):
            # torch.can_cast(torch.int16, torch.uint8) 返回 True，
            # 实际上这不是安全的转换。这个函数在这种情况下返回 False。
            
            # 内部函数，检查数据类型是否为无符号整数类型
            def is_unsigned_int(dtype):
                return dtype is torch.uint8
            
            # 如果目标数据类型是无符号整数类型，则源数据类型也必须是无符号整数类型才能进行转换
            if is_unsigned_int(dst_dtype):
                return is_unsigned_int(src_dtype)
            
            # 否则，使用 torch.can_cast 检查是否可以进行类型转换
            return torch.can_cast(src_dtype, dst_dtype)

        # 定义内部函数 make_tensor_wrapper，根据形状和数据类型生成张量
        def make_tensor_wrapper(shape, dtype):
            # 如果数据类型不是 torch.complex32，使用 make_tensor 函数生成张量
            if dtype is not torch.complex32:
                return make_tensor(shape, device=device, dtype=dtype)
            # 否则，使用 torch.randn 生成 complex32 类型的张量
            return torch.randn(shape, device=device, dtype=dtype)

        # 使用 make_tensor_wrapper 函数生成一个张量 t，形状为 (50,)，数据类型为参数传入的 dtype
        t = make_tensor_wrapper((50,), dtype)
        
        # 获取所有类型及复杂类型，并包括 torch.bool, torch.half, torch.bfloat16, torch.complex32
        src_dtypes = all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.complex32)
        
        # 遍历所有源数据类型
        for src_dtype in src_dtypes:
            # 使用 make_tensor_wrapper 函数生成一个形状为 (50,)，数据类型为 src_dtype 的张量 src
            src = make_tensor_wrapper((50,), dtype=src_dtype)
            
            # 将 t 的内容复制到 src 中
            t.copy_(src)
            
            # 使用 make_tensor_wrapper 函数生成一个形状为 (50,)，数据类型为 src_dtype 的张量 dst
            dst = make_tensor_wrapper((50, ), dtype=src_dtype)
            
            # 如果可以将 src_dtype 转换为 dtype
            if can_cast(src_dtype, dtype):
                rtol = None
                atol = None
                
                # 如果目标数据类型是 torch.half 或 torch.complex32，设置 rtol 和 atol 的值为 1e-3
                if dtype in (torch.half, torch.complex32):
                    rtol = 1e-3
                    atol = 1e-3
                
                # 如果目标数据类型是 torch.bfloat16，设置 rtol 和 atol 的值为 1e-2
                if dtype in (torch.bfloat16,):
                    rtol = 1e-2
                    atol = 1e-2
                
                # 断言 src 和 dst.copy_(t) 的内容相等，允许的相对误差为 rtol，绝对误差为 atol
                self.assertEqual(src, dst.copy_(t), rtol=rtol, atol=atol)

    # 使用装饰器 @dtypes 指定测试方法 test_item 的参数类型
    @dtypes(*all_types_and_complex_and(
        torch.bool, torch.half, torch.bfloat16, torch.complex32,
        torch.uint16, torch.uint32, torch.uint64))
    # 定义测试方法 test_item
    def test_item(self, device, dtype):
        # 如果设备类型为 'xla' 且数据类型为 torch.uint16, torch.uint32, torch.uint64，则跳过测试
        if torch.device(device).type == 'xla' and dtype in [torch.uint16, torch.uint32, torch.uint64]:
            self.skipTest('uint16,32,64 not implemented on XLA')
        
        # 生成一个形状为空的张量 t，设备为参数传入的 device，数据类型为 dtype
        t = torch.ones((), device=device, dtype=dtype)
        
        # 断言 t.item() 的值为 1
        self.assertEqual(1, t.item())

    # 使用装饰器 @onlyNativeDeviceTypes
    # 定义一个测试方法，用于测试非连续张量的原位掩码赋值操作
    def test_masked_scatter_inplace_noncontiguous(self, device):
        # 创建一个形状为 (5, 2) 的长整型全零张量，设备为指定设备
        t = torch.zeros(5, 2, dtype=torch.long, device=device)
        # 对 t 进行转置，使其变为非连续张量
        t_non_contig = t.transpose(0, 1)
        # 对 t_non_contig 进行连续化处理
        t_contig = t_non_contig.contiguous()

        # 断言 t_contig 是否为连续张量
        assert t_contig.is_contiguous()
        # 断言 t_non_contig 是否为非连续张量
        assert not t_non_contig.is_contiguous()

        # 创建一个形状为 (2, 5) 的布尔类型张量作为掩码，设备为指定设备
        mask = torch.tensor([[False, True], [False, True], [False, False], [True, True], [True, True]], device=device)
        # 对 mask 进行转置，使其变为非连续张量
        mask_non_contig = mask.transpose(0, 1)
        # 对 mask_non_contig 进行连续化处理
        mask_contig = mask_non_contig.contiguous()

        # 断言 mask_contig 是否为连续张量
        assert mask_contig.is_contiguous()
        # 断言 mask_non_contig 是否为非连续张量
        assert not mask_non_contig.is_contiguous()

        # 创建一个形状为 (2, 5) 的张量作为数据源，设备为指定设备
        source = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 9]], device=device)

        # 对 t_contig 执行原位掩码赋值操作，结果存储在 expected 中
        expected = t_contig.masked_scatter_(mask_contig, source)

        # 对 t_non_contig 执行原位掩码赋值操作，结果存储在 actual 中
        actual = t_non_contig.masked_scatter_(mask_non_contig, source)
        # 断言 actual 与 expected 是否相等
        self.assertEqual(actual, expected)

        # 对 t_contig 执行原位掩码赋值操作，使用 mask_non_contig，结果存储在 actual 中
        actual = t_contig.masked_scatter_(mask_non_contig, source)
        # 断言 actual 与 expected 是否相等
        self.assertEqual(actual, expected)

        # 对 t_non_contig 执行原位掩码赋值操作，使用 mask_contig，结果存储在 actual 中
        actual = t_non_contig.masked_scatter_(mask_contig, source)
        # 断言 actual 与 expected 是否相等
        self.assertEqual(actual, expected)
# Tests that compare a device's computation with the (gold-standard) CPU's.
class TestDevicePrecision(TestCase):
    exact_dtype = True

    # FIXME: move to indexing test suite
    # 仅在使用CUDA设备时运行该测试
    @onlyCUDA
    def test_index_add_bfloat16(self, device):
        # 创建一个在CPU上的随机张量，并转换为bfloat16数据类型
        inp_tensor = torch.randn(5, 3, device='cpu').bfloat16()
        # 创建一个CPU上的张量t，指定数据类型为bfloat16
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.bfloat16, device='cpu')
        # 创建一个CPU上的索引张量
        index = torch.tensor([0, 4, 2], device='cpu')
        # 在inp_tensor上执行index_add操作，将t张量加到指定索引位置
        out_cpu = inp_tensor.index_add(0, index, t)

        # 将inp_tensor和t张量移动到指定的设备上
        inp_tensor = inp_tensor.to(device=device)
        t = t.to(device=device)
        index = index.to(device=device)
        # 在指定设备上执行index_add操作
        out_gpu = inp_tensor.index_add(0, index, t)

        # 断言两个操作结果在指定的容差范围内相等
        self.assertEqual(out_cpu, out_gpu, atol=1e-2, rtol=0)

    # FIXME: move to serialization test suite
    # 测试在指定设备上对张量进行序列化和反序列化操作
    def test_device_serialization(self, device):
        # 创建一个在指定设备上的随机张量
        x = torch.randn(4, 4, device=device)

        # 使用临时文件进行张量的保存和加载
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)

        # 断言保存和加载后的张量相等，并且类型和设备一致
        self.assertEqual(x_copy, x)
        self.assertIs(type(x_copy), type(x))
        self.assertEqual(x_copy.device, x.device)

    # FIXME: move to serialization test suite
    # 测试多设备环境下的张量序列化和反序列化操作
    @deviceCountAtLeast(2)
    def test_multidevice_serialization(self, devices):
        # 在不同设备上创建两个随机张量
        x = [torch.randn(4, 4, device=devices[0]),
             torch.randn(4, 4, device=devices[1])]

        # 使用临时文件进行张量列表的保存和加载
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            x_copy = torch.load(f)

        # 断言保存和加载后的每个张量列表元素相等，并且类型和设备一致
        for original, cp in zip(x, x_copy):
            self.assertEqual(cp, original)
            self.assertIs(type(cp), type(original))
            self.assertEqual(cp.device, original.device)

    # FIXME: move to data movement test suite
    # 测试不同设备之间的张量复制操作
    @deviceCountAtLeast(1)
    def test_copy_noncontig(self, devices):
        # 定义一个测试函数，用于在指定设备上执行复制测试
        def do_test(d0, d1):
            # 在指定设备d0上创建一个浮点型张量x
            x = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], device=d0)
            # 在指定设备d1上创建一个整型张量y
            y = torch.tensor([0, 0, 0, 0, 0, 0], device=d1)
            # 断言x和y的数据类型不相同
            self.assertNotEqual(x.dtype, y.dtype)

            # 将x的奇数索引位置的值复制到y中
            y[::2].copy_(x[::2])
            # 断言复制后y张量的值
            self.assertEqual(y, [1, 0, 3, 0, 5, 0])

        # 在CPU和指定设备之间执行复制测试
        do_test('cpu', devices[0])
        do_test(devices[0], 'cpu')

        # 如果有多个设备，则在第一个和第二个设备之间执行复制测试
        if len(devices) > 1:
            do_test(devices[0], devices[1])

    # FIXME: move to type conversion test suite
    # 在同一设备上测试张量类型的转换
    @deviceCountAtLeast(2)
    def test_type_conversions_same_device(self, devices):
        # 在第二个设备上创建一个随机张量x
        x = torch.randn(5, 5, device=devices[1])
        # 断言将x转换为整型后的设备与原设备一致
        self.assertEqual(x.int().device, torch.device(devices[1]))
        self.assertEqual(x.type(torch.int).device, torch.device(devices[1]))
        self.assertEqual(x.to(torch.int).device, torch.device(devices[1]))

    # 根据CUDA是否可用返回指定的浮点类型
    @dtypesIfCUDA(torch.half, torch.float, torch.double,
                  torch.int8, torch.short, torch.int, torch.long,
                  torch.uint8)
    # 返回指定的浮点类型
    @dtypes(torch.float, torch.double,
            torch.int8, torch.short, torch.int, torch.long,
            torch.uint8)
    # 定义一个测试方法，用于测试从序列创建张量的功能
    def test_from_sequence(self, device, dtype):
        # 创建一个包含嵌套列表的序列
        seq = [list(range(i * 4, i * 4 + 4)) for i in range(5)]
        # 创建一个参考张量，包含从0到19的数，并调整形状为5行4列
        reference = torch.arange(0, 20).resize_(5, 4)
        # 使用断言比较创建的张量与参考张量是否相等，可以指定数据类型，并关闭精确数据类型检查
        self.assertEqual(torch.tensor(seq, dtype=dtype, device=device), reference, exact_dtype=False)

    # FIXME: 已迁移到索引测试套件
    @deviceCountAtLeast(1)
    def test_advancedindex_mixed_cpu_devices(self, devices) -> None:
        # 定义一个测试函数，测试张量的高级索引操作
        def test(x: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> None:
            # 测试使用getitem方法获取元素
            self.assertEqual(x[:, ia, None, ib, 0].cpu(),
                             x.cpu()[:, ia.cpu(), None, ib.cpu(), 0])
            self.assertEqual(x[ia], x.cpu()[ia.cpu()])
            # 测试使用setitem方法设置元素
            x_clone1 = x.clone()
            x_clone2 = x.clone()
            first_shape = x[:, ia, None, ib, 0].shape
            second_shape = x[ia].shape
            x_clone1[:, ia, None, ib, 0] = torch.randn(first_shape).to(x_clone1)
            x_clone2[ia] = torch.randn(second_shape).to(x_clone2)

        # 将CPU设备赋给变量cpu
        cpu = torch.device('cpu')
        # 遍历设备列表中的每个设备
        for device in devices:
            # 创建一个形状为(3, 4, 4, 4, 3)的随机张量x
            x = torch.randn(3, 4, 4, 4, 3)
            # 创建一个包含索引的张量ia和ib，将它们转换到cpu设备
            ia = torch.tensor([0, 2, 1]).to(cpu)
            ib = torch.tensor([0, 2, 1]).to(cpu)

            # 使用cpu设备对x和ia、ib进行索引操作的测试
            test(x.to(device), ia, ib)

            # 使用混合的cpu和设备张量对x和ia、ib进行索引操作的测试
            ia = ia.to(cpu)
            ib = ib.to(device)
            test(x.to(device), ia, ib)

    # FIXME: 已移至数据移动测试套件

    @deviceCountAtLeast(1)
    def test_advancedindex_mixed_devices_error(self, devices) -> None:
        # 定义一个测试函数，测试混合设备上的错误索引操作
        def test(x: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> None:
            # 测试使用getitem方法获取元素时的错误情况
            with self.assertRaisesRegex(RuntimeError, fr"indices should be either .* \({x.device}\)"):
                value = x[:, ia, None, ib, 0]
            with self.assertRaisesRegex(RuntimeError, fr"indices should be either .* \({x.device}\)"):
                value = x[ib]

        # 将CPU设备赋给变量cpu
        cpu = torch.device('cpu')
        # 遍历设备列表中的每个设备
        for device in devices:
            # 创建一个形状为(3, 4, 4, 4, 3)的随机张量x
            x = torch.randn(3, 4, 4, 4, 3)
            # 创建一个包含索引的张量ia和ib，并将它们转换到当前设备
            ia = torch.tensor([0, 2, 1]).to(device)
            ib = torch.tensor([0, 2, 1]).to(device)

            # 使用设备张量对x和ia、ib进行错误索引操作的测试
            test(x, ia, ib)

            # 使用混合的cpu和设备张量对x和ia、ib进行错误索引操作的测试
            x = x.to(cpu)
            ia = ia.to(cpu)
            ib = ib.to(device)
            test(x, ia, ib)

            # 如果设备列表中的设备数大于1，则对不同设备上的x和ia、ib进行错误索引操作的测试
            if len(devices) > 1:
                other_device = devices[0] if device == devices[1] else devices[1]
                x = x.to(device)
                ia = ia.to(cpu)
                ib = ib.to(other_device)
                test(x, ia, ib)
    # 测试在给定设备上的张量复制和广播操作
    def test_copy_broadcast(self, device) -> None:
        # 创建一个形状为 (10, 5) 的张量 x，数据为随机数
        x = torch.randn(10, 5)
        # 创建一个在指定设备上的形状为 (5) 的随机张量 y
        y = torch.randn(5, device=device)
        # 将张量 y 的数据复制到张量 x 中
        x.copy_(y)
        # 断言 x 的第 3 行数据与 y 相等
        self.assertEqual(x[3], y)

        # 创建一个在指定设备上的形状为 (10, 5) 的随机张量 x
        x = torch.randn(10, 5, device=device)
        # 创建一个形状为 (5) 的随机张量 y
        y = torch.randn(5)
        # 将张量 y 的数据复制到张量 x 中
        x.copy_(y)
        # 断言 x 的第 3 行数据与 y 相等
        self.assertEqual(x[3], y)

    # FIXME: move to an elementwise ternary test suite
    # 使用装饰器定义的测试函数，测试 clamp 方法在不同设备和数据类型下的行为
    @dtypes(torch.int64, torch.float32, torch.float64)
    def test_clamp(self, device, dtype):
        # 定义测试参数的组合
        test_args = [
            *product(
                [(100, 50), (10, 64), (97,)],  # shape
                (True, False),  # non-contiguous
            )
        ]

        # 遍历测试参数组合
        for shape, noncontig in test_args:
            # 创建指定设备上指定数据类型的张量 x
            x = make_tensor(shape, device=device, dtype=dtype,
                            noncontiguous=noncontig)
            # 创建与 x 相同形状的张量 ub 和 lb
            ub = make_tensor(shape, device=device, dtype=dtype,
                             noncontiguous=noncontig)
            lb = make_tensor(shape, device=device, dtype=dtype,
                             noncontiguous=noncontig)

            # 使用 clamp 方法限制 x 的取值范围，并与期望结果进行比较
            expect = x.max(lb).min(ub)
            actual = x.clamp(lb, ub)
            self.assertEqual(expect, actual)

            # 使用 numpy 的 clip 方法对比 x 的 CPU 版本与 lb、ub 的 clip 结果
            expect = np.clip(x.cpu().numpy(), lb.cpu().numpy(), ub.cpu().numpy())
            self.assertEqual(expect, actual)

            # 使用 clamp 方法限制 x 的最小值，并与期望结果进行比较
            expect = x.max(lb)
            actual = x.clamp(min=lb)
            self.assertEqual(expect, actual)

            # 使用 clamp 方法限制 x 的最大值，并与期望结果进行比较
            expect = x.min(ub)
            actual = x.clamp(max=ub)
            self.assertEqual(expect, actual)

            # 测试最小值和最大值的广播
            expect = x.max(lb[0]).min(ub[..., :1])
            actual = x.clamp(lb[0], ub[..., :1])
            self.assertEqual(expect, actual)

            # 测试张量 x 的广播
            expect = x[..., :1].max(lb).min(ub)
            actual = x[..., :1].clamp(lb, ub)
            self.assertEqual(expect, actual)

    # 测试 CUDA 设备上的设备索引
    def test_cuda_device_idx(self, device):
        # 创建一个在指定设备上的形状为 (3) 的零张量 x
        x = torch.zeros(3, device=device)
        # 调用一个特定函数创建在指定设备上的形状为 (3) 的零张量 y
        y = torch._efficientzerotensor(3, device=device)
        # 断言张量 x 和 y 的设备相同
        self.assertEqual(x.device, y.device)
# 我们为子类实现了自定义的释放操作，因此我们需要确保所有这些部分都正常工作。我们将使用 __del__ 方法来追踪对象是否被正确释放。
class Tracker:
    def __init__(self, marker):
        self.marker = marker

    @staticmethod
    def make():
        marker = [False]
        return marker, Tracker(marker)

    def __del__(self):
        self.marker[0] = True

# 定义一个上下文管理器，用于临时禁用 Python 的垃圾回收机制
@contextlib.contextmanager
def disable_gc():
    if gc.isenabled():
        try:
            gc.disable()
            yield
        finally:
            gc.enable()
    else:
        yield

# 测试类，继承自 TestCase
class TestTorch(TestCase):
    exact_dtype = True

    # 测试 torch 模块的 dir() 方法
    def test_dir(self):
        dir(torch)

    # 测试使用 exec 导入 torch 的所有模块
    def test_wildcard_import(self):
        exec('from torch import *')

    # 测试新轴语法与 numpy 比较
    def test_newaxis_numpy_comparison(self):
        def run_test(tensor, *idx):
            npt = tensor.numpy()
            self.assertEqual(tensor[idx], npt[idx])

        # 1维张量测试
        x = torch.arange(0, 10)
        cases = [
            [None],
            [None, None],
            [Ellipsis, None],
            [None, Ellipsis],
            [2, None],
            [None, 2],
            [Ellipsis, None, 2],
            [Ellipsis, 2, None],
            [2, Ellipsis, None],
            [2, None, Ellipsis],
            [None, 2, Ellipsis],
            [None, Ellipsis, 2],
        ]

        # 遍历测试用例
        for case in cases:
            run_test(x, *case)

        # 2维张量测试
        x = torch.arange(0, 12).view(3, 4)
        cases = [
            [None],
            [None, None],
            [None, None, None],
            [Ellipsis, None],
            [Ellipsis, None, None],
            [None, Ellipsis],
            [None, Ellipsis, None],
            [None, None, Ellipsis],
            [2, None],
            [2, None, Ellipsis],
            [2, Ellipsis, None],
            [None, 2, Ellipsis],
            [Ellipsis, 2, None],
            [Ellipsis, None, 2],
            [None, Ellipsis, 2],
            [1, 2, None],
            [1, 2, Ellipsis, None],
            [1, Ellipsis, 2, None],
            [Ellipsis, 1, None, 2],
            [Ellipsis, 1, 2, None],
            [1, None, 2, Ellipsis],
            [None, 1, Ellipsis, 2],
            [None, 1, 2, Ellipsis],
        ]

        # 遍历测试用例
        for case in cases:
            run_test(x, *case)

    # 生成一个连续递增的张量
    def _consecutive(self, size, start=1):
        sequence = torch.ones(torch.tensor(size).prod(0)).cumsum(0)
        sequence.add_(start - 1)
        return sequence.resize_(*size)
    # 定义测试方法 `test_newindex`，用于测试索引赋值操作

        reference = self._consecutive((3, 3, 3))
        # 创建一个参考值，用于比较测试结果；_consecutive() 方法在其他测试中已验证

        # 定义内部函数 checkPartialAssign，用于部分索引赋值的测试
        def checkPartialAssign(index):
            reference = torch.zeros(3, 3, 3)
            # 创建一个全零张量作为参考
            reference[index] = self._consecutive((3, 3, 3))[index]
            # 对部分索引赋值，将其与预期结果进行比较
            self.assertEqual(reference[index], self._consecutive((3, 3, 3))[index], atol=0, rtol=0)
            # 将部分索引再次赋值为零，与全零张量进行比较
            reference[index] = 0
            self.assertEqual(reference, torch.zeros(3, 3, 3), atol=0, rtol=0)

        # 分别测试不同的索引赋值情况
        checkPartialAssign(0)
        checkPartialAssign(1)
        checkPartialAssign(2)
        checkPartialAssign((0, 1))
        checkPartialAssign((1, 2))
        checkPartialAssign((0, 2))
        checkPartialAssign(torch.LongTensor((0, 2)))

        # 使用 assertRaises 测试超出索引范围的赋值操作是否会触发 IndexError 异常
        with self.assertRaises(IndexError):
            reference[1, 1, 1, 1] = 1
        with self.assertRaises(IndexError):
            reference[1, 1, 1, (1, 1)] = 1
        with self.assertRaises(IndexError):
            reference[3, 3, 3, 3, 3, 3, 3, 3] = 1
        with self.assertRaises(IndexError):
            reference[0.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, :, 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, ..., 0.0:2.0] = 1
        with self.assertRaises(IndexError):
            reference[0.0, :, 0.0] = 1

    # Test `torch._check*` functions
    # 定义一个测试方法，用于检查各种 Torch 库函数的行为
    def test_check(self):
        # 定义测试用例列表，每个元素包含一个 Torch 库函数和预期的错误类型
        test_cases = [
            # check function, expected error
            (torch._check, RuntimeError),          # 检查 RuntimeError 报错
            (torch._check_index, IndexError),      # 检查 IndexError 报错
            (torch._check_value, ValueError),      # 检查 ValueError 报错
            (torch._check_type, TypeError),        # 检查 TypeError 报错
            (torch._check_not_implemented, NotImplementedError),  # 检查 NotImplementedError 报错
        ]

        # 遍历每个测试用例
        for check_fn, expected_error in test_cases:
            # 条件为 True 时不应该引发错误
            check_fn(True)

            # 测试 cond=False 时的默认错误消息
            default_message = 'Expected cond to be True'
            with self.assertRaisesRegex(expected_error, default_message):
                check_fn(False)

            # 测试带有简单错误消息的情况
            message = 'message'
            with self.assertRaisesRegex(expected_error, message):
                check_fn(False, lambda: message)

            # 测试带有 tensor 的消息
            def message():
                return torch.arange(4)

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(False, message)

            # 测试带有格式化字符串消息的情况
            def message():
                return f"{'test'} {[1, 2, 'a', True]} {True} {100} {torch.arange(4)}"

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(False, message)

            # 测试错误的 `cond` 参数类型
            with self.assertRaisesRegex(TypeError, 'cond must be a bool'):
                check_fn('wrong type')

            with self.assertRaisesRegex(TypeError, 'cond must be a bool'):
                check_fn(torch.tensor(True))

    # FIXME: move to indexing test suite
    # 定义一个测试方法，用于测试索引加法功能
    def test_index_add(self):
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 遍历布尔值组合，测试不同情况下的目标、源和索引是否连续
            for dest_contig, src_contig, index_contig in product([True, False], repeat=3):
                # 遍历其他尺寸的组合，可以为空或者包含两个元素
                for other_sizes in ((), (4, 5)):
                    # 遍历数据类型为torch.int和torch.long
                    for dtype in [torch.int, torch.long]:
                        # 初始化复制数量和目标数量
                        num_copy, num_dest = 3, 3
                        # 生成指定设备上随机的目标张量
                        dest = torch.randn(num_dest, *other_sizes, device=device)
                        # 如果目标张量不是连续的，则使用非连续版本
                        if not dest_contig:
                            dest = make_tensor(dest.shape, device=device, dtype=dest.dtype, noncontiguous=True)
                        # 生成指定设备上随机的源张量
                        src = torch.randn(num_copy, *other_sizes, device=device)
                        # 如果源张量不是连续的，则使用非连续版本
                        if not src_contig:
                            src = noncontiguous_like(src)
                        # 生成随机的索引，并确保其在指定设备上，选择其数据类型
                        idx = torch.randperm(num_dest, dtype=dtype, device=device).narrow(0, 0, num_copy)
                        # 如果索引不是连续的，则使用非连续版本
                        if not index_contig:
                            idx = noncontiguous_like(idx)
                        # 使用原子加法的索引加法操作，不带alpha参数
                        dest2 = dest.clone()
                        dest.index_add_(0, idx, src)
                        # 使用循环实现相同的操作，以便进行断言比较
                        for i in range(idx.size(0)):
                            dest2[idx[i]] += src[i]
                        self.assertEqual(dest, dest2)
                        # 使用原子加法的索引加法操作，带alpha参数为2
                        dest2 = dest.clone()
                        dest.index_add_(0, idx, src, alpha=2)
                        # 使用循环实现相同的操作，以便进行断言比较
                        for i in range(idx.size(0)):
                            dest2[idx[i]] += src[i] * 2
                        self.assertEqual(dest, dest2)

    # FIXME: 解决下面的注释问题，并将其移到索引测试套件中
    # 为特定数据类型在cuda上出现的原子加法问题添加测试覆盖，详细情况见GitHub issue:
    # https://github.com/pytorch/pytorch/issues/29153
    def test_index_add_all_dtypes(self):
        # 遍历所有设备类型
        for device in get_all_device_types():
            # 遍历所有数学数据类型
            for dtype in get_all_math_dtypes(device):
                # 遍历索引数据类型为torch.int和torch.long
                for idx_dtype in [torch.int, torch.long]:
                    size = [5, 5]
                    # 根据数据类型不同生成不同的张量
                    if dtype.is_floating_point or dtype.is_complex:
                        tensor = torch.rand(size, dtype=dtype, device=device)
                    elif dtype.is_signed:
                        tensor = torch.randint(-5, 15, size, dtype=dtype, device=device)
                    else:
                        tensor = torch.randint(0, 10, size, dtype=dtype, device=device)

                    # 使用原子加法的索引加法操作，将张量tensor添加到zeros上
                    zeros = torch.zeros(size, dtype=dtype, device=device)
                    added = zeros.index_add(0, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor)
                    self.assertEqual(added, tensor)

                    # 使用原子加法的索引加法操作，将张量tensor添加到zeros上，并带有alpha参数为-1
                    added = zeros.index_add(0, torch.arange(0, size[0], dtype=idx_dtype, device=device), tensor, alpha=-1)
                    self.assertEqual(added, -tensor)
    @unittest.mock.patch.object(torch._dynamo.config, "suppress_errors", False)
    @set_default_dtype(torch.double)
    def test_index_add_correctness(self):
        # 测试 index_add 方法的正确性
        # 当 alpha 等于 1，index 的数据类型为 torch.long 时，
        # 即使用 scatter_add 方法时，检查是否能得到正确的结果
    
        def helper(dim, dtype, device, size_result, size_source):
            # 辅助函数：根据给定的维度、数据类型、设备和尺寸创建相关的 tensor
            tensor = torch.zeros(size_result, dtype=dtype, device=device)
            index = torch.randint(0, size_result[dim], (size_source[dim],),
                                  dtype=torch.long, device=device)
            if dtype.is_floating_point or dtype.is_complex:
                source = torch.rand(size_source, dtype=dtype, device=device)
            elif dtype.is_signed:
                source = torch.randint(-2, 5, size_source, dtype=dtype, device=device)
            else:
                source = torch.randint(0, 5, size_source, dtype=dtype, device=device)
    
            ref_out = tensor.index_add(dim, index, source, alpha=2.) / 2.
            ref_out = ref_out.to(dtype=dtype)
            out = tensor.index_add(dim, index, source)
            if device == 'cuda':
                # 在 CUDA 设备上，使用相对误差和绝对误差进行比较
                self.assertEqual(out, ref_out, atol=1e-2, rtol=1e-2)
            else:
                # 在其他设备上，scatter_add 使用 fp32 作为累加类型，而 index_add 不是这样的
                self.assertEqual(out, ref_out.to(dtype=dtype), atol=1e-2, rtol=1e-2)
    
        # 遍历维度的负数索引
        for dim in [-1, -2, -3]:
            # 遍历所有类型和复杂类型，包括 torch.half 和 torch.bfloat16
            for dtype in all_types_and_complex_and(torch.half, torch.bfloat16):
                # 遍历所有设备类型
                for device in get_all_device_types():
                    # 遍历不同的尺寸
                    for size in [(2, 512, 256), (5, 256, 256)]:
                        helper(dim, dtype, device, size, size)
    
                # 检查边界情况
                result = torch.zeros(1, 512, 256, dtype=dtype)
                source = torch.ones(1, 512, 256, dtype=dtype)
                index = torch.ones(257).to(dtype=torch.long)
                # 当 index 超出边界时，引发 RuntimeError
                self.assertRaises(RuntimeError, lambda: result.index_add_(dim, index, source))
                index = (torch.ones(256) * 257).to(dtype=torch.long)
                # 当 index 超出边界时，引发 RuntimeError
                self.assertRaises(RuntimeError, lambda: result.index_add_(dim, index, source))
    
    def test_index_add_cornercase(self):
        # 测试 index_add 方法的边界情况
        for device in get_all_device_types():
            dest = torch.randn((), device=device)
            index = torch.tensor([0], device=device)
            source = torch.randn(1, 1, 1, device=device)
            with self.assertRaisesRegex(
                RuntimeError,
                r"source tensor shape must match self tensor shape, excluding the specified dimension",
            ):
                # 当源张量的形状与目标张量的形状不匹配时，引发 RuntimeError
                dest.index_add(0, index, source)
    def test_linspace_logspace(self):
        # 确保输出不需要梯度，无论输入是否需要梯度。
        # 工厂函数的输出不应该成为任何计算图的一部分。

        # 设置起始和结束点
        start = 0.0
        end = 3.0

        # 循环遍历不同的步数
        for step in [0, 1, 2]:
            # 测试 torch.linspace 函数
            self.assertFalse(
                torch.linspace(
                    torch.tensor(start, requires_grad=True),  # 起始点张量化，并要求梯度
                    torch.tensor(end, requires_grad=True),    # 结束点张量化，并要求梯度
                    step                                     # 步数
                ).requires_grad                               # 检查输出是否需要梯度
            )

            # 测试不同组合下的 torch.linspace 函数
            self.assertFalse(
                torch.linspace(torch.tensor(start, requires_grad=True), end, step).requires_grad
            )
            self.assertFalse(
                torch.linspace(start, torch.tensor(end, requires_grad=True), step).requires_grad
            )

            # 测试 torch.logspace 函数
            self.assertFalse(
                torch.logspace(
                    torch.tensor(start, requires_grad=True),  # 起始点张量化，并要求梯度
                    torch.tensor(end, requires_grad=True),    # 结束点张量化，并要求梯度
                    step                                     # 步数
                ).requires_grad                               # 检查输出是否需要梯度
            )

            # 测试不同组合下的 torch.logspace 函数
            self.assertFalse(
                torch.logspace(torch.tensor(start, requires_grad=True), end, step).requires_grad
            )
            self.assertFalse(
                torch.logspace(start, torch.tensor(end, requires_grad=True), step).requires_grad
            )

    # FIXME: move to shape ops test suite


这段代码是一个测试函数，用于验证 `torch.linspace` 和 `torch.logspace` 函数的行为。循环测试了这些函数在不同参数组合下的输出是否需要梯度，以及确保这些输出不会成为计算图的一部分。
    # 定义测试方法 test_unflatten，用于测试 torch.tensor 的 unflatten 方法
    def test_unflatten(self):
        # 测试参数：tensor 为空时进行 unflatten 操作，期望得到一个空的 tensor
        self.assertEqual(torch.tensor([]).unflatten(0, (0, 1)), torch.empty(0, 1))
        # 测试参数：tensor 包含一个元素时进行 unflatten 操作，期望得到一个包含单个元素的二维 tensor
        self.assertEqual(torch.tensor([1]).unflatten(0, (1, 1)), torch.tensor([[1]]))
        # 测试参数：tensor 包含四个元素时进行 unflatten 操作，期望得到一个按给定尺寸重新组织的二维 tensor
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, (2, 2)), torch.tensor([[1, 2], [3, 4]]))
        # 测试参数：使用列表形式指定尺寸进行 unflatten 操作
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, [2, 2]), torch.tensor([[1, 2], [3, 4]]))
        # 测试参数：使用 torch.Size 形式指定尺寸进行 unflatten 操作
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, torch.Size([2, 2])), torch.tensor([[1, 2], [3, 4]]))
        # 测试参数：tensor 形状为 (2, 10)，在第一维度上按指定尺寸重新组织 tensor
        self.assertEqual(torch.ones(2, 10).unflatten(1, (5, 2)), torch.ones(2, 5, 2))
        # 测试参数：在不明确指定第一维度大小的情况下进行 unflatten 操作
        self.assertEqual(torch.tensor([1, 2, 3, 4]).unflatten(0, (-1, 2)),
                         torch.tensor([[1, 2], [3, 4]]))
        # 测试参数：在不明确指定第二维度大小的情况下进行 unflatten 操作
        self.assertEqual(torch.ones(2, 10).unflatten(1, (5, -1)),
                         torch.ones(2, 5, 2))
        # 测试参数：在不需要改变 tensor 形状的情况下进行 unflatten 操作
        self.assertEqual(torch.ones(2, 10).unflatten(1, (-1,)),
                         torch.ones(2, 10))
        # 测试参数：tensor 形状为 (2, 3*4*5*6)，按指定尺寸重新组织 tensor
        self.assertEqual(torch.ones(2, 3 * 4 * 5 * 6).unflatten(1, (3, 4, -1, 6)),
                         torch.ones(2, 3, 4, 5, 6))
        # 测试参数：tensor 形状为 (2, 0, 2)，在第一维度上按指定尺寸重新组织 tensor
        self.assertEqual(torch.ones(2, 0, 2).unflatten(1, (3, -1, 4, 5)),
                         torch.ones(2, 3, 0, 4, 5, 2))

        # 测试无效参数：指定维度为字符串时抛出 TypeError 异常
        with self.assertRaisesRegex(TypeError, r"unflatten\(\): argument 'dim' \(position 1\) must be int, not str"):
            torch.tensor([1]).unflatten('A', (1, 1))

        # 测试无效参数：使用未定义的维度名时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"Name 'A' not found in Tensor\[None\]."):
            torch.ones(4).unflatten('A', (('A', 2), ('B', 2)))

        # 测试其他无效参数：尺寸为空时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"sizes must be non-empty"):
            torch.tensor([1]).unflatten(0, [])
        # 测试其他无效参数：提供的尺寸与指定维度大小不匹配时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"Provided sizes \[2, 2\] don't multiply up to the size of dim 0 \(1\)"):
            torch.tensor([1]).unflatten(0, [2, 2])
        # 测试其他无效参数：尝试在零维度的标量 tensor 上进行 unflatten 操作时抛出 IndexError 异常
        with self.assertRaisesRegex(IndexError, r"Dimension specified as 0 but tensor has no dimensions"):
            torch.tensor(1).unflatten(0, [0])
        # 测试其他无效参数：尝试在多个维度上同时使用 -1 作为尺寸时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, r"only one dimension can be inferred"):
            torch.randn(5, 10).unflatten(1, (-1, -1))
        # 测试其他无效参数：提供的尺寸与指定维度大小不匹配时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError,
                                    r"Provided sizes \[-1, 4\] don't multiply up to the size of dim 1 \(10\)"):
            torch.randn(5, 10).unflatten(1, (-1, 4))
        # 测试其他无效参数：在含有未指定大小的维度上进行 unflatten 操作时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError,
                                    r"the unspecified dimension size -1 can be any value and is ambiguous"):
            torch.randn(2, 0).unflatten(1, (2, -1, 0))

    # 测试从 C++ 生成的警告是否被正确转换为相应的类型
    def test_warn_types(self):
        # 定义测试用例列表，每个元素是一个三元组：函数、警告类型、消息正则表达式
        test_cases = [
            # function, warning type, message
            (torch._C._warn, UserWarning, r"Test message for TORCH_WARN"),
            (torch._C._warn_deprecation, DeprecationWarning, r"Test message for TORCH_WARN_DEPRECATION"),
        ]

        # 遍历测试用例
        for fn, warning_type, message in test_cases:
            # 使用 warnings 模块捕获警告信息
            with warnings.catch_warnings(record=True) as w:
                warnings.resetwarnings()
                # 设置过滤条件，仅记录特定类型的警告
                warnings.filterwarnings('always', category=warning_type)
                # 调用被测试函数
                fn()

                # 断言：确保恰好捕获到一个警告
                self.assertEqual(len(w), 1, msg=f'{warning_type} not raised')
                # 获取第一个捕获到的警告对象
                warning = w[0].message
                # 断言：确保捕获的警告类型是预期的警告类型
                self.assertTrue(isinstance(warning, warning_type), msg=f'{warning_type} not raised')
                # 断言：确保警告消息符合预期的正则表达式
                self.assertTrue(re.search(
                    message,
                    str(warning)))

    def test_structseq_repr(self):
        # 创建一个 tensor a
        a = torch.arange(250).reshape(5, 5, 10)
        # 预期的字符串表示，包含 tensor.max() 的返回结果
        expected = """
        torch.return_types.max(
        values=tensor([[ 40,  41,  42,  43,  44,  45,  46,  47,  48,  49],
                [ 90,  91,  92,  93,  94,  95,  96,  97,  98,  99],
                [140, 141, 142, 143, 144, 145, 146, 147, 148, 149],
                [190, 191, 192, 193, 194, 195, 196, 197, 198, 199],
                [240, 241, 242, 243, 244, 245, 246, 247, 248, 249]]),
        indices=tensor([[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]))"""
        # 断言：tensor a 的字符串表示应与预期的字符串一致
        self.assertEqual(repr(a.max(1)), textwrap.dedent(expected).strip())

    def test_is_same_size(self):
        # 创建几个不同大小的 tensor
        t1 = torch.empty(3, 4, 9, 10)
        t2 = torch.empty(3, 4)
        t3 = torch.empty(1, 9, 3, 3)
        t4 = torch.empty(3, 4, 9, 10)

        # 断言：t1 和 t2 不是相同大小的 tensor
        self.assertFalse(t1.is_same_size(t2))
        # 断言：t1 和 t3 不是相同大小的 tensor
        self.assertFalse(t1.is_same_size(t3))
        # 断言：t1 和 t4 是相同大小的 tensor
        self.assertTrue(t1.is_same_size(t4))

        # 创建几个 nested tensor
        nt1 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(3, 4), torch.ones(5, 4)])
        nt2 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4)])
        nt3 = torch.nested.nested_tensor([torch.ones(2, 4, 5), torch.ones(2, 6, 5)])
        nt4 = torch.nested.nested_tensor([torch.ones(2, 4), torch.ones(3, 4), torch.ones(5, 4)])

        # 断言：nt1 和 nt2 不是相同大小的 nested tensor
        self.assertFalse(nt1.is_same_size(nt2))
        # 断言：nt1 和 nt3 不是相同大小的 nested tensor
        self.assertFalse(nt1.is_same_size(nt3))
        # 断言：nt1 和 nt4 是相同大小的 nested tensor
        self.assertTrue(nt1.is_same_size(nt4))
        
        # 断言：尝试比较 tensor 和 nested tensor 会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "Expected both self and other to be nested tensors."):
            t1.is_same_size(nt1)

        # 断言：尝试比较 nested tensor 和 tensor 会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "Expected both self and other to be nested tensors."):
            nt1.is_same_size(t1)
    # 定义一个测试方法，用于测试张量的设置操作
    def test_tensor_set(self):
        # 创建一个空的张量 t1
        t1 = torch.tensor([])
        # 创建一个形状为 (3, 4, 9, 10) 的未初始化张量 t2，然后用均匀分布填充
        t2 = torch.empty(3, 4, 9, 10).uniform_()
        # 使用 t2 设置 t1 的值
        t1.set_(t2)
        # 断言 t1 和 t2 的存储地址相同
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 创建一个尺寸为 [9, 3, 4, 10] 的张量尺寸对象
        size = torch.Size([9, 3, 4, 10])
        # 使用 t2 的存储设置 t1，从偏移量 0 开始，并指定尺寸
        t1.set_(t2.storage(), 0, size)
        # 断言 t1 的尺寸为 size
        self.assertEqual(t1.size(), size)
        # 使用 t2 的存储设置 t1，从偏移量 0 开始，并使用元组形式的尺寸
        t1.set_(t2.storage(), 0, tuple(size))
        # 再次断言 t1 的尺寸为 size
        self.assertEqual(t1.size(), size)
        # 断言 t1 的步长为 (120, 40, 10, 1)
        self.assertEqual(t1.stride(), (120, 40, 10, 1))
        # 创建一个步长为 (10, 360, 90, 1)
        stride = (10, 360, 90, 1)
        # 使用 t2 的存储设置 t1，从偏移量 0 开始，使用指定的尺寸和步长
        t1.set_(t2.storage(), 0, size, stride)
        # 断言 t1 的步长为 stride
        self.assertEqual(t1.stride(), stride)
        # 使用 t2 的存储设置 t1，从偏移量 0 开始，使用指定的尺寸和步长
        t1.set_(t2.storage(), 0, size=size, stride=stride)
        # 再次断言 t1 的尺寸和步长为指定的 size 和 stride
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        # 测试参数名称
        t1 = torch.tensor([])
        # 情况 1：当源是张量时，使用 set_ 方法设置 t1
        t1.set_(source=t2)
        # 再次断言 t1 和 t2 的存储地址相同
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 情况 2：当源是存储时，使用 set_ 方法设置 t1
        t1.set_(source=t2.storage())
        # 再次断言 t1 和 t2 的存储地址相同
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        # 情况 3：当源是存储，并且同时指定了其他参数，使用 set_ 方法设置 t1
        t1.set_(source=t2.storage(), storage_offset=0, size=size, stride=stride)
        # 断言 t1 的尺寸和步长为指定的 size 和 stride
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), stride)

        # 创建一个布尔类型张量 t1
        t1 = torch.tensor([True, True], dtype=torch.bool)
        # 创建一个布尔类型张量 t2
        t2 = torch.tensor([False, False], dtype=torch.bool)
        # 使用 t2 设置 t1 的值
        t1.set_(t2)
        # 再次断言 t1 和 t2 的存储地址相同
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)

    # 定义一个测试方法，用于测试张量设置操作中的错误情况
    def test_tensor_set_errors(self):
        # 创建一个 dtype 为 float32 的随机张量 f_cpu
        f_cpu = torch.randn((2, 3), dtype=torch.float32)
        # 创建一个 dtype 为 float64 的随机张量 d_cpu
        d_cpu = torch.randn((2, 3), dtype=torch.float64)

        # 改变数据类型，预期抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu.storage()))
        # 改变存储，同时指定尺寸和步长，预期抛出 RuntimeError 异常
        self.assertRaises(RuntimeError,
                          lambda: f_cpu.set_(d_cpu.storage(), 0, d_cpu.size(), d_cpu.stride()))
        # 直接设置不同 dtype 的张量，预期抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: f_cpu.set_(d_cpu))

        # 改变设备，如果 CUDA 可用
        if torch.cuda.is_available():
            # 创建一个 dtype 为 float32，在 CUDA 设备上的随机张量 f_cuda
            f_cuda = torch.randn((2, 3), dtype=torch.float32, device='cuda')

            # 从 CPU 到 CUDA 设备，预期抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_cuda.storage()))
            # 从 CPU 到 CUDA 设备，同时指定尺寸和步长，预期抛出 RuntimeError 异常
            self.assertRaises(RuntimeError,
                              lambda: f_cpu.set_(f_cuda.storage(), 0, f_cuda.size(), f_cuda.stride()))
            # 直接设置从 CUDA 到 CPU，预期抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: f_cpu.set_(f_cuda))

            # 从 CUDA 到 CPU 设备，预期抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: f_cuda.set_(f_cpu.storage()))
            # 从 CUDA 到 CPU 设备，同时指定尺寸和步长，预期抛出 RuntimeError 异常
            self.assertRaises(RuntimeError,
                              lambda: f_cuda.set_(f_cpu.storage(), 0, f_cpu.size(), f_cpu.stride()))
            # 直接设置从 CUDA 到 CPU，预期抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: f_cuda.set_(f_cpu))

    # FIXME: 将这个测试移动到 test_testing.py 中（与所有 close 测试一起移动）
    # NOTE: test_equal 将会被 torch.testing.assert_close 替代，一旦 torch.testing 稳定版发布
    # （beta 版）
    def test_permute(self):
        # 创建原始列表
        orig = [1, 2, 3, 4, 5, 6, 7]
        # 生成一个随机的排列索引，并转换为列表
        perm = torch.randperm(7).tolist()
        # 使用原始形状创建一个空的张量，并填充为0
        x = torch.empty(*orig).fill_(0)
        # 计算通过 perm 进行排列后的新形状
        new = [i - 1 for i in x.permute(*perm).size()]
        # 断言 perm 和 new 应相等
        self.assertEqual(perm, new)
        # 断言 x 的形状应与 orig 相等
        self.assertEqual(x.size(), orig)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_reversed(self):
        # 创建一个从0到9的张量
        val = torch.arange(0, 10)
        # 断言反转后的张量与预期的张量相等
        self.assertEqual(reversed(val), torch.arange(9, -1, -1))

        # 创建一个从1到9的张量，并将其视图改为3x3
        val = torch.arange(1, 10).view(3, 3)
        # 断言反转后的张量与预期的张量相等
        self.assertEqual(reversed(val), torch.tensor([[7, 8, 9], [4, 5, 6], [1, 2, 3]]))

        # 创建一个标量张量
        val = torch.tensor(42)
        # 断言反转后的张量与预期的张量相等
        self.assertEqual(reversed(val), torch.tensor(42))

    def test_contains(self):
        # 创建一个从0到9的张量
        x = torch.arange(0, 10)
        # 断言4是否在张量 x 中
        self.assertEqual(4 in x, True)
        # 断言12是否在张量 x 中
        self.assertEqual(12 in x, False)

        # 创建一个从1到9的张量，并将其视图改为3x3
        x = torch.arange(1, 10).view(3, 3)
        # 创建一个从1到3的张量
        val = torch.arange(1, 4)
        # 断言张量 val 是否在张量 x 中
        self.assertEqual(val in x, True)
        # 将 val 中的所有元素加上10
        val += 10
        # 断言张量 val 是否在张量 x 中
        self.assertEqual(val in x, False)

        # 抛出一个运行时错误，指出张量中不支持字符串 "foo"
        self.assertRaisesRegex(
            RuntimeError,
            f"Tensor.__contains__ only supports Tensor or scalar, but you passed in a {str}.",
            lambda: "foo" in x)
        # 抛出一个运行时错误，指出张量中不支持类型为列表的 [1, 2]
        self.assertRaisesRegex(
            RuntimeError,
            f"Tensor.__contains__ only supports Tensor or scalar, but you passed in a {type([1, 2])}.",
            lambda: [1, 2] in x)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_deepcopy_parameter(self):
        # 导入深度复制函数
        from copy import deepcopy
        # 创建一个线性层，输入维度为10，输出维度为1
        l = torch.nn.Linear(10, 1)
        # 获取线性层的状态字典，并保持变量
        s = l.state_dict(keep_vars=True)
        # 断言状态字典中的 weight 和 bias 是 Parameter 类型
        self.assertEqual(torch.nn.Parameter, type(s['weight']))
        self.assertEqual(torch.nn.Parameter, type(s['bias']))

        # 深度复制状态字典
        s2 = deepcopy(s)
        # 断言复制后的状态字典中的 weight 和 bias 是 Parameter 类型
        self.assertEqual(torch.nn.Parameter, type(s2['weight']))
        self.assertEqual(torch.nn.Parameter, type(s2['bias']))

    def test_pickle(self):
        # 导入 pickle 模块
        import pickle
        # 创建一个5x5的随机张量
        a = torch.randn(5, 5)
        # 序列化张量 a
        serialized = pickle.dumps(a)
        # 反序列化为张量 b
        b = pickle.loads(serialized)
        # 断言张量 a 和 b 相等
        self.assertEqual(a, b)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_pickle_parameter(self):
        # 导入 pickle 模块
        import pickle
        # 创建一个随机张量的参数
        a = torch.nn.Parameter(torch.randn(5, 5))
        # 序列化参数 a
        serialized = pickle.dumps(a)
        # 反序列化为参数 b
        b = pickle.loads(serialized)
        # 断言 b 是 Parameter 类型
        self.assertTrue(isinstance(b, torch.nn.Parameter))
        # 断言参数 a 和 b 的 requires_grad 相等
        self.assertEqual(a.requires_grad, b.requires_grad)
        # 断言参数 a 和 b 的值相等
        self.assertEqual(a, b)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_pickle_parameter_no_requires_grad(self):
        # 导入 pickle 模块
        import pickle
        # 创建一个不需要梯度的随机张量的参数
        a = torch.nn.Parameter(torch.randn(5, 5), requires_grad=False)
        # 序列化参数 a
        serialized = pickle.dumps(a)
        # 反序列化为参数 b
        b = pickle.loads(serialized)
        # 断言 b 是 Parameter 类型
        self.assertTrue(isinstance(b, torch.nn.Parameter))
        # 断言参数 a 和 b 的 requires_grad 相等
        self.assertEqual(a.requires_grad, b.requires_grad)
        # 断言参数 a 和 b 的值相等
        self.assertEqual(a, b)
    def test_pickle_dtype(self):
        # 创建一个 torch.float32 的张量
        t = torch.float32
        # 对张量进行序列化
        serialized = pickle.dumps(t)
        # 反序列化得到的对象
        b = pickle.loads(serialized)
        # 断言反序列化后的对象是 torch.dtype 类型
        self.assertTrue(isinstance(b, torch.dtype))
        # 断言反序列化后的对象与原始对象具有相同的内存地址
        self.assertEqual(id(b), id(t))

    def test_pickle_size(self):
        # 创建一个包含10个随机数的张量并获取其大小
        a = torch.rand(10).size()
        # 对大小信息进行序列化
        serialized = pickle.dumps(a)
        # 反序列化得到的对象
        b = pickle.loads(serialized)
        # 断言反序列化后的对象是 torch.Size 类型
        self.assertTrue(isinstance(b, torch.Size))
        # 断言反序列化后的对象与原始对象相等
        self.assertEqual(a, b)

    def test_pickle_function(self):
        # 设置一个 torch.tanh 的函数对象
        a = torch.tanh
        # 对函数对象进行序列化
        serialized = pickle.dumps(a)
        # 反序列化得到的对象
        b = pickle.loads(serialized)
        # 断言反序列化后的对象与原始对象相等
        self.assertEqual(a, b)

    def test_generator_cpu(self):
        # 测试默认生成器是否相等
        self.assertEqual(torch.default_generator, torch.default_generator)

        # 测试 Generator API
        g1 = torch.Generator()
        g2 = torch.Generator()
        # 设置两个生成器的种子值
        g1.manual_seed(12345)
        g2.manual_seed(12345)
        # 断言两个生成器的初始种子值相等
        self.assertEqual(g1.initial_seed(), g2.initial_seed())

        # 重新设置种子值，并断言初始种子值不相等
        g1.seed()
        g2.seed()
        self.assertNotEqual(g1.initial_seed(), g2.initial_seed())

        # 使用一个生成器生成正态分布随机数，将状态设置到另一个生成器，并比较生成的随机数
        g1 = torch.Generator()
        g2_state = g2.get_state()
        g2_randn = torch.randn(1, generator=g2)
        g1.set_state(g2_state)
        g1_randn = torch.randn(1, generator=g1)
        self.assertEqual(g1_randn, g2_randn)

        # 获取默认生成器的状态，使用一个生成器生成正态分布随机数，将状态设置到另一个生成器，并比较生成的随机数
        default_state = torch.default_generator.get_state()
        q = torch.empty(100)
        g1_normal = q.normal_()
        g2 = torch.Generator()
        g2.set_state(default_state)
        g2_normal = q.normal_(generator=g2)
        self.assertEqual(g1_normal, g2_normal)

    def test_invalid_generator_raises(self):
        # 断言尝试创建无效生成器会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.Generator('opengl'))

    def test_pickle_generator(self) -> None:
        devices = ['cpu']
        if torch.cuda.is_available():
            devices += ['cuda']

        for device in devices:
            with self.subTest(device=device):
                # 创建指定设备上的生成器，并设置种子
                generator = torch.Generator(device=device).manual_seed(12345)
                if device != "cpu":
                    generator.set_offset(100)
                # 生成随机数以推进 RNG 状态
                torch.randn((100, 100), generator=generator, device=device)

                # 对生成器进行序列化和反序列化
                reserialized: torch.Generator = pickle.loads(pickle.dumps(generator))

                # 断言反序列化后的生成器的设备与原始生成器相同
                self.assertEqual(generator.device, reserialized.device)
                # 断言反序列化后的生成器的初始种子值与原始生成器相同
                self.assertEqual(generator.initial_seed(), reserialized.initial_seed())
                # 如果设备不是 CPU，则断言反序列化后的生成器的偏移值与原始生成器相同
                if device != "cpu":
                    self.assertEqual(generator.get_offset(), reserialized.get_offset())
                # 检查生成器状态是否一致
                torch.testing.assert_close(generator.get_state(), reserialized.get_state())
    # 生成 Sobol 序列的参考样本，不进行混洗
    def _sobol_reference_samples(self, scramble: bool) -> torch.Tensor:
        if not scramble:
            # 使用 Joe Kuo 2010 年的理论数值
            return torch.tensor(
                [
                    [0., 0.],
                    [0.5, 0.5],
                    [0.75, 0.25],
                    [0.25, 0.75],
                    [0.375, 0.375],
                    [0.875, 0.875],
                    [0.625, 0.125],
                    [0.125, 0.625],
                ],
            )
        else:
            # 理论数值未知：已验证收敛性质
            return torch.tensor(
                [
                    [0.50860737, 0.29320504],
                    [0.07116939, 0.89594537],
                    [0.49354145, 0.11524881],
                    [0.93097717, 0.70244044],
                    [0.87266153, 0.23887917],
                    [0.31021884, 0.57600391],
                    [0.13687253, 0.42054182],
                    [0.69931293, 0.77336788],
                ],
            )

    # 测试 SobolEngine 的范围是否在 [0, 1] 之间
    def test_sobolengine_bounds(self, scramble: bool = False):
        engine = torch.quasirandom.SobolEngine(100, scramble=scramble, seed=123456)
        sample = engine.draw(512)
        self.assertTrue(torch.all(sample >= 0))
        self.assertTrue(torch.all(sample <= 1))

    # 测试混洗后的 SobolEngine 的范围是否在 [0, 1] 之间
    def test_sobolengine_bounds_scrambled(self):
        self.test_sobolengine_bounds(scramble=True)

    # 测试 SobolEngine 的随机抽样功能
    def test_sobolengine_draw(self, scramble: bool = False):
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        sample = engine.draw(n=len(ref_sample))
        self.assertEqual(sample, ref_sample)
        self.assertEqual(engine.num_generated, len(ref_sample))

    # 测试混洗后的 SobolEngine 的随机抽样功能
    def test_sobolengine_draw_scrambled(self):
        self.test_sobolengine_draw(scramble=True)

    # 测试 SobolEngine 的首个点是否为零向量
    def test_sobolengine_first_point(self):
        for dtype in (torch.float, torch.double):
            engine = torch.quasirandom.SobolEngine(2, scramble=False)
            sample = engine.draw(1, dtype=dtype)
            self.assertTrue(torch.all(sample == 0))
            self.assertEqual(sample.dtype, dtype)
        for dtype in (torch.float, torch.double):
            engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=123456)
            sample = engine.draw(1, dtype=dtype)
            self.assertTrue(torch.all(sample != 0))
            self.assertEqual(sample.dtype, dtype)

    # 测试 SobolEngine 的连续抽样功能
    def test_sobolengine_continuing(self, scramble: bool = False):
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        n_half = len(ref_sample) // 2
        _ = engine.draw(n=n_half)
        sample = engine.draw(n=n_half)
        torch.testing.assert_close(sample, ref_sample[n_half:])

    # 测试混洗后的 SobolEngine 的连续抽样功能
    def test_sobolengine_continuing_scrambled(self):
        self.test_sobolengine_continuing(scramble=True)
    # 测试 SobolEngine 的 reset 方法，验证引擎重置后生成数目为 0
    def test_sobolengine_reset(self, scramble: bool = False):
        # 获取参考样本（无序列号）以供比较，根据需要进行混淆
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        # 创建 SobolEngine 实例，设定维度为 2，选择是否混淆，设置种子为 123456
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        # 绘制一半数量的样本
        _ = engine.draw(n=len(ref_sample) // 2)
        # 重置引擎
        engine.reset()
        # 验证引擎生成数目是否为 0
        self.assertEqual(engine.num_generated, 0)
        # 再次绘制完整数量的样本，并验证其与参考样本的接近程度
        sample = engine.draw(n=len(ref_sample))
        torch.testing.assert_close(sample, ref_sample)

    # 测试 SobolEngine 在混淆状态下的 reset 方法
    def test_sobolengine_reset_scrambled(self):
        # 直接调用非混淆状态下的 reset 测试函数
        self.test_sobolengine_reset(scramble=True)

    # 测试 SobolEngine 的 fast_forward 方法
    def test_sobolengine_fast_forward(self, scramble: bool = False):
        # 获取参考样本（无序列号）以供比较，根据需要进行混淆
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        # 创建 SobolEngine 实例，设定维度为 2，选择是否混淆，设置种子为 123456
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        # 快速前进 4 步
        engine.fast_forward(4)
        # 绘制 4 个样本并验证其与参考样本的接近程度
        sample = engine.draw(n=4)
        torch.testing.assert_close(sample, ref_sample[4:])
        # 使用绘制和快速前进交替操作，验证结果与预期的接近程度
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.draw())
            else:
                engine.fast_forward(1)
        torch.testing.assert_close(
            ref_sample[[i for i in range(8) if i % 2 == 0]],
            torch.from_numpy(np.concatenate(even_draws)),
        )

    # 测试 SobolEngine 在混淆状态下的 fast_forward 方法
    def test_sobolengine_fast_forward_scrambled(self):
        # 直接调用非混淆状态下的 fast_forward 测试函数
        self.test_sobolengine_fast_forward(scramble=True)

    # 测试 SobolEngine 的默认数据类型处理
    def test_sobolengine_default_dtype(self):
        # 创建 SobolEngine 实例，设定维度为 3，选择混淆，设置种子为 123456
        engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123456)
        # 验证默认数据类型是否正确处理为 torch.float32
        self.assertEqual(engine.draw(n=5).dtype, torch.float32)
        # 在默认数据类型设定为 torch.float64 的上下文中，再次创建 SobolEngine 实例，验证数据类型设定是否生效
        with set_default_dtype(torch.float64):
            engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123456)
            # 验证默认数据类型是否正确处理为 torch.float64（在设置为 float64 时）
            self.assertEqual(engine.draw(n=5).dtype, torch.float64)
            # 验证明确传递的数据类型是否被遵守
            self.assertEqual(engine.draw(n=5, dtype=torch.float32).dtype, torch.float32)
            # 重新初始化引擎并验证首次绘制时的数据类型处理是否正确
            engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123456)
            self.assertEqual(engine.draw(n=5, dtype=torch.float32).dtype, torch.float32)

    # 跳过对于 "np.float64 restored as float32 after graph break." 的测试
    @skipIfTorchDynamo("np.float64 restored as float32 after graph break.")
    # 测试 SobolEngine 分布的方法，可以选择是否打乱顺序
    def test_sobolengine_distribution(self, scramble=False):
        # 设置维度为 50 的 SobolEngine 引擎，可以选择是否打乱顺序和种子值
        d = 50
        engine = torch.quasirandom.SobolEngine(d, scramble=scramble, seed=123456)
        # 从引擎中生成 1024 个样本
        sample = engine.draw(1024)
        # 检查样本均值是否接近全为 0.5，允许绝对误差和相对误差为 2
        torch.testing.assert_close(
            torch.mean(sample, dim=0), torch.full((d,), 0.5), atol=2, rtol=2
        )
        # 检查样本的第 25 百分位数是否接近全为 0.25，允许绝对误差和相对误差为 2
        torch.testing.assert_close(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, d), atol=2, rtol=2
        )
        # 检查样本的第 75 百分位数是否接近全为 0.75，允许绝对误差和相对误差为 2
        torch.testing.assert_close(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, d), atol=2, rtol=2
        )

    # 如果 TorchDynamo 可用，跳过测试的标记为“np.float64 restored as float32 after graph break.”
    @skipIfTorchDynamo("np.float64 restored as float32 after graph break.")
    def test_sobolengine_distribution_scrambled(self):
        # 测试 SobolEngine 分布的方法，强制打乱顺序
        self.test_sobolengine_distribution(scramble=True)

    # 测试 SobolEngine draw_base2 方法，可以选择是否打乱顺序
    def test_sobolengine_draw_base2(self, scramble=False):
        # 使用 SobolEngine 引擎生成维度为 2 的基于 2 的样本，可以选择是否打乱顺序和种子值
        ref_sample = self._sobol_reference_samples(scramble=scramble)
        engine = torch.quasirandom.SobolEngine(2, scramble=scramble, seed=123456)
        sample = engine.draw_base2(2)
        # 检查生成的样本与参考样本前 4 个是否一致
        self.assertEqual(ref_sample[:4], sample)
        # 再次生成基于 2 的样本，检查生成的样本与参考样本后 4 个是否一致
        sample = engine.draw_base2(2)
        self.assertEqual(ref_sample[4:8], sample)

    # 测试 SobolEngine draw_base2 方法，强制打乱顺序
    def test_sobolengine_draw_base2_scrambled(self):
        self.test_sobolengine_draw_base2(scramble=True)

    # 测试 SobolEngine 引发异常的情况
    def test_sobolengine_raise(self):
        # 获取 SobolEngine 的最大维度值
        maxdim = torch.quasirandom.SobolEngine.MAXDIM
        # 检查是否引发 ValueError 异常，当维度超过最大维度值时
        with self.assertRaises(ValueError):
            torch.quasirandom.SobolEngine(maxdim + 1)

    # 测试高维度情况下 SobolEngine 的抽样行为
    def test_sobolengine_high_dim(self):
        # 使用维度为 1111 的 SobolEngine 引擎，不打乱顺序且使用种子值
        engine = torch.quasirandom.SobolEngine(1111, scramble=False, seed=123456)
        # 第一次抽样
        samples1 = engine.draw()
        # 统计第一次抽样的唯一值和出现次数
        vals1, counts1 = torch.unique(samples1, return_counts=True)
        # 第二次抽样
        samples2 = engine.draw()
        # 统计第二次抽样的唯一值和出现次数
        vals2, counts2 = torch.unique(samples2, return_counts=True)
        # 检查第一次抽样的唯一值是否为 0.0，出现次数是否为 1111
        self.assertEqual(vals1.item(), 0.0)
        self.assertEqual(counts1.item(), 1111)
        # 检查第二次抽样的唯一值是否为 0.5，出现次数是否为 1111
        self.assertEqual(vals2.item(), 0.5)
        self.assertEqual(counts1.item(), 1111)

    # 测试解析 int64 类型的情况
    def test_parsing_int64(self):
        # 接受整数作为参数
        x = torch.cumsum(torch.ones(5, 5), 0)
        self.assertEqual(x, torch.cumsum(torch.ones(5, 5), torch.tensor(0)))
        # 不接受浮点数变量
        self.assertRaises(TypeError, lambda: torch.cumsum(torch.ones(5, 5), torch.tensor(0.)))
    def test_parsing_double(self):
        # accepts floating point and integer arguments
        x = torch.randn(2, 3)  # 创建一个大小为 (2, 3) 的随机张量 x
        torch.isclose(x, x, 1, 1)  # 检查张量 x 和自身是否在绝对或相对误差容限内接近
        self.assertTrue(torch.isclose(x, x, 1, 1).all())  # 断言：所有元素都在误差容限内接近
        self.assertTrue(torch.isclose(x, x, 1.5, 1.).all())  # 断言：所有元素在不同的误差容限内接近
        # accepts floating point and integer tensors
        self.assertTrue(torch.isclose(x, x, torch.tensor(1), torch.tensor(1)).all())  # 断言：使用张量作为误差容限
        self.assertTrue(torch.isclose(x, x, torch.tensor(1.5), torch.tensor(1.)).all())  # 断言：使用不同的张量作为误差容限
        # doesn't accept variables with requires_grad
        self.assertRaises(TypeError,
                          lambda: torch.isclose(x, x, torch.tensor(1.5), torch.tensor(1., requires_grad=True)).all())  # 断言：不接受具有 requires_grad 的张量

    def test_parsing_intlist(self):
        # parse with integer variables
        self.assertEqual(torch.Size([3, 4]), torch.ones((torch.tensor(3), torch.tensor(4))).shape)  # 断言：使用整数张量作为参数创建全为1的张量的形状
        self.assertEqual(torch.Size([3, 4]), torch.ones(torch.tensor(3), torch.tensor(4)).shape)  # 断言：使用整数张量直接作为参数创建全为1的张量的形状
        # parse with numpy integers
        self.assertEqual(torch.Size([3, 4]), torch.ones((np.array(3), np.int64(4))).shape)  # 断言：使用 numpy 整数创建全为1的张量的形状
        self.assertEqual(torch.Size([3, 4]), torch.ones(np.array(3), np.int64(4)).shape)  # 断言：使用 numpy 整数直接作为参数创建全为1的张量的形状
        self.assertEqual(torch.Size([3, 4]), torch.ones((np.int64(3), np.array(4))).shape)  # 断言：使用 numpy 整数和数组作为参数创建全为1的张量的形状
        self.assertEqual(torch.Size([3, 4]), torch.ones(np.int64(3), np.array(4)).shape)  # 断言：使用 numpy 整数直接作为参数创建全为1的张量的形状

        # fail parse with float variables
        self.assertRaises(TypeError, lambda: torch.ones((torch.tensor(3.), torch.tensor(4))))  # 断言：不接受包含浮点数的元组作为参数
        # fail parse with numpy floats
        self.assertRaises(TypeError, lambda: torch.ones((3., torch.tensor(4))))  # 断言：不接受包含浮点数的元组作为参数
        self.assertRaises(TypeError, lambda: torch.ones((np.array(3.), torch.tensor(4))))  # 断言：不接受包含浮点数的元组作为参数

        # fail parse with > 1 element variables
        self.assertRaises(TypeError, lambda: torch.ones(torch.tensor(3, 3)))  # 断言：不接受包含两个元素的张量作为参数
        self.assertRaises(TypeError, lambda: torch.ones(torch.tensor(3, 3)))  # 断言：不接受包含两个元素的张量作为参数
        self.assertRaises(TypeError, lambda: torch.ones(np.array(3, 3)))  # 断言：不接受包含两个元素的张量作为参数
        self.assertRaises(TypeError, lambda: torch.ones(np.array(3, 3)))  # 断言：不接受包含两个元素的张量作为参数

        # fail parse with additional positional args after intlist arg
        self.assertRaisesRegex(TypeError,
                               "received an invalid combination of arguments",
                               lambda: torch.LongTensor((6, 0), 1, 1, 0))  # 断言：不接受在整数列表参数后跟随额外位置参数
        self.assertRaisesRegex(TypeError,
                               "missing 1 required positional arguments",
                               lambda: torch.tensor().new_zeros((5, 5), 0))  # 断言：在创建张量时缺少必需的位置参数
    # 定义一个测试方法，用于测试从缓冲区创建不同类型的存储对象

    # 创建一个字节数组a，包含 [1, 2, 3, 4]
    a = bytearray([1, 2, 3, 4])
    # 使用 torch 的 ByteStorage 类从字节数组a创建存储对象，并将其转换为列表进行断言
    self.assertEqual(torch.ByteStorage.from_buffer(a).tolist(), [1, 2, 3, 4])

    # 使用 torch 的 ShortStorage 类从字节数组a创建存储对象，使用大端字节序 'big'
    shorts = torch.ShortStorage.from_buffer(a, 'big')
    # 断言存储对象的大小为2
    self.assertEqual(shorts.size(), 2)
    # 将存储对象转换为列表进行断言
    self.assertEqual(shorts.tolist(), [258, 772])

    # 使用 torch 的 IntStorage 类从字节数组a创建存储对象，使用小端字节序 'little'
    ints = torch.IntStorage.from_buffer(a, 'little')
    # 断言存储对象的大小为1
    self.assertEqual(ints.size(), 1)
    # 断言存储对象的第一个元素值为67305985
    self.assertEqual(ints[0], 67305985)

    # 创建一个字节数组f，包含 [0x40, 0x10, 0x00, 0x00]
    f = bytearray([0x40, 0x10, 0x00, 0x00])
    # 使用 torch 的 FloatStorage 类从字节数组f创建存储对象，使用大端字节序 'big'
    floats = torch.FloatStorage.from_buffer(f, 'big')
    # 断言存储对象的大小为1
    self.assertEqual(floats.size(), 1)
    # 断言存储对象的第一个元素值为2.25
    self.assertEqual(floats[0], 2.25)

    # 创建一个字节数组f，包含 [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40]
    f = bytearray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x10, 0x40])
    # 使用 torch 的 BoolStorage 类从字节数组f创建存储对象，使用大端字节序 'big'
    bools = torch.BoolStorage.from_buffer(f, 'big')
    # 断言存储对象的大小为8
    self.assertEqual(bools.size(), 8)
    # 断言存储对象转换为列表后的值
    self.assertEqual(bools.tolist(), [False, True, True, True, True, True, True, True])
    # 断言存储对象的类型为 'torch.BoolStorage'
    self.assertEqual(bools.type(), 'torch.BoolStorage')
    # 断言存储对象是否是 torch.BoolStorage 类型的实例
    self.assertTrue(isinstance(bools, torch.BoolStorage))

    # 创建一个字节数组f，包含特定字节序列的字节数组
    f = bytearray(b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9')
    # 使用 torch 的 BoolStorage 类从字节数组f创建存储对象，使用大端字节序 'big'
    bools = torch.BoolStorage.from_buffer(f, 'big')
    # 断言存储对象的大小为19
    self.assertEqual(bools.size(), 19)

    # 创建一个字节数组f，包含 [0x4A]
    f = bytearray(b'\0x4A')
    # 使用 torch 的 BoolStorage 类从字节数组f创建存储对象，使用大端字节序 'big'
    bools = torch.BoolStorage.from_buffer(f, 'big')
    # 断言存储对象的大小为4
    self.assertEqual(bools.size(), 4)
    # 断言存储对象转换为列表后的值
    self.assertEqual(bools.tolist(), [False, True, True, True])

    # 使用 torch 的 ByteStorage 类从字节数组a创建存储对象
    bytes = torch.ByteStorage.from_buffer(a)
    # 断言存储对象的字节大小为4
    self.assertEqual(bytes.nbytes(), 4)
    # 断言存储对象转换为列表后的值
    self.assertEqual(bytes.tolist(), [1, 2, 3, 4])
    # 断言存储对象是否是 torch.ByteStorage 类型的实例
    self.assertTrue(isinstance(bytes, torch.ByteStorage))
    # 定义测试函数，用于测试字节交换功能
    def test_storage_byteswap(self):
        # 输入数据列表
        input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # 预期交换后的8字节数据
        swapped_8bytes = [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]
        # 预期交换后的4字节数据
        swapped_4bytes = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
        # 预期交换后的2字节数据
        swapped_2bytes = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
        # 预期交换后的1字节数据
        swapped_1byte = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # 创建 TypedStorage 对象，并获取其底层未类型化的存储
        storage = torch.storage.TypedStorage(input, dtype=torch.uint8)._untyped_storage

        # 复制存储以备份数据，并对备份进行不同类型的字节交换操作
        storage_f64 = storage.__copy__()
        storage_f64.byteswap(torch.float64)
        self.assertEqual(storage_f64.tolist(), swapped_8bytes)

        storage_f32 = storage.__copy__()
        storage_f32.byteswap(torch.float32)
        self.assertEqual(storage_f32.tolist(), swapped_4bytes)

        storage_f16 = storage.__copy__()
        storage_f16.byteswap(torch.float16)
        self.assertEqual(storage_f16.tolist(), swapped_2bytes)

        storage_bf16 = storage.__copy__()
        storage_bf16.byteswap(torch.bfloat16)
        self.assertEqual(storage_bf16.tolist(), swapped_2bytes)

        storage_i64 = storage.__copy__()
        storage_i64.byteswap(torch.int64)
        self.assertEqual(storage_i64.tolist(), swapped_8bytes)

        storage_i32 = storage.__copy__()
        storage_i32.byteswap(torch.int32)
        self.assertEqual(storage_i32.tolist(), swapped_4bytes)

        storage_i16 = storage.__copy__()
        storage_i16.byteswap(torch.int16)
        self.assertEqual(storage_i16.tolist(), swapped_2bytes)

        storage_i8 = storage.__copy__()
        storage_i8.byteswap(torch.int8)
        self.assertEqual(storage_i8.tolist(), swapped_1byte)

        storage_ui8 = storage.__copy__()
        storage_ui8.byteswap(torch.uint8)
        self.assertEqual(storage_ui8.tolist(), swapped_1byte)

        storage_bool = storage.__copy__()
        storage_bool.byteswap(torch.bool)
        self.assertEqual(storage_bool.tolist(), swapped_1byte)

        storage_c128 = storage.__copy__()
        storage_c128.byteswap(torch.complex128)
        self.assertEqual(storage_c128.tolist(), swapped_8bytes)

        storage_c64 = storage.__copy__()
        storage_c64.byteswap(torch.complex64)
        self.assertEqual(storage_c64.tolist(), swapped_4bytes)

    # 测试相关于 TypedStorage 的内部函数版本是否不会产生弃用警告
    # 定义一个测试函数，用于测试 TypedStorage 内部功能不产生警告
    def test_typed_storage_internal_no_warning(self):
        # 创建一个大小为 10 的 FloatStorage 对象
        s0 = torch.FloatStorage(10)
        # 获取未类型化的 FloatStorage 对象
        s0_untyped = s0.untyped()
        # 创建一个大小为 10 的张量 t0，元素值为随机数
        t0 = torch.randn(10)

        # 定义包含多个 lambda 表达式的列表 funcs，每个表达式执行 TypedStorage 的相关操作
        funcs = [
            lambda: torch.FloatStorage(_internal=True),  # 创建一个内部使用的 FloatStorage 对象
            lambda: torch.TypedStorage(
                dtype=torch.float,
                device='cpu',
                _internal=True),  # 创建一个内部使用的 TypedStorage 对象
            lambda: torch.TypedStorage(
                wrap_storage=s0_untyped,
                dtype=s0.dtype,
                _internal=True),  # 创建一个内部使用的 TypedStorage 对象，包装已存在的未类型化存储
            lambda: torch.FloatStorage._dtype,  # 获取 FloatStorage 的数据类型
            lambda: s0._resize_(20),  # 调整 s0 的大小为 20
            lambda: s0._size(),  # 获取 s0 的大小
            lambda: s0._untyped_storage,  # 获取 s0 的未类型化存储
            lambda: s0._is_shared(),  # 检查 s0 是否是共享的
            lambda: s0._share_memory_(),  # 共享 s0 的内存
            lambda: s0._pickle_storage_type(),  # 获取 s0 的存储类型的 pickle 表示
            lambda: s0._setitem(slice(0, s0._size()), 1),  # 设置 s0 的部分元素值
            lambda: s0._element_size(),  # 获取 s0 元素的大小
            lambda: s0._deepcopy({}),  # 深拷贝 s0 的内容
            lambda: s0._data_ptr(),  # 获取 s0 数据的指针
            lambda: s0._nbytes(),  # 获取 s0 占用的字节数
            lambda: t0._typed_storage(),  # 获取 t0 的类型化存储
        ]

        # 如果 CUDA 可用，则继续添加 CUDA 相关操作到 funcs 列表中
        if torch.cuda.is_available():
            # 创建一个大小为 10 的 CUDA FloatStorage 对象
            s1 = torch.cuda.FloatStorage(10)
            # 获取未类型化的 CUDA FloatStorage 对象
            s1_untyped = s1.untyped()
            # 创建一个在 CUDA 上的大小为 10 的张量 t1，元素值为随机数
            t1 = torch.randn(10, device='cuda')

            funcs += [
                lambda: torch.cuda.FloatStorage(_internal=True),  # 创建一个内部使用的 CUDA FloatStorage 对象
                lambda: torch.TypedStorage(
                    dtype=torch.float,
                    device='cuda',
                    _internal=True),  # 创建一个内部使用的 CUDA TypedStorage 对象
                lambda: torch.TypedStorage(
                    wrap_storage=s1_untyped,
                    dtype=s1.dtype,
                    _internal=True),  # 创建一个内部使用的 CUDA TypedStorage 对象，包装已存在的未类型化存储
                lambda: torch.cuda.FloatStorage._dtype,  # 获取 CUDA FloatStorage 的数据类型
                lambda: s1._resize_(20),  # 调整 s1 的大小为 20
                lambda: s1._size(),  # 获取 s1 的大小
                lambda: s1._untyped_storage,  # 获取 s1 的未类型化存储
                lambda: s1._is_shared(),  # 检查 s1 是否是共享的
                lambda: s1._share_memory_(),  # 共享 s1 的内存
                lambda: s1._pickle_storage_type(),  # 获取 s1 的存储类型的 pickle 表示
                lambda: s1._setitem(slice(0, s1._size()), 1),  # 设置 s1 的部分元素值
                lambda: s1._element_size(),  # 获取 s1 元素的大小
                lambda: s1._deepcopy({}),  # 深拷贝 s1 的内容
                lambda: s1._data_ptr(),  # 获取 s1 数据的指针
                lambda: s1._nbytes(),  # 获取 s1 占用的字节数
                lambda: t1._typed_storage(),  # 获取 t1 的类型化存储
            ]

        # 遍历 funcs 列表中的每个函数，并检查是否会产生 TypedStorage 废弃警告
        for f in funcs:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', "TypedStorage is deprecated")
                f()

    # 使用装饰器 skipIfTorchInductor("FIXME") 标记的测试函数，跳过 Torch Inductor FIXME 情况下的测试
    @skipIfTorchInductor("FIXME")
    # 定义一个测试函数，用于测试 TypedStorage 的弃用警告
    def test_typed_storage_deprecation_warning(self):
        # 创建一个大小为 10 的 FloatStorage 对象 s0
        s0 = torch.FloatStorage(10)
        # 函数列表，包含多个 lambda 表达式，每个表达式都是调用 TypedStorage 相关方法的函数
        funcs = [
            lambda: torch.FloatStorage(),  # 创建一个未初始化的 FloatStorage 对象
            lambda: torch.FloatStorage.dtype,  # 返回 FloatStorage 类型的 dtype
            lambda: s0.fill_(0),  # 使用值 0 填充 FloatStorage 对象 s0
            lambda: s0.is_cuda,  # 检查 FloatStorage 对象 s0 是否在 CUDA 设备上
            lambda: s0.untyped(),  # 获取 FloatStorage 对象 s0 的未类型化版本
            lambda: len(s0),  # 获取 FloatStorage 对象 s0 的长度
            lambda: s0[0],  # 获取 FloatStorage 对象 s0 的第一个元素
        ]

        # 如果 CUDA 可用，创建一个大小为 10 的 CUDA FloatStorage 对象 s1，并添加对应的 lambda 函数
        if torch.cuda.is_available():
            s1 = torch.cuda.FloatStorage(10)
            funcs += [
                lambda: torch.cuda.FloatStorage(),  # 创建一个未初始化的 CUDA FloatStorage 对象
                lambda: torch.cuda.FloatStorage.dtype,  # 返回 CUDA FloatStorage 类型的 dtype
                lambda: s1.fill_(0),  # 使用值 0 填充 CUDA FloatStorage 对象 s1
                lambda: s1.is_cuda,  # 检查 CUDA FloatStorage 对象 s1 是否在 CUDA 设备上
                lambda: s1.untyped(),  # 获取 CUDA FloatStorage 对象 s1 的未类型化版本
                lambda: len(s1),  # 获取 CUDA FloatStorage 对象 s1 的长度
                lambda: s1[0],  # 获取 CUDA FloatStorage 对象 s1 的第一个元素
            ]

        # 遍历 funcs 列表中的每个函数，并使用 AlwaysWarnTypedStorageRemoval 类确保 TypedStorage 的警告总是发出
        for f in funcs:
            with AlwaysWarnTypedStorageRemoval(True):  # 设置警告为总是发出
                with warnings.catch_warnings(record=True) as w:
                    warnings.resetwarnings()  # 重置警告状态
                    f()  # 调用函数 f
                    # 断言警告的数量为 1
                    self.assertEqual(len(w), 1, msg=str([str(a) for a in w]))
                    warning = w[0].message  # 获取第一个警告信息
                    # 断言警告类型为 DeprecationWarning
                    self.assertTrue(isinstance(warning, DeprecationWarning))
                    # 断言警告信息匹配 TypedStorage is deprecated
                    self.assertTrue(re.search(
                        '^TypedStorage is deprecated',
                        str(warning)))

        # 测试默认情况下只会触发第一个警告
        torch.storage._reset_warn_typed_storage_removal()
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            torch.FloatStorage()
            torch.randn(10).storage()
            self.assertEqual(len(w), 1, msg=str([str(a) for a in w]))
            warning = w[0].message
            # 断言警告信息匹配 TypedStorage is deprecated
            self.assertTrue(re.search(
                '^TypedStorage is deprecated',
                str(warning)))
            # 检查警告堆栈中的代码行
            with open(w[0].filename, encoding="utf-8") as f:
                code_line = f.readlines()[w[0].lineno - 1]
            # 断言警告的代码行包含 'torch.FloatStorage()'
            self.assertTrue(re.search(re.escape('torch.FloatStorage()'), code_line))

        # 检查如果过去已经发出警告，则不会再次发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.resetwarnings()
            torch.FloatStorage()
            torch.randn(10).storage()
            # 断言没有新的警告被记录
            self.assertEqual(len(w), 0, msg=str([str(a) for a in w]))
    def test_from_file(self):
        # 定义一个内部函数，用于测试从文件中读取数据并进行断言
        def assert_with_filename(filename):
            # 设置存储空间的大小为10000
            size = 10000
            # 从文件中创建一个浮点数存储对象，并指定是否共享数据，以及存储空间大小
            s1 = torch.FloatStorage.from_file(filename, True, size)
            # 从 s1 创建一个浮点数张量，并复制随机生成的数据到 t1
            t1 = torch.FloatTensor(s1).copy_(torch.randn(size))
            # 断言两个张量的数据指针是否相同
            self.assertEqual(s1.data_ptr(), torch.FloatTensor(s1).data_ptr())

            # 检查映射关系
            # 再次从文件中创建一个浮点数存储对象 s2，并创建对应的张量 t2
            s2 = torch.FloatStorage.from_file(filename, True, size)
            t2 = torch.FloatTensor(s2)
            # 断言 t1 和 t2 的值是否相等，不考虑绝对或相对误差
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # 检查从 t1 到 t2 的变化
            rnum = random.uniform(-1, 1)
            t1.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # 检查从 t2 到 t1 的变化
            rnum = random.uniform(-1, 1)
            t2.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # 释放张量
            del s1, t1, s2, t2

        # 使用临时文件名调用 assert_with_filename 函数进行测试
        with TemporaryFileName() as fname:
            assert_with_filename(fname)

        # 如果文件系统支持 UTF-8 编码
        if IS_FILESYSTEM_UTF8_ENCODING:
            # 使用临时目录名和文件名进行测试
            with TemporaryDirectoryName(suffix='\u4e2d\u6587') as dname, TemporaryFileName(dir=dname) as fname:
                assert_with_filename(fname)

    def test_torch_from_file(self):
        # 定义一个内部函数，用于测试从文件中读取张量数据并进行断言
        def assert_with_filename(filename):
            # 设置存储空间的大小为10000
            size = 10000
            # 从文件中创建一个浮点数张量，并复制随机生成的数据到 t1
            s1 = torch.from_file(filename, True, size, dtype=torch.float)
            t1 = torch.FloatTensor(s1).copy_(torch.randn(size))

            # 检查映射关系
            # 再次从文件中创建一个浮点数张量 s2，并创建对应的张量 t2
            s2 = torch.from_file(filename, True, size, dtype=torch.float)
            t2 = torch.FloatTensor(s2)
            # 断言 t1 和 t2 的值是否相等，不考虑绝对或相对误差
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # 检查从 t1 到 t2 的变化
            rnum = random.uniform(-1, 1)
            t1.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # 检查从 t2 到 t1 的变化
            rnum = random.uniform(-1, 1)
            t2.fill_(rnum)
            self.assertEqual(t1, t2, atol=0, rtol=0)

            # 释放张量
            del s1, t1, s2, t2

        # 使用临时文件名调用 assert_with_filename 函数进行测试
        with TemporaryFileName() as fname:
            assert_with_filename(fname)

        # 如果文件系统支持 UTF-8 编码
        if IS_FILESYSTEM_UTF8_ENCODING:
            # 使用临时目录名和文件名进行测试
            with TemporaryDirectoryName(suffix='\u4e2d\u6587') as dname, TemporaryFileName(dir=dname) as fname:
                assert_with_filename(fname)
# 测试一个复杂张量的字符串表示是否符合预期

tensor([4.0000+0.j,    inf+0.j, 1.5000+infj,   -inf+4.j, 0.0000+0.j,    nan+infj,
        3.0000+nanj])'''
self.assertExpectedInline(str(y), expected_str)

# 测试张量的数据类型是否正确
with set_default_dtype(torch.float):
    # 创建一个张量，使用指定的浮点数精度，并检查其字符串表示是否符合预期
    x = torch.tensor([1e-324, 1e-323, 1e-322, 1e307, 1e308, 1e309], dtype=torch.float64)
    self.assertEqual(x.__repr__(), str(x))
    expected_str = '''\
tensor([ 0.0000e+00, 9.8813e-324, 9.8813e-323, 1.0000e+307, 1.0000e+308,
                inf], dtype=torch.float64)'''
    # 检查张量的字符串表示是否符合预期
    self.assertExpectedInline(str(x), expected_str)

# 测试修改默认数据类型后的张量字符串表示
with set_default_dtype(torch.float64):
    # 检查张量的字符串表示是否符合预期
    self.assertEqual(x.__repr__(), str(x))
    expected_str = '''\
# test integral floats and special values in a tensor
# 测试张量中的整数浮点数和特殊值
x = torch.tensor([ 0.0000e+00, 9.8813e-324, 9.8813e-323, 1.0000e+307, 1.0000e+308,
                inf])
self.assertExpectedInline(str(x), expected_str)

# test summary
# 测试张量的摘要信息
x = torch.zeros(10000)
self.assertEqual(x.__repr__(), str(x))
self.assertExpectedInline(str(x), '''tensor([0., 0., 0.,  ..., 0., 0., 0.])''')

# test internal summary function
# 测试内部摘要函数
x = torch.rand(1, 20, 5, 30)
summary = torch._tensor_str.get_summarized_data(x)
self.assertEqual(summary.shape, (1, 6, 5, 6))
first_and_last = [0, 1, 2, -3, -2, -1]
self.assertEqual(summary, x[:, first_and_last][..., first_and_last])

# test device
# 测试设备
if torch.cuda.is_available():
    x = torch.tensor([123], device='cuda:0')
    self.assertEqual(x.__repr__(), str(x))
    self.assertExpectedInline(str(x), '''tensor([123], device='cuda:0')''')

    # test changing default to cuda
    # 测试将默认张量类型更改为 cuda
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    self.assertEqual(x.__repr__(), str(x))
    self.assertExpectedInline(str(x), '''tensor([123])''')

    # test printing a tensor on a different gpu than current one.
    # 在与当前不同的 GPU 上打印张量
    if torch.cuda.device_count() >= 2:
        with torch.cuda.device(1):
            self.assertEqual(x.__repr__(), str(x))
            self.assertExpectedInline(str(x), '''tensor([123], device='cuda:0')''')

    # test printing cpu tensor when default device is cuda
    # 当默认设备是 cuda 时，测试打印 CPU 张量
    y = torch.tensor([123], device='cpu')
    self.assertEqual(y.__repr__(), str(y))
    self.assertExpectedInline(str(y), '''tensor([123], device='cpu')''')
torch.set_default_tensor_type(default_type)

# test integral floats and requires_grad
# 测试整数浮点数和 requires_grad
x = torch.tensor([123.], requires_grad=True)
self.assertEqual(x.__repr__(), str(x))
self.assertExpectedInline(str(x), '''tensor([123.], requires_grad=True)''')

# test non-contiguous print
# 测试非连续打印
# sliced tensor should have > PRINT_OPTS.threshold elements
# 切片张量应具有大于 PRINT_OPTS.threshold 的元素
x = torch.ones(100, 2, 2, 10)
y = x.as_strided(size=(100, 2, 10), stride=(2 * 2 * 10, 2 * 10, 1))
self.assertEqual(str(y), y.__repr__())
expected_str = '''\
tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        ...,

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]]])\
'''
# 断言预期的内联字符串与实际输出是否一致
self.assertExpectedInline(str(y), expected_str)

# 创建一个大小为 (100, 2, 2, 10) 的复数张量 x，所有元素均为 1 + 1j
x = torch.ones(100, 2, 2, 10) * (1 + 1j)

# 使用 as_strided 方法从张量 x 生成一个新的张量 y，大小为 (100, 2, 10)，设定步长为 (2 * 2 * 10, 2 * 10, 1)
y = x.as_strided(size=(100, 2, 10), stride=(2 * 2 * 10, 2 * 10, 1))

# 断言张量 y 的字符串表示与其 repr() 方法输出是否一致
self.assertEqual(str(y), y.__repr__())

# 预期的字符串表示，包含一个复数张量的字符串表示
expected_str = '''\
tensor([[[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        ...,

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]],

        [[1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j],
         [1.+1.j, 1.+1.j, 1.+1.j,  ..., 1.+1.j, 1.+1.j, 1.+1.j]]])
'''
        self.assertExpectedInline(str(y), expected_str)
        # 调用断言方法，比较 y 的字符串表示和期望字符串是否相等

        # test print 0-dim tensor: there's no 0-dim in Numpy, we match arrayprint style
        # 测试打印零维张量：在 NumPy 中不存在零维，这里我们匹配数组打印样式
        x = torch.tensor(0.00002)
        # 创建一个包含单个浮点数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor(2.0000e-05)''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # test print boolean tensor
        # 测试打印布尔张量
        x = torch.tensor([True])
        # 创建一个包含布尔值的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([True])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        x = torch.tensor(True)
        # 创建一个包含单个布尔值的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor(True)''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test print float in sci_mode when min < 0.0001.
        # [Numpy] 测试在科学计数法模式下打印浮点数，当最小值小于 0.0001 时
        x = torch.tensor([0.00002])
        # 创建一个包含单个浮点数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([2.0000e-05])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test print complex in sci_mode when real_min < 0.0001 and (or) imag_min < 0.0001.
        # [Numpy] 测试在科学计数法模式下打印复数，当实部最小值小于 0.0001 并且（或）虚部最小值小于 0.0001 时
        x = torch.tensor([0.00002]) * (1 + 1j)
        # 创建一个包含复数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([2.0000e-05+2.0000e-05j])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test print float in sci_mode when max > 1e8.
        # [Numpy] 测试在科学计数法模式下打印浮点数，当最大值大于 1e8 时
        # TODO: Pytorch uses fixed precision to print, while Numpy uses dragon4_scientific
        # to do automatic trimming and padding.
        # 注意：PyTorch 使用固定精度进行打印，而 NumPy 使用 dragon4_scientific 进行自动修剪和填充。
        x = torch.tensor([123456789.])
        # 创建一个包含单个浮点数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([1.2346e+08])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test print float in sci_mode when max / min > 1000.
        # [Numpy] 测试在科学计数法模式下打印浮点数，当最大值除以最小值大于 1000 时
        x = torch.tensor([0.01, 11])
        # 创建一个包含多个浮点数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([1.0000e-02, 1.1000e+01])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test print int max / min > 1000, no sci_mode
        # [Numpy] 测试打印整数，当最大值除以最小值大于 1000 时，不使用科学计数法模式
        x = torch.tensor([1, 1010])
        # 创建一个包含多个整数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([   1, 1010])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test print int > 1e8, no sci_mode
        # [Numpy] 测试打印整数，大于 1e8，不使用科学计数法模式
        x = torch.tensor([1000000000])  # 1e9
        # 创建一个包含单个整数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([1000000000])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test printing float in int_mode
        # [Numpy] 测试以整数模式打印浮点数
        x = torch.tensor([1., 1000.])
        # 创建一个包含多个浮点数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([   1., 1000.])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等

        # [Numpy] test printing float in int_mode in sci format when max / min > 1000.
        # [Numpy] 测试以整数模式打印浮点数，当最大值除以最小值大于 1000 时，使用科学计数法格式
        x = torch.tensor([1., 1010.])
        # 创建一个包含多个浮点数的张量 x
        self.assertEqual(x.__repr__(), str(x))
        # 断言张量 x 的字符串表示与 str(x) 的结果相等
        self.assertExpectedInline(str(x), '''tensor([1.0000e+00, 1.0100e+03])''')
        # 调用断言方法，比较张量 x 的字符串表示与期望的字符串是否相等
    # 测试方法，用于验证存储对象的大小计算是否正确
    def test_sizeof(self) -> None:
        # 创建大小为 0 的随机张量，并获取其存储对象的大小
        sizeof_empty = torch.randn(0).storage().__sizeof__()
        # 创建大小为 10 的随机张量，并获取其存储对象的大小
        sizeof_10 = torch.randn(10).storage().__sizeof__()
        # 创建大小为 100 的随机张量，并获取其存储对象的大小
        sizeof_100 = torch.randn(100).storage().__sizeof__()
        # 断言：验证计算得到的扩展比例是否为 10
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        # 断言：验证计算得到的余数是否为 0
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

        # 转换为 uint8 类型后，再次获取存储对象的大小
        sizeof_empty = torch.randn(0).to(torch.uint8).storage().__sizeof__()
        sizeof_10 = torch.randn(10).to(torch.uint8).storage().__sizeof__()
        sizeof_100 = torch.randn(100).to(torch.uint8).storage().__sizeof__()
        # 断言：验证计算得到的扩展比例是否为 10
        self.assertEqual((sizeof_100 - sizeof_empty) // (sizeof_10 - sizeof_empty), 10)
        # 断言：验证计算得到的余数是否为 0
        self.assertEqual((sizeof_100 - sizeof_empty) % (sizeof_10 - sizeof_empty), 0)

    # TorchDynamo 不适合进行此测试时跳过
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # 测试方法，验证存储对象的可调整性
    def test_resizable(self) -> None:
        # 创建大小为 5 的随机张量
        x = torch.randn(5)
        # 断言：验证其存储对象是否可调整
        self.assertTrue(x.storage().resizable())
        # 转换为 numpy 数组
        x.numpy()
        # 断言：验证其存储对象是否不可调整
        self.assertFalse(x.storage().resizable())

    # 测试方法，验证张量的迭代行为
    def test_iter(self) -> None:
        # 创建大小为 5x5 的随机张量
        x = torch.randn(5, 5)
        # 遍历张量的每一个子张量，并验证其与原张量索引处的值相等
        for i, sub in enumerate(x):
            self.assertEqual(sub, x[i])  # noqa: PLR1736

        # 创建空张量
        x = torch.tensor([])
        # 断言：将空张量转换为列表应为空列表
        self.assertEqual(list(x), [])

    # 测试方法，验证 torch.autograd.Variable 的创建和操作
    def test_new(self) -> None:
        # 创建空张量，并将其封装为 torch.autograd.Variable
        x = torch.autograd.Variable(torch.tensor([]))
        # 创建大小为 4x4 的随机张量，并封装为 torch.autograd.Variable
        y = torch.autograd.Variable(torch.randn(4, 4))
        # 创建包含 [1, 2, 3] 的整数张量，并封装为 torch.autograd.Variable
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))

        # 断言：验证创建的新张量形状为 [0]
        self.assertEqual(x.new().shape, [0])
        # 断言：验证创建的新张量与原张量相等
        self.assertEqual(x.new(), x)
        # 断言：验证创建的新张量形状为 [1, 2]
        self.assertEqual(x.new(1, 2).shape, [1, 2])
        # 断言：验证创建的新张量形状为 [3, 4]
        self.assertEqual(x.new(torch.Size([3, 4])).shape, [3, 4])
        # 断言：验证创建的新张量形状为 [2]
        self.assertEqual(x.new([3, 4]).shape, [2])
        # 断言：验证创建的新张量转换为列表后应为 [3, 4]
        self.assertEqual(x.new([3, 4]).tolist(), [3, 4])
        # 断言：验证创建的新张量转换为列表后应为 [3, 4]
        self.assertEqual(x.new((3, 4)).tolist(), [3, 4])
        # 断言：验证创建的新张量转换为列表后应为 [3, 4]
        self.assertEqual(x.new([np.int32(3), np.float64(4)]).tolist(), [3, 4])
        # 断言：验证创建的新张量转换为列表后应为 [3, 4]
        self.assertEqual(x.new(np.array((3, 4))).tolist(), [3, 4])
        # 断言：验证创建的新张量转换为列表后应为 [3, 4]
        self.assertEqual(x.new([z[2], z[0] + 3]).tolist(), [3, 4])
        # 断言：验证创建的新张量形状为 [3, 4]
        self.assertEqual(x.new(size=(3, 4)).shape, [3, 4])
        # 断言：验证创建的新张量形状为 [0]
        self.assertEqual(x.new(()).shape, [0])
        # 断言：验证创建的新张量的数据指针与 y 的数据指针相等
        self.assertEqual(x.new(y.storage()).data_ptr(), y.data_ptr())
        # 断言：验证创建的新张量的数据指针与 y 的数据指针相等
        self.assertEqual(x.new(y).data_ptr(), y.data_ptr())
        # 断言：验证创建的新张量与 y 不是同一对象
        self.assertIsNot(x.new(y), y)

        # 引发 TypeError 异常，因为 z 不是有效的张量形状参数
        self.assertRaises(TypeError, lambda: x.new(z))
        # 引发 RuntimeError 异常，因为 z.storage() 不是有效的张量参数
        self.assertRaises(RuntimeError, lambda: x.new(z.storage()))

    # 如果 PYTORCH_CUDA_MEMCHECK 为真，则跳过此测试
    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    # 测试是否可以在 CPU 上使用 pin_memory
    def test_pin_memory(self):
        # 创建一个形状为 (3, 5) 的随机张量
        x = torch.randn(3, 5)
        # 断言张量 x 没有被 pin
        self.assertFalse(x.is_pinned())
        # 如果 CUDA 可用，预期会抛出 RuntimeError 异常
        if not torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: x.pin_memory())
        else:
            # 在内存中 pin 住张量 x
            pinned = x.pin_memory()
            # 断言张量已经被 pin
            self.assertTrue(pinned.is_pinned())
            # 断言 pin 后的张量与原张量相等
            self.assertEqual(pinned, x)
            # 断言 pin 后的张量与原张量的数据指针不同
            self.assertNotEqual(pinned.data_ptr(), x.data_ptr())
            # 测试对已经 pin 的张量再次 pin，预期不会改变
            self.assertIs(pinned, pinned.pin_memory())
            # 断言 pin 后的张量与再次 pin 后的数据指针相同
            self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())

    # 测试错误消息类型转换
    def test_error_msg_type_translation(self):
        with self.assertRaisesRegex(
                RuntimeError,
                # 错误消息同时包含 Double 和 Long
                '(?=.*Double)(?=.*Long)'):

            # 使用 LongTensor 输入但使用 DoubleTensor 权重调用模型
            input = torch.zeros(1, 1, 1, 6, dtype=torch.long)
            weight = torch.nn.Parameter(torch.zeros(1, 1, 1, 3, dtype=torch.double))
            model = torch.nn.Conv2d(1, 1, (1, 3), stride=1, padding=0, bias=False)
            model.weight = weight
            out = model(input)

    # 测试 apply_ 方法
    def test_apply(self):
        # 创建一个张量 x，包含元素 1 到 5
        x = torch.arange(1, 6)
        # 对 x 应用 lambda 函数加倍每个元素
        res = x.clone().apply_(lambda k: k + k)
        # 断言应用后的结果与 x 的每个元素乘以 2 相等
        self.assertEqual(res, x * 2)
        # 测试当 lambda 函数返回字符串时，应抛出 TypeError 异常
        self.assertRaises(TypeError, lambda: x.apply_(lambda k: "str"))

    # 测试 map_ 方法
    def test_map(self):
        # 创建一个随机 Variable 张量 x
        x = torch.autograd.Variable(torch.randn(3, 3))
        # 创建一个随机 Variable 张量 y
        y = torch.autograd.Variable(torch.randn(3))
        # 复制 x 到 res
        res = x.clone()
        # 使用 y 和 lambda 函数对 res 应用 map_ 方法
        res.map_(y, lambda a, b: a + b)
        # 断言应用 map_ 后的结果与 x 加上 y 相等
        self.assertEqual(res, x + y)
        # 测试当传入非函数字符串时，应抛出 TypeError 异常
        self.assertRaisesRegex(TypeError, "not callable", lambda: res.map_(y, "str"))

    # 测试 map2_ 方法
    def test_map2(self):
        # 创建一个随机 Variable 张量 x
        x = torch.autograd.Variable(torch.randn(3, 3))
        # 创建一个随机 Variable 张量 y
        y = torch.autograd.Variable(torch.randn(3))
        # 创建一个随机 Variable 张量 z
        z = torch.autograd.Variable(torch.randn(1, 3))
        # 复制 x 到 res
        res = x.clone()
        # 使用 y、z 和 lambda 函数对 res 应用 map2_ 方法
        res.map2_(y, z, lambda a, b, c: a + b * c)
        # 断言应用 map2_ 后的结果与 x 加上 y 乘以 z 相等
        self.assertEqual(res, x + y * z)
        # 设置 z 为 requires_grad=True，预期抛出 RuntimeError 异常
        z.requires_grad = True
        self.assertRaisesRegex(
            RuntimeError, "requires grad",
            lambda: res.map2_(y, z, lambda a, b, c: a + b * c))

    # 测试 torch.Size 类
    def test_Size(self):
        # 创建一个 torch.Size 对象 x，包含维度 [1, 2, 3]
        x = torch.Size([1, 2, 3])
        # 断言 x 是 tuple 的实例
        self.assertIsInstance(x, tuple)
        # 断言 x 的第一个元素是 1
        self.assertEqual(x[0], 1)
        # 断言 x 的第二个元素是 2
        self.assertEqual(x[1], 2)
        # 断言 x 的第三个元素是 3
        self.assertEqual(x[2], 3)
        # 断言 x 的长度是 3
        self.assertEqual(len(x), 3)
        # 测试当尝试用 torch.ones(3) 创建 torch.Size 时，应抛出 TypeError 异常
        self.assertRaises(TypeError, lambda: torch.Size(torch.ones(3)))

        # 断言对 x 进行数学运算后返回的对象类型仍为 torch.Size
        self.assertIsInstance(x * 2, torch.Size)
        # 断言对 x 进行切片后返回的对象类型仍为 torch.Size
        self.assertIsInstance(x[:-1], torch.Size)
        # 断言对 x 进行加法运算后返回的对象类型仍为 torch.Size
        self.assertIsInstance(x + x, torch.Size)

    # 测试 torch.Size 类的标量输入
    def test_Size_scalar(self):
        # 创建标量张量 three 和 two
        three = torch.tensor(3)
        two = torch.tensor(2)
        # 创建一个包含标量 three 和 two 的 torch.Size 对象 x
        x = torch.Size([0, 1, two, three, 4])
        # 断言 x 的各元素值正确
        for i in range(1, 5):
            self.assertEqual(x[i], i)
    # 测试 torch.Size 的迭代功能
    def test_Size_iter(self):
        # 使用两种不同的迭代器生成器，分别进行测试
        for sizes in [iter([1, 2, 3, 4, 5]), range(1, 6)]:
            # 创建 torch.Size 对象 x
            x = torch.Size(sizes)
            # 遍历 torch.Size 对象，验证每个索引对应的值是否符合预期
            for i in range(0, 5):
                self.assertEqual(x[i], i + 1)

    # 测试当张量不是二维时，t() 和 t_() 方法是否会引发 RuntimeError
    def test_t_not_2d_error(self):
        # 调用 t() 方法应引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.randn(2, 3, 4).t())
        # 调用 t_() 方法应引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.randn(2, 3, 4).t_())

    # 暂时跳过该测试，因为它会影响所有测试结果
    @unittest.skipIf(True, "flush_denormal not supported")
    def test_set_flush_denormal(self):
        # 定义浮点数和双精度浮点数的极小值
        tiny_float = 1e-42
        tiny_double = 1e-320
        # 创建浮点数张量和双精度浮点数张量
        float_tensor = torch.FloatTensor([1.0, tiny_float])
        double_tensor = torch.DoubleTensor([1.0, tiny_float, tiny_double])

        # 验证初始状态下张量中元素的值
        self.assertEqual(float_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(float_tensor[1], tiny_float, atol=tiny_float / 16, rtol=0)
        self.assertEqual(double_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[1], tiny_float, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[2], tiny_double, atol=0.0, rtol=0)

        # 启用 flush_denormal 模式
        torch.set_flush_denormal(True)
        # 验证启用后张量中元素的值
        self.assertEqual(float_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(float_tensor[1], 0.0, atol=0.0, rtol=0)  # 将 tiny_float 转换为零
        self.assertEqual(double_tensor[0], 1.0, atol=0.0, rtol=0)
        self.assertEqual(double_tensor[1], tiny_float, atol=0.0, rtol=0)  # 双精度类型中的 tiny_float 不会被转换为零
        self.assertEqual(double_tensor[2], 0.0, atol=0.0, rtol=0)  # 将 tiny_double 转换为零
        # 恢复 flush_denormal 模式为禁用
        torch.set_flush_denormal(False)

    # 测试显示 Torch 配置信息
    def test_show_config(self):
        # 调用显示配置信息的方法，验证其不会导致崩溃
        torch.__config__.show()

    # 如果是在 FBCODE 环境中，跳过该测试（CXX_FLAGS 仅适用于 OSS 构建）
    @unittest.skipIf(IS_FBCODE, "CXX_FLAGS is only for OSS build.")
    def test_cxx_flags(self):
        # 调用获取 C++ 编译标志的方法
        torch.__config__._cxx_flags()

    # 测试并行信息
    def test_parallel_info(self):
        # 调用获取并行信息的方法
        torch.__config__.parallel_info()

    # 测试获取 CPU 能力信息
    def test_get_cpu_capability(self):
        # 此方法主要为 torchvision 的 resize 提供支持
        torch.backends.cpu.get_cpu_capability()

        # 确保该方法在 TorchScript 中可用，以支持 torchvision 的 resize
        torch.jit.script(torch.backends.cpu.get_cpu_capability)

    # 一个耗时测试的示例，用于确保 slowTest 装饰器正常工作
    @slowTest
    def test_slow_test(self):
        # 仅进行一个简单的 smoke test，验证测试装饰器是否正常工作
        pass
    def test_is_nonzero(self):
        # 测试空张量抛出 RuntimeError 异常，指定错误消息
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch.tensor([]).is_nonzero()
        # 测试包含多个值的张量抛出 RuntimeError 异常，指定错误消息
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch.tensor([0, 0]).is_nonzero()
        # 测试零值张量返回 False
        self.assertFalse(torch.tensor(0).is_nonzero())
        # 测试非零值张量返回 True
        self.assertTrue(torch.tensor(1).is_nonzero())
        # 测试包含单个零值的张量返回 False
        self.assertFalse(torch.tensor([0]).is_nonzero())
        # 测试包含单个非零值的张量返回 True
        self.assertTrue(torch.tensor([1]).is_nonzero())
        # 测试包含单个嵌套零值的张量返回 False
        self.assertFalse(torch.tensor([[0]]).is_nonzero())
        # 测试包含单个嵌套非零值的张量返回 True
        self.assertTrue(torch.tensor([[1]]).is_nonzero())
        # 测试非整数值张量返回 True
        self.assertTrue(torch.tensor(0.1).is_nonzero())
        # 测试负非零值张量返回 True
        self.assertTrue(torch.tensor(-0.1).is_nonzero())
        # 测试零浮点值张量返回 False
        self.assertFalse(torch.tensor(0.0).is_nonzero())
        # 测试布尔 True 张量返回 True
        self.assertTrue(torch.tensor(True).is_nonzero())
        # 测试布尔 False 张量返回 False
        self.assertFalse(torch.tensor(False).is_nonzero())
        # 测试复数零值张量返回 False
        self.assertFalse(torch.tensor(0 + 0j).is_nonzero())
        # 测试复数非零值张量返回 True
        self.assertTrue(torch.tensor(0 + 0.1j).is_nonzero())

    def test_assert_async(self):
        # 测试空张量抛出 RuntimeError 异常，指定错误消息
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch._assert_async(torch.tensor([]))
        # 测试包含多个值的张量抛出 RuntimeError 异常，指定错误消息
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous"):
            torch._assert_async(torch.tensor([0, 0]))
        # 测试期望单个非零值张量，但获取到零值时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0))
        # 测试期望单个非零值张量，获取到非零值时不抛出异常
        torch._assert_async(torch.tensor(1))
        # 测试期望单个非零值张量，获取到非零浮点值时不抛出异常
        torch._assert_async(torch.tensor(0.1))
        torch._assert_async(torch.tensor(-0.1))
        # 测试期望单个非零值张量，但获取到零浮点值时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0.0))
        # 测试期望单个非零值张量，获取到布尔 True 时不抛出异常
        torch._assert_async(torch.tensor(True))
        # 测试期望单个非零值张量，但获取到布尔 False 时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(False))
        # 测试期望单个非零值张量，获取到复数非零值时不抛出异常
        torch._assert_async(torch.tensor(0 + 0.1j))
        # 测试期望单个非零值张量，但获取到复数零值时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Expected Tensor with single nonzero value, but got zero"):
            torch._assert_async(torch.tensor(0 + 0j))

    # NB: 我们必须不是使用 CUDA 构建的；如果我们是使用 CUDA 构建的，但没有可用的 CUDA，我们会得到不同的错误。
    @unittest.skipIf(torch.backends.cuda.is_built() or IS_SANDCASTLE, "CUDA is built, can't test CUDA not built error")
    # 测试CUDA未构建的情况
    def test_cuda_not_built(self):
        msg = "Torch not compiled with CUDA enabled"
        # 断言抛出带有指定消息的AssertionError异常，验证CUDA未启用
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.cuda.current_device())
        # 断言抛出带有指定消息的AssertionError异常，验证在CUDA设备上创建张量失败
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1], device="cuda"))
        # 断言抛出带有指定消息的AssertionError异常，验证在CUDA设备上创建张量失败
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1]).cuda())
        # 断言抛出带有指定消息的TypeError异常，验证在CUDA设备上创建浮点张量失败
        self.assertRaisesRegex(TypeError, msg, lambda: torch.cuda.FloatTensor())
        # 断言抛出带有指定消息的TypeError异常，验证设置默认张量类型为CUDA浮点型失败
        self.assertRaisesRegex(TypeError, msg, lambda: torch.set_default_tensor_type(torch.cuda.FloatTensor))
        # 断言抛出带有指定消息的AssertionError异常，验证在CUDA设备上创建张量失败
        self.assertRaisesRegex(AssertionError, msg, lambda: torch.tensor([1]).to(device="cuda"))

    # 测试张量是否具有内部重叠
    def test_has_internal_overlap(self):
        OVERLAP_NO = 0
        OVERLAP_YES = 1
        OVERLAP_TOO_HARD = 2

        # 检查连续张量
        a = torch.randn(3, 3)
        self.assertEqual(torch._debug_has_internal_overlap(a), OVERLAP_NO)

        # 检查零步长的张量
        b = torch.randn(1, 3)
        b_expanded = b.expand(4, 3)
        self.assertEqual(torch._debug_has_internal_overlap(b_expanded), OVERLAP_YES)

        # 检查零步长的张量，尺寸为1的轴，在非连续存储中（gh-33812）
        c = torch.randn(10).as_strided([2, 1, 5], [1, 0, 2])
        self.assertEqual(torch._debug_has_internal_overlap(c), OVERLAP_NO)
        c = torch.randn(2, 1, 10)[::2].as_strided((2, 1, 5), (10, 0, 2))
        self.assertEqual(torch._debug_has_internal_overlap(c), OVERLAP_TOO_HARD)

    # 测试允许张量元数据更改
    def test_allow_tensor_metadata_change(self):
        a = torch.ones(2, 3)
        # 允许从detach()创建的视图张量上进行元数据更改。

    # 测试内存格式
    def test_memory_format(self):
        def test_helper(x, memory_format):
            # 使用指定的内存格式创建连续张量，并进行相关断言验证
            y = x.contiguous(memory_format=memory_format)
            self.assertFalse(y.is_contiguous())
            self.assertTrue(y.is_contiguous(memory_format=memory_format))
            self.assertEqual(y, x)

        test_helper(torch.randn(4, 3, 8, 8), torch.channels_last)
        test_helper(torch.randn(4, 3, 8, 8, 8), torch.channels_last_3d)

    # 测试内存格式，如果已满足条件，则contiguous()返回相同张量
    def test_memory_format_contiguous_returns_same_tensor_if_already_satisfies(self):
        def test_helper(x, memory_format):
            # 对已满足条件的张量进行contiguous()操作，应返回相同张量，并进行断言验证
            alias = x.contiguous(memory_format=memory_format)
            alias.fill_(7)
            self.assertEqual(x, alias)

        test_helper(torch.randn(4, 8, 8, 3).permute(0, 3, 1, 2), torch.channels_last)
        test_helper(torch.randn(4, 8, 8, 8, 3).permute(0, 4, 1, 2, 3), torch.channels_last_3d)

    # 测试空的内存格式
    def test_memory_format_empty(self):
        def test_helper(dim1, dim2, memory_format):
            # 断言运行时异常，不能创建空的指定内存格式的张量
            with self.assertRaises(RuntimeError):
                x = torch.empty(dim1, memory_format=memory_format)
            # 创建指定内存格式的非空张量，并验证其连续性
            x = torch.empty(dim2, memory_format=memory_format)
            self.assertTrue(x.is_contiguous(memory_format=memory_format))

        test_helper((3, 3), (3, 3, 3, 3), torch.channels_last)
        test_helper((3, 3, 3), (3, 3, 3, 3, 3), torch.channels_last_3d)
    def test_dim_order(self):
        # 定义一个形状为 (2, 3, 5, 7) 的张量
        shape = (2, 3, 5, 7)

        # 创建一个未初始化的张量 t，形状为 shape
        t = torch.empty(shape)
        # 断言 t 的维度顺序应为 (0, 1, 2, 3)，使用元组来表示
        self.assertSequenceEqual(t.dim_order(), (0, 1, 2, 3), seq_type=tuple)
        # transpose 操作并不实际改变底层物理内存的排列方式
        # 因此预期 dim_order 应该反映出这种变化（比如步长）
        self.assertSequenceEqual(t.transpose(0, 1).dim_order(), (1, 0, 2, 3))

        # 创建一个带有 channels_last 内存格式的未初始化张量 t
        t = torch.empty(shape, memory_format=torch.channels_last)
        # 断言 t 的维度顺序应为 (0, 2, 3, 1)
        self.assertSequenceEqual(t.dim_order(), (0, 2, 3, 1))

        # 创建一个带有 channels_last_3d 内存格式的未初始化张量 t
        t = torch.empty((2, 3, 5, 7, 8), memory_format=torch.channels_last_3d)
        # 断言 t 的维度顺序应为 (0, 2, 3, 4, 1)
        self.assertSequenceEqual(t.dim_order(), (0, 2, 3, 4, 1))

        # 遍历 0 到 3 的排列组合
        for dim_order in itertools.permutations(range(4)):
            # 断言 torch.empty_permuted 函数对 shape 和 dim_order 的结果的 dim_order 应与 dim_order 本身相同
            self.assertSequenceEqual(
                dim_order, torch.empty_permuted(shape, dim_order).dim_order()
            )

    def test_subclass_tensors(self):
        # 当试图以 FloatTensor 为基类进行子类化时，预期引发 TypeError
        with self.assertRaisesRegex(TypeError, "type 'torch.FloatTensor' is not an acceptable base type"):
            class Foo1(torch.FloatTensor):
                pass

        # 允许以 Tensor 为基类进行子类化
        class Foo2(torch.Tensor):
            # 定义一个返回值为 5 的方法 foo
            def foo(self):
                return 5
        # 创建 Foo2 类的实例 f
        f = Foo2()
        # 断言调用 f 的 foo 方法返回 5
        self.assertEqual(f.foo(), 5)

    def test_ndim(self):
        # 创建一个形状为 (1, 2, 3) 的随机张量 a
        a = torch.randn(1, 2, 3)
        # 断言 a 的维度数为 3
        self.assertEqual(3, a.ndim)

        # 创建一个标量随机张量 b
        b = torch.randn(())
        # 断言 b 的维度数为 0
        self.assertEqual(0, b.ndim)

        # 创建一个形状为 (1, 0) 的随机张量 c
        c = torch.randn(1, 0)
        # 断言 c 的维度数为 2
        self.assertEqual(2, c.ndim)

    def test_nbytes(self):
        # 创建一个形状为 (1, 2, 3)、数据类型为 torch.float64 的随机张量 a
        a = torch.randn(1, 2, 3, dtype=torch.float64)
        # 断言 a 的 nbytes 等于其元素数乘以元素的字节大小
        self.assertEqual(a.numel() * a.element_size(), a.nbytes)

        # 创建一个标量随机张量 b
        b = torch.randn(())
        # 断言 b 的 nbytes 等于其元素数乘以元素的字节大小
        self.assertEqual(b.numel() * b.element_size(), b.nbytes)

        # 创建一个形状为 (1, 0) 的随机张量 c
        c = torch.randn(1, 0)
        # 断言 c 的 nbytes 等于其元素数乘以元素的字节大小
        self.assertEqual(c.numel() * c.element_size(), c.nbytes)

    def test_fill_diagonal(self):
        # 创建一个形状为 (7, 3) 的随机张量 a1，并创建其副本 a2
        a1 = torch.randn(7, 3)
        a2 = a1.clone()
        v = 1
        # 使用循环将 a2 的对角线元素设为 v
        for i in range(3):
            a2[i][i] = v
        # 使用 fill_diagonal_ 方法将 a1 的对角线元素填充为 v
        a1.fill_diagonal_(v)
        # 断言 a1 等于 a2
        self.assertEqual(a1, a2)

        # 创建一个形状为 (7, 3) 的随机张量 b1，并创建其副本 b2
        b1 = torch.randn(7, 3)
        b2 = b1.clone()
        # 使用循环将 b2 的主对角线和 wrap 后的次对角线元素设为 v
        for i in range(3):
            b2[i][i] = v
            b2[i + 4][i] = v
        # 使用 fill_diagonal_ 方法将 b1 的对角线元素填充为 v，并启用 wrap 模式
        b1.fill_diagonal_(v, wrap=True)
        # 断言 b1 等于 b2
        self.assertEqual(b1, b2)

        # 创建一个形状为 (3, 3, 3) 的随机张量 c1，并创建其副本 c2
        c1 = torch.rand(3, 3, 3)
        c2 = c1.clone()
        # 使用循环将 c2 的三维对角线元素设为 v
        for i in range(3):
            c2[i][i][i] = v
        # 使用 fill_diagonal_ 方法将 c1 的对角线元素填充为 v
        c1.fill_diagonal_(v)
        # 断言 c1 等于 c2
        self.assertEqual(c1, c2)

        # 创建一个非连续张量 d1，形状为 (3, 3, 3)，选取第二维后的所有元素，并创建其副本 d2
        d1 = torch.rand(3, 3, 3)[:, 1, ...]
        d2 = d1.clone()
        # 使用循环将 d2 的二维对角线元素设为 v
        for i in range(3):
            d2[i][i] = v
        # 使用 fill_diagonal_ 方法将 d1 的对角线元素填充为 v
        d1.fill_diagonal_(v)
        # 断言 d1 等于 d2
        self.assertEqual(d1, d2)

        # 创建一个非连续张量 e1，形状为 (7, 3, 3)，选取第二维后的所有元素，并创建其副本 e2
        e1 = torch.rand(7, 3, 3)[:, 1, ...]
        e2 = e1.clone()
        # 使用循环将 e2 的主对角线和 wrap 后的次对角线元素设为 v
        for i in range(3):
            e2[i][i] = v
            e2[i + 4][i] = v
        # 使用 fill_diagonal_ 方法将 e1 的对角线元素填充为 v，并启用 wrap 模式
        e1.fill_diagonal_(v, wrap=True)
        # 断言 e1 等于 e2
        self.assertEqual(e1, e2)
    # 测试将实部和虚部设置为数字的功能
    def test_setting_real_imag_to_a_number(self):
        # 创建一个大小为 4 的随机复数张量
        x = torch.randn(4, dtype=torch.cfloat)
        # 将张量 x 的实部和虚部都设置为 0
        x.real = 0
        x.imag = 0
        # 创建一个大小为 4 的全零张量
        zeros = torch.zeros(4)
        # 断言张量 x 的实部等于全零张量
        self.assertEqual(x.real, zeros)
        # 断言张量 x 的虚部等于全零张量
        self.assertEqual(x.imag, zeros)

    # 测试在 CPU 推断下的批标准化
    def test_batch_norm_cpu_inference(self):
        # 输入数据 nchw 分别为 (2,1,1,1) 和 (2,2,2,2)
        inputs = [
            torch.tensor([[[[-0.5000]]], [[[0.5000]]]]),
            torch.tensor([
                [
                    [[-0.5000, 0.5000], [-1.0000, 1.0000]],
                    [[-0.2500, -0.5000], [0.2500, 0.5000]]
                ],
                [
                    [[0.1000, 1.0000], [1.0000, 0.1000]],
                    [[1.0000, 0.5000], [1.5000, -1.5000]]
                ]
            ])
        ]
        # 预期输出数据 nchw 分别为 (2,1,1,1) 和 (2,2,2,2)
        outputs = [
            torch.tensor([
                [[[-0.499997496604919433593750000]]],
                [[[0.499997496604919433593750000]]]]
            ),
            torch.tensor([
                [[[-0.499997496604919433593750000, 0.499997496604919433593750000],
                  [-0.999994993209838867187500000, 0.999994993209838867187500000]],
                 [[-0.249998748302459716796875000, -0.499997496604919433593750000],
                  [0.249998748302459716796875000, 0.499997496604919433593750000]]],
                [[[0.099999502301216125488281250, 0.999994993209838867187500000],
                  [0.999994993209838867187500000, 0.099999502301216125488281250]],
                 [[0.999994993209838867187500000, 0.499997496604919433593750000],
                  [1.499992489814758300781250000, -1.499992489814758300781250000]]]]
            )
        ]

        # 遍历输入数据列表
        for i in range(len(inputs)):
            # 遍历仿射参数值为 False 和 True 的情况
            for affine in [False, True]:
                # 创建一个 BatchNorm2d 模块，设置仿射参数
                m = torch.nn.BatchNorm2d(inputs[i].size()[1], 1e-05, 0.1, affine=affine)
                # 将 BatchNorm2d 设置为评估模式
                m.eval()
                # 对于连续的情况，使用 contiguous 方法
                input1 = inputs[i].contiguous()
                output1 = m(input1)
                # 对于非连续的情况，使用 permute 方法重新排列维度
                input2 = input1.permute(0, 1, 3, 2)
                output2 = m(input2).permute(0, 1, 3, 2)
                # 对于通道优先的情况，使用 contiguous 方法，并指定内存格式为通道优先
                input3 = input1.contiguous(memory_format=torch.channels_last)
                output3 = m(input3)
                # 断言 BatchNorm2d 输出与预期输出相等
                self.assertEqual(output3, outputs[i])
                self.assertEqual(output3, output1)
                self.assertEqual(output3, output2)

    # FIXME: 将这些元测试移到它们自己的测试套件/类中，或将它们分发到适当的测试套件中
    # 测试空元数据
    @skipIfTorchDynamo("Fails after Triton update, see https://github.com/pytorch/pytorch/issues/94687")
    def test_empty_meta(self):
        # 在设备 'meta' 上创建一个巨大的空张量
        x = torch.empty(2 ** 20, 2 ** 20, device='meta')
        y = torch.empty(2 ** 20, device='meta')
        # 计算两个张量的和
        z = x + y
        # 断言结果张量的大小符合预期
        self.assertEqual(z.size(), (2 ** 20, 2 ** 20))
        # 使用 lambda 函数断言对结果张量中特定元素的访问引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: z[0][0].item())
    @skipIfTorchDynamo("Fails after Triton update, see https://github.com/pytorch/pytorch/issues/94687")
    # 装饰器，用于标记测试为在 TorchDynamo 环境下跳过的测试用例，原因是在 Triton 更新后出现问题，详见 GitHub issue 94687
    def test_format_scalar_meta(self):
        # 创建一个空的标量张量 x，设备类型为 'meta'
        x = torch.empty((), device='meta')
        # 断言格式化后的 x 是否等于其自身的字符串表示形式
        self.assertEqual(format(x), repr(x))

    def test_upsample_nearest1d_meta(self):
        # TODO: this test should be triggered by test_nn.py but right
        # now meta is not enabled (and even if it was, we are probably
        # missing too many meta functions to get through the test unmolested)
        # 此测试应该由 test_nn.py 触发，但目前 meta 模式未启用（即使启用了，我们可能也缺少太多 meta 函数以使测试顺利通过）

        # NB: Can't make the exponent too big, or it will overflow
        # signed 64-bit integer
        # 创建一个非常大的张量 x，设备类型为 'meta'，用于测试最近邻插值是否能处理大尺寸张量
        x = torch.empty(2 * 10 ** 8, 3, 2 * 10 ** 8, device='meta')
        # 对张量 x 进行尺寸放大两倍的最近邻插值操作
        z = torch.nn.functional.interpolate(x, scale_factor=2)
        # 断言插值后张量 z 的尺寸是否为 (2 * 10 ** 8, 3, 4 * 10 ** 8)
        self.assertEqual(z.size(), (2 * 10 ** 8, 3, 4 * 10 ** 8))
        # 断言尝试获取 z[0][0][0] 的标量值会触发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: z[0][0][0].item())

        # TODO: the out tests cannot be triggered by test_nn.py because
        # we don't actually do out= arguments for nn functions, so there
        # is no public API by which to get the out version
        # out= 参数在 nn 函数中未被实际使用，因此无法通过 test_nn.py 触发 out 测试，也没有公共 API 可以获取 out 版本

        # interpolate doesn't seem to support out=
        # (not sure why passing None here doesn't work? How strange...)
        # 创建一个空张量 z，用于测试最近邻一维上采样函数 upsample_nearest1d 的 out= 参数
        z = torch.empty(0, device='meta')
        # 调用 torch._C._nn.upsample_nearest1d 对 x 进行一维最近邻上采样，结果存入 z
        torch._C._nn.upsample_nearest1d(x, (4 * 10 ** 8,), 2, out=z)
        # 断言张量 z 的尺寸是否为 (2 * 10 ** 8, 3, 4 * 10 ** 8)
        self.assertEqual(z.size(), (2 * 10 ** 8, 3, 4 * 10 ** 8))
        # 断言尝试获取 z[0][0][0] 的标量值会触发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: z[0][0][0].item())
    # 测试函数：test_upsample_nearest2d_meta
    def test_upsample_nearest2d_meta(self):
        # TODO: 由于 nn 函数中没有 out= 参数的实际使用，因此 test_nn.py 无法触发 out 测试
        # 没有公共 API 可以获取 out 版本

        # 确保不覆盖 out 张量的步长。注意：此测试必须在 2d/3d 上执行，因为 1d 没有任何有意义的布局支持
        x = torch.empty(4, 3, 8, 8, device='meta')
        out = torch.empty(4, 3, 16, 16, device='meta', memory_format=torch.channels_last)
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        x = torch.empty(4, 3, 8, 8, device='meta', memory_format=torch.channels_last)
        out = torch.empty(4, 3, 16, 16, device='meta')
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous())

        # 但如果发生 resize，则会覆盖
        x = torch.empty(4, 3, 8, 8, device='meta', memory_format=torch.channels_last)
        out = torch.empty(0, device='meta')
        torch._C._nn.upsample_nearest2d(x, (16, 16), out=out)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        # 如果 out 的 dtype 不匹配，则报错
        x = torch.empty(4, 3, 8, 8, device='meta', dtype=torch.float)
        out = torch.empty(4, 3, 16, 16, device='meta', dtype=torch.double)
        self.assertExpectedRaisesInline(
            RuntimeError, lambda: torch._C._nn.upsample_nearest2d(x, (16, 16), out=out),
            """Expected out tensor to have dtype torch.float32 but got torch.float64 instead"""
        )

        # 如果 out 的设备不匹配，则报错
        x = torch.empty(0, 3, 8, 8, device='meta')
        out = torch.empty(0, 3, 16, 16, device='cpu')
        # FIXME: 编译时应该正确地报告设备不匹配的错误
        if not TEST_WITH_TORCHINDUCTOR:
            self.assertExpectedRaisesInline(
                RuntimeError, lambda: torch._C._nn.upsample_nearest2d(x, (16, 16), out=out),
                """Attempting to copy from device meta to device cpu, but cross-device copies are not allowed!"""
            )

    # 测试函数：test_add_meta_scalar
    def test_add_meta_scalar(self):
        # 来自 https://github.com/pytorch/pytorch/issues/53815
        x = torch.empty(2, device='meta')
        y = x + 2
        self.assertEqual(y.size(), x.size())

    # 测试以确保在移除之前仍正确处理 .data
    # Tests to make sure we still handle .data properly until it is removed
    def test_dot_data_use(self):
        # 测试 .data 允许原地更改张量类型，在权重类型错误时抛出合适的错误

        # 使用断言检查是否抛出 RuntimeError，并验证错误消息包含 Double 和 ComplexFloat
        with self.assertRaisesRegex(
                RuntimeError,
                '(?=.*Double)(?=.*ComplexFloat)'):

            # 创建一个双精度浮点类型的随机张量作为输入
            input = torch.randn(1, 1, 1, 6, dtype=torch.double)
            # 创建一个复数类型的零张量作为权重
            weight = torch.zeros(1, 1, 1, 3, dtype=torch.complex64)
            # 创建一个二维卷积神经网络模型
            model = torch.nn.Conv2d(1, 1, (1, 3), stride=1, padding=0, bias=False)
            # 将模型的权重数据设置为上述的复数类型零张量
            model.weight.data = weight
            # 对输入进行模型计算
            out = model(input)

    def test_empty_storage_view(self):
        # 测试对空数组的切片操作不会触发尝试调整存储空间大小的错误

        # 创建一个空的 NumPy 数组，并将其转换为 PyTorch 张量
        t = torch.from_numpy(np.empty((0, 4)))
        # 对张量 t 进行切片并尝试原地乘以 1
        t[:, 1::2] *= 1

    def test_has_storage(self):
        # 测试不同情况下张量是否具有存储空间

        # 使用空列表创建张量，并检查其是否有存储空间
        self.assertIsNotNone(torch.tensor([]).storage())
        # 使用空张量创建张量，并检查其是否有存储空间
        self.assertIsNotNone(torch.empty(0).storage())
        # 使用克隆方法创建空列表张量，并检查其是否有存储空间
        self.assertIsNotNone(torch.tensor([]).clone().storage())
        # 使用非零元素的张量创建张量，并检查其是否有存储空间
        self.assertIsNotNone(torch.tensor([0, 0, 0]).nonzero().storage())
        # 使用 new 方法创建空张量，并检查其是否有存储空间
        self.assertIsNotNone(torch.tensor([]).new().storage())

    # FIXME: Extend this test and put in a TensorProperties test class
    def test_numel(self):
        # 测试张量的元素数量方法是否正确计算

        # 创建一个字节类型的三维张量，并检查其元素总数是否正确
        b = torch.ByteTensor(3, 100, 100)
        self.assertEqual(b.nelement(), 3 * 100 * 100)
        self.assertEqual(b.numel(), 3 * 100 * 100)

    # Verifies that (deep)copies of dtypes are the same objects
    def test_copy_dtypes(self):
        # 测试数据类型的深拷贝是否得到相同的对象引用

        # 对所有数据类型进行迭代，并深度拷贝每个数据类型，检查其引用是否相同
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
            copied_dtype = copy.deepcopy(dtype)
            self.assertIs(dtype, copied_dtype)

    def test_dtype_is_signed(self):
        # 测试数据类型是否为有符号类型

        # 对所有数据类型进行迭代，并检查其是否为有符号类型
        for dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.half):
            self.assertEqual(dtype.is_signed, torch.is_signed(torch.tensor(0, dtype=dtype)))

        # 检查对于量化类型是否会触发 RuntimeError
        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.quint8.is_signed)
        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.qint8.is_signed)
        self.assertRaisesRegex(RuntimeError, 'not supported for quantized', lambda: torch.qint32.is_signed)

    # FIXME: Put the following random tests into their own test class or test suite
    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNGState(self):
        # 测试随机数生成器状态的保存和恢复功能

        # 获取当前随机数生成器状态，并进行克隆
        state = torch.get_rng_state()
        stateCloned = state.clone()
        # 生成一千个随机数
        before = torch.rand(1000)

        # 检查两个随机数生成器状态是否相等
        self.assertEqual(state.ne(stateCloned).long().sum(), 0, atol=0, rtol=0)

        # 恢复之前保存的随机数生成器状态，并再次生成一千个随机数
        torch.set_rng_state(state)
        after = torch.rand(1000)
        # 检查恢复状态后生成的随机数与原先生成的随机数是否相等
        self.assertEqual(before, after, atol=0, rtol=0)

    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    # 定义一个测试方法，用于验证随机数生成器状态的分叉（aliasing）
    def test_RNGStateAliasing(self):
        # 在此处分叉随机数流
        gen = torch.Generator()
        # 设置生成器的状态为当前全局随机数生成器的状态
        gen.set_state(torch.get_rng_state())
        # 断言生成器的状态与全局随机数生成器的状态相同
        self.assertEqual(gen.get_state(), torch.get_rng_state())

        # 目标值为生成一组随机数
        target_value = torch.rand(1000)
        # 大幅度改变主随机数生成器的内部状态
        _ = torch.rand(100000)
        # 使用分叉后的生成器生成一组随机数
        forked_value = torch.rand(1000, generator=gen)
        # 断言目标值与分叉后的值相等，不考虑绝对误差和相对误差，如果不等则输出错误信息
        self.assertEqual(target_value, forked_value, atol=0, rtol=0, msg="RNG has not forked correctly.")

    # 如果不支持 Torch Dynamo，跳过此测试方法
    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_RNG_after_pickle(self):
        # 手动设定随机数种子
        torch.random.manual_seed(100)
        # 记录随机数生成器状态下生成的一组随机数
        before = torch.rand(10)

        # 重新设定随机数种子
        torch.random.manual_seed(100)
        # 创建一个字节流缓冲区
        buf = io.BytesIO()
        # 序列化一个张量对象并写入缓冲区
        tensor = torch.tensor([1, 2, 3])
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(tensor)
        # 记录重新设定种子后生成的一组随机数
        after = torch.rand(10)

        # 断言两次生成的随机数序列相同，不考虑绝对误差和相对误差
        self.assertEqual(before, after, atol=0, rtol=0)

    # 如果不支持 Torch Dynamo，跳过此测试方法
    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    def test_boxMullerState(self):
        # 手动设定随机数种子
        torch.manual_seed(123)
        # 生成一个奇数长度的正态分布随机数序列
        odd_number = 101
        seeded = torch.randn(odd_number)
        # 获取当前随机数生成器的状态
        state = torch.get_rng_state()
        # 中间生成一组随机数
        midstream = torch.randn(odd_number)
        # 恢复到之前保存的随机数生成器状态
        torch.set_rng_state(state)
        # 重新生成相同种子下的一组随机数
        repeat_midstream = torch.randn(odd_number)
        # 再次手动设定相同的随机数种子
        torch.manual_seed(123)
        # 重新生成相同种子下的一组随机数
        reseeded = torch.randn(odd_number)
        # 断言中间生成和重新生成的随机数序列相同，不考虑绝对误差和相对误差
        self.assertEqual(midstream, repeat_midstream, atol=0, rtol=0,
                         msg='get_rng_state/set_rng_state not generating same sequence of normally distributed numbers')
        # 断言两次手动设定相同种子生成的随机数序列相同，不考虑绝对误差和相对误差
        self.assertEqual(seeded, reseeded, atol=0, rtol=0,
                         msg='repeated calls to manual_seed not generating same sequence of normally distributed numbers')

    # 如果不支持 Torch Dynamo，跳过此测试方法
    @skipIfTorchDynamo("requires https://github.com/pytorch/torchdynamo/pull/1098")
    # 定义一个测试函数，用于验证手动设置种子功能
    def test_manual_seed(self):
        # 获取当前随机数生成器状态
        rng_state = torch.get_rng_state()
        # 设置随机种子为2
        torch.manual_seed(2)
        # 生成一个包含100个标准正态分布随机数的张量
        x = torch.randn(100)
        # 断言当前的初始种子为2
        self.assertEqual(torch.initial_seed(), 2)
        # 再次设置随机种子为2
        torch.manual_seed(2)
        # 生成另一个包含100个标准正态分布随机数的张量
        y = torch.randn(100)
        # 断言两个张量的值相等
        self.assertEqual(x, y)

        # 定义整型64位最大值和最小值
        max_int64 = 0x7fff_ffff_ffff_ffff
        min_int64 = -max_int64 - 1
        # 定义无符号整型64位最大值
        max_uint64 = 0xffff_ffff_ffff_ffff
        # 检查所有有效种子值的边界情况
        test_cases = [
            # (seed, expected_initial_seed)
            # 正数种子应该不变
            (max_int64, max_int64),
            (max_int64 + 1, max_int64 + 1),
            (max_uint64, max_uint64),
            (0, 0),
            # 负数种子应该从最大种子值开始循环
            (-1, max_uint64),
            (min_int64, max_int64 + 1)
        ]
        # 遍历测试用例
        for seed, expected_initial_seed in test_cases:
            # 设置随机种子
            torch.manual_seed(seed)
            # 获取实际的初始种子值
            actual_initial_seed = torch.initial_seed()
            # 构建错误信息字符串
            msg = (f"expected initial_seed() = {expected_initial_seed:x} "
                   f"after calling manual_seed({seed:x}), but got {actual_initial_seed:x} instead")
            # 断言期望的初始种子与实际初始种子相等，如果不等则输出错误信息
            self.assertEqual(expected_initial_seed, actual_initial_seed, msg=msg)
        # 针对无效的种子值进行测试
        for invalid_seed in [min_int64 - 1, max_uint64 + 1]:
            # 断言设置无效种子时抛出 RuntimeError 异常，并且异常信息包含特定字符串
            with self.assertRaisesRegex(RuntimeError, r'Overflow when unpacking long'):
                torch.manual_seed(invalid_seed)

        # 恢复随机数生成器状态
        torch.set_rng_state(rng_state)

    # FIXME: 描述此测试并将其移植到更合适的测试套件中，用于复制操作
    def test_copy_transpose(self):
        # 生成一个大小为100x100的浮点型张量，并转置
        x = torch.arange(100 * 100, dtype=torch.float).reshape(100, 100).t()
        # 生成一个空的100x100浮点型张量，并将x的数据复制到y中
        y = torch.empty(100, 100, dtype=torch.float)
        y.copy_(x)
        # 断言y的第一列与0到99的序列相等
        self.assertEqual(y[:, 0], range(100))
        # 断言y的第41列与4000到4099的序列相等
        self.assertEqual(y[:, 40], range(4000, 4100))

        # 生成一个空的100x100双精度浮点型张量，并将x的数据复制到y中
        y = torch.empty(100, 100, dtype=torch.double)
        y.copy_(x)
        # 断言y的第一列与0到99的序列相等
        self.assertEqual(y[:, 0], range(100))
        # 断言y的第41列与4000到4099的序列相等
        self.assertEqual(y[:, 40], range(4000, 4100))

        # 验证报告的回归问题 https://github.com/pytorch/pytorch/issues/45269
        # 生成一个大小为100x100的复数浮点型张量，并转置
        x = torch.arange(100 * 100).reshape(100, 100).to(dtype=torch.cfloat).t()
        # 生成一个空的100x100复数浮点型张量，并将x的数据复制到y中
        y = torch.empty(100, 100, dtype=torch.cfloat)
        y.copy_(x)
        # 断言y的第一列与0到99的序列相等
        self.assertEqual(y[:, 0], range(100))
        # 断言y的第41列与4000到4099的序列相等
        self.assertEqual(y[:, 40], range(4000, 4100))

        # 生成一个大小为100x100的32位复数张量，并转置
        x = torch.arange(100 * 100).reshape(100, 100).to(dtype=torch.complex32).t()
        # 生成一个空的100x100 32位复数张量，并将x的数据复制到y中
        y = torch.empty(100, 100, dtype=torch.complex32)
        y.copy_(x)
        # 断言y的第一列与0到99的序列相等
        self.assertEqual(y[:, 0], range(100))
        # 断言y的第41列与4000到4099的序列相等
        self.assertEqual(y[:, 40], range(4000, 4100))

    # FIXME: 将此测试移植到更合适的测试套件中
    def test_copy_broadcast(self):
        # 复制一个大小为5x6的零张量，并从大小为6的张量复制
        torch.zeros(5, 6).copy_(torch.zeros(6))
        # 断言复制大小不匹配时抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.zeros(5, 6).copy_(torch.zeros(30)))
    # 在 inductor（和 aot_eager）中失败，因为 functionalization 替换了 copy_ 为 copy，
    # 这在处理不良输入时无法正确报错。
    def test_copy_many_to_one(self):
        # 测试就地复制，尝试将多个内存存储中的数据复制到单个存储中会引发 RuntimeError。
        self.assertRaises(RuntimeError, lambda: torch.zeros(1, 6).expand(5, 6).copy_(torch.zeros(5, 6)))

    def test_copy_float16(self):
        # 检查 fbgemm 代码不再读取超出内存边界的情况，参见 copy_impl 和 fbgemm::Float16ToFloat_ref。
        # https://github.com/pytorch/pytorch/issues/88543

        # 用于测试不同代码路径的数据类型。
        dtypes = (
            # out_dtype, src_dtype
            (torch.float32, torch.float16),  # fbgemm
            (torch.float16, torch.float32),  # fbgemm
            (torch.float32, torch.float32),  # TensorIterator
        )

        cases = (
            # out_shape, src_shape, is_ok
            # 这些情况过去在 fbgemm 中会崩溃，确保在 TensorIterator 中也会抛出异常。
            ((1, 2, 3), (0, 2, 3), False),  # 相同步长，TensorIterator 不允许
            ((1, 5, 6), (4, 5, 6), False),  # 相同步长，TensorIterator 不允许
            (1, (0, 2, 3), False),  # 不同步长
            ((4, 5, 6), (0, 2, 3), False),  # 不同步长
            ((4, 5, 6), (1, 2, 3), False),  # 不同步长
            ((4, 5, 6), (6, 5, 4), False),  # 相同元素数

            # 这些情况应该在 fbgemm 和 TensorIterator 中通过。
            ((4, 5, 6), (1, 5, 6), True),  # 相同步长
            ((4, 5, 6), (4, 5, 6), True),  # 相同步长
            ((0, 2, 3), 1, True),  # 不同步长，TensorIterator 允许
            ((4, 5, 6), (4, 5, 1), True),  # 不同步长，TensorIterator 允许
        )

        for (out_shape, src_shape, is_ok), (out_dtype, src_dtype) in itertools.product(cases, dtypes):
            out = torch.zeros(out_shape, dtype=out_dtype, device=torch.device('cpu'))
            src = torch.ones(src_shape, dtype=src_dtype, device=torch.device('cpu'))
            if is_ok:
                if torch.cuda.is_available():
                    out_cuda = out.cuda()
                    src_cuda = src.cuda()
                res = out.copy_(src)
                if torch.cuda.is_available():
                    res_cuda = out_cuda.copy_(src_cuda)
                    self.assertEqual(res, res_cuda)
            else:
                self.assertRaises(RuntimeError, lambda: out.copy_(src))

    # FIXME: 移植到更合适的测试套件中
    def test_to(self):
        self._test_to_with_layout(torch.strided)
        is_cuda10_2_or_higher = (
            (torch.version.cuda is not None)
            and ([int(x) for x in torch.version.cuda.split(".")] >= [10, 2]))
        if is_cuda10_2_or_higher:  # 在 cuda10_1 中 sparse_csr 是 beta 版本
            self._test_to_with_layout(torch.sparse_csr)
    # FIXME: describe this test
    # 定义一个测试函数，用于测试`as_subclass`方法的行为
    def test_as_subclass(self):
        # 定义一个继承自torch.Tensor的子类SubTensor，设置一个成员变量
        class SubTensor(torch.Tensor):
            member_var = object()

        # 创建三个不同形状的Tensor对象
        t0 = torch.tensor(0)
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([[3, 4], [5, 6]])

        # 分别调用as_subclass方法将t0、t1、t2转换为SubTensor子类对象
        s0 = t0.as_subclass(SubTensor)
        s1 = t1.as_subclass(SubTensor)
        s2 = t2.as_subclass(SubTensor)

        # 断言确保返回的对象类型是SubTensor子类
        self.assertTrue(type(s0) is SubTensor)
        self.assertTrue(type(s1) is SubTensor)
        self.assertTrue(type(s2) is SubTensor)

        # 断言检查转换后的数据仍然与原始数据相等
        self.assertEqual(t0, s0)
        self.assertEqual(t1, s1)
        self.assertEqual(t2, s2)

        # 修改原始Tensor数据，检查转换后的数据仍然相等
        t0[()] = 1
        t1[1] = 3
        t2[1, 1] = 7
        self.assertEqual(t0, s0)
        self.assertEqual(t1, s1)
        self.assertEqual(t2, s2)

        # 断言检查成员变量是否正确传递
        self.assertTrue(s0.member_var is SubTensor.member_var)
        self.assertTrue(s1.member_var is SubTensor.member_var)
        self.assertTrue(s2.member_var is SubTensor.member_var)

        # 测试自动求导的传播
        t = torch.tensor(5, dtype=torch.float32, requires_grad=True)

        # 对张量进行计算
        exp_t = torch.exp(t)

        # 将计算结果exp_t转换为SubTensor子类对象
        exp_s = exp_t.as_subclass(SubTensor)

        # 断言检查原始张量的梯度初始值为None
        self.assertTrue(t.grad is None)

        # 运行自动求导计算
        exp_s.backward()

        # 断言检查自动求导是否传播到了声明requires_grad=True的原始张量上
        self.assertTrue(t.grad is not None)

        # 断言检查当传入一个不正确的子类时是否会抛出正确的异常信息
        class BadSubTensor:
            member_var = object()

        err_msg = "Creating a Tensor subclass from a class that does not inherit from Tensor"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            s0 = t0.as_subclass(BadSubTensor)

    # FIXME: Port to a test suite that better fits slicing
    # 将此测试迁移到更适合切片的测试套件中
    def test_slice(self):
        empty = torch.empty(0, 4)
        x = torch.arange(0., 16).view(4, 4)

        # 检查整体切片是否与原始张量相等
        self.assertEqual(x[:], x)
        self.assertEqual(x[:4], x)

        # start和stop被限制在维度大小内
        self.assertEqual(x[:5], x)

        # 如果start >= stop，则结果为空
        self.assertEqual(x[2:1], empty)
        self.assertEqual(x[2:2], empty)

        # 超出边界的切片结果为空
        self.assertEqual(x[10:12], empty)

        # 其他正确性检查
        self.assertEqual(x[:1].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(x[:-3].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(x[:, -2:3].tolist(), [[2], [6], [10], [14]])
        self.assertEqual(x[0:-1:2].tolist(), [[0, 1, 2, 3], [8, 9, 10, 11]])
    # 测试函数：test_split_with_sizes_copy_out，用于测试张量分割和复制功能
    def test_split_with_sizes_copy_out(self):
        # 检查CUDA是否可用，选择设备
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # 定义张量形状
        shape = (30, 40, 50)
        # 生成指定形状的随机张量，并放到指定设备上
        x = torch.rand(*shape, device=device)
        # 定义多组测试用例，每个测试用例包含维度和分割大小列表
        cases = [
            (0, [3, 7, 8, 12]),
            (1, [3, 7, 10, 20]),
            (-2, [3, 7, 10, 20]),
            (2, [3, 7, 10, 12, 18]),
            (-1, [3, 7, 10, 12, 18]),
            (2, [3, 7, 10, 0, 30]),
        ]
        # 对每个测试用例进行迭代
        for dim, split_sizes in cases:
            # 使用给定大小在指定维度上分割张量x，并返回视图列表
            views = x.split_with_sizes(split_sizes, dim=dim)
            # 创建期望的视图列表，每个视图是原视图的克隆
            expects = [v.clone() for v in views]
            # 创建与视图列表相同大小的空张量列表out
            out = [torch.zeros_like(v) for v in views]
            # 验证每个视图是否不全等于对应的空张量t
            for expect, t in zip(expects, out):
                if expect.numel() != 0:
                    self.assertFalse(expect.eq(t).all().item())

            # 使用split_with_sizes_copy函数将x按split_sizes分割并复制到out中
            torch.split_with_sizes_copy(x, split_sizes, dim=dim, out=out)
            # 验证每个期望的视图是否全部等于复制后的out中的对应视图t
            for expect, t in zip(expects, out):
                self.assertTrue(expect.eq(t).all().item())

            # 如果CUDA不可用，跳过下面的测试步骤
            if not torch.cuda.is_available():
                continue

            # 使用CUDA图形进行测试
            out = [torch.zeros_like(v) for v in views]
            # 验证每个视图是否不全等于对应的空张量t
            for expect, t in zip(expects, out):
                if expect.numel() != 0:
                    self.assertFalse(expect.eq(t).all().item())

            # 创建CUDA图形对象g，并使用图形上下文
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # 使用split_with_sizes_copy函数将x按split_sizes分割并复制到out中
                torch.split_with_sizes_copy(x, split_sizes, dim=dim, out=out)

            # 回放CUDA图形操作
            g.replay()
            # 验证每个期望的视图是否全部等于复制后的out中的对应视图t
            for expect, t in zip(expects, out):
                self.assertTrue(expect.eq(t).all().item())

    # 测试函数：test_type，用于测试张量类型转换功能
    def test_type(self):
        # 生成一个3x3的双精度随机张量x
        x = torch.randn(3, 3).double()
        # 验证将x转换为torch.FloatTensor后的数据类型是否为torch.float32
        self.assertEqual(x.type('torch.FloatTensor').dtype, torch.float32)
        # 验证将x转换为torch.FloatTensor后的数据类型是否为torch.float32
        self.assertEqual(x.type(torch.FloatTensor).dtype, torch.float32)
        # 验证将x转换为整型后，再转换回张量的默认数据类型
        self.assertEqual(x.int().type(torch.Tensor).dtype, torch.get_default_dtype())
        # 验证将x转换为torch.int32后的数据类型是否为torch.int32
        self.assertEqual(x.type(torch.int32).dtype, torch.int32)

    # FIXME: port to a quantization test suite
    # 测试函数：test_qengine，测试量化引擎设置
    def test_qengine(self):
        # 获取支持的量化引擎列表
        qengines = torch.backends.quantized.supported_engines
        # 保存当前的量化引擎设置
        original_qe = torch.backends.quantized.engine
        # 遍历每个量化引擎并设置
        for qe in qengines:
            torch.backends.quantized.engine = qe
            # 断言当前的量化引擎设置是否成功
            assert torch.backends.quantized.engine == qe, 'qengine not set successfully'
        # 恢复原始的量化引擎设置
        torch.backends.quantized.engine = original_qe

    # 测试函数：test_terminate_handler_on_crash，测试异常情况下的终止处理
    def test_terminate_handler_on_crash(self):
        # 定义一个命令列表，其中包含一个导致异常终止的Python命令
        cmd = [sys.executable, '-c', "import os; os.environ[\"TORCH_CUSTOM_TERMINATE\"] ='1'; \
               import torch; import torch._C; torch._C._abort()"]
        # 使用断言检测异常是否抛出
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            subprocess.check_output(cmd, shell=False)
        e = cm.exception
        output = e.stdout.decode("utf-8")
        # 验证异常的返回码不为0且输出不为空
        self.assertNotEqual(e.returncode, 0)
        self.assertNotEqual(output, None)
        # 验证输出中是否包含特定的异常信息
        self.assertIn('Unhandled exception caught in c10/util/AbortHandler.h', output)
    # FIXME: 将代码迁移到一个分布式测试套件中——此外……在 Windows CUDA 上如何导致内存耗尽？
    @slowTest  # 标记为慢速测试
    @unittest.skipIf(NO_MULTIPROCESSING_SPAWN, "在不支持 spawn 启动方法的环境下禁用")
    @unittest.skipIf(IS_WINDOWS, 'FIXME: Windows 上 CUDA 内存耗尽错误')
    def test_multinomial_invalid_probs(self):
        def _spawn_method(self, method, arg):
            try:
                mp.set_start_method('spawn')  # 设置启动方法为 spawn
            except RuntimeError:
                pass
            with mp.Pool(1) as pool:  # 使用单个进程池
                out = pool.map(method, [arg])  # 调用方法并映射结果
                self.assertTrue(out[0])  # 断言第一个结果为真

        def _test_multinomial_invalid_probs(probs):
            try:
                # n_sample = 1 是特殊情况，测试更一般的 n_sample=2
                torch.multinomial(probs.to('cpu'), 2)  # 使用 CPU 上的概率张量进行多项式采样
                return False  # 不应该执行到这里
            except RuntimeError as e:
                return 'probability tensor contains either `inf`, `nan` or element < 0' in str(e)

            # 分别测试不同的概率张量是否触发异常
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., -1., 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., inf, 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., -inf, 1.]))
            _spawn_method(_test_multinomial_invalid_probs, torch.tensor([1., 1., nan]))

    # FIXME: 将代码迁移到更合适的测试套件中
    def test_to_with_tensor(self):
        a = torch.tensor(5)
        self.assertEqual(a.device, a.to(a).device)  # 测试张量 a 的设备是否与转换后的设备相同

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5., device=cuda)
                    self.assertEqual(b.device, b.to(b, non_blocking=non_blocking).device)  # 测试张量 b 转换后设备的一致性
                    self.assertEqual(a.device, b.to(a, non_blocking=non_blocking).device)  # 测试不同张量间的设备一致性
                    self.assertEqual(b.device, a.to(b, non_blocking=non_blocking).device)  # 再次测试不同张量间的设备一致性

    # 测试 use_deterministic_flag 是否可以如预期设置
    @wrapDeterministicFlagAPITest
    # 定义一个测试方法，用于测试确定性标志的设置
    def test_deterministic_flag(self):
        # 遍历确定性和仅警告标志的所有组合
        for deterministic, warn_only in product([True, False], [True, False]):
            # 设置 Torch 使用确定性算法，并指定是否仅警告
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            # 断言当前 Torch 的确定性算法是否与设置一致
            self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
            # 断言当前 Torch 的确定性算法是否设置为仅警告模式
            self.assertEqual(warn_only, torch.is_deterministic_algorithms_warn_only_enabled())

            # 根据确定性和仅警告标志设置调试模式
            if deterministic:
                if warn_only:
                    debug_mode = 1
                else:
                    debug_mode = 2
            else:
                debug_mode = 0

            # 断言当前 Torch 的调试模式是否与设置一致
            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())

        # 遍历所有调试模式，并设置 Torch 的调试模式，然后断言是否设置成功
        for debug_mode in [0, 1, 2]:
            torch.set_deterministic_debug_mode(debug_mode)
            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())
            # 根据调试模式设置确定性和仅警告标志，并断言是否设置成功
            deterministic = debug_mode in [1, 2]
            warn_only = debug_mode == 1
            self.assertEqual(deterministic, torch.are_deterministic_algorithms_enabled())
            self.assertEqual(warn_only, torch.is_deterministic_algorithms_warn_only_enabled())

        # 遍历预定义的调试模式及其对应字符串，设置 Torch 的调试模式，并断言是否设置成功
        for debug_mode, debug_mode_str in [(0, 'default'), (1, 'warn'), (2, 'error')]:
            torch.set_deterministic_debug_mode(debug_mode_str)
            self.assertEqual(debug_mode, torch.get_deterministic_debug_mode())

        # 使用断言检测是否能捕获设置非预期类型参数的异常
        with self.assertRaisesRegex(
                TypeError,
                r"_set_deterministic_algorithms\(\): argument 'mode' \(position 1\) must be bool, not int"):
            torch.use_deterministic_algorithms(1)

        # 使用断言检测是否能捕获设置非预期类型参数的异常
        with self.assertRaisesRegex(
                TypeError,
                r"_set_deterministic_algorithms\(\): argument 'warn_only' must be bool, not int"):
            torch.use_deterministic_algorithms(False, warn_only=1)

    # 测试 torch.utils.deterministic.fill_uninitialized_memory 是否能如预期设置
    # 测试在填充未初始化内存时的确定性保护
    def test_deterministic_fill_uninitialized_memory(self):
        # 使用 DeterministicGuard 开启确定性保护，禁止填充未初始化内存
        with DeterministicGuard(True, fill_uninitialized_memory=False):
            # 验证 torch.utils.deterministic.fill_uninitialized_memory 为 False
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            # 验证 torch._C._get_deterministic_fill_uninitialized_memory() 返回 False
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            # 使用 DeterministicGuard 开启确定性保护，并允许填充未初始化内存
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                # 验证 torch.utils.deterministic.fill_uninitialized_memory 为 True
                self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
                # 验证 torch._C._get_deterministic_fill_uninitialized_memory() 返回 True
                self.assertTrue(torch._C._get_deterministic_fill_uninitialized_memory())

            # 验证再次验证禁止填充未初始化内存
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            # 修改 torch.utils.deterministic.fill_uninitialized_memory 为 False
            torch.utils.deterministic.fill_uninitialized_memory = False
            # 验证 torch.utils.deterministic.fill_uninitialized_memory 确实为 False
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            # 验证 torch._C._get_deterministic_fill_uninitialized_memory() 仍为 False
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            # 修改 torch.utils.deterministic.fill_uninitialized_memory 为 True
            torch.utils.deterministic.fill_uninitialized_memory = True
            # 验证 torch.utils.deterministic.fill_uninitialized_memory 确实为 True
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
            # 验证 torch._C._get_deterministic_fill_uninitialized_memory() 确实为 True
            self.assertTrue(torch._C._get_deterministic_fill_uninitialized_memory())

            # 使用 torch._C._set_deterministic_fill_uninitialized_memory() 设置为 False
            torch._C._set_deterministic_fill_uninitialized_memory(False)
            # 验证 torch.utils.deterministic.fill_uninitialized_memory 确实为 False
            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
            # 验证 torch._C._get_deterministic_fill_uninitialized_memory() 仍为 False
            self.assertFalse(torch._C._get_deterministic_fill_uninitialized_memory())

            # 使用 torch._C._set_deterministic_fill_uninitialized_memory() 设置为 True
            torch._C._set_deterministic_fill_uninitialized_memory(True)
            # 验证 torch.utils.deterministic.fill_uninitialized_memory 确实为 True
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
            # 验证 torch._C._get_deterministic_fill_uninitialized_memory() 确实为 True
            self.assertTrue(torch._C._get_deterministic_fill_uninitialized_memory())

            # 使用 self.assertRaisesRegex 检查设定为非布尔值时抛出异常
            with self.assertRaisesRegex(RuntimeError, r"expected a bool, but got int"):
                torch.utils.deterministic.fill_uninitialized_memory = 1

    # 测试通过 dtype 名称进行类型转换
    def test_type_conversion_via_dtype_name(self):
        # 创建一个包含单个元素的张量 x
        x = torch.tensor([1])
        # 验证 x 转换为 torch.uint8 后的数据类型
        self.assertEqual(x.byte().dtype, torch.uint8)
        # 验证 x 转换为 torch.bool 后的数据类型
        self.assertEqual(x.bool().dtype, torch.bool)
        # 验证 x 转换为 torch.int8 后的数据类型
        self.assertEqual(x.char().dtype, torch.int8)
        # 验证 x 转换为 torch.float64 后的数据类型
        self.assertEqual(x.double().dtype, torch.float64)
        # 验证 x 转换为 torch.float32 后的数据类型
        self.assertEqual(x.float().dtype, torch.float32)
        # 验证 x 转换为 torch.float16 后的数据类型
        self.assertEqual(x.half().dtype, torch.float16)
        # 验证 x 转换为 torch.int32 后的数据类型
        self.assertEqual(x.int().dtype, torch.int32)
        # 验证 x 转换为 torch.bfloat16 后的数据类型
        self.assertEqual(x.bfloat16().dtype, torch.bfloat16)
        
        # 创建一个复数张量 cfloat，验证其实部分与 x 转换为 float 后的实部相等
        cfloat = x.cfloat()
        self.assertEqual(cfloat.dtype, torch.complex64)
        self.assertEqual(cfloat.real, x.float())
        # 验证 cfloat 的虚部为与其形状相同的零张量
        self.assertEqual(cfloat.imag, torch.zeros_like(cfloat.imag))
        
        # 创建一个双精度复数张量 cdouble，验证其实部分与 x 转换为 double 后的实部相等
        cdouble = x.cdouble()
        self.assertEqual(cdouble.dtype, torch.complex128)
        self.assertEqual(cdouble.real, x.double())
        # 验证 cdouble 的虚部为与其形状相同的零张量
        self.assertEqual(cdouble.imag, torch.zeros_like(cdouble.imag))
        
        # 创建一个半精度复数张量 chalf，验证其实部分与 x 转换为 half 后的实部相等
        chalf = x.chalf()
        self.assertEqual(chalf.dtype, torch.complex32)
        self.assertEqual(chalf.real, x.half())
        # 验证 chalf 的虚部为与其形状相同的零张量
        self.assertEqual(chalf.imag, torch.zeros_like(chalf.imag))
    # 测试类型别名映射是否正确
    def test_type_alias(self):
        # 定义 Torch 数据类型到别名的映射字典
        type_alias_map = {torch.float64: torch.double,
                          torch.float32: torch.float,
                          torch.int32: torch.int,
                          torch.int64: torch.long,
                          torch.int16: torch.short,
                          torch.float16: torch.half,
                          torch.complex32: torch.chalf,
                          torch.complex64: torch.cfloat}
        
        # 遍历映射字典，验证别名是否正确
        for dtype, alias in type_alias_map.items():
            self.assertIs(alias, dtype)

    # 测试文档模板是否被所有公共 API 使用
    def test_doc_template(self) -> None:
        """
        Test that all public API doc strings use the same standard template for
        all common arguments such as tensor or dim
        """
        # 导入文档模板相关的内容
        from torch._torch_docs import __file__ as doc_file
        from torch._torch_docs import multi_dim_common, single_dim_common, factory_common_args, factory_like_common_args

        # 打开并读取文档文件
        with open(doc_file, encoding="utf-8") as f:
            doc_strs = f.read()

        # 使用正则表达式找出所有文档字符串中的模板匹配
        matches = re.findall(
            r'add_docstr\(([^,]+?),[^"\']*?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')(?:\.|,?[^,\)]*?\))',
            doc_strs,
            re.MULTILINE | re.DOTALL,
        )
        # 断言至少找到一个匹配
        self.assertTrue(matches)

        # 遍历所有匹配项
        for m in matches:
            func = m[0].strip()  # 函数名
            desc = m[1].strip()  # 描述文本

            # 遍历公共参数字典，验证描述中是否有用到这些参数
            for common_args in [multi_dim_common, single_dim_common, factory_common_args, factory_like_common_args]:
                for k, v in common_args.items():
                    self.assertNotIn(v, desc, f'The argument description "{v}" in {func} can be '
                                              f'replaced by {{{k}}}')

    # FIXME: deprecate torch.Tensor constructor
    # 测试 Tensor 构造函数接受标量
    def test_tensor_ctor_scalar(self):
        # 使用标量初始化 Tensor
        x = torch.Tensor(torch.tensor(1.0))
        # 断言 Tensor 的值是否正确
        self.assertEqual(x, torch.tensor(1.0))

    # 测试深拷贝梯度
    def test_deepcopy_gradient(self):
        # 导入深拷贝模块
        from copy import deepcopy
        # 创建一个全零 Tensor，并设定梯度为全一
        a = torch.zeros(10)
        a.grad = torch.ones(10)
        # 断言深拷贝后的梯度是否与原始 Tensor 的梯度相同
        self.assertEqual(a.grad, deepcopy(a).grad)
        # 创建一个稀疏 Tensor，并设定梯度为稀疏全一
        s = torch.zeros(10).to_sparse()
        s.grad = torch.ones(10).to_sparse()
        # 断言稀疏 Tensor 的深拷贝后梯度是否相同
        self.assertEqual(s.grad, deepcopy(s).grad)
        # 确保共享不被打破
        # 深拷贝列表，验证梯度是否依旧共享
        c = deepcopy([a, a.grad])
        self.assertTrue(c[0].grad is c[1])

    # 测试 Tensor 基础初始化
    def test_tensor_base_init(self):
        # 直接构造不可以
        # 断言直接构造 TensorBase 会抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch._C.TensorBase())

        # 子类化直接不可以
        # 使用断言验证直接子类化 TensorBase 会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Cannot subclass"):
            class Tfail(torch._C.TensorBase):
                pass

        # 使用 Tensor 进行子类化可以
        # 定义一个允许子类化的 Tensor 类
        class T(torch.Tensor):
            pass

        # 实例化一个 Tensor 子类
        T()

    # 测试 Storage 基础初始化
    def test_storage_base_init(self):
        # 直接构造不可以
        # 断言直接构造 StorageBase 会抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch._C.StorageBase())

        # 但是子类化是可以的
        # 定义一个允许子类化的 Storage 类
        class T(torch._C.StorageBase):
            pass

        # 实例化一个 Storage 子类
        T()
    def test_tensor_base_new(self):
        # 定义一个测试类 TestTensor，继承自 torch.Tensor
        # 使用静态方法 __new__ 重写父类的构造方法，调用父类的 __new__ 方法创建实例
        class TestTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, x, *args, **kwargs):
                return super().__new__(cls, x, *args, **kwargs)

        # 创建一个包含5个1的张量
        x = torch.ones(5)
        # 创建 TestTensor 的实例
        test_tensor = TestTensor(x)

    def test_storage_base_new(self):
        # 定义一个测试类 TestStorage，继承自 torch._C.StorageBase
        # 使用静态方法 __new__ 重写父类的构造方法，调用父类的 __new__ 方法创建实例
        class TestStorage(torch._C.StorageBase):
            @staticmethod
            def __new__(cls, x, *args, **kwargs):
                return super().__new__(cls, x, *args, **kwargs)

        # 创建一个包含5个未初始化元素的 UntypedStorage 对象
        x = torch.UntypedStorage(5)
        # 创建 TestStorage 的实例
        test_storage = TestStorage(x)

    def test_pyobj_preserved(self):
        # 创建一个包含2个未初始化元素的张量
        x = torch.empty(2)
        # 给张量的 __dict__ 属性添加一个字段 'foo'
        x.foo = 2
        # 创建另一个包含2个未初始化元素的张量
        y = torch.empty(2)
        # 将 x 赋值给 y.grad，保留了 x 的引用
        y.grad = x
        # 删除变量 x，释放其在 Python 中的引用
        del x
        # 断言 y.grad.foo 的值为 2，验证字段 'foo' 被保留
        self.assertEqual(y.grad.foo, 2)
        # 将 y.grad 赋值给变量 z，保留了 y.grad 的引用
        z = y.grad
        # 删除变量 z，释放其在 Python 中的引用
        del z
        # 断言 y.grad.foo 的值为 2，验证字段 'foo' 被保留
        self.assertEqual(y.grad.foo, 2)

    def test_subclass_preserved(self):
        # 定义一个 MyTensor 类，继承自 torch.Tensor
        class MyTensor(torch.Tensor):
            pass

        # 创建一个包含2个未初始化元素的 MyTensor 实例
        x = MyTensor(torch.empty(2))
        # 创建另一个包含2个未初始化元素的张量
        y = torch.empty(2)
        # 将 x 赋值给 y.grad，保留了 x 的引用
        y.grad = x
        # 删除变量 x，释放其在 Python 中的引用
        del x
        # 断言 y.grad 的类型为 MyTensor，验证类型信息被保留
        self.assertEqual(type(y.grad), MyTensor)
        # 将 y.grad 赋值给变量 z，保留了 y.grad 的引用
        z = y.grad
        # 删除变量 z，释放其在 Python 中的引用
        del z
        # 断言 y.grad 的类型为 MyTensor，验证类型信息被保留
        self.assertEqual(type(y.grad), MyTensor)

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc(self):
        # 创建一个 Tracker 对象 m 和一个追踪器对象 t
        m, t = Tracker.make()
        # 创建一个包含10个未初始化元素的 UntypedStorage 对象
        s0 = torch.UntypedStorage(10)
        # 将 s0 赋值给 s1
        s1 = s0
        # 将追踪器对象 t 分配给 s0 的 _tracker 属性
        s0._tracker = t
        # 删除变量 t，释放其在 Python 中的引用

        # 断言 m[0] 为 False，验证追踪器对象未触发
        self.assertFalse(m[0])
        # 删除变量 s0，释放其在 Python 中的引用
        del s0
        # 断言 m[0] 为 False，验证追踪器对象未触发
        self.assertFalse(m[0])
        # 删除变量 s1，释放其在 Python 中的引用
        del s1
        # 断言 m[0] 为 True，验证追踪器对象已触发

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_from_tensor_dealloc(self):
        # 创建一个 Tracker 对象 m 和一个追踪器对象 t
        m, t = Tracker.make()
        # 创建一个包含10个随机元素的张量
        a = torch.randn(10)
        # 从张量 a 中获取其对应的 UntypedStorage 对象 s0
        s0 = a.untyped_storage()
        # 将追踪器对象 t 分配给 s0 的 _tracker 属性
        s0._tracker = t
        # 删除变量 t，释放其在 Python 中的引用

        # 从张量 a 中获取其另一个 UntypedStorage 对象 s1
        s1 = a.untyped_storage()
        # 断言 s0 和 s1 是同一个对象，验证存储对象的一致性
        self.assertTrue(s0 is s1)
        # 断言 s1 具有 '_tracker' 属性，验证追踪器对象被保留

        # 删除变量 a，释放其在 Python 中的引用
        del a

        # 断言 m[0] 为 False，验证追踪器对象未触发
        self.assertFalse(m[0])
        # 删除变量 s0，释放其在 Python 中的引用
        del s0
        # 断言 m[0] 为 False，验证追踪器对象未触发
        self.assertFalse(m[0])
        # 删除变量 s1，释放其在 Python 中的引用
        del s1
        # 断言 m[0] 为 True，验证追踪器对象已触发

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_from_tensor_dealloc_zombie(self):
        # 创建一个 Tracker 对象 m 和一个追踪器对象 t
        m, t = Tracker.make()
        # 创建一个包含10个随机元素的张量
        a = torch.randn(10)
        # 从张量 a 中获取其对应的 UntypedStorage 对象 s0
        s0 = a.untyped_storage()
        # 将追踪器对象 t 分配给 s0 的 _tracker 属性
        s0._tracker = t
        # 删除变量 t，释放其在 Python 中的引用

        # 从张量 a 中获取其另一个 UntypedStorage 对象 s1
        s1 = a.untyped_storage()
        # 断言 s0 和 s1 是同一个对象，验证存储对象的一致性
        self.assertTrue(s0 is s1)
        # 断言 s1 具有 '_tracker' 属性，验证追踪器对象被保留

        # 断言 m[0] 为 False，验证追踪器对象未触发
        self.assertFalse(m[0])
        # 删除变量 s0，释放其在 Python 中的引用
        del s0
        # 断言 m[0] 为 False，验证追踪器对象未触发
        self.assertFalse(m[0])
        # 删除变量 s1，释放其在 Python 中的引用
        del s1
        # 断言 m[0] 为 False，验证追踪器对象未触发
        self.assertFalse(m[0])
        # 删除变量 a，释放其在 Python 中的引用
        del a
        # 断言 m[0] 为 True，验证追踪器对象已触发

    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    # 定义一个测试方法，测试从张量中创建的存储对象在释放后是否能正确恢复
    def test_storage_from_tensor_dealloc_resurrected(self):
        # 创建跟踪器和张量
        m, t = Tracker.make()
        a = torch.randn(10)
        # 获取张量的未类型化存储
        s0 = a.untyped_storage()
        # 将跟踪器附加到存储上，并释放跟踪器对象
        s0._tracker = t
        del t

        # 再次获取相同张量的未类型化存储
        s1 = a.untyped_storage()
        # 断言两个存储对象是同一个对象
        self.assertTrue(s0 is s1)
        # 断言存储对象有 '_tracker' 属性
        self.assertTrue(hasattr(s1, '_tracker'))

        # 断言跟踪器已经被释放
        self.assertFalse(m[0])
        # 释放 s0，但不会触发跟踪器
        del s0
        # 再次断言跟踪器未被释放
        self.assertFalse(m[0])
        # 释放 s1，跟踪器未被释放
        del s1
        # 再次断言跟踪器未被释放
        self.assertFalse(m[0])

        # 获取张量的未类型化存储
        s0 = a.untyped_storage()
        # 断言 s0 是 torch.UntypedStorage 类型的对象
        self.assertTrue(isinstance(s0, torch.UntypedStorage))

        # 释放张量对象 a
        del a
        # 断言跟踪器已被释放
        self.assertFalse(m[0])
        # 释放 s0
        del s0
        # 断言跟踪器已被触发
        self.assertTrue(m[0])

    # 如果在 TorchDynamo 中，跳过这个测试（因为跟踪器钩子在 TorchDynamo 中不工作）
    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc_resurrected(self):
        # 创建跟踪器和未类型化存储对象
        m, t = Tracker.make()
        s = torch.UntypedStorage(10)
        # 将跟踪器附加到存储对象上，并释放跟踪器对象
        s._tracker = t
        del t

        # 使用存储对象创建张量
        a = torch.tensor(s)
        # 断言跟踪器未被释放
        self.assertFalse(m[0])
        # 释放存储对象 s
        del s
        # 再次断言跟踪器未被释放
        self.assertFalse(m[0])

        # 获取张量的未类型化存储
        s = a.untyped_storage()
        # 断言 s 是 torch.UntypedStorage 类型的对象
        self.assertTrue(isinstance(s, torch.UntypedStorage))

        # 释放张量对象 a
        del a
        # 断言跟踪器已被释放
        self.assertFalse(m[0])
        # 释放存储对象 s
        del s
        # 再次断言跟踪器已被触发
        self.assertTrue(m[0])

    # 如果在 TorchDynamo 中，跳过这个测试（因为跟踪器钩子在 TorchDynamo 中不工作）
    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc_subclass_zombie(self):
        # 定义一个继承自 torch.UntypedStorage 的子类 MyStorage
        class MyStorage(torch.UntypedStorage):
            finalized_count = 0

            # 定义析构函数，用于跟踪对象销毁次数
            def __del__(self):
                MyStorage.finalized_count += 1

        # 创建跟踪器和 MyStorage 类的对象
        m, t = Tracker.make()
        s = MyStorage(10)
        # 将跟踪器附加到 MyStorage 对象上，并释放跟踪器对象
        s._tracker = t
        del t

        # 使用 MyStorage 对象创建张量
        a = torch.tensor(s)
        # 断言跟踪器未被释放
        self.assertFalse(m[0])
        # 释放 MyStorage 对象 s
        del s

        # 断言 MyStorage 的析构计数为 0
        self.assertEqual(MyStorage.finalized_count, 0)
        # 再次断言跟踪器未被释放
        self.assertFalse(m[0])

        # 释放张量对象 a
        del a
        # 断言 MyStorage 的析构计数为 1
        self.assertEqual(MyStorage.finalized_count, 1)
        # 断言跟踪器已被释放
        self.assertTrue(m[0])

    # 如果在 TorchDynamo 中，跳过这个测试（因为跟踪器钩子在 TorchDynamo 中不工作）
    @skipIfTorchDynamo("Tracker hook does not work in TorchDynamo")
    def test_storage_dealloc_subclass_resurrected(self):
        # 定义一个继承自 torch.UntypedStorage 的子类 MyStorage
        class MyStorage(torch.UntypedStorage):
            finalized_count = 0

            # 定义析构函数，用于跟踪对象销毁次数
            def __del__(self):
                MyStorage.finalized_count += 1

        # 创建跟踪器和 MyStorage 类的对象
        m, t = Tracker.make()
        s = MyStorage(10)
        # 将跟踪器附加到 MyStorage 对象上，并释放跟踪器对象
        s._tracker = t
        del t

        # 使用 MyStorage 对象创建张量
        a = torch.tensor(s)
        # 断言跟踪器未被释放
        self.assertFalse(m[0])
        # 释放 MyStorage 对象 s
        del s

        # 断言 MyStorage 的析构计数为 0
        self.assertEqual(MyStorage.finalized_count, 0)
        # 再次断言跟踪器未被释放
        self.assertFalse(m[0])

        # 获取张量的未类型化存储
        s = a.untyped_storage()
        # 释放张量对象 a
        del a
        # 断言跟踪器未被释放
        self.assertFalse(m[0])
        # 断言 MyStorage 的析构计数为 0
        self.assertEqual(MyStorage.finalized_count, 0)
        # 断言 s 是 MyStorage 类的实例
        self.assertTrue(isinstance(s, MyStorage))
        # 释放存储对象 s
        del s
        # 断言 MyStorage 的析构计数为 1
        self.assertEqual(MyStorage.finalized_count, 1)
        # 断言跟踪器已被释放
        self.assertTrue(m[0])
    # 定义一个测试函数，用于测试特定情况下张量对象的析构过程
    def test_tensor_slot_dealloc(self):

        # 定义一个继承自torch.Tensor的子类SlotTensor1，限定其只能有一个名为'slot1'的属性
        class SlotTensor1(torch.Tensor):
            __slots__ = ['slot1']

        # 定义一个继承自SlotTensor1的子类SlotTensor2，限定其只能有一个名为'slot2'的属性
        class SlotTensor2(SlotTensor1):
            __slots__ = ['slot2']

        # 创建两个追踪对象m1和t1，m2和t2，使用Tracker.make()方法
        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()

        # 创建一个SlotTensor2对象，初始化时传入一个大小为2的空张量
        slot_tensor = SlotTensor2(torch.empty(2))

        # 将t1赋值给slot_tensor的slot1属性，将t2赋值给slot_tensor的slot2属性
        slot_tensor.slot1 = t1
        slot_tensor.slot2 = t2

        # 删除t1和t2对象
        del t1
        del t2

        # 断言m1[0]和m2[0]均为False
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])

        # 删除slot_tensor对象
        del slot_tensor

        # 断言m1[0]和m2[0]均为True
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    # 定义一个测试函数，用于测试特定情况下存储对象的析构过程
    def test_storage_slot_dealloc(self):

        # 定义一个继承自torch._C.StorageBase的子类SlotStorage1，限定其只能有一个名为'slot1'的属性
        class SlotStorage1(torch._C.StorageBase):
            __slots__ = ['slot1']

        # 定义一个继承自SlotStorage1的子类SlotStorage2，限定其只能有一个名为'slot2'的属性
        class SlotStorage2(SlotStorage1):
            __slots__ = ['slot2']

        # 创建两个追踪对象m1和t1，m2和t2，使用Tracker.make()方法
        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()

        # 创建一个SlotStorage2对象，初始化时传入一个大小为2的UntypedStorage对象
        slot_storage = SlotStorage2(torch.UntypedStorage(2))

        # 将t1赋值给slot_storage的slot1属性，将t2赋值给slot_storage的slot2属性
        slot_storage.slot1 = t1
        slot_storage.slot2 = t2

        # 删除t1和t2对象
        del t1
        del t2

        # 断言m1[0]和m2[0]均为False
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])

        # 删除slot_storage对象
        del slot_storage

        # 断言m1[0]和m2[0]均为True
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    # 跳过TorchDynamo平台的测试，因为这些测试不适用于TorchDynamo
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_tensor_dict_dealloc(self):
        # 创建一个追踪对象m和t，使用Tracker.make()方法
        m, t = Tracker.make()

        # 创建一个大小为2的空张量x
        x = torch.empty(2)

        # 将t对象赋值给x的arf属性
        x.arf = t

        # 删除t对象
        del t

        # 断言m[0]为False
        self.assertFalse(m[0])

        # 删除x对象
        del x

        # 断言m[0]为True
        self.assertTrue(m[0])

    # 跳过TorchDynamo平台的测试，因为这些测试不适用于TorchDynamo
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_storage_dict_dealloc(self):
        # 创建一个追踪对象m和t，使用Tracker.make()方法
        m, t = Tracker.make()

        # 创建一个大小为2的UntypedStorage对象x
        x = torch.UntypedStorage(2)

        # 将t对象赋值给x的arf属性
        x.arf = t

        # 删除t对象
        del t

        # 断言m[0]为False
        self.assertFalse(m[0])

        # 删除x对象
        del x

        # 断言m[0]为True
        self.assertTrue(m[0])

    # 定义一个测试函数，用于测试特定情况下张量对象的析构过程
    def test_tensor_finalizer_dealloc(self):
        # 创建一个包含False的列表m
        m = [False]

        # 定义一个继承自torch.Tensor的子类FinalizerTensor，覆写__del__方法将m[0]设置为True
        class FinalizerTensor(torch.Tensor):
            def __del__(self):
                m[0] = True

        # 创建一个FinalizerTensor对象，初始化时传入一个大小为2的空张量
        fin_tensor = FinalizerTensor(torch.empty(2))

        # 断言m[0]为False
        self.assertFalse(m[0])

        # 删除fin_tensor对象
        del fin_tensor

        # 断言m[0]为True
        self.assertTrue(m[0])

    # 定义一个测试函数，用于测试特定情况下存储对象的析构过程
    def test_storage_finalizer_dealloc(self):
        # 创建一个包含False的列表m
        m = [False]

        # 定义一个继承自torch._C.StorageBase的子类FinalizerStorage，覆写__del__方法将m[0]设置为True
        class FinalizerStorage(torch._C.StorageBase):
            def __del__(self):
                m[0] = True

        # 创建一个FinalizerStorage对象，初始化时传入一个大小为2的UntypedStorage对象
        fin_storage = FinalizerStorage(torch.UntypedStorage(2))

        # 断言m[0]为False
        self.assertFalse(m[0])

        # 删除fin_storage对象
        del fin_storage

        # 断言m[0]为True
        self.assertTrue(m[0])

    # 跳过TorchDynamo平台的测试，因为这些测试不适用于TorchDynamo
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_tensor_weakref_dealloc(self):
        # 创建一个大小为2的空张量x
        x = torch.empty(2)

        # 创建一个包含False的列表m
        m = [False]

        # 定义一个回调函数cb，将m[0]设置为True
        def cb(r):
            m[0] = True

        # 创建一个x对象的弱引用wref，指定回调函数cb
        wref = weakref.ref(x, cb)

        # 删除x对象
        del x

        # 断言m[0]为True
        self.assertTrue(m[0])

        # 断言wref()返回None，表示弱引用已经失效
        self.assertEqual(wref(), None)

    # 跳过TorchDynamo平台的测试，因为这些测试不适用于TorchDynamo
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_storage_weakref_dealloc(self):
        # 创建一个大小为2的UntypedStorage对象x
        x = torch.UntypedStorage(2)

        # 创建一个包含False的列表m
        m = [False]

        # 定义一个回调函数cb，将m[0]设置为True
        def cb(r):
            m[0] = True

        # 创建一个x对象的弱引用wref，指定回调函数cb
        wref = weakref.ref(x, cb)

        # 删除x对象
        del x

        # 断言m[0]为True
        self.assertTrue(m[0])

        # 断言wref()返回None，表示弱引用已经失效
        self.assertEqual(wref(), None)
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # 标记为跳过 TorchDynamo 的测试，因为这不适合 TorchDynamo
    def test_tensor_cycle_via_dict(self):
        # 创建跟踪器对象 m1 和对应的张量对象 t1
        m1, t1 = Tracker.make()
        # 创建一个空的张量对象 x
        x = torch.empty(2)
        # 将张量对象 x 的跟踪器属性设置为 t1
        x._tracker = t1
        # 删除 t1 变量

        del t1

        # 创建另一个跟踪器对象 m2 和对应的张量对象 t2
        m2, t2 = Tracker.make()
        # 创建一个空的张量对象 y
        y = torch.empty(2)
        # 将张量对象 y 的跟踪器属性设置为 t2
        y._tracker = t2
        # 删除 t2 变量

        del t2

        # 设置张量对象 x 和 y 之间的循环引用关系
        x._loop = y
        y._loop = x

        # 创建一个新的张量对象 z
        # 将 z.grad 属性设置为 x，这创建了一个 C++ 引用循环
        # 该循环不会被 Python GC 中断，因为它完全在 C++ 中完成
        z = torch.empty(2)
        z.grad = x

        # 删除变量 x 和 y
        del x
        del y

        # 手动触发 GC 垃圾回收
        gc.collect()

        # 断言 m1[0] 和 m2[0] 都为 False
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])

        # 禁用 GC
        with disable_gc():
            # 删除变量 z
            del z
            # 断言 m1[0] 和 m2[0] 都为 False
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        # 再次手动触发 GC 垃圾回收
        gc.collect()

        # 断言 m1[0] 和 m2[0] 都为 True
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # 标记为跳过 TorchDynamo 的测试，因为这不适合 TorchDynamo
    def test_storage_cycle_via_dict(self):
        # 创建跟踪器对象 m1 和对应的张量存储对象 t1
        m1, t1 = Tracker.make()
        # 创建一个未类型化的存储对象 x，容量为 2
        x = torch.UntypedStorage(2)
        # 将存储对象 x 的跟踪器属性设置为 t1
        x._tracker = t1
        # 删除 t1 变量

        del t1

        # 创建另一个跟踪器对象 m2 和对应的张量存储对象 t2
        m2, t2 = Tracker.make()
        # 创建一个未类型化的存储对象 y，容量为 2
        y = torch.UntypedStorage(2)
        # 将存储对象 y 的跟踪器属性设置为 t2
        y._tracker = t2
        # 删除 t2 变量

        del t2

        # 设置存储对象 x 和 y 之间的循环引用关系
        x._loop = y
        y._loop = x

        # 创建一个新的未类型化存储对象 z
        # 将 z.grad 属性设置为 x，这创建了一个 C++ 引用循环
        # 该循环不会被 Python GC 中断，因为它完全在 C++ 中完成
        z = torch.UntypedStorage(2)
        z.grad = x

        # 删除变量 x 和 y
        del x
        del y

        # 手动触发 GC 垃圾回收
        gc.collect()

        # 断言 m1[0] 和 m2[0] 都为 False
        self.assertFalse(m1[0])
        self.assertFalse(m2[0])

        # 禁用 GC
        with disable_gc():
            # 删除变量 z
            del z
            # 断言 m1[0] 和 m2[0] 都为 False
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        # 再次手动触发 GC 垃圾回收
        gc.collect()

        # 断言 m1[0] 和 m2[0] 都为 True
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])
    # 定义一个测试函数，用于测试通过 __slots__ 实现的循环引用处理机制
    def test_tensor_cycle_via_slots(self):
        # 设置标志位，用于检测对象是否被删除
        m1 = [False]
        m2 = [False]

        # 定义继承自 torch.Tensor 的类 SlotTensor1，限定只能有一个额外的实例属性 slot1
        class SlotTensor1(torch.Tensor):
            __slots__ = ['slot1']

            # 对象销毁时执行的方法，标记 m1 为 True
            def __del__(self):
                m1[0] = True

        # 定义继承自 SlotTensor1 的类 SlotTensor2，限定只能有一个额外的实例属性 slot2
        class SlotTensor2(SlotTensor1):
            __slots__ = ['slot2']

            # 对象销毁时执行的方法，标记 m2 为 True
            def __del__(self):
                m2[0] = True

        # 创建 SlotTensor1 的实例 x，传入一个形状为 (2,) 的空张量
        x = SlotTensor1(torch.empty(2))
        # 创建 SlotTensor2 的实例 y，传入一个形状为 (2,) 的空张量
        y = SlotTensor2(torch.empty(2))

        # 设定对象之间的相互引用关系
        x.slot1 = y
        y.slot2 = x

        # 删除对象 x
        del x
        # 禁用垃圾回收，并删除对象 y
        with disable_gc():
            del y
            # 验证 m1 和 m2 仍然为 False
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        # 执行垃圾回收
        gc.collect()
        # 验证 m1 和 m2 变为 True，表示对象已被删除
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    # 定义一个测试函数，用于测试通过 __slots__ 实现的循环引用处理机制（用于 StorageBase 类）
    def test_storage_cycle_via_slots(self):
        # 设置标志位，用于检测对象是否被删除
        m1 = [False]
        m2 = [False]

        # 定义继承自 torch._C.StorageBase 的类 SlotStorage1，限定只能有一个额外的实例属性 slot1
        class SlotStorage1(torch._C.StorageBase):
            __slots__ = ['slot1']

            # 对象销毁时执行的方法，标记 m1 为 True
            def __del__(self):
                m1[0] = True

        # 定义继承自 SlotStorage1 的类 SlotStorage2，限定只能有一个额外的实例属性 slot2
        class SlotStorage2(SlotStorage1):
            __slots__ = ['slot2']

            # 对象销毁时执行的方法，标记 m2 为 True
            def __del__(self):
                m2[0] = True

        # 创建 SlotStorage1 的实例 x，传入一个大小为 2 的未类型化 Storage
        x = SlotStorage1(torch.UntypedStorage(2))
        # 创建 SlotStorage2 的实例 y，传入一个大小为 2 的未类型化 Storage
        y = SlotStorage2(torch.UntypedStorage(2))

        # 设定对象之间的相互引用关系
        x.slot1 = y
        y.slot2 = x

        # 删除对象 x
        del x
        # 禁用垃圾回收，并删除对象 y
        with disable_gc():
            del y
            # 验证 m1 和 m2 仍然为 False
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        # 执行垃圾回收
        gc.collect()
        # 验证 m1 和 m2 变为 True，表示对象已被删除
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])

    # 根据 TorchDynamo 不适用于 hooks 的情况，将此测试移动到 test_autograd 中
    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_backward_hooks_traverse(self):
        # 创建 Tracker 实例 m1 和 m2，用于检测对象是否被删除
        m1, t1 = Tracker.make()
        m2, t2 = Tracker.make()

        # 创建两个 requires_grad=True 的张量 x 和 y
        x = torch.empty(2, requires_grad=True)
        x._tracker = t1
        y = torch.empty(2, requires_grad=True)
        y._tracker = t2

        # 删除 t1 和 t2
        del t1
        del t2

        # 设定对象之间的相互引用关系
        x._backward_hooks = y
        y._backward_hooks = x

        # 删除对象 x
        del x
        # 禁用垃圾回收，并删除对象 y
        with disable_gc():
            del y
            # 验证 m1 和 m2 仍然为 False
            self.assertFalse(m1[0])
            self.assertFalse(m2[0])

        # 执行垃圾回收
        gc.collect()

        # 验证 m1 和 m2 变为 True，表示对象已被删除
        self.assertTrue(m1[0])
        self.assertTrue(m2[0])
    # 根据条件跳过测试，条件是在 TorchDynamo 中存在问题（GitHub 上的 issue #1993）
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    # 测试弱引用在 Tensor 死亡时的行为
    def test_tensor_dead_weak_ref(self):
        # 创建一个空的 Tensor
        x = torch.empty(2)
        # 创建对 x 的弱引用
        w_x = weakref.ref(x)
        # 创建另一个 Tensor
        y = torch.empty(2)
        # 将 y 的梯度设置为 x
        y.grad = x
        # 删除 x 的引用
        del x

        # 重新获取 x，此时 x 应为 None
        x = w_x()
        # 理想情况下，x 应该保持 Tensor 存活。但是 CPython 没有提供足够的钩子来实现这一点。
        # 因此，x 将变成一个未定义的 Tensor。这不是最好的情况，但我们能做到的最好。
        del y

        # 断言会抛出 RuntimeError 异常，因为 x 已经是一个未定义的 Tensor
        self.assertRaises(RuntimeError, lambda: x.sigmoid())

    # 根据条件跳过测试，条件是在 TorchDynamo 中存在问题（GitHub 上的 issue #1993）
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    # 测试弱引用在 Storage 死亡时的行为
    def test_storage_dead_weak_ref(self):
        # 创建一个 UntypedStorage
        x = torch.UntypedStorage(2)
        # 创建对 x 的弱引用
        w_x = weakref.ref(x)
        # 创建一个 Tensor，其数据源于 x
        y = torch.tensor(x)
        # 删除 x 的引用
        del x

        # 重新获取 x，此时 x 应为 None
        x = w_x()
        # 理想情况下，x 应该保持 Storage 存活。但是 CPython 没有提供足够的钩子来实现这一点。
        # 因此，x 将变成一个具有空 StorageImpl 的 Storage。这不是最好的情况，但我们能做到的最好。
        del y

        # 断言会抛出 RuntimeError 异常，因为 x 的 StorageImpl 为空
        self.assertRaisesRegex(RuntimeError, "Got a null Storage", lambda: x[0])
        self.assertRaisesRegex(RuntimeError, "Got a null Storage", lambda: x.float())

    # 测试弱引用在 Tensor 重新复活时的行为
    def test_tensor_resurrected_weak_ref(self):
        # 创建一个空的 Tensor
        x = torch.empty(2)
        # 创建对 x 的弱引用
        w_x = weakref.ref(x)
        # 创建另一个 Tensor
        y = torch.empty(2)
        # 将 y 的梯度设置为 x
        y.grad = x
        # 删除 x 的引用
        del x

        # 重新获取 x，此时 x 应为 None
        x = w_x()
        # 使用 _fix_weakref 手动修复弱引用
        x._fix_weakref()
        del y
        # 调用 Tensor 的 sigmoid 方法，验证 Tensor 是否仍然可用
        x.sigmoid()

    # 测试弱引用在 Storage 重新复活时的行为
    def test_storage_resurrected_weak_ref(self):
        # 创建一个 UntypedStorage
        x = torch.UntypedStorage(2)
        # 创建对 x 的弱引用
        w_x = weakref.ref(x)
        # 创建一个 Tensor，其数据源于 x
        y = torch.tensor(x)
        # 删除 x 的引用
        del x

        # 重新获取 x，此时 x 应为 None
        x = w_x()
        # 使用 _fix_weakref 手动修复弱引用
        x._fix_weakref()
        del y
        # 调用 Tensor 的 float 方法，验证 Storage 是否仍然可用
        x.float()

    # 根据条件跳过测试，条件是在 TorchDynamo 中存在问题（GitHub 上的 issue #1993）
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    # 测试修复弱引用后不会泄漏的情况
    def test_tensor_fix_weakref_no_leak(self):
        import weakref

        called = False

        # 创建一个随机张量
        a = torch.randn(1)

        # 定义回调函数，在弱引用被调用时设置 called 为 True
        def callback(w):
            nonlocal called
            called = True
        # 创建对 a 的弱引用，并注册回调函数
        wa = weakref.ref(a, callback)
        # 手动修复弱引用
        a._fix_weakref()
        del a

        # 断言回调函数被调用
        self.assertTrue(called)

    # 根据条件跳过测试，条件是在 TorchDynamo 中存在问题（GitHub 上的 issue #1993）
    # 测试修复弱引用后不会泄漏的情况
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1993")
    def test_storage_fix_weakref_no_leak(self):
        import weakref

        called = False

        # 创建一个 UntypedStorage
        a = torch.UntypedStorage(1)

        # 定义回调函数，在弱引用被调用时设置 called 为 True
        def callback(w):
            nonlocal called
            called = True
        # 创建对 a 的弱引用，并注册回调函数
        wa = weakref.ref(a, callback)
        # 手动修复弱引用
        a._fix_weakref()
        del a

        # 断言回调函数被调用
        self.assertTrue(called)

    # FIXME: move to test_linalg
    # 这个测试应该移动到 test_linalg 中进行
    @torch.inference_mode()
    # 定义一个测试方法，用于测试多线程下的批量矩阵乘法
    def test_bmm_multithreaded(self):
        # 设置计算设备为 CPU
        device = 'cpu'
        # 获取当前线程数
        num_threads = torch.get_num_threads()

        # 设置线程数为 4
        torch.set_num_threads(4)
        # 批量大小列表
        batch_sizes = [1, 10]
        # 矩阵维度
        M, N, O = 23, 8, 12
        # 数据类型为 32 位浮点数
        dtype = torch.float32
        # 对应的 NumPy 数据类型
        numpy_dtype = dtype

        # 定义一个函数用于反转排列顺序
        def invert_perm(p):
            # 构建从值到索引的映射字典
            d = {x: i for i, x in enumerate(p)}
            # 返回反转后的排列
            return (d[0], d[1], d[2])

        # 定义一个生成输入数据的函数，根据给定的批次数生成不同类型的张量
        def generate_inputs(num_batches):
            # 生成转置后的张量对
            for perm1, perm2 in itertools.product(itertools.permutations((0, 1, 2)), repeat=2):
                # 生成随机张量 b1 和 b2，并对其进行排列操作
                b1 = make_tensor((num_batches, M, N), dtype=dtype, device=device, low=-1, high=1)
                b2 = make_tensor((num_batches, N, O), dtype=dtype, device=device, low=-1, high=1)
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                yield b1, b2
            
            # 生成广播操作后的张量对
            for b1, b2, b3, b4, b5, b6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)
                shape2 = (num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, M, N)
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1).expand(num_batches, N, O)
                yield b1, b2
            
            # 生成零大小的张量对
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = torch.randn(shape1, dtype=dtype, device=device)
                b2 = torch.randn(shape2, dtype=dtype, device=device)
                yield b1, b2

        try:
            # 遍历不同的批量大小
            for num_batches in batch_sizes:
                # 遍历生成的输入数据对及其排列方式
                for (b1, b2), perm3 in itertools.product(generate_inputs(num_batches), itertools.permutations((0, 1, 2))):
                    # 计算 b1 和 b2 的矩阵乘法结果 res1
                    res1 = torch.bmm(b1, b2)
                    # 创建一个全为 NaN 的张量 res2，并对其进行排列操作
                    res2 = torch.full((num_batches, M, O), math.nan, dtype=dtype, device=device) \
                        .permute(perm3).contiguous().permute(invert_perm(perm3))
                    # 将 b1 和 b2 的乘法结果存入 res2
                    torch.bmm(b1, b2, out=res2)
                    # 计算期望的乘法结果 expect，使用 NumPy 进行计算，并转换为指定设备和数据类型
                    expect = torch.from_numpy(
                        b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()).to(device=device, dtype=dtype)
                    # 断言 res1 和 res2 与期望结果 expect 相等
                    self.assertEqual(expect, res1)
                    self.assertEqual(expect, res2)
        finally:
            # 恢复原来的线程数设置
            torch.set_num_threads(num_threads)

    # 定义一个测试方法，测试共轭和负数操作转换为列表的功能
    def test_conj_neg_tolist(self):
        # 生成一个随机复数张量 x
        x = torch.randn(2, dtype=torch.cfloat)
        # 对 x 进行共轭操作得到 y1
        y1 = x.conj()
        # 期望的共轭物理操作结果
        y1_expect = x.conj_physical()
        # 获取 y1 的虚部 y2
        y2 = y1.imag
        # 断言 y1 等于其期望转换为列表后的结果
        self.assertEqual(y1, y1_expect.tolist())
        # 断言 y2 等于 y1 的虚部转换为列表后的结果
        self.assertEqual(y2, y1_expect.imag.tolist())
    # 使用 unittest 模块的装饰器 @unittest.skipIf 来标记该测试方法，在 CUDA 构建已启用时跳过执行
    @unittest.skipIf(torch.backends.cuda.is_built(), "Skipped for cuda-enabled build")
    def test_no_cuda_monkeypatch(self):
        # 注意：由于整个文件在 CUDA 不可用时被跳过，因此这个测试方法并不位于 test_cuda.py 文件中。
        
        # 断言在尝试实例化虚基类 Stream 时抛出 RuntimeError 异常，并检查异常消息是否包含指定文本
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class Stream"):
            torch.cuda.Stream()

        # 断言在尝试实例化虚基类 Event 时抛出 RuntimeError 异常，并检查异常消息是否包含指定文本
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class Event"):
            torch.cuda.Event()

        # 断言在尝试实例化虚基类 CUDAGraph 时抛出 RuntimeError 异常，并检查异常消息是否包含指定文本
        with self.assertRaisesRegex(RuntimeError, "Tried to instantiate dummy base class CUDAGraph"):
            torch.cuda.graphs.CUDAGraph()

    def test_tensor_where_scalar(self):

        # 创建一个包含四个元素的张量 a，值从 0 到 3
        a = torch.arange(4.0)
        not_zero = 0.001

        # 使用 torch.where 函数生成张量 b，其中 not_zero 是一个标量参数
        b = torch.where(a != 0, a, not_zero)
        # 使用 Tensor.where 方法生成张量 c，其中 not_zero 是一个标量参数
        c = a.where(a != 0, not_zero)

        # 断言张量 b 和张量 c 相等
        self.assertEqual(b, c)

    def test_data_ptr_of_empty_tensor_with_storage(self):
        # 创建一个形状为 (2, 2) 的空张量 t
        t = torch.empty((2, 2))
        # 断言张量 t 的数据指针不等于 0
        self.assertNotEqual(t.data_ptr(), 0)
        # 将张量 t 重新调整形状为 (0, 2)
        t.resize_((0, 2))
        # 断言张量 t 的数据指针等于 0
        self.assertEqual(t.data_ptr(), 0)

    def test_data_ptr_of_empty_view_with_storage(self):
        # 创建一个形状为 (2, 2) 的空张量 t
        t = torch.empty((2, 2))
        # 断言张量 t 的数据指针不等于 0
        self.assertNotEqual(t.data_ptr(), 0)
        # 通过切片和视图操作创建一个空视图 t2，形状为 (0, 1)
        t2 = t[0:0].view(0, 1)
        # 断言空视图 t2 的数据指针等于 0
        self.assertEqual(t2.data_ptr(), 0)

    def test_size_stride(self) -> None:
        # 创建一个形状为 (2, 3)、数据类型为 torch.float32 的随机张量 t
        t = torch.rand(2, 3, dtype=torch.float32)
        # 断言张量 t 在维度 0 上的大小为 2
        self.assertEqual(t.size(0), 2)
        # 断言张量 t 的大小为 torch.Size([2, 3])
        self.assertEqual(t.size(dim=None), torch.Size([2, 3]))
        # 断言张量 t 的步长为 torch.Size([3, 1])
        self.assertEqual(t.stride(dim=None), torch.Size([3, 1]))
        # 断言张量 t 的转置张量 t.t() 的步长为 torch.Size([1, 3])
        self.assertEqual(t.t().stride(), torch.Size([1, 3]))

    def test_invalid_arg_error_handling(self) -> None:
        """ Tests that errors from old TH functions are propagated back """
        # 遍历无效参数 [-1, 2**65]，分别断言调用 torch.set_num_threads 和 torch.set_num_interop_threads 时抛出 RuntimeError 异常
        for invalid_val in [-1, 2**65]:
            self.assertRaises(RuntimeError, lambda: torch.set_num_threads(invalid_val))
            self.assertRaises(RuntimeError, lambda: torch.set_num_interop_threads(invalid_val))

    def _get_tensor_prop(self, t):
        # 保存张量 t 的 id 和引用计数（如果在测试 Torch Dynamo 框架时，引用计数会被修改）
        preserved = (
            id(t),
            0 if TEST_WITH_TORCHDYNAMO else sys.getrefcount(t),
        )
        # 获取张量 t 类的序列化名称列表
        slotnames = copyreg._slotnames(t.__class__)
        # 获取张量 t 的移动状态，包括其字典、属性名和各个属性值
        moved = (
            slotnames,
            id(t.__dict__),
            tuple(t.__dict__.keys()),
            [getattr(t, name, None) for name in slotnames]
        )
        # 返回保存的状态信息和移动状态信息的元组
        return preserved, moved
    # 定义一个内部方法，用于检查并交换两个张量的属性和值
    def _checked_swap(self, t1, t2):
        # 获取张量 t1 的属性：是否持久化和是否移动
        t1_pres, t1_moved = self._get_tensor_prop(t1)
        # 获取张量 t2 的属性：是否持久化和是否移动
        t2_pres, t2_moved = self._get_tensor_prop(t2)

        # 使用 torch.utils.swap_tensors 方法交换 t1 和 t2 的实际数据
        torch.utils.swap_tensors(t1, t2)

        # 再次获取交换后的张量 t1 和 t2 的属性
        new_t1_pres, new_t1_moved = self._get_tensor_prop(t1)
        new_t2_pres, new_t2_moved = self._get_tensor_prop(t2)
        
        # 断言交换后张量 t1 和 t2 的属性是否符合预期
        self.assertEqual(t1_pres, new_t1_pres)
        self.assertEqual(t2_pres, new_t2_pres)
        self.assertEqual(t1_moved, new_t2_moved)
        self.assertEqual(t2_moved, new_t1_moved)

        # 进行额外的验证，确保在交换张量后，张量的 PyObject slots 被正确地交换了
        # 通过填充张量并比较其返回的引用来验证
        self.assertEqual(id(t1.fill_(0.5)), id(t1))
        self.assertEqual(id(t2.fill_(0.5)), id(t2))

    # 如果当前测试环境为 TorchDynamo，则跳过该测试
    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo adds weakrefs")
    def test_swap_basic(self):
        # 创建多个张量的列表，用于交换测试
        ts = [
            torch.rand(2),
            torch.rand(3, 3),
            torch.empty(3, dtype=torch.int),
            TwoTensor(torch.rand(4), torch.rand(4))  # TwoTensor 是一个自定义的张量类型
        ]

        # 遍历所有可能的张量组合
        for t1, t2 in itertools.combinations(ts, 2):
            # 对每个张量进行克隆，以保留原始张量的状态
            t1 = t1.clone()
            t2 = t2.clone()
            # 为 t2 添加一个自定义属性 foo
            t2.foo = "bar"
            # 创建一个列表来保存 t1 的引用
            holder = []
            holder.append(t1)

            # 调用 _checked_swap 方法来交换 t1 和 t2
            self._checked_swap(t1, t2)

            # 断言 holder 中的第一个元素仍然是 t1
            self.assertIs(holder[0], t1)
            # 断言 t1 的自定义属性 foo 的值为 "bar"
            self.assertEqual(t1.foo, "bar")

            # 如果 t1 是浮点型张量，则进行进一步的验证
            if t1.is_floating_point():
                # 克隆 t1 并将 requires_grad 设置为 True
                t3 = t1.clone().detach().requires_grad_(True)
                # 执行张量操作
                out = t3 * 2
                # 再次交换 t3 和 t2
                torch.utils.swap_tensors(t3, t2)
                # 使用断言验证是否抛出特定异常
                with self.assertRaisesRegex(RuntimeError, "AccumulateGrad node that was poisoned by swap_tensors"):
                    out.sum().backward()

            # 创建 t1 的弱引用
            wr = weakref.ref(t1)
            # 使用断言验证是否抛出特定异常，确保在存在弱引用时交换失败
            with self.assertRaisesRegex(RuntimeError, "has weakref"):
                torch.utils.swap_tensors(t1, t2)

    # 如果当前测试环境为 TorchDynamo，则跳过该测试
    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "Dynamo adds weakrefs")
    # 定义一个测试方法，用于测试在不同情况下交换张量对象的功能
    def test_swap_fail_slots(self):
        # 定义一个继承自 TwoTensor 的子类，指定两个特定的 __slots__
        class MyTwoTensor(TwoTensor):
            __slots__ = ("a", "b")

        # 定义另一个继承自 TwoTensor 的子类，特定的 __slots__ 顺序与上一个相反
        class MyTwoTensor2(TwoTensor):
            __slots__ = ("b", "a")

        # 定义另一个继承自 TwoTensor 的子类，指定四个特定的 __slots__
        class MyTwoTensor3(TwoTensor):
            __slots__ = ("a", "b", "c", "d")

        # 定义另一个继承自 TwoTensor 的子类，指定两个特定的 __slots__ 中间有一个不同
        class MyTwoTensor4(TwoTensor):
            __slots__ = ("a", "c")

        # 创建一个普通的 PyTorch 张量对象
        t1 = torch.rand(4)
        
        # 创建一个自定义的 TwoTensor 对象，包含两个张量属性
        t2 = TwoTensor(torch.rand(4), torch.rand(4))
        
        # 创建一个特定 __slots__ 的 MyTwoTensor 对象
        t3 = MyTwoTensor(torch.rand(4), torch.rand(4))
        
        # 创建另一个特定 __slots__ 的 MyTwoTensor 对象，与 t3 类似
        t4 = MyTwoTensor(torch.rand(4), torch.rand(4))
        
        # 创建一个与 t3 不同 __slots__ 顺序的 MyTwoTensor2 对象
        t5 = MyTwoTensor2(torch.rand(4), torch.rand(4))
        
        # 创建一个具有额外两个 __slots__ 的 MyTwoTensor3 对象
        t6 = MyTwoTensor3(torch.rand(4), torch.rand(4))
        
        # 再次创建具有相同 __slots__ 的 MyTwoTensor3 对象
        t7 = MyTwoTensor3(torch.rand(4), torch.rand(4))
        
        # 创建一个与 t3 共享一个 __slots__ 但不同的 MyTwoTensor4 对象
        t8 = MyTwoTensor4(torch.rand(4), torch.rand(4))

        # 调用测试方法，试图交换 t1 和 t2（普通张量和 TwoTensor），预期抛出 RuntimeError 异常
        self._checked_swap(t1, t2)
        
        # 使用 assertRaisesRegex 验证尝试交换 t1 和 t3 时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t1, t3)
        
        # 使用 assertRaisesRegex 验证尝试交换 t2 和 t3 时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t2, t3)
        
        # 使用 assertRaisesRegex 验证尝试交换 t2 和 t8 时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t2, t8)
        
        # 调用测试方法，成功交换 t3 和 t4（相同 __slots__ 的 MyTwoTensor 对象）
        self._checked_swap(t3, t4)
        
        # 调用测试方法，成功交换 t3 和 t5（不同顺序的相同 __slots__ 的 MyTwoTensor 和 MyTwoTensor2 对象）
        self._checked_swap(t3, t5)
        
        # 使用 assertRaisesRegex 验证尝试交换 t3 和 t6 时抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Cannot swap t1 and t2 if they have different slots"):
            torch.utils.swap_tensors(t3, t6)
        
        # 修改 t3 对象的一个额外属性值
        t3.c = "foo"
        
        # 修改 t4 对象的一个额外属性值
        t4.d = "bar"
        
        # 调用测试方法，成功交换 t3 和 t4，验证交换后 t4 的属性 c 是否为 "foo"
        self._checked_swap(t3, t4)
        self.assertEqual(t4.c, "foo")
        
        # 验证交换后 t3 的属性 d 是否为 "bar"
        self.assertEqual(t3.d, "bar")
        
        # 修改 t6 对象的一个额外属性值
        t6.c = "cat"
        
        # 修改 t7 对象的一个额外属性值
        t7.d = "dog"
        
        # 调用测试方法，成功交换 t6 和 t7，验证交换后 t6 的属性 c 是否为 "cat"
        self._checked_swap(t6, t7)
# 扩展 TestTorch 类，增加负维度包装测试
# FIXME: 用 OpInfo 的示例输入或系统化的 OpInfo 测试替换这些
# 测试负维度包装的函数

# 方法类型常量定义
METHOD = 1
INPLACE_METHOD = 2
FUNCTIONAL = 4

# 维度参数的占位符
DIM_ARG: None = None

# 创建负维度测试的函数，返回一个测试函数
def make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim=0):
    def neg_dim_test(self):
        # 根据 tensor_arg 的类型创建输入张量 x，并确定其维度 ndim
        if isinstance(tensor_arg, list):
            assert METHOD not in types and INPLACE_METHOD not in types
            x = [torch.randn(arg) for arg in tensor_arg]
            ndim = len(tensor_arg[-1])
        else:
            x = torch.randn(*tensor_arg)
            ndim = len(tensor_arg)
        ndim += extra_dim  # 添加额外的维度

        # 计算需要测试的维度数 n_dim_to_test
        n_dim_to_test = sum(e is DIM_ARG for e in arg_constr())

        # 对于每组维度值的组合，执行测试
        for dims_val in combinations(range(ndim), n_dim_to_test):
            arg = arg_constr()  # 构造参数列表
            arg_neg = copy.deepcopy(arg)  # 深拷贝参数列表
            idx = 0
            # 替换参数列表中的 DIM_ARG 为实际的维度值和负维度值
            for i, v in enumerate(arg):
                if v is DIM_ARG:
                    arg[i] = dims_val[idx]
                    arg_neg[i] = dims_val[idx] - ndim
                    idx += 1

            # 如果方法类型包含 METHOD
            if METHOD in types:
                a = getattr(x, name)(*arg)  # 调用张量方法并获取结果 a
                b = getattr(x, name)(*arg_neg)  # 使用负维度参数调用方法并获取结果 b
                self.assertEqual(a, b)  # 断言结果相等

            # 如果方法类型包含 INPLACE_METHOD
            if INPLACE_METHOD in types:
                a = x.clone()  # 克隆张量 x
                getattr(a, name + '_')(*arg)  # 调用张量的原地方法
                b = x.clone()  # 再次克隆张量 x
                getattr(b, name + '_')(*arg_neg)  # 使用负维度参数调用原地方法
                self.assertEqual(a, b)  # 断言结果相等

            # 如果方法类型包含 FUNCTIONAL
            if FUNCTIONAL in types:
                a = getattr(torch, name)(x, *arg)  # 调用 Torch 的函数式方法并获取结果 a
                b = getattr(torch, name)(x, *arg_neg)  # 使用负维度参数调用函数式方法并获取结果 b
                self.assertEqual(a, b)  # 断言结果相等

    return neg_dim_test  # 返回测试函数

# 创建返回随机索引张量的函数
def idx_tensor(size, max_val):
    return torch.LongTensor(*size).random_(0, max_val - 1)

# 添加负维度测试函数的入口函数
def add_neg_dim_tests():
    # 定义一个包含多个测试样例的列表，每个样例包含测试名称、输入维度、参数生成函数、期望的方法列表
    neg_dim_tests = [
        ('narrow', (10, 20, 30), lambda: [DIM_ARG, 0, 5], [METHOD]),
        # 测试名称为'narrow'，输入维度为(10, 20, 30)，参数生成函数返回[DIM_ARG, 0, 5]，期望的方法列表为[METHOD]
    
        ('transpose', (10, 20, 30), lambda: [DIM_ARG, DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        # 测试名称为'transpose'，输入维度为(10, 20, 30)，参数生成函数返回[DIM_ARG, DIM_ARG]，期望的方法列表为[METHOD, INPLACE_METHOD, FUNCTIONAL]
    
        ('size', (10, 20, 30), lambda: [DIM_ARG], [METHOD]),
        # 测试名称为'size'，输入维度为(10, 20, 30)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD]
    
        ('cat', [(2, 3, 4), (2, 3, 4)], lambda: [DIM_ARG], [FUNCTIONAL]),
        # 测试名称为'cat'，输入维度为[(2, 3, 4), (2, 3, 4)]，参数生成函数返回[DIM_ARG]，期望的方法列表为[FUNCTIONAL]
    
        ('chunk', (10, 20, 30), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'chunk'，输入维度为(10, 20, 30)，参数生成函数返回[5, DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('gather', (10, 20), lambda: [DIM_ARG, idx_tensor((10, 20), 10)], [METHOD, FUNCTIONAL]),
        # 测试名称为'gather'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG, idx_tensor((10, 20), 10)]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('index_select', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10)], [METHOD, FUNCTIONAL]),
        # 测试名称为'index_select'，输入维度为(10, 10)，参数生成函数返回[DIM_ARG, idx_tensor((10,), 10)]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('split', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'split'，输入维度为(10, 20)，参数生成函数返回[5, DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('squeeze', (10, 1, 20, 1), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
        # 测试名称为'squeeze'，输入维度为(10, 1, 20, 1)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, INPLACE_METHOD, FUNCTIONAL]
    
        ('unbind', (2, 3, 4), lambda: [DIM_ARG], [FUNCTIONAL]),
        # 测试名称为'unbind'，输入维度为(2, 3, 4)，参数生成函数返回[DIM_ARG]，期望的方法列表为[FUNCTIONAL]
    
        ('unsqueeze', (10, 20), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL], 1),
        # 测试名称为'unsqueeze'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, INPLACE_METHOD, FUNCTIONAL]，期望的附加参数为1
    
        ('logcumsumexp', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'logcumsumexp'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('cumprod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'cumprod'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('cumsum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'cumsum'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('cummax', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'cummax'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('cummin', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'cummin'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('mean', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'mean'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('median', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'median'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('nanmedian', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'nanmedian'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('mode', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'mode'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('norm', (10, 20), lambda: [2, DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'norm'，输入维度为(10, 20)，参数生成函数返回[2, DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('prod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'prod'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('std', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'std'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('sum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'sum'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('var', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'var'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('kthvalue', (10, 20), lambda: [3, DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'kthvalue'，输入维度为(10, 20)，参数生成函数返回[3, DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('max', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'max'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD, FUNCTIONAL]
    
        ('min', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
        # 测试名称为'min'，输入维度为(10, 20)，参数生成函数返回[DIM_ARG]，期望的方法列表为[METHOD,
    # 对于每个测试声明在 neg_dim_tests 列表中进行迭代
    for decl in neg_dim_tests:
        # 如果声明的长度为4，解构赋值
        if len(decl) == 4:
            name, tensor_arg, arg_constr, types = decl
            # 针对额外维度设置默认值为0
            extra_dim = 0
        # 如果声明的长度为5，解构赋值
        elif len(decl) == 5:
            name, tensor_arg, arg_constr, types, extra_dim = decl

        # 根据声明的名称生成测试名称
        test_name = 'test_' + name + '_neg_dim'

        # 确保在 TestTorch 类中不存在同名的测试函数
        assert not hasattr(TestTorch, test_name), "Duplicated test name: " + test_name
        # 动态地将生成的测试函数设置为 TestTorch 类的属性，使用 make_neg_dim_test 函数创建
        setattr(TestTorch, test_name, make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim))
# TODO: 这些空的类是为了 XLA 兼容性暂时实例化的
# 一旦 XLA 更新他们的测试套件，应该移除这些类

# 生成测试用例
# 注意：测试生成必须在文件范围内进行，而不是在主函数内进行，否则 pytest 会失败
add_neg_dim_tests()  # 添加负维度测试
instantiate_device_type_tests(TestViewOps, globals())  # 实例化测试视图操作的设备类型测试
instantiate_device_type_tests(TestVitalSignsCuda, globals())  # 实例化测试 CUDA 下的 VitalSigns
instantiate_device_type_tests(TestTensorDeviceOps, globals())  # 实例化测试张量设备操作的设备类型测试
instantiate_device_type_tests(TestTorchDeviceType, globals())  # 实例化测试 Torch 设备类型的设备类型测试
instantiate_device_type_tests(TestDevicePrecision, globals(), except_for='cpu')  # 实例化测试设备精度的设备类型测试，排除 CPU

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True  # 设置默认的数据类型检查为启用状态
    run_tests()  # 运行所有测试用例
```