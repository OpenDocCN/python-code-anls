# `.\pytorch\test\test_view_ops.py`

```py
# Owner(s): ["module: tests"]
import random  # 导入随机数生成模块

import unittest  # 导入单元测试框架
from functools import partial  # 导入偏函数功能
from itertools import combinations, permutations, product  # 导入组合、排列、笛卡尔积生成器

import numpy as np  # 导入 NumPy 库

import torch  # 导入 PyTorch 库

from torch.testing import make_tensor  # 导入创建测试张量的函数
from torch.testing._internal.common_device_type import (  # 导入测试设备类型相关函数
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    onlyNativeDeviceTypes,
    skipLazy,
    skipMeta,
    skipXLA,
)
from torch.testing._internal.common_dtype import (  # 导入数据类型相关函数
    all_types_and,
    all_types_and_complex_and,
    complex_types,
    floating_and_complex_types_and,
)
from torch.testing._internal.common_utils import (  # 导入通用工具函数
    gradcheck,
    gradgradcheck,
    IS_FBCODE,
    numpy_to_torch_dtype_dict,
    run_tests,
    skipIfTorchDynamo,
    suppress_warnings,
    TestCase,
)


# TODO: replace this with make_tensor() in common_utils.py
# 根据指定的形状、数据类型、设备生成输入张量
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)  # 创建空张量
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # 处理浮点数和复数类型
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)  # 转换为 bfloat16 类型
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(
                    30, 100
                )
            x[torch.randn(*shape) > 0.5] = 0  # 随机将部分元素设为零
            if with_extremal and dtype.is_floating_point:
                # 使用极端值
                x[torch.randn(*shape) > 0.5] = float("nan")  # NaN
                x[torch.randn(*shape) > 0.5] = float("inf")  # 正无穷
                x[torch.randn(*shape) > 0.5] = float("-inf")  # 负无穷
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex("nan")  # NaN
                x[torch.randn(*shape) > 0.5] = complex("inf")  # 正无穷
                x[torch.randn(*shape) > 0.5] = complex("-inf")  # 负无穷
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True  # 随机将部分元素设为 True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)  # 生成整数张量

    return x


# TODO: replace this with make_tensor() in common_utils.py
# 根据指定的维度数、最小和最大尺寸生成随机形状
def _rand_shape(dim, min_size, max_size):
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)


# TODO: refactor tests to avoid this function
# 当设备为 CPU 且数据类型为 half/bfloat16 时将其转换为 float 类型
def _convert_t(dtype, device):
    if device == "cpu" and dtype in {torch.half, torch.bfloat16}:
        return torch.float
    return dtype


# TODO: replace this with make_tensor() in common_utils.py
# 返回请求形状、数据类型和设备的张量
# 在请求半精度 CPU 张量时返回可以由半精度表示的浮点 CPU 张量。
# 初始化使用 randint 用于非浮点类型，使用 randn 用于浮点类型。
# 返回一个填充了一的张量
def _make_tensor(shape, dtype, device, fill_ones=False) -> torch.Tensor:
    if fill_ones:
        # 如果 fill_ones 为 True，则返回一个形状为 shape，数据类型为 dtype，设备为 device 的张量，所有元素为 1
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    if not (dtype.is_floating_point or dtype.is_complex):
        # 如果 dtype 不是浮点数或复数，则生成一个在 [0, 10) 范围内的随机整数张量，形状为 shape，设备为 device
        t = torch.randint(0, 10, shape, device=device)
        if dtype != torch.uint8:
            # 如果 dtype 不是 uint8，则将生成的张量减去 5，以生成负值
            t = t - 5
        return t.to(_convert_t(dtype, device))  # 转换为指定的 dtype 和 device 类型

    if dtype == torch.half and device == "cpu":
        # 如果 dtype 是半精度（half）并且设备是 CPU，则生成一个随机浮点数张量，然后转换为半精度
        return torch.randn(*shape, dtype=torch.float, device=device).half().float()
    if dtype == torch.bfloat16 and device == "cpu":
        # 如果 dtype 是 bfloat16 并且设备是 CPU，则生成一个随机浮点数张量，然后转换为 bfloat16
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16().float()

    # 默认情况下，生成一个随机浮点数张量，形状为 shape，数据类型为 dtype，设备为 device
    return torch.randn(shape, dtype=dtype, device=device).to(dtype=dtype)


# Tests ops and indexing to ensure they return views (and new tensors) as
# appropriate.
class TestViewOps(TestCase):
    exact_dtype = True

    def is_view_of(self, base, other):
        # 检查 other 是否是 base 的视图，并且它们不是同一个对象
        if (
            not other._is_view()
            or other is base
            or other._base is not base
            or base.device != other.device
        ):
            return False
        # 注意：只有在本机设备类型上验证存储，因为某些加速器（如 XLA）不公开存储
        if base.device.type == "cpu" or base.device.type == "cuda":
            if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                return False

        return True

    def is_view_of_same_base(self, v1, v2):
        # 返回 True，如果 v1 和 v2 是同一个基张量的视图
        if not v1._is_view() or v1 is v2:
            return False
        return self.is_view_of(v1._base, v2)

    def _do_transpose(self, x, contiguous=False, dim0=0, dim1=1):
        # 如果 contiguous 为 True，则执行转置操作，否则返回原始张量 x
        if contiguous:
            return x
        else:
            return x.transpose(dim0, dim1)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_conj_self(self, device, dtype):
        # 在设备 device 上创建一个形状为 (5, 5) 的张量 t，所有元素为 1
        t = torch.ones(5, 5, device=device)
        # 对张量 t 执行共轭操作，返回结果张量 s
        s = t.conj()
        self.assertTrue(s is t)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    # 测试视图的错误检查，当视图的数据类型比原始数据类型具有更大的元素大小时
    # 定义一个测试函数，用于测试视图数据类型扩展时的错误情况
    def test_view_dtype_upsize_errors(self, device, dtype):
        # 获取指定数据类型的字节大小
        dtype_size = torch._utils._element_size(dtype)

        # 遍历所有可能的视图数据类型，包括半精度、bfloat16和布尔类型
        for view_dtype in all_types_and_complex_and(
            torch.half, torch.bfloat16, torch.bool
        ):
            # 获取当前视图数据类型的字节大小
            view_dtype_size = torch._utils._element_size(view_dtype)
            # 如果当前视图数据类型的大小小于等于原数据类型的大小，则跳过
            if view_dtype_size <= dtype_size:
                continue

            # 计算当前视图数据类型相对于原数据类型的大小比率
            size_ratio = view_dtype_size // dtype_size

            # 创建一个张量，形状为(4, 4, size_ratio + 1)，指定数据类型和设备，并在指定范围内随机初始化
            a = make_tensor(
                (4, 4, size_ratio + 1), dtype=dtype, device=device, low=-5, high=5
            )
            # 断言视图操作抛出指定异常，验证尺寸要求
            with self.assertRaisesRegex(
                RuntimeError, rf"self.size\(-1\) must be divisible by {size_ratio}"
            ):
                a.view(view_dtype)

            # 创建一个张量，形状为(4, 4, size_ratio)，指定数据类型和设备，并在指定范围内随机初始化
            a = make_tensor(
                (4, 4, size_ratio), dtype=dtype, device=device, low=-5, high=5
            )
            # 通过扩展视图创建张量，并指定步长，验证扩展操作抛出指定异常
            a = a.as_strided((4, 4, size_ratio), (size_ratio, 1, 1))
            with self.assertRaisesRegex(
                RuntimeError, rf"self.stride\(1\) must be divisible by {size_ratio}"
            ):
                a.view(view_dtype)

    @onlyNativeDeviceTypes


这段代码是一个用于测试视图数据类型扩展时可能出现的错误情况的测试函数。
    # 定义一个测试函数，用于测试 torch.view_as_complex 方法
    def test_view_as_complex(self, device):
        # 定义内部函数 fn，用于执行具体的测试
        def fn(contiguous_input=True, dim0=0, dim1=1):
            # 创建一个随机张量 t，形状为 (3, 2, 2)，并指定设备
            t = torch.randn(3, 2, 2, device=device)
            # 生成复数张量 c_t，通过取 t 的第三维度的前两个切片分别作为实部和虚部
            c_t = t[:, :, 0] + 1j * t[:, :, 1]

            # 调用 self._do_transpose 方法对 t 进行转置操作
            input = self._do_transpose(t, contiguous_input, dim0, dim1)

            # 如果 input 的最后一个维度不是 2，则抛出 RuntimeError 异常
            if input.size()[-1] != 2:
                self.assertRaisesRegex(
                    RuntimeError,
                    "Tensor must have a last dimension of size 2",
                    lambda: torch.view_as_complex(input),
                )
                return

            # 如果 input 的最后一个维度的步长不是 1，则抛出 RuntimeError 异常
            if input.stride()[-1] != 1:
                self.assertRaisesRegex(
                    RuntimeError,
                    "Tensor must have a last dimension with stride 1",
                    lambda: torch.view_as_complex(input),
                )
                return

            # 调用 torch.view_as_complex 方法，将 input 转换为复数张量 res
            res = torch.view_as_complex(input)
            # 断言 res 和 c_t 经过相同的转置操作后是否相等
            self.assertEqual(res, self._do_transpose(c_t, contiguous_input, dim0, dim1))
            # 断言 res 是否为 t 的视图
            self.assertTrue(self.is_view_of(t, res))

        # 分别调用 fn 进行不同参数配置的测试
        fn()
        fn(contiguous_input=False)
        # 在 dim0=0, dim1=2 的情况下，期望抛出 RuntimeError 异常
        fn(contiguous_input=False, dim0=0, dim1=2)
        # 在 dim0=1, dim1=2 的情况下，期望抛出 RuntimeError 异常
        fn(contiguous_input=False, dim0=1, dim1=2)

        # 创建一个形状为 (3, 3) 的张量 x
        x = torch.randn(3, 3, device=device)
        # 通过 torch.as_strided 方法创建一个新的张量 t，形状为 (2, 2)，步长为 (1, 1)
        t = torch.as_strided(x, (2, 2), (1, 1))
        # 期望在这种情况下，抛出 RuntimeError 异常，要求张量的非最后一个维度的步长必须能被 2 整除
        self.assertRaisesRegex(
            RuntimeError,
            "Tensor must have a stride divisible by 2 for all but last dimension",
            lambda: torch.view_as_complex(t),
        )

        # 创建一个零元素张量 x，形状为 torch.Size([0])
        x = torch.tensor([], device=device)
        # 期望在这种情况下，抛出 RuntimeError 异常，要求张量的最后一个维度大小为 2
        self.assertRaisesRegex(
            RuntimeError,
            "Tensor must have a last dimension of size 2",
            lambda: torch.view_as_complex(x),
        )

        # 创建一个零维张量 z，值为 2.0
        z = torch.tensor(2.0)
        # 期望在这种情况下，抛出 RuntimeError 异常，要求输入张量至少有一个维度
        self.assertRaisesRegex(
            RuntimeError,
            "Input tensor must have one or more dimensions",
            lambda: torch.view_as_complex(z),
        )

        # 将 x 重塑为形状 (0, 2)，即 torch.Size([0, 2])
        y = x.reshape(0, 2)
        # 将 y 转换为复数张量 res
        res = torch.view_as_complex(y)
        # 断言 res 是否为 x 的视图
        self.assertTrue(self.is_view_of(x, res))
        # 断言 res 的形状是否为 torch.Size([0])
        self.assertEqual(res.shape, torch.Size([0]))

    @onlyNativeDeviceTypes
    @dtypes(*complex_types(), torch.complex32)
    # 定义一个测试函数，用于测试 torch.view_as_real 方法在不同情况下的行为
    def test_view_as_real(self, device, dtype):
        # 定义内部函数 fn，用于测试给定类型和设备的随机张量的视图转换
        def fn(contiguous_input=True):
            # 创建一个指定类型和设备的随机张量
            t = torch.randn(3, 4, dtype=dtype, device=device)
            # 调用 _do_transpose 方法对随机张量进行转置操作
            input = self._do_transpose(t, contiguous_input)
            # 调用 torch.view_as_real 方法将输入张量转换为实部和虚部组成的复数张量
            res = torch.view_as_real(input)
            # 断言转换后的结果的实部与原始输入的实部相等
            self.assertEqual(res[:, :, 0], input.real)
            # 断言转换后的结果的虚部与原始输入的虚部相等
            self.assertEqual(res[:, :, 1], input.imag)
            # 断言转换后的结果是原始随机张量的视图
            self.assertTrue(self.is_view_of(t, res))

        # 调用 fn 内部函数，分别测试 contiguous_input 为 True 和 False 的情况
        fn()
        fn(contiguous_input=False)

        # 创建一个空张量 x，数据类型和设备与输入一致
        x = torch.tensor([], dtype=dtype, device=device)
        # 调用 torch.view_as_real 方法将空张量转换为实部和虚部组成的复数张量
        res = torch.view_as_real(x)
        # 断言转换后的结果是空张量 x 的视图
        self.assertTrue(self.is_view_of(x, res))
        # 断言转换后的结果的形状为 torch.Size([0, 2])
        self.assertEqual(res.shape, torch.Size([0, 2]))

        # 创建一个零维张量 x，包含一个复数元素
        x = torch.tensor(2 + 3j, dtype=dtype, device=device)
        # 调用 torch.view_as_real 方法将零维张量转换为包含实部和虚部的一维张量
        res = torch.view_as_real(x)
        # 断言转换后的结果是原始零维张量 x 的视图
        self.assertTrue(self.is_view_of(x, res))
        # 断言转换后的结果的形状为 torch.Size([2])
        self.assertEqual(res.shape, torch.Size([2]))

    # 标记为仅适用于本地设备类型的测试函数，并指定数据类型为所有类型及复数类型
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_view_tensor_split(self, device, dtype):
        # 创建一个指定形状和数据类型的张量 a
        a = make_tensor((40, 30), dtype=dtype, device=device, low=-9, high=9)
        # 使用 tensor_split 方法在指定维度上对张量 a 进行分割
        a_split_dim0 = a.tensor_split(7, 0)
        # 遍历分割后的张量列表，断言每个分割出来的张量是原始张量 a 的视图
        for a_split_dim0_tensor in a_split_dim0:
            self.assertTrue(self.is_view_of(a, a_split_dim0_tensor))
        # 使用 tensor_split 方法在另一维度上对张量 a 进行分割
        a_split_dim1 = a.tensor_split(7, 1)
        # 遍历分割后的张量列表，断言每个分割出来的张量是原始张量 a 的视图
        for a_split_dim1_tensor in a_split_dim1:
            self.assertTrue(self.is_view_of(a, a_split_dim1_tensor))

    # 标记为仅适用于本地设备类型的测试函数，并指定数据类型为所有类型及复数类型
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_view_tensor_hsplit(self, device, dtype):
        # 创建一个指定形状和数据类型的张量 t
        t = make_tensor((4, 4, 4), dtype=dtype, device=device, low=-9, high=9)
        # 使用 torch.hsplit 方法在第一个维度上对张量 t 进行分割
        t_hsplit = torch.hsplit(t, 2)
        # 遍历分割后的张量列表，断言每个分割出来的张量是原始张量 t 的视图
        for t_hsplit_tensor in t_hsplit:
            self.assertTrue(self.is_view_of(t, t_hsplit_tensor))
        # 修改原始张量 t 的一个元素
        t[2, 2, 2] = 7
        # 断言分割后的第二个张量中的特定元素与原始张量 t 中的对应元素相等
        self.assertEqual(t_hsplit[1][2, 0, 2], t[2, 2, 2])

    # 标记为仅适用于本地设备类型的测试函数，并指定数据类型为所有类型及复数类型
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_view_tensor_vsplit(self, device, dtype):
        # 创建一个指定形状和数据类型的张量 t
        t = make_tensor((4, 4, 4), dtype=dtype, device=device, low=-9, high=9)
        # 使用 torch.vsplit 方法在第一个维度上对张量 t 进行分割
        t_vsplit = torch.vsplit(t, 2)
        # 遍历分割后的张量列表，断言每个分割出来的张量是原始张量 t 的视图
        for t_vsplit_tensor in t_vsplit:
            self.assertTrue(self.is_view_of(t, t_vsplit_tensor))
        # 修改原始张量 t 的一个元素
        t[2, 2, 2] = 7
        # 断言分割后的第二个张量中的特定元素与原始张量 t 中的对应元素相等
        self.assertEqual(t_vsplit[1][0, 2, 2], t[2, 2, 2])

    # 标记为仅适用于本地设备类型的测试函数，并指定数据类型为所有类型
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_view_tensor_dsplit(self, device, dtype):
        # 创建一个指定形状和数据类型的张量 t
        t = make_tensor((4, 4, 4), dtype=dtype, device=device, low=-9, high=9)
        # 使用 torch.dsplit 方法在第一个维度上对张量 t 进行分割
        t_dsplit = torch.dsplit(t, 2)
        # 遍历分割后的张量列表，断言每个分割出来的张量是原始张量 t 的视图
        for t_dsplit_tensor in t_dsplit:
            self.assertTrue(self.is_view_of(t, t_dsplit_tensor))
        # 修改原始张量 t 的一个元素
        t[2, 2, 2] = 7
        # 断言分割后的第二个张量中的特定元素与原始张量 t 中的对应元素相等
        self.assertEqual(t_dsplit[1][2, 2, 0], t[2, 2, 2])
    # 定义一个测试方法，测试对于非复数张量的情况，调用 torch.imag 应引发 RuntimeError 异常
    def test_imag_noncomplex(self, device, dtype):
        # 创建一个全为1的张量，指定设备和数据类型
        t = torch.ones((5, 5), dtype=dtype, device=device)

        # 使用 assertRaises 上下文管理器来断言调用 torch.imag(t) 会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            torch.imag(t)

    # 应用装饰器，只在本地设备类型上运行，测试各种复数数据类型
    @onlyNativeDeviceTypes
    @dtypes(*complex_types())
    # 定义测试方法，测试实部和虚部视图
    def test_real_imag_view(self, device, dtype):
        # 定义一个内部函数，与 NumPy 进行比较
        def compare_with_numpy(contiguous_input=True):
            # 创建一个随机张量，指定设备和数据类型
            t = torch.randn(3, 3, dtype=dtype, device=device)
            if not contiguous_input:
                # 如果输入不连续，则使用转置后的张量
                u = t.T
            else:
                u = t

            # 获取张量的实部
            re = u.real
            # 从 NumPy 数组中创建预期的实部张量，然后转换为指定设备
            exp = torch.from_numpy(u.cpu().numpy().real).to(device=device)
            # 断言实部张量与预期张量相等
            self.assertEqual(re, exp)
            # 断言实部张量是原始张量 t 的视图
            self.assertTrue(self.is_view_of(t, re))

            # 获取张量的虚部
            im = u.imag
            # 从 NumPy 数组中创建预期的虚部张量，然后转换为指定设备
            exp = torch.from_numpy(u.cpu().numpy().imag).to(device=device)
            # 断言虚部张量与预期张量相等
            self.assertEqual(im, exp)
            # 断言虚部张量是原始张量 t 的视图
            self.assertTrue(self.is_view_of(t, im))

        # 调用内部函数，测试连续和非连续输入的情况
        compare_with_numpy()
        compare_with_numpy(contiguous_input=False)

        # 确保正确设置存储偏移量
        a = torch.randn(10, dtype=dtype)
        # 断言切片后的实部与切片后的实部视图相等
        self.assertEqual(a[5:].real, a.real[5:])
        # 断言切片后的虚部与切片后的虚部视图相等
        self.assertEqual(a[5:].imag, a.imag[5:])

    # 应用装饰器，只在本地设备类型上运行，测试共轭和虚部视图
    @onlyNativeDeviceTypes
    @dtypes(*complex_types())
    # 定义测试方法，测试共轭视图
    def test_conj_imag_view(self, device, dtype) -> None:
        # 创建一个指定形状、数据类型和设备的张量
        t = _make_tensor(
            (
                4,
                5,
            ),
            dtype,
            device,
        )
        # 使用 NumPy 的共轭方法创建预期的张量，并转换为指定设备
        t_numpy_conj = torch.from_numpy(t.cpu().numpy().conj()).to(device=device)
        # 调用张量的共轭方法
        v = t.conj()
        # 断言共轭张量是原始张量 t 的视图
        self.assertTrue(self.is_view_of(t, v))
        # 断言共轭张量与预期的 NumPy 共轭张量相等
        self.assertEqual(v, t_numpy_conj)

        # 如果张量是复数类型
        if t.is_complex():
            # 获取共轭张量的虚部
            v_imag = v.imag
            # 断言共轭虚部是原始张量 t 的视图
            self.assertTrue(self.is_view_of(t, v_imag))
            # 断言共轭虚部与预期的 NumPy 共轭虚部相等
            self.assertEqual(v_imag, t_numpy_conj.imag)
            # 断言共轭虚部为负数
            self.assertTrue(v_imag.is_neg())

    # 应用装饰器，只在本地设备类型上运行，测试共轭视图和共享内存
    @onlyNativeDeviceTypes
    # 定义测试方法，测试共轭视图在共享内存情况下的行为
    def test_conj_view_with_shared_memory(self, device) -> None:
        # 创建一个指定形状、数据类型为复数类型的张量
        a = _make_tensor(
            (
                4,
                5,
            ),
            torch.cfloat,
            device,
        )
        # 调用张量的共轭方法
        b = a.conj()
        c = a.conj()

        # 断言张量 a 与张量 b 的加法结果与原地加法操作的结果相等
        self.assertEqual(torch.add(a, b), a.add_(b))
        # 断言张量 b 与张量 c 的加法结果与使用 out 参数进行的加法操作的结果相等
        self.assertEqual(torch.add(b, c), torch.add(b, c, out=a))
        # 断言张量 b 与张量 c 的加法结果与原地加法操作的结果相等
        self.assertEqual(torch.add(b, c), b.add_(c))

    # 应用装饰器，只在本地设备类型上运行，测试复数类型和指定数据类型的组合
    @onlyNativeDeviceTypes
    @dtypes(
        *product(
            complex_types(),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    @suppress_warnings
    # 测试设置张量的实部和虚部，根据给定的设备和数据类型
    def test_set_real_imag(self, device, dtypes):
        # 创建一个形状为 (10,) 的张量 x，使用指定的数据类型和设备
        x = torch.randn(10, dtype=dtypes[0], device=device)

        # 创建一个新的实部张量 new_real 和一个新的虚部张量 new_imag
        new_real = _make_tensor((10,), dtypes[1], device)
        new_imag = _make_tensor((10,), dtypes[1], device)

        # 设置张量 x 的实部和虚部为新创建的张量
        x.real = new_real
        x.imag = new_imag

        # 如果数据类型 dtypes[1] 是复数类型
        if dtypes[1].is_complex:
            # 断言 x 的实部等于 new_real 的实部，允许数据类型不完全匹配
            self.assertEqual(x.real, new_real.real, exact_dtype=False)
            # 断言 x 的虚部等于 new_imag 的实部，允许数据类型不完全匹配
            self.assertEqual(x.imag, new_imag.real, exact_dtype=False)

        else:
            # 如果数据类型不是复数类型，断言 x 的实部等于 new_real，允许数据类型不完全匹配
            self.assertEqual(x.real, new_real, exact_dtype=False)
            # 断言 x 的虚部等于 new_imag，允许数据类型不完全匹配
            self.assertEqual(x.imag, new_imag, exact_dtype=False)

    # 测试获取张量对角线视图的行为
    def test_diagonal_view(self, device) -> None:
        # 创建一个全为 1 的 5x5 张量 t，使用指定的设备
        t = torch.ones((5, 5), device=device)
        # 获取张量 t 的对角线视图 v
        v = torch.diagonal(t)
        # 断言 v 是张量 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的第一个元素为 0，检查 t 的对应元素是否也被修改
        v[0] = 0
        self.assertEqual(t[0, 0], v[0])

        # 创建一个全为 1 的 3x3x3 张量 t，使用指定的设备
        t = torch.ones((3, 3, 3), device=device)
        # 获取张量 t 指定偏移、维度的对角线视图 v
        v = torch.diagonal(t, offset=1, dim1=1, dim2=2)
        # 断言 v 是张量 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的第一个元素的第一个元素为 0，检查 t 的对应元素是否也被修改
        v[0, 0] = 0
        self.assertEqual(t[0, 0, 1], v[0, 0])

    # 测试选择张量的视图
    def test_select_view(self, device) -> None:
        # 创建一个全为 1 的 5x5 张量 t，使用指定的设备
        t = torch.ones((5, 5), device=device)
        # 选择张量 t 的第 0 维度上索引为 2 的视图 v
        v = t.select(0, 2)
        # 断言 v 是张量 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的第一个元素为 0，检查 t 的对应元素是否也被修改
        v[0] = 0
        self.assertEqual(t[2, 0], v[0])

    # TODO: 懒加载还未实现 unbind 方法
    @skipLazy
    def test_unbind_view(self, device) -> None:
        # 创建一个全为 0 的 5x5 张量 t，使用指定的设备
        t = torch.zeros((5, 5), device=device)
        # 使用 unbind 方法解绑张量 t，返回元组 tup
        tup = torch.unbind(t)

        # 遍历元组 tup 中的索引 idx 和元素 v
        for idx, v in enumerate(tup):
            # 断言 v 是张量 t 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 的第一个元素为 idx + 1，检查 t 的对应元素是否也被修改
            v[0] = idx + 1
            self.assertEqual(t[idx, 0], v[0])

    # TODO: opinfo 或者移动到 unbind 的测试套件
    def test_unbind(self):
        # 创建一个随机张量 stacked，形状为 (3, 10, 10)，需要梯度计算
        stacked = torch.randn(3, 10, 10, requires_grad=True)
        # 使用 unbind 方法解绑张量 stacked，得到 x, y, z
        x, y, z = stacked.unbind()
        # 创建一个随机张量 grad，形状为 (3, 10, 10)
        grad = torch.randn(3, 10, 10)
        # 对 x, y, z 分别进行反向传播，传入对应的梯度 grad.unbind()
        torch.autograd.backward([x, y, z], grad.unbind())
        # 断言 stacked 的梯度与 grad 相等
        self.assertEqual(stacked.grad, grad)
        # 对于每一个索引 i，检查是否仅提供一个梯度的情况下仍然有效
        for i in range(3):
            # 创建一个随机张量 stacked，形状为 (3, 10, 10)，需要梯度计算
            stacked = torch.randn(3, 10, 10, requires_grad=True)
            # 使用 unbind 方法解绑张量 stacked，得到 outs
            outs = stacked.unbind()
            # 获取 grad.unbind()[i] 的值
            gi = grad.unbind()[i]
            # 计算 outs[i] 关于 stacked 的梯度 g
            (g,) = torch.autograd.grad(outs[i], stacked, gi)
            # 创建一个期望的梯度张量 g_expected
            g_expected = torch.stack(
                [gi if j == i else torch.zeros_like(gi) for j in range(3)], dim=0
            )
            # 断言计算得到的梯度 g 与期望的梯度 g_expected 相等
            self.assertEqual(g, g_expected)
        # 使用 gradcheck 进行检查
        stacked = torch.randn(3, 10, 10, dtype=torch.double, requires_grad=True)
        gradcheck(lambda x: x.unbind(), (stacked,), check_forward_ad=True)

    # TODO: 修复 LTC 的测试，这里存在与动态形状交互导致断言触发的问题
    @skipLazy
    def test_expand_view(self, device) -> None:
        # 创建一个全为 1 的 5x1 张量 t，使用指定的设备
        t = torch.ones((5, 1), device=device)
        # 使用 expand 方法扩展张量 t 至形状 (5, 5)，得到视图 v
        v = t.expand(5, 5)
        # 断言 v 是张量 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的第三行第三列元素为 0，检查 t 的对应元素是否也被修改
        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])
    # 测试函数，用于验证 torch.tensor.expand_as 方法的功能
    def test_expand_as_view(self, device):
        # 创建一个形状为 (5, 1) 的张量，元素都为 1，指定设备为 device
        t = torch.ones((5, 1), device=device)
        # 创建一个形状为 (5, 5) 的空张量，指定设备为 device
        e = torch.empty((5, 5), device=device)
        # 使用 t 张量的形状来扩展 e 张量，使其形状变为 (5, 5)，并赋值给 v
        v = t.expand_as(e)
        # 断言函数 is_view_of 返回 t 是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 张量的第 (2, 2) 位置的值为 0
        v[2, 2] = 0
        # 断言 t 张量的第 (2, 0) 位置的值与 v 张量的第 (2, 2) 位置的值相等
        self.assertEqual(t[2, 0], v[2, 2])

    # 测试函数，用于验证 torch.narrow 方法创建的视图
    def test_narrow_view(self, device):
        # 创建一个形状为 (5, 5) 的张量，元素都为 1，指定设备为 device
        t = torch.ones((5, 5), device=device)
        # 使用 torch.narrow 在第 1 维上从索引 2 开始，长度为 2，创建视图 v
        v = torch.narrow(t, 1, 2, 2)
        # 断言函数 is_view_of 返回 t 是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 张量的第 (0, 0) 位置的值为 0
        v[0, 0] = 0
        # 断言 t 张量的第 (0, 2) 位置的值与 v 张量的第 (0, 0) 位置的值相等
        self.assertEqual(t[0, 2], v[0, 0])

    # 测试函数，用于验证 torch.tensor.permute 方法创建的视图
    def test_permute_view(self, device) -> None:
        # 创建一个形状为 (5, 5) 的张量，元素都为 1，指定设备为 device
        t = torch.ones((5, 5), device=device)
        # 使用 permute 方法交换维度 0 和 1，创建视图 v
        v = t.permute(1, 0)
        # 断言函数 is_view_of 返回 t 是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 张量的第 (0, 1) 位置的值为 0
        v[0, 1] = 0
        # 断言 t 张量的第 (1, 0) 位置的值与 v 张量的第 (0, 1) 位置的值相等
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于验证 torch.swapdims、torch.swapaxes 和 torch.transpose 方法创建的视图
    def test_transpose_view(self, device):
        # 遍历三种方法：torch.swapdims、torch.swapaxes、torch.transpose
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            # 创建一个形状为 (5, 5) 的张量，元素都为 1，指定设备为 device
            t = torch.ones((5, 5), device=device)
            # 使用当前方法 fn 对 t 张量进行操作，创建视图 v
            v = fn(t, 0, 1)
            # 断言函数 is_view_of 返回 t 是 v 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 张量的第 (0, 1) 位置的值为 0
            v[0, 1] = 0
            # 断言 t 张量的第 (1, 0) 位置的值与 v 张量的第 (0, 1) 位置的值相等
            self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于验证 torch.tensor.view_as 和 inplace 方法创建的视图
    def test_transpose_inplace_view(self, device):
        # 创建一个形状为 (5, 5) 的张量，元素都为 1，指定设备为 device
        t = torch.ones(5, 5, device=device)
        # 使用 view_as 方法创建 t 的视图 v
        v = t.view_as(t)
        # 使用 swapdims_ 方法交换维度 0 和 1，结果赋值给 v
        v = v.swapdims_(0, 1)
        # 断言函数 is_view_of 返回 t 是 v 的视图
        self.assertTrue(self.is_view_of(t, v))
        # 修改 v 张量的第 (0, 1) 位置的值为 0
        v[0, 1] = 0
        # 断言 t 张量的第 (1, 0) 位置的值与 v 张量的第 (0, 1) 位置的值相等
        self.assertEqual(t[1, 0], v[0, 1])

        # 重复上述过程，使用 swapaxes_ 方法
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapaxes_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        # 重复上述过程，使用 transpose_ 方法
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.transpose_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于验证 torch.tensor.t 方法创建的转置视图
    def test_t_view(self, device):
        # 创建一个形状为 (5, 5) 的张量，元素都为 1，指定设备为 device
        t = torch.ones((5, 5), device=device)
        # 使用 t 的 t 方法创建转置视图 v
        v = t.t()
        # 断言函数 is_view_of 返回 t 是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 张量的第 (0, 1) 位置的值为 0
        v[0, 1] = 0
        # 断言 t 张量的第 (1, 0) 位置的值与 v 张量的第 (0, 1) 位置的值相等
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于验证 torch.tensor.t_ 方法创建的转置视图（inplace）
    def test_t_inplace_view(self, device):
        # 创建一个形状为 (5, 5) 的张量，元素都为 1，指定设备为 device
        t = torch.ones(5, 5, device=device)
        # 使用 view_as 方法创建 t 的视图 v
        v = t.view_as(t)
        # 使用 t_ 方法进行转置，结果赋值给 v
        v = v.t_()
        # 断言函数 is_view_of 返回 t 是 v 的视图
        self.assertTrue(self.is_view_of(t, v))
        # 修改 v 张量的第 (0, 1) 位置的值为 0
        v[0, 1] = 0
        # 断言 t 张量的第 (1, 0) 位置的值与 v 张量的第 (0, 1) 位置的值相等
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于验证 torch.tensor.T、H、mT、mH 属性创建的转置视图
    def test_T_view(self, device):
        # 遍历四种属性：T、H、mT、mH
        for op in ("T", "H", "mT", "mH"):
            # 创建一个形状为 (5, 5) 的张量，元素都为 1，指定设备为 device
            t = torch.ones((5, 5
    # 测试函数：测试在原地操作下的squeeze视图
    def test_squeeze_inplace_view(self, device):
        # 创建一个5x5的张量，所有元素为1，指定设备
        t = torch.ones(5, 5, device=device)
        # 使用view_as函数创建一个与t相同形状的视图v
        v = t.view_as(t)
        # 对v进行原地squeeze操作，去除所有维度为1的维度
        v = v.squeeze_()
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))
        # 修改v的元素(0,1)为0
        v[0, 1] = 0
        # 断言t与v的基础张量相等
        self.assertEqual(t, v._base)

    # 测试函数：测试unsqueeze操作后的视图
    def test_unsqueeze_view(self, device):
        # 创建一个5x5的张量，所有元素为1，指定设备
        t = torch.ones(5, 5, device=device)
        # 对t进行unsqueeze操作，在第1维度上增加一个维度
        v = torch.unsqueeze(t, 1)
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改v的元素(0,0,1)为0
        v[0, 0, 1] = 0
        # 断言t的元素(0,1)与v的元素(0,0,1)相等
        self.assertEqual(t[0, 1], v[0, 0, 1])

    # 测试函数：测试原地unsqueeze操作后的视图
    def test_unsqueeze_inplace_view(self, device):
        # 创建一个5x5的张量，所有元素为1，指定设备
        t = torch.ones(5, 5, device=device)
        # 使用view_as函数创建一个与t相同形状的视图v
        v = t.view_as(t)
        # 对v进行原地unsqueeze操作，在第1维度上增加一个维度
        v = v.unsqueeze_(1)
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))
        # 修改v的元素(0,0,1)为0
        v[0, 0, 1] = 0
        # 断言t的元素(0,1)与v的元素(0,0,1)相等
        self.assertEqual(t[0, 1], v[0, 0, 1])

    # 测试函数：测试使用as_strided函数创建的视图
    def test_as_strided_view(self, device):
        # 创建一个5x5的张量，所有元素为1，指定设备
        t = torch.ones(5, 5, device=device)
        # 使用as_strided函数从t创建一个形状为(25,)的视图v，步长为(1,)
        v = torch.as_strided(t, (25,), (1,))
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改v的第6个元素为0
        v[6] = 0
        # 断言t的元素(1,1)与v的第6个元素相等
        self.assertEqual(t[1, 1], v[6])

    # 测试函数：测试原地使用as_strided函数创建的视图
    def test_as_strided_inplace_view(self, device):
        # 创建一个5x5的张量，所有元素为1，指定设备
        t = torch.ones(5, 5, device=device)
        # 使用view_as函数创建一个与t相同形状的视图v
        v = t.view_as(t)
        # 对v进行原地使用as_strided函数创建一个形状为(25,)的视图，步长为(1,)
        v = v.as_strided_((25,), (1,))
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))
        # 修改v的第6个元素为0
        v[6] = 0
        # 断言t的元素(1,1)与v的第6个元素相等
        self.assertEqual(t[1, 1], v[6])
    def test_as_strided_gradients(self):
        def test(x, prepro_fn, size, strides, offset=None):
            # 将输入张量转换为双精度类型，并标记为需要梯度
            x = x.to(torch.double).detach().requires_grad_()

            # 使用 torch.no_grad() 上下文管理器，确保前向传播不会调整存储空间大小
            # 这可能导致输出中出现 NaN，并因此无法通过数值雅可比检查
            with torch.no_grad():
                # 如果存在预处理函数，应用预处理函数；否则保持不变
                y = prepro_fn(x) if prepro_fn is not None else x
                # 计算最大偏移量，避免重新调整存储空间大小
                max_offset = sum((si - 1) * st for si, st in zip(size, strides))
                max_offset += offset if offset is not None else y.storage_offset()
                # 检查是否需要调整存储空间大小，抛出断言异常以防止测试失败
                assert max_offset < len(y.storage()), "test case resizes storage"

            # 定义闭包函数 closure，返回通过 as_strided 方法得到的张量
            def closure(x):
                if prepro_fn is not None:
                    x = prepro_fn(x)
                return x.as_strided(size, strides, offset)

            # 对闭包函数进行梯度检查，验证其对输入张量的梯度计算正确性
            gradcheck(closure, [x], check_forward_ad=True)
            # 对二阶梯度进行检查
            gradgradcheck(closure, [x])

        # 第一个测试：调用 test 函数，测试张量 [0, 1, 2, ..., 24] 的情况
        test(torch.arange(0, 25), lambda x: x.view(5, 5), [3, 3], [6, 2], 2)

        # 第二个测试：测试具有奇怪步长的情况，且维度大小为 1 的情况
        test(torch.randn(12), None, [1, 2, 1, 5], [0, 5, 100, 1], 2)

        # 第三个测试：测试扩展的情况
        test(torch.randn(5), None, [3, 3, 3], [0, 1, 0], 2)
        test(torch.randn(5), None, [3, 3, 3], [0, 0, 0], 4)
        test(torch.randn(5), lambda x: x.expand(5, 5), [5, 5], [0, 1], 0)

        # 第四个测试：测试非扩展且重叠的情况
        test(torch.randn(35), None, [6, 6], [5, 1], 2)
        test(torch.randn(15), None, [3, 2], [3, 6], 2)

        # 第五个测试：测试转置的情况
        test(torch.randn(3, 4), None, [4, 3], [1, 4])

        # 第六个测试：测试“获取超出输入范围”的情况
        x = torch.randn(6, 2)
        test(x[3:], None, [3, 2], [2, 1], 0)  # 应该全为零
        self.assertEqual(x[3:].as_strided([3, 2], [2, 1], 0), x[:3])

        # 第七个测试：测试在扩展输入上的 select 情况
        test(torch.randn(2, 3), lambda x: x.expand(10, 2, 3), [2, 3], [3, 1], 0)

    def test_view_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view(25)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_as_view(self, device):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,))
        v = t.view_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_contiguous_self(self, device):
        t = torch.ones(5, 5, device=device)
        s = t.contiguous()
        self.assertTrue(s is t)

    @skipMeta
    # self.is_view_of 对于懒惰情况会报告误报
    @skipLazy
    def test_contiguous_nonview(self, device):
        t = torch.ones(5, 5, device=device)
        nv = t.t().contiguous()
        self.assertTrue(not self.is_view_of(t, nv))

        nv[0, 0] = 0
        self.assertNotEqual(t[0, 0], nv[0, 0])
    # 定义测试函数，验证 reshape 方法是否正确创建视图
    def test_reshape_view(self, device):
        # 创建一个 5x5 的张量，所有元素为 1，指定设备
        t = torch.ones(5, 5, device=device)
        # 使用 reshape 方法将张量 t 转换为形状为 (25,) 的视图 v
        v = torch.reshape(t, (25,))
        # 断言 v 是否为 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 中索引为 6 的元素为 0
        v[6] = 0
        # 断言 t 中索引为 (1, 1) 的元素与 v 中索引为 6 的元素相等
        self.assertEqual(t[1, 1], v[6])

    # 定义测试函数，验证 reshape_as 方法是否正确创建视图
    def test_reshape_as_view(self, device):
        # 创建一个 5x5 的张量，所有元素为 1，指定设备
        t = torch.ones(5, 5, device=device)
        # 创建一个形状为 (25,) 的空张量 e，指定设备
        e = torch.empty((25,), device=device)
        # 使用 reshape_as 方法将张量 t 转换为与 e 相同形状的视图 v
        v = t.reshape_as(e)
        # 断言 v 是否为 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 中索引为 6 的元素为 0
        v[6] = 0
        # 断言 t 中索引为 (1, 1) 的元素与 v 中索引为 6 的元素相等
        self.assertEqual(t[1, 1], v[6])

    @skipMeta
    # self.is_view_of 报告对于懒加载存在误报
    @skipLazy
    # 定义测试函数，验证 reshape 方法是否正确创建非视图
    def test_reshape_nonview(self, device):
        # 创建一个 5x5 的张量，所有元素为 1，指定设备
        t = torch.ones(5, 5, device=device)
        # 使用 reshape 方法将 t 的转置作为形状为 (25,) 的新张量 nv
        nv = torch.reshape(t.t(), (25,))
        # 断言 nv 是否不是 t 的视图
        self.assertTrue(not self.is_view_of(t, nv))

        # 修改 nv 中索引为 6 的元素为 0
        nv[6] = 0
        # 断言 t 中索引为 (1, 1) 的元素与 nv 中索引为 6 的元素不相等
        self.assertNotEqual(t[1, 1], nv[6])

    # 此测试使用 as_strided 构造具有重叠内存的张量，
    # 这种情况下不适用于函数化处理。
    @skipLazy
    @skipXLA
    # 定义测试函数，验证 flatten 方法是否正确创建视图
    def test_flatten_view(self, device):
        # 定义嵌套函数，用于测试写入是否传播
        def test_writes_propagate(t, v):
            # 设置张量 t 和视图 v 的索引为零的元素为 0
            idx_t = (0,) * t.ndim
            idx_v = (0,) * v.ndim
            v[idx_v] = 0
            # 断言张量 t 和视图 v 中索引为零的元素相等
            self.assertEqual(t[idx_t], v[idx_v])

        # 创建一个 1x2x3x4 的张量，所有元素为 1，指定设备
        t = torch.ones(1, 2, 3, 4, device=device)
        # 使用 flatten 方法将 t 转换为形状为 (24,) 的视图 v
        v = t.flatten()
        # 断言 v 是否为 t 的视图
        self.assertTrue(self.is_view_of(t, v))
        # 调用嵌套函数测试写入是否传播
        test_writes_propagate(t, v)

        # 创建一个标量张量，值为 1，指定设备
        t = torch.tensor(1, device=device)
        # 使用 flatten 方法将标量张量 t 转换为形状为 (1,) 的视图 v
        v = t.flatten()
        # 调用嵌套函数测试写入是否传播
        test_writes_propagate(t, v)
        # 断言 v 是否为 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 创建一个 1x2x3x4 的张量，所有元素为 1，转置后指定设备
        t = torch.ones(1, 2, 3, 4, device=device).transpose(2, 3)
        # 使用 flatten 方法将 t 转换为形状为 (24,) 的视图 v
        v = t.flatten(0, 1)
        # 调用嵌套函数测试写入是否传播
        test_writes_propagate(t, v)
        # 断言 v 是否与 t 具有相同的基础张量
        self.assertTrue(self.is_view_of_same_base(t, v))

        # 创建一个 720 元素的张量，指定设备，并使用 as_strided 方法创建
        t = torch.ones(720, device=device).as_strided(
            (2, 3, 2, 3, 5, 4), (6, 2, 15, 5, 1, 0)
        )
        # 使用 flatten 方法将 t 转换为形状为 (24,) 的视图 v1
        v1 = t.flatten(0, 1)
        # 使用 flatten 方法将 v1 转换为形状为 (6,) 的视图 v2
        v2 = v1.flatten(1, 3)
        # 使用 flatten 方法将 v2 转换为形状为 (2,) 的视图 v3
        v3 = v2.flatten(2, 2)
        # 调用嵌套函数测试写入是否传播
        test_writes_propagate(t, v1)
        # 断言 v1 是否与 t 具有相同的基础张量
        self.assertTrue(self.is_view_of_same_base(t, v1))
        # 调用嵌套函数测试写入是否传播
        test_writes_propagate(t, v2)
        # 断言 v2 是否与 t 具有相同的基础张量
        self.assertTrue(self.is_view_of_same_base(t, v2))
        # 调用嵌套函数测试写入是否传播
        test_writes_propagate(t, v3)
        # 断言 v3 是否与 t 具有相同的基础张量
        self.assertTrue(self.is_view_of_same_base(t, v3))

    @onlyNativeDeviceTypes
    # 测试函数：验证在非视图情况下的扁平化操作
    def test_flatten_nonview(self, device):
        # 辅助函数：验证张量是否为非视图
        def assert_is_nonview(t, nv):
            # 生成对应张量的索引
            idx_t = (0,) * t.ndim
            idx_nv = (0,) * nv.ndim
            # 断言：验证 nv 不是视图
            self.assertTrue(not nv._is_view())
            # 将 nv 的索引位置设为 0
            nv[idx_nv] = 0
            # 如果设备不是 "meta"，断言 t 和 nv 在对应索引位置上的值不相等
            if device != "meta":
                self.assertNotEqual(t[idx_t], nv[idx_nv])

        # 创建一个张量 t，全为 1，形状为 (2, 3, 2, 3)，并对其进行维度转置
        t = torch.ones(2, 3, 2, 3, device=device).transpose(2, 3)
        # 对 t 进行扁平化操作，从维度 1 到维度 3
        nv = t.flatten(1, 3)
        # 调用辅助函数 assert_is_nonview 验证 t 和 nv 是否为非视图
        assert_is_nonview(t, nv)

        # 创建一个张量 t，形状为 (2, 2)，全为 1，并进行转置
        t = torch.ones(2, 2, device=device).T
        # 对 t 进行扁平化操作，默认从第一个维度开始扁平化
        nv = t.flatten()
        # 再次调用 assert_is_nonview 验证 t 和 nv 是否为非视图
        assert_is_nonview(t, nv)

        # 对于 t 的扁平化操作，如果起始维度等于结束维度，返回原始对象
        t = torch.ones(2, 2, device=device)
        nv = t.flatten(1, 1)
        # 断言 t 和 nv 指向相同对象
        self.assertTrue(t is nv)

    # 测试函数：验证基本索引和切片生成的视图
    def test_basic_indexing_slice_view(self, device):
        # 创建一个张量 t，形状为 (5, 5)，全为 1
        t = torch.ones(5, 5, device=device)
        # 对 t 进行切片操作，获取部分视图 v
        v = t[:2, :3]
        # 断言：验证 v 是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的元素值，验证 t 和 v 对应元素的值是否相等
        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    # 测试函数：验证基本索引和省略号生成的视图
    def test_basic_indexing_ellipses_view(self, device):
        # 创建一个张量 t，形状为 (5, 5)，全为 1
        t = torch.ones(5, 5, device=device)
        # 使用省略号进行索引，获取部分视图 v
        v = t[..., :2]
        # 断言：验证 v 是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的元素值，验证 t 和 v 对应元素的值是否相等
        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    # 测试函数：验证基本索引和新轴生成的视图
    def test_basic_indexing_newaxis_view(self, device):
        # 创建一个张量 t，形状为 (5, 5)，全为 1
        t = torch.ones(5, 5, device=device)
        # 使用新轴进行索引，获取部分视图 v
        v = t[None, :2, 3]
        # 断言：验证 v 是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的元素值，验证 t 和 v 对应元素的值是否相等
        v[0, 0] = 0
        self.assertEqual(t[0, 3], v[0, 0])

    # 测试函数：验证高级索引生成的非视图
    def test_advanced_indexing_nonview(self, device):
        # 创建一个张量 t，形状为 (3, 3)，全为 1
        t = torch.ones(3, 3, device=device)
        # 创建行索引和列索引张量
        rows = torch.tensor([[0, 0], [2, 2]], device=device)
        cols = torch.tensor([[0, 1], [2, 2]], device=device)
        # 使用高级索引获取部分非视图 nv
        nv = t[rows, cols]
        # 断言：验证 nv 不是 t 的视图
        self.assertTrue(not self.is_view_of(t, nv))

        # 修改 nv 的元素值，验证 t 和 nv 对应元素的值是否不相等
        nv[1, 1] = 0
        self.assertNotEqual(t[2, 2], nv[1, 1])

    # 测试函数：验证高级索引的赋值操作
    @unittest.skipIf(
        IS_FBCODE, "TorchScript backend not yet supported in FBCODE/OVRSOURCE builds"
    )
    def test_advanced_indexing_assignment(self, device):
        # 创建一个张量 t，形状为 (3, 3)，全为 1
        t = torch.ones(3, 3, device=device)
        # 创建行索引和列索引张量
        rows = torch.tensor([[0, 0], [2, 2]], device=device)
        cols = torch.tensor([[0, 1], [2, 2]], device=device)
        # 使用高级索引赋值操作
        t[rows, cols] = 0
        # 断言：验证赋值后 t 对应位置的值是否为 0
        self.assertEqual(t[2, 2], 0)

    # 测试函数：验证块视图的行为
    @unittest.skip("See https://github.com/pytorch/pytorch/pull/32720")
    def test_chunk_view(self, device):
        # 创建一个形状为 (3, 3) 的零张量 t
        t = torch.zeros(3, 3, device=device)
        # 将 t 分块成多个小块 l
        l = torch.chunk(t, 3)

        # 遍历每个小块的索引和值
        for idx, v in enumerate(l):
            # 断言：验证 v 是 t 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 的元素值，验证 t 和 v 对应位置的值是否相等
            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    # 测试函数：验证分割视图的行为
    @unittest.skip("See https://github.com/pytorch/pytorch/pull/32720")
    def test_split_view(self, device):
        # 创建一个形状为 (3, 3) 的零张量 t
        t = torch.zeros(3, 3, device=device)
        # 将 t 按指定长度分割成多个部分 l
        l = torch.split(t, [1, 1, 1])

        # 遍历每个部分的索引和值
        for idx, v in enumerate(l):
            # 断言：验证 v 是 t 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 的元素值，验证 t 和 v 对应位置的值是否相等
            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])
    # 定义测试函数 test_movedim_view，测试 movedim 和 moveaxis 函数的视图变换功能
    def test_movedim_view(self, device):
        # 定义内部函数 run_test，用于执行单个测试操作
        def run_test(device, op):
            # 创建一个设备上的全零张量
            t = torch.zeros(3, 3, device=device)
            # 执行操作 op，并将结果存储在 out 中
            out = op(t)

            # 断言 out 是 t 的视图
            self.assertTrue(self.is_view_of(t, out))

            # 随机修改输出 out 中的值，并验证原始张量 t 是否也被修改
            for _ in range(3):
                idx_1, idx_2 = random.randint(0, 2), random.randint(0, 2)
                out[idx_1, idx_2] = random.random()
                # 检查 t 的对应位置是否被正确修改
                self.assertEqual(t[idx_2, idx_1], out[idx_1, idx_2])

        # 遍历测试的函数列表 [torch.movedim, torch.moveaxis]
        for fn in [torch.movedim, torch.moveaxis]:
            # 部分应用函数 fn，将源维度从 (0, 1) 转换到 (1, 0)，并执行测试
            op = partial(fn, source=(0, 1), destination=(1, 0))
            run_test(device, op)

            # 部分应用函数 fn，将源维度从 0 转换到 1，并执行测试
            op = partial(fn, source=0, destination=1)
            run_test(device, op)

    # 测试视图拷贝核心函数及其导数是否正确实现
    def test_view_copy(self, device):
        # 创建一个设备上的随机张量 a，并要求其梯度跟踪
        a = torch.randn(4, device=device, requires_grad=True)
        # 克隆张量 a，并将其分离并重新要求梯度跟踪
        a_ref = a.clone().detach().requires_grad_()
        # 创建 a_ref 的视图，形状为 (2, 2)
        a_view = a_ref.view(2, 2)
        # 使用 torch.view_copy 函数复制张量 a，并形状为 (2, 2)
        a_view_copy = torch.view_copy(a, (2, 2))

        # 断言视图拷贝操作不保留视图关系
        self.assertTrue(self.is_view_of(a_ref, a_view))
        self.assertFalse(self.is_view_of(a, a_view_copy))

        # 对 a_view_copy 的和进行反向传播
        a_view_copy.sum().backward()
        # 对 a_view 的和进行反向传播
        a_view.sum().backward()

        # 断言前向和反向传播结果的形状和值相同
        self.assertEqual(a_view_copy, a_view)
        self.assertEqual(a.grad, a_ref.grad)

    # 测试视图拷贝核心函数输出是否连续（默认情况下）
    def test_view_copy_output_contiguous(self, device):
        # 创建一个四维张量 a，在通道优先的内存格式下
        a = torch.randn(4, 4, 4, 4, device=device).to(memory_format=torch.channels_last)
        # 使用 torch.ops.aten.slice_copy 函数从 a 中切片复制，从第 0 轴的 0 到 2 的部分
        b = torch.ops.aten.slice_copy(a, 0, 0, 2)
        # 断言 b 是否是连续的张量
        self.assertTrue(b.is_contiguous())

    # 测试视图拷贝核心函数的输出情况
    def test_view_copy_out(self, device):
        # 创建一个设备上的随机二维张量 a
        a = torch.randn(2, 2, device=device)
        # 创建一个空的张量 out，形状为 (2,)
        out = torch.empty(2, device=device)

        # 使用 torch.diagonal_copy 函数，将 a 的对角线拷贝到 out 中
        torch.diagonal_copy(a, out=out)
        # 使用 torch.diagonal_copy 函数，返回 a 的对角线拷贝到 expected 中
        expected = torch.diagonal_copy(a)

        # 断言 out 和 expected 是否相等
        self.assertEqual(expected, out)

        # 创建一个设备上的随机四维张量 a
        a = torch.randn(4, device=device)
        # 创建两个空的张量 out1 和 out2，形状都为 (2,)
        out1 = torch.empty(2, device=device)
        out2 = torch.empty(2, device=device)

        # 使用 torch.split_copy 函数，将 a 分割为两部分，每部分形状为 (2,)，并将结果分别存储在 out1 和 out2 中
        torch.split_copy(a, 2, out=(out1, out2))
        # 使用 torch.split_copy 函数，将 a 分割为两部分，每部分形状为 (2,)，并返回结果 expected1 和 expected2
        expected1, expected2 = torch.split_copy(a, 2)

        # 断言 out1 和 expected1 是否相等
        self.assertEqual(expected1, out1)
        # 断言 out2 和 expected2 是否相等
        self.assertEqual(expected2, out2)
# 定义一个测试类 TestOldViewOps，继承自 TestCase 类，用于测试旧版本的视图操作
class TestOldViewOps(TestCase):

    # 定义一个测试方法 test_ravel，接受一个 device 参数
    def test_ravel(self, device):
        
        # 定义一个内部方法 _test_ravel，接受 tensors（张量列表）、size（期望的扁平化后大小）、nc（是否非连续）
        def _test_ravel(tensors, size, nc=False):
            
            # 遍历输入的张量列表 tensors
            for src in tensors:
                
                # 对连续的张量进行扁平化操作
                flat = src.ravel()
                
                # 断言扁平化后的形状与预期的大小相同
                self.assertEqual(flat.shape, torch.Size([size]))
                
                # 断言原始张量通过 view(-1) 返回的结果与扁平化后的结果相同
                self.assertEqual(src.view(-1), flat)
                
                # 断言扁平化后的张量的 _base 属性指向原始张量 src
                self.assertIs(flat._base, src)
                
                # 断言扁平化后的张量是连续的
                self.assertTrue(flat.is_contiguous())

                # 对非连续的张量进行扁平化操作
                if nc:
                    # 转置非连续张量 src
                    nc_src = src.t()
                    
                    # 对转置后的张量进行扁平化操作
                    nc_flat = nc_src.ravel()
                    
                    # 断言扁平化后的形状与预期的大小相同
                    self.assertEqual(nc_flat.shape, torch.Size([size]))
                    
                    # 断言转置后的连续视图通过 view(-1) 返回的结果与扁平化后的结果相同
                    self.assertEqual(nc_src.contiguous().view(-1), nc_flat)
                    
                    # 断言扁平化后的张量的 _base 属性不指向原始张量 src
                    self.assertIsNot(nc_flat._base, src)
                    
                    # 断言扁平化后的张量是连续的
                    self.assertTrue(nc_flat.is_contiguous())

        # 测试 flatten 方法是否对零维张量返回一维张量
        zero_dim_tensor = torch.tensor(123, device=device)
        flat0 = zero_dim_tensor.ravel()
        
        # 测试结果是否符合预期的形状
        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        self.assertEqual(flat0.shape, torch.Size([1]))

        # 测试一维张量的扁平化操作
        one_dim_tensor = torch.tensor([123], device=device)
        flat1 = zero_dim_tensor.ravel()
        
        # 测试结果是否符合预期的形状
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        self.assertEqual(flat1.shape, torch.Size([1]))

        # 测试非连续张量的扁平化操作
        nc_ones_tensor = torch.ones(10, device=device)[::2]
        flat2 = nc_ones_tensor.ravel()
        
        # 测试结果是否符合预期的形状
        self.assertEqual(nc_ones_tensor.shape, torch.Size([5]))
        self.assertEqual(flat2.shape, torch.Size([5]))

        # 断言不同张量的扁平化结果相同
        self.assertEqual(flat0, one_dim_tensor)
        self.assertEqual(flat0, flat1)
        
        # 断言扁平化后的张量的形状相同
        self.assertEqual(flat0.shape, flat1.shape)
        
        # 断言扁平化后的张量是连续的
        self.assertTrue(flat0.is_contiguous())
        self.assertTrue(flat1.is_contiguous())
        self.assertTrue(flat2.is_contiguous())

        # 测试浮点张量和量化张量的扁平化操作
        tensors = [
            torch.randn(5, 5, 5, 5, device=device),
            torch._empty_affine_quantized(
                [5, 5, 5, 5], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        
        # 调用 _test_ravel 方法进行测试
        _test_ravel(tensors, 625)

        tensors = [
            torch.randn(0, 2, 3, device=device),
            torch.randn(3, 0, 2, device=device),
            torch._empty_affine_quantized(
                [0, 2, 3], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
            torch._empty_affine_quantized(
                [3, 0, 2], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        
        # 调用 _test_ravel 方法进行测试
        _test_ravel(tensors, 0)

        tensors = [
            torch.randn(5, 5, device=device),
            torch._empty_affine_quantized(
                [5, 5], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        
        # 调用 _test_ravel 方法进行测试，指定 nc 参数为 True
        _test_ravel(tensors, 25, True)

    # TODO: this should be refactored into the view ops test suite
    # 定义一个测试方法，用于测试在指定设备上对空张量进行reshape操作
    def test_empty_reshape(self, device):
        # 创建一个形状为(0, 6)的随机张量x，使用指定设备
        x = torch.randn(0, 6, device=device)
        # 断言reshape后张量的形状为(1, 0, 6, 1, 1)
        self.assertEqual((1, 0, 6, 1, 1), x.reshape(1, 0, 6, 1, 1).shape)
        # 断言reshape后张量的数据指针与原张量相同，保证数据可共享
        self.assertEqual(x.data_ptr(), x.reshape(1, 0, 6, 1, 1).data_ptr())

        # 测试与NumPy语义一致性，不推断具有自由度的维度的大小
        self.assertRaises(RuntimeError, lambda: x.reshape(0, -1))

    # 在TorchDynamo出现问题时跳过测试
    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    # 定义测试方法，用于测试张量的expand操作
    def test_expand(self, device):
        # 创建一个形状为(1, 8, 1)的随机张量tensor，使用指定设备
        tensor = torch.rand(1, 8, 1, device=device)
        # 创建一个形状为(5)的随机张量tensor2，使用指定设备
        tensor2 = torch.rand(5, device=device)
        # 创建一个形状为(4, 8, 5)的随机张量template，使用指定设备
        template = torch.rand(4, 8, 5, device=device)
        # 获取目标形状
        target = template.size()
        # 断言使用expand_as方法扩展后张量的形状与目标形状相同
        self.assertEqual(tensor.expand_as(template).size(), target)
        # 断言使用expand方法扩展后张量的形状与目标形状相同
        self.assertEqual(tensor.expand(4, 8, 5).size(), target)
        # 断言使用expand方法扩展后张量的形状与目标形状相同
        self.assertEqual(tensor.expand(target).size(), target)
        # 断言使用expand_as方法扩展后张量tensor2的形状与目标形状相同
        self.assertEqual(tensor2.expand_as(template).size(), target)
        # 断言使用expand方法扩展后张量tensor2的形状与目标形状相同
        self.assertEqual(tensor2.expand(4, 8, 5).size(), target)
        # 断言使用expand方法扩展后张量tensor2的形状与目标形状相同
        self.assertEqual(tensor2.expand(target).size(), target)

        # 测试双重扩展
        self.assertEqual(tensor2.expand(1, 5).expand(2, 2, 5), tensor2.repeat(2, 2, 1))

        # 测试非连续张量
        noncontig = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        # 断言非连续张量的连续版本，使用repeat方法进行扩展后的形状与expand方法扩展后的形状相同
        self.assertEqual(
            noncontig.expand(2, 5, 4, 3), noncontig.contiguous().repeat(2, 1, 4, 1)
        )

        # 确保与unsqueeze兼容
        expanded = tensor2.expand(1, 1, 5)
        unsqueezed = tensor2.unsqueeze(0).unsqueeze(1)
        # 断言扩展后的张量与unsqueeze后的张量相同
        self.assertEqual(expanded, unsqueezed)
        # 断言扩展后的张量的步长与unsqueeze后的张量的步长相同
        self.assertEqual(expanded.stride(), unsqueezed.stride())

        # 测试使用-1作为目标大小
        self.assertEqual(tensor.expand(4, -1, 5), tensor.expand(4, 8, 5))
        # 断言使用两个-1作为参数时会引发运行时错误
        self.assertRaises(RuntimeError, lambda: tensor2.expand(-1, -1))

        # 测试将空张量扩展到空张量
        self.assertEqual(
            torch.zeros(0, device=device).expand((0,)), torch.zeros(0, device=device)
        )

    # TODO: 将此测试重构为视图操作测试套件的一部分
    # 定义测试方法，用于测试空张量的视图操作
    def test_view_empty(self, device):
        # 创建一个形状为(0, 6)的随机张量x，使用指定设备
        x = torch.randn(0, 6, device=device)
        # 断言使用view方法后张量的形状为(1, 0, 6, 1, 1)
        self.assertEqual((1, 0, 6, 1, 1), x.view(1, 0, 6, 1, 1).shape)

    # TODO: 将此测试重构为视图操作测试套件的一部分
    @onlyNativeDeviceTypes
    # 定义测试方法，接受一个设备参数
    def test_reshape(self, device):
        # 创建一个形状为 (3, 3) 的随机张量 x，使用指定设备
        x = torch.randn(3, 3, device=device)
        # 断言 x 和其展平后的数据指针相同
        self.assertEqual(x.data_ptr(), x.reshape(-1).data_ptr())
        # 断言 x 和重新形状为 (1, 9, 1) 后的数据指针相同
        self.assertEqual(x.data_ptr(), x.reshape(1, 9, 1).data_ptr())
        # 使用 torch.reshape 将 x 重新形状为 (9,)，与 x.reshape(9) 进行比较
        self.assertEqual(torch.reshape(x, (9,)), x.reshape(9))
        # 预期运行时错误，因为尝试将 x 重新形状为 (-1, -1)
        self.assertRaises(RuntimeError, lambda: x.reshape(-1, -1))

        # 创建一个形状为 (4, 4, 4) 的随机张量 y，取其第一维的所有元素和第三维的所有元素
        y = torch.randn(4, 4, 4, device=device)[:, 0, :]
        # 如果设备不是 "meta"，则断言 y 和其展平后的数据指针不同
        # 因为在 "meta" 设备上，元数据张量的数据指针始终为 0，无论形状如何
        if device != "meta":
            self.assertNotEqual(y.data_ptr(), y.reshape(-1).data_ptr())
        # 断言 y 连续化后展平的结果与 reshape(-1) 的结果相等
        self.assertEqual(y.contiguous().view(-1), y.reshape(-1))
        # 断言将 y 重新形状为 (2, 2, 4) 后的数据指针与原始 y 的数据指针相同
        self.assertEqual(y.reshape(2, 2, 4).data_ptr(), y.data_ptr())

        # 创建一个标量张量 s，随机值，形状为 ()
        s = torch.randn((), device=device)
        # 断言 s 和将其形状重新设为 () 后的数据指针相同
        self.assertEqual(s.data_ptr(), s.reshape(()).data_ptr())
        # 断言 s 重新形状为 (-1) 后的形状为 (1,)
        self.assertEqual(s.reshape(-1).shape, (1,))
        # 预期运行时错误，因为尝试将 s 重新形状为 (2)，但 s 为标量，不可形状化为大于 1 的形状
        self.assertRaises(RuntimeError, lambda: s.reshape(2))

        # 创建一个空张量 empty，形状为空，使用指定设备
        empty = torch.tensor([], device=device)
        # 断言 empty 和其展平后的结果相等
        self.assertEqual(empty, empty.reshape(-1))
        # 断言 empty 和其形状为 [0] 后的结果相等
        self.assertEqual(empty, empty.reshape([0]))
        # TODO: 一旦有多维空张量支持，修复这些测试用例
        # 断言 empty 重新形状为 [0, 1] 后的形状为 (0, 1)
        self.assertEqual(empty.reshape([0, 1]).shape, (0, 1))
        # 断言 empty 重新形状为 [1, -1] 后的形状为 (1, 0)
        self.assertEqual(empty.reshape([1, -1]).shape, (1, 0))
        # 预期运行时错误，因为尝试将 empty 重新形状为 (1)，但其张量为空，不能形状化为大于 0 的形状
        self.assertRaises(RuntimeError, lambda: empty.reshape(1))

        # 重新定义张量 x，形状为 (3, 3)，使用指定设备
        x = torch.randn(3, 3, device=device)
        # 断言 x 和根据 torch.rand(9) 的形状重新形状化后的数据指针相同
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(9)).data_ptr())
        # 断言 x 和根据 torch.rand(1, 9, 1) 的形状重新形状化后的数据指针相同
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(1, 9, 1)).data_ptr())
        # 预期运行时错误，因为尝试将 x 重新形状为与设备上的 10 大小不匹配的形状
        self.assertRaises(
            RuntimeError, lambda: x.reshape_as(torch.rand(10, device=device))
        )
    # 定义一个测试方法，用于测试 flatten 方法的行为
    def test_flatten(self, device):
        # 测试当给定一个零维张量时，flatten 方法返回一个一维张量
        zero_dim_tensor = torch.tensor(123, device=device)
        # 使用 flatten 方法对零维张量进行扁平化操作
        flat0 = zero_dim_tensor.flatten()
        # 创建一个包含单个元素的一维张量
        one_dim_tensor = torch.tensor([123], device=device)
        # 用相同的方法对单元素张量进行扁平化
        flat1 = zero_dim_tensor.flatten()

        # 断言零维张量的形状为一个空的 torch.Size 对象
        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        # 断言扁平化后的零维张量的形状为 torch.Size([1])
        self.assertEqual(flat0.shape, torch.Size([1]))
        # 断言单元素张量的形状为 torch.Size([1])
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        # 断言两个扁平化后的张量具有相同的值
        self.assertEqual(flat0, one_dim_tensor)
        # 进一步断言两个扁平化后的零维张量相等
        self.assertEqual(flat0, flat1)
        # 最后断言两个扁平化后张量的形状相同
        self.assertEqual(flat0.shape, flat1.shape)

        # 测试浮点张量和量化张量的扁平化行为
        tensors = [
            torch.randn(5, 5, 5, 5, device=device),
            torch._empty_affine_quantized(
                [5, 5, 5, 5], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        # 遍历不同类型的张量进行扁平化操作
        for src in tensors:
            # 对张量进行从头到尾的扁平化
            flat = src.flatten(0, -1)
            # 断言扁平化后的形状为 torch.Size([625])
            self.assertEqual(flat.shape, torch.Size([625]))
            # 断言扁平化后的值与使用 view(-1) 相同
            self.assertEqual(src.view(-1), flat.view(-1))

            # 对张量进行从头到部分的扁平化
            flat = src.flatten(0, 2)
            # 断言扁平化后的形状为 torch.Size([125, 5])
            self.assertEqual(flat.shape, torch.Size([125, 5]))
            # 断言扁平化后的值与使用 view(-1) 相同
            self.assertEqual(src.view(-1), flat.view(-1))

            # 对张量进行从头到一部分的扁平化
            flat = src.flatten(0, 1)
            # 断言扁平化后的形状为 torch.Size([25, 5, 5])
            self.assertEqual(flat.shape, torch.Size([25, 5, 5]))
            # 断言扁平化后的值与使用 view(-1) 相同
            self.assertEqual(src.view(-1), flat.view(-1))

            # 对张量进行一部分到另一部分的扁平化
            flat = src.flatten(1, 2)
            # 断言扁平化后的形状为 torch.Size([5, 25, 5])
            self.assertEqual(flat.shape, torch.Size([5, 25, 5]))
            # 断言扁平化后的值与使用 view(-1) 相同
            self.assertEqual(src.view(-1), flat.view(-1))

            # 对张量进行一部分到末尾的扁平化
            flat = src.flatten(2, 3)
            # 断言扁平化后的形状为 torch.Size([5, 5, 25])
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            # 断言扁平化后的值与使用 view(-1) 相同
            self.assertEqual(src.view(-1), flat.view(-1))

            # 对张量进行从倒数第二维到最后一维的扁平化
            flat = src.flatten(-2, -1)
            # 断言扁平化后的形状为 torch.Size([5, 5, 25])
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            # 断言扁平化后的值与使用 view(-1) 相同
            self.assertEqual(src.view(-1), flat.view(-1))

            # 对张量进行从一个维度到同一个维度的扁平化，此时应该返回原张量
            flat = src.flatten(2, 2)
            self.assertEqual(flat, src)

            # 测试超出边界的索引
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                src.flatten(5, 10)

            # 测试无效的起始和结束维度
            with self.assertRaisesRegex(
                RuntimeError, "start_dim cannot come after end_dim"
            ):
                src.flatten(2, 0)

    # TODO: update to work on CUDA, too
    @onlyCPU
    # 定义一个测试方法，用于测试 torch.tensor 对象的 narrow 方法在给定设备上的行为
    def test_narrow(self, device):
        # 创建一个 3x3 的张量 x
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        # 测试 narrow 方法，从第 0 维索引 0 开始，长度为 1
        self.assertEqual(x.narrow(0, 0, 1), torch.tensor([[0, 1, 2]]))
        # 测试 narrow 方法，从第 0 维索引 0 开始，长度为 2
        self.assertEqual(x.narrow(0, 0, 2), torch.tensor([[0, 1, 2], [3, 4, 5]]))
        # 测试 narrow 方法，从第 0 维索引 1 开始，长度为 1
        self.assertEqual(x.narrow(0, 1, 1), torch.tensor([[3, 4, 5]]))
        # 测试 narrow 方法，从第 0 维索引 -1（即倒数第一行）开始，长度为 1
        self.assertEqual(x.narrow(0, -1, 1), torch.tensor([[6, 7, 8]]))
        # 测试 narrow 方法，从第 0 维索引 -2（即倒数第二行）开始，长度为 2
        self.assertEqual(x.narrow(0, -2, 2), torch.tensor([[3, 4, 5], [6, 7, 8]]))
        # 测试 narrow 方法，从第 0 维索引 -3（即倒数第三行）开始，长度为 3
        self.assertEqual(
            x.narrow(0, -3, 3), torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        )
        # 测试 narrow 方法，从第 -1 维索引 -1（即倒数第一列）开始，长度为 1
        self.assertEqual(x.narrow(-1, -1, 1), torch.tensor([[2], [5], [8]]))
        # 测试 narrow 方法，从第 -2 维索引 -1 开始，长度为 1
        self.assertEqual(x.narrow(-2, -1, 1), torch.tensor([[6, 7, 8]]))

    # TODO: update to work on CUDA, too
    @onlyCPU
    # 使用装饰器 onlyCPU，标识该方法仅在 CPU 上运行
    def test_narrow_tensor(self, device):
        # 创建一个 3x3 的张量 x
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        # 测试 narrow 方法，使用 tensor(0) 作为索引，预期结果为 [[0, 1, 2]]
        self.assertEqual(x.narrow(0, torch.tensor(0), 1), torch.tensor([[0, 1, 2]]))
        # 测试 narrow 方法，使用 tensor(0.0) 作为索引，预期引发异常
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor(0.0), 1)
        # 测试 narrow 方法，使用 tensor([0]) 作为索引，预期引发异常
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0]), 1)
        # 测试 narrow 方法，使用 tensor([0, 1]) 作为索引，预期引发异常
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0, 1]), 1)

    # TODO: make work on CUDA, too
    @onlyCPU
    # 使用装饰器 onlyCPU，标识该方法仅在 CPU 上运行
    def test_t(self, device):
        # 测试 0 维张量的转置
        x = torch.randn(())
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # 测试 1 维张量的转置
        x = torch.arange(4)
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # 测试 2 维张量的转置
        x = torch.rand((2, 2))
        # 测试 t() 方法与 transpose(0, 1) 方法的等效性
        self.assertEqual(x.t(), x.transpose(0, 1))
        x = x.to_sparse()
        self.assertEqual(x.t(), x.transpose(0, 1))

        # 测试 3 维张量的转置
        x = torch.rand((2, 2, 2))
        # 测试 t() 方法对 3 维张量的异常处理
        with self.assertRaisesRegex(
            RuntimeError, "expects a tensor with <= 2 dimensions, but self is 3D"
        ):
            x.t()
        x = x.to_sparse()
        with self.assertRaisesRegex(
            RuntimeError, "expects a tensor with <= 2 sparse and 0 dense dimensions"
        ):
            x.t()

    @onlyCPU
    # 使用装饰器 onlyCPU，标识该方法仅在 CPU 上运行
    # 定义一个名为 test_split 的测试方法，接受一个设备参数
    def test_split(self, device):
        # 创建一个大小为 (7, 4) 的随机张量
        tensor = torch.rand(7, 4)
        # 指定切分大小为 3
        split_size = 3
        # 指定切分维度为 0
        dim = 0
        # 预期的每个切分后张量的大小
        target_sizes = ([3, 4], [3, 4], [1, 4])
        # 对张量进行切分操作，按照指定的大小和维度
        splits = tensor.split(split_size, dim)
        # 初始化起始位置
        start = 0
        # 遍历每个目标大小和对应的切分张量
        for target_size, split in zip(target_sizes, splits):
            # 断言切分后张量的大小与预期目标大小相同
            self.assertEqual(split.size(), target_size)
            # 断言切分后张量与原始张量在指定维度上的子张量相等
            self.assertEqual(
                tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0
            )
            # 更新起始位置
            start = start + target_size[dim]

        # 变量大小的切分操作
        tensor = torch.randn(20, 10)
        # 指定切分维度为 0
        dim = 0
        # 指定每个切分段的大小
        split_sizes = [5, 5, 10]
        # 预期的每个切分后张量的大小
        target_sizes = [[5, 10], [5, 10], [10, 10]]
        # 对张量进行切分操作，按照指定的大小和维度
        splits = tensor.split(split_sizes, dim)
        # 初始化起始位置
        start = 0
        # 遍历每个目标大小和对应的切分张量
        for target_size, split in zip(target_sizes, splits):
            # 断言切分后张量的大小与预期目标大小相同
            self.assertEqual(split.size(), target_size)
            # 断言切分后张量与原始张量在指定维度上的子张量相等
            self.assertEqual(
                tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0
            )
            # 更新起始位置
            start = start + target_size[dim]

        # 指定切分大小
        split_sizes = [2, 2, 6]
        # 预期的每个切分后张量的大小
        target_sizes = ([20, 2], [20, 2], [20, 6])
        # 指定切分维度为 1
        dim = 1
        # 对张量进行切分操作，按照指定的大小和维度
        splits = tensor.split(split_sizes, dim)
        # 初始化起始位置
        start = 0
        # 遍历每个目标大小和对应的切分张量
        for target_size, split in zip(target_sizes, splits):
            # 断言切分后张量的大小与预期目标大小相同
            self.assertEqual(split.size(), target_size)
            # 断言切分后张量与原始张量在指定维度上的子张量相等
            self.assertEqual(
                tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0
            )
            # 更新起始位置
            start = start + target_size[dim]

    # 标记仅在 CPU 上运行的测试方法
    @onlyCPU
    # 定义一个名为 test_chunk 的测试方法，接受一个设备参数
    def test_chunk(self, device):
        # 创建一个大小为 (4, 7) 的随机张量
        tensor = torch.rand(4, 7)
        # 指定要划分的块数
        num_chunks = 3
        # 指定切分维度为 1
        dim = 1
        # 预期的每个切分后张量的大小
        target_sizes = ([4, 3], [4, 3], [4, 1])
        # 对张量进行均匀切分操作，按照指定的块数和维度
        splits = tensor.chunk(num_chunks, dim)
        # 初始化起始位置
        start = 0
        # 遍历每个目标大小和对应的切分张量
        for target_size, split in zip(target_sizes, splits):
            # 断言切分后张量的大小与预期目标大小相同
            self.assertEqual(split.size(), target_size)
            # 断言切分后张量与原始张量在指定维度上的子张量相等
            self.assertEqual(
                tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0
            )
            # 更新起始位置
            start = start + target_size[dim]

        # 非法的切分大小情况
        error_regex = "chunk expects.*greater than 0"
        # 断言在指定情况下会抛出运行时错误并匹配特定的错误消息正则表达式
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(0)
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(-2)

    # 标记仅在 CPU 上运行的测试方法
    @onlyCPU
    # 定义一个名为 test_unsqueeze 的测试方法，接受一个设备参数，并返回空值
    def test_unsqueeze(self, device) -> None:
        # 创建一个大小为 (2, 3, 4) 的随机张量
        x = torch.randn(2, 3, 4)
        # 在指定维度上对张量进行 unsqueeze 操作
        y = x.unsqueeze(1)
        # 断言 unsqueeze 后的张量与使用 view 方法重新组织维度后的张量相等
        self.assertEqual(y, x.view(2, 1, 3, 4))
        # 在原地对张量进行 unsqueeze 操作
        y = x.clone().unsqueeze_(2)
        # 断言 unsqueeze 后的张量与使用 view 方法重新组织维度后的张量相等
        self.assertEqual(y, x.view(2, 3, 1, 4))

        # 选择张量的第二个维度切片
        x = x[:, 1]
        # 断言张量在操作前后不是连续的
        self.assertFalse(x.is_contiguous())
        # 在指定维度上对张量进行 unsqueeze 操作
        y = x.unsqueeze(1)
        # 断言 unsqueeze 后的张量与使用 contiguous 方法重新组织维度后的张量相等
        self.assertEqual(y, x.contiguous().view(2, 1, 4))
        # 在原地对张量进行 unsqueeze 操作
        y = x.clone().unsqueeze_(2)
        # 断言 unsqueeze 后的张量与使用 contiguous 方法重新组织维度后的张量相等
        self.assertEqual(y, x.contiguous().view(2, 4, 1))

    # 用于特殊情况下的转置复制的单元测试（参见 ATen/native/Copy.cpp 了解详细信息）
    # 测试大矩阵的转置操作，使用给定的设备
    def test_big_transpose(self, device):
        # 创建一个随机数填充的张量，形状为 (456, 789)，在指定设备上
        t = torch.rand(456, 789, device=device)
        # 执行转置操作，并确保返回一个连续的张量
        t1 = t.t().contiguous()
        # 使用 numpy 转置创建张量，并从 CPU 上的张量复制数据
        t2 = torch.from_numpy(t.cpu().numpy().transpose())
        # 断言 t1 和 t2 相等
        self.assertEqual(t1, t2)

    # 测试张量的转置操作，使用给定的设备
    def test_T(self, device):
        # 创建一个随机数填充的张量，形状为 (2, 3, 4)，在指定设备上
        a = torch.randn(2, 3, 4, device=device)
        # 执行转置操作
        t1 = a.T
        # 使用 permute 进行维度重排，等效于转置
        t2 = a.permute(2, 1, 0)
        # 断言 t1 和 t2 相等
        self.assertEqual(t2, t1)
        # 创建一个随机数填充的一维张量，形状为 (10)，在指定设备上
        b = torch.randn(10, device=device)
        # 断言 b 和 b.T 相等，因为一维张量的转置等于其本身
        self.assertEqual(b, b.T)

    # 使用不同数据类型测试张量的转置操作，指定设备和数据类型
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_transposes(self, device, dtype):
        # 遍历不同的转置操作：转置（T）、共轭转置（H）、主对角线转置（mT）、副对角线转置（mH）、伴随转置（adjoint）
        for op in ("T", "H", "mT", "mH", "adjoint"):
            # 根据操作类型选择不同的张量形状
            shapes = (
                ((2, 3), (2, 3, 4)) if op[0] == "m" or op == "adjoint" else ((2, 3),)
            )
            # 遍历不同形状的张量
            for shape in shapes:
                # 创建指定形状的张量，指定设备和数据类型
                a = make_tensor(shape, device=device, dtype=dtype)
                # 根据操作类型获取对应的转置结果
                t1 = getattr(a, op)
                # 对于 adjoint 操作，需再次调用获取结果
                if op == "adjoint":
                    t1 = t1()
                # 执行转置操作，将张量在指定维度进行转置
                t2 = a.transpose(-2, -1)
                # 对于共轭转置操作或伴随转置，需执行共轭操作
                if op[-1] == "H" or op == "adjoint":
                    t2 = t2.conj()
                # 断言转置后的结果 t2 和操作后的结果 t1 相等
                self.assertEqual(t2, t1)

    # 使用不同数据类型测试张量转置操作中的错误情况，指定设备和数据类型
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_transposes_errors(self, device, dtype):
        # 遍历不同的转置操作：共轭转置（H）、主对角线转置（mT）、副对角线转置（mH）、伴随转置（adjoint）
        for op in ("H", "mT", "mH", "adjoint"):
            # 根据操作类型选择不同的张量形状
            shapes = ((2,), (2, 3, 4)) if op == "H" else ((2,),)
            # 遍历不同形状的张量
            for shape in shapes:
                # 创建指定形状的张量，指定设备和数据类型
                a = make_tensor(shape, device=device, dtype=dtype)
                # 使用断言捕获期望的运行时错误，当尝试在非矩阵上执行操作时
                with self.assertRaisesRegex(RuntimeError, "only supported on matrices"):
                    t1 = getattr(a, op)
                    # 对于伴随转置操作，需再次调用获取结果
                    if op == "adjoint":
                        t1 = t1()

    # 测试张量的数据类型转换操作，使用给定的设备
    def test_python_types(self, device):
        # 创建具有指定数据类型的随机数填充的张量，形状为 (1, 2)，在指定设备上
        a1 = torch.randn((1, 2), device=device, dtype=torch.float64)
        # 创建具有 Python 原生数据类型的随机数填充的张量，形状为 (1, 2)，在指定设备上
        a2 = torch.randn((1, 2), device=device, dtype=float)
        # 断言 a1 和 a2 的数据类型相等
        self.assertEqual(a1.dtype, a2.dtype)

        # 创建具有指定数据类型的从 10 到 19 的整数张量，设备上
        b1 = torch.arange(10, 20, dtype=torch.int64, device=device)
        # 创建具有 Python 原生整数数据类型的从 10 到 19 的整数张量，设备上
        b2 = torch.arange(10, 20, dtype=int, device=device)
        # 断言 b1 和 b2 的数据类型相等
        self.assertEqual(b1.dtype, b2.dtype)

        # 创建具有指定数据类型的布尔张量，设备上
        c1 = torch.tensor([True, False], dtype=torch.bool, device=device)
        # 创建具有 Python 原生布尔数据类型的布尔张量，设备上
        c2 = torch.tensor([True, False], dtype=bool, device=device)
        # 断言 c1 和 c2 的数据类型相等
        self.assertEqual(c1.dtype, c2.dtype)

    # 测试张量的 resize_as_ 方法，确保保留步长
    # TODO: is resize best put in test_view_ops?
    def test_resize_as_preserves_strides(self, device):
        # 创建一个空的张量，形状为 (2, 3)，并对其进行转置
        x = torch.empty(2, 3).t()
        # 记录转置前的步长信息
        old_strides = x.stride()
        # 调整张量大小，使其与自身大小相同
        x.resize_as_(x)
        # 断言调整大小后的步长信息与之前相同
        self.assertEqual(x.stride(), old_strides)
    # 定义一个测试方法，用于验证在指定设备上的内存格式调整功能
    def test_memory_format_resize_as(self, device):
        # 定义内部辅助函数，用于测试给定形状和内存格式的张量调整
        def test_helper(shape, memory_format, device):
            # 创建一个在指定设备上的随机张量，并确保其连续性和指定的内存格式
            xc = torch.randn(shape, device=device).contiguous(
                memory_format=memory_format
            )
            # 创建一个与 xc 元素数量相同的随机张量，并将其大小调整为与 xc 相同，保留原有的内存格式
            flat = torch.randn(xc.numel(), device=device)
            flat.resize_as_(xc, memory_format=torch.preserve_format)
            # 断言调整后的张量保持连续性和指定的内存格式
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        # 测试使用 channels_last 内存格式的不同形状的张量调整
        test_helper((10, 3, 32, 32), torch.channels_last, device)
        # 测试使用 channels_last_3d 内存格式的不同形状的张量调整
        test_helper((3, 10, 3, 32, 32), torch.channels_last_3d, device)

    # 定义一个测试方法，用于验证在指定设备上的 resize_ 方法与内存格式相关的功能
    def test_memory_format_resize_(self, device):
        # 定义内部辅助函数，用于测试在给定形状、元素数量和内存格式下的张量 resize_ 功能
        def test_helper(shape, numel, memory_format, device):
            # 创建一个在指定设备上具有随机值的扁平张量
            flat = torch.randn(numel, device=device)
            # 使用指定的形状和内存格式调整此扁平张量
            flat.resize_(shape, memory_format=memory_format)
            # 断言调整后的张量保持连续性和指定的内存格式
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        # 测试使用 channels_last 内存格式的不同形状的扁平张量 resize_
        test_helper((10, 3, 32, 32), 10 * 3 * 32 * 32, torch.channels_last, device)
        # 测试使用 channels_last_3d 内存格式的不同形状的扁平张量 resize_
        test_helper(
            (3, 10, 3, 32, 32), 3 * 10 * 3 * 32 * 32, torch.channels_last_3d, device
        )

    # 标记为仅适用于本地设备类型的测试装饰器，以及指定的数据类型
    @onlyNativeDeviceTypes
    @dtypes(torch.int64, torch.float, torch.complex128)
    # 定义一个测试方法，用于验证在指定设备上的转置操作的异常情况处理
    def test_transpose_invalid(self, device, dtype):
        # 遍历三种不同的转置函数
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            # 生成一个随机形状的输入张量和指定设备上的数据类型
            shape = _rand_shape(4, min_size=5, max_size=10)
            x = _generate_input(shape, dtype, device, False)

            # 测试当源维度或目标维度超出范围时，函数是否能正确抛出 IndexError 异常
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

    # 标记为仅适用于指定数据类型的测试装饰器
    @dtypes(torch.int64, torch.float, torch.complex128)
    # 定义测试函数，用于比较 torch 中的维度操作函数和 numpy 中的对应函数
    def test_transpose_vs_numpy(self, device, dtype):
        # 遍历三个 torch 的维度操作函数：swapdims, swapaxes, transpose
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            # 遍历不同维度数目，从 0 到 4 维
            for nd in range(5):
                # 随机生成一个形状为 nd 的张量的形状
                shape = _rand_shape(nd, min_size=5, max_size=10)
                # 生成一个输入张量 x，具体实现未给出，用于后续比较
                x = _generate_input(shape, dtype, device, with_extremal=False)
                # 对随机生成的负数进行处理，随机选取维度对进行操作
                for random_negative in [True, False]:
                    # 生成随机概率
                    for src_dim, dst_dim in permutations(range(nd), r=2):
                        random_prob = random.random()

                        # 根据概率和随机负数标志调整源维度和目标维度
                        if random_negative and random_prob > 0.66:
                            src_dim = src_dim - nd
                        elif random_negative and random_prob > 0.33:
                            dst_dim = dst_dim - nd
                        elif random_negative:
                            src_dim = src_dim - nd
                            dst_dim = dst_dim - nd

                        # 部分映射表，将 torch 函数映射到其偏函数，用于后续调用
                        partial_map = {
                            torch.swapdims: partial(
                                torch.swapdims, dim0=src_dim, dim1=dst_dim
                            ),
                            torch.swapaxes: partial(
                                torch.swapaxes, axis0=src_dim, axis1=dst_dim
                            ),
                            torch.transpose: partial(
                                torch.transpose, dim0=src_dim, dim1=dst_dim
                            ),
                        }

                        # 根据当前的 fn 选择对应的 torch 函数
                        torch_fn = partial_map[fn]
                        # 生成对应的 numpy 函数偏函数
                        np_fn = partial(np.swapaxes, axis1=src_dim, axis2=dst_dim)
                        # 调用 self.compare_with_numpy 方法，比较 torch_fn 和 np_fn 的结果
                        self.compare_with_numpy(
                            torch_fn, np_fn, x, device=None, dtype=None
                        )

            # 移动维度到相同位置的特殊情况处理
            # 生成一个形状为 (2, 3, 5, 7, 11) 的张量 x
            x = torch.randn(2, 3, 5, 7, 11)
            # 部分映射表，将 torch 函数映射到其偏函数，用于后续调用
            partial_map = {
                torch.swapdims: partial(torch.swapdims, dim0=0, dim1=0),
                torch.swapaxes: partial(torch.swapaxes, axis0=0, axis1=0),
                torch.transpose: partial(torch.transpose, dim0=0, dim1=0),
            }
            # 根据当前的 fn 选择对应的 torch 函数
            torch_fn = partial_map[fn]
            # 生成对应的 numpy 函数偏函数
            np_fn = partial(np.swapaxes, axis1=0, axis2=0)
            # 调用 self.compare_with_numpy 方法，比较 torch_fn 和 np_fn 的结果
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)
    # 定义一个测试函数，用于检查输入张量的最小维度要求
    def _test_atleast_dim(self, torch_fn, np_fn, device, dtype):
        # 循环不同的维度数量，从0到4
        for ndims in range(0, 5):
            # 生成一个随机形状的张量
            shape = _rand_shape(ndims, min_size=5, max_size=10)
            # 遍历当前维度数量加一次，以测试不同张量尺寸的情况
            for n in range(ndims + 1):
                # 遍历是否使用极端值的情况（False/True）
                for with_extremal in [False, True]:
                    # 遍历是否要求张量是连续的（False/True）
                    for contiguous in [False, True]:
                        # 生成输入张量
                        x = _generate_input(shape, dtype, device, with_extremal)
                        # 如果需要连续的张量，进行转置操作
                        if contiguous:
                            x = x.T
                        # 调用函数，比较结果与NumPy的实现
                        self.compare_with_numpy(
                            torch_fn, np_fn, x, device=None, dtype=None
                        )

                        # 比较序列输入的情况
                        torch_sequence_x = (x,) * random.randint(3, 10)
                        np_sequence_x = tuple(
                            np.array(x.detach().cpu().numpy()) for x in torch_sequence_x
                        )
                        torch_res = torch_fn(*torch_sequence_x)
                        np_res = np_fn(*np_sequence_x)

                        # 转换为与NumPy兼容的格式
                        torch_res = tuple(x.cpu() for x in torch_res)
                        np_res = tuple(torch.from_numpy(x) for x in np_res)
                        # 断言两者结果相等
                        self.assertEqual(np_res, torch_res)

    # TODO: 这些是视图操作吗？
    @dtypes(*all_types_and_complex_and(torch.half))
    # 定义一个测试函数，测试至少扩展到1维、2维、3维的情况
    def test_atleast(self, device, dtype):
        self._test_atleast_dim(torch.atleast_1d, np.atleast_1d, device, dtype)
        self._test_atleast_dim(torch.atleast_2d, np.atleast_2d, device, dtype)
        self._test_atleast_dim(torch.atleast_3d, np.atleast_3d, device, dtype)

    # TODO: 对这个进行OpInfo
    # 定义一个测试函数，用于测试不同维度的情况
    def _test_atleast(self, device, torch_fn):
        # 0维情况
        s = torch.tensor(0.5, dtype=torch.double, requires_grad=True)

        # 检查梯度
        gradcheck(lambda x: torch_fn(x), s)
        # 检查二阶梯度
        gradgradcheck(lambda x: torch_fn(x), s)

        # 1维情况
        a = torch.rand(4, dtype=torch.double, requires_grad=True)

        # 检查梯度
        gradcheck(lambda x: torch_fn(x), a)
        # 检查二阶梯度
        gradgradcheck(lambda x: torch_fn(x), a)

        # 2, 3, 4维情况
        b = torch.rand(4, 3, dtype=torch.double, requires_grad=True)
        c = torch.rand(4, 3, 2, dtype=torch.double, requires_grad=True)
        d = torch.rand(4, 3, 2, 1, dtype=torch.double, requires_grad=True)

        input_tuple = (s, a, b, c, d)
        # 检查梯度
        gradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)
        # 检查二阶梯度
        gradgradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)

    # 定义一个测试函数，测试至少扩展函数在不同维度的情况下的梯度
    def test_atleast_gradient(self, device):
        self._test_atleast(device, torch.atleast_1d)
        self._test_atleast(device, torch.atleast_2d)
        self._test_atleast(device, torch.atleast_3d)

    # 仅在CPU上执行，且仅对torch.float类型执行
    @onlyCPU
    @dtypes(torch.float)
    # 定义一个测试函数，用于测试广播张量的功能，参数包括设备和数据类型
    def test_broadcast_tensors(self, device, dtype):
        # 创建一个形状为 (2, 1, 3) 的张量 x0，数据类型为 dtype，在指定设备上生成随机数据
        x0 = torch.randn(2, 1, 3, dtype=dtype, device=device)
        # 创建一个形状为 (3,) 的张量 x1，数据类型为 dtype，在指定设备上生成随机数据
        x1 = torch.randn(3, dtype=dtype, device=device)
        # 创建一个形状为 (3, 1) 的张量 x2，数据类型为 dtype，在指定设备上生成随机数据
        x2 = torch.randn(3, 1, dtype=dtype, device=device)
        # 预期的输出张量形状应为 (2, 3, 3)
        expected_size = (2, 3, 3)

        # 使用 torch.broadcast_tensors 函数对 x0, x1, x2 进行广播
        y0, y1, y2 = torch.broadcast_tensors(x0, x1, x2)
        # 断言 y0 的形状与预期形状相同
        self.assertTrue(y0.size() == expected_size)
        # 断言 y1 的形状与预期形状相同
        self.assertTrue(y1.size() == expected_size)
        # 断言 y2 的形状与预期形状相同
        self.assertTrue(y2.size() == expected_size)

    # 以下是一个装饰器，限制该测试函数仅在 CPU 上运行
    @onlyCPU
    # 定义一个测试函数，用于测试广播形状的功能，接受设备参数
    def test_broadcast_shapes(self, device):
        # 定义多个示例形状
        examples = [(), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)]
        # 遍历每个示例形状
        for s0 in examples:
            # 根据当前形状创建随机张量 x0
            x0 = torch.randn(s0)
            # 计算使用 broadcast_tensors 函数广播后的形状，并取第一个张量的形状
            expected = torch.broadcast_tensors(x0)[0].shape
            # 调用被测试的函数，计算其返回的形状
            actual = torch.broadcast_shapes(s0)
            # 断言期望的形状与实际计算得到的形状相等
            self.assertEqual(expected, actual)

            # 再次遍历每个示例形状，用于测试两个张量的广播
            for s1 in examples:
                # 根据当前形状创建随机张量 x1
                x1 = torch.randn(s1)
                # 计算使用 broadcast_tensors 函数广播后的形状，并取第一个张量的形状
                expected = torch.broadcast_tensors(x0, x1)[0].shape
                # 调用被测试的函数，计算其返回的形状
                actual = torch.broadcast_shapes(s0, s1)
                # 断言期望的形状与实际计算得到的形状相等
                self.assertEqual(expected, actual)

        # 定义包含整数输入的列表，用于测试不同输入的广播形状
        inputs_list = [[1, 4], [4, 1], [1, 1, 3]]
        # 遍历整数输入的列表
        for integral_inputs in inputs_list:
            # 调用被测试的函数，计算其返回的形状
            res1 = torch.broadcast_shapes(*integral_inputs)
            # 使用 map 函数创建空张量后，调用 broadcast_tensors 函数计算广播后的形状
            res2 = torch.broadcast_tensors(*map(torch.empty, integral_inputs))[0].shape
            # 断言期望的形状与实际计算得到的形状相等
            self.assertEqual(res1, res2)

        # 包含负值的输入列表，用于测试广播形状时出现负维度的情况
        inputs_with_neg_vals = [
            [1, 1, -12],
            [-1, 1],
            [-11,  # 注意：这里是一个单独的负数输入
        ]
        # 遍历包含负值的输入列表
        for integral_inputs_with_neg_vals in inputs_with_neg_vals:
            # 用于断言异常，测试是否会抛出 RuntimeError 异常并包含指定的错误信息
            with self.assertRaisesRegex(
                RuntimeError, "Trying to create tensor with negative dimension"
            ):
                # 调用被测试的函数，测试其在有负值输入时的行为
                torch.broadcast_shapes(*integral_inputs_with_neg_vals)

        # 包含引发错误情况的整数输入，用于测试不匹配的形状广播
        integral_inputs_error_case = [(3, 5), (2, 4, 1)]
        # 遍历引发错误情况的整数输入列表
        for error_input in integral_inputs_error_case:
            # 用于断言异常，测试是否会抛出 RuntimeError 异常并包含指定的错误信息
            with self.assertRaisesRegex(
                RuntimeError,
                "Shape mismatch: objects cannot be broadcast to a single shape",
            ):
                # 调用被测试的函数，测试其在形状不匹配时的行为
                torch.broadcast_shapes(*error_input)

        # 包含负数输入的列表，用于测试当输入包含负数时的异常情况
        negative_inputs = [(-1,), (1, -12), (4, -11), (-4, 1), (1, 1, -2)]
        # 遍历包含负数输入的列表
        for s0 in negative_inputs:
            # 用于断言异常，测试是否会抛出 RuntimeError 异常并包含指定的错误信息
            with self.assertRaisesRegex(
                RuntimeError, "Trying to create tensor with negative dimension"
            ):
                # 调用被测试的函数，测试其在有负数输入时的行为
                torch.broadcast_shapes(s0)

            # 再次遍历负数输入列表，用于测试两个负数输入的情况
            for s1 in negative_inputs:
                # 用于断言异常，测试是否会抛出 RuntimeError 异常并包含指定的错误信息
                with self.assertRaisesRegex(
                    RuntimeError, "Trying to create tensor with negative dimension"
                ):
                    # 调用被测试的函数，测试其在有两个负数输入时的行为
                    torch.broadcast_shapes(s0, s1)

        # 浮点数输入引发错误情况的列表，用于测试输入类型不匹配的异常情况
        float_inputs_error_case = [(1.1, 2.0), (1.1, 1.0)]
        # 遍历浮点数输入引发错误情况的列表
        for error_case in float_inputs_error_case:
            # 遍历每个浮点数输入
            for float_input in error_case:
                # 用于断言异常，测试是否会抛出 RuntimeError 异常并包含指定的错误信息
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Input shapes "
                    "should be of type ints, a tuple of ints, or a list of ints",
                ):
                    # 调用被测试的函数，测试其在浮点数输入时的行为
                    torch.broadcast_shapes(float_input)

        # 包含不同输入类型的列表，用于测试不同类型输入的广播形状
        diff_input_types = [(1, (5,)), (3, (1,)), (1, (3, 4))]
        # 遍历包含不同输入类型的列表
        for s0 in diff_input_types:
            # 调用被测试的函数，计算其返回的形状
            res1 = torch.broadcast_shapes(*s0)
            # 使用 map 函数创建空张量后，调用 broadcast_tensors 函数计算广播后的形状
            res2 = torch.broadcast_tensors(*map(torch.empty, s0))[0].shape
            # 断言期望的形状与实际计算得到的形状相等
            self.assertEqual(res1, res2)

    # 由于 numpy 不支持 BFloat16，跳过相关测试
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    def test_broadcast_to(self, device, dtype):
        # 内部函数，判断两个尺寸是否可以进行广播
        def can_broadcast(s0, s1):
            # 将 s0 和 s1 反转以便比较尾部维度
            s0 = tuple(reversed(s0))
            s1 = tuple(reversed(s1))
            for i in range(len(s0)):
                # 如果 s0[i] 不是 1 且不等于 s1[i]，则不能广播
                if s0[i] != 1 and s0[i] != s1[i]:
                    return False
            return True

        # 不同的尺寸组合
        sizes = ((), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2))
        # 对于每一对尺寸 s0 和 s1 进行测试
        for s0, s1 in combinations(sizes, r=2):
            # 创建指定尺寸的 tensor，用于测试
            t = make_tensor(s0, dtype=dtype, device=device, low=-9, high=9)
            # 将 tensor 转换为 numpy 数组
            t_np = t.cpu().numpy()

            # 如果 s0 可以广播到 s1
            if can_broadcast(s0, s1):
                # 使用 torch 的广播功能创建结果 tensor
                res = torch.broadcast_to(t, s1)
                # 使用 numpy 的广播功能创建 numpy 数组作为对比
                np_res = np.broadcast_to(t_np, s1)
                # 断言 torch 广播结果和 numpy 广播结果相等
                self.assertEqual(res, np_res)
            else:
                # 如果 s0 不能广播到 s1，则预期引发 RuntimeError 异常
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"The expanded size of the tensor \(\d\) "
                    r"must match the existing size \(\d\)",
                ):
                    torch.broadcast_to(t, s1)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    # 测试 tensor 的 reshape 视图语义
    def test_reshape_view_semantics(self, device, dtype):
        # 创建指定尺寸和类型的 tensor
        tensor = make_tensor((15, 4), dtype=dtype, device=device)
        # 目标 reshape 尺寸
        target = (20, 3)

        # 可以作为视图返回的情况
        view_tensor = tensor.reshape(target)
        # 断言视图 tensor 的尺寸和目标尺寸相等
        self.assertEqual((view_tensor.size()), target)
        # 断言原始 tensor 和视图 tensor 的存储地址相同
        self.assertEqual(tensor.storage().data_ptr(), view_tensor.storage().data_ptr())

        # 必须复制的情况（transpose 使得 tensor 不连续，需要复制）
        copy_tensor = tensor.transpose(0, 1).reshape(target)
        # 断言复制 tensor 的尺寸和目标尺寸相等
        self.assertEqual(copy_tensor.size(), target)
        # 断言原始 tensor 和复制 tensor 的存储地址不同
        self.assertNotEqual(
            tensor.storage().data_ptr(), copy_tensor.storage().data_ptr()
        )

    # 测试 tensor 的连续性
    def test_contiguous(self, device):
        # 创建指定设备上的随机 tensor
        x = torch.randn(1, 16, 5, 5, device=device)
        # 断言 tensor 是连续的
        self.assertTrue(x.is_contiguous())
        # 获取 tensor 的 stride 列表
        stride = list(x.stride())
        # 修改第 0 维的 stride，但是 tensor 仍然是连续的，因为 size[0] 是 1
        x.set_(x.storage(), 0, x.size(), stride)
        # 再次断言 tensor 是连续的
        self.assertTrue(x.is_contiguous())

    @onlyNativeDeviceTypes
    # 跳过 torch.bfloat16，因为 numpy 不支持它
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    # 定义测试方法，用于测试张量在指定设备和数据类型下的分割操作
    def test_tensor_split_sections(self, device, dtype):
        # 定义不同的输入尺寸列表，每个元素表示一个张量的大小
        input_sizes = [
            (0,),        # 空张量
            (10,),       # 一维张量，长度为10
            (10, 0),     # 二维张量，第二维长度为0
            (0, 10),     # 二维张量，第一维长度为0
            (4, 10),     # 二维张量，各维长度为4和10
            (12, 3),     # 二维张量，各维长度为12和3
        ]
        # 遍历每个输入尺寸
        for input_size in input_sizes:
            # 创建基础张量，设备为指定设备，数据类型为指定数据类型，值在-9到9之间
            a_base = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)
            # 如果张量的维度大于2，则测试其转置后的输入
            # 否则，只使用原始张量进行测试
            for a in [a_base, a_base.t()] if a_base.dim() > 2 else [a_base]:
                # 将张量转换为numpy数组，以便后续使用numpy进行比较
                a_n = a.cpu().numpy()
                # 遍历张量的每一个维度
                for dim in range(-a.dim(), a.dim()):
                    # 对于每个维度，遍历1到2倍于该维度大小的分割数
                    for sections in range(1, 2 * a.size(dim)):
                        # 创建用于打印消息的字符串，包含输入尺寸、分割数和维度信息
                        msg = f"input_size {input_size}, sections {sections}, dim {dim}"
                        # 使用torch.tensor_split函数按指定维度分割张量，并得到两个结果
                        result1 = torch.tensor_split(a, sections, dim)
                        result2 = torch.tensor_split(
                            a, torch.tensor(sections, dtype=torch.int64), dim
                        )
                        # 遍历两种分割结果，比较设备和数据类型是否一致
                        for r1, r2 in zip(result1, result2):
                            self.assertEqual(r1.device, torch.device(device), msg=msg)
                            self.assertEqual(r1.dtype, dtype, msg=msg)
                            self.assertEqual(r2.device, torch.device(device), msg=msg)
                            self.assertEqual(r2.dtype, dtype, msg=msg)
                        # 使用numpy的array_split函数按指定维度分割numpy数组
                        result_n = np.array_split(a_n, sections, dim)
                        # 断言torch分割结果与numpy分割结果是否一致
                        self.assertEqual(result_n, result1, msg=msg)
                        self.assertEqual(result_n, result2, msg=msg)

    @onlyNativeDeviceTypes
    # 由于numpy不支持BFloat16类型，跳过这种类型的测试
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    # 定义一个测试方法，用于测试在给定设备和数据类型下的张量分割操作
    def test_tensor_split_indices(self, device, dtype):
        # 定义多个输入尺寸的示例
        input_sizes = [
            (0,),        # 空张量
            (10,),       # 一维张量，长度为10
            (10, 0),     # 二维张量，第一维长度为10，第二维长度为0
            (0, 10),     # 二维张量，第一维长度为0，第二维长度为10
            (4, 10),     # 二维张量，第一维长度为4，第二维长度为10
            (12, 3),     # 二维张量，第一维长度为12，第二维长度为3
        ]
        # 定义多组索引参数的示例
        indices_args = [
            (),          # 空索引
            (0,),        # 单个索引为0
            (3,),        # 单个索引为3
            (10,),       # 单个索引为10
            (-1,),       # 单个负索引为-1
            (-10,),      # 单个负索引为-10
            (2, -1),     # 多个索引，包括正索引2和负索引-1
            (3, 4, 10),   # 多个索引，包括3、4、10
            (0, -1, 0, 10),  # 多个索引，包括0、-1、0、10
            (1, 5, 2, 8),    # 多个索引，包括1、5、2、8
        ]
        # 遍历每个输入尺寸示例
        for input_size in input_sizes:
            # 创建基础张量a_base，根据给定设备、数据类型、范围生成
            a_base = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)
            # 如果a_base的维度大于2，则在转置后的输入上运行测试
            for a in [a_base, a_base.t()] if a_base.dim() > 2 else [a_base]:
                # 将张量a转换为numpy数组a_n
                a_n = a.cpu().numpy()
                # 对张量a的每个维度进行迭代
                for dim in range(-a.dim(), a.dim()):
                    # 对每组索引参数进行迭代
                    for indices in indices_args:
                        # 使用torch.tensor_split函数分割张量a，返回结果为result_1
                        result_1 = torch.tensor_split(a, indices, dim)
                        # 使用torch.tensor_split函数分割张量a，将indices转换为torch.int64，返回结果为result_2
                        result_2 = torch.tensor_split(
                            a, torch.tensor(indices, dtype=torch.int64), dim
                        )
                        # 设置测试消息msg，包括input_size、indices和dim的信息
                        msg = f"input_size {input_size}, indices {indices}, dim {dim}"
                        # 对result_1和result_2中的每对张量进行比较
                        for r1, r2 in zip(result_1, result_2):
                            # 断言r1的设备与给定设备一致
                            self.assertEqual(r1.device, torch.device(device), msg=msg)
                            # 断言r1的数据类型与给定数据类型一致
                            self.assertEqual(r1.dtype, dtype, msg=msg)
                            # 断言r2的设备与给定设备一致
                            self.assertEqual(r2.device, torch.device(device), msg=msg)
                            # 断言r2的数据类型与给定数据类型一致
                            self.assertEqual(r2.dtype, dtype, msg=msg)
                        # 使用numpy的array_split函数，以indices和dim对a_n进行分割，返回结果为result_n
                        result_n = np.array_split(a_n, indices, dim)
                        # 断言result_n与result_1相等
                        self.assertEqual(result_n, result_1, msg=msg)
                        # 断言result_n与result_2相等
                        self.assertEqual(result_n, result_2, msg=msg)
    # 定义测试函数 test_tensor_split_errors，接受参数 self 和 device
    def test_tensor_split_errors(self, device):
        # 定义变量 S 并赋值为 10
        S = 10
        # 定义测试用例列表 test_cases
        test_cases = [
            # 第一个测试用例
            # input size, sections or indices, dim, error type, error message, numpy error type
            [(S,), 10, 1, IndexError, r"Dimension out of range", IndexError],
            # 第二个测试用例
            [
                (),
                10,
                0,
                RuntimeError,
                r"tensor_split expected at least a 1-dimensional tensor, "
                + "but got a tensor with 0 dims",
                IndexError,
            ],
            # 第三个测试用例
            [(S,), (10,), 1, IndexError, r"Dimension out of range", IndexError],
            # 第四个测试用例
            [
                (),
                (10,),
                0,
                RuntimeError,
                r"tensor_split expected at least a 1-dimensional tensor, "
                + "but got a tensor with 0 dims",
                IndexError,
            ],
            # 第五个测试用例
            [
                (S,),
                0,
                0,
                RuntimeError,
                r"number of sections must be larger than 0, got 0",
                ValueError,
            ],
            # 第六个测试用例
            [
                (S,),
                -1,
                0,
                RuntimeError,
                r"number of sections must be larger than 0, got -1",
                ValueError,
            ],
        ]
        # 遍历测试用例
        for input_size, sections_or_indices, dim, err, err_msg, numpy_err in test_cases:
            # 使用 torch.randn 创建张量 a，形状为 input_size，设备为 device
            a = torch.randn(input_size, device=device)
            # 格式化消息字符串
            msg = f"input_size {input_size}, sections_or_indices {sections_or_indices}, dim {dim}"
            # 断言抛出异常 err，异常消息为 err_msg，消息为 msg
            with self.assertRaisesRegex(err, err_msg, msg=msg):
                torch.tensor_split(a, sections_or_indices, dim)
            # 再次断言，使用张量形式的 sections_or_indices
            with self.assertRaisesRegex(err, err_msg, msg=msg):
                torch.tensor_split(a, torch.tensor(sections_or_indices), dim)
            # 最后使用 numpy 的 array_split 进行断言，期待抛出 numpy_err 异常
            with self.assertRaises(numpy_err, msg=msg):
                np.array_split(a.cpu().numpy(), sections_or_indices, dim)

        # 针对 tensor_split 和 tensor_indices_or_sections 的附加测试
        # 断言抛出 RuntimeError 异常，消息中包含指定的错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            r"tensor_split expected tensor_indices_or_sections to have dtype of long, but got Float",
        ):
            torch.tensor_split(a, torch.tensor(1.1), dim)

        # 再次断言，期待抛出 RuntimeError 异常，消息中包含指定的错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            r"tensor_split expected tensor_indices_or_sections to be a"
            + " zero-dimensional or one-dimensional tensor, but got a tensor with 2 dims",
        ):
            torch.tensor_split(torch.rand(S, device=device), torch.tensor(((1,),)), 0)

    # 定义测试函数 test_resize_all_dtypes_and_devices，接受参数 self 和 device
    def test_resize_all_dtypes_and_devices(self, device):
        # 定义形状 shape 为 (2, 2)
        shape = (2, 2)
        # 遍历所有数据类型和复杂数据类型，包括 torch.half, torch.bfloat16, torch.bool
        for dt in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
            # 使用指定的数据类型 dt 和设备 device 创建张量 x
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            # 调整张量 x 的大小为 shape
            x.resize_(shape)
            # 断言张量 x 的形状与 shape 相等
            self.assertEqual(shape, x.shape)
    # 对所有数据类型和设备执行测试，调整张量大小以匹配另一个张量的大小
    def test_resize_as_all_dtypes_and_devices(self, device):
        # 遍历所有数据类型和复杂类型以及指定的设备
        for dt in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
            # 创建一个张量 x，指定数据类型和设备，形状为 3x2
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            # 创建一个张量 y，指定数据类型和设备，形状为 2x3
            y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dt, device=device)
            # 调整张量 x 的大小以匹配张量 y 的大小
            x.resize_as_(y)
            # 断言调整后的张量 x 的形状与张量 y 的形状相同
            self.assertEqual(y.shape, x.shape)

    # 仅在本机设备类型上执行测试
    @onlyNativeDeviceTypes
    def test_resize_overflow(self, device):
        # 创建一个空张量 x，数据类型为 torch.float64
        x = torch.empty((), dtype=torch.float64)
        # 使用断言捕获 RuntimeError 异常，并检查是否包含指定的错误信息
        with self.assertRaisesRegex(
            RuntimeError, "Storage size calculation overflowed"
        ):
            # 尝试调整张量 x 的大小为 [2, 4, 2**29, 2**29]
            x.resize_([2, 4, 2**29, 2**29])
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            # 尝试调整张量 x 的大小为 [8, 8, 2**29, 2**29]
            x.resize_([8, 8, 2**29, 2**29])
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            # 尝试调整张量 x 的大小为 [0, 4, 2305843009213693952]
            x.resize_([0, 4, 2305843009213693952])

    # 对所有数据类型和设备执行测试，检查张量视图的形状
    def test_view_all_dtypes_and_devices(self, device):
        # 遍历所有数据类型和复杂类型以及指定的设备
        for dt in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
            # 创建一个张量 x，指定数据类型和设备，形状为 3x2
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            # 断言将张量 x 转换为视图后的形状为 [6]
            self.assertEqual(x.view(6).shape, [6])

    # 如果 Torch Dynamo 中未实现 "conj bit"，则跳过在 CPU 上执行测试
    @skipIfTorchDynamo("conj bit not implemented in TensorVariable yet")
    @onlyCPU
    def test_conj_neg_view_numpy_error(self, device):
        # 使用断言捕获 RuntimeError 异常，并检查是否包含指定的错误信息
        self.assertRaisesRegex(
            RuntimeError,
            "has conjugate bit set",
            # 调用 lambda 函数以捕获 torch.tensor([1 + 2j]).conj().numpy() 的错误信息
            lambda: torch.tensor([1 + 2j]).conj().numpy(),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "has negative bit set",
            # 调用 lambda 函数以捕获 torch.tensor([1 + 2j]).conj().imag.numpy() 的错误信息
            lambda: torch.tensor([1 + 2j]).conj().imag.numpy(),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "not supported for conjugate view tensors",
            # 调用 lambda 函数以捕获 torch.tensor([1 + 2j]).conj().view(torch.float64) 的错误信息
            lambda: torch.tensor([1 + 2j]).conj().view(torch.float64),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "not supported for tensors with negative bit set",
            # 调用 lambda 函数以捕获 torch.tensor([1 + 2j]).conj().imag.view(torch.int32) 的错误信息
            lambda: torch.tensor([1 + 2j]).conj().imag.view(torch.int32),
        )

    # 仅在 CPU 上执行测试，检查稀疏张量的行索引和列索引
    @onlyCPU
    def test_crow_col_indices(self, device):
        # 定义稀疏张量的行索引、列索引和值
        crow_indices = (0, 1, 2)
        col_indices = (1, 0)
        values = (1, 2)
        # 创建一个稀疏 CSR 张量 t，指定行索引、列索引和值，形状为 (2, 2)
        t = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 2))
        # 调用 t 的 crow_indices() 方法，触发内部断言以检查使用计数是否大于 1（在调试版本中）
        t.crow_indices()
        # 调用 t 的 col_indices() 方法
        t.col_indices()
# 使用函数 instantiate_device_type_tests 实例化测试类 TestViewOps 的设备类型相关测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestViewOps, globals(), include_lazy=True)

# 使用函数 instantiate_device_type_tests 实例化测试类 TestOldViewOps 的设备类型相关测试，并将其添加到全局命名空间中
instantiate_device_type_tests(TestOldViewOps, globals())

# 如果当前脚本作为主程序运行，则执行函数 run_tests()，通常用于运行测试套件
if __name__ == "__main__":
    run_tests()
```