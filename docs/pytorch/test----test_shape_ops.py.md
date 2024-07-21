# `.\pytorch\test\test_shape_ops.py`

```
# Owner(s): ["module: tests"]

import random  # 导入随机数模块
import unittest  # 导入单元测试模块
import warnings  # 导入警告模块
from functools import partial  # 导入 functools 模块中的 partial 函数

from itertools import chain, combinations, permutations, product  # 导入 itertools 中的若干函数

import numpy as np  # 导入 NumPy 库，使用别名 np

import torch  # 导入 PyTorch 库

from torch import nan  # 从 torch 模块中导入 nan 常量
from torch.testing import make_tensor  # 从 torch.testing 模块导入 make_tensor 函数
from torch.testing._internal.common_device_type import (
    dtypes,  # 从 torch.testing._internal.common_device_type 导入 dtypes
    dtypesIfCUDA,  # 从 torch.testing._internal.common_device_type 导入 dtypesIfCUDA
    instantiate_device_type_tests,  # 从 torch.testing._internal.common_device_type 导入 instantiate_device_type_tests
    largeTensorTest,  # 从 torch.testing._internal.common_device_type 导入 largeTensorTest
    onlyCPU,  # 从 torch.testing._internal.common_device_type 导入 onlyCPU
    onlyCUDA,  # 从 torch.testing._internal.common_device_type 导入 onlyCUDA
    onlyNativeDeviceTypes,  # 从 torch.testing._internal.common_device_type 导入 onlyNativeDeviceTypes
)
from torch.testing._internal.common_dtype import (
    all_types,  # 从 torch.testing._internal.common_dtype 导入 all_types
    all_types_and,  # 从 torch.testing._internal.common_dtype 导入 all_types_and
    all_types_and_complex_and,  # 从 torch.testing._internal.common_dtype 导入 all_types_and_complex_and
)
from torch.testing._internal.common_utils import (
    IS_JETSON,  # 从 torch.testing._internal.common_utils 导入 IS_JETSON 常量
    run_tests,  # 从 torch.testing._internal.common_utils 导入 run_tests 函数
    skipIfTorchDynamo,  # 从 torch.testing._internal.common_utils 导入 skipIfTorchDynamo 装饰器
    TEST_PRIVATEUSE1_DEVICE_TYPE,  # 从 torch.testing._internal.common_utils 导入 TEST_PRIVATEUSE1_DEVICE_TYPE 常量
    TestCase,  # 从 torch.testing._internal.common_utils 导入 TestCase 类
    torch_to_numpy_dtype_dict,  # 从 torch.testing._internal.common_utils 导入 torch_to_numpy_dtype_dict
)


# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    # 根据不同的 shape 创建不同的张量
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # 对浮点数和复数类型的张量进行处理
            # 由于 torch.randn 对 bfloat16 类型未实现，需要进行特殊处理
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(
                    30, 100
                )
            # 随机将部分元素置零
            x[torch.randn(*shape) > 0.5] = 0
            # 如果需要，使用极端值
            if with_extremal and dtype.is_floating_point:
                # 使用浮点数的极端值
                x[torch.randn(*shape) > 0.5] = float("nan")
                x[torch.randn(*shape) > 0.5] = float("inf")
                x[torch.randn(*shape) > 0.5] = float("-inf")
            elif with_extremal and dtype.is_complex:
                # 使用复数的极端值
                x[torch.randn(*shape) > 0.5] = complex("nan")
                x[torch.randn(*shape) > 0.5] = complex("inf")
                x[torch.randn(*shape) > 0.5] = complex("-inf")
        elif dtype == torch.bool:
            # 对布尔类型的张量进行处理
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            # 对其它类型的张量进行处理
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x


class TestShapeOps(TestCase):
    # TODO: update to work on CUDA, too
    @onlyCPU  # 限定只在 CPU 上运行的测试方法
    def test_unbind(self, device):
        # 创建一个形状为 (2, 3, 4, 5) 的张量 x
        x = torch.rand(2, 3, 4, 5)
        # 遍历张量 x 的每一个维度
        for dim in range(4):
            # 使用 dim 维度对张量 x 进行解绑操作
            res = torch.unbind(x, dim)
            res2 = x.unbind(dim)
            # 断言解绑后的列表长度与该维度的长度相等
            self.assertEqual(x.size(dim), len(res))
            self.assertEqual(x.size(dim), len(res2))
            # 对解绑后的张量列表的每一个张量进行断言，验证其正确性
            for i in range(dim):
                self.assertEqual(x.select(dim, i), res[i])
                self.assertEqual(x.select(dim, i), res2[i])

    # TODO: update to work on CUDA, too?
    @skipIfTorchDynamo("TorchDynamo fails with an unknown error")  # 跳过 TorchDynamo 测试
    @onlyCPU  # 限定只在 CPU 上运行的测试方法
    # 定义一个测试方法，用于测试 torch.tensor 的 tolist 方法
    def test_tolist(self, device):
        # 创建一个空的 Python 列表 list0D
        list0D = []
        # 使用 torch.tensor 将空列表 list0D 转换为一个 0 维的张量 tensor0D
        tensor0D = torch.tensor(list0D)
        # 断言 tensor0D 转换为列表后与原始列表 list0D 相等
        self.assertEqual(tensor0D.tolist(), list0D)

        # 创建一个包含浮点数的 1 维列表 table1D
        table1D = [1.0, 2.0, 3.0]
        # 使用 torch.tensor 将 table1D 转换为一个 1 维的张量 tensor1D
        tensor1D = torch.tensor(table1D)
        # 使用 torch.Storage 创建与 table1D 相同内容的 storage 对象
        storage = torch.Storage(table1D)
        # 断言 tensor1D 和 storage 转换为列表后与原始列表 table1D 相等
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)

        # 创建一个包含整数的 2 维列表 table2D
        table2D = [[1, 2], [3, 4]]
        # 使用 torch.tensor 将 table2D 转换为一个 2 维的张量 tensor2D
        tensor2D = torch.tensor(table2D)
        # 断言 tensor2D 转换为列表后与原始列表 table2D 相等
        self.assertEqual(tensor2D.tolist(), table2D)

        # 创建一个包含 3 维数组的张量 tensor3D
        tensor3D = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        # 使用 select 方法从 tensor3D 中选择第一个维度为 1 的张量 tensorNonContig
        tensorNonContig = tensor3D.select(1, 1)
        # 断言 tensorNonContig 是否是连续的张量（返回 False）
        self.assertFalse(tensorNonContig.is_contiguous())
        # 断言 tensorNonContig 转换为列表后与预期列表 [[3, 4], [7, 8]] 相等
        self.assertEqual(tensorNonContig.tolist(), [[3, 4], [7, 8]])

    # 使用 @dtypes 装饰器声明多种数据类型的测试
    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_movedim_invalid(self, device, dtype):
        # 随机生成一个形状为 shape 的张量 x
        shape = self._rand_shape(4, min_size=5, max_size=10)
        x = _generate_input(shape, dtype, device, False)

        # 遍历 movedim 和 moveaxis 方法的测试用例
        for fn in [torch.movedim, torch.moveaxis]:
            # 测试当源维度或目标维度超出范围时是否抛出 IndexError 异常
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

            # 测试当源维度和目标维度大小不匹配时是否抛出 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, "movedim: Invalid source or destination dims:"
            ):
                fn(x, (1, 0), (0,))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `source`"
            ):
                fn(x, (0, 0), (0, 1))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `source`"
            ):
                fn(x, (0, 1, 0), (0, 1, 2))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `destination`"
            ):
                fn(x, (0, 1), (1, 1))

            with self.assertRaisesRegex(
                RuntimeError, "movedim: repeated dim in `destination`"
            ):
                fn(x, (0, 1, 2), (1, 0, 1))

    # 使用 @dtypes 装饰器声明多种数据类型的测试
    @dtypes(torch.int64, torch.float, torch.complex128)
    @dtypes(torch.float, torch.bool)
    def test_diag(self, device, dtype):
        # 根据不同的数据类型生成不同的随机张量 x
        if dtype is torch.bool:
            x = torch.rand(100, 100, device=device) >= 0.5
        else:
            x = torch.rand(100, 100, dtype=dtype, device=device)

        # 使用 torch.diag 方法获取 x 的对角线元素 res1
        res1 = torch.diag(x)
        # 创建一个空张量 res2，用于存储 torch.diag 的结果
        res2 = torch.tensor((), dtype=dtype, device=device)
        # 使用 torch.diag 将 x 的对角线元素填充到 res2 中
        torch.diag(x, out=res2)
        # 断言 res1 和 res2 相等
        self.assertEqual(res1, res2)
    # 定义一个测试函数，用于测试 torch.diagonal 函数在给定设备上的行为
    def test_diagonal(self, device):
        # 创建一个形状为 (100, 100) 的随机张量 x，位于指定设备上
        x = torch.randn((100, 100), device=device)
        # 使用 torch.diagonal 获取张量 x 的主对角线元素
        result = torch.diagonal(x)
        # 使用 torch.diag 获取张量 x 的主对角线元素（预期结果）
        expected = torch.diag(x)
        # 断言两个结果张量相等
        self.assertEqual(result, expected)

        # 再次创建一个形状为 (100, 100) 的随机张量 x，位于指定设备上
        x = torch.randn((100, 100), device=device)
        # 使用 torch.diagonal 获取张量 x 位于第 17 条对角线上的元素
        result = torch.diagonal(x, 17)
        # 使用 torch.diag 获取张量 x 位于第 17 条对角线上的元素（预期结果）
        expected = torch.diag(x, 17)
        # 断言两个结果张量相等
        self.assertEqual(result, expected)

    # 标记为仅在 CPU 上运行的测试，并指定 torch.float 作为数据类型
    @onlyCPU
    @dtypes(torch.float)
    def test_diagonal_multidim(self, device, dtype):
        # 创建一个具有多维度的随机张量 x，指定设备和数据类型
        x = torch.randn(10, 11, 12, 13, dtype=dtype, device=device)
        # 获得 x 的 NumPy 表示
        xn = x.numpy()
        # 遍历参数列表 [(2, 2, 3), (2,), (-2, 1, 2), (0, -2, -1)]
        for args in [(2, 2, 3), (2,), (-2, 1, 2), (0, -2, -1)]:
            # 使用 torch.diagonal 获取张量 x 的对角线元素，根据 args 参数指定
            result = torch.diagonal(x, *args)
            # 使用 NumPy 获取张量 x 的对角线元素，根据 args 参数指定（预期结果）
            expected = xn.diagonal(*args)
            # 断言预期结果的形状与实际结果的形状相等
            self.assertEqual(expected.shape, result.shape)
            # 断言预期结果与实际结果相等
            self.assertEqual(expected, result)
        
        # 对非连续的张量进行测试
        # 将张量 x 按指定顺序重新排列
        xp = x.permute(1, 2, 3, 0)
        # 使用 torch.diagonal 获取重新排列后的张量 xp 的对角线元素
        result = torch.diagonal(xp, 0, -2, -1)
        # 使用 NumPy 获取重新排列后的张量 xp 的对角线元素（预期结果）
        expected = xp.numpy().diagonal(0, -2, -1)
        # 断言预期结果的形状与实际结果的形状相等
        self.assertEqual(expected.shape, result.shape)
        # 断言预期结果与实际结果相等
        self.assertEqual(expected, result)

    # 标记为仅在原生设备类型上运行的测试，并指定所有数据类型
    @onlyNativeDeviceTypes
    @dtypes(*all_types())
    @dtypesIfCUDA(*all_types_and(torch.half))
    def test_trace(self, device, dtype):
        # 定义内部函数用于测试给定形状的张量
        def test(shape):
            # 创建一个指定设备和数据类型的随机张量
            tensor = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            # 获取张量元素总和的数据类型，并映射为 NumPy 数据类型
            expected_dtype = tensor.sum().dtype
            expected_dtype = torch_to_numpy_dtype_dict[expected_dtype]

            # 使用 NumPy 计算张量的迹，得到期望结果
            result = np.trace(tensor.cpu().numpy(), dtype=expected_dtype)
            # 创建一个张量表示期望结果，并指定设备
            expected = torch.tensor(result, device=device)
            # 断言张量的迹与期望结果相等
            self.assertEqual(tensor.trace(), expected)

        # 测试的张量形状列表
        shapes = (
            [10, 1],
            [1, 10],
            [100, 100],
            [20, 100],
            [100, 20],
        )
        # 对每个形状进行测试
        for shape in shapes:
            test(shape)

    # 生成一个基准测试，用于创建指定设备和数据类型的随机张量，并计算预期的 clamped 值
    def generate_clamp_baseline(self, device, dtype, *, min_vals, max_vals, with_nans):
        """
        Creates a random tensor for a given device and dtype, and computes the expected clamped
        values given the min_vals and/or max_vals.
        If with_nans is provided, then some values are randomly set to nan.
        """
        # 创建一个在 [0, 1) 范围内的随机张量 X，乘以 50 并减去 25，均匀分布在 [-25, 25] 区间
        X = torch.rand(100, device=device).mul(50).add(-25)
        # 将 X 转换为指定的数据类型 dtype
        X = X.to(dtype)
        # 如果 with_nans 为 True，则随机设置一些值为 NaN
        if with_nans:
            # 创建一个与 X 相同形状的随机布尔掩码
            mask = torch.randint(0, 2, X.shape, dtype=torch.bool, device=device)
            # 将掩码位置上的元素设置为 NaN
            X[mask] = nan

        # 如果 min_vals 是 torch.Tensor 类型，则转换为 NumPy 数组
        if isinstance(min_vals, torch.Tensor):
            min_vals = min_vals.cpu().numpy()

        # 如果 max_vals 是 torch.Tensor 类型，则转换为 NumPy 数组
        if isinstance(max_vals, torch.Tensor):
            max_vals = max_vals.cpu().numpy()

        # 使用 NumPy 的 clip 函数，将张量 X 的值限制在 [min_vals, max_vals] 区间内，并指定设备
        X_clamped = torch.tensor(
            np.clip(X.cpu().numpy(), a_min=min_vals, a_max=max_vals), device=device
        )
        return X, X_clamped

    # 测试 clamp 函数及其别名 clip
    @dtypes(torch.int64, torch.float32)
    # 定义一个测试函数，用于测试 clamp 相关的函数
    def test_clamp(self, device, dtype):
        # 定义操作列表，包括 torch.clamp 等函数的多种版本
        op_list = (
            torch.clamp,
            torch.Tensor.clamp,
            torch.Tensor.clamp_,
            torch.clip,
            torch.Tensor.clip,
            torch.Tensor.clip_,
        )

        # 生成 min/max 参数的组合
        args = product((-10, None), (10, None))

        # 遍历每种操作
        for op in op_list:
            # 遍历参数组合
            for min_val, max_val in args:
                # 跳过 min_val 和 max_val 都为 None 的情况
                if min_val is None and max_val is None:
                    continue

                # 生成基准测试数据 X 和预期结果 Y_expected
                X, Y_expected = self.generate_clamp_baseline(
                    device, dtype, min_vals=min_val, max_vals=max_val, with_nans=False
                )

                # 复制 X 以便于测试中的原地操作不改变 X 的内容
                X1 = X.clone()
                # 执行操作 op
                Y_actual = op(X1, min_val, max_val)
                # 断言操作后的实际结果 Y_actual 与预期结果 Y_expected 相等
                self.assertEqual(Y_expected, Y_actual)

                # 测试 op-out 行为（方法版本中不存在 out 参数）
                if op in (torch.clamp, torch.clip):
                    # 创建一个与 X 相同大小的空张量 Y_out
                    Y_out = torch.empty_like(X)
                    # 使用 out 参数执行操作 op
                    op(X, min=min_val, max=max_val, out=Y_out)
                    # 断言操作后的输出 Y_out 与预期结果 Y_expected 相等
                    self.assertEqual(Y_expected, Y_out)

    # 测试 clamp 函数在处理 NaN 值时的行为
    def test_clamp_propagates_nans(self, device):
        # 定义操作列表，包括 torch.clamp 等函数的多种版本
        op_list = (
            torch.clamp,
            torch.Tensor.clamp,
            torch.Tensor.clamp_,
            torch.clip,
            torch.Tensor.clip,
            torch.Tensor.clip_,
        )

        # 生成 min/max 参数的组合
        args = product((-10, None), (10, None))

        # 遍历每种操作
        for op in op_list:
            # 遍历参数组合
            for min_val, max_val in args:
                # 跳过 min_val 和 max_val 都为 None 的情况
                if min_val is None and max_val is None:
                    continue

                # 生成包含 NaN 的测试数据 X 和预期结果 Y_expected
                X, Y_expected = self.generate_clamp_baseline(
                    device,
                    torch.float,
                    min_vals=min_val,
                    max_vals=max_val,
                    with_nans=True,
                )
                # 将预期结果 Y_expected 转换为表示 NaN 的布尔张量
                Y_expected = torch.isnan(Y_expected)

                # 复制 X 以便于测试中的原地操作不改变 X 的内容
                X1 = X.clone()
                # 执行操作 op
                Y_actual = op(X1, min_val, max_val)
                # 断言操作后的实际结果 Y_actual 包含预期的 NaN 结果
                self.assertEqual(Y_expected, torch.isnan(Y_actual))

                # 测试 op-out 行为（方法版本中不存在 out 参数）
                if op in (torch.clamp, torch.clip):
                    # 创建一个与 X 相同大小的空张量 Y_out
                    Y_out = torch.empty_like(X)
                    # 使用 out 参数执行操作 op
                    op(X, min_val, max_val, out=Y_out)
                    # 断言操作后的输出 Y_out 包含预期的 NaN 结果
                    self.assertEqual(Y_expected, torch.isnan(Y_out))

    # 测试 clamp 函数在缺少 min 或 max 参数时是否会引发错误
    def test_clamp_raises_arg_errors(self, device):
        # 生成随机数据 X
        X = torch.randn(100, dtype=torch.float, device=device)
        # 定义错误消息
        error_msg = "At least one of 'min' or 'max' must not be None"
        
        # 断言调用 X.clamp() 会引发 RuntimeError 并包含指定错误消息
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp()
        # 断言调用 X.clamp_() 会引发 RuntimeError 并包含指定错误消息
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp_()
        # 断言调用 torch.clamp(X) 会引发 RuntimeError 并包含指定错误消息
        with self.assertRaisesRegex(RuntimeError, error_msg):
            torch.clamp(X)
    # 装饰器，指定多种数据类型和复杂类型，包括 torch.half, torch.bool, torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_flip_errors(self, device, dtype):
        # 创建部分函数，生成指定设备和数据类型的张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)
        # 生成指定形状的张量数据
        data = make_arg((2, 2, 2))

        # 测试：不允许在同一维度上多次翻转
        self.assertRaises(RuntimeError, lambda: data.flip(0, 1, 1))
        # 测试：不允许空列表作为输入
        self.assertRaises(TypeError, lambda: data.flip())

        # 测试：不允许维度数大于最大维度数
        self.assertRaises(IndexError, lambda: data.flip(0, 1, 2, 3))
        self.assertRaises(IndexError, lambda: data.flip(3))

    # 生成随机形状的张量
    def _rand_shape(self, dim, min_size, max_size):
        return tuple(torch.randint(min_size, max_size + 1, (dim,)))

    # 装饰器，指定多种数据类型和复杂类型，包括 torch.half, torch.bool, torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_flip_numpy(self, device, dtype):
        # 创建部分函数，生成指定设备和数据类型的张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        # 遍历不同维度的形状
        for ndim in [3, 4]:
            shape = self._rand_shape(ndim, 5, 10)
            # 生成指定形状的张量数据
            data = make_arg(shape)

            # Axis to sample for given shape.
            # 针对给定形状的轴进行抽样
            for i in range(1, ndim + 1):
                # Check all combinations of `i` axis.
                # 检查所有 `i` 轴的组合
                for flip_dim in combinations(range(ndim), i):
                    # 创建 torch.flip 和 np.flip 的部分函数
                    torch_fn = partial(torch.flip, dims=flip_dim)
                    np_fn = partial(np.flip, axis=flip_dim)
                    # 将 torch 函数和 numpy 函数进行比较
                    self.compare_with_numpy(torch_fn, np_fn, data)

    # 装饰器，限定只在 CUDA 环境下运行（CPU 太慢）
    @onlyCUDA
    # 装饰器，标记测试数据为大型张量，约 17GB
    @largeTensorTest("17GB")
    # 装饰器，标记测试数据为大型张量，约 81GB，在 CPU 环境下运行
    @largeTensorTest("81GB", "cpu")
    # 装饰器，如果运行在 Jetson 硬件上则跳过测试（对于 Jetson 太大）
    @unittest.skipIf(IS_JETSON, "Too large for Jetson")
    def test_flip_large_tensor(self, device):
        # 生成指定大小的随机张量数据
        t_in = torch.empty(2**32 + 1, dtype=torch.uint8).random_()
        # 创建部分函数，使用 torch.flip 对张量进行翻转
        torch_fn = partial(torch.flip, dims=(0,))
        # 创建部分函数，使用 np.flip 对数组进行翻转
        np_fn = partial(np.flip, axis=0)
        # 将 torch 函数和 numpy 函数进行比较
        self.compare_with_numpy(torch_fn, np_fn, t_in)
        # 删除张量数据，释放内存
        del t_in

    # 测试函数，比较 torch.fliplr 和 np.fliplr 的功能
    def _test_fliplr_flipud(self, torch_fn, np_fn, min_dim, max_dim, device, dtype):
        # 遍历指定维度范围内的形状
        for dim in range(min_dim, max_dim + 1):
            shape = self._rand_shape(dim, 5, 10)
            # 如果数据类型是浮点型或复数型，使用随机数生成数据；否则使用随机整数
            if dtype.is_floating_point or dtype.is_complex:
                data = torch.randn(*shape, device=device, dtype=dtype)
            else:
                data = torch.randint(0, 10, shape, device=device, dtype=dtype)
            # 将 torch 函数和 numpy 函数进行比较
            self.compare_with_numpy(torch_fn, np_fn, data)

    # 装饰器，指定数据类型为 torch.int64, torch.double, torch.cdouble
    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_fliplr(self, device, dtype):
        # 测试 fliplr_flipud 函数，指定维度范围为 2 到 4
        self._test_fliplr_flipud(torch.fliplr, np.fliplr, 2, 4, device, dtype)

    # 装饰器，指定数据类型为 torch.int64, torch.double, torch.cdouble
    @dtypes(torch.int64, torch.double, torch.cdouble)
    # 定义测试函数，用于验证 torch.fliplr 的异常情况
    def test_fliplr_invalid(self, device, dtype):
        # 生成一个大小为 42 的随机张量，并将其转换为指定数据类型
        x = torch.randn(42).to(dtype)
        # 预期捕获 RuntimeError 异常，其中包含 "Input must be >= 2-d." 字符串
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            torch.fliplr(x)
        # 预期捕获 RuntimeError 异常，其中包含 "Input must be >= 2-d." 字符串
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            torch.fliplr(torch.tensor(42, device=device, dtype=dtype))

    # 装饰器指定的数据类型测试函数，验证 torch.flipud 的正常情况
    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_flipud(self, device, dtype):
        # 调用 _test_fliplr_flipud 函数测试 torch.flipud
        self._test_fliplr_flipud(torch.flipud, np.flipud, 1, 4, device, dtype)

    # 装饰器指定的数据类型测试函数，验证 torch.flipud 的异常情况
    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_flipud_invalid(self, device, dtype):
        # 预期捕获 RuntimeError 异常，其中包含 "Input must be >= 1-d." 字符串
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 1-d."):
            torch.flipud(torch.tensor(42, device=device, dtype=dtype))

    # 测试函数，验证 torch.rot90 的不同旋转角度情况
    def test_rot90(self, device):
        # 创建一个大小为 2x2 的张量
        data = torch.arange(1, 5, device=device).view(2, 2)
        # 测试旋转 0 度的情况，期望与原始数据相同
        self.assertEqual(torch.tensor([1, 2, 3, 4]).view(2, 2), data.rot90(0, [0, 1]))
        # 测试逆时针旋转 90 度的情况
        self.assertEqual(torch.tensor([2, 4, 1, 3]).view(2, 2), data.rot90(1, [0, 1]))
        # 测试逆时针旋转 180 度的情况
        self.assertEqual(torch.tensor([4, 3, 2, 1]).view(2, 2), data.rot90(2, [0, 1]))
        # 测试逆时针旋转 270 度的情况
        self.assertEqual(torch.tensor([3, 1, 4, 2]).view(2, 2), data.rot90(3, [0, 1]))

        # 测试默认参数下的旋转，即 k=1, dims=[0, 1]
        self.assertEqual(data.rot90(), data.rot90(1, [0, 1]))

        # 测试维度顺序颠倒的情况
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(1, [1, 0]))

        # 测试 k 取模的情况
        self.assertEqual(data.rot90(5, [0, 1]), data.rot90(1, [0, 1]))
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(-1, [0, 1]))
        self.assertEqual(data.rot90(-5, [0, 1]), data.rot90(-1, [0, 1]))

        # 测试超出范围的维度错误
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, -3]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 2]))

        # 测试大于二维张量的情况
        data = torch.arange(1, 9, device=device).view(2, 2, 2)
        self.assertEqual(
            torch.tensor([2, 4, 1, 3, 6, 8, 5, 7]).view(2, 2, 2), data.rot90(1, [1, 2])
        )
        self.assertEqual(data.rot90(1, [1, -1]), data.rot90(1, [1, 2]))

        # 测试错误情况
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 3]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [1, 1]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 1, 2]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0]))

    # 装饰器用于跳过特定情况下的测试
    @skipIfTorchDynamo("TorchDynamo fails with an unknown error")
    # 装饰器指定的数据类型测试函数，验证 torch.rot90 在复数数据类型下的情况
    @dtypes(torch.cfloat, torch.cdouble)
    def test_complex_rot90(self, device, dtype):
        # 随机生成张量的形状，维度数在 2 到 4 之间，每维的大小在 5 到 10 之间
        shape = self._rand_shape(random.randint(2, 4), 5, 10)
        # 对每个旋转次数进行循环测试
        for rot_times in range(4):
            # 在指定设备和数据类型下生成随机张量
            data = torch.randn(*shape, device=device, dtype=dtype)
            # 创建一个部分函数，用于调用 torch.rot90 进行测试，指定 k 和 dims
            torch_fn = partial(torch.rot90, k=rot_times, dims=[0, 1])
            # 创建一个部分函数，用于调用 np.rot90 进行测试，指定 k 和 axes
            np_fn = partial(np.rot90, k=rot_times, axes=[0, 1])
            # 调用 compare_with_numpy 函数，将 torch_fn 和 np_fn 作为参数进行比较
            self.compare_with_numpy(torch_fn, np_fn, data)
    # TODO: update once warning flag is available to always trigger ONCE warnings
    # 一旦警告标志可用，确保非零值不会触发警告，即使未提供 as_tuple 参数。
    # 该方法用于测试在给定设备上的非零值操作是否会触发警告。
    def test_nonzero_no_warning(self, device):
        # 创建一个随机张量 t，形状为 (2, 2)，并指定设备。
        t = torch.randn((2, 2), device=device)
        
        # 使用 warnings 模块捕获警告，记录到列表 w 中。
        with warnings.catch_warnings(record=True) as w:
            # 设置警告过滤器，始终触发警告。
            warnings.simplefilter("always")
            
            # 执行 torch.nonzero(t) 操作，检查是否会产生警告。
            torch.nonzero(t)
            
            # 执行 t.nonzero() 操作，检查是否会产生警告。
            t.nonzero()
            
            # 断言捕获的警告列表 w 的长度为 0，即没有警告被触发。
            self.assertEqual(len(w), 0)

    # 在下面的测试装饰器中，列出了所有数据类型及其组合，包括 torch.half, torch.bool, torch.bfloat16。
    # 定义测试函数 test_nonzero，接受设备和数据类型作为参数
    def test_nonzero(self, device, dtype):
        # 定义不同形状的张量列表
        shapes = [
            torch.Size((12,)),
            torch.Size((12, 1)),
            torch.Size((1, 12)),
            torch.Size((6, 2)),
            torch.Size((3, 2, 2)),
            torch.Size((5, 5, 5)),
        ]

        # 定义生成非平凡输入的函数
        def gen_nontrivial_input(shape, dtype, device):
            # 如果数据类型不是 torch.bfloat16，则生成整数随机张量
            if dtype != torch.bfloat16:
                return torch.randint(2, shape, device=device, dtype=dtype)
            else:
                # 对于 torch.bfloat16，生成 float 类型的随机张量，并转换为 bfloat16 类型
                return torch.randint(2, shape, device=device, dtype=torch.float).to(
                    dtype
                )

        # 遍历各种形状的张量
        for shape in shapes:
            # 生成指定形状的张量
            tensor = gen_nontrivial_input(shape, dtype, device)
            # 使用 torch.nonzero 获取非零元素的索引，结果作为张量返回
            dst1 = torch.nonzero(tensor, as_tuple=False)
            # 使用 tensor.nonzero 获取非零元素的索引，结果作为张量返回
            dst2 = tensor.nonzero(as_tuple=False)
            # 创建一个空张量 dst3，数据类型为 torch.long，存储在指定设备上
            dst3 = torch.empty([], dtype=torch.long, device=device)
            # 使用 torch.nonzero 将非零元素的索引存储到 dst3 中
            torch.nonzero(tensor, out=dst3)
            # 如果设备类型不是 "xla"，验证是否会抛出运行时错误
            if self.device_type != "xla":
                self.assertRaisesRegex(
                    RuntimeError,
                    "scalar type Long",
                    lambda: torch.nonzero(
                        tensor, out=torch.empty([], dtype=torch.float, device=device)
                    ),
                )
            # 如果设备类型是 "cuda" 或 TEST_PRIVATEUSE1_DEVICE_TYPE，验证是否会抛出运行时错误
            if (
                self.device_type == "cuda"
                or self.device_type == TEST_PRIVATEUSE1_DEVICE_TYPE
            ):
                self.assertRaisesRegex(
                    RuntimeError,
                    "on the same device",
                    lambda: torch.nonzero(
                        tensor, out=torch.empty([], dtype=torch.long)
                    ),
                )
            # 将张量转移到 CPU，并转换为 numpy 数组
            np_array = (
                tensor.cpu().numpy()
                if dtype != torch.bfloat16
                else tensor.float().cpu().numpy()
            )
            # 使用 numpy 的 nonzero 函数得到非零元素的索引，并转换为 PyTorch 张量，转置处理
            np_result = torch.from_numpy(np.stack(np_array.nonzero())).t()
            # 断言 torch.nonzero 的结果与 numpy 结果相等，指定容差为 0
            self.assertEqual(dst1.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst2.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst3.cpu(), np_result, atol=0, rtol=0)
            # 使用 torch.nonzero(as_tuple=True) 获取非零元素的索引，并转换为元组形式
            tup1 = torch.nonzero(tensor, as_tuple=True)
            tup2 = tensor.nonzero(as_tuple=True)
            # 将元组形式的索引转换为 PyTorch 张量，并进行转置，转移到 CPU
            tup1 = torch.stack(tup1).t().cpu()
            tup2 = torch.stack(tup2).t().cpu()
            # 断言 torch.nonzero(as_tuple=True) 的结果与 numpy 结果相等，指定容差为 0
            self.assertEqual(tup1, np_result, atol=0, rtol=0)
            self.assertEqual(tup2, np_result, atol=0, rtol=0)
    # 测试 torch.nonzero 函数的 as_tuple 参数设置输出的正确性
    def test_nonzero_astuple_out(self, device):
        # 创建一个形状为 (3, 3, 3) 的随机张量 t，并移动到指定设备上
        t = torch.randn((3, 3, 3), device=device)
        # 创建一个与 t 具有相同形状的空张量 out，数据类型为 long
        out = torch.empty_like(t, dtype=torch.long)

        # 使用 assertRaises 检查运行时错误是否会被抛出
        with self.assertRaises(RuntimeError):
            # 调用 torch.nonzero 函数，期望抛出 RuntimeError，同时指定 as_tuple=True 和 out=out
            torch.nonzero(t, as_tuple=True, out=out)

        # 使用 assertEqual 检查 torch.nonzero 的两种调用方式的结果是否相同
        self.assertEqual(
            torch.nonzero(t, as_tuple=False, out=out), torch.nonzero(t, out=out)
        )

        # 验证 JIT 脚本不能处理 as_tuple 关键字参数
        # 参考问题 https://github.com/pytorch/pytorch/issues/45499.
        def _foo(t):
            # 使用 as_tuple=True 调用 torch.nonzero 函数
            tuple_result = torch.nonzero(t, as_tuple=True)
            # 使用 as_tuple=False 调用 torch.nonzero 函数
            nontuple_result = torch.nonzero(t, as_tuple=False)
            # 创建一个与 nontuple_result 相同形状的空张量 out，并使用 as_tuple=False 调用 torch.nonzero 函数
            out = torch.empty_like(nontuple_result)
            torch.nonzero(t, as_tuple=False, out=out)
            return tuple_result, nontuple_result, out

        # 使用 assertRaises 检查是否抛出运行时错误
        with self.assertRaises(RuntimeError):
            # 对 _foo 函数进行 JIT 脚本化
            scripted_foo = torch.jit.script(_foo)

        # 验证 JIT 追踪功能能够正常工作
        traced_foo = torch.jit.trace(_foo, t)
        # 使用 traced_foo 调用 t，获取返回的 tuple_result、nontuple_result 和 out
        traced_tuple, traced_nontuple, traced_out = traced_foo(t)
        # 获取 torch.nonzero(as_tuple=True) 的期望输出
        expected_tuple = torch.nonzero(t, as_tuple=True)
        # 获取 torch.nonzero() 的期望输出
        expected_nontuple = torch.nonzero(t)

        # 使用 assertEqual 检查 JIT 追踪后的结果与预期结果是否相同
        self.assertEqual(traced_tuple, expected_tuple)
        self.assertEqual(traced_nontuple, expected_nontuple)
        self.assertEqual(traced_out, expected_nontuple)

    # 仅针对本地设备类型的测试
    @onlyNativeDeviceTypes
    def test_nonzero_discontiguous(self, device):
        # 创建一个形状为 (4, 4) 的随机整数张量 tensor，移动到指定设备上
        shape = (4, 4)
        tensor = torch.randint(2, shape, device=device)
        # 创建一个与 tensor 形状相同，但不连续的张量 tensor_nc
        tensor_nc = torch.empty(shape[0], shape[1] * 2, device=device)[:, ::2].copy_(
            tensor
        )
        # 调用 tensor 和 tensor_nc 的 torch.nonzero(as_tuple=False)，并比较它们的结果是否相等
        dst1 = tensor.nonzero(as_tuple=False)
        dst2 = tensor_nc.nonzero(as_tuple=False)
        self.assertEqual(dst1, dst2, atol=0, rtol=0)
        # 创建一个与 dst1 相同形状的空张量 dst3
        dst3 = torch.empty_like(dst1)
        # 获取 dst3 的数据指针
        data_ptr = dst3.data_ptr()
        # 使用 torch.nonzero(tensor, out=dst3) 将 tensor 的非零元素索引存储到 dst3
        torch.nonzero(tensor, out=dst3)
        # 使用 assertEqual 检查 dst3 的数据指针与之前的 dst3 是否相同，以及 dst1 和 dst3 是否相等
        self.assertEqual(data_ptr, dst3.data_ptr())
        self.assertEqual(dst1, dst3, atol=0, rtol=0)
        # 创建一个形状为 (dst1.size(0), dst1.size(1) * 2) 的 long 类型空张量 dst4，以及其数据指针和步长
        dst4 = torch.empty(
            dst1.size(0), dst1.size(1) * 2, dtype=torch.long, device=device
        )[:, ::2]
        data_ptr = dst4.data_ptr()
        strides = dst4.stride()
        # 使用 torch.nonzero(tensor, out=dst4) 将 tensor 的非零元素索引存储到 dst4
        torch.nonzero(tensor, out=dst4)
        # 使用 assertEqual 检查 dst4 的数据指针与之前的 dst4 是否相同，以及 dst1 和 dst4 是否相等，以及它们的步长是否相同
        self.assertEqual(data_ptr, dst4.data_ptr())
        self.assertEqual(dst1, dst4, atol=0, rtol=0)
        self.assertEqual(strides, dst4.stride())

    # 测试 torch.nonzero 函数在 requires_grad=True 的张量上的行为
    def test_nonzero_non_diff(self, device):
        # 创建一个形状为 (10,) 的随机张量 x，并设置 requires_grad=True
        x = torch.randn(10, requires_grad=True)
        # 调用 x.nonzero()，并使用 assertFalse 检查其 requires_grad 属性是否为 False
        nz = x.nonzero()
        self.assertFalse(nz.requires_grad)

    # 针对指定数据类型的测试，包括 torch.int64、torch.float 和 torch.complex128
    @dtypes(torch.int64, torch.float, torch.complex128)
    # 定义一个测试方法，用于测试稀疏张量和密集张量的维度属性
    def test_sparse_dense_dim(self, device, dtype):
        # 对于给定的形状列表进行迭代测试
        for shape in [(), (2,), (2, 3)]:
            # 根据数据类型是否是复数或浮点型来生成张量 x
            if dtype.is_complex or dtype.is_floating_point:
                # 如果是复数或浮点型，使用 torch.rand 生成指定设备和数据类型的随机张量
                x = torch.rand(shape, device=device, dtype=dtype)
            else:
                # 如果不是复数或浮点型，使用 torch.randint 生成指定设备和数据类型的随机整数张量
                x = torch.randint(-9, 9, shape, device=device, dtype=dtype)
            # 断言稀疏维度为 0
            self.assertEqual(x.sparse_dim(), 0)
            # 断言密集维度为形状的长度
            self.assertEqual(x.dense_dim(), len(shape))
# 在全局范围内实例化设备类型测试，使用 TestShapeOps 作为测试类，globals() 函数返回全局符号表的字典
instantiate_device_type_tests(TestShapeOps, globals())

# 如果当前脚本作为主程序执行，调用 run_tests() 函数来运行测试
if __name__ == "__main__":
    run_tests()
```