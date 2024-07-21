# `.\pytorch\test\test_binary_ufuncs.py`

```
# Owner(s): ["module: tests"]

# 导入所需的模块和库
import itertools  # 提供用于迭代的工具
import math  # 提供数学函数
import operator  # 提供标准操作符的函数
import random  # 提供随机数生成器
import warnings  # 提供警告处理功能
from functools import partial  # 导入 partial 函数，用于创建偏函数
from itertools import chain, product  # 提供更多迭代工具
from numbers import Number  # 提供数字抽象基类

import numpy as np  # 导入 NumPy 库，用于数值计算

import torch  # 导入 PyTorch 库

import torch.autograd.forward_ad as fwAD  # 导入 PyTorch 的自动求导模块
from torch import inf, nan  # 导入 inf 和 nan 常量
from torch.testing import make_tensor  # 导入测试相关的函数和类
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,  # 检查设备数量是否至少为指定值的函数
    dtypes,  # 支持的数据类型列表
    dtypesIfCPU,  # 如果在 CPU 上支持的数据类型列表
    dtypesIfCUDA,  # 如果在 CUDA 上支持的数据类型列表
    expectedFailureMeta,  # 用于标记预期失败测试的元信息
    instantiate_device_type_tests,  # 实例化设备类型测试的函数
    onlyCPU,  # 仅在 CPU 上运行的装饰器
    onlyCUDA,  # 仅在 CUDA 上运行的装饰器
    onlyNativeDeviceTypes,  # 仅在本机设备类型上运行的装饰器
    OpDTypes,  # 操作支持的数据类型
    ops,  # 测试的操作列表
    precisionOverride,  # 用于精度覆盖的装饰器
    skipIf,  # 根据条件跳过测试的装饰器
    skipMeta,  # 跳过测试的元信息
)
from torch.testing._internal.common_dtype import (
    all_types_and,  # 所有支持数据类型的交集
    all_types_and_complex_and,  # 所有支持复数数据类型的交集
    complex_types,  # 所有复数数据类型
    floating_and_complex_types,  # 所有浮点数和复数数据类型的交集
    floating_types_and,  # 所有浮点数数据类型的交集
    get_all_int_dtypes,  # 获取所有整数数据类型的函数
    get_all_math_dtypes,  # 获取所有数学相关数据类型的函数
    integral_types,  # 所有整数数据类型
    integral_types_and,  # 所有整数数据类型的交集
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,  # 二元通用函数的测试集合
    binary_ufuncs_and_refs,  # 二元通用函数及其参考的测试集合
    generate_elementwise_binary_broadcasting_tensors,  # 生成用于广播的二元元素级张量的函数
    generate_elementwise_binary_extremal_value_tensors,  # 生成极端值的二元元素级张量的函数
    generate_elementwise_binary_large_value_tensors,  # 生成大值的二元元素级张量的函数
    generate_elementwise_binary_small_value_tensors,  # 生成小值的二元元素级张量的函数
    generate_elementwise_binary_tensors,  # 生成二元元素级张量的函数
    generate_elementwise_binary_with_scalar_and_type_promotion_samples,  # 生成包含标量和类型提升样本的二元元素级张量的函数
    generate_elementwise_binary_with_scalar_samples,  # 生成包含标量样本的二元元素级张量的函数
)
from torch.testing._internal.common_utils import (
    gradcheck,  # 梯度检查函数
    iter_indices,  # 迭代索引的函数
    numpy_to_torch_dtype_dict,  # NumPy 数据类型到 PyTorch 数据类型的映射字典
    run_tests,  # 运行测试的函数
    set_default_dtype,  # 设置默认数据类型的函数
    skipIfTorchDynamo,  # 如果在 Torch Dynamo 模式下跳过的装饰器
    slowTest,  # 标记为慢速测试的装饰器
    TEST_SCIPY,  # 是否导入了 SciPy
    TestCase,  # 测试用例基类
    torch_to_numpy_dtype_dict,  # PyTorch 数据类型到 NumPy 数据类型的映射字典
    xfailIfTorchDynamo,  # 如果在 Torch Dynamo 模式下预期失败的装饰器
)

if TEST_SCIPY:
    import scipy.integrate  # 导入 SciPy 的积分模块
    import scipy.special  # 导入 SciPy 的特殊函数模块


# TODO: update to use opinfos consistently
class TestBinaryUfuncs(TestCase):
    # Generic tests for elementwise binary (AKA binary universal (u) functions (funcs))
    # TODO: below contiguous tensor results are compared with a variety of noncontiguous results.
    #   It would be interesting to have the lhs and rhs have different discontiguities.

    # Helper for comparing torch tensors and NumPy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(
        self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs
    ):
        # 断言实际值是一个 torch.Tensor 对象
        assert isinstance(actual, torch.Tensor)

        # 一些 NumPy 函数返回标量而不是数组
        if isinstance(expected, Number):
            # 如果期望值是一个数字，直接比较实际值的 item() 方法与期望值
            self.assertEqual(actual.item(), expected, msg=msg, **kwargs)
        elif isinstance(expected, np.ndarray):
            # 处理数组与张量之间的精确 dtype 比较
            if exact_dtype:
                # 允许数组的 dtype 在与 bfloat16 张量比较时为 float32
                #   因为 NumPy 不支持 bfloat16 dtype
                # 同时像 scipy.special.erf, scipy.special.erfc 等操作会将 float16 提升为 float32
                if expected.dtype == np.float32:
                    assert actual.dtype in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                    )
                else:
                    # 断言期望的 dtype 与 actual 的 dtype 相匹配
                    assert expected.dtype == torch_to_numpy_dtype_dict[actual.dtype]

            # 使用 torch.from_numpy 将 NumPy 数组 expected 转换为与 actual 相同的 dtype，并比较
            self.assertEqual(
                actual,
                torch.from_numpy(expected).to(actual.dtype),
                msg,
                exact_device=False,
                **kwargs,
            )
        else:
            # 如果期望值既不是数字也不是数组，则直接比较实际值与期望值
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    # 测试函数及其（接受数组的）参考函数在给定张量上产生相同的值
    # 定义测试方法：测试给定数据类型、操作和生成器的参考数值运算
    def _test_reference_numerics(self, dtype, op, gen, equal_nan=True):
        # 定义辅助函数：比较预期值和实际值，并可选地检查精确数据类型和 NaN 相等性
        def _helper_reference_numerics(
            expected, actual, msg, exact_dtype, equal_nan=True
        ):
            # 检查是否可以转换预期值的数据类型为 Torch 数据类型
            if not torch.can_cast(
                numpy_to_torch_dtype_dict[expected.dtype.type], dtype
            ):
                exact_dtype = False

            # 特定情况下的数据类型比较和数值比较
            if dtype is torch.bfloat16 and expected.dtype == np.float32:
                # 使用自定义精度和误差边界比较
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    exact_dtype=exact_dtype,
                    rtol=16e-3,
                    atol=1e-5,
                )
            else:
                # 使用默认设置比较
                self.assertEqualHelper(
                    actual,
                    expected,
                    msg,
                    dtype=dtype,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                )

        # 遍历生成器中的每个样本数据
        for sample in gen:
            # 获取左侧和右侧的输入张量
            l = sample.input
            r = sample.args[0]

            # 获取样本数据的 NumPy 表示
            numpy_sample = sample.numpy()
            l_numpy = numpy_sample.input
            r_numpy = numpy_sample.args[0]

            # 进行操作并获取实际和预期结果
            actual = op(l, r)
            expected = op.ref(l_numpy, r_numpy)

            # 构造适用于小型可打印张量的自定义错误消息
            def _numel(x):
                if isinstance(x, torch.Tensor):
                    return x.numel()
                # 假定 x 是标量
                return 1

            # 如果左右输入张量小于等于 100，则包含详细的错误信息
            if _numel(l) <= 100 and _numel(r) <= 100:
                msg = (
                    "Failed to produce expected results! Input lhs tensor was"
                    f" {l}, rhs tensor was {r}, torch result is {actual}, and reference result is"
                    f" {expected}."
                )
            else:
                msg = None

            # 默认要求精确的数据类型匹配
            exact_dtype = True
            if isinstance(actual, torch.Tensor):
                # 调用辅助函数进行比较
                _helper_reference_numerics(
                    expected, actual, msg, exact_dtype, equal_nan
                )
            else:
                # 处理多个输出的结果
                for x, y in zip(expected, actual):
                    # 对多个输出结果进行比较
                    _helper_reference_numerics(x, y, msg, exact_dtype, equal_nan)

    # 下列测试仅适用于具有参考实现的逐元素二元运算符
    binary_ufuncs_with_references = list(
        filter(lambda op: op.ref is not None and op.ref is not None, binary_ufuncs)
    )

    # 使用 ops 装饰器注册测试方法：测试逐元素二元运算的参考数值计算
    @ops(binary_ufuncs_with_references)
    def test_reference_numerics(self, device, dtype, op):
        # 生成逐元素二元张量的生成器
        gen = generate_elementwise_binary_tensors(op, device=device, dtype=dtype)
        # 调用测试方法
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
    # 使用装饰器 ops(binary_ufuncs_with_references)，将该测试方法标记为使用二元函数操作和参考数值的测试
    def test_reference_numerics_small_values(self, device, dtype, op):
        # 如果数据类型是 torch.bool，则跳过该测试，因为不支持布尔类型
        if dtype is torch.bool:
            self.skipTest("Doesn't support bool!")
    
        # 生成小值张量作为测试数据，使用 op 运算符，在指定设备和数据类型上生成
        gen = generate_elementwise_binary_small_value_tensors(
            op, device=device, dtype=dtype
        )
        # 调用内部方法进行数值参考测试，确保数值正确性，包括处理 NaN 值
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
    
    # 使用装饰器 ops(...)，标记为使用二元函数操作和参考数值的测试，并指定允许的数据类型
    def test_reference_numerics_large_values(self, device, dtype, op):
        # 生成大值张量作为测试数据，使用 op 运算符，在指定设备和数据类型上生成
        gen = generate_elementwise_binary_large_value_tensors(
            op, device=device, dtype=dtype
        )
        # 调用内部方法进行数值参考测试，确保数值正确性，包括处理 NaN 值
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
    
    # 使用装饰器 ops(...)，标记为使用二元函数操作和参考数值的测试，并指定允许的数据类型
    def test_reference_numerics_extremal_values(self, device, dtype, op):
        # 生成极端值张量作为测试数据，使用 op 运算符，在指定设备和数据类型上生成
        gen = generate_elementwise_binary_extremal_value_tensors(
            op, device=device, dtype=dtype
        )
        # 调用内部方法进行数值参考测试，确保数值正确性，包括处理 NaN 值
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
    
    # 使用装饰器 ops(...)，标记为测试广播和非连续广播行为的方法
    def test_broadcasting(self, device, dtype, op):
        # 生成广播张量作为测试数据，使用 op 运算符，在指定设备和数据类型上生成
        gen = generate_elementwise_binary_broadcasting_tensors(
            op, device=device, dtype=dtype
        )
        # 调用内部方法进行数值参考测试，确保数值正确性，包括处理 NaN 值
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
    
    # 使用装饰器 ops(...)，标记为使用二元函数操作的测试方法
    def test_scalar_support(self, device, dtype, op):
        # 生成带有标量样本的张量作为测试数据，使用 op 运算符，在指定设备和数据类型上生成
        gen = generate_elementwise_binary_with_scalar_samples(
            op, device=device, dtype=dtype
        )
        # 调用内部方法进行数值参考测试，确保数值正确性，包括处理 NaN 值
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
        
        # 生成带有标量和类型提升样本的张量作为测试数据，使用 op 运算符，在指定设备和数据类型上生成
        gen = generate_elementwise_binary_with_scalar_and_type_promotion_samples(
            op, device=device, dtype=dtype
        )
        # 再次调用内部方法进行数值参考测试，确保数值正确性，包括处理 NaN 值
        self._test_reference_numerics(dtype, op, gen, equal_nan=True)
    
    # 使用装饰器 ops(binary_ufuncs)，将该测试方法标记为使用二元函数操作的测试
    # 定义一个测试方法，用于比较连续存储和非连续存储的张量操作
    def test_contig_vs_every_other(self, device, dtype, op):
        # 创建一个形状为 (1026,) 的张量 lhs，使用 op.lhs_make_tensor_kwargs 提供的参数
        lhs = make_tensor(
            (1026,), device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
        )
        # 创建一个形状为 (1026,) 的张量 rhs，使用 op.rhs_make_tensor_kwargs 提供的参数
        rhs = make_tensor(
            (1026,), device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
        )

        # 对 lhs 和 rhs 进行步长为2的切片，生成非连续存储的张量
        lhs_non_contig = lhs[::2]
        rhs_non_contig = rhs[::2]

        # 断言 lhs 和 rhs 是连续存储的张量
        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        # 断言 lhs_non_contig 和 rhs_non_contig 是非连续存储的张量
        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        # 计算期望结果，对 lhs 和 rhs 进行操作后再进行步长为2的切片
        expected = op(lhs, rhs)[::2]
        # 对非连续存储的张量进行相同操作，获取实际结果
        actual = op(lhs_non_contig, rhs_non_contig)
        # 断言期望结果与实际结果相等
        self.assertEqual(expected, actual)

    # 使用二元通用函数装饰器，定义测试方法，用于比较转置和连续存储的张量操作
    @ops(binary_ufuncs)
    def test_contig_vs_transposed(self, device, dtype, op):
        # 创建形状为 (789, 357) 的 lhs 和 rhs 张量，使用 op.lhs_make_tensor_kwargs 和 op.rhs_make_tensor_kwargs 提供的参数
        lhs = make_tensor(
            (789, 357), device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
        )
        rhs = make_tensor(
            (789, 357), device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
        )

        # 对 lhs 和 rhs 进行转置操作，生成非连续存储的张量
        lhs_non_contig = lhs.T
        rhs_non_contig = rhs.T

        # 断言 lhs 和 rhs 是连续存储的张量
        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        # 断言 lhs_non_contig 和 rhs_non_contig 是非连续存储的张量
        self.assertFalse(lhs_non_contig.is_contiguous())
        self.assertFalse(rhs_non_contig.is_contiguous())

        # 计算期望结果，对 lhs 和 rhs 进行操作后再转置
        expected = op(lhs, rhs).T
        # 对非连续存储的张量进行相同操作，获取实际结果
        actual = op(lhs_non_contig, rhs_non_contig)
        # 断言期望结果与实际结果相等
        self.assertEqual(expected, actual)

    # 使用二元通用函数装饰器，定义测试方法，用于比较非连续存储的张量操作
    @ops(binary_ufuncs)
    def test_non_contig(self, device, dtype, op):
        # 定义多组不同形状的张量形状
        shapes = ((5, 7), (1024,))
        for shape in shapes:
            # 创建形状为 shape 的 lhs 和 rhs 张量，使用 op.lhs_make_tensor_kwargs 和 op.rhs_make_tensor_kwargs 提供的参数
            lhs = make_tensor(
                shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
            )
            rhs = make_tensor(
                shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
            )

            # 创建形状为 shape + (2,) 的非连续存储的 lhs_non_contig 和 rhs_non_contig 张量
            lhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[
                ..., 0
            ]
            lhs_non_contig.copy_(lhs)

            rhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[
                ..., 0
            ]
            rhs_non_contig.copy_(rhs)

            # 断言 lhs 和 rhs 是连续存储的张量
            self.assertTrue(lhs.is_contiguous())
            self.assertTrue(rhs.is_contiguous())

            # 断言 lhs_non_contig 和 rhs_non_contig 是非连续存储的张量
            self.assertFalse(lhs_non_contig.is_contiguous())
            self.assertFalse(rhs_non_contig.is_contiguous())

            # 计算期望结果，对 lhs 和 rhs 进行操作
            expected = op(lhs, rhs)
            # 对非连续存储的张量进行相同操作，获取实际结果
            actual = op(lhs_non_contig, rhs_non_contig)
            # 断言期望结果与实际结果相等
            self.assertEqual(expected, actual)

    # 使用二元通用函数装饰器，定义测试方法
    @ops(binary_ufuncs)
    # 测试非连续索引的情况
    def test_non_contig_index(self, device, dtype, op):
        # 定义张量的形状
        shape = (2, 2, 1, 2)
        # 创建左操作数张量，使用 make_tensor 函数生成
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        # 创建右操作数张量，使用 make_tensor 函数生成
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        # 获取左操作数的非连续切片，省略第二维后面的所有维度
        lhs_non_contig = lhs[:, 1, ...]
        # 将非连续的左操作数切片转换为连续张量
        lhs = lhs_non_contig.contiguous()

        # 获取右操作数的非连续切片，省略第二维后面的所有维度
        rhs_non_contig = rhs[:, 1, ...]
        # 将非连续的右操作数切片转换为连续张量
        rhs = rhs_non_contig.contiguous()

        # 断言左操作数已经是连续的张量
        self.assertTrue(lhs.is_contiguous())
        # 断言右操作数已经是连续的张量
        self.assertTrue(rhs.is_contiguous())

        # 断言左操作数的非连续切片确实是非连续的张量
        self.assertFalse(lhs_non_contig.is_contiguous())
        # 断言右操作数的非连续切片确实是非连续的张量
        self.assertFalse(rhs_non_contig.is_contiguous())

        # 计算期望的运算结果，使用操作符 op 对连续张量进行运算
        expected = op(lhs, rhs)
        # 计算实际的运算结果，使用操作符 op 对非连续张量进行运算
        actual = op(lhs_non_contig, rhs_non_contig)
        # 断言期望的运算结果等于实际的运算结果
        self.assertEqual(expected, actual)

    # 使用 binary_ufuncs 中的操作符来测试非连续张量的扩展
    @ops(binary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        # 定义不同形状的张量
        shapes = [(1, 3), (1, 7), (5, 7)]
        # 遍历所有形状
        for shape in shapes:
            # 创建左操作数张量，使用 make_tensor 函数生成
            lhs = make_tensor(
                shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
            )
            # 创建右操作数张量，使用 make_tensor 函数生成
            rhs = make_tensor(
                shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
            )

            # 对左操作数进行克隆并扩展为三倍大小，保持非连续性
            lhs_non_contig = lhs.clone().expand(3, -1, -1)
            # 对右操作数进行克隆并扩展为三倍大小，保持非连续性
            rhs_non_contig = rhs.clone().expand(3, -1, -1)

            # 断言左操作数已经是连续的张量
            self.assertTrue(lhs.is_contiguous())
            # 断言右操作数已经是连续的张量
            self.assertTrue(rhs.is_contiguous())

            # 断言左操作数的非连续版本确实是非连续的张量
            self.assertFalse(lhs_non_contig.is_contiguous())
            # 断言右操作数的非连续版本确实是非连续的张量
            self.assertFalse(rhs_non_contig.is_contiguous())

            # 计算期望的运算结果，使用操作符 op 对连续张量进行运算
            expected = op(lhs, rhs)
            # 计算实际的运算结果，使用操作符 op 对非连续张量进行运算
            actual = op(lhs_non_contig, rhs_non_contig)
            # 遍历每个扩展后的结果并断言期望的运算结果等于实际的运算结果
            for i in range(3):
                self.assertEqual(expected, actual[i])

    # 使用 binary_ufuncs 中的操作符测试尺寸为1的连续张量
    @ops(binary_ufuncs)
    def test_contig_size1(self, device, dtype, op):
        # 定义形状为 (5, 100) 的张量
        shape = (5, 100)
        # 创建左操作数张量，使用 make_tensor 函数生成
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        # 创建右操作数张量，使用 make_tensor 函数生成
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        # 对左操作数进行切片，保留第一维和前50列，使其成为尺寸为1的连续张量
        lhs = lhs[:1, :50]
        # 创建左操作数的备份，使用 torch.empty 创建相同形状的张量
        lhs_alt = torch.empty(lhs.size(), device=device, dtype=dtype)
        # 将切片后的左操作数复制到备份中
        lhs_alt.copy_(lhs)

        # 对右操作数进行切片，保留第一维和前50列，使其成为尺寸为1的连续张量
        rhs = rhs[:1, :50]
        # 创建右操作数的备份，使用 torch.empty 创建相同形状的张量
        rhs_alt = torch.empty(rhs.size(), device=device, dtype=dtype)
        # 将切片后的右操作数复制到备份中
        rhs_alt.copy_(rhs)

        # 断言左操作数是连续的张量
        self.assertTrue(lhs.is_contiguous())
        # 断言右操作数是连续的张量
        self.assertTrue(rhs.is_contiguous())

        # 断言备份的左操作数也是连续的张量
        self.assertTrue(lhs_alt.is_contiguous())
        # 断言备份的右操作数也是连续的张量
        self.assertTrue(rhs_alt.is_contiguous())

        # 计算期望的运算结果，使用操作符 op 对连续张量进行运算
        expected = op(lhs, rhs)
        # 计算实际的运算结果，使用操作符 op 对备份的连续张量进行运算
        actual = op(lhs_alt, rhs_alt)
        # 断言期望的运算结果等于实际的运算结果
        self.assertEqual(expected, actual)
    # 定义一个测试方法，用于测试具有大尺寸维度的连续切片操作
    def test_contig_size1_large_dim(self, device, dtype, op):
        # 定义一个大尺寸的张量形状
        shape = (5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4)
        # 创建左操作数张量，使用给定的设备和数据类型，并根据操作数的参数生成
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        # 创建右操作数张量，使用给定的设备和数据类型，并根据操作数的参数生成
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        # 对左操作数进行切片操作，保留第一个元素，并创建副本张量
        lhs = lhs[:1, :, :, :, :, :, :, :, :, :, :, :]
        lhs_alt = torch.empty(lhs.size(), device=device, dtype=dtype)
        lhs_alt.copy_(lhs)

        # 对右操作数进行切片操作，保留第一个元素，并创建副本张量
        rhs = rhs[:1, :, :, :, :, :, :, :, :, :, :, :]
        rhs_alt = torch.empty(rhs.size(), device=device, dtype=dtype)
        rhs_alt.copy_(rhs)

        # 断言左操作数和右操作数是连续的张量
        self.assertTrue(lhs.is_contiguous())
        self.assertTrue(rhs.is_contiguous())

        # 断言副本张量 lhs_alt 和 rhs_alt 是连续的张量
        self.assertTrue(lhs_alt.is_contiguous())
        self.assertTrue(rhs_alt.is_contiguous())

        # 计算期望的结果，使用给定的二元操作符 op
        expected = op(lhs, rhs)
        actual = op(lhs_alt, rhs_alt)
        # 断言期望的结果与实际结果相等
        self.assertEqual(expected, actual)

    # 标记一组二元函数操作的装饰器，并定义一个测试方法来比较批处理和切片操作的结果
    @ops(binary_ufuncs)
    def test_batch_vs_slicing(self, device, dtype, op):
        # 定义一个形状为 (32, 512) 的张量
        shape = (32, 512)
        # 创建左操作数张量，使用给定的设备和数据类型，并根据操作数的参数生成
        lhs = make_tensor(
            shape, dtype=dtype, device=device, **op.lhs_make_tensor_kwargs
        )
        # 创建右操作数张量，使用给定的设备和数据类型，并根据操作数的参数生成
        rhs = make_tensor(
            shape, dtype=dtype, device=device, **op.rhs_make_tensor_kwargs
        )

        # 计算期望的结果，使用给定的二元操作符 op
        expected = op(lhs, rhs)

        # 初始化一个空列表，用于存储每个批次索引对应的操作结果
        actual = []
        for idx in range(32):
            actual.append(op(lhs[idx], rhs[idx]))  # 对每个批次索引应用二元操作符
        actual = torch.stack(actual)  # 将结果堆叠成张量

        # 断言期望的结果与实际结果相等
        self.assertEqual(expected, actual)

    # 测试元素级二元操作符在类型提升中的行为是否正确
    # 注意：由于所有可能的类型提升测试的组合非常庞大，此处仅手工选择了一些例子进行测试
    # 注意：可能可以重构此测试以简化逻辑
    @ops(binary_ufuncs_and_refs, dtypes=OpDTypes.none)
    # TODO: move to error input test
    @ops(binary_ufuncs, allowed_dtypes=(torch.float32,))
    def test_not_broadcastable(self, device, dtype, op):
        # 遍历一组不可广播的张量形状对
        for shape_lhs, shape_rhs in (
            ((2,), (3,)),
            ((3, 1), (2, 1)),
            ((1, 3, 2), (3,)),
            ((3, 1, 2), (2, 1, 2)),
        ):
            # 创建左操作数张量，使用给定的设备、数据类型和操作数的参数生成
            lhs = make_tensor(
                shape_lhs, device=device, dtype=dtype, **op.lhs_make_tensor_kwargs
            )
            # 创建右操作数张量，使用给定的设备、数据类型和操作数的参数生成
            rhs = make_tensor(
                shape_rhs, device=device, dtype=dtype, **op.rhs_make_tensor_kwargs
            )

            try:
                # 尝试应用二元操作符 op 到左右操作数，获取广播后的形状
                broadcasted_shape = op(lhs, rhs).shape
            except RuntimeError:
                continue

            # 构建断言失败时的错误消息
            msg = (
                f"On {device}, torch.{op.name} broadcasts inputs shapes {shape_lhs} and {shape_rhs} into "
                f"{broadcasted_shape}, although they are not broadcastable."
            )
            # 抛出断言错误，显示错误消息
            raise AssertionError(msg)
    # 测试当操作数为空时的广播行为

    def test_add_broadcast_empty(self, device):
        # 预期抛出 RuntimeError 异常，因为两个张量都是空的
        self.assertRaises(
            RuntimeError,
            lambda: torch.randn(5, 0, device=device) + torch.randn(0, 5, device=device),
        )

        # 验证相同维度但一个是空张量的情况下，结果是空张量
        self.assertEqual(
            torch.randn(5, 0, device=device),
            torch.randn(0, device=device) + torch.randn(5, 0, device=device),
        )

        # 验证一个操作数为三维空张量，另一个为非空张量时的广播行为
        self.assertEqual(
            torch.randn(5, 0, 0, device=device),
            torch.randn(0, device=device) + torch.randn(5, 0, 1, device=device),
        )

        # 验证标量与空张量相加的结果
        self.assertEqual(
            torch.randn(5, 0, 6, device=device),
            torch.randn((), device=device) + torch.randn(5, 0, 6, device=device),
        )

        # 验证一个非空张量加一个空张量的结果
        self.assertEqual(
            torch.randn(0, device=device),
            torch.randn(0, device=device) + torch.randn(1, device=device),
        )

        # 验证多维度非空与空张量的广播行为
        self.assertEqual(
            torch.randn(0, 7, 0, 6, 5, 0, 7, device=device),
            torch.randn(0, 7, 0, 6, 5, 0, 1, device=device)
            + torch.randn(1, 1, 5, 1, 7, device=device),
        )

        # 预期抛出 RuntimeError 异常，因为维度不匹配无法广播
        self.assertRaises(
            RuntimeError,
            lambda: torch.randn(7, 0, device=device) + torch.randn(2, 1, device=device),
        )
    # 测试位操作的功能，包括张量与张量、张量与标量的操作
    def test_bitwise_ops(self, device, dtype):
        # 定义位操作符和就地操作符
        ops = (
            operator.and_,      # 按位与
            operator.iand,      # 就地按位与
            operator.or_,       # 按位或
            operator.ior,       # 就地按位或
            operator.xor,       # 按位异或
            operator.ixor,      # 就地按位异或
        )
        inplace_ops = (operator.iand, operator.ior, operator.ixor)  # 只包含就地操作的操作符集合
        shapes = ((5,), (15, 15), (500, 500))  # 定义不同形状的张量作为测试用例

        # 遍历所有操作符和形状的组合
        for op, shape in itertools.product(ops, shapes):
            # 测试张量 x 张量的情况
            a = make_tensor(shape, device=device, dtype=dtype)
            b = make_tensor(shape, device=device, dtype=dtype)
            a_np = a.cpu().clone().numpy()
            b_np = b.cpu().clone().numpy()
            # 断言张量操作和对应的NumPy操作结果相同
            self.assertEqual(op(a, b), op(a_np, b_np))

            # 测试张量 x 标量的情况
            a = make_tensor(shape, device=device, dtype=dtype)
            b_scalar = make_tensor((), device="cpu", dtype=dtype).item()
            a_np = a.cpu().clone().numpy()
            # 断言张量操作和对应的NumPy操作结果相同
            self.assertEqual(op(a, b_scalar), op(a_np, b_scalar))

            # 测试标量 x 张量的情况
            a_scalar = make_tensor((), device="cpu", dtype=dtype).item()
            b = make_tensor(shape, device=device, dtype=dtype)
            b_np = b.cpu().clone().numpy()
            # 断言张量操作和对应的NumPy操作结果相同
            self.assertEqual(op(a_scalar, b), op(a_scalar, b_np))

            # 对于不是就地操作的操作符，再进行一轮测试
            if op in inplace_ops:
                # 再次测试张量 x 张量的情况
                a = make_tensor(shape, device=device, dtype=dtype)
                b = make_tensor(shape, device=device, dtype=dtype)
                a_np = a.cpu().clone().numpy()
                b_np = b.cpu().clone().numpy()
                # 执行就地操作
                op(a, b)
                op(a_np, b_np)
                # 断言就地操作后张量a与a_np相等
                self.assertEqual(a, a_np)

                # 再次测试张量 x 标量的情况
                a = make_tensor(shape, device=device, dtype=dtype)
                b_scalar = make_tensor((), device="cpu", dtype=dtype).item()
                a_np = a.cpu().clone().numpy()
                # 执行就地操作
                op(a, b_scalar)
                op(a_np, b_scalar)
                # 断言就地操作后张量a与a_np相等
                self.assertEqual(a, a_np)
    # 定义一个测试函数，用于测试除法的不同舍入模式
    def test_div_rounding_modes(self, device, dtype):
        # 如果数据类型是浮点数
        if dtype.is_floating_point:
            # 设置低和高的取值范围
            low, high = -10.0, 10.0
        else:
            # 获取数据类型的信息（整数类型），设置低和高的取值范围
            info = torch.iinfo(dtype)
            low, high = info.min, info.max

        # 创建两个张量 a 和 b，用于测试
        a = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)
        b = make_tensor((100,), dtype=dtype, device=device, low=low, high=high)

        # 避免除以零，以便测试 (a / b) * b == a
        if dtype.is_floating_point:
            # 设置一个很小的值 eps，避免 b 接近零
            eps = 0.1
            # 将 b 中绝对值小于 eps 的元素替换为 eps
            b[(-eps < b) & (b < eps)] = eps
        else:
            # 对于非浮点数类型，将 b 中值为零的元素替换为 1
            b[b == 0] = 1

        # 如果数据类型不是浮点数
        if not dtype.is_floating_point:
            # 如果 a 小于零，稍微修正 floor(a / b) * b 以避免下溢
            a = torch.where(a < 0, a + b, a)

        # 计算真实的除法结果 d_true，不进行舍入
        d_true = torch.divide(a, b, rounding_mode=None)
        # 断言 d_true 是浮点数类型
        self.assertTrue(d_true.is_floating_point())
        # 断言 d_true * b 等于 a，转换到 d_true 的数据类型
        self.assertEqual(d_true * b, a.to(d_true.dtype))

        # 计算 floor 舍入模式的除法结果 d_floor
        d_floor = torch.divide(a, b, rounding_mode="floor")
        # 如果数据类型不是 torch.bfloat16 或 torch.half
        if dtype not in (torch.bfloat16, torch.half):
            # 断言 d_floor * b 加上 a 除以 b 的余数，等于 a
            self.assertEqual(d_floor * b + torch.remainder(a, b), a)
        else:
            # 断言 d_floor * b 加上 a 转换为 float 后除以 b 转换为 float 的余数，等于 a
            self.assertEqual(
                d_floor * b + torch.remainder(a.float(), b.float()),
                a,
                exact_dtype=False,
            )

        # 计算 trunc 舍入模式的除法结果 d_trunc
        d_trunc = torch.divide(a, b, rounding_mode="trunc")
        # 检查是否存在不支持的舍入模式
        rounding_unsupported = (
            dtype == torch.half
            and device != "cuda"
            or dtype == torch.bfloat16
            and device != "cpu"
        )
        # 如果存在不支持的舍入模式，将 d_true 转换为 float
        d_ref = d_true.float() if rounding_unsupported else d_true
        # 断言 d_trunc 等于 d_ref 取整后转换为 dtype 类型
        self.assertEqual(d_trunc, d_ref.trunc().to(dtype))

    # 使用指定的浮点数类型和 torch.bfloat16, torch.float16 进行测试
    @dtypes(*floating_types_and(torch.bfloat16, torch.float16))
    def test_floor_div_extremal(self, device, dtype):
        # 对于给定的 num, denom 和 shape 组合，进行测试
        for num, denom, shape in itertools.product(
            [torch.finfo(dtype).max * 0.7],
            [0.5, -0.5, 0.0],
            [(), (32,)],  # 标量和矢量化
        ):
            # 创建 num 和 denom 的张量 a 和 b
            a = torch.full(shape, num, dtype=dtype, device=device)
            b = torch.full(shape, denom, dtype=dtype, device=device)

            # 计算参考值 ref，使用 np.floor_divide 计算并转换为标量
            ref = np.floor_divide(num, denom).item()
            # 如果 ref 大于 dtype 的最大值，将 ref 设置为正无穷
            if ref > torch.finfo(dtype).max:
                ref = np.inf
            # 如果 ref 小于 dtype 的最小值，将 ref 设置为负无穷
            elif ref < torch.finfo(dtype).min:
                ref = -np.inf
            # 创建期望的张量 expect，填充为 ref
            expect = torch.full(shape, ref, dtype=dtype, device=device)
            # 计算实际结果 actual，使用 floor 舍入模式进行除法计算
            actual = torch.div(a, b, rounding_mode="floor")
            # 断言期望值与实际值相等
            self.assertEqual(expect, actual)

    # 使用指定的数据类型进行测试：torch.bfloat16, torch.half, torch.float32, torch.float64
    # 测试特殊浮点数值的除法运算，与 NumPy 进行比较
    def test_div_rounding_nonfinite(self, device, dtype):
        # 创建包含特殊浮点数值的张量
        num = torch.tensor(
            [1.0, -1.0, 0, 0.1, -0.1, np.pi, -np.pi, np.inf, -np.inf, np.nan],
            dtype=dtype,
            device=device,
        )
        # 将除数设为非零的特殊浮点数
        denom = num[num != 0]

        # 将 num 扩展为二维张量 a，denom 扩展为二维张量 b
        a, b = num[None, :].clone(), denom[:, None].clone()

        # 将 a 和 b 转换为 NumPy 数组，根据需要选择精确的数据类型
        exact_dtype = dtype != torch.bfloat16
        if exact_dtype:
            an, bn = a.cpu().numpy(), b.cpu().numpy()
        else:
            an, bn = a.float().cpu().numpy(), b.float().cpu().numpy()

        # 遍历模式和相应的 NumPy 函数（true_divide 或 floor_divide）
        for mode, np_ref in ((None, np.true_divide), ("floor", np.floor_divide)):
            # 使用 NumPy 函数计算预期结果
            expect = np_ref(an, bn)
            # 根据模式选择相应的参数
            kwargs = dict(rounding_mode=mode) if mode is not None else {}
            # 使用双精度浮点数作为默认数据类型进行计算
            with set_default_dtype(torch.double):
                actual = torch.divide(a, b, **kwargs)
            # 断言计算结果与 NumPy 数组中的预期结果相等
            self.assertEqual(
                actual,
                torch.from_numpy(expect),
                exact_device=False,
                exact_dtype=exact_dtype,
            )

        # 比较连续的张量（可能是向量化的）与非连续的张量（不是向量化的）
        a_noncontig = torch.empty([2 * i for i in a.shape], dtype=dtype, device=device)[
            ::2, ::2
        ]
        a_noncontig[:] = a
        b_noncontig = torch.empty([2 * i for i in b.shape], dtype=dtype, device=device)[
            ::2, ::2
        ]
        b_noncontig[:] = b

        # 遍历舍入模式，计算预期结果并进行比较
        for rounding_mode in (None, "trunc", "floor"):
            expect = torch.divide(a_noncontig, b_noncontig, rounding_mode=rounding_mode)
            actual = torch.divide(a, b, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect)

    # 使用不同数据类型测试除零操作的舍入
    @dtypes(torch.bfloat16, torch.half, torch.float32, torch.float64)
    def test_divide_by_zero_rounding(self, device, dtype):
        # 创建包含特殊浮点数值的张量 a
        a = torch.tensor(
            [1.0, -1.0, 0, 0.1, -0.1, np.pi, -np.pi, np.inf, -np.inf, np.nan],
            dtype=dtype,
        )
        # 根据数据类型选择是否精确
        exact_dtype = dtype != torch.bfloat16
        if exact_dtype:
            an = a.cpu().numpy()
        else:
            an = a.float().cpu().numpy()

        # 创建与 a 形状相同的零张量 zero
        zero = torch.zeros_like(a)

        # 计算预期结果，使用 NumPy 进行除零运算
        expect = np.divide(an, 0)
        # 遍历舍入模式，进行计算并断言结果与预期相等
        for rounding_mode in (None, "floor"):
            # CPU 标量
            actual = torch.divide(a, 0, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect, exact_dtype=exact_dtype)
            # 设备张量
            actual = torch.divide(a, zero, rounding_mode=rounding_mode)
            self.assertEqual(actual, expect, exact_dtype=exact_dtype)

    # 使用所有类型（包括 torch.half）进行数据类型测试
    @dtypes(*all_types_and(torch.half))
    # 在测试中使用 NumPy 进行除法精度比较
    def test_div_rounding_numpy(self, device, dtype):
        # 根据数据类型是浮点数还是整数，获取相应的信息
        info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
        low, high = info.min, info.max

        # 创建指定设备和数据类型的随机张量 a 和 b
        a = make_tensor((4096,), dtype=dtype, device=device, low=low, high=high)
        b = make_tensor((4096,), dtype=dtype, device=device, low=low, high=high)

        # 避免整数除以零，对于浮点数，NumPy 1.20 之后将 floor_divide 改为按 IEEE 规则处理 inf/nan
        b[b == 0] = 1

        # 如果数据类型不是 bfloat16，则将 a 和 b 转换为 NumPy 数组
        exact_dtype = dtype != torch.bfloat16
        if exact_dtype:
            an, bn = a.cpu().numpy(), b.cpu().numpy()
        else:
            an, bn = a.float().cpu().numpy(), b.float().cpu().numpy()

        # 遍历不同的模式和对应的 NumPy 函数进行比较
        for mode, np_ref in (
            (None, np.true_divide),
            ("floor", np.floor_divide),
            ("trunc", lambda a, b: np.trunc(np.true_divide(a, b)).astype(a.dtype)),
        ):
            # 使用 NumPy 计算期望的除法结果
            expect = torch.from_numpy(np_ref(an, bn))

            kwargs = dict(rounding_mode=mode) if mode is not None else {}
            
            # 使用默认的 double 数据类型进行计算，确保结果一致性
            with set_default_dtype(torch.double):
                actual = torch.divide(a, b, **kwargs)
            
            # 断言实际计算结果与期望结果相等
            self.assertEqual(
                actual, expect, exact_device=False, exact_dtype=exact_dtype
            )

            # 对非连续张量进行相同的测试（不向量化）
            expect = expect[::2]
            with set_default_dtype(torch.double):
                actual = torch.divide(a[::2], b[::2], **kwargs)

            self.assertEqual(
                actual, expect, exact_device=False, exact_dtype=exact_dtype
            )

    @dtypes(*complex_types())
    # 测试复杂除法不会产生下溢或上溢的情况
    def test_complex_div_underflow_overflow(self, device, dtype):
        # 确保复杂除法在计算过程中不会产生下溢或上溢
        # 注意：如果计算结果超过了 finfo.max / 2，仍然会产生错误，
        #       但希望人们意识到这是一个危险的区域
        finfo = torch.finfo(dtype)
        # 分子列表
        nom_lst = [
            complex(finfo.min / 2, finfo.min / 2),
            complex(finfo.max / 2, finfo.max / 2),
            complex(finfo.tiny, finfo.tiny),
            complex(finfo.tiny, 0.0),
            complex(0.0, 0.0),
        ]
        # 分母列表
        denom_lst = [
            complex(finfo.min / 2, finfo.min / 2),
            complex(finfo.max / 2, finfo.max / 2),
            complex(finfo.tiny, finfo.tiny),
            complex(0.0, finfo.tiny),
            complex(finfo.tiny, finfo.tiny),
        ]
        # 预期结果列表
        expected_lst = [
            complex(1.0, 0.0),
            complex(1.0, 0.0),
            complex(1.0, 0.0),
            complex(0.0, -1.0),
            complex(0.0, 0.0),
        ]
        # 创建分子张量
        nom = torch.tensor(nom_lst, dtype=dtype, device=device)
        # 创建分母张量
        denom = torch.tensor(denom_lst, dtype=dtype, device=device)
        # 创建预期结果张量
        expected = torch.tensor(expected_lst, dtype=dtype, device=device)
        # 执行复杂除法操作
        res = nom / denom
        # 断言结果与预期结果相等
        self.assertEqual(res, expected)

    # 测试尝试将 CUDA 张量就地加到 CPU 张量会抛出正确的错误消息
    @onlyCUDA
    def test_cross_device_inplace_error_msg(self, device):
        # 创建一个 CPU 张量 a
        a = torch.tensor(2.0)
        # 创建一个 CUDA 张量 b
        b = torch.tensor(2.0, device=device)
        # 使用 assertRaisesRegex 断言捕获 RuntimeError，并验证错误消息
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            # 尝试将 b 就地加到 a 上
            a += b

    # TODO: refactor this test into a more generic one, it's parked here currently
    @onlyNativeDeviceTypes
    # 测试输出尺寸调整时的警告情况
    def test_out_resize_warning(self, device):
        # 创建张量 a 和 b，指定设备和数据类型
        a = torch.tensor((1, 2, 3), device=device, dtype=torch.float32)
        b = torch.tensor((4, 5, 6), device=device, dtype=torch.float32)

        # 定义一元和二元操作的输入
        unary_inputs = (a,)
        binary_inputs = (a, b)
        
        # 定义一元和二元操作的列表
        unary_ops = (torch.ceil, torch.exp)
        binary_ops = (torch.add, torch.sub)
        
        # 遍历所有操作
        for op in unary_ops + binary_ops:
            # 捕获操作可能引发的警告
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # 根据当前操作选择输入
                inputs = unary_inputs if op in unary_ops else binary_inputs

                # 执行操作，输出到预先创建的空张量，期望不产生警告
                op(*inputs, out=torch.empty(3, device=device))
                op(*inputs, out=torch.empty(0, device=device))
                
                # 验证警告列表长度为 0
                self.assertEqual(len(w), 0)

                # 执行可能引发警告的操作
                op(*inputs, out=torch.empty(2, device=device))
                
                # 验证警告列表长度为 1
                self.assertEqual(len(w), 1)
        
        # 测试多维输出不触发段错误
        arg1 = (torch.ones(2, 1, device=device), torch.ones(1, device=device))
        arg2 = (torch.ones(2, device=device), torch.ones(1, 1, device=device))
        outs = (
            torch.ones(2, 1, 1, 1, device=device),
            torch.ones(2, 2, 2, 2, device=device),
        )

        # 遍历输入和输出组合，检查是否引发警告
        for a1, a2, o in zip(arg1, arg2, outs):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # 执行操作，输出到预先创建的张量，期望引发警告
                torch.mul(a1, a2, out=o)
                
                # 验证警告列表长度为 1
                self.assertEqual(len(w), 1)

    # 验证原地操作（如 idiv）确实是原地操作
    @expectedFailureMeta  # 用户未触发 UserWarning
    @onlyNativeDeviceTypes
    def test_inplace_dunders(self, device):
        # 创建张量 t，随机初始化，指定设备
        t = torch.randn((1,), device=device)
        
        # 记录初始数据指针
        expected = t.data_ptr()
        
        # 进行一系列原地操作
        t += 1
        t -= 1
        t *= 1
        t /= 1
        t **= 1
        t //= 1
        t %= 1
        
        # 验证数据指针未改变
        self.assertEqual(expected, t.data_ptr())

    # 检查内部内存重叠情况
    def check_internal_mem_overlap(
        self, inplace_op, num_inputs, dtype, device, expected_failure=False
    ):
        # 如果 inplace_op 是字符串，则获取对应的张量方法
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        
        # 创建输入张量和其它随机张量作为参数
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input) for i in range(num_inputs - 1)]
        
        # 如果不是预期的失败情况，确保引发 RuntimeError 异常
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, "single memory location"):
                inplace_op(*inputs)
        else:
            # 如果预期失败，使用 Assertion Error 确保不引发异常
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, "single memory location"):
                    inplace_op(*inputs)

    # 检查一元操作的输入输出内存重叠情况
    def unary_check_input_output_mem_overlap(
        self, data, sz, op, expected_failure=False
    ):
        # 定义一个内部测试函数，用于验证操作 op 的输出是否符合预期
        def _test(op, output, input):
            # 创建一个与输出张量相同大小的空张量，用于存储操作结果的期望输出
            output_exp = torch.empty_like(output)
            # 执行操作 op，将结果存入 output_exp 中
            op(input, out=output_exp)
            # 断言操作 op 对于给定的输入和输出是正确的，使用 op 的名称作为消息
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # 输出与输入完全相同的情况：
        _test(op, output=data[0:sz], input=data[0:sz])
        # 输出与输入独立的情况：
        _test(op, output=data[0:sz], input=data[sz : 2 * sz])
        # 输出与输入部分重叠的情况：
        if not expected_failure:
            # 预期不会失败的情况下，使用断言来验证运行时错误信息中包含特定字符串
            with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                _test(op, data[0:sz], data[1 : sz + 1])
        else:
            # 预期会失败的情况下，使用双重断言验证是否会引发断言错误和运行时错误
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                    _test(op, data[0:sz], data[1 : sz + 1])

    # 检查输入与输出之间的内存重叠情况
    def binary_check_input_output_mem_overlap(self, op, device, expected_failure=False):
        # 定义数据大小
        sz = 3
        # 创建一个在指定设备上随机初始化的数据张量
        data = torch.randn(2 * sz, device=device)
        # 创建另一个在指定设备上随机初始化的数据张量
        other = torch.randn(sz, device=device)

        # 调用内部函数来检查输入与输出之间的内存重叠情况
        self.unary_check_input_output_mem_overlap(
            data,
            sz,
            # 使用 lambda 表达式来定义操作函数，将 other 作为第一个参数传入
            lambda input, out: op(other, input, out=out),
            expected_failure=expected_failure,
        )

        # 再次调用内部函数来检查输入与输出之间的内存重叠情况
        self.unary_check_input_output_mem_overlap(
            data,
            sz,
            # 使用 lambda 表达式来定义操作函数，将 input 作为第一个参数传入
            lambda input, out: op(input, other, out=out),
            expected_failure=expected_failure,
        )

    # 标记问题链接以及相关说明
    @xfailIfTorchDynamo
    @dtypes(torch.double)
    # 定义测试函数，用于测试二进制操作的内存重叠情况
    def test_binary_op_mem_overlap(self, device, dtype):
        # 定义包含各种操作和相关参数的列表
        ops = [
            ("add", True, True, "cpu"),      # 加法操作，检查输入输出内存是否重叠，在 CPU 上执行
            ("add", True, True, "cuda"),     # 加法操作，在 CUDA 上执行
            ("mul", True, True, "cpu"),      # 乘法操作，在 CPU 上执行
            ("mul", True, True, "cuda"),     # 乘法操作，在 CUDA 上执行
            ("sub", True, True, "cpu"),      # 减法操作，在 CPU 上执行
            ("sub", True, True, "cuda"),     # 减法操作，在 CUDA 上执行
            ("div", True, True, "cpu"),      # 除法操作，在 CPU 上执行
            ("div", True, True, "cuda"),     # 除法操作，在 CUDA 上执行
            ("pow", True, True, "cpu"),      # 指数操作，在 CPU 上执行
            ("pow", True, True, "cuda"),     # 指数操作，在 CUDA 上执行
            ("fmod", True, True, "cpu"),     # 取模操作，在 CPU 上执行
            ("fmod", True, True, "cuda"),    # 取模操作，在 CUDA 上执行
            ("atan2", True, True, "cpu"),    # 反正切操作，在 CPU 上执行
            ("atan2", True, True, "cuda"),   # 反正切操作，在 CUDA 上执行
            ("hypot", True, True, "cpu"),    # 求hypot操作，在 CPU 上执行
            ("hypot", True, True, "cuda"),   # 求hypot操作，在 CUDA 上执行
            ("igamma", True, True, "cpu"),   # 伽玛函数操作，在 CPU 上执行
            ("igamma", True, True, "cuda"),  # 伽玛函数操作，在 CUDA 上执行
            ("igammac", True, True, "cpu"),  # 补伽玛函数操作，在 CPU 上执行
            ("igammac", True, True, "cuda"), # 补伽玛函数操作，在 CUDA 上执行
            ("nextafter", True, True, "cpu"),# 下一个浮点数操作，在 CPU 上执行
            ("nextafter", True, True, "cuda"),# 下一个浮点数操作，在 CUDA 上执行
            ("le", True, True, "cpu"),       # 小于等于比较操作，在 CPU 上执行
            ("le", True, True, "cuda"),      # 小于等于比较操作，在 CUDA 上执行
            ("lt", True, True, "cpu"),       # 小于比较操作，在 CPU 上执行
            ("lt", True, True, "cuda"),      # 小于比较操作，在 CUDA 上执行
            ("ge", True, True, "cpu"),       # 大于等于比较操作，在 CPU 上执行
            ("ge", True, True, "cuda"),      # 大于等于比较操作，在 CUDA 上执行
            ("gt", True, True, "cpu"),       # 大于比较操作，在 CPU 上执行
            ("gt", True, True, "cuda"),      # 大于比较操作，在 CUDA 上执行
            ("eq", True, True, "cpu"),       # 等于比较操作，在 CPU 上执行
            ("eq", True, True, "cuda"),      # 等于比较操作，在 CUDA 上执行
            ("ne", True, True, "cpu"),       # 不等于比较操作，在 CPU 上执行
            ("ne", True, True, "cuda"),      # 不等于比较操作，在 CUDA 上执行
            ("logical_and", True, True, "cpu"),   # 逻辑与操作，在 CPU 上执行
            ("logical_and", True, True, "cuda"),  # 逻辑与操作，在 CUDA 上执行
            ("logical_or", True, True, "cpu"),    # 逻辑或操作，在 CPU 上执行
            ("logical_or", True, True, "cuda"),   # 逻辑或操作，在 CUDA 上执行
            ("logical_xor", True, True, "cpu"),   # 逻辑异或操作，在 CPU 上执行
            ("logical_xor", True, True, "cuda"),  # 逻辑异或操作，在 CUDA 上执行
        ]

        # 遍历操作列表，逐个进行测试
        for (
            fn,
            has_input_output_mem_overlap_check,
            has_internal_mem_overlap_check,
            dev,
        ) in ops:
            # 如果当前设备不匹配操作所需的设备，则跳过
            if dev != device:
                continue

            # 获取当前操作的输出操作符和就地操作符
            out_op = getattr(torch, fn)
            inplace_op = getattr(torch.Tensor, fn + "_")

            # 调用函数检查就地操作是否存在内存重叠，期望失败取决于操作是否支持内存重叠检查
            self.check_internal_mem_overlap(
                inplace_op,
                2,           # 两个输入参数
                dtype,       # 数据类型
                device,
                expected_failure=not has_internal_mem_overlap_check,  # 是否期望失败
            )

            # 调用函数检查输入输出操作是否存在内存重叠，期望失败取决于操作是否支持输入输出内存重叠检查
            self.binary_check_input_output_mem_overlap(
                out_op,      # 输出操作符
                device,      # 设备
                expected_failure=not has_input_output_mem_overlap_check  # 是否期望失败
            )
    # 对给定张量的指数进行幂运算，用于处理多种指数值
    def _do_pow_for_exponents(self, m1, exponents, pow_fn, atol):
        # 遍历每个指数值
        for num in exponents:
            # 检查是否为负整数，且不是浮点数或复数类型的张量
            if (
                isinstance(num, int)
                and num < 0
                and not m1.is_floating_point()
                and not m1.is_complex()
            ):
                # 断言错误，负整数不能作为指数
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"Integers to negative integer powers are not allowed\.",
                ):
                    torch.pow(m1[4], num)
            else:
                # 对于正常情况，执行张量的指数运算
                # contiguous case
                # 使用 torch.pow 计算张量 m1[4] 的 num 次幂
                res1 = torch.pow(m1[4], num)
                res2 = res1.clone().zero_()
                # 使用传入的 pow_fn 函数计算，处理复数指数的特殊情况
                for i in range(res2.size(0)):
                    res2[i] = pow_fn(m1[4][i], num)
                rtol = 0 if atol is not None else None
                # 断言两种计算方法的结果相等
                self.assertEqual(res1, res2, atol=atol, rtol=rtol)

                # non-contiguous case
                # 对 m1[:, 4] 执行 num 次幂运算
                res1 = torch.pow(m1[:, 4], num)
                res2 = res1.clone().zero_()
                for i in range(res2.size(0)):
                    res2[i] = pow_fn(m1[i, 4], num)
                # 断言两种计算方法的结果相等
                self.assertEqual(res1, res2, atol=atol, rtol=rtol)

                # scalar ** tensor case，确保正确处理 __rpow__() 的数据类型
                expected_dtype = torch.result_type(num, m1)
                # 使用 num ** m1[4] 进行计算
                res1 = num ** m1[4]
                # 使用 torch.tensor 创建期望的数据类型张量，进行 m1[4] 的 num 次幂运算
                res2 = (
                    torch.tensor(num, dtype=expected_dtype, device=m1.device) ** m1[4]
                )
                # 断言两种计算方法的结果相等
                self.assertEqual(res1, res2)
                # 断言结果的数据类型符合预期
                self.assertEqual(res1.dtype, expected_dtype)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    # 定义一个测试方法，用于测试指数运算的不同情况
    def test_pow(self, device, dtype):
        # 创建一个空的张量 m1，数据类型为指定的 dtype，在指定的设备上
        m1 = torch.empty(0, dtype=dtype, device=device)
        # 如果 m1 是浮点数或复数类型
        if m1.is_floating_point() or m1.is_complex():
            # 使用 make_tensor 函数创建一个大小为 (100, 100) 的张量，值范围在 [0, 1) 之间，并加上 0.5
            m1 = (
                make_tensor((100, 100), low=0, high=1, dtype=dtype, device=device) + 0.5
            )
        else:
            # 如果 m1 是整数类型，根据 dtype 确定范围上限，用 make_tensor 创建一个大小为 (100, 100) 的张量
            range_high = 4 if dtype in (torch.int8, torch.uint8) else 10
            m1 = make_tensor(
                (100, 100), low=1, high=range_high, dtype=dtype, device=device
            )

        # 定义指数的一系列值，包括负数、零、正数、复数等
        exponents = [-2.8, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 3.3, True, False]
        complex_exponents = [
            -2.5j,
            -1.0j,
            0j,
            1.0j,
            2.5j,
            1.0 + 1.0j,
            -1.0 - 1.5j,
            3.3j,
        ]
        # 如果 m1 是复数类型，调用 _do_pow_for_exponents 方法处理所有指数情况
        if m1.is_complex():
            self._do_pow_for_exponents(m1, exponents + complex_exponents, pow, 10e-4)
        else:
            # 否则，只处理实数指数，调用 _do_pow_for_exponents 方法，使用 math.pow 函数
            self._do_pow_for_exponents(m1, exponents, math.pow, None)
            # 如果 dtype 是半精度（half）且设备是 CPU，则预期会出现错误
            will_raise_error = (
                dtype is torch.half and torch.device(device).type == "cpu"
            )
            if will_raise_error:
                # 在 CPU 上，半精度张量使用复数指数会导致运算数据类型为 ComplexHalf，目前不支持此操作
                with self.assertRaisesRegex(
                    RuntimeError, "not implemented for 'ComplexHalf'"
                ):
                    self._do_pow_for_exponents(m1, complex_exponents, pow, 10e-4)
            else:
                # 否则，正常处理复数指数情况
                self._do_pow_for_exponents(m1, complex_exponents, pow, 10e-4)

        # 对于基数是数字，指数是张量的情况，进行指数运算并比较结果
        # 连续内存的情况
        res1 = torch.pow(3, m1[4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = pow(3, m1[4, i])
        self.assertEqual(res1, res2)

        # 非连续内存的情况
        res1 = torch.pow(3, m1[:, 4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = pow(3, m1[i][4])
        self.assertEqual(res1, res2)
    # 定义一个测试函数，用于测试幂运算的不同情况
    def _test_pow(self, base, exponent, np_exponent=None):
        # 如果未提供 np_exponent 参数，则使用 exponent 的值
        if np_exponent is None:
            np_exponent = exponent

        # 定义一个辅助函数，将 torch.Tensor 转换为 numpy 数组
        def to_np(value):
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy()
            return value

        # 尝试进行 numpy 的幂运算
        try:
            np_res = np.power(to_np(base), to_np(np_exponent))
            # 根据 numpy 结果创建 torch.Tensor 作为期望的输出
            expected = (
                torch.from_numpy(np_res)
                if isinstance(np_res, np.ndarray)
                else torch.tensor(np_res, dtype=base.dtype)
            )
        # 捕获到 ValueError 异常时执行以下操作
        except ValueError as e:
            # 错误信息应该是 "Integers to negative integer powers are not allowed."
            err_msg = "Integers to negative integer powers are not allowed."
            # 断言异常信息与预期的错误信息相符
            self.assertEqual(str(e), err_msg)
            # 创建一个空的 tensor 作为输出
            out = torch.empty_like(base)
            # 定义多个测试用例，每个用例都应该引发 RuntimeError 异常，并且异常信息匹配 err_msg
            test_cases = [
                lambda: base.pow(exponent),
                lambda: base.pow_(exponent),
                lambda: torch.pow(base, exponent),
                lambda: torch.pow(base, exponent, out=out),
            ]
            # 对每个测试用例进行断言，确保引发了预期的异常
            for test_case in test_cases:
                self.assertRaisesRegex(RuntimeError, err_msg, test_case)
        # 没有捕获到异常时执行以下操作
        else:
            # 如果 base 是 torch.Tensor 类型
            if isinstance(base, torch.Tensor):
                # 计算 base 的 exponent 次幂，与预期的输出进行比较
                actual = base.pow(exponent)
                self.assertEqual(actual, expected.to(actual))
                # 克隆 base，并进行 in-place 操作的测试
                actual = base.clone()
                # 当 base 是 0 维度的 cpu tensor，而 exp 是 cuda tensor 时，测试 pow 和 pow_ 方法的行为
                if (
                    isinstance(exponent, torch.Tensor)
                    and base.dim() == 0
                    and base.device.type == "cpu"
                    and exponent.device.type == "cuda"
                ):
                    # 期望引发的异常信息
                    regex = "Expected all tensors to be on the same device, but found at least two devices, cuda.* and cpu!"
                    self.assertRaisesRegex(RuntimeError, regex, base.pow_, exponent)
                # 如果可以将 base 和 exponent 的数据类型进行转换
                elif torch.can_cast(torch.result_type(base, exponent), base.dtype):
                    # 测试 pow_ 方法的结果是否与预期一致
                    actual2 = actual.pow_(exponent)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual2, expected)
                else:
                    # 如果无法转换数据类型，则测试 pow_ 方法是否引发预期的异常
                    self.assertRaisesRegex(
                        RuntimeError,
                        "Found dtype \\w+ but expected \\w+",
                        lambda: actual.pow_(exponent),
                    )

            # 对 base 和 exponent 使用 torch.pow 方法进行计算，并与预期结果进行比较
            actual = torch.pow(base, exponent)
            self.assertEqual(actual, expected.to(actual))

            # 使用 torch.pow 方法的另一种形式进行计算，同时指定输出变量 actual2
            actual2 = torch.pow(base, exponent, out=actual)
            self.assertEqual(actual, expected.to(actual))
            self.assertEqual(actual2, expected.to(actual))

    # 可能将此注释合并到 OpInfo 中，但一个障碍是第一个输入必须是标量。这并不像只是简单地将其包装在其中
    # 定义一个测试方法，用于测试将第二个输入作为标量的情况。
    # 包含更多逻辑的包装器可能需要实现。
    def test_pow_scalar_base(self, device):
        # 创建一个张量 a，范围从 1 到 12，数据类型为双精度浮点型，存储于指定设备上，并设置梯度跟踪
        a = (
            torch.arange(1, 13, dtype=torch.double, device=device)
            .view(3, 4)  # 将张量视图调整为 3 行 4 列的形状
            .requires_grad_()
        )
        # 使用 gradcheck 方法测试 torch.pow 函数，将 2 作为底数，a 作为指数
        gradcheck(lambda a: torch.pow(2, a), (a,))

    # 测试 pow() 函数，针对整型和浮点型张量及整型和浮点型指数（张量或标量）进行测试。
    # 同时测试非连续张量的情况。
    # 测试当底数张量因 PyTorch 的广播语义而无法调整大小时，pow 的原地变体会引发运行时错误。
    def test_pow_inplace_resizing_exception(self, device):
        # 定义多组测试用例，每组包含底数和指数的形状
        test_cases = (
            ((), (3,)),
            ((2,), (2, 1)),
            ((2, 1), (2, 2)),
            ((2, 2), (2, 1, 1)),
        )
        # 生成测试输入列表，每个输入为一对张量（底数和指数）
        test_inputs = [
            (
                make_tensor(
                    base_size, dtype=torch.float64, device=device, high=10.0, low=0.0
                ),
                make_tensor(
                    exp_size, dtype=torch.float64, device=device, high=10.0, low=0.0
                ),
            )
            for base_size, exp_size in test_cases
        ]
        # 对于每组测试输入，验证 pow_ 方法在调整大小时是否会引发 RuntimeError
        for base, exponent in test_inputs:
            regex = "doesn't match the broadcast shape"
            self.assertRaisesRegex(RuntimeError, regex, base.pow_, exponent)

    # 测试整型张量的负整数指数情况
    def test_int_tensor_pow_neg_ints(self, device):
        # 定义整型张量的一组负整数作为指数
        ints = [
            torch.iinfo(torch.int32).min,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            torch.iinfo(torch.int32).max,
        ]
        # 创建整型张量 tensor，数据类型为 int32，存储于指定设备上
        tensor = torch.tensor(ints, dtype=torch.int32, device=device)
        # 对于每个负整数 pow，调用 _test_pow 方法测试底数为 tensor 的 pow
        for pow in neg_ints:
            self._test_pow(tensor, pow)

    # 测试长整型张量与浮点数指数的情况
    def test_long_tensor_pow_floats(self, device):
        # 定义一组整数作为底数 ints 和一组浮点数作为指数 floats
        ints = [0, 1, 23, 4567]
        floats = [0.0, 1 / 3, 1 / 2, 1.0, 3 / 2, 2.0]
        # 创建长整型张量 tensor，数据类型为 int64，存储于指定设备上
        tensor = torch.tensor(ints, dtype=torch.int64, device=device)
        # 对于每个浮点数 pow，调用 _test_pow 方法测试底数为 tensor 的 pow
        for pow in floats:
            self._test_pow(tensor, pow)

    # 测试浮点数标量与浮点张量底数的情况
    @dtypes(*[torch.float32, torch.float64])
    def test_float_scalar_pow_float_tensor(self, device, dtype):
        # 定义一组浮点数作为底数 floats 和一组不同形状的浮点张量作为指数 tensors
        floats = [2.0, -3 / 2, -1.0, -1 / 2, -1 / 3, 0.0, 1 / 3, 1 / 2, 1.0, 3 / 2, 2.0]
        exponent_shapes = (
            (1,),
            (2, 2),
            (2, 1),
            (2, 2, 2),
        )
        tensors = [
            make_tensor(shape, dtype=dtype, device=device, low=0)
            for shape in exponent_shapes
        ]
        # 创建浮点张量 floats_tensor，数据类型为 dtype，存储于指定设备上
        floats_tensor = torch.tensor(floats, dtype=dtype, device=device)
        # 对于每个浮点数 base，以及每个 tensors 中的张量 tensor，调用 _test_pow 方法测试 pow(base, tensor)
        for base in floats:
            self._test_pow(base, floats_tensor)
            for tensor in tensors:
                self._test_pow(base, tensor)

    # 仅在 CUDA 设备上运行
    # 测试 CUDA 张量与标量张量的幂运算
    def test_cuda_tensor_pow_scalar_tensor(self, device):
        # 创建 CUDA 张量列表
        cuda_tensors = [
            torch.randn((3, 3), device=device),  # 创建一个随机张量
            torch.tensor(3.0, device=device),    # 创建一个标量张量
        ]
        # 创建标量张量列表
        scalar_tensors = [
            torch.tensor(5.0, device="cpu"),  # 创建一个标量张量（在 CPU 上）
            torch.tensor(-3),                 # 创建一个标量张量
            torch.tensor(1),                  # 创建一个标量张量
        ]
        # 对 CUDA 张量列表和标量张量列表进行笛卡尔积遍历
        for base, exp in product(cuda_tensors, scalar_tensors):
            # 调用 _test_pow 方法进行幂运算测试
            self._test_pow(base, exp)

    # 仅在 CUDA 设备上执行的测试，测试 CPU 张量与 CUDA 标量张量的幂运算
    @onlyCUDA
    def test_cpu_tensor_pow_cuda_scalar_tensor(self, device):
        # 创建 CUDA 张量列表
        cuda_tensors = [
            torch.tensor(5.0, device="cuda"),  # 创建一个 CUDA 张量
            torch.tensor(-3, device="cuda"),   # 创建一个 CUDA 张量
        ]
        # 遍历 CUDA 张量列表
        for exp in cuda_tensors:
            # 创建一个在 CPU 上的随机张量
            base = torch.randn((3, 3), device="cpu")
            # 预期引发运行时错误的正则表达式
            regex = "Expected all tensors to be on the same device, but found at least two devices, cuda.* and cpu!"
            # 断言调用 torch.pow 方法时引发 RuntimeError 异常并匹配预期的正则表达式
            self.assertRaisesRegex(RuntimeError, regex, torch.pow, base, exp)
        # 再次遍历 CUDA 张量列表
        for exp in cuda_tensors:
            # 创建一个标量张量（在 CPU 上）
            base = torch.tensor(3.0, device="cpu")
            # 调用 _test_pow 方法进行幂运算测试
            self._test_pow(base, exp)

    # 仅在 CUDA 设备上执行的测试，测试复杂数幂运算的极端失败情况
    @onlyCUDA
    @dtypes(torch.complex64, torch.complex128)
    def test_pow_cuda_complex_extremal_failing(self, device, dtype):
        # 创建一个复杂数张量 t，包含一个复数值
        t = torch.tensor(complex(-1.0, float("inf")), dtype=dtype, device=device)
        # 断言调用 t.pow(2) 时引发 AssertionError 异常
        with self.assertRaises(AssertionError):
            cuda_out = t.pow(2)
            cpu_out = t.cpu().pow(2)
            # 断言 CPU 和 CUDA 计算的结果相等
            self.assertEqual(cpu_out, cuda_out)

    # 跳过 Torch Dynamo 的测试，仅在本地设备类型上执行的测试，测试复杂数标量幂运算与张量的情况
    @skipIfTorchDynamo()
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_complex_scalar_pow_tensor(self, device, dtype):
        # 复杂数列表
        complexes = [0.5j, 1.0 + 1.0j, -1.5j, 2.2 - 1.6j, 1 + 0j]
        # 创建第一个指数张量
        first_exp = make_tensor((100,), dtype=dtype, device=device, low=-2, high=2)
        # 创建第二个指数张量（非连续）
        second_exp = make_tensor(
            (100,), dtype=dtype, device=device, low=-2, high=2, noncontiguous=True
        )
        # 设置第一个指数张量的特定索引为零
        first_exp[0] = first_exp[10] = first_exp[20] = 0
        # 设置第二个指数张量的特定索引为零
        second_exp[0] = second_exp[10] = second_exp[20] = 0
        # 遍历复杂数列表
        for base in complexes:
            # 在 CPU 上，对于 ComplexHalf 数据类型，如果 base 不是 (1+0j)，将引发 NotImplementedError
            will_raise_error = (
                torch.device(device).type == "cpu"
                and dtype is torch.half
                and base != (1 + 0j)
            )
            # 如果将引发错误
            if will_raise_error:
                # 断言调用 _test_pow 方法时引发 RuntimeError 异常，并匹配 "not implemented for 'ComplexHalf'" 的正则表达式
                with self.assertRaisesRegex(
                    RuntimeError, "not implemented for 'ComplexHalf'"
                ):
                    self._test_pow(base, first_exp)
                    self._test_pow(base, second_exp)
            else:
                # 否则，调用 _test_pow 方法进行幂运算测试
                self._test_pow(base, first_exp)
                self._test_pow(base, second_exp)

    # 仅在本地设备类型上执行的测试，跳过元测试
    @onlyNativeDeviceTypes
    @skipMeta
    # 测试幂运算函数对标量和非标量输入的行为
    def test_pow_scalar_type_promotion(self, device):
        # 定义不同类型的输入进行测试
        inputs = [17, [17]]
        for input in inputs:
            # 使用给定设备创建 torch.tensor 对象，数据类型为 uint8
            input_tensor_uint8 = torch.tensor(input, dtype=torch.uint8, device=device)
            # 进行幂运算，期望结果先以 uint8 进行计算（可能溢出为0），然后转换为 int64
            out_uint8_computation = torch.pow(
                2,
                input_tensor_uint8,
                out=torch.tensor(0, dtype=torch.int64, device=device),
            )

            # 使用给定设备创建 torch.tensor 对象，数据类型为 int64
            input_tensor_int64 = torch.tensor(input, dtype=torch.int64, device=device)
            # 进行幂运算，期望结果直接在 int64 范围内计算，不会溢出
            out_int64_computation = torch.pow(
                2,
                input_tensor_int64,
                out=torch.tensor(0, dtype=torch.int64, device=device),
            )

            # 断言 uint8 计算结果和 int64 计算结果不相等
            self.assertNotEqual(out_uint8_computation, out_int64_computation)
            # 将计算结果转换为 uint8 类型后进行断言，期望两者相等
            self.assertEqual(
                out_uint8_computation.to(dtype=torch.uint8),
                out_int64_computation.to(dtype=torch.uint8),
            )

    # 测试张量幂运算函数
    def test_tensor_pow_tensor(self, device):
        # 定义列表旋转函数
        def rotate(l, n):
            return l[-n:] + l[:-n]

        # 定义测试张量与张量幂运算的函数
        def test_tensor_pow_tensor(values, torch_type, numpy_type):
            # 使用给定设备创建 torch.tensor 对象，数据类型为 torch_type
            vals_tensor = torch.tensor(values, dtype=torch_type, device=device)
            for i in range(len(values)):
                # 将 values 列表按照 i 进行旋转，并创建对应的 torch.tensor 对象
                pows = rotate(values, i)
                pows_tensor = torch.tensor(pows, dtype=torch_type, device=device)
                # 调用 _test_pow 方法进行计算结果的断言
                self._test_pow(vals_tensor, pows_tensor)

        # 测试整数类型的幂运算
        ints = [0, 1, 2, 3]
        test_tensor_pow_tensor(ints, torch.uint8, np.uint8)
        test_tensor_pow_tensor(ints, torch.int8, np.int8)
        test_tensor_pow_tensor(ints, torch.int16, np.int16)
        test_tensor_pow_tensor(ints, torch.int32, np.int32)
        test_tensor_pow_tensor(ints, torch.int64, np.int64)

        # 测试浮点数类型的幂运算
        floats = [-3.0, -2.0, -1.0, -1 / 2, -1 / 3, 0.0, 1 / 3, 1 / 2, 1.0, 2.0, 3.0]
        test_tensor_pow_tensor(floats, torch.float16, np.float16)
        test_tensor_pow_tensor(floats, torch.float32, np.float32)
        test_tensor_pow_tensor(floats, torch.float64, np.float64)
    # 在给定设备上测试逻辑异或操作，设备参数由测试框架传入
    def test_logical_xor_with_nontrivial_alignment(self, device):
        # 创建大小为 128 的随机张量，并将其转换为布尔类型张量
        size = 128
        a = torch.randn(size, device=device) > 0
        b = torch.randn(size, device=device) > 0
        c = torch.randn(size, device=device) > 0
        # 非常规对齐方式的示例列表
        non_trivial_alignment = [1, 2, 4, 8, 15]
        # 循环遍历所有非常规对齐方式的组合
        for i in non_trivial_alignment:
            for j in non_trivial_alignment:
                for k in non_trivial_alignment:
                    # 从张量 a、b、c 中切片出对应的子张量
                    a_ = a[i : 100 + i]
                    b_ = b[j : 100 + j]
                    c_ = c[k : 100 + k]
                    # 使用逻辑异或操作符计算 a_ 和 b_，结果存储在 c_ 中
                    torch.logical_xor(a_, b_, out=c_)
                    # 验证每个元素的异或操作是否正确
                    for x, y, z in zip(a_.tolist(), b_.tolist(), c_.tolist()):
                        self.assertEqual(x ^ y, z)

    @dtypes(torch.float)
    def test_add_with_tail(self, device, dtype):
        # 测试包含尾部的张量，尾部大小不是 GPU warp 大小的倍数
        for tail_size in [1, 63, 67, 130]:
            size = 4096 + tail_size
            # 在给定设备上创建指定类型的随机张量 a 和 b
            a = torch.randn(size, device=device, dtype=dtype)
            b = torch.randn(size, device=device, dtype=dtype)
            # 计算张量 a 和 b 的加法
            c = a + b
            # 验证每个元素的加法是否正确
            for x, y, z in zip(a.tolist(), b.tolist(), c.tolist()):
                self.assertEqual(x + y, z)

    # 测试 CUDA 张量在不同设备上进行二元操作的情况，
    # 以及 CUDA "标量" 与非标量 CPU 张量进行二元操作的情况
    @deviceCountAtLeast(2)
    @onlyCUDA
    def test_cross_device_binary_ops(self, devices):
        vals = (1.0, (2.0,))
        cpu_tensor = torch.randn(2, 2)

        def do_test(op, a, b):
            # 检查不同设备上的张量进行二元操作时是否抛出预期的异常
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(a, b)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(b, a)
            # 检查 CUDA "标量" 与非标量 CPU 张量进行二元操作时是否抛出预期的异常
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(a, cpu_tensor)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors.+"):
                op(cpu_tensor, a)

        # 遍历不同的二元操作符和值组合
        for op in (
            operator.add,
            torch.add,
            operator.sub,
            torch.sub,
            operator.mul,
            torch.mul,
            operator.truediv,
            torch.true_divide,
            operator.floordiv,
            torch.floor_divide,
        ):
            for a, b in product(vals, vals):
                # 在给定设备上创建张量 a 和 b
                a = torch.tensor(a, device=devices[0])
                b = torch.tensor(b, device=devices[1])

            # 执行测试函数，验证各种组合的二元操作是否符合预期
            do_test(op, a, b)

    # 此测试确保标量张量可以安全地与所有可用 CUDA 设备上的张量进行二元操作
    @deviceCountAtLeast(2)
    @onlyCUDA
    # 测试二进制运算与标量，设备未指定情况下的行为
    def test_binary_op_scalar_device_unspecified(self, devices):
        # 创建标量张量，值为1.0
        scalar_val = torch.tensor(1.0)
        # 遍历设备列表
        for default_device in devices:
            # 在 CUDA 设备上设置默认设备
            with torch.cuda.device(default_device):
                # 再次遍历设备列表
                for device in devices:
                    # 创建指定设备的张量对象
                    device_obj = torch.device(device)
                    # 在指定设备上创建形状为(3,)的随机张量
                    x = torch.rand(3, device=device)
                    # 标量与张量的乘法，结果张量的设备应与标量的设备一致
                    y0 = x * scalar_val
                    self.assertEqual(y0.device, device_obj)
                    # 张量与标量的乘法，结果张量的设备应与标量的设备一致
                    y1 = scalar_val * x
                    self.assertEqual(y1.device, device_obj)
                    # 检查两种乘法结果应相等
                    self.assertEqual(y0, y1)

    # 测试除法和整数除法与 Python 行为的对比
    def test_div_and_floordiv_vs_python(self, device):
        # 辅助函数，测试处理标量的除法操作
        def _scalar_helper(python_op, torch_op):
            # 遍历(-10, 10)的组合
            for a, b in product(range(-10, 10), range(-10, 10)):
                # 遍历两种操作：乘以0.5和向下取整
                for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                    a = op(a)
                    b = op(b)

                    # 跳过除数为零的情况
                    if b == 0:
                        continue

                    # 计算预期结果
                    expected = python_op(a, b)

                    # 遍历真除和torch.true_divide两种操作
                    for op in (operator.truediv, torch.true_divide):
                        # 计算标量的实际结果
                        actual_scalar = torch_op(a, b)

                        # 创建张量
                        a_t = torch.tensor(a, device=device)
                        b_t = torch.tensor(b, device=device)

                        # 计算张量的实际结果
                        actual_tensor = torch_op(a_t, b_t)
                        actual_first_tensor = torch_op(a_t, b)
                        actual_second_tensor = torch_op(a, b_t)

                        # 断言标量和张量的结果应与预期相等
                        self.assertEqual(actual_scalar, expected)
                        self.assertEqual(actual_tensor.item(), expected)
                        self.assertEqual(actual_first_tensor, actual_tensor)
                        self.assertEqual(actual_second_tensor, actual_tensor)

        # 使用真除法测试标量操作
        _scalar_helper(operator.truediv, operator.truediv)
        _scalar_helper(operator.truediv, torch.true_divide)
        # 使用整数除法测试标量操作
        _scalar_helper(lambda a, b: math.floor(a / b), operator.floordiv)
        _scalar_helper(lambda a, b: math.floor(a / b), torch.floor_divide)

    # 只对本地设备类型进行测试，跳过TorchDynamo环境
    @onlyNativeDeviceTypes
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_div_and_floordiv_script_vs_python(self, device):
        # Creates jitted functions of two tensors
        # 定义对两个张量进行JIT编译的函数
        def _wrapped_div(a, b):
            return a / b

        def _wrapped_floordiv(a, b):
            return a // b

        # 对_wrapped_div和_wrapped_floordiv进行JIT编译
        scripted_div = torch.jit.script(_wrapped_div)
        scripted_floordiv = torch.jit.script(_wrapped_floordiv)

        # 遍历范围为-10到10的所有整数对(a, b)
        for a, b in product(range(-10, 10), range(-10, 10)):
            # 针对a和b应用lambda函数(op)，lambda函数为乘以0.5或向下取整
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)
                b = op(b)

                # 跳过除数为零的情况
                if b == 0:
                    continue

                # 计算预期的除法和整除结果
                expected_div = a / b
                expected_floordiv = math.floor(a / b)
                a_t = torch.tensor(a, device=device)
                b_t = torch.tensor(b, device=device)

                # 使用JIT编译的函数进行断言，验证结果是否符合预期
                self.assertEqual(scripted_div(a_t, b_t), expected_div)
                self.assertEqual(scripted_floordiv(a_t, b_t), expected_floordiv)

        # Creates jitted functions of one tensor
        # 定义对单个张量进行JIT编译的函数
        def _wrapped_div_scalar(a):
            return a / 5

        # NOTE: the JIT implements division as torch.reciprocal(a) * 5
        # 注意：JIT将除法实现为torch.reciprocal(a) * 5
        def _wrapped_rdiv_scalar(a):
            return 5 / a

        def _wrapped_floordiv_scalar(a):
            return a // 5

        # NOTE: this fails if the input is not an integer tensor
        # 如果输入不是整数张量，则会失败
        # See https://github.com/pytorch/pytorch/issues/45199
        def _wrapped_rfloordiv_scalar(a):
            return 5 // a

        # 对_wrapped_div_scalar、_wrapped_rdiv_scalar、
        # _wrapped_floordiv_scalar和_wrapped_rfloordiv_scalar进行JIT编译
        scripted_div_scalar = torch.jit.script(_wrapped_div_scalar)
        scripted_rdiv_scalar = torch.jit.script(_wrapped_rdiv_scalar)
        scripted_floordiv_scalar = torch.jit.script(_wrapped_floordiv_scalar)
        scripted_rfloordiv_scalar = torch.jit.script(_wrapped_rfloordiv_scalar)

        # 遍历范围为-10到10的所有整数a
        for a in range(-10, 10):
            # 针对a应用lambda函数(op)，lambda函数为乘以0.5或向下取整
            for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                a = op(a)

                a_t = torch.tensor(a, device=device)

                # 断言使用JIT编译的函数的结果是否与预期的除法结果相等
                self.assertEqual(a / 5, scripted_div_scalar(a_t))

                # 跳过除数为零的情况
                if a == 0:
                    continue

                # 断言使用JIT编译的函数的结果是否与预期的逆除法结果相等
                self.assertEqual(5 / a, scripted_rdiv_scalar(a_t))

                # 处理Issue 45199（参见上面的注释）
                if a_t.is_floating_point():
                    # 如果是浮点数张量，断言会抛出RuntimeError异常
                    with self.assertRaises(RuntimeError):
                        scripted_rfloordiv_scalar(a_t)
                else:
                    # 否则，验证使用JIT编译的函数的结果是否与预期的逆整除结果相等
                    self.assertEqual(5 // a, scripted_rfloordiv_scalar(a_t))

    @onlyNativeDeviceTypes
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # Tests binary op equivalence with Python builtin ops
    # Also tests that reverse operations are equivalent to forward ops
    # NOTE: division ops are tested separately above
    # 使用 pytest 框架进行测试，测试二元操作函数与标量操作
    def test_binary_ops_with_scalars(self, device):
        # 对于每一对 python_op 和 torch_op 组合进行循环测试
        for python_op, torch_op in (
            (operator.add, torch.add),
            (operator.sub, torch.sub),
            (operator.mul, torch.mul),
            (operator.truediv, torch.div),
        ):
            # 对于每一对整数 a 和 b，以及每一个操作 op 进行循环测试
            for a, b in product(range(-10, 10), range(-10, 10)):
                for op in (lambda x: x * 0.5, lambda x: math.floor(x)):
                    # 应用操作 op 到 a 和 b
                    a = op(a)
                    b = op(b)

                    # 跳过除数为零的情况
                    if b == 0 or a == 0:
                        continue

                    # 创建 torch 张量 a 和 b，并指定设备
                    a_tensor = torch.tensor(a, device=device)
                    b_tensor = torch.tensor(b, device=device)
                    # 将张量移动到 CPU
                    a_tensor_cpu = a_tensor.cpu()
                    b_tensor_cpu = b_tensor.cpu()
                    # 构建值元组
                    vals = (a, b, a_tensor, b_tensor, a_tensor_cpu, b_tensor_cpu)

                    # 对于每一对 vals 中的元素进行测试
                    for args in product(vals, vals):
                        first, second = args

                        # 如果 first 不是 torch.Tensor，则将其作为标量处理
                        first_scalar = (
                            first
                            if not isinstance(first, torch.Tensor)
                            else first.item()
                        )
                        # 如果 second 不是 torch.Tensor，则将其作为标量处理
                        second_scalar = (
                            second
                            if not isinstance(second, torch.Tensor)
                            else second.item()
                        )
                        # 计算预期结果
                        expected = python_op(first_scalar, second_scalar)

                        # 断言预期结果与 python_op 对 first 和 second 的结果相同
                        self.assertEqual(expected, python_op(first, second))
                        # 断言预期结果与 torch_op 对 first 和 second 的结果相同
                        self.assertEqual(expected, torch_op(first, second))

    # 使用 @dtypes 装饰器，测试最大和最小函数的类型提升
    @dtypes(
        *product(
            all_types_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_maximum_minimum_type_promotion(self, device, dtypes):
        # 创建 torch 张量 a 和 b，指定设备和数据类型
        a = torch.tensor((0, 1), device=device, dtype=dtypes[0])
        b = torch.tensor((1, 0), device=device, dtype=dtypes[1])
        # 对于每一个最大或最小函数 op 进行循环测试
        for op in (
            torch.maximum,
            torch.max,
            torch.fmax,
            torch.minimum,
            torch.min,
            torch.fmin,
        ):
            # 计算 op(a, b) 的结果
            result = op(a, b)
            # 断言结果的数据类型与 torch.result_type(a, b) 相同
            self.assertEqual(result.dtype, torch.result_type(a, b))

    # 使用 @dtypes 装饰器，测试整数类型和 torch.bool 类型的函数
    @dtypes(*integral_types_and(torch.bool))
    # 定义一个测试函数，用于测试整数和布尔值的最大最小操作
    def test_maximum_minimum_int_and_bool(self, device, dtype):
        # 定义不同的操作函数和对应的别名和numpy函数
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )
        # 使用numpy的随机数生成器创建一组随机整数数组a_np和b_np
        rng = np.random.default_rng()
        a_np = np.array(
            rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype]
        )
        b_np = np.array(
            rng.integers(-100, 100, size=10), dtype=torch_to_numpy_dtype_dict[dtype]
        )

        # 遍历每一种操作
        for torch_op, alias, numpy_op in ops:
            # 将numpy数组a_np转换为torch张量a_tensor，并发送到指定的设备和数据类型
            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            # 将numpy数组b_np转换为torch张量b_tensor，并发送到指定的设备和数据类型
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            # 使用torch_op计算张量a_tensor和b_tensor的结果
            tensor_result = torch_op(a_tensor, b_tensor)

            # 创建一个形状与a_tensor相同的空张量out
            out = torch.empty_like(a_tensor)
            # 将torch_op的结果存储到out张量中
            torch_op(a_tensor, b_tensor, out=out)

            # 使用numpy_op计算numpy数组a_np和b_np的结果
            numpy_result = numpy_op(a_np, b_np)

            # 如果存在别名函数，计算别名函数在a_tensor和b_tensor上的结果，并断言它与tensor_result相等
            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            # 断言torch_op的结果与numpy_op的结果相等
            self.assertEqual(tensor_result, numpy_result)
            # 断言out张量与numpy_op的结果相等
            self.assertEqual(out, numpy_result)

    # 用于测试浮点数的最大最小操作，考虑了特定的精度覆盖和数据类型
    @precisionOverride({torch.bfloat16: 1e-2})
    @dtypes(*(floating_types_and(torch.half, torch.bfloat16)))
    def test_maximum_minimum_float(self, device, dtype):
        # 定义不同的操作函数和对应的别名和numpy函数
        ops = (
            (torch.maximum, torch.max, np.maximum),
            (torch.minimum, torch.min, np.minimum),
            (torch.fmax, None, np.fmax),
            (torch.fmin, None, np.fmin),
        )

        # 根据dtype选择不同的numpy随机数生成策略来生成a_np和b_np
        if dtype == torch.bfloat16:
            # 当dtype为torch.bfloat16时，生成浮点数数组a_np和b_np
            a_np = np.random.randn(10).astype(np.float64)
            b_np = np.random.randn(10).astype(np.float64)
        else:
            # 当dtype为其他浮点类型时，根据dtype将随机生成的numpy数组a_np和b_np转换为相应的类型
            a_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])
            b_np = np.random.randn(10).astype(torch_to_numpy_dtype_dict[dtype])

        # 遍历每一种操作
        for torch_op, alias, numpy_op in ops:
            # 使用numpy_op计算numpy数组a_np和b_np的结果
            numpy_result = numpy_op(a_np, b_np)

            # 将numpy数组a_np转换为torch张量a_tensor，并发送到指定的设备和数据类型
            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            # 将numpy数组b_np转换为torch张量b_tensor，并发送到指定的设备和数据类型
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            # 使用torch_op计算张量a_tensor和b_tensor的结果
            tensor_result = torch_op(a_tensor, b_tensor)
            
            # 创建一个形状与a_tensor相同的空张量out
            out = torch.empty_like(a_tensor)
            # 将torch_op的结果存储到out张量中
            torch_op(a_tensor, b_tensor, out=out)

            # 如果存在别名函数，计算别名函数在a_tensor和b_tensor上的结果，并断言它与tensor_result相等
            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result, exact_dtype=False)

            # 断言torch_op的结果与numpy_op的结果相等
            self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
            # 断言out张量与numpy_op的结果相等
            self.assertEqual(out, numpy_result, exact_dtype=False)
    def test_maximum_minimum_float_nan_and_inf(self, device, dtype):
        # np.maximum and np.minimum functions compare input arrays element-wisely.
        # if one of the elements being compared is a NaN, then that element is returned.
        # 定义一系列操作函数，每个元组包含了 Torch 和 NumPy 的最大、最小、fmax、fmin函数
        ops = (
            (torch.maximum, torch.max, np.maximum),  # 最大值函数
            (torch.minimum, torch.min, np.minimum),  # 最小值函数
            (torch.fmax, None, np.fmax),              # fmax 函数
            (torch.fmin, None, np.fmin),              # fmin 函数
        )
        # 定义两个输入数组，包括正负无穷、NaN以及普通数值
        a_vals = (
            float("inf"),
            -float("inf"),
            float("nan"),
            float("inf"),
            float("nan"),
            float("nan"),
            1,
            float("nan"),
        )
        b_vals = (
            -float("inf"),
            float("inf"),
            float("inf"),
            float("nan"),
            float("nan"),
            0,
            float("nan"),
            -5,
        )
        # 根据数据类型选择正确的 NumPy 数据类型
        if dtype == torch.bfloat16:
            a_np = np.array(a_vals, dtype=np.float64)
            b_np = np.array(b_vals, dtype=np.float64)
        else:
            a_np = np.array(a_vals, dtype=torch_to_numpy_dtype_dict[dtype])
            b_np = np.array(b_vals, dtype=torch_to_numpy_dtype_dict[dtype])

        # 对每种操作进行迭代测试
        for torch_op, alias, numpy_op in ops:
            # 使用 NumPy 计算预期结果
            numpy_result = numpy_op(a_np, b_np)

            # 将 NumPy 数组转换为 Torch 张量，并设置设备和数据类型
            a_tensor = torch.from_numpy(a_np).to(device=device, dtype=dtype)
            b_tensor = torch.from_numpy(b_np).to(device=device, dtype=dtype)
            
            # 使用 Torch 函数计算张量结果
            tensor_result = torch_op(a_tensor, b_tensor)

            # 创建一个与输入张量相同大小的空张量，并使用 Torch 函数计算结果
            out = torch.empty_like(a_tensor)
            torch_op(a_tensor, b_tensor, out=out)

            # 如果存在别名函数，验证其结果与 Torch 函数结果一致
            if alias is not None:
                alias_result = alias(a_tensor, b_tensor)
                self.assertEqual(alias_result, tensor_result)

            # 根据数据类型验证 Torch 计算结果与 NumPy 预期结果一致性
            if dtype == torch.bfloat16:
                self.assertEqual(tensor_result, numpy_result, exact_dtype=False)
                self.assertEqual(out, numpy_result, exact_dtype=False)
            else:
                self.assertEqual(tensor_result, numpy_result)
                self.assertEqual(out, numpy_result)

    @dtypes(
        *product(
            complex_types(),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    # 定义一个测试方法，用于测试多种 torch 操作符的最大值和最小值函数
    def test_maximum_minimum_complex(self, device, dtypes):
        # 遍历多种 torch 操作符
        for torch_op in (
            torch.maximum,
            torch.minimum,
            torch.max,
            torch.min,
            torch.fmax,
            torch.fmin,
        ):
            # 断言对于两个相同维度的张量，调用这些操作符会抛出 RuntimeError 异常，表示不支持操作
            with self.assertRaisesRegex(RuntimeError, ".+not implemented for.+"):
                torch_op(
                    torch.ones(1, device=device, dtype=dtypes[0]),
                    torch.ones(1, device=device, dtype=dtypes[1]),
                )

            # 再次断言，但是将输入张量的顺序颠倒，仍然会抛出相同的 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, ".+not implemented for.+"):
                torch_op(
                    torch.ones(1, device=device, dtype=dtypes[1]),
                    torch.ones(1, device=device, dtype=dtypes[0]),
                )

    # 仅在 CUDA 设备上执行的测试方法，用于测试在不同设备上调用最大值和最小值函数时的行为
    @onlyCUDA
    def test_maximum_minimum_cross_device(self, device):
        # 创建两个张量 a 和 b，其中 b 在指定的设备上
        a = torch.tensor((1, 2, -1))
        b = torch.tensor((3, 0, 4), device=device)
        # 操作符列表，包括最大值和最小值
        ops = (torch.maximum, torch.minimum)

        # 遍历操作符列表
        for torch_op in ops:
            # 断言调用操作符时会抛出 RuntimeError 异常，因为张量 a 和 b 不在同一设备上
            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch_op(a, b)

            # 再次断言，但是顺序颠倒，同样会抛出 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch_op(b, a)

        # 测试 CUDA 张量和 CPU 标量的情况
        # ops 包括 torch 操作符和对应的 numpy 操作符
        ops = ((torch.maximum, np.maximum), (torch.minimum, np.minimum))
        a_np = np.array(1)
        b_np = np.array([3, 0, 4])

        # 遍历 ops 列表
        for torch_op, numpy_op in ops:
            # 创建 torch 张量 a_tensor 和 b_tensor，将 b_tensor 放在指定设备上
            a_tensor = torch.from_numpy(a_np)
            b_tensor = torch.from_numpy(b_np).to(device=device)
            # 调用 torch 操作符和 numpy 操作符，进行计算并比较结果
            tensor_result_1 = torch_op(a_tensor, b_tensor)
            numpy_result_1 = numpy_op(a_np, b_np)
            tensor_result_2 = torch_op(b_tensor, a_tensor)
            numpy_result_2 = numpy_op(b_np, a_np)

            # 断言 torch 张量和 numpy 数组的计算结果相同
            self.assertEqual(tensor_result_1, numpy_result_1)
            self.assertEqual(tensor_result_2, numpy_result_2)

    # 使用装饰器 dtypes，测试 torch 的最大值和最小值函数的子梯度行为
    @dtypes(
        *product(
            floating_types_and(torch.half, torch.bfloat16),
            floating_types_and(torch.half, torch.bfloat16),
        )
    )
    def test_maximum_and_minimum_subgradient(self, device, dtypes):
        # 定义一个内部函数 run_test，用于运行测试
        def run_test(f, a, b, expected_a_grad, expected_b_grad):
            # 创建 torch 张量 a 和 b，指定为可计算梯度
            a = torch.tensor(a, requires_grad=True, device=device, dtype=dtypes[0])
            b = torch.tensor(b, requires_grad=True, device=device, dtype=dtypes[1])
            # 计算函数 f(a, b) 的结果，并对结果进行求和求梯度
            z = f(a, b)
            z.sum().backward()
            # 断言张量 a 和 b 的梯度与预期值相等
            self.assertEqual(a.grad, expected_a_grad)
            self.assertEqual(b.grad, expected_b_grad)

        # 调用 run_test 函数测试 torch.maximum 函数
        run_test(
            torch.maximum,
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0],
        )

        # 调用 run_test 函数测试 torch.minimum 函数
        run_test(
            torch.minimum,
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 1.0],
        )
    # 定义测试方法，用于测试最大值和最小值函数在浮点数（float32）上的正向自动微分
    def test_maximum_minimum_forward_ad_float32(self, device):
        # TODO: 这部分应该由 OpInfo 覆盖，但实际未覆盖。问题在于我们的梯度测试使用的是 float64，但也应该测试 float32
        # 创建随机张量 x, y, tx, ty，设备为指定设备，数据类型为 float32
        x = torch.randn(3, device=device, dtype=torch.float32)
        y = torch.randn(3, device=device, dtype=torch.float32)
        tx = torch.randn(3, device=device, dtype=torch.float32)
        ty = torch.randn(3, device=device, dtype=torch.float32)

        # 进入双重级别的自动微分环境
        with fwAD.dual_level():
            # 将 x, tx 封装成对偶数（dual number）
            x_dual = fwAD.make_dual(x, tx)
            # 将 y, ty 封装成对偶数（dual number）
            y_dual = fwAD.make_dual(y, ty)
            # 计算 x_dual 和 y_dual 的最大值
            result = torch.maximum(x_dual, y_dual)
            # 解包 result 得到原始值和对应的切线值（tangent）
            _, result_tangent = fwAD.unpack_dual(result)

        # 期望的结果是根据 x 和 y 的大小关系，选择 tx 或 ty
        expected = torch.where(x > y, tx, ty)
        # 断言计算得到的切线值与期望值相等
        self.assertEqual(result_tangent, expected)

        # 再次进入双重级别的自动微分环境
        with fwAD.dual_level():
            # 重新封装 x, tx 和 y, ty 成对偶数（dual number）
            x_dual = fwAD.make_dual(x, tx)
            y_dual = fwAD.make_dual(y, ty)
            # 计算 x_dual 和 y_dual 的最小值
            result = torch.minimum(x_dual, y_dual)
            # 解包 result 得到原始值和对应的切线值（tangent）
            _, result_tangent = fwAD.unpack_dual(result)

        # 期望的结果是根据 x 和 y 的大小关系，选择 tx 或 ty
        expected = torch.where(x < y, tx, ty)
        # 断言计算得到的切线值与期望值相等
        self.assertEqual(result_tangent, expected)

    # TODO: 这样的测试应该是通用的
    # 根据CUDA设备类型选择数据类型（torch.half, torch.float, torch.double），并测试不同数据类型的乘法运算
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_mul_intertype_scalar(self, device, dtype):
        # 创建一个浮点数张量 x，数据类型为指定的 dtype，设备为指定设备
        x = torch.tensor(1.5, dtype=dtype, device=device)
        # 创建一个整数张量 y，数据类型为 torch.int32，设备为指定设备
        y = torch.tensor(3, dtype=torch.int32, device=device)

        # 断言 x 乘以 y 的结果等于 4.5
        self.assertEqual(x * y, 4.5)
        # 断言 y 乘以 x 的结果等于 4.5
        self.assertEqual(y * x, 4.5)

        # 使用断言检查运行时错误，确保不能将 y 原地乘以 x
        with self.assertRaisesRegex(
            RuntimeError, "can't be cast to the desired output type"
        ):
            y *= x
        # 将 x 原地乘以 y
        x *= y
        # 断言 x 的值等于 4.5
        self.assertEqual(x, 4.5)

    # 只在 CPU 上执行的测试
    # 测试各种数据类型（包括 torch.half, torch.bfloat16, torch.bool）的乘法运算
    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    # 定义一个测试函数 test_sub，接受参数 self（测试类实例）、device（设备类型）、dtype（张量数据类型）
    def test_sub(self, device, dtype):
        # 如果 dtype 是整数类型（包括 int、long 等），执行以下代码块
        if dtype in integral_types():
            # 创建包含整数的张量 m1 和 m2，并指定设备和数据类型
            m1 = torch.tensor([2, 4], dtype=dtype, device=device)
            m2 = torch.tensor([1, 2], dtype=dtype, device=device)
            # 创建差值张量 diff，数据类型为 dtype
            diff = torch.tensor([1, 2], dtype=dtype)
        else:
            # 否则，创建包含浮点数的张量 m1 和 m2，并指定设备和数据类型
            m1 = torch.tensor([2.34, 4.44], dtype=dtype, device=device)
            m2 = torch.tensor([1.23, 2.33], dtype=dtype, device=device)
            # 创建差值张量 diff，数据类型为 dtype
            diff = torch.tensor([1.11, 2.11], dtype=dtype)

        # 如果 dtype 是 torch.bool 类型，预期引发 RuntimeError 异常
        if dtype == torch.bool:
            self.assertRaises(RuntimeError, lambda: m1 - m2)
        elif dtype == torch.bfloat16 or dtype == torch.half:
            # 如果 dtype 是 torch.bfloat16 或 torch.half 类型，使用自定义容差进行相等断言
            # bfloat16 具有较低的精度，因此需要单独检查
            self.assertEqual(m1 - m2, diff, atol=0.01, rtol=0)
        else:
            # 否则，使用默认容差进行相等断言
            self.assertEqual(m1 - m2, diff)

    # TODO: what is this test testing?
    # 用装饰器标记此测试仅适用于 CPU，并且 dtype 为 torch.float 类型
    @onlyCPU
    @dtypes(torch.float)
    # 定义一个测试函数 test_csub，接受参数 self（测试类实例）、device（设备类型）、dtype（张量数据类型）
    def test_csub(self, device, dtype):
        # 创建随机张量 a，数据类型为 dtype，指定设备类型为 device
        a = torch.randn(100, 90, dtype=dtype, device=device)
        # 克隆张量 a 并对其执行正态分布初始化，得到张量 b
        b = a.clone().normal_()

        # 计算 a - b 的结果，并指定 alpha=-1
        res_add = torch.add(a, b, alpha=-1)
        # 克隆张量 a 作为 res_csub，执行就地减法操作 b
        res_csub = a.clone()
        res_csub.sub_(b)
        # 断言 res_add 与 res_csub 相等
        self.assertEqual(res_add, res_csub)

        # 创建随机张量 a，数据类型为 dtype，设备类型为 device
        a = torch.randn(100, 100, dtype=dtype, device=device)

        # 定义一个标量 scalar
        scalar = 123.5
        # 计算 a - scalar 的结果
        res_add = torch.add(a, -scalar)
        # 克隆张量 a 作为 res_csub，执行就地减法操作 scalar
        res_csub = a.clone()
        res_csub.sub_(scalar)
        # 断言 res_add 与 res_csub 相等
        self.assertEqual(res_add, res_csub)

    # TODO: reconcile with minimum/maximum tests
    # 用于标记测试函数的装饰器，如果在 CUDA 上，dtype 可选 torch.half、torch.float、torch.double
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    # 用于标记测试函数的装饰器，dtype 可选 torch.float、torch.double
    @dtypes(torch.float, torch.double)
    # 定义测试函数，用于测试 torch.min 和 torch.max 函数对包含 NaN 的张量的行为
    def test_min_max_binary_op_nan(self, device, dtype):
        # 创建两个随机张量 a 和 b，形状为 (1000,)，指定设备和数据类型
        a = torch.rand(1000, dtype=dtype, device=device)
        b = torch.rand(1000, dtype=dtype, device=device)

        # 设置张量 a 的前 250 个元素为 NaN
        a[:250] = float("nan")
        # 设置张量 b 的第 250 到 500 个元素为 NaN
        b[250:500] = float("nan")
        # 设置张量 a 和 b 的第 500 到 750 个元素为 NaN
        a[500:750] = float("nan")
        b[500:750] = float("nan")
        # 设置张量 a 和 b 的剩余元素为非 NaN

        # 计算张量 a 和 b 每个位置的最大值和最小值
        ma = torch.max(a, b)
        mi = torch.min(a, b)

        # 验证前 750 个元素的最大值和最小值是否为 NaN
        for i in range(750):
            self.assertTrue(
                torch.isnan(ma[i]),
                f"max(a, b): {ma[i]}, a: {a[i]}, b: {b[i]}",
            )
            self.assertTrue(
                torch.isnan(mi[i]),
                f"min(a, b): {mi[i]}, a: {a[i]}, b: {b[i]}",
            )

        # 验证剩余元素的最大值和最小值是否不为 NaN
        for i in range(750, 1000):
            self.assertFalse(
                torch.isnan(ma[i]),
                f"max(a, b): {ma[i]}, a: {a[i]}, b: {b[i]}",
            )
            self.assertFalse(
                torch.isnan(mi[i]),
                f"min(a, b): {mi[i]}, a: {a[i]}, b: {b[i]}",
            )
    # 测试 torch.copysign 函数在多种输入情况下的子梯度计算
    def test_copysign_subgradient(self, device, dtypes):
        # Input is 0.0
        # 创建一个包含三个元素的张量 x，数据类型为 dtypes[0]，存储在指定设备上，并且需要计算梯度
        x = torch.tensor(
            [0.0, 0.0, 0.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        # 创建一个包含三个元素的张量 y，数据类型为 dtypes[1]，存储在指定设备上，并且需要计算梯度
        y = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        # 使用 torch.copysign 计算 x 和 y 元素对应位置的带符号值，并赋给 out
        out = torch.copysign(x, y)
        # 对 out 求和并反向传播梯度
        out.sum().backward()
        # 断言 x 的梯度为 [0.0, 0.0, 0.0]
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        # 断言 y 的梯度为 [0.0, 0.0, 0.0]
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Input is -0.0
        # 创建一个包含三个元素的张量 x，数据类型为 dtypes[0]，存储在指定设备上，并且需要计算梯度
        x = torch.tensor(
            [-0.0, -0.0, -0.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        # 创建一个包含三个元素的张量 y，数据类型为 dtypes[1]，存储在指定设备上，并且需要计算梯度
        y = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        # 使用 torch.copysign 计算 x 和 y 元素对应位置的带符号值，并赋给 out
        out = torch.copysign(x, y)
        # 对 out 求和并反向传播梯度
        out.sum().backward()
        # 断言 x 的梯度为 [0.0, 0.0, 0.0]
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        # 断言 y 的梯度为 [0.0, 0.0, 0.0]
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is 0.0
        # 创建一个包含三个元素的张量 x，数据类型为 dtypes[0]，存储在指定设备上，并且需要计算梯度
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        # 创建一个包含三个元素的张量 y，数据类型为 dtypes[1]，存储在指定设备上，并且需要计算梯度
        y = torch.tensor(
            [0.0, 0.0, 0.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        # 使用 torch.copysign 计算 x 和 y 元素对应位置的带符号值，并赋给 out
        out = torch.copysign(x, y)
        # 对 out 求和并反向传播梯度
        out.sum().backward()
        # 断言 x 的梯度为 [-1.0, 0.0, 1.0]
        self.assertEqual(x.grad.tolist(), [-1.0, 0.0, 1.0])
        # 断言 y 的梯度为 [0.0, 0.0, 0.0]
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is -0.0
        # 创建一个包含三个元素的张量 x，数据类型为 dtypes[0]，存储在指定设备上，并且需要计算梯度
        x = torch.tensor(
            [-1.0, 0.0, 1.0], dtype=dtypes[0], device=device, requires_grad=True
        )
        # 创建一个包含三个元素的张量 y，数据类型为 dtypes[1]，存储在指定设备上，并且需要计算梯度
        y = torch.tensor(
            [-0.0, -0.0, -0.0], dtype=dtypes[1], device=device, requires_grad=True
        )
        # 使用 torch.copysign 计算 x 和 y 元素对应位置的带符号值，并赋给 out
        out = torch.copysign(x, y)
        # 对 out 求和并反向传播梯度
        out.sum().backward()
        # 断言 x 的梯度为 [1.0, 0.0, -1.0]
        self.assertEqual(x.grad.tolist(), [1.0, 0.0, -1.0])
        # 断言 y 的梯度为 [0.0, 0.0, 0.0]
        self.assertEqual(y.grad.tolist(), [0.0] * 3)
    # 测试 true_divide 函数对两个张量进行元素级真实除法，并将结果存储在预先分配的张量 res 中
    def test_true_divide_out(self, device, dtype):
        # 创建两个张量 a1 和 a2，分别包含指定设备和数据类型的值
        a1 = torch.tensor([4.2, 6.2], dtype=dtype, device=device)
        a2 = torch.tensor([2.0, 2.0], dtype=dtype, device=device)
        # 创建一个和 a1 相同设备和数据类型的空张量 res
        res = torch.empty_like(a1)
        # 断言 true_divide 函数的结果与预期的张量值相等，指定允许的绝对误差 atol
        self.assertEqual(
            torch.true_divide(a1, a2, out=res),
            torch.tensor([2.1, 3.1], dtype=dtype, device=device),
            atol=0.01,
            rtol=0,
        )

    # 使用 torch.div 和 torch.mul 函数测试标量与张量的除法和乘法运算
    @dtypes(torch.half)
    def test_divmul_scalar(self, device, dtype):
        # 创建标量张量 x，并将其转换为指定设备和数据类型
        x = torch.tensor(100.0, device=device, dtype=dtype)
        # 创建 x 的浮点参考值 x_ref
        x_ref = x.float()
        scale = 1e5
        # 使用 torch.div 对 x 进行除法操作，并将结果与预期的浮点值进行比较
        res = x.div(scale)
        expected = x_ref.div(scale)
        self.assertEqual(res, expected.to(dtype), atol=0.0, rtol=0.0)
        # 创建另一个标量张量 x，并将其转换为指定设备和数据类型
        x = torch.tensor(1e-5, device=device, dtype=dtype)
        # 创建 x 的浮点参考值 x_ref
        x_ref = x.float()
        # 使用 torch.mul 对 x 进行乘法操作，并将结果与预期的浮点值进行比较
        res = x.mul(scale)
        expected = x_ref.mul(scale)
        self.assertEqual(res, expected.to(dtype), atol=0.0, rtol=0.0)
        # 使用乘法操作符 * 对 x 进行乘法操作，并将结果与预期的浮点值进行比较
        res = scale * x
        self.assertEqual(res, expected.to(dtype), atol=0.0, rtol=0.0)

    # 测试 floor_divide 操作符对张量进行元素级向下取整除法
    @dtypesIfCUDA(
        *set(get_all_math_dtypes("cuda")) - {torch.complex64, torch.complex128}
    )
    @dtypes(*set(get_all_math_dtypes("cpu")) - {torch.complex64, torch.complex128})
    def test_floor_divide_tensor(self, device, dtype):
        # 创建一个在指定设备上的随机张量 x，并将其转换为指定数据类型 dtype
        x = torch.randn(10, device=device).mul(30).to(dtype)
        # 创建一个包含从 1 到 10 的张量 y，数据类型为 dtype
        y = torch.arange(1, 11, dtype=dtype, device=device)

        # 使用 floor_divide 操作符对张量 x 和 y 进行元素级向下取整除法
        z = x // y
        # 使用双精度浮点数计算 x 和 y 的除法结果，并进行向下取整后转换为 dtype
        z_alt = torch.floor(x.double() / y.double()).to(dtype)

        # 断言 z 的数据类型与 x 相同，并且 z 的值与 z_alt 相等
        self.assertEqual(z.dtype, x.dtype)
        self.assertEqual(z, z_alt)

    # 测试 floor_divide 操作符对标量与张量进行元素级向下取整除法
    @dtypesIfCUDA(
        *set(get_all_math_dtypes("cuda")) - {torch.complex64, torch.complex128}
    )
    @dtypes(*set(get_all_math_dtypes("cpu")) - {torch.complex64, torch.complex128})
    def test_floor_divide_scalar(self, device, dtype):
        # 创建一个在指定设备上的随机张量 x，并将其转换为指定数据类型 dtype
        x = torch.randn(100, device=device).mul(10).to(dtype)

        # 使用 floor_divide 操作符对张量 x 和标量 3 进行元素级向下取整除法
        z = x // 3
        # 创建一个张量，其元素为 x 中每个元素除以 3 的向下取整结果，并转换为 x 的数据类型 dtype
        z_alt = torch.tensor(
            [math.floor(v.item() / 3.0) for v in x], dtype=x.dtype, device=device
        )

        # 断言 z 的数据类型与 x 相同，并且 z 的值与 z_alt 相等
        self.assertEqual(z.dtype, x.dtype)
        self.assertEqual(z, z_alt)

    # 测试 rdiv 操作符对张量进行元素级右除法（即，30 除以张量 x 的每个元素）
    @onlyCPU
    @dtypes(*get_all_math_dtypes("cpu"))
    def test_rdiv(self, device, dtype):
        # 若数据类型为 torch.float16，则直接返回，不进行后续操作
        if dtype is torch.float16:
            return
        # 若数据类型为复数，则创建一个张量 x，包含指定设备上的随机值并加 1，乘以 4
        elif dtype.is_complex:
            x = torch.rand(100, dtype=dtype, device=device).add(1).mul(4)
        # 否则，创建一个在指定设备上的随机张量 x，并加 1，乘以 4，并转换为指定数据类型 dtype
        else:
            x = torch.rand(100, device=device).add(1).mul(4).to(dtype)
        # 使用 rdiv 操作符对标量 30 和张量 x 进行元素级右除法
        y = 30 / x
        # 创建一个张量，其元素为 30 除以 x 中每个元素的结果，并与 x 的数据类型相同
        z = torch.tensor([30 / v.item() for v in x], device=device)
        # 断言 y 与 z 相等，允许浮点数比较时的数据类型精度不同
        self.assertEqual(y, z, exact_dtype=False)

    # 测试浮点类型和 torch.half 数据类型的所有组合
    @dtypes(*floating_types_and(torch.half))
    def test_fmod_remainder_by_zero_float(self, device, dtype):
        fn_list = (torch.fmod, torch.remainder)
        # 对每个函数 fn in fn_list 进行测试
        for fn in fn_list:
            # 创建一个大小为 (10, 10) 的张量 x，设备为指定 device，数据类型为 dtype，数值范围在 -9 到 9 之间
            x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
            zero = torch.zeros_like(x)
            # 断言：对于所有的 fn(x, 0.0)，结果应该是 NaN，无论是在 CPU 还是 GPU 上
            self.assertTrue(torch.all(fn(x, 0.0).isnan()))
            # 断言：对于所有的 fn(x, zero)，结果应该是 NaN，无论是在 CPU 还是 GPU 上
            self.assertTrue(torch.all(fn(x, zero).isnan()))

    @onlyNativeDeviceTypes  # 检查问题 https://github.com/pytorch/pytorch/issues/48130
    @dtypes(*integral_types())
    def test_fmod_remainder_by_zero_integral(self, device, dtype):
        fn_list = (torch.fmod, torch.remainder)
        # 对每个函数 fn in fn_list 进行测试
        for fn in fn_list:
            # 创建一个大小为 (10, 10) 的张量 x，设备为指定 device，数据类型为 dtype，数值范围在 -9 到 9 之间
            x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
            zero = torch.zeros_like(x)
            # 如果运行在 CPU 上
            if self.device_type == "cpu":
                # 断言：调用 fn(x, zero) 应该抛出 RuntimeError，并且错误信息中包含 "ZeroDivisionError"
                with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
                    fn(x, zero)
            elif torch.version.hip is not None:
                # ROCm 行为：x % 0 是一个无操作；应返回 x 自身
                self.assertEqual(fn(x, zero), x)
            else:
                # CUDA 行为：对于不同的数据类型，行为不确定
                # 对于 int64 类型，CUDA 返回所有位为 1 的模式，负数返回所有位为 1，正数返回一半位为 1
                # uint8: 0xff -> 255
                # int32: 0xffffffff -> -1
                if dtype == torch.int64:
                    # 断言：对于 int64 类型，fn(x, zero) 应该返回 4294967295（所有位为 1）或者 -1，具体取决于 x 的正负
                    self.assertEqual(fn(x, zero) == 4294967295, x >= 0)
                    self.assertEqual(fn(x, zero) == -1, x < 0)
                else:
                    value = 255 if dtype == torch.uint8 else -1
                    # 断言：对于其他整数类型，fn(x, zero) 应该返回固定值，分别是 255 或者 -1
                    self.assertTrue(torch.all(fn(x, zero) == value))

    @dtypes(*all_types_and(torch.half))
    # 定义一个测试函数，用于测试 torch.fmod 和 torch.remainder 函数的行为
    def test_fmod_remainder(self, device, dtype):
        # 作为参考，使用 numpy 实现的相关功能

        # 定义一个内部辅助函数，用于执行具体的测试
        def _helper(x, mod, fns_list):
            # 遍历传入的函数列表，分别测试每种函数的行为
            for fn, inplace_fn, ref_fn in fns_list:
                # 如果 x 是 torch 张量，则转换为 numpy 数组，否则保持不变
                np_x = x.cpu().numpy() if torch.is_tensor(x) else x
                # 如果 mod 是 torch 张量，则转换为 numpy 数组，否则保持不变
                np_mod = mod.cpu().numpy() if torch.is_tensor(mod) else mod
                # 使用 numpy 中对应的函数计算期望值
                exp = ref_fn(np_x, np_mod)
                # 将 numpy 数组转换为 torch 张量
                exp = torch.from_numpy(exp)
                # 调用 torch 中的函数计算结果
                res = fn(x, mod)

                # 断言计算结果与期望值相等，允许数据类型不完全一致
                self.assertEqual(res, exp, exact_dtype=False)

                # 如果 x 是 torch 张量
                if torch.is_tensor(x):
                    # 测试输出到指定张量 out 的结果是否正确
                    out = torch.empty(0, device=device, dtype=res.dtype)
                    fn(x, mod, out=out)
                    self.assertEqual(out, exp, exact_dtype=False)
                    # 断言 out 的形状为 [10, 10]
                    self.assertEqual(out.size(), torch.Size([10, 10]))
                    # 测试原地操作时的类型转换错误
                    try:
                        inplace_fn(x, mod)
                        self.assertEqual(x, exp, exact_dtype=False)
                    except RuntimeError as e:
                        # 断言捕获到的运行时错误信息匹配预期格式
                        self.assertRegex(
                            str(e),
                            "result type (Half|Float|Double) "
                            "can't be cast to the desired output "
                            "type (Byte|Char|Short|Int|Long)",
                        )

        # 生成一个大小为 (10, 10) 的测试张量 x，指定设备和数据类型
        x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # 生成一个与 x 具有相同数据类型的大小为 (10, 10) 的测试张量 mod
        mod = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
        # 将 mod 中的零元素替换为 1，确保不出现除以零的情况
        mod[mod == 0] = 1

        # 生成测试 mod 的列表，包括整数、浮点数、张量和非连续张量
        mods = [3, 2.3, mod, mod.t()]
        # 如果 dtype 是整数类型，则添加一个具有浮点数据类型的 mod
        if dtype in integral_types():
            mod_float = make_tensor(
                (10, 10), device=device, dtype=torch.float, low=-9, high=9
            )
            mod[mod == 0] = 1
            mods.append(mod_float)

        # 对 x 和 mods 中的每对元素执行测试
        for dividend, mod in product([x, x.t()], mods):
            _helper(
                dividend,
                mod,
                (
                    (torch.fmod, torch.Tensor.fmod_, np.fmod),
                    (torch.remainder, torch.Tensor.remainder_, np.remainder),
                ),
            )

        # 针对 torch.remainder(scalar, tensor) 的测试
        for dividend, mod in product([5, 3.14], mods):
            # 如果 mod 是张量，则执行测试
            if torch.is_tensor(mod):
                _helper(
                    dividend,
                    mod,
                    ((torch.remainder, torch.Tensor.remainder_, np.remainder),),
                )

    # 使用 @dtypes 装饰器指定测试数据类型为 torch.float 和 torch.double
    @dtypes(torch.float, torch.double)
    # 定义一个测试方法，用于测试 torch.remainder 和 torch.fmod 的行为，当被除数很大时
    def test_remainder_fmod_large_dividend(self, device, dtype):
        # 定义一个很大的被除数
        alarge = 1e9
        # 定义圆周率
        pi = 3.14159265358979
        # 遍历两种被除数和除数的组合：一个很大的被除数和圆周率的正负值
        for avalue in [alarge, -alarge]:
            for bvalue in [pi, -pi]:
                # 创建 tensor 变量 a，使用指定的设备和数据类型
                a = torch.tensor([avalue], dtype=dtype, device=device)
                # 创建 tensor 变量 b，使用指定的设备和数据类型
                b = torch.tensor([bvalue], dtype=dtype, device=device)
                # 计算 a 除以 b 的余数，使用 torch.remainder
                c = torch.remainder(a, b)
                # 计算 a 除以 b 的余数，使用 torch.fmod
                d = torch.fmod(a, b)
                # 断言：余数 c 的符号与除数 b 的符号相同
                self.assertTrue(
                    (b[0] > 0) == (c[0] > 0)
                )  # remainder has same sign as divisor
                # 断言：fmod d 的符号与被除数 a 的符号相同
                self.assertTrue(
                    (a[0] > 0) == (d[0] > 0)
                )  # fmod has same sign as dividend
                # 断言：余数 c 的绝对值小于除数 b 的绝对值
                self.assertTrue(
                    abs(c[0]) < abs(b[0])
                )  # remainder is within range of divisor
                # 断言：fmod d 的绝对值小于除数 b 的绝对值
                self.assertTrue(
                    abs(d[0]) < abs(b[0])
                )  # fmod is within range of divisor
                # 如果被除数 a 和除数 b 的符号相同，则断言余数 c 等于 fmod d
                if (a[0] > 0) == (b[0] > 0):
                    self.assertTrue(c[0] == d[0])  # remainder is same as fmod
                # 否则，断言余数 c 与 fmod d 的绝对值之差等于除数 b 的绝对值
                else:
                    self.assertTrue(
                        abs(c[0] - d[0]) == abs(b[0])
                    )  # differ by one divisor

    # 根据设备和数据类型设置适用的测试用例，测试 torch.hypot 函数
    @dtypesIfCPU(torch.bfloat16, torch.half, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    def test_hypot(self, device, dtype):
        # 定义多组输入
        inputs = [
            (
                torch.randn(10, device=device).to(dtype),
                torch.randn(10, device=device).to(dtype),
            ),
            (
                torch.randn((3, 3, 3), device=device).to(dtype),
                torch.randn((3, 3, 3), device=device).to(dtype),
            ),
            (
                torch.randn((10, 1), device=device).to(dtype),
                torch.randn((10, 1), device=device).to(dtype).transpose(0, 1),
            ),
            (
                torch.randint(100, (10,), device=device, dtype=torch.long),
                torch.randn(10, device=device).to(dtype),
            ),
        ]
        # 遍历输入，执行测试
        for input in inputs:
            # 计算 torch.hypot 的实际输出
            actual = torch.hypot(input[0], input[1])
            # 根据数据类型选择不同的预期输出
            if dtype in [torch.bfloat16, torch.half]:
                expected = torch.sqrt(input[0] * input[0] + input[1] * input[1])
            else:
                expected = np.hypot(input[0].cpu().numpy(), input[1].cpu().numpy())
            # 断言实际输出等于预期输出
            self.assertEqual(actual, expected, exact_dtype=False)

    # 限制仅在原生设备类型上执行测试
    @onlyNativeDeviceTypes
    # 定义适用的数据类型进行测试：torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    # 定义一个测试函数，用于测试 torch.gcd 方法
    def test_gcd(self, device, dtype):
        # Tests gcd(0, 0), gcd(0, a) cases
        # 创建包含三个元素的张量 t1 和 t2，分别包含 [0, 10, 0] 和 [0, 0, 10]
        t1 = torch.tensor([0, 10, 0], dtype=dtype, device=device)
        t2 = torch.tensor([0, 0, 10], dtype=dtype, device=device)
        # 计算张量 t1 和 t2 的最大公约数
        actual = torch.gcd(t1, t2)
        # 使用 NumPy 计算 [0, 10, 0] 和 [0, 0, 10] 的最大公约数
        expected = np.gcd([0, 10, 0], [0, 0, 10])
        # 断言实际计算结果与预期结果相等
        self.assertEqual(actual, expected, exact_dtype=False)

        if dtype == torch.uint8:
            # Test unsigned integers with potential sign issues (i.e., uint8 with value >= 128)
            # 当数据类型为 uint8 时，测试无符号整数可能的符号问题（即值大于等于 128 的 uint8）
            a = torch.tensor([190, 210], device=device, dtype=dtype)
            b = torch.tensor([190, 220], device=device, dtype=dtype)
            # 计算张量 a 和 b 的最大公约数
            actual = torch.gcd(a, b)
            # 预期的最大公约数结果为 [190, 10]
            expected = torch.tensor([190, 10], device=device, dtype=dtype)
            # 断言实际计算结果与预期结果相等
            self.assertEqual(actual, expected)
        else:
            # Compares with NumPy
            # 使用 NumPy 进行比较
            # 创建两个长度为 1024 的张量 a 和 b，数据类型为 dtype
            a = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
            b = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
            # 计算张量 a 和 b 的最大公约数
            actual = torch.gcd(a, b)
            # 使用 NumPy 计算张量 a 和 b 的最大公约数
            expected = np.gcd(a.cpu().numpy(), b.cpu().numpy())
            # 断言实际计算结果与预期结果相等
            self.assertEqual(actual, expected)

    @onlyNativeDeviceTypes
    @dtypes(torch.int16, torch.int32, torch.int64)
    # 定义一个测试函数，用于测试 torch.lcm 方法
    def test_lcm(self, device, dtype):
        # Tests lcm(0, 0), lcm(0, a) cases
        # 创建包含三个元素的张量 t1 和 t2，分别包含 [0, 10, 0] 和 [0, 0, 10]
        t1 = torch.tensor([0, 10, 0], dtype=dtype, device=device)
        t2 = torch.tensor([0, 0, 10], dtype=dtype, device=device)
        # 计算张量 t1 和 t2 的最小公倍数
        actual = torch.lcm(t1, t2)
        # 使用 NumPy 计算 [0, 10, 0] 和 [0, 0, 10] 的最小公倍数
        expected = np.lcm([0, 10, 0], [0, 0, 10])
        # 断言实际计算结果与预期结果相等
        self.assertEqual(actual, expected, exact_dtype=False)

        # Compares with NumPy
        # 使用 NumPy 进行比较
        # 创建两个长度为 1024 的张量 a 和 b，数据类型为 dtype
        a = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
        b = torch.randint(-20, 20, (1024,), device=device, dtype=dtype)
        # 计算张量 a 和 b 的最小公倍数
        actual = torch.lcm(a, b)
        # 使用 NumPy 计算张量 a 和 b 的最小公倍数
        expected = np.lcm(a.cpu().numpy(), b.cpu().numpy())
        # 断言实际计算结果与预期结果相等
        self.assertEqual(actual, expected, exact_dtype=False)

    @onlyNativeDeviceTypes
    @dtypesIfCPU(torch.float32, torch.float64, torch.float16)
    @dtypes(torch.float32, torch.float64)
    # 定义一个测试函数，用于测试 torch.nextafter 函数在不同情况下的行为
    def test_nextafter(self, device, dtype):
        # 测试特殊情况：当其中一个参数为零，另一个为正无穷、负无穷或普通数值时
        t1 = torch.tensor([0, 0, 10], device=device, dtype=dtype)
        t2 = torch.tensor([inf, -inf, 10], device=device, dtype=dtype)
        # 调用 torch.nextafter 函数计算实际结果
        actual = torch.nextafter(t1, t2)
        # 使用 NumPy 的 nextafter 函数计算期望结果
        expected = np.nextafter(t1.cpu().numpy(), t2.cpu().numpy())
        # 断言实际结果与期望结果相等，允许的绝对误差和相对误差均为零
        self.assertEqual(actual, expected, atol=0, rtol=0)

        # 再次测试不同情况下的 torch.nextafter 行为
        actual = torch.nextafter(t2, t1)
        expected = np.nextafter(t2.cpu().numpy(), t1.cpu().numpy())
        self.assertEqual(actual, expected, atol=0, rtol=0)

        # 测试特殊情况：当其中一个参数为零，另一个参数为 NaN
        t1 = torch.tensor([0, nan], device=device, dtype=dtype)
        t2 = torch.tensor([nan, 0], device=device, dtype=dtype)
        # 调用 torch.nextafter 函数，判断结果是否全部为 NaN
        self.assertTrue(torch.nextafter(t1, t2).isnan().all())

        # 生成随机数数组，测试 torch.nextafter 在普通情况下的表现
        a = torch.randn(100, device=device, dtype=dtype)
        b = torch.randn(100, device=device, dtype=dtype)
        actual = torch.nextafter(a, b)
        expected = np.nextafter(a.cpu().numpy(), b.cpu().numpy())
        self.assertEqual(actual, expected, atol=0, rtol=0)

    # 使用装饰器，指定只在本地设备类型上运行的测试
    @onlyNativeDeviceTypes
    # 使用指定数据类型 bfloat16 运行的测试函数
    @dtypes(torch.bfloat16)
    def test_nextafter_bfloat16(self, device, dtype):
        # 定义特殊数值 nan 和 inf
        nan = float("nan")
        inf = float("inf")
        # 定义测试案例，包括各种输入和期望的输出
        cases = (
            # (from, to, expected)
            (0, 1, 9.183549615799121e-41),
            (0, -1, -9.183549615799121e-41),
            (1, -2, 0.99609375),
            (1, 0, 0.99609375),
            (1, 2, 1.0078125),
            (-1, -2, -1.0078125),
            (-1, 0, -0.99609375),
            (2, -1, 1.9921875),
            (2, 1, 1.9921875),
            (20, 3000, 20.125),
            (20, -3000, 19.875),
            (3000, -20, 2992.0),
            (-3000, 20, -2992.0),
            (65536, 0, 65280.0),
            (65536, inf, 66048.0),
            (-65536, 0, -65280.0),
            (-65536, -inf, -66048.0),
            (nan, 0, nan),
            (0, nan, nan),
            (nan, nan, nan),
            (nan, inf, nan),
            (inf, nan, nan),
            (inf, -inf, 3.3895313892515355e38),
            (-inf, inf, -3.3895313892515355e38),
            (inf, 0, 3.3895313892515355e38),
            (0, inf, 9.183549615799121e-41),
            (-inf, 0, -3.3895313892515355e38),
            (0, -inf, -9.183549615799121e-41),
        )

        # 遍历测试案例，逐一执行测试
        for from_v, to_v, expected in cases:
            from_t = torch.tensor([from_v], device=device, dtype=dtype)
            to_t = torch.tensor([to_v], device=device, dtype=dtype)
            actual = torch.nextafter(from_t, to_t).item()
            # 断言实际结果与期望结果相等，允许的绝对误差和相对误差均为零
            self.assertEqual(actual, expected, atol=0, rtol=0)
    # 定义一个内部函数_test_cop，用于测试给定的 torchfn 和 mathfn 函数对于指定的 dtype 和 device 的操作
    def _test_cop(self, torchfn, mathfn, dtype, device):
        # 定义一个参考实现函数 reference_implementation，用于计算结果 res2
        def reference_implementation(res2):
            # 使用 iter_indices 函数迭代 sm1 的索引 i, j
            for i, j in iter_indices(sm1):
                # 计算 sm1 和 sm2 中的元素索引 idx1d
                idx1d = i * sm1.size(0) + j
                # 使用 mathfn 函数对 sm1[i, j] 和 sm2[idx1d] 进行操作，并将结果存入 res2[i, j]
                res2[i, j] = mathfn(sm1[i, j], sm2[idx1d])
            # 返回计算结果 res2
            return res2

        # 创建一个随机的 10x10x10 的张量 m1，并设定其数据类型和设备
        m1 = torch.randn(10, 10, 10, dtype=dtype, device=device)
        # 创建一个随机的 10x100 的张量 m2，并设定其数据类型和设备
        m2 = torch.randn(10, 10 * 10, dtype=dtype, device=device)
        # 选择 m1 的第 4 行作为 sm1
        sm1 = m1[4]
        # 选择 m2 的第 4 行作为 sm2
        sm2 = m2[4]

        # 使用 torchfn 函数对 sm1 和 sm2.view(10, 10) 进行操作，计算结果存入 res1
        res1 = torchfn(sm1, sm2.view(10, 10))
        # 使用 reference_implementation 函数计算预期结果 res2，并进行克隆
        res2 = reference_implementation(res1.clone())
        # 断言 res1 和 res2 相等
        self.assertEqual(res1, res2)

        # 创建一个随机的 10x10x10 的张量 m1，并设定其数据类型和设备
        m1 = torch.randn(10, 10, 10, dtype=dtype, device=device)
        # 创建一个随机的 100x100 的张量 m2，并设定其数据类型和设备
        m2 = torch.randn(10 * 10, 10 * 10, dtype=dtype, device=device)
        # 选择 m1 的所有行的第 4 列作为 sm1
        sm1 = m1[:, 4]
        # 选择 m2 的所有行的第 4 列作为 sm2
        sm2 = m2[:, 4]
        
        # 将 sm2 视图设置为与 sm1 相同的大小和步幅
        sm2.set_(
            sm2.storage(),
            sm2.storage_offset(),
            sm1.size(),
            (sm2.stride()[0] * 10, sm2.stride()[0]),
        )
        # 使用 torchfn 函数对 sm1 和 sm2 进行操作，计算结果存入 res1
        res1 = torchfn(sm1, sm2)
        # 将 sm2 视图设置为与 m2 的第 4 列相同的大小和步幅，用于 reference_implementation 函数
        sm2.set_(
            sm2.storage(), sm2.storage_offset(), m2[:, 4].size(), m2[:, 4].stride()
        )
        # 使用 reference_implementation 函数计算预期结果 res2，并进行克隆
        res2 = reference_implementation(res1.clone())
        # 断言 res1 和 res2 相等
        self.assertEqual(res1, res2)

    # 使用 onlyCPU 和 dtypes 修饰器，测试 torch.div 函数在指定设备和数据类型上的操作
    @onlyCPU
    @dtypes(torch.float)
    def test_cdiv(self, device, dtype):
        # 调用 _test_cop 函数，测试 torch.div 函数在指定设备和数据类型上的操作
        self._test_cop(torch.div, operator.truediv, dtype, device)

    # 使用 onlyCPU 和 dtypes 修饰器，测试 torch.remainder 函数在指定设备和数据类型上的操作
    @onlyCPU
    @dtypes(torch.float)
    def test_cremainder(self, device, dtype):
        # 调用 _test_cop 函数，测试 torch.remainder 函数在指定设备和数据类型上的操作
        self._test_cop(torch.remainder, operator.mod, dtype, device)

    # 使用 onlyCPU 和 dtypes 修饰器，测试 torch.mul 函数在指定设备和数据类型上的操作
    @onlyCPU
    @dtypes(torch.float)
    def test_cmul(self, device, dtype):
        # 调用 _test_cop 函数，测试 torch.mul 函数在指定设备和数据类型上的操作
        self._test_cop(torch.mul, operator.mul, dtype, device)

    # 使用 onlyCPU 和 dtypes 修饰器，测试 torch.pow 函数在指定设备和数据类型上的操作
    @onlyCPU
    @dtypes(torch.float)
    def test_cpow(self, device, dtype):
        # 调用 _test_cop 函数，测试 torch.pow 函数在指定设备和数据类型上的操作
        self._test_cop(
            torch.pow, lambda x, y: nan if x < 0 else math.pow(x, y), dtype, device
        )

    # 使用 dtypes 修饰器，测试 floor_divide_zero 函数在指定设备和数据类型上的操作
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_floor_divide_zero(self, device, dtype):
        # 创建一个包含 0 和 1 的张量 a，设定其数据类型和设备
        a = torch.tensor([0, 1], dtype=dtype, device=device)
        # 创建一个包含 0 和 1 的张量 b，设定其数据类型和设备
        b = torch.tensor([0, 1], dtype=dtype, device=device)
        # 使用 assertRaisesRegex 检查 RuntimeError 中是否包含 "ZeroDivisionError" 的异常
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError"):
            # 使用 assertWarnsOnceRegex 检查 UserWarning 中是否包含 "floor_divide" 的警告，执行 a // b 操作
            with self.assertWarnsOnceRegex(UserWarning, "floor_divide"):
                a // b

    # 使用 dtypes 修饰器，测试 muldiv_scalar 函数在指定设备和数据类型上的操作
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_muldiv_scalar(self, device, dtype):
        # 创建一个形状为 (10, 3) 的随机张量 x，设定其数据类型和设备
        x = make_tensor((10, 3), dtype=dtype, device=device, low=None, high=None)
        # 创建一个形状为 (1,) 的随机张量 s，设定其数据类型为 dtype=torch.float，设备为 "cpu"，并获取其数值
        s = make_tensor((1,), dtype=dtype, device="cpu", low=None, high=None).item()
        # 使用 torch.full_like 创建一个与 x 形状相同，值为 s 的张量 y
        y = torch.full_like(x, s)
        # 断言 x 乘以 s 的结果与 x 乘以 y 的结果相等
        self.assertEqual(x * s, x * y)
        # 断言 s 乘以 x 的结果与 y 乘以 x 的结果相等
        self.assertEqual(s * x, y * x)
        # 断言 x 除以 s 的结果与 x 除以 y 的结果相等
        self.assertEqual(x / s, x / y)
        # 断言 s 除以 x 的结果与 y 除以 x 的结果相等
        self.assertEqual(s / x, y / x)

    # TODO: update make_tensor to support extremal additions and remove this in favor of make_tensor
    # 定义一个方法用于生成输入数据张量，根据给定的形状、数据类型、设备和是否包含极端值来生成不同类型的张量
    def _generate_input(self, shape, dtype, device, with_extremal):
        # 如果形状为 ()，创建一个空张量
        if shape == ():
            x = torch.tensor((), dtype=dtype, device=device)
        else:
            # 如果数据类型是浮点数或复数
            if dtype.is_floating_point or dtype.is_complex:
                # 对于 bfloat16 类型，使用 torch.randn 的变通方法生成张量，并将其转换为 bfloat16 类型
                if dtype == torch.bfloat16:
                    x = torch.randn(*shape, device=device) * random.randint(30, 100)
                    x = x.to(torch.bfloat16)
                else:
                    # 使用 torch.randn 生成指定形状的张量，并乘以一个随机整数
                    x = torch.randn(
                        *shape, dtype=dtype, device=device
                    ) * random.randint(30, 100)
                # 随机将张量中大于 0.5 的元素置为 0
                x[torch.randn(*shape) > 0.5] = 0
                # 如果需要生成极端值，并且数据类型是浮点数
                if with_extremal and dtype.is_floating_point:
                    # 使用极端值来替换张量中随机选取的元素
                    x[torch.randn(*shape) > 0.5] = float("nan")
                    x[torch.randn(*shape) > 0.5] = float("inf")
                    x[torch.randn(*shape) > 0.5] = float("-inf")
                # 如果需要生成极端值，并且数据类型是复数
                elif with_extremal and dtype.is_complex:
                    x[torch.randn(*shape) > 0.5] = complex("nan")
                    x[torch.randn(*shape) > 0.5] = complex("inf")
                    x[torch.randn(*shape) > 0.5] = complex("-inf")
            # 如果数据类型是布尔类型
            elif dtype == torch.bool:
                # 创建一个形状为 shape 的零张量，并随机将部分元素置为 True
                x = torch.zeros(shape, dtype=dtype, device=device)
                x[torch.randn(*shape) > 0.5] = True
            else:
                # 对于其它数据类型，使用 torch.randint 生成指定范围内的随机整数张量
                x = torch.randint(15, 100, shape, dtype=dtype, device=device)

        # 返回生成的张量 x
        return x

    @dtypes(
        *tuple(
            itertools.combinations_with_replacement(
                # 生成所有类型及其组合，并包括 torch.half, torch.bfloat16, torch.bool
                all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), 2
            )
        )
    )
    def test_comparison_ops_type_promotion_and_broadcasting(self, device, dtypes):
        # issue #42660
        # testing all combinations of broadcasting and type promotion
        # with a range of dtypes and input shapes, and with extremal values
        
        # 定义一个比较函数，用来比较 torch_fn 和 np_fn 的结果
        def compare_with_numpy_bin_op(torch_fn, np_fn, x, y, out=None):
            # 解决 numpy 不支持 bfloat16 的问题
            # 将 torch.bfloat16 类型的张量转换为 torch.float32 类型
            x_np = x if x.dtype != torch.bfloat16 else x.to(torch.float32)
            # 将 y 转换为 numpy 数组，因为 numpy 不支持 torch.bfloat16
            y_np = (
                y.cpu().numpy()
                if y.dtype != torch.bfloat16
                else y.to(torch.float32).cpu().numpy()
            )
            # 使用 self.compare_with_numpy 方法比较 torch_fn 和 np_fn 的结果
            self.compare_with_numpy(
                lambda inp: torch_fn(inp, y, out=out) if out else torch_fn(inp, y),
                lambda inp: np_fn(inp, y_np, out=out) if out else np_fn(inp, y_np),
                x_np,
            )

        # 不支持复数操作的函数列表
        complex_op_denylist = [
            torch.lt,
            torch.le,
            torch.gt,
            torch.ge,
        ]  # complex not supported
        
        # 输入数据的大小列表，用来进行广播操作
        input_sizes = [(1,), (10,), (10, 1), (1, 10), (4, 10), (64, 10), (12, 3)]
        
        # 操作对列表，包含了 torch 和 numpy 中对应的操作函数
        op_pairs = [
            (torch.lt, np.less),
            (torch.le, np.less_equal),
            (torch.gt, np.greater),
            (torch.ge, np.greater_equal),
            (torch.eq, np.equal),
            (torch.ne, np.not_equal),
            (torch.logical_and, np.logical_and),
            (torch.logical_or, np.logical_or),
            (torch.logical_xor, np.logical_xor),
        ]

        # 遍历所有输入大小
        for size1 in input_sizes:
            size2 = (2,) + size1  # 执行广播操作
            # 对是否包含极端值进行循环测试
            for with_extremal in [False, True]:
                # 生成输入数据 a 和 b
                a = self._generate_input(size1, dtypes[0], device, with_extremal)
                b = self._generate_input(size2, dtypes[1], device, with_extremal)
                # 遍历操作对
                for torch_op, numpy_op in op_pairs:
                    # 如果数据类型为复数且操作在复数不支持列表中，则跳过
                    if (
                        dtypes[0].is_complex or dtypes[1].is_complex
                    ) and torch_op in complex_op_denylist:
                        continue
                    # 使用 functional 版本的操作函数进行比较
                    compare_with_numpy_bin_op(torch_op, numpy_op, a, b)

                    # functional 比较操作始终返回布尔类型的张量
                    self.assertEqual(torch_op(a, b).dtype, torch.bool)

                    # 使用 out 版本的操作函数进行比较
                    out = torch.zeros(
                        1, dtype=torch.complex128
                    )  # 所有转换为 complex128 的转换都是安全的
                    compare_with_numpy_bin_op(torch_op, numpy_op, a, b, out=out)
    def test_shift_limits(self, device, dtype):
        "Ensure that integer bit shifting works as expected with out-of-limits shift values."
        # 定义测试函数，验证整数位移在超出限制的情况下的表现
        iinfo = torch.iinfo(dtype)
        bits = iinfo.bits
        low = iinfo.min
        high = iinfo.max
        exact_dtype = (
            dtype != torch.uint8
        )  # 标记精确的数据类型，因为对于超出限制的位移值，numpy 会将 uint8 类型更改为 int16
        for input in (
            torch.tensor(
                [-1, 0, 1], device=device, dtype=dtype
            ),  # 小规模非矢量操作
            torch.tensor(
                [low, high], device=device, dtype=dtype
            ),  # 小规模非矢量操作
            make_tensor(
                (64, 64, 64), low=low, high=high, device=device, dtype=dtype
            ),  # 大规模矢量操作
        ):
            shift_left_expected = torch.zeros_like(input)
            shift_right_expected = torch.clamp(input, -1, 0)
            # 遍历超出限制的位移范围
            for shift in chain(range(-100, -1), range(bits, 100)):
                shift_left = input << shift
                self.assertEqual(shift_left, shift_left_expected, msg=f"<< {shift}")
                self.compare_with_numpy(
                    lambda x: x << shift,
                    lambda x: np.left_shift(x, shift),
                    input,
                    exact_dtype=exact_dtype,
                    msg=f"<< {shift}",
                )
                shift_right = input >> shift
                self.assertEqual(shift_right, shift_right_expected, msg=f">> {shift}")
                self.compare_with_numpy(
                    lambda x: x >> shift,
                    lambda x: np.right_shift(x, shift),
                    input,
                    exact_dtype=exact_dtype,
                    msg=f">> {shift}",
                )
    # 测试 Heaviside 函数在给定设备和数据类型下的行为
    def test_heaviside(self, device, dtypes):
        # 提取输入和值的数据类型
        input_dtype = dtypes[0]
        values_dtype = dtypes[1]

        # 使用 NumPy 随机数生成器创建输入数组，类型与指定数据类型对应
        rng = np.random.default_rng()
        input = np.array(
            rng.integers(-10, 10, size=10),
            dtype=torch_to_numpy_dtype_dict[
                input_dtype if (input_dtype != torch.bfloat16) else torch.float64
            ],
        )
        # 将数组中指定位置置为零
        input[0] = input[3] = input[7] = 0

        # 使用 NumPy 随机数生成器创建值数组，类型与指定数据类型对应
        values = np.array(
            rng.integers(-10, 10, size=10),
            dtype=torch_to_numpy_dtype_dict[
                values_dtype if (values_dtype != torch.bfloat16) else torch.float64
            ],
        )

        # 使用 NumPy 计算 Heaviside 函数结果
        np_result = torch.from_numpy(np.heaviside(input, values)).to(
            device=device, dtype=input_dtype
        )

        # 将输入和值转换为 PyTorch 张量，并指定设备和数据类型
        input = torch.from_numpy(input).to(device=device, dtype=input_dtype)
        values = torch.from_numpy(values).to(device=device, dtype=values_dtype)
        # 创建与输入相同类型和设备的空张量
        out = torch.empty_like(input)

        # 根据输入和值的数据类型进行不同的测试
        if input_dtype == values_dtype:
            # 调用 PyTorch 的 Heaviside 函数，并验证结果与 NumPy 的结果一致
            torch_result = torch.heaviside(input, values)
            self.assertEqual(np_result, torch_result)

            # 调用张量的实例方法 Heaviside，并验证结果与 NumPy 的结果一致
            torch_result = input.heaviside(values)
            self.assertEqual(np_result, torch_result)

            # 在指定的输出张量中计算 Heaviside 函数，并验证结果与 NumPy 的结果一致
            torch.heaviside(input, values, out=out)
            self.assertEqual(np_result, out)

            # 就地修改输入张量，应用 Heaviside 函数，并验证结果与 NumPy 的结果一致
            input.heaviside_(values)
            self.assertEqual(np_result, input)
        else:
            # 当输入和值的数据类型不同时，检查是否引发 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                torch.heaviside(input, values)
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                input.heaviside(values)
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                torch.heaviside(input, values, out=out)
            with self.assertRaisesRegex(
                RuntimeError,
                "heaviside is not yet implemented for tensors with different dtypes.",
            ):
                input.heaviside_(values)

    @onlyCUDA
    # 测试函数，用于测试 torch.heaviside() 跨设备操作
    def test_heaviside_cross_device(self, device):
        # 创建张量 x，指定设备，包含元素：[-9, 5, 0, 6, -2, 2]
        x = torch.tensor([-9, 5, 0, 6, -2, 2], device=device)
        # 创建标量张量 y，设备与 x 相同
        y = torch.tensor(0)
        # 对 x 进行 Heaviside 函数操作，阈值为 y，结果为 result
        result = torch.heaviside(x, y)
        # 期望的结果张量 expect：[0, 1, 0, 1, 0, 1]
        expect = torch.tensor([0, 1, 0, 1, 0, 1], device=device)
        # 断言结果张量与期望张量相等
        self.assertEqual(result, expect)

        # 再次调用 Heaviside 函数，交换参数顺序
        result = torch.heaviside(y, x)
        # 期望的结果张量 expect 与 x 保持一致：[-9, 5, 0, 6, -2, 2]
        expect = torch.tensor([-9, 5, 0, 6, -2, 2], device=device)
        # 断言结果张量与期望张量相等
        self.assertEqual(result, expect)

        # 创建张量 x，未指定设备，包含元素：[-9, 5, 0, 6, -2, 2]
        x = torch.tensor([-9, 5, 0, 6, -2, 2])
        # 创建标量张量 y，设备与之前指定的 device 相同
        y = torch.tensor(0, device=device)
        # 使用 with 语句断言运行时错误，预期所有张量在同一设备上
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            # 跨设备调用 Heaviside 函数
            torch.heaviside(x, y)

        # 再次使用 with 语句断言运行时错误，预期所有张量在同一设备上
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            # 跨设备调用 Heaviside 函数，交换参数顺序
            torch.heaviside(y, x)

    # 测试函数，用于测试复数类型的 Heaviside 函数
    @dtypes(*list(product(complex_types(), complex_types())))
    def test_heaviside_complex(self, device, dtypes):
        # 选择输入数据类型和值数据类型
        input_dtype = dtypes[0]
        values_dtype = dtypes[1]

        # 准备复数数据
        data = (complex(0, -6), complex(-1, 3), complex(1, 1))
        # 创建输入张量 input，指定设备和数据类型
        input = torch.tensor(data, device=device, dtype=input_dtype)
        # 创建值张量 values，指定设备和数据类型
        values = torch.tensor(data, device=device, dtype=values_dtype)
        # 创建与 input 相同类型和设备的输出张量 out
        out = torch.empty_like(input)
        # 获取输入张量的实部 real
        real = input.real

        # 使用 with 语句断言运行时错误，预期复数张量不能执行 Heaviside 函数
        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            # 调用 Heaviside 函数
            torch.heaviside(input, real)
        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            # 调用实部的 Heaviside 函数
            real.heaviside(values)
        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            # 在原地调用 Heaviside 函数
            input.heaviside_(values)
        with self.assertRaisesRegex(
            RuntimeError, "heaviside is not yet implemented for complex tensors."
        ):
            # 调用 Heaviside 函数，指定输出张量 out
            torch.heaviside(real, real, out=out)

    # 测试函数，用于测试逻辑运算操作
    def _test_logical(self, device, dtypes, op, a_, b_, expected_res_):
        # 创建预期结果张量 expected_res，指定数据类型和设备
        expected_res = torch.tensor(expected_res_, dtype=dtypes[0], device=device)
        # 创建张量 a，指定数据类型和设备
        a = torch.tensor(a_, dtype=dtypes[0], device=device)
        # 创建张量 b，指定数据类型和设备
        b = torch.tensor(b_, dtype=dtypes[1], device=device)

        # 测试新创建的张量
        self.assertEqual(expected_res.bool(), getattr(a, op)(b))
        # 测试指定输出张量
        c = torch.empty(0, dtype=torch.bool, device=device)
        getattr(torch, op)(a, b, out=c)
        self.assertEqual(expected_res.bool(), c)

        # 在张量 a 上调用指定的逻辑操作，并断言结果与期望结果相等
        getattr(a, op + "_")(b)
        self.assertEqual(expected_res, a)

    # 测试函数，用于测试逻辑异或操作
    @dtypes(
        *product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_logical_xor(self, device, dtypes):
        # 调用 _test_logical 函数测试逻辑异或操作
        self._test_logical(
            device, dtypes, "logical_xor", [10, 0, 1, 0], [1, 0, 0, 10], [0, 0, 1, 1]
        )
    @dtypes(
        *product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    # 使用装饰器定义测试函数，测试逻辑与运算
    def test_logical_and(self, device, dtypes):
        # 调用内部方法 _test_logical 进行逻辑与运算的测试
        self._test_logical(
            device, dtypes, "logical_and", [10, 0, 1, 0], [1, 0, 0, 10], [1, 0, 0, 0]
        )

    @dtypes(
        *product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    # 使用装饰器定义测试函数，测试逻辑或运算
    def test_logical_or(self, device, dtypes):
        # 调用内部方法 _test_logical 进行逻辑或运算的测试
        self._test_logical(
            device, dtypes, "logical_or", [10, 0, 1, 0], [1, 0, 0, 10], [1, 0, 1, 1]
        )

    # 测试取余运算是否溢出
    def test_remainder_overflow(self, device):
        # 创建整数张量 x，设备为给定的设备
        x = torch.tensor(23500, dtype=torch.int64, device=device)
        q = 392486996410368
        # 验证 x % q 等于 x
        self.assertEqual(x % q, x)
        # 验证 -x % q 等于 q - x
        self.assertEqual(-x % q, q - x)
        # 验证 x % -q 等于 x - q
        self.assertEqual(x % -q, x - q)
        # 验证 -x % -q 等于 -x
        self.assertEqual(-x % -q, -x)

    # 测试幂运算
    def test_rpow(self, device):
        # 创建大小为 10x10 的随机张量 m，设备为给定的设备
        m = torch.randn(10, 10, device=device)
        # 验证 torch.pow(2, m) 等于 2 的 m 次方
        self.assertEqual(torch.pow(2, m), 2**m)

        # 使用标量进行测试
        m = torch.randn(1, device=device).squeeze()
        assert m.dim() == 0, "m is intentionally a scalar"
        # 验证 torch.pow(2, m) 等于 2 的 m 次方
        self.assertEqual(torch.pow(2, m), 2**m)

    # 使用 onlyCPU 装饰器定义测试函数，测试 ldexp 运算
    @onlyCPU
    def test_ldexp(self, device):
        # 创建大小为 64 的随机张量 mantissas，设备为给定的设备
        mantissas = torch.randn(64, device=device)
        # 创建大小为 64 的随机整数张量 exponents，范围为 -31 到 31
        exponents = torch.randint(-31, 31, (64,), device=device, dtype=torch.int32)

        # 基本测试
        np_outcome = np.ldexp(mantissas.numpy(), exponents.numpy())
        pt_outcome_1 = torch.ldexp(mantissas, exponents)
        pt_outcome_2 = mantissas.ldexp(exponents)
        # 验证 torch.ldexp 和 .ldexp 方法的结果与 numpy 中的 ldexp 函数结果相等
        self.assertEqual(np_outcome, pt_outcome_1)
        self.assertEqual(np_outcome, pt_outcome_2)
        mantissas.ldexp_(exponents)
        # 验证 .ldexp_ 方法的结果与 numpy 中的 ldexp 函数结果相等
        self.assertEqual(np_outcome, mantissas)

        # 测试边界情况
        mantissas = torch.tensor(
            [float("inf"), float("-inf"), float("inf"), float("nan")], device=device
        )
        exponents = torch.randint(0, 31, (4,), device=device, dtype=torch.int32)
        np_outcome = np.ldexp(mantissas.numpy(), exponents.numpy())
        pt_outcome = torch.ldexp(mantissas, exponents)
        # 验证边界情况下 torch.ldexp 的结果与 numpy 中的 ldexp 函数结果相等
        self.assertEqual(np_outcome, pt_outcome)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 使用装饰器定义测试函数，设置支持的数据类型为 float、double、复数 float 和复数 double
    # 定义测试函数 test_lerp，用于测试 torch.lerp 函数在不同设备和数据类型下的行为
    def test_lerp(self, device, dtype):
        # 定义起始点、终点和权重的形状组合
        start_end_weight_shapes = [(), (5,), (5, 5)]
        # 遍历所有形状组合的笛卡尔积
        for shapes in product(
            start_end_weight_shapes, start_end_weight_shapes, start_end_weight_shapes
        ):
            # 随机生成起始点张量，并将其移动到指定设备上，并指定数据类型
            start = torch.randn(shapes[0], device=device, dtype=dtype)
            # 随机生成终点张量，并将其移动到指定设备上，并指定数据类型
            end = torch.randn(shapes[1], device=device, dtype=dtype)

            # 定义权重张量列表
            weights = [
                torch.randn(shapes[2], device=device, dtype=dtype),
                random.random(),
            ]
            # 如果数据类型是复数类型，则添加复数权重
            if dtype.is_complex:
                weights += [complex(0, 1), complex(0.4, 1.2)]

            # 遍历权重列表中的每个权重值
            for weight in weights:
                # 使用 torch.lerp 函数计算加权线性插值的实际结果
                actual = torch.lerp(start, end, weight)
                # 使用张量对象方法 start.lerp(end, weight) 计算加权线性插值的实际结果
                actual_method = start.lerp(end, weight)
                # 断言两种方式计算得到的结果应相等
                self.assertEqual(actual, actual_method)
                # 创建一个指定值为 1.0 的张量 actual_out，并指定设备和数据类型
                actual_out = torch.tensor(1.0, dtype=dtype, device=device)
                # 使用 torch.lerp 函数计算加权线性插值，并将结果存入 actual_out
                torch.lerp(start, end, weight, out=actual_out)
                # 断言 torch.lerp 计算结果与 actual_out 中的值相等
                self.assertEqual(actual, actual_out)
                # 计算预期的加权线性插值结果
                expected = start + weight * (end - start)
                # 断言计算得到的实际结果与预期结果相等
                self.assertEqual(expected, actual)

    # 标记为仅在 CUDA 设备上运行的测试函数，测试 torch.lerp 在低精度数据类型下的行为
    @onlyCUDA
    @dtypes(torch.half, torch.bfloat16)
    def test_lerp_lowp(self, device, dtype):
        # 定义 x 和 y 值的组合
        xvals = (0.0, -30000.0)
        yvals = (0.1, -20000.0)
        # 创建 x 张量列表，每个张量填充相同的 x 值，并将其移动到指定设备上，并指定数据类型
        xs = [torch.full((4,), xval, device=device, dtype=dtype) for xval in xvals]
        # 创建 y 张量列表，每个张量填充相同的 y 值，并将其移动到指定设备上，并指定数据类型
        ys = [torch.full((4,), yval, device=device, dtype=dtype) for yval in yvals]
        # 定义权重列表，包括一个标量和一个张量
        weights = [70000, torch.full((4,), 8, device=device, dtype=dtype)]
        # 遍历 x、y 和权重的组合
        for x, y, w in zip(xs, ys, weights):
            # 将 x 张量转换为 float 类型的参考张量 xref
            xref = x.float()
            # 将 y 张量转换为 float 类型的参考张量 yref
            yref = y.float()
            # 如果权重是张量，则将其转换为 float 类型的参考张量 wref
            wref = w.float() if isinstance(w, torch.Tensor) else w
            # 使用 torch.lerp 函数计算加权线性插值的实际结果
            actual = torch.lerp(x, y, w)
            # 使用参考张量 xref、yref 和 wref 计算加权线性插值的预期结果，并转换为指定数据类型
            expected = torch.lerp(xref, yref, wref).to(dtype)
            # 断言实际结果与预期结果相等，允许的误差为 0
            self.assertEqual(actual, expected, atol=0.0, rtol=0.0)

    # 标记为仅在 CPU 设备上运行的测试函数，测试 torch.lerp 在低精度数据类型下的行为
    @onlyCPU
    @dtypes(torch.half, torch.bfloat16)
    def test_lerp_lowp_cpu(self, device, dtype):
        # 定义 x 和 y 值的组合
        xvals = (0.0, -30000.0)
        yvals = (0.1, -20000.0)
        # 遍历不同形状的组合
        for shape in [(4,), (20,), (3, 10, 10)]:
            # 创建 x 张量列表，每个张量填充相同的 x 值，并将其移动到指定设备上，并指定数据类型
            xs = [torch.full(shape, xval, device=device, dtype=dtype) for xval in xvals]
            # 创建 y 张量列表，每个张量填充相同的 y 值，并将其移动到指定设备上，并指定数据类型
            ys = [torch.full(shape, yval, device=device, dtype=dtype) for yval in yvals]
            # 定义权重列表，包括一个标量和一个张量
            weights = [70000, torch.full(shape, 8, device=device, dtype=dtype)]
            # 遍历 x、y 和权重的组合
            for x, y, w in zip(xs, ys, weights):
                # 将 x 张量转换为 float 类型的参考张量 xref
                xref = x.float()
                # 将 y 张量转换为 float 类型的参考张量 yref
                yref = y.float()
                # 如果权重是张量，则将其转换为 float 类型的参考张量 wref
                wref = w.float() if isinstance(w, torch.Tensor) else w
                # 使用 torch.lerp 函数计算加权线性插值的实际结果
                actual = torch.lerp(x, y, w)
                # 使用参考张量 xref、yref 和 wref 计算加权线性插值的预期结果，并转换为指定数据类型
                expected = torch.lerp(xref, yref, wref).to(dtype)
                # 断言实际结果与预期结果相等，允许的误差为 0
                self.assertEqual(actual, expected, atol=0.0, rtol=0.0)
    # 定义测试函数 _test_logaddexp，用于比较 numpy 和 torch 中的 logaddexp 或 logaddexp2 函数
    def _test_logaddexp(self, device, dtype, base2):
        # 如果 base2 为真，则选择 numpy 的 logaddexp2 和 torch 的 logaddexp2 函数
        if base2:
            ref_func = np.logaddexp2
            our_func = torch.logaddexp2
        # 如果数据类型为复数类型（torch.complex64 或 torch.complex128），因为 numpy 没有实现复数的 logaddexp，所以定义一个新的函数 _ref_func
        elif dtype in (torch.complex64, torch.complex128):
            def _ref_func(x, y):
                return scipy.special.logsumexp(np.stack((x, y), axis=0), axis=0)

            ref_func = _ref_func
            our_func = torch.logaddexp
        # 否则选择 numpy 的 logaddexp 和 torch 的 logaddexp 函数
        else:
            ref_func = np.logaddexp
            our_func = torch.logaddexp

        # 定义测试辅助函数 _test_helper，用于比较 ref_func 和 our_func 的结果是否相等
        def _test_helper(a, b):
            # 如果数据类型为 torch.bfloat16，则先将数据转换为 float 类型进行比较
            if dtype == torch.bfloat16:
                ref = ref_func(a.cpu().float().numpy(), b.cpu().float().numpy())
                v = our_func(a, b)
                self.assertEqual(ref, v.float(), atol=0.01, rtol=0.01)
            else:
                ref = ref_func(a.cpu().numpy(), b.cpu().numpy())
                v = our_func(a, b)
                self.assertEqual(ref, v)

        # 简单的测试用例
        a = torch.randn(64, 2, dtype=dtype, device=device) - 0.5
        b = torch.randn(64, 2, dtype=dtype, device=device) - 0.5
        _test_helper(a, b)
        _test_helper(a[:3], b[:3])

        # 数值稳定性测试，用于大数值的情况
        a *= 10000
        b *= 10000
        _test_helper(a, b)
        _test_helper(a[:3], b[:3])

        # 包含特殊值的测试用例，如 inf, -inf, nan
        a = torch.tensor(
            [float("inf"), float("-inf"), float("inf"), float("nan")],
            dtype=dtype,
            device=device,
        )
        b = torch.tensor(
            [float("inf"), float("-inf"), float("-inf"), float("nan")],
            dtype=dtype,
            device=device,
        )
        _test_helper(a, b)

    # 装饰器，如果在 TorchDynamo 环境下则跳过测试（复数情况下的 inf/nan 在 Dynamo/Inductor 下会有差异）
    @skipIfTorchDynamo()
    # 装饰器，指定 CUDA 环境下测试的数据类型为 torch.float32, torch.float64, torch.bfloat16
    @dtypesIfCUDA(torch.float32, torch.float64, torch.bfloat16)
    # 装饰器，指定测试的数据类型为 torch.float32, torch.float64, torch.bfloat16, torch.complex64, torch.complex128
    @dtypes(
        torch.float32, torch.float64, torch.bfloat16, torch.complex64, torch.complex128
    )
    # 测试函数，测试 logaddexp 函数
    def test_logaddexp(self, device, dtype):
        self._test_logaddexp(device, dtype, base2=False)

    # 装饰器，指定测试的数据类型为 torch.float32, torch.float64, torch.bfloat16
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    # 测试函数，测试 logaddexp2 函数
    def test_logaddexp2(self, device, dtype):
        self._test_logaddexp(device, dtype, base2=True)

    # 装饰器，指定只在 CUDA 环境下进行测试
    @onlyCUDA
    # 测试函数，测试在半精度张量上的加法和减法操作
    def test_addsub_half_tensor(self, device):
        x = torch.tensor([60000.0], dtype=torch.half, device=device)
        for op, y, alpha in (
            (torch.add, torch.tensor([-60000.0], dtype=torch.half, device=device), 2),
            (torch.sub, torch.tensor([60000.0], dtype=torch.half, device=device), 2),
            (torch.add, -70000.0, 1),
            (torch.sub, 70000.0, 1),
        ):
            actual = op(x, y, alpha=alpha)
            self.assertTrue(not (actual.isnan() or actual.isinf()))
    # 测试减法操作在布尔张量上的异常情况
    def test_sub_typing(self, device):
        # 创建第一个布尔张量 m1
        m1 = torch.tensor(
            [True, False, False, True, False, False], dtype=torch.bool, device=device
        )
        # 创建第二个布尔张量 m2
        m2 = torch.tensor(
            [True, True, False, False, False, True], dtype=torch.bool, device=device
        )
        # 断言减法操作在两个布尔张量上会引发 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            r"Subtraction, the `\-` operator, with two bool tensors is not supported. "
            r"Use the `\^` or `logical_xor\(\)` operator instead.",
            lambda: m1 - m2,
        )
        # 断言减法操作在布尔张量和标量上会引发 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            r"Subtraction, the `\-` operator, with a bool tensor is not supported. "
            r"If you are trying to invert a mask, use the `\~` or `logical_not\(\)` operator instead.",
            lambda: 1 - m1,
        )
        # 断言减法操作在布尔张量和标量上会引发 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            r"Subtraction, the `\-` operator, with a bool tensor is not supported. "
            r"If you are trying to invert a mask, use the `\~` or `logical_not\(\)` operator instead.",
            lambda: m2 - 1,
        )

        # 测试不匹配的 alpha 参数
        # 创建第一个整数张量 m1
        m1 = torch.tensor([1], dtype=torch.int8, device=device)
        # 创建第二个整数张量 m2
        m2 = torch.tensor([2], dtype=torch.int8, device=device)
        # 断言在整数张量上使用 alpha 参数会引发 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            r"Boolean alpha only supported for Boolean results\.",
            lambda: torch.sub(m1, m2, alpha=True),
        )
        # 断言在整数张量上使用浮点型 alpha 参数会引发 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            r"For integral input tensors, argument alpha must not be a floating point number\.",
            lambda: torch.sub(m1, m2, alpha=1.0),
        )

    # 测试乘法操作
    def test_mul(self, device):
        # 创建一个随机张量 m1
        m1 = torch.randn(10, 10, device=device)
        # 克隆 m1 到 res1
        res1 = m1.clone()
        # 对 res1 的第三列元素乘以 2（原地操作）
        res1[:, 3].mul_(2)
        # 克隆 m1 到 res2
        res2 = m1.clone()
        # 循环对 res2 的每一行的第三列元素乘以 2
        for i in range(res1.size(0)):
            res2[i, 3] = res2[i, 3] * 2
        # 断言 res1 和 res2 相等
        self.assertEqual(res1, res2)

        # 创建两个布尔张量 a1 和 a2
        a1 = torch.tensor([True, False, False, True], dtype=torch.bool, device=device)
        a2 = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
        # 断言布尔张量 a1 和 a2 的逐元素乘积是否正确
        self.assertEqual(
            a1 * a2,
            torch.tensor([True, False, False, False], dtype=torch.bool, device=device),
        )

        # 如果设备为 CPU，测试 bfloat16 类型的乘法操作
        if device == "cpu":
            a1 = torch.tensor([0.1, 0.1], dtype=torch.bfloat16, device=device)
            a2 = torch.tensor([1.1, 0.1], dtype=torch.bfloat16, device=device)
            # 断言 bfloat16 类型张量的乘法结果是否符合预期，使用绝对误差和相对误差容忍度
            self.assertEqual(
                a1 * a2,
                torch.tensor([0.11, 0.01], dtype=torch.bfloat16, device=device),
                atol=0.01,
                rtol=0,
            )
            # 断言 bfloat16 类型张量的乘法结果是否与 mul 方法结果相等
            self.assertEqual(a1.mul(a2), a1 * a2)
    # 定义测试方法，用于测试布尔张量的比较操作
    def test_bool_tensor_comparison_ops(self, device):
        # 创建张量 a，包含布尔值 True 和 False，指定设备为 device
        a = torch.tensor(
            [True, False, True, False, True, False], dtype=torch.bool, device=device
        )
        # 创建张量 b，包含布尔值 True 和 False，指定设备为 device
        b = torch.tensor(
            [True, False, True, True, True, True], dtype=torch.bool, device=device
        )
        # 断言 a 与 b 的相等性，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a == b, torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.bool, device=device)
        )
        # 断言 a 与 b 的不等性，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a != b, torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool, device=device)
        )
        # 断言 a 小于 b，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a < b, torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool, device=device)
        )
        # 断言 a 大于 b，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a > b, torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.bool, device=device)
        )
        # 断言 a 大于等于 b，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a >= b, torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.bool, device=device)
        )
        # 断言 a 小于等于 b，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a <= b, torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool, device=device)
        )
        # 断言 a 大于 False，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a > False, torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool, device=device)
        )
        # 断言 a 等于 True，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a == torch.tensor(True, dtype=torch.bool, device=device),
            torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool, device=device),
        )
        # 断言 a 等于 0，预期结果是包含相应布尔值的张量
        self.assertEqual(
            a == torch.tensor(0, dtype=torch.bool, device=device),
            torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.bool, device=device),
        )
        # 使用 assertFalse 断言 a 不等于 b
        self.assertFalse(a.equal(b))

    # 使用 @dtypes 装饰器，标记所有数据类型及 torch.half, torch.bfloat16, torch.bool
    # 定义一个测试逻辑函数，接受设备和数据类型作为参数
    def test_logical(self, device, dtype):
        # 如果数据类型不是 torch.bool
        if dtype != torch.bool:
            # 创建一个张量 x，包含元素 [1, 2, 3, 4]，使用给定的设备和数据类型
            x = torch.tensor([1, 2, 3, 4], device=device, dtype=dtype)
            # 创建一个张量 b，包含元素 [2]，使用给定的设备和数据类型
            b = torch.tensor([2], device=device, dtype=dtype)
            # 断言 x 中小于 2 的元素为 True，其余为 False
            self.assertEqual(x.lt(2), torch.tensor([True, False, False, False]))
            # 断言 x 中小于等于 2 的元素为 True，其余为 False
            self.assertEqual(x.le(2), torch.tensor([True, True, False, False]))
            # 断言 x 中大于等于 2 的元素为 True，其余为 False
            self.assertEqual(x.ge(2), torch.tensor([False, True, True, True]))
            # 断言 x 中大于 2 的元素为 True，其余为 False
            self.assertEqual(x.gt(2), torch.tensor([False, False, True, True]))
            # 断言 x 中等于 2 的元素为 True，其余为 False
            self.assertEqual(x.eq(2), torch.tensor([False, True, False, False]))
            # 断言 x 中不等于 2 的元素为 True，其余为 False
            self.assertEqual(x.ne(2), torch.tensor([True, False, True, True]))

            # 使用张量 b 替代 x 中的元素进行相同的比较
            self.assertEqual(x.lt(b), torch.tensor([True, False, False, False]))
            self.assertEqual(x.le(b), torch.tensor([True, True, False, False]))
            self.assertEqual(x.ge(b), torch.tensor([False, True, True, True]))
            self.assertEqual(x.gt(b), torch.tensor([False, False, True, True]))
            self.assertEqual(x.eq(b), torch.tensor([False, True, False, False]))
            self.assertEqual(x.ne(b), torch.tensor([True, False, True, True]))
        else:
            # 如果数据类型是 torch.bool，则创建一个张量 x，包含元素 [True, False, True, False]，使用给定的设备
            x = torch.tensor([True, False, True, False], device=device)
            # 断言 x 中小于 True 的元素为 False，其余为 True
            self.assertEqual(x.lt(True), torch.tensor([False, True, False, True]))
            # 断言 x 中小于等于 True 的元素为 True，其余为 True
            self.assertEqual(x.le(True), torch.tensor([True, True, True, True]))
            # 断言 x 中大于等于 True 的元素为 True，其余为 False
            self.assertEqual(x.ge(True), torch.tensor([True, False, True, False]))
            # 断言 x 中大于 True 的元素为 False，其余为 False
            self.assertEqual(x.gt(True), torch.tensor([False, False, False, False]))
            # 断言 x 中等于 True 的元素为 True，其余为 False
            self.assertEqual(x.eq(True), torch.tensor([True, False, True, False]))
            # 断言 x 中不等于 True 的元素为 False，其余为 True
            self.assertEqual(x.ne(True), torch.tensor([False, True, False, True]))
    # 定义测试函数 test_atan2，用于测试 torch.atan2 方法在不同情况下的行为
    def test_atan2(self, device):
        
        # 定义内部函数 _test_atan2_with_size，测试指定大小的张量在给定设备上的 atan2 方法
        def _test_atan2_with_size(size, device):
            # 创建随机张量 a 和 b，大小为 size，数据类型为 torch.double，位于指定设备上
            a = torch.rand(size=size, device=device, dtype=torch.double)
            b = torch.rand(size=size, device=device, dtype=torch.double)
            # 计算 a 和 b 的 atan2 结果
            actual = a.atan2(b)
            # 将 a 和 b 展平为一维张量
            x = a.view(-1)
            y = b.view(-1)
            # 计算期望的 atan2 结果，使用 math.atan2 函数逐元素计算
            expected = torch.tensor(
                [math.atan2(x[i].item(), y[i].item()) for i in range(x.numel())],
                device=device,
                dtype=torch.double,
            )
            # 断言实际结果与期望结果一致
            self.assertEqual(expected, actual.view(-1), rtol=0, atol=0.02)

            # 测试 bfloat16/float16 数据类型的情况
            for lowp_dtype in [torch.bfloat16, torch.float16]:
                if lowp_dtype == torch.bfloat16:
                    rtol = 0
                    atol = 0.02
                else:
                    rtol = 0
                    atol = 0.001
                # 将 a 和 b 转换为低精度数据类型
                a_16 = a.to(dtype=lowp_dtype)
                b_16 = b.to(dtype=lowp_dtype)
                # 计算低精度数据类型下的 atan2 结果
                actual_16 = a_16.atan2(b_16)
                # 断言低精度结果与高精度结果的一致性
                self.assertEqual(actual_16, actual.to(dtype=lowp_dtype))
                self.assertEqual(
                    expected,
                    actual_16.view(-1),
                    exact_dtype=False,
                    rtol=rtol,
                    atol=atol,
                )

        # 分别使用不同大小的张量进行测试
        _test_atan2_with_size((2, 2), device)
        _test_atan2_with_size((3, 3), device)
        _test_atan2_with_size((5, 5), device)

    # 定义测试函数 test_atan2_edgecases，测试 atan2 方法的边界情况
    def test_atan2_edgecases(self, device):
        
        # 定义内部函数 _test_atan2，测试给定 x、y 值的 atan2 结果是否符合期望
        def _test_atan2(x, y, expected, device, dtype):
            # 创建期望结果的张量
            expected_tensor = torch.tensor([expected], dtype=dtype, device=device)
            # 创建 x 和 y 的张量
            x_tensor = torch.tensor([x], dtype=dtype, device=device)
            y_tensor = torch.tensor([y], dtype=dtype, device=device)
            # 计算 atan2 结果
            actual = torch.atan2(y_tensor, x_tensor)
            # 断言实际结果与期望结果一致
            self.assertEqual(expected_tensor, actual, rtol=0, atol=0.02)

        # 遍历不同的数据类型进行测试
        for dtype in [torch.float, torch.double]:
            _test_atan2(0, 0, 0, device, dtype)
            _test_atan2(0, 1, math.pi / 2, device, dtype)
            _test_atan2(0, -1, math.pi / -2, device, dtype)
            _test_atan2(-1, 0, math.pi, device, dtype)
            _test_atan2(1, 0, 0, device, dtype)
            _test_atan2(-1, -1, math.pi * -3 / 4, device, dtype)
            _test_atan2(1, 1, math.pi / 4, device, dtype)
            _test_atan2(1, -1, math.pi / -4, device, dtype)
            _test_atan2(-1, 1, math.pi * 3 / 4, device, dtype)
    # 定义一个测试方法，用于测试 torch.trapezoid 函数在不同输入条件下的行为
    def test_trapezoid(self, device):
        
        # 定义一个内部函数，用于测试在给定大小、维度、dx 和设备下的 torch.trapezoid 调用
        def test_dx(sizes, dim, dx, device):
            # 生成一个指定大小的随机张量，使用指定设备
            t = torch.randn(sizes, device=device)
            # 调用 torch.trapezoid 函数计算梯形面积，以及与之对比的 NumPy 实现
            actual = torch.trapezoid(t, dx=dx, dim=dim)
            expected = np.trapz(t.cpu().numpy(), dx=dx, axis=dim)  # noqa: NPY201
            # 断言预期输出形状与实际输出形状相同
            self.assertEqual(expected.shape, actual.shape)
            # 断言预期输出与实际输出值相等（不要求精确类型匹配）
            self.assertEqual(expected, actual, exact_dtype=False)

        # 定义一个内部函数，用于测试在给定大小、维度、x 和设备下的 torch.trapezoid 调用
        def test_x(sizes, dim, x, device):
            # 生成一个指定大小的随机张量，使用指定设备
            t = torch.randn(sizes, device=device)
            # 调用 torch.trapezoid 函数计算梯形面积，以及与之对比的 NumPy 实现
            actual = torch.trapezoid(t, x=torch.tensor(x, device=device), dim=dim)
            expected = np.trapz(t.cpu().numpy(), x=x, axis=dim)  # noqa: NPY201
            # 断言预期输出形状与实际输出形状相同
            self.assertEqual(expected.shape, actual.shape)
            # 断言预期输出与实际输出值相等（需将实际输出转移到 CPU 上比较，不要求精确类型匹配）
            self.assertEqual(expected, actual.cpu(), exact_dtype=False)

        # 分别调用 test_dx 和 test_x 函数，进行多组不同参数下的测试
        test_dx((2, 3, 4), 1, 1, device)
        test_dx((10, 2), 0, 0.1, device)
        test_dx((1, 10), 0, 2.3, device)
        test_dx((0, 2), 0, 1.0, device)
        test_dx((0, 2), 1, 1.0, device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x(
            (10, 2), 0, [2.0, 3.0, 4.0, 7.0, 11.0, 14.0, 22.0, 26.0, 26.1, 30.3], device
        )
        test_x((1, 10), 0, [1.0], device)
        test_x((0, 2), 0, [], device)
        test_x((0, 2), 1, [1.0, 2.0], device)
        test_x((2, 3, 4), -1, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 3, 4), 0, [1.0, 2.0], device)
        test_x((2, 3, 4), 1, [1.0, 2.0, 3.0], device)
        test_x((2, 3, 4), 2, [1.0, 2.0, 3.0, 4.0], device)
        test_x((2, 2, 4), -1, [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], device)
        
        # 使用 assertRaisesRegex 断言捕获预期的 IndexError 异常
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            test_x((2, 3), 2, [], device)
            test_dx((2, 3), 2, 1.0, device)
        
        # 使用 assertRaisesRegex 断言捕获预期的 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "There must be one `x` value for each sample point"
        ):
            test_x((2, 3), 1, [1.0, 2.0], device)
            test_x((2, 3), 1, [1.0, 2.0, 3.0, 4.0], device)

    # 使用装饰器指定条件下跳过测试，如果未安装 Scipy，则跳过当前测试
    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    # 使用装饰器指定当前测试为 meta 跳过
    @skipMeta
    # 使用装饰器指定当前测试函数接受 torch.double 类型参数
    @dtypes(torch.double)
    # 定义测试方法，用于测试幂函数的重载和内存重叠
    def test_pow_scalar_overloads_mem_overlap(self, device, dtype):
        # 定义常量 sz 为 3
        sz = 3
        # 生成指定大小的随机张量，使用指定数据类型和设备
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        # 使用 self.check_internal_mem_overlap 方法检查内存重叠情况
        self.check_internal_mem_overlap(lambda t: t.pow_(42), 1, dtype, device)
        # 使用 self.unary_check_input_output_mem_overlap 方法检查输入和输出的内存重叠情况
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: torch.pow(input, 42, out=out)
        )
        # 使用 self.unary_check_input_output_mem_overlap 方法检查输入和输出的内存重叠情况
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: torch.pow(42, input, out=out)
        )

    # 使用装饰器指定当前测试函数接受多种数据类型参数的组合
    @dtypes(
        *list(
            product(
                all_types_and_complex_and(torch.half, torch.bfloat16),
                all_types_and_complex_and(torch.half, torch.bfloat16),
            )
        )
    )
    # 定义测试方法，用于测试 torch.float_power 函数在异常情况下的行为
    def test_float_power_exceptions(self, device):
        
        # 定义内部辅助函数 _promo_helper，用于确定输出的数据类型
        def _promo_helper(x, y):
            # 遍历 x 和 y
            for i in (x, y):
                # 如果元素是复数类型，则返回 torch.complex128
                if type(i) == complex:
                    return torch.complex128
                # 如果元素是 torch.Tensor 类型且是复数，则返回 torch.complex128
                elif type(i) == torch.Tensor and i.is_complex():
                    return torch.complex128
            # 默认返回 torch.double
            return torch.double

        # 定义测试用例，包括 base 和 exp 的不同组合
        test_cases = (
            (torch.tensor([-2, -1, 0, 1, 2], device=device), -0.25),
            (
                torch.tensor([-1.0j, 0j, 1.0j, 1.0 + 1.0j, -1.0 - 1.5j], device=device),
                2.0,
            ),
        )
        
        # 遍历测试用例
        for base, exp in test_cases:
            # 遍历输出数据类型的可能取值
            for out_dtype in (torch.long, torch.float, torch.double, torch.cdouble):
                # 创建一个空张量 out，指定设备和数据类型
                out = torch.empty(1, device=device, dtype=out_dtype)
                # 调用 _promo_helper 函数获取所需的数据类型
                required_dtype = _promo_helper(base, exp)

                # 如果 out 的数据类型与所需的数据类型一致
                if out.dtype == required_dtype:
                    # 调用 torch.float_power 函数，并将结果存储在 out 中
                    torch.float_power(base, exp, out=out)
                else:
                    # 如果 out 的数据类型与所需的数据类型不一致，预期抛出 RuntimeError 异常
                    with self.assertRaisesRegex(
                        RuntimeError, "operation's result requires dtype"
                    ):
                        torch.float_power(base, exp, out=out)

                # 如果 base 的数据类型与所需的数据类型一致
                if base.dtype == required_dtype:
                    # 调用 torch.Tensor.float_power_ 函数
                    torch.Tensor.float_power_(base.clone(), exp)
                else:
                    # 如果 base 的数据类型与所需的数据类型不一致，预期抛出 RuntimeError 异常
                    with self.assertRaisesRegex(
                        RuntimeError, "operation's result requires dtype"
                    ):
                        torch.Tensor.float_power_(base.clone(), exp)

    # 装饰器函数 @skipIf 标记这个测试依赖于 TEST_SCIPY 变量为真
    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    # 装饰器函数 @dtypes 指定测试用例的数据类型组合
    @dtypes(
        *product(
            all_types_and(torch.half, torch.bool), all_types_and(torch.half, torch.bool)
        )
    )
    # 装饰器函数 @dtypes 指定测试用例的数据类型为 torch.float64
    @dtypes(torch.float64)
    # 定义测试方法 test_xlogy_xlog1py_gradients，用于测试 torch.special.xlogy 和 torch.special.xlog1py 函数的梯度
    def test_xlogy_xlog1py_gradients(self, device, dtype):
        # 创建部分应用函数 make_arg，用于创建具有指定数据类型、设备和梯度的张量
        make_arg = partial(torch.tensor, dtype=dtype, device=device, requires_grad=True)

        # 创建全零张量 zeros，指定数据类型和设备
        zeros = torch.zeros((2,), dtype=dtype, device=device)

        # 创建 x 和 y 张量，分别使用 make_arg 函数创建
        x = make_arg([0.0, 0.0])
        y = make_arg([-1.5, 0.0])
        # 调用 torch.special.xlogy 函数计算 x 和 y 的结果，对结果求和并反向传播
        torch.special.xlogy(x, y).sum().backward()
        # 断言 x 的梯度与 zeros 张量相等
        self.assertEqual(x.grad, zeros)

        # 重复上述过程，但使用 torch.special.xlog1py 函数
        x = make_arg([0.0, 0.0])
        y = make_arg([-2.5, -1.0])
        torch.special.xlog1py(x, y).sum().backward()
        self.assertEqual(x.grad, zeros)
    # 定义一个测试函数，用于测试 xlogy 和 xlog1py 函数的标量类型提升行为
    def test_xlogy_xlog1py_scalar_type_promotion(self, device):
        # 测试 Python 数字在类型提升时与零维张量不在同一优先级
        # 生成一个随机的零维张量 t，数据类型为 float32，设备为指定的 device
        t = torch.randn((), dtype=torch.float32, device=device)

        # 断言 t 的数据类型与 torch.xlogy(t, 5) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.xlogy(t, 5).dtype)
        # 断言 t 的数据类型与 torch.xlogy(t, 5.0) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.xlogy(t, 5.0).dtype)
        # 断言 t 的数据类型与 torch.special.xlog1py(t, 5) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.special.xlog1py(t, 5).dtype)
        # 断言 t 的数据类型与 torch.special.xlog1py(t, 5.0) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.special.xlog1py(t, 5.0).dtype)

        # 断言 t 的数据类型与 torch.xlogy(5, t) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.xlogy(5, t).dtype)
        # 断言 t 的数据类型与 torch.xlogy(5.0, t) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.xlogy(5.0, t).dtype)
        # 断言 t 的数据类型与 torch.special.xlog1py(5, t) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.special.xlog1py(5, t).dtype)
        # 断言 t 的数据类型与 torch.special.xlog1py(5.0, t) 的结果数据类型相同
        self.assertEqual(t.dtype, torch.special.xlog1py(5.0, t).dtype)

    # 根据是否存在 TEST_SCIPY 变量来跳过当前测试
    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    # 定义一个测试函数，测试 torch.bfloat16 类型的函数
    def test_xlogy_xlog1py_bfloat16(self, device):
        # 定义一个辅助函数，比较 torch_fn 和 reference_fn 的输出结果是否一致
        def _compare_helper(x, y, torch_fn, reference_fn):
            # 如果 x 和 y 不是 float 类型，则将它们转换成 numpy 数组
            x_np = x if isinstance(x, float) else x.cpu().to(torch.float).numpy()
            y_np = y if isinstance(y, float) else y.cpu().to(torch.float).numpy()
            # 使用 reference_fn 计算期望结果，并转换成 PyTorch tensor
            expected = torch.from_numpy(reference_fn(x_np, y_np))
            # 使用 torch_fn 计算实际结果
            actual = torch_fn(x, y)
            # 断言期望结果和实际结果是否相等，不要求严格的数据类型匹配
            self.assertEqual(expected, actual, exact_dtype=False)

        # 设置 x 和 y 的数据类型为 torch.bfloat16
        x_dtype, y_dtype = torch.bfloat16, torch.bfloat16

        # 创建不同形状的张量 x, y, z 以及 x_1p, y_1p, z_1p，数据类型为 torch.bfloat16
        x = make_tensor((3, 2, 4, 5), dtype=x_dtype, device=device, low=0.5, high=1000)
        y = make_tensor((3, 2, 4, 5), dtype=y_dtype, device=device, low=0.5, high=1000)
        z = make_tensor((4, 5), dtype=y_dtype, device=device, low=0.5, high=1000)

        x_1p = make_tensor(
            (3, 2, 4, 5), dtype=x_dtype, device=device, low=-0.8, high=1000
        )
        y_1p = make_tensor(
            (3, 2, 4, 5), dtype=y_dtype, device=device, low=-0.8, high=1000
        )
        z_1p = make_tensor((4, 5), dtype=y_dtype, device=device, low=-0.8, high=1000)

        # 定义 xlogy_fns 和 xlog1py_fns 为 torch 和 scipy.special 中的函数对
        xlogy_fns = torch.xlogy, scipy.special.xlogy
        xlog1py_fns = torch.special.xlog1py, scipy.special.xlog1py

        # 对不同的输入参数组合调用 _compare_helper 进行比较
        _compare_helper(x, x, *xlogy_fns)
        _compare_helper(x, y, *xlogy_fns)
        _compare_helper(x, z, *xlogy_fns)
        _compare_helper(x, 3.14, *xlogy_fns)
        _compare_helper(y, 3.14, *xlogy_fns)
        _compare_helper(z, 3.14, *xlogy_fns)

        _compare_helper(x_1p, x_1p, *xlog1py_fns)
        _compare_helper(x_1p, y_1p, *xlog1py_fns)
        _compare_helper(x_1p, z_1p, *xlog1py_fns)
        _compare_helper(x_1p, 3.14, *xlog1py_fns)
        _compare_helper(y_1p, 3.14, *xlog1py_fns)
        _compare_helper(z_1p, 3.14, *xlog1py_fns)

        # 创建包含特殊值的张量 t，数据类型为 device
        t = torch.tensor(
            [-1.0, 0.0, 1.0, 2.0, float("inf"), -float("inf"), float("nan")],
            device=device,
        )
        # 创建一个值为 7 的张量 zeros，数据类型为 y_dtype
        zeros = torch.tensor(7, dtype=y_dtype, device=device)

        # 对不同的输入参数组合调用 _compare_helper 进行比较
        _compare_helper(t, zeros, *xlogy_fns)
        _compare_helper(t, 0.0, *xlogy_fns)

        _compare_helper(t, zeros, *xlog1py_fns)
        _compare_helper(t, 0.0, *xlog1py_fns)

    # 标记测试用例的数据类型和设备，并且要求 scipy 的支持，该测试用例被认为是一个较慢的测试
    @dtypes(*product(all_types_and(torch.bool), all_types_and(torch.bool)))
    @skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @slowTest
    # 定义测试函数 test_zeta，用于测试 torch.special.zeta 函数
    def test_zeta(self, device, dtypes):
        # 从 dtypes 中获取 x 和 q 的数据类型
        x_dtype, q_dtype = dtypes

        # 辅助函数 test_helper，用于执行 zeta 函数的测试
        def test_helper(x, q):
            # 如果 x 是浮点数，直接使用 x；否则将 x 转换为 NumPy 数组
            x_np = x if isinstance(x, float) else x.cpu().numpy()
            # 如果 q 是浮点数，直接使用 q；否则将 q 转换为 NumPy 数组
            q_np = q if isinstance(q, float) else q.cpu().numpy()
            # 计算预期结果，使用 scipy.special.zeta 函数计算并转换为 PyTorch 张量
            expected = torch.from_numpy(scipy.special.zeta(x_np, q_np))
            # 计算实际结果，使用 torch.special.zeta 函数计算
            actual = torch.special.zeta(x, q)

            # 如果设备类型是 "cpu"，设置相对误差和绝对误差的阈值
            rtol, atol = None, None
            if self.device_type == "cpu":
                rtol, atol = 1e-6, 1e-6
            # 断言预期值与实际值相等，使用给定的相对误差和绝对误差阈值
            self.assertEqual(expected, actual, rtol=rtol, atol=atol, exact_dtype=False)

        # 测试用例1：x 和 q 张量具有相同的大小
        x = make_tensor((2, 3, 4), dtype=x_dtype, device=device)
        q = make_tensor((2, 3, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # 测试用例2：x 张量比 q 张量广播，左边广播
        x = make_tensor((2, 1, 4), dtype=x_dtype, device=device)
        q = make_tensor((2, 3, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # 测试用例3：x 张量比 q 张量广播，右边广播
        x = make_tensor((2, 3, 4), dtype=x_dtype, device=device)
        q = make_tensor((2, 1, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # 测试用例4：x 张量和 q 张量都广播
        x = make_tensor((2, 3, 1), dtype=x_dtype, device=device)
        q = make_tensor((2, 1, 4), dtype=q_dtype, device=device)
        test_helper(x, q)

        # 测试用例5：x 是标量，q 是张量，对 x 在指定范围内的值进行迭代测试
        for x in np.linspace(-5, 5, num=10).tolist():
            # 如果 q 的数据类型不是浮点数，使用默认的数据类型
            if not q_dtype.is_floating_point:
                q_dtype = torch.get_default_dtype()
            # 创建 q 张量
            q = make_tensor((2, 3, 4), dtype=q_dtype, device=device)
            test_helper(x, q)

        # 测试用例6：x 是张量，q 是标量，对 q 在指定范围内的值进行迭代测试
        for q in np.linspace(-5, 5, num=10).tolist():
            # 如果 x 的数据类型不是浮点数，使用默认的数据类型
            if not x_dtype.is_floating_point:
                x_dtype = torch.get_default_dtype()
            # 创建 x 张量
            x = make_tensor((2, 3, 4), dtype=x_dtype, device=device)
            test_helper(x, q)

    # 仅在 CUDA 环境下运行的测试装饰器，用于测试 chalf 类型的张量与 CPU 标量的乘法
    @onlyCUDA
    # 指定测试数据类型为 torch.chalf
    @dtypes(
        torch.chalf,
    )
    def test_mul_chalf_tensor_and_cpu_scalar(self, device, dtype):
        # 测试张量与 CPU 标量的乘法是否正常工作，针对 chalf 类型
        # 理想情况下，应该由 test_ops.py 中的 test_complex_half_reference_testing 函数覆盖
        # 通过检查 OpInfo 中的 reference_samples 来验证
        # 但目前由于 sample 生成需要对 complex32 支持 `index_select`，而这在编写该测试时尚未实现
        # TODO: 修复上述问题后，删除此测试
        # 参考：https://github.com/pytorch/pytorch/pull/76364
        # 创建 chalf 类型的张量 x
        x = make_tensor((2, 2), device=device, dtype=dtype)
        # 断言 x 乘以 2.5 等于 x 乘以 torch.tensor(2.5, device=device, dtype=dtype)
        self.assertEqual(x * 2.5, x * torch.tensor(2.5, device=device, dtype=dtype))
# 定义包含所有二进制操作符名称的列表
tensor_binary_ops = [
    "__lt__",        # 小于操作符 <
    "__le__",        # 小于等于操作符 <=
    "__gt__",        # 大于操作符 >
    "__ge__",        # 大于等于操作符 >=
    "__eq__",        # 等于操作符 ==
    "__ne__",        # 不等于操作符 !=
    "__add__",       # 加法操作符 +
    "__radd__",      # 右向加法操作符 +
    "__iadd__",      # 原地加法操作符 +=
    "__sub__",       # 减法操作符 -
    "__rsub__",      # 右向减法操作符 -
    "__isub__",      # 原地减法操作符 -=
    "__mul__",       # 乘法操作符 *
    "__rmul__",      # 右向乘法操作符 *
    "__imul__",      # 原地乘法操作符 *=
    "__matmul__",    # 矩阵乘法操作符 @
    "__rmatmul__",   # 右向矩阵乘法操作符 @
    "__truediv__",   # 真除操作符 /
    "__rtruediv__",  # 右向真除操作符 /
    "__itruediv__",  # 原地真除操作符 /=
    "__floordiv__",  # 地板除操作符 //
    "__rfloordiv__", # 右向地板除操作符 //
    "__ifloordiv__", # 原地地板除操作符 //=
    "__mod__",       # 取模操作符 %
    "__rmod__",      # 右向取模操作符 %
    "__imod__",      # 原地取模操作符 %=
    "__pow__",       # 幂操作符 **
    "__rpow__",      # 右向幂操作符 **
    "__ipow__",      # 原地幂操作符 **=
    "__lshift__",    # 左移操作符 <<
    "__rlshift__",   # 右向左移操作符 <<
    "__ilshift__",   # 原地左移操作符 <<=
    "__rshift__",    # 右移操作符 >>
    "__rrshift__",   # 右向右移操作符 >>
    "__irshift__",   # 原地右移操作符 >>=
    "__and__",       # 按位与操作符 &
    "__rand__",      # 右向按位与操作符 &
    "__iand__",      # 原地按位与操作符 &=
    "__xor__",       # 按位异或操作符 ^
    "__rxor__",      # 右向按位异或操作符 ^
    "__ixor__",      # 原地按位异或操作符 ^=
    "__or__",        # 按位或操作符 |
    "__ror__",       # 右向按位或操作符 |
    "__ior__",       # 原地按位或操作符 |=
    # 不支持的操作符
    # '__imatmul__',
    # '__divmod__', '__rdivmod__', '__idivmod__',
]

# 测试二进制数学操作是否对未知类型返回NotImplemented
def generate_not_implemented_tests(cls):
    # 定义一个未知类型类，用于测试
    class UnknownType:
        pass

    # 待测试的数据类型列表
    _types = [
        torch.half,
        torch.float,
        torch.double,
        torch.int8,
        torch.short,
        torch.int,
        torch.long,
        torch.uint8,
    ]

    # 创建测试函数的生成器
    def create_test_func(op):
        @dtypes(*_types)
        def test(self, device, dtype):
            # 生成输入张量
            tensor = torch.empty((), device=device, dtype=dtype)

            # 在设备上执行张量操作
            result = getattr(tensor, op)(UnknownType())
            self.assertEqual(result, NotImplemented)

        return test

    # 为每个操作符创建测试函数并添加到测试类中
    for op in tensor_binary_ops:
        test_name = f"test_{op}_not_implemented"
        assert not hasattr(cls, test_name), f"{test_name} already in {cls.__name__}"

        setattr(cls, test_name, create_test_func(op))


# 生成不支持操作的测试函数
generate_not_implemented_tests(TestBinaryUfuncs)
# 实例化设备类型测试
instantiate_device_type_tests(TestBinaryUfuncs, globals())

# 如果是主程序，则运行测试
if __name__ == "__main__":
    run_tests()
```