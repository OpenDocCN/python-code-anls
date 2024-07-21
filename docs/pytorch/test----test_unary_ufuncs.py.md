# `.\pytorch\test\test_unary_ufuncs.py`

```py
# Owner(s): ["module: tests"]

import torch  # 导入 PyTorch 库
import numpy as np  # 导入 NumPy 库

import math  # 导入数学函数
from numbers import Number  # 导入 Number 类型
import random  # 导入随机数模块
import unittest  # 导入单元测试模块

from torch import inf, nan  # 导入无穷大和NaN常量
from torch.testing._internal.common_utils import (  # 导入测试中常用的工具函数
    TestCase,
    run_tests,
    torch_to_numpy_dtype_dict,
    numpy_to_torch_dtype_dict,
    suppress_warnings,
    TEST_SCIPY,
    slowTest,
    skipIfNoSciPy,
    IS_WINDOWS,
    gradcheck,
    is_iterable_of_tensors,
    xfailIfTorchDynamo,
)
from torch.testing._internal.common_methods_invocations import (  # 导入测试中常用的方法和调用
    unary_ufuncs,
    generate_elementwise_unary_tensors,
    generate_elementwise_unary_small_value_tensors,
    generate_elementwise_unary_large_value_tensors,
    generate_elementwise_unary_extremal_value_tensors,
)
from torch.testing._internal.common_device_type import (  # 导入测试中常用的设备类型
    instantiate_device_type_tests,
    ops,
    dtypes,
    onlyCPU,
    onlyNativeDeviceTypes,
    onlyCUDA,
    dtypesIfCUDA,
    precisionOverride,
    dtypesIfCPU,
)
from torch.utils import _pytree as pytree  # 导入私有模块 _pytree

from torch.testing import make_tensor  # 导入创建张量的测试工具函数
from torch.testing._internal.common_dtype import (  # 导入测试中常用的数据类型相关函数
    floating_types_and,
    all_types_and_complex_and,
    integral_types_and,
    get_all_math_dtypes,
    complex_types,
    floating_and_complex_types_and,
)

if TEST_SCIPY:  # 如果 TEST_SCIPY 变量为真，则导入 scipy 库
    import scipy

# Refer [scipy reference filter]
# Filter operators for which the reference function
# is available in the current environment (for reference_numerics tests).
# 使用 lambda 表达式过滤出在当前环境中存在参考函数的操作符（用于参考数值测试）。
reference_filtered_ops = list(filter(lambda op: op.ref is not None, unary_ufuncs))

# Tests for unary "universal functions (ufuncs)" that accept a single
# tensor and have common properties like:
#   - they are elementwise functions
#   - the input shape is the output shape
#   - they typically have method and inplace variants
#   - they typically support the out kwarg
#   - they typically have NumPy or SciPy references
# 测试一元“通用函数（ufuncs）”，这些函数接受单个张量并具有以下常见特性：
#   - 它们是逐元素函数
#   - 输入形状等于输出形状
#   - 通常具有方法和原地变体
#   - 通常支持 out 参数
#   - 通常有 NumPy 或 SciPy 参考实现

# See NumPy's universal function documentation
# (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
# about the concept of ufuncs.
# 更多关于通用函数的概念，请参阅 NumPy 的通用函数文档（https://numpy.org/doc/1.18/reference/ufuncs.html）。

# TODO: port test_unary_out_op_mem_overlap
# TODO: add test for inplace variants erroring on broadcasted inputs
# 待完成：迁移 test_unary_out_op_mem_overlap 测试
# 待完成：添加对广播输入错误使用原地变体的测试
class TestUnaryUfuncs(TestCase):  # 定义测试类 TestUnaryUfuncs，继承自单元测试基类 TestCase
    exact_dtype = True  # 精确的数据类型匹配标志为 True

    @ops(  # 使用 ops 装饰器，指定测试的操作符列表和允许的数据类型
        [_fn for _fn in unary_ufuncs if _fn.domain != (None, None)],  # 过滤出 domain 不为空的 ufunc
        allowed_dtypes=floating_types_and(torch.bfloat16, torch.half),  # 允许的数据类型包括浮点数和半精度浮点数
    )
    # 测试浮点数操作的定义域是否正确，参数包括设备、数据类型和操作
    def test_float_domains(self, device, dtype, op):
        # 定义一组epsilon值，用于测试浮点数的边界情况
        eps = (1e-5, 1e-3, 1e-1, 1, 2, 10, 20, 50, 100)

        # 获取操作的下界和上界
        low, high = op.domain
        # 注意：下面的两个循环是为了提高可读性而分开的

        # 检查下界是否存在
        if low is not None:
            # 创建代表下界的张量，使用指定的设备和数据类型
            low_tensor = torch.tensor(low, device=device, dtype=dtype)
            # 遍历epsilon值
            for epsilon in eps:
                # 计算低于下界epsilon的张量
                lower_tensor = low_tensor - epsilon

                # 如果差值与原始值相等，跳过测试
                # 例如，如果差值很小且数据类型不精确（例如bfloat16）
                if lower_tensor.item() == low_tensor.item():
                    continue

                # 执行操作并断言结果为NaN
                result = op(lower_tensor)
                self.assertEqual(
                    result.item(),
                    float("nan"),
                    msg=(
                        f"input of {lower_tensor.item()} outside lower domain boundary"
                        f" {low} produced {result.item()}, not nan!"
                    ),
                )

        # 检查上界是否存在
        if high is not None:
            # 创建代表上界的张量，使用指定的设备和数据类型
            high_tensor = torch.tensor(high, device=device, dtype=dtype)
            # 遍历epsilon值
            for epsilon in eps:
                # 计算高于上界epsilon的张量
                higher_tensor = high_tensor + epsilon

                # 如果差值与原始值相等，跳过测试
                if higher_tensor.item() == high_tensor.item():
                    continue

                # 执行操作并断言结果为NaN
                result = op(higher_tensor)
                self.assertEqual(
                    result.item(),
                    float("nan"),
                    msg=(
                        f"input of {higher_tensor.item()} outside upper domain boundary"
                        f" {high} produced {result.item()}, not nan!"
                    ),
                )

    # 用于比较torch张量和numpy数组的辅助函数
    # TODO: 这个函数或assertEqual是否应该验证步幅（strides）也相等？
    def assertEqualHelper(
        self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs
    ):
        # 断言 actual 是 torch.Tensor 类型
        assert isinstance(actual, torch.Tensor)

        # 一些 NumPy 函数返回标量而不是数组
        if isinstance(expected, Number):
            # 如果 expected 是数字类型，则比较 actual 的值与 expected
            self.assertEqual(actual.item(), expected, msg, **kwargs)
        elif isinstance(expected, np.ndarray):
            # 处理数组与张量之间的精确数据类型比较
            if exact_dtype:
                # 如果要求精确的数据类型比较
                if (
                    actual.dtype is torch.bfloat16
                    or expected.dtype != torch_to_numpy_dtype_dict[actual.dtype]
                ):
                    # 允许数组的数据类型为 float32，用于与 bfloat16 张量的比较，因为 NumPy 不支持 bfloat16 数据类型
                    # 而且像 scipy.special.erf、scipy.special.erfc 等操作会将 float16 提升为 float32
                    if expected.dtype == np.float32:
                        assert actual.dtype in (
                            torch.float16,
                            torch.bfloat16,
                            torch.float32,
                        )
                    elif expected.dtype == np.float64:
                        assert actual.dtype in (
                            torch.float16,
                            torch.bfloat16,
                            torch.float32,
                            torch.float64,
                        )
                    else:
                        # 报告错误，指出预期的数据类型与实际的数据类型不匹配
                        self.fail(
                            f"Expected dtype {expected.dtype} but got {actual.dtype}!"
                        )

            # 使用 torch.from_numpy(expected) 将 expected 转换为与 actual 相同的数据类型，并进行比较
            self.assertEqual(
                actual,
                torch.from_numpy(expected).to(actual.dtype),
                msg,
                exact_device=False,
                **kwargs
            )
        else:
            # 如果 expected 是普通类型，则直接比较 actual 和 expected
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    # 测试函数及其（接受数组的）参考函数在给定张量上产生相同的值
    # 测试函数及其（接受数组的）参考函数在一系列张量上产生相同的值，包括空张量、标量张量、
    #   1D 张量和包含有趣极端值和非连续性的大 2D 张量。
    @suppress_warnings
    @ops(reference_filtered_ops)
    def test_reference_numerics_normal(self, device, dtype, op):
        # 生成逐元素一元张量，用于正常测试，不需要梯度
        tensors = generate_elementwise_unary_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        # 使用 _test_reference_numerics 方法进行测试
        self._test_reference_numerics(dtype, op, tensors)

    @suppress_warnings
    @ops(reference_filtered_ops)
    def test_reference_numerics_small(self, device, dtype, op):
        # 对于 bool 类型的张量，跳过测试，因为 bool 类型没有小值
        if dtype in (torch.bool,):
            raise self.skipTest("bool has no small values")

        # 生成逐元素一元小值张量，用于测试小值情况，不需要梯度
        tensors = generate_elementwise_unary_small_value_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        # 使用 _test_reference_numerics 方法进行测试
        self._test_reference_numerics(dtype, op, tensors)
    # 应用 @suppress_warnings 装饰器，用于抑制测试中的警告
    # 应用 @ops 装饰器，指定操作集合为 reference_filtered_ops
    def test_reference_numerics_large(self, device, dtype, op):
        # 如果数据类型是 torch.bool, torch.uint8, torch.int8 中的一种，跳过测试
        if dtype in (torch.bool, torch.uint8, torch.int8):
            raise self.skipTest("bool, uint8, and int8 dtypes have no large values")

        # 生成大值的元素级一元张量，并且不需要梯度
        tensors = generate_elementwise_unary_large_value_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        # 执行具体的数值参考测试
        self._test_reference_numerics(dtype, op, tensors)

    # 应用 @suppress_warnings 装饰器，用于抑制测试中的警告
    # 应用 @ops 装饰器，指定操作集合为 reference_filtered_ops，并且允许浮点数和复数类型以及 torch.bfloat16, torch.half
    def test_reference_numerics_extremal(self, device, dtype, op):
        # 生成极端值的元素级一元张量，并且不需要梯度
        tensors = generate_elementwise_unary_extremal_value_tensors(
            op, device=device, dtype=dtype, requires_grad=False
        )
        # 执行具体的数值参考测试
        self._test_reference_numerics(dtype, op, tensors)

    # 用于测试（非）连续性一致性的测试
    # 应用 @ops 装饰器，指定操作集合为 unary_ufuncs
    def test_contig_vs_every_other(self, device, dtype, op):
        # 创建连续张量 contig
        contig = make_tensor(
            (1026,), device=device, dtype=dtype, low=op.domain[0], high=op.domain[1]
        )
        # 创建非连续张量 non_contig，是 contig 的每隔一个元素切片
        non_contig = contig[::2]

        # 断言 contig 是连续的，non_contig 是非连续的
        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        # 为操作 op 生成样本参数
        torch_kwargs, _ = op.sample_kwargs(device, dtype, non_contig)
        # 对 non_contig 进行操作 op，得到期望的结果
        expected = op(non_contig, **torch_kwargs)
        # 对 contig 进行操作 op，得到实际结果
        result = op(contig, **torch_kwargs)
        # 将实际结果按照与 non_contig 相同的方式进行映射
        result = pytree.tree_map(lambda x: x[::2], result)
        # 断言结果相等
        self.assertEqual(result, expected)

    # 应用 @ops 装饰器，指定操作集合为 unary_ufuncs
    def test_contig_vs_transposed(self, device, dtype, op):
        # 创建连续张量 contig
        contig = make_tensor(
            (789, 357), device=device, dtype=dtype, low=op.domain[0], high=op.domain[1]
        )
        # 创建非连续张量 non_contig，是 contig 的转置
        non_contig = contig.T

        # 断言 contig 是连续的，non_contig 是非连续的
        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        # 为操作 op 生成样本参数
        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        # 对 non_contig 进行操作 op，得到期望的结果
        expected = op(non_contig, **torch_kwargs)
        # 对 contig 进行操作 op，得到实际结果
        result = op(contig, **torch_kwargs)
        # 将实际结果按照与 non_contig 相同的方式进行映射
        result = pytree.tree_map(lambda x: x.T, result)
        # 断言结果相等
        self.assertEqual(result, expected)

    # 应用 @ops 装饰器，指定操作集合为 unary_ufuncs
    def test_non_contig(self, device, dtype, op):
        # 需要测试的形状列表
        shapes = [(5, 7), (1024,)]
        for shape in shapes:
            # 创建连续张量 contig
            contig = make_tensor(
                shape, dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
            )
            # 创建非连续张量 non_contig，从 contig 复制而来
            non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
            non_contig.copy_(contig)

            # 断言 contig 是连续的，non_contig 是非连续的
            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            # 为操作 op 生成样本参数
            torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
            # 断言对 contig 和 non_contig 应用 op 的结果相等
            self.assertEqual(op(contig, **torch_kwargs), op(non_contig, **torch_kwargs))
    # 测试非连续索引操作的函数
    def test_non_contig_index(self, device, dtype, op):
        # 创建一个连续的张量
        contig = make_tensor(
            (2, 2, 1, 2),
            dtype=dtype,
            device=device,
            low=op.domain[0],
            high=op.domain[1],
        )
        # 使用非连续的索引操作获取子张量
        non_contig = contig[:, 1, ...]
        # 将非连续的张量变成连续的
        contig = non_contig.contiguous()

        # 断言张量是否为连续的
        self.assertTrue(contig.is_contiguous())
        # 断言非连续的张量是否为非连续的
        self.assertFalse(non_contig.is_contiguous())

        # 生成操作所需的参数字典和其他信息
        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        # 断言对连续张量和非连续张量进行相同操作的结果是否一致
        self.assertEqual(op(contig, **torch_kwargs), op(non_contig, **torch_kwargs))

    # 使用装饰器 ops(unary_ufuncs)标记的测试函数，测试非连续扩展操作
    @ops(unary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        # 定义不同形状的张量进行测试
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            # 创建一个连续的张量
            contig = make_tensor(
                shape, dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
            )
            # 克隆张量并使用 expand 扩展为非连续的
            non_contig = contig.clone().expand(3, -1, -1)

            # 断言张量是否为连续的
            self.assertTrue(contig.is_contiguous())
            # 断言非连续的张量是否为非连续的
            self.assertFalse(non_contig.is_contiguous())

            # 生成操作所需的参数字典和其他信息
            torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
            # 对连续张量和非连续张量进行操作，并断言它们的结果是否相等
            contig = op(contig, **torch_kwargs)
            non_contig = op(non_contig, **torch_kwargs)
            for i in range(3):
                non_contig_i = pytree.tree_map(lambda x: x[i], non_contig)
                self.assertEqual(
                    contig, non_contig_i, msg="non-contiguous expand[" + str(i) + "]"
                )

    # 使用装饰器 ops(unary_ufuncs)标记的测试函数，测试连续大小为1的情况
    @ops(unary_ufuncs)
    def test_contig_size1(self, device, dtype, op):
        # 创建一个连续的张量
        contig = make_tensor(
            (5, 100), dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
        )
        # 对张量进行切片操作，使其变为连续大小为1
        contig = contig[:1, :50]
        # 创建一个与 contig 大小相同的空张量，并复制 contig 的数据
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        # 断言张量是否为连续的
        self.assertTrue(contig.is_contiguous())
        # 断言 contig2 是否为连续的
        self.assertTrue(contig2.is_contiguous())

        # 生成操作所需的参数字典和其他信息
        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        # 对连续张量和 contig2 进行操作，并断言它们的结果是否相等
        self.assertEqual(op(contig, **torch_kwargs), op(contig2, **torch_kwargs))

    # 使用装饰器 ops(unary_ufuncs)标记的测试函数，测试连续大小为1的情况（大维度）
    @ops(unary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        # 创建一个大维度的连续张量
        contig = make_tensor(
            (5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4),
            dtype=dtype,
            device=device,
            low=op.domain[0],
            high=op.domain[1],
        )
        # 对张量进行切片操作，使其变为连续大小为1
        contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
        # 创建一个与 contig 大小相同的空张量，并复制 contig 的数据
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        # 断言张量是否为连续的
        self.assertTrue(contig.is_contiguous())
        # 断言 contig2 是否为连续的
        self.assertTrue(contig2.is_contiguous())

        # 生成操作所需的参数字典和其他信息
        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        # 对连续张量和 contig2 进行操作，并断言它们的结果是否相等
        self.assertEqual(op(contig, **torch_kwargs), op(contig2, **torch_kwargs))

    # 该测试函数检验多批次计算与逐批次计算的结果是否一致
    @ops(unary_ufuncs)
    # 定义一个测试方法，用于测试批处理与切片的操作
    def test_batch_vs_slicing(self, device, dtype, op):
        # 创建一个指定大小的张量作为输入数据
        input = make_tensor(
            (1024, 512), dtype=dtype, device=device, low=op.domain[0], high=op.domain[1]
        )

        # 从操作对象中获取用于调用操作的关键字参数和其它信息
        torch_kwargs, _ = op.sample_kwargs(device, dtype, input)
        # 使用输入数据调用操作，获取实际输出
        actual = op(input, **torch_kwargs)

        # 对输入数据进行切片操作，分别调用操作，并将结果存储在列表中
        all_outs = [op(slice, **torch_kwargs) for slice in input]
        # 如果实际输出是张量的可迭代对象，则将所有切片的结果堆叠成张量列表
        if is_iterable_of_tensors(actual):
            expected = [torch.stack([out[i] for out in all_outs]) for i in range(len(actual))]
        else:
            # 否则直接将所有切片的结果堆叠成一个张量
            expected = torch.stack(all_outs)

        # 断言实际输出与预期输出相等
        self.assertEqual(actual, expected)

    # 使用指定的数据类型测试 nan_to_num 方法
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half))
    def test_nan_to_num(self, device, dtype):
        # 遍历是否连续存储的标志
        for contiguous in [False, True]:
            # 创建一个指定大小的张量，并设置其值在指定范围内
            x = make_tensor((64, 64), low=0.0, high=100.0, dtype=dtype, device=device)

            # 如果数据类型是浮点型
            if dtype.is_floating_point:
                # 添加极端值
                extremals = [float("nan"), float("inf"), -float("inf")]
                # 将极端值设置到指定的行上
                for idx, extremal in zip(torch.randint(0, 63, (3,)), extremals):
                    x[idx, :] = extremal

            # 如果数据不是连续存储，则对其进行转置
            if not contiguous:
                x = x.T

            # 使用随机生成的参数值进行测试 nan_to_num 方法
            nan = random.random()
            posinf = random.random() * 5
            neginf = random.random() * 10

            # 使用自定义的 compare_with_numpy 方法，分别与 numpy 中的处理结果进行比较
            self.compare_with_numpy(
                lambda x: x.nan_to_num(nan=nan, posinf=posinf),
                lambda x: np.nan_to_num(x, nan=nan, posinf=posinf),
                x,
            )
            self.compare_with_numpy(
                lambda x: x.nan_to_num(posinf=posinf, neginf=neginf),
                lambda x: np.nan_to_num(x, posinf=posinf, neginf=neginf),
                x,
            )

            # 创建一个与输入张量相同大小的空张量 out
            out = torch.empty_like(x)
            # 使用 torch 的 nan_to_num 方法对输入张量进行处理，并将结果存储到 result 中
            result = torch.nan_to_num(x)
            # 将处理结果存储到指定的 out 张量中
            torch.nan_to_num(x, out=out)
            # 断言处理结果与 out 张量相等
            self.assertEqual(result, out)

            # 使用指定的参数对输入张量进行处理，并将结果存储到 out 张量中
            result = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
            torch.nan_to_num(x, out=out, nan=nan, posinf=posinf, neginf=neginf)
            # 断言处理结果与 out 张量相等
            self.assertEqual(result, out)
    # 定义一个测试函数，测试 torch.nan_to_num 在 bfloat16 数据类型上的行为
    def test_nan_to_num_bfloat16(self, device):
        
        # 定义一个内部函数，用于测试指定数据类型下的函数行为
        def test_dtype(fn, input, dtype):
            # 分离输入张量，克隆并转换数据类型，并要求梯度计算
            input = input.detach().clone().to(dtype=dtype).requires_grad_(True)
            # 克隆输入张量并转换为 float 数据类型，并要求梯度计算
            input2 = input.detach().clone().float().requires_grad_(True)
            # 对 fn 应用于 input 的结果进行计算，并对其求和进行反向传播
            out = fn(input)
            out.sum().backward()
            # 对 fn 应用于 input2 的结果进行计算，并对其求和进行反向传播
            out2 = fn(input2)
            out2.sum().backward()
            # 断言输出的数据类型与指定的数据类型一致
            self.assertEqual(out.dtype, dtype)
            # 断言输入的梯度的数据类型与指定的数据类型一致
            self.assertEqual(input.grad.dtype, dtype)
            # 断言两个输出张量的内容相等，不需要精确的数据类型匹配
            self.assertEqual(out, out2, exact_dtype=False)
            # 断言两个输入张量的梯度相等，不需要精确的数据类型匹配
            self.assertEqual(input.grad, input2.grad, exact_dtype=False)

        # 定义一个返回 torch.nan_to_num 函数的函数
        def func():
            return torch.nan_to_num

        # 定义不同形状的张量作为测试数据
        shapes = [[1, 3, 6, 6], [1, 3, 6, 128], [1, 3, 256, 256]]
        # 遍历不同形状的张量
        for shape in shapes:
            # 创建一个指定形状的随机张量，并分配到指定设备
            x = torch.randn(shape, device=device)
            # 定义极值数据，包括 nan、inf 和 -inf
            extremals = [float('nan'), float('inf'), -float('inf')]
            # 遍历极值数据，并将其赋值给随机张量 x 的指定位置
            for id1, id2, extremal in zip(torch.randint(0, 2, (3,)), torch.randint(0, 5, (3,)), extremals):
                x[0, id1, id2, :] = extremal
            # 使用 func 函数对 x 进行类型转换和 nan 处理后，与 torch.bfloat16 进行测试
            test_dtype(func(), x, torch.bfloat16)

    # 使用 @dtypes 装饰器，测试 torch.nan_to_num 在复数类型数据上的行为
    @dtypes(torch.complex64, torch.complex128)
    def test_nan_to_num_complex(self, device, dtype):
        # 获取值数据类型的实部数据类型
        value_dtype = torch.tensor([], dtype=dtype).real.dtype

        # 定义一个生成复数张量的函数
        def gen_tensor(a):
            return torch.view_as_complex(torch.tensor(a, dtype=value_dtype, device=device))

        # 定义极值和关键字名称的对应关系
        for extremal, kwarg_name in zip(['nan', 'inf', '-inf'], ['nan', 'posinf', 'neginf']):
            # 根据给定的极值创建复数张量 a，并使用 torch.nan_to_num 进行处理
            a = gen_tensor([123, float(extremal)])
            res = torch.nan_to_num(a, **{kwarg_name: 12})
            res_check = gen_tensor([123, 12])
            # 断言处理后的结果与预期结果相等
            self.assertEqual(res, res_check)

            # 根据给定的极值创建复数张量 a，并使用 torch.nan_to_num 进行处理
            a = gen_tensor([float(extremal), 456])
            res = torch.nan_to_num(a, **{kwarg_name: 21})
            res_check = gen_tensor([21, 456])
            # 断言处理后的结果与预期结果相等
            self.assertEqual(res, res_check)

    # 使用 @dtypes 装饰器，测试 torch.sqrt 和 torch.acos 在边缘值情况下的表现
    @dtypes(torch.cdouble)
    def test_complex_edge_values(self, device, dtype):
        # 对于指定的复数张量 x 进行测试，根据不同情况调用 torch.sqrt 或 torch.acos
        x = torch.tensor(0.0 - 1.0e20j, dtype=dtype, device=device)
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)
        # 在 Windows 上跳过 torch.acos 测试，因为 CUDA 返回共轭值，参见 issue 链接
        if not (IS_WINDOWS and dtype == torch.cdouble and "cuda" in device):
            self.compare_with_numpy(torch.acos, np.arccos, x)

        # 对于指定的复数张量 x 进行测试，根据不同情况调用 torch.sqrt 或 torch.acos
        x = torch.tensor(
            (-1.0e60 if dtype == torch.cdouble else -1.0e20) - 4988429.2j,
            dtype=dtype,
            device=device,
        )
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)

    # 使用 unittest.skipIf 装饰器，如果未安装 SciPy 则跳过测试
    @unittest.skipIf(not TEST_SCIPY, "Requires SciPy")
    @dtypes(torch.float, torch.double)
    # 测试 torch.digamma 函数在特殊情况下的行为，基于 SciPy 的测试用例
    def test_digamma_special(self, device, dtype):
        # 欧拉常数
        euler = 0.57721566490153286
        # 定义包含特殊值的数据集
        dataset = [
            (0.0, -0.0),
            (1, -euler),
            (0.5, -2 * math.log(2) - euler),
            (1 / 3, -math.pi / (2 * math.sqrt(3)) - 3 * math.log(3) / 2 - euler),
            (1 / 4, -math.pi / 2 - 3 * math.log(2) - euler),
            (
                1 / 6,
                -math.pi * math.sqrt(3) / 2
                - 2 * math.log(2)
                - 3 * math.log(3) / 2
                - euler,
            ),
            (
                1 / 8,
                -math.pi / 2
                - 4 * math.log(2)
                - (math.pi + math.log(2 + math.sqrt(2)) - math.log(2 - math.sqrt(2)))
                / math.sqrt(2)
                - euler,
            ),
        ]
        # 创建 tensor，并指定设备和数据类型
        x = torch.tensor(dataset, device=device, dtype=dtype)
        # 使用 self.compare_with_numpy 方法比较 torch.digamma 和 scipy.special.digamma 的结果

    # 如果未安装 SciPy，则跳过该测试
    @unittest.skipIf(not TEST_SCIPY, "Requires SciPy")
    @dtypes(torch.float, torch.double)
    def test_digamma(self, device, dtype):
        # 测试极点行为
        tensor = torch.tensor(
            [
                -0.999999994,
                -1.999999994,
                -2.0000000111,
                -100.99999994,
                0.000000111,
                -1931.99999994,
                -0.000000111,
                0,
                -0,
                -1,
                -2,
                -931,
            ],
            dtype=dtype,
            device=device,
        )
        # 使用 self.compare_with_numpy 方法比较 torch.digamma 和 scipy.special.digamma 的结果

    # 测试 torch.frexp 函数的行为
    @dtypes(*floating_types_and(torch.half))
    def test_frexp(self, device, dtype):
        # 创建指定设备和数据类型的输入 tensor
        input = make_tensor((50, 50), dtype=dtype, device=device)
        # 调用 torch.frexp 分解输入 tensor 成尾数和指数
        mantissa, exponent = torch.frexp(input)
        # 使用 numpy 进行同样的操作
        np_mantissa, np_exponent = np.frexp(input.cpu().numpy())

        # 断言 torch.frexp 的输出与 numpy 的输出相等
        self.assertEqual(mantissa, np_mantissa)
        self.assertEqual(exponent, np_exponent)

        # torch.frexp 返回的指数类型为 int32，以保持与 np.frexp 的兼容性
        self.assertTrue(exponent.dtype == torch.int32)
        self.assertTrue(torch_to_numpy_dtype_dict[exponent.dtype] == np_exponent.dtype)
    # 定义一个测试函数，用于验证 torch.frexp 在不支持的数据类型上是否会引发异常
    def test_frexp_assert_raises(self, device):
        # 列出所有不支持的输入数据类型，包括整数类型和复数类型
        invalid_input_dtypes = integral_types_and(torch.bool) + complex_types()
        # 遍历每种不支持的数据类型
        for dtype in invalid_input_dtypes:
            # 生成一个指定数据类型和设备的张量作为输入
            input = make_tensor((50, 50), dtype=dtype, device=device)
            # 断言调用 torch.frexp(input) 会引发 RuntimeError，并检查异常消息是否包含特定文本
            with self.assertRaisesRegex(
                RuntimeError, r"torch\.frexp\(\) only supports floating-point dtypes"
            ):
                torch.frexp(input)

        # 针对所有浮点数数据类型（包括半精度浮点数）
        for dtype in floating_types_and(torch.half):
            # 生成一个指定数据类型和设备的张量作为输入
            input = make_tensor((50, 50), dtype=dtype, device=device)

            # 列出除当前数据类型外的所有数据类型，并创建与输入张量相同形状的空张量作为输出
            dtypes = list(
                all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16)
            )
            dtypes.remove(dtype)
            # 遍历每种不支持的数据类型，用作 mantissa 的数据类型
            for mantissa_dtype in dtypes:
                mantissa = torch.empty_like(input, dtype=mantissa_dtype)
                exponent = torch.empty_like(input, dtype=torch.int)
                # 断言调用 torch.frexp(input, out=(mantissa, exponent)) 会引发 RuntimeError，并检查异常消息是否包含特定文本
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"torch\.frexp\(\) expects mantissa to have dtype .+ but got .+",
                ):
                    torch.frexp(input, out=(mantissa, exponent))

            dtypes.append(dtype)
            dtypes.remove(torch.int)
            # 遍历每种不支持的数据类型，用作 exponent 的数据类型
            for exponent_dtype in dtypes:
                mantissa = torch.empty_like(input)
                exponent = torch.empty_like(input, dtype=exponent_dtype)
                # 断言调用 torch.frexp(input, out=(mantissa, exponent)) 会引发 RuntimeError，并检查异常消息是否包含特定文本
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"torch\.frexp\(\) expects exponent to have int dtype but got .+",
                ):
                    torch.frexp(input, out=(mantissa, exponent))

    # 验证 torch.polygamma 在 n 为负数时是否会引发异常
    def test_polygamma_neg(self, device):
        with self.assertRaisesRegex(
            RuntimeError, r"polygamma\(n, x\) does not support negative n\."
        ):
            torch.polygamma(-1, torch.tensor([1.0, 2.0], device=device))

    # 使用 @onlyCPU 装饰器标记的测试函数，测试按位取反操作的正确性
    @onlyCPU
    def test_op_invert(self, device):
        # 计算 0xFFFF - [0, 1, 2, ..., 126]，结果作为参考值 res
        res = 0xFFFF - torch.arange(127, dtype=torch.int8)
        # 遍历每种数据类型（uint8, int8, int16, int32, int64），生成对应的输入张量 a
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            a = torch.arange(127, dtype=dtype)
            # 断言取反操作 (~a) 的结果与参考值 res 转换到当前数据类型后是否一致
            self.assertEqual(res.to(dtype), ~a)

        # 断言对布尔张量进行取反操作时的预期结果
        self.assertEqual(torch.tensor([True, False]), ~torch.tensor([False, True]))

        # 针对浮点数数据类型（half, float, double），验证取反操作是否引发 TypeError 异常
        for dtype in (torch.half, torch.float, torch.double):
            a = torch.zeros(10, dtype=dtype)
            # 断言尝试对浮点数张量执行取反操作时会引发 TypeError
            with self.assertRaises(TypeError):
                b = ~a

    # 使用 @dtypes 装饰器标记的测试函数，用于检查内部内存重叠情况
    def check_internal_mem_overlap(
        self, inplace_op, num_inputs, dtype, device, expected_failure=False
        # TODO resolve with opinfos
    ):
        # 如果 inplace_op 是字符串，则将其转换为对应的 torch.Tensor 方法
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        
        # 创建一个随机的张量作为输入，数据类型为 dtype，设备为 device，并扩展为 3x3 的张量
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        
        # 创建输入列表，包含一个上面创建的 input 张量，以及 num_inputs - 1 个与 input 相同大小的随机张量
        inputs = [input] + [torch.randn_like(input) for i in range(num_inputs - 1)]
        
        # 如果不是预期的失败情况
        if not expected_failure:
            # 使用 assertRaisesRegex 检查 RuntimeError 异常，确保抛出的异常信息包含 "single memory location"
            with self.assertRaisesRegex(RuntimeError, "single memory location"):
                inplace_op(*inputs)
        else:
            # 使用 assertRaises 检查 AssertionError 异常
            with self.assertRaises(AssertionError):
                # 嵌套使用 assertRaisesRegex 检查 RuntimeError 异常，确保抛出的异常信息包含 "single memory location"
                with self.assertRaisesRegex(RuntimeError, "single memory location"):
                    inplace_op(*inputs)

    def unary_check_input_output_mem_overlap(
        self, data, sz, op, expected_failure=False
    ):
        def _test(op, output, input):
            # 创建一个和 output 张量相同大小的空张量 output_exp
            output_exp = torch.empty_like(output)
            # 调用 op 函数，将 input 张量作为输入，output_exp 张量作为输出
            op(input, out=output_exp)
            # 使用 self.assertEqual 检查 op 函数的输出和 output_exp 张量是否相等，msg 参数为 op 函数的名称
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output 等同于 input:
        _test(op, output=data[0:sz], input=data[0:sz])
        
        # output 和 input 是独立的:
        _test(op, output=data[0:sz], input=data[sz : 2 * sz])
        
        # output 部分重叠于 input:
        if not expected_failure:
            # 使用 assertRaisesRegex 检查 RuntimeError 异常，确保抛出的异常信息包含 "unsupported operation"
            with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                _test(op, data[0:sz], data[1 : sz + 1])
        else:
            # 使用 assertRaises 检查 AssertionError 异常
            with self.assertRaises(AssertionError):
                # 嵌套使用 assertRaisesRegex 检查 RuntimeError 异常，确保抛出的异常信息包含 "unsupported operation"
                with self.assertRaisesRegex(RuntimeError, "unsupported operation"):
                    _test(op, data[0:sz], data[1 : sz + 1])

    # TODO: run on non-native device types
    # https://github.com/pytorch/pytorch/issues/126474
    @xfailIfTorchDynamo
    @dtypes(torch.double)
    # TODO: opinfo hardshrink
    @onlyCPU
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardshrink(self, device, dtype):
        # 创建一个张量 data，数据为 [1, 0.5, 0.3, 0.6]，数据类型为 dtype，设备为 device，并将其视图调整为 2x2
        data = torch.tensor([1, 0.5, 0.3, 0.6], dtype=dtype, device=device).view(2, 2)
        
        # 使用 self.assertEqual 检查 hardshrink 函数的输出是否符合预期结果，msg 参数为 hardshrink 函数的名称
        self.assertEqual(
            torch.tensor([1, 0.5, 0, 0.6], dtype=dtype, device=device).view(2, 2),
            data.hardshrink(0.3),
        )
        
        # 使用 self.assertEqual 检查 hardshrink 函数的输出是否符合预期结果，msg 参数为 hardshrink 函数的名称
        self.assertEqual(
            torch.tensor([1, 0, 0, 0.6], dtype=dtype, device=device).view(2, 2),
            data.hardshrink(0.5),
        )

        # 测试默认的 lambd=0.5 情况下的 hardshrink 函数
        self.assertEqual(data.hardshrink(), data.hardshrink(0.5))

        # 测试非连续情况下的 hardshrink 函数
        self.assertEqual(
            torch.tensor([1, 0, 0.5, 0.6], dtype=dtype, device=device).view(2, 2),
            data.t().hardshrink(0.3),
        )

    @onlyCPU
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardshrink_edge_cases(self, device, dtype) -> None:
        # 定义内部函数 h，用于测试 hardshrink 方法的边界情况
        def h(values, l_expected):
            # 遍历给定的 l_expected 字典，每个键值对包含期望的收缩阈值 l 和对应的期望输出 expected
            for l, expected in l_expected.items():
                # 将输入的数值列表转换为 PyTorch 张量 values_tensor，并指定设备和数据类型
                values_tensor = torch.tensor(
                    [float(v) for v in values], dtype=dtype, device=device
                )
                # 将期望的输出列表转换为 PyTorch 张量 expected_tensor，并指定设备和数据类型
                expected_tensor = torch.tensor(
                    [float(v) for v in expected], dtype=dtype, device=device
                )
                # 使用 hardshrink 方法对 values_tensor 进行收缩，并与期望的输出进行断言比较
                self.assertEqual(
                    expected_tensor == values_tensor.hardshrink(l),
                    torch.ones_like(values_tensor, dtype=torch.bool),
                )

        # 定义测试助手函数 test_helper，测试不同的最小值和最大值情况
        def test_helper(min, max):
            # 调用 h 函数进行测试，传入不同的最小值和最大值情况
            h(
                [0.0, min, -min, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
                {
                    0.0: [0.0, min, -min, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
                    min: [0.0, 0.0, 0.0, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
                    0.1: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, max, -max, inf, -inf],
                    1.0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, max, -max, inf, -inf],
                    max: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, -inf],
                    inf: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            )

        # 调用测试助手函数 test_helper，传入具体的数据类型的最小值和最大值
        test_helper(torch.finfo(dtype).tiny, torch.finfo(dtype).max)

    @onlyCPU
    @slowTest
    @dtypes(torch.float)
    @unittest.skipIf(True, "Insufficient memory on linux.(2|4)xlarge")
    def test_exp_slow(self, device, dtype):
        # 对 https://github.com/pytorch/pytorch/issues/17271 进行测试
        # 在我的 MacBook 上速度比较慢，但在强大的 Xeon 服务器上只需要几秒钟
        a = torch.exp(torch.ones(2**31, dtype=dtype, device=device))
        b = torch.exp(torch.ones(1, dtype=dtype, device=device))
        # 使用 expand 方法比较张量 a 和 b，期望它们相等
        self.assertEqual(a, b.expand(2**31))

    @precisionOverride(
        {torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002}
    )
    @dtypes(torch.float, torch.double, torch.bfloat16)
    def test_hardswish(self, device, dtype):
        # 定义输入数值列表和期望输出数组的乘积
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        expectedOutput = np.multiply(
            inputValues, np.minimum(np.maximum((np.add(inputValues, 3)), 0), 6) / 6.0
        )

        # 将输入数值列表转换为 PyTorch 张量 inputTensor，并指定设备和数据类型
        inputTensor = torch.tensor(inputValues, dtype=dtype, device=device)
        # 将期望输出数组转换为 PyTorch 张量 expectedOutputTensor，并指定设备和数据类型
        expectedOutputTensor = torch.tensor(expectedOutput, dtype=dtype, device=device)

        # 测试 torch.nn.functional.hardswish 函数的正常使用情况
        self.assertEqual(
            torch.nn.functional.hardswish(inputTensor), expectedOutputTensor
        )

        # 测试 torch.nn.functional.hardswish 函数的原位操作
        inputTensorCpy = inputTensor.clone().detach()
        torch.nn.functional.hardswish(inputTensorCpy, inplace=True)
        # 断言原位操作后的张量与期望的输出张量相等
        self.assertEqual(inputTensorCpy, expectedOutputTensor)
    # 定义测试函数 test_hardsigmoid，用于测试 torch.nn.functional.hardsigmoid 函数
    def test_hardsigmoid(self, device, dtype):
        # 输入的测试数值列表
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        # 预期的输出值，经过 hardsigmoid 函数处理后的结果
        expectedOutput = np.minimum(np.maximum((np.add(inputValues, 3)), 0), 6) / 6.0

        # 创建输入的 Tensor，指定设备和数据类型
        inputTensor = torch.tensor(inputValues, dtype=dtype, device=device)

        # 测试正常调用 hardsigmoid 函数
        self.assertEqual(
            torch.nn.functional.hardsigmoid(inputTensor),
            torch.tensor(expectedOutput, dtype=dtype, device=device),
        )

        # 测试 inplace 模式下的 hardsigmoid 函数调用
        inputTensorCpy = inputTensor.clone().detach()
        self.assertEqual(
            torch.nn.functional.hardsigmoid(inputTensorCpy, inplace=True),
            torch.tensor(expectedOutput, dtype=dtype, device=device),
        )

    # 使用 precisionOverride 装饰器设置不同数据类型的精度要求
    @precisionOverride(
        {torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002}
    )
    # 使用 dtypes 装饰器指定测试的数据类型
    @dtypes(torch.float, torch.double, torch.bfloat16)
    # 定义测试函数 test_hardsigmoid_backward，用于测试 hardsigmoid 函数的反向传播
    def test_hardsigmoid_backward(self, device, dtype):
        # 输入的测试数值列表
        inputValues = [-3.0, 3.0, -2.0, 2.0, -6.0, 6.0]
        # 预期的输出梯度值
        expectedValues = [0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0]
        # 创建输入的 Tensor，并开启梯度追踪
        inputTensor = torch.tensor(
            inputValues, dtype=dtype, device=device
        ).requires_grad_()
        # 创建预期的梯度 Tensor
        expectedTensor = torch.tensor(expectedValues, dtype=dtype, device=device)
        # 对输入 Tensor 应用 hardsigmoid 函数
        out = torch.nn.functional.hardsigmoid(inputTensor)
        # 计算反向传播
        out.backward(torch.ones_like(inputTensor))
        # 断言输入 Tensor 的梯度与预期的梯度相等
        self.assertEqual(inputTensor.grad, expectedTensor)

    # 使用 skipIfNoSciPy 装饰器，跳过若没有 SciPy 的情况下的测试
    @skipIfNoSciPy
    # 使用 dtypes 装饰器指定测试的数据类型
    @dtypes(torch.float, torch.double)
    # 定义测试函数 test_silu，用于测试 torch.nn.functional.silu 函数
    def test_silu(self, device, dtype):
        # 创建一个随机的 numpy 数组作为输入
        input_np = np.random.randn(5, 8)
        # 创建一个特殊的输入列表
        special_input = [[-1000, -1, -0.1, 0, 0.5, 1, 2, 1000]]
        # 将特殊输入连接到原始输入数组中，并转换为指定的数据类型
        input_np = np.concatenate((input_np, special_input), axis=0).astype(
            torch_to_numpy_dtype_dict[dtype]
        )
        # 期望的输出值，经过 silu 函数处理后的结果
        expected_output_np = input_np * scipy.special.expit(input_np)

        # 创建期望的输出 Tensor，并将其移动到指定设备
        expected_output = torch.from_numpy(expected_output_np).to(device)
        # 创建非连续的期望输出 Tensor
        expected_output_noncontig = expected_output.transpose(0, 1)

        # 设置绝对误差和相对误差的容忍度
        atol = 1e-6
        rtol = 1e-6

        # 创建连续的输入 Tensor，并将其移动到指定设备
        input = torch.from_numpy(input_np).clone().contiguous().to(device)
        # 测试正常调用 silu 函数
        self.assertEqual(
            torch.nn.functional.silu(input), expected_output, atol=atol, rtol=rtol
        )
        # 测试 inplace 模式下的 silu 函数调用
        self.assertEqual(
            torch.nn.functional.silu(input, inplace=True),
            expected_output,
            atol=atol,
            rtol=rtol,
        )

        # 创建非连续的输入 Tensor，并将其移动到指定设备
        input = torch.from_numpy(input_np).clone().to(device)
        input_noncontig = input.transpose(0, 1)
        # 测试非连续输入的 silu 函数调用
        self.assertEqual(
            torch.nn.functional.silu(input_noncontig),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )
        # 测试 inplace 模式下非连续输入的 silu 函数调用
        self.assertEqual(
            torch.nn.functional.silu(input_noncontig, inplace=True),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )

    # 使用 dtypes 装饰器指定测试的数据类型为复数类型
    @dtypes(torch.complex64, torch.complex128)
    # 定义一个测试方法，用于测试 torch.nn.functional.silu 函数的复杂输入情况
    def test_silu_complex(self, device, dtype):
        # 定义绝对误差和相对误差的阈值
        atol = 1e-6
        rtol = 1e-6
        # 定义多组复数输入及其预期输出
        inouts = [
            (0.2 + 0.3j, 0.08775215595960617065 + 0.18024823069572448730j),
            (1e-19 + 1e-18j, 4.99999984132761269448e-20 + 5.00000022906852482872e-19j),
            (-1.0 + 2.0j, -0.78546208143234252930 - 0.44626939296722412109j),
            (0.0 + 0.5j, -0.06383547931909561157 + 0.25000000000000000000j),
            (2.0j, -1.55740761756896972656 + 0.99999988079071044922j)
        ]

        # 对每一组输入输出进行测试
        for inp, out in inouts:
            # 调用 silu 函数计算结果
            res = torch.nn.functional.silu(torch.tensor(inp, dtype=dtype, device=device))
            # 断言结果中没有 NaN 值
            self.assertFalse(torch.any(torch.isnan(res)))
            # 检查实部是否满足预期值的实数部分
            self.assertEqual(res.real, out.real, atol=atol, rtol=rtol)
            # 检查实部是否满足预期值的虚数部分
            self.assertEqual(res.imag, out.imag, atol=atol, rtol=rtol)

        # 对每一组输入输出进行 inplace 操作的测试
        for inp, out in inouts:
            # 调用 silu 函数进行 inplace 操作，计算结果
            res = torch.nn.functional.silu(torch.tensor(inp, dtype=dtype, device=device), inplace=True)
            # 断言结果中没有 NaN 值
            self.assertFalse(torch.any(torch.isnan(res)))
            # 检查实部是否满足预期值的实数部分
            self.assertEqual(res.real, out.real, atol=atol, rtol=rtol)
            # 检查实部是否满足预期值的虚数部分
            self.assertEqual(res.imag, out.imag, atol=atol, rtol=rtol)

    # 由于这些输入对于 gradcheck 是成功的，但对于 gradgradcheck 预期是失败的，
    # 所以不明显如何将其合并到 OpInfo 中。
    @dtypes(torch.double)
    def test_sinc(self, device, dtype):
        # 在 x=0 处 sinc(x) 的导数需要特殊处理。
        # 一个简单的计算将会得到 0/0 -> NaN。
        # 我们还需要在接近 0 的情况下特别小心，因为导数的分母是平方的，
        # 有些浮点数是正数，其平方是零。
        a = torch.tensor(
            [0.0, torch.finfo(torch.double).tiny, 1.0],
            dtype=dtype,
            requires_grad=True,
            device=device,
        )
        # 对 sinc 函数进行梯度检查
        gradcheck(torch.sinc, a)

    # 如果没有安装 SciPy，跳过这个测试
    @skipIfNoSciPy
    @dtypes(torch.float, torch.double)
    # 定义测试方法 `test_mish`，用于测试 `torch.nn.functional.mish` 函数的不同输入情况
    def test_mish(self, device, dtype):
        # 生成一个形状为 (5, 8) 的随机正态分布数组作为输入
        input_np = np.random.randn(5, 8)
        # 添加一个特殊输入列表到输入数组的末尾，并转换为指定的 PyTorch 数据类型
        special_input = [[-1000, -1, -0.1, 0, 0.5, 1, 2, 1000]]
        input_np = np.concatenate((input_np, special_input), axis=0).astype(
            torch_to_numpy_dtype_dict[dtype]
        )
        # 计算预期输出，使用 Mish 激活函数的定义
        expected_output_np = input_np * np.tanh(np.log1p(np.exp(input_np)))

        # 将预期输出转换为 PyTorch 张量，并发送到指定的计算设备
        expected_output = torch.from_numpy(expected_output_np).to(device)
        # 转置预期输出张量的第一和第二维度
        expected_output_noncontig = expected_output.transpose(0, 1)

        # 定义数值误差的绝对和相对容差
        atol = 1e-6
        rtol = 1e-6

        # 克隆输入数组为 PyTorch 张量，并确保其在内存中是连续的，然后发送到指定的计算设备
        input = torch.from_numpy(input_np).clone().contiguous().to(device)
        # 使用 `torch.nn.functional.mish` 函数计算输出，并断言其与预期输出相等
        self.assertEqual(
            torch.nn.functional.mish(input), expected_output, atol=atol, rtol=rtol
        )
        # 在原地使用 `torch.nn.functional.mish` 函数计算输出，并断言其与预期输出相等
        self.assertEqual(
            torch.nn.functional.mish(input, inplace=True),
            expected_output,
            atol=atol,
            rtol=rtol,
        )

        # 克隆输入数组为 PyTorch 张量，并发送到指定的计算设备，不保证其在内存中是连续的
        input = torch.from_numpy(input_np).clone().to(device)
        # 转置输入张量的第一和第二维度
        input_noncontig = input.transpose(0, 1)
        # 使用 `torch.nn.functional.mish` 函数计算非连续输入的输出，并断言其与预期输出相等
        self.assertEqual(
            torch.nn.functional.mish(input_noncontig),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )
        # 在原地使用 `torch.nn.functional.mish` 函数计算非连续输入的输出，并断言其与预期输出相等
        self.assertEqual(
            torch.nn.functional.mish(input_noncontig, inplace=True),
            expected_output_noncontig,
            atol=atol,
            rtol=rtol,
        )

    # 定义一个装饰器 `@dtypes`，用于指定复数类型的测试参数
    @dtypes(torch.complex64, torch.complex128)
    def test_log1p_complex(self, device, dtype):
        # 这里的输出值是使用任意精度数学（mpmath）获得的，并且使用WolframAlpha进行了双重检查。
        # 在此处不使用numpy的log1p，因为到撰写本文时，np.log1p在小复数输入值上存在精度问题，请参见：
        # https://github.com/numpy/numpy/issues/22609
        inouts = [
            (0.2 + 0.3j, 0.21263386770217202 + 0.24497866312686414j),
            (1e-19 + 1e-18j, 1e-19 + 1e-18j),
            (1e-18 + 0.1j, 0.00497517 + 0.0996687j),
            (0.1 + 1e-18j, 0.0953102 + 9.090909090909090909e-19j),
            (0.5 + 0j, 0.40546510810816 + 0j),
            (0.0 + 0.5j, 0.111571776 + 0.463647609j),
            (2.0 + 1.0j, 1.151292546497023 + 0.3217505543966422j),
            (-1.0 + 2.0j, 0.6931471805599453 + 1.570796326794897j),
            (2.0j, 0.80471895621705014 + 1.1071487177940904j),
            (-2.0j, 0.80471895621705014 - 1.1071487177940904j),
        ]
        # 测试极端值
        if dtype == torch.complex128:
            inouts += [
                (-1 + 1e250j, 575.6462732485114 + 1.5707963267948966j),
                (1e250 + 1j, 575.6462732485114 + 1e-250j),
                (1e250 + 1e250j, 575.9928468387914 + 0.7853981633974483j),
                (1e-250 + 1e250j, 575.6462732485114 + 1.5707963267948966j),
                (1e-250 + 2e-250j, 1e-250 + 2e-250j),
                (1e250 + 1e-250j, 575.6462732485114 + 0.0j),
            ]
        elif dtype == torch.complex64:
            inouts += [
                (-1 + 1e30j, 69.07755278982137 + 1.5707963267948966j),
                (1e30 + 1j, 69.07755278982137 + 1e-30j),
                (1e30 + 1e30j, 69.42412638010134 + 0.7853981633974483j),
                (1e-30 + 1e30j, 69.07755278982137 + 1.5707963267948966j),
                (1e-30 + 2e-30j, 1e-30 + 2e-30j),
                (1e30 + 1e-30j, 69.07755278982137 + 0.0j),
            ]

        # 分别测试log1p函数
        for inp, out in inouts:
            res = torch.log1p(torch.tensor(inp, dtype=dtype, device=device))
            self.assertFalse(torch.any(torch.isnan(res)))
            # 设置atol == 0.0，因为某些部分具有非常小的值
            self.assertEqual(res.real, out.real, atol=0.0, rtol=1e-6)
            self.assertEqual(res.imag, out.imag, atol=0.0, rtol=1e-6)

        # 在张量中测试log1p函数
        inp_lst, out_lst = (list(elmt) for elmt in zip(*inouts))
        inp_tens = torch.tensor(inp_lst, dtype=dtype, device=device)
        out_tens = torch.tensor(out_lst, dtype=dtype, device=device)
        res_tens = torch.log1p(inp_tens)
        self.assertEqual(res_tens.real, out_tens.real, atol=0.0, rtol=1e-6)
        self.assertEqual(res_tens.imag, out_tens.imag, atol=0.0, rtol=1e-6)

    # 是否需要为诸如threshold之类的操作编写test_unary(_nonufunc)测试套件？
    @onlyCPU
    @dtypes(*get_all_math_dtypes("cpu"))
    # 测试给定设备和数据类型的阈值函数
    def test_threshold(self, device, dtype):
        # 如果数据类型不是 torch.uint8 且不是 torch.float16 且不是复数类型
        if dtype != torch.uint8 and dtype != torch.float16 and not dtype.is_complex:
            # 生成一个包含随机数的张量，使用设备 device，数据类型为 torch.float，然后取符号并转换为指定的 dtype 类型
            x = (
                torch.randn(100, dtype=torch.float, device=device)
                .sign()
                .to(dtype=dtype)
            )
            # 对张量 x 应用阈值函数，将小于等于 0 的元素置为 0
            y = torch.threshold(x, 0, 0)
            # 断言 y 中存在小于等于 0 的元素
            self.assertTrue(y.le(0).any())

    # 辅助函数，用于测试 igamma 函数
    def _helper_test_igamma(self, loglo, loghi, device, dtype, torch_fcn, scipy_fcn):
        # 自然常数 e 的近似值
        exp1 = 2.71828182846
        # 生成一个对数空间张量，从 10^loglo 到 10^loghi，共 500 个步长，使用指定的设备和数据类型
        vec1 = torch.logspace(
            loglo, loghi, steps=500, base=exp1, dtype=torch.float64, device=device
        ).unsqueeze(-1)
        # 将张量 vec1 转换为指定的数据类型 dtype
        vec1 = vec1.to(dtype)
        # 定义多组输入参数，用于测试 igamma 函数
        inputs = [
            (vec1, vec1.transpose(0, 1)),
            (vec1, vec1),  # 对于大数，结果应该接近 0.5
            (vec1, 0.5 * vec1),  # 测试一个较大的比率
            (vec1, 2.0 * vec1),
            (vec1[::2, :], vec1[::2, :]),  # 连续和非连续测试
            (vec1[::2, :], vec1[: vec1.shape[0] // 2, :]),
            (vec1[: vec1.shape[0] // 2, :], vec1[::2, :]),
        ]
        # 如果数据类型为半精度类型（torch.bfloat16 或 torch.float16）
        half_prec = dtype in [torch.bfloat16, torch.float16]
        # 遍历输入参数对
        for input0, input1 in inputs:
            # 调用 torch_fcn 函数计算实际结果
            actual = torch_fcn(input0, input1)
            # 如果数据类型为半精度，将输入参数转换为 float 类型
            if half_prec:
                input0 = input0.to(torch.float)
                input1 = input1.to(torch.float)
            # 使用 scipy_fcn 函数计算期望结果，并将其转换为指定的数据类型 dtype
            expected = scipy_fcn(input0.cpu().numpy(), input1.cpu().numpy())
            expected = torch.from_numpy(expected).to(dtype)
            # 断言实际结果等于期望结果
            self.assertEqual(actual, expected)

    # 根据设备和数据类型测试 igamma 函数的常见用例
    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @onlyNativeDeviceTypes
    def test_igamma_common(self, device, dtype):
        # 对合理范围的值测试 igamma 函数
        loglo = -4  # 约为 0.018
        loghi = 4  # 约为 54.6
        self._helper_test_igamma(
            loglo, loghi, device, dtype, torch.igamma, scipy.special.gammainc
        )

    # 根据设备和数据类型测试 igammac 函数的常见用例
    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @onlyNativeDeviceTypes
    def test_igammac_common(self, device, dtype):
        # 对合理范围的值测试 igammac 函数
        loglo = -4  # 约为 0.018
        loghi = 4  # 约为 54.6
        self._helper_test_igamma(
            loglo, loghi, device, dtype, torch.igammac, scipy.special.gammaincc
        )
    # 测试 torch.igamma 在边界情况下的行为
    def test_igamma_edge_cases(self, device, dtype):
        # 定义张量的参数
        tkwargs = {"dtype": dtype, "device": device}
        # 创建包含 inf 的张量
        infs = torch.zeros((3,), **tkwargs) + float("inf")
        # 创建全零张量
        zeros = torch.zeros((3,), **tkwargs)
        # 创建全一张量
        ones = torch.ones((3,), **tkwargs)
        # 创建包含从 0 到一个大数的张量
        zero_to_large = torch.tensor([0.0, 1.0, 1e3], **tkwargs)
        # 创建包含从一个很小的数到 inf 的张量
        small_to_inf = torch.tensor([1e-3, 1.0, float("inf")], **tkwargs)
        # 创建包含 nan 的张量
        nans = torch.zeros((3,), **tkwargs) + float("nan")
        # 创建输入列表，包含多组测试输入和期望输出
        inpouts = [
            # (a    ,    x),       out
            ((zeros, small_to_inf), ones),        # 测试 igamma(0, inf) 的结果是否为 1
            ((small_to_inf, zeros), zeros),       # 测试 igamma(inf, 0) 的结果是否为 0
            ((infs, zero_to_large), zeros),       # 测试 igamma(inf, 大数) 的结果是否为 0
            ((zero_to_large, infs), ones),        # 测试 igamma(大数, inf) 的结果是否为 1
            ((zeros, zeros), nans),               # 测试 igamma(0, 0) 的结果是否为 nan
            ((infs, infs), nans),                 # 测试 igamma(inf, inf) 的结果是否为 nan
            ((-small_to_inf, small_to_inf), nans),# 测试 igamma(-inf, inf) 的结果是否为 nan
        ]
        # 遍历输入列表进行测试
        for inputs, output in inpouts:
            input0, input1 = inputs
            # 计算 torch.igamma 的结果
            calc = torch.igamma(input0, input1)
            # 如果期望输出包含 nan，则验证计算结果是否全为 nan
            if torch.all(torch.isnan(output)):
                self.assertTrue(torch.all(torch.isnan(calc)))
            else:
                # 否则，验证计算结果与期望输出是否相等
                self.assertEqual(calc, output)

    # 测试 torch.igammac 在边界情况下的行为
    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @onlyNativeDeviceTypes
    def test_igammac_edge_cases(self, device, dtype):
        # 定义张量的参数
        tkwargs = {"dtype": dtype, "device": device}
        # 创建包含 inf 的张量
        infs = torch.zeros((3,), **tkwargs) + float("inf")
        # 创建全零张量
        zeros = torch.zeros((3,), **tkwargs)
        # 创建全一张量
        ones = torch.ones((3,), **tkwargs)
        # 创建包含从 0 到一个大数的张量
        zero_to_large = torch.tensor([0.0, 1.0, 1e3], **tkwargs)
        # 创建包含从一个很小的数到 inf 的张量
        small_to_inf = torch.tensor([1e-3, 1.0, float("inf")], **tkwargs)
        # 创建包含 nan 的张量
        nans = torch.zeros((3,), **tkwargs) + float("nan")
        # 创建输入列表，包含多组测试输入和期望输出
        inpouts = [
            # (a    ,    x),       out
            ((zeros, small_to_inf), zeros),        # 测试 igammac(0, inf) 的结果是否为 0
            ((small_to_inf, zeros), ones),         # 测试 igammac(inf, 0) 的结果是否为 1
            ((infs, zero_to_large), ones),         # 测试 igammac(inf, 大数) 的结果是否为 1
            ((zero_to_large, infs), zeros),        # 测试 igammac(大数, inf) 的结果是否为 0
            ((zeros, zeros), nans),                # 测试 igammac(0, 0) 的结果是否为 nan
            ((infs, infs), nans),                  # 测试 igammac(inf, inf) 的结果是否为 nan
            ((-small_to_inf, small_to_inf), nans), # 测试 igammac(-inf, inf) 的结果是否为 nan
        ]
        # 遍历输入列表进行测试
        for inputs, output in inpouts:
            input0, input1 = inputs
            # 计算 torch.igammac 的结果
            calc = torch.igammac(input0, input1)
            # 如果期望输出包含 nan，则验证计算结果是否全为 nan
            if torch.all(torch.isnan(output)):
                self.assertTrue(torch.all(torch.isnan(calc)))
            else:
                # 否则，验证计算结果与期望输出是否相等
                self.assertEqual(calc, output)

    # 辅助函数，用于测试 torch.i0 与 scipy 的一致性
    def _i0_helper(self, t):
        # 获取数据类型
        dtype = t.dtype
        # 计算 torch.i0 的结果
        actual = torch.i0(t)
        # 对于 bfloat16 类型，需要将输入转换为 float32
        if dtype is torch.bfloat16:
            t = t.to(torch.float32)
        # 调用 scipy 计算 i0 的期望结果
        expected = scipy.special.i0(t.cpu().numpy())
        # 对于 float16 类型，由于 scipy 将结果升级到 float32，需要降级回 float16
        if dtype is torch.bfloat16 or dtype is torch.float16:
            expected = torch.from_numpy(expected).to(dtype)
        # 验证 torch.i0 的计算结果与 scipy 的期望结果是否相等
        self.assertEqual(actual, expected)
    def _i0_range_helper(self, range, device, dtype):
        # i0 tests are broken up by the domain for which the function does not overflow for each dtype
        # This is done to ensure that the function performs well across all possible input values, without worrying
        # about inf or nan possibilities
        # 循环处理给定范围及其相反数，用于测试函数在不溢出的情况下的表现
        for r in (range, -range):
            # 生成在设备上的指定类型的随机数张量，并按照范围缩放
            t = torch.rand(1000, device=device).to(dtype) * r
            # 调用辅助函数进行处理
            self._i0_helper(t)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range1(self, device, dtype):
        # This tests the domain for i0 for which float16 does not overflow
        # The domain is (-13.25, 13.25)
        # 测试 i0 函数在 float16 类型不溢出的定义域 (-13.25, 13.25) 内的情况
        self._i0_range_helper(13.25, device, dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range2(self, device, dtype):
        # This tests the domain for i0 for which float32 and bfloat16 does not overflow
        # The domain is (-88.5, 88.5)
        # 测试 i0 函数在 float32 和 bfloat16 类型不溢出的定义域 (-88.5, 88.5) 内的情况
        self._i0_range_helper(88.5, device, dtype)

    @dtypes(torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range3(self, device, dtype):
        # This tests the domain for i0 for which float64 does not overflow
        # The domain is (-709.75, 709.75)
        # 测试 i0 函数在 float64 类型不溢出的定义域 (-709.75, 709.75) 内的情况
        self._i0_range_helper(709.75, device, dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_special(self, device, dtype):
        # Test special cases for i0 function
        # 测试 i0 函数的特殊情况
        t = torch.tensor([], device=device, dtype=dtype)
        self._i0_helper(t)

        t = torch.tensor([inf, -inf, nan], device=device, dtype=dtype)
        # Assert that the result of i0 function applied on t contains NaN values
        # 断言对 t 使用 i0 函数后的结果包含 NaN 值
        self.assertTrue(torch.i0(t).isnan().all())

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    # 定义测试方法，用于比较特殊函数 i0 和 i1 的 Torch 实现与 SciPy 的对比
    def test_special_i0_i1_vs_scipy(self, device, dtype):
        # 定义内部函数，用于检查 Torch 函数与对应的 SciPy 函数结果是否一致
        def check_equal(t, torch_fn, scipy_fn):
            # 通过与 SciPy 的比较来测试
            actual = torch_fn(t)
            # 如果数据类型是 torch.bfloat16，则需将数据转换为 float32，因为 SciPy 会向上转换为 float32
            if dtype is torch.bfloat16:
                t = t.to(torch.float32)
            expected = scipy_fn(t.cpu().numpy())

            # 对于 float16 数据类型，需要向下转换，因为 SciPy 会向上转换为 float32
            if dtype is torch.bfloat16 or dtype is torch.float16:
                expected = torch.from_numpy(expected).to(dtype)
            self.assertEqual(actual, expected)

        # 创建空的 tensor t，设备为指定设备，数据类型为指定类型
        t = torch.tensor([], device=device, dtype=dtype)
        # 检查 i0 函数
        check_equal(t, torch.i0, scipy.special.i0)
        # 检查 i0e 函数
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        # 如果数据类型不是 torch.half 或 torch.bfloat16，则检查 i1 函数
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            # 检查 i1e 函数
            check_equal(t, torch.special.i1e, scipy.special.i1e)

        # 设置范围，默认为 (-1e7, 1e7)
        range = (-1e7, 1e7)
        # 如果数据类型是 torch.half，则设置范围为 (-65000, 65000)
        if dtype == torch.half:
            range = (-65000, 65000)

        # 创建均匀分布的 tensor t，范围为指定范围，元素数量为 1e4，设备和数据类型与之前相同
        t = torch.linspace(*range, int(1e4), device=device, dtype=dtype)
        # 检查 i0 函数
        check_equal(t, torch.i0, scipy.special.i0)
        # 检查 i0e 函数
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        # 如果数据类型不是 torch.half 或 torch.bfloat16，则检查 i1 函数
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            # 检查 i1e 函数
            check_equal(t, torch.special.i1e, scipy.special.i1e)

        # 测试特定值：最小值、最大值、eps、tiny
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        # 创建包含特定值的 tensor t，数据类型和设备与之前相同
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        # 检查 i0 函数
        check_equal(t, torch.i0, scipy.special.i0)
        # 检查 i0e 函数
        check_equal(t, torch.special.i0e, scipy.special.i0e)
        # 如果数据类型不是 torch.half 或 torch.bfloat16，则检查 i1 函数
        if dtype not in [torch.half, torch.bfloat16]:
            check_equal(t, torch.special.i1, scipy.special.i1)
            # 检查 i1e 函数
            check_equal(t, torch.special.i1e, scipy.special.i1e)

    # 标记测试函数，用于比较特殊函数 ndtr 的 Torch 实现与 SciPy 的对比
    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_special_ndtr_vs_scipy(self, device, dtype):
        # 定义内部函数，用于检查 Torch 函数与对应的 SciPy 函数结果是否一致
        def check_equal(t):
            # 通过与 SciPy 的比较来测试
            actual = torch.special.ndtr(t)
            expected = scipy.special.ndtr(t.cpu().numpy())
            self.assertEqual(actual, expected)

        # 设置范围，默认为 (-10, 10)
        range = (-10, 10)
        # 创建均匀分布的 tensor t，范围为指定范围，元素数量为 1，设备和数据类型与之前相同
        t = torch.linspace(*range, 1, device=device, dtype=dtype)
        # 检查 ndtr 函数
        check_equal(t)

        # 跳过 NaN、inf 和 -inf 的测试，因为它们在 reference_numerics 测试中已经包含

        # 测试特定值：最小值、最大值、eps、tiny
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        # 创建包含特定值的 tensor t，数据类型和设备与之前相同
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        # 检查 ndtr 函数
        check_equal(t)
    # 测试特殊的 torch.special.log_ndtr 方法与 scipy 的对比
    def test_special_log_ndtr_vs_scipy(self, device, dtype):
        # 定义内部函数 check_equal，用于比较 torch.special.log_ndtr 的结果与 scipy 的结果
        def check_equal(t):
            # 调用 torch.special.log_ndtr 计算结果
            actual = torch.special.log_ndtr(t)
            # 使用 scipy.special.log_ndtr 计算期望结果
            expected = scipy.special.log_ndtr(t.cpu().numpy())
            # 断言两者结果相等
            self.assertEqual(actual, expected)

        # 跳过 NaN、inf、-inf 的测试，因为这些情况在 reference_numerics 测试中已经涵盖
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        # 创建包含最小值、最大值、机器精度和极小正数的张量 t
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        # 调用 check_equal 函数进行测试
        check_equal(t)

    # TODO: 允许通过元数据选择大的 opinfo 值
    @dtypes(torch.long)
    def test_abs_big_number(self, device, dtype):
        # 定义一个大数 bignumber
        bignumber = 2**31 + 1
        # 创建包含 bignumber 的张量 res
        res = torch.tensor([bignumber], device=device, dtype=dtype)
        # 断言 res 的绝对值大于 0
        self.assertGreater(res.abs()[0], 0)

    # TODO: 在 opinfos 中添加对有符号零的测试
    @dtypes(torch.float, torch.double)
    def test_abs_signed_zero(self, device, dtype):
        # 测试 abs(0.0) 和 abs(-0.0) 都应该结果为 0.0
        size = 128 + 1  # 选择足够大的数值以确保测试矢量化和非矢量化操作
        # 创建大小为 size 的零张量 inp
        inp = torch.zeros(size, device=device, dtype=dtype)
        inp[::2] = -0.0  # 设置偶数索引位置为 -0.0
        inp = inp.abs()  # 对 inp 求绝对值
        # 遍历 inp 中的每个值，断言其符号为正数
        for v in inp:
            self.assertGreater(math.copysign(1.0, v), 0.0)

    # TODO: 更新以与 NumPy 比较，通过与 OpInfo 合理化
    @onlyCUDA
    @dtypes(torch.float, torch.double)
    def test_abs_zero(self, device, dtype):
        # 测试 abs(0.0) 和 abs(-0.0) 都应该结果为 0.0
        # 创建包含 0.0 和 -0.0 的张量 abs_zeros，并将结果转换为列表
        abs_zeros = torch.tensor([0.0, -0.0], device=device, dtype=dtype).abs().tolist()
        # 遍历 abs_zeros 中的每个数值，断言其符号为正数
        for num in abs_zeros:
            self.assertGreater(math.copysign(1.0, num), 0.0)

    # @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_isposinf_isneginf_non_boolean_output(self, device, dtype):
        # 测试非布尔类型的 `out=` 参数
        # 布尔类型的输出在上述测试用例中已经覆盖
        vals = (float("inf"), -float("inf"), 1.2)
        # 创建包含 vals 值的张量 t
        t = torch.tensor(vals, device=device)
        # 遍历 torch.isposinf 和 torch.isneginf 函数
        for torch_op in (torch.isposinf, torch.isneginf):
            # 创建与 t 相同大小和数据类型的空张量 out
            out = torch.empty_like(t, dtype=dtype)
            # 断言调用 torch_op 时抛出 RuntimeError，提示不支持非布尔类型的输出
            with self.assertRaisesRegex(
                RuntimeError, "does not support non-boolean outputs"
            ):
                torch_op(t, out=out)
    # 定义一个测试方法，用于测试零维张量的非零元素索引功能，使用指定的设备
    def test_nonzero_empty(self, device):
        # 定义一个内部辅助函数，用于断言元组中的张量是否为空且维度正确
        def assert_tuple_empty(tup, dim):
            # 断言元组长度与给定维度相等
            self.assertEqual(dim, len(tup))
            # 遍历元组中的每个张量，断言其形状为 torch.Size([0])
            for t in tup:
                self.assertEqual(torch.Size([0]), t.shape)

        # 创建一个形状为 (0, 2, 0, 5, 0) 的随机张量 x，指定设备为输入设备
        x = torch.randn(0, 2, 0, 5, 0, device=device)
        # 获取张量 x 中非零元素的索引
        y = torch.nonzero(x)
        # 使用 as_tuple=True 获取非零元素索引的元组
        z = torch.nonzero(x, as_tuple=True)

        # 断言 y 张量中的元素数量为 0
        self.assertEqual(0, y.numel())
        # 断言 y 张量的形状为 torch.Size([0, 5])
        self.assertEqual(torch.Size([0, 5]), y.shape)
        # 断言 z 元组的长度为 5，并且其中每个张量的形状为 torch.Size([0])
        assert_tuple_empty(z, 5)

        # 创建一个标量张量 x，其值为 0.5，指定设备为输入设备
        x = torch.tensor(0.5, device=device)
        # 获取张量 x 中非零元素的索引
        y = torch.nonzero(x)
        # 使用 as_tuple=True 获取非零元素索引的元组
        # 对于零维张量，返回的元组长度为 1，以匹配 NumPy 的行为
        z = torch.nonzero(x, as_tuple=True)
        # 断言 z 元组的长度为 1
        self.assertEqual(1, len(z))
        # 断言 z[0] 张量的值为 torch.zeros(1, dtype=torch.long)
        self.assertEqual(torch.zeros(1, dtype=torch.long), z[0])

        # 创建一个零维张量 x，其值为 0，指定设备为输入设备
        x = torch.zeros((), device=device)
        # 获取张量 x 中非零元素的索引
        y = torch.nonzero(x)
        # 使用 as_tuple=True 获取非零元素索引的元组
        z = torch.nonzero(x, as_tuple=True)
        # 断言 y 张量的形状为 torch.Size([0, 0])
        self.assertEqual(torch.Size([0, 0]), y.shape)
        # 断言 z 元组的长度为 1
        self.assertEqual(1, len(z))
        # 断言 z[0] 张量的值为 torch.empty(0, dtype=torch.long)
        self.assertEqual(torch.empty(0, dtype=torch.long), z[0])

    # TODO: rationalize with exp OpInfo
    # 使用装饰器设置参数化测试，包含浮点数和复数类型以及 torch.bfloat16 类型
    @dtypes(*floating_and_complex_types_and(torch.bfloat16))
    # 如果在 CUDA 上，使用装饰器设置参数化测试，包含浮点数和复数类型以及 torch.half 和 torch.bfloat16 类型
    @dtypesIfCUDA(*floating_and_complex_types_and(torch.half, torch.bfloat16))
# 实例化设备类型测试，针对全局中的 TestUnaryUfuncs 类
instantiate_device_type_tests(TestUnaryUfuncs, globals())

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```