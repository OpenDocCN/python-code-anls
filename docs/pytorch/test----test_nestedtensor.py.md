# `.\pytorch\test\test_nestedtensor.py`

```py
# 导入所需的模块和库
import io  # 导入io模块，用于处理文件和流
import itertools  # 导入itertools模块，提供迭代工具函数
import math  # 导入math模块，提供数学运算函数
import sys  # 导入sys模块，提供系统相关的函数
import unittest  # 导入unittest模块，用于编写和运行测试
from functools import partial  # 导入functools模块的partial函数，用于创建偏函数
from typing import Optional, Tuple  # 导入typing模块，用于类型提示

import numpy as np  # 导入numpy库，用于科学计算

import torch  # 导入PyTorch深度学习库
import torch._dynamo  # 导入torch._dynamo模块
import torch._dynamo.testing  # 导入torch._dynamo.testing模块
import torch.nn  # 导入torch.nn模块，包含神经网络相关的函数和类
import torch.nn.functional as F  # 导入torch.nn.functional模块，包含函数实现神经网络的各种操作

from torch.nested._internal.nested_tensor import (  # 从torch.nested._internal.nested_tensor模块导入以下函数和类
    buffer_from_jagged,
    jagged_from_list,
    nested_view_from_values_offsets,
    NestedTensor,
    ViewNestedFromBuffer,
)
from torch.testing._internal.common_cuda import (  # 从torch.testing._internal.common_cuda模块导入以下常量
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    SM70OrLater,
    SM80OrLater,
)
from torch.testing._internal.common_device_type import (  # 从torch.testing._internal.common_device_type模块导入以下函数和常量
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    PYTORCH_CUDA_MEMCHECK,
    skipCUDAIf,
    skipCUDAIfRocm,
    skipMeta,
)
from torch.testing._internal.common_dtype import (  # 从torch.testing._internal.common_dtype模块导入以下函数和常量
    floating_types_and_half,
)
from torch.testing._internal.common_utils import (  # 从torch.testing._internal.common_utils模块导入以下函数和常量
    decorateIf,
    freeze_rng_state,
    gradcheck,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_WINDOWS,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    skipIfSlowGradcheckEnv,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_ROCM,
    TestCase,
    xfailIfTorchDynamo,
)

# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor easier in the future.


def _iter_constructors():
    # yield as_nested_tensor
    yield torch.nested.nested_tensor


# Returns True if the function recompiles between inputs1 and inputs2 with the
# specified dynamic setting.
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    compile_count = [0]

    def counter(gm, example_inputs):
        compile_count[0] += 1
        return gm

    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)
    compiled_f(*inputs2)
    return compile_count[0] > 1


# Helper function to generate a pair of random nested tensors
# one is contiguous, the other is not, but they appear to have same entries
# an output nested tensor consists of
# * `len(ragged_sizes)` matrices
# * matrices[i].shape == (20, ragged_sizes[i])


def random_nt_noncontiguous_pair(ragged_sizes, device="cpu", dtype=torch.float16):
    xs = []
    for size in ragged_sizes:
        xs.append(torch.randn((size, 20), device=device, dtype=dtype))
    # contiguous nested tensor
    ys = []
    for x in xs:
        ys.append(x.transpose(-1, -2))
    nt_contiguous = torch.nested.nested_tensor(ys)
    # noncontiguous nested tensor
    n = len(ragged_sizes)
    nt_noncontiguous = torch.nested.nested_tensor(xs).transpose(-1, -2)
    return nt_contiguous, nt_noncontiguous


# Helper functions to pad a noncontiguous nested tensor
# can be replaced once to_padded_tensor supports noncontiguous memory


def noncontiguous_to_padded_tensor(input, shape=None):
    tensors = input.unbind()
    ntensors = len(tensors)
    assert ntensors > 0
    # 如果形状为None，则初始化形状为空列表
    if shape is None:
        shape = []
        # 遍历第一个张量的每个维度大小，将其添加到形状列表中
        for size in tensors[0].shape:
            shape.append(size)
        # 遍历其他张量，更新形状列表中每个维度的最大值
        for i in range(1, ntensors):
            new_shape = tensors[i].shape
            for j in range(len(shape)):
                shape[j] = max(shape[j], new_shape[j])
        # 将张量数量作为第一个元素添加到形状列表的开头
        shape = [ntensors] + shape
    
    # 根据计算出的形状创建一个全零张量作为结果
    result = tensors[0].new_zeros(shape)
    
    # 遍历每个张量，将其复制到结果张量的相应视图中
    for itensor in range(ntensors):
        tensor = tensors[itensor]
        view = result[itensor]
        # 遍历张量的每个维度，缩小视图的尺寸以匹配张量的尺寸
        for idim in range(tensor.dim()):
            view = view.narrow(idim, 0, tensor.size(idim))
        # 将当前张量的数据复制到视图中
        view.copy_(tensor)
    
    # 返回最终结果张量
    return result
# Helper function to generate a random nested tensor
def random_nt(
    device,
    dtype,
    num_tensors,
    max_dims,
    min_dims=None,
    layout=torch.strided,
    require_non_empty=True,
):
    if min_dims is None:
        min_dims = tuple([0] * len(max_dims))  # 如果未指定最小维度，将其初始化为与最大维度相同的元组

    assert len(max_dims) == len(min_dims)  # 断言最大维度和最小维度的长度相等
    for min_dim, max_dim in zip(min_dims, max_dims):
        assert max_dim > min_dim, "random_nt: max_dim must be greater than min_dim"  # 断言最大维度大于最小维度
        assert min_dim >= 0, "random_nt: min_dim must be non-negative"  # 断言最小维度非负
        if require_non_empty:
            assert not (
                min_dim == 0 and max_dim == 1
            ), "random_nt: zero cannot be the only possible value if require_non_empty is True"  # 断言如果需要非空，则不能只有可能为零

    if require_non_empty:
        # 选择一个随机索引，该索引需要是非空的
        non_zero_idx = torch.randint(low=0, high=num_tensors, size=(1,)).item()

    ts1 = []
    for i, _ in enumerate(range(num_tensors)):
        tensor_dims = []
        for min_dim, max_dim in zip(min_dims, max_dims):
            new_min_dim = min_dim
            if require_non_empty and i == non_zero_idx and min_dim == 0:
                new_min_dim = 1  # 如果需要非空且当前索引为非零索引，则将最小维度调整为1
            tensor_dims.append(
                torch.randint(low=new_min_dim, high=max_dim, size=(1,)).item()  # 在指定范围内生成随机维度大小
            )
        t1 = torch.randn(tensor_dims, device=device, dtype=dtype)  # 使用生成的维度大小生成随机张量
        ts1.append(t1)

    return torch.nested.nested_tensor(ts1, device=device, dtype=dtype, layout=layout)  # 返回生成的嵌套张量


# Alternate approach to generating a random NT.
# dims should be something like [5, None, 10], with None indicating that a
# random ragged structure should be used
def random_nt_from_dims(
    dims, device=None, dtype=None, layout=torch.strided, requires_grad=False
):
    sizes = [
        [
            d if d is not None else torch.randint(2, 10, size=(1,)).item()  # 如果维度为None，则在2到10之间随机选择一个维度大小
            for d in dims[1:]
        ]
        for d in range(dims[0])
    ]
    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in sizes],  # 使用生成的大小列表创建随机张量
        device=device,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
    )


# Creates an NT matching another NT's number of components and
# shape / ragged structure for all dims specified to be -1.
def random_nt_from_similar(other, dims=None):
    if dims is None:
        return torch.randn_like(other)  # 返回与输入张量相同大小的随机张量
    assert len(dims) == other.dim()  # 断言指定维度与输入张量维度相同
    assert dims[0] == -1 or dims[0] == other.size(0)  # 断言指定的第一个维度为-1或与输入张量的第一维大小相同

    ret_sizes = []
    for t in other.unbind():  # 对输入张量进行解绑操作
        other_size = t.shape  # 获取每个张量的形状
        ret_size = []
        for i, d in enumerate(dims[1:]):
            if d == -1:
                ret_size.append(other_size[i])  # 如果指定为-1，则使用输入张量相应维度的大小
            else:
                ret_size.append(d)  # 否则使用指定的维度大小
        ret_sizes.append(ret_size)

    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in ret_sizes], device=other.device  # 使用生成的大小列表创建随机张量
    )


# makes naming nice for tests that parametrize over layout.
def layout_name(layout):
    # e.g. "torch.jagged" -> "jagged"
    # 调用 layout 对象的 __repr__() 方法，获取其字符串表示形式，并以点号 "." 分割为列表
    # 取列表中最后一个元素，即获取字符串中最后一部分
    return layout.__repr__().split(".")[-1]
# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_dense_to_nested_tensor(values):
    # 创建一个偏移量张量，用于定义每个密集张量块的起始偏移量
    offsets = torch.arange(
        0, values.shape[0] * values.shape[1] + 1, values.shape[1], device=values.device
    )
    # 存储一些关于数据的元数据，如最大序列长度和最小序列长度
    metadata_cache = {"max_seqlen": values.shape[1], "min_seqlen": 1}
    # 使用自定义的PyTorch函数将密集张量转换为嵌套张量
    nt = ViewNestedFromBuffer.apply(
        values.view(-1, values.shape[-1]), offsets, metadata_cache
    )
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_jagged_to_nested_tensor(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    # 存储一些关于数据的元数据，如最大序列长度和最小序列长度
    metadata_cache = {"max_seqlen": max_length, "min_seqlen": 1}
    # 使用自定义的PyTorch函数将不规则张量转换为嵌套张量
    nt = ViewNestedFromBuffer.apply(values, offsets, metadata_cache)
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_nt_to_jagged(nt):
    # 使用自定义的PyTorch函数将嵌套张量转换为不规则张量
    return buffer_from_jagged(nt)


@markDynamoStrictTest
class TestNestedTensor(TestCase):
    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_2d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        # 准备测试数据
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            # 生成随机的数据行并添加到数据列表中
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            data.append(row)
            # 将生成的数据行转换为PyTorch张量并添加到参考列表中
            nested_tensor_ref_list.append(torch.Tensor(row))
        # 使用PyTorch的嵌套张量函数创建嵌套张量
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        # 解绑嵌套张量以获取单独的张量列表
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            # 断言解绑后的张量与参考张量列表中的张量类型和值相等
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.int64)
            )

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_3d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        # 准备测试数据
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            # 生成随机的数据行并转换为包含子列表的数据行
            row = [list(item * np.arange(max_seq_len)) for item in
                   list(np.random.randint(low=0, high=vocab_size, size=(length,)))]
            data.append(row)
            # 将生成的数据行转换为PyTorch张量并添加到参考列表中
            nested_tensor_ref_list.append(torch.Tensor(row))
        # 使用PyTorch的嵌套张量函数创建嵌套张量
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        # 解绑嵌套张量以获取单独的张量列表
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            # 断言解绑后的张量与参考张量列表中的张量类型和值相等
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.int64)
            )
    @torch.inference_mode()
    def test_3d_nested_tensor_float(self, batch_size, max_seq_len, vocab_size):
        # 初始化一个空列表用于存储数据
        data = []
        # 初始化一个空列表用于存储参考的嵌套张量列表
        nested_tensor_ref_list = []
        # 循环生成指定数量的样本
        for _ in range(batch_size):
            # 根据最大序列长度确定当前序列的长度
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            # 生成随机长度和值的行
            row = list(
                np.random.randint(low=0, high=vocab_size, size=(length,)).astype(float)
            )
            # 对每个元素进行一定的处理，生成二维列表
            row = [list(item * np.arange(max_seq_len)) for item in row]
            # 将当前行添加到数据列表中
            data.append(row)
            # 将当前行的 Torch 张量添加到嵌套张量参考列表中
            nested_tensor_ref_list.append(torch.Tensor(row))
        # 使用数据列表创建嵌套张量对象
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.float)
        # 对嵌套张量对象进行解绑操作，得到张量列表
        nested_tensor_list = nested_tensor.unbind()
        # 遍历样本数量，比较解绑后的张量和参考张量的类型是否一致
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.float)
            )

    @torch.inference_mode()
    def _test_unbind_case(self, a, b):
        # 创建一个包含两个张量的嵌套张量对象
        nt = torch.nested.nested_tensor([a, b])
        # 对嵌套张量进行解绑操作，获取解绑后的两个张量
        a1, b1 = nt.unbind()
        # 断言解绑后的第一个张量和原始的第一个张量不是同一个对象
        self.assertTrue(a is not a1)
        # 断言解绑后的第二个张量和原始的第二个张量不是同一个对象
        self.assertTrue(b is not b1)

        # 使用指定的数据类型创建嵌套张量对象
        nt = torch.nested.nested_tensor([a, b], dtype=a.dtype)
        # 按指定维度对嵌套张量进行解绑操作，获取解绑后的两个张量
        a1, b1 = nt.unbind(0)
        # 断言解绑后的第一个张量与原始的第一个张量相等
        self.assertEqual(a, a1)
        # 断言解绑后的第二个张量与原始的第二个张量相等
        self.assertEqual(b, b1)

        # 创建一个随机生成的张量
        a = torch.randn((2, 3)).add_(1)
        # 使用单个张量创建嵌套张量对象
        nt = torch.nested.nested_tensor([a])
        # 断言解绑后的第一个张量与原始的第一个张量相等
        self.assertEqual(a, nt.unbind(0)[0])

    @torch.inference_mode()
    def test_unbind_0(self):
        # 调用 _test_unbind_case 方法，传入两个张量作为参数
        self._test_unbind_case(
            torch.tensor([1, 2]),
            torch.tensor([7, 8]),
        )

    @torch.inference_mode()
    def test_unbind_1(self):
        # 调用 _test_unbind_case 方法，传入两个张量作为参数
        self._test_unbind_case(
            torch.tensor([1]),
            torch.tensor([7]),
        )

    @torch.inference_mode()
    def test_unbind_3(self):
        # 调用 _test_unbind_case 方法，传入两个张量作为参数
        self._test_unbind_case(
            torch.tensor([1.0]),
            torch.tensor([]),
        )

    @torch.inference_mode()
    def test_unbind_4(self):
        # 调用 _test_unbind_case 方法，传入两个张量作为参数
        self._test_unbind_case(
            torch.tensor([]),
            torch.tensor([]),
        )

    @torch.inference_mode()
    def test_unbind_dim(self):
        # 定义一个内部函数，用于测试解绑函数的异常情况
        def _test_fn(unbind_fn):
            # 创建两个随机张量
            a = torch.rand(3, 2)
            b = torch.rand(2, 3)
            # 使用这两个张量创建嵌套张量对象
            nt = torch.nested.nested_tensor([a, b])
            # 断言在解绑指定维度时会触发 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: unbind_fn(nt, 1))

        # 调用 _test_fn 函数，传入解绑函数作为参数
        _test_fn(lambda x, dim: x.unbind(dim))
        # TODO: Re-enable this once using torch_dispatch
        # _test_fn(lambda x, dim: torch.unbind(x, dim))

    @torch.inference_mode()
    def test_nested_tensor(self):
        # 断言创建包含非嵌套张量的嵌套张量对象会触发 TypeError 异常
        self.assertRaises(
            TypeError, lambda: torch.nested.nested_tensor(torch.tensor([3.0]))
        )
        # 断言创建包含非张量的嵌套张量对象会触发 TypeError 异常
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor(4.0))
    # 定义测试函数，用于测试嵌套张量的维度匹配情况
    def test_nested_tensor_matching_dim(self):
        # 断言捕获异常信息为运行时错误，指出索引为1的张量维度为1，索引为0的张量维度为0
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 1 and dimension 0 for Tensor at index 0.",
            # 使用 lambda 表达式调用 nested_tensor 函数，传入两个张量作为参数
            lambda: torch.nested.nested_tensor([torch.tensor(1.0), torch.tensor([])]),
        )
        # 断言捕获异常信息为运行时错误，指出索引为2的张量维度为1，索引为1的张量维度为0
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 2 and dimension 0 for Tensor at index 1.",
            # 使用 lambda 表达式调用 nested_tensor 函数，传入三个张量作为参数
            lambda: torch.nested.nested_tensor(
                [torch.tensor(1.0), torch.tensor(2.0), torch.tensor([])]
            ),
        )

    # 声明测试函数，使用 Torch 的推断模式进行测试
    @torch.inference_mode()
    def test_default_nested_tensor(self):
        # 断言捕获异常信息为类型错误，测试 nested_tensor 函数不带参数时的异常情况
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor())
        # 创建默认的嵌套张量，其包含空列表
        default_nested_tensor = torch.nested.nested_tensor([])
        # 创建默认的张量，空列表表示
        default_tensor = torch.tensor([])
        # 断言默认嵌套张量的维度与默认张量的维度相同
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        # 断言默认嵌套张量的布局与默认张量的布局相同
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        # 断言默认嵌套张量的设备与默认张量的设备相同
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        # 断言默认嵌套张量的数据类型与默认张量的数据类型相同
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        # 断言默认嵌套张量的梯度要求与默认张量的梯度要求相同
        self.assertEqual(
            default_nested_tensor.requires_grad, default_tensor.requires_grad
        )
        # 断言默认张量的梯度为 None
        self.assertIsNone(default_tensor.grad)
        # TODO: 待实现性能驱动的用例和实现后，重新启用以下断言
        # 断言默认嵌套张量的固定性与默认张量的固定性相同
        # self.assertEqual(default_nested_tensor.is_pinned(),
        #                  default_tensor.is_pinned())

    # 声明测试函数，使用 Torch 的推断模式进行测试
    @torch.inference_mode()
    def test_dim(self):
        # 遍历所有构造器的生成器
        for constructor in _iter_constructors():
            # 使用构造器创建一个空列表作为参数构造张量 a1
            a1 = constructor([])
            # 断言张量 a1 的维度为 1
            self.assertEqual(a1.dim(), 1)
            # 使用构造器创建包含一个张量的列表作为参数构造张量 a1
            a1 = constructor([torch.tensor(3.0)])
            # 断言张量 a1 的维度为 1
            self.assertEqual(a1.dim(), 1)
            # 使用构造器创建包含一个整数张量的列表作为参数构造张量 a1
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            # 断言张量 a1 的维度为 2
            self.assertEqual(a1.dim(), 2)

    # 跳过测试，如果运行环境为 FBCODE（Facebook 专用的代码库）
    @unittest.skipIf(IS_FBCODE, "numel is not virtual in fbcode.")
    # 使用 Torch 的推断模式进行测试
    @torch.inference_mode()
    def test_numel(self):
        # 对于每个构造函数，迭代执行测试
        for constructor in _iter_constructors():
            # 使用空列表构造张量 a1
            a1 = constructor([])
            # 断言张量 a1 的元素数为 0
            self.assertEqual(a1.numel(), 0)
            # 使用包含两个张量的列表构造张量 a1
            a1 = constructor([torch.tensor(3.0), torch.tensor(4.0)])
            # 断言张量 a1 的元素数为 2
            self.assertEqual(a1.numel(), 2)
            # 使用包含一个形状为 [2, 2, 2] 的张量的列表构造张量 a1
            a1 = constructor([torch.randn(2, 2, 2)])
            # 断言张量 a1 的元素数为 8
            self.assertEqual(a1.numel(), 8)
            # 使用包含两个不同形状的张量的列表构造张量 a1
            a1 = constructor([torch.randn([1, 2, 3]), torch.randn(3, 2, 1)])
            # 断言张量 a1 的元素数为 12
            self.assertEqual(a1.numel(), 12)
            # 使用包含两个不同形状的张量的列表构造张量 a1
            a1 = constructor([torch.randn([1, 1, 3]), torch.randn(3, 2, 4)])
            # 断言张量 a1 的元素数为 27
            self.assertEqual(a1.numel(), 27)
            # 使用包含两个不同形状的张量的列表构造张量 a1
            a1 = constructor([torch.randn([5, 5, 5]), torch.randn(6, 6, 6)])
            # 断言张量 a1 的元素数为 341
            self.assertEqual(a1.numel(), 341)

            # 有趣的边缘情况
            # 使用包含一个形状为 [1, 2, 3] 和一个形状为 [1, 2, 0] 的张量的列表构造张量 a1
            self.assertEqual(a1.numel(), 6)

    @torch.inference_mode()
    def test_size(self):
        # 对于每个构造函数，迭代执行测试
        for constructor in _iter_constructors():
            # 使用空列表构造张量 a1
            a1 = constructor([])
            # 引发运行时错误，指出 NestedTensorImpl 不支持尺寸操作
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support sizes",
                lambda: a1.size(),
            )

    def test_size_dim(self):
        # 创建空的 nested tensor a
        a = torch.nested.nested_tensor([])
        # 断言 nested tensor a 的第 0 维尺寸为 0
        self.assertEqual(a.size(0), 0)

        # 创建包含一个张量的 nested tensor a
        a = torch.nested.nested_tensor([torch.tensor(1)])
        # 断言 nested tensor a 的第 0 维尺寸为 1
        self.assertEqual(a.size(0), 1)

        # 创建包含两个张量的 nested tensor a
        a = torch.nested.nested_tensor([torch.tensor(1), torch.tensor(2)])
        # 断言 nested tensor a 的第 0 维尺寸为 2
        self.assertEqual(a.size(0), 2)

        # 创建包含两个形状分别为 [1, 2] 和 [1, 8] 的张量的 nested tensor a
        a = torch.nested.nested_tensor([torch.rand(1, 2), torch.rand(1, 8)])
        # 断言 nested tensor a 的第 0 维尺寸为 2
        self.assertEqual(a.size(0), 2)
        # 断言 nested tensor a 的第 1 维尺寸为 1
        self.assertEqual(a.size(1), 1)
        # 引发运行时错误，指出给定的第 2 维不规则，没有尺寸
        self.assertRaisesRegex(
            RuntimeError,
            "Given dimension 2 is irregular and does not have a size",
            lambda: a.size(2),
        )

        # 创建包含两个形状分别为 [3, 4] 和 [5, 4] 的张量的 nested tensor a
        a = torch.nested.nested_tensor([torch.rand(3, 4), torch.rand(5, 4)])
        # 断言 nested tensor a 的第 0 维尺寸为 2
        self.assertEqual(a.size(0), 2)
        # 引发运行时错误，指出给定的第 1 维不规则，没有尺寸
        self.assertRaisesRegex(
            RuntimeError,
            "Given dimension 1 is irregular and does not have a size",
            lambda: a.size(1),
        )
        # 断言 nested tensor a 的第 2 维尺寸为 4
        self.assertEqual(a.size(2), 4)

    @unittest.skipIf(IS_FBCODE, "stride is not virtual in fbcode.")
    @torch.inference_mode()
    def test_stride(self):
        # 对于每个构造函数，迭代执行测试
        for constructor in _iter_constructors():
            # 使用空列表构造张量 a1
            a1 = constructor([])
            # 引发运行时错误，指出 NestedTensorImpl 不支持步长操作
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support strides",
                lambda: a1.stride(),
            )

    @unittest.skipIf(IS_FBCODE, "is_contiguous is not virtual in fbcode.")
    @torch.inference_mode()
    # 定义测试类中的方法，用于测试是否连续
    def test_is_contiguous(self):
        # 测试空情况
        nt_empty = torch.nested.nested_tensor([])
        # 断言空张量是否连续
        assert nt_empty.is_contiguous()
        # 断言空张量与其连续版本是否相等
        self.assertEqual(nt_empty, nt_empty.contiguous())

        # 获取一个连续和一个非连续的随机嵌套张量对
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))

        # 测试连续情况
        assert nt_contiguous.is_contiguous()
        # 断言连续张量与其连续版本是否相等
        self.assertEqual(nt_contiguous, nt_contiguous.contiguous())

        # 测试非连续情况
        assert not nt_noncontiguous.is_contiguous()
        # 断言连续张量与非连续张量的连续版本是否相等
        self.assertEqual(nt_contiguous, nt_noncontiguous.contiguous())

        # 测试使用 memory_format 参数查询是否连续
        self.assertTrue(
            nt_contiguous.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            not nt_noncontiguous.is_contiguous(memory_format=torch.contiguous_format)
        )

    # 使用 torch.inference_mode 装饰器定义测试类中的方法
    @torch.inference_mode()
    def test_repr_string(self):
        # 创建空嵌套张量，并定义预期输出
        a = torch.nested.nested_tensor([])
        expected = "nested_tensor([\n\n])"
        # 断言转换为字符串后是否符合预期输出
        self.assertEqual(str(a), expected)
        # 断言使用 repr 函数后是否符合预期输出
        self.assertEqual(repr(a), expected)

        # 创建包含一个张量的嵌套张量，并定义预期输出
        a = torch.nested.nested_tensor([torch.tensor(1.0)])
        expected = "nested_tensor([\n  tensor(1.)\n])"
        # 断言转换为字符串后是否符合预期输出
        self.assertEqual(str(a), expected)
        # 断言使用 repr 函数后是否符合预期输出
        self.assertEqual(repr(a), expected)

        # 创建包含两个张量的嵌套张量，并定义预期输出
        a = torch.nested.nested_tensor([torch.tensor([[1, 2]]), torch.tensor([[4, 5]])])
        expected = "nested_tensor([\n  tensor([[1, 2]]),\n  tensor([[4, 5]])\n])"
        # 断言转换为字符串后是否符合预期输出
        self.assertEqual(str(a), expected)
        # 断言使用 repr 函数后是否符合预期输出
        self.assertEqual(repr(a), expected)

    # 定义测试类中的方法，测试对空张量应用 to_padded_tensor 的情况
    def test_to_padded_tensor_on_empty_tensor(self):
        # 创建空嵌套张量
        nt = torch.nested.nested_tensor([])
        # 将其转换为填充张量
        empty = torch.nested.to_padded_tensor(nt, 4)
        # 断言结果是否为空张量
        self.assertEqual(empty, torch.tensor([]))

    # 定义测试类中的方法，测试嵌套命名空间
    def test_nested_namespace(self):
        # 创建包含两个随机张量的嵌套张量
        nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 5)])
        # 将嵌套张量转换为填充张量
        result = nt.to_padded_tensor(4)
        # 使用 torch.nested.to_padded_tensor 将嵌套张量转换为填充张量
        nested_namespace_result = torch.nested.to_padded_tensor(nt, 4)
        # 断言结果是否相等
        self.assertEqual(result, nested_namespace_result)
    # 定义测试方法，用于测试张量复制操作
    def test_copy_(self):
        # 设置张量数量
        ntensors = 4
        # 创建随机张量集合，分配在 CPU 上，数据类型为 float32，大小为 (4, 4)
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        # 创建与 nt 相同大小的空张量 nt_copy
        nt_copy = torch.empty_like(nt)
        # 将 nt 的数据复制到 nt_copy 中
        nt_copy.copy_(nt)

        # 遍历解绑后的 nt 和 nt_copy 张量，逐一比较是否相等
        for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
            self.assertEqual(nt_ub, nt_copy_ub)

        # 创建一个错误的嵌套张量 nt_error
        nt_error = torch.nested.nested_tensor([torch.tensor([0, 0])])
        # 断言操作会抛出 RuntimeError 异常，检查异常消息是否正确
        self.assertRaisesRegex(
            RuntimeError,
            "copy_ only supports tensors that are the same size for Nested implementations",
            lambda: nt_error.copy_(nt),
        )

        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 创建 CUDA 上的随机张量集合 nt，数据类型为 float32，大小为 (4, 4)
            nt = random_nt(torch.device("cuda"), torch.float32, ntensors, (4, 4))
            # 创建与 nt 相同大小的空张量 nt_copy，分配在 CPU 上
            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            # 使用非阻塞方式将 nt 的数据复制到 nt_copy 中
            nt_copy.copy_(nt, non_blocking=True)
            # 同步 CUDA 当前设备上的当前流
            torch.cuda.current_stream(torch.cuda.current_device()).synchronize()
            # 再次遍历解绑后的 nt 和 nt_copy 张量，逐一比较是否相等
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

            # 创建与 nt 相同大小的空张量 nt_copy，分配在 CPU 上
            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            # 使用阻塞方式将 nt 的数据复制到 nt_copy 中
            nt_copy.copy_(nt, non_blocking=False)
            # 再次遍历解绑后的 nt 和 nt_copy 张量，逐一比较是否相等
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

    # 定义测试方法，用于测试张量填充操作
    def test_fill_(self):
        # 设置张量数量
        ntensors = 4
        # 创建随机张量集合，分配在 CPU 上，数据类型为 float32，大小为 (4, 4)
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        # 将 nt 中所有元素填充为 10.0
        nt.fill_(10.0)
        # 遍历解绑后的 nt 张量，逐一比较每个张量元素是否等于 10.0
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(10.0)
            self.assertEqual(nt_ub, t)

        # 创建一个填充张量 fill_tensor，内容为 [11.0]
        fill_tensor = torch.tensor([11.0])
        # 断言操作会抛出 RuntimeError 异常，检查异常消息是否正确
        self.assertRaisesRegex(
            RuntimeError,
            "fill_ only supports 0-dimension value tensor",
            lambda: nt.fill_(fill_tensor),
        )

        # 使用 fill_tensor[0] 填充 nt 中所有元素
        nt.fill_(fill_tensor[0])
        # 再次遍历解绑后的 nt 张量，逐一比较每个张量元素是否等于 11.0
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(11.0)
            self.assertEqual(nt_ub, t)

    # 定义测试方法，用于测试张量清零操作
    def test_zero_(self):
        # 设置张量数量
        ntensors = 4
        # 创建随机张量集合，分配在 CPU 上，数据类型为 float32，大小为 (4, 4)
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        # 将 nt 中所有元素清零
        nt.zero_()
        # 遍历解绑后的 nt 张量，逐一比较每个张量元素是否等于 0.0
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(0.0)
            self.assertEqual(nt_ub, t)

    # 使用参数化装饰器定义测试方法，用于测试类似的张量操作函数
    @parametrize(
        "func",
        [torch.ones_like, torch.zeros_like, torch.randn_like],
        # 将函数名作为参数化名称
        name_fn=lambda f: f.__name__,
    )
    def test_like_functions(self, func):
        # 设置张量数量
        ntensors = 4
        # 创建随机张量集合，分配在 CPU 上，数据类型为 float32，大小为 (4, 4)
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        # 设置随机数种子为 1
        torch.manual_seed(1)
        # 使用 func 函数生成与 nt 相同大小的张量集合 nt_like
        nt_like = func(nt)

        # 再次设置随机数种子为 1
        torch.manual_seed(1)
        # 遍历解绑后的 nt_like 张量集合，使用 func 函数逐一生成相同大小的张量并比较是否相等
        for nt_ub in nt_like.unbind():
            t_like = func(nt_ub)
            self.assertEqual(nt_ub, t_like)
# 使用特定标记对该测试类进行严格的 Dynamo 测试
@markDynamoStrictTest
class TestNestedTensorDeviceType(TestCase):
    # 辅助函数，生成一对随机的嵌套张量
    # 这两个嵌套张量具有相同的形状
    def random_nt_pair(self, device, dtype, num_tensors, max_dims):
        ts1 = []
        ts2 = []
        for _ in range(num_tensors):
            # 生成随机的张量维度，每个维度在给定范围内随机取值
            tensor_dims = tuple(
                [
                    torch.randint(low=0, high=max_dim, size=(1,)).item()
                    for max_dim in max_dims
                ]
            )
            # 创建指定设备和数据类型的随机张量
            t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
            t2 = torch.randn(tensor_dims, device=device, dtype=dtype)
            ts1.append(t1)
            ts2.append(t2)
        # 返回创建的两个嵌套张量对象
        return (
            torch.nested.nested_tensor(ts1, device=device, dtype=dtype),
            torch.nested.nested_tensor(ts2, device=device, dtype=dtype),
        )

    @dtypes(*floating_types_and_half())
    def test_detach(self, device, dtype):
        # 创建指定设备和数据类型的随机张量 a 和 b
        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=False)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=False)
        # 创建嵌套张量 x 包含张量 a 和 b，并设置需要梯度计算
        x = torch.nested.nested_tensor([a, b], requires_grad=True)

        # 对 x 进行 detach 操作，生成新的不需要梯度的张量 x_detach
        x_detach = x.detach()

        # 对 x_detach 进行数值操作，生成新的张量 z
        z = x_detach * 4
        # 验证 x_detach 和 z 均不需要梯度
        self.assertFalse(x_detach.requires_grad)
        self.assertFalse(z.requires_grad)

        # 重新生成随机需要梯度计算的张量 a 和 b
        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)
        # 创建嵌套张量 x 包含张量 a 和 b
        x = torch.nested.as_nested_tensor([a, b])

        # 对 x 进行数值操作，并进行 detach 操作得到 y，使得 y 不需要梯度
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

        # 对 x 和 y 进行数值操作，并转换为填充张量进行求和和反向传播
        z = x + y
        torch.nested.to_padded_tensor(z, 0).sum().backward()
        # 这是一个不正确的梯度结果，但我们假定这是用户想要的。detach() 是一个高级选项。
        # 验证张量 a 和 b 的梯度是否为全1张量
        self.assertEqual(a.grad, torch.ones(2, 4, device=device, dtype=dtype))
        self.assertEqual(b.grad, torch.ones(5, 4, device=device, dtype=dtype))

    @dtypes(torch.float, torch.float16, torch.double)
    def test_unbind_noncontiguous(self, device, dtype):
        # 生成随机的非连续嵌套张量对 nt_contiguous 和 nt_noncontiguous
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        # 对连续和非连续嵌套张量进行解绑操作
        ub_contiguous = nt_contiguous.unbind()
        ub_noncontiguous = nt_noncontiguous.unbind()
        # 验证解绑后的张量列表长度是否一致
        self.assertEqual(len(ub_contiguous), len(ub_noncontiguous))
        n = len(ub_contiguous)
        # 逐个验证解绑后的张量是否相等
        for i in range(n):
            self.assertEqual(ub_contiguous[i], ub_noncontiguous[i])

    @dtypes(torch.float)
    @skipMeta
    # 定义一个测试函数，测试不进行任何转换的嵌套张量操作
    def test_to_then_from_padded_tensor_no_transform0213(self, device, dtype):
        # 生成一个指定设备和数据类型的随机张量
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        # 解绑张量，得到张量列表
        ts = list(torch.unbind(t))
        # 修改列表中第一个张量的内容，去掉最后一个元素
        ts[0] = ts[0][:-1]
        # 使用修改后的张量列表创建嵌套张量
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        # 将嵌套张量转换为填充后的张量
        padded = torch.nested.to_padded_tensor(nt, 0)

        # 从填充后的张量和原始嵌套张量示例中重新构造嵌套张量
        nt_to = torch._nested_from_padded_and_nested_example(padded, nt)

        # 对比原始嵌套张量和重新构造的嵌套张量的每个张量是否相等
        for t1, t2 in zip(nt.unbind(), nt_to.unbind()):
            self.assertEqual(t1, t2)
        # 断言原始嵌套张量和重新构造的嵌套张量在同一个设备上
        self.assertEqual(nt.device, nt_to.device)

    # 使用指定的浮点数据类型进行测试的装饰器
    @dtypes(torch.float)
    # 如果是在CUDA环境下，则额外使用浮点和半精度数据类型进行测试
    @dtypesIfCUDA(torch.float, torch.half)
    # 跳过此测试的元数据装饰器
    @skipMeta
    # 进入推理模式的装饰器
    @torch.inference_mode()
    # 使用指定的浮点数据类型进行测试的装饰器
    @dtypes(torch.float)
    # 如果是在CUDA环境下，则额外使用浮点和半精度数据类型进行测试
    @dtypesIfCUDA(torch.float, torch.half)
    # 跳过此测试的元数据装饰器
    @skipMeta
    # 进入推理模式的装饰器
    @torch.inference_mode()
    # 测试函数，用于测试LayerNorm在嵌套张量上的行为
    def test_layer_norm_breaking(self, device, dtype):
        # 定义张量的尺寸
        size = 128
        # 创建具有随机值的张量t0
        t0 = torch.randn(
            4, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        # 创建具有随机值的张量t1
        t1 = torch.randn(
            10, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        # 创建具有随机值的张量t2
        t2 = torch.randn(
            7, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        # 将创建的张量组成列表
        ts = [t0, t1, t2, t0, t2]
        # 使用张量列表创建嵌套张量
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        # 创建LayerNorm对象，设定嵌套张量的维度作为参数
        layer_norm = torch.nn.LayerNorm((4, size, size, 4), device=device, dtype=dtype)
        # 断言运行时错误，检查LayerNorm对嵌套张量的规范化形状是否超出规则维度
        self.assertRaisesRegex(
            RuntimeError,
            "normalized_shape extends into irregular dimensions for the nested tensor",
            lambda: layer_norm(nt),
        )
        # 创建LayerNorm对象，设定不规则维度作为参数
        layer_norm = torch.nn.LayerNorm((size + 1, size, 4), device=device, dtype=dtype)
        # 断言运行时错误，检查LayerNorm对嵌套张量的规范化维度形状是否不合法
        self.assertRaisesRegex(
            RuntimeError,
            "The shape at dimension 0",
            lambda: layer_norm(nt),
        )

    # 根据装饰条件决定是否跳过测试的装饰器
    @decorateIf(
        xfailIfTorchDynamo,
        # 仅在Python 3.11下失败。TODO: 确保在视图正常工作后修复此问题！
        lambda params: params["layout"] == torch.jagged and sys.version_info >= (3, 11),
    )
    # 参数化测试，测试不同布局下的嵌入操作
    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    # 测试函数，测试嵌入操作
    def test_embedding(self, device, layout):
        # 创建包含随机整数的张量列表
        inputs = [
            torch.randint(100, (L,), device=device, dtype=torch.int64)
            for L in torch.randint(5, 50, (8,))
        ]
        # 使用输入张量列表创建嵌套张量
        x = torch.nested.nested_tensor(
            inputs, device=device, dtype=torch.int64, layout=layout
        )
        # 创建嵌入层对象，设定嵌套张量的维度作为参数
        emb = torch.nn.Embedding(100, 8, device=device)
        # 对嵌套张量进行嵌入操作
        y = emb(x)
        # 解绑嵌入后的张量
        ys = y.unbind()
        # 对比每个输入张量与对应嵌入结果的一致性
        for i, inp in enumerate(inputs):
            self.assertEqual(emb(inp), ys[i])

    # 跳过此测试的元数据装饰器
    @skipMeta
    # 进入推理模式的装饰器
    @torch.inference_mode()
    # 使用浮点类型和半精度类型作为数据类型的装饰器
    @dtypes(*floating_types_and_half())
    # 定义一个测试函数，用于测试 masked_fill 方法在不同条件下的行为
    def test_masked_fill(self, device, dtype):
        # 生成随机的嵌套张量对 (nt, mask)
        (nt, mask) = self.random_nt_pair(device, dtype, 4, (4, 4))
        # 创建一个布尔类型的嵌套张量 mask，表示每个元素是否小于 0
        mask = torch.nested.nested_tensor([m < 0 for m in mask.unbind()])
        # 根据每个元素的 mask，对 nt 中的元素进行 masked_fill 操作，填充为 0，并生成参考结果 ref
        ref = torch.nested.nested_tensor(
            [t.masked_fill(m, 0) for (t, m) in zip(nt.unbind(), mask.unbind())]
        )
        # 对原始的嵌套张量 nt，根据 mask 进行 masked_fill 操作，填充为 0，并生成输出结果 out
        out = nt.masked_fill(mask, 0)
        # 断言参考结果和输出结果相等
        self.assertEqual(ref, out)

    # 使用指定的数据类型 dtype，在给定设备上测试 to_padded_tensor 方法
    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_simple(self, device, dtype):
        # 创建一个随机的张量 t，形状为 (4, 4, 4)，在给定设备上，使用指定数据类型 dtype
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        # 将张量 t 拆解成列表 ts
        ts = list(torch.unbind(t))
        # 修改列表中的第一个张量，去除其最后一个维度
        ts[0] = ts[0][:-1]
        # 使用 ts 创建嵌套张量 nt，指定设备和数据类型 dtype
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        # 针对两种填充值 (0 和 1)，测试 to_padded_tensor 方法
        for padding_value in (0, 1):
            # 对嵌套张量 nt 使用 to_padded_tensor 方法，填充值为 padding_value
            padded = torch.nested.to_padded_tensor(nt, padding_value)

            # 创建正确的输出 correct_output，克隆张量 t
            correct_output = t.clone()
            # 根据填充值设置 correct_output 的最后一个元素
            if padding_value == 0:
                correct_output[0][-1] = torch.zeros_like(correct_output[0][-1])
            else:
                correct_output[0][-1] = torch.ones_like(correct_output[0][-1])

            # 断言填充后的张量 padded 和正确的输出 correct_output 相等
            self.assertEqual(padded, correct_output)
            # 断言填充后的张量 padded 的设备与指定设备一致
            self.assertEqual(padded.device, torch.device(device))
            # 断言填充后的张量 padded 的数据类型与指定数据类型 dtype 一致
            self.assertEqual(padded.dtype, dtype)

    # 使用指定的数据类型 dtype，在给定设备上测试 to_padded_tensor 方法，同时验证输出尺寸
    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_output_size(self, device, dtype):
        # 创建一个随机的张量 t，形状为 (4, 4, 4)，在给定设备上，使用指定数据类型 dtype
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        # 设置期望的输出尺寸为 (4, 6, 5)
        output_size = (4, 6, 5)
        # 将张量 t 拆解成列表 ts
        ts = list(torch.unbind(t))
        # 修改列表中的第一个张量，去除其最后一个维度
        ts[0] = ts[0][:-1]
        # 使用 ts 创建嵌套张量 nt，指定设备和数据类型 dtype
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        # 针对两种填充值 (0 和 1)，测试 to_padded_tensor 方法，同时指定输出尺寸
        for padding_value in (0, 1):
            # 对嵌套张量 nt 使用 to_padded_tensor 方法，填充值为 padding_value，输出尺寸为 output_size
            padded = torch.nested.to_padded_tensor(
                nt, padding_value, output_size=output_size
            )
            # 创建正确的输出 correct_output，使用指定填充值和输出尺寸
            correct_output = (
                torch.ones(output_size, device=device, dtype=dtype) * padding_value
            )
            # 将原始张量 t 的值复制到正确的输出 correct_output 的指定位置
            correct_output[:4:, :4, :4] = t.clone()
            # 根据填充值设置 correct_output 的一个元素
            if padding_value == 0:
                correct_output[0][3] = torch.zeros_like(correct_output[0][3])
            else:
                correct_output[0][3] = torch.ones_like(correct_output[0][3])

            # 断言填充后的张量 padded 和正确的输出 correct_output 相等
            self.assertEqual(padded, correct_output)
            # 断言填充后的张量 padded 的设备与指定设备一致
            self.assertEqual(padded.device, torch.device(device))
            # 断言填充后的张量 padded 的数据类型与指定数据类型 dtype 一致
            self.assertEqual(padded.dtype, dtype)
    # 定义测试方法，用于测试二维张量的填充操作
    def test_to_padded_tensor_dim2(self, device, dtype):
        # 创建三个随机张量，指定设备和数据类型
        ts = [
            torch.randn(160, device=device, dtype=dtype),
            torch.randn(1240, device=device, dtype=dtype),
            torch.randn(2400, device=device, dtype=dtype),
        ]
        # 使用 nested_tensor 函数创建嵌套张量 nt
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        # 指定填充值为 42
        pad = 42
        # 用于存储正确输出结果的列表
        correct_output = []
        # 遍历每个张量 t
        for t in ts:
            # 创建与 ts[2] 相同形状的全为填充值 pad 的张量 next_output
            next_output = torch.ones_like(ts[2]) * pad
            # 将张量 t 的数据复制到 next_output 的前 t.size(0) 个元素
            next_output[: t.size(0)].copy_(t)
            # 将 next_output 添加到 correct_output 中
            correct_output.append(next_output)
        # 将 correct_output 列表堆叠成一个张量
        correct_output = torch.stack(correct_output)
        # 调用 to_padded_tensor 函数对嵌套张量 nt 进行填充操作
        padded = torch.nested.to_padded_tensor(nt, pad)
        # 使用断言检查填充后的张量与预期的 correct_output 是否相等
        self.assertEqual(padded, correct_output)

    # 使用 dtypes 装饰器，测试三维张量的填充操作
    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim3(self, device, dtype):
        # 创建三个随机三维张量，指定设备和数据类型
        ts = [
            torch.randn(16, 21, device=device, dtype=dtype),
            torch.randn(24, 32, device=device, dtype=dtype),
            torch.randn(40, 53, device=device, dtype=dtype),
        ]
        # 使用 nested_tensor 函数创建嵌套张量 nt
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        # 指定填充值为 42
        pad = 42
        # 用于存储正确输出结果的列表
        correct_output = []
        # 遍历每个张量 t
        for t in ts:
            # 创建与 ts[2] 相同形状的全为填充值 pad 的张量 next_output
            next_output = torch.ones_like(ts[2]) * pad
            # 将张量 t 的数据复制到 next_output 的前 t.size(0) 行、t.size(1) 列
            next_output[: t.size(0), : t.size(1)].copy_(t)
            # 将 next_output 添加到 correct_output 中
            correct_output.append(next_output)
        # 将 correct_output 列表堆叠成一个张量
        correct_output = torch.stack(correct_output)
        # 调用 to_padded_tensor 函数对嵌套张量 nt 进行填充操作
        padded = torch.nested.to_padded_tensor(nt, pad)
        # 使用断言检查填充后的张量与预期的 correct_output 是否相等
        self.assertEqual(padded, correct_output)

    # 使用 dtypes 装饰器，测试四维张量的填充操作
    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim4(self, device, dtype):
        # 创建三个随机四维张量，指定设备和数据类型
        ts = [
            torch.randn(16, 21, 13, device=device, dtype=dtype),
            torch.randn(24, 32, 14, device=device, dtype=dtype),
            torch.randn(40, 53, 16, device=device, dtype=dtype),
        ]
        # 使用 nested_tensor 函数创建嵌套张量 nt
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        # 指定填充值为 42
        pad = 42
        # 用于存储正确输出结果的列表
        correct_output = []
        # 遍历每个张量 t
        for t in ts:
            # 创建与 ts[2] 相同形状的全为填充值 pad 的张量 next_output
            next_output = torch.ones_like(ts[2]) * pad
            # 将张量 t 的数据复制到 next_output 的前 t.size(0)、t.size(1)、t.size(2) 维度
            next_output[: t.size(0), : t.size(1), : t.size(2)].copy_(t)
            # 将 next_output 添加到 correct_output 中
            correct_output.append(next_output)
        # 将 correct_output 列表堆叠成一个张量
        correct_output = torch.stack(correct_output)
        # 调用 to_padded_tensor 函数对嵌套张量 nt 进行填充操作
        padded = torch.nested.to_padded_tensor(nt, pad)
        # 使用断言检查填充后的张量与预期的 correct_output 是否相等
        self.assertEqual(padded, correct_output)

    # TODO: test noncontiguous to_padded_tensor
    # 暂时测试非连续的 to_padded_tensor 功能
    # 和 to_padded_tensor 的错误消息
    # 因为 to_padded_tensor 尚不支持非连续缓冲区
    @dtypes(torch.float, torch.float16, torch.double)
    @torch.inference_mode()
    # 定义一个测试方法，用于测试非连续的嵌套张量转换为填充张量的功能
    def test_to_padded_tensor_noncontiguous(self, device, dtype):
        # 生成一个连续和一个非连续的随机嵌套张量对
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        # 调用非连续到填充张量的转换功能，并断言结果是否与预期的连续嵌套张量相等
        self.assertEqual(
            torch.nested.to_padded_tensor(nt_contiguous, 0.0),
            noncontiguous_to_padded_tensor(nt_noncontiguous),
        )
        # 测试当非连续嵌套张量作为输入时，to_padded_tensor 是否会抛出运行时错误并包含特定的错误信息
        self.assertRaisesRegex(
            RuntimeError,
            r"for now to_padded_tensor only supports contiguous nested tensor",
            lambda: torch.nested.to_padded_tensor(nt_noncontiguous, 0.0),
        )

    # 标记为跳过测试的元信息，用于测试设备检查功能
    @skipMeta
    # 测试设备检查功能，验证嵌套张量是否在给定设备上运行
    def test_device_checks(self, device):
        # 创建一个空的嵌套张量，设备由参数指定
        nt = torch.nested.nested_tensor([], device=device)
        # 检查设备是否为 CUDA 设备
        is_cuda = "cuda" in str(device)
        # 断言嵌套张量的 is_cuda 属性是否与预期的设备类型一致
        self.assertEqual(nt.is_cuda, is_cuda)

    # 用于测试多种数据类型的装饰器，验证张量操作的类型转换功能
    @dtypes(torch.float, torch.float16, torch.double)
    def test_nested_tensor_indexing(self, device, dtype):
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        # 断言：索引空的嵌套张量会引发 IndexError
        self.assertRaises(IndexError, lambda: nt0[0])
        
        # normal case
        x0 = torch.randn((2, 5), device=device, dtype=dtype)
        x1 = torch.randn((3, 4), device=device, dtype=dtype)
        # 创建嵌套张量 nt 包含 x0 和 x1
        nt = torch.nested.nested_tensor([x0, x1])
        
        # single index: only support integer in the batch dimension
        # 单个索引：只支持在批次维度使用整数索引
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        self.assertRaises(IndexError, lambda: nt[2])
        self.assertRaises(IndexError, lambda: nt[-3])
        self.assertRaises(NotImplementedError, lambda: nt[:])
        self.assertEqual(nt[...], nt)
        
        # tuple of indices: only support integer in the batch dimension
        #                 + all possible indexing in the original tensor dimensions
        # 索引元组：只支持在批次维度使用整数索引，并支持原始张量维度的所有可能索引
        self.assertEqual(nt[0, 0, 0], x0[0, 0])
        self.assertEqual(nt[0, 1, :], x0[1, :])
        self.assertEqual(nt[1, ...], x1)
        self.assertRaises(IndexError, lambda: nt[1, 4, 2])
        self.assertRaises(NotImplementedError, lambda: nt[:, 1, 1])
        
        # test select on non-batch dimensions
        # 在非批次维度上测试 select 方法
        self.assertEqual(nt.select(1, 0)[0], x0.select(0, 0))
        self.assertEqual(nt.select(1, 0)[1], x1.select(0, 0))
        self.assertRaises(IndexError, lambda: nt.select(1, 3))
        self.assertEqual(nt.select(2, 0)[0], x0.select(1, 0))
        self.assertEqual(nt.select(2, 0)[1], x1.select(1, 0))
        self.assertRaises(IndexError, lambda: nt.select(2, 5))
        
        # make sure indexing returns a view
        # 确保索引返回一个视图
        nt[0].fill_(100.0)
        answer = torch.tensor(100.0, device=device, dtype=dtype).expand((2, 5))
        self.assertEqual(nt[0], answer)
        nt[1, 1, :].fill_(200.0)
        answer = torch.tensor(200.0, device=device, dtype=dtype).expand(4)
        self.assertEqual(nt[1, 1, :], answer)

        # Test that indexing works when requires_grad_(True)
        # previously this was failing because the backward kernel for select.int uses .sizes()
        # 测试在 requires_grad_(True) 时索引是否工作
        # 以前由于 select.int 的反向内核使用 .sizes() 导致测试失败
        nt = torch.nested.nested_tensor([x0, x1]).requires_grad_(True)
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        grad_x0 = torch.randn((2, 5), device=device, dtype=dtype)
        nt[0].backward(grad_x0)
        expected_grad = torch.nested.nested_tensor(
            [grad_x0, torch.zeros((3, 4), device=device, dtype=dtype)]
        )
        self.assertEqual(nt.grad, expected_grad)
    @parametrize(
        "func",
        [
            subtest(torch.nn.functional.relu, name="relu"),
            subtest(torch.nn.functional.relu_, name="relu_"),
            subtest(torch.nn.functional.gelu, name="gelu"),
            subtest(torch._C._nn.gelu_, name="gelu_"),
            subtest(torch.tanh, name="tanh"),
            subtest(torch.tanh_, name="tanh_"),
            subtest(torch.neg, name="neg"),
            subtest(torch.nn.functional.silu, name="silu"),
            subtest(partial(torch.nn.functional.silu, inplace=True), name="silu_"),
            subtest(torch.abs, name="abs"),
            subtest(torch.abs_, name="abs_"),
            subtest(torch.sgn, name="sgn"),
            subtest(torch.logical_not, name="logical_not"),
            subtest(torch.sin, name="sin"),
            subtest(torch.cos, name="cos"),
        ],
    )
    # 参数化测试不同的激活函数和操作函数
    def test_activations(self, device, func):
        # 生成包含连续和非连续 NestedTensor 对的随机对
        nt, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device=device, dtype=torch.float32
        )
        # 使用当前函数 func 对 nt 进行嵌套计算
        nested_result = func(nt)
        # 断言嵌套结果是嵌套的
        self.assertTrue(nested_result.is_nested)
        # 对每个张量 t 和对应的 func(t) 结果 t_res 进行断言
        for t, t_res in zip(nt.unbind(), nested_result.unbind()):
            self.assertEqual(func(t), t_res)
        # 断言对非连续 NestedTensor 抛出 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            "NestedTensor must be contiguous to get buffer.",
            lambda: func(nt_noncontiguous),
        )

    @parametrize("func", [subtest(torch.ge, name="ge"), subtest(torch.eq, name="eq")])
    # 参数化测试使用不同的二元操作函数和标量
    def test_binary_ops_with_scalar(self, device, func):
        # 生成包含连续和非连续 NestedTensor 对的随机对
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device=device, dtype=torch.float32
        )
        scalar = 0.0

        # 对于每个 nt，进行二元操作 func(nt, scalar)
        for nt in (nt_contiguous, nt_noncontiguous):
            nested_result = func(nt, scalar)
            # 断言嵌套结果是嵌套的
            self.assertTrue(nested_result.is_nested)
            # 对每个张量 t 和对应的 func(t, scalar) 结果 t_res 进行断言
            for t, t_res in zip(nt.unbind(), nested_result.unbind()):
                self.assertEqual(func(t, scalar), t_res)

    @dtypes(*floating_types_and_half())
    # 定义一个测试方法，用于测试嵌套张量的分块操作
    def test_nested_tensor_chunk(self, device, dtype):
        # Transformer use case
        # 创建三个随机张量，形状为 (3, 12)，在指定设备上，并使用指定数据类型
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype)
        c = torch.randn(1, 3 * 4, device=device, dtype=dtype)
        # 在最后一个维度上将张量 a, b, c 分块为 3 份
        a_chunks = a.chunk(3, dim=-1)
        b_chunks = b.chunk(3, dim=-1)
        c_chunks = c.chunk(3, dim=-1)

        # 将每个张量的第一个分块组成列表 a_nt, b_nt, c_nt
        a_nt = [a_chunks[0], b_chunks[0], c_chunks[0]]
        b_nt = [a_chunks[1], b_chunks[1], c_chunks[1]]
        c_nt = [a_chunks[2], b_chunks[2], c_chunks[2]]

        # 创建嵌套张量 nt，包含张量 a, b, c
        nt = torch.nested.nested_tensor([a, b, c])
        # 在最后一个维度上将嵌套张量 nt 分块为 3 份
        chunked = nt.chunk(3, dim=-1)

        # 断言分块后的结果与预期的嵌套张量列表 a_nt, b_nt, c_nt 相等
        self.assertEqual(chunked[0], torch.nested.nested_tensor(a_nt))
        self.assertEqual(chunked[1], torch.nested.nested_tensor(b_nt))
        self.assertEqual(chunked[2], torch.nested.nested_tensor(c_nt))

        # 验证每个分块张量都不是连续的
        for chunk in chunked:
            self.assertFalse(chunk.is_contiguous())

        # 在不规则维度上尝试进行分块操作，预期抛出 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=0),
        )

        # 当嵌套张量 nt 非连续时，尝试进行分块操作，预期抛出 RuntimeError 异常
        _, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "chunk expects `self` to be contiguous.",
            lambda: torch.chunk(nt_noncontiguous, 5, dim=-1),
        )

        # 当分块数不可整除张量最后一个维度长度时，预期抛出 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is only supported for nested tensors with trailing dimension divisible by chunks.",
            lambda: torch.chunk(nt, 5, dim=-1),
        )

        # 在分块张量上调用反向传播，预期抛出 RuntimeError 异常
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        # 创建需要梯度的嵌套张量 nt_grad
        nt_grad = torch.nested.as_nested_tensor([a, b])
        # 在最后一个维度上将嵌套张量 nt_grad 分块为 2 份
        chunked = torch.chunk(nt_grad, 2, dim=-1)
        # 尝试在分块张量上调用反向传播，预期抛出 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            "derivative for aten::chunk is not implemented",
            lambda: chunked[0].backward(chunked[0].clone()),
        )

    @dtypes(*floating_types_and_half())
    # 定义一个测试方法，用于测试嵌套张量在指定设备和数据类型下的分割操作
    def test_nested_tensor_split_with_sizes(self, device, dtype):
        # 创建三个形状为 (3, 20) 的随机张量，分别命名为 a, b, c
        a = torch.randn(3, 20, device=device, dtype=dtype)
        b = torch.randn(2, 20, device=device, dtype=dtype)
        c = torch.randn(1, 20, device=device, dtype=dtype)

        # 指定分割大小为 [4, 6, 10]
        split_sizes = [4, 6, 10]
        # 分别按照指定大小在最后一个维度上对张量 a, b, c 进行分割
        a_splits = a.split_with_sizes(split_sizes, dim=-1)
        b_splits = b.split_with_sizes(split_sizes, dim=-1)
        c_splits = c.split_with_sizes(split_sizes, dim=-1)

        # 创建嵌套张量 nt，包含张量 a, b, c
        nt = torch.nested.nested_tensor([a, b, c])
        # 对嵌套张量 nt 进行按照 split_sizes 在最后一个维度上的分割
        nt_splits = nt.split_with_sizes(split_sizes, dim=-1)

        # 遍历分割后的嵌套张量 nt_splits
        for i, nt_split in enumerate(nt_splits):
            # 断言分割后的 nt_split 应与单独分割的 a_splits[i], b_splits[i], c_splits[i] 构成的嵌套张量相等
            self.assertEqual(
                nt_split,
                torch.nested.nested_tensor([a_splits[i], b_splits[i], c_splits[i]]),
            )
            # 创建一个张量，其中包含 a_splits[i], b_splits[i], c_splits[i] 的步长信息
            dense_strides = torch.stack(
                [
                    torch.tensor(a_splits[i].stride()),
                    torch.tensor(b_splits[i].stride()),
                    torch.tensor(c_splits[i].stride()),
                ]
            )
            # 断言 nt_split 的步长信息应与 dense_strides 相等
            self.assertEqual(nt_split._nested_tensor_strides(), dense_strides)
            # 断言 nt_split 不是连续的张量
            self.assertFalse(nt_split.is_contiguous())

        # 在不规则维度上调用时失败的情况
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes for nested tensors is currently only supported for the last dimension.",
            lambda: torch.split_with_sizes(nt, split_sizes, dim=1),
        )

        # 在非最后一个维度上调用时失败的情况
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes for nested tensors is currently only supported for the last dimension.",
            lambda: torch.split_with_sizes(nt, split_sizes, dim=0),
        )

        # 在非连续的 nt 上调用时失败的情况
        _, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes expects `self` to be contiguous.",
            lambda: torch.split_with_sizes(nt_noncontiguous, split_sizes, dim=-1),
        )

        # 使用不完全覆盖维度大小的 split_sizes 调用时失败的情况
        bad_split_sizes = [4, 6, 9]  # 总和不等于 20
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes expects split_sizes to sum exactly to 20",
            lambda: torch.split_with_sizes(nt, bad_split_sizes, dim=-1),
        )
    # 使用参数化测试框架，对transpose参数进行测试，分别测试True和False两种情况
    @parametrize("transpose", [True, False])
    def test_nested_tensor_add(self, device, dtype, transpose):
        # 如果transpose为True，生成随机张量a和b，并进行转置和连续化操作
        if transpose:
            a = torch.randn(2, 2, 2, device=device, dtype=dtype)
            b = torch.rand(2, 2, 2, device=device, dtype=dtype)
            c = a.transpose(-1, -2).contiguous()
            d = b.transpose(-1, -2).contiguous()
            # 创建嵌套张量nt1和nt2，并对nt2进行维度变换
            nt1 = torch.nested.nested_tensor([a, b, a, b])
            nt2 = torch.nested.nested_tensor([c, d, c, d]).transpose(-1, -2)
        else:
            # 如果transpose为False，调用random_nt_pair方法生成随机嵌套张量对nt1和nt2
            (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        # 计算参考结果ref，使用zip和unbind方法分别对nt1和nt2的元素进行加法运算
        ref = torch.nested.nested_tensor(
            [t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        # 计算输出结果out，对nt1和nt2进行加法运算
        out = nt1 + nt2
        # 断言输出结果与参考结果相等
        self.assertEqual(ref, out)

    # 使用参数化测试框架，对transpose参数进行测试，分别测试True和False两种情况
    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    @parametrize("transpose", [True, False])
    def test_nested_tensor_sub(self, device, dtype, transpose):
        # 如果transpose为True，生成随机张量a和b，并进行转置和连续化操作
        if transpose:
            a = torch.randn(2, 2, 2, device=device, dtype=dtype)
            b = torch.rand(2, 2, 2, device=device, dtype=dtype)
            c = a.transpose(-1, -2).contiguous()
            d = b.transpose(-1, -2).contiguous()
            # 创建嵌套张量nt1和nt2，并对nt2进行维度变换
            nt1 = torch.nested.nested_tensor([a, b, a, b])
            nt2 = torch.nested.nested_tensor([c, d, c, d]).transpose(-1, -2)
        else:
            # 如果transpose为False，调用random_nt_pair方法生成随机嵌套张量对nt1和nt2
            (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        # 计算参考结果ref，使用zip和unbind方法分别对nt1和nt2的元素进行减法运算
        ref = torch.nested.nested_tensor(
            [t1 - t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        # 计算输出结果out，对nt1和nt2进行减法运算
        out = nt1 - nt2
        # 断言输出结果与参考结果相等
        self.assertEqual(ref, out)

    # 使用参数化测试框架，对embedding_dim参数进行测试，测试不同的embedding维度
    @onlyCUDA
    @dtypes(torch.float, torch.float16)
    @torch.inference_mode()
    @parametrize("embedding_dim", [8, 128, 256, 384])
    def test_nested_tensor_dense_elementwise(self, device, dtype, embedding_dim):
        # 定义内部测试函数_test_add_mul，用于测试add和mul方法
        def _test_add_mul(nt, t):
            # 计算参考结果ref_add，使用zip和unbind方法分别对nt和t的元素进行加法运算
            ref_add = torch.nested.nested_tensor(
                [t1 + t2 for (t1, t2) in zip(nt.unbind(), t.unbind())]
            )
            # 计算参考结果ref_mul，使用zip和unbind方法分别对nt和t的元素进行乘法运算
            ref_mul = torch.nested.nested_tensor(
                [t1 * t2 for (t1, t2) in zip(nt.unbind(), t.unbind())]
            )
            # 断言nt调用add方法与参考结果ref_add相等
            self.assertEqual(nt.add(t), ref_add)
            # 断言nt调用mul方法与参考结果ref_mul相等
            self.assertEqual(nt.mul(t), ref_mul)

        batch_size = 32
        # 生成随机整数序列seq_lens，用作生成张量的长度
        seq_lens = torch.randint(low=0, high=10, size=(batch_size,))

        # [B, *, D], [B, 1, D] 情况
        # 生成随机张量列表ts，每个张量长度为seq_lens中的随机整数
        ts = [torch.randn((seq_len, embedding_dim)) for seq_len in seq_lens]
        # 创建嵌套张量nt和张量t，分别指定设备和数据类型
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        t = torch.randn((batch_size, 1, embedding_dim), device=device, dtype=dtype)
        # 调用内部测试函数_test_add_mul，测试nt和t的add和mul方法
        _test_add_mul(nt, t)

        # [B, *], [B, 1] 情况
        # 生成随机张量列表ts，每个张量长度为seq_lens中的随机整数
        ts = [torch.randn(seq_len) for seq_len in seq_lens]
        # 创建嵌套张量nt和张量t，分别指定设备和数据类型
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        t = torch.randn((batch_size, 1), device=device, dtype=dtype)
        # 调用内部测试函数_test_add_mul，测试nt和t的add和mul方法
        _test_add_mul(nt, t)
    # 定义测试方法，用于测试嵌套张量的乘法操作
    def test_nested_tensor_mul(self, device, dtype):
        # nested tensor * nested tensor
        # 生成随机的嵌套张量对(nt1, nt2)，包含4个子张量，每个子张量形状为(4, 4)
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        # 创建参考结果(ref)，通过逐个解绑子张量并逐元素相乘得到
        ref = torch.nested.nested_tensor(
            [t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        # 执行嵌套张量的乘法操作，得到计算结果(out)
        out = nt1 * nt2
        # 断言计算结果与参考结果相等
        self.assertEqual(ref, out)

        # nested tensor * scalar
        number = 10.0
        # 创建标量张量scalar，并将其转换到指定设备和数据类型
        scalar = torch.tensor(number).to(dtype).to(device)
        # 创建参考结果(ref)，通过逐个解绑子张量并逐元素与标量相乘得到
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        # 执行嵌套张量与标量的乘法操作，得到四种不同的计算结果
        out_number0 = nt1 * number
        out_number1 = number * nt1
        out_scalar0 = nt1 * scalar
        out_scalar1 = scalar * nt1
        # 断言四种计算结果均与参考结果相等
        self.assertEqual(out_number0, ref)
        self.assertEqual(out_number1, ref)
        self.assertEqual(out_scalar0, ref)
        self.assertEqual(out_scalar1, ref)

        # error case: numel == 1 but dim > 0
        # 创建包含单个元素的向量vector，并将其转换到指定设备和数据类型
        vector = torch.tensor([number]).to(dtype).to(device)
        # 断言执行乘法操作时出现预期的运行时错误，指示不允许非嵌套张量与嵌套张量进行乘法运算
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul(vector),
        )
        # 断言执行乘法操作时出现预期的运行时错误，指示不允许嵌套张量与非嵌套张量进行乘法运算
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul(nt1),
        )

    # 使用指定的数据类型进行测试，跳过元数据，进入推断模式，测试嵌套张量的除法操作
    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_div(self, device, dtype):
        # 生成随机的嵌套张量对(nt, nt2)，包含4个子张量，每个子张量形状为(4, 4)
        nt, nt2 = self.random_nt_pair(device, dtype, 4, (4, 4))
        scale = 4.0
        # 创建参考结果(ref)，通过逐个解绑子张量并逐元素除以标量得到
        ref = torch.nested.nested_tensor([t / scale for t in nt.unbind()])
        # 执行嵌套张量的除法操作，除以标量得到计算结果(out)
        out = nt / 4.0
        # 断言计算结果与参考结果相等
        self.assertEqual(ref, out)

        # 创建参考结果(ref_transposed)，通过逐个解绑子张量并逐元素除以标量后转置得到
        ref_transposed = ref.transpose(1, 2)
        # 执行嵌套张量的转置操作后除以标量，得到计算结果(out)，再与参考结果进行比较
        out = nt.transpose(1, 2) / 4.0
        self.assertEqual(ref_transposed, out)

        # 创建参考结果(ref)，通过逐个解绑子张量并逐元素除以对应位置的子张量得到
        ref = torch.nested.nested_tensor(
            [t / t2 for (t, t2) in zip(nt.unbind(), nt2.unbind())]
        )
        # 执行嵌套张量的逐元素除法操作，得到计算结果(out)，再与参考结果进行比较
        out = nt / nt2
        self.assertEqual(ref, out)

        # 执行嵌套张量的转置操作后逐元素除法，得到计算结果(out)，再与转置后的参考结果进行比较
        out = nt.transpose(1, 2) / nt2.transpose(1, 2)
        self.assertEqual(ref.transpose(1, 2), out)

        # 创建嵌套张量的拷贝(nt_transpose_copy)，先将每个子张量进行轴交换后重新封装
        nt_transpose_copy = torch.nested.nested_tensor(
            [t.transpose(0, 1) for t in nt.unbind()]
        )
        # 断言执行除法操作时出现预期的运行时错误，指示需要在给定嵌套张量时保持步幅匹配
        self.assertRaisesRegex(
            RuntimeError,
            "div requires strides to match when given NestedTensors",
            lambda: nt_transpose_copy.transpose(1, 2) / nt2,
        )

        # 创建嵌套张量(nt)，包含3个子张量，分别形状为[3, 4]，并将其转换到指定设备和数据类型
        nt = torch.nested.nested_tensor(
            [torch.randn(i, 4) for i in [3, 4, 5]], device=device, dtype=dtype
        )
        # 将嵌套张量(nt)按照指定轴进行分块操作，得到分块结果(nt_chunks)
        nt_chunks = nt.chunk(2, -1)
        # 断言执行除法操作时出现预期的运行时错误，指示需要在给定嵌套张量时保持偏移量匹配
        self.assertRaisesRegex(
            RuntimeError,
            "div requires offsets to match when given NestedTensors",
            lambda: nt_chunks[0] / nt_chunks[1],
        )
    # 定义测试方法，用于测试嵌套张量的原地加法
    def test_nested_tensor_add_in_place(self, device, dtype):
        # 生成随机的嵌套张量对
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        # 计算预期结果的参考值，即两个嵌套张量中每对张量元素的加法结果组成的嵌套张量
        ref = torch.nested.nested_tensor(
            [t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        # 原地更新第一个嵌套张量，执行原地加法操作
        nt1 += nt2
        # 断言原地加法后的结果与预期的参考值相等
        self.assertEqual(ref, nt1)

    # 使用装饰器定义测试方法，测试嵌套张量的原地乘法
    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul_in_place(self, device, dtype):
        # nested tensor * nested tensor
        # 生成随机的嵌套张量对
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        # 计算预期结果的参考值，即两个嵌套张量中每对张量元素的乘法结果组成的嵌套张量
        ref = torch.nested.nested_tensor(
            [t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        # 原地更新第一个嵌套张量，执行原地乘法操作
        nt1 *= nt2
        # 断言原地乘法后的结果与预期的参考值相等
        self.assertEqual(ref, nt1)

        # nested tensor * scalar
        number = 10.0
        scalar = torch.tensor(number).to(dtype).to(device)
        # 计算预期结果的参考值，即第一个嵌套张量中每个张量元素与标量乘法结果组成的嵌套张量
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        # 创建一个输出张量，复制第一个嵌套张量并进行乘以标量的操作
        out_number = nt1.clone()
        out_number *= number
        # 创建一个输出张量，复制第一个嵌套张量并进行乘以另一个张量标量的操作
        out_scalar = nt1.clone()
        out_scalar *= scalar
        # 断言两种乘法操作的输出结果与预期的参考值相等
        self.assertEqual(out_number, ref)
        self.assertEqual(out_scalar, ref)

        # 预期会引发运行时错误，因为标量乘法会导致形状不匹配
        self.assertRaisesRegex(
            RuntimeError,
            r"output with shape \[.*\] doesn't match the broadcast shape \[.*\]",
            lambda: scalar.mul_(nt1),
        )

        # 错误情况：张量元素个数为1，但维度大于0
        vector = torch.tensor([number]).to(dtype).to(device)
        # 断言会引发运行时错误，因为乘法操作期望两个操作数都是嵌套张量，但是其中一个不是
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul_(vector),
        )
        # 断言会引发运行时错误，因为乘法操作期望两个操作数都是嵌套张量，但是其中一个不是
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul_(nt1),
        )

    # 使用装饰器定义测试方法，仅在 CPU 上运行，用于测试特定的嵌套张量操作
    @onlyCPU
    @skipMeta
    @dtypes(torch.float)
    # 定义一个测试函数，用于测试嵌套张量的维度求和功能
    def test_nested_tensor_sum_dim(self, device, dtype):
        # 定义测试参数，每个元组包含张量数量和每个张量的最大尺寸
        params = ((2, (1, 1)), ((4), (4, 4)), (10, (3, 5, 7)))

        # 定义内部测试函数，用于测试张量的求和操作
        def test_sum(device, dtype, ntensors, max_sizes, dim, keepdim=True):
            # 使用随机生成的嵌套张量进行测试
            nt = random_nt(device, dtype, ntensors, max_sizes, require_non_empty=False)
            # 克隆一个张量用于后续的梯度计算
            nt2 = nt.clone()
            # 对克隆的张量进行解绑操作，得到一个张量列表
            ub2 = nt2.unbind()
            # 开启主张量的梯度追踪
            nt.requires_grad_(True)
            # 对解绑后的张量列表中的每个张量开启梯度追踪
            [t.requires_grad_(True) for t in ub2]
            # 对主张量进行指定维度的求和操作，保持维度
            nt_sum = nt.sum(dim=dim, keepdim=keepdim)
            # 对解绑后的张量列表中的每个张量进行相同维度的求和操作，保持维度
            ub2_sum = [t.sum(-1, keepdim=keepdim) for t in ub2]
            # 断言主张量求和结果与解绑后张量列表的张量求和结果是否相等
            self.assertEqual(nt_sum, torch.nested.nested_tensor(ub2_sum))

            # 测试反向传播
            # 生成与输出张量大小相同的梯度张量
            size = nt_sum._nested_tensor_size()
            gt2 = []
            for i in range(ntensors):
                gt2.append(torch.randn(size[i].tolist(), device=device, dtype=dtype))
            gt = torch.nested.nested_tensor(gt2).clone()
            # 执行主张量的反向传播
            nt_sum.backward(gt)
            # 对解绑后的张量列表中的每个张量执行反向传播
            for t2, g2 in zip(ub2_sum, gt2):
                t2.backward(g2)
            # 断言主张量的梯度与解绑后的张量列表中每个张量的梯度是否相等
            self.assertEqual(nt.grad, torch.nested.nested_tensor([t.grad for t in ub2]))
            return

        # 遍历参数列表，依次测试不同张量数量和最大尺寸的情况
        for ntensors, max_sizes in params:
            test_sum(device, dtype, ntensors, max_sizes, len(max_sizes))

        # 测试错误输入情况
        # 断言当尝试在非最后一个维度上对嵌套张量求和时会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor can only be reduced across the last"
        ):
            torch.nested.nested_tensor(
                [torch.tensor([3, 4, 5]), torch.tensor([1, 2])]
            ).sum(0, keepdim=True)

        # 断言当尝试在多个维度上对嵌套张量求和时会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor only allows reduction of a single"
        ):
            torch.nested.nested_tensor(
                [torch.tensor([[3, 4, 5]]), torch.tensor([[1, 2]])]
            ).sum([0, 1], keepdim=True)

        # 断言当前版本要求对嵌套张量的求和操作必须保持 keepdim=True
        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor always requires keepdim=True for now."
        ):
            torch.nested.nested_tensor(
                [torch.tensor([3, 4, 5]), torch.tensor([1, 2])]
            ).sum(-1)

    @dtypes(torch.float, torch.float16)
    # 定义一个测试方法，用于测试连续性的情况，接收设备和数据类型作为参数
    def test_contiguous(self, device, dtype):
        # 由于在 Python 中无法直接访问缓冲区，所以很难展示我们正在测试的内容。
        # 当我们对一个具有一致维度的嵌套张量进行 chunk 操作时
        # 对于 chunk_size > 1 的情况下，结果张量是原始嵌套张量的视图
        # 它们的元素数现在比缓冲区的大小要小。以前的 clone 操作是创建一个新的嵌套张量，
        # 其缓冲区的大小与原始张量相同。
        nt_contiguous = torch.nested.nested_tensor(
            [
                torch.randn(2, 20, device=device, dtype=dtype),
                torch.randn(4, 20, device=device, dtype=dtype),
            ]
        )
        # 将最后一个维度（长度为20）分成5个块
        chunks = nt_contiguous.chunk(5, dim=-1)

        # 遍历分块后的张量列表，验证每个块在调用 contiguous 后是否是连续的
        for chunk in chunks:
            self.assertFalse(chunk.is_contiguous())  # 检查块是否不连续
            self.assertTrue(chunk.contiguous().is_contiguous())  # 确保调用 contiguous 后块是连续的

    # 为了测试克隆操作，跳过对 torch.float16 的测试，因为 "bernoulli_scalar_cpu_" 操作在 'Half' 类型上未实现
    @dtypes(torch.float, torch.float16)
    @skipMeta
    def test_clone(self, device, dtype):
        # 创建一个随机嵌套张量 nt1
        nt1 = random_nt(device, dtype, 4, (4, 4), (1, 1))
        # 克隆 nt1，得到 nt2
        nt2 = nt1.clone()
        # 验证两个张量的值是否相等
        self.assertEqual(nt1, nt2)
        # 验证修改 nt2 是否不会影响 nt1
        nt2.mul_(nt1)
        ub1 = nt1.unbind()
        ub2 = nt2.unbind()
        for i in range(len(ub1)):
            self.assertNotEqual(ub1[i], ub2[i])

        # 测试保留内存格式（Preserve）的克隆操作
        nt1.clone(memory_format=torch.preserve_format)
        # 设置错误消息，用于验证调用 channels_last 内存格式时是否会引发 RuntimeError
        msg = "Nested tensor clone supports Preserve and Contiguous memory formats, called clone with memory format: ChannelsLast"
        with self.assertRaisesRegex(RuntimeError, msg):
            nt1.clone(memory_format=torch.channels_last)

    # 为了测试 torch.float16，跳过 torch.jagged 布局的测试
    @decorateIf(xfailIfTorchDynamo, lambda params: params["layout"] == torch.jagged)
    @dtypes(torch.float, torch.double)
    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    # 定义测试函数 `test_dropout`，用于测试 dropout 功能
    def test_dropout(self, device, dtype, layout):
        # edge case: empty nested tensor
        # 边界情况：空的嵌套张量
        # TODO: support empty NT in jagged layout
        # TODO: 支持不规则布局中的空 NT（暂未实现）
        if layout == torch.strided:
            # 创建一个空的嵌套张量 `nt0`，使用指定的布局 `layout`
            nt0 = torch.nested.nested_tensor([], layout=layout)
            # 对 `nt0` 应用 dropout 操作，保留概率为 0.5
            y = torch.nn.functional.dropout(nt0, 0.5)
            # 断言 `nt0` 和 `y` 应该相等
            self.assertEqual(nt0, y)
        # normal nested tensor
        # 普通的嵌套张量
        ntensors = 4
        if layout == torch.jagged:
            # 在指定设备和数据类型上创建随机嵌套张量 `nt`，使用不规则布局 `jagged`
            nt = random_nt(device, dtype, ntensors, (4, 4), (0, 3), layout=layout)
        else:
            # 在指定设备和数据类型上创建随机嵌套张量 `nt`，使用普通布局 `layout`
            nt = random_nt(device, dtype, ntensors, (4, 4), layout=layout)
        # edge case: invalid dropout
        # 边界情况：无效的 dropout 概率
        # 检查是否会引发 ValueError 异常
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(-0.1))
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(1.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, -0.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, 1.1))
        # edge case: no dropout
        # 边界情况：不应用 dropout
        # 创建 dropout 操作对象 `dropouter`，保留概率为 0.0
        dropouter = torch.nn.Dropout(0.0)
        # 对 `nt` 应用 `dropouter`
        y0 = dropouter(nt)
        # 对 `nt` 应用 `torch.nn.functional.dropout`，保留概率为 0.0
        y1 = torch.nn.functional.dropout(nt, 0.0)
        # 断言 `nt` 和 `y0` 应该相等
        self.assertEqual(nt, y0)
        # 断言 `nt` 和 `y1` 应该相等
        self.assertEqual(nt, y1)
        # edge case: all dropout
        # 边界情况：全部应用 dropout
        # 创建 dropout 操作对象 `dropouter`，保留概率为 1.0
        dropouter = torch.nn.Dropout(1.0)
        # 对 `nt` 应用 `dropouter`
        y0 = dropouter(nt)
        # 对 `nt` 应用 `torch.nn.functional.dropout`，保留概率为 1.0
        y1 = torch.nn.functional.dropout(nt, 1.0)
        # 创建一个与 `nt` 形状相同的全零张量 `nt0`
        nt0 = torch.zeros_like(nt)
        # 断言 `y0` 和 `nt0` 应该相等
        self.assertEqual(nt0, y0)
        # 断言 `y1` 和 `nt0` 应该相等
        self.assertEqual(nt0, y1)
        # normal case: normal dropout
        # 普通情况：普通的 dropout
        p = 0.2
        # 对 `nt` 应用 dropout 操作，保留概率为 `p`
        y = torch.nn.functional.dropout(nt, p)
        # 创建期望结果 `expect`，复制 `nt`
        expect = nt.clone()
        if layout == torch.jagged:
            # 如果布局是不规则的，按元素更新 `expect` 以反映 dropout 的影响
            expect = torch.where(y == 0.0, y, nt)
            expect /= 1.0 - p
            # 断言 `y` 和 `expect` 应该相等
            self.assertEqual(y, expect)
        else:
            # 如果布局是普通的，按张量和元素更新 `expect` 以反映 dropout 的影响
            expect = nt.clone()
            for i in range(ntensors):
                actual_tensor = y[i].view(-1)
                expect_tensor = expect[i].view(-1)
                for j in range(actual_tensor.shape[0]):
                    if actual_tensor[j].item() == 0.0:
                        expect_tensor[j] = 0.0
                    else:
                        expect_tensor[j] /= 1.0 - p
            # 断言 `y` 和 `expect` 应该相等
            self.assertEqual(y, expect)
        with freeze_rng_state():
            # 使用冻结的随机数生成状态创建 `dropouter` 对象，保留概率为 `p`
            dropouter = torch.nn.Dropout(p)
            # 对 `nt` 应用 `dropouter`
            y0 = dropouter(nt)
        with freeze_rng_state():
            # 使用冻结的随机数生成状态对 `nt` 应用 `torch.nn.functional.dropout`，保留概率为 `p`
            y1 = torch.nn.functional.dropout(nt, p)
        # 断言 `y0` 和 `y1` 应该相等
        self.assertEqual(y0, y1)

    @dtypes(torch.float, torch.double)
    # 定义测试函数 `test_dropout_noncontiguous`，测试非连续内存布局下的 dropout
    def test_dropout_noncontiguous(self, device, dtype):
        ntensors = 4
        # 创建一个随机的嵌套张量 `nt0`，在指定设备和数据类型上
        nt0 = random_nt(device, dtype, ntensors, (4, 4))
        # 将 `nt0` 沿着最后两个维度进行转置，得到 `nt1`
        nt1 = nt0.transpose(-1, -2)
        p = 0.3
        with freeze_rng_state():
            # 使用冻结的随机数生成状态创建 `dropouter` 对象，保留概率为 `p`
            dropouter = torch.nn.Dropout(p)
            # 对 `nt0` 应用 `dropouter`
            y0 = dropouter(nt0)
        with freeze_rng_state():
            # 使用冻结的随机数生成状态对 `nt1` 应用 `torch.nn.functional.dropout`，保留概率为 `p`，然后再进行转置
            y1 = torch.nn.functional.dropout(nt1, p).transpose(-1, -2)
        # 断言 `y0` 和 `y1` 应该相等
        self.assertEqual(y0, y1)
    # 定义测试 softmax 的函数，接受设备和数据类型作为参数
    def test_softmax(self, device, dtype):
        # 创建包含 4 个张量的嵌套张量，形状为 (4, 4)，使用指定的设备和数据类型
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))
        
        # 错误情况：在嵌套维度 0 上应用 softmax 是不允许的
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, 0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, -3),
        )
        
        # 错误情况：维度超出范围
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, 3))
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, -4))
        
        # 正常情况：使用 Softmax 对象 softmaxer，沿着维度 1 计算 softmax
        softmaxer = torch.nn.Softmax(1)
        y0 = softmaxer(nt)
        y1 = torch.nn.functional.softmax(nt, 1)
        self.assertEqual(y0, y1)
        
        # 将嵌套张量 nt 转换为填充张量 pt，填充值为负无穷
        pt = torch.nested.to_padded_tensor(nt, float("-inf"))
        
        # 如果整个切片被填充，则 softmax 将返回 nan，这里将其转换为 0.0
        expect = torch.nn.functional.softmax(pt, 1).nan_to_num_(0.0)
        self.assertEqual(torch.nested.to_padded_tensor(y0, 0.0), expect)
        
        # 边界情况：空的嵌套张量 nt0，对其沿维度 1 计算 softmax 应返回自身
        nt0 = torch.nested.nested_tensor([])
        y = torch.nn.functional.softmax(nt0, 1)
        self.assertEqual(nt0, y)
        
        # 边界情况：嵌套标量 nt1，尝试在维度 0 上应用 softmax 会引发 RuntimeError
        nt1 = torch.nested.nested_tensor([torch.tensor(0.0), torch.tensor(1.0)])
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.softmax(nt1, 0))
        # 尝试在维度 1 上应用 softmax 会引发 IndexError
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt1, 1))
    # 测试非连续情况下的 bmm 函数
    def test_bmm_noncontiguous(self, device, dtype):
        # 生成两组张量，一组连续，一组非连续
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3), device, dtype
        )
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair(
            (6, 7), device, dtype
        )
        # 断言两组张量经转置后使用 bmm 函数的结果相等
        self.assertEqual(
            nt0_contiguous.transpose(-1, -2).bmm(nt1_contiguous),
            nt0_noncontiguous.transpose(-1, -2).bmm(nt1_noncontiguous),
        )

    # 使用指定数据类型进行测试，支持浮点数和双精度
    @dtypes(torch.float, torch.double)
    def test_matmul_with_bmm_path(self, device, dtype):
        # 定义解绑和重新绑定 matmul 函数
        def unbind_rebind_matmul(nt1, nt2):
            # 将第一个张量和第二个张量解绑，进行 matmul 操作后返回
            t1s = nt1.unbind()
            t2s = nt2.unbind()
            out_ts = [t1.matmul(t2) for t1, t2 in zip(t1s, t2s)]
            return torch.nested.nested_tensor(out_ts)

        # [N, n_head, *, head_dim], [N, n_head, head_dim, *] 维度参数
        Ns = [1, 2, 5]
        n_heads = np.random.randint(2, 5)
        head_dim = 3
        t1s = []
        t2s = []
        for N in Ns:
            for _ in range(N):
                seq_len1 = np.random.randint(2, 5)
                seq_len2 = np.random.randint(2, 5)
                t1s.append(torch.randn(n_heads, seq_len1, head_dim))
                t2s.append(torch.randn(n_heads, head_dim, seq_len2))
            # 使用给定设备和数据类型创建 nested_tensor
            nt1 = torch.nested.nested_tensor(t1s, device=device, dtype=dtype)
            nt2 = torch.nested.nested_tensor(t2s, device=device, dtype=dtype)
            # 断言使用 matmul 函数和自定义函数的结果相等
            self.assertEqual(torch.matmul(nt1, nt2), unbind_rebind_matmul(nt1, nt2))

        # 测试非连续情况
        t3s = []
        t4s = []
        for _ in range(N):
            seq_len = np.random.randint(2, 5)
            t3s.append(torch.randn(seq_len, n_heads, head_dim))
            t4s.append(torch.randn(seq_len, n_heads, head_dim))
        # 对 nt3 和 nt4 进行转置操作
        nt3 = torch.nested.nested_tensor(t3s, device=device, dtype=dtype).transpose(
            1, 2
        )
        nt4 = (
            torch.nested.nested_tensor(t4s, device=device, dtype=dtype)
            .transpose(1, 2)
            .transpose(2, 3)
        )
        # 断言使用 matmul 函数和自定义函数的结果相等
        self.assertEqual(torch.matmul(nt3, nt4), unbind_rebind_matmul(nt3, nt4))

    # 不能测试 torch.float16 类型，因为: RuntimeError: "bmm" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    # 目前仅在 CUDA 上支持
    @dtypes(torch.float, torch.double)
    def test_matmul_nt_with_broadcasted_t(self, device, dtype):
        # NT (B, *, C, D) 与 T (D, E) 的广播情况
        nt = random_nt_from_dims([3, None, 4, 5], device=device, dtype=dtype)
        t = torch.randn(5, 6, device=device, dtype=dtype)
        output = torch.matmul(nt, t)

        # 应当等同于将每个组件与密集张量进行 matmul 运算
        self.assertEqual(nt.size(0), output.size(0))
        for component, out_component in zip(nt, output):
            self.assertEqual(out_component, torch.matmul(component, t))

    # 不能测试 torch.float16 类型，因为: RuntimeError: "bmm" not implemented for 'Half'
    # 使用装饰器指定此测试方法支持的数据类型为 torch.float 和 torch.double
    @dtypes(torch.float, torch.double)
    # 定义一个测试方法，用于测试非连续存储的矩阵乘法
    def test_matmul_noncontiguous(self, device, dtype):
        # 生成两组随机的非连续存储的矩阵对，一个是连续存储，一个是非连续存储，设备为指定的设备，数据类型为指定的类型
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3), device, dtype
        )
        # 同上，生成另一组随机的非连续存储的矩阵对
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair(
            (6, 7), device, dtype
        )
        # 断言：对两组矩阵执行转置后的乘法运算结果应该相等
        self.assertEqual(
            torch.matmul(nt0_contiguous.transpose(-1, -2), nt1_contiguous),
            torch.matmul(nt0_noncontiguous.transpose(-1, -2), nt1_noncontiguous),
        )
    
    # 使用装饰器指定此测试方法支持的数据类型为 torch.float 和 torch.double
    @dtypes(torch.float, torch.double)
    # 定义一个测试函数，用于测试线性函数的不同情况
    def test_linear(self, device, dtype):
        # 创建一个形状为 (1, 2) 的随机张量 a，使用指定设备和数据类型
        a = torch.randn(1, 2, device=device, dtype=dtype)
        # 创建一个形状为 (2, 2) 的随机张量 b，使用指定设备和数据类型
        b = torch.randn(2, 2, device=device, dtype=dtype)
        # 创建一个形状为 (3, 2) 的随机张量 c，使用指定设备和数据类型
        c = torch.randn(3, 2, device=device, dtype=dtype)
        # 创建一个嵌套张量 nt，包含张量 a、b、c
        nt = torch.nested.nested_tensor([a, b, c])

        # 创建一个形状为 (2, 2) 的随机权重张量，使用指定设备和数据类型
        weight = torch.randn(2, 2, device=device, dtype=dtype)
        # 创建一个形状为 (2,) 的随机偏置张量，使用指定设备和数据类型
        bias = torch.randn(2, device=device, dtype=dtype)

        # 成功的情况下，应用线性函数 F.linear 到嵌套张量 nt 上，使用给定的权重和偏置
        torch.functional.F.linear(nt, weight, bias)

        # 当嵌套张量的维度不正确时，引发 RuntimeError 异常，验证错误消息格式
        msg = r"Linear requires nested_tensor.dim == 3 and dense_matrix.dim == 2. Nested tensor dim: 2. Dense tensor dim: 2"
        nt1 = torch.nested.nested_tensor(
            [
                torch.randn(1, device=device, dtype=dtype),
                torch.randn(2, device=device, dtype=dtype),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt1, weight, bias)

        # 当权重张量的形状不正确时，引发 RuntimeError 异常，验证错误消息格式
        msg = r"Linear requires nested_tensor.dim == 3 and dense_matrix.dim == 2. Nested tensor dim: 3. Dense tensor dim: 3"
        weight1 = torch.randn(2, 2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, weight1, bias)

        # 当嵌套张量中张量的最后一个维度不一致时，引发 RuntimeError 异常，验证错误消息格式
        msg = r"Expected all tensors in nested tensor to have the same trailing dimension, instead last dimension equals:"
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(1, 2, device=device, dtype=dtype),
                torch.randn(2, 3, device=device, dtype=dtype),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt2, weight, bias)

        # 当嵌套张量的最后一个维度与权重张量的第二个维度不匹配时，引发 RuntimeError 异常，验证错误消息格式
        weight2 = torch.randn(2, 4, device=device, dtype=dtype)
        msg = (
            r"Shape mismatch for NestedTensor Linear: Expected input's \(a nested tensor\) 'last_dim'"
            r" to equal 'weight.size\(1\), but got: last_dim = 2, and weight.size\(1\) = 4"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, weight2, bias)

        # 当输入是嵌套张量且权重也是嵌套张量时，引发 RuntimeError 异常，验证错误消息格式
        nt_weight = nt.clone()
        msg = r"Linear does not support nested weight when input is a nested tensor."
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, nt_weight, bias)
    # 定义测试函数，测试非连续线性张量的情况
    def test_linear_noncontiguous(self, device, dtype):
        # 获取一个非连续和一个连续的随机张量对
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        # 随机初始化权重张量
        weight = torch.randn((8, 5), device=device, dtype=dtype)
        # 断言期望运行时错误，并验证错误消息
        self.assertRaisesRegex(
            RuntimeError,
            r"for now linear only supports contiguous nested tensor",
            # 调用线性函数，传入非连续张量和权重
            lambda: torch.nn.functional.linear(nt_noncontiguous, weight),
        )

    # 使用指定数据类型测试填充为零的张量处理函数，检测非零元素数目为零的错误
    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_zero_numel_errors(self, device, dtype):
        # 创建两个大小为 [1, 0] 和 [0, 0] 的张量
        ts = [torch.ones(1, 0), torch.ones(0, 0)]
        # 创建嵌套张量，指定设备、数据类型和布局
        nt = torch.nested.nested_tensor(
            ts, device=device, dtype=dtype, layout=torch.strided
        )
        # 断言期望运行时错误，并验证错误消息
        self.assertRaisesRegex(
            RuntimeError,
            r"at least one constituent tensor should have non-zero numel",
            # 调用转换为填充张量函数，传入嵌套张量和填充值
            lambda: torch.nested.to_padded_tensor(nt, 0.0),
        )

    # 使用指定数据类型测试转置操作函数
    @dtypes(torch.float, torch.float16, torch.double)
    def test_transpose(self, device, dtype):
        # 创建随机嵌套张量
        nt = random_nt(device, dtype, 4, (4, 4))
        # 错误情况：尝试转置嵌套维度
        self.assertRaisesRegex(
            RuntimeError,
            "Nested tensor dimension 0 cannot be transposed",
            # 调用转置函数，尝试转置维度 0 和 1
            lambda: nt.transpose(0, 1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Nested tensor dimension 0 cannot be transposed",
            # 调用转置函数，尝试转置维度 1 和 -3
            lambda: nt.transpose(1, -3),
        )
        # 错误情况：维度超出范围
        self.assertRaises(IndexError, lambda: nt.transpose(1, 3))
        self.assertRaises(IndexError, lambda: nt.transpose(-4, -1))
        # 正常情况：执行转置操作，并转换为填充张量
        ntT = nt.transpose(-1, -2)
        ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        ptT = pt.transpose(-1, -2)
        # 断言期望相等的填充张量转置结果
        self.assertEqual(ptT, ptT_from_ntT)
    # 定义一个测试方法，用于测试 squeeze 和 unsqueeze 操作在 nested tensors 上的行为
    def test_squeeze_unsqueeze(self, device, dtype):
        # 创建一个形状为 (2, 3) 的张量 a，包含元素 0 到 5
        a = torch.arange(6).reshape(2, 3)
        # 创建一个形状为 (5, 3) 的张量 b，包含元素 0 到 14
        b = torch.arange(15).reshape(5, 3)
        # 使用 nested_tensor 函数创建一个 nested tensor nt，包含张量 a 和 b
        nt = torch.nested.nested_tensor([a, b], device=device, dtype=dtype)

        # 错误情况：在没有指定维度的情况下进行 squeeze 操作
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeeze without the dim argument",
            lambda: nt.squeeze(),
        )

        # 错误情况：对嵌套维度进行 squeeze 操作
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeezing dimension 0",
            lambda: nt.squeeze(0),
        )

        # 错误情况：维度超出范围的 squeeze 操作
        self.assertRaises(IndexError, lambda: nt.squeeze(3))

        # 错误情况：对包含单个张量的嵌套张量进行 squeeze 操作
        c = torch.ones(1)
        nt_singleton = torch.nested.nested_tensor([c, c], device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeezing a nested tensor of singleton",
            lambda: nt_singleton.squeeze(1),
        )

        # 对维度大小不为 1 的维度进行 squeeze 操作，预期操作不变
        nt2 = nt.squeeze(-1)
        self.assertEqual(nt, nt2)

        # 测试应该成功的情况
        nt_sizes = nt._nested_tensor_size()
        nt_strides = nt._nested_tensor_strides()
        # 遍历一系列维度值，测试 unsqueeze 操作
        for i in range(-2, 4):
            if i == 0:
                # 不能对批次维度进行 unsqueeze 操作
                continue
            # 执行 unsqueeze 操作
            nt_unsqueezed = nt.unsqueeze(i)
            # 对于负的维度索引，会对应到 dim = dim + nt.dim() + 1 的 unsqueeze() 操作
            wrapped_i = i + nt.dim() + 1 if i < 0 else i
            # 对应到 nt 大小张量的索引需要减去 1 来忽略批次维度
            size_idx = wrapped_i - 1
            # 断言新增维度的尺寸为 1
            self.assertEqual(
                nt_unsqueezed._nested_tensor_size()[:, size_idx],
                torch.ones(2, dtype=torch.long),
            )
            # 获取 unsqueeze 后的步长
            unsqueezed_stride = nt_unsqueezed._nested_tensor_strides()[:, size_idx]
            if i == nt.ndim or i == -1:
                self.assertEqual(unsqueezed_stride, torch.ones(2, dtype=torch.long))
            else:
                # 计算 unsqueeze 后的步长与原始张量的步长和大小的乘积的关系
                stride_col_after = nt_strides[:, size_idx]
                size_col_after = nt_sizes[:, size_idx]
                self.assertEqual(unsqueezed_stride, stride_col_after * size_col_after)
            # 执行 squeeze 操作
            nt_squeezed = nt_unsqueezed.squeeze(i)
            # 断言 squeeze 后的张量与原始张量相等
            self.assertEqual(nt_squeezed, nt)
            # 断言 squeeze 后的张量尺寸与步长与原始张量相同
            self.assertEqual(nt_squeezed._nested_tensor_size(), nt_sizes)
            self.assertEqual(nt_squeezed._nested_tensor_strides(), nt_strides)
    # 定义测试函数，测试转置和推断模式的交互
    def test_transpose_inference_mode_interaction(self, device, dtype):
        # 生成随机的嵌套张量
        nt = random_nt(device, dtype, 4, (4, 4))

        # 进入推断模式，构建默认模式并在推断模式下进行转置操作
        with torch.inference_mode():
            # 对嵌套张量进行转置操作
            ntT = nt.transpose(-1, -2)
            # 将非连续的张量转换为填充张量
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            # 将嵌套张量转换为填充张量
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            # 对填充张量进行转置操作
            ptT = pt.transpose(-1, -2)
            # 断言两个填充张量是否相等
            self.assertEqual(ptT, ptT_from_ntT)

        # 再次进入推断模式，构建并在推断模式下进行转置操作
        with torch.inference_mode():
            # 生成随机的嵌套张量
            nt = random_nt(device, dtype, 4, (4, 4))
            # 对嵌套张量进行转置操作
            ntT = nt.transpose(-1, -2)
            # 将非连续的张量转换为填充张量
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            # 将嵌套张量转换为填充张量
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            # 对填充张量进行转置操作
            ptT = pt.transpose(-1, -2)
            # 断言两个填充张量是否相等
            self.assertEqual(ptT, ptT_from_ntT)

    # 使用装饰器定义允许的数据类型
    @dtypes(torch.float, torch.float16, torch.double)
    # 定义一个测试方法，用于测试 nested_tensor 对象的视图操作
    def test_view(self, device, dtype):
        # 创建一个随机的 nested_tensor 对象
        nt = random_nt(device, dtype, 4, (4, 4))
        
        # 错误情况: 空的形状
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[\]' is invalid for a nested tensor",
            lambda: nt.view(()),
        )
        
        # 错误情况: 空的 nested_tensor 对象
        nt_empty = torch.nested.nested_tensor([])
        self.assertRaisesRegex(
            RuntimeError,
            "empty nested tensor cannot be reshaped",
            lambda: nt_empty.view(-1),
        )
        
        # 错误情况: -1 作为批处理大小
        self.assertRaisesRegex(
            RuntimeError,
            r"view: For now nested view cannot change or infer the implicit batch dimension",
            lambda: nt.view(-1, 2, 3),
        )
        
        # 错误情况: 给定形状与实际输入大小不匹配
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[.*\]' is invalid for input of size [0-9]+",
            lambda: nt.view(4, 2, 3),
        )
        
        # 正常情况下的测试
        x0 = torch.randn((2, 20), device=device, dtype=dtype)
        x1 = torch.randn((3, 20), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        
        # 错误情况: 尝试将批处理维度重塑为合法形状
        self.assertRaisesRegex(
            RuntimeError,
            r"For now nested view cannot change or infer the implicit batch dimension",
            lambda: nt.transpose(-1, -2).view(40, -1),
        )
        
        # 继承不规则维度，将 (2, 20) -> (2, 5, 4)，(3, 20) -> (3, 5, 4)
        nt1 = nt.view(2, -1, 5, 4)
        
        # 将 (2, 3, 20) -> (2, 3, 5, 4) -> (2, 4, 5, 4)
        pt1 = pt.view(2, -1, 5, 4)
        
        # 断言非连续的 nested_tensor 转换后与预期的 padded_tensor 相等
        self.assertEqual(noncontiguous_to_padded_tensor(nt1), pt1)
        
        # 多个 -1 （即“旧”维度），应该失败
        # 尝试将 (2, (2, 3), 5, 4) -> (2, (2, 3), 5, 2, 2)，但我们禁止多个维度推断旧行为
        self.assertRaisesRegex(
            RuntimeError,
            r"only one dimension can be inferred",
            lambda: nt1.view(2, -1, -1, 2, 2),
        )

    @dtypes(torch.float, torch.float16, torch.double)
    # 定义测试方法，用于测试视图推理模式下的交互操作
    def test_view_inference_mode_interaction(self, device, dtype):
        # 在默认模式下构造 nested_tensor，并在推理模式下进行视图操作
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 20)), torch.randn((3, 20))], device=device, dtype=dtype
        )
        # 进入推理模式上下文
        with torch.inference_mode():
            # 对 nested_tensor 进行视图重塑
            ntT = nt.view(2, -1, 4, 5)
            # 调用非连续数据到填充张量的转换函数
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            # 使用 nested_tensor 创建填充张量
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            # 对填充张量进行视图重塑
            ptT = pt.view(2, -1, 4, 5)
            # 断言填充后的张量与从 nested_tensor 得到的张量相等
            self.assertEqual(ptT, ptT_from_ntT)
        # 再次进入推理模式上下文
        with torch.inference_mode():
            # 构造 nested_tensor 并在推理模式下进行视图操作
            nt = torch.nested.nested_tensor(
                [torch.randn((2, 20)), torch.randn((3, 20))], device=device, dtype=dtype
            )
            # 对 nested_tensor 进行视图重塑
            ntT = nt.view(2, -1, 4, 5)
            # 调用非连续数据到填充张量的转换函数
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            # 使用 nested_tensor 创建填充张量
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            # 对填充张量进行视图重塑
            ptT = pt.view(2, -1, 4, 5)
            # 断言填充后的张量与从 nested_tensor 得到的张量相等
            self.assertEqual(ptT, ptT_from_ntT)

    @dtypes(torch.float, torch.float16, torch.double)
    # 定义测试函数 test_reshape，用于测试张量的重塑操作
    def test_reshape(self, device, dtype):
        # 生成一个随机的嵌套张量 nt，形状为 (4, (4, 4))
        nt = random_nt(device, dtype, 4, (4, 4))
        
        # 错误情况: 空的形状
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[\]' is invalid for a nested tensor",
            lambda: nt.reshape(()),
        )
        
        # 错误情况: 空的嵌套张量
        nt_empty = torch.nested.nested_tensor([])
        self.assertRaisesRegex(
            RuntimeError,
            "empty nested tensor cannot be reshaped",
            lambda: nt_empty.reshape(-1),
        )
        
        # 错误情况: 尝试改变或推断隐式批处理维度
        self.assertRaisesRegex(
            RuntimeError,
            r"reshape: For now nested reshape cannot change or infer the implicit batch dimension",
            lambda: nt.reshape(-1, 2, 3),
        )
        
        # 错误情况: 形状不匹配当前输入大小
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[.*\]' is invalid for input of size [0-9]+",
            lambda: nt.reshape(4, 2, 3),
        )
        
        # 正常情况: 创建两个随机张量 x0 和 x1，形状分别为 (2, 20) 和 (3, 20)
        x0 = torch.randn((2, 20), device=device, dtype=dtype)
        x1 = torch.randn((3, 20), device=device, dtype=dtype)
        
        # 创建嵌套张量 nt，包含 x0 和 x1，形状为 (2, (2, 3), 20)
        nt = torch.nested.nested_tensor([x0, x1])
        
        # 将 nt 转换为填充张量 pt，填充值为 0.0
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        
        # 错误情况: 尝试将批处理维度重塑为合法的形状
        self.assertRaisesRegex(
            RuntimeError,
            r"reshape: For now nested reshape cannot change or infer the implicit batch dimension",
            lambda: nt.transpose(-1, -2).reshape(40, -1),
        )
        
        # 继承不规则维度，并重塑为 (2, -1, 5, 4) 的形状
        nt1 = nt.reshape(2, -1, 5, 4)
        
        # 将填充张量 pt 也按相同的形状 (2, -1, 5, 4) 进行重塑
        pt1 = pt.reshape(2, -1, 5, 4)
        
        # 断言非连续数据转换为填充张量后的结果相等
        self.assertEqual(noncontiguous_to_padded_tensor(nt1), pt1)
        
        # 错误情况: 尝试使用超过一个 -1 来推断维度，应该失败
        # 这里尝试从 (2, (2, 3), 5, 4) -> (2, (2, 3), 5, 2, 2)
        # 但我们禁止对多个维度进行“继承旧行为”的操作
        self.assertRaisesRegex(
            RuntimeError,
            r"only one dimension can be inferred",
            lambda: nt1.reshape(2, -1, -1, 2, 2),
        )
    # 定义一个测试方法，用于测试嵌套张量的 narrow 操作
    def test_narrow(self, device, dtype):
        # 生成一个随机的嵌套张量，包含5个维度，其他维度大小随机
        nt = random_nt_from_dims([5, None, None, None], device=device, dtype=dtype)

        # 在 dim=0 上进行 narrow 操作，从起始到结束位置
        bounds = [(0, 5), (0, 3), (1, 2), (1, 5), (2, 4)]
        for start, end in bounds:
            length = end - start
            # 执行 narrow 操作
            narrowed = nt.narrow(dim=0, start=start, length=length)
            # 确保输出是一个视图
            self.assertTrue(narrowed._base is nt)
            # 检查每个分片是否与原始张量对应部分相等
            for nc, c in zip(narrowed.unbind(), nt.unbind()[start:end]):
                self.assertEqual(nc, c)

        # 如果 dim != 0，则抛出异常，不支持非零维度的 narrow 操作
        for dim in range(1, nt.dim()):
            with self.assertRaisesRegex(
                RuntimeError, "only dim=0 supported for nested tensors"
            ):
                nt.narrow(dim=dim, start=0, length=1)

        # 错误情况: 非连续的嵌套张量无法进行 narrow 操作
        _, nt_noncont = random_nt_noncontiguous_pair((2, 3, 4))
        with self.assertRaisesRegex(
            RuntimeError, "only contiguous nested tensors supported"
        ):
            nt_noncont.narrow(dim=0, start=0, length=1)
    # 定义一个测试函数，用于测试 torch.empty_like() 方法
    def test_empty_like(self, device, dtype):
        # 生成随机的嵌套张量
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))

        # 在与原始嵌套张量相同的设备上创建一个空张量
        nt_empty = torch.empty_like(nt)
        # 断言空张量与原始张量具有相同的大小
        assert nt.is_same_size(nt_empty)
        # 断言空张量与原始张量具有相同的数据类型
        self.assertEqual(nt.dtype, nt_empty.dtype)
        # 断言空张量与原始张量在相同的设备上
        self.assertEqual(nt.device, nt_empty.device)
        # 断言空张量与原始张量具有相同的布局
        self.assertEqual(nt.layout, nt_empty.layout)

        # 检查更改 empty_like 嵌套张量输出的数据类型
        dtype_set = {torch.float, torch.float16, torch.double}
        for other_dtype in dtype_set - {dtype}:
            nt_empty_other_dtype = torch.empty_like(nt, dtype=other_dtype)
            self.assertEqual(nt.dtype, dtype)
            self.assertEqual(nt_empty_other_dtype.dtype, other_dtype)
            self.assertEqual(nt.device, nt_empty.device)
            self.assertEqual(nt.layout, nt_empty.layout)

        # 为自动求导创建张量
        nt_empty_req_grad = torch.empty_like(nt, requires_grad=True)
        self.assertEqual(nt_empty_req_grad.requires_grad, True)

        # 测试非连续张量不会失败
        nt_cont, nt_noncont = random_nt_noncontiguous_pair((2, 3, 6, 7))
        nt_empty = torch.empty_like(nt_cont)
        assert nt_cont.is_same_size(nt_empty)
        nt_empty_non_contig = torch.empty_like(nt_noncont)
        assert nt_noncont.is_same_size(nt_empty_non_contig)

        # 测试连续内存格式选项
        nt_empty_contig = torch.empty_like(
            nt_cont, memory_format=torch.contiguous_format
        )
        assert nt_cont.is_same_size(nt_empty_contig)
        assert nt_empty_contig.is_contiguous()

        nt_empty_non_contig = torch.empty_like(
            nt_noncont, memory_format=torch.contiguous_format
        )
        assert nt_noncont.is_same_size(nt_empty_non_contig)
        assert nt_empty_non_contig.is_contiguous()

        # 测试其他内存格式失败
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_cont, memory_format=torch.channels_last),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_noncont, memory_format=torch.channels_last),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_cont, memory_format=torch.channels_last_3d),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_noncont, memory_format=torch.channels_last_3d),
        )
@markDynamoStrictTest
class TestNestedTensorAutograd(TestCase):
    # Note [Gradcheck args check_batched_grad=False] the common_utils testing version of gradcheck
    # includes the default parameters used for testing ops with gradcheck. However nested tensor
    # does not support the stack op therefore we turn it off for these tests

    # 定义测试类 TestNestedTensorAutograd，继承自 TestCase

    def _create_leaf_nested_tensor_from_list(self, tensor_device, requires_grad=False):
        # 创建叶子节点的嵌套张量，从列表中创建，可以选择是否需要梯度
        return torch.nested.nested_tensor(
            [
                torch.randn(
                    1,
                    2,
                ),
                torch.randn(7, 8),
            ],
            requires_grad=requires_grad,
            device=tensor_device,
        )

    def _create_nested_tensor_from_list(self, tensor_device, requires_grad=False):
        # 从列表中创建嵌套张量，可以选择是否需要梯度
        return torch.nested.as_nested_tensor(
            [
                torch.randn(1, 2, requires_grad=requires_grad),
                torch.randn(7, 8, requires_grad=requires_grad),
            ],
            device=tensor_device,
        )

    def _create_nested_tensor_from_mask(self, tensor_device, requires_grad=False):
        # 根据掩码创建嵌套张量，可以选择是否需要梯度
        data = torch.randn(2, 3, 4, requires_grad=requires_grad, device=tensor_device)
        mask = torch.ones_like(data[:, :, 0]).bool()
        return torch._nested_tensor_from_mask(data, mask)

    def test_as_nested_tensor_propagates_gradients(self, device):
        # 测试嵌套张量是否能正确传播梯度

        a = torch.arange(3, dtype=torch.float, device=device)
        b = torch.arange(5, dtype=torch.float, device=device)
        nt = torch.nested.as_nested_tensor([a, b])

        # tensors with requires_grad=False are leaves
        # 非需要梯度的张量是叶子节点
        self.assertTrue(nt.is_leaf)
        self.assertTrue(not nt.requires_grad)

        a = torch.arange(3, dtype=torch.float, requires_grad=True, device=device)
        b = torch.arange(5, dtype=torch.float, requires_grad=True, device=device)
        nt2 = torch.nested.as_nested_tensor([a, b])

        fake_grad = torch.nested.nested_tensor(
            [torch.ones_like(a), torch.zeros_like(b)], device=device
        )
        nt2.backward(fake_grad)

        self.assertEqual(a.grad, fake_grad[0])
        self.assertEqual(b.grad, fake_grad[1])

    def test_nested_tensor_generates_leaf(self, device):
        # 测试嵌套张量是否生成叶子节点

        a = torch.arange(3, dtype=torch.float, requires_grad=True, device=device)
        b = torch.arange(5, dtype=torch.float, requires_grad=True, device=device)

        nt = torch.nested.nested_tensor([a, b], requires_grad=False)
        self.assertTrue(nt.is_leaf)
        self.assertTrue(not nt.requires_grad)

        nt2 = torch.nested.nested_tensor([a, b], requires_grad=True)
        self.assertTrue(nt2.is_leaf)
        self.assertTrue(nt2.requires_grad)

        fake_grad = torch.nested.nested_tensor(
            [torch.ones_like(a), torch.zeros_like(b)], device=device
        )
        nt2.backward(fake_grad)

        self.assertEqual(nt2.grad, fake_grad)
        self.assertEqual(a.grad, None)
        self.assertEqual(b.grad, None)
    # 测试设置梯度要求，从列表中创建嵌套张量，并确保其需要梯度计算
    def test_set_requires_grad_from_list(self, device):
        # 从列表创建嵌套张量
        nt = self._create_nested_tensor_from_list(device)
        # 设置张量需要梯度计算
        nt.requires_grad_()
        # 断言张量确实需要梯度计算
        assert nt.requires_grad

    # 测试设置梯度要求，从掩码中创建嵌套张量，并确保其需要梯度计算
    def test_set_requires_grad_from_mask(self, device):
        # 从掩码创建嵌套张量
        nt = self._create_nested_tensor_from_mask(device)
        # 设置张量需要梯度计算
        nt.requires_grad_()
        # 断言张量确实需要梯度计算
        assert nt.requires_grad

    # 测试加法操作的反向传播，确保正确计算梯度
    def test_backward_for_add_op(self, device):
        # 从掩码中创建第一个嵌套张量
        nt_1 = self._create_nested_tensor_from_mask(device)
        # 从掩码中创建第二个嵌套张量
        nt_2 = self._create_nested_tensor_from_mask(device)

        # 设置第一个张量需要梯度计算
        nt_1.requires_grad_()
        # 执行张量加法操作
        c = nt_1 + nt_2

        # 断言第一个张量需要梯度计算
        assert nt_1.requires_grad
        # 断言结果张量需要梯度计算
        assert c.requires_grad

        # 创建梯度输出张量
        grad_output = self._create_nested_tensor_from_mask(device)
        # 执行反向传播
        c.backward(grad_output)

        # 断言第一个张量的梯度计算结果
        self.assertEqual(nt_1.grad, grad_output)

    # 测试减法操作的反向传播，确保正确计算梯度
    def test_backward_for_sub_op(self, device):
        # 从掩码中创建第一个嵌套张量
        nt_1 = self._create_nested_tensor_from_mask(device)
        # 从掩码中创建第二个嵌套张量
        nt_2 = self._create_nested_tensor_from_mask(device)

        # 设置第一个张量需要梯度计算
        nt_1.requires_grad_()
        # 设置第二个张量需要梯度计算
        nt_2.requires_grad_()
        # 执行张量减法操作
        c = nt_1 - nt_2

        # 断言第一个张量需要梯度计算
        assert nt_1.requires_grad
        # 断言第二个张量需要梯度计算
        assert nt_2.requires_grad
        # 断言结果张量需要梯度计算
        assert c.requires_grad

        # 创建梯度输出张量
        grad_output = self._create_nested_tensor_from_mask(device)
        # 执行反向传播
        c.backward(grad_output)

        # 断言第一个张量的梯度计算结果
        self.assertEqual(nt_1.grad, grad_output)
        # 断言第二个张量的梯度计算结果
        self.assertEqual(nt_2.grad, -1 * grad_output)

    # 测试具有步长的减法操作的反向传播，确保正确计算梯度
    def test_backward_sub_strided(self, device):
        # 创建包含随机张量的嵌套张量a和b，两个张量都需要梯度计算
        a = torch.nested.nested_tensor(
            [torch.randn(9, 2, 4), torch.randn(12, 2, 4)],
            requires_grad=True,
            device=device,
        )
        b = torch.nested.nested_tensor(
            [torch.randn(9, 4, 2), torch.randn(12, 4, 2)],
            requires_grad=True,
            device=device,
        )
        # 执行张量减法操作，并对结果张量进行克隆作为梯度输出
        c = a - b.transpose(-1, -2)
        grad_output = c.clone()
        # 执行反向传播
        c.backward(grad_output)
        # 断言张量a的梯度计算结果
        self.assertEqual(a.grad, grad_output)
        # 断言张量b的梯度计算结果
        self.assertEqual(b.grad, -1 * grad_output.transpose(-1, -2))

    # 测试具有步长的加法操作的反向传播，确保正确计算梯度
    def test_backward_add_strided(self, device):
        # 创建包含随机张量的嵌套张量a和b，两个张量都需要梯度计算
        a = torch.nested.nested_tensor(
            [torch.randn(9, 2, 4), torch.randn(12, 2, 4)],
            requires_grad=True,
            device=device,
        )
        b = torch.nested.nested_tensor(
            [torch.randn(9, 4, 2), torch.randn(12, 4, 2)],
            requires_grad=True,
            device=device,
        )
        # 执行张量加法操作，并对结果张量进行克隆作为梯度输出
        c = a + b.transpose(-1, -2)
        grad_output = c.clone()
        # 执行反向传播
        c.backward(grad_output)
        # 断言张量a的梯度计算结果
        self.assertEqual(a.grad, grad_output)
        # 断言张量b的梯度计算结果
        self.assertEqual(b.grad, grad_output.transpose(-1, -2))
    # 定义一个测试函数，用于测试嵌套张量到填充张量的转换，并进行梯度检查
    def test_nested_tensor_to_padded_tensor(self, device):
        # 遍历不同的填充值进行测试
        for padding_val in [0, 1]:
            # 创建一个从列表创建的叶子嵌套张量，设备为指定的设备，需要计算梯度
            nt = self._create_leaf_nested_tensor_from_list(
                tensor_device=device, requires_grad=True
            )

            # 将嵌套张量转换为填充张量
            out = torch.nested.to_padded_tensor(nt, padding_val)
            
            # 创建一个与输出张量形状相同的全一张量，设备为指定设备
            grad_output = torch.ones(out.shape, device=device)
            
            # 计算输出张量的梯度
            out.backward(grad_output)

            # 断言叶子嵌套张量的梯度等于指定的嵌套张量结构
            self.assertEqual(
                nt.grad,
                torch.nested.nested_tensor(
                    [torch.ones(1, 2), torch.ones(7, 8)], device=device
                ),
            )

    # 定义一个测试函数，用于从掩码创建嵌套张量并转换为填充张量的测试
    def test_nested_tensor_from_mask_and_to_padded(self, device):
        N, L, D = 2, 4, 4
        
        # 创建一个全一掩码张量，形状为 (N, L)，设备为指定设备
        mask = torch.ones(N, L, device=device)
        
        # 遍历掩码张量的第二维，随机设置非全一的位置为零
        for i in range(1, N):
            end = torch.randint(1, L - 1, (1,), device=device)
            mask[i, end:] = 0

        # 将第一个样本的掩码设为全一
        mask[0, :] = 1
        
        # 将掩码张量转换为布尔类型
        mask = mask.bool()

        # 创建一个随机数据张量，形状为 (N, L, D)，需要计算梯度，数据类型为双精度浮点数，设备为指定设备
        data = torch.randn(
            N, L, D, requires_grad=True, dtype=torch.float64, device=device
        )

        # 定义一个梯度测试函数，输入为 data
        def grad_test_func(inpt):
            # 从掩码张量创建嵌套张量
            nt = torch._nested_tensor_from_mask(inpt, mask)
            # 隐式测试填充张量的梯度
            return torch.nested.to_padded_tensor(nt, 0)

        # 使用梯度检查函数 gradcheck 进行梯度检查，禁用批次梯度检查
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # 定义一个测试函数，用于从填充张量创建嵌套张量并进行填充张量的测试
    def test_nested_tensor_from_padded(self, device):
        # 创建一个形状为 (2, 2) 的嵌套大小张量
        nested_size = torch.tensor([[1, 2], [2, 2]])
        
        # 创建一个形状为 (2, 2, 2) 的随机填充张量，数据类型为双精度浮点数，设备为指定设备
        padded_tensor = torch.randn(2, 2, 2, dtype=torch.float64, device=device)
        
        # 将填充张量第一个样本的第二行设为零
        padded_tensor[0, 1, :] = 0
        
        # 设置填充张量需要计算梯度
        padded_tensor.requires_grad_()

        # 定义一个梯度测试函数，输入为填充张量和嵌套大小张量
        def grad_test_func(tensor, nested_size):
            # 从填充张量创建嵌套张量
            nt = torch._nested_from_padded(
                tensor, nested_size, fuse_transform_0213=False
            )
            # 隐式测试填充张量的梯度
            return torch.nested.to_padded_tensor(nt, 0)

        # 数据为填充张量和嵌套大小张量的元组
        data = (padded_tensor, nested_size)
        
        # 使用梯度检查函数 gradcheck 进行梯度检查，禁用批次梯度检查
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # 定义一个测试函数，用于从融合填充张量创建嵌套张量并进行填充张量的测试
    def test_nested_tensor_from_padded_fused(self, device):
        # 创建一个形状为 (2, 2) 的嵌套大小张量
        nested_size = torch.tensor([[1, 8], [2, 8]])
        
        # 创建一个形状为 (2, 2, 2, 4) 的随机填充张量，数据类型为双精度浮点数，设备为指定设备
        padded_tensor = torch.randn(2, 2, 2, 4, dtype=torch.float64, device=device)
        
        # 将填充张量第一个样本的第二行设为零
        padded_tensor[0, 1, :] = 0
        
        # 设置填充张量需要计算梯度
        padded_tensor.requires_grad_()

        # 定义一个梯度测试函数，输入为填充张量和嵌套大小张量
        def grad_test_func(tensor, nested_size):
            # 从填充张量创建嵌套张量（使用融合变换）
            nt = torch._nested_from_padded(
                tensor, nested_size, fuse_transform_0213=True
            )
            # 隐式测试填充张量的梯度
            return torch.nested.to_padded_tensor(nt, 0)

        # 数据为填充张量和嵌套大小张量的元组
        data = (padded_tensor, nested_size)
        
        # 使用梯度检查函数 gradcheck 进行梯度检查，禁用批次梯度检查
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)
    # 定义一个测试函数，用于测试嵌套张量的梯度
    def test_nested_tensor_from_list(self, device):
        # 创建具有指定形状和属性的张量 `a`
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        # 创建具有指定形状和属性的张量 `b`
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        # 创建具有指定形状和属性的张量 `c`
        c = torch.randn(10, 2, requires_grad=True, dtype=torch.float64, device=device)

        # 定义一个内部函数 `grad_test_func`，将张量 `a`, `b`, `c` 转换为嵌套张量
        def grad_test_func(a, b, c):
            c = torch.nested.as_nested_tensor([a, b, c])
            # 隐式测试 `to_padded_tensor` 函数的梯度
            return torch.nested.to_padded_tensor(c, 0)

        # 将张量 `a`, `b`, `c` 打包为元组 `data`
        data = (a, b, c)
        # 使用 `gradcheck` 函数进行梯度检查，关闭批处理梯度检查选项
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # 通过装饰器进行条件修饰，当不满足特定条件时跳过测试
    @decorateIf(
        xfailIfTorchDynamo,
        # 只有在 Python 3.11 中失败。TODO: 调试此问题！
        lambda params: params["layout"] == torch.jagged and sys.version_info >= (3, 11),
    )
    # 参数化测试函数，测试不同布局下的 dropout 反向传播
    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    def test_dropout_backward(self, layout):
        # 根据不同的布局类型创建嵌套张量 `nt`
        if layout == torch.jagged:
            nt = torch.nested.nested_tensor(
                [torch.randn((2, 5)), torch.randn((3, 5))],
                requires_grad=True,
                layout=layout,
            )
        else:
            nt = torch.nested.nested_tensor(
                [torch.randn((2, 5)), torch.randn((3, 4))],
                requires_grad=True,
                layout=layout,
            )
        # 设置 dropout 概率 `p`
        p = 0.2
        # 应用 dropout 函数到 `nt`，得到 `y`
        y = torch.nn.functional.dropout(nt, p)
        # 对 `y` 进行反向传播
        y.backward(nt.clone().detach())
        # 断言 `nt` 的梯度等于 `y`
        self.assertEqual(nt.grad, y)

    # 测试函数，测试嵌套张量的 `bmm` 方法的梯度
    def test_nested_tensor_bmm_gradcheck(self, device):
        # 创建具有指定形状和属性的张量 `a`
        a = torch.randn(2, 6, requires_grad=True, dtype=torch.float64, device=device)
        # 创建具有指定形状和属性的张量 `b`
        b = torch.randn(3, 6, requires_grad=True, dtype=torch.float64, device=device)
        # 创建具有指定形状和属性的张量 `c`
        c = torch.randn(6, 4, requires_grad=True, dtype=torch.float64, device=device)
        # 创建具有指定形状和属性的张量 `d`
        d = torch.randn(6, 5, requires_grad=True, dtype=torch.float64, device=device)

        # 定义一个内部函数 `grad_test_func`，将张量 `a`, `b` 转换为嵌套张量，并执行 `bmm` 运算
        def grad_test_func(a, b, c, d):
            nt0 = torch.nested.as_nested_tensor([a, b])
            nt1 = torch.nested.as_nested_tensor([c, d])
            result = nt0.bmm(nt1)
            return torch.nested.to_padded_tensor(result, 0.0)

        # 将张量 `a`, `b`, `c`, `d` 打包为元组 `data`
        data = (a, b, c, d)
        # 使用 `gradcheck` 函数进行梯度检查
        assert torch.autograd.gradcheck(grad_test_func, inputs=data)
    # 定义测试函数，用于测试嵌套张量的双向矩阵乘法的反向传播
    def test_nested_tensor_bmm_backward(self, device):
        # 创建两个包含随机张量的嵌套张量 nt0 和 nt1，设置需要梯度计算，并指定设备
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 6)), torch.randn((3, 6))],
            requires_grad=True,
            device=device,
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((6, 4)), torch.randn((6, 5))],
            requires_grad=True,
            device=device,
        )
        # 使用 torch.no_grad() 上下文管理器，创建填充后的张量 pt0 和 pt1，需要梯度计算
        with torch.no_grad():
            pt0 = torch.nested.to_padded_tensor(nt0, 0.0).requires_grad_(True)
            pt1 = torch.nested.to_padded_tensor(nt1, 0.0).requires_grad_(True)

        # 计算嵌套张量 nt0 和 nt1 的批次乘积，并赋值给 ynt
        ynt = nt0.bmm(nt1)
        # 计算填充后的张量 pt0 和 pt1 的批次乘积，并赋值给 ypt
        ypt = pt0.bmm(pt1)
        # 对 ynt 和 ypt 分别执行反向传播，使用 ynt 和 ypt 的克隆张量作为参数
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        # 断言嵌套张量 nt0 和 pt0 的梯度是否相等
        self.assertEqual(torch.nested.to_padded_tensor(nt0.grad, 0.0), pt0.grad)
        # 断言嵌套张量 nt1 和 pt1 的梯度是否相等
        self.assertEqual(torch.nested.to_padded_tensor(nt1.grad, 0.0), pt1.grad)

    # 定义测试函数，用于检查嵌套张量的矩阵乘法是否通过梯度检查
    def test_nested_tensor_matmul_gradcheck(self, device):
        # 创建四个随机张量 a, b, c, d，需要梯度计算，指定数据类型和设备
        a = torch.randn(2, 6, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 6, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(6, 4, requires_grad=True, dtype=torch.float64, device=device)
        d = torch.randn(6, 5, requires_grad=True, dtype=torch.float64, device=device)

        # 定义梯度检查函数 grad_test_func，接受参数 a, b, c, d
        def grad_test_func(a, b, c, d):
            # 将张量 a 和 b 转换为嵌套张量 nt0，将张量 c 和 d 转换为嵌套张量 nt1
            nt0 = torch.nested.as_nested_tensor([a, b])
            nt1 = torch.nested.as_nested_tensor([c, d])
            # 计算 nt0 和 nt1 的矩阵乘积 result，并将结果转换为填充后的张量，填充值为 0.0
            result = torch.matmul(nt0, nt1)
            return torch.nested.to_padded_tensor(result, 0.0)

        # 执行梯度检查，使用 grad_test_func 函数和数据 (a, b, c, d)
        assert torch.autograd.gradcheck(grad_test_func, inputs=(a, b, c, d))

    # 定义测试函数，用于测试嵌套张量的矩阵乘法的反向传播
    def test_nested_tensor_matmul_backward(self, device):
        # 创建包含随机张量的嵌套张量 nt0 和 nt1，设置需要梯度计算，并指定设备
        nt0 = torch.nested.nested_tensor(
            [torch.randn((7, 2, 6)), torch.randn((7, 3, 6))],
            requires_grad=True,
            device=device,
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((7, 6, 4)), torch.randn((7, 6, 5))],
            requires_grad=True,
            device=device,
        )
        # 使用 torch.no_grad() 上下文管理器，创建填充后的张量 pt0 和 pt1，需要梯度计算
        with torch.no_grad():
            pt0 = torch.nested.to_padded_tensor(nt0, 0.0).requires_grad_(True)
            pt1 = torch.nested.to_padded_tensor(nt1, 0.0).requires_grad_(True)

        # 计算嵌套张量 nt0 和 nt1 的矩阵乘积，并赋值给 ynt
        ynt = torch.matmul(nt0, nt1)
        # 计算填充后的张量 pt0 和 pt1 的矩阵乘积，并赋值给 ypt
        ypt = torch.matmul(pt0, pt1)
        # 对 ynt 和 ypt 分别执行反向传播，使用 ynt 和 ypt 的克隆张量作为参数
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        # 断言嵌套张量 nt0 和 pt0 的梯度是否相等
        self.assertEqual(torch.nested.to_padded_tensor(nt0.grad, 0.0), pt0.grad)
        # 断言嵌套张量 nt1 和 pt1 的梯度是否相等
        self.assertEqual(torch.nested.to_padded_tensor(nt1.grad, 0.0), pt1.grad)
    def test_nested_tensor_transpose_gradcheck(self, device):
        # 创建一个形状为 (2, 5) 的随机张量 a，需要计算梯度，放置在指定设备上
        a = torch.randn(2, 5, requires_grad=True, device=device)
        # 创建一个形状为 (3, 4) 的随机张量 b，需要计算梯度，放置在指定设备上
        b = torch.randn(3, 4, requires_grad=True, device=device)

        # 定义一个用于梯度检查的函数 grad_test_func，接受两个张量 a 和 b 作为输入
        def grad_test_func(a, b):
            # 将张量 a 和 b 转换为嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b])
            # 对嵌套张量进行两次转置操作，-2 和 -1 表示倒数第二和最后一个维度
            result = nt.transpose(-2, -1).transpose(-2, -1)
            # 将结果转换为填充张量，并用 0.0 填充空白部分
            return torch.nested.to_padded_tensor(result, 0.0)

        # 将输入数据设为 (a, b)，并进行自动梯度检查，设置误差范围为 1e-3
        data = (a, b)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data, eps=1e-3)

    def test_nested_tensor_transpose_backward(self, device):
        # 创建一个嵌套张量 nt，包含形状为 (2, 5) 和 (3, 4) 的随机张量，需要计算梯度，放置在指定设备上
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 5)), torch.randn((3, 4))],
            requires_grad=True,
            device=device,
        )
        # 在不计算梯度的情况下，将嵌套张量转换为填充张量 pt，并要求计算梯度
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        # 对嵌套张量 nt 和填充张量 pt 进行转置操作，-2 和 -1 表示倒数第二和最后一个维度
        ynt = nt.transpose(-2, -1)
        ypt = pt.transpose(-2, -1)
        # 对 ynt 和 ypt 进行反向传播，使用它们的克隆作为梯度
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        # 断言嵌套张量 nt 的梯度转换为填充张量后与 pt 的梯度相等
        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_reshape_gradcheck(self, device):
        # 创建一个形状为 (2, 6) 的随机张量 a，需要计算梯度，放置在指定设备上
        a = torch.randn(2, 6, requires_grad=True, device=device)
        # 创建一个形状为 (3, 6) 的随机张量 b，需要计算梯度，放置在指定设备上
        b = torch.randn(3, 6, requires_grad=True, device=device)

        # 定义一个用于梯度检查的函数 grad_test_func，接受两个张量 a 和 b 作为输入
        def grad_test_func(a, b):
            # 将张量 a 和 b 转换为嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b])
            # 对嵌套张量进行 reshape 操作，重塑为形状 (2, -1, 2, 3)
            result = nt.reshape(2, -1, 2, 3)
            # 将结果转换为填充张量，并用 0.0 填充空白部分
            return torch.nested.to_padded_tensor(result, 0.0)

        # 将输入数据设为 (a, b)，并进行自动梯度检查，设置误差范围为 1e-3
        data = (a, b)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data, eps=1e-3)

    def test_nested_tensor_reshape_backward(self):
        # 创建一个嵌套张量 nt，包含形状为 (2, 6) 和 (3, 6) 的随机张量，需要计算梯度
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 6)), torch.randn((3, 6))], requires_grad=True
        )
        # 在不计算梯度的情况下，将嵌套张量转换为填充张量 pt，并要求计算梯度
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        # 对嵌套张量 nt 和填充张量 pt 进行 reshape 操作，重塑为形状 (2, -1, 2, 3)
        ynt = nt.reshape(2, -1, 2, 3)
        ypt = pt.reshape(2, -1, 2, 3)
        # 对 ynt 和 ypt 进行反向传播，使用它们的克隆作为梯度
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        # 断言嵌套张量 nt 的梯度转换为填充张量后与 pt 的梯度相等
        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_squeeze_backward(self, device):
        # 创建一个嵌套张量 nt，包含形状为 (2, 6, 1) 和 (3, 6, 1) 的随机张量，需要计算梯度，放置在指定设备上
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 6, 1)), torch.randn((3, 6, 1))],
            requires_grad=True,
            device=device,
        )
        # 在不计算梯度的情况下，将嵌套张量转换为填充张量 pt，并要求计算梯度
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        # 对嵌套张量 nt 和填充张量 pt 进行 squeeze 操作，去除最后一个维度
        ynt = nt.squeeze(-1)
        ypt = pt.squeeze(-1)
        # 对 ynt 和 ypt 进行反向传播，使用它们的克隆作为梯度
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        # 断言嵌套张量 nt 的梯度转换为填充张量后与 pt 的梯度相等
        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)
    def test_nested_tensor_squeeze_gradcheck(self, device):
        # 创建随机张量a和b，形状为(2, 6, 1)，数据类型为torch.float64，允许计算梯度，放置在指定设备上
        a = torch.randn((2, 6, 1), dtype=torch.float64, requires_grad=True, device=device)
        b = torch.randn((3, 6, 1), dtype=torch.float64, requires_grad=True, device=device)

        def grad_test_func(a, b):
            # 将a和b作为嵌套张量输入，创建嵌套张量nt
            nt = torch.nested.as_nested_tensor([a, b])
            # 对nt执行squeeze操作，去除维度为-1的单维度
            result = nt.squeeze(-1)
            # 将squeeze后的结果转换为填充张量，填充值为0.0
            return torch.nested.to_padded_tensor(result, 0.0)

        # 使用gradcheck函数检查grad_test_func函数的梯度计算是否正确，设置数值微分的eps为1e-3
        assert torch.autograd.gradcheck(grad_test_func, inputs=(a, b), eps=1e-3)

    def test_nested_tensor_unsqueeze_backward(self, device):
        # 创建随机张量nt作为嵌套张量输入，形状分别为(2, 6)和(3, 6)，允许计算梯度，放置在指定设备上
        nt = torch.nested.nested_tensor(
            [torch.randn((2, 6)), torch.randn((3, 6))],
            requires_grad=True,
            device=device,
        )
        # 使用torch.no_grad()上下文管理器，创建填充张量pt，其值为0.0，允许计算梯度，并放置在指定设备上
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        # 对nt进行unsqueeze操作，在第2维度上添加一个维度
        ynt = nt.unsqueeze(2)
        # 对pt进行unsqueeze操作，在第2维度上添加一个维度
        ypt = pt.unsqueeze(2)
        # 对ynt执行反向传播，使用ynt的克隆进行梯度计算
        ynt.backward(ynt.clone())
        # 对ypt执行反向传播，使用ypt的克隆进行梯度计算
        ypt.backward(ypt.clone())

        # 断言nt的梯度与pt的梯度转换为填充张量后是否相等，填充值为0.0
        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_unsqueeze_gradcheck(self, device):
        # 创建随机张量a和b，形状为(2, 6)，数据类型为torch.float64，允许计算梯度，放置在指定设备上
        a = torch.randn((2, 6), dtype=torch.float64, requires_grad=True, device=device)
        b = torch.randn((3, 6), dtype=torch.float64, requires_grad=True, device=device)

        def grad_test_func(a, b):
            # 将a和b作为嵌套张量输入，创建嵌套张量nt
            nt = torch.nested.as_nested_tensor([a, b])
            # 对nt执行unsqueeze操作，在最后一个维度上添加一个维度
            result = nt.unsqueeze(-1)
            # 将unsqueeze后的结果转换为填充张量，填充值为0.0
            return torch.nested.to_padded_tensor(result, 0.0)

        # 使用gradcheck函数检查grad_test_func函数的梯度计算是否正确，设置数值微分的eps为1e-3
        assert torch.autograd.gradcheck(grad_test_func, inputs=(a, b), eps=1e-3)

    def test_nested_tensor_linear(self, device):
        # 创建随机张量a，形状为(1, 2)，数据类型为torch.float64，允许计算梯度，放置在指定设备上
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        # 创建随机张量b，形状为(2, 2)，数据类型为torch.float64，允许计算梯度，放置在指定设备上
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        # 创建随机张量c，形状为(3, 2)，数据类型为torch.float64，允许计算梯度，放置在指定设备上
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        # 创建随机张量weight，形状为(2, 2)，数据类型为torch.float64，允许计算梯度，放置在指定设备上
        weight = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        # 创建随机张量bias，形状为(2)，数据类型为torch.float64，允许计算梯度，放置在指定设备上
        bias = torch.randn(2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, weight, bias=None):
            # 将a、b和c作为嵌套张量输入，创建嵌套张量nt
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 对nt执行线性变换操作，使用weight和bias参数
            d = torch.functional.F.linear(nt, weight, bias)
            # 将线性变换后的结果转换为填充张量，填充值为0
            return torch.nested.to_padded_tensor(d, 0)

        # 准备数据，包括a、b、c、weight和bias
        data = (a, b, c, weight, bias)
        # 使用gradcheck函数检查grad_test_func函数的梯度计算是否正确，不检查批量梯度，eps设置为1e-3
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

        # 准备数据，包括a、b、c和weight（不包括bias）
        data = (a, b, c, weight)
        # 使用gradcheck函数检查grad_test_func函数的梯度计算是否正确，不检查批量梯度，eps设置为1e-3
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)
    def test_nested_tensor_linear_plus_transpose(self, device):
        # 创建随机张量 a, b, c，形状分别为 (1, 2), (2, 2), (3, 2)，需要计算梯度，数据类型为 float64，在指定的设备上
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        # 创建随机权重张量和偏置张量，形状分别为 (2, 2), (2)，需要计算梯度，数据类型为 float64，在指定的设备上
        weight = torch.randn(
            2, 2, requires_grad=True, dtype=torch.float64, device=device
        )
        bias = torch.randn(2, requires_grad=True, dtype=torch.float64, device=device)

        # 定义梯度测试函数，输入参数为 a, b, c, weight, bias（可选），将它们作为嵌套张量处理
        def grad_test_func(a, b, c, weight, bias=None):
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 调用线性函数 F.linear 对嵌套张量进行线性变换，使用给定的权重和偏置
            d = torch.functional.F.linear(nt, weight, bias)
            # 对结果张量进行转置操作，交换最后两个维度，并保证内存连续性
            d = d.transpose(-1, -2).contiguous()
            # 将结果张量转换回填充张量形式，填充值为 0
            return torch.nested.to_padded_tensor(d, 0)

        # 定义输入数据为 (a, b, c, weight, bias)，并断言梯度检查结果为 True，不检查批次梯度
        data = (a, b, c, weight, bias)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

        # 测试不加偏置的线性变换
        data = (a, b, c, weight)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_softmax(self, device):
        # 创建随机张量 a, b, c，形状分别为 (1, 2), (2, 2), (3, 2)，需要计算梯度，数据类型为 float64，在指定的设备上
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        # 定义梯度测试函数，输入参数为 a, b, c 和维度 dim，在 dim 维度上应用 softmax 函数
        def grad_test_func(a, b, c, dim):
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 调用 softmax 函数对嵌套张量 nt 在 dim 维度上进行 softmax 计算
            d = torch.functional.F.softmax(nt, dim=dim)
            # 将结果张量转换回填充张量形式，填充值为 0
            return torch.nested.to_padded_tensor(d, 0)

        # 在最后一个维度上应用 softmax 函数，即 dim=-1
        data = (a, b, c, -1)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_linear_backward(self, device):
        # 创建随机张量 a, b, c，形状分别为 (1, 2), (2, 2), (3, 2)，不需要计算梯度，位于指定的设备上
        a = torch.randn(1, 2, requires_grad=False, device=device)
        b = torch.randn(2, 2, requires_grad=False, device=device)
        c = torch.randn(3, 2, requires_grad=False, device=device)

        # 创建随机权重张量和偏置张量，形状分别为 (2, 2)，需要计算梯度，位于指定的设备上
        weight = torch.randn(2, 2, requires_grad=True, device=device)
        bias = torch.randn(2, requires_grad=True, device=device)

        # 将 a, b, c 组成嵌套张量 nt，位于指定的设备上
        nt = torch.nested.as_nested_tensor([a, b, c], device=device)

        # 对嵌套张量 nt 应用线性变换，并使用给定的权重和偏置
        out = torch.functional.F.linear(nt, weight, bias)

        # 对结果张量 out 执行反向传播，使用 out.clone() 作为梯度
        out.backward(out.clone())

        # 断言权重和偏置张量的梯度不为 None
        assert weight.grad is not None
        assert bias.grad is not None

        # 断言 a, b, c 的梯度为 None
        assert a.grad is None
        assert b.grad is None
        assert c.grad is None
    def test_values_grad_with_broadcast(self, device):
        # 创建随机张量 a, b, c，形状分别为 (1, 2, 4)，(2, 2, 4)，(3, 2, 4)，指定需要计算梯度，数据类型为 float64，设备为 device
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            # 将张量 a, b, c 组成嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 提取嵌套张量 nt 的值到 buffer
            buffer = nt.values()
            # 返回 buffer 的所有元素的和
            return buffer.sum()

        data = (a, b, c)
        # 使用 gradcheck 验证 grad_test_func 的梯度
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_to_buffer_series_ops_grad_with_broadcast(self, device):
        # 创建随机张量 a, b, c，形状分别为 (1, 1, 2)，指定需要计算梯度，数据类型为 float64，设备为 device
        a = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            # 将张量 a, b, c 组成嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 提取嵌套张量 nt 的值到 buffer
            buffer = nt.values()
            # 将 buffer 中的所有元素乘以 2
            buffer = buffer * 2
            # 返回 buffer 中的所有元素的指数值
            return buffer.exp()

        data = (a, b, c)
        # 使用 gradcheck 验证 grad_test_func 的梯度
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_unbind_flow_through(self, device):
        # 创建随机张量 a, b, c，形状分别为 (1, 2, 4)，(2, 2, 4)，(3, 2, 4)，指定需要计算梯度，数据类型为 float64，设备为 device
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            # 将张量 a, b, c 组成嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 将嵌套张量 nt 在最后两个维度上进行转置
            ntT = nt.transpose(-1, -2)
            # 在最后一个维度上解绑嵌套张量 ntT
            unbound = ntT.unbind()
            # 提取解绑后的第一个张量 d
            d = unbound[0]
            # 将张量 d 中的每个元素求平方
            d = torch.pow(d, 2)
            # 返回结果张量 d
            return d

        data = (a, b, c)
        # 使用 gradcheck 验证 grad_test_func 的梯度
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_split_with_sizes_flow_through(self, device):
        # 创建随机张量 a, b, c，形状分别为 (2, 5)，(3, 5)，(4, 5)，指定需要计算梯度，数据类型为 float64，设备为 device
        a = torch.randn(2, 5, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 5, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 5, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            # 将张量 a, b, c 组成嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 在最后一个维度上按指定大小分割嵌套张量 nt
            splits = nt.split_with_sizes([2, 3], dim=-1)
            # 在分割后的张量列表中解绑第二个张量，并提取解绑后的第一个张量 d
            unbound = splits[1].unbind()
            d = unbound[0]
            # 将张量 d 中的每个元素求平方
            d = torch.pow(d, 2)
            # 返回结果张量 d
            return d

        data = (a, b, c)
        # 使用 gradcheck 验证 grad_test_func 的梯度
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)
    # 测试反向索引功能，使用给定设备生成随机张量 x0 和 x1
    def test_indexing_backward(self, device):
        x0 = torch.randn((2, 5))  # 生成大小为 (2, 5) 的随机张量 x0
        x1 = torch.randn((3, 4))  # 生成大小为 (3, 4) 的随机张量 x1
        # 创建嵌套张量 nt，要求计算梯度，并使用给定设备
        nt = torch.nested.nested_tensor([x0, x1], device=device, requires_grad=True)
        self.assertEqual(nt[0], x0)  # 断言索引 0 返回张量 x0
        self.assertEqual(nt[-1], x1)  # 断言索引 -1 返回张量 x1
        grad_x0 = torch.randn((2, 5), device=device)  # 生成大小为 (2, 5) 的随机梯度 grad_x0
        nt[0].backward(grad_x0)  # 对索引为 0 的张量执行反向传播
        # 期望的梯度张量，包含 grad_x0 和大小为 (3, 4) 的零张量
        expected_grad = torch.nested.nested_tensor(
            [grad_x0, torch.zeros((3, 4), device=device)]
        )
        self.assertEqual(nt.grad, expected_grad)  # 断言计算的梯度与期望的梯度相等

    # 测试带掩码填充的反向传播功能，使用给定设备生成随机张量 a, b, c
    def test_masked_fill_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 a
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 b
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 c

        # 定义带掩码填充的梯度测试函数
        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])  # 将张量 a, b, c 转换为嵌套张量 nt
            mask = nt.detach().clone().to(bool)  # 创建掩码，用于标记 nt 的非零值
            out = nt.masked_fill(mask, 0)  # 使用掩码将 nt 中的非零值填充为 0
            out = torch.nested.to_padded_tensor(out, 0)  # 将填充后的 nt 转换为填充张量
            return out

        data = (a, b, c)  # 将随机张量 a, b, c 组成数据元组
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)  # 断言执行梯度检查

    # 测试 GELU 激活函数的反向传播功能，使用给定设备生成随机张量 a, b, c
    def test_gelu_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 a
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 b
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 c

        # 定义 GELU 函数的梯度测试函数
        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])  # 将张量 a, b, c 转换为嵌套张量 nt
            nt_gelu = torch.nn.functional.gelu(nt)  # 应用 GELU 激活函数到 nt
            return torch.nested.to_padded_tensor(nt_gelu, 0)  # 将 GELU 结果转换为填充张量

        data = (a, b, c)  # 将随机张量 a, b, c 组成数据元组
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)  # 断言执行梯度检查

    # 测试 ReLU 激活函数的反向传播功能，使用给定设备生成随机张量 a, b, c
    def test_relu_backward(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 a
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 b
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)  # 生成具有梯度的随机张量 c

        # 定义 ReLU 函数的梯度测试函数
        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])  # 将张量 a, b, c 转换为嵌套张量 nt
            nt_relu = torch.nn.functional.relu(nt)  # 应用 ReLU 激活函数到 nt
            return torch.nested.to_padded_tensor(nt_relu, 0)  # 将 ReLU 结果转换为填充张量

        data = (a, b, c)  # 将随机张量 a, b, c 组成数据元组
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)  # 断言执行梯度检查
    # 定义一个测试函数，用于测试 torch.selu 的反向传播
    def test_selu_backward(self, device):
        # 创建三个张量 a, b, c，形状分别为 (1, 2, 4)，(2, 2, 4)，(3, 2, 4)，要求计算梯度，数据类型为 torch.float64，设备为指定设备
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        # 定义一个内部函数 grad_test_func，接收参数 a, b, c
        def grad_test_func(a, b, c):
            # 将输入张量 a, b, c 转换为嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 对 nt 应用 torch.silu 函数，得到 nt_relu
            nt_relu = torch.nn.functional.silu(nt)
            # 将 nt_relu 转换为填充张量，填充值为 0
            return torch.nested.to_padded_tensor(nt_relu, 0)

        # 准备数据元组 (a, b, c)
        data = (a, b, c)
        # 断言 grad_test_func 的梯度检查结果，关闭批量梯度检查
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # 定义一个测试函数，用于测试 torch.abs 的反向传播
    def test_abs_backward(self, device):
        # 创建三个张量 a, b, c，形状分别为 (1, 2, 4)，(2, 2, 4)，(3, 2, 4)，要求计算梯度，数据类型为 torch.float64，设备为指定设备
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        # 定义一个内部函数 grad_test_func，接收参数 a, b, c
        def grad_test_func(a, b, c):
            # 将输入张量 a, b, c 转换为嵌套张量 nt
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 对 nt 应用 torch.abs 函数，得到 nt_abs
            nt_abs = torch.abs(nt)
            # 将 nt_abs 转换为填充张量，填充值为 0
            return torch.nested.to_padded_tensor(nt_abs, 0)

        # 准备数据元组 (a, b, c)
        data = (a, b, c)
        # 断言 grad_test_func 的梯度检查结果，关闭批量梯度检查
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # 测试层归一化的反向传播，处理特定情况下的边缘案例
    def test_layer_norm_backward_edge_case(self, device):
        # 定义大小为 4 的随机张量 a，不要求计算梯度，数据类型为 torch.float64，设备为指定设备
        size = 4
        a = torch.randn(
            1, 2, size, requires_grad=False, dtype=torch.float64, device=device
        )
        # 将张量 a 转换为嵌套张量 nt
        nt = torch.nested.nested_tensor([a])
        # 创建层归一化实例 nt_layer_norm，指定设备和数据类型为 torch.float64
        nt_layer_norm = torch.nn.LayerNorm(
            nt.size(-1), device=device, dtype=torch.float64
        )
        # 对 nt_layer_norm 应用层归一化，得到 out
        out = nt_layer_norm(nt)
        # 对 out 执行反向传播，传播 out 的克隆张量的梯度
        out.backward(out.clone())

    # 测试不同步长的累积梯度
    def test_accumulate_grad_different_strides(self, device):
        # 创建两个形状为 (1, 4, 2) 和 (1, 8, 2) 的随机张量 a, b，要求计算梯度，数据类型为 torch.float64，设备为指定设备
        a = torch.rand(1, 4, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.rand(1, 8, 2, requires_grad=True, dtype=torch.float64, device=device)

        # 定义一个内部函数 grad_test_func，接收参数 a, b
        def grad_test_func(a, b):
            # 将输入张量 a, b 转换为嵌套张量 nt_1
            nt_1 = torch.nested.as_nested_tensor([a, b])
            # 克隆 nt_1 得到 nt_2
            nt_2 = nt_1.clone()
            # 对 nt_1, nt_2 应用 torch.scaled_dot_product_attention 函数，得到 out
            out = torch.nn.functional.scaled_dot_product_attention(nt_1, nt_2, nt_2)
            # 将 out 转换为填充张量，填充值为 0
            return torch.nested.to_padded_tensor(out, 0)

        # 准备数据元组 (a, b)
        data = (a, b)
        # 断言 grad_test_func 的梯度检查结果，关闭批量梯度检查
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # 注解：链接到 GitHub 上的一个 issue，描述了一个问题和相关的环境
    @skipIfSlowGradcheckEnv
    @parametrize("size", [1024, 1023, 513, 512, 256, 128, 32, 4, 2])
    # 定义一个测试方法，用于测试LayerNorm的反向传播功能，接受设备和尺寸参数
    def test_layer_norm_backward(self, device, size):
        # 生成随机张量a，形状为[1, 2, size]，需要计算梯度，数据类型为torch.float64，放置在指定设备上
        a = torch.randn(
            1, 2, size, requires_grad=True, dtype=torch.float64, device=device
        )
        # 生成随机张量b，形状为[2, 2, size]，需要计算梯度，数据类型为torch.float64，放置在指定设备上
        b = torch.randn(
            2, 2, size, requires_grad=True, dtype=torch.float64, device=device
        )
        # 生成随机张量c，形状为[3, 2, size]，需要计算梯度，数据类型为torch.float64，放置在指定设备上
        c = torch.randn(
            3, 2, size, requires_grad=True, dtype=torch.float64, device=device
        )

        # 定义梯度测试函数grad_test_func，接受a、b、c作为输入参数
        def grad_test_func(a, b, c):
            # 将a、b、c作为嵌套张量组成一个NestedTensor
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 创建一个LayerNorm模块，对最后一个维度进行归一化，指定设备和数据类型
            layer_norm = torch.nn.LayerNorm(
                nt.size(-1), device=device, dtype=torch.float64
            )
            # 对NestedTensor应用LayerNorm
            nt_layer_norm = layer_norm(nt)
            # 将LayerNorm后的NestedTensor转换为填充后的张量，用0填充
            return torch.nested.to_padded_tensor(nt_layer_norm, 0)

        # 将a、b、c作为数据元组传递给梯度检查函数gradcheck，禁用批次梯度检查
        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    # 标记为在慢速梯度检查环境下跳过测试
    @skipIfSlowGradcheckEnv
    # 参数化测试方法，使用不同的size参数进行多次测试
    @parametrize("size", [128, 32, 4, 2])
    # 定义测试方法，用于测试5维输入的LayerNorm反向传播功能，接受设备和尺寸参数
    def test_layer_norm_backward_5d(self, device, size):
        # 生成随机张量a，形状为[4, size, size, 4]，需要计算梯度，数据类型为torch.float64，放置在指定设备上
        a = torch.randn(
            4, size, size, 4, requires_grad=True, dtype=torch.float64, device=device
        )
        # 生成随机张量b，形状为[7, size, size, 4]，需要计算梯度，数据类型为torch.float64，放置在指定设备上
        b = torch.randn(
            7, size, size, 4, requires_grad=True, dtype=torch.float64, device=device
        )
        # 生成随机张量c，形状为[10, size, size, 4]，需要计算梯度，数据类型为torch.float64，放置在指定设备上
        c = torch.randn(
            10, size, size, 4, requires_grad=True, dtype=torch.float64, device=device
        )

        # 定义梯度测试函数grad_test_func，接受a、b、c作为输入参数
        def grad_test_func(a, b, c):
            # 将a、b、c作为嵌套张量组成一个NestedTensor
            nt = torch.nested.as_nested_tensor([a, b, c])
            # 创建一个LayerNorm模块，对最后三个维度进行归一化，指定设备和数据类型
            layer_norm = torch.nn.LayerNorm(
                (size, size, nt.size(-1)), device=device, dtype=torch.float64
            )
            # 对NestedTensor应用LayerNorm
            nt_layer_norm = layer_norm(nt)
            # 将LayerNorm后的NestedTensor转换为填充后的张量，用0填充
            return torch.nested.to_padded_tensor(nt_layer_norm, 0)

        # 将a、b、c作为数据元组传递给梯度检查函数gradcheck，禁用批次梯度检查
        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)
# 默认的绝对误差容差字典，针对不同数据类型的张量
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
# 默认的相对误差容差字典，针对不同数据类型的张量
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    # 计算真实值与计算值之间的偏差
    deviation = true_value - computed_value
    # 计算相对误差，并取绝对值
    deviation = torch.abs(deviation / true_value)
    # 将NaN值替换为对应数据类型的默认相对误差容差
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    # 返回最大的相对误差
    return deviation.max().item()


def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    # 计算真实值与计算值之间的偏差，并取绝对值后的最大值作为绝对误差容差
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    # 如果没有指定调整因子，则默认为1.0
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    # 获取绝对误差容差
    atol = get_atol(true_value, computed_value)
    # 获取相对误差容差
    rtol = get_rtol(true_value, computed_value)

    # 根据调整因子调整绝对误差容差和相对误差容差的值
    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    
    # 如果相对误差容差超过1e30，使用默认的相对误差容差值
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    
    return atol, rtol


# 我们可以考虑通过参数化现有的测试来支持更多的操作，而不是使用单独的测试类。
# 同时，也许可以通过OpInfos重写测试。
@markDynamoStrictTest
class TestNestedTensorSubclass(TestCase):
    # TODO: consolidate with the below
    # 获取用于嵌套张量的列表，支持不同的尺寸、设备，并可选是否需要梯度
    def _get_list_for_jagged_tensor(self, nested_size, device, requires_grad=True):
        Ds = nested_size[1:]
        out = []
        for s in nested_size[0]:
            out.append(
                torch.randn(
                    s,
                    *Ds,
                    requires_grad=requires_grad,
                    device=device,
                    dtype=torch.float64,
                )
            )
        return out

    # 获取示例张量列表，可以包含嵌套列表，并可选是否包含梯度信息
    def _get_example_tensor_lists(
        self, include_list_of_lists=True, include_requires_grad=True
    ):
        # 定义一个内部函数 _make_tensor，用于创建具有指定形状的张量
        def _make_tensor(
            *shape, include_requires_grad=include_requires_grad, requires_grad=True
        ):
            return torch.randn(
                *shape,
                requires_grad=(requires_grad if include_requires_grad else False),
            )

        # 故意引入混合的 requires_grad 设置以测试组件
        # 当 include_requires_grad=True 时，张量的 requires_grad 有不同的设定。
        example_lists = [
            # (B, *, D) 其中 B=4
            [
                _make_tensor(2, 5),
                _make_tensor(3, 5, requires_grad=False),
                _make_tensor(4, 5, requires_grad=False),
                _make_tensor(6, 5),
            ],
            # (B, *, D_0, D_1) 其中 B=5
            [
                _make_tensor(2, 5, 6),
                _make_tensor(3, 5, 6),
                _make_tensor(4, 5, 6, requires_grad=False),
                _make_tensor(5, 5, 6),
                _make_tensor(6, 5, 6),
            ],
            # (B, *, D_0, D_1, D_2) 其中 B=6
            [
                _make_tensor(2, 5, 6, 7),
                _make_tensor(3, 5, 6, 7),
                _make_tensor(4, 5, 6, 7, requires_grad=False),
                _make_tensor(5, 5, 6, 7),
                _make_tensor(6, 5, 6, 7),
                _make_tensor(7, 5, 6, 7),
            ],
        ]

        # 如果 include_list_of_lists 为 True，则添加一个列表的示例
        if include_list_of_lists:
            example_lists.append(
                # (B, *, D) 其中 B=3，以列表形式表示
                [
                    _make_tensor(2, 5, requires_grad=False).tolist(),
                    _make_tensor(3, 5).tolist(),
                    _make_tensor(4, 5).tolist(),
                ]
            )

        # 返回生成的 example_lists
        return example_lists

    # 测试张量的属性
    def test_tensor_attributes(self, device):
        # 创建具有指定形状、需要梯度的张量 a, b, c
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)
        
        # 使用 torch.nested.as_nested_tensor 创建嵌套张量 nt
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        
        # 获取嵌套张量的偏移量 _offsets
        _offsets = nt.offsets()

        # 遍历一系列操作符 op 并应用于 nt
        for op in (
            torch.ops.aten.is_non_overlapping_and_dense.default,
            torch.ops.aten.sym_size.default,
            torch.ops.aten.dim.default,
            torch.ops.aten.numel.default,
            torch.ops.aten.sym_numel.default,
            torch.ops.aten.sym_stride.default,
            torch.ops.aten.sym_storage_offset.default,
        ):
            op(nt)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，消息为 "directly calling torch.ops.aten.size"
        with self.assertRaisesRegex(
            RuntimeError, "directly calling torch.ops.aten.size"
        ):
            torch.ops.aten.size.default(nt)

        # 使用 torch.nested._internal.nested_tensor.get_tensor_symint 获取 nested_int
        nested_int = torch.nested._internal.nested_tensor.get_tensor_symint(
            _offsets, coeff=1
        )

        # 断言验证 nt 的 size、shape、dim 和 numel 属性
        self.assertEqual(nt.size(), (3, nested_int, 3))
        self.assertEqual(nt.shape, (3, nested_int, 3))
        self.assertEqual(nt.dim(), 3)
        self.assertEqual(nt.numel(), 27)

    # 使用 parametrize 装饰器定义参数化测试
    @parametrize("nt_dim", [3, 4, 5])
    def test_linear(self, device, nt_dim):
        # 根据输入的维度选择固定的张量形状
        if nt_dim == 3:
            fixed_shape = (3,)
        elif nt_dim == 4:
            fixed_shape = (4, 3)
        elif nt_dim == 5:
            fixed_shape = (5, 4, 3)

        # 生成随机张量a，形状为(2, *fixed_shape)，需要梯度计算，数据类型为torch.float64，存储设备为device
        a = torch.randn(
            2, *fixed_shape, requires_grad=True, dtype=torch.float64, device=device
        )
        # 生成随机张量b，形状为(3, *fixed_shape)，需要梯度计算，数据类型为torch.float64，存储设备为device
        b = torch.randn(
            3, *fixed_shape, requires_grad=True, dtype=torch.float64, device=device
        )
        # 生成随机张量c，形状为(4, *fixed_shape)，需要梯度计算，数据类型为torch.float64，存储设备为device
        c = torch.randn(
            4, *fixed_shape, requires_grad=True, dtype=torch.float64, device=device
        )
        # 生成随机权重张量weight，形状为(4, 3)，需要梯度计算，数据类型为torch.float64，存储设备为device
        weight = torch.randn(
            4, 3, requires_grad=True, dtype=torch.float64, device=device
        )

        # 定义梯度测试函数grad_test_func，输入为a, b, c, weight，将它们组成嵌套张量nt
        def grad_test_func(a, b, c, weight):
            nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            # 对nt进行线性操作，使用权重张量weight
            out = torch.nn.functional.linear(nt, weight)
            # 返回线性操作的结果
            return out.values()

        # 进行梯度检查，验证grad_test_func的梯度计算是否正确
        gradcheck(grad_test_func, inputs=(a, b, c, weight), check_batched_grad=False)

    def test_unary_pointwise(self, device):
        # 生成随机张量a，形状为(2, 3)，需要梯度计算，数据类型为torch.float64，存储设备为device
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        # 生成随机张量b，形状为(3, 3)，需要梯度计算，数据类型为torch.float64，存储设备为device
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        # 生成随机张量c，形状为(4, 3)，需要梯度计算，数据类型为torch.float64，存储设备为device
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        # 定义梯度测试函数grad_test_func，输入为a, b, c，将它们组成嵌套张量nt
        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            # 对nt进行逐元素操作，先sin再cos，然后应用sigmoid linear unit（SiLU）函数
            out = torch.nn.functional.silu(nt.sin().cos())
            # 返回SiLU函数的结果
            return out.values()

        # 进行梯度检查，验证grad_test_func的梯度计算是否正确
        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    def test_unary_pointwise_transposed_inputs(self, device):
        # 生成随机张量a, b, c，分别形状为(i+2, 5)，需要梯度计算，数据类型为torch.float64，存储设备为device
        a, b, c = (
            torch.randn(
                i + 2, 5, requires_grad=True, dtype=torch.float64, device=device
            )
            for i in range(3)
        )

        # 分别将a, b, c转换为嵌套张量nt，使用torch.jagged布局
        nt = torch.nested.nested_tensor(
            [a.detach(), b.detach(), c.detach()], layout=torch.jagged
        )
        # 对nt进行维度转置，将第1和第2个维度交换
        nt_t = nt.transpose(1, 2)
        # 断言nt_t不是连续的张量
        self.assertFalse(nt_t.is_contiguous())
        # 对nt_t进行逐元素操作，先sin再cos，然后应用SiLU函数
        out = torch.nn.functional.silu(nt_t.sin().cos())
        # 断言out是连续的张量，验证操作正确性
        self.assertEqual(
            out.is_contiguous(),
            torch.nn.functional.silu(b.transpose(-1, -2).sin().cos()).is_contiguous(),
        )

        # 断言nt_t和out的形状相同
        self.assertEqual(nt_t.shape, out.shape)

        # 重新生成随机张量a, b, c，分别形状为(i+2, 5)，需要梯度计算，数据类型为torch.float64，存储设备为device
        a, b, c = (
            torch.randn(
                i + 2, 5, requires_grad=True, dtype=torch.float64, device=device
            )
            for i in range(3)
        )

        # 定义梯度测试函数grad_test_func，输入为a, b, c，将它们组成嵌套张量nt
        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            # 对nt进行维度转置，将第1和第2个维度交换
            nt_t = nt.transpose(1, 2)
            # 对nt_t进行逐元素操作，先sin再cos，然后应用SiLU函数
            out = torch.nn.functional.silu(nt_t.sin().cos())
            # 返回SiLU函数的结果
            return out.values()

        # 进行梯度检查，验证grad_test_func的梯度计算是否正确
        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)
    # 定义一个测试方法，用于测试二进制逐点操作
    def test_binary_pointwise(self, device):
        # 创建随机张量 a, b, c，要求梯度计算，使用双精度浮点数，指定设备
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        # 错误用法示例：如果偏移张量不是完全相同的张量对象，则形状检查将失败
        nt1 = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        nt2 = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)

        # 断言捕获异常，确保在形状不匹配时抛出 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            "cannot call binary pointwise function .* with inputs of shapes",
            lambda: nt1 * nt2,
        )

        # 正确用法示例：使用相同的偏移张量对象进行调用链
        def grad_test_func(a, b, c):
            nt1 = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            # TODO: Switch to public API that takes in (values, offsets) once it exists
            nt2, offsets = jagged_from_list([a, b, c], nt1.offsets())
            out = nt1 * nt2  # 执行逐点乘法操作
            return out.values()  # 返回输出张量的值

        # 使用 gradcheck 函数对梯度进行检查，禁用批处理梯度检查
        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)

    # 定义另一个测试方法，用于测试二进制逐点操作的转置情况
    def test_binary_pointwise_transposed(self, device):
        # 创建具有不同大小的随机张量 a, b, c，指定设备为双精度浮点数
        a, b, c = (
            torch.randn(i + 2, 5, dtype=torch.float64, device=device) for i in range(3)
        )

        # 创建具有不同偏移的 nt1 和 nt2，来自于 jagged_from_list 函数
        nt1, offsets = jagged_from_list([a, b, c], None)
        nt2, offsets = jagged_from_list([a, b, c], offsets)

        # 对 nt1 和 nt2 进行转置操作
        nt1_t = nt1.transpose(1, 2)
        nt2_t = nt2.transpose(1, 2)

        # 断言捕获异常，确保在形状不匹配时抛出 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError,
            "cannot call binary pointwise function mul.Tensor with inputs of shapes",
            lambda: nt1 * nt2_t,
        )

        # 重新创建具有梯度要求的随机张量 a, b, c
        a, b, c = (
            torch.randn(
                i + 2, 5, requires_grad=True, dtype=torch.float64, device=device
            )
            for i in range(3)
        )

        # 正确用法示例：使用相同的偏移张量对象进行调用链
        def grad_test_func(a, b, c):
            nt1, offsets = jagged_from_list([a, b, c], None)
            nt2, offsets = jagged_from_list([a, b, c], offsets)
            nt1_t = nt1.transpose(1, 2)
            nt2_t = nt2.transpose(1, 2)
            out = nt1_t * nt2_t  # 执行逐点乘法操作
            return out.values()  # 返回输出张量的值

        # 使用 gradcheck 函数对梯度进行检查，禁用批处理梯度检查
        gradcheck(grad_test_func, inputs=(a, b, c), check_batched_grad=False)
    # 定义一个测试函数，用于测试 torch.split 方法在 NestedTensor 上的操作
    def test_split(self, device):
        # 创建三个随机张量 a, b, c，形状为 (2, 3), (3, 3), (4, 3)，要求计算梯度，使用双精度浮点数，指定设备
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        # 将张量 a, b, c 转换为 NestedTensor，使用 torch.jagged 布局
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        
        # 在最后一个维度 (-1) 上对 NestedTensor 进行分割，每个分割的长度为 2
        out = torch.split(nt, 2, -1)
        # 断言分割后的结果长度为 2
        self.assertEqual(len(out), 2)
        # 断言分割后的第一个部分与预期结果相等
        self.assertEqual(
            out[0],
            torch.nested.as_nested_tensor(
                [a[:, 0:2], b[:, 0:2], c[:, 0:2]], layout=torch.jagged
            ),
        )
        # 断言分割后的第二个部分与预期结果相等
        self.assertEqual(
            out[1],
            torch.nested.as_nested_tensor(
                [a[:, 2:], b[:, 2:], c[:, 2:]], layout=torch.jagged
            ),
        )

        # 使用断言检查在维度 1 上使用 split 方法会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"split\(\): not supported for NestedTensor on dim=1",
        ):
            torch.split(nt, 2, 1)

    # 定义一个测试函数，测试带有指定分割大小的 torch.split 方法在 NestedTensor 上的操作
    def test_split_with_sizes(self, device):
        # 创建三个随机张量 a, b, c，形状为 (2, 3), (3, 3), (4, 3)，要求计算梯度，使用双精度浮点数，指定设备
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)

        # 将张量 a, b, c 转换为 NestedTensor，使用 torch.jagged 布局
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        
        # 在最后一个维度 (-1) 上对 NestedTensor 进行分割，分割的大小分别为 [1, 2]
        out = torch.split(nt, [1, 2], -1)
        # 断言分割后的结果长度为 2
        self.assertEqual(len(out), 2)
        # 断言分割后的第一个部分与预期结果相等
        self.assertEqual(
            out[0],
            torch.nested.as_nested_tensor(
                [a[:, 0:1], b[:, 0:1], c[:, 0:1]], layout=torch.jagged
            ),
        )
        # 断言分割后的第二个部分与预期结果相等
        self.assertEqual(
            out[1],
            torch.nested.as_nested_tensor(
                [a[:, 1:], b[:, 1:], c[:, 1:]], layout=torch.jagged
            ),
        )
        
        # 使用断言检查在维度 1 上使用 split_with_sizes 方法会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"split_with_sizes\(\): not supported for NestedTensor on dim=1",
        ):
            torch.split(nt, [1, 2], 1)

    # 定义一个测试函数，测试 NestedTensor 上的 softmax 方法
    def test_softmax(self, device):
        # 从指定维度生成随机 NestedTensor nt，形状为 [3, None, 5]，使用单精度浮点数，torch.jagged 布局
        nt = random_nt_from_dims(
            [3, None, 5], device=device, dtype=torch.float32, layout=torch.jagged
        )

        # 在维度 2 上对 NestedTensor 进行 softmax 操作
        output = nt.softmax(dim=2)

        # 定义一个内部函数 _compare_to_ref，用于比较计算结果和参考结果的 softmax 操作
        @torch._dynamo.disable
        def _compare_to_ref(nt, output, dim):
            # 对 NestedTensor 拆解后的每个部分进行比较
            for in_component, out_component in zip(nt.unbind(), output.unbind()):
                self.assertEqual(in_component.softmax(dim=dim), out_component)

        # 在拆解后的维度 2 上进行比较
        _compare_to_ref(nt, output, dim=1)

        # 在维度 -1 上对 NestedTensor 进行 softmax 操作
        output2 = nt.softmax(dim=-1)
        # 使用 torch._dynamo.disable 方法对比 output 和 output2 的结果
        torch._dynamo.disable(self.assertEqual)(output, output2)
        # 在拆解后的维度 -1 上进行比较
        _compare_to_ref(nt, output2, dim=-1)
    def test_views_inherit_ragged_dim(self, device):
        # view
        # 创建一个随机的具有不规则维度的命名张量（nt），在指定设备上使用torch.float32数据类型，并采用torch.jagged布局
        nt = random_nt_from_dims(
            [4, None, 8, 10], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # inherit ragged dim via -1
        # 通过-1继承不规则维度
        view = nt.view(4, -1, 80)
        # 断言原始张量和视图的第二维度相同
        self.assertEqual(nt.shape[1], view.shape[1])
        # inherit batch and ragged dims via -1
        # 通过-1继承批次和不规则维度
        view2 = nt.view(-1, -1, 80)
        # 断言原始张量和视图的前两个维度相同
        self.assertEqual(nt.shape[:2], view2.shape[:2])

        # expand
        # 创建一个随机的具有不规则维度的命名张量（nt），在指定设备上使用torch.float32数据类型，并采用torch.jagged布局
        nt = random_nt_from_dims(
            [3, None, 1], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # inherit batch and ragged dims via -1
        # 通过-1继承批次和不规则维度
        view = nt.expand(-1, -1, 5)
        # 断言原始张量和视图的前两个维度相同
        self.assertEqual(nt.shape[:2], view.shape[:2])

    def test_view_ragged_idx_not_one(self, device):
        # 创建一个随机的具有不规则维度的命名张量（nt），在指定设备上使用torch.float32数据类型，并采用torch.jagged布局
        nt = random_nt_from_dims(
            [2, None, 20], device=device, dtype=torch.float32, layout=torch.jagged
        )

        # transpose and view
        # 将第1和第2维度进行转置，然后视图化
        view_transposed = nt.transpose(1, 2).view(2, 20, nt.size(1))
        # 断言视图的大小与预期大小相同
        self.assertEqual((2, 20, nt.size(1)), (view_transposed.size()))
        # 断言视图的基本属性与原始张量相同
        self.assertEqual(view_transposed._base, nt._base)

    def test_unsafe_view(self, device):
        # 创建一个随机的具有不规则维度的命名张量（nt），在指定设备上使用torch.float32数据类型，并采用torch.jagged布局
        nt = random_nt_from_dims(
            [4, None, 8, 10], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # basic view
        # 使用_aten._unsafe_view函数进行基本视图操作
        view1 = torch.ops.aten._unsafe_view(nt, (4, -1, 80))
        # 断言视图的大小与预期大小相同
        self.assertEqual((4, nt.size(1), 80), tuple(view1.size()))
        # _unsafe_view differs from view in that the view information is not tracked
        # _unsafe_view与view不同，它不会跟踪视图信息
        self.assertTrue(view1._base is None)

        # test an unsafe_view when ragged_idx != 1, currently only supports identity view
        # 当不规则索引不等于1时测试unsafe_view，目前仅支持恒等视图
        nt_t = nt.transpose(1, 2)
        # 使用_aten._unsafe_view函数进行视图操作
        view2 = torch.ops.aten._unsafe_view(nt_t, (4, 8, nt.size(1), 10))
        # 断言视图的大小与预期大小相同
        self.assertEqual((4, 8, nt.size(1), 10), tuple(view2.size()))
        # 断言视图的基本属性为None
        self.assertTrue(view2._base is None)

    @xfailIfTorchDynamo
    @parametrize("requires_grad", [False, True])
    # 定义测试函数，用于测试 reshape 操作在不同情况下的表现
    def test_reshape_decomp(self, device, requires_grad):
        # 创建一个非连续的 NamedTensor（NT），形状为 [3, None, 10]
        # 使用给定的设备和数据类型，布局为 torch.jagged
        nt = (
            random_nt_from_dims(
                [3, None, 10],
                device=device,
                dtype=torch.float32,
                layout=torch.jagged,
            )
            .detach()  # 分离出新的 tensor
            .requires_grad_(requires_grad)  # 根据 requires_grad 参数设置是否需要梯度
        )
        # 对 NT 进行 reshape 操作，变换成新的形状 (-1, -1, 5, 2)
        view = nt.reshape(-1, -1, 5, 2)
        # 断言新形状的前两个维度与原始 NT 的前两个维度相同
        self.assertEqual(view.shape[:2], nt.shape[:2])
        # 断言 view 是一个视图，并且其基础对象是 nt
        self.assertTrue(view._is_view() and view._base is nt)
        # 如果需要梯度，确保梯度可以正确传播回去
        if requires_grad:
            view.backward(torch.ones_like(view))
            # 断言 nt 的梯度与全为 1 的 tensor 相同
            self.assertEqual(nt.grad, torch.ones_like(nt))

        # 创建一个非连续的 NamedTensor（NT），形状为 [3, None, 5, 2]
        # 使用给定的设备、数据类型和布局，根据 requires_grad 参数决定是否需要梯度
        nt = random_nt_from_dims(
            [3, None, 5, 2],
            device=device,
            dtype=torch.float32,
            layout=torch.jagged,
            requires_grad=requires_grad,
        )
        # 将 nt 进行维度转置，使其变为非连续
        nt_noncontig = nt.transpose(-1, -2)
        # 断言 nt_noncontig 不是连续的
        self.assertFalse(nt_noncontig.is_contiguous())
        # 对 nt_noncontig 进行 reshape 操作，变换成新的形状 (-1, -1, 10)
        copy = nt_noncontig.reshape(-1, -1, 10)
        # 断言 copy 是连续的
        self.assertTrue(copy.is_contiguous())
        # 断言 copy 的前两个维度与原始 NT 的前两个维度相同
        self.assertEqual(copy.shape[:2], nt.shape[:2])
        # 如果需要梯度，确保梯度可以正确传播回去
        if requires_grad:
            copy.backward(torch.ones_like(copy))
            # 断言 nt 的梯度与全为 1 的 tensor 相同
            self.assertEqual(nt.grad, torch.ones_like(nt))

    # 定义测试函数，用于测试 flatten 操作在不同情况下的表现
    def test_flatten_decomp(self, device):
        # 创建一个 NamedTensor（NT），形状为 [3, None, 5, 2]
        # 使用给定的设备、数据类型和布局
        nt = random_nt_from_dims(
            [3, None, 5, 2], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # 对 NT 进行 flatten 操作，将最后两个维度合并为一个
        flattened = nt.flatten(-2, -1)
        # 断言 flatten 后的形状与手动 view 后的形状相同
        self.assertEqual(flattened.shape, nt.view(3, -1, 10).shape)

        # 创建一个 NamedTensor（NT），形状为 [3, None, 5, 2, 6]
        # 使用给定的设备、数据类型和布局
        nt = random_nt_from_dims(
            [3, None, 5, 2, 6], device=device, dtype=torch.float32, layout=torch.jagged
        )
        # 对 NT 进行 flatten 操作，将倒数第三和倒数第二个维度合并为一个
        flattened = nt.flatten(-3, -2)
        # 断言 flatten 后的形状与手动 view 后的形状相同
        self.assertEqual(flattened.shape, nt.view(3, -1, 10, 6).shape)
    # 测试 chunk 方法
    def test_chunk(self, device):
        # 常规情况
        D = 30
        B = 8
        # 使用指定设备生成随机的 NestedTensor，包含 B 个样本，每个样本有 D 个特征，布局为 torch.jagged
        nt = random_nt_from_dims(
            [B, None, D], device=device, dtype=torch.float32, layout=torch.jagged
        )
        NUM_CHUNKS = 3
        # 在最后一个维度上将 nt 分成 NUM_CHUNKS 个块
        chunks = nt.chunk(NUM_CHUNKS, dim=-1)
        self.assertEqual(len(chunks), NUM_CHUNKS)
        for i in range(NUM_CHUNKS):
            # 检查每个块的最后一个维度是否符合预期
            self.assertEqual(chunks[i].shape[-1], D // NUM_CHUNKS)

        # 在批次维度上进行 chunk
        chunks = nt.chunk(NUM_CHUNKS, dim=0)
        self.assertEqual(len(chunks), NUM_CHUNKS)
        chunk_size = math.ceil(B / NUM_CHUNKS)
        for i in range(NUM_CHUNKS):
            if i < NUM_CHUNKS - 1:
                # 检查每个块的批次维度是否符合预期（非最后一个块）
                self.assertEqual(chunks[i].shape[0], chunk_size)
            else:
                # 检查最后一个块的批次维度是否符合预期
                self.assertEqual(chunks[i].shape[0], B - chunk_size * (NUM_CHUNKS - 1))
            # 检查块的偏移量是否符合预期
            offsets_expected = (
                nt._offsets[i * chunk_size + 1 : (i + 1) * chunk_size + 1]
                - nt._offsets[i * chunk_size]
            )
            self.assertEqual(chunks[i]._offsets[1:], offsets_expected)
        # 检查拼接后的值是否与原始 NestedTensor 的值相等
        self.assertEqual(nt._values, torch.cat([x._values for x in chunks], dim=0))

        # 在 ragged 维度上进行 chunk 不支持
        with self.assertRaisesRegex(
            RuntimeError, "chunk.* not supported for NestedTensor on dim=1"
        ):
            nt.chunk(2, dim=1)

    # 测试 squeeze 方法
    def test_squeeze(self, device):
        B = 4
        D = 6
        # 挤压中间维度
        nt = random_nt_from_dims(
            [B, None, 1, D], device=device, dtype=torch.float32, layout=torch.jagged
        )
        j0 = nt.shape[1]

        for dim_arg in [-2, 2]:
            out = nt.squeeze(dim_arg)
            self.assertEqual(out.shape, (B, j0, D))
            self.assertEqual(out.unsqueeze(-2), nt)

        # 挤压最后一个维度
        nt = random_nt_from_dims(
            [B, None, 1], device=device, dtype=torch.float32, layout=torch.jagged
        )
        j1 = nt.shape[1]

        for dim_arg in [-1, 2]:
            out = nt.squeeze(dim_arg)
            self.assertEqual(out.shape, (B, j1))
            self.assertEqual(out.unsqueeze(-1), nt)

        # 在批次维度上进行挤压不支持
        with self.assertRaisesRegex(
            RuntimeError, "squeeze.* not supported for NestedTensor on dim=0"
        ):
            nt.squeeze(0)

        # 在 ragged 维度上进行挤压不支持
        with self.assertRaisesRegex(
            RuntimeError, "squeeze.* not supported for NestedTensor on dim=1"
        ):
            nt.squeeze(1)
    def test_binary_pointwise_broadcasting(self, device):
        # 定义一个测试函数，用于测试二进制逐点广播操作
        # (B, j0, 3, 4)
        # 使用 self._get_list_for_jagged_tensor 方法获取一个嵌套张量列表，形状为 ((2, 3, 4), 3, 4)，在指定设备上要求梯度计算
        ts = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 3, 4), device, requires_grad=True
        )
        
        # (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
        # (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
        # (B, j0, ?, ?) + (1, ?, ?) -> (B, j0, ?, ?)
        # Unsupported: (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)
        # 定义支持的以及不支持的广播操作示例
        t_sizes = (
            (4,),
            (1, 4),
            (3, 1),
            (1, 3, 1),
            (1, 1, 1, 4),
            # (1, 1, 1, 1, 4), (unsupported today)
        )

        def grad_test_func(t, *ts):
            # 使用 torch.nested.as_nested_tensor 将列表 ts 转换为嵌套张量 nt，布局为 torch.jagged
            nt = torch.nested.as_nested_tensor(list(ts), layout=torch.jagged)
            # 执行 nt + t 操作
            out = nt + t
            return out.values()

        # 遍历不同的 t_size 进行测试
        for t_size in t_sizes:
            # 创建形状为 t_size 的随机张量 t，要求梯度计算，数据类型为 torch.float64，在指定设备上
            t = torch.rand(
                t_size, requires_grad=True, device=device, dtype=torch.float64
            )
            # 使用 gradcheck 函数检查 grad_test_func 的梯度
            gradcheck(grad_test_func, inputs=(t, *ts), check_batched_grad=False)

    def test_threshold_backward(self, device):
        # 使用 self._get_list_for_jagged_tensor 方法获取两个嵌套张量列表 ts1 和 ts2，形状为 ((2, 3, 4), 16)，在指定设备上不要求梯度计算
        ts1 = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 16), device=device, requires_grad=False
        )
        ts2 = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 16), device=device, requires_grad=False
        )

        # 使用 jagged_from_list 函数将 ts1 和 ts2 转换为嵌套张量 nt1 和 nt2，并获取偏移量 offsets
        nt1, offsets = jagged_from_list(ts1, None)
        nt2, offsets = jagged_from_list(ts2, offsets)
        # 分别对 nt1 和 nt2 的值进行深拷贝并分离出张量 buf1 和 buf2
        buf1 = nt1.values().detach().clone()
        buf2 = nt2.values().detach().clone()

        # 调用 torch.ops.aten.threshold_backward 执行 nt1 和 nt2 的阈值反向传播，阈值设定为 0.0
        res_nt = torch.ops.aten.threshold_backward(nt1, nt2, 0.0)
        # 对 buf1 和 buf2 执行阈值反向传播，阈值设定为 0.0
        res_dense = torch.ops.aten.threshold_backward(buf1, buf2, 0.0)

        # 使用 self.assertEqual 检查 res_dense 和 res_nt.values() 的相等性
        self.assertEqual(res_dense, res_nt.values())

    @parametrize("keepdim", [False, True])
    # 定义一个测试函数，用于测试求和操作对不同维度列表形状的处理效果
    def test_sum_int_DimList(self, device, keepdim):
        # (B, j0, 3, 4)
        # 获取一个嵌套张量列表，形状为((2, 3, 4), 3, 4)，在指定设备上，要求支持梯度计算
        ts = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 3, 4), device=device, requires_grad=True
        )

        # 检查形状的正确性
        reduce_dims = (
            # dims, expected shape, expected keepdim shape
            # j0 表示为 None
            ((0, 1), (3, 4), (1, 1, 3, 4)),
            ((1, 2), None, None),
            ((2, 3), (3, None), (3, None, 1, 1)),
            ((0, 1, 3), (3,), (1, 1, 3, 1)),
            ((0, 1, 2), (4,), (1, 1, 1, 4)),
            ((0, 1, 2, 3), tuple(), (1, 1, 1, 1)),
        )

        # 遍历不同的减少维度配置
        for rd, ref_shape_no_keepdim, ref_shape_keepdim in reduce_dims:
            # 如果维度包含 0 或 1，但不同时包含两者
            if (0 in rd) ^ (1 in rd):
                # 断言引发 RuntimeError，提示“应用在不规则维度上，而非批量维度”
                with self.assertRaisesRegex(
                    RuntimeError,
                    "applying over the ragged dimension, but not the batch dimension",
                ):
                    # 将 ts 转换为嵌套张量，布局为 torch.jagged，然后在指定维度上进行求和，保留维度配置
                    nt = torch.nested.as_nested_tensor(ts, layout=torch.jagged)
                    out = torch.sum(nt, dim=rd, keepdim=keepdim)
                continue

            # 将 ts 转换为嵌套张量，布局为 torch.jagged，然后在指定维度上进行求和，保留维度配置
            nt = torch.nested.as_nested_tensor(ts, layout=torch.jagged)
            out = torch.sum(nt, dim=rd, keepdim=keepdim)
            # 根据 keepdim 确定参考形状
            ref_shape = ref_shape_keepdim if keepdim else ref_shape_no_keepdim
            # 断言输出张量的维度数与参考形状长度相等
            self.assertEqual(len(out.shape), len(ref_shape))
            # 遍历输出张量的维度和参考形状，进行逐一比较
            for o, r in zip(out.shape, ref_shape):
                if r is not None:
                    self.assertEqual(o, r)
                else:
                    self.assertTrue(isinstance(o, torch.SymInt))

        # 检查数值的正确性
        # 不规则性未被减少
        nt = torch.nested.as_nested_tensor(ts, layout=torch.jagged)
        out = torch.sum(nt, dim=(2, 3), keepdim=keepdim)
        out_ref = torch.sum(nt.values(), dim=(1, 2))
        # 断言输出为 NestedTensor 类型
        self.assertIsInstance(out, NestedTensor)
        # 展平以避免根据 keepdim 复制 unsqueeze 逻辑
        self.assertTrue(torch.allclose(out.values().view(-1), out_ref.view(-1)))

        # 不规则性被减少
        nt = torch.nested.as_nested_tensor(ts, layout=torch.jagged)
        out = torch.sum(nt, dim=(0, 1), keepdim=keepdim)
        out_ref = torch.sum(nt.values(), dim=(0,))
        # 断言输出不是 NestedTensor 类型
        self.assertNotIsInstance(out, NestedTensor)
        # 检查输出张量与参考张量是否全部接近
        self.assertTrue(torch.allclose(out, out_ref))
    # 定义一个测试方法，用于测试序列化功能
    def test_serialization(self, device, dtype, requires_grad, weights_only):
        # 定义一个比较元数据的内部函数，用于比较两个嵌套张量的元数据是否相等
        def compare_metadata(nt1, nt2):
            self.assertEqual(nt1._nested_tensor_size(), nt2._nested_tensor_size())
            self.assertEqual(nt1._nested_tensor_strides(), nt2._nested_tensor_strides())
            self.assertEqual(
                nt1._nested_tensor_storage_offsets(),
                nt2._nested_tensor_storage_offsets(),
            )

        # 获取一个连续和一个非连续的随机嵌套张量对
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))
        
        # 对每一个随机嵌套张量执行以下操作
        for a in [nt_contiguous, nt_noncontiguous]:
            # 创建一个字节流对象
            buffer = io.BytesIO()
            # 将当前嵌套张量a序列化到字节流中
            serialized = torch.save(a, buffer)
            # 将读取指针移动到字节流的开头
            buffer.seek(0)
            # 从字节流中加载序列化后的张量b
            b = torch.load(buffer, weights_only=weights_only)
            # 断言a和b在概念上相等且元数据相等
            self.assertEqual(a, b)
            compare_metadata(a, b)
            # 断言b与nt_contiguous和nt_noncontiguous在概念上相等，但不一定是元数据相等的
            self.assertEqual(b, nt_contiguous)
            self.assertEqual(b, nt_noncontiguous)

    # 标记为CUDA测试时跳过，用于测试内存检查
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    @onlyCUDA
    def test_pin_memory(self, device):
        # 获取一个连续和一个非连续的随机嵌套张量对
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))
        
        # 对每一个随机嵌套张量执行以下操作
        for nt in [nt_contiguous, nt_noncontiguous]:
            # 断言该张量未被固定到内存中
            self.assertFalse(nt.is_pinned())
            # 固定该张量到指定设备的内存中
            pinned = nt.pin_memory(device)
            # 断言固定后的张量已经被固定到内存中
            self.assertTrue(pinned.is_pinned())
            # 断言原始张量与固定后的张量在概念上相等
            self.assertEqual(nt, pinned)
            # 断言原始张量与固定后张量的数据指针不相同
            self.assertNotEqual(nt.data_ptr(), pinned.data_ptr())
            # 测试对已固定张量再次调用固定操作，应该返回原对象
            self.assertIs(pinned, pinned.pin_memory())
            # 断言固定后张量与其再次固定后的张量的数据指针相同
            self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())

    # 禁用torch编译器
    @torch.compiler.disable
    def _validate_nt(
        self, nt, device, dtype, layout, requires_grad, dim, batch_size, base=None
    ):
        # 验证嵌套张量构造后的多个属性
        device = torch.device(device)
        self.assertEqual(nt.dim(), dim)
        self.assertEqual(nt.device, device)
        self.assertEqual(nt.dtype, dtype)
        self.assertEqual(nt.layout, layout)
        self.assertEqual(nt.requires_grad, requires_grad)

        # 如果布局为torch.jagged，则进一步验证其属性
        if layout == torch.jagged:
            self.assertEqual(nt._values.device, device)
            self.assertEqual(nt._offsets.device, device)
            self.assertEqual(nt.shape[0], batch_size)
            self.assertTrue(isinstance(nt.shape[1], torch.SymInt))

        # 如果存在基张量base，则验证当前张量为视图且其基张量为base
        if base is not None:
            self.assertTrue(nt._is_view() and nt._base is base)

    # 使用指定的数据类型和参数测试嵌套张量的构造（针对torch.float, torch.double, torch.half）
    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("components_require_grad", [False, True])
    def test_jagged_layout_construction_nested_tensor(
        self, device, dtype, requires_grad, components_require_grad
    ):
    ):
        # 遍历所有的示例张量列表，包括列表的列表，根据需要设置是否需要梯度
        for tensor_list in self._get_example_tensor_lists(
            include_list_of_lists=True, include_requires_grad=components_require_grad
        ):
            # 使用torch.nested.nested_tensor()创建嵌套张量nt，指定设备、数据类型、布局和是否需要梯度
            nt = torch.nested.nested_tensor(
                tensor_list,
                device=device,
                dtype=dtype,
                layout=torch.jagged,
                requires_grad=requires_grad,
            )

            # 预期的维度是第一个张量的维度加一
            expected_dim = torch.as_tensor(tensor_list[0]).dim() + 1
            # 预期的批量大小是张量列表的长度
            expected_batch_size = len(tensor_list)
            # 验证创建的嵌套张量nt的属性和维度是否正确
            self._validate_nt(
                nt,
                device,
                dtype,
                torch.jagged,
                requires_grad,
                expected_dim,
                expected_batch_size,
            )

            # 确保在使用requires_grad=True时，梯度不会流回原始张量
            if requires_grad:
                (nt * 2).backward(torch.ones_like(nt))
            # 确保每个张量t的梯度为None
            for t in tensor_list:
                t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                self.assertTrue(t.grad is None)

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("components_require_grad", [False, True])
    def test_jagged_layout_construction_as_nested_tensor(
        self, device, dtype, components_require_grad
    ):
        # 注意：as_nested_tensor(tensor_list)不支持张量列表中包含子列表的情况
        # 遍历所有的示例张量列表，排除包含子列表的情况，并根据需要设置是否需要梯度
        for tensor_list in self._get_example_tensor_lists(
            include_list_of_lists=False, include_requires_grad=components_require_grad
        ):
            # 使用torch.nested.as_nested_tensor()创建嵌套张量nt，指定设备、数据类型和布局为torch.jagged
            nt = torch.nested.as_nested_tensor(
                tensor_list, device=device, dtype=dtype, layout=torch.jagged
            )

            # 如果至少一个组件需要梯度，则应设置nt.requires_grad=True
            expected_dim = tensor_list[0].dim() + 1
            expected_batch_size = len(tensor_list)
            # 验证创建的嵌套张量nt的属性和维度是否正确
            self._validate_nt(
                nt,
                device,
                dtype,
                torch.jagged,
                components_require_grad,
                expected_dim,
                expected_batch_size,
            )

            # 确保在使用components_require_grad=True时，梯度会流回原始张量
            if components_require_grad:
                (nt * 2).backward(torch.ones_like(nt))
                # 确保每个张量t的梯度为2
                for t in tensor_list:
                    if t.requires_grad:
                        self.assertEqual(t.grad, torch.ones_like(t) * 2)
                    else:
                        self.assertTrue(t.grad is None)

    @xfailIfTorchDynamo
    @unittest.skipIf(
        PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property"
    )
    @onlyCUDA
    # 测试使用固定内存构建具有不规则布局的嵌套张量
    def test_jagged_layout_construction_with_pinned_memory(self, device):
        # 对每个张量列表进行迭代，从_get_example_tensor_lists()方法获取示例张量列表
        for tensor_list in self._get_example_tensor_lists():
            # 使用torch.nested.nested_tensor函数创建嵌套张量，指定布局为torch.jagged，
            # 设备为CPU，并且将内存固定在GPU上
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device="cpu", pin_memory=True
            )

            # 预期维度是第一个张量的维度加1
            expected_dim = torch.as_tensor(tensor_list[0]).dim() + 1
            # 预期批量大小等于张量列表的长度
            expected_batch_size = len(tensor_list)
            # 调用验证函数_validate_nt，验证嵌套张量的属性
            self._validate_nt(
                nt,
                device="cpu",
                dtype=torch.float32,
                layout=torch.jagged,
                requires_grad=False,
                dim=expected_dim,
                batch_size=expected_batch_size,
            )
            # 断言嵌套张量已经固定在内存中
            self.assertTrue(nt.is_pinned())

    # 使用视图从值和偏移量创建嵌套张量的测试函数
    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("values_is_view", [False, True])
    def test_jagged_view_from_values_offsets(
        self, device, dtype, requires_grad, values_is_view
    ):
        if values_is_view:
            # 将values设置为base的视图
            base = torch.randn(
                2, 3, 4, 5, 6, device=device, dtype=dtype, requires_grad=requires_grad
            )
            values = base.flatten(0, -2)
        else:
            # 创建具有随机值的张量values
            values = torch.randn(
                10, 5, device=device, dtype=dtype, requires_grad=requires_grad
            )
        # 创建包含偏移量的张量offsets
        offsets = torch.tensor([0, 2, 4, 6, 10], device=device, dtype=torch.int64)

        # 调用nested_view_from_values_offsets函数创建嵌套视图nt
        nt = nested_view_from_values_offsets(values, offsets)

        # 预期维度是values的维度加1
        expected_dim = values.dim() + 1
        # 预期批量大小是offsets数组的长度减去1
        expected_batch_size = offsets.shape[0] - 1
        # 如果values_is_view为True，则预期base等于base；否则等于values
        expected_base = base if values_is_view else values
        # 调用验证函数_validate_nt，验证嵌套张量的属性
        self._validate_nt(
            nt,
            device,
            dtype,
            torch.jagged,
            requires_grad,
            expected_dim,
            expected_batch_size,
            # 确保nt是一个正确的视图
            base=expected_base,
        )

        if requires_grad:
            # 确保梯度能够回传
            (nt * 2).backward(torch.ones_like(nt))

            @torch.compiler.disable
            def _check_grad(t):
                # 断言张量的梯度不为None
                self.assertTrue(t.grad is not None)
                # 断言张量的梯度等于张量本身乘以2
                self.assertEqual(t.grad, torch.ones_like(t) * 2)

            # 如果values_is_view为True，则检查base的梯度；否则检查values的梯度
            _check_grad(base if values_is_view else values)

    # 仅使用torch.float作为参数的测试函数
    @dtypes(torch.float)
    # 测试函数：从不规则张量创建嵌套张量
    def test_nested_tensor_from_jagged(self, device, dtype):
        # 从 (values, offsets) 构造
        values = torch.randn(10, 5, device=device, dtype=dtype)
        offsets = torch.tensor([0, 2, 4, 6, 10], device=device, dtype=torch.int64)
        nt = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
        # 断言返回值是 NestedTensor 类型
        self.assertTrue(isinstance(nt, NestedTensor))
        # 断言是视图且基础张量是 values
        self.assertTrue(nt._is_view() and nt._base is values)
        # 断言张量维度为 3
        self.assertEqual(nt.dim(), 3)
        # 断言第一个维度大小与 offsets 大小减一相等
        self.assertEqual(nt.size(0), offsets.size(0) - 1)
        # 断言最后一个维度大小与 values 最后一个维度大小相等
        self.assertEqual(nt.size(-1), values.size(-1))
        # 断言长度属性为 None
        self.assertIsNone(nt._lengths)
        # 断言张量是否连续
        self.assertTrue(nt.is_contiguous())

        # 从 (values, offsets, lengths) 构造
        lengths = torch.tensor([2, 1, 1, 2], device=device)
        nt = torch.nested.nested_tensor_from_jagged(
            values, offsets=offsets, lengths=lengths
        )
        # 断言返回值是 NestedTensor 类型
        self.assertTrue(isinstance(nt, NestedTensor))
        # 断言是视图且基础张量是 values
        self.assertTrue(nt._is_view() and nt._base is values)
        # 断言张量维度为 3
        self.assertEqual(nt.dim(), 3)
        # 断言第一个维度大小与 offsets 大小减一相等
        self.assertEqual(nt.size(0), offsets.size(0) - 1)
        # 断言最后一个维度大小与 values 最后一个维度大小相等
        self.assertEqual(nt.size(-1), values.size(-1))
        # 断言长度属性与 lengths 相等
        self.assertEqual(nt._lengths, lengths)
        # 当 offsets 和 lengths 都被指定时，预期张量不连续
        self.assertFalse(nt.is_contiguous())

        # 从 (values, lengths) 构造
        values = torch.randn(14, 5, device=device, dtype=dtype)
        lengths = torch.tensor([2, 3, 4, 5], device=device)
        nt = torch.nested.nested_tensor_from_jagged(values, lengths=lengths)
        # 断言返回值是 NestedTensor 类型
        self.assertTrue(isinstance(nt, NestedTensor))
        # 断言是视图且基础张量是 values
        self.assertTrue(nt._is_view() and nt._base is values)
        # 断言张量维度为 3
        self.assertEqual(nt.dim(), 3)
        # 断言第一个维度大小与 lengths 大小相等
        self.assertEqual(nt.size(0), lengths.size(0))
        # 断言最后一个维度大小与 values 最后一个维度大小相等
        self.assertEqual(nt.size(-1), values.size(-1))
        # 当只有 lengths 被指定时，根据最佳核心集成转换为 offsets
        expected_offsets = torch.tensor([0, 2, 5, 9, 14], device=device)
        expected_nt = torch.nested.nested_tensor_from_jagged(
            values, offsets=expected_offsets
        )
        for n1, n2 in zip(nt.unbind(), expected_nt.unbind()):
            self.assertEqual(n1, n2)

        # 错误情况：未指定 offsets 或 lengths
        with self.assertRaisesRegex(
            RuntimeError, "At least one of offsets or lengths is required"
        ):
            torch.nested.nested_tensor_from_jagged(values, offsets=None, lengths=None)
    ):
        # 对于每个输入的维度，创建一个新的张量 t
        if dim == 0:
            # 如果维度是 0，创建一个标量张量并指定是否需要梯度
            t = torch.tensor(3.0, requires_grad=requires_grad)
        else:
            # 如果维度大于 0，创建一个随机张量，维度由参数 dim 决定，同样指定是否需要梯度
            t = torch.randn(*(3 for _ in range(dim)), requires_grad=requires_grad)
        # 确保张量 t 的维度等于预期维度 dim
        assert t.dim() == dim

        if dim < 2:
            # 对于 0 或 1 维度的张量，无法转换为嵌套张量，预期会抛出 RuntimeError 异常
            with self.assertRaisesRegex(
                RuntimeError, "Expected tensor argument to have dim"
            ):
                nt = torch.nested.as_nested_tensor(
                    t, device=device, dtype=dtype, layout=layout
                )
            return

        orig_t = t
        if not contiguous:
            # 如果张量不是连续的，则转置张量 t
            t = t.transpose(0, 1)

        # 将张量 t 转换为嵌套张量 nt，指定设备、数据类型和布局
        nt = torch.nested.as_nested_tensor(t, device=device, dtype=dtype, layout=layout)
        expected_dim = t.dim()
        expected_batch_size = t.size(0)
        # 验证转换后的嵌套张量 nt 的属性和预期值是否一致
        self._validate_nt(
            nt, device, dtype, layout, requires_grad, expected_dim, expected_batch_size
        )

        if torch.device(device) == t.device and dtype == t.dtype and contiguous:
            # 如果设备和数据类型与原始张量 t 相同，并且张量是连续的，则 nt 应该是 t 的视图而非复制
            self.assertTrue(nt._is_view() and nt._base is t)

        # 确保通过从未绑定的张量列表构建的嵌套张量与直接转换得到的 nt 相等
        nt_from_unbind = torch.nested.as_nested_tensor(
            list(t.unbind(0)), device=device, dtype=dtype, layout=layout
        )
        self.assertEqual(nt, nt_from_unbind)

        # 确保对具有相同属性的嵌套张量 nt 再次调用转换函数返回的是 nt 本身
        nt2 = torch.nested.as_nested_tensor(
            nt, device=device, dtype=dtype, layout=layout
        )
        self.assertTrue(nt is nt2)

        # 目前不支持通过此方法在不同布局之间进行转换，预期会抛出 RuntimeError 异常
        other_layout = torch.strided if layout == torch.jagged else torch.jagged
        with self.assertRaisesRegex(
            RuntimeError, "Converting between nested tensor layouts is not supported"
        ):
            torch.nested.as_nested_tensor(
                nt, device=device, dtype=dtype, layout=other_layout
            )

        if requires_grad:
            # 确保梯度能够正确地流回输入张量中
            (nt * 2).backward(torch.ones_like(nt))
            self.assertEqual(orig_t.grad, torch.ones_like(orig_t) * 2)

    @dtypes(torch.double, torch.half)
    @onlyCUDA
    def test_device_dtype_transfer_updates_offsets(self, device, dtype):
        # 遍历获取的示例张量列表
        for tensor_list in self._get_example_tensor_lists():
            orig_device = torch.device("cpu")
            orig_dtype = torch.float32
            # 使用给定设备和数据类型创建嵌套张量 nt，指定布局为 torch.jagged
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=orig_device, dtype=orig_dtype
            )

            # 确保偏移量的数据类型为 int64
            self.assertEqual(torch.int64, nt.offsets().dtype)
            # 将嵌套张量 nt 转换到指定设备和数据类型
            nt = nt.to(device=device).to(dtype=dtype)

            # 新设备上偏移量仍应为 int64 数据类型
            self.assertEqual(nt.values().device, nt.offsets().device)
            self.assertEqual(torch.int64, nt.offsets().dtype)
    # 定义测试方法，用于测试解绑操作对于不同情况的行为
    def test_unbind(self, device):
        # 遍历获取的示例张量列表
        for tensor_list in self._get_example_tensor_lists():
            # 使用 torch.nested.nested_tensor 方法创建嵌套张量 nt，使用 torch.jagged 布局，并指定设备
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=device
            )  # ragged_idx = 1
            # 对嵌套张量进行解绑操作
            out = nt.unbind()
            # 断言解绑后的张量列表长度与原张量列表长度相同
            self.assertEqual(len(out), len(tensor_list))
            # 验证每个解绑后的张量与原始张量列表中的对应元素相等
            for i, t in enumerate(out):
                self.assertEqual(t, tensor_list[i])

    # 使用参数化装饰器，定义测试方法，测试在转置操作后的解绑行为
    @parametrize("ragged_idx", [2, 3])
    def test_unbind_transpose(self, device, ragged_idx):
        # 遍历获取的示例张量列表
        for tensor_list in self._get_example_tensor_lists():
            # 使用 torch.nested.nested_tensor 方法创建嵌套张量 nt，使用 torch.jagged 布局，并指定设备
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=device
            )
            # 如果 ragged_idx 小于嵌套张量的维度数，则进行转置操作
            if ragged_idx < nt.dim():
                nt = nt.transpose(1, ragged_idx)  # set ragged_idx
                # 对转置后的嵌套张量进行解绑操作
                out = nt.unbind()
                # 断言解绑后的张量列表长度与原张量列表长度相同
                self.assertEqual(len(out), len(tensor_list))
                # 验证每个解绑后的张量经过还原转置后与原始张量列表中的对应元素相等
                for i, t in enumerate(out):
                    self.assertEqual(
                        t.transpose(0, ragged_idx - 1), tensor_list[i]
                    )  # transpose back each element of result

    # 定义测试方法，测试在将 ragged_idx 设置为最后一个维度后的解绑行为
    def test_unbind_transpose_ragged_idx_last_dim(self, device):
        # 遍历获取的示例张量列表
        for tensor_list in self._get_example_tensor_lists():
            # 使用 torch.nested.nested_tensor 方法创建嵌套张量 nt，使用 torch.jagged 布局，并指定设备，
            # 然后对其进行转置操作，将 ragged_idx 设置为最后一个维度
            nt = torch.nested.nested_tensor(
                tensor_list, layout=torch.jagged, device=device
            ).transpose(
                1, -1
            )  # set ragged_idx = last dimension
            # 对转置后的嵌套张量进行解绑操作
            out = nt.unbind()
            # 断言解绑后的张量列表长度与原张量列表长度相同
            self.assertEqual(len(out), len(tensor_list))
            # 验证每个解绑后的张量经过还原转置后与原始张量列表中的对应元素相等
            for i, t in enumerate(out):
                self.assertEqual(
                    t.transpose(0, -1), tensor_list[i]
                )  # transpose back each element of result

    # 定义测试方法，测试从 jagged 张量构造出的嵌套张量的解绑行为
    def test_unbind_lengths(self, device):
        # 创建示例值张量和偏移量、长度张量
        values = torch.randn(16, 128, device=device)
        offsets = torch.tensor([0, 8, 12, 13, 16], device=device)
        lengths = torch.tensor([6, 2, 1, 2], device=device)
        # 使用 torch.nested.nested_tensor_from_jagged 方法创建 3D 嵌套张量 nt
        nt = torch.nested.nested_tensor_from_jagged(
            values, offsets=offsets, lengths=lengths
        )  # 3D nested tensor

        tensor_list = []
        # 根据偏移量和长度构造出张量列表
        for i in range(offsets.shape[0] - 1):
            tensor_list.append(values[offsets[i] : (offsets[i] + lengths[i])])

        # 对嵌套张量 nt 进行解绑操作
        out = nt.unbind()
        # 断言解绑后的张量列表长度与构造的张量列表长度相同
        self.assertEqual(len(out), len(tensor_list))
        # 验证每个解绑后的张量与构造的张量列表中的对应元素相等
        for i, t in enumerate(out):
            self.assertEqual(t, tensor_list[i])
    # 使用给定的设备生成一个大小为 [16, 8, 128] 的随机张量
    values = torch.randn(16, 8, 128, device=device)
    # 定义偏移量张量，表示每个子张量在 values 中的起始索引
    offsets = torch.tensor([0, 8, 12, 13, 16], device=device)
    # 定义长度张量，表示每个子张量的长度
    lengths = torch.tensor([6, 2, 1, 2], device=device)
    # 定义 ragged_idx 变量，指定 NestedTensor 的类型
    ragged_idx = 1
    # 创建 NestedTensor 对象，基于给定的 values, offsets, lengths 和 ragged_idx
    nt = torch.nested._internal.nested_tensor.NestedTensor(
        values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
    )  # 4D nested tensor

    # 创建一个空列表，用于存储拆分后的子张量
    tensor_list = []
    # 遍历 offsets 张量，每次取一个子张量的索引范围，并添加到 tensor_list 中
    for i in range(offsets.shape[0] - 1):
        tensor_list.append(values[offsets[i] : (offsets[i] + lengths[i]), :, :])

    # 执行 NestedTensor 对象的 unbind 方法，返回拆分后的子张量列表
    out = nt.unbind()

    # 断言输出列表的长度与 tensor_list 相同
    self.assertEqual(len(out), len(tensor_list))
    # 逐个比较每个输出张量与 tensor_list 中对应位置的张量是否相等
    for i, t in enumerate(out):
        self.assertEqual(t, tensor_list[i])


    # 使用给定的设备生成一个大小为 [16, 8, 128] 的随机张量
    values = torch.randn(16, 8, 128, device=device)
    # 定义偏移量张量，表示每个子张量在 values 中的起始索引
    offsets = torch.tensor([0, 8, 12, 13, 16], device=device)
    # 定义长度张量，表示每个子张量的长度
    lengths = torch.tensor([6, 2, 1, 2], device=device)
    # 定义 ragged_idx 变量，指定 NestedTensor 的类型
    ragged_idx = 2
    # 创建 NestedTensor 对象，基于给定的 values, offsets, lengths 和 ragged_idx
    nt = torch.nested._internal.nested_tensor.NestedTensor(
        values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
    )  # 4D nested tensor

    # 断言调用 unbind 方法时抛出 RuntimeError 异常，异常消息包含指定内容
    self.assertRaisesRegex(
        RuntimeError,
        r"unbind\(\): nested tensor offsets and lengths.*",
        lambda: nt.unbind(),
    )


    # 使用给定的设备生成一个大小为 [16, 8, 128] 的随机张量
    values = torch.randn(16, 8, 128, device=device)
    # 定义偏移量张量，表示每个子张量在 values 中的起始索引
    offsets = torch.tensor([0, 2, 4, 8], device=device)
    # 定义长度张量，表示每个子张量的长度
    lengths = torch.tensor([2, 1, 3], device=device)
    # 定义 ragged_idx 变量，指定 NestedTensor 的类型
    ragged_idx = 2
    # 创建 NestedTensor 对象，基于给定的 values, offsets, lengths 和 ragged_idx
    nt = torch.nested._internal.nested_tensor.NestedTensor(
        values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
    )  # 4D nested tensor

    # 创建一个空列表，用于存储拆分后的子张量
    tensor_list = []
    # 遍历 offsets 张量，每次取一个子张量的索引范围，并添加到 tensor_list 中
    for i in range(offsets.shape[0] - 1):
        tensor_list.append(values[:, offsets[i] : (offsets[i] + lengths[i]), :])

    # 执行 NestedTensor 对象的 unbind 方法，返回拆分后的子张量列表
    out = nt.unbind()

    # 断言输出列表的长度与 tensor_list 相同
    self.assertEqual(len(out), len(tensor_list))
    # 逐个比较每个输出张量与 tensor_list 中对应位置的张量是否相等
    for i, t in enumerate(out):
        self.assertEqual(t, tensor_list[i])


    # 使用给定的设备生成一个大小为 [16, 8, 128] 的随机张量
    values = torch.randn(16, 8, 128, device=device)
    # 定义偏移量张量，表示每个子张量在 values 中的起始索引
    offsets = torch.tensor([0, 100, 128], device=device)
    # 定义长度张量，表示每个子张量的长度
    lengths = torch.tensor([50, 28], device=device)
    # 定义 ragged_idx 变量，指定 NestedTensor 的类型
    ragged_idx = 3
    # 创建 NestedTensor 对象，基于给定的 values, offsets, lengths 和 ragged_idx
    nt = torch.nested._internal.nested_tensor.NestedTensor(
        values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
    )  # 4D nested tensor

    # 创建一个空列表，用于存储拆分后的子张量
    tensor_list = []
    # 遍历 offsets 张量，每次取一个子张量的索引范围，并添加到 tensor_list 中
    for i in range(offsets.shape[0] - 1):
        tensor_list.append(values[:, :, offsets[i] : (offsets[i] + lengths[i])])

    # 执行 NestedTensor 对象的 unbind 方法，返回拆分后的子张量列表
    out = nt.unbind()

    # 断言输出列表的长度与 tensor_list 相同
    self.assertEqual(len(out), len(tensor_list))
    # 逐个比较每个输出张量与 tensor_list 中对应位置的张量是否相等
    for i, t in enumerate(out):
        self.assertEqual(t, tensor_list[i])
    # 定义一个测试方法，用于测试解绑不连续张量的长度超出范围的情况
    def test_unbind_lengths_ragged_idx_0(self, device):
        # 创建一个形状为 (16, 8, 128) 的张量，元素为随机数，存储在指定设备上
        values = torch.randn(16, 8, 128, device=device)
        # 创建一个包含偏移量的张量，指定设备上
        offsets = torch.tensor([0, 100, 128], device=device)
        # 创建一个包含长度信息的张量，指定设备上
        lengths = torch.tensor([50, 28], device=device)
        # 定义不连续张量的索引
        ragged_idx = 0
        # 创建一个嵌套张量对象，包含给定的值、偏移量、长度和不连续张量的索引
        nt = torch.nested._internal.nested_tensor.NestedTensor(
            values, offsets=offsets, lengths=lengths, _ragged_idx=ragged_idx
        )  # 4D nested tensor

        # 初始化一个空列表，用于存储解绑后的子张量
        tensor_list = []
        # 遍历偏移量张量的前 n-1 个元素
        for i in range(offsets.shape[0] - 1):
            # 将从 values 张量中切片出偏移量和长度范围内的子张量，并添加到列表中
            tensor_list.append(values[:, :, offsets[i] : (offsets[i] + lengths[i])])

        # 断言异常信息中包含特定的运行时错误信息
        self.assertRaisesRegex(
            RuntimeError,
            r"unbind\(\): nested tensor.*out of bounds",
            lambda: nt.unbind(),
        )

    # 如果不支持 Torch Dynamo，标记为预期失败的测试方法
    @xfailIfTorchDynamo
    # 定义一个测试方法，用于测试 layer_norm 函数在嵌套张量上的行为
    def test_layer_norm_2(self, device):
        # 获取用于测试的嵌套张量列表
        test_tensor_list = self._get_list_for_jagged_tensor(
            ((2, 3, 4), 3), device=device, requires_grad=True
        )
        # 创建一个偏置张量，形状为 (3,)，元素为随机数，不需要梯度，存储在指定设备上
        bias = torch.randn(3, requires_grad=False, dtype=torch.float64, device=device)

        # 定义用于梯度检查的函数
        def grad_test_func(a, b, c, bias):
            # 将输入张量列表转换为嵌套张量，使用 torch.jagged 布局
            nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
            # 对嵌套张量执行 layer_norm 操作，指定偏置
            out = torch.nn.functional.layer_norm(nt, (nt.shape[-1],), bias=bias)
            # 返回 layer_norm 操作的值
            return out.values()

        # 执行梯度检查，检查函数 grad_test_func 的梯度，不批量化梯度检查
        gradcheck(
            grad_test_func, inputs=(*test_tensor_list, bias), check_batched_grad=False
        )

        # 断言特定的运行时错误信息，当 layer_norm 函数尝试在嵌套张量的不连续维度上进行归一化时
        with self.assertRaisesRegex(
            RuntimeError,
            r"layer_norm\(\): normalizing over ragged dim not supported for nested tensors",
        ):
            # 将测试张量列表转换为嵌套张量，使用 torch.jagged 布局
            nt = torch.nested.as_nested_tensor(test_tensor_list, layout=torch.jagged)
            # 执行 layer_norm 操作，尝试在嵌套张量的不连续维度上进行归一化
            _ = torch.nn.functional.layer_norm(nt, (nt.shape[-2], nt.shape[-1]))

    # 定义一个测试方法，用于测试 narrow 函数在嵌套张量上的行为
    def test_narrow(self, device):
        # 创建一个包含起始位置的张量，指定设备上，数据类型为 torch.int64
        starts = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int64)
        # 创建一个包含长度信息的张量，指定设备上，数据类型为 torch.int64
        lengths = torch.tensor([3, 2, 2, 1, 5], device=device, dtype=torch.int64)
        # 创建一个张量 buffer，其值为从 0 到 9 的整数，形状为 (5, 10)，存储在指定设备上，数据类型为 torch.int64
        buffer = (
            torch.arange(0, 10, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(5, -1)
            .clone()
            .detach()
        )
        # 使用 narrow 函数创建一个嵌套张量 nt，指定布局为 torch.jagged
        nt = torch.nested.narrow(buffer, 1, starts, lengths, layout=torch.jagged)

        # 断言 nt 是一个视图，且其基本张量是 buffer
        self.assertTrue(nt._is_view() and nt._base is buffer)

        # TODO: 当 unbind 函数可用时使用此方法
        # unbinded_nt = nt.unbind()
        # for i in range(starts.shape[0]):
        #     self.assertEqual(torch.arange(starts[i], starts[i] + lengths[i], device=device, dtype=torch.int64), unbinded_nt[i])

        # 遍历 starts 张量的每个元素
        for i in range(starts.shape[0]):
            # 断言嵌套张量中特定偏移和长度范围内的值等于预期的范围值
            self.assertEqual(
                torch.arange(
                    starts[i], starts[i] + lengths[i], device=device, dtype=torch.int64
                ),
                nt.values()[nt.offsets()[i] : (nt.offsets()[i] + nt.lengths()[i])],
            )
    # 测试函数，用于检查是否为连续存储
    def test_is_contiguous(self, device):
        # 创建三个随机张量 a, b, c，设备为指定设备，类型为 float64，并且需要梯度跟踪
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64, device=device)
        # 使用 nested.as_nested_tensor 将 a, b, c 组合成一个嵌套张量，布局为 jagged
        nt_contiguous = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)

        # 创建起始索引和长度张量 starts_nc 和 lengths_nc，设备为指定设备，类型为 int64
        starts_nc = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int64)
        lengths_nc = torch.tensor([3, 2, 2, 1, 5], device=device, dtype=torch.int64)
        # 创建一个基础张量 narrow_base，值为 0 到 9，设备为指定设备，类型为 int64
        narrow_base = (
            torch.arange(0, 10, device=device, dtype=torch.int64)
            .unsqueeze(0)  # 在第一维度添加一个维度
            .expand(5, -1)  # 扩展为 5 行，与原张量行数一致
            .clone()  # 克隆张量，确保独立性
        )
        # 使用 nested.narrow 函数对 narrow_base 进行裁剪，起始索引为 starts_nc，长度为 lengths_nc，布局为 jagged
        nt_noncontiguous = torch.nested.narrow(
            narrow_base, 1, starts_nc, lengths_nc, layout=torch.jagged
        )

        # 创建起始索引和长度张量 starts_c 和 lengths_c，设备为指定设备，类型为 int64
        starts_c = torch.tensor([1, 0, 0, 0, 0], device=device, dtype=torch.int64)
        lengths_c = torch.tensor([9, 10, 10, 10, 8], device=device, dtype=torch.int64)
        # 使用 nested.narrow 函数对 narrow_base 进行裁剪，起始索引为 starts_c，长度为 lengths_c，布局为 jagged
        nt_contiguous_narrow = torch.nested.narrow(
            narrow_base, 1, starts_c, lengths_c, layout=torch.jagged
        )

        # 测试连续情况
        # 断言 nt_contiguous 是否为连续存储
        assert nt_contiguous.is_contiguous()

        # 测试非连续情况
        # 断言 nt_noncontiguous 是否为连续存储
        assert not nt_noncontiguous.is_contiguous()
        # 断言 nt_contiguous_narrow 是否为连续存储
        assert nt_contiguous_narrow.is_contiguous()

        # 测试按内存格式查询
        # 使用 memory_format 参数检查 nt_contiguous 是否为连续存储
        self.assertTrue(
            nt_contiguous.is_contiguous(memory_format=torch.contiguous_format)
        )
        # 使用 memory_format 参数检查 nt_noncontiguous 是否为连续存储
        self.assertTrue(
            not nt_noncontiguous.is_contiguous(memory_format=torch.contiguous_format)
        )
        # 使用 memory_format 参数检查 nt_contiguous_narrow 是否为连续存储
        self.assertTrue(
            nt_contiguous_narrow.is_contiguous(memory_format=torch.contiguous_format)
        )
    # 测试函数，用于验证函数 func 处理类似结构的张量的正确性
    def test_like_value(self, func):
        # 从指定维度随机生成一个嵌套张量 nt，布局为 torch.jagged，使用随机数据，放在 CPU 上
        nt = random_nt_from_dims(
            [2, None, 3], torch.device("cpu"), torch.float32, layout=torch.jagged
        )
        # 使用 func 处理 nt，生成类似结构的新张量 nt_like
        nt_like = func(nt)

        # 对 nt_like 中的每个子张量进行解绑操作
        for nt_ub in nt_like.unbind():
            # 对解绑后的每个子张量 nt_ub，再次使用 func 进行处理，期望其等于原始解绑前的 nt_ub
            t_like = func(nt_ub)
            self.assertEqual(nt_ub, t_like)

    # 测试非连续操作在嵌套张量上的表现
    def test_noncontiguous_pointwise(self, device):
        # 创建三个随机张量 a, b, c，布局为 torch.jagged，数据类型为 torch.float64，在指定设备上，且要求梯度计算
        a = torch.randn(2, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(4, 3, 4, requires_grad=True, dtype=torch.float64, device=device)
        # 创建嵌套张量 nt，包含上述三个张量，布局为 torch.jagged
        nt = torch.nested.nested_tensor([a, b, c], layout=torch.jagged)
        # 在 nt 上进行维度转置操作，交换第 1 和第 2 维
        transposed = nt.transpose(1, 2)
        # 断言转置后的张量不是连续的
        self.assertFalse(transposed.is_contiguous())
        # 对转置后的张量进行克隆操作
        clone = transposed.clone()

        # 定义函数用于检查两个嵌套张量的相等性
        def check_nt_equality(x, y):
            # 检查两个张量的值是否相等
            self.assertEqual(x.values(), y.values())
            # 检查两个张量的偏移量是否相等
            self.assertEqual(x.offsets(), y.offsets())
            # 检查两个张量的 _ragged_idx 属性是否相等
            self.assertEqual(x._ragged_idx, y._ragged_idx)
            # 检查两个张量的形状是否相等
            self.assertEqual(x.shape, y.shape)

        # 再次断言克隆后的张量不是连续的
        self.assertFalse(clone.is_contiguous())
        # 检查克隆后的张量与转置后的张量是否相等
        check_nt_equality(clone, transposed)

        # 将转置后的张量克隆为连续内存格式的张量
        clone_contig = transposed.clone(memory_format=torch.contiguous_format)
        # 断言克隆后的连续内存格式张量是连续的
        self.assertTrue(clone_contig.is_contiguous())
        # 检查克隆后的连续内存格式张量与转置后的张量是否相等
        check_nt_equality(clone_contig, transposed)

        # 对转置后的张量进行分离操作
        detached = transposed.detach()
        # 断言分离后的张量不是连续的
        self.assertFalse(clone.is_contiguous())
        # 检查分离后的张量与转置后的张量是否相等
        check_nt_equality(detached, transposed)

    # 测试在指定设备上，将嵌套张量转换为指定数据类型的副本
    def test_to_copy(self, device):
        # 创建嵌套张量 nt，其中包含三个随机形状不同的张量，数据类型为 torch.float64，布局为 torch.jagged
        nt = torch.nested.nested_tensor(
            [
                torch.randn(
                    i + 2, 3, 4, requires_grad=True, dtype=torch.float64, device=device
                )
                for i in range(3)
            ],
            layout=torch.jagged,
        )

        # 使用底层操作函数 _to_copy 将 nt 转换为指定数据类型的副本 nt_copy_dtype
        nt_copy_dtype = torch.ops.aten._to_copy(nt, dtype=torch.float16)
        # 断言副本的数据类型为 torch.float16
        self.assertEqual(torch.float16, nt_copy_dtype.dtype)

        # 对 nt 进行维度转置操作
        nt_t = nt.transpose(1, 2)
        # 使用底层操作函数 _to_copy 将转置后的 nt_t 转换为指定数据类型的副本 nt_t_copy_dtype
        nt_t_copy_dtype = torch.ops.aten._to_copy(nt_t, dtype=torch.float16)
        # 断言副本的数据类型为 torch.float16
        self.assertEqual(torch.float16, nt_t_copy_dtype.dtype)

    # 如果使用 Torch Dynamo，跳过具有 prof.events() 跟踪功能的测试
    @skipIfTorchDynamo("Dynamo doesn't know how to trace prof.events()")
    # 定义一个测试方法，用于测试性能分析器的序列号功能
    def test_profiler_sequence_nr(self):
        # 使用 torch.profiler.profile() 创建性能分析器 prof
        with torch.profiler.profile() as prof:
            # 创建一个大小为 (4, 6) 的张量 values，开启梯度追踪
            values = torch.randn(4, 6, requires_grad=True)
            # 创建一个偏移量张量 offsets
            offsets = torch.tensor([0, 2, 4])
            # values 张量每个元素乘以 2
            values = values * 2
            # 创建一个输入维度为 6，输出维度为 8 的线性层 l
            l = torch.nn.Linear(6, 8)
            # 使用 values 和 offsets 创建一个 nested tensor nt
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)

            # 将 nt 输入线性层 l 中
            nt = l(nt)
            # 获取 nt 的值
            val = nt.values()

            # 计算值的和作为损失
            loss = val.sum()
            # 反向传播损失
            loss.backward()

        # 初始化前向传播事件的序列号列表 fwd_seq_nrs
        fwd_seq_nrs = []
        # 遍历 prof 中的事件
        for evt in prof.events():
            # 如果事件名中包含 "linear"，且不包含 "backward"，且序列号不为 -1
            if (
                "linear" in evt.name.lower()
                and "backward" not in evt.name.lower()
                and evt.sequence_nr != -1
            ):
                # 将事件的序列号加入 fwd_seq_nrs
                fwd_seq_nrs.append(evt.sequence_nr)

        # 初始化反向传播事件的序列号列表 bwd_seq_nrs
        bwd_seq_nrs = []
        # 再次遍历 prof 中的事件
        for evt in prof.events():
            # 如果事件名中包含 "linear"，且包含 "backward"，且不包含 "evaluate_function"，且序列号不为 -1
            if (
                "linear" in evt.name.lower()
                and "backward" in evt.name.lower()
                and "evaluate_function" not in evt.name.lower()
                and evt.sequence_nr != -1
            ):
                # 将事件的序列号加入 bwd_seq_nrs
                bwd_seq_nrs.append(evt.sequence_nr)

        # 断言前向传播事件的序列号列表长度为 1
        self.assertEqual(len(fwd_seq_nrs), 1)
        # 断言反向传播事件的序列号列表长度为 1
        self.assertEqual(len(bwd_seq_nrs), 1)
        # 断言前向传播事件的第一个序列号与反向传播事件的第一个序列号相等
        self.assertEqual(fwd_seq_nrs[0], bwd_seq_nrs[0])

    # 定义一个测试方法，用于测试在指定设备上是否相同大小的张量
    def test_is_same_size(self, device):
        # 定义一个内部方法，返回三个张量列表，每个张量大小逐渐增加
        def get_3_tensors():
            return [
                torch.randn(
                    i + 2, 3, 4, requires_grad=True, dtype=torch.float64, device=device
                )
                for i in range(3)
            ]

        # 使用 get_3_tensors() 获取第一组张量和偏移量 nt1, offsets1
        nt1, offsets1 = jagged_from_list(get_3_tensors(), None)
        # 使用 get_3_tensors() 获取第二组张量和偏移量 nt2, offsets1
        nt2, offsets1 = jagged_from_list(get_3_tensors(), offsets1)

        # 使用 get_3_tensors() 获取第三组张量和偏移量 nt3, offsets2
        nt3, offsets2 = jagged_from_list(get_3_tensors(), None)
        # 使用 get_3_tensors() 获取第四组张量和偏移量 nt4, offsets2
        nt4, offsets2 = jagged_from_list(get_3_tensors(), offsets2)

        # 定义一个内部方法，用于检查四组张量的大小是否一致
        def check_size(nt1, nt2, nt3, nt4):
            # 断言 nt1 和 nt2 大小相同
            self.assertTrue(torch.ops.aten.is_same_size(nt1, nt2))
            # 断言 nt3 和 nt4 大小相同
            self.assertTrue(torch.ops.aten.is_same_size(nt3, nt4))
            # 断言 nt1 和 nt3 大小不相同
            self.assertFalse(torch.ops.aten.is_same_size(nt1, nt3))

        # 检查第一组和第二组张量的大小
        check_size(nt1, nt2, nt3, nt4)

        # 对四组张量分别进行转置，并检查转置后的大小
        nt1_t, nt2_t, nt3_t, nt4_t = (x.transpose(1, 2) for x in (nt1, nt2, nt3, nt4))
        check_size(nt1_t, nt2_t, nt3_t, nt4_t)

    # 根据条件选择跳过测试的装饰器：如果 TorchDynamo 不支持，则跳过
    @skipIfTorchDynamo("compiles internally")
    # 根据条件选择跳过测试的装饰器：如果运行环境为 Windows，则跳过
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    # 根据条件选择跳过测试的装饰器：如果 GPU 的计算能力小于 SM70，则跳过
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    # 定义一个测试方法，用于测试动态形状的特化
    def test_specialize_dynamic_shape(self, device):
        # 生成一个指定设备上的随机张量 values，形状为 (18, 16)
        values = torch.randn((18, 16), device=device)
        # 创建一个偏移张量 offsets，指定设备上的值为 [0, 2, 3, 6, 15, 18]
        offsets = torch.tensor([0, 2, 3, 6, 15, 18], device=device)
        # 生成一个与 values 相同形状的随机张量 like_values
        like_values = torch.randn_like(values)

        # 使用 nested_tensor_from_jagged 方法将 values 标记为动态形状
        nt = torch.nested.nested_tensor_from_jagged(values, offsets)

        # 定义一个内部函数 fn，接受 values 和 same_size 参数
        def fn(values, same_size):
            # 在这里，通过 same_size 的形状特化动态形状
            # 参考：https://github.com/pytorch/pytorch/issues/127097
            # 确保在 torch.compile 中不会报错
            return values + same_size

        # 使用 assertEqual 检查未编译前和编译后的 fn(values, like_values) 是否相等
        self.assertEqual(
            fn(values, like_values),
            torch.compile(fn)(values, like_values),
        )

    # 跳过在 Torch Dynamo 中编译的测试
    @skipIfTorchDynamo("compiles internally")
    # 在 Windows 系统上跳过测试，因为 torch.compile 尚不支持
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    # 如果 GPU 能力小于 SM70，则跳过 CUDA 测试
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    def test_specialize_dynamic_shape_recompile(self, device):
        # 定义一个生成输入的函数，参数为 total_len
        def generate_inp(total_len):
            # 在指定设备上生成形状为 (total_len, 16) 的随机张量 values
            values = torch.randn((total_len, 16), device=device)
            # 创建一个偏移张量 offsets，指定设备上的值为 [0, 2, 3, 6, 15, total_len]
            offsets = torch.tensor([0, 2, 3, 6, 15, total_len], device=device)
            # 生成一个与 values 相同形状的随机张量 like_values
            like_values = torch.randn_like(values)
            return values, offsets, like_values

        # 定义一个检查结果的函数，接受参考函数 ref_fn、结果函数 res_fn 和参数 args
        def check_results(ref_fn, res_fn, args):
            # 解包参数 args
            values, offsets, like_values = args
            # 使用 nested_tensor_from_jagged 方法标记 values 的动态形状
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)

            # 使用 assertEqual 检查 ref_fn(values, like_values) 和 res_fn(values, like_values) 是否相等
            self.assertEqual(
                ref_fn(values, like_values),
                res_fn(values, like_values),
            )

        # 定义一个函数 fn，接受 values 和 same_size 参数，返回它们的加法结果
        def fn(values, same_size):
            return values + same_size

        # 创建一个 CompileCounter 对象来统计编译次数
        compile_counter = torch._dynamo.testing.CompileCounter()

        # 使用 torch._dynamo.optimize 方法优化 fn 函数的编译过程，禁用 Python 优化
        compiled_fn = torch._dynamo.optimize(compile_counter, nopython=True)(fn)
        # 检查生成输入为 18 时的结果
        check_results(fn, compiled_fn, generate_inp(18))
        # 确保编译次数为 1
        self.assertEqual(compile_counter.frame_count, 1)

        # 检查生成输入为 19 时的结果
        check_results(fn, compiled_fn, generate_inp(19))
        # 可能会由于动态形状而再次编译，编译次数为 1 或 2 都是可以接受的
        frame_count_2 = compile_counter.frame_count
        self.assertIn(frame_count_2, [1, 2])

        # 确保此时已经使用动态形状编译过，所以额外的形状不应触发额外的重新编译
        # 检查生成输入为 20 时的结果
        check_results(fn, compiled_fn, generate_inp(20))
        self.assertEqual(compile_counter.frame_count, frame_count_2)

    # 注释：Math fallback 在 CUDA 上不支持 bfloat16
    # 注释：ROCm 不支持 NT 的 flash attention 或 mem_efficient attention
    @unittest.skipIf(
        TEST_WITH_ROCM,
        "ROCm doesn't support flash attention or mem_efficient attention for NT",
    )
    # 使用装饰器指定函数参数的数据类型，根据条件选择不同的浮点数类型
    @dtypes(
        *(
            [torch.float16, torch.bfloat16, torch.float32]
            if SM80OrLater
            else [torch.float16, torch.float32]
        )
    )
    # 在Torch Dynamo环境下跳过测试，因为SDPA测试在内部编译
    @skipIfTorchDynamo("SDPA test compiles internally")
    # 如果运行环境是Windows，则跳过此测试，因为torch.compile尚不支持Windows
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    # 如果CUDA设备的计算能力小于SM70，则跳过测试
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    # 在ROCm环境下，由于sqrt()的保护不起作用，跳过此测试
    @skipCUDAIfRocm
    # 仅在CUDA环境下运行测试
    @onlyCUDA
    # 再次使用装饰器指定函数参数的数据类型，根据条件选择不同的浮点数类型
    @dtypes(
        *(
            [torch.float16, torch.bfloat16, torch.float32]
            if SM80OrLater
            else [torch.float16, torch.float32]
        )
    )
    # 指定函数参数的数据类型为torch.float32、torch.double和torch.half
    @dtypes(torch.float32, torch.double, torch.half)
    # 定义一个测试函数，测试SDPA（Scaled Dot-Product Attention）与常数序列长度的情况
    def test_sdpa_with_constant_sequence_length(self, device, dtype):
        # 定义一个随机生成的嵌套张量query，形状为 [4, None, 8, 10]
        # 其中B为批大小，P*为不规则的提示数量，S为常数序列长度，D为嵌入大小
        query = random_nt_from_dims(
            [4, None, 8, 10],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )
        # 从query生成一个随机的key和value
        key = random_nt_from_similar(query)
        value = random_nt_from_similar(query)
        # 使用Scaled Dot-Product Attention计算输出
        output = F.scaled_dot_product_attention(query, key, value)
        # 断言输出是NestedTensor类型
        self.assertTrue(isinstance(output, NestedTensor))
        # 对输出的所有值求和并进行反向传播
        output.values().sum().backward()

        # 克隆query并设置requires_grad为True，用于后续的比较和梯度计算
        query_dense = query.clone().detach().requires_grad_(True)
        # 对query_dense、key和value的values应用Scaled Dot-Product Attention
        output_dense = F.scaled_dot_product_attention(
            query_dense.values(), key.values(), value.values()
        )
        # 使用torch._dynamo.disable方法来执行断言，验证两个输出是否相等
        torch._dynamo.disable(self.assertEqual)(output._values, output_dense)
        # 对output_dense的所有值求和并进行反向传播
        output_dense.sum().backward()
        # 使用torch._dynamo.disable方法来执行断言，验证query和query_dense的梯度是否相等
        torch._dynamo.disable(self.assertEqual)(query.grad, query_dense.grad)

    # 仅在CUDA环境下运行测试
    @onlyCUDA
    # 如果平台不支持融合注意力（fused attention）或内存高效注意力（mem-efficient attention），则跳过测试
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Platform doesn't support flash or mem-efficient attention",
    )
    # 再次使用装饰器指定函数参数的数据类型，根据条件选择不同的浮点数类型
    @dtypes(
        *(
            [torch.float16, torch.bfloat16, torch.float32]
            if SM80OrLater
            else [torch.float16, torch.float32]
        )
    )
    # 定义一个测试函数，用于测试带有输入投影的分散注意力机制（SDPA）实现
    def test_sdpa_with_packed_in_proj(self, device, dtype):
        # 创建一个具有形状 (B, *, D) 的随机输入数据，使用 torch.jagged 布局
        input_packed = random_nt_from_dims(
            [5, None, 10], device=device, dtype=dtype, layout=torch.jagged
        )

        # 执行输入投影
        num_heads = 2
        # 为了使用效率更高的内核（例如 flash / mem-efficient），head_dim 应该是 4 的倍数
        head_dim = 8
        # 创建一个线性层 qkv_linear，将输入投影到 Q、K、V 向量
        qkv_linear = torch.nn.Linear(10, num_heads * head_dim * 3).to(
            device=device, dtype=dtype
        )

        # 定义输入投影函数 in_proj，将输入数据投影到 Q、K、V 向量
        def in_proj(input_packed, qkv_linear=qkv_linear):
            qkv_post_proj = qkv_linear(input_packed)
            # 将投影后的结果分割成 Q、K、V 向量，这些向量是非连续的，用于触发 _is_safe_to_get_storage_as_tensor() 函数
            q, k, v = qkv_post_proj.chunk(3, dim=-1)
            # 将 Q、K、V 向量重新整形为 (num_heads, head_dim) 的形状，并转置最后两个维度
            q = q.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            k = k.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            v = v.unflatten(-1, [num_heads, head_dim]).transpose(-2, -3)
            return q, k, v

        # 对输入数据进行投影，获取 Q、K、V 向量
        q, k, v = in_proj(input_packed)
        # 使用 scaled_dot_product_attention 函数计算输出
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=None)

        # 与单独运行未绑定组件的输出进行比较
        for in_component, out_component in zip(
            input_packed.unbind(), output.transpose(-2, -3).unbind()
        ):
            # 对每个组件执行输入投影
            q, k, v = in_proj(in_component)
            # 使用 scaled_dot_product_attention 函数计算输出，并转置最后两个维度
            out = F.scaled_dot_product_attention(q, k, v).transpose(-2, -3)

            # 获取低精度数学引用结果
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(q, k, v)[0].transpose(-2, -3)
            # 获取比较的容差
            output_ref_atol, output_ref_rtol = get_tolerances(
                out, out_lp_ref, fudge_factor=2
            )

            # 断言输出与预期的组件输出相等
            self.assertEqual(
                out, out_component, atol=output_ref_atol, rtol=output_ref_rtol
            )
    # 定义一个测试方法，测试使用反向传播的情况
    def test_sdpa_backwards(self, device, dtype):
        # 生成一个形状为 (9, 3, 256) 的张量，要求梯度，放在指定的设备上，并使用指定的数据类型
        values = torch.randn(9, 3, 256, requires_grad=True, device=device, dtype=dtype)
        # 创建一个张量，表示偏移量，使用指定的设备和数据类型
        offsets = torch.tensor([0, 1, 3, 5, 9], device=device, dtype=torch.int64)

        # 使用 @torch.compile 进行编译，定义一个函数 f，接受 values 和 offsets 作为参数
        @torch.compile
        def f(values, offsets):
            # 将 values 和 offsets 转换成嵌套张量，最大长度限制为 4
            nt = convert_jagged_to_nested_tensor(values, offsets, max_length=4)
            # 将张量的维度 -2 和 -3 进行转置操作
            nt = nt.transpose(-2, -3)
            # 故意中断图以触发子类视图输入的视图重播
            torch.tensor(1).item()  # 这里是故意中断图的地方
            # 对 nt 使用缩放点积注意力机制，输出结果进行 -2 和 -3 维度的转置
            output = F.scaled_dot_product_attention(nt, nt, nt).transpose(-2, -3)
            # 将输出转换回嵌套张量的形式
            return convert_nt_to_jagged(output)

        # 调用函数 f，传入 values 和 offsets，得到输出
        output = f(values, offsets)
        # 对输出求和并进行反向传播
        output.sum().backward()
        # 断言 values 的梯度与一个与 values 形状相同且元素为1的张量相等
        self.assertEqual(values.grad, torch.ones_like(values))

    # 将内部定义的 NT 使用案例最大化到此处以进行最大的测试真实性。
    # 当 ViewNestedFromBuffer 等被弃用时，应移除这些。
    @skipCUDAIfRocm  # 不需要在 Rocm 平台上跳过
    @skipIfTorchDynamo("compiles internally")  # 在内部编译时跳过
    @unittest.skipIf(IS_WINDOWS, reason="Windows 尚未支持 torch.compile")  # 在 Windows 上跳过，因为尚未支持 torch.compile
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")  # 如果 GPU 能力小于 SM70，则跳过
    # 定义一个测试函数，测试带有正常测试的多头注意力机制
    def test_dummy_mha_with_nt(self, device):
        # 设定批大小、维度信息
        bs = 3
        d1 = 2
        d2 = 4
        d3 = 6
        # 设置注意力头数和每个头的维度
        n_heads = 2
        d_head = d3 // n_heads
        # 设置两个最大长度
        max_length_1 = 10
        max_length_2 = 20
        # 设置随机种子
        torch.manual_seed(0)

        # 定义一个内部类 mha，继承自 torch.nn.Module
        class mha(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 设置随机种子
                torch.manual_seed(0)
                # 定义一个线性层
                self.linear = torch.nn.Linear(d2, d3, device=device)

            # 前向传播函数
            def forward(self, query, value, offsets):
                # 对 value 应用线性变换
                value = self.linear(value)
                # 将 value 转换为嵌套张量形式，根据给定的偏移和最大长度
                key = convert_jagged_to_nested_tensor(value, offsets, max_length_1)
                # 同样将 value 转换为嵌套张量形式，根据给定的偏移和最大长度
                value = convert_jagged_to_nested_tensor(value, offsets, max_length_2)
                # 将 query 转换为密集嵌套张量
                query = convert_dense_to_nested_tensor(query)
                # 将 query 重塑为形状为 (bs, n_heads, d_head) 的张量，并转置
                q = query.view(bs, -1, n_heads, d_head).transpose(1, 2)
                # 将 key 重塑为形状为 (bs, n_heads, d_head) 的张量，并转置
                k = key.view(bs, -1, n_heads, d_head).transpose(1, 2)
                # 将 value 重塑为形状为 (bs, n_heads, d_head) 的张量，并转置
                v = value.view(bs, -1, n_heads, d_head).transpose(1, 2)
                # 使用缩放点积注意力计算注意力输出
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                # 将注意力输出的维度再次转置
                attn_output = attn_output.transpose(1, 2)
                # 将注意力输出转换回嵌套张量的形式
                attn_output = convert_nt_to_jagged(attn_output)
                # 返回注意力输出、key 的最大序列长度、value 的最大序列长度
                return attn_output, key._max_seqlen, value._max_seqlen

        # 生成随机的 query 张量
        query = torch.rand(bs, d1, d3, device=device)
        # 生成随机的 value 张量，并设置 requires_grad 为 True
        value = torch.rand(6, d2, requires_grad=True, device=device)
        # 定义偏移张量
        offsets = torch.tensor([0, 2, 3, 6], device=device)

        # 实例化 mha 类
        m = mha()
        # 对 mha 类进行符号化跟踪
        symbolic_traced: torch.fx.GraphModule = torch.fx.symbolic_trace(m)
        # 编译符号化跟踪后的模型
        m = torch.compile(symbolic_traced)
        # 在编译后的模型上执行前向传播，获取注意力输出、key 的最大序列长度、value 的最大序列长度
        attn_output, cached_key_max_seqlen, cached_value_max_seqlen = m(
            query, value, offsets
        )
        # 计算注意力输出的和作为损失
        loss = attn_output.sum()
        # 对损失进行反向传播
        loss.backward()

        # 检查 value 是否在跟踪和编译后依然具有 requires_grad
        value_grad = value.grad  # 保存以便后续比较
        self.assertIsNotNone(value_grad)
        # 检查 max_seqlen 是否正确缓存
        self.assertEqual(cached_key_max_seqlen, max_length_1)
        self.assertEqual(cached_value_max_seqlen, max_length_2)

        # 检查输出是否在计算时与即时模式的输出数值上等价
        m_eager = mha()
        value.grad = None
        attn_output_eager, _, _ = m_eager(query, value, offsets)
        attn_output_eager.sum().backward()
        self.assertTrue(torch.allclose(attn_output_eager, attn_output))
        self.assertTrue(torch.allclose(value_grad, value.grad))

    @dtypes(torch.float32)
    # 定义测试方法 test_apply_，接受设备和数据类型作为参数
    def test_apply_(self, device, dtype):
        # 生成一个随机的 NamedTensor，其中维度为 [5, None, 10]
        nt = random_nt_from_dims(
            [5, None, 10],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )

        # 定义一个函数 f，对输入的张量 x 进行乘以 2 的操作
        def f(x):
            return x * 2

        # 如果设备不是 "cpu"，则应该抛出 TypeError 异常，提示 apply_ 方法只能在 CPU 上使用
        if device != "cpu":
            with self.assertRaisesRegex(
                TypeError, "apply_ is only implemented on CPU tensors"
            ):
                nt.apply_(f)
            return

        # 在应用 apply_ 方法之前，复制并分离原始 nt._values 的值
        before = nt._values.clone().detach()

        # 调用 nt.apply_(f) 方法，对 nt 中的每个元素应用函数 f
        nt.apply_(f)
        # 计算期望的值，即应用函数 f 后的结果
        expected = f(before)
        # 断言 nt._values 的值等于预期值 expected
        self.assertEqual(expected, nt._values)
        # apply_ 方法应该在不添加到自动求导图中的情况下原地交换值
        self.assertIsNone(nt.grad)
        self.assertIsNone(nt._values.grad_fn)

    # 使用装饰器定义测试方法 test_compile_preserves_metadata_cache，接受设备和数据类型作为参数
    @dtypes(torch.float64, torch.float32, torch.half)
    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    # 定义测试方法 test_compile_preserves_metadata_cache
    def test_compile_preserves_metadata_cache(self, device, dtype):
        # 生成一个随机的 NamedTensor，其中维度为 [4, None, 3, 16]
        nt = random_nt_from_dims(
            [4, None, 3, 16],
            device=device,
            dtype=dtype,
            layout=torch.jagged,
            requires_grad=True,
        )

        # 期望 _metadata_cache 中存储最小和最大序列长度
        cache = dict(nt._metadata_cache)

        # 使用 @torch.compile 装饰器定义函数 f，对输入的 NamedTensor nt 进行操作
        @torch.compile
        def f(nt):
            # 对输入 nt 进行维度转置操作
            q = nt.transpose(-3, -2)
            # 使用 scaled_dot_product_attention 函数进行注意力计算
            output = F.scaled_dot_product_attention(q, q, q).transpose(-3, -2)
            return output

        # 调用函数 f，并将结果存储在 output 中
        output = f(nt)
        # 对 output 进行反向传播，使用全一张量作为梯度
        output.backward(torch.ones_like(output))
        # 断言 output 的 _metadata_cache 与预期的 cache 相等
        self.assertEqual(output._metadata_cache, cache)

    # 使用装饰器定义测试方法 test_compile_preserves_metadata_cache，接受设备和数据类型作为参数
    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    def test_compile_with_dynamic_max_seq_len(self, device, dtype):
        # 定义一个测试函数，用于测试动态最大序列长度情况下的编译
        # shape (B, *, D)
        # 定义一个包含嵌套张量的对象nt，包含三个张量，每个张量的形状为 (B_i, *, D)，
        # 其中第一个张量有2行，第二个张量有3行，第三个张量有18行。
        nt = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),
                torch.randn(3, 5),
                torch.randn(18, 5),
            ],
            layout=torch.jagged,
        )

        # 定义另一个包含嵌套张量的对象nt2，形状与nt相同，但第三个张量有19行。
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),
                torch.randn(3, 5),
                torch.randn(19, 5),
            ],
            layout=torch.jagged,
        )

        def f(nt):
            # 定义一个函数f，参数为一个嵌套张量nt，返回一个与nt形状相同的全为1的张量，
            # 乘以nt的最大序列长度。
            # TODO: 当我们可以使用 @properties 时，替换为公共 API
            return torch.ones_like(nt) * nt._get_max_seqlen()

        # 对于动态性参数dynamic的三种情况进行循环测试
        for dynamic in [False, True, None]:
            # 调用_recompiles_for_inputs函数，检查在给定动态性参数的情况下，函数f在nt和nt2上的行为是否一致。
            self.assertFalse(_recompiles_for_inputs(f, (nt,), (nt2,), dynamic=dynamic))

    @dtypes(torch.float32)
    @skipIfTorchDynamo("Test compiles internally")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @skipCUDAIfRocm
    def test_compile_with_dynamic_min_seq_len(self, device, dtype):
        # 定义一个测试函数，用于测试动态最小序列长度情况下的编译
        # shape (B, *, D)
        # 定义一个包含嵌套张量的对象nt，包含三个张量，每个张量的形状为 (B_i, *, D)，
        # 其中第一个张量有7行，第二个张量有8行，第三个张量有9行。
        nt = torch.nested.nested_tensor(
            [
                torch.randn(7, 5),
                torch.randn(8, 5),
                torch.randn(9, 5),
            ],
            layout=torch.jagged,
        )

        # 定义另一个包含嵌套张量的对象nt2，形状与nt相同，但第一个张量有8行。
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(8, 5),
                torch.randn(9, 5),
                torch.randn(10, 5),
            ],
            layout=torch.jagged,
        )

        def f(nt):
            # 定义一个函数f，参数为一个嵌套张量nt，返回一个与nt形状相同的全为1的张量，
            # 乘以nt的最小序列长度。
            # TODO: 当我们可以使用 @properties 时，替换为公共 API
            return torch.ones_like(nt) * nt._get_min_seqlen()

        # 对于动态性参数dynamic的三种情况进行循环测试
        for dynamic in [False, True, None]:
            # 调用_recompiles_for_inputs函数，检查在给定动态性参数的情况下，函数f在nt和nt2上的行为是否一致。
            self.assertFalse(_recompiles_for_inputs(f, (nt,), (nt2,), dynamic=dynamic))
    # 定义一个测试方法，用于测试编译时的动态最大序列长度传播
    def test_compile_with_propagated_dynamic_max_seq_len(self, device, dtype):
        # shape (B, *, D)
        # max seq len: 18
        # 创建一个嵌套张量，包含三个子张量，布局为不规则布局（jagged），每个子张量形状为 (不定, 5)
        nt = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),   # 第一个子张量形状为 (2, 5)
                torch.randn(3, 5),   # 第二个子张量形状为 (3, 5)
                torch.randn(18, 5),  # 第三个子张量形状为 (18, 5)，设置最大序列长度为 18
            ],
            layout=torch.jagged,
        )

        # max seq len: 19
        # 创建另一个嵌套张量，形状和布局与上面类似，但是第三个子张量形状为 (19, 5)，设置最大序列长度为 19
        nt2 = torch.nested.nested_tensor(
            [
                torch.randn(2, 5),
                torch.randn(3, 5),
                torch.randn(19, 5),
            ],
            layout=torch.jagged,
        )

        # 定义一个函数 f，接受一个嵌套张量作为输入
        def f(nt):
            # 对输入的嵌套张量进行 sin 函数操作并加上 1
            nt2 = nt.sin() + 1
            # TODO: 当我们可以使用 @properties 时，替换为公共 API
            # 返回一个与输入形状相同的全为 1 的张量，乘以输入张量的最大序列长度
            return torch.ones_like(nt2) * nt2._get_max_seqlen()

        # 调用函数 f 对 nt 进行计算，得到参考结果 ref
        ref = f(nt)
        # 使用 torch.compile 对函数 f 进行编译，关闭动态计算，返回编译后的输出
        output = torch.compile(f, fullgraph=True, dynamic=False)(nt)
        # 使用断言检查编译后的输出是否与参考结果相等
        self.assertEqual(ref, output)

        # 遍历动态参数的不同设置
        for dynamic in [False, True, None]:
            # 使用 _recompiles_for_inputs 函数检查输入 nt 和 nt2 是否会导致函数 f 重新编译
            self.assertFalse(_recompiles_for_inputs(f, (nt,), (nt2,), dynamic=dynamic))

    # 使用不同的数据类型进行测试
    @dtypes(torch.float32, torch.double, torch.half)
    # 测试反向解绑操作
    def test_unbind_backward(self, device, dtype):
        # 创建一个嵌套张量，包含三个子张量，每个子张量形状为 (不定, 4)，设置 requires_grad 为 True
        nt = torch.nested.nested_tensor(
            [
                torch.randn(2, 4, device=device),
                torch.randn(5, 4, device=device),
                torch.randn(3, 4, device=device),
            ],
            layout=torch.jagged,
            requires_grad=True,
        )

        # 对嵌套张量进行解绑操作，分别得到子张量 a, b, c
        a, b, c = nt.unbind()
        # 对子张量 b 的所有元素求和并进行反向传播
        b.sum().backward()

        # 创建一个与 nt 形状相同的全零张量，并对其进行解绑操作，将第二个子张量的梯度增加 1.0
        expected_grad = torch.zeros_like(nt)
        expected_grad.unbind()[1].add_(1.0)
        # 使用 torch._dynamo.disable(self.assertEqual) 来禁用动态计算，检查 nt 的梯度是否与期望的梯度相等
        torch._dynamo.disable(self.assertEqual)(nt.grad, expected_grad)
# 实例化参数化测试，针对 TestNestedTensor 类
instantiate_parametrized_tests(TestNestedTensor)

# 实例化设备类型相关的测试，针对 TestNestedTensorDeviceType 类，使用全局变量
instantiate_device_type_tests(TestNestedTensorDeviceType, globals())

# 实例化设备类型相关的测试，针对 TestNestedTensorAutograd 类，使用全局变量
instantiate_device_type_tests(TestNestedTensorAutograd, globals())

# 实例化设备类型相关的测试，针对 TestNestedTensorSubclass 类，使用全局变量
instantiate_device_type_tests(TestNestedTensorSubclass, globals())

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```