# `.\pytorch\test\test_decomp.py`

```py
# Owner(s): ["module: decompositions"]

import functools  # 导入 functools 库，用于高阶函数和操作函数对象的工具

import itertools  # 导入 itertools 库，用于高效循环和迭代操作
import re  # 导入 re 模块，用于正则表达式的匹配和操作
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from collections import defaultdict  # 导入 defaultdict 类，用于创建默认字典
from functools import partial  # 导入 partial 函数，用于部分函数应用

import torch.autograd  # 导入 torch.autograd 模块，PyTorch 自动求导功能
from torch import Tensor  # 导入 Tensor 类型，PyTorch 张量类型
from torch._decomp import core_aten_decompositions, decomposition_table  # 导入核心的 ATen 分解表和分解函数
from torch._dispatch.python import enable_python_dispatcher  # 导入 enable_python_dispatcher 函数，启用 Python 调度器
from torch._ops import DispatchKey  # 导入 DispatchKey 类，PyTorch 分发键
from torch.testing import make_tensor  # 导入 make_tensor 函数，用于创建测试用张量
from torch.testing._internal.common_cuda import tf32_off  # 导入 tf32_off 函数，关闭 TF32 模式
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,  # 导入 instantiate_device_type_tests 函数，实例化设备类型测试
    onlyCPU,  # 导入 onlyCPU 修饰器，限制仅在 CPU 上运行测试
    onlyCUDA,  # 导入 onlyCUDA 修饰器，限制仅在 CUDA 上运行测试
    onlyNativeDeviceTypes,  # 导入 onlyNativeDeviceTypes 修饰器，限制仅在本机设备类型上运行测试
    ops,  # 导入 ops 模块，包含测试操作
)
from torch.testing._internal.common_methods_invocations import (
    op_db,  # 导入 op_db，包含操作数据库
    skip,  # 导入 skip 修饰器，跳过测试
    skipOps,  # 导入 skipOps 修饰器，跳过特定操作的测试
    xfail,  # 导入 xfail 修饰器，标记预期失败的测试
)
from torch.testing._internal.common_modules import module_db, modules  # 导入模块数据库和模块列表
from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,  # 导入 is_iterable_of_tensors 函数，检查是否为张量迭代对象
    IS_MACOS,  # 导入 IS_MACOS 常量，检查是否在 macOS 上运行
    run_tests,  # 导入 run_tests 函数，运行测试
    skipIfCrossRef,  # 导入 skipIfCrossRef 修饰器，如果存在交叉引用，则跳过测试
    skipIfTorchDynamo,  # 导入 skipIfTorchDynamo 修饰器，如果在 Torch Dynamo 模式下则跳过测试
    suppress_warnings,  # 导入 suppress_warnings 函数，抑制警告
    TEST_WITH_ASAN,  # 导入 TEST_WITH_ASAN 常量，检查是否使用地址安全性工具
    TEST_WITH_SLOW,  # 导入 TEST_WITH_SLOW 常量，检查是否运行慢速测试
    TestCase,  # 导入 TestCase 类，用于编写测试用例
    unMarkDynamoStrictTest,  # 导入 unMarkDynamoStrictTest 函数，取消严格测试标记
)
from torch.utils import _pytree as pytree  # 导入 _pytree 模块作为 pytree
from torch.utils._python_dispatch import TorchDispatchMode  # 导入 TorchDispatchMode 类，PyTorch 分发模式

from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten  # 导入 pytree 相关函数

aten = torch.ops.aten  # 设置 aten 变量为 torch.ops.aten 操作的别名


# TODO: this isn't going to work with non-aten namespaces
def overload_to_aten_name(op):
    return op._schema.name.split("::")[1]  # 将操作对象的名称转换为 ATen 操作名称


# All operators that can have decomp tests
decomposition_names = {
    overload_to_aten_name(k)
    for k in decomposition_table
    if isinstance(k, torch._ops.OpOverload)  # 获取可以进行分解测试的所有操作名称集合
}
core_decomposition_names = {
    overload_to_aten_name(k)
    for k in core_aten_decompositions()
    if isinstance(k, torch._ops.OpOverload)  # 获取核心 ATen 分解操作的名称集合
}
_decomp_test_ops = [
    op
    for op in op_db
    if op.aten_name in decomposition_names
    or op.aten_backward_name in decomposition_names  # 获取所有可进行分解测试的操作对象列表
]
_decomp_test_ops_core_autograd = [
    op
    for op in op_db
    if op.aten_name in core_decomposition_names and op.supports_autograd
    # 获取支持自动求导的核心 ATen 分解测试操作对象列表
]
_sdpa_op_info = [
    op for op in op_db if "scaled_dot_product_attention" in op.aten_name
    # 获取包含 "scaled_dot_product_attention" 的操作对象信息列表
]


def diff_arg(arg, requires_grad=True):
    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()

    if is_iterable_of_tensors(arg):
        if all(is_differentiable_arg(a) for a in arg):
            return True
        if all(not is_differentiable_arg(a) for a in arg):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)
    # 检查参数是否为可微分参数的函数


# Version of autograd.grad with some differences:
#   - pytree inputs is allowed (but leaves of the pytree have to all
#     be tensors)
#   - if an input is not used as part of derivatives, we will return a
#     zero-filled tensor for the result
def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    # 自动求导的版本，支持一些不同之处：
    #   - 允许 pytree 输入（但是 pytree 的叶子必须全部为张量）
    #   - 如果输入不作为导数的一部分，则返回填充零的张量作为结果
    # 将输入参数展平为一维数组，并返回展平后的结果及其结构描述
    inputs, inputs_spec = tree_flatten(inputs)
    # 从展平后的输入中筛选出需要梯度计算的部分
    diff_inputs = tuple(inp for inp in inputs if inp.requires_grad)
    # 如果梯度输出未提供，则从输出中筛选出需要梯度计算的部分
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        # 从输出和梯度输出中筛选出需要梯度计算的部分，组成元组列表
        diff_grad_outputs = [
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        ]
        # 如果没有找到需要计算梯度的输出，则设置为空元组
        if len(diff_grad_outputs) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            # 分离出需要计算梯度的输出和对应的梯度输出
            diff_outputs, grad_outputs = zip(*diff_grad_outputs)
    # 使用 PyTorch 的自动求导函数计算输入对应的梯度
    grad_inputs = torch.autograd.grad(
        diff_outputs,                    # 待计算梯度的输出
        diff_inputs,                     # 待计算梯度的输入
        grad_outputs,                    # 梯度输出
        retain_graph=retain_graph,       # 是否保留计算图
        create_graph=create_graph,       # 是否创建计算图
        allow_unused=True,               # 是否允许未使用的梯度
    )
    # 初始化结果列表
    result = []
    # 创建梯度输入的迭代器
    grad_inputs_iter = iter(grad_inputs)
    # 遍历所有输入参数
    for inp in inputs:
        # 如果当前参数需要计算梯度
        if inp.requires_grad:
            # 获取对应的梯度输入
            grad_input = next(grad_inputs_iter)
            # 如果梯度输入为空，则使用与输入相同形状的零张量作为结果
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                result.append(grad_input)
        else:
            # 如果当前参数不需要计算梯度，则使用与输入相同形状的零张量作为结果
            result.append(torch.zeros_like(inp))
    # 根据输入的结构描述，将结果列表还原为原始结构的输入参数
    return tree_unflatten(result, inputs_spec)
# 将输入值转换为元组，如果已经是元组则直接返回
def _as_tuple(val):
    if isinstance(val, tuple):
        return val
    return (val,)


# 对一个函数进行反向传播（vjp），但不创建新图
def ref_vjp_no_create(f, *primals):
    # 调用给定函数 f，传入参数 primals，并保存结果
    result = f(*primals)

    # 定义一个包装函数 wrapped，用于计算梯度
    def wrapped(cotangents):
        return _autograd_grad(
            _as_tuple(result),  # 结果转换为元组
            primals,            # 原始输入
            _as_tuple(cotangents),  # 梯度转换为元组
            create_graph=False,    # 不创建新的计算图
            retain_graph=True,     # 保留现有计算图
        )

    # 返回结果和计算梯度的函数 wrapped
    return result, wrapped


# 不同数据类型的默认相对误差（rtol）和绝对误差（atol）
dtype_precisions = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-4),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}


# 返回给定数据类型的默认相对误差（rtol）和绝对误差（atol）
def _getDefaultRtolAndAtol(dtype0, dtype1):
    # 获取给定数据类型的默认 rtol 和 atol
    rtol = max(
        dtype_precisions.get(dtype0, (0, 0))[0],  # 获取 dtype0 对应的 rtol，若不存在则返回默认值 (0, 0)
        dtype_precisions.get(dtype1, (0, 0))[0],  # 获取 dtype1 对应的 rtol，若不存在则返回默认值 (0, 0)
    )
    # 获取给定数据类型的默认 atol
    atol = max(
        dtype_precisions.get(dtype0, (0, 0))[1],  # 获取 dtype0 对应的 atol，若不存在则返回默认值 (0, 0)
        dtype_precisions.get(dtype1, (0, 0))[1],  # 获取 dtype1 对应的 atol，若不存在则返回默认值 (0, 0)
    )
    # 返回计算得到的 rtol 和 atol
    return rtol, atol


# 对比操作的原始张量 orig 和分解后的张量 decomp 的属性和形状
def op_assert_ref(test_case, op, test_dtype, i, orig, decomp, ref, args, kwargs):
    # 断言原始张量和分解后张量的数据类型相同
    assert orig.dtype == decomp.dtype, f"{i} Operation:  {op}"
    # 如果原始张量或分解后张量的元素个数为 0，则直接断言它们的元素个数相同并返回
    if orig.numel() == 0 or decomp.numel() == 0:
        assert orig.numel() == decomp.numel()
        return
    # 断言原始张量和分解后张量的形状相同
    assert orig.shape == decomp.shape, f"{i} Operation:  {op}"
    tol_table = {
        # 定义容差表，存储不同数据类型和操作的容差阈值
        (torch.bfloat16, torch.ops.aten.native_layer_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_layer_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_layer_norm_backward.default): 1e-3,
        (torch.bfloat16, torch.ops.aten.native_layer_norm_backward.default): 2e-2,
        (torch.bfloat16, torch.ops.aten.native_batch_norm.default): 1e-5,
        (torch.float16, torch.ops.aten.native_batch_norm.default): 1e-5,
        (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
        (torch.bfloat16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
        (torch.float16, torch.ops.aten._native_batch_norm_legit.default): 1e-5,
        (torch.float16, torch.ops.aten._native_batch_norm_legit.no_stats): 1e-5,
        (torch.bfloat16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
        (torch.float16, torch.ops.aten.linalg_vector_norm.default): 1e-4,
        (torch.bfloat16, torch.ops.aten.var_mean.correction): 5e-7,
        (torch.float16, torch.ops.aten.var_mean.correction): 5e-7,
        (torch.bfloat16, torch.ops.aten.var_mean.dim): 5e-7,
        (torch.float16, torch.ops.aten.var_mean.dim): 5e-7,
        (torch.float16, torch.ops.aten.nll_loss_forward.default): 1e-2,
        (torch.bfloat16, torch.ops.aten.nll_loss_forward.default): 1e-1,
        (torch.float16, torch.ops.aten.nll_loss2d_forward.default): 1e-2,
        (torch.bfloat16, torch.ops.aten.nll_loss2d_forward.default): 2e-1,
        (torch.float16, torch.ops.aten.hardswish.default): 2e-7,
        (torch.bfloat16, torch.ops.aten.hardswish.default): 2e-7,
        (torch.float16, torch.ops.aten.multi_margin_loss.default): 3e-2,
        (torch.bfloat16, torch.ops.aten.multi_margin_loss.default): 3e-2,
        (torch.float16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
        (torch.bfloat16, torch.ops.aten.multilabel_margin_loss_forward.default): 3e-2,
        # 定义一个例外情况，详细说明见链接 https://github.com/pytorch/pytorch/pull/96264
        (torch.float16, torch.ops.aten.mv.default): 1e-5,
    }
    
    # 如果参考值是浮点数类型
    if ref.is_floating_point():
        # 计算原始值与参考值之间的绝对差的最大值
        orig_diff = (orig - ref).abs().max()
        # 计算分解值与参考值之间的绝对差的最大值
        decomp_diff = (decomp - ref).abs().max()
        # 获取容差表中对应测试数据类型和操作的容差阈值，如果不存在则使用默认值 1e-7
        atol = tol_table.get((test_dtype, op), 1e-7)
        # 如果分解值的最大差异大于原始值的最大差异加上容差阈值
        if decomp_diff > orig_diff + atol:
            # 抛出运行时异常，显示关于分解操作的详细信息和差异
            raise RuntimeError(
                f"Difference from float64 is larger with decomposition {op.__name__}"
                f" than original on output {i}. Original max diff: {orig_diff}, Decomp max diff: {decomp_diff}\n"
                f"atol = {atol}\n"
                f"args = {args}\n"
                f"kwargs = {kwargs}"
            )
    # 如果参考值不是浮点数类型
    else:
        # 使用测试框架的断言，比较原始值和分解值是否相等，否则显示操作、参数和关键字参数信息
        test_case.assertEqual(
            orig, decomp, msg=f"{op.__name__}\nargs = {args}\nkwargs = {kwargs}"
        )
def op_assert_equal(test_case, op, test_dtype, orig, decomp, args, kwargs):
    # 使用测试框架的断言方法，比较原始张量和分解后张量的数据类型是否相同
    test_case.assertEqual(
        orig.dtype,
        decomp.dtype,
        f"Operation: {op}, orig.dtype: {orig.dtype}, decomp.dtype: {decomp.dtype}, {args}, {kwargs}",
    )
    # 在添加条目到容差表之前，请确保你的分解是正确的 :)
    # 容差表，包含了一系列的元组，每个元组包含了数据类型和操作的组合以及相应的容差值
    tol_table = {
        # 由于奇怪的 epsilon 行为，参见 https://github.com/pytorch/pytorch/issues/73161
        (torch.float32, torch.ops.aten.native_layer_norm.default): (1e-3, 1e-3),
        (torch.float32, torch.ops.aten.native_layer_norm_backward.default): (
            1e-3,
            1e-3,
        ),
        (torch.float64, torch.ops.aten.native_layer_norm.default): (1e-6, 1e-6),
        # 这超出了 CPU 上的默认容差，CUDA 上是正常的
        (torch.float32, torch.ops.aten.grid_sampler_2d.default): (7e-6, 3e-5),
        # 在 CUDA 上超出了容差，可能是由于 fma
        (torch.float32, torch.ops.aten.mv.default): (1e-5, 3e-5),
        (torch.complex64, torch.ops.aten.mv.default): (5e-5, 5e-5),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.vec): (1e-5, 5e-4),
        (torch.float64, torch.ops.aten.upsample_bicubic2d.default): (1e-5, 5e-4),
        # 这个分解太准确了。它计算所有内容都是 int64，因此有时会有偏差。参见
        # https://github.com/pytorch/pytorch/issues/81996
        # https://github.com/pytorch/pytorch/issues/82230
        (torch.int8, torch.ops.aten.linspace.default): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.default): (0, 1),
        (torch.int16, torch.ops.aten.linspace.default): (0, 1),
        (torch.int32, torch.ops.aten.linspace.default): (0, 1),
        (torch.int64, torch.ops.aten.linspace.default): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Tensor_Tensor): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Tensor_Scalar): (0, 1),
        (torch.int8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.uint8, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int16, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int32, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
        (torch.int64, torch.ops.aten.linspace.Scalar_Tensor): (0, 1),
    }
    # 如果在容差表中找到了对应的数据类型和操作的条目，则获取相应的相对误差和绝对误差
    if (decomp.dtype, op) in tol_table:
        rtol, atol = tol_table[(decomp.dtype, op)]
    # 否则，获取默认的相对误差和绝对误差
    rtol, atol = _getDefaultRtolAndAtol(orig.dtype, decomp.dtype)
    # 使用测试框架的断言方法进行比较
    test_case.assertEqual(
        orig,  # 原始数据
        decomp,  # 解压缩后的数据
        rtol=rtol,  # 相对误差容差
        atol=atol,  # 绝对误差容差
        msg=f"{op.__name__}\nargs = {args}\nkwargs = {kwargs}",  # 错误消息格式化字符串
    )
# 定义函数 normalize_op_input_output2，接受以下参数：
# - f：原始函数
# - args：位置参数的元组
# - kwargs：关键字参数的字典
# - output_process_fn_grad：处理函数的梯度（可选）
# - requires_grad：是否需要梯度计算（默认为 True）
def normalize_op_input_output2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=True
):
    # 将 args 展平，并记录其结构到 args_spec 中
    flat_args, args_spec = tree_flatten(args)
    # 找出需要梯度的参数在 flat_args 中的索引
    diff_argnums = tuple(
        i
        for i, arg in enumerate(flat_args)
        if diff_arg(arg, requires_grad=requires_grad)
    )
    # 断言至少有一个参数需要梯度
    assert len(diff_argnums) > 0
    # 从 flat_args 中提取需要梯度的参数，组成 primals 元组
    primals = tuple(flat_args[i] for i in diff_argnums)

    # 定义一个包装函数 wrapped，用于接受 primals 作为位置参数
    @functools.wraps(f)
    def wrapped(*primals):
        # 复制 flat_args 到 _args
        _args = list(flat_args)
        # 将 primals 的值填充回 _args 中对应的位置
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        # 恢复 _args 的结构到原始的 args_spec 结构
        _args = tree_unflatten(_args, args_spec)
        # 调用原始函数 f，并传入 _args 和 kwargs
        result = f(*_args, **kwargs)
        # 如果有 output_process_fn_grad 函数，则对结果进行处理
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        # 如果结果是元组，则检查其中的 Tensor 是否为浮点数或复数类型
        if isinstance(result, tuple):
            # TODO 应检查整数输出是否也符合要求
            result = tuple(
                r
                for r in result
                if isinstance(r, Tensor) and (r.is_floating_point() or r.is_complex())
            )
            # 断言结果元组中至少有一个元素
            assert len(result) > 0
        # 返回处理后的结果
        return result

    # 返回 wrapped 函数和 primals 元组
    return wrapped, primals


# NB: 这也将提升 dtype 参数
# TODO: 正确处理复数类型
def upcast_tensor(x, dtype=torch.float32):
    # 如果 x 是 Tensor 类型且其 dtype 是浮点数类型
    if isinstance(x, Tensor) and x.dtype.is_floating_point:
        # 将 x 转换为指定的 dtype 类型
        return x.to(dtype=dtype)
    # 如果 x 是 torch.dtype 类型且其值在指定的浮点数类型中
    elif isinstance(x, torch.dtype) and x in [
        torch.float16,
        torch.bfloat16,
        torch.float,
    ]:
        # 返回指定的 dtype
        return dtype
    else:
        # 其他情况下直接返回 x
        return x


# 定义函数 normalize_op_input_output，接受以下参数：
# - f：原始函数
# - sample：包含 input、args 和 kwargs 的对象
# - requires_grad：是否需要梯度计算（默认为 True）
def normalize_op_input_output(f, sample, requires_grad=True):
    # 构造参数 args，包括 sample.input 和 sample.args
    args = tuple([sample.input] + list(sample.args))
    # 调用 normalize_op_input_output2 函数处理 f 和 args，返回结果
    return normalize_op_input_output2(
        f,
        args,
        sample.kwargs,
        sample.output_process_fn_grad,
        requires_grad=requires_grad,
    )


# 定义 CROSS_REF_EXCLUDE_SET 集合，包含以下元组：
# - 第一个元素是字符串 "cuda"
# - 第二个元素是 torch.bfloat16 类型
# - 第三个元素是字符串 "nn.functional.bilinear"
# 每个元组表示需要排除的情况，对应特定的错误或功能
CROSS_REF_EXCLUDE_SET = {
    # 当调用 `cublasGemmStridedBatchedExFix(handle, opa, opb, (int)m, (int)n, (int)k,
    # (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea, b, CUDA_R_16BF,
    # (int)ldb, strideb, (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec,
    # (int)num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)` 时出现 CUBLAS_STATUS_NOT_SUPPORTED 错误
    ("cuda", torch.bfloat16, "nn.functional.bilinear"),
    # 随机性
    (None, None, "special.ndtr"),  # aten.special_ndtr 未被拆解
    (None, None, "new_empty"),
    (None, None, "empty_like"),
    (None, None, "empty"),
    # AssertionError: False 不是 true : aten.item 未被拆解，出现了对 aten._local_scalar_dense.default 的调用。
    (None, None, "item"),
    # 这是唯一一个在 Python API 中没有对应的原位操作
    # 其 OpInfo 错误地将其注册为 `torch.zero_(x.clone())`。
    (None, None, "zero_"),
    # 对这里发生的情况一无所知
}
    # logsumexp.default 在递归测试中使用 torch.tensor(-math.inf) 和空列表 args 时失败
    # 但在本地测试和 logsumexp 测试中似乎通过了
    (None, torch.float32, "masked.logsumexp"),

    # exp_vml_cpu 在 Half 数据类型上未实现
    (None, torch.float16, "signal.windows.exponential"),

    # sin_vml_cpu 在 Half 数据类型上未实现
    (torch.cpu, torch.float16, "signal.windows.cosine"),

    # CompositeAutogradImplicit
    # 参考 https://github.com/pytorch/pytorch/issues/81669
    (None, None, "nn.functional.relu6"),

    # 这个分解在 autograd 之前运行
    (None, None, "nn.functional.rrelu"),

    # meshgrid 函数
    (None, None, "meshgrid"),

    # Decomposition registered as Autograd
    (None, None, "nn.functional.hardshrink"),
    (None, None, "nn.functional.softshrink"),

    # diag 函数未进行分解（仅注册了 diag_out 的分解，torch.diag 是 CompImplicit）
    (None, None, "diag"),

    # _softmax_backward_data 的 CPU 内核对于 bfloat16 始终返回 grad_input 作为 float32
    ("cpu", torch.bfloat16, "_softmax_backward_data"),

    # native_batch_norm 仅在 Python 调度器打开时是隐式的（否则为非组合）
    (None, None, "native_batch_norm"),

    # _upsample_bilinear2d_aa 函数
    (None, None, "_upsample_bilinear2d_aa"),

    # aten.empty_strided 未进行分解
    (None, None, "empty_strided"),
}

# 定义一个集合，用于存储不进行反向传播交叉引用的情况
CROSS_REF_BACKWARD_EXCLUDE_SET = {
    # 不精确的反向分解公式
    ("cpu", torch.bfloat16, "nn.functional.hardswish"),
    ("cuda", torch.float16, "nn.functional.cross_entropy"),
}

# 创建空集合，用于存储所有分解过的操作
all_decomposed = set()

# 创建默认字典，用于计算每个操作被调用的次数
all_called = defaultdict(int)

# 用于测试覆盖率的有用片段
"""
import atexit
def check_coverage():
    print("missing coverage:")
    print("\n".join(map(str, decomposition_table.keys() - all_decomposed)))
atexit.register(check_coverage)
"""

# 用于 Horace 创建 Google 表格的有用片段
"""
import atexit
def dump_ops():
    with open('run_ops.txt', 'w') as f, open('count_ops.txt', 'w') as g:
        for op, count in sorted(all_called.items(), key=lambda x: x[0].__name__):
            f.write(f'{op.__name__}\n')
            g.write(f'{count}\n')
    with open('run_decompositions.txt', 'w') as f:
        for op in sorted([i.__name__ for i in all_decomposed]):
            f.write(f'{op}\n')

atexit.register(dump_ops)
"""


def any_unsupported(args, kwargs):
    # 内部函数：测试是否存在不受支持的操作
    def test_unsupported(t):
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            # 检查是否有未正确处理的张量类型
            return any(
                [
                    t.is_sparse_csr,
                    t.is_sparse,
                    t.is_mkldnn,
                    t.is_quantized,
                    t.is_nested,
                    torch._is_functional_tensor(t),
                ]
            )
        elif torch.overrides.is_tensor_like(t):
            # Tensor-like 子类一般会改变分解行为，因此在这种情况下绕过测试
            return True
        else:
            return False

    # 扁平化参数列表
    flat_args = pytree.arg_tree_leaves(*args, **kwargs)
    # 返回是否存在不受支持的操作
    return any(test_unsupported(x) for x in flat_args)


# 定义一个集合，包含核心反向传播失败的情况
core_backward_failures = {
    skip("_softmax_backward_data"),  # 缓慢：超时 360 秒后失败
    xfail("addcdiv"),
    skip("addcmul"),  # 缓慢：超时 360 秒后失败
    skip("deg2rad"),  # 缓慢：超时 360 秒后失败
    skip("diag_embed"),  # 缓慢：超时 360 秒后失败
    skip("frac"),  # 缓慢：超时 360 秒后失败
    skip("grid_sampler_2d"),  # 缓慢：超时 360 秒后失败
    xfail("lerp"),
    skip("logaddexp"),  # 缓慢：超时 360 秒后失败
    skip("native_dropout_backward"),  # 缓慢：超时 360 秒后失败
    xfail("nn.functional.binary_cross_entropy_with_logits"),
    skip("nn.functional.glu"),  # 缓慢：超时 360 秒后失败
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.softshrink"),
    skip("nn.functional.unfold"),  # 缓慢：超时 360 秒后失败
    xfail("norm"),
    xfail("norm", "fro"),
    xfail("norm", "inf"),
    xfail("norm", "nuc"),
    skip("rad2deg"),  # 缓慢：超时 360 秒后失败
    skip("renorm"),  # 缓慢：超时 360 秒后失败
}
    skip("rot90"),  # 跳过测试函数 "rot90"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    skip("rsub"),  # 跳过测试函数 "rsub"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    skip("sgn"),  # 跳过测试函数 "sgn"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    skip("special.xlog1py"),  # 跳过测试函数 "special.xlog1py"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    xfail("stack"),  # 标记测试函数 "stack" 为预期失败，预期失败的原因未明确注明
    skip("tril"),  # 跳过测试函数 "tril"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    skip("triu"),  # 跳过测试函数 "triu"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    skip("unfold_copy"),  # 跳过测试函数 "unfold_copy"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    skip("xlogy"),  # 跳过测试函数 "xlogy"，原因是执行时间长，在 --timeout=360 秒时会超时失败
    xfail("zero_"),  # 标记测试函数 "zero_" 为预期失败，预期失败的原因未明确注明
}
# 如果不是在使用慢速测试，更新core_backward_failures字典
if not TEST_WITH_SLOW:
    core_backward_failures.update(
        {
            skip("addr"),  # slow: takes 46 sec on A100
            skip("baddbmm"),  # slow: takes 800+ sec on A100
            skip("clamp_min"),  # slow: takes 800 sec on A100
            skip("clamp_max"),  # slow: takes 800 sec on A100
            skip("logit"),  # slow: takes 44 sec on A100
            skip("nn.functional.hardswish"),  # slow: takes 60 sec on A100
            skip("std_mean"),  # slow: takes 170 sec on A100
            skip("split", variant_name="list_args"),  # slow: takes 118 sec on A100
            skip("transpose"),  # slow: takes 50 sec on A100
            skip("unbind"),  # slow: takes 70 sec on A100
            skip("unsafe_split"),  # slow: takes 49 sec on A100
        }
    )

# 创建comprehensive_failures字典
comprehensive_failures = {
    xfail(
        "nn.functional.interpolate", "bilinear", dtypes=(torch.uint8,)
    ),  # off by one error
    xfail(
        "nn.functional.interpolate", "bicubic", dtypes=(torch.uint8,)
    ),  # off by one error
    xfail(
        "nn.functional.upsample_bilinear", "", dtypes=(torch.uint8,)
    ),  # off by one error
}

# 测试类标记为unMarkDynamoStrictTest
@unMarkDynamoStrictTest
# TestCase类的子类，设置长消息模式
class TestDecomp(TestCase):
    longMessage = True

    # 注意事项：这个测试与test_comprehensive重叠，但只运行在已分解的内容上，因此速度更快
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops)
    # 定义测试函数test_quick，接受device、dtype和op参数
    def test_quick(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=False)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipOps("TestDecomp", "test_quick_core_backward", core_backward_failures)
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @suppress_warnings
    @ops(_decomp_test_ops_core_autograd, allowed_dtypes=(torch.float64,))
    # 定义测试函数test_quick_core_backward，接受device、dtype和op参数
    def test_quick_core_backward(self, device, dtype, op):
        # 对于op.sample_inputs产生的具有梯度的输入样本，迭代处理
        for sample_input in op.sample_inputs(device, dtype, requires_grad=True):
            aten_name = op.decomp_aten_name or op.aten_name
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            func = partial(op.get_op(), **kwargs)
            # 使用enable_python_dispatcher进入self.DecompCrossRefMode上下文
            with self.DecompCrossRefMode(
                self, self.precision, self.rel_tol, dtype, run_all=False
            ) as mode, enable_python_dispatcher():
                # 使用torch.autograd.gradcheck检查函数func的梯度
                torch.autograd.gradcheck(func, args)
            # 检查已分解的aten_name在mode上下文中的情况
            self.check_decomposed(aten_name, mode)

    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    @skipOps("TestDecomp", "test_comprehensive", comprehensive_failures)
    @suppress_warnings
    @ops(op_db)
    # 定义测试函数test_comprehensive，接受device、dtype和op参数
    def test_comprehensive(self, device, dtype, op):
        self.do_cross_ref(device, dtype, op, run_all=True)
    # 定义一个测试方法，用于测试在指定设备上生成特定大小和数据类型的张量
    def test_uniform(self, device):
        size = (2, 3, 4, 5)
        dtype = torch.float32
        x = make_tensor(size, dtype=dtype, device=device)  # 调用自定义函数生成指定设备上的张量
        low = 0.3
        high = 0.9

        torch.manual_seed(123)  # 设置随机种子
        ref = torch.ops.aten.uniform(x, low, high)  # 调用内置函数计算均匀分布
        torch.manual_seed(123)  # 重新设置相同的随机种子
        res = torch._decomp.decompositions.uniform(x, low=low, high=high)  # 调用自定义分解函数计算均匀分布
        self.assertEqual(ref, res)  # 断言结果一致

    # 定义一个测试方法，验证在指定设备上的广播索引复制功能
    def test_broadcasting_index_copy(self, device):
        x = torch.zeros([1, 10], device=device)  # 创建全零张量
        xs = torch.ones([2, 10], device=device)  # 创建全一张量

        def index_copy(xs, x):
            torch._decomp.decompositions.index_copy_(  # 调用自定义分解函数进行索引复制
                xs, 0, torch.tensor(0).to(device), x
            )

        index_copy(xs, x)  # 调用索引复制函数

        xs_two = torch.ones([2, 10], device=device)  # 创建全一张量
        xs_two[0] = x  # 将第一个张量替换为 x

        self.assertEqual(xs, xs_two)  # 断言两个张量相等

    # 定义一个测试方法，验证带噪声的随机整流线性单元（RReLU）的行为
    def test_rrelu_with_noise(self, device):
        # rrelu_with_noise 的行为依赖于输入元素是否 <= 0 和是否处于训练模式，覆盖所有情况：
        dtype = torch.float64
        x = torch.tensor(
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
            dtype=dtype,
            device=device,
        )
        lower = 1.0
        upper = 4.0
        training = False

        torch.manual_seed(123)  # 设置随机种子
        noise_ref = torch.zeros(x.shape, dtype=dtype, device=device)  # 创建全零张量作为噪声参考
        ref = torch.ops.aten.rrelu_with_noise(x, noise_ref, lower, upper, training)  # 调用内置函数计算 RReLU
        torch.manual_seed(123)  # 重新设置相同的随机种子
        noise_res = torch.zeros(x.shape, dtype=dtype, device=device)  # 创建全零张量作为噪声结果
        res = torch._decomp.decompositions.rrelu_with_noise(  # 调用自定义分解函数计算 RReLU
            x,
            noise_res,
            lower,
            upper,
            training,
        )
        self.assertEqual(ref, res)  # 断言 RReLU 结果一致
        self.assertEqual(noise_ref, noise_res)  # 断言噪声结果一致

        # 现在测试训练模式下的情况：
        training = True

        torch.manual_seed(123)  # 设置随机种子
        noise_ref = torch.zeros(x.shape, dtype=dtype, device=device)  # 创建全零张量作为噪声参考
        ref = torch.ops.aten.rrelu_with_noise(x, noise_ref, lower, upper, training)  # 调用内置函数计算 RReLU
        torch.manual_seed(123)  # 重新设置相同的随机种子
        noise_res = torch.zeros(x.shape, dtype=dtype, device=device)  # 创建全零张量作为噪声结果
        res = torch._decomp.decompositions.rrelu_with_noise(  # 调用自定义分解函数计算 RReLU
            x,
            noise_res,
            lower,
            upper,
            training,
        )
        self.assertEqual(ref, res)  # 断言 RReLU 结果一致
        self.assertEqual(noise_ref, noise_res)  # 断言噪声结果一致

    # 使用 ASAN 时跳过测试，只测试循环神经网络（RNN）、长短期记忆网络（LSTM）和门控循环单元（GRU）模块
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @suppress_warnings
    @tf32_off()
    # 仅测试 RNN，因为我们对它们有 py 分发器分解
    @modules(
        filter(
            lambda m: m.module_cls in (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU),
            module_db,
        )
    )
    # 测试RNN解构模块的功能，接收设备、数据类型、模块信息和训练状态作为参数
    def test_rnn_decomp_module(self, device, dtype, module_info, training):
        # 获取模块类
        module_cls = module_info.module_cls
        # 生成模块输入数据
        module_inputs = module_info.module_inputs_func(
            module_info,
            device=device,
            dtype=dtype,
            requires_grad=True,
            training=training,
        )
        # 遍历模块输入数据
        for module_input in module_inputs:
            # 如果模块的前向输入为空，则跳过
            if module_input.forward_input is None:
                continue
            # 获取构造函数的参数和关键字参数
            args, kwargs = (
                module_input.constructor_input.args,
                module_input.constructor_input.kwargs,
            )
            # 使用参数和关键字参数创建模块实例
            m = module_cls(*args, **kwargs)
            # 将模块实例移动到指定的设备和数据类型上
            m.to(device).to(dtype)

            # 获取前向输入的参数和关键字参数
            args, kwargs = (
                module_input.forward_input.args,
                module_input.forward_input.kwargs,
            )
            # 在特定模式下运行模块的前向计算，用于交叉引用检查
            with self.DecompCrossRefMode(
                self, self.precision, self.rel_tol, dtype, run_all=True
            ), enable_python_dispatcher():
                # 执行模块的前向计算
                decomp_out = m(*args, **kwargs)

            # 再次执行模块的前向计算，获取非解构版本的输出
            non_decomp_out = m(*args, **kwargs)
            # 断言解构版本和非解构版本的输出应该相等
            self.assertEqual(decomp_out, non_decomp_out)

    # 测试批归一化层权重和偏置的扁平化操作
    def test_batch_norm_unflatten_weight_bias(self, device):
        # 定义输入张量的形状
        shape = (1, 3, 2, 2)
        # 生成随机输入张量
        input = torch.randn(shape, device=device)
        # 生成随机权重张量
        weight = torch.randn((3, 1, 1, 1), device=device)
        # 生成随机偏置张量
        bias = torch.randn(3, device=device)
        # 生成随机均值张量
        mean = torch.randn(3, device=device)
        # 生成随机方差张量
        var = torch.randn(3, device=device)
        # 调用PyTorch内部的批归一化函数
        res = torch._decomp.decompositions.native_batch_norm(
            input, weight, bias, mean, var, False, 1, 1e-05
        )
        # 断言结果张量的形状应该与预期相同
        self.assertEqual(shape, res[0].shape)

    # 测试创建张量序列的图形化表示
    def test_arange_graph(self, device):
        # 导入生成张量的函数
        from torch.fx.experimental.proxy_tensor import make_fx

        # 定义生成张量序列的函数
        def func(x, start):
            # 获取张量的长度
            le = x.shape[-1]
            # 根据起始索引创建张量序列
            if start is None:
                a = torch.arange(le, dtype=torch.float32, device=x.device)
            else:
                a = torch.arange(start, le, dtype=torch.float32, device=x.device)
            return a

        # 定义用于匹配删除设备和requires_grad信息的模式
        pattern = r", device = device\(.+\), requires_grad = False"

        # 使用函数生成张量序列的图形化表示
        cfunc = make_fx(func, decomposition_table=decomposition_table)
        # 执行生成的图形化表示的代码
        fx_g = cfunc(torch.rand(10, device=device), None)
        # 获取生成的图形化表示的代码并去除设备和requires_grad信息
        fx_g_code = fx_g.code.strip()
        # 删除设备和requires_grad信息
        fx_g_code = re.sub(pattern, "", fx_g_code)
        # 断言生成的图形化表示的代码应该与预期的内联代码相同
        self.assertExpectedInline(
            fx_g_code,
            """\
    # 定义一个名为 forward 的方法，接受参数 self, x_1, start_1
    def forward(self, x_1, start_1):
        # 使用 torch.ops.prims.iota.default 方法生成一个整数序列 iota，从 0 开始，步长为 1，长度为 10，数据类型为 torch.int64
        iota = torch.ops.prims.iota.default(10, start=0, step=1, dtype=torch.int64)
        # 使用 torch.ops.prims.mul.default 方法将整数序列 iota 中的每个元素与标量 1 相乘，并将结果赋给 mul 变量；然后将 iota 变量置为 None
        mul = torch.ops.prims.mul.default(iota, 1);  iota = None
        # 使用 torch.ops.prims.add.default 方法将 mul 变量中的每个元素与标量 0 相加，并将结果赋给 add 变量；然后将 mul 变量置为 None
        add = torch.ops.prims.add.default(mul, 0);  mul = None
        # 使用 torch.ops.prims.convert_element_type.default 方法将 add 变量中的每个元素转换为 torch.float32 类型，并将结果赋给 convert_element_type 变量；然后将 add 变量置为 None
        convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        # 返回 convert_element_type 变量作为方法的结果
        return convert_element_type""",
        )

        # 使用 cfunc 方法调用 forward 方法，传入参数 torch.rand(10, device=device), 1，并将结果保存到 fx_g 变量中
        fx_g = cfunc(torch.rand(10, device=device), 1)
        # 获取 fx_g.code 属性，并去除其两侧的空白字符，并将结果保存到 fx_g_code 变量中
        fx_g_code = fx_g.code.strip()
        # 使用正则表达式替换模式 pattern 在 fx_g_code 中的内容，将其替换为空字符串，并将结果保存回 fx_g_code 变量
        # Remove device and requires_grad
        fx_g_code = re.sub(pattern, "", fx_g_code)
        # 使用 self.assertExpectedInline 方法断言 fx_g_code 的内容与预期的字符串匹配
        self.assertExpectedInline(
            fx_g_code,
            """\
def forward(self, x_1, start_1):
    iota = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64)
    mul = torch.ops.prims.mul.default(iota, 1);  iota = None
    add = torch.ops.prims.add.default(mul, 1);  mul = None
    convert_element_type = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
    return convert_element_type""",
        )

    # 定义一个名为 test_masked_fill 的方法，接受参数 self, device
    def test_masked_fill(self, device):
        # 导入 make_fx 方法和 decomposition_table 对象
        from torch.fx.experimental.proxy_tensor import make_fx

        # 如果设备不是 "xpu", "cuda" 或者 torch._C._get_privateuse1_backend_name()，则跳过测试
        if torch.device(device).type not in [
            "xpu",
            "cuda",
            torch._C._get_privateuse1_backend_name(),
        ]:
            self.skipTest("only runs on XPU and CUDA and PrivateUse1.")

        # 定义一个名为 func 的方法，接受参数 scores, mask, value，返回 scores 中根据 mask 进行填充的结果
        def func(scores, mask, value):
            return scores.masked_fill(mask, value)

        # 创建 tensor 对象 scores_t, mask_t, value_t，并将它们保存到相应的变量中
        scores_t = torch.tensor([1, 2, 3, 4], device=device)
        mask_t = torch.tensor([True, True, True, True], device=device)
        value_t = torch.tensor(0, dtype=scores_t.dtype)
        # 使用 make_fx 方法将 func 方法转换为 cfunc 对象，传入 decomposition_table 作为参数
        cfunc = make_fx(func, decomposition_table=decomposition_table)
        # 使用 cfunc 调用 func 方法，传入 scores_t, mask_t, value_t 作为参数，并将结果保存到 fx_g 变量中
        fx_g = cfunc(scores_t, mask_t, value_t)
        # 获取 fx_g.code 属性，并去除其两侧的空白字符，并将结果保存到 fx_g_code 变量中
        fx_g_code = fx_g.code.strip()
        # 使用 self.assertExpectedInline 方法断言 fx_g_code 的内容与预期的字符串匹配
        self.assertExpectedInline(
            fx_g_code,
            """\
def forward(self, scores_1, mask_1, value_1):
    where = torch.ops.prims.where.default(mask_1, value_1, scores_1);  mask_1 = value_1 = scores_1 = None
    return where""",
        )

    # 定义一个名为 check_decomposed 的方法，接受参数 self, aten_name, mode
    def check_decomposed(self, aten_name, mode):
        # 断言 mode.decomposed 中是否有任何一个函数的名字与给定的 aten_name 相匹配
        self.assertTrue(
            any(overload_to_aten_name(c) == aten_name for c in mode.decomposed),
            msg=(
                f"aten.{aten_name} was not decomposed, saw calls for: "
                f"{', '.join(map(str, list(mode.called)))}. If your op is  "
                f"CompositeImplicitAutograd you should skip this test "
                f"by updating CROSS_REF_EXCLUDE_SET."
            ),
        )

    # 使用 skipIfTorchDynamo 装饰器，跳过该测试在 TorchDynamo 环境下的执行
    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    instantiate_device_type_tests(TestDecomp, globals())

# 定义一个名为 DecompOneOffTests 的类，继承自 TestCase 类
class DecompOneOffTests(TestCase):
    # 使用 unittest.skipIf 装饰器，如果在 ASAN 下则跳过该测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 使用 onlyNativeDeviceTypes 装饰器，限定只在本地设备上执行测试
    @onlyNativeDeviceTypes
    # 使用 skipIfCrossRef 装饰器，如果不满足特定条件则跳过该测试
    @skipIfCrossRef
    # 测试连续的 softmax 函数在给定设备上的实现
    def test_contiguous_softmax(self, device):
        # 定义张量的大小
        size = (2, 4, 3, 3)
        # 定义张量的步长
        stride = (9, 18, 3, 1)
        # 定义张量的数据类型
        dtype = torch.float32

        # 生成指定大小和类型的随机张量，并移动到指定设备上
        x = torch.randn(size, dtype=dtype, device=device)
        # 使用指定的大小和步长创建一个视图张量
        x = torch.as_strided(x, size, stride)

        # 调用底层的 ATen 函数计算 softmax
        ref = torch.ops.aten._softmax(x, -1, False)
        # 调用底层的分解模块函数计算 softmax
        res = torch._decomp.decompositions._softmax(x, -1, False)
        # 断言两个张量的步长是否相等
        self.assertEqual(ref.stride(), res.stride())

    # 在特定条件下跳过测试，条件为 ASAN 开启时
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    # 测试连续的 log_softmax 函数在给定设备上的实现
    def test_contiguous_log_softmax(self, device):
        # 定义张量的大小
        size = (2, 4, 3, 3)
        # 定义张量的步长
        stride = (9, 18, 3, 1)
        # 定义张量的数据类型
        dtype = torch.float32
        # 生成指定大小和类型的随机张量，并移动到指定设备上
        x = torch.randn(size, dtype=dtype, device=device)
        # 使用指定的大小和步长创建一个视图张量
        x = torch.as_strided(x, size, stride)

        # 调用底层的 ATen 函数计算 log_softmax
        ref = torch.ops.aten._log_softmax(x, -1, False)
        # 调用底层的分解模块函数计算 log_softmax
        res = torch._decomp.decompositions._log_softmax(x, -1, False)
        # 断言两个张量的步长是否相等
        self.assertEqual(ref.stride(), res.stride())

    # 在仅 CUDA 设备上运行的测试函数
    @onlyCUDA
    # 测试指数函数在非无穷范围内的行为
    def test_exponential_non_inf(self, device):
        # 在指定设备上创建空张量
        inp = torch.empty((4, 400, 256), device=device)

        # 保存随机数生成器的状态，并使用 inp 上的指数函数就地修改
        with torch._dynamo.utils.preserve_rng_state():
            exp_ref = inp.exponential_()
        # 使用 ref 模块中的指数函数计算 inp 的指数
        exp = torch._refs.exponential(inp)

        # 断言两个张量的值是否相等
        self.assertEqual(exp, exp_ref)
        # 断言张量中是否没有包含无穷值
        self.assertFalse(exp.isinf().any())

    # 在特定条件下跳过测试，条件为 ASAN 开启时
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @skipIfCrossRef
    @onlyCUDA
    # 测试在自动混合精度模式下批标准化反向传播的行为
    def test_amp_batch_norm_backward(self):
        # 指定设备为 CUDA
        device = "cuda"
        # 创建随机梯度输出张量
        grad_out = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        # 创建随机输入张量
        x = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
        # 创建随机权重张量
        weight = torch.randn((2,), dtype=torch.float32, device=device)
        # 创建随机 running mean 张量
        rmean = torch.randn((2,), dtype=torch.float32, device=device)
        # 创建随机 running variance 张量
        rvar = torch.randn((2,), dtype=torch.float32, device=device)
        # 创建空的 mean 张量
        mean = torch.randn((0,), dtype=torch.float32, device=device)

        # 调用底层的 ATen 函数计算批标准化反向传播
        ref = torch.ops.aten.native_batch_norm_backward(
            grad_out,
            x,
            weight,
            rmean,
            rvar,
            mean,
            mean,
            False,
            1e-05,
            [True, True, True],
        )
        # 调用底层的分解模块函数计算批标准化反向传播
        res = torch._decomp.decompositions.native_batch_norm_backward(
            grad_out,
            x,
            weight,
            rmean,
            rvar,
            mean,
            mean,
            False,
            1e-05,
            [True, True, True],
        )
        # 逐一断言每对张量的步长和数据类型是否相等
        for a, b in zip(ref, res):
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.dtype, b.dtype)

    # 在特定条件下跳过测试，条件为 ASAN 开启时
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @skipIfCrossRef
    # 定义测试 ELU 激活函数的反向传播的方法，接受一个设备参数
    def test_elu_backward(self, device):
        # 设置张量大小为 (2, 4, 3, 3)
        size = (2, 4, 3, 3)
        # 设置张量数据类型为 torch.float32
        dtype = torch.float32
        # 生成指定大小的随机梯度张量，设备为指定设备
        grad_out = torch.randn(size, dtype=dtype, device=device)
        # 生成指定大小的随机输出张量，设备为指定设备
        out = torch.randn(size, dtype=dtype, device=device)

        # 调用 ATen 库中的 ELU 梯度计算函数
        ref = torch.ops.aten.elu_backward(grad_out, 1.0, 1, 1, True, out)
        # 调用自定义的 ELU 梯度计算函数
        res = torch._decomp.decompositions.elu_backward(grad_out, 1.0, 1, 1, True, out)
        # 断言两个结果是否相等
        self.assertEqual(ref, res)

    # 跳过 ASAN 环境下的测试用例
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 只在本地设备上执行测试
    @onlyNativeDeviceTypes
    # 跳过交叉引用的测试
    @skipIfCrossRef
    # 测试阈值函数反向传播时的数据类型
    def test_threshold_backward_dtype(self, device):
        # 生成指定设备上的随机梯度张量
        grad = torch.randint(10, (4,), device=device)
        # 生成指定设备上的随机输入张量
        input_tensor = torch.randint(10, (4,), device=device)

        # 调用 ATen 库中的阈值函数反向传播计算
        ref = torch.ops.aten.threshold_backward(grad, input_tensor, 1)
        # 调用自定义的阈值函数反向传播计算
        res = torch._decomp.decompositions.threshold_backward(grad, input_tensor, 1)
        # 断言两个结果的数据类型是否相等
        self.assertEqual(ref.dtype, res.dtype)

    # 跳过 ASAN 环境下的测试用例
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 只在 CPU 设备上执行测试
    @onlyCPU
    # 跳过交叉引用的测试
    @skipIfCrossRef
    # 测试权重归一化接口函数
    def test_weight_norm_interface(self, device):
        # 生成指定设备上的随机张量 g
        g = torch.randn((3, 10, 10), device=device)
        # 生成指定设备上的随机张量 v
        v = torch.randn((1, 1, 10), device=device)

        # 调用 ATen 库中的权重归一化接口函数
        ref = torch.ops.aten._weight_norm_interface(g, v, 2)
        # 调用自定义的权重归一化接口函数
        res = torch._decomp.decompositions._weight_norm_interface(g, v, 2)
        # 断言两个结果是否在数值上近似相等
        self.assertTrue(torch.allclose(ref[0], res[0]))
        self.assertTrue(torch.allclose(ref[1], res[1]))

        # 生成指定设备上的随机张量 inp 和 inp2
        inp = torch.rand([30, 10], device=device)
        inp2 = torch.rand([30, 1], device=device)

        # 断言 ATen 和自定义接口在给定输入上的输出是否相等
        self.assertEqual(
            torch.ops.aten._weight_norm_interface(inp, inp2),
            torch._decomp.decompositions._weight_norm_interface(inp, inp2),
        )

    # 跳过 ASAN 环境下的测试用例
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 只在 CPU 设备上执行测试
    @onlyCPU
    # 跳过交叉引用的测试
    @skipIfCrossRef
    # 跳过指定的操作测试
    @skipOps(
        "DecompOneOffTests",
        "test_sdpa",
        [
            xfail(
                "nn.functional.scaled_dot_product_attention",
                dtypes=[torch.half] + ([torch.bfloat16] if IS_MACOS else []),
            ),
        ],
    )
    # 使用 _sdpa_op_info 进行操作测试
    @ops(_sdpa_op_info)
    def test_sdpa(self, device, dtype, op):
        # SDPA doesn't support float16, this is aligned with aten/src/ATen/native/transformers/attention.cpp. If we
        # add support for float16 over there we should update this test as well.
        
        # 定义一个名为ScaledDotProductAttention的类，继承自torch.nn.Module，用于执行缩放点积注意力机制
        class ScaledDotProductAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 前向传播函数，接收查询、键、值张量，以及掩码和是否因果的参数
            def forward(
                self, query_layer, key_layer, value_layer, mask=None, is_causal=True
            ):
                # 使用给定的操作符op计算注意力输出
                attn_output = op(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
                return attn_output

        # 生成随机的查询、键、值张量，大小为(1, 128, 100, 64)，使用给定的设备和数据类型
        query_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        key_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        value_layer = torch.randn(1, 128, 100, 64, device=device, dtype=dtype)
        # 定义不同的掩码，一个为None，一个为全True的布尔张量
        masks = [None, torch.ones((1, 1, 100, 100), device=device, dtype=torch.bool)]

        # 获取当前数据类型的绝对误差和相对误差容限
        atol, rtol = dtype_precisions[dtype]

        # 遍历不同的掩码
        for mask in masks:
            # 判断是否为因果模式
            is_causal = mask is None
            # 创建一个ScaledDotProductAttention的实例
            attention = ScaledDotProductAttention()
            # 调用torch._decomp.decompositions.scaled_dot_product_flash_attention_for_cpu函数，执行分解注意力计算
            decomposed_res = (
                torch._decomp.decompositions.scaled_dot_product_flash_attention_for_cpu(
                    query_layer, key_layer, value_layer, 0.0, is_causal, attn_mask=mask
                )
            )
            # 使用给定的操作符op计算注意力输出
            eager_res = op(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

            # 使用assertTrue断言，验证两个结果是否在给定的误差容限内相等
            self.assertTrue(
                torch.allclose(decomposed_res[0], eager_res, atol=atol, rtol=rtol)
            )
# 调用函数以实例化设备类型测试，并将其应用到全局环境中
instantiate_device_type_tests(DecompOneOffTests, globals())

# 定义一个测试类，用于测试是否存在分解操作
class HasDecompTest(TestCase):
    # 在每个测试方法运行之前设置测试环境
    def setUp(self):
        super().setUp()
        # 设置最大差异为无限制
        self.maxDiff = None

    # 静态方法：检查操作是否出现在追踪中
    @staticmethod
    def _can_appear_in_trace(op: torch._ops.OpOverload) -> bool:
        # 检查操作是否有张量类型的参数
        has_tensor_arg = any(
            "Tensor" in str(a.type)
            for a in itertools.chain(op._schema.arguments, op._schema.returns)
        )
        if not has_tensor_arg:
            return False

        try:
            # CompositeImplicitAutograd 操作对追踪器是透明的，因此不需要分解
            return not op.has_kernel_for_dispatch_key(
                DispatchKey.CompositeImplicitAutograd
            )
        except RuntimeError as e:
            # 对于某些 jit-registered 操作，has_key 可能会失败，但在此情况下不应该是相关的
            if "does not exist" in str(e):
                return False
            raise

    # 测试方法：检查所有 aten 操作的重载是否都存在分解
    def test_has_decomposition(self):
        def all_aten_overloads():
            for name in torch._C._dispatch_get_all_op_names():
                if not name.startswith("aten::"):
                    continue

                name = name[6:]
                if "." in name:
                    packet_name, overload_name = name.split(".")
                else:
                    packet_name, overload_name = name, "default"

                packet = getattr(aten, packet_name)
                assert isinstance(packet, torch._ops.OpOverloadPacket)
                op = getattr(packet, overload_name)
                yield op

        # 用于一些 CI 配置中注册的操作，避免导致测试失败
        allow_list = {aten.get_gradients.default}

        # 获取所有可能出现在追踪中的操作，并且它们需要存在分解
        overloads_wanting_decomp = {
            op for op in all_aten_overloads() if self._can_appear_in_trace(op)
        }
        # 计算缺少分解的操作集合
        ops_missing_decomp = overloads_wanting_decomp - decomposition_table.keys()
        ops_missing_decomp -= allow_list
        # 断言预期结果，检查缺少分解的操作列表
        self.assertExpected(
            "".join(sorted(op.name() + "\n" for op in ops_missing_decomp))
        )
    def test_aten_core_operators(self):
        # 如果一个分解未包含在核心分解中，
        # 那么它必须分解为一个核心 ATen 操作符。
        #
        # 参见注释 [Core ATen Ops]
        #
        # 如果此测试失败，则可能是以下原因之一：
        # - 将分解添加到 torch._decomp.core_aten_decompositions 中，
        #   如果分解应该被感应器使用（而不是核心操作符）。
        # - 使用 EXPECTTEST_ACCEPT=1 再次运行此测试，以更新核心 ATen 操作符的列表
        #   （感应器将不使用分解）。

        # 一些分解已注册为 CompositeImplicitAutograd 操作符，
        # 它们永远不会出现在 AOTAutograd 的图中，因此永远不会被使用。
        useful_decomps = {
            op
            for op in decomposition_table.keys()  # 遍历分解表中的所有操作符
            if isinstance(op, torch._ops.OpOverload) and self._can_appear_in_trace(op)  # 如果操作符是 OpOverload 类型并且可以出现在跟踪中
        }
        core_decomps = torch._decomp.core_aten_decompositions().keys()  # 获取核心 ATen 分解的键集合
        core_aten_ops = useful_decomps - core_decomps  # 找出有用的分解中不属于核心分解的部分
        self.assertExpected("".join(sorted(op.name() + "\n" for op in core_aten_ops)))
# 如果当前脚本作为主程序运行（而不是被导入），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```