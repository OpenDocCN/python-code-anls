# `.\pytorch\test\functorch\test_ops.py`

```py
# 导入 functools 模块，提供了高阶函数支持
import functools
# 导入 itertools 模块，提供了操作迭代器的函数
import itertools
# 导入 unittest 模块，用于编写和运行单元测试
import unittest

# 从 common_utils 导入一系列函数和变量
from common_utils import (
    check_vmap_fallback,
    decorate,
    expectedFailureIf,
    generate_vmap_inputs,
    get_fallback_and_vmap_exhaustive,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    loop,
    loop2,
    opsToleranceOverride,
    skip,
    skipOps,
    tol1,
    tol2,
    xfail,
)

# 从 functorch_additional_op_db 导入 additional_op_db 变量
from functorch_additional_op_db import additional_op_db

# 导入 torch 模块
import torch
# 导入 torch.autograd.forward_ad 模块，提供前向自动微分支持
import torch.autograd.forward_ad as fwAD

# 从 functorch 模块导入一系列函数：grad, jacfwd, jacrev, vjp, vmap
from functorch import grad, jacfwd, jacrev, vjp, vmap
# 从 torch 模块导入 Tensor 类型
from torch import Tensor
# 从 torch._functorch.eager_transforms 模块导入 _as_tuple, jvp 函数
from torch._functorch.eager_transforms import _as_tuple, jvp
# 从 torch.testing._internal.autograd_function_db 模块导入 autograd_function_db 变量
from torch.testing._internal.autograd_function_db import autograd_function_db
# 从 torch.testing._internal.common_cuda 模块导入 with_tf32_off 函数
from torch.testing._internal.common_cuda import with_tf32_off
# 从 torch.testing._internal.common_device_type 模块导入多个函数和变量
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
# 从 torch.testing._internal.common_methods_invocations 模块导入多个函数
from torch.testing._internal.common_methods_invocations import op_db

# 从 torch.testing._internal.common_utils 模块导入多个函数和变量
from torch.testing._internal.common_utils import (
    is_iterable_of_tensors,
    IS_MACOS,
    IS_X86,
    noncontiguous_like,
    parametrize,
    run_tests,
    runOnRocm,
    skipIfRocm,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TestCase,
    unMarkDynamoStrictTest,
)

# 从 torch.testing._internal.opinfo.core 模块导入 SampleInput 类
from torch.testing._internal.opinfo.core import SampleInput
# 从 torch.utils._pytree 模块导入 pytree 相关函数和类
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

# 导入 torch.ops.aten 模块
aten = torch.ops.aten


# 自定义版本的 autograd.grad 函数，具有以下不同点：
#   - 允许使用 pytree 输入（但是 pytree 的叶子必须全部是张量）
#   - 如果一个输入在导数中没有使用，将返回一个用零填充的张量作为结果
def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    # 使用 tree_flatten 函数将输入展平成列表，inputs_spec 是展平后的结果和规格说明
    inputs, inputs_spec = tree_flatten(inputs)
    # 提取所有需要梯度的输入张量
    diff_inputs = tuple(inp for inp in inputs if inp.requires_grad)
    if grad_outputs is None:
        # 如果未提供梯度输出，提取所有需要梯度的输出张量
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        # 否则，提取需要梯度的输出张量和对应的梯度输出
        diff_grad_outputs = [
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        ]
        if len(diff_grad_outputs) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*diff_grad_outputs)
    # 使用 torch.autograd.grad 函数计算梯度
    grad_inputs = torch.autograd.grad(
        diff_outputs,
        diff_inputs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    result = []
    # 创建梯度输入的迭代器
    grad_inputs_iter = iter(grad_inputs)
    # 遍历输入列表中的每个输入张量
    for inp in inputs:
        # 检查当前输入张量是否需要梯度计算
        if inp.requires_grad:
            # 如果需要梯度计算，从梯度输入迭代器中获取下一个梯度输入
            grad_input = next(grad_inputs_iter)
            # 如果获取的梯度输入为 None，则将一个与输入张量相同形状的零张量加入结果列表
            if grad_input is None:
                result.append(torch.zeros_like(inp))
            else:
                # 否则，将获取的梯度输入加入结果列表
                result.append(grad_input)
        else:
            # 如果当前输入张量不需要梯度计算，则将一个与输入张量相同形状的零张量加入结果列表
            result.append(torch.zeros_like(inp))
    
    # 使用输入规范重新构造结果列表，返回最终结果
    return tree_unflatten(result, inputs_spec)
# 定义一个函数，用于检查参数是否可微分
def diff_arg(arg, requires_grad=True):
    # 定义内部函数，检查参数是否可微分
    def is_differentiable_arg(arg):
        # 如果需要梯度，检查是否需要梯度
        if requires_grad:
            return arg.requires_grad
        else:
            # 否则，检查参数是否为浮点数或复数
            return arg.is_floating_point() or arg.is_complex()

    # 如果参数是张量的可迭代对象
    if is_iterable_of_tensors(arg):
        # 如果所有参数都可微分，则返回True
        if all(is_differentiable_arg(a) for a in arg):
            return True
        # 如果所有参数都不可微分，则返回False
        if all(not is_differentiable_arg(a) for a in arg):
            return False
        # 否则抛出运行时错误，当前测试运行器无法处理此情况
        raise RuntimeError("NYI: The test runner can't handle this")
    # 如果参数是张量且可微分，则返回True
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# 定义一个函数，返回一个函数 f'，满足以下条件：
# - f' 只接受位置参数
# - 所有 f' 的参数都是浮点数张量
# - f' 的所有输出都是浮点数张量
def normalize_op_input_output2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=True
):
    # 展平参数列表，并获取参数结构信息
    flat_args, args_spec = tree_flatten(args)
    # 找出可微分参数的索引
    diff_argnums = tuple(
        i
        for i, arg in enumerate(flat_args)
        if diff_arg(arg, requires_grad=requires_grad)
    )
    # 确保至少有一个可微分参数
    assert len(diff_argnums) > 0
    # 提取原始参数中的可微分参数
    primals = tuple(flat_args[i] for i in diff_argnums)

    # 定义一个包装函数，用于接收 primals 作为参数
    @functools.wraps(f)
    def wrapped(*primals):
        # 复制一份参数列表
        _args = list(flat_args)
        # 将 primals 替换回参数列表中的对应位置
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        # 重建参数结构
        _args = tree_unflatten(_args, args_spec)
        # 调用原始函数 f，并传入重建后的参数列表和关键字参数
        result = f(*_args, **kwargs)
        # 如果定义了输出处理函数，则对结果应用该函数
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        # 如果结果是元组，则保留其中的浮点数张量
        if isinstance(result, tuple):
            result = tuple(r for r in result if torch.is_floating_point(r))
            # 确保至少有一个输出是浮点数张量
            assert len(result) > 0
        # 返回处理后的结果
        return result

    # 返回包装函数和原始参数的元组
    return wrapped, primals


# TODO: 与 normalize_op_input_output2 合并
# 定义一个函数，返回一个函数 f'，满足以下条件：
# - f' 只接受位置参数
# - 所有 f' 的参数都是浮点数张量
# - f' 的所有输出都是浮点数张量
def normalize_op_input_output3(
    f, args, kwargs, sample_args, output_process_fn_grad=None
):
    # 展平参数列表，并获取参数结构信息
    flat_args, args_spec = tree_flatten(args)
    # 展平示例参数列表
    flat_sample_args = pytree.tree_leaves(sample_args)
    # 找出可微分参数的索引
    diff_argnums = tuple(
        i
        for i, (arg, sample) in enumerate(zip(flat_args, flat_sample_args))
        if diff_arg(sample, requires_grad=True)
    )
    # 确保至少有一个可微分参数
    assert len(diff_argnums) > 0
    # 提取原始参数中的可微分参数
    primals = tuple(flat_args[i] for i in diff_argnums)

    # 定义一个包装函数，用于接收 primals 作为参数
    @functools.wraps(f)
    def wrapped(*primals):
        # 复制一份参数列表
        _args = list(flat_args)
        # 将 primals 替换回参数列表中的对应位置
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        # 重建参数结构
        _args = tree_unflatten(_args, args_spec)
        # 调用原始函数 f，并传入重建后的参数列表和关键字参数
        result = f(*_args, **kwargs)
        # 如果定义了输出处理函数，则对结果应用该函数
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        # 如果结果是元组，则保留其中的浮点数张量
        if isinstance(result, tuple):
            result = tuple(r for r in result if torch.is_floating_point(r))
            # 确保至少有一个输出是浮点数张量
            assert len(result) > 0
        # 返回处理后的结果
        return result

    # 返回包装函数和原始参数的元组
    return wrapped, primals


# 定义一个函数，返回一个函数 f'，满足以下条件：
# - f' 只接受位置参数
# - 所有 f' 的参数都是浮点数张量
# - f' 的所有输出都是浮点数张量
def normalize_op_input_output(f, sample, requires_grad=True):
    # 构建参数列表，将样本输入作为第一个参数，其余作为剩余参数
    args = tuple([sample.input] + list(sample.args))
    # 调用 normalize_op_input_output2 函数，返回结果
    return normalize_op_input_output2(
        f,
        args,
        sample.kwargs,
        sample.output_process_fn_grad,
        requires_grad=requires_grad,
    )
# 返回一个函数 g(*args, *cotangents)，该函数计算 vjps 和 (*args, cotangents)
def ref_vjp(f, *primals):
    # 调用函数 f，传入参数 primals，并接收结果
    result = f(*primals)

    # 定义内部函数 wrapped，接收 cotangents 参数，并调用 _autograd_grad 函数处理
    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents))

    # 返回结果 result 和内部函数 wrapped
    return result, wrapped


# 模拟 JVP（Jacobian-Vector Product）操作
def simulate_jvp(f, primals, tangents):
    # 使用 torch.autograd.functional.jvp 计算函数 f 在 primals 和 tangents 上的结果
    primals_out, tangents_out = torch.autograd.functional.jvp(f, primals, tangents)
    # 返回计算结果 primals_out 和 tangents_out
    return primals_out, tangents_out


# 返回一个函数 g(*args, *cotangents)，该函数计算 jvps 和 (*args, *cotangents)
def ref_jvp(f, primals, tangents):
    # 进入 fwAD 的双重级别上下文管理器
    with fwAD.dual_level():
        # 使用 fwAD.make_dual 将 primals 和 tangents 转换为双向张量 duals
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        # 调用函数 f，传入 duals 参数，并接收结果 result_duals
        result_duals = f(*duals)
        # 使用 tree_flatten 将 result_duals 展平并获取规范 spec
        result_duals, spec = tree_flatten(result_duals)
        # 使用 fwAD.unpack_dual 将 result_duals 拆分为 primals_out 和 tangents_out
        primals_out, tangents_out = zip(*(fwAD.unpack_dual(d) for d in result_duals))
        # 使用 tree_unflatten 将 primals_out 和 tangents_out 还原成原始结构
        return tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec)


# 获取对应函数 f 和样本 sample 的 cotangents，返回处理结果的新函数
def get_sample_cotangents(f, sample):
    # 调用 normalize_op_input_output 函数，标准化 f 和 sample 的输入输出
    fn, primals = normalize_op_input_output(f, sample)
    # 调用 fn 函数，传入 primals 参数，并接收输出结果 output
    output = fn(*primals)
    # 使用 tree_map 将 torch.randn_like 函数映射到 output 上，生成 cotangents
    return tree_map(torch.randn_like, output)


# 返回一个新函数 g(*args, *cotangents)，该函数计算 vjps 和 (*args, cotangents)
# 以及 sample (*args, *cotangents)
def get_vjp_fn_and_args_with_cotangents(f, sample, cotangents):
    # 构建参数 args 和 kwargs
    args = tuple([sample.input] + list(sample.args))
    kwargs = sample.kwargs
    # 使用 tree_flatten 将 args 和 cotangents 展平并获取规范
    flat_args, args_spec = tree_flatten(args)
    flat_cotangents, cotangents_spec = tree_flatten(cotangents)

    # 定义一个包装函数 wrapped，接收参数 *args
    @functools.wraps(f)
    def wrapped(*args):
        # 断言传入参数的长度正确
        assert len(args) == len(flat_args) + len(flat_cotangents)
        # 提取实际参数 actual_args 和 cotangents
        actual_args = args[: len(flat_args)]
        cotangents = args[len(flat_args) :]
        # 使用 tree_unflatten 将 actual_args 和 cotangents 还原成原始结构
        actual_args = tree_unflatten(actual_args, args_spec)
        cotangents = tree_unflatten(cotangents, cotangents_spec)

        # 调用 normalize_op_input_output3 函数，标准化 f、actual_args、kwargs 和 flat_args
        fn, primals = normalize_op_input_output3(
            f, actual_args, kwargs, flat_args, sample.output_process_fn_grad
        )
        # 调用 vjp 函数，传入 fn 和 primals 参数，并获取结果
        _, vjp_fn = vjp(fn, *primals)
        # 调用 vjp_fn 函数，传入 cotangents 参数，并返回结果
        return vjp_fn(cotangents)

    # 返回包装后的函数 wrapped，以及展平后的参数 flat_args + flat_cotangents
    return wrapped, tuple(flat_args + flat_cotangents)


# 返回一个新函数 g(*args, *cotangents)，该函数计算 vjps 和 sample (*args, *cotangents)
def get_vjpfull_variant(f, sample):
    # 调用 normalize_op_input_output 函数，标准化 f 和 sample 的输入输出
    fn, primals = normalize_op_input_output(f, sample)
    # 调用 _get_vjpfull_variant 函数，传入 fn 和 primals 参数，并返回结果
    return _get_vjpfull_variant(fn, primals)


# 返回一个新函数 g(*args, *cotangents)，该函数计算 vjps 和 sample (*args, *cotangents)
def get_vjpfull_variant2(f, args, kwargs):
    # 调用 normalize_op_input_output2 函数，标准化 f、args 和 kwargs 的输入输出
    fn, primals = normalize_op_input_output2(f, args, kwargs)
    # 调用 _get_vjpfull_variant 函数，传入 fn 和 primals 参数，并返回结果
    return _get_vjpfull_variant(fn, primals)


# 内部函数，返回一个新函数 g(*args, *cotangents)，该函数计算 vjps 和 (*args, *cotangents)
def _get_vjpfull_variant(fn, primals):
    # 调用 fn 函数，传入 primals 参数，并接收结果 result
    result = fn(*primals)
    # 使用 tree_map 将 torch.randn_like 函数映射到 result 上，生成 cotangents
    cotangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x, requires_grad=True), result)
    )
    # 获取 primals 的数量
    num_primals = len(primals)
    # 构建参数 args，包括 primals 和 cotangents
    args = (*primals, *cotangents)

    # 定义一个包装函数 wrapped，接收参数 *args
    @functools.wraps(fn)
    def wrapped(*args):
        # 提取 primals 和 cotangents
        primals = args[:num_primals]
        cotangents = args[num_primals:]
        # 调用 vjp 函数，传入 fn 和 primals 参数，并获取结果 result 和 vjp_fn
        result, vjp_fn = vjp(fn, *primals)
        # 如果 result 是 torch.Tensor 类型，则断言 cotangents 的长度为 1
        if isinstance(result, torch.Tensor):
            assert len(cotangents) == 1
            cotangents = cotangents[0]
        # 调用 vjp_fn 函数，传入 cotangents 参数，并返回结果
        return vjp_fn(cotangents)

    # 返回包装后的函数 wrapped，以及参数 args
    return wrapped, args


# 返回一个新函数 g(*args, *cotangents)，该函数计算 jvps 和 (*args, *cotangents)
def get_jvp_variant(f, sample):
    # 我们希望这是 jvp 的高阶变体，以便可以用于包装 vmap
    # 此函数目前没有实现内容，仅有注释说明
    # 使用 normalize_op_input_output 函数对输入和输出进行标准化处理，获取标量函数 fn 和原始输入 primals
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    # 使用 torch.randn_like(x) 生成一个与 primals 中每个元素形状相同的随机张量，并将其转换为元组
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))

    # 使用 functools.wraps(f) 装饰器创建一个函数 wrapped，其行为与 f 相同
    @functools.wraps(f)
    def wrapped(*args):
        # 将输入参数 args 赋值给 tangents
        tangents = args
        # 调用 jvp 函数计算 fn 在 (primals, tangents) 上的 Jacobian 向量积
        primals_out, tangents_out = jvp(fn, primals, tangents)

        # 如果 primals_out 是 torch.Tensor 类型，则返回 (primals_out, tangents_out) 元组
        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            # 否则，将 primals_out 和 tangents_out 展平为列表，并返回它们的连接结果
            flat_primals_out = pytree.tree_leaves(primals_out)
            flat_tangents_out = pytree.tree_leaves(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    # 返回 wrapped 函数和 tangents 元组作为结果
    return wrapped, tangents
def get_jvp_variant_primals_tangents2(
    f, args, kwargs, output_process_fn_grad=None, requires_grad=False
):
    # 根据参数获取规范化后的操作函数和原始输入
    fn, primals = normalize_op_input_output2(
        f, args, kwargs, output_process_fn_grad, requires_grad
    )
    # 为每个原始输入生成随机张量作为切线
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))
    # 返回 JVP 变种的结果
    return _get_jvp_variant(fn, primals, tangents)


def get_jvp_variant_primals_tangents(f, sample):
    # 获取规范化后的操作函数和原始输入（不需要梯度）
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    # 为每个原始输入生成随机张量作为切线
    tangents = _as_tuple(tree_map(lambda x: torch.randn_like(x), primals))
    # 返回 JVP 变种的结果
    return _get_jvp_variant(fn, primals, tangents)


def _get_jvp_variant(fn, primals, tangents):
    # 定义一个装饰器函数，用于计算 JVP 变种
    @functools.wraps(fn)
    def wrapped(*args):
        # 将输入分为原始输入和切线输入
        primals_in = args[: len(primals)]
        tangents_in = args[len(primals) :]
        # 计算 JVP
        primals_out, tangents_out = jvp(fn, primals_in, tangents_in)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            # 如果输出不是张量，则将输出的树结构展开为列表
            flat_primals_out = pytree.tree_leaves(primals_out)
            flat_tangents_out = pytree.tree_leaves(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, primals + tangents


def is_inplace(op, variant):
    # 检查变体是否是就地操作的包装
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()


vjp_fail = {
    xfail("tensor_split"),  # 由于数据指针复合兼容性问题而失败
    # https://github.com/pytorch/pytorch/issues/96560
    decorate("nn.functional.scaled_dot_product_attention", decorator=skipIfRocm),
}

aliasing_ops = {
    "T",
    "broadcast_to",
    "conj",
    "contiguous",
    "diagonal",  # linalg.diagonal 是一个别名
    "expand",
    "flatten",
    "imag",
    "mH",  # adjoint 是一个别名
    "mT",
    "movedim",  # moveaxis 是一个别名
    "narrow",
    "permute",
    "positive",
    # 'ravel' 是复合隐式自动梯度，可能调用 clone
    "real",
    "reshape",
    "resolve_conj",
    "resolve_neg",
    "select",
    "squeeze",
    "transpose",  # swapdims 和 swapaxes 是别名
    "unflatten",
    "unfold",
    "unsqueeze",
    "view",
    "view_as",
    "view_as_complex",
    "view_as_real",
}

aliasing_ops_list_return = {
    "chunks",
    "dsplit",
    "hsplit",
    "split",
    "unbind",
    "vsplit",
    # 'tensor_split' 不符合复合兼容性，参见 vjp_fail
}

skip_noncontig = {
    "_batch_norm_with_update",
    "as_strided_copy",
}


@unittest.skipIf(TEST_WITH_ASAN, "tests time out with asan, are probably redundant")
@unMarkDynamoStrictTest
class TestOperators(TestCase):
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",
        "test_grad",
        vjp_fail.union(
            {
                xfail(
                    "chalf", "", device_type="cpu"
                ),  # 在 CPU 设备上因未实现 'ComplexHalf' 而跳过测试
                xfail(
                    "sparse.sampled_addmm", ""
                ),  # 稀疏 CSR 张量不具备步幅，因此跳过测试
                xfail(
                    "sparse.mm", "reduce"
                ),  # 稀疏 CSR 张量不具备步幅，因此跳过测试
                # 非连续性 Bug
                #
                # 断言错误：张量不相似！
                xfail("_softmax_backward_data", device_type="cpu"),
                xfail("as_strided"),
                xfail("as_strided", "partial_views"),
                # 运行时错误：！self.requires_grad() || self.is_contiguous()
                xfail("as_strided_scatter"),
                # 运行时错误：张量必须具有最后维度步幅为 1
                xfail("view_as_complex"),
                # 查询：最后维度必须是连续的
                # 融合注意力核需要最后维度连续
                xfail("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                # 运行时错误：预期连续张量，但对于参数 #2 'grad_output' 得到非连续张量
                decorate(
                    "_batch_norm_with_update",
                    decorator=expectedFailureIf(TEST_WITH_ROCM),
                    device_type="cuda",
                ),
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_grad",
        (
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
            ),
            tol1("masked.cumprod", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            tol1(
                "svd_lowrank",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "linalg.tensorsolve",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=8e-04, rtol=1e-03)},
            ),
            tol1(
                "__rmatmul__",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            tol1(
                "matmul",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
        ),
    )


注释：


    # 标记要跳过的操作，针对测试类 "TestOperators" 中的 "test_grad" 测试方法
    @skipOps(
        "TestOperators",
        "test_grad",
        vjp_fail.union(
            {
                # 在 CPU 设备上，由于 'ComplexHalf' 未实现，故跳过测试
                xfail(
                    "chalf", "", device_type="cpu"
                ),
                # 稀疏 CSR 张量在执行 sampled_addmm 操作时，不支持步幅，故跳过测试
                xfail(
                    "sparse.sampled_addmm", ""
                ),
                # 稀疏 CSR 张量在执行 mm 操作时，不支持步幅，故跳过测试
                xfail(
                    "sparse.mm", "reduce"
                ),
                # 对于 _softmax_backward_data 操作，当在 CPU 设备上运行时，存在张量非连续性导致的断言错误，因此跳过测试
                xfail("_softmax_backward_data", device_type="cpu"),
                # 对于 as_strided 操作，存在不明确的非连续性问题，因此跳过测试
                xfail("as_strided"),
                # 对于 as_strided 操作，存在部分视图的非连续性问题，因此跳过测试
                xfail("as_strided", "partial_views"),
                # 对于 as_strided_scatter 操作，存在张量非连续性导致的运行时错误，因此跳过测试
                xfail("as_strided_scatter"),
                # 对于 view_as_complex 操作，要求最后一个维度必须是连续的，故跳过测试
                xfail("view_as_complex"),
                # 对于 nn.functional.scaled_dot_product_attention 操作，要求最后一个维度必须是连续的，故跳过测试
                xfail("nn.functional.scaled_dot_product_attention"),
                # 对于 torch.ops.aten._flash_attention_forward 操作，要求最后一个维度必须是连续的，故跳过测试
                xfail("torch.ops.aten._flash_attention_forward"),
                # 对于 torch.ops.aten._efficient_attention_forward 操作，要求最后一个维度必须是连续的，故跳过测试
                xfail("torch.ops.aten._efficient_attention_forward"),
                # 对于 _batch_norm_with_update 操作，若在 ROCM 环境下，要求连续的张量，故跳过测试
                decorate(
                    "_batch_norm_with_update",
                    decorator=expectedFailureIf(TEST_WITH_ROCM),
                    device_type="cuda",
                ),
            }
        ),
    )
    # 重写操作容差，针对测试类 "TestOperators" 中的 "test_grad" 测试方法
    @opsToleranceOverride(
        "TestOperators",
        "test_grad",
        (
            # 设置 nn.functional.binary_cross_entropy_with_logits 操作的容差，针对 torch.float32 类型的张量
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
            ),
            # 设置 masked.cumprod 操作的容差，针对 torch.float32 类型的张量
            tol1("masked.cumprod", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),
            # 设置 svd_lowrank 操作的容差，针对 torch.float32 类型的张量，限定在 CUDA 设备上
            tol1(
                "svd_lowrank",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            # 设置 linalg.tensorsolve 操作的容差，针对 torch.float32 类型的张量，限定在 CUDA 设备上
            tol1(
                "linalg.tensorsolve",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            # 设置 nn.functional.multi_head_attention_forward 操作的容差，针对 torch.float32 类型的张量
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=8e-04, rtol=1e-03)},
            ),
            # 设置 __rmatmul__ 操作的容差，针对 torch.float32 类型的张量，限定在 CUDA 设备上
            tol1(
                "__rmatmul__",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
            # 设置 matmul 操作的容差，针对 torch.float32 类型的张量，限定在 CUDA 设备上
            tol1(
                "matmul",
                {torch.float32: tol(atol=3e-04, rtol=3e-04)},
                device_type="cuda",
            ),
        ),
    )
    # 定义测试函数 test_grad，接受设备、数据类型和操作对象作为参数
    def test_grad(self, device, dtype, op):
        # 如果操作对象在 vjp_fail 列表中，跳过测试并输出消息
        if op.name in vjp_fail:
            self.skipTest("Skipped; Expected failures")
            return

        # 如果操作对象不支持自动求导，跳过测试并输出消息
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        # 获取操作对象生成的样本输入数据，要求支持梯度
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 如果操作是原地操作，跳过测试并输出消息
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped for redundancy. test_vjp handles in-place testing.")
            return

        # 对每个样本进行迭代
        for sample in samples:
            # 构建参数列表，包括输入和其他参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            # 如果操作对象不在 skip_noncontig 列表中，处理非连续的样本输入
            if op.name not in skip_noncontig:
                noncontig_sample = sample.noncontiguous()
                noncontig_args = [noncontig_sample.input] + list(noncontig_sample.args)
                noncontig_kwargs = noncontig_sample.kwargs

            # 找到需要计算梯度的参数的索引
            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            # 确保至少有一个参数需要计算梯度
            assert len(diff_argnums) > 0
            # 获取需要计算梯度的参数
            diff_args = tuple(args[i] for i in diff_argnums)

            # 定义包装函数 wrapped_fn，执行操作并处理输出结果
            def wrapped_fn(*args, **kwargs):
                # 执行操作
                result = op(*args, **kwargs)
                # 如果定义了输出结果处理函数，应用该函数
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                # 定义函数 abs_if_complex，用于处理复杂数的绝对值
                def abs_if_complex(t):
                    if t.dtype.is_complex:
                        return t.abs()
                    return t

                # 对梯度结果进行归约为单个值
                if isinstance(result, torch.Tensor):
                    return abs_if_complex(result.sum())
                # 对多个结果进行归约处理
                result = sum(abs_if_complex(res.sum()) for res in result)
                return result

            # 计算 wrapped_fn 的梯度结果
            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            # 计算预期的梯度结果
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)
            # 断言计算的梯度结果与预期结果相等
            self.assertEqual(result, expected)

            # 如果操作对象不在 skip_noncontig 列表中，对非连续样本输入也进行测试
            if op.name not in skip_noncontig:
                result_noncontig = grad(wrapped_fn, diff_argnums)(
                    *noncontig_args, **noncontig_kwargs
                )
                # 断言非连续样本输入的计算结果与预期结果相等
                self.assertEqual(result_noncontig, expected)

    # 应用装饰器 with_tf32_off，用于关闭 TF32 模式，解决特定问题
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # 应用装饰器 ops，传入操作数据库、附加操作数据库和自动求导函数数据库，指定允许的数据类型为 torch.float
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestOperators",  # 跳过测试运算符为 "TestOperators" 的测试
        "test_jvp",  # 跳过测试名为 "test_jvp" 的测试
        set(
            {
                # 下列是表现异常的复合操作，需要在 PyTorch 核心中修复
                # 在 tensor_split 操作中会出现 RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
                xfail("tensor_split"),
                # BUG: 静默错误：在大多数环境中运行时产生数值差异
                skip("nn.functional.max_unpool1d"),  # 除了 macOS 外，在其他环境中均失败
                skip(
                    "nn.functional.max_unpool2d"
                ),  # 除了 Windows 外，在其他环境中均失败
                skip("nn.functional.max_unpool3d"),  # 除了 macOS 外，在其他环境中均失败
                xfail(
                    "native_batch_norm"
                ),  # TODO: 比较 None 和保存的均值/方差张量的零张量失败
                xfail(
                    "_native_batch_norm_legit"
                ),  # TODO: 比较 None 和保存的均值/方差张量的零张量失败
                xfail(
                    "_batch_norm_with_update"
                ),  # TODO: 比较 None 和保存的均值/方差张量的零张量失败
                xfail("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                xfail(
                    "nn.functional.rrelu"
                ),  # 在就地操作测试中出错，没有实现公式
                xfail(
                    "NumpyExpMarkDirtyAutogradFunction"
                ),  # TODO: https://github.com/pytorch/pytorch/issues/91280
                # --- 非连续失败！ ---
                # 预期作为操作者失败，因为操作者期望最后一个维度具有 stride=1
                xfail("view_as_complex"),
                # BUG
                # AssertionError: Tensor-likes are not close!
                xfail("as_strided"),
                xfail("as_strided", "partial_views"),
                xfail("as_strided_scatter"),
                decorate(
                    "linalg.det",
                    "singular",
                    decorator=expectedFailureIf(IS_MACOS and IS_X86),
                ),  # 标记 "linalg.det" 在 macOS 和 x86 架构下的失败
            }
        ),
    )
    @opsToleranceOverride(
        "TestOperators",
        "test_jvp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-06)},
                device_type="cuda",
            ),
            tol1(
                "linalg.tensorsolve",
                {torch.float32: tol(atol=1e-04, rtol=1.3e-05)},
                device_type="cuda",
            ),
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",
                {torch.float32: tol(atol=4e-04, rtol=4e-04)},
            ),
            tol1(
                "nn.functional.batch_norm", {torch.float32: tol(atol=4e-05, rtol=5e-05)}
            ),
            tol1("nn.functional.conv2d", {torch.float32: tol(atol=4e-05, rtol=5e-05)}),
            tol1("svd_lowrank", {torch.float32: tol(atol=5e-05, rtol=5e-05)}),
            tol1("pca_lowrank", {torch.float32: tol(atol=5e-05, rtol=5e-05)}),
            tol1(
                "nn.functional.multi_head_attention_forward",
                {torch.float32: tol(atol=6e-05, rtol=2e-05)},
            ),
        ),
    )
    # 重写测试函数 test_jvp，设置不同操作的容差值
    def test_jvp(self, device, dtype, op):
        # 定义需要使用 VJP 分解的操作
        VJP_DECOMP = {
            "nn.functional.logsigmoid",
        }
        # 根据操作是否支持前向自动微分和是否在 VJP_DECOMP 中，决定是否跳过测试
        if op.name in VJP_DECOMP:
            fixme_ref_jvp_local = simulate_jvp
        else:
            fixme_ref_jvp_local = ref_jvp

        if not op.supports_forward_ad and op.name not in VJP_DECOMP:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        # 生成操作的输入样本
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 根据操作是否支持原地操作，选择不同的变体
        outplace_variant = op if not is_inplace(op, op.get_op()) else None
        inplace_variant = op.inplace_variant if op.supports_inplace_autograd else None

        # 遍历样本，测试不同变体的操作
        for sample in samples:
            if outplace_variant:
                # 测试非原地操作变体
                self.jvp_opinfo_test(
                    outplace_variant,
                    sample,
                    sample.output_process_fn_grad,
                    clone_inputs=False,
                    fixme_ref_jvp_local=fixme_ref_jvp_local,
                    test_noncontig=op.name not in skip_noncontig,
                )
            if is_valid_inplace_sample_input(sample, op, inplace_variant):
                # 测试原地操作变体
                self.jvp_opinfo_test(
                    inplace_variant,
                    sample,
                    sample.output_process_fn_grad,
                    clone_inputs=True,
                    fixme_ref_jvp_local=fixme_ref_jvp_local,
                    test_noncontig=op.name not in skip_noncontig,
                )

    # 测试 JVP 操作信息
    def jvp_opinfo_test(
        self,
        fn,
        sample,
        output_process_fn,
        clone_inputs,
        fixme_ref_jvp_local,
        test_noncontig,
    ):
        # NB: we used requires_grad=True to determine where the primals are,
        # but don't need that information otherwise
        # 将输入参数中的 `sample.input` 放入 args 中，并且不需要额外的 requires_grad=True 信息
        args = (sample.input,) + sample.args
        kwargs = sample.kwargs
        # 根据给定的参数和函数 `fn`，规范化操作的输入输出，确保需要梯度计算
        contig_fn, primals = normalize_op_input_output2(
            fn, args, kwargs, output_process_fn, requires_grad=True
        )
        # 将原始的 primals 分离出来，使其不再保留梯度信息
        orig_primals = tree_map(lambda x: x.detach(), primals)
        # 根据 primals 的形状生成与之相同的随机张量，作为原始 tangents
        orig_tangents = tree_map(lambda x: torch.randn_like(x), primals)

        def maybe_clone_inputs():
            # 如果 clone_inputs 为 True，则克隆原始的 primals 和 tangents
            if clone_inputs:
                primals = tree_map(torch.clone, orig_primals)
                tangents = tree_map(torch.clone, orig_tangents)
                return primals, tangents
            # 否则返回原始的 primals 和 tangents
            return orig_primals, orig_tangents

        # 可能克隆输入参数
        primals, tangents = maybe_clone_inputs()
        # 使用 fixme_ref_jvp_local 函数计算期望的 primal_outs 和 tangent_outs
        expected_primal_outs, expected_tangent_outs = fixme_ref_jvp_local(
            contig_fn, primals, tangents
        )

        # 再次可能克隆输入参数
        primals, tangents = maybe_clone_inputs()
        # 使用 jvp 函数计算 primal_outs 和 tangent_outs
        primal_outs, tangent_outs = jvp(contig_fn, primals, tangents)

        # 使用断言方法验证计算得到的 primal_outs 和 tangent_outs 是否等于期望值
        self.assertEqual(primal_outs, expected_primal_outs)
        self.assertEqual(tangent_outs, expected_tangent_outs)

        # 如果需要测试非连续情况
        if test_noncontig:
            # 获取非连续的样本
            noncontig_sample = sample.noncontiguous()
            # 将非连续的输入参数放入 args 中
            noncontig_args = (noncontig_sample.input,) + noncontig_sample.args
            noncontig_kwargs = sample.kwargs
            # 根据给定的参数和函数 `fn`，规范化非连续操作的输入输出，确保需要梯度计算
            noncontig_fn, primals = normalize_op_input_output2(
                fn,
                noncontig_args,
                noncontig_kwargs,
                output_process_fn,
                requires_grad=True,
            )
            # 将非连续操作的 primals 分离出来，使其不再保留梯度信息
            noncontig_primals = tree_map(lambda x: x.detach(), primals)
            # 将原始 tangents 转换为与非连续操作相同的非连续张量
            noncontig_tangents = tree_map(
                lambda x: noncontiguous_like(x), orig_tangents
            )
            # 使用 jvp 函数计算非连续操作的 primal_outs 和 tangent_outs
            noncontig_primal_outs, noncontig_tangent_outs = jvp(
                noncontig_fn, noncontig_primals, noncontig_tangents
            )

            # 使用断言方法验证计算得到的非连续 primal_outs 和 tangent_outs 是否等于期望值
            self.assertEqual(noncontig_primal_outs, expected_primal_outs)
            self.assertEqual(noncontig_tangent_outs, expected_tangent_outs)

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    # 装饰器，跳过指定测试操作
    @skipOps(
        "TestOperators",  # 测试操作的名称
        "test_vjp",  # 测试函数的名称
        vjp_fail.union(  # 使用 union 函数将多个异常情况合并
            {
                xfail("sparse.sampled_addmm", ""),  # 标记为预期失败，空字符串是额外信息
                xfail("sparse.mm", "reduce"),  # 标记为预期失败，额外信息是 "reduce"
                # ---- 非连续性失败 ----
                # 预计会失败，因为操作期望最后一个维度具有 stride=1
                xfail("view_as_complex"),
                # 运行时错误：query: 最后一个维度必须是连续的
                # 融合注意力核心需要最后一个维度是连续的
                xfail("nn.functional.scaled_dot_product_attention"),
                xfail("torch.ops.aten._flash_attention_forward"),
                xfail("torch.ops.aten._efficient_attention_forward"),
                # BUG
                # 断言错误：Tensor-likes are not close!
                xfail("as_strided"),
                xfail("as_strided_scatter"),
                xfail("_softmax_backward_data", device_type="cpu"),  # 标记为预期失败，设备类型是 "cpu"
                xfail("as_strided", "partial_views"),  # 标记为预期失败，额外信息是 "partial_views"
            }
        ),
    )
    # 装饰器，操作容差覆盖
    @opsToleranceOverride(
        "TestOperators",  # 测试操作的名称
        "test_vjp",  # 测试函数的名称
        (  # 容差设置的元组
            tol1(
                "nn.functional.conv_transpose3d",  # 操作的名称
                {torch.float32: tol(atol=5e-05, rtol=9e-05)},  # 容差设置
                device_type="cuda",  # 设备类型为 "cuda"
            ),
            tol1(
                "nn.functional.binary_cross_entropy_with_logits",  # 操作的名称
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},  # 容差设置
            ),
            tol1(
                "nn.functional.multi_head_attention_forward",  # 操作的名称
                {torch.float32: tol(atol=2e-03, rtol=2e-04)},  # 容差设置
            ),
            tol1("__rmatmul__", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),  # 操作的名称，容差设置
            tol1("matmul", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),  # 操作的名称，容差设置
            tol2(
                "linalg.pinv", "hermitian",  # 操作的名称和额外信息
                {torch.float32: tol(atol=1e-05, rtol=1e-05)},  # 容差设置
            ),
            tol1("linalg.tensorsolve", {torch.float32: tol(atol=1e-05, rtol=1e-05)}),  # 操作的名称，容差设置
            tol1("linalg.multi_dot", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),  # 操作的名称，容差设置
            tol1("svd_lowrank", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),  # 操作的名称，容差设置
            tol1("pca_lowrank", {torch.float32: tol(atol=1e-04, rtol=1e-04)}),  # 操作的名称，容差设置
        ),
    )
    # 定义一个测试方法，用于测试反向传播函数（VJP），接受设备类型、数据类型和操作对象作为参数
    def test_vjp(self, device, dtype, op):
        # 如果操作对象不支持自动求导，则跳过测试，并输出跳过原因
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        # 使用操作对象生成包含梯度信息的样本数据
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 定义内部测试函数，接受操作对象和是否原地操作作为参数
        def _test(_op, inplace=False):
            # 遍历样本数据
            for sample in samples:
                # 如果是原地操作并且样本数据不符合原地操作的要求，则继续下一个循环
                if inplace and not is_valid_inplace_sample_input(
                    sample, op, op.inplace_variant
                ):
                    continue
                # 根据样本数据和操作对象，规范化输入和输出
                fn, primals = normalize_op_input_output(_op, sample)
                # 执行操作函数，计算结果
                result = fn(*primals)
                # 生成随机梯度
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                # 计算向量雅可比积（VJP）
                out, vjp_fn = vjp(fn, *primals)
                # 断言计算得到的输出与预期结果相等
                self.assertEqual(out, result)
                # 使用雅可比积函数计算结果的梯度
                result_vjps = vjp_fn(cotangents)

                # 获取参考的雅可比积函数
                _, vjp_fn = ref_vjp(fn, *primals)
                # 使用参考的雅可比积函数计算预期的梯度
                expected_vjps = vjp_fn(cotangents)

                # 断言计算得到的雅可比积与预期的雅可比积相等
                self.assertEqual(result_vjps, expected_vjps)

                # 如果操作对象的名称不在跳过非连续张量计算的列表中
                if op.name not in skip_noncontig:
                    # 规范化非连续张量输入和输出
                    noncontig_fn, noncontig_primals = normalize_op_input_output(
                        _op, sample.noncontiguous()
                    )
                    # 对梯度生成非连续张量的副本
                    noncontig_cotangents = tree_map(
                        lambda x: noncontiguous_like(x), cotangents
                    )
                    # 计算非连续张量的向量雅可比积
                    out_noncontig, vjp_fn = vjp(noncontig_fn, *noncontig_primals)
                    # 断言非连续张量计算得到的输出与预期结果相等
                    self.assertEqual(out_noncontig, result)
                    # 使用非连续张量的雅可比积函数计算结果的梯度
                    noncontig_result_vjps = vjp_fn(noncontig_cotangents)
                    # 断言计算得到的非连续张量的雅可比积与预期的雅可比积相等
                    self.assertEqual(noncontig_result_vjps, expected_vjps)

        # 调用内部测试函数，传入操作对象
        _test(op)
        # 对于操作对象的每个别名，也调用内部测试函数
        for a_op in op.aliases:
            _test(a_op)
        # 如果操作对象具有原地操作变体
        if op.inplace_variant:

            # 定义一个函数 f，接受输入 inp 和其他参数，并返回原地操作的结果
            def f(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)

            # 调用内部测试函数，传入原地操作的函数 f 和标志 inplace=True
            _test(f, inplace=True)

    # 装饰器，将该测试方法应用于指定的操作数据库和自动求导函数数据库，允许的数据类型为 torch.float
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    # 跳过某些操作的装饰器，用于测试用例 "TestOperators" 中的 "test_vjpvjp"
    @skipOps(
        "TestOperators",
        "test_vjpvjp",
        vjp_fail.union(
            {
                skip("nn.functional.max_unpool1d"),  # 静默错误; 不稳定
                skip("nn.functional.max_unpool2d"),  # 静默错误; 不稳定
                xfail("nn.functional.ctc_loss"),  # 未实现
                xfail(
                    "native_layer_norm", ""
                ),  # 期望一个适当的张量，但参数 #1 'other' 传入了 None
                xfail("sparse.sampled_addmm", ""),  # 稀疏张量没有步长
                xfail("sparse.mm", "reduce"),  # 稀疏张量没有步长
                skip("nn.functional.scaled_dot_product_attention"),  # 跳过缩放点积注意力
                xfail("torch.ops.aten._flash_attention_forward"),  # 失败的注意力前向操作
                xfail("torch.ops.aten._efficient_attention_forward"),  # 失败的高效注意力前向操作
                # 下面是详细失败信息：
                # AssertionError: 张量不相似！
                # 不匹配元素数：1 / 15 (6.7%)
                # 最大绝对差异：索引 (2, 4) 处的 24.0 (允许最多 1e-05)
                # 最大相对差异：索引 (2, 4) 处的 1.7933241714393998e-06 (允许最多 1.3e-06)
                # 故障出现在项目 [0] 处
                xfail("masked.prod"),  # 失败的掩码乘积操作
            }
        ),
    )
    
    # 覆盖操作容差的装饰器，用于测试用例 "TestOperators" 中的 "test_vjpvjp"
    @opsToleranceOverride(
        "TestOperators",
        "test_vjpvjp",
        (
            tol1(
                "nn.functional.conv_transpose3d",
                {torch.float32: tol(atol=5e-05, rtol=9e-05)},
                device_type="cuda",
            ),  # 卷积转置操作的容差覆盖
            tol1("prod", {torch.float32: tol(atol=2e-05, rtol=1e-04)}),  # 点乘操作的容差覆盖
            tol1("masked.cumprod", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),  # 掩码累积乘积操作的容差覆盖
            tol1("cumprod", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),  # 累积乘积操作的容差覆盖
            tol1("linalg.vander", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),  # 维特比乘积操作的容差覆盖
            tol2(
                "linalg.det", "singular", {torch.float32: tol(atol=2e-05, rtol=2e-05)}
            ),  # 行列式操作的容差覆盖，异常情况
        ),
    )
    # 定义测试方法，用于测试特定操作的向量雅可比积
    def test_vjpvjp(self, device, dtype, op):
        # 如果操作不支持自动求导，则跳过测试并输出相应信息
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        # 如果操作不支持二阶导数，则跳过测试并输出相应信息
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        # 使用操作对象生成样本输入，设置为需要梯度计算
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 定义测试函数
        def test(_op, inplace=False):
            # 遍历样本
            for sample in samples:
                # 如果是原地操作且样本不符合原地操作的要求，则继续下一个循环
                if inplace and not is_valid_inplace_sample_input(
                    sample, op, op.inplace_variant
                ):
                    continue
                # 获取操作的向量雅可比积全变量
                fn, args = get_vjpfull_variant(_op, sample)
                # 计算操作结果
                result = fn(*args)
                # 生成随机梯度的余切向量
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                # 计算向量雅可比积的雅可比积
                _, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)

                # 计算向量雅可比积的参考雅可比积
                _, vjp_fn = ref_vjp(fn, *args)
                expected_vjps = vjp_fn(cotangents)

                # 断言操作的向量雅可比积与参考雅可比积相等
                self.assertEqual(result_vjps, expected_vjps)

        # 对当前操作进行测试
        test(op)
        # 如果存在原地操作变体
        if op.inplace_variant:
            # 定义原地操作的函数
            def fn(inp, *args, **kwargs):
                return op.inplace_variant(inp.clone(), *args, **kwargs)

            # 对原地操作进行测试
            test(fn, inplace=True)

    # 设置 TF32 禁用装饰器，用于禁用某些特定测试用例
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # 注册一系列操作和自动求导函数的测试
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    # 设置操作的容差覆盖，以允许容差范围内的数值误差
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    # 设置测试函数的容差覆盖，针对特定测试用例设置容差范围
    @opsToleranceOverride(
        "TestOperators",
        "test_vmapvjpvjp",
        (
            tol1("linalg.svd", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
            tol1("linalg.lu_factor", {torch.float32: tol(atol=2e-03, rtol=2e-02)}),
            tol1("svd", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
            tol1("matrix_exp", {torch.float32: tol(atol=1e-03, rtol=5e-04)}),
        ),
    )
    # 设置要跳过的操作测试，用于标记不应运行的测试用例
    @skipOps(
        "TestOperators",
        "test_vmapvjpvjp",
        {
            xfail("as_strided", "partial_views"),
        },
    )
    def test_vmapvjpvjp(self, device, dtype, op):
        # 测试 `vjpvjp` 独立运行时，
        # 确保 vmap 的 `vjpvjp` 正确性。
        
        # 如果操作不支持自动微分，则跳过测试
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        
        # 如果操作不支持二阶梯度，则跳过测试
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        # 获取在指定设备和数据类型上的输入样本，要求计算梯度
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: 测试原地操作
        # 如果操作是原地操作，则跳过测试
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        # 对每个样本进行测试
        for sample in samples:
            # 获取 vjpfull 变体的函数和参数
            fn, args = get_vjpfull_variant(op, sample)
            # 执行函数计算结果
            result = fn(*args)
            # 生成与结果形状相同的随机张量作为余切向量
            cotangents = tree_map(lambda x: torch.randn_like(x), result)
            cotangents = pytree.tree_leaves(cotangents)
            num_args = len(args)

            # 将参数和余切向量组合成元组
            args_and_cotangents = tuple(args) + tuple(cotangents)

            # 定义双重 vjp 的函数
            def vjp_of_vjp(*args_and_cotangents):
                args = args_and_cotangents[:num_args]
                cotangents = args_and_cotangents[num_args:]
                result, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)
                result = pytree.tree_leaves(result)
                result_vjps = pytree.tree_leaves(result_vjps)
                return (*result, *result_vjps)

            # 检查是否为批归一化并处于训练状态
            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            # 获取回退和 vmap 的穷尽生成器
            generator = get_fallback_and_vmap_exhaustive(
                vjp_of_vjp,
                args_and_cotangents,
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
            )
            # 遍历生成器，比较循环输出和批处理输出是否相等
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)
    # 使用 @skipOps 装饰器跳过指定的测试用例，详细见函数定义及其参数说明
    @skipOps(
        "TestOperators",  # 跳过名为 "TestOperators" 的测试类中的测试用例
        "test_vmapvjp",  # 跳过名为 "test_vmapvjp" 的测试用例
        vmapvjp_fail.union(  # 联合额外的跳过条件，这些条件被装饰为无效测试
            {
                xfail("as_strided"),  # 标记 "as_strided" 为预期失败
                xfail("as_strided_copy"),  # 标记 "as_strided_copy" 为预期失败
                xfail("as_strided", "partial_views"),  # 标记 "as_strided" 和 "partial_views" 为预期失败
            }
        ),
    )
    # 定义名为 test_vmapvjp 的测试方法，接受 device、dtype 和 op 作为参数
    def test_vmapvjp(self, device, dtype, op):
        # 如果 op 不支持自动求导，则跳过当前测试用例
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        # 生成带有梯度信息的样本输入
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place，如果操作是原地操作，则跳过当前测试用例
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return
        
        # 遍历每个样本输入
        for sample in samples:
            # 获取样本输入对应的 cotangents（余切向量）
            cotangents = get_sample_cotangents(op, sample)
            
            # 获取包含 cotangents 的 VJP 函数及其参数
            fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
            
            # 检查是否是批处理标准化并且正在训练
            is_batch_norm_and_training = is_batch_norm_training(op.name, sample.kwargs)
            
            # 获取 fallback 和 vmap 详尽测试的生成器
            generator = get_fallback_and_vmap_exhaustive(
                fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training
            )
            
            # 遍历生成器的输出，并逐一断言每个输出是否相等
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)
    }
    # 定义一个测试方法，用于测试 vmapjvpall 操作
    def test_vmapjvpall(self, device, dtype, op):
        # 检查操作是否支持原地运算，如果是，则跳过测试并输出相应信息
        if is_inplace(op, op.get_op()):
            # TODO: 测试原地操作
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        # 从操作中获取样本输入
        samples = op.sample_inputs(device, dtype, requires_grad=False)

        # 如果操作不支持前向自动微分，跳过测试并输出相应信息
        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        # 遍历样本集合中的每个样本
        for sample in samples:
            # 构建参数值列表，包括输入和其他参数
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            args = tuple(arg_values) + tuple(kwarg_values)
            # 获取 JVP 变体的原始值和切线
            fn, args = get_jvp_variant_primals_tangents(op, sample)
            # 检查是否是批量归一化且处于训练状态
            is_batch_norm_and_training = is_batch_norm_training(op.name, kwarg_values)
            # 获取回退和 vmap 全面测试生成器
            generator = get_fallback_and_vmap_exhaustive(
                fn, args, {}, is_batch_norm_and_training=is_batch_norm_and_training
            )
            # 遍历生成器中的每对循环输出和批处理输出，进行断言比较
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    # 应用操作数据库、附加操作数据库和自动求导函数数据库中的操作，指定允许的数据类型为 torch.float
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    # 装饰器：跳过特定的操作测试用例，包括特定的跳过和失败的测试用例
    @skipOps(
        "TestOperators",
        "test_vmapjvpall_has_batch_rule",
        vmapjvpall_fail.union(
            {
                skip(
                    "to"
                ),  # 运行时错误：要求秩为4的张量使用 channels_last 格式
                xfail(
                    "cdouble"
                ),  # 运行时错误：要求秩为4的张量使用 channels_last 格式
                xfail("cumprod"),
                xfail("masked_fill"),
                xfail("fill"),
                skip("masked.mean"),  # ???
                xfail("masked_scatter"),
                xfail("put"),
                xfail("take"),
                xfail("nn.functional.feature_alpha_dropout", "without_train"),
                xfail("nn.functional.dropout2d", ""),
                xfail("pca_lowrank", ""),
                xfail("svd_lowrank", ""),
                xfail("nn.functional.feature_alpha_dropout", "with_train"),
                xfail("special.log_ndtr", ""),
                xfail("fft.ihfft2"),  # conj_physical 回退
                xfail("fft.ihfftn"),  # conj_physical 回退
                xfail("nn.functional.max_unpool3d", "grad"),
                xfail("nn.functional.max_unpool2d", "grad"),
                xfail("nn.functional.soft_margin_loss", ""),
                xfail("nn.functional.max_unpool1d", "grad"),
                xfail("nn.functional.embedding", ""),
                xfail(
                    "scatter_reduce", "sum"
                ),  # aten::scatter_reduce.two 命中了 vmap 回退
                xfail(
                    "scatter_reduce", "mean"
                ),  # aten::scatter_reduce.two 命中了 vmap 回退
                xfail(
                    "scatter_reduce", "amin"
                ),  # aten::scatter_reduce.two 命中了 vmap 回退
                xfail(
                    "scatter_reduce", "amax"
                ),  # aten::scatter_reduce.two 命中了 vmap 回退
                xfail("nn.functional.glu"),
                xfail("nn.functional.bilinear"),  # trilinear 没有批处理规则
                xfail("linalg.lu", ""),
                xfail("nn.functional.dropout3d", ""),
                xfail("as_strided_scatter", ""),
                xfail("masked.cumprod", ""),
                xfail("renorm"),  # 命中了禁用的 vmap 回退
            }
        ).difference(
            {
                # as_strided_copy 在 test_vmapvjp 中失败，在此处成功
                xfail("as_strided_copy", ""),
            }
        ),
    )
    # 装饰器：设置特定浮点数精度的容差
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    # 检查是否为原地操作，如果是，跳过测试，并输出相应的跳过信息
    if is_inplace(op, op.get_op()):
        # TODO: 测试原地操作
        self.skipTest("Skipped! NYI: inplace-testing not supported.")
        return

    # 生成操作的样本输入数据，包括设备类型和数据类型
    samples = op.sample_inputs(device, dtype, requires_grad=False)

    # 如果操作不支持前向自动微分，跳过测试，并输出相应的跳过信息
    if not op.supports_forward_ad:
        self.skipTest("Skipped! Forward AD not supported.")
        return

    # 定义测试函数
    def test():
        # 遍历样本数据集
        for sample in samples:
            # 构建参数值列表，包括输入值和参数值
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            args = tuple(arg_values) + tuple(kwarg_values)
            # 获取 JVP 变体的原始值和切线值函数
            fn, args = get_jvp_variant_primals_tangents(op, sample)
            # 检查是否为批量归一化且处于训练模式
            is_batch_norm_and_training = is_batch_norm_training(
                op.name, kwarg_values
            )
            # 获取回退和批量映射详尽检查的结果
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                fn,
                args,
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
                compute_loop_out=False,
            ):
                pass

    # 使用检查批量映射回退的函数进行测试
    check_vmap_fallback(self, test, op, dry_run=False)
    # 检查是否操作支持自动求导，如果不支持则跳过测试
    def test_vmapvjp_has_batch_rule(self, device, dtype, op):
        if not op.supports_autograd:
            # 输出跳过测试的信息并返回
            self.skipTest("Skipped! Autograd not supported.")
            return

        # 生成需要求导的样本数据
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        # 如果操作支持原地运算，则跳过测试
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        # 定义测试函数
        def test():
            # 遍历样本数据集
            for sample in samples:
                # 获取样本数据对应的余切值
                cotangents = get_sample_cotangents(op, sample)
                # 获取关于变量的梯度值和相关的函数
                fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
                # 检查当前操作是否是批量归一化且处于训练状态
                is_batch_norm_and_training = is_batch_norm_training(
                    op.name, sample.kwargs
                )
                # 获取回退值和 Vmap 的详尽遍历
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                    fn,
                    args,
                    {},
                    is_batch_norm_and_training=is_batch_norm_and_training,
                    compute_loop_out=False,
                ):
                    pass
                # 对于操作的所有别名进行测试
                for a_op in op.aliases:
                    # 获取别名操作的函数和参数
                    fn, args = get_vjp_fn_and_args_with_cotangents(
                        a_op, sample, cotangents
                    )
                    # 获取回退值和 Vmap 的详尽遍历
                    for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                        fn,
                        args,
                        {},
                        is_batch_norm_and_training=is_batch_norm_and_training,
                        compute_loop_out=False,
                    ):
                        pass

        # 调用检查 Vmap 回退的函数，并传入测试函数和操作对象
        check_vmap_fallback(self, test, op, dry_run=False)

    # 注解：@ops 标签用于标识操作数据库中的运算，设置允许的数据类型为 torch.float
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    # 定义一个测试函数，用于测试批处理自动求导规则的操作
    def test_vjpvmap(self, device, dtype, op):
        # 注意: 没有 vjpvmap_has_batch_rule 的测试，因为在 test_vmap.py 中的 vmap_has_batch_rule 测试几乎是多余的

        # 如果操作是 nn.functional.dropout，则跳过该测试
        if op.name == "nn.functional.dropout":
            self.skipTest("Skipped!")

        # 如果操作不支持自动求导，则跳过测试
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        # TODO: 测试原地操作
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        # 生成操作的样本输入
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        
        # 一些与批处理归一化相关的函数名称
        batch_norm_fns = (
            "nn.functional.batch_norm",
            "nn.functional.instance_norm",
        )  # instance norm 调用 batch norm

        # 判断当前操作是否是批处理归一化操作
        is_batch_norm = op.name in batch_norm_fns

        # 遍历样本输入
        for sample in samples:
            # 准备参数和关键字参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            # 判断是否是批处理归一化并且处于训练模式
            is_batch_norm_and_training = is_batch_norm and is_batch_norm_training(
                op.name, kwargs
            )

            # 生成输入的批处理版本
            generator = generate_vmap_inputs(
                args, kwargs, is_batch_norm_and_training=is_batch_norm_and_training
            )

            # 遍历生成的输入
            for batched_args, in_dims, kwargs in generator:
                # 对操作应用 vmap 函数
                vmapped_op = vmap(op, in_dims)
                
                # 标准化操作的输入输出
                fn, primals = normalize_op_input_output2(
                    vmapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )
                
                # 执行操作
                result = fn(*primals)
                
                # 创建随机的余切向量
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                # 计算操作的 VJP 函数
                _, vjp_fn = vjp(fn, *primals)
                
                # 计算 VJP 的结果
                result_vjps = vjp_fn(cotangents)

                # 计算参考 VJP 函数
                _, vjp_fn = ref_vjp(fn, *primals)
                
                # 计算参考 VJP 的结果
                expected_vjps = vjp_fn(cotangents)

                # 断言两者的结果应该相等
                self.assertEqual(result_vjps, expected_vjps)

    # 比较 VJP 的雅可比矩阵
    def _compare_jacobians_of_vjp(
        self, fn, cotangents_and_primals, argnums=None, atol_rtol=None
    ):
        # 如果没有指定参数编号，则默认使用所有参数
        if argnums is None:
            argnums = tuple(range(len(cotangents_and_primals)))

        # 定义获取 VJP 函数的函数
        def get_vjp(cotangents, *primals):
            _, vjp_fn = vjp(fn, *primals)
            return vjp_fn(cotangents)

        # 计算正向 JVP 的雅可比矩阵
        jacobian_jvp = jacfwd(get_vjp, argnums)(*cotangents_and_primals)
        
        # 计算反向 VJP 的雅可比矩阵
        jacobian_vjp = jacrev(get_vjp, argnums)(*cotangents_and_primals)

        # 对于数据类型变化的操作，两个雅可比矩阵可能具有不同的数据类型
        jacobian_jvp = tree_map(lambda x: x.to(torch.float), jacobian_jvp)
        jacobian_vjp = tree_map(lambda x: x.to(torch.float), jacobian_vjp)

        # 如果指定了绝对误差和相对误差，则使用它们进行断言
        if atol_rtol is not None:
            (atol, rtol) = atol_rtol
            self.assertEqual(jacobian_jvp, jacobian_vjp, atol=atol, rtol=rtol)
        else:
            # 否则，直接断言两个雅可比矩阵应该相等
            self.assertEqual(jacobian_jvp, jacobian_vjp)

    # 应用于所有操作的装饰器，用于测试自动求导函数
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    )
    @opsToleranceOverride(
        "TestOperators",  # 设置容差覆盖的测试类别为 "TestOperators"
        "test_jvpvjp",   # 设置容差覆盖的测试方法为 "test_jvpvjp"
        (                 # 开始定义容差覆盖的具体内容，使用元组表示多个条目
            tol1("masked.prod", {torch.float32: tol(atol=1e-04, rtol=1.3e-05)}),  # 设置 "masked.prod" 的容差参数
            tol1("masked.cumprod", {torch.float32: tol(atol=1e-04, rtol=5e-04)}),  # 设置 "masked.cumprod" 的容差参数
            tol1(           # 设置 "cumprod" 的容差参数
                "cumprod",  # 操作名称为 "cumprod"
                {torch.float32: tol(atol=1e-04, rtol=1.3e-05)},  # 指定数据类型为 torch.float32 的容差值
                device_type="cuda",  # 指定设备类型为 "cuda"
            ),
            tol1(           # 设置 "linalg.vander" 的容差参数
                "linalg.vander",  # 操作名称为 "linalg.vander"
                {torch.float32: tol(atol=1e-04, rtol=1.3e-05)},  # 指定数据类型为 torch.float32 的容差值
                device_type="cuda",  # 指定设备类型为 "cuda"
            ),
            tol1(           # 设置 "nn.functional.group_norm" 的容差参数
                "nn.functional.group_norm",  # 操作名称为 "nn.functional.group_norm"
                {torch.float32: tol(atol=1e-03, rtol=1e-03)}  # 指定数据类型为 torch.float32 的容差值
            ),
            tol2(           # 设置 "linalg.pinv" 的容差参数
                "linalg.pinv",  # 操作名称为 "linalg.pinv"
                "hermitian",   # 指定 "linalg.pinv" 的额外修饰条件为 "hermitian"
                {torch.float32: tol(atol=5e-03, rtol=5e-03)}  # 指定数据类型为 torch.float32 的容差值
            ),
        ),
    )
    # 定义一个测试方法，用于测试某个操作在指定设备和数据类型上的行为
    def test_jvpvjp(self, device, dtype, op):
        # 如果操作不支持自动求导，则跳过测试并输出相应信息
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        # 生成操作所需的输入样本，要求对其进行梯度计算
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: 测试原地操作（in-place），如果是原地操作则跳过测试
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        # 遍历每个样本
        for sample in samples:
            # 规范化操作的输入输出
            fn, primals = normalize_op_input_output(op, sample)
            # 执行操作，并得到结果
            result = fn(*primals)
            # 对结果执行随机生成切线向量的操作
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            # 对原始输入和切线向量执行随机生成切线向量的操作
            primals_tangents = tree_map(lambda x: torch.randn_like(x), primals)
            cotangents_tangents = tree_map(lambda x: torch.randn_like(x), cotangents)

            # 定义一个函数用于执行 vjp 操作
            def push_vjp(primals, cotangents):
                _, vjp_fn = vjp(fn, *primals)
                return vjp_fn(cotangents)

            # 执行 jvp 操作，计算结果
            result = jvp(
                push_vjp, (primals, cotangents), (primals_tangents, cotangents_tangents)
            )
            # 断言结果的长度为 2
            self.assertEqual(len(result), 2)

            # 定义一个函数，用于对输入和切线向量执行双重水平求导
            def tree_map2(fn, first, second):
                flat_first, spec_first = tree_flatten(first)
                flat_second, spec_second = tree_flatten(second)
                assert spec_first == spec_second
                flat_result = [fn(f, s) for f, s in zip(flat_first, flat_second)]
                return tree_unflatten(flat_result, spec_first)

            # 定义一个参考函数，用于执行参考水平的求导操作
            def reference(primals, cotangents, primals_tangents, cotangents_tangents):
                with fwAD.dual_level():
                    # 将原始输入转换为双重数值
                    primal_duals = tree_map2(fwAD.make_dual, primals, primals_tangents)
                    _, vjp_fn = ref_vjp(fn, *primal_duals)

                    # 将切线向量转换为双重数值
                    cotangent_duals = tree_map2(
                        fwAD.make_dual, cotangents, cotangents_tangents
                    )
                    # 执行 vjp 操作
                    result = vjp_fn(cotangent_duals)

                    # 将结果展平并返回原始值和切线值
                    flat_result, spec = tree_flatten(result)
                    primals_out, tangents_out = zip(
                        *[fwAD.unpack_dual(r) for r in flat_result]
                    )
                    tangents_out = [
                        t if t is not None else torch.zeros_like(p)
                        for p, t in zip(primals_out, tangents_out)
                    ]
                    expected = (
                        tree_unflatten(primals_out, spec),
                        tree_unflatten(tangents_out, spec),
                    )
                return expected

            # 计算预期结果
            expected = reference(
                primals, cotangents, primals_tangents, cotangents_tangents
            )
            # 断言结果与预期一致
            self.assertEqual(result, expected)

    # 添加装饰器，禁用 TF32 模式，解决特定问题
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # 使用指定的操作数据库和自动求导函数数据库，并限定数据类型为 torch.float32
    @ops(op_db + additional_op_db + autograd_function_db, allowed_dtypes=(torch.float,))
    # 设置容差覆盖，指定浮点数容差范围
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    # 用于修饰测试方法，指定特定操作符的容差覆盖
    @opsToleranceOverride(
        "TestOperators",  # 测试操作符的类名
        "test_vmapjvpvjp",  # 测试方法名
        (
            tol1("linalg.svd", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),  # 设置 linalg.svd 操作的容差
            tol1(
                "linalg.householder_product",  # 设置 linalg.householder_product 操作的容差
                {torch.float32: tol(atol=5e-03, rtol=5e-03)},
            ),
            tol1("linalg.multi_dot", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),  # 设置 linalg.multi_dot 操作的容差
            tol2(
                "linalg.pinv", "hermitian",  # 设置 linalg.pinv 操作的容差，指定 hermitian 属性
                {torch.float32: tol(atol=5e-04, rtol=5e-04)}
            ),
            tol1(
                "nn.functional.conv_transpose2d",  # 设置 nn.functional.conv_transpose2d 操作的容差
                {torch.float32: tol(atol=5e-04, rtol=5e-04)},
            ),
            tol1("svd", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),  # 设置 svd 操作的容差
            tol1("matrix_exp", {torch.float32: tol(atol=5e-04, rtol=5e-04)}),  # 设置 matrix_exp 操作的容差
        ),
    )
    # 测试 vmapjvpvjp 方法
    def test_vmapjvpvjp(self, device, dtype, op):
        # 如果操作不支持自动求导，则跳过测试
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        # 生成操作的样本输入数据，并标记需要梯度信息
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 如果操作是原地操作，则跳过测试
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        # 对每个样本进行测试
        for sample in samples:
            # 标准化操作的输入和输出
            fn, primals = normalize_op_input_output(op, sample)
            # 执行操作
            result = fn(*primals)

            # 对结果执行随机梯度生成
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            # 生成标准化操作输入和输出的切线
            primals_tangents = tree_map(lambda x: torch.randn_like(x), primals)
            cotangents_tangents = tree_map(lambda x: torch.randn_like(x), cotangents)

            # 定义推送 VJP 的函数
            def push_vjp(primals, cotangents):
                _, vjp_fn = vjp(fn, *primals)
                return vjp_fn(cotangents)

            # 展开输入和输出数据，以便进行 JVP 计算
            args, spec = tree_flatten(
                ((primals, cotangents), (primals_tangents, cotangents_tangents))
            )

            # 定义 VJP 的 JVP
            def jvp_of_vjp(*args):
                (primals, tangents) = tree_unflatten(args, spec)
                primals_out, tangents_out = jvp(push_vjp, primals, tangents)

                flat_primals_out = pytree.tree_leaves(primals_out)
                flat_tangents_out = pytree.tree_leaves(tangents_out)
                return tuple(flat_primals_out + flat_tangents_out)

            # 检查是否批量归一化并处于训练状态
            is_batch_norm_and_training = is_batch_norm_training(op, sample.kwargs)

            # 获取回退和 exhaustive vmap
            generator = get_fallback_and_vmap_exhaustive(
                jvp_of_vjp,
                args,
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
            )

            # 对生成的结果进行迭代，比较循环和批量化的输出
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)
    # 生成极端输入数据，用于测试
    def _make_extremal_inputs(self, shape, device):
        # 如果形状为None，则返回一个包含None的元组
        if shape is None:
            return (None,)
        # 否则返回包含三种极端值的元组：全为-1000.0、全为0、全为1000.0
        return (
            torch.full(shape, -1000.0, device=device),
            torch.zeros(shape, device=device),
            torch.full(shape, 1000.0, device=device),
        )

    # 生成位置参数和关键字参数的组合
    def _arg_and_kwarg_options(self, args_options, kwargs_options):
        # 返回位置参数和关键字参数的笛卡尔积
        return itertools.product(*args_options, kwargs_options)

    # 测试极端数值情况下的负对数似然损失函数
    def test_extremal_numerics_nll_loss(self, device):
        N, C = 3, 4
        d1, d2, d3 = 5, 6, 7
        # 定义不同输入和目标的形状
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
        )
        # 定义关键字参数的选项
        kwargs_options = (
            {"ignore_index": 0, "reduction": "mean"},
            {"reduction": "sum"},
            {"reduction": "none"},
            {},
        )
        # 遍历每种输入和目标的形状
        for input_shape, target_shape, weight_shape in shapes:
            # 生成输入的极端数据选项
            input_options = self._make_extremal_inputs(input_shape, device)
            # 遍历所有输入数据和关键字参数选项的组合
            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                # 如果权重的形状为None，则将权重设为None，否则生成随机权重数据
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)
                # 生成随机目标数据
                target = torch.randint(0, C, target_shape, device=device)
                # 忽略索引0，确保目标中至少有一个非零元素
                target[0] = 1

                # 使用偏函数创建带有特定参数的负对数似然损失函数
                fn = functools.partial(
                    torch.nn.functional.nll_loss, target=target, weight=weight, **kwargs
                )
                # 计算损失函数的结果
                result = fn(input)
                # 生成与结果同形状的随机梯度数据
                cotangents = torch.randn_like(result, device=device)
                # 比较变换的雅可比矩阵
                self._compare_jacobians_of_vjp(fn, (cotangents, input))

    # 测试极端数值情况下的L1损失函数
    def test_extremal_numerics_l1_loss(self, device):
        N, C, H, W = 3, 4, 5, 6
        # 定义不同形状的输入
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        # 定义关键字参数的选项
        kwargs_options = ({"reduction": "sum"}, {"reduction": "none"}, {})
        # 遍历每种输入形状
        for shape in shapes:
            # 生成输入和目标的极端数据选项
            input_options = self._make_extremal_inputs(shape, device)
            target_options = self._make_extremal_inputs(shape, device)
            # 遍历所有输入数据、目标数据和关键字参数选项的组合
            for input, target, kwargs in self._arg_and_kwarg_options(
                (input_options, target_options), kwargs_options
            ):
                # 计算L1损失函数的结果
                result = torch.nn.functional.l1_loss(input, target)
                # 生成与结果同形状的随机梯度数据
                cotangents = torch.randn_like(result, device=device)
                # 比较变换的雅可比矩阵
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.l1_loss, (cotangents, input, target)
                )
    # 定义测试函数，用于测试极端数值情况下的均方误差损失函数
    def test_extremal_numerics_mse_loss(self, device):
        # 定义不同维度的输入形状
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        # 定义不同的损失函数参数选项
        kwargs_options = ({"reduction": "sum"}, {"reduction": "none"}, {})
        # 遍历每一种输入形状
        for shape in shapes:
            # 创建极端输入选项
            input_options = self._make_extremal_inputs(shape, device)
            # 创建极端目标选项
            target_options = self._make_extremal_inputs(shape, device)
            # 遍历每一对输入和目标选项以及损失函数参数选项
            for input, target, kwargs in self._arg_and_kwarg_options(
                (input_options, target_options), kwargs_options
            ):
                # 计算均方误差损失
                result = torch.nn.functional.mse_loss(input, target)
                # 创建与结果形状相同的随机张量，作为梯度的初始值
                cotangents = torch.randn_like(result, device=device)
                # 比较损失函数的梯度变化
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.mse_loss, (cotangents, input, target)
                )

    # 定义测试函数，用于测试极端数值情况下的softmax函数
    def test_extremal_numerics_softmax(self, device):
        # 定义不同维度的输入形状
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        # 定义不同的softmax函数参数选项
        kwargs_options = ({"dim": 1}, {})
        # 遍历每一种输入形状
        for shape in shapes:
            # 创建极端输入选项
            input_options = self._make_extremal_inputs(shape, device)
            # 遍历每一对输入选项和softmax函数参数选项
            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                # 计算softmax
                result = torch.nn.functional.softmax(input)
                # 创建与结果形状相同的随机张量，作为梯度的初始值
                cotangents = torch.randn_like(result, device=device)
                # 比较softmax函数的梯度变化
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.softmax, (cotangents, input)
                )

    # 定义测试函数，用于测试极端数值情况下的log_softmax函数
    def test_extremal_numerics_log_softmax(self, device):
        # 定义不同维度的输入形状
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        # 定义不同的log_softmax函数参数选项
        kwargs_options = ({"dim": 1}, {})
        # 遍历每一种输入形状
        for shape in shapes:
            # 创建极端输入选项
            input_options = self._make_extremal_inputs(shape, device)
            # 遍历每一对输入选项和log_softmax函数参数选项
            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                # 计算log_softmax
                result = torch.nn.functional.log_softmax(input)
                # 创建与结果形状相同的随机张量，作为梯度的初始值
                cotangents = torch.randn_like(result, device=device)
                # 比较log_softmax函数的梯度变化
                self._compare_jacobians_of_vjp(
                    torch.nn.functional.log_softmax, (cotangents, input)
                )
    # 定义一个测试函数，用于测试极端数值情况下的交叉熵计算
    def test_extremal_numerics_cross_entropy(self, device):
        # 设定输入数据的维度
        N, C = 3, 4
        # 设定额外维度的尺寸
        d1, d2, d3 = 5, 6, 7
        # 定义多组输入形状，每组包含输入形状、目标形状和权重形状的组合
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C), (N, C), (C,)),
            ((N, C), (N, C), None),
            ((C,), (), (C,)),
            ((C,), (), None),
            ((C,), (C,), (C,)),
            ((C,), (C,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), None),
        )
        # 遍历所有形状组合
        for input_shape, target_shape, weight_shape in shapes:
            # 根据输入形状生成极端输入选项
            input_options = self._make_extremal_inputs(input_shape, device)
            # 定义关键字参数选项，包括不同的约减(reduction)方式和忽略索引(ignore_index)
            kwargs_options = [{"reduction": "sum"}, {"reduction": "none"}, {}]
            if input_shape != target_shape:
                kwargs_options.append({"ignore_index": 0, "reduction": "mean"})

            # 遍历参数和关键字参数选项的组合
            for input, kwargs in self._arg_and_kwarg_options(
                (input_options,), kwargs_options
            ):
                # 如果权重形状为None，则权重设为None；否则生成随机权重
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)

                # 根据输入形状生成目标张量
                if input_shape == target_shape:
                    target = torch.rand(target_shape, device=device)
                elif len(target_shape) == 0:
                    target = torch.tensor(
                        1, device=device
                    )  # 必须是非零值，因为ignore_index可能为0
                else:
                    target = torch.randint(0, C, target_shape, device=device)

                # 部分函数应用，生成特定参数和关键字参数的交叉熵函数
                fn = functools.partial(
                    torch.nn.functional.cross_entropy,
                    target=target,
                    weight=weight,
                    **kwargs,
                )
                # 计算交叉熵的结果
                result = fn(input)
                # 生成与结果张量相同形状的随机张量，用作梯度的传递
                cotangents = torch.randn_like(result, device=device)
                # 比较对VJP（Vector-Jacobian Product，向量-雅可比积）的Jacobian矩阵的比较
                self._compare_jacobians_of_vjp(
                    fn, (cotangents, input), atol_rtol=(1e-4, 1e-5)
                )
    # 定义一个测试函数，用于测试极端数值情况下的二元交叉熵损失函数
    def test_extremal_numerics_binary_cross_entropy(self, device):
        # 定义不同维度的输入形状
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        
        # 遍历不同的输入形状
        for shape in shapes:
            # 创建极端输入的权重选项
            weight_options = self._make_extremal_inputs(shape, device)
            # 定义不同的参数选项，包括损失函数的降维方式
            kwargs_options = [{"reduction": "sum"}, {"reduction": "none"}, {}]

            # 遍历权重选项和参数选项的组合
            for weight, kwargs in self._arg_and_kwarg_options(
                (weight_options,), kwargs_options
            ):
                # 创建随机输入数据和目标数据
                input = torch.rand(shape, device=device)
                target = torch.rand(shape, device=device)
                
                # 利用偏函数创建二元交叉熵损失函数
                fn = functools.partial(
                    torch.nn.functional.binary_cross_entropy,
                    target=target,
                    weight=weight,
                    **kwargs,
                )
                
                # 计算损失函数的结果
                result = fn(input)
                
                # 创建与结果相同形状的随机梯度张量
                cotangents = torch.randn_like(result, device=device)
                
                # 比较损失函数的向量-雅可比积分（VJP）的雅可比矩阵
                self._compare_jacobians_of_vjp(
                    fn, (cotangents, input), atol_rtol=(1e-4, 2e-5)
                )

    # 定义一个测试函数，用于测试极端数值情况下的层归一化函数
    def test_extremal_numerics_layer_norm(self, device):
        # 定义不同维度的输入形状
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        
        # 遍历不同的输入形状
        for shape in shapes:
            # 创建极端输入的输入选项
            input_options = self._make_extremal_inputs(shape, device)
            # 根据输入形状的维度创建归一化维度
            normalized_shape = shape[1:]
            # 创建极端输入的权重和偏置选项
            weight_options = self._make_extremal_inputs(normalized_shape, device)
            bias_options = self._make_extremal_inputs(normalized_shape, device)

            # 遍历输入、偏置和权重选项的组合
            for input, bias, weight in self._arg_and_kwarg_options(
                (input_options, bias_options, weight_options), ()
            ):
                # 定义一个函数，应用层归一化到输入数据
                def fn(input, weight, bias):
                    return torch.nn.functional.layer_norm(
                        input, normalized_shape, weight=weight, bias=bias
                    )

                # 计算层归一化的结果
                result = fn(input, weight, bias)
                
                # 创建与结果相同形状的随机梯度张量
                cotangents = torch.randn_like(result, device=device)
                
                # 比较层归一化函数的VJP的雅可比矩阵
                self._compare_jacobians_of_vjp(fn, (cotangents, input, weight, bias))

    # 应用装饰器，在执行测试之前关闭 TF32 模式
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # 指定允许的数据类型，包括 torch.float32 和 torch.double
    @ops(
        op_db + additional_op_db + autograd_function_db,
        allowed_dtypes=(torch.float32, torch.double),
    )
    # 装饰器函数，用于跳过指定测试用例
    @skipOps(
        "TestOperators",  # 跳过整个 "TestOperators" 测试组
        "test_vmap_autograd_grad",  # 跳过 "test_vmap_autograd_grad" 测试
        {
            # 以下是具体要跳过的测试函数及其相关描述
            xfail("masked_select"),  # 对 "masked_select" 函数的测试预期失败
            xfail("nn.functional.max_unpool2d", "grad"),  # 对 "nn.functional.max_unpool2d" 函数的 "grad" 模式测试预期失败
            xfail("nn.functional.max_unpool2d"),  # 对 "nn.functional.max_unpool2d" 函数的测试预期失败
            xfail("to_sparse"),  # 对 "to_sparse" 函数的测试预期失败，由于分派键问题
            xfail("torch.ops.aten._efficient_attention_forward"),  # 对 "torch.ops.aten._efficient_attention_forward" 函数的测试预期失败，因为输出是整数
            decorate("xlogy", decorator=skipIfRocm),  # 对 "xlogy" 函数的测试装饰为在 ROCm 环境下跳过
            skip(
                "matrix_exp", dtypes=(torch.float32,), device_type="cuda"
            ),  # 对 "matrix_exp" 函数在指定条件下的测试跳过，如在 CUDA 设备上使用 float32 类型
            skip(
                "ldexp", dtypes=(torch.float32,), device_type="cpu"
            ),  # 对 "ldexp" 函数在指定条件下的测试跳过，如在 CPU 上使用 float32 类型
            skip("__rmatmul__"),  # 对 "__rmatmul__" 函数的测试跳过，由于不稳定性需要进一步调查
            skip("matmul"),  # 对 "matmul" 函数的测试跳过，由于不稳定性需要进一步调查
            skip("nn.functional.conv_transpose3d"),  # 对 "nn.functional.conv_transpose3d" 函数的测试跳过，由于不稳定性需要进一步调查
            skip("nn.functional.conv_transpose2d"),  # 对 "nn.functional.conv_transpose2d" 函数的测试跳过，由于不稳定性需要进一步调查
            skip("nn.functional.conv_transpose1d"),  # 对 "nn.functional.conv_transpose1d" 函数的测试跳过，由于不稳定性需要进一步调查
            skip(
                "nn.functional.layer_norm", dtypes=(torch.float32,), device_type="cpu"
            ),  # 对 "nn.functional.layer_norm" 函数在指定条件下的测试跳过，如在 CPU 上使用 float32 类型
            skip(
                "linalg.lu_factor", dtypes=(torch.float32,), device_type="cuda"
            ),  # 对 "linalg.lu_factor" 函数在指定条件下的测试跳过，如在 CUDA 设备上使用 float32 类型
            skip(
                "linalg.lu_factor_ex", dtypes=(torch.float32,), device_type="cuda"
            ),  # 对 "linalg.lu_factor_ex" 函数在指定条件下的测试跳过，如在 CUDA 设备上使用 float32 类型
            skip("linalg.multi_dot", "", device_type="cpu"),  # 对 "linalg.multi_dot" 函数在指定条件下的测试跳过，如在 CPU 上的测试
            skip("sparse.sampled_addmm", ""),  # 对 "sparse.sampled_addmm" 函数的测试跳过
            skip("sparse.mm", "reduce"),  # 对 "sparse.mm" 函数的 "reduce" 模式测试跳过
            skip("native_layer_norm", "", device_type="cpu"),  # 对 "native_layer_norm" 函数在指定条件下的测试跳过，如在 CPU 上的测试
            decorate(
                "_batch_norm_with_update",
                decorator=expectedFailureIf(TEST_WITH_ROCM),  # 对 "_batch_norm_with_update" 函数的测试装饰为在 ROCm 环境下预期失败
                device_type="cuda",
            ),  # 对 "_batch_norm_with_update" 函数在 CUDA 设备上的测试装饰为在 ROCm 环境下预期失败
        },
    )
    # 定义一个装饰器函数，用于设置操作符容差的覆盖值
    @opsToleranceOverride(
        # 第一个参数是测试操作符的名称
        "TestOperators",
        # 第二个参数是测试用例的名称
        "test_vmap_autograd_grad",
        # 第三个参数是一个元组，包含不同操作符的容差设置
        (
            # 第一个元组项：针对 "linalg.householder_product" 操作符的容差设置
            tol1(
                "linalg.householder_product",
                # 设置在 CUDA 设备上的容差值，类型为 torch.float32
                {torch.float32: tol(atol=5e-04, rtol=9e-03)},
                device_type="cuda",
            ),
            # 第二个元组项：针对 "linalg.householder_product" 操作符的容差设置
            tol1(
                "linalg.householder_product",
                # 设置在 CPU 设备上的容差值，类型为 torch.float32
                {torch.float32: tol(atol=1e-04, rtol=1e-04)},
                device_type="cpu",
            ),
            # 第三个元组项：针对 "linalg.multi_dot" 操作符的容差设置
            tol1(
                "linalg.multi_dot",
                # 设置在 CUDA 设备上的容差值，类型为 torch.float32
                {torch.float32: tol(atol=2e-04, rtol=1e-04)},
                device_type="cuda",
            ),
            # 第四个元组项：针对 "linalg.pinv" 操作符的容差设置，要求是 "hermitian"
            tol2(
                "linalg.pinv",
                "hermitian",
                # 设置在 torch.float32 类型上的容差值
                {torch.float32: tol(atol=5e-06, rtol=5e-06)},
            ),
            # 第五个元组项：针对 "nn.functional.conv3d" 操作符的容差设置
            tol1(
                "nn.functional.conv3d",
                # 设置在 torch.float32 类型上的容差值
                {torch.float32: tol(atol=5e-04, rtol=9e-03)},
            ),
            # 第六个元组项：针对 "svd_lowrank" 操作符的容差设置
            tol1(
                "svd_lowrank",
                # 设置在 torch.float32 类型上的容差值
                {torch.float32: tol(atol=5e-05, rtol=5e-05)},
            ),
            # 第七个元组项：针对 "pca_lowrank" 操作符的容差设置
            tol1(
                "pca_lowrank",
                # 设置在 torch.float32 类型上的容差值
                {torch.float32: tol(atol=5e-05, rtol=5e-05)},
            ),
        ),
    )
    def test_vmap_autograd_grad(self, device, dtype, op):
        # 定义一个函数，用于检查输入是否可微
        def is_differentiable(inp):
            return isinstance(inp, Tensor) and (
                inp.grad_fn is not None or inp.requires_grad
            )

        # 获取树结构中所有可微分的叶子节点
        def get_flat_differentiable(tree):
            flattened = pytree.tree_leaves(tree)
            return tuple(i for i in flattened if is_differentiable(i))

        # 获取两个列表中可微分的配对项
        def get_differentiable_linked(list1, list2):
            # 将两个列表按位置配对
            paired_list = zip(list1, list2)
            # 筛选出可微分的配对项
            paired_list = tuple(
                (first, second)
                for (first, second) in paired_list
                if is_differentiable(first)
            )
            # 返回两个列表中可微分项的配对
            return zip(*paired_list)

        # 过滤掉为 None 的项
        def filter_none(out):
            flattened = pytree.tree_leaves(out)
            return tuple(o for o in flattened if o is not None)

        # 如果操作不支持自动求导，则跳过测试
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        # 获取操作的样本输入，确保其需要梯度
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)

        # 对于每个样本输入
        for sample_input in sample_inputs:
            # 根据操作和样本输入，标准化输入输出
            fn, primals = normalize_op_input_output(op, sample_input)
            # 调用操作函数，计算输出
            out = fn(*primals)
            # 对输出进行树映射，应用随机生成的共切向量
            cotangents = tree_map(torch.randn_like, out)

            # 定义计算梯度的函数
            def compute_grad(cotangents):
                # 展平输出和共切向量
                out_flattened = out
                cotangents_flattened = cotangents
                if not isinstance(out_flattened, torch.Tensor):
                    out_flattened = pytree.tree_leaves(out)
                    cotangents_flattened = pytree.tree_leaves(cotangents)
                    # 获取可微分项的链接
                    out_flattened, cotangents_flattened = get_differentiable_linked(
                        out_flattened, cotangents_flattened
                    )

                # 使用 PyTorch 的自动求导计算梯度
                return filter_none(
                    torch.autograd.grad(
                        out_flattened,
                        get_flat_differentiable(primals),
                        cotangents_flattened,
                        retain_graph=True,
                        allow_unused=True,
                    )
                )

            # 检查是否为批量归一化且处于训练状态
            is_batch_norm_and_training = is_batch_norm_training(op, sample_input.kwargs)
            # 获取回退和 vmap 的全面生成器
            generator = get_fallback_and_vmap_exhaustive(
                compute_grad,
                (cotangents,),
                {},
                is_batch_norm_and_training=is_batch_norm_and_training,
            )
            # 对于生成器中的每个循环输出和批量输出，断言它们相等
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)
    def test_vmapvmapjvp_linalg_solve(self):
        # 从操作数据库中筛选出名称为 "linalg.solve" 的操作
        ops = [op for op in op_db if op.name == "linalg.solve"]
        # 断言找到的操作数量大于0
        assert len(ops) > 0

        # 下面的代码从 get_fallback_and_vmap_exhaustive 测试中大量专门化。
        # 如果需要更通用的方法，可能需要进行重构。

        # 设置 B0 和 B1 的值
        B0 = 2
        B1 = 3

        # 我们希望检查当 A 被 jvp 视为连续时，但在 vmap 调用期间会变为不连续的情况。
        # 这将在两个级别的 vmap 过程中发生。
        A = torch.randn(4, 4)
        k = torch.randn(4, 5, B1, B0)

        # 调用 get_jvp_variant_primals_tangents 函数，获取处理 torch.linalg.solve 的函数和参数
        fn, args = get_jvp_variant_primals_tangents(
            torch.linalg.solve, SampleInput(A, args=(k,))
        )

        # 定义输入维度为 (None, -1, None, -1)
        in_dims_all = (None, -1, None, -1)

        # 对 fn 使用两层 vmap，其中 in_dims_all 是输入维度的规格
        batched_out = vmap(vmap(fn, in_dims=in_dims_all), in_dims=in_dims_all)(*args)

        # 调用 loop2 函数，使用 fn 和输入维度规格计算输出，以及 B0 和 B1 的值
        loop_out = loop2(fn, in_dims_all, in_dims_all, 0, 0, B0, B1, *args)

        # 断言两种计算方式的输出应该相等
        self.assertEqual(loop_out, batched_out)

    @ops(
        # 从操作数据库中筛选出名称在 aliasing_ops 中的操作，并合并 op_db 和 additional_op_db 的结果
        filter(lambda op: op.name in aliasing_ops, op_db + additional_op_db),
        # 指定允许的数据类型为 torch.float
        allowed_dtypes=(torch.float,),
    )
    # 参数化测试，grad_op 参数取值为 "jvp" 或 "vjp"
    @parametrize("grad_op", ["jvp", "vjp"])
    def test_view_then_inplace(self, device, dtype, op, grad_op):
        # 遍历操作 op 的样本输入
        for sample_input in op.sample_inputs(device, dtype):

            def f(x):
                # 执行操作 op 的不带梯度的版本，并将结果复制到 x 中
                op(sample_input.input, *sample_input.args, **sample_input.kwargs).copy_(
                    x
                )
                return x

            # 执行操作 op 的不带梯度的版本
            without_grad = op(
                sample_input.input, *sample_input.args, **sample_input.kwargs
            )

            if grad_op == "jvp":
                # 断言在执行 jvp 操作时会抛出 RuntimeError 异常，提示尝试调用原位操作
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    jvp(
                        f,
                        (torch.randn_like(without_grad),),
                        (torch.randn_like(without_grad),),
                    )
            else:
                # 对于 grad_op == "vjp" 的情况，断言会抛出 RuntimeError 异常，提示尝试调用原位操作
                assert grad_op == "vjp"
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    vjp(f, torch.randn_like(without_grad))

    @ops(
        # 从操作数据库中筛选出名称在 aliasing_ops_list_return 中的操作，并合并 op_db 和 additional_op_db 的结果
        filter(
            lambda op: op.name in aliasing_ops_list_return, op_db + additional_op_db
        ),
        # 指定允许的数据类型为 torch.float
        allowed_dtypes=(torch.float,),
    )
    # 参数化测试，grad_op 参数取值为 "jvp" 或 "vjp"
    @parametrize("grad_op", ["jvp", "vjp"])
    def test_view_then_inplace_list_return(self, device, dtype, op, grad_op):
        # 遍历操作的样本输入
        for sample_input in op.sample_inputs(device, dtype):

            def f(x):
                # 执行操作，并在第一个元素上进行就地操作（in-place），复制 x 的值
                op(sample_input.input, *sample_input.args, **sample_input.kwargs)[
                    0
                ].copy_(x)
                return x

            # 执行操作，获取没有梯度的结果
            without_grad = op(
                sample_input.input, *sample_input.args, **sample_input.kwargs
            )[0]
            # 断言在执行梯度操作时会抛出 RuntimeError，指示试图调用就地操作
            with self.assertRaisesRegex(
                RuntimeError, "During a grad .* attempted to call in-place operation"
            ):
                if grad_op == "jvp":
                    # 如果是 jvp 梯度操作，使用 jvp 函数
                    jvp(
                        f,
                        (torch.randn_like(without_grad),),
                        (torch.randn_like(without_grad),),
                    )
                else:
                    assert grad_op == "vjp"
                    # 如果是 vjp 梯度操作，使用 vjp 函数
                    vjp(f, torch.randn_like(without_grad))

    @parametrize("grad_op", ["jvp", "vjp"])
    def test_view_then_inplace_special(self, grad_op):
        # 一些 __getitem__ 中使用了 at::index，它不会别名，因此这里测试了一些会别名的情况
        ops = [
            lambda x: x[0],
            lambda x: x[0, 0, 0],
            lambda x: x[:1],
            lambda x: x[:, :1],
            lambda x: x[:, :1, :],
        ]

        # 遍历操作列表
        for op in ops:

            def f(x):
                # 执行操作，并在 captured 上进行就地操作（in-place），复制 x 的值
                op(captured).copy_(x)
                return x

            captured = torch.randn(4, 3, 3)
            # 执行操作，获取没有梯度的结果
            without_grad = op(captured)
            # 根据梯度操作类型断言会抛出 RuntimeError，指示试图调用就地操作
            if grad_op == "jvp":
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    jvp(
                        f,
                        (torch.randn_like(without_grad),),
                        (torch.randn_like(without_grad),),
                    )
            else:
                assert grad_op == "vjp"
                with self.assertRaisesRegex(
                    RuntimeError,
                    "During a grad .* attempted to call in-place operation",
                ):
                    vjp(f, torch.randn_like(without_grad))

    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # NOTE: [three-transform testing]
    # We only test the autograd_function_db tests here.
    #
    # Usually testing the composition of two transforms is sufficient to convince
    # ourselves that an operator is correctly implemented. For the following cases,
    # we want to be extra sure, so we send those through some three-transform tests:
    # - autograd.Function. The mechanism is via PyDispatcher/HigherOrderOperator, not the
    #   regular PyTorch dispatcher, so it's good to exercise more caution.
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    # 使用 @skipOps 装饰器跳过指定测试用例
    @skipOps(
        "TestOperators",  # 跳过名为 TestOperators 的测试类
        "test_vmapvjpvmap",  # 跳过名为 test_vmapvjpvmap 的测试方法
        {  # 跳过带有以下 xfail 标记的测试条件
            xfail("NumpyCubeNotComposableAutogradFunction"),  # 不可组合的 NumpyCubeNotComposableAutogradFunction
        },
    )
    # 定义测试方法 test_vmapvjpvmap，接受 device、dtype、op 参数
    def test_vmapvjpvmap(self, device, dtype, op):
        # 使用 op 的 sample_inputs 方法生成带有梯度的样本数据集 samples
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2  # 设置批处理大小为 2
        # 遍历每个样本
        for sample in samples:
            # 构建参数列表 args，包含 sample.input 和 sample.args
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs  # 获取 kwargs
            # 生成批处理输入数据，使用 generate_vmap_inputs 函数
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            # 遍历生成器产生的每个批次的参数、输入维度、kwargs
            for batched_args, in_dims, kwargs in generator:
                # 创建 inner_vmapped_op 和 inner_mapped_op 函数
                inner_vmapped_op = vmap(op, in_dims)
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                # 规范化操作的输入输出，获取 inner_vmapped_fn 和 primals
                inner_vmapped_fn, primals = normalize_op_input_output2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )
                # 规范化操作的输入输出，获取 inner_mapped_fn 和 _
                inner_mapped_fn, _ = normalize_op_input_output2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )
                # 执行 inner_mapped_fn 并获取结果
                result = inner_mapped_fn(*primals)
                # 使用 tree_map 创建 cotangents，其中 lambda 函数生成与 result 相同形状的随机张量
                cotangents = tree_map(lambda x: torch.rand_like(x), result)

                # 定义 apply_vjp 函数，生成处理 vjp 的函数
                def apply_vjp(fn):
                    def inner(primals, cotangents):
                        _, vjp_fn = vjp(fn, *primals)  # 调用 vjp 获取函数及其导数函数
                        return vjp_fn(cotangents)  # 返回导数函数应用于 cotangents 的结果

                    return inner

                # 生成 vjpvmap_fn 和 vjpmap_fn 函数，应用 apply_vjp 函数
                vjpvmap_fn = apply_vjp(inner_vmapped_fn)
                vjpmap_fn = apply_vjp(inner_mapped_fn)
                batched_args = (primals, cotangents)  # 更新批处理参数为 primals 和 cotangents
                generator = generate_vmap_inputs(batched_args, {})

                # 遍历批处理生成器产生的每个批次的参数、输入维度、_
                for batched_args, in_dims, _ in generator:
                    # 策略：比较 vmap(vjp(vmap(op)) vs map(vjp(map(op))
                    vmapvjpvmap_fn = vmap(vjpvmap_fn, in_dims)
                    mapvjpmap_fn = functools.partial(loop, vjpmap_fn, in_dims, 0, B)

                    # 执行 vmapvjpvmap_fn 和 mapvjpmap_fn 函数
                    result = vmapvjpvmap_fn(*batched_args)
                    expected = mapvjpmap_fn(*batched_args)
                    # 使用 self.assertEqual 断言结果与期望值相等
                    self.assertEqual(result, expected)

    # See NOTE: [three-transform testing]
    # 使用 @ops 装饰器注册自动求导函数数据库 autograd_function_db 中的测试
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    # 使用 @skipOps 装饰器跳过指定测试用例
    @skipOps(
        "TestOperators",  # 跳过名为 TestOperators 的测试类
        "test_vjpvmapvmap",  # 跳过名为 test_vjpvmapvmap 的测试方法
        {  # 跳过带有以下 xfail 标记的测试条件
            xfail("NumpyCubeNotComposableAutogradFunction"),  # 不可组合的 NumpyCubeNotComposableAutogradFunction
        },
    )
    # 定义一个测试方法 test_vjpvmapvmap，用于测试指定操作 op 在不同输入条件下的向量化映射效果
    def test_vjpvmapvmap(self, device, dtype, op):
        # 使用 op 的方法 sample_inputs 生成一组样本数据，设备为 device，数据类型为 dtype，要求梯度计算
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2  # 定义批处理大小为 2
        # 遍历样本数据
        for sample in samples:
            # 构建参数列表，包括 sample.input 和 sample.args 的元素，并转换为列表
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs  # 获取样本的关键字参数
            # 使用 generate_vmap_inputs 生成输入参数的向量化映射生成器，批处理大小为 B
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            # 遍历向量化映射生成器生成的每个批次参数、内部输入维度和关键字参数
            for batched_args, inner_in_dims, kwargs in generator:
                # 使用 vmap 对操作 op 进行内部的向量化映射
                inner_vmapped_op = vmap(op, inner_in_dims)
                # 使用 functools.partial 部分应用 loop 函数，对内部映射后的操作 op 进行映射
                inner_mapped_op = functools.partial(loop, op, inner_in_dims, 0, B)
                # 使用 generate_vmap_inputs 生成新的向量化映射生成器，处理批次参数和关键字参数
                generator = generate_vmap_inputs(batched_args, kwargs)
                # 再次遍历生成器生成的每个批次参数、输入维度和关键字参数
                for batched_args, in_dims, kwargs in generator:
                    # 策略：比较 vjp(vmap(vmap(op)) 与 vjp(map(map(op)) 的效果
                    # 对内部向量化映射后的操作 inner_vmapped_op 进行最终的向量化映射
                    vmapped_op = vmap(inner_vmapped_op, in_dims)
                    # 使用 functools.partial 部分应用 loop 函数，对内部映射后的操作 inner_mapped_op 进行映射
                    mapped_op = functools.partial(loop, inner_mapped_op, in_dims, 0, B)

                    # 对 vmapped_op 进行规范化，获取其规范化后的函数和原始输入输出
                    vmapped_fn, primals = normalize_op_input_output2(
                        vmapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )
                    # 对 mapped_op 进行规范化，获取其规范化后的函数和原始输入输出
                    mapped_fn, _ = normalize_op_input_output2(
                        mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )

                    # 执行 mapped_fn 函数，传入 primals 作为参数，并生成结果
                    result = mapped_fn(*primals)
                    # 使用 tree_map 生成与结果相同形状的随机张量，作为 cotangents
                    cotangents = tree_map(lambda x: torch.rand_like(x), result)

                    # 调用 vjp 函数，获取 mapped_fn 函数的梯度函数
                    _, vjp_fn = vjp(mapped_fn, *primals)
                    # 计算预期的 vjps，使用 cotangents 作为参数
                    expected_vjps = vjp_fn(cotangents)

                    # 调用 vjp 函数，获取 vmapped_fn 函数的梯度函数
                    _, vjp_fn = vjp(vmapped_fn, *primals)
                    # 计算结果的 vjps，使用 cotangents 作为参数
                    result_vjps = vjp_fn(cotangents)

                    # 断言：验证 result_vjps 是否等于 expected_vjps
                    self.assertEqual(result_vjps, expected_vjps)

    # 查看注意事项 [three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_vjpvjpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # 无法组合
        },
    )
    # 定义测试方法 test_vjpvjpvmap，用于测试 vjp 和 vmap 的组合效果
    def test_vjpvjpvmap(self, device, dtype, op):
        # 生成操作的样本输入，要求梯度可计算
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2  # 定义批大小为 2
        # 遍历每个样本输入
        for sample in samples:
            # 准备参数列表，包括输入和其他参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 生成 vmap 输入的生成器
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            # 遍历生成器产生的每批参数、输入维度和关键字参数
            for batched_args, in_dims, kwargs in generator:
                # 对操作进行 vmap 化，根据输入维度创建内部 vmap 操作
                inner_vmapped_op = vmap(op, in_dims)
                # 部分应用循环函数，生成内部映射操作
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                # 获取 vjpfull 变体2 的函数和参数
                vjpmap_fn, args = get_vjpfull_variant2(
                    inner_mapped_op, batched_args, kwargs
                )
                # 获取 vjpfull 变体2 的函数和参数
                vjpvmap_fn, _ = get_vjpfull_variant2(
                    inner_vmapped_op, batched_args, kwargs
                )

                # 获取 vjpfull 变体2 的函数和参数，针对 vjpvmap_fn
                vjpvjpvmap_fn, new_args = get_vjpfull_variant2(vjpvmap_fn, args, {})
                # 获取 vjpfull 变体2 的函数和参数，针对 vjpmap_fn
                vjpvjpmap_fn, _ = get_vjpfull_variant2(vjpmap_fn, args, {})

                # 计算预期结果
                expected = vjpvjpmap_fn(*new_args)
                # 计算实际结果
                result = vjpvjpvmap_fn(*new_args)
                # 断言实际结果与预期结果相等
                self.assertEqual(result, expected)

    # 我们一般确信 jvp x vmap 的工作方式（vmap 将一个操作符转换为另一个操作符，并测试操作符的 jvp 支持）。因此，我们仅在我们不确定的情况下进行测试：
    # - autograd.Function <> functorch 交互
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_jvpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # 不可组合
        },
    )
    # 定义测试方法 test_jvpvmap，用于测试 jvp 和 vmap 的组合效果
    def test_jvpvmap(self, device, dtype, op):
        # 生成操作的样本输入，要求梯度可计算
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        B = 2  # 定义批大小为 2
        # 遍历每个样本输入
        for sample in samples:
            # 准备参数列表，包括输入和其他参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 生成 vmap 输入的生成器
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            # 遍历生成器产生的每批参数、输入维度和关键字参数
            for batched_args, in_dims, kwargs in generator:
                # 对操作进行 vmap 化，根据输入维度创建内部 vmap 操作
                inner_vmapped_op = vmap(op, in_dims)
                # 部分应用循环函数，生成内部映射操作
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                # 获取 jvpfull 变体2 的函数和参数
                jvpvmap_op, primals = get_jvp_variant_primals_tangents2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )
                # 获取 jvpfull 变体2 的函数和参数
                jvpmap_op, _ = get_jvp_variant_primals_tangents2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )

                # 计算预期结果
                expected = jvpmap_op(*primals)
                # 计算实际结果
                result = jvpvmap_op(*primals)
                # 断言实际结果与预期结果相等
                self.assertEqual(result, expected)

    # 见注释: [three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_jvpvmapvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # 不可组合
        },
    )
    # 定义测试方法，用于测试 jvpvmapvmap 函数
    def test_jvpvmapvmap(self, device, dtype, op):
        # 从操作 op 获取样本输入数据，设备为 device，数据类型为 dtype，并设置为需要梯度计算
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        # 设置批处理大小 B 为 2
        B = 2
        # 对于每个样本数据进行循环处理
        for sample in samples:
            # 构建参数列表，包括样本输入和其他参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 生成 vmap 输入的生成器，用于生成批处理的参数和维度信息
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            # 遍历生成器中的批处理参数、内部输入维度和关键字参数
            for batched_args, inner_in_dims, kwargs in generator:
                # 使用 vmap 对操作 op 进行内部 vmap 映射，得到新的操作对象
                inner_vmapped_op = vmap(op, inner_in_dims)
                # 使用 functools.partial 创建内部映射操作对象
                inner_mapped_op = functools.partial(loop, op, inner_in_dims, 0, B)
                # 重新生成 vmap 输入的生成器，用于下一级映射操作
                generator = generate_vmap_inputs(batched_args, kwargs)
                # 遍历生成器中的批处理参数、输入维度和关键字参数
                for batched_args, in_dims, kwargs in generator:
                    # strategy: compare jvp(vmap(vmap(op)) vs jvp(map(map(op))
                    # 对内部 vmap 映射操作对象进行 vmap 映射，得到最终的 vmap 操作对象
                    vmapped_op = vmap(inner_vmapped_op, in_dims)
                    # 使用 functools.partial 创建映射操作对象
                    mapped_op = functools.partial(loop, inner_mapped_op, in_dims, 0, B)

                    # 调用 get_jvp_variant_primals_tangents2 函数，获取 jvpvmapvmap_fn 函数和原始数据
                    jvpvmapvmap_fn, primals = get_jvp_variant_primals_tangents2(
                        vmapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )
                    # 调用 get_jvp_variant_primals_tangents2 函数，获取 jvpmapmap_fn 函数和原始数据
                    jvpmapmap_fn, _ = get_jvp_variant_primals_tangents2(
                        mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                    )

                    # 计算预期结果
                    expected = jvpmapmap_fn(*primals)
                    # 计算实际结果
                    result = jvpvmapvmap_fn(*primals)
                    # 使用断言验证实际结果与预期结果是否相等
                    self.assertEqual(result, expected)

    # See NOTE: [three-transform testing]
    # 应用装饰器，设置 TF32 禁用状态，解决特定问题
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # 应用操作装饰器，设置自动求导函数数据库和允许的数据类型为 torch.float32
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    # 跳过指定的操作测试
    @skipOps(
        "TestOperators",
        "test_vmapjvpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # 不可组合的情况
        },
    )
    # 定义测试方法 test_vmapjvpvmap，用于测试 vmap、jvp 和其组合的函数
    def test_vmapjvpvmap(self, device, dtype, op):
        # 使用操作 op 的样本输入生成输入样本，要求梯度信息为 True
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        # 固定批量大小为 B = 2
        B = 2
        # 遍历每个样本
        for sample in samples:
            # 将输入样本的 input 和 args 转换为列表，并加入 input，构成参数列表 args
            args = [sample.input] + list(sample.args)
            # 获取 kwargs
            kwargs = sample.kwargs
            # 使用 generate_vmap_inputs 生成输入 generator
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            # 遍历 generator 中的 batched_args、in_dims 和 kwargs
            for batched_args, in_dims, kwargs in generator:
                # 对操作 op 和输入维度 in_dims 进行 vmap 化
                inner_vmapped_op = vmap(op, in_dims)
                # 使用 functools.partial 创建部分应用了 loop 函数的 inner_mapped_op
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                # 获取 jvpvmap_fn 和其原始值 primals
                jvpvmap_fn, primals = get_jvp_variant_primals_tangents2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )
                # 获取 jvpmap_fn 和其原始值（不使用切线）_
                jvpmap_fn, _ = get_jvp_variant_primals_tangents2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )

                # 生成 primals 的 vmap 输入 generator
                generator = generate_vmap_inputs(primals, {})

                # 遍历 generator 中的 batched_args、in_dims 和 _
                for batched_args, in_dims, _ in generator:
                    # 策略：比较 vmap(jvp(vmap(op))) 和 map(jvp(map(op))) 的结果
                    vmapjvpvmap_fn = vmap(jvpvmap_fn, in_dims)
                    mapjvpmap_fn = functools.partial(loop, jvpmap_fn, in_dims, 0, B)

                    # 计算 vmapjvpvmap_fn 的结果
                    result = vmapjvpvmap_fn(*batched_args)
                    # 计算期望结果
                    expected = mapjvpmap_fn(*batched_args)
                    # 使用 self.assertEqual 进行结果断言
                    self.assertEqual(result, expected)

    # 查看笔记：[three-transform testing]
    @ops(autograd_function_db, allowed_dtypes=(torch.float32,))
    @skipOps(
        "TestOperators",
        "test_jvpjvpvmap",
        {
            xfail("NumpyCubeNotComposableAutogradFunction"),  # 不可组合的 NumpyCubeNotComposableAutogradFunction
        },
    )
    # 定义一个测试方法，测试 JVPJVPVMAP 函数
    def test_jvpjvpvmap(self, device, dtype, op):
        # 使用给定的操作对象生成样本输入，要求支持梯度计算
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        # 设置批量大小 B
        B = 2
        # 对于每个样本进行迭代
        for sample in samples:
            # 准备参数列表，包括输入和其他参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 生成 VMAP 输入的生成器，返回批量参数、输入维度和关键字参数
            generator = generate_vmap_inputs(args, kwargs, batch_size=B)
            # 对于生成器中的每组批量参数、输入维度和关键字参数进行迭代
            for batched_args, in_dims, kwargs in generator:
                # 创建内部的 VMAP 操作
                inner_vmapped_op = vmap(op, in_dims)
                # 创建内部的映射操作，使用 functools.partial 创建
                inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

                # 获取内部映射操作的 JVP 变体的原始值和切线值函数，返回新的参数列表
                jvpmap_fn, args = get_jvp_variant_primals_tangents2(
                    inner_mapped_op, batched_args, kwargs, sample.output_process_fn_grad
                )
                # 获取内部 VMAP 操作的 JVP 变体的原始值和切线值函数，返回新的参数列表
                jvpvmap_fn, _ = get_jvp_variant_primals_tangents2(
                    inner_vmapped_op,
                    batched_args,
                    kwargs,
                    sample.output_process_fn_grad,
                )

                # 获取 JVPVMAP 函数的 JVP 变体的原始值和切线值函数，返回新的参数列表
                jvpjvpvmap_fn, new_args = get_jvp_variant_primals_tangents2(
                    jvpvmap_fn, args, {}
                )
                # 获取 JVPMAP 函数的 JVP 变体的原始值和切线值函数，返回新的参数列表
                jvpjvpmap_fn, _ = get_jvp_variant_primals_tangents2(jvpmap_fn, args, {})

                # 计算预期输出
                expected = jvpjvpmap_fn(*new_args)
                # 计算实际输出
                result = jvpjvpvmap_fn(*new_args)
                # 断言实际输出与预期输出相等
                self.assertEqual(result, expected)

# 请参考注释：[three-transform testing]
@ops(autograd_function_db, allowed_dtypes=(torch.float32,))
@skipOps(
    "TestOperators",
    "test_jvpvjpvmap",
    {
        xfail("NumpyCubeNotComposableAutogradFunction"),  # 不可组合的情况
    },
)
# 定义测试 JVPVJPVMAP 函数
def test_jvpvjpvmap(self, device, dtype, op):
    # 使用给定的操作对象生成样本输入，要求支持梯度计算
    samples = op.sample_inputs(device, dtype, requires_grad=True)
    # 设置批量大小 B
    B = 2
    # 对于每个样本进行迭代
    for sample in samples:
        # 准备参数列表，包括输入和其他参数
        args = [sample.input] + list(sample.args)
        kwargs = sample.kwargs
        # 生成 VMAP 输入的生成器，返回批量参数、输入维度和关键字参数
        generator = generate_vmap_inputs(args, kwargs, batch_size=B)
        # 对于生成器中的每组批量参数、输入维度和关键字参数进行迭代
        for batched_args, in_dims, kwargs in generator:
            # 创建内部的 VMAP 操作
            inner_vmapped_op = vmap(op, in_dims)
            # 创建内部的映射操作，使用 functools.partial 创建
            inner_mapped_op = functools.partial(loop, op, in_dims, 0, B)

            # 获取内部映射操作的 VJP 变体的原始值和切线值函数，返回新的参数列表
            vjpmap_fn, args = get_vjpfull_variant2(
                inner_mapped_op, batched_args, kwargs
            )
            # 获取内部 VMAP 操作的 VJP 变体的原始值和切线值函数，返回新的参数列表
            vjpvmap_fn, _ = get_vjpfull_variant2(
                inner_vmapped_op, batched_args, kwargs
            )

            # 获取 VJPVJPMAP 函数的 JVP 变体的原始值和切线值函数，返回新的参数列表
            jvpvjpvmap_fn, new_args = get_jvp_variant_primals_tangents2(
                vjpvmap_fn, args, {}
            )
            # 获取 JVPVJPAMAP 函数的 JVP 变体的原始值和切线值函数，返回新的参数列表
            jvpvjpmap_fn, _ = get_jvp_variant_primals_tangents2(vjpmap_fn, args, {})

            # 计算预期输出
            expected = jvpvjpmap_fn(*new_args)
            # 计算实际输出
            result = jvpvjpvmap_fn(*new_args)
            # 断言实际输出与预期输出相等
            self.assertEqual(result, expected)
    # 在设备上执行数据写入错误测试
    def test_data_write_errors_under_transform(self, device):
        # 创建一个形状为 (3, 3) 的张量 t，放置在指定的设备上
        t = torch.randn(3, 3, device=device)

        # 定义一个函数 fn，接受一个张量 t 作为参数
        def fn(t):
            # 用另一个形状为 (3, 3) 的随机张量替换 t 的数据部分
            t.data = torch.randn(3, 3)
            # 返回张量 t 的和
            return t.sum()

        # 错误消息字符串，用于断言抛出异常时的比对
        msg = "mutating directly with `.data` inside functorch transform"

        # 断言调用 grad 函数时抛出 RuntimeError 异常，并且异常信息符合 msg
        with self.assertRaisesRegex(RuntimeError, msg):
            grad(fn)(t)

        # 断言调用 vjp 函数时抛出 RuntimeError 异常，并且异常信息符合 msg
        with self.assertRaisesRegex(RuntimeError, msg):
            vjp(fn, t)

        # 断言调用 jvp 函数时抛出 RuntimeError 异常，并且异常信息符合 msg
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(fn, (t,), (torch.randn_like(t),))

    # 测试处理包含标量列表的张量
    def test_tensor_with_scalar_list(self, device):
        # 创建一个形状为 () 的张量 x，放置在指定的设备上
        x = torch.randn((), device=device)

        # 定义一个函数 func_list_of_scalar，接受一个标量 x 作为输入，返回一个张量，包含 x 的值
        def func_list_of_scalar(x):
            return torch.tensor([x], device=device)

        # 定义一个函数 func，接受一个标量 x 作为输入，返回一个张量，形状为 (1,)
        def func(x):
            return torch.tensor(x, device=device).view(1)

        # 使用 vjp 函数计算 func_list_of_scalar 对 x 的值的雅可比积分
        actual_o, actual_fn = vjp(func_list_of_scalar, x)
        # 使用 vjp 函数计算 func 对 x 的值的雅可比积分
        expected_o, expected_fn = vjp(func, x)

        # 断言 actual_o 和 expected_o 的值相等
        self.assertEqual(actual_o, expected_o)
        
        # 断言 actual_fn 和 expected_fn 在传入全为 1 的张量时的返回值相等
        self.assertEqual(
            expected_fn(torch.ones_like(expected_o)),
            actual_fn(torch.ones_like(actual_o)),
        )
# 只允许在以下设备类型上运行测试：CPU 和 CUDA
only_for = ("cpu", "cuda")

# 实例化设备类型相关的测试用例，基于给定的测试类和全局变量，限定仅对指定设备类型运行
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

# 如果这个脚本被直接执行（而不是被导入），则执行测试函数
if __name__ == "__main__":
    run_tests()
```