# `.\pytorch\torch\_functorch\eager_transforms.py`

```py
# 忽略类型检查错误
# 版权声明，版权归 Facebook 及其关联公司所有
# 本源代码使用 BSD 风格许可证授权，许可证可以在源树的根目录下的 LICENSE 文件中找到

# 引入上下文管理模块
import contextlib
# 从 functools 模块导入 partial 和 wraps 函数
from functools import partial, wraps
# 从 typing 模块导入 Any、Callable、List、Optional、Tuple 和 Union 类型
from typing import Any, Callable, List, Optional, Tuple, Union

# 导入 torch 库
import torch
# 导入 torch.autograd.forward_ad 模块作为 fwAD
import torch.autograd.forward_ad as fwAD

# 从 torch._C._functorch 模块导入多个函数
from torch._C._functorch import (
    _assert_wrapped_functional,
    _func_decrement_nesting,
    _func_increment_nesting,
    _grad_decrement_nesting,
    _grad_increment_nesting,
    _jvp_decrement_nesting,
    _jvp_increment_nesting,
    _propagate_functional_input_mutation,
    _unwrap_for_grad,
    _unwrap_functional_tensor,
    _wrap_for_grad,
    _wrap_functional_tensor,
    get_inplace_requires_grad_allowed,
    set_inplace_requires_grad_allowed,
)
# 从 torch._functorch.utils 模块导入 argnums_t 和 exposed_in 函数
from torch._functorch.utils import argnums_t, exposed_in
# 从 torch._subclasses.functional_tensor 模块导入 FunctionalTensor 类
from torch._subclasses.functional_tensor import FunctionalTensor
# 从 torch.fx.experimental 模块导入 const_fold 函数
from torch.fx.experimental import const_fold
# 从 torch.fx.experimental.proxy_tensor 模块导入 make_fx 函数
from torch.fx.experimental.proxy_tensor import make_fx
# 从 torch.utils 模块导入 pytree 模块作为 pytree
from torch.utils import _pytree as pytree
# 从 torch.utils._pytree 模块导入多个函数和类
from torch.utils._pytree import (
    tree_flatten,
    tree_map,
    tree_map_,
    tree_map_only,
    tree_unflatten,
    treespec_pprint,
)
# 从当前模块的 apis 中导入 vmap 函数
from .apis import vmap
# 从当前模块的 vmap 中导入 doesnt_support_saved_tensors_hooks 和 get_chunk_sizes 函数

# 定义 lazy_dynamo_disallow 装饰器函数，导入 torch._dynamo 模块
def lazy_dynamo_disallow(func):
    import torch._dynamo
    return torch._dynamo.disallow_in_graph(func)

# 定义上下文管理器函数 enable_inplace_requires_grad
@contextlib.contextmanager
def enable_inplace_requires_grad(enabled):
    # 获取当前 inplace requires grad 允许状态
    prev_state = get_inplace_requires_grad_allowed()
    # 设置 inplace requires grad 允许状态
    set_inplace_requires_grad_allowed(enabled)
    try:
        # 执行 yield 语句
        yield
    finally:
        # 恢复 inplace requires grad 允许状态到之前的状态
        set_inplace_requires_grad_allowed(prev_state)

# 定义 _vjp_treespec_compare 函数，比较原始输出和余切的树形结构
def _vjp_treespec_compare(primals_out, cotangents):
    # 暂时解开树形结构
    _, primals_out_spec = tree_flatten(primals_out)
    _, cotangents_spec = tree_flatten(cotangents)
    # 由于 Dynamo 无法跟踪操作符.ne，为了绕过此限制，此函数不会被内联
    if primals_out_spec != cotangents_spec:
        raise RuntimeError(
            f"Expected pytree structure of cotangents to be the same "
            f"as pytree structure of outputs to the function. "
            f"cotangents: {treespec_pprint(cotangents_spec)}, "
            f"primal output: {treespec_pprint(primals_out_spec)}"
        )

# 定义 _jvp_treespec_compare 函数，比较原始值和切线的树形结构
def _jvp_treespec_compare(primals, tangents):
    # 暂时解开树形结构
    _, primals_spec = tree_flatten(primals)
    _, tangents_spec = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(
            f"{jvp_str}: Expected primals and tangents to have the same python "
            f"structure. For example, if primals is a tuple of 3 tensors, "
            f"tangents also must be. Got primals with structure {primals_spec} "
            f"and tangents with structure {tangents_spec}"
        )

# 定义 _linearize_treespec_compare 函数，比较原始值和切线的树形结构
def _linearize_treespec_compare(primals, tangents):
    # 当修复了 issue #116264 之后，需要撤销这个修改
    
    # 使用 tree_flatten 函数分别展开 primals 和 tangents，获取它们的参数规范
    _, primals_argspec = tree_flatten(primals)
    _, tangent_argspec = tree_flatten(tangents)
    
    # 检查 tangents 和 primals 的参数规范是否一致，如果不一致则抛出运行时错误
    if tangent_argspec != primals_argspec:
        raise RuntimeError(
            f"Expected the tangents {tangent_argspec} to have "
            f"the same argspec as the primals {primals_argspec}"
        )
# 避免 x.requires_grad_() 调用时破坏计算图
# https://github.com/pytorch/pytorch/pull/110053
def _set_tensor_requires_grad(x):
    return x.requires_grad_()

# 创建可微分对象的辅助函数，对输入进行处理
def _create_differentiable(inps, level=None):
    def create_differentiable(x):
        # 如果 x 是 torch.Tensor 类型
        if isinstance(x, torch.Tensor):
            # 启用就地修改 requires_grad 的上下文管理器
            with enable_inplace_requires_grad(True):
                return _set_tensor_requires_grad(x)
        # 如果不是 Tensor，则抛出 ValueError 异常
        raise ValueError(
            f"Thing passed to transform API must be Tensor, " f"got {type(x)}"
        )

    # 对输入 inps 的每个元素应用 create_differentiable 函数
    return tree_map(create_differentiable, inps)

# 取消创建可微分对象的辅助函数，对输入进行处理
def _undo_create_differentiable(inps, level=None):
    def unwrap_tensors(x):
        # 如果 x 是 torch.Tensor 类型
        if isinstance(x, torch.Tensor):
            return _unwrap_for_grad(x, level)
        # 如果 x 是元组类型，则递归应用 unwrap_tensors 函数
        if isinstance(x, tuple):
            return tree_map(unwrap_tensors, tuple(x))

        # 如果类型不支持，则抛出 RuntimeError 异常
        raise RuntimeError(f"Expected tensors, got unsupported type {type(x)}")

    # 对输入 inps 的每个元素应用 unwrap_tensors 函数
    return tree_map(unwrap_tensors, inps)

# 检查输入是否为可微分对象（torch.Tensor，并且 requires_grad=True）
def _is_differentiable(maybe_tensor):
    if not isinstance(maybe_tensor, torch.Tensor):
        return False
    return maybe_tensor.requires_grad

# 检查输入中是否有任何可微分对象（torch.Tensor 或 torch.Tensor 元组）
def _any_differentiable(tensor_or_tuple_of_tensors):
    # 展平 tensor_or_tuple_of_tensors，获取 flat_args
    flat_args, _ = tree_unflatten(tensor_or_tuple_of_tensors)
    # 检查 flat_args 中是否有任何可微分对象
    return any(tuple(map(_is_differentiable, flat_args)))

# 包装 tensor 对象以准备进行梯度计算
def _wrap_tensor_for_grad(maybe_tensor, level):
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    return _wrap_for_grad(maybe_tensor, level)

# 包装 tensor_pytree 中的所有 tensor 对象以准备进行梯度计算
def _wrap_all_tensors(tensor_pytree, level):
    return tree_map(partial(_wrap_tensor_for_grad, level=level), tensor_pytree)

# 将值 val 转换为元组，如果已经是元组则直接返回
def _as_tuple(val):
    if isinstance(val, tuple):
        return val
    return (val,)

# 用于处理不依赖输入的输出的 autograd.grad 的版本
def _autograd_grad(
    outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
):
    if grad_outputs is None:
        # 选出需要计算梯度的输出
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        # 选择需要计算梯度的输出及其对应的梯度
        result = tuple(
            (out, go) for out, go in zip(outputs, grad_outputs) if out.requires_grad
        )
        if len(result) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*result)
    if len(diff_outputs) == 0:
        # 如果没有需要计算梯度的输出，则返回输入 tensor_pytree 的零梯度
        return tuple(torch.zeros_like(inp) for inp in inputs)
    # 使用 torch.autograd.grad 计算梯度
    grad_inputs = torch.autograd.grad(
        diff_outputs,
        inputs,
        grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    # 处理计算得到的梯度，确保每个输入都有梯度信息
    grad_inputs = tuple(
        torch.zeros_like(inp) if gi is None else gi
        for gi, inp in zip(grad_inputs, inputs)
    )
    return grad_inputs

# NOTE [grad and vjp interaction with no_grad]
#
# def f(x):
#   with torch.no_grad():
#     c = x ** 2
#   return x - c
#
# 考虑在调用 grad 前 enable_grad 的开启/关闭状态
#
# Case 1: enable_grad is on.
# grad(f)(x)
# 在这种情况下，`grad` 应该遵守内部的 torch.no_grad。
#
# Case 2: enable_grad is off
# 在 torch.no_grad() 内部:
#   grad(f)(x)
# 在这种情况下，`grad` 应该遵守内部的 torch.no_grad，但不遵守外部的。
# 这是因为 `grad` 是一个“函数变换”：其结果不应依赖于 `f` 外部上下文管理器的结果。
#
# 这给我们带来了以下期望的行为：
# - （嵌套的）grad 变换必须遵守其内部的 torch.no_grad
# - （嵌套的）grad 变换不应遵守其外部的 torch.no_grad
#
# 要实现这种行为，在进入 grad/vjp 时：
# - 我们保存当前的（“前一个”）is_grad_enabled (*)
# - 我们无条件启用 grad。
#
# 在 DynamicLayerBackFallback 中，当我们临时弹出 `grad` 层时：
# - 如果 grad_mode 被禁用，则不执行任何操作。（有一个 torch.no_grad 活动，所有后续 grad 变换必须遵守它）。
# - 如果 grad_mode 被启用，并且先前的 is_grad_enabled (*) 是 False，则临时恢复先前的 `is_grad_enabled`。
#   这是因为我们正在从外部 no_grad 的 `grad` 进入内部 no_grad 的 `grad` 的边界。
#
# 注意：vjp 具有一些有趣的行为，因为 vjp 的可调用对象可以在不同的 grad_mode 下调用，与前向计算不同...
#
# 注意：前向模式自动微分：前向模式自动微分不遵守 torch.no_grad，但遵守 c10::AutoFwGradMode。
# 我们为我们的 jvp 变换实现了相同的逻辑（如果 FwGradMode 被禁用，则会有特殊处理）。
#
# 如何增加和减少嵌套？我认为我们不能。
@exposed_in("torch.func")
def vjp(func: Callable, *primals, has_aux: bool = False):
    """
    代表向量-Jacobian 乘积，返回一个包含应用于 `primals` 的 `func` 的结果和一个函数的元组，
    该函数在给定 `cotangents` 时计算相对于 `primals` 的反向模式 Jacobian 乘以 `cotangents`。

    Args:
        func (Callable): 接受一个或多个参数的 Python 函数。必须返回一个或多个 Tensor。
        primals (Tensors): `func` 的位置参数，必须都是 Tensor。返回的函数也将计算相对于这些参数的导数。
        has_aux (bool): 表示 `func` 返回一个 `(output, aux)` 元组，其中第一个元素是要求导的函数的输出，
            第二个元素是其他辅助对象，不会被求导。默认值: False。
    # 返回一个 ``(output, vjp_fn)`` 元组，其中包含将 ``func`` 应用于 ``primals`` 的结果，
    # 以及一个计算 ``func`` 相对于所有 ``primals`` 的 VJP（向量雅可比积）的函数。
    # 如果 ``has_aux`` 为 True，则返回一个 ``(output, vjp_fn, aux)`` 元组。
    # 返回的 ``vjp_fn`` 函数将返回每个 VJP 的元组。

    # 在简单情况下使用时，:func:`vjp` 的行为与 :func:`grad` 相同

    # :func:`vjp` 可以支持具有多个输出的函数，通过为每个输出传递相应的余切来实现
    # 示例中展示了一个简单的多输出函数的使用方法

    # :func:`vjp` 可以支持输出为 Python 结构的函数
    # 示例中展示了如何处理输出为字典的情况

    # :func:`vjp` 返回的函数将计算相对于每个 ``primals`` 的偏导数

    # ``primals`` 是 ``f`` 的位置参数。所有 kwargs 使用其默认值
    # 返回调用 `_vjp_with_argnums` 函数的结果，传入参数 `func` 和 `*primals`
    # `has_aux` 参数指示是否存在辅助信息
    return _vjp_with_argnums(func, *primals, has_aux=has_aux)
@contextlib.contextmanager
def grad_increment_nesting():
    try:
        # 调用内部函数 _grad_increment_nesting()，获取梯度增加的层数
        grad_level = _grad_increment_nesting()
        # 使用生成器 yield 传递 grad_level，使其可以在 with 语句块内使用
        yield grad_level
    finally:
        # 在 finally 中调用 _grad_decrement_nesting()，减少梯度嵌套层数
        _grad_decrement_nesting()


def enter_jvp_nesting():
    global JVP_NESTING
    # 调用 _jvp_increment_nesting()，增加 JVP（Jacobian-Vector Product）嵌套层数，并获取当前层数
    jvp_level = _jvp_increment_nesting()
    # 全局变量 JVP_NESTING 增加 1，记录当前 JVP 嵌套层数
    JVP_NESTING += 1
    return jvp_level


def exit_jvp_nesting():
    global JVP_NESTING
    # 调用 _jvp_decrement_nesting()，减少 JVP 嵌套层数
    _jvp_decrement_nesting()
    # 全局变量 JVP_NESTING 减少 1，更新当前 JVP 嵌套层数


@contextlib.contextmanager
def jvp_increment_nesting():
    try:
        # 使用生成器 yield 进入 JVP 嵌套，调用 enter_jvp_nesting() 获取当前 JVP 层级
        yield enter_jvp_nesting()
    finally:
        # 在 finally 中退出 JVP 嵌套，调用 exit_jvp_nesting() 减少 JVP 层级
        exit_jvp_nesting()


@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(
    func: Callable, *primals, argnums: Optional[argnums_t] = None, has_aux: bool = False
):
    # 这个函数与 vjp 函数相同，但是还接受 argnums 参数
    # 所有的参数都与 vjp 函数相同，只是增加了一个参数 argnums
    # argnums (Optional[int or tuple[int]]): 可选参数，指定要计算梯度的参数（用于 vjp）。默认值为 None
    #
    # 警告：用户不应直接调用此函数，应该只调用 vjp 函数。
    # 这只是为了确保传递给 jacrev 但不进行微分的输入得到正确的包装。
    #
    # 注意：所有错误消息都会被产生为如果调用了 vjp 函数，即使这个函数被 jacrev 调用。
    #
    # 返回与 :func:`vjp` 相同的两个元素，但是返回的函数 vjp_fn 返回仅由 argnums 指定的原始元素的 VJP（Jacobian-Vector Products）元组。
    # 使用 grad_increment_nesting() 上下文管理器，增加梯度计算的嵌套级别
    with grad_increment_nesting() as level:
        # 启用 Torch 的梯度计算
        with torch.enable_grad():
            # 将所有张量包装起来，使其可追踪梯度
            primals = _wrap_all_tensors(primals, level)
            # 下面是一个注释，供审阅者参考：
            # 这非常奇怪，但它在 symbolic_convert.py 的断言 "len(self.block_stack) == 1" 中通过了
            # 相同的 "if argnums is None" 在某些情况下会失败
            if not isinstance(argnums, int) and not argnums:
                # 创建可微的 primals
                diff_primals = _create_differentiable(primals, level)
            else:
                # 切片处理 argnums，生成不可变为元组的 diff_primals
                diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                # 使用 tree_map_ 函数，部分应用 _create_differentiable 函数，level 作为参数
                tree_map_(partial(_create_differentiable, level=level), diff_primals)
            # 调用函数 func，传入 primals，得到 primals_out
            primals_out = func(*primals)

            # 如果存在辅助变量
            if has_aux:
                # 确保 primals_out 是一个包含两个元素的元组
                if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                    raise RuntimeError(
                        "vjp(f, *primals): output of function f should be a tuple: (output, aux) "
                        "if has_aux is True"
                    )
                # 将 primals_out 拆分为主输出和辅助输出
                primals_out, aux = primals_out
                # 恢复创建的不可变对象 aux
                aux = _undo_create_differentiable(aux, level)

            # 将 primals_out 展平，获取规范
            flat_primals_out, primals_out_spec = tree_flatten(primals_out)
            # 断言 flat_primals_out 不为空的张量输出
            assert_non_empty_tensor_output(flat_primals_out, "vjp(f, *primals)")
            # 将 diff_primals 展平，获取 primals 的规范
            flat_diff_primals, primals_spec = tree_flatten(diff_primals)
            # 使用 _undo_create_differentiable 函数，恢复创建的不可变对象 primals_out
            results = _undo_create_differentiable(primals_out, level)

            # 遍历 flat_primals_out 中的每一个 primal_out
            for primal_out in flat_primals_out:
                # 确保 primal_out 是 torch.Tensor 类型
                assert isinstance(primal_out, torch.Tensor)
                # 如果 primal_out 是浮点数或复数类型，则继续循环
                if primal_out.is_floating_point() or primal_out.is_complex():
                    continue
                # 否则，抛出运行时错误，指出不支持的数据类型
                raise RuntimeError(
                    "vjp(f, ...): All outputs of f must be "
                    "floating-point or complex Tensors, got Tensor "
                    f"with dtype {primal_out.dtype}"
                )

        # 定义一个包装器函数 wrapper，接受 cotangents 作为参数
        def wrapper(cotangents, retain_graph=True, create_graph=None):
            # 如果 create_graph 为 None，则根据 torch.is_grad_enabled() 来设定
            if create_graph is None:
                create_graph = torch.is_grad_enabled()
            # 将 cotangents 展平，获取 cotangents 的规范
            flat_cotangents, cotangents_spec = tree_flatten(cotangents)
            # 比较 primals_out 和 cotangents 的树结构规范
            _vjp_treespec_compare(primals_out, cotangents)
            # 调用 _autograd_grad 函数，计算梯度并返回结果
            result = _autograd_grad(
                flat_primals_out,
                flat_diff_primals,
                flat_cotangents,
                retain_graph=retain_graph,
                create_graph=create_graph,
            )
            # 使用 tree_unflatten 函数，将结果展开为 primals_spec 的规范
            return tree_unflatten(result, primals_spec)

    # 如果存在辅助变量，返回 results、wrapper 和 aux
    if has_aux:
        return results, wrapper, aux
    # 否则，只返回 results 和 wrapper
    else:
        return results, wrapper
# 确保输入列表 x 的长度为 1
def _safe_zero_index(x):
    assert len(x) == 1
    # 返回列表中唯一的元素
    return x[0]


# jacrev 和 jacfwd 不支持复杂函数
# 辅助函数，用于抛出适当的错误
def error_if_complex(func_name, args, is_input):
    # 将 args 展平为列表
    flat_args = pytree.tree_leaves(args)
    # 遍历展平后的参数列表
    for idx, arg in enumerate(flat_args):
        # 如果参数是 torch.Tensor 类型且数据类型为复数
        if isinstance(arg, torch.Tensor) and arg.dtype.is_complex:
            # 根据 is_input 确定是输入还是输出
            input_or_output = "inputs" if is_input else "outputs"
            # 构造错误信息
            err_msg = (
                f"{func_name}: Expected all {input_or_output} "
                f"to be real but received complex tensor at flattened input idx: {idx}"
            )
            # 抛出运行时错误
            raise RuntimeError(err_msg)


@exposed_in("torch.func")
def jacrev(
    func: Callable,
    argnums: Union[int, Tuple[int]] = 0,
    *,
    has_aux=False,
    chunk_size: Optional[int] = None,
    _preallocate_and_copy=False,
):
    """
    使用反向模式自动微分计算 ``func`` 对于参数（参数）索引 ``argnum`` 的雅可比矩阵

    .. note::
        使用 :attr:`chunk_size=1` 等同于使用 for 循环逐行计算雅可比矩阵，即 :func:`vmap` 的约束不适用。

    Args:
        func (function): 接受一个或多个参数的 Python 函数，其中之一必须是 Tensor，并返回一个或多个 Tensor
        argnums (int or Tuple[int]): 可选，整数或整数元组，指定要相对于其计算雅可比矩阵的参数。默认为 0。
        has_aux (bool): 标志，指示 ``func`` 是否返回一个 ``(output, aux)`` 元组，
            其中第一个元素是要进行微分的函数的输出，第二个元素是不进行微分的辅助对象。默认为 False。
        chunk_size (None or int): 如果为 None（默认），使用最大的块大小（相当于在 vjp 上执行单个 vmap 来计算雅可比矩阵）。
            如果为 1，则使用 for 循环逐行计算雅可比矩阵。
            如果不为 None，则以 :attr:`chunk_size` 行为单位计算雅可比矩阵（相当于在 vjp 上执行多个 vmap）。如果在计算雅可比矩阵时遇到内存问题，请尝试指定非 None 的 chunk_size。
        _preallocate_and_copy (bool): 内部参数，预分配内存并进行复制。默认为 False。

    Returns:
        返回一个函数，该函数接受与 ``func`` 相同的输入，并返回相对于 ``argnums`` 的参数的雅可比矩阵。
        如果 ``has_aux is True``，则返回的函数会返回一个 ``(jacobian, aux)`` 元组，其中 ``jacobian``
        是雅可比矩阵，``aux`` 是 ``func`` 返回的辅助对象。

    基本使用点对点的一元操作将给出对角线数组
    """
    as the Jacobian

        >>> from torch.func import jacrev  # 导入 torch 库中的 jacrev 函数，用于计算雅可比矩阵
        >>> x = torch.randn(5)  # 生成一个大小为 5 的随机张量 x
        >>> jacobian = jacrev(torch.sin)(x)  # 计算 sin 函数在 x 处的雅可比矩阵
        >>> expected = torch.diag(torch.cos(x))  # 生成一个对角矩阵，对角线元素为 x 中每个元素的余弦值
        >>> assert torch.allclose(jacobian, expected)  # 断言计算得到的雅可比矩阵与预期的对角矩阵相近

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacrev  # 导入 torch 库中的 jacrev 函数
        >>> x = torch.randn(5)  # 生成一个大小为 5 的随机张量 x
        >>>
        >>> def f(x):  # 定义一个函数 f，接受输入 x
        >>>   return x.sin()  # 返回 x 的正弦值
        >>>
        >>> def g(x):  # 定义一个函数 g，接受输入 x
        >>>   result = f(x)  # 调用函数 f 计算结果
        >>>   return result, result  # 返回结果的元组
        >>>
        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)  # 计算函数 g 在 x 处的雅可比矩阵以及输出值
        >>> assert torch.allclose(f_x, f(x))  # 断言计算得到的输出值与直接调用函数 f(x) 的结果相近

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacrev, vmap  # 导入 torch 库中的 jacrev 和 vmap 函数
        >>> x = torch.randn(64, 5)  # 生成一个大小为 (64, 5) 的随机张量 x
        >>> jacobian = vmap(jacrev(torch.sin))(x)  # 使用 vmap 和 jacrev 计算 x 中每个样本的 sin 函数的雅可比矩阵
        >>> assert jacobian.shape == (64, 5, 5)  # 断言计算得到的雅可比矩阵的形状为 (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from torch.func import jacrev  # 导入 torch 库中的 jacrev 函数
        >>> def f(x):  # 定义一个函数 f，接受输入 x
        >>>   return x.sin().sum()  # 返回 x 的正弦值的总和
        >>>
        >>> x = torch.randn(5)  # 生成一个大小为 5 的随机张量 x
        >>> hessian = jacrev(jacrev(f))(x)  # 计算函数 f 在 x 处的海森矩阵（二阶导数的雅可比矩阵）
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))  # 断言计算得到的海森矩阵与预期的对角矩阵相近

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacrev  # 导入 torch 库中的 jacrev 函数
        >>> def f(x, y):  # 定义一个函数 f，接受输入 x 和 y
        >>>   return x + y ** 2  # 返回 x + y^2 的结果
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)  # 生成两个大小为 5 的随机张量 x 和 y
        >>> jacobian = jacrev(f, argnums=1)(x, y)  # 计算函数 f 在 (x, y) 处关于第二个参数 y 的雅可比矩阵
        >>> expected = torch.diag(2 * y)  # 生成一个对角矩阵，对角线元素为 2*y
        >>> assert torch.allclose(jacobian, expected)  # 断言计算得到的雅可比矩阵与预期的对角矩阵相近

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacrev  # 导入 torch 库中的 jacrev 函数
        >>> def f(x, y):  # 定义一个函数 f，接受输入 x 和 y
        >>>   return x + y ** 2  # 返回 x + y^2 的结果
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)  # 生成两个大小为 5 的随机张量 x 和 y
        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)  # 计算函数 f 在 (x, y) 处关于两个参数 x 和 y 的雅可比矩阵
        >>> expectedX = torch.diag(torch.ones_like(x))  # 生成一个对角矩阵，对角线元素为 1
        >>> expectedY = torch.diag(2 * y)  # 生成一个对角矩阵，对角线元素为 2*y
        >>> assert torch.allclose(jacobian[0], expectedX)  # 断言计算得到的第一个雅可比矩阵与预期的对角矩阵相近
        >>> assert torch.allclose(jacobian[1], expectedY)  # 断言计算得到的第二个雅可比矩阵与预期的对角矩阵相近
    if not (chunk_size is None or chunk_size > 0):
        # 如果 `chunk_size` 不是 None 且大于 0，否则抛出数值错误异常
        raise ValueError("jacrev: `chunk_size` should be greater than 0.")

    # Dynamo 不支持 HOP 组合，如果它们的内部函数使用 @functools.wraps(...) 注解。
    # 我们通过仅在不使用 Dynamo 进行跟踪时应用 wraps 来绕过此问题。
    if not torch._dynamo.is_compiling():
        # 如果不是在 Dynamo 编译状态下，将 func 函数应用 wraps 包装到 wrapper_fn 中
        wrapper_fn = wraps(func)(wrapper_fn)

    # 返回经过包装的 wrapper_fn 函数
    return wrapper_fn
# NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
#
# Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
# It turns out we can compute the jacobian of this function with a single
# call to autograd.grad by using vmap over the correct grad_outputs.
#
# Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
# into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
#
# To get the first row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
# To get the 2nd row of the jacobian, we call
# >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
# and so on.
#
# Using vmap, we can vectorize all 4 of these computations into one by
# passing the standard basis for R^4 as the grad_output.
# vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
#
# Now, how do we compute the jacobian *without stacking the output*?
# We can just split the standard basis across the outputs. So to
# compute the jacobian of f(x), we'd use
# >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
# The grad_outputs looks like the following:
# ( torch.tensor([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 1],
#                 [0, 0, 0]]),
#   torch.tensor([[0],
#                 [0],
#                 [0],
#                 [1]]) )
#
# But we're not done yet!
# >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
# returns a Tensor of shape [4, 3]. We have to remember to split the
# jacobian of shape [4, 3] into two:
# - one of shape [3, 3] for the first output
# - one of shape [   3] for the second output


def _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
    # This function:
    # - constructs a N=sum(tensor_numels) standard basis. i.e. an NxN identity matrix.
    # - Splits the identity matrix into chunks with each chunk size determined by `tensor_numels`.
    # - Each chunk corresponds to one tensor. The chunk has the same dtype and
    #   device as the tensor
    #
    # For example, with tensor_numels = [1, 2, 1], this function returns:
    # ( tensor([[1],     tensor([[0, 0],      tensor([[0],
    #           [0],             [1, 0],              [0],
    #           [0],             [0, 1],              [0],
    #           [0]])  ,         [0, 0]])  ,          [1]])  )
    #
    # Precondition: tensor_numels == tuple(tensor.numel() for tensor in tensors)
    # Precondition: tensors always has at least one element.
    #
    # See NOTE: [Computing jacobian with vmap and grad for multiple tensors]
    # for context behind this function.
    # NOTE: Argument `chunk_size` is used to generate chunked basis instead of
    #       one huge basis matrix. `chunk_size` dictates the maximum size of the
    #       basis matrix along dim=0.
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    assert chunk_size is None or chunk_size > 0
    total_numel = sum(tensor_numels)
    # 如果 chunk_size 存在且小于 total_numel，则计算各个 chunk 的大小
    if chunk_size and chunk_size < total_numel:
        chunk_numels = get_chunk_sizes(total_numel, chunk_size)
    else:  # 如果 chunk_size 为 None 或者大于等于 total_numel，则设定 chunk_size 为 total_numel，并且将其作为唯一的 chunk 大小
        chunk_size = total_numel
        chunk_numels = [total_numel]

    # 计算每个张量对角线开始的索引
    diag_start_indices = (
        0,  # 第一个张量对角线开始的索引为 0
        *torch.tensor(tensor_numels).cumsum(dim=0)[:-1].neg().unbind(),  # 其他张量对角线开始的索引
    )

    # 对每个 chunk 进行迭代
    for chunk_idx, total_numel in enumerate(chunk_numels):
        # 创建与输入张量相同数量和形状的全零张量作为 chunk
        chunks = tuple(
            tensor.new_zeros(total_numel, tensor_numel)
            for tensor, tensor_numel in zip(tensors, tensor_numels)
        )

        # 对每个 chunk 和对应的对角线开始索引填充 1
        for chunk, diag_start_idx in zip(chunks, diag_start_indices):
            chunk.diagonal(diag_start_idx + chunk_idx * chunk_size).fill_(1)

        # 将每个 chunk 调整为与对应的张量相同的形状
        chunks = tuple(
            chunk.view(total_numel, *tensor.shape)
            for chunk, tensor in zip(chunks, tensors)
        )

        # 使用生成器语法返回当前 chunk
        yield chunks
# 为给定张量和张量元素数量构建标准基础，使用默认的块大小
def _construct_standard_basis_for(tensors, tensor_numels):
    # 调用 _chunked_standard_basis_for_ 函数，返回第一个生成的标准基础张量
    for basis in _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
        return basis

# 验证并包装参数索引，确保其为整数且在有效范围内
def _validate_and_wrap_argnum(argnum, num_args):
    if not isinstance(argnum, int):
        raise RuntimeError(f"argnum must be int, got: {type(argnum)}")
    if argnum >= 0 and argnum < num_args:
        return argnum
    if argnum < 0 and argnum >= -num_args:
        return argnum + num_args
    raise RuntimeError(f"Got argnum={argnum}, but only {num_args} positional inputs")

# 检查元组中参数索引的唯一性和非空性
def _check_unique_non_empty(argnums):
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError("argnums must be non-empty")
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f"argnums elements must be unique, got {argnums}")

# 根据给定的参数索引替换旧参数列表中的参数
def _replace_args(old_args, new_args, argnums):
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(
                f"new_args should be of size 1, was of size {len(new_args)}"
            )
        # 根据参数索引进行替换
        return tuple(
            new_args[0] if i == argnums else old_args[i] for i in range(len(old_args))
        )
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(
                "new_args should have the same size as argnums. "
                f"Argnums size {len(argnums)}, new_args size {len(new_args)}"
            )
        
        # 定义函数用于根据索引替换对应的参数
        def get_right_elem(i):
            return new_args[argnums.index(i)] if i in argnums else old_args[i]

        return tuple(get_right_elem(i) for i in range(len(old_args)))
    raise RuntimeError(f"argnums must be int or Tuple[int, ...], got: {type(argnums)}")

# 验证并包装参数索引或索引元组，确保它们是有效的，并返回经过验证的结果
def _validate_and_wrap_argnums(argnums, num_args):
    if isinstance(argnums, int):
        return _validate_and_wrap_argnum(argnums, num_args)
    if isinstance(argnums, tuple):
        return tuple(_validate_and_wrap_argnum(argnum, num_args) for argnum in argnums)
    raise AssertionError("Should never get here")

# 根据给定的参数索引或索引元组，从参数列表中切片选定的参数，并作为元组返回
def _slice_argnums(args, argnums, as_tuple=True):
    if not isinstance(argnums, int) and not isinstance(argnums, tuple):
        raise RuntimeError(
            f"argnums must be int or Tuple[int, ...], got: {type(argnums)}"
        )
    argnums = _validate_and_wrap_argnums(argnums, len(args))
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        if as_tuple:
            return (args[argnums],)
        else:
            return args[argnums]
    return tuple(args[i] for i in argnums)

# 设置 JVP_NESTING 的初始值为 0
JVP_NESTING = 0

# 断言元素为扁平的张量元组，并在不符合条件时引发异常
def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None:
    if not isinstance(elts, tuple):
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got {type(elts)}"
        )
    # 对于 elts 中的每个元素进行迭代
    for elt in elts:
        # 检查元素是否为 torch.Tensor 类型，如果是则继续下一次迭代
        if isinstance(elt, torch.Tensor):
            continue
        # 如果元素不是 torch.Tensor 类型，则抛出运行时异常
        raise RuntimeError(
            f"{api}: Expected {argname} to be a tuple of Tensors, got "
            f"a tuple with an element of type {type(elt)}"
        )
    # 检查 elts 的长度是否为零
    if len(elts) == 0:
        # 如果 elts 长度为零，则抛出运行时异常
        raise RuntimeError(
            f"{api}: Expected {argname} to be a non-empty tuple of Tensors."
        )
# 确保输出的张量列表非空，用于断言函数返回的输出列表至少包含一个非空张量
def assert_non_empty_tensor_output(output: List[Any], api: str) -> None:
    # 如果输出列表长度为1且第一个元素为None，或者输出列表长度小于1，则抛出运行时错误
    if (len(output) == 1 and output[0] is None) or len(output) < 1:
        raise RuntimeError(
            f"{api}: Expected f to be a function that has non-empty output (got output = {output})"
        )
    # 检查输出列表中的每个元素是否都是torch.Tensor类型
    for o in output:
        if not isinstance(o, torch.Tensor):
            raise RuntimeError(
                f"{api}: expected f(*primals) to return only tensors"
                f", got unsupported type {type(o)}"
            )


# 确保输出是张量或张量元组，用于断言函数的输出是张量或张量元组
def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None:
    # 如果输出是单个张量，则直接返回
    if isinstance(output, torch.Tensor):
        return
    # 如果输出不是元组类型，则抛出运行时错误
    if not isinstance(output, tuple):
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got "
            f"{type(output)}"
        )
    # 如果输出元组长度为0，则抛出运行时错误
    if len(output) == 0:
        raise RuntimeError(
            f"{api}: Expected output of f to be a non-empty tuple of Tensors."
        )
    # 检查输出元组中的每个元素是否都是torch.Tensor类型
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected output of f to be a Tensor or Tensors, got "
            f"{type(out)} as an output"
        )


# 确保输出列表中的元素都是张量，用于断言给定的张量列表中的每个元素都是张量
def assert_non_empty_list_of_tensors(
    output: List[torch.Tensor], api: str, argname: str
) -> None:
    # 如果输出列表长度为0，则抛出运行时错误
    if len(output) == 0:
        raise RuntimeError(f"{api}: Expected {argname} to contain at least one Tensor.")
    # 检查输出列表中的每个元素是否都是torch.Tensor类型
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(
            f"{api}: Expected {argname} to only contain Tensors, got " f"{type(out)}"
        )


# "jvp(f, primals, tangents)" 字符串的全局定义
jvp_str = "jvp(f, primals, tangents)"


# 安全地解包双重对象，用于从双重对象中解包原始值和切线值
def safe_unpack_dual(dual, strict):
    # 如果双重对象不是torch.Tensor类型，则抛出运行时错误
    if not isinstance(dual, torch.Tensor):
        raise RuntimeError(
            f"{jvp_str}: expected f(*args) to return only tensors"
            f", got unsupported type {type(dual)}"
        )

    # 调用fwAD.unpack_dual函数，将双重对象拆分为原始值和切线值
    primal, tangent = fwAD.unpack_dual(dual)
    # 如果切线值为None且strict为True，则抛出运行时错误
    if tangent is None:
        if strict:
            raise RuntimeError(
                "jvp(f, primals, tangents, strict=True): "
                "The output of f is independent of "
                "the inputs. This is not allowed with strict=True."
            )
        # 否则，将切线值设为与原始值相同形状的全零张量
        tangent = torch.zeros_like(primal)
    # 返回解包后的原始值和切线值
    return primal, tangent


# "torch.func" 的装饰器函数，用于声明jvp函数可作为torch模块的一部分公开
@exposed_in("torch.func")
def jvp(
    func: Callable,
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    has_aux: bool = False,
):
    """
    代表雅可比向量积，返回一个元组，其中包含`func(*primals)`的输出和"在`primals`处评估的`func`的雅可比矩阵"乘以`tangents`。这也称为前向模式自动微分。
    """
    Returns a ``(output, jvp_out)`` tuple containing the output of ``func``
    evaluated at ``primals`` and the Jacobian-vector product.
    If ``has_aux is True``, then instead returns a ``(output, jvp_out, aux)`` tuple.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.

    jvp is useful when you wish to compute gradients of a function R^1 -> R^N

        >>> from torch.func import jvp
        >>> x = torch.randn([])
        >>> f = lambda x: x * torch.tensor([1., 2., 3])
        >>> value, grad = jvp(f, (x,), (torch.tensor(1.),))
        >>> assert torch.allclose(value, f(x))
        >>> assert torch.allclose(grad, torch.tensor([1., 2, 3]))

    :func:`jvp` can support functions with multiple inputs by passing in the
    tangents for each of the inputs

         >>> from torch.func import jvp
         >>> x = torch.randn(5)
         >>> y = torch.randn(5)
         >>> f = lambda x, y: (x * y)
         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
         >>> assert torch.allclose(output, x + y)
@doesnt_support_saved_tensors_hooks
# 定义一个装饰器，指示函数不支持保存张量钩子
def _jvp_with_argnums(
    func: Callable,
    primals: Any,
    tangents: Any,
    argnums: Optional[argnums_t],
    *,
    strict: bool = False,
    has_aux: bool,
):
    # 这个函数与 jvp 函数相同，但是还接受 argnums 参数
    # 大多数参数与 jvp 函数相同，除了新增的参数
    # argnums (Optional[int or tuple[int]]): 可选参数，指定要计算梯度的参数位置。
    #         如果为 None，则计算对所有输入的梯度（用于 jvp 函数）。默认为 None
    # 因此，tangents 的长度必须为 argnums 的长度，并与由 argnums 指定的对应 primal 匹配
    #
    # 警告: 用户不应直接调用此函数，应该直接调用 jvp 函数。
    # 它只是分开来确保传递给 jacfwd 但没有被微分的输入得到正确的包装。
    #
    # 注意: 所有的错误消息都像调用 jvp 函数一样生成，即使是通过 jacfwd 调用的。
    #
    # 返回与 :func:`jvp` 相同的两个元素，但返回的元组 `jvp_out` 只包含相对于由 argnums 指定的 primals 的 JVPs
    if not isinstance(primals, tuple):
        raise RuntimeError(
            f"{jvp_str}: Expected primals to be a tuple. "
            f"E.g. it should be valid to call f(*primals)."
        )
    # 将 primals 树状展开成列表 flat_primals，同时保留其结构信息到 primals_spec
    diff_args = primals if argnums is None else _slice_argnums(primals, argnums)
    flat_primals, primals_spec = tree_flatten(diff_args)
    # 将 tangents 树状展开成列表 flat_tangents，同时保留其结构信息到 tangents_spec
    flat_tangents, tangents_spec = tree_flatten(tangents)
    # 比较 diff_args 和 tangents 的树结构
    _jvp_treespec_compare(diff_args, tangents)
    # 断言 flat_primals 不是空的张量列表
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, "primals")
    # 断言 flat_tangents 不是空的张量列表
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, "tangents")

    global JVP_NESTING
    # 使用 jvp_increment_nesting 上下文管理器增加 JVP 嵌套级别
    with jvp_increment_nesting() as level:
        # 启用 fwAD 的前向梯度计算
        with fwAD._set_fwd_grad_enabled(True):
            # 根据 JVP_NESTING 的值选择上下文管理器，如果 JVP_NESTING == 1 使用 fwAD.dual_level，否则使用 contextlib.nullcontext
            ctx = fwAD.dual_level if JVP_NESTING == 1 else contextlib.nullcontext
            with ctx():
                # 使用 flat_primals 和 flat_tangents 创建平坦的双重数值（duals）
                flat_duals = tuple(
                    fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
                )
                # 将平坦的双重数值重新组合成树结构（duals）
                duals = tree_unflatten(flat_duals, primals_spec)

                # 如果 argnums 是整数或元组类型，则调用 _wrap_all_tensors 函数包装 primals
                # 并使用 _replace_args 替换参数
                if isinstance(argnums, (int, tuple)):
                    primals = _wrap_all_tensors(primals, level)
                    duals = _replace_args(primals, duals, argnums)

                # 调用函数 func，并传入 duals 作为参数
                result_duals = func(*duals)

                # 如果函数有辅助输出（auxiliary output）
                if has_aux:
                    # 确保函数返回值是一个元组且长度为 2，否则抛出 RuntimeError
                    if not (isinstance(result_duals, tuple) and len(result_duals) == 2):
                        raise RuntimeError(
                            f"{jvp_str}: output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    # 分离主输出和辅助输出
                    result_duals, aux = result_duals
                    # 对辅助输出进行反向操作
                    aux = _undo_create_differentiable(aux, level)

                # 将主输出结果扁平化，并返回扁平化后的结果和其结构描述 spec
                result_duals, spec = tree_flatten(result_duals)
                # 确保主输出结果非空张量
                assert_non_empty_tensor_output(result_duals, jvp_str)

                # 解包主输出结果中的 primals_out 和 tangents_out，并安全地解包为可用的数据
                primals_out, tangents_out = zip(
                    *[safe_unpack_dual(dual, strict) for dual in result_duals]
                )

                # 对 primals_out 和 tangents_out 中的每个元素进行反向操作
                primals_out = tree_map(
                    partial(_undo_create_differentiable, level=level), primals_out
                )
                tangents_out = tree_map(
                    partial(_undo_create_differentiable, level=level), tangents_out
                )

                # 将反向操作后的 primals_out 和 tangents_out 重新组合成树结构
                primals_out_unflatten = tree_unflatten(primals_out, spec)
                tangents_out_unflatten = tree_unflatten(tangents_out, spec)

                # 如果有辅助输出，返回主输出、切向输出和辅助输出
                if has_aux:
                    return primals_out_unflatten, tangents_out_unflatten, aux

                # 否则，只返回主输出和切向输出
                return primals_out_unflatten, tangents_out_unflatten
def safe_unflatten(tensor, dim, shape):
    # 如果 shape 的长度为 0，表示要操作的维度是标量，需要确保该维度的大小为 1，然后压缩该维度
    if len(shape) == 0:
        assert tensor.shape[dim] == 1
        return tensor.squeeze(dim)
    # 否则，使用 tensor 对象的 unflatten 方法，重新构造具有给定 shape 的 tensor
    return tensor.unflatten(dim, shape)


@exposed_in("torch.func")
def jacfwd(
    func: Callable,
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
):
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using forward-mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        randomness(str): Flag indicating what type of randomness to use.
            See :func:`vmap` for more detail. Allowed: "different", "same", "error".
            Default: "error"

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use :func:`jacrev`, which has better operator coverage.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>> jacobian = jacfwd(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacfwd` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacfwd, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacfwd(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output

    """
    # 实现 forward-mode 自动微分计算 Jacobian 矩阵
    # 返回一个函数，接受与 func 相同的输入，并返回相应参数的 Jacobian 矩阵
    pass
    as an auxiliary object:

        >>> from torch.func import jacfwd             # 导入 torch.func 模块中的 jacfwd 函数
        >>> x = torch.randn(5)                        # 创建一个包含随机数的张量 x
        >>>
        >>> def f(x):                                 # 定义一个函数 f，接受输入 x
        >>>   return x.sin()                          # 返回 x 的正弦值
        >>>
        >>> def g(x):                                 # 定义一个函数 g，接受输入 x
        >>>   result = f(x)                           # 调用函数 f 计算结果
        >>>   return result, result                    # 返回两次计算结果
        >>>
        >>> jacobian_f, f_x = jacfwd(g, has_aux=True)(x)  # 使用 jacfwd 计算函数 g 的雅可比矩阵，has_aux=True 表示 g 返回辅助输出
        >>> assert torch.allclose(f_x, f(x))           # 断言辅助输出 f_x 与 f(x) 的近似相等

    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`
    to produce Hessians

        >>> from torch.func import jacfwd, jacrev      # 导入 torch.func 模块中的 jacfwd 和 jacrev 函数
        >>> def f(x):                                  # 定义一个函数 f，接受输入 x
        >>>   return x.sin().sum()                     # 返回 x 的正弦值的总和
        >>>
        >>> x = torch.randn(5)                         # 创建一个包含随机数的张量 x
        >>> hessian = jacfwd(jacrev(f))(x)             # 使用 jacfwd 和 jacrev 计算函数 f 的 Hessian 矩阵
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))  # 断言计算得到的 Hessian 矩阵与预期的负正弦值对角矩阵近似相等

    By default, :func:`jacfwd` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacfwd             # 导入 torch.func 模块中的 jacfwd 函数
        >>> def f(x, y):                              # 定义一个函数 f，接受输入 x 和 y
        >>>   return x + y ** 2                       # 返回 x + y 的平方
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)     # 创建包含随机数的张量 x 和 y
        >>> jacobian = jacfwd(f, argnums=1)(x, y)     # 使用 jacfwd 计算函数 f 对第二个参数 y 的雅可比矩阵
        >>> expected = torch.diag(2 * y)               # 计算预期的对角矩阵，每个元素为 2*y[i]
        >>> assert torch.allclose(jacobian, expected)  # 断言计算得到的雅可比矩阵与预期的对角矩阵近似相等

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacfwd             # 导入 torch.func 模块中的 jacfwd 函数
        >>> def f(x, y):                              # 定义一个函数 f，接受输入 x 和 y
        >>>   return x + y ** 2                       # 返回 x + y 的平方
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)     # 创建包含随机数的张量 x 和 y
        >>> jacobian = jacfwd(f, argnums=(0, 1))(x, y)  # 使用 jacfwd 计算函数 f 对参数 x 和 y 的雅可比矩阵
        >>> expectedX = torch.diag(torch.ones_like(x)) # 计算预期的对角矩阵，每个元素为 1
        >>> expectedY = torch.diag(2 * y)              # 计算预期的对角矩阵，每个元素为 2*y[i]
        >>> assert torch.allclose(jacobian[0], expectedX)  # 断言计算得到的雅可比矩阵的第一个部分与预期的对角矩阵相等
        >>> assert torch.allclose(jacobian[1], expectedY)  # 断言计算得到的雅可比矩阵的第二个部分与预期的对角矩阵相等
    def wrapper_fn(*args):
        # 检查是否复杂，如果是则抛出错误
        error_if_complex("jacfwd", args, is_input=True)
        # 根据 argnums 切片参数
        primals = args if argnums is None else _slice_argnums(args, argnums)
        # 展平参数并获取展平后的规格
        flat_primals, primals_spec = tree_flatten(primals)
        # 计算展平后每个参数的元素数量
        flat_primals_numels = tuple(p.numel() for p in flat_primals)
        # 构建标准基向量集合，用于计算 Jacobi 矩阵
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        # 将展平后的基向量重新组合成树形结构
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis):
            # 调用 _jvp_with_argnums 计算偏导数
            output = _jvp_with_argnums(
                func, args, basis, argnums=argnums, has_aux=has_aux
            )
            # 检查函数输出是否复杂，如果是则抛出错误
            error_if_complex("jacfwd", output[0], is_input=False)
            if has_aux:
                _, jvp_out, aux = output
                return jvp_out, aux
            _, jvp_out = output
            return jvp_out

        # 使用 vmap 对 push_jvp 函数进行向量化映射
        results = vmap(push_jvp, randomness=randomness)(basis)
        if has_aux:
            # 如果有辅助输出，则解构 results 和 aux
            results, aux = results
            # aux 在标准基格式中，例如 NxN 矩阵，需要获取第一个元素作为原始 `func` 的输出
            flat_aux, aux_spec = tree_flatten(aux)
            flat_aux = [value[0] for value in flat_aux]
            aux = tree_unflatten(flat_aux, aux_spec)

        # 将结果展平成列表
        jac_outs, spec = tree_flatten(results)
        # 检查输出是否为空，如果是则抛出错误（理论上不会出现）
        # assert_non_empty_output(jac_outs, 'jacfwd(f, ...)(*args)')

        # 将结果映射回原始参数的结构
        jac_outs_ins = tuple(
            tuple(
                # 安全解构 jac_out_in，重构成原始参数的形状
                safe_unflatten(jac_out_in, -1, primal.shape)
                for primal, jac_out_in in zip(
                    flat_primals,
                    jac_out.movedim(0, -1).split(flat_primals_numels, dim=-1),
                )
            )
            for jac_out in jac_outs
        )
        # 将结果映射回原始参数的结构
        jac_outs_ins = tuple(
            tree_unflatten(jac_ins, primals_spec) for jac_ins in jac_outs_ins
        )

        # 如果 argnums 是整数，则取第一个元素
        if isinstance(argnums, int):
            jac_outs_ins = tuple(jac_ins[0] for jac_ins in jac_outs_ins)
        # 如果有辅助输出，则返回映射回原始参数的结构和 aux
        if has_aux:
            return tree_unflatten(jac_outs_ins, spec), aux
        # 否则，只返回映射回原始参数的结构
        return tree_unflatten(jac_outs_ins, spec)

    # 如果不是在 Dynamo 编译环境下，应用 functools.wraps(func)
    if not torch._dynamo.is_compiling():
        wrapper_fn = wraps(func)(wrapper_fn)

    # 返回包装后的函数
    return wrapper_fn
# 将函数标记为在"torch.func"中可见
@exposed_in("torch.func")
#
    with grad_increment_nesting() as level:
        output, aux, grad_input = None, None, None
        # See NOTE [grad and vjp interaction with no_grad]
        # 使用 torch.enable_grad() 启用梯度计算上下文管理器
        with torch.enable_grad():
            # 将所有的参数（args）和关键字参数（kwargs）封装成张量
            args = _wrap_all_tensors(args, level)
            kwargs = _wrap_all_tensors(kwargs, level)
            # 从参数中选择需要求导的部分
            diff_args = _slice_argnums(args, argnums, as_tuple=False)
            # 对所选参数进行转换，使其支持自动求导
            tree_map_(partial(_create_differentiable, level=level), diff_args)

            # 调用函数 func，计算输出
            output = func(*args, **kwargs)
            # 如果函数有辅助输出（aux），则检查输出是否为一个包含两个元素的元组
            if has_aux:
                if not (isinstance(output, tuple) and len(output) == 2):
                    raise RuntimeError(
                        "grad_and_value(f)(*args): output of function f should be a tuple: (output, aux) "
                        "if has_aux is True"
                    )
                output, aux = output

            # 检查输出是否为 torch.Tensor 类型
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    "grad_and_value(f)(*args): Expected f(*args) "
                    f"to return a Tensor, got {type(output)}"
                )
            # 检查输出是否为标量张量
            if output.dim() != 0:
                raise RuntimeError(
                    "grad_and_value(f)(*args): Expected f(*args) "
                    "to return a scalar Tensor, got tensor with "
                    f"{output.dim()} dims. Maybe you wanted to "
                    "use the vjp or jacrev APIs instead?"
                )

            # 将求导参数扁平化并记录其结构
            flat_diff_args, spec = tree_flatten(diff_args)

            # 使用 _as_tuple 将输出扁平化
            flat_outputs = _as_tuple(output)
            # 使用 _autograd_grad 计算梯度，支持创建计算图
            flat_grad_input = _autograd_grad(
                flat_outputs, flat_diff_args, create_graph=True
            )
            # 将扁平化的梯度张量还原成原始结构
            grad_input = tree_unflatten(flat_grad_input, spec)

            # 取消对求导参数的创建可微处理
            grad_input = _undo_create_differentiable(grad_input, level)
            # 取消对输出的创建可微处理
            output = _undo_create_differentiable(output, level)
            # 如果有辅助输出，取消对辅助输出的创建可微处理
            if has_aux:
                aux = _undo_create_differentiable(aux, level)

        # 如果有辅助输出，返回梯度输入和输出及辅助输出的元组
        if has_aux:
            return grad_input, (output, aux)
        # 否则，只返回梯度输入和输出的元组
        return grad_input, output
# 根据给定参数计算函数 `func` 的梯度。
def grad_impl(func: Callable, argnums: argnums_t, has_aux: bool, args, kwargs):
    # 调用内部函数 `grad_and_value_impl` 计算函数梯度及其值
    results = grad_and_value_impl(func, argnums, has_aux, args, kwargs)
    if has_aux:
        # 如果 `has_aux` 为 True，则结果包含梯度和辅助信息
        grad, (_, aux) = results
        return grad, aux
    else:
        # 如果 `has_aux` 为 False，则结果仅包含梯度
        grad, _ = results
        return grad


# 将可能是 torch.Tensor 的对象包装成函数式张量，根据需要进行功能化处理
def _maybe_wrap_functional_tensor(
    maybe_tensor, level, *, _python_functionalize: bool = False
):
    if not isinstance(maybe_tensor, torch.Tensor):
        # 如果 `maybe_tensor` 不是 torch.Tensor 对象，则直接返回
        return maybe_tensor
    # 使用 `_wrap_functional_tensor` 函数对 `maybe_tensor` 进行包装
    wrapped = _wrap_functional_tensor(maybe_tensor, level)
    # 断言包装后的张量与原始张量的一致性
    _assert_wrapped_functional(maybe_tensor, wrapped)
    if _python_functionalize:
        # 如果需要进行 Python 功能化，则创建 FunctionalTensor 对象
        out = FunctionalTensor(wrapped)
        # 将原始张量的自动求导元信息镜像到 `out`
        torch._mirror_autograd_meta_to(maybe_tensor, out)
        return out
    else:
        return wrapped


# 对输入的张量树应用 `_maybe_wrap_functional_tensor` 函数，实现功能化处理
def _wrap_all_tensors_to_functional(
    tensor_pytree, level, *, _python_functionalize: bool = False
):
    return tree_map(
        partial(
            lambda x: _maybe_wrap_functional_tensor(
                x, level, _python_functionalize=_python_functionalize
            )
        ),
        tensor_pytree,
    )


# 可能将功能化张量解包成原始张量，根据需要重新应用视图
def _maybe_unwrap_functional_tensor(maybe_tensor, *, reapply_views: bool):
    if not isinstance(maybe_tensor, torch.Tensor):
        # 如果 `maybe_tensor` 不是 torch.Tensor 对象，则直接返回
        return maybe_tensor
    if isinstance(maybe_tensor, FunctionalTensor):
        # 如果 `maybe_tensor` 是 FunctionalTensor，则获取其内部的原始张量
        maybe_tensor = maybe_tensor.elem

    if not torch._is_functional_tensor(maybe_tensor):
        # 如果 `maybe_tensor` 不是功能化张量，则直接返回
        # 这种情况发生在功能化函数返回未正确包装的全局变量时
        return maybe_tensor
    # 同步输出张量的任何挂起更新
    torch._sync(maybe_tensor)
    return _unwrap_functional_tensor(maybe_tensor, reapply_views)


# 对输入的张量树应用 `_maybe_unwrap_functional_tensor` 函数，实现解包处理
def _unwrap_all_tensors_from_functional(tensor_pytree, *, reapply_views: bool):
    return tree_map(
        lambda t: _maybe_unwrap_functional_tensor(t, reapply_views=reapply_views),
        tensor_pytree,
    )


# 在 Torch 框架中暴露为 "torch.func" 的功能化函数装饰器
@exposed_in("torch.func")
def functionalize(func: Callable, *, remove: str = "mutations") -> Callable:
    """
    functionalize 是一种转换，可用于移除函数中的（中间）变异和别名，同时保留函数的语义。

    `functionalize(func)` 返回一个新函数，其语义与 `func` 相同，但移除了所有中间变异。
    对中间张量进行的每个原位操作 `intermediate.foo_()` 都被其非原位等效操作替换：
    `intermediate_updated = intermediate.foo()`。

    functionalize 对于将 PyTorch 程序传送到无法轻松表示变异或别名操作符的后端或编译器非常有用。
    """
    # 定义一个函数 functionalize，接受两个参数：
    # - func：一个 Python 函数，接受一个或多个参数。
    # - remove：一个可选的字符串参数，取值可以是 'mutations' 或 'mutations_and_views'。
    #   - 'mutations' 表示所有具有变异效果的操作符将被替换为它们的非变异等效操作符。
    #   - 'mutations_and_views' 表示除了替换变异操作符外，还会替换所有别名操作符为它们的非别名等效操作符。
    #   默认取值为 'mutations'。
    def functionalize(func, remove='mutations'):
        # 返回一个新的 "functionalized" 函数。该函数接受与 func 相同的输入，并具有相同的行为，
        # 但是在函数中对中间张量进行的任何变异操作（和可选的别名操作）都将被移除。
        # functionalize 也会移除在函数输入上执行的变异（和视图）。
        # 然而，为了保持语义，functionalize 将在转换运行结束后通过检测任何张量输入“应该已经”
        # 被变异，并在必要时将新数据复制回输入来修复变异。
        pass
    >>> # xdoctest: +SKIP
    >>> import torch
    >>> from torch.fx.experimental.proxy_tensor import make_fx
    >>> from torch.func import functionalize
    >>>
    >>> # 定义一个函数，对中间张量进行突变和视图操作
    >>> def f(a):
    ...     # 对输入张量 a 加 1，得到 b
    ...     b = a + 1
    ...     # 将 b 进行视图变换，变成一维张量 c
    ...     c = b.view(-1)
    ...     # 在张量 c 上执行 in-place 加 1 操作
    ...     c.add_(1)
    ...     # 返回 b
    ...     return b
    ...
    >>> # 创建一个随机输入张量
    >>> inpt = torch.randn(2)
    >>>
    >>> # 使用函数 f 处理输入张量 inpt，得到输出 out1
    >>> out1 = f(inpt)
    >>> # 对函数 f 进行功能化处理后，再处理输入张量 inpt，得到输出 out2
    >>> out2 = functionalize(f)(inpt)
    >>>
    >>> # 验证两种方式的输出是否相等
    >>> # 输出 True 表示两种方法的输出在语义上是等价的
    >>> print(torch.allclose(out1, out2))
    True
    >>>
    >>> # 使用 make_fx 将函数 f 转换为 FX 模块，打印其代码
    >>> f_traced = make_fx(f)(inpt)
    >>> print(f_traced.code)
    def forward(self, a_1):
        add = torch.ops.aten.add(a_1, 1);  a_1 = None
        view = torch.ops.aten.view(add, [-1])
        add_ = torch.ops.aten.add_(view, 1);  view = None
        return add
    >>>
    >>> # 使用 make_fx 将经过功能化处理后的函数 f 转换为 FX 模块，打印其代码
    >>> f_no_mutations_traced = make_fx(functionalize(f))(inpt)
    >>> print(f_no_mutations_traced.code)
    def forward(self, a_1):
        add = torch.ops.aten.add(a_1, 1);  a_1 = None
        view = torch.ops.aten.view(add, [-1]);  add = None
        add_1 = torch.ops.aten.add(view, 1);  view = None
        view_1 = torch.ops.aten.view(add_1, [2]);  add_1 = None
        return view_1
    >>>
    >>> # 使用 make_fx 将经过功能化处理并移除 mutations_and_views 的函数 f 转换为 FX 模块，打印其代码
    >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
    >>> print(f_no_mutations_and_views_traced.code)
    def forward(self, a_1):
        add = torch.ops.aten.add(a_1, 1);  a_1 = None
        view_copy = torch.ops.aten.view_copy(add, [-1]);  add = None
        add_1 = torch.ops.aten.add(view_copy, 1);  view_copy = None
        view_copy_1 = torch.ops.aten.view_copy(add_1, [2]);  add_1 = None
        return view_copy_1
    >>>
    >>> # 定义一个函数，对其输入张量进行突变操作
    >>> def f(a):
    ...     # 将输入张量 a 进行视图变换，变成一维张量 b
    ...     b = a.view(-1)
    ...     # 在张量 b 上执行 in-place 加 1 操作
    ...     b.add_(1)
    ...     # 返回未变异的输入张量 a
    ...     return a
    ...
    >>> # 使用 make_fx 将经过功能化处理并移除 mutations_and_views 的函数 f 转换为 FX 模块，打印其代码
    >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
    >>> # 打印说明，所有的变异和视图已经被移除，但为了正确地将变异应用于输入，在函数完成后需要额外的 copy_ 操作
    >>> print(f_no_mutations_and_views_traced.code)
    def forward(self, a_1):
        view_copy = torch.ops.aten.view_copy(a_1, [-1])
        add = torch.ops.aten.add(view_copy, 1);  view_copy = None
        view_copy_1 = torch.ops.aten.view_copy(add, [2]);  add = None
        copy_ = torch.ops.aten.copy_(a_1, view_copy_1);  a_1 = None
        return view_copy_1
    """
    There are a few "failure modes" for functionalize that are worth calling out:
      (1) Like other torch.func transforms, `functionalize()` doesn't work with functions
          that directly use `.backward()`. The same is true for torch.autograd.grad.
          If you want to use autograd, you can compute gradients directly
          with `functionalize(grad(f))`.
      (2) Like other torch.func transforms, `functionalize()` doesn't work with global state.
          If you call `functionalize(f)` on a function that takes views / mutations of
          non-local state, functionalization will simply no-op and pass the view/mutation
          calls directly to the backend.
          One way to work around this is to ensure that any non-local state creation
          is wrapped into a larger function, which you then call functionalize on.
      (3) `resize_()` has some limitations: functionalize will only work on programs
          that use `resize_()` as long as the tensor being resized is not a view.
      (4) `as_strided()` has some limitations: functionalize will not work on
          `as_strided()` calls that result in tensors with overlapping memory.


    Finally, a helpful mental model for understanding functionalization is that
    most user pytorch programs are written with the public torch API.
    When executed, torch operators are generally decomposed into
    our internal C++ "ATen" API.
    The logic for functionalization happens entirely at the level of ATen.
    Functionalization knows how to take every aliasing operator in ATen,
    and map it to its non-aliasing equivalent
    (e.g. ``tensor.view({-1})`` -> ``at::view_copy(tensor, {-1})``),
    and how to take every mutating operator in ATen,
    and map it to its non-mutating equivalent
    (e.g. ``tensor.add_(1)`` -> ``at::add(tensor, -1)``),
    while tracking aliases and mutations out-of-line to know when to fix things up.
    Information about which ATen operators are aliasing or mutating all comes from
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml.
    """

    # 根据不同的 remove 参数设置 reapply_views 变量的值
    if remove == "mutations":
        reapply_views = True
    elif remove == "mutations_and_views":
        reapply_views = False
    else:
        # 如果 remove 参数既不是 "mutations" 也不是 "mutations_and_views"，抛出运行时错误
        raise RuntimeError(
            f"functionalize(f, remove='mutations'): received invalid argument for remove={remove}."
            " Valid options are:\n"
            "     remove='mutations': all inplace and out= operators will be removed from the program, and replaced"
            " with their out-of-place equivalents.\n"
            "     remove='mutations_and_views': In addition to the above, all aliasing operators {view} will be"
            " replaced with their non-aliasing counterparts, {view}_copy.\n"
        )

    # 将 @doesnt_support_saved_tensors_hooks 装饰器应用到 func 函数上，并保留原函数的元数据
    # 这里假设 @doesnt_support_saved_tensors_hooks 是一个装饰器函数，用于处理不支持保存张量 hooks 的情况
    @doesnt_support_saved_tensors_hooks
    # 使用 func 函数来装饰当前函数
    @wraps(func)
    # 定义一个内部函数 wrapped，用于包装给定的 func 函数
    def wrapped(*args, **kwargs):
        # 尝试执行以下代码块，无论是否发生异常
        try:
            # 调用 _func_increment_nesting 函数，增加函数嵌套层级
            func_level = _func_increment_nesting(reapply_views)
            # 对所有的位置参数进行包装，使其变成功能性对象
            func_args = _wrap_all_tensors_to_functional(args, func_level)
            # 对所有的关键字参数进行包装，使其变成功能性对象
            func_kwargs = _wrap_all_tensors_to_functional(kwargs, func_level)

            # 获取未包装参数的扁平化列表
            flattened_unwrapped_args = pytree.arg_tree_leaves(*args)
            # 获取已包装参数的扁平化列表
            flattened_wrapped_args = pytree.arg_tree_leaves(*func_args)
            # 获取未包装关键字参数的扁平化列表
            flattened_unwrapped_kwargs = pytree.arg_tree_leaves(**kwargs)
            # 获取已包装关键字参数的扁平化列表
            flattened_wrapped_kwargs = pytree.arg_tree_leaves(**func_kwargs)

            # 调用 func 函数，传入已包装的参数和关键字参数，获取函数输出
            func_outputs = func(*func_args, **func_kwargs)
            # 将函数输出中的功能性对象解包成普通张量
            outputs = _unwrap_all_tensors_from_functional(
                func_outputs, reapply_views=reapply_views
            )
            # 将输出结果进行扁平化处理，获取扁平化后的输出和函数输出的结构
            flat_outputs, func_out_spec = tree_flatten(outputs)

            # 遍历已包装的参数和关键字参数，如果是 torch.Tensor 类型，则调用 sync_() 方法以确保所有挂起的变更已应用
            for a in flattened_wrapped_args + flattened_wrapped_kwargs:
                if isinstance(a, torch.Tensor):
                    # 在输入张量上调用 sync_()，确保应用了所有挂起的变更
                    torch._sync(a)

            # 如果输入张量有任何变更，需要将这些变更传播回给用户
            # 遍历未包装的参数和关键字参数，将变更传播给已包装的对应张量
            for unwrapped, wrapped in zip(
                flattened_unwrapped_args, flattened_wrapped_args
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(
                    wrapped, torch.Tensor
                ):
                    # 将功能性输入的变更传播到未包装的输入张量上
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for unwrapped, wrapped in zip(
                flattened_unwrapped_kwargs, flattened_wrapped_kwargs
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(
                    wrapped, torch.Tensor
                ):
                    # 将功能性输入的变更传播到未包装的关键字输入张量上
                    _propagate_functional_input_mutation(unwrapped, wrapped)

            # 返回函数的输出结果
            return outputs
        finally:
            # 无论如何，都要执行 _func_decrement_nesting 函数，减少函数嵌套层级
            _func_decrement_nesting()

    # 返回内部函数 wrapped，作为包装后的 func 函数的结果
    return wrapped
# 将该函数标记为在"torch.func"中公开的函数
@exposed_in("torch.func")
# 定义函数linearize，用于计算func在给定primals处的值及其线性近似
def linearize(func: Callable, *primals) -> Tuple[Any, Callable]:
    """
    Returns the value of ``func`` at ``primals`` and linear approximation
    at ``primals``.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. These are the values at which the function is linearly approximated.

    Returns:
        Returns a ``(output, jvp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the jvp of
        ``func`` evaluated at ``primals``.

    linearize is useful if jvp is to be computed multiple times at ``primals``. However,
    to achieve this, linearize saves intermediate computation and has higher memory requirements
    than directly applying `jvp`. So, if all the ``tangents`` are known, it maybe more efficient
    to compute vmap(jvp) instead of using linearize.

    .. note::
        linearize evaluates ``func`` twice. Please file an issue for an implementation
        with a single evaluation.

    Example::
        >>> import torch
        >>> from torch.func import linearize
        >>> def fn(x):
        ...     return x.sin()
        ...
        >>> output, jvp_fn = linearize(fn, torch.zeros(3, 3))
        >>> jvp_fn(torch.ones(3, 3))
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])
        >>>

    """
    # 注意：我们对`fn`进行了两次评估。
    # 一次是为了返回输出，另一次是在跟踪图时。
    # 如果这成为瓶颈，我们应该更新make_fx，使其也返回输出。

    # 调用func函数，计算primals处的输出
    output = func(*primals)
    # 使用tree_flatten函数分解输出，获取展平后的结果和结构信息
    _, output_spec = tree_flatten(output)

    # 使用tree_flatten函数展平primals中的张量，并获取其结构信息
    flat_primals, primals_argspec = tree_flatten(primals)

    # 为跟踪准备tangents
    # 每个张量p都创建一个空的同形张量t，并存储在flat_tangents元组中
    flat_tangents = tuple(p.new_empty(()).expand_as(p) for p in flat_primals)

    # 定义用于跟踪的函数
    def trace_fn(flat_tangents):
        # 使用fwAD.dual_level()上下文管理器
        with fwAD.dual_level():
            # 创建fwAD.make_dual的结果，并与flat_primals和flat_tangents一一对应
            flat_duals = tuple(
                fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)
            )
            # 使用tree_unflatten函数重构flat_duals，得到duals
            duals = tree_unflatten(flat_duals, primals_argspec)
            # 调用func函数，计算duals处的输出
            output = func(*duals)
            # 使用tree_map_only函数，提取output中的张量并解包fwAD.unpack_dual的结果，得到tangents
            tangents = tree_map_only(
                torch.Tensor, lambda t: fwAD.unpack_dual(t)[1], output
            )

        return tangents

    # 使用lazy_dynamo_disallow(make_fx)函数创建jvp_graph，调用trace_fn，并传入flat_tangents
    jvp_graph = lazy_dynamo_disallow(make_fx)(trace_fn)(flat_tangents)
    # 使用lazy_dynamo_disallow(const_fold.split_const_subgraphs)函数创建const_folded_jvp_graph，传入jvp_graph
    const_folded_jvp_graph = lazy_dynamo_disallow(const_fold.split_const_subgraphs)(
        jvp_graph
    )

    # 仅保留关于primals的元数据信息
    # 获取flat_primals中每个张量的形状、设备和数据类型，并存储在对应元组中
    flat_primals_shape = tuple(p.shape for p in flat_primals)
    flat_primals_device = tuple(p.device for p in flat_primals)
    flat_primals_dtype = tuple(p.dtype for p in flat_primals)
    # 定义一个函数用于执行前向自动微分检查
    def forward_ad_checks(flat_tangents):
        # 遍历平坦化切线列表中的每个元素及其索引
        for idx, t in enumerate(flat_tangents):
            # 检查当前切线张量的形状是否与相应原始数据的形状匹配
            if t.shape != flat_primals_shape[idx]:
                # 如果不匹配，生成包含错误消息的异常对象并抛出
                msg = (
                    f"tangent:{idx} with shape {t.shape} in flattened "
                    f"pytree doesn't match the shape {flat_primals_shape[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            # 检查当前切线张量的设备是否与相应原始数据的设备匹配
            if t.device != flat_primals_device[idx]:
                # 如果不匹配，生成包含错误消息的异常对象并抛出
                msg = (
                    f"tangent:{idx} with device {t.device} in flattened "
                    f"pytree doesn't match the device {flat_primals_device[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

            # 检查当前切线张量的数据类型是否与相应原始数据的数据类型匹配
            if t.dtype != flat_primals_dtype[idx]:
                # 如果不匹配，生成包含错误消息的异常对象并抛出
                msg = (
                    f"tangent:{idx} with dtype {t.dtype} in flattened "
                    f"pytree doesn't match the dtype {flat_primals_dtype[idx]} "
                    "of the corresponding primal."
                )
                raise RuntimeError(msg)

    # 定义一个函数用于计算Jacobian向量积（JVP）
    # jvp_fn : callable to return
    #   It takes care of checking the argspec of tangents,
    #   calling the folded fx graph and unflattening fx graph output
    def jvp_fn(*tangents):
        # 将输入的切线参数展平，并返回展平后的切线列表和切线参数的规范
        flat_tangents, tangent_argspec = tree_flatten(tangents)
        
        # 使用_primals和tangents进行线性化树规范比较
        _linearize_treespec_compare(primals, tangents)

        # 对前向自动微分进行检查，确保切线与原始数据匹配
        forward_ad_checks(flat_tangents)

        # 调用常数折叠的JVP图，并返回展平后的输出
        flat_output = const_folded_jvp_graph(*flat_tangents)
        
        # 由于常数折叠的图可能返回展平输出，因此需要对输出进行变换
        return tree_unflatten(flat_output, output_spec)

    # 返回计算得到的输出及JVP函数
    return output, jvp_fn
```