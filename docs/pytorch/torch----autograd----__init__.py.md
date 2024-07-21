# `.\pytorch\torch\autograd\__init__.py`

```
"""
``torch.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.

It requires minimal changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""

# 引入警告模块，用于处理警告信息
import warnings
# 引入类型提示模块中的各种类型
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Union

# 引入 PyTorch 模块
import torch

# 从 torch.types 中引入特定类型
from torch.types import _size, _TensorOrTensors, _TensorOrTensorsOrGradEdge

# 从内部模块导入相关功能
from .. import _vmap_internals
# 从 overrides 模块导入特定功能
from ..overrides import handle_torch_function, has_torch_function, is_tensor_like
# 从当前目录下的子模块导入相关功能
from . import forward_ad, functional, graph
# 从 anomaly_mode 模块导入异常检测相关功能
from .anomaly_mode import detect_anomaly, set_detect_anomaly
# 从 function 模块导入函数相关功能
from .function import Function, NestedIOFunction
# 从 grad_mode 模块导入梯度模式相关功能
from .grad_mode import (
    _force_original_view_tracking,
    _unsafe_preserve_version_counter,
    enable_grad,
    inference_mode,
    no_grad,
    set_grad_enabled,
    set_multithreading_enabled,
)
# 从 gradcheck 模块导入梯度检查相关功能
from .gradcheck import gradcheck, gradgradcheck
# 从 graph 模块导入图运行相关功能
from .graph import _engine_run_backward
# 从 variable 模块导入 Variable 类
from .variable import Variable

# 定义公开的模块成员列表
__all__ = [
    "Variable",
    "Function",
    "backward",
    "grad_mode",
    "NestedIOFunction",
    "detect_anomaly",
    "enable_grad",
    "grad",
    "gradcheck",
    "gradgradcheck",
    "inference_mode",
    "no_grad",
    "set_detect_anomaly",
    "set_grad_enabled",
    "set_multithreading_enabled",
    "variable",
]

# 定义可选的 Tensor 类型
_OptionalTensor = Optional[torch.Tensor]
# 定义形状或嵌套形状类型
_ShapeorNestedShape = Union[_size, Sequence[_size], torch.Tensor]


# 定义计算形状的函数
def _calculate_shape(
    output: torch.Tensor, grad: torch.Tensor, is_grads_batched: bool
) -> Tuple[_ShapeorNestedShape, _ShapeorNestedShape]:
    # is_same_size ensures that both tensors are either nested or non nested
    # circular import
    from torch.nested._internal.nested_tensor import NestedTensor

    # 如果输出是嵌套的且不是 NestedTensor 类型，则抛出运行时错误
    if output.is_nested and not isinstance(output, NestedTensor):
        if is_grads_batched:
            raise RuntimeError("Batched grads are not supported with Nested Tensor.")
        # 获取输出和梯度的嵌套形状
        out_shape = output._nested_tensor_size()
        grad_shape = grad._nested_tensor_size()

        return out_shape, grad_shape

    # 否则返回常规张量的形状
    reg_out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return reg_out_shape, reg_grad_shape


# 创建梯度的函数
def _make_grads(
    outputs: Sequence[torch.Tensor],
    grads: Sequence[_OptionalTensor],
    is_grads_batched: bool,
) -> Tuple[_OptionalTensor, ...]:
    # 创建一个空的新梯度列表
    new_grads: List[_OptionalTensor] = []
    # 返回新梯度的元组形式
    return tuple(new_grads)


# 将 Tensor 或 Tensors 转换为元组的函数
def _tensor_or_tensors_to_tuple(
    tensors: Optional[_TensorOrTensors], length: int
) -> Tuple[_OptionalTensor, ...]:
    # 如果输入为 None，则返回长度为 length 的 None 元组
    if tensors is None:
        return (None,) * length
    # 如果输入为单个 Tensor，则返回单元素元组
    if isinstance(tensors, torch.Tensor):
        return (tensors,)
    # 否则将输入转换为元组并返回
    return tuple(tensors)
# 定义了一个函数 backward，用于计算给定张量相对于图叶子节点的梯度之和
def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensorsOrGradEdge] = None,
) -> None:
    r"""Compute the sum of gradients of given tensors with respect to graph leaves.

    # 使用链式法则对图进行微分。如果任何张量是非标量（即其数据有多个元素）且需要梯度，则会计算雅可比向量积，
    # 在这种情况下，还需要指定 grad_tensors。它应该是匹配长度的序列，包含雅可比向量积中的“向量”，通常是相应张量对应的梯度
    # （对于所有不需要梯度张量，None 是可接受的值）。
    # 这个函数会在叶子节点累积梯度 - 在调用之前可能需要将 .grad 属性清零或设置为 None。
    # 有关累积梯度的内存布局详情，请参见：Default gradient layouts。
    # 参见链接：https://pytorch.org/docs/stable/notes/autograd.html#default-gradient-layouts

    .. note::
        # 使用 create_graph=True 和参数的梯度会创建参数和其梯度之间的引用循环，可能导致内存泄漏。
        # 我们建议在创建图时使用 autograd.grad 来避免这种情况。
        # 如果必须使用此函数，请确保在使用后将参数的 .grad 字段重置为 None，以打破循环并避免泄漏。
        # 参见链接：https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward

    .. note::
        # 如果在用户指定的 CUDA 流上下文中运行任何前向操作、创建 grad_tensors 或调用 backward，
        # 请参见：Stream semantics of backward passes。
        # 参见链接：https://pytorch.org/docs/stable/notes/cuda.html#stream-semantics-of-backward-passes

    .. note::
        # 当提供 inputs 并且给定输入不是叶子节点时，当前实现将调用其 grad_fn（即使不严格需要获取这些梯度）。
        # 这是用户不应依赖的实现细节。
        # 参见链接：https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780
    Args:
        tensors (Sequence[Tensor] or Tensor): 需要计算导数的张量，可以是单个张量或张量序列。
        grad_tensors (Sequence[Tensor or None] or Tensor, optional): 雅可比向量积中的向量，通常是对应张量每个元素的梯度。
            对于不需要梯度的标量张量或不需要梯度的情况，可以指定为 None。如果所有的 grad_tensors 都可以接受 None 值，
            则此参数是可选的。
        retain_graph (bool, optional): 如果为 ``False``，则释放用于计算梯度的计算图。注意，在几乎所有情况下，将此选项设置为 ``True``
            是不必要的，通常可以通过更有效的方式解决问题。默认为 ``create_graph`` 的值。
        create_graph (bool, optional): 如果为 ``True``，将构建导数的计算图，允许计算高阶导数乘积。默认为 ``False``。
        inputs (Sequence[Tensor] or Tensor or Sequence[GradientEdge], optional): 用于累积梯度的输入，梯度将累积到 ``.grad`` 中。
            所有其他张量将被忽略。如果未提供，则梯度将累积到用于计算 :attr:`tensors` 的所有叶子张量中。
    """
    # 检查是否在 functorch 转换内部调用 backward()，如果是，则抛出 RuntimeError
    if torch._C._are_functorch_transforms_active():
        raise RuntimeError(
            "backward() called inside a functorch transform. This is not "
            "supported, please use functorch.grad or functorch.vjp instead "
            "or call backward() outside of functorch transforms."
        )

    # 检查是否使用了已弃用的 `grad_variables` 参数，提出警告并使用 `grad_tensors` 替代
    if grad_variables is not None:
        warnings.warn(
            "`grad_variables` is deprecated. Use `grad_tensors` instead.",
            FutureWarning,
            stacklevel=2,
        )
        # 如果 `grad_tensors` 未提供，则使用 `grad_variables`
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            # 如果同时传递了 `grad_tensors` 和已弃用的 `grad_variables`，则引发 RuntimeError
            raise RuntimeError(
                "`grad_tensors` and `grad_variables` (deprecated) "
                "arguments both passed to `backward()`. Please only "
                "use `grad_tensors`."
            )

    # 检查 `inputs` 是否为空列表，如果是则抛出 RuntimeError
    if inputs is not None and len(inputs) == 0:
        raise RuntimeError("`inputs` argument to `backward()` cannot be empty.")

    # 将 `tensors` 转换为元组，确保是元组形式
    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)

    # 将 `inputs` 转换为元组形式，支持多种输入类型
    inputs = (
        (inputs,)
        if isinstance(inputs, (torch.Tensor, graph.GradientEdge))
        else tuple(inputs)
        if inputs is not None
        else tuple()
    )

    # 将 `grad_tensors` 转换为元组形式，确保与 `tensors` 的长度一致
    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))

    # 根据 `tensors` 和 `grad_tensors_` 创建梯度，不支持批量梯度
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)

    # 如果 `retain_graph` 未指定，则与 `create_graph` 相同
    if retain_graph is None:
        retain_graph = create_graph

    # 以下注释重复是因为某些 Python 版本会打印多行函数的第一行注释
    # 调用函数 `_engine_run_backward` 来执行反向传播计算梯度。
    # 该函数接受多个参数，包括 `tensors`（张量）、`grad_tensors_`（梯度张量）、
    # `retain_graph`（是否保留计算图）、`create_graph`（是否创建计算图）、
    # `inputs`（输入参数）等。
    # `allow_unreachable=True` 表示允许访问不可达对象（即已经被销毁的对象）。
    # `accumulate_grad=True` 表示在张量的 `grad` 属性中累积梯度值。
# 定义一个函数 grad，用于计算并返回关于输入的梯度之和
def grad(
    outputs: _TensorOrTensors,  # 输出张量或张量的元组
    inputs: _TensorOrTensorsOrGradEdge,  # 输入张量、张量元组或梯度边缘对象
    grad_outputs: Optional[_TensorOrTensors] = None,  # 可选参数，与输出张量长度相同的梯度向量
    retain_graph: Optional[bool] = None,  # 可选参数，保留计算图
    create_graph: bool = False,  # 是否创建计算图的标志
    only_inputs: bool = True,  # 已弃用的参数，现在默认为 True，仅用于输入
    allow_unused: Optional[bool] = None,  # 可选参数，允许未使用的输入张量
    is_grads_batched: bool = False,  # 是否批量化梯度的标志
    materialize_grads: bool = False,  # 是否生成梯度的标志
) -> Tuple[torch.Tensor, ...]:  # 函数返回类型为一个张量元组
    r"""Compute and return the sum of gradients of outputs with respect to the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    .. note::

        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        ``only_inputs`` argument is deprecated and is ignored now (defaults to ``True``).
        To accumulate gradient for other parts of the graph, please use
        ``torch.autograd.backward``.

    """
    """
    Args:
        outputs (sequence of Tensor): differentiated function 的输出。
        inputs (sequence of Tensor or GradientEdge): 相对于其计算梯度的输入。
        grad_outputs (sequence of Tensor): vector-Jacobian 乘积中的“向量”，通常是每个输出的梯度。
            可以为标量张量或不需要梯度的张量指定 None 值。如果所有 grad_tensors 都可以接受 None 值，
            则此参数是可选的。默认为 None。
        retain_graph (bool, optional): 如果为 ``False``，则用于计算梯度的计算图将被释放。
            注意，在几乎所有情况下，设置此选项为 ``True`` 是不需要的，并且通常可以以更高效的方式解决。
            默认为 ``create_graph`` 的值。
        create_graph (bool, optional): 如果为 ``True``，将构造导数的计算图，允许计算更高阶导数乘积。
            默认为 ``False``。
        allow_unused (Optional[bool], optional): 如果为 ``False``，指定未在计算输出时使用的输入是一个错误。
            默认为 ``materialize_grads`` 的值。
        is_grads_batched (bool, optional): 如果为 ``True``，每个 ``grad_outputs`` 张量的第一维将被解释为批处理维度。
            而不是计算单个 vector-Jacobian 乘积，我们为批次中的每个“向量”计算一批 vector-Jacobian 乘积。
            我们使用 vmap 原型特性作为后端来向量化对自动求导引擎的调用，以便可以在单次调用中执行此计算。
            这应该导致性能改进，与手动循环并多次向后传播相比。请注意，由于此功能是实验性的，可能存在性能突变。
            请使用 ``torch._C._debug_only_display_vmap_fallback_warnings(True)`` 显示任何性能警告，
            如果在您的用例中存在警告，请在 GitHub 上提交问题。
            默认为 ``False``。
        materialize_grads (bool, optional): 如果为 ``True``，则将未使用的输入的梯度设置为零，而不是 None。
            在计算更高阶导数时，这是有用的。如果 ``materialize_grads`` 为 ``True``，并且 ``allow_unused`` 为 ``False``，
            将引发错误。默认为 ``False``。
    """
    if materialize_grads and allow_unused is False:
        raise ValueError(
            "Expected allow_unused to be True or not passed when materialize_grads=True, "
            "but got: allow_unused=False."
        )
    if allow_unused is None:
        allow_unused = materialize_grads
    # 将 outputs 转换为 Tuple[torch.Tensor, ...] 类型，如果 outputs 不是 tensor-like 则转换为元组
    t_outputs = cast(
        Tuple[torch.Tensor, ...],
        (outputs,) if is_tensor_like(outputs) else tuple(outputs),
    )
    
    # 如果 inputs 是 tensor-like 或者是 graph.GradientEdge 的实例，则转换为元组 _TensorOrTensorsOrGradEdge
    if is_tensor_like(inputs) or isinstance(inputs, graph.GradientEdge):
        inputs = cast(_TensorOrTensorsOrGradEdge, (inputs,))
    else:
        inputs = tuple(inputs)
    
    # 从 inputs 中选出所有的 tensor-like 对象，组成元组 t_inputs
    t_inputs = tuple(i for i in inputs if is_tensor_like(i))
    
    # 将 t_outputs 和 t_inputs 合并成 overridable_args
    overridable_args = t_outputs + t_inputs
    
    # 如果 overridable_args 中有需要重载的 Torch 函数，则调用 handle_torch_function 处理
    if has_torch_function(overridable_args):
        return handle_torch_function(
            grad,
            overridable_args,
            t_outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
            is_grads_batched=is_grads_batched,
            materialize_grads=materialize_grads,
        )
    
    # 如果 only_inputs 不为 True，则发出警告，因为该参数已经被废弃
    if not only_inputs:
        warnings.warn(
            "only_inputs argument is deprecated and is ignored now "
            "(defaults to True). To accumulate gradient for other "
            "parts of the graph, please use torch.autograd.backward.",
            FutureWarning,
            stacklevel=2,
        )
    
    # 将 grad_outputs 转换为元组，长度与 t_outputs 相同
    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(t_outputs))
    
    # 根据 t_outputs 和 grad_outputs_ 构建梯度
    grad_outputs_ = _make_grads(
        t_outputs, grad_outputs_, is_grads_batched=is_grads_batched
    )
    
    # 如果 retain_graph 未指定，则设为 create_graph 的值
    if retain_graph is None:
        retain_graph = create_graph
    
    # 如果 is_grads_batched 为 True，则定义函数 vjp，通过 _engine_run_backward 处理反向传播
    if is_grads_batched:
    
        def vjp(gO):
            return _engine_run_backward(
                t_outputs,
                gO,
                retain_graph,
                create_graph,
                inputs,
                allow_unused,
                accumulate_grad=False,
            )
    
        # 使用 _vmap_internals._vmap 对 vjp 进行批处理映射，allow_none_pass_through=True 表示允许 None 通过
        result = _vmap_internals._vmap(vjp, 0, 0, allow_none_pass_through=True)(
            grad_outputs_
        )
    else:
        # 否则直接调用 _engine_run_backward 处理反向传播
        result = _engine_run_backward(
            t_outputs,
            grad_outputs_,
            retain_graph,
            create_graph,
            inputs,
            allow_unused,
            accumulate_grad=False,
        )
    
    # 如果 materialize_grads 为 True，则确保结果中没有 None 的项，对应 inputs 是 GradientEdge 的情况引发 RuntimeError
    if materialize_grads:
        if any(
            result[i] is None and not is_tensor_like(inputs[i])
            for i in range(len(inputs))
        ):
            raise RuntimeError(
                "materialize_grads cannot be used when the given input is a GradientEdge"
            )
        # 将结果中的 None 替换为与对应的 input 形状相同的零张量，并确保需要梯度计算
        result = tuple(
            output
            if output is not None
            else torch.zeros_like(input, requires_grad=True)
            for (output, input) in zip(result, inputs)
        )
    
    # 返回最终的结果
    return result
# 这个函数适用于梯度检查点以进行内存优化。目前，只有在通过torch.autograd.backward()调用执行引擎并且没有传递其inputs参数时，才支持梯度检查点。不支持torch.autograd.grad()。这是因为如果指定了inputs，梯度将不会计算其他内容，例如模型参数如权重、偏置等。
# 此函数返回检查点是否有效，即torch.autograd.backward还是torch.autograd.grad。实现通过在torch/csrc/autograd/engine.cpp中维护一个线程本地变量来工作，该变量查看堆栈中的NodeTask，并在evaluate_function中执行NodeTask之前检查是否需要执行可重入的反向传播。
# 有关更多讨论/背景，请参见https://github.com/pytorch/pytorch/pull/4594
def _is_checkpoint_valid():
    return Variable._execution_engine.is_checkpoint_valid()


def variable(*args, **kwargs):  # noqa: D103
    raise RuntimeError(
        "torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead"
    )


# 为了修复FX代码生成，对variable.Variable进行Monkey patching。FX通过大致执行f"{fn.__module__}.{fn.__name__}(...)来生成调用。这会在FX图的输出中产生torch.autograd.variable.Variable(...)。不幸的是，模块名称torch.autograd.variable被已弃用的函数 - variable(...)所遮蔽。
variable.Variable = Variable  # type: ignore[attr-defined]

if not torch._C._autograd_init():
    raise RuntimeError("autograd initialization failed")

# 导入所有本地方法/类
from torch._C._autograd import (
    _add_metadata_json,
    _disable_profiler,
    _disable_profiler_legacy,
    _enable_profiler,
    _enable_profiler_legacy,
    _enable_record_function,
    _get_sequence_nr,
    _kineto_step,
    _KinetoEvent,
    _pop_saved_tensors_default_hooks,
    _prepare_profiler,
    _profiler_enabled,
    _ProfilerResult,
    _push_saved_tensors_default_hooks,
    _record_function_with_args_enter,
    _record_function_with_args_exit,
    _set_empty_test_observer,
    _supported_activities,
    DeviceType,
    kineto_available,
    ProfilerEvent,
    SavedTensor,
)

from torch._C._profiler import ProfilerActivity, ProfilerConfig, ProfilerState

from . import profiler


def _register_py_tensor_class_for_device(device, cls):
    if not isinstance(cls, type):
        raise RuntimeError("cls isn't a typeinfo object")
    torch._C._register_py_class_for_device(device, cls)


is_multithreading_enabled = torch._C._is_multithreading_enabled
torch._C._add_docstr(
    is_multithreading_enabled, "Returns True if multithreading is currently enabled."
)

is_view_replay_enabled = torch._C._is_view_replay_enabled
torch._C._add_docstr(
    is_view_replay_enabled, "Returns True if view-replay is currently enabled."
)
```