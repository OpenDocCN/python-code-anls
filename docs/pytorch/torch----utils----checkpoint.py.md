# `.\pytorch\torch\utils\checkpoint.py`

```py
# 设置 mypy 来允许未类型化的函数定义
# 导入必要的库和模块
import contextlib  # 上下文管理器相关功能的标准库
import platform  # 获取平台信息的标准库
import uuid  # 生成唯一标识符的标准库
import warnings  # 警告相关功能的标准库
import weakref  # 弱引用相关功能的标准库
from collections import defaultdict  # 默认字典的标准库
from typing import *  # 导入类型提示的所有内容，忽略 F403 错误
import enum  # 枚举类型相关功能的标准库
from weakref import ReferenceType  # 引用类型的标准库

# 导入 PyTorch 库及其模块
import torch  # PyTorch 主库
import torch.fx.traceback as fx_traceback  # PyTorch FX 的 traceback 模块
from torch._functorch._aot_autograd.functional_utils import is_fun  # PyTorch 的 AOT 自动求导功能的工具函数
from torch.utils._pytree import tree_map  # PyTorch 工具模块中的树映射函数
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode  # 用于测试中的内部日志记录相关模块
from torch.utils._python_dispatch import TorchDispatchMode  # PyTorch 的分发模式

__all__ = [
    "checkpoint",  # 导出的符号名称列表
    "checkpoint_sequential",
    "CheckpointError",
    "CheckpointFunction",
    "check_backward_validity",
    "detach_variable",
    "get_device_states",
    "set_device_states",
    "noop_context_fn",
    "set_checkpoint_early_stop",
    "DefaultDeviceType",
    "set_checkpoint_debug_enabled",
    "CheckpointPolicy",
    "SelectiveCheckpointContext",
    "create_selective_checkpoint_contexts",
    "SAC_IGNORED_OPS",
]

# 默认的确定性模式
_DEFAULT_DETERMINISM_MODE = "default"

# 用于控制是否启用检查点调试信息的全局变量
_checkpoint_debug_enabled: Optional[bool] = None


@contextlib.contextmanager
def set_checkpoint_debug_enabled(enabled: Optional[bool]):
    """
    Context manager that sets whether checkpoint should print additional debug
    information when running. See the ``debug`` flag for
    :func:`~torch.utils.checkpoint.checkpoint` for more information. Note that
    when set, this context manager overrides the value of ``debug`` passed to
    checkpoint. To defer to the local setting, pass ``None`` to this context.

    Args:
        enabled (bool): Whether checkpoint should print debug information.
            Default is 'None'.
    """
    global _checkpoint_debug_enabled
    try:
        prev = _checkpoint_debug_enabled
        _checkpoint_debug_enabled = enabled
        yield
    finally:
        _checkpoint_debug_enabled = prev


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    """
    Detaches tensors from the computation graph and retains certain properties.

    Args:
        inputs (Tuple[Any, ...]): Tuple of input tensors.

    Returns:
        Tuple[torch.Tensor, ...]: Tuple of detached tensors with retained properties.
    """
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()  # 分离张量，使其不再连接到计算图
            x.requires_grad = inp.requires_grad  # 保留 requires_grad 属性
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )


def check_backward_validity(inputs: Iterable[Any]) -> None:
    """
    Checks if any input tensors require gradients and warns if none do.

    Args:
        inputs (Iterable[Any]): Iterable containing input tensors.
    """
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn(
            "None of the inputs have requires_grad=True. Gradients will be None"
        )


def _get_device_module(device="cuda"):
    """
    Helper function to get the device module based on device type.

    Args:
        device (str): Device type string. Defaults to "cuda".

    Returns:
        module: Torch module corresponding to the device type.
    """
    device_module = getattr(torch, device)  # 获取指定设备类型的 Torch 模块
    return device_module


class DefaultDeviceType:
    """
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    """
    r"""
    用于管理检查点默认设备类型的类。

    如果没有非 CPU 张量存在，则默认设备类型将会
    """
    # 默认的设备类型，初始化为 'cuda'
    _default_device_type = "cuda"

    @staticmethod
    def set_device_type(device: str = "cuda"):
        """
        设置用于检查点的默认设备类型。

        Args:
            device (str): 要设置为默认的设备类型。默认为 'cuda'。
        """
        # 将默认设备类型设置为指定的设备类型
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        获取当前用于检查点的默认设备类型。

        Returns:
            str: 当前的默认设备类型。
        """
        # 返回当前默认的设备类型
        return DefaultDeviceType._default_device_type
def _infer_device_type(*args):
    device_types = []

    def add_device_types(arg):
        nonlocal device_types
        # 检查参数是否为 torch.Tensor 类型且不在 CPU 设备上
        if isinstance(arg, torch.Tensor) and not arg.device.type == "cpu":
            # 将设备类型添加到列表中
            device_types.append(arg.device.type)
    # 对参数列表应用 add_device_types 函数
    tree_map(add_device_types, args)

    # 将设备类型列表转换为集合，以便去除重复的设备类型
    device_types_set = set(device_types)
    # 如果检测到多于一个设备类型
    if len(device_types_set) > 1:
        # 发出警告，说明发现了至少两种类型设备上的张量参数（不包括 CPU 设备）
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
            f"\nDevice types: {sorted(device_types_set)} first device type: {device_types[0]}"
        )
    # 如果没有检测到任何设备类型
    if len(device_types) == 0:
        # 返回默认设备类型
        return DefaultDeviceType.get_device_type()
    # 如果检测到 "cuda" 设备类型
    elif "cuda" in device_types_set:
        return "cuda"
    # 否则返回第一个检测到的设备类型
    else:
        return device_types[0]


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_device_ids = []

    def add_device_ids(arg):
        nonlocal fwd_device_ids
        # 检查参数是否为 torch.Tensor 类型且不在 CPU 设备上
        if isinstance(arg, torch.Tensor) and not arg.device.type == "cpu":
            # 将设备 ID 添加到列表中
            fwd_device_ids.append(arg.get_device())
    # 对参数列表应用 add_device_ids 函数
    tree_map(add_device_ids, args)

    fwd_device_states = []
    # 确定设备模块，根据推断出的设备类型
    device_module = _get_device_module(_infer_device_type(*args))

    # 对每个设备 ID 执行以下操作
    for device_id in fwd_device_ids:
        # 将设备切换到指定设备 ID 上
        with device_module.device(device_id):
            # 获取该设备的 RNG 状态并添加到列表中
            fwd_device_states.append(device_module.get_rng_state())

    return fwd_device_ids, fwd_device_states


def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """
    # 如果未指定设备类型，则使用默认设备类型
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    # 根据设备类型获取相应的设备模块
    device_module = _get_device_module(device_type)
    
    # 遍历设备列表和状态列表，逐一设置设备的随机数生成器状态
    for device, state in zip(devices, states):
        # 在设备模块上下文中，将当前设备设为活跃设备
        with device_module.device(device):
            # 设置当前设备的随机数生成器状态为给定的状态
            device_module.set_rng_state(state)
    # 定义一个静态方法 forward，用于执行前向传播
    def forward(ctx, run_function, preserve_rng_state, *args):
        # 检查反向传播的有效性
        check_backward_validity(args)
        # 将运行函数和是否保留 RNG 状态保存到上下文对象
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        
        # 推断设备类型（CPU 还是 CUDA）
        ctx.device_type = _infer_device_type(*args)
        # 获取自动混合精度的配置参数
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        
        # 如果要保留 RNG 状态
        if preserve_rng_state:
            # 在 CPU 状态下保存当前的 RNG 状态
            ctx.fwd_cpu_state = torch.get_rng_state()
            # 避免意外地提前初始化 CUDA 上下文
            ctx.had_device_in_fwd = False
            # 获取设备模块，并检查设备是否已初始化
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                # 如果设备已初始化，保存设备状态和设备的状态信息
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)
        
        # 将非张量输入保存在 ctx.inputs 中，并为张量输入保留占位符
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                # 如果是张量，则将其索引添加到 tensor_indices 中，并将输入保留为 None
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                # 如果不是张量，则直接添加到输入列表中
                ctx.inputs.append(arg)

        # 保存张量输入，以备反向传播时使用
        ctx.save_for_backward(*tensor_inputs)

        # 在不计算梯度的情况下执行运行函数
        with torch.no_grad():
            outputs = run_function(*args)
        
        # 返回前向传播的输出
        return outputs
    # 定义一个函数 backward，接受上下文 ctx 和任意数量的参数 args
    def backward(ctx, *args):
        # 检查当前的 checkpoint 状态是否有效，若无效则抛出 RuntimeError
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )
        
        # 复制输入列表，避免修改原始列表
        inputs = list(ctx.inputs)
        # 获取保存的张量索引
        tensor_indices = ctx.tensor_indices
        # 获取保存的张量
        tensors = ctx.saved_tensors
        # 根据设备类型获取设备模块
        device_module = _get_device_module(ctx.device_type)

        # 使用保存的张量填充输入列表的相应位置
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # 存储周围的 RNG 状态，并模拟前向传播期间的状态。完成后恢复周围的状态。
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)
            
            # 分离输入变量
            detached_inputs = detach_variable(tuple(inputs))

            # 自动混合精度上下文管理器
            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()
            
            # 启用梯度计算，并在自动混合精度上下文中执行
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                # 调用上下文中的运行函数，传入分离后的输入
                outputs = ctx.run_function(*detached_inputs)

        # 如果输出是 torch.Tensor，则转换为元组
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # 仅对需要梯度的张量运行 backward()
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        # 若没有需要梯度的输出张量，则抛出 RuntimeError
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary"
            )
        
        # 执行反向传播，只对需要梯度的张量执行
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        
        # 收集梯度结果，如果输入不是 torch.Tensor，则对应位置填入 None
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        # 返回一个元组，包含两个 None 和收集到的梯度
        return (None, None) + grads
# 定义一个名为 noop_context_fn 的函数，返回两个空的上下文管理器
def noop_context_fn():
    return contextlib.nullcontext(), contextlib.nullcontext()

# 使用装饰器 @torch._disable_dynamo 禁用 TorchDynamo 对该函数的优化
@torch._disable_dynamo
# 定义名为 checkpoint 的函数，用于模型或其部分的检查点操作
def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.
    # 定义一个注释块，详细说明了带有 `use_reentrant=True` 和 `use_reentrant=False` 两种参数的 `checkpoint` 函数的差异：
    
    .. note::
    
        The reentrant variant of checkpoint (`use_reentrant=True`) and
        the non-reentrant variant of checkpoint (`use_reentrant=False`)
        differ in the following ways:
    
        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes `function` in its entirety during
          the backward pass.
    
        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under :func:`torch.no_grad`.
          The non-reentrant version does record the autograd graph, allowing one
          to perform backward on the graph within checkpointed regions.
    
        * The reentrant checkpoint only supports the :func:`torch.autograd.backward`
          API for the backward pass without its `inputs` argument, while the
          non-reentrant version supports all ways of performing the backward pass.
    
        * At least one input and output must have `requires_grad=True` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.
    
        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc.) as participating in autograd,
          while the non-reentrant version does.
    
        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the non-reentrant
          version does. For the reentrant variant, if the checkpointed segment
          contains tensors detached using `detach()` or with :func:`torch.no_grad`,
          the backward pass will raise an error. This is because `checkpoint` makes
          all the outputs require gradients and this causes issues when a tensor
          is defined to have no gradient in the model. To avoid this, detach the
          tensors outside of the `checkpoint` function.
    # 定义函数，用于实现模型或模型的部分的前向传播。
    # 函数需要能够处理作为元组传递的输入。例如，在LSTM中，如果用户传递了``(activation, hidden)``，
    # :attr:`function` 应该正确使用第一个输入作为 ``activation``，第二个输入作为 ``hidden``。
    Args:
        function: 描述在模型或模型的前向传播中运行的内容。应该能够正确处理作为元组传递的输入。
        preserve_rng_state(bool, optional): 是否保留每个检查点期间的随机数生成器状态。在 torch.compile 下，此标志不起作用，总是会保留 RNG 状态。默认为 ``True``。
        use_reentrant(bool): 指定是否使用需要可重入自动求导的激活检查点变体。必须显式传递此参数。在2.4版本中，如果未传递 ``use_reentrant``，将会引发异常。如果 ``use_reentrant=False``，``checkpoint`` 将使用不需要可重入自动求导的实现。这使得 ``checkpoint`` 能够支持额外的功能，例如与 ``torch.autograd.grad`` 预期的工作以及支持传递给检查点函数的关键字参数。
        context_fn(Callable, optional): 一个可调用对象，返回两个上下文管理器的元组。函数及其重新计算将分别在第一个和第二个上下文管理器下运行。仅当 ``use_reentrant=False`` 时支持此参数。
        determinism_check(str, optional): 指定要执行的确定性检查的字符串。默认设置为 ``"default"``，将重新计算的张量的形状、dtype 和设备与保存的张量进行比较。要关闭此检查，请指定 ``"none"``。目前仅支持这两个值。如果需要更多确定性检查，请提交问题。仅当 ``use_reentrant=False`` 支持此参数；如果 ``use_reentrant=True``，则始终禁用确定性检查。
        debug(bool, optional): 如果为 ``True``，错误消息还将包括原始前向计算期间运行的操作以及重新计算时的跟踪。仅当 ``use_reentrant=False`` 支持此参数。
        args: 包含传递给 :attr:`function` 的输入的元组。

    Returns:
        运行 :attr:`function` 在 :attr:`*args` 上的输出。
    # 如果 use_reentrant 参数为 None，则发出警告，建议显式传递该参数
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. In version 2.4 we will raise an exception "
            "if use_reentrant is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=2
        )
        # 将 use_reentrant 设置为 True，作为默认行为的保留
        use_reentrant = True

    # 用于在 Python 2.7 兼容的方式中混合 *args 和 **kwargs 的小技巧
    preserve = kwargs.pop("preserve_rng_state", True)
    # 如果 kwargs 非空且 use_reentrant 为 True，则抛出 ValueError 异常
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    # 如果 use_reentrant 为 True
    if use_reentrant:
        # 如果 context_fn 不是 noop_context_fn 或 debug 不是 False，则抛出 ValueError 异常
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        # 调用 CheckpointFunction.apply 方法，返回其结果
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        # 调用 _checkpoint_without_reentrant_generator 方法生成 gen
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # 运行前向逻辑的预处理
        next(gen)
        # 调用 function(*args, **kwargs)，得到返回值 ret
        ret = function(*args, **kwargs)
        # 运行前向逻辑的后处理
        try:
            next(gen)
        except StopIteration:
            # 如果 gen 已经迭代完毕，则直接返回 ret
            return ret
# 定义一个函数来对顺序模型进行分段检查点，以节省内存空间
def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    r"""Checkpoint a sequential model to save memory.
    
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will not store
    the intermediate activations. The inputs of each checkpointed segment will
    be saved for re-running the segment in the backward pass.
    
    .. warning::
        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True` variant, please see
        :func:`~torch.utils.checkpoint.checkpoint` for
        the important considerations and limitations of this variant. It is
        recommended that you use ``use_reentrant=False``.
    
    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.
    
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.4 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
    
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    
    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # 如果 use_reentrant 参数为 None，则发出警告并设置为 True
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint.checkpoint_sequential: the use_reentrant "
            "parameter should be passed explicitly. "
            "In version 2.4 we will raise an exception if use_reentrant "
            "is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants."
        )
        use_reentrant = True

    # 为了在兼容 Python 2.7 的方式下处理关键字参数，提取并移除 'preserve_rng_state' 参数
    preserve = kwargs.pop("preserve_rng_state", True)
    # 如果 kwargs 字典非空，抛出异常，显示出现了意外的关键字参数
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    # 定义一个嵌套函数 run_function，用于执行一系列函数
    def run_function(start, end, functions):
        def forward(input):
            # 对从 start 到 end 的函数列表进行迭代，依次应用到输入上
            for j in range(start, end + 1):
                input = functions[j](input)
            return input

        return forward

    # 如果 functions 是 torch.nn.Sequential 类型，则将其转换为其子模块列表
    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # 计算每个段的大小
    segment_size = len(functions) // segments
    # 最后一个块必须是非易失的
    end = -1
    # 遍历每个段的起始位置
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        # 使用 checkpoint 函数对当前段的函数序列进行处理，传递相应参数
        input = checkpoint(
            run_function(start, end, functions),
            input,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve,
        )
    # 执行剩余的函数并返回最终结果
    return run_function(end + 1, len(functions) - 1, functions)(input)
# 定义一个内部断言函数，用于检查条件是否为真，如果条件为假则抛出断言错误
def _internal_assert(cond):
    # 如果条件不满足，则抛出 AssertionError 异常，包含详细的错误信息
    if not cond:
        raise AssertionError(
            "Something went unexpectedly wrong in activation checkpoint. "
            "Please report this bug by filing an issue to PyTorch."
        )


# NOTE [ Nestable Checkpoint ]
#
# 嵌套检查点的语义可以通过两个基本规则定义。
# 遵循这两个规则会导致一个重要的推论，这对于推动设计至关重要。
#
# 规则 1. 保存的张量仅由最内部的检查点管理，并且对任何外层检查点都是隐藏的。
#
# 规则 2. 内部检查点的输入被视为保存到其父检查点的张量。
#
# 推论：要重新计算任何给定的保存张量，我们需要重新计算包围它的所有检查点。
#
# 为什么会有这个推论？为了在反向传播过程中解包一个保存的张量 X，我们需要重新计算最内层的检查点（规则 1），
# 而要重新计算该检查点，我们需要其输入，这些输入由该检查点的父检查点管理（规则 2），因此必须首先重新计算父检查点。
# 持续这样的推理，我们意识到为了解包 X，需要重新计算 X 被保存时活动的所有检查点。（除非我们在该反向传播中已经为其他保存的张量做过了）
#
# 在实践中，我们使用一个空操作的 autograd 函数将输入保存为保存的张量。
# 在解包时调用 ctx.saved_tensor 会触发父检查点的重新计算。
#
# 规则 3. 我们应该从没有当前活动检查点的状态开始重新计算。遇到的检查点在重新计算期间仍然被尊重。
#
# 当我们开始重新计算时，我们将用于重新计算的保存变量钩子推入堆栈。请参见规则 6 中的示例以获取更多上下文。
#
#                                  * * * *
#
# 除了嵌套检查点特有的基本语义外，我们还施加了几个适用于检查点的一般约束。
#
# 规则 4. 重新计算张量的生命周期
#
#         重新计算的张量被视为特定于特定反向传播调用，并且在解包时立即清除。
#         特别地，即使 retain_graph=True，我们也要求立即清除这些张量。
#
# [ 规则 4 的实现细节 ]
#
# 如果我们可以接受在 retain_graph=True 情况下保留重新计算的张量，则可以将重新计算的变量存储为 WeakKeyDictionary 的值，
# 并且打包强引用到键，这样在反向传播时，只要 retain_graph=False，那些打包的键就会被清除。清除打包的键会清除 WKD 中的对应条目。
#
# 如果我们希望在 retain_graph=True 情况下立即清除解包时的重新计算变量，我们不能依赖于自动通过反向传播清除打包的键。
# 相反，我们需要使用强引用键的方式来立即清除它们。
# 直接地，我们封装一个容器对象，当我们解包时手动清除。
# 
# 一个重要的细节是，如果第二次反向传播发生，第二次重新计算需要用新创建的键重置容器。
# 
# 规则 5. 一旦我们已经重新计算了我们知道需要的保存的张量，就停止重新计算。
# 
# [ 规则 5 的实现细节 ]
# 
# 在重新计算过程中，如果重新计算的张量数量与我们预期要重新计算的张量数量相匹配，则抛出异常。
# 我们用 try-catch 包裹重新计算调用以捕获这个特定异常。参见下面的规则 6 的一些示例。
# 
# 规则 6. 我们支持在检查点上下文中执行反向传播
# 
# [ retain_graph is True ]
# 
# def fn(x):
#   y = x.sin()
#   z = y.cos()
#   gx, = torch.autograd.grad(z, x, retain_graph=True)
#   return gx, z
# 
# out = checkpoint(fn)(inp)
# out.backward()
# 
# 因为 z 是在启用检查点时由 cos 保存的，它实际上并没有保存，所以在 .grad() 调用中必须触发重新计算。
# 
# 在重新计算期间，“内部打包钩子”有两个责任：
# 
# 1) 像往常一样，填充 WeakKeyDictionary 存储重新计算的张量
# 2) 打包实际的张量（分离的），以便可以在重新计算图上执行反向传播。保存到该图的张量将存活直到重新计算结束，或者如果有人在 retain_graph=False 时执行了反向传播，则提前死亡。
# 
# 更普遍地，在重新计算图上执行反向传播发生在以下情况下：
# - 如果在前向过程中执行反向传播，
#   - 在原始前向过程中如果禁用了早停
#   - 在原始反向传播中
# - 如果有多个 .grad()/.backward() 调用，即使启用了早停，我们也会在重新计算图上执行反向传播（见下面的示例）
# 
# [ retain_graph is False ]
# 
# 下面的示例展示了在重新计算期间，如果我们发现某些要重新计算的张量已经被清除了会发生什么。
# 
# 结果：我们不做任何特殊处理，只是跳过它们！
# 
# def fn(x):
#   y = x.sin()                           # (1)
#   z = y.cos()                           # (2)
#   gx, = torch.autograd.grad(z, x)       # (3)
#   return x.cos() * gx                   # (4)
# 
# out = checkpoint(fn)(inp)
# out.backward()                          # (5)
# 
# 1, 2. 由于我们在检查点内部，不保存 x 和 y。
# 3. 触发 fn 的重新计算，因为 x 和 y 没有被保存。取决于是否启用了早停，要么在 (2) 停止，要么继续运行函数。
#    因为我们在 backward 时使用了 retain_graph=False，清除了 x 和 y 的持有者。
# 4. 由于我们在检查点内部，不保存 x。
# 5. 调用 backward 触发 fn 的另一个重新计算。在重新计算过程中，我们看到 x 和 y 在原始图中已经被清除，如所示。
# 开启或关闭早停检查点功能的全局变量，默认为 True
_enable_checkpoint_early_stop = True


@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    """设置检查点是否应该尽早停止重新计算的上下文管理器。

    默认情况下，非可重入检查点会在计算完所有所需张量后停止重新计算。
    此上下文管理器可以用于禁用特定应用程序中可能有问题的此功能。

    此上下文管理器只在运行前向传播时需要激活，不需要在后向传播期间激活。

    示例::

        >>> # xdoctest: +SKIP(failing)
        >>> message = "saved tensors default hooks are disabled"
        >>> with set_checkpoint_early_stop(False):
        ...     # 任何处于此上下文管理器下的检查点都将遵守此上下文管理器的设置，
        ...     # 即使其后向传播是在外部执行的。
        ...     out = checkpoint(fn, inputs)
        ...
        >>> out.backward()
    """
    global _enable_checkpoint_early_stop
    try:
        prev = _enable_checkpoint_early_stop
        _enable_checkpoint_early_stop = enable
        yield
    finally:
        _enable_checkpoint_early_stop = prev


class _Handle:
    pass


class _Holder:
    def __init__(self):
        # 用于存储持有对象的字典，键为整数，值为可选的 _Handle 对象
        self.handles: Dict[int, Optional[_Handle]] = dict()


class _NoopSaveInputs(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        # 返回一个空的张量作为前向传播的输出
        return torch.empty((0,))

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        # 只有张量可以使用 ctx.save_for_backward 保存，其它一切都被 get_args 捕获，
        # 并直接保存在 ctx 上
        tensor_indices, tensors = zip(
            *[(i, o) for i, o in enumerate(inputs) if isinstance(o, torch.Tensor)]
        )
        # 创建一个索引映射，将原始输入张量的索引映射到保存张量的索引
        idx2saved_idx = {b: a for a, b in enumerate(tensor_indices)}
        # 将张量替换为 None 作为占位符，其它输入保持不变
        args = [None if isinstance(o, torch.Tensor) else o for o in inputs]

        def get_args(saved_tensors):
            # 使用 ctx.saved_tensors 恢复占位符，并将其替换为原始张量
            ret = [
                saved_tensors[idx2saved_idx[i]] if i in tensor_indices else o
                for i, o in enumerate(args)
            ]
            # 返回除了第一个元素（占位符）之外的其余元素作为结果
            return ret[1:]

        # 将 get_args 函数绑定到 ctx 上
        ctx.get_args = get_args
        # 使用 ctx.save_for_backward 保存所有张量
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 不期望在此图上执行后向传播，抛出异常
        raise AssertionError("Did not expect to backward on this graph")


class _CheckpointFrame:
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn):
        # 初始化对象时，接收重新计算函数、早期停止标志、解包错误回调和元数据函数作为参数
        self.recompute_fn = recompute_fn
        # 初始化输入保存器为None
        self.input_saver = None
        # 初始化弱引用持有者列表为空列表
        self.weak_holders: List[ReferenceType] = []

        # 将self.recomputed初始化为默认字典，键为int类型，值为WeakKeyDictionary类型的弱引用字典
        # 这样做是为了在部分反向传播时，字典中的条目会随着Holder被清除而清除，
        # 当SavedVariable被清除时，Holder也会被移除。
        self.recomputed: DefaultDict[
            int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]
        ] = defaultdict(weakref.WeakKeyDictionary)

        # 初始化self.recomp_counter为默认字典，键为int类型，值为int类型的计数器，默认值为0
        self.recomp_counter: DefaultDict[int, int] = defaultdict(int)

        # 初始化self.is_recomputed为默认字典，键为int类型，值为bool类型，默认值为False
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(bool)

        # 设置早期停止标志
        self.early_stop = early_stop

        # 调试信息
        # 设置元数据函数
        self.metadata_fn = metadata_fn
        # 设置解包错误回调函数
        self.unpack_error_cb = unpack_error_cb
        # 初始化x_metadatas列表为空列表
        self.x_metadatas = []
        # 设置前向传播完成标志为False
        self.forward_completed = False
        # 设置忽略保存不匹配标志为False
        self.ignore_saved_mismatch = False
# 定义模板字符串，用于展示错误日志和操作记录，当使用 `torch.utils.checkpoint.checkpoint()` 函数时出错时使用
_checkpoint_error_template = """ \
An error happened while unpacking tensors; dumping logs of latest computation
because you passed `debug=True` to `torch.utils.checkpoint.checkpoint()`.
Scroll all the way down for guidance on how to navigate these logs.

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        1. Stack traces of the operators that ran in the original forward     |
+------------------------------------------------------------------------------+

{forward_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        2. Stack traces of the operators that ran during recomputation        |
+------------------------------------------------------------------------------+

{recompute_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|       3. Log of operators in the original forward and recomputation          |
+------------------------------------------------------------------------------+
(Scroll up to correlate stack traces with each operation listed below. This
 helps identify their source in the code.)

IMPORTANT: Differences in "detach" calls between the original forward and the
           recomputation are expected. They are introduced by the checkpointing
           mechanism and can be ignored.

Operations executed during the original forward:

{forward_ops}

Operations executed during recomputation:

{recompute_ops}

+------------------------------------------------------------------------------+
 ERROR: Detected non-determinism while running activation checkpointing

 You are seeing this error because you passed `debug=True` to checkpoint and
 tensors to be saved during the original forward and differ between those saved
 during recomputation. This can happen if different operators were ran in the
 original forward and in the recomputation.

 To identify where the mismatch may be coming from, you can do the following:

 1) Compare the operators ran during original forward and recomputation to
    see where they differ. These operators are printed above in the order they
    were executed.

 2) Review the stack trace for each operator to locate its invocation source.
    Each operator's stack trace is printed in their execution order.

 Note that the logs can be quite long. Here's how they are structured:
 (Tip: you can Ctrl-f for these headers)

 1. Stack traces of the operators that ran in the original forward
 2. Stack traces of the operators that ran during recomputation
 3. Log of operators in the original forward and recomputation
 4. Error message                                             <--- You are here
--------------------------------------------------------------------------------
"""

# 定义一个自定义的异常类 CheckpointError，继承自 RuntimeError
class CheckpointError(RuntimeError):
    pass


# 定义一个返回元组的函数，包含两个类型为 Callable 的函数作为返回值
def _get_debug_context_and_cb() -> Tuple[Callable[[], Any], Callable[[CheckpointError], None]]:
    # This function returns the context_fn and error_cb to be used by the
    # 返回 context_fn 和 error_cb 函数，供外部调用使用
    # 检查点机制。当在解压过程中检测到错误时，调用 error_cb。

    # 检查当前系统是否为 Linux 平台且是 x86_64 架构，以确定是否支持 record_context_cpp。
    cpp_tb = platform.machine() == 'x86_64' and platform.system() == 'Linux'

    # 定义一个 CaptureLogs 类，用于捕获日志和堆栈信息。
    class CaptureLogs:
        def __init__(self):
            self.logs = None  # 日志内容
            self.tbs = None   # 堆栈信息

        def get_context_manager(self):
            @contextlib.contextmanager
            def logging_mode():
                # 进入日志张量模式，并捕获日志信息和堆栈信息
                with LoggingTensorMode(), \
                     capture_logs(True, python_tb=True, script_tb=True, cpp_tb=cpp_tb) as logs_and_tb:
                    self.logs, self.tbs = logs_and_tb  # 将捕获的日志和堆栈信息保存到实例中
                    yield logs_and_tb  # 返回捕获的日志和堆栈信息
            return logging_mode()

    # 创建两个 CaptureLogs 的实例，用于前向传播和重新计算过程中的日志捕获
    capture_logs_fwd = CaptureLogs()
    capture_logs_recompute = CaptureLogs()

    # 定义 unpack_error_cb 函数，用于处理 CheckpointError 异常
    def unpack_error_cb(e: CheckpointError):
        # 获取日志和堆栈信息的字符串表示
        def get_str_tb(label, capture_logs):
            out = ""
            total_len = len(capture_logs.logs)
            for i, (log, tb) in enumerate(zip(capture_logs.logs, capture_logs.tbs)):
                out += f"{log}   ({i + 1} of {total_len} in {label})\n\n"
                found_torch_dispatch = False
                for line in tb:
                    # 只在找到 '__torch_dispatch__' 后开始打印堆栈跟踪
                    is_torch_dispatch = line['name'] == '__torch_dispatch__'
                    if not found_torch_dispatch and not is_torch_dispatch:
                        continue
                    elif is_torch_dispatch:
                        found_torch_dispatch = True
                        continue
                    out += f"{line['filename']}:{line['line']}:{line['name']}\n"
                out += "\n\n"
            return out
        
        # 断言确保捕获的日志不为 None
        assert capture_logs_fwd.logs is not None
        assert capture_logs_recompute.logs is not None
        
        # 抛出 CheckpointError 异常，包含格式化的错误消息和捕获的日志信息
        raise CheckpointError(
            _checkpoint_error_template.format(
                forward_traces=get_str_tb("original", capture_logs_fwd),
                recompute_traces=get_str_tb("recompute", capture_logs_recompute),
                forward_ops="\n".join(capture_logs_fwd.logs),
                recompute_ops="\n".join(capture_logs_recompute.logs)
            )
        ) from e

    # 定义 context_fn 函数，返回前向传播和重新计算过程中的日志捕获上下文管理器
    def context_fn():
        return capture_logs_fwd.get_context_manager(), capture_logs_recompute.get_context_manager()

    # 返回 context_fn 和 unpack_error_cb 函数作为结果
    return context_fn, unpack_error_cb
# 定义一个函数，用于提取 Torch 张量的元信息，返回一个包含形状、数据类型和设备的字典
def _default_meta_extractor(x: torch.Tensor) -> Dict[str, Any]:
    # 这些属性检查速度快且易于理解
    return {
        "shape": x.shape,   # 获取张量的形状
        "dtype": x.dtype,   # 获取张量的数据类型
        "device": x.device  # 获取张量所在设备信息
    }

# 允许的确定性检查到函数的映射字典，包括默认的确定性模式和"none"模式
_allowed_determinism_checks_to_fns: Dict[str, Callable[[torch.Tensor], Any]] = {
    _DEFAULT_DETERMINISM_MODE: _default_meta_extractor,  # 默认模式使用默认元信息提取函数
    "none": lambda _: None,  # "none"模式返回空值的匿名函数
}

# 定义一个自定义异常类，用于表示停止重新计算错误
class _StopRecomputationError(Exception):
    pass

# 定义一个继承自torch.autograd.graph.saved_tensors_hooks的类
class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: int):
        # 定义一个打包钩子函数，用于保存重新计算过程中的张量
        def pack_hook(x):
            target_frame = target_frame_ref()
            assert target_frame is not None  # appease mypy

            # 获取当前重新计算的索引，并增加计数
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1

            # 检查是否超过了保存张量的容器的长度
            if recomp_idx >= len(target_frame.weak_holders):
                assert not target_frame.early_stop
                if not target_frame.forward_completed:
                    # 当不允许提前停止并在检查点内进行梯度时，设置标志以避免后续错误
                    target_frame.ignore_saved_mismatch = True
                    return x.detach()
                # 如果超过了长度，抛出错误
                raise CheckpointError(
                    "torch.utils.checkpoint: trying to save more tensors during "
                    "recomputation than during the original forward pass."
                )

            # 获取当前索引处的弱引用对象
            holder = target_frame.weak_holders[recomp_idx]()

            # 如果弱引用对象存在，则保存当前张量到重新计算的容器中
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x.detach()

            # 如果设置了提前停止，并且当前计数等于弱引用对象的数量，则抛出停止重新计算错误
            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                target_frame.weak_holders
            ):
                raise _StopRecomputationError

            # 返回张量的分离副本
            return x.detach()

        # 定义一个解包钩子函数，返回未修改的输入张量
        def unpack_hook(x):
            # 返回未修改的张量，用于展示重新计算时可能出现的后向计算
            return x

        # 调用父类初始化方法，传入打包和解包钩子函数
        super().__init__(pack_hook, unpack_hook)


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame):
        # 定义打包钩子函数，用于创建一个新的 _Holder 对象并将其弱引用添加到 frame 的 weak_holders 列表中
        def pack_hook(x):
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # 如果定义了 metadata_fn 函数，则使用 torch.no_grad() 上下文保存 x 的元数据到 frame 的 x_metadatas 列表中
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        # 定义解包钩子函数
        def unpack_hook(holder):
            # 获取当前图任务的 ID
            gid = torch._C._current_graph_task_id()
            if gid == -1:
                # 如果当前不是在反向传播调用期间触发解包，则生成临时 ID
                gid = int(uuid.uuid4())

            # 如果未重新计算过 gid 对应的节点，则执行重新计算过程
            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)

                try:
                    # 使用 _recomputation_hook 上下文管理器和 torch.autograd.enable_grad() 启用梯度计算，执行重新计算函数
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        frame.recompute_fn(*args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                # 检查重新计算的张量是否匹配
                frame.check_recomputed_tensors_match(gid)

            # 断言 gid 在 holder.handles 中
            _internal_assert(gid in holder.handles)

            # 如果 holder.handles[gid] 为 None，则抛出 CheckpointError
            if holder.handles[gid] is None:
                raise CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    "unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do "
                    "so only once. Otherwise please open an issue with details on your use case."
                )
            # 断言 holder.handles[gid] 在 frame.recomputed[gid] 中
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            # 返回 frame.recomputed[gid][holder.handles[gid]]，并将 holder.handles[gid] 设置为 None
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        # 如果定义了 unpack_error_cb 回调函数，则定义一个带错误处理的解包钩子函数
        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            # 调用父类的构造函数，使用 pack_hook 和 unpack_hook_with_error_cb 函数初始化对象
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            # 调用父类的构造函数，使用 pack_hook 和 unpack_hook 函数初始化对象
            super().__init__(pack_hook, unpack_hook)
# 检查是否在 AOTAutograd 追踪下编译
# 可能有更好的方法来执行此检查...
# TODO: 统一所有编译堆栈上的 _is_compiling
def _is_compiling(func, args, kwargs):
    for arg in args:
        # 检查参数是否为 torch.Tensor 类型且是函数
        if isinstance(arg, torch.Tensor) and is_fun(arg):
            return True
    return False


class _VersionWrapper:
    # 检查缓存的张量是否被修改
    def __init__(self, val):
        self.val: Union[torch.Tensor, Any] = val
        # 如果 val 是 torch.Tensor 类型，则记录其版本号；否则为 None
        self.version: Optional[int] = val._version if isinstance(val, torch.Tensor) else None

    def get_val(self, allow_cache_entry_mutation):
        # 如果版本号不为 None 且不允许缓存条目修改
        if self.version is not None and not allow_cache_entry_mutation:
            # 检查张量的版本是否被修改，如果被修改则抛出 RuntimeError
            if self.val._version != self.version:
                raise RuntimeError(
                    "Tensor cached during selective activation checkpoint has been mutated"
                )
        # 返回张量的值
        return self.val


def _maybe_detach(x, any_ret_has_alias_info):
    # 可能会分离张量的原因有两个：
    # - 对于视图操作，确保从 CachedDispatchMode 返回时，as_view 看到 AutogradMeta 是 nullptr
    # - 避免引用循环
    # 对于第一种情况，仅检查 x 是否具有可微分 dtype 是不够的，因为非可微分 dtype 可能有非 nullptr 的 AutogradMeta，例如当张量是视图时。
    if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex() or any_ret_has_alias_info):
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ADInplaceOrView, False):
            # 确保在 Autograd 下层视图操作正确传播版本计数器。
            # TODO: 使用 reentrant_dispatch 而不是手动操作 dispatch keys。使用 reentrant_dispatch 将尊重 inference_mode，但对此情况不相关。
            x = x.detach()
    return x


class SelectiveCheckpointContext:
    """
    在选择性检查点期间传递给策略函数的上下文。

    此类用于在选择性检查点期间向策略函数传递相关的元数据。元数据包括当前策略函数调用是否在重新计算期间。

    示例:
        >>> # xdoctest: +SKIP(stub)
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    print(ctx.is_recompute)
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    def __init__(self, *, is_recompute):
        # 是否在重新计算期间
        self.is_recompute = is_recompute


class CheckpointPolicy(enum.Enum):
    """
    检查点策略枚举类。
    """
    # 定义枚举，用于指定反向传播期间的检查点策略。
    
    # 支持以下策略：
    # - `MUST_SAVE`：操作的输出在前向传播期间将被保存，不会在反向传播期间重新计算
    # - `PREFER_SAVE`：操作的输出在前向传播期间将被保存，但可能会在反向传播期间重新计算
    # - `MUST_RECOMPUTE`：操作的输出在前向传播期间不会被保存，必须在反向传播期间重新计算
    # - `PREFER_RECOMPUTE`：操作的输出在前向传播期间不会被保存，可能会在反向传播期间重新计算
    
    # 使用 `MUST_*` 而不是 `PREFER_*` 可以表明该策略不应被像 `torch.compile` 这样的子系统覆盖。
    
    # 注意：
    # - 始终返回 `PREFER_RECOMPUTE` 的策略函数等同于普通的检查点技术。
    # - 如果每个操作都返回 `PREFER_SAVE`，则不等同于不使用检查点技术。这样的策略会保存额外的张量，
    #   不限于实际需要用于梯度计算的张量。
# 根据布尔值创建对应的检查点策略，用于向后兼容
def _policy_from_bool(b):
    return CheckpointPolicy.MUST_SAVE if b else CheckpointPolicy.PREFER_RECOMPUTE

# SAC_IGNORED_OPS 定义了一组被忽略的操作集合，用于 SAC 策略实现
SAC_IGNORED_OPS = {
    # 在前向传播和重计算期间，AC 插入了不同数量的 detach 操作。
    torch.ops.aten.detach.default,
    # AC 的确定性检查在前向传播期间调用了额外的元数据操作。
    # 如果这些操作被选中缓存，并且涉及子类，这些元数据操作变成可分派的，可能导致不正确性。
    torch.ops.prim.device.default,
} | set(torch._subclasses.functional_tensor.FunctionalTensor.metadata_fns)

# _CachingTorchDispatchMode 类继承自 TorchDispatchMode 类，用于实现 SAC 策略
class _CachingTorchDispatchMode(TorchDispatchMode):
    # 与 _CachedTorchDispatchMode 一起使用，实现 SAC。
    def __init__(self, policy_fn, storage):
        self.policy_fn = policy_fn
        self.storage = storage

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 如果 func 在 SAC_IGNORED_OPS 中，直接调用 func(*args, **kwargs)
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        # 获取检查点策略
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=False),
                                func, *args, **kwargs)
        # 如果策略是布尔型，则转换为相应的检查点策略对象
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        # 检查是否正在编译
        is_compiling = _is_compiling(func, args, kwargs)

        if is_compiling:
            # 覆盖每个节点的 "recompute" 标签以添加用户注释。
            fx_traceback.current_meta["recompute"] = policy

        # 调用 func(*args, **kwargs) 执行函数
        out = func(*args, **kwargs)

        # 检查函数返回中是否有任何别名信息
        any_ret_has_alias_info = any(ret.alias_info is not None for ret in func._schema.returns)

        # 如果策略是 MUST_SAVE 或 PREFER_SAVE，或者正在编译，则将结果存储起来
        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE) or is_compiling:
            self.storage[func].append(tree_map(lambda x: _VersionWrapper(_maybe_detach(x, any_ret_has_alias_info)), out))

        # 返回函数执行的结果 out
        return out

# _CachedTorchDispatchMode 类继承自 TorchDispatchMode 类，用于实现 SAC 策略
class _CachedTorchDispatchMode(TorchDispatchMode):
    # 与 _CachingTorchDispatchMode 一起使用，实现 SAC。
    def __init__(self, policy_fn, storage, allow_cache_entry_mutation):
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
    # 定义一个特殊方法 __torch_dispatch__，用于处理特定的 Torch 函数调用
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 如果 func 在 SAC_IGNORED_OPS 中，则直接调用 func 函数并返回结果
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        # 如果 kwargs 为 None，则初始化为空字典
        kwargs = {} if kwargs is None else kwargs
        # 根据当前的策略生成一个 policy 对象，使用 SelectiveCheckpointContext 作为参数
        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=True),
                                func, *args, **kwargs)
        # 如果 policy 是布尔类型，则将其转换成对应的策略对象
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        # 检查当前函数是否处于编译状态
        is_compiling = _is_compiling(func, args, kwargs)

        # 如果需要保存检查点或者当前正在编译中
        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE) or is_compiling:
            # 从存储中获取与 func 对应的数据
            storage = self.storage.get(func)
            # 如果 storage 为 None，则抛出运行时错误，表示在反向传播过程中找不到 func 对应的数据
            if storage is None:
                raise RuntimeError(f"{func} encountered during backward, but not found in storage")
            # 如果 storage 的长度为 0，则抛出运行时错误，表示尝试在选择性激活检查点下的同一区域多次执行反向传播
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            # 使用 tree_map 将 storage 的第一个元素的值映射出来，并从 storage 中移除该元素
            out = tree_map(lambda x: x.get_val(self.allow_cache_entry_mutation), storage.pop(0))
        else:
            # 否则，直接调用 func 函数，并传入相应的参数和关键字参数
            out = func(*args, **kwargs)
        
        # 返回处理结果 out
        return out
# 创建选择性检查点上下文的辅助函数，用于在激活检查点期间避免重新计算特定操作。

# Args:
#   policy_fn_or_list (Callable or List):
#     - 如果提供的是一个策略函数，它应该接受一个 SelectiveCheckpointContext 对象、OpOverload、操作的参数和关键字参数，
#       并返回一个 CheckpointPolicy 枚举值，指示是否应重新计算操作。
#     - 如果提供的是一个操作列表，相当于策略函数对指定操作返回 CheckpointPolicy.MUST_SAVE，
#       对其他操作返回 CheckpointPolicy.PREFER_RECOMPUTE。
#   allow_cache_entry_mutation (bool, optional):
#     默认情况下，如果选择性激活检查点缓存的任何张量被修改，会引发错误以确保正确性。
#     如果设置为 True，则禁用此检查。

# Returns:
#   返回两个上下文管理器的元组。

def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    # 如果 grad_mode 被禁用，checkpoint 在 context_fn 下不会运行前向传播，因此继续如常进行。
    # 如果 policy_fn_or_list 是一个列表
    if isinstance(policy_fn_or_list, list):
        # 遍历 policy_fn_or_list 中的每个操作 op
        for op in policy_fn_or_list:
            # 如果 op 不是 torch._ops.OpOverload 类型的对象
            if not isinstance(op, torch._ops.OpOverload):
                # 准备额外的错误信息，提示用户更新 OpOverloadPacket 到特定的 OpOverload
                _extra_msg = (
                    "Please update the OpOverloadPacket to a specific OpOverload."
                    "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                ) if isinstance(op, torch._ops.OpOverloadPacket) else ""
                # 抛出值错误，指出预期的 op 应为 OpOverload，但得到了不符合要求的 op 对象和类型
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        # 定义 policy_fn 函数，接受 ctx、op 和任意位置参数和关键字参数
        def policy_fn(ctx, op, *args, **kwargs):
            # 如果 op 在 policy_fn_or_list 中
            if op in policy_fn_or_list:
                # 返回必须保存策略
                return CheckpointPolicy.MUST_SAVE
            else:
                # 否则返回优先重新计算策略
                return CheckpointPolicy.PREFER_RECOMPUTE
    
    # 如果 policy_fn_or_list 是可调用对象
    elif callable(policy_fn_or_list):
        # 将 policy_fn 设置为 policy_fn_or_list
        policy_fn = policy_fn_or_list
    
    # 如果 policy_fn_or_list 不是列表也不是可调用对象，则抛出类型错误
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")
    
    # 初始化一个空的字典，存储任意类型到列表的映射关系
    storage: Dict[Any, List[Any]] = defaultdict(list)
    
    # 返回两个对象：使用 policy_fn 的缓存分发模式和带缓存的分发模式
    return (
        _CachingTorchDispatchMode(policy_fn, storage),
        _CachedTorchDispatchMode(policy_fn, storage, allow_cache_entry_mutation),
    )
# NB: this helper wraps fn before calling checkpoint_impl. kwargs and
#     saving/restoring of global state is handled here.

def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    *args,
    **kwargs
):
    """Checkpointing without reentrant autograd.

    Args:
        fn: The function to be checkpointed. It describes what to run in the forward pass of the model or part of the model.
        preserve_rng_state(bool, optional): Determines whether to omit stashing and restoring the RNG state during each checkpoint. Default: ``True``.
        context_fn(Callable, optional): A callable returning a tuple of two context managers. The function and its recomputation will be run under the first and second context managers respectively.
        determinism_check(str, optional): Specifies the type of determinism check to perform. Defaults to ``_DEFAULT_DETERMINISM_MODE``, which compares shapes, dtypes, and devices of recomputed tensors against saved tensors. Can be set to ``"none"`` to disable this check.
        debug(bool, optional): If ``True``, error messages will include a trace of operators ran during the original forward computation and the recomputation.
        *args: Positional arguments to pass to the given ``fn``.
        **kwargs: Keyword arguments to pass to the given ``fn``.
    """
    # Initialize unpack_error_cb to None
    unpack_error_cb = None

    # Check if debug is enabled and context_fn is not the default noop_context_fn
    if _checkpoint_debug_enabled if _checkpoint_debug_enabled is not None else debug:
        if context_fn != noop_context_fn:
            # Raise an error if debug=True is used with a non-default context_fn
            raise ValueError(
                "debug=True is incompatible with non-default context_fn"
            )
        # Obtain debug context and unpack error callback
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    # Validate determinism_check value against allowed checks
    if determinism_check in _allowed_determinism_checks_to_fns:
        # Retrieve the metadata function corresponding to determinism_check
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        # Raise an error if determinism_check is not in allowed checks
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    # Infer device type from args
    device_type = _infer_device_type(*args)
    # Get device module based on inferred device type
    device_module = _get_device_module(device_type)
    # Obtain context managers for forward pass and recomputation
    forward_context, recompute_context = context_fn()
    # 如果正在编译并且上下文函数不是 noop_context_fn
    if _is_compiling(fn, args, kwargs) and context_fn != noop_context_fn:
        # 断言：forward_context 和 recompute_context 必须是 TorchDispatchMode 类型
        assert (
            isinstance(forward_context, TorchDispatchMode) and
            isinstance(recompute_context, TorchDispatchMode)
        ), \
            "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` " + \
            "must generate a tuple of two `TorchDispatchMode`s."
    
    # 处理可能（远程）存在自动类型转换同时启用于 CPU 和 GPU 的情况
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device_type)

    # 如果需要保留 RNG 状态
    if preserve_rng_state:
        # 获取当前 CPU 的 RNG 状态
        fwd_cpu_state = torch.get_rng_state()
        # 如果用户意图稍后在其 run_function 内部初始化 CUDA 上下文，则不要意外地初始化 CUDA 上下文。
        # （如果用户打算在运行函数之前初始化上下文，我们实际上应该在这里存储 CUDA 状态。
        # 不幸的是，我们无法预测这将在我们运行函数之前发生。如果他们这样做，我们会引发错误。）
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            # 如果设备模块已初始化，则标记为 True
            had_device_in_fwd = True
            # 获取所有参数的设备状态和设备状态
            fwd_devices, fwd_device_states = get_device_states(*args)

    # 定义 recompute_fn 函数，用于稍后的重新计算
    def recompute_fn(*inputs):
        kwargs, *args = inputs
        # 如果需要保留 RNG 状态并且之前有设备在前向传播中被标记
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        # 在一个新的 RNG 上下文中运行函数
        with torch.random.fork_rng(
            devices=rng_devices, enabled=preserve_rng_state, device_type=device_type
        ):
            # 如果需要保留 RNG 状态
            if preserve_rng_state:
                # 恢复之前存储的 CPU RNG 状态
                torch.set_rng_state(fwd_cpu_state)
                # 如果之前有设备在前向传播中被标记，则恢复设备状态
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states, device_type=device_type)

            # 创建 device_autocast_ctx 上下文管理器，用于自动混合精度训练
            device_autocast_ctx = torch.amp.autocast(
                device_type=device_type, **device_autocast_kwargs
            ) if torch.amp.is_autocast_available(device_type) else contextlib.nullcontext()
            
            # 进入自动混合精度上下文管理器和 CPU 自动混合精度上下文管理器，以及 recompute_context 上下文管理器
            with device_autocast_ctx, torch.amp.autocast("cpu", **cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
                # 调用传入的函数 fn
                fn(*args, **kwargs)

    # 创建一个新的 _CheckpointFrame 对象，用于检查点操作
    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop,
        unpack_error_cb,
        metadata_fn
    )
    
    # 创建一个空的张量 dummy，用于保存输入
    dummy = torch.empty((0,), requires_grad=True)
    # 使用 _NoopSaveInputs.apply 方法保存输入
    new_frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    # 当环境的 grad_mode 是 False 时
    if new_frame.input_saver.grad_fn is None:
        # 返回一个空的生成器
        yield
        return

    # 进入 _checkpoint_hook 上下文管理器和 forward_context 上下文管理器
    with _checkpoint_hook(new_frame), forward_context:
        # 返回一个空的生成器
        yield
    # 标记前向传播已完成
    new_frame.forward_completed = True
    # 检查 device_module 对象中的 _initialized 属性是否为 True，且 preserve_rng_state 为 True，
    # 并且 had_device_in_fwd 为 False。这里的 type: ignore[possibly-undefined] 是类型提示，表示
    # 可能未定义的情况需要忽略类型检查。

    # 如果设备在前向传播之前未初始化，也就是 had_device_in_fwd 为 False，那么我们没有保存设备状态，
    # 因此抛出 RuntimeError 异常。

    raise RuntimeError(
        "PyTorch's device state was initialized in the forward pass "
        "of a Checkpoint, which is not allowed. Please open an issue "
        "if you need this feature."
    )

    # 返回语句，这里没有返回值，函数执行到这里将会结束。
```