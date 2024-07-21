# `.\pytorch\torch\autograd\graph.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和库
import abc  # 引入抽象基类模块
import collections  # 引入集合模块
import contextlib  # 引入上下文管理模块
import functools  # 引入函数工具模块
import logging  # 引入日志记录模块
import threading  # 引入线程模块
import weakref  # 引入弱引用模块
from collections import defaultdict, namedtuple  # 从集合模块中引入 defaultdict 和 namedtuple
from typing import (  # 引入类型提示相关的模块
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch  # 引入 PyTorch 库
from torch.autograd.variable import Variable  # 从 PyTorch 自动求导变量模块中引入 Variable 类
from torch.utils._python_dispatch import TorchDispatchMode  # 从 PyTorch 工具的 Python 调度模块中引入 TorchDispatchMode
from torch.utils.hooks import RemovableHandle  # 从 PyTorch 工具的钩子模块中引入 RemovableHandle 类

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 导出的变量名列表
__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "register_multi_grad_hook",
    "allow_mutation_on_saved_tensors",
    "Node",
    "GradientEdge",
    "get_gradient_edge",
    "increment_version",
]


class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        r"""Return the name.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> print(b.grad_fn.name())
            CloneBackward0
        """
        ...

    @property
    @abc.abstractmethod
    def next_functions(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        ...

    @abc.abstractmethod
    def metadata(self) -> dict:
        r"""Return the metadata."""
        ...

    @abc.abstractmethod
    def _register_hook_dict(self, tensor: torch.Tensor) -> None:
        ...
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_inputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove() # Removes the hook
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        ...

    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward pre-hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_prehook(lambda gI: (gI[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove()
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        ...
    # 定义一个特殊方法 __subclasshook__，用于类的子类检查
    def __subclasshook__(cls, C):
        # 如果当前类是 Node 类
        if cls is Node:
            # 如果传入的类 C 不为空，并且是 torch._C._functions 模块下的某个类，
            # 或者是 torch.autograd.function.BackwardCFunction 的子类
            if (
                C is not None and C is getattr(torch._C._functions, C.__name__, None)
            ) or issubclass(C, torch.autograd.function.BackwardCFunction):
                # 返回 True，表示 C 是 Node 的子类或符合条件
                return True
        # 返回 NotImplemented，表示未匹配到子类关系
        return NotImplemented
# 定义一个函数，用于获取给定张量的梯度函数或梯度累积器的梯度边缘
def _get_grad_fn_or_grad_acc(t):
    # 如果张量需要梯度且没有梯度函数，则返回其视图作为的梯度函数链中的下一个函数
    if t.requires_grad and t.grad_fn is None:
        return t.view_as(t).grad_fn.next_functions[0][0]
    else:
        return t.grad_fn


# 使用 namedtuple 定义一个名为 GradientEdge 的对象，表示自动求导图中的梯度边缘
GradientEdge = namedtuple("GradientEdge", ("node output_nr"))
# 设置 GradientEdge 的文档字符串，描述其在自动求导图中的作用
GradientEdge.__doc__ = """\
Object representing a given gradient edge within the autograd graph.
To get the gradient edge where a given Tensor gradient will be computed,
you can do ``edge = autograd.graph.get_gradient_edge(tensor)``.
"""


def get_gradient_edge(tensor):
    """获取给定张量的梯度边缘，用于计算其梯度。

    具体来说，等价于调用 ``g = autograd.grad(loss, input)`` 和 ``g = autograd.grad(loss, get_gradient_edge(input))``。
    """
    # 如果张量不需要梯度，则抛出运行时错误
    if not tensor.requires_grad:
        raise RuntimeError(
            "It is not possible to get the gradient edge for a Tensor that does not require gradients"
        )
    # 调用内部函数 _get_grad_fn_or_grad_acc 获取张量的梯度函数或梯度累积器
    grad_fn = _get_grad_fn_or_grad_acc(tensor)

    # 注意：output_nr 默认为 0，适用于 AccumulateGrad 节点。
    return GradientEdge(grad_fn, tensor.output_nr)


def increment_version(tensor):
    """更新自动求导元数据，跟踪给定张量是否被原地修改过。

    这样做是为了在自动求导引擎中进行更精确的错误检查。
    PyTorch 函数和自定义 Function 内部调用 mark_dirty() 时会自动完成这一操作，
    因此只有在 PyTorch 不知道的情况下，如自定义内核通过张量数据指针原地修改张量数据时，
    才需要显式调用这个函数。

    注意，为单个原地操作多次递增版本计数器并不会导致问题。
    """
    torch._C._increment_version(tensor)


class saved_tensors_hooks:
    """上下文管理器，为保存的张量设置打包/解包钩子。

    使用这个上下文管理器定义操作的中间结果在保存前如何打包和在检索时如何解包。

    在这个上下文中，每当一个操作保存一个张量用于反向传播（包括使用
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` 保存的中间结果，
    以及由 PyTorch 定义的操作记录的结果），都会调用 ``pack_hook`` 函数。
    ``pack_hook`` 的输出将存储在计算图中，而不是原始张量。

    当需要访问保存的张量时（即执行 :func:`torch.Tensor.backward()` 或
    :func:`torch.autograd.grad()` 时），将调用 ``unpack_hook``。它以由 ``pack_hook`` 返回的
    *packed* 对象作为参数，并应返回一个内容与原始张量相同的张量。
    """
    """
    The hooks should have the following signatures:
    
        pack_hook(tensor: Tensor) -> Any
        
        unpack_hook(Any) -> Tensor
    
    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.
    
    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.
    
    Example::
    
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Packing tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Unpacking tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
    
    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.
    
    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.
    """

    # 初始化函数，接受两个回调函数作为参数
    def __init__(
        self,
        pack_hook: Callable[[torch.Tensor], Any],  # 接受一个输入为 torch.Tensor 类型参数的回调函数，返回任意类型
        unpack_hook: Callable[[Any], torch.Tensor],  # 接受一个任意类型参数，返回类型为 torch.Tensor 的回调函数
    ):
        self.pack_hook = pack_hook  # 将 pack_hook 参数赋值给对象的 pack_hook 属性
        self.unpack_hook = unpack_hook  # 将 unpack_hook 参数赋值给对象的 unpack_hook 属性

    # 进入上下文管理器时调用的方法
    def __enter__(self):
        # 调用 Torch 库中的私有函数，将 pack_hook 和 unpack_hook 注册为默认的张量保存钩子
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    # 离开上下文管理器时调用的方法
    def __exit__(self, *args: object):
        # 调用 Torch 库中的私有函数，弹出默认的张量保存钩子
        torch._C._autograd._pop_saved_tensors_default_hooks()
class save_on_cpu(saved_tensors_hooks):
    """Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.

    当前上下文管理器用于在前向传播期间保存的张量将会存储在 CPU 上，并在反向传播时重新获取。

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    在这个上下文管理器内执行操作时，在前向传播期间保存的中间结果将会被移动到 CPU，
    然后在需要进行反向传播时复制回原始设备。如果计算图已经在 CPU 上，则不会进行张量复制。

    Use this context-manager to trade compute for GPU memory usage (e.g.
    when your model doesn't fit in GPU memory during training).

    使用这个上下文管理器来在计算和 GPU 内存使用之间进行权衡（例如，在训练期间模型不适合放在 GPU 内存中时）。

    Args:
        pin_memory (bool): If ``True`` tensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.

        pin_memory（bool）：如果为“True”，则在打包期间张量将被保存到 CPU 固定内存，
                           并在解包时异步复制到 GPU。默认为“False”。
                           参见：:ref:`cuda-memory-pinning`。

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>>
        >>> def f(a, b, c):
        ...     prod_1 = a * b           # a and b are saved on GPU
        ...     with torch.autograd.graph.save_on_cpu():
        ...         prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
        ...     y = prod_2 * a           # prod_2 and a are saved on GPU
        ...     return y
        >>>
        >>> y = f(a, b, c)
        >>> del a, b, c  # for illustration only
        >>> # the content of a, b, and prod_2 are still alive on GPU
        >>> # the content of prod_1 and c only live on CPU
        >>> y.sum().backward()  # all CPU tensors are moved back to GPU, for backward
        >>> # all intermediary tensors are released (deleted) after the call to backward

    """

    def __init__(self, pin_memory=False, device_type="cuda"):
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


@contextlib.contextmanager
def disable_saved_tensors_hooks(error_message):
    """Context-manager that disables the saved tensors default hooks feature.

    用于禁用保存的张量默认钩子功能的上下文管理器。

    Useful for if you are creating a feature that does not work with saved
    tensors default hooks.

    适用于创建与保存的张量默认钩子功能不兼容的功能时使用。
    Args:
        error_message (str): 当保存的张量默认钩子被禁用时，抛出带有此错误消息的 RuntimeError。

    Example::

        >>> # xdoctest: +SKIP(failing)
        >>> message = "saved tensors default hooks are disabled"
        >>> with torch.autograd.graph.disable_saved_tensors_hooks(message):
        ...     # Raises RuntimeError: saved tensors default hooks are disabled
        ...     with torch.autograd.graph.save_on_cpu():
        ...         pass

    """
    # 尝试获取可能之前的错误消息
    try:
        maybe_prev_message = (
            torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        )
        # 禁用保存张量钩子，设置新的错误消息
        torch._C._autograd._saved_tensors_hooks_disable(error_message)
        # 执行 yield，允许执行嵌套的上下文
        yield
    finally:
        # See NOTE: [disabled_error_message invariant]
        # 如果之前的错误消息为 None，则重新启用保存张量钩子
        if maybe_prev_message is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            # 否则，重新禁用之前的错误消息
            torch._C._autograd._saved_tensors_hooks_disable(maybe_prev_message)
class _MultiHandle(RemovableHandle):
    handles: Tuple[RemovableHandle, ...]  # 定义一个元组类型的属性 handles，包含 RemovableHandle 实例

    def __init__(self, handles: Tuple[RemovableHandle, ...]):
        self.handles = handles  # 初始化 handles 属性为传入的 handles 参数元组

    def remove(self):
        for handle in self.handles:
            handle.remove()  # 调用每个 handle 对象的 remove() 方法来移除相应的 hook

    def __getstate__(self):
        return self.handles  # 返回当前对象的 handles 属性作为其状态信息

    def __setstate__(self, state):
        self.handles = state  # 设置对象的 handles 属性为传入的状态信息 state


def register_multi_grad_hook(
    tensors: Sequence[torch.Tensor],
    fn: Union[
        Callable[[Sequence[Optional[torch.Tensor]]], None],
        Callable[[torch.Tensor], None],
    ],
    *,
    mode: str = "all",
):
    r"""Register a multi-grad backward hook.

    There are two supported modes: ``"all"`` and ``"any"``.

    Under the ``"all"`` mode, the hook will be called after gradients with respect to every tensor in
    :attr:`tensors` have been computed. If a tensor is in :attr:`tensors` but
    is not part of the graph, or if a tensor is not needed to compute the gradients
    for any ``inputs`` specified for the current ``.backward()`` or ``.grad()`` call,
    this tensor will be ignored and the hook will not wait for its gradient to be
    computed.

    After every non-ignored tensor's gradient has been computed, :attr:`fn` will be
    called with those gradients. ``None`` will be passed for tensors that did not
    have their gradients computed.

    Under the ``"any"`` mode, the hook will be called after the first gradient
    with respect to a tensor in :attr:`tensors` has been computed. The hook
    will be called with that gradient as its argument.

    The hook should not modify its arguments.

    This function returns a handle with a method ``handle.remove()`` that removes the hook.

    .. note::
        See :ref:`backward-hooks-execution` for more information on how when this hook
        is executed, and how its execution is ordered relative to other hooks.

    Example::

        >>> import torch
        >>>
        >>> a = torch.rand(2, 3, requires_grad=True)
        >>> b = torch.rand(2, 3, requires_grad=True)
        >>> c = a * b
        >>> d = a * b
        >>>
        >>> def fn(grads):
        ...     print([g is not None for g in grads])
        ...
        >>> torch.autograd.graph.register_multi_grad_hook((a, b, c, d), fn)
        >>>
        >>> c.sum().backward(retain_graph=True)
        [True, True, True, False]
        >>> c.sum().backward(inputs=(a,), retain_graph=True)
        [True, False, True, False]
        >>>
    """
    supported_modes = ("all", "any")
    if mode not in supported_modes:
        raise ValueError(f"Expects mode to be one of {supported_modes} but got {mode}")  # 如果 mode 不在支持的模式列表中，则抛出 ValueError 异常
    # 如果模式为 "all"，执行以下逻辑
    if mode == "all":
        # count 是一个计数器字典，记录每个任务 ID 的调用次数
        count: Dict[int, int] = dict()
        # nb_calls 初始化为 None，用于记录需要执行的总次数
        nb_calls = None
        # buffer 是一个字典，用于存储每个任务 ID 对应的梯度列表
        buffer: Dict[int, List[Optional[torch.Tensor]]] = dict()

        # 获取每个张量的梯度函数或者梯度累加器的列表
        grad_fns = list(map(_get_grad_fn_or_grad_acc, tensors))
        # 计算张量的数量
        len_tensors = len(tensors)

        # 定义内部钩子函数，用于注册到每个张量的梯度计算中
        def get_inner_hook(idx):
            def inner_hook(grad: torch.Tensor):
                nonlocal count, nb_calls, buffer, fn
                # 获取当前图任务的 ID
                id = torch._C._current_graph_task_id()
                # 断言任务 ID 不为 -1，确保钩子函数在反向传播中被调用
                assert (
                    id != -1
                ), "expected this hook to be called inside a backward call"
                # 初始化当前任务 ID 的计数器和缓冲区
                count[id] = count.get(id, 0)
                buffer[id] = buffer.get(id, [None] * len_tensors)

                # 若当前任务 ID 的计数为 0，则计算实际的 nb_calls 和缓冲区内容
                if count[id] == 0:
                    nb_calls = sum(torch._C._will_engine_execute_node(g) for g in grad_fns)  # 计算需要执行的总次数

                # 将当前张量的梯度存入对应的缓冲区
                buffer[id][idx] = grad
                count[id] += 1

                # 如果当前任务 ID 的计数等于 nb_calls，执行给定的回调函数 fn
                if count[id] == nb_calls:
                    fn = cast(Callable[[Sequence[Optional[torch.Tensor]]], None], fn)
                    fn(buffer[id])
                    # 清理计数器和缓冲区
                    del count[id]
                    del buffer[id]

            return inner_hook

        # 为每个张量注册内部钩子函数，并返回相应的句柄
        handles: Tuple[RemovableHandle] = tuple(
            t.register_hook(get_inner_hook(i)) for i, t in enumerate(tensors)
        )
    # 如果模式为 "any"，执行以下逻辑
    elif mode == "any":
        # 将 fn 转换为 Callable[[torch.Tensor], None] 类型
        fn = cast(Callable[[torch.Tensor], None], fn)
        # 创建一个线程锁对象
        lock = threading.Lock()
        # ran_hook 是一个字典，记录每个任务 ID 是否已经执行过钩子函数的状态
        ran_hook: Dict[int, bool] = defaultdict(bool)

        # 定义装饰后的回调函数，用于注册到每个需要梯度的张量上
        @functools.wraps(fn)
        def wrapped_fn(grad: torch.Tensor):
            nonlocal ran_hook
            # 获取当前图任务的 ID
            id = torch._C._current_graph_task_id()
            # 断言任务 ID 不为 -1，确保钩子函数在反向传播中被调用
            assert id != -1, "expected this hook to be called inside a backward call"
            # 使用线程锁确保同一时刻只有一个任务 ID 的钩子函数在执行
            with lock:
                prev, ran_hook[id] = ran_hook[id], True
            # 如果之前已经执行过该任务 ID 的钩子函数，则直接返回
            if prev:
                return
            # 否则执行给定的回调函数 fn
            fn(grad)

        # 为每个需要梯度的张量注册装饰后的钩子函数，并返回相应的句柄
        handles = tuple(
            tensor.register_hook(wrapped_fn)
            for tensor in tensors
            if tensor.requires_grad
        )

    # 返回多句柄对象 _MultiHandle，用于管理所有注册的钩子函数句柄
    return _MultiHandle(handles)  # type: ignore[possibly-undefined]
# NOTE [Allow mutation on tensors saved for backward]
#
# 1. Tensor gets saved for backward
#    - remember the python object id and the version of the tensor
#    - remember aliasing information (data_ptr of base + version)
#    - save the original so we control its lifetime
# 2. Any time a tensor gets in-placed
#    - for each tensor aliased to it:
#      - check using its object id and version to see if it has been saved
#      - if it has been saved, clone it
#      - delete the reference to the original
# 3. during backward
#    - if the clone exists, the tensor must've been modified in-place
_allow_mutation_on_saved_tensors_enabled = False


def _get_tid(t) -> Tuple[int, int, int]:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        t,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        # If the tensor is a fake or functional tensor, set data_ptr to 0
        data_ptr = 0
    else:
        # Otherwise, retrieve the data pointer using t.data_ptr()
        data_ptr = t.data_ptr()
    # Return a tuple of object id, data pointer, and tensor version
    return (id(t), data_ptr, t._version)


def _get_sid(t) -> Tuple[int, int]:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        t,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        # If the tensor is a fake or functional tensor, set data_ptr to 0
        data_ptr = 0
    else:
        # Otherwise, retrieve the data pointer using t.data_ptr()
        data_ptr = t.data_ptr()
    # Return a tuple of data pointer and tensor version
    return (data_ptr, t._version)


class _Handle:
    pass


class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx):
        # Initialize the pack_hook function for saving tensors
        def pack_hook(t):
            # Obtain tensor id, data pointer, and version
            tid = _get_tid(t)
            sid = _get_sid(t)
            # Tensors saved for backward have an entry in _tid_to_weakhandle
            handle: Optional[_Handle] = None

            # Save aliasing information in ctx.sid_to_tid
            ctx.sid_to_tid[sid].add(tid)

            # Check if tensor is already saved; if not, create a handle
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = t
            else:
                # Store an additional strong reference to the existing handle
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        # Initialize the unpack_hook function for retrieving tensors during backward
        def unpack_hook(tup):
            handle = tup
            error_msg = (
                "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
                "in which the graph was originally recorded."
            )
            # Ensure that mutation on saved tensors is allowed
            assert _allow_mutation_on_saved_tensors_enabled, error_msg
            # Retrieve the tensor from ctx.cloned if it exists; otherwise from ctx.original
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                assert handle in ctx.original, error_msg
                res = ctx.original[handle]
            return res

        super().__init__(pack_hook, unpack_hook)


class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __init__(self, ctx):
        self.ctx = ctx
    # 定义一个特殊方法用于 Torch 模型的分发操作，根据函数、类型和参数进行分发
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 如果 kwargs 为空，则初始化为空字典
        kwargs = kwargs or {}

        # 遍历函数的参数列表
        for idx, arg in enumerate(func._schema.arguments):
            # 如果参数有别名信息并且是写操作
            if arg.alias_info is not None and arg.alias_info.is_write:
                # 如果参数是输出参数，则取 kwargs 中的 "out"，否则取 args 中对应索引的参数
                t = kwargs["out"] if arg.is_out else args[idx]
                # 获取 tensor 的类型 ID 和存储 ID
                tid = _get_tid(t)
                sid = _get_sid(t)
                # 获取当前上下文
                ctx = self.ctx
                # 如果存储 ID 在上下文的 sid_to_tid 中
                if sid in ctx.sid_to_tid:
                    # 遍历该 sid 对应的所有类型 ID
                    for tid in ctx.sid_to_tid[sid]:
                        # 如果该类型 ID 不在上下文的 tid_to_weakhandle 中
                        if tid not in ctx.tid_to_weakhandle:
                            # 如果 tid 在 sid_to_tid 中，则也应该在 tid_to_weakhandle 中，
                            # 但可能在反向传播清除该 tensor 后，再进行原地修改。例如：
                            #
                            # >>> a = torch.randn(2, 3, requires_grad=True).clone()
                            # >>> out = (a**2).sum()
                            # >>> out.backward()
                            # >>> a.sin_()
                            continue
                        # 获取该类型 ID 对应的弱引用句柄
                        handle = ctx.tid_to_weakhandle[tid]
                        # 如果句柄已经在克隆集合中
                        if handle in ctx.cloned:
                            # 表示已经克隆了相同的 tensor，跳过处理
                            continue
                        # 克隆原始 tensor 并将其添加到克隆集合中
                        ctx.cloned[handle] = ctx.original[handle].clone()
                        # 删除原始 tensor
                        del ctx.original[handle]

        # 调用函数并返回结果
        rs = func(*args, **kwargs)
        return rs
class _AllowMutationOnSavedContext:
    # 定义私有类 _AllowMutationOnSavedContext，用于管理允许在保存的上下文中进行变异的状态
    def __init__(self):
        # 初始化三个 WeakKeyDictionary 对象，用于存储对象引用和弱引用
        self.cloned: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.original: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.tid_to_weakhandle: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )
        # 初始化 defaultdict，用于存储特定格式的键值对
        self.sid_to_tid: Dict[Tuple[int, int], Set[Tuple[int, int, int]]] = defaultdict(
            set
        )

    # 清空所有存储在实例中的数据
    def clear(self):
        self.cloned.clear()
        self.original.clear()
        self.tid_to_weakhandle.clear()
        self.sid_to_tid.clear()


@contextlib.contextmanager
def allow_mutation_on_saved_tensors():
    """Context manager under which mutating tensors saved for backward is allowed.

    Under this context manager, tensors saved for backward are cloned on mutation,
    so the original version can still be used during backward. Normally, mutating a tensor
    saved for backward will result in an error raised when it's used during backward.

    To ensure the correct behavior, both the forward and backward should be run under
    the same context manager.

    returns:
        An _AllowMutationOnSavedContext object storing the state managed by this
        context manager. This object can be useful for debugging purposes. The state
        managed by the context manager is automatically cleared upon exiting.

    Example::

        >>> import torch
        >>> with torch.autograd.graph.allow_mutation_on_saved_tensors():
        ...     # forward
        ...     a = torch.ones(2, 3, requires_grad=True)
        ...     b = a.clone()
        ...     out = (b**2).sum()
        ...     b.sin_()
        ...     # backward
        ...     out.sum().backward()
        ...
        tensor([[0.8415, 0.8415, 0.8415],
                [0.8415, 0.8415, 0.8415]], grad_fn=<SinBackward0>)
    """
    global _allow_mutation_on_saved_tensors_enabled

    # 创建 _AllowMutationOnSavedContext 对象作为上下文管理器的状态对象
    ctx = _AllowMutationOnSavedContext()

    # 使用 _swap_with_cloned 和 _CloneArgBeforeMutateMode 上下文管理器，以便在变异前进行克隆
    with _swap_with_cloned(ctx), _CloneArgBeforeMutateMode(ctx):
        try:
            # 检查是否已经启用允许在保存的张量上进行变异的功能，如果已经启用则抛出 RuntimeError
            if _allow_mutation_on_saved_tensors_enabled:
                raise RuntimeError(
                    "allow_mutation_on_saved_tensors contexts cannot be nested"
                )
            # 启用允许在保存的张量上进行变异的功能
            _allow_mutation_on_saved_tensors_enabled = True
            # 返回上下文管理器的状态对象
            yield ctx
        finally:
            # 清理上下文管理器中的状态数据
            ctx.clear()
            # 禁用允许在保存的张量上进行变异的功能
            _allow_mutation_on_saved_tensors_enabled = False


def _register_logging_hooks_on_whole_graph(t_outputs: List[torch.Tensor]):
    # 对传入的张量列表中的每个张量应用 _get_grad_fn_or_grad_acc 函数，并将结果存储在 grad_fns 列表中
    grad_fns = list(map(_get_grad_fn_or_grad_acc, t_outputs))
    # 定义一个生成器函数，用于遍历给定的节点列表 roots 所构成的图
    def iter_graph(roots):
        # 如果 roots 列表为空，则直接返回，不进行任何操作
        if not roots:
            return
        # 初始化一个集合 seen，用于记录已经处理过的节点
        seen = set()
        # 初始化一个双端队列 q，用于存放待处理的节点
        q: Deque = collections.deque()
        # 遍历 roots 列表中的每个节点
        for node in roots:
            # 如果节点不为 None，则将其添加到 seen 集合和队列 q 中
            if node is not None:
                seen.add(node)
                q.append(node)

        # 开始处理队列中的节点
        while q:
            # 从队列左侧取出一个节点
            node = q.popleft()
            # 遍历当前节点的后续函数列表
            for fn, _idx in node.next_functions:
                # 如果函数已经在 seen 集合中或者为 None，则跳过当前循环
                if fn in seen or fn is None:
                    continue
                # 将新的函数添加到 seen 集合和队列 q 中
                seen.add(fn)
                q.append(fn)

            # 使用生成器语法，返回当前处理的节点
            yield node

    # 定义一个函数 fmt，用于格式化给定的对象 t
    def fmt(t):
        # 避免循环导入问题，从 torch.testing._internal.common_utils 中导入 dtype_abbrs
        from torch.testing._internal.common_utils import dtype_abbrs

        # 如果 t 为 None，则返回字符串 "None"
        if t is None:
            return "None"
        # 否则返回格式化后的字符串，包含数据类型的简称和形状信息
        return f"{dtype_abbrs[t.dtype]}[{', '.join(map(str, t.shape))}]"

    # 定义一个预处理函数 prehook，用于注册在自动求导过程中的钩子
    def prehook(grad_outputs):
        # 获取当前的自动求导节点
        node = torch._C._current_autograd_node()
        # 格式化梯度输出列表 grad_outputs
        grad_outputs_str = f"[{','.join(fmt(t) for t in grad_outputs)}]"
        # 构建日志字符串，记录当前节点和梯度输出信息
        log_str = f"Executing: {node} with grad_outputs: {grad_outputs_str}"
        # 使用日志模块记录调试信息
        log.debug(log_str)

    # 初始化一个空列表 handles，用于存放注册的钩子处理函数的句柄
    handles = []
    # 遍历通过 iter_graph 函数生成的节点迭代器，注册钩子函数 prehook
    for node in iter_graph(grad_fns):
        handles.append(node.register_prehook(prehook))

    # 定义一个函数 unregister_hooks，用于移除所有已注册的钩子函数
    def unregister_hooks():
        # 遍历 handles 列表中的每个句柄，逐个移除钩子函数
        for handle in handles:
            handle.remove()

    # 返回 unregister_hooks 函数，以便外部调用者可以使用该函数来移除注册的钩子
    return unregister_hooks
# 定义一个函数 `_engine_run_backward`，用于执行反向传播操作
def _engine_run_backward(t_outputs, *args, **kwargs):
    # 判断是否需要附加日志钩子，判断全局日志级别是否为 DEBUG
    attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    # 如果需要附加日志钩子
    if attach_logging_hooks:
        # 在整个计算图上注册日志钩子
        unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    try:
        # 调用 C++ 引擎执行反向传播，参数包括 `t_outputs` 以及其他位置参数和关键字参数
        return Variable._execution_engine.run_backward(
            t_outputs, *args, **kwargs
        )
    finally:
        # 如果需要附加日志钩子
        if attach_logging_hooks:
            # 取消注册日志钩子
            unregister_hooks()  # type: ignore[possibly-undefined]
```