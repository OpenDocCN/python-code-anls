# `.\pytorch\torch\autograd\function.py`

```py
# 使用 mypy: allow-untyped-defs 来允许未类型化的函数定义，用于类型检查工具mypy
import functools  # 导入 functools 模块，提供了创建和使用高阶函数的工具
import inspect  # 导入 inspect 模块，用于获取有关活动对象（如模块、类、方法、函数等）的信息
import itertools  # 导入 itertools 模块，提供了用于创建和操作迭代器的函数
import warnings  # 导入 warnings 模块，用于警告处理
from collections import OrderedDict  # 从 collections 模块导入 OrderedDict 类，有序字典的实现
from typing import Any, List, Optional, Tuple  # 导入 typing 模块中的类型，用于类型提示
from typing_extensions import deprecated  # 导入 deprecated 类型，用于标记过时的内容

import torch  # 导入 torch 模块，PyTorch 主模块
import torch._C as _C  # 导入 torch._C 模块，包含了 PyTorch 的C++扩展接口
import torch._functorch as _functorch  # 导入 torch._functorch 模块，包含了 functorch 扩展
import torch.utils.hooks as hooks  # 导入 torch.utils.hooks 模块，用于管理钩子功能
from torch._C import _functions  # 从 torch._C 中导入 _functions 模块，包含了底层函数接口
from torch._functorch.autograd_function import custom_function_call  # 从 torch._functorch.autograd_function 中导入 custom_function_call 函数

__all__ = [  # 定义 __all__ 列表，指定模块中的公共接口
    "FunctionCtx",  # 类 FunctionCtx 的公共接口
    "BackwardCFunction",  # 类 BackwardCFunction 的公共接口
    "FunctionMeta",  # 类 FunctionMeta 的公共接口
    "Function",  # 类 Function 的公共接口
    "once_differentiable",  # 函数 once_differentiable 的公共接口
    "InplaceFunction",  # 类 InplaceFunction 的公共接口
    "NestedIOFunction",  # 类 NestedIOFunction 的公共接口
]

# 用于为每个从 Function 继承的类提供唯一的 id
# 这在 FunctionMeta 类中的类定义过程中进行增加
AUTOGRAD_FUNCTION_COUNTER = itertools.count()


# 原名为：_ContextMethodMixin 的类
class FunctionCtx:
    # 将给定的张量保存起来，以备将来调用 :func:`~Function.backward` 时使用。

    # ``save_for_backward`` 应该最多调用一次，在 :func:`setup_context` 或 :func:`forward` 方法中，并且仅接受张量作为参数。
    
    # 所有打算在反向传播中使用的张量都应该使用 ``save_for_backward`` 进行保存（而不是直接在 ``ctx`` 上保存），以防止不正确的梯度和内存泄漏，并启用保存张量钩子的应用。参见 :class:`torch.autograd.graph.saved_tensors_hooks`。

    # 注意，如果保存了中间张量（即不是 :func:`forward` 的输入或输出），你的自定义 Function 可能不支持双向传播。
    # 不支持双向传播的自定义 Function 应该在其 :func:`backward` 方法上使用 ``@once_differentiable`` 进行修饰，以便执行双向传播时引发错误。
    # 如果你希望支持双向传播，可以在反向传播时基于输入重新计算中间结果，或者将中间结果作为自定义 Function 的输出返回。参见 `double backward tutorial <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_ 获取更多细节。

    # 在 :func:`backward` 方法中，保存的张量可以通过 :attr:`saved_tensors` 属性进行访问。
    # 在将它们返回给用户之前，会检查确保它们没有被任何改变其内容的原地操作使用。

    # 参数也可以是 ``None``。这是一个空操作。

    # 查看 :ref:`extending-autograd` 获取更多关于如何使用这个方法的细节。

    # 示例::
    #     >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
    #     >>> class Func(Function):
    #     >>>     @staticmethod
    #     >>>     def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
    #     >>>         w = x * z
    #     >>>         out = x * y + y * z + w * y
    #     >>>         ctx.save_for_backward(x, y, w, out)
    #     >>>         ctx.z = z  # z is not a tensor
    #     >>>         return out
    #     >>>
    #     >>>     @staticmethod
    #     >>>     @once_differentiable
    #     >>>     def backward(ctx, grad_out):
    #     >>>         x, y, w, out = ctx.saved_tensors
    #     >>>         z = ctx.z
    #     >>>         gx = grad_out * (y + y * z)
    #     >>>         gy = grad_out * (x + z + w)
    #     >>>         gz = None
    #     >>>         return gx, gy, gz
    #     >>>
    #     >>> a = torch.tensor(1., requires_grad=True, dtype=torch.double)
    #     >>> b = torch.tensor(2., requires_grad=True, dtype=torch.double)
    #     >>> c = 4
    #     >>> d = Func.apply(a, b, c)

    def save_for_backward(self, *tensors: torch.Tensor):
        self.to_save = tensors
    # 定义一个实例方法 `save_for_forward`，用于保存传入的张量，以备后续调用 `jvp` 方法时使用。
    r"""Save given tensors for a future call to :func:`~Function.jvp`.

    ``save_for_forward`` should be called at most once, in either the
    :func:`setup_context` or :func:`forward` methods, and all arguments
    should be tensors.

    In :func:`jvp`, saved objects can be accessed through the :attr:`saved_tensors`
    attribute.

    Arguments can also be ``None``. This is a no-op.

    See :ref:`extending-autograd` for more details on how to use this method.

    Example::
        >>> # xdoctest: +SKIP
        >>> class Func(torch.autograd.Function):
        >>>     @staticmethod
        >>>     def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
        >>>         ctx.save_for_backward(x, y)
        >>>         ctx.save_for_forward(x, y)
        >>>         ctx.z = z
        >>>         return x * y * z
        >>>
        >>>     @staticmethod
        >>>     def jvp(ctx, x_t, y_t, _):
        >>>         x, y = ctx.saved_tensors
        >>>         z = ctx.z
        >>>         return z * (y * x_t + x * y_t)
        >>>
        >>>     @staticmethod
        >>>     def vjp(ctx, grad_out):
        >>>         x, y = ctx.saved_tensors
        >>>         z = ctx.z
        >>>         return z * grad_out * y, z * grad_out * x, None
        >>>
        >>>     a = torch.tensor(1., requires_grad=True, dtype=torch.double)
        >>>     t = torch.tensor(1., dtype=torch.double)
        >>>     b = torch.tensor(2., requires_grad=True, dtype=torch.double)
        >>>     c = 4
        >>>
        >>>     with fwAD.dual_level():
        >>>         a_dual = fwAD.make_dual(a, t)
        >>>         d = Func.apply(a_dual, b, c)

    """
    # 对传入的每个张量进行类型检查，确保都是 `torch.Tensor` 类型或者为 `None`
    for tensor in tensors:
        assert isinstance(tensor, torch.Tensor) or tensor is None, (
            "save_for_forward expects all arguments to be tensors; you should "
            "save non-tensors as attributes on ctx."
        )

    # 将传入的张量保存到当前对象的 `saved_for_forward` 属性中
    self.saved_for_forward = tensors
    def mark_dirty(self, *args: torch.Tensor):
        r"""Mark given tensors as modified in an in-place operation.

        This method is used to indicate that tensors have been modified in-place 
        during the execution of the model's forward pass. It should be called 
        either in the `setup_context` or `forward` methods. All tensors that have 
        been modified in-place should be passed as arguments to this method.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class Inplace(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         x_npy = x.numpy() # x_npy shares storage with x
            >>>         x_npy += 1
            >>>         ctx.mark_dirty(x)
            >>>         return x
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, grad_output):
            >>>         return grad_output
            >>>
            >>> a = torch.tensor(1., requires_grad=True, dtype=torch.double).clone()
            >>> b = a * a
            >>> Inplace.apply(a)  # This would lead to wrong gradients!
            >>>                   # but the engine would not know unless we mark_dirty
            >>> # xdoctest: +SKIP
            >>> b.backward() # RuntimeError: one of the variables needed for gradient
            >>>              # computation has been modified by an inplace operation

        Args:
            *args (torch.Tensor): Tensors that have been modified in-place.

        """
        self.dirty_tensors = args

    @deprecated(
        "`mark_shared_storage` is deprecated. "
        "Tensors with shared storages are automatically tracked. "
        "Note that calls to `set_()` are not tracked",
        category=FutureWarning,
    )
    def mark_shared_storage(self, *pairs):
        pass
    def mark_non_differentiable(self, *args: torch.Tensor):
        r"""Mark outputs as non-differentiable.

        This should be called at most once, in either the :func:`setup_context`
        or :func:`forward` methods, and all arguments should be tensor outputs.

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.

        This is used e.g. for indices returned from a sort. See example::
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         sorted, idx = x.sort()
            >>>         ctx.mark_non_differentiable(idx)
            >>>         ctx.save_for_backward(x, idx)
            >>>         return sorted, idx
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, g1, g2):  # still need to accept g2
            >>>         x, idx = ctx.saved_tensors
            >>>         grad_input = torch.zeros_like(x)
            >>>         grad_input.index_add_(0, idx, g1)
            >>>         return grad_input

        """
        self.non_differentiable = args


注释：


# 将输出标记为不可微分

# 将此方法最多调用一次，通常在 setup_context 或 forward 方法中调用，所有参数应为张量输出。

# 标记这些输出为不需要梯度，增加反向计算的效率。在 Function.backward 方法中，仍然需要接受每个输出的梯度，
# 但这些梯度始终是一个与相应输出形状相同的零张量。

# 例如，用于排序返回的索引。参见示例::
#     >>> class Func(Function):
#     >>>     @staticmethod
#     >>>     def forward(ctx, x):
#     >>>         sorted, idx = x.sort()
#     >>>         ctx.mark_non_differentiable(idx)
#     >>>         ctx.save_for_backward(x, idx)
#     >>>         return sorted, idx
#     >>>
#     >>>     @staticmethod
#     >>>     @once_differentiable
#     >>>     def backward(ctx, g1, g2):  # 仍然需要接受 g2
#     >>>         x, idx = ctx.saved_tensors
#     >>>         grad_input = torch.zeros_like(x)
#     >>>         grad_input.index_add_(0, idx, g1)
#     >>>         return grad_input
    # 设置是否将梯度张量实例化。默认为 True。
    # 这应该只能从 setup_context 或 forward 方法中调用。

    # 如果为 True，则在调用 backward 和 jvp 方法之前，未定义的梯度张量将被扩展为全零张量。

    # 示例：
    # >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
    # >>> class SimpleFunc(Function):
    # >>>     @staticmethod
    # >>>     def forward(ctx, x):
    # >>>         return x.clone(), x.clone()
    # >>>
    # >>>     @staticmethod
    # >>>     @once_differentiable
    # >>>     def backward(ctx, g1, g2):
    # >>>         return g1 + g2  # 不需要检查 None
    # >>>
    # >>> # 我们修改 SimpleFunc 来处理未实例化梯度输出
    # >>> class Func(Function):
    # >>>     @staticmethod
    # >>>     def forward(ctx, x):
    # >>>         ctx.set_materialize_grads(False)
    # >>>         ctx.save_for_backward(x)
    # >>>         return x.clone(), x.clone()
    # >>>
    # >>>     @staticmethod
    # >>>     @once_differentiable
    # >>>     def backward(ctx, g1, g2):
    # >>>         x, = ctx.saved_tensors
    # >>>         grad_input = torch.zeros_like(x)
    # >>>         if g1 is not None:  # 现在必须检查 None
    # >>>             grad_input += g1
    # >>>         if g2 is not None:
    # >>>             grad_input += g2
    # >>>         return grad_input
    # >>>
    # >>> a = torch.tensor(1., requires_grad=True)
    # >>> b, _ = Func.apply(a)  # 使得 g2 变为未定义

    # 设置是否实例化梯度张量的属性值
    self.materialize_grads = value
# DO NOT USE: This is only defined to be able to load old serialized models
# 定义一个上下文方法混合类，用于加载旧的序列化模型
_ContextMethodMixin = FunctionCtx


class _HookMixin:
    @staticmethod
    def _register_hook(backward_hooks, hook):
        # 如果 backward_hooks 为 None，则初始化为有序字典
        if backward_hooks is None:
            backward_hooks = OrderedDict()
        # 创建一个可移除的句柄，并将 hook 添加到 backward_hooks 中
        handle = hooks.RemovableHandle(backward_hooks)
        backward_hooks[handle.id] = hook
        return backward_hooks, handle


class BackwardCFunction(_C._FunctionBase, FunctionCtx, _HookMixin):
    r"""
    This class is used for internal autograd work. Do not use.
    """
    
    def apply(self, *args):
        r"""
        Apply method used when executing this Node during the backward
        """
        # _forward_cls 由派生类定义
        # 用户应该只定义 backward 或 vjp 中的一个，而不是同时定义两者
        backward_fn = self._forward_cls.backward  # type: ignore[attr-defined]
        vjp_fn = self._forward_cls.vjp  # type: ignore[attr-defined]
        if backward_fn is not Function.backward and vjp_fn is not Function.vjp:
            raise RuntimeError(
                "Implementing both 'backward' and 'vjp' for a custom "
                "Function is not allowed. You should only implement one "
                "of them."
            )
        # 根据条件选择要执行的用户定义函数（backward 或 vjp）
        user_fn = vjp_fn if vjp_fn is not Function.vjp else backward_fn
        return user_fn(self, *args)

    def apply_jvp(self, *args):
        r"""
        Apply method used when executing forward mode AD during the forward
        """
        # _forward_cls 由派生类定义，执行正向自动求导时应用
        return self._forward_cls.jvp(self, *args)  # type: ignore[attr-defined]

    def _compiled_autograd_key(self):
        # 返回由 _forward_cls 定义的编译自动求导关键字
        return self._forward_cls._compiled_autograd_key(self)  # type: ignore[attr-defined]


class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """

    def __init__(cls, name, bases, attrs):
        # 创建一个反向函数类 backward_fn，其基类为 BackwardCFunction，_forward_cls 为当前类 cls
        backward_fn = type(
            name + "Backward", (BackwardCFunction,), {"_forward_cls": cls}
        )
        # 为反向函数类设置自动求导函数 ID
        backward_fn._autograd_function_id = next(AUTOGRAD_FUNCTION_COUNTER)  # type: ignore[attr-defined]
        # 设置是否应提升编译自动求导标志位
        backward_fn._compiled_autograd_should_lift = attrs.get(  # type: ignore[attr-defined]
            "_compiled_autograd_should_lift", True
        )
        # 将反向函数类设置为当前类的 _backward_cls 属性
        cls._backward_cls = backward_fn

        super().__init__(name, bases, attrs)


class _SingleLevelFunction(
    _C._FunctionBase, FunctionCtx, _HookMixin, metaclass=FunctionMeta
):
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        r"""Define the forward of the custom autograd Function.

        This function is to be overridden by all subclasses.
        There are two ways to define forward:

        Usage 1 (Combined forward and ctx)::

            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass

        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details

        Usage 2 (Separate forward and ctx)::

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass

            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass

        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details

        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.
        """
        raise NotImplementedError(
            "You must implement the forward function for custom autograd.Function."
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
        r"""There are two ways to define the forward pass of an autograd.Function.

        Either:

        1. Override forward with the signature ``forward(ctx, *args, **kwargs)``.
           ``setup_context`` is not overridden. Setting up the ctx for backward
           happens inside the ``forward``.
        2. Override forward with the signature ``forward(*args, **kwargs)`` and
           override ``setup_context``. Setting up the ctx for backward happens
           inside ``setup_context`` (as opposed to inside the ``forward``)

        See :meth:`torch.autograd.Function.forward` and :ref:`extending-autograd` for more details.
        """
        raise NotImplementedError("setup_context is not implemented.")
    # 定义用于反向模式自动微分的操作的微分公式
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        r"""Define a formula for differentiating the operation with backward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.
        """
        raise NotImplementedError(
            "You must implement either the backward or vjp method for "
            "your custom autograd.Function to use it with backward "
            "mode AD."
        )

    # vjp and backward are alias of each other
    # vjp 和 backward 是彼此的别名
    vjp = backward

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        r"""Define a formula for differentiating the operation with forward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many inputs as the :func:`forward` got (None will be passed in
        for non tensor inputs of the forward function),
        and it should return as many tensors as there were outputs to
        :func:`forward`. Each argument is the gradient w.r.t the given input,
        and each returned value should be the gradient w.r.t. the
        corresponding output. If an output is not a Tensor or the function is not
        differentiable with respect to that output, you can just pass None as a
        gradient for that input.

        You can use the :attr:`ctx` object to pass any value from the forward to this
        functions.
        """
        raise NotImplementedError(
            "You must implement the jvp function for custom "
            "autograd.Function to use it with forward mode AD."
        )
    """
    Base class to create custom `autograd.Function`.

    To create a custom `autograd.Function`, subclass this class and implement
    the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
    op in the forward pass, call the class method ``apply``. Do not call
    :meth:`forward` directly.

    To ensure correctness and best performance, make sure you are calling the
    correct methods on ``ctx`` and validating your backward function using
    :func:`torch.autograd.gradcheck`.

    See :ref:`extending-autograd` for more details on how to use this class.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> class Exp(Function):
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> # Use it by calling the apply method:
        >>> # xdoctest: +SKIP
        >>> output = Exp.apply(input)
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            f"{self.__class__} should not be instantiated. Methods on autograd functions"
            "are all static, so you should invoke them on the class itself. "
            "Instantiating an autograd function will raise an "
            "error in a future version of PyTorch.",
            DeprecationWarning,
            stacklevel=2,
        )
        # 发出警告，不应直接实例化此类，因为其方法均为静态方法

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Legacy autograd function with non-static forward method is deprecated. "
            "Please use new-style autograd function with static forward method. "
            "(Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)"
        )
        # 抛出运行时错误，不支持使用非静态前向方法的旧式自动求导函数

    """
    Bool that specifies if PyTorch should attempt to autogenerate
    :func:`torch.vmap` support for this autograd.Function. You may set this to
    True only if this autograd.Function's forward, backward, and jvp (if they
    exist) are written using PyTorch operations; otherwise, please override
    :meth:`torch.autograd.Function.vmap` to add support for :func:`torch.vmap`.

    Please see :ref:`func-autograd-function` for more details.
    """
    generate_vmap_rule = False
    # 控制是否尝试为此 autograd.Function 自动生成 torch.vmap 支持的布尔值。仅当该 autograd.Function 的 forward、backward 和 jvp 方法均使用 PyTorch 操作编写时，才应将其设置为 True；否则，请覆盖 torch.autograd.Function.vmap 方法以添加对 torch.vmap 的支持。
    def vmap(info, in_dims, *args):
        r"""Define the behavior for this autograd.Function underneath :func:`torch.vmap`.

        For a :func:`torch.autograd.Function` to support
        :func:`torch.vmap`, you must either override this static method, or set
        ``generate_vmap_rule`` to ``True`` (you may not do both).

        If you choose to override this staticmethod: it must accept

        - an ``info`` object as the first argument. ``info.batch_size``
          specifies the size of the dimension being vmapped over,
          while ``info.randomness`` is the randomness option passed to
          :func:`torch.vmap`.
        - an ``in_dims`` tuple as the second argument.
          For each arg in ``args``, ``in_dims`` has a corresponding
          ``Optional[int]``. It is ``None`` if the arg is not a Tensor or if
          the arg is not being vmapped over, otherwise, it is an integer
          specifying what dimension of the Tensor is being vmapped over.
        - ``*args``, which is the same as the args to :meth:`~Function.forward`.

        The return of the vmap staticmethod is a tuple of ``(output, out_dims)``.
        Similar to ``in_dims``, ``out_dims`` should be of the same structure as
        ``output`` and contain one ``out_dim`` per output that specifies if the
        output has the vmapped dimension and what index it is in.

        Please see :ref:`func-autograd-function` for more details.
        """
        raise NotImplementedError(
            "To use autograd.Function with vmap, you must either override the "
            "vmap staticmethod or set generate_vmap_rule=True."
        )

    @classmethod
    def apply(cls, *args, **kwargs):
        def bind_default_args(func, *args, **kwargs):
            signature = inspect.signature(func)  # 获取函数的签名
            bound_args = signature.bind(*args, **kwargs)  # 绑定函数参数
            bound_args.apply_defaults()  # 应用默认参数值

            return bound_args.args  # 返回绑定后的参数列表

        is_setup_ctx_defined = _is_setup_context_defined(cls.setup_context)  # 检查是否定义了 setup_context
        if is_setup_ctx_defined:
            args = bind_default_args(cls.forward, *args, **kwargs)  # 绑定默认参数到 forward 方法的参数中

        if not torch._C._are_functorch_transforms_active():
            # See NOTE: [functorch vjp and autograd interaction]
            args = _functorch.utils.unwrap_dead_wrappers(args)  # 解封装死包装器
            return super().apply(*args, **kwargs)  # 调用父类的 apply 方法（即 torch.autograd.Function 中的 apply 方法）；类型忽略警告

        if not is_setup_ctx_defined:
            raise RuntimeError(
                "In order to use an autograd.Function with functorch transforms "
                "(vmap, grad, jvp, jacrev, ...), it must override the setup_context "
                "staticmethod. For more details, please see "
                "https://pytorch.org/docs/main/notes/extending.func.html"
            )

        return custom_function_call(cls, *args, **kwargs)  # 调用自定义函数调用方法

    @staticmethod
    def _compiled_autograd_key(ctx):
        return (ctx._autograd_function_id,)  # 返回包含 autograd_function_id 的元组作为静态方法的输出
# 判断给定函数是否定义了设置上下文，返回布尔值
def _is_setup_context_defined(fn):
    return fn != _SingleLevelFunction.setup_context


# 标记函数为仅可微分一次的装饰器
def once_differentiable(fn):
    # 包装器函数，保留原始函数的元数据
    @functools.wraps(fn)
    def wrapper(ctx, *args):
        # 使用 torch.no_grad() 上下文，禁止梯度计算
        with torch.no_grad():
            # 调用原始函数，获取输出结果
            outputs = fn(ctx, *args)

        # 如果未启用梯度计算，则直接返回输出结果
        if not torch.is_grad_enabled():
            return outputs

        # 检查输入参数中是否有 requires_grad=True 的张量
        requires_grad = any(
            isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args
        )
        if not requires_grad:
            return outputs

        # 如果输出结果不是元组，则转换为元组形式
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # 创建延迟错误函数对象，用于标记尝试对已标记为 @once_differentiable 的函数进行二次微分的错误
        err_fn = _functions.DelayedError(
            b"trying to differentiate twice a function that was marked "
            b"with @once_differentiable",
            len(outputs),
        )

        # 创建每个需要梯度的输出结果的别名。至少需要一个输入参数依赖于梯度，以便输出结果具有 grad_fn。
        def fake_requires_grad(var):
            if var is not None:
                var = var.detach()
                var.requires_grad = True
            return var

        # 调用错误函数，将每个需要梯度的输出结果作为参数传入
        return err_fn(*[fake_requires_grad(v) for v in outputs])

    return wrapper


# InplaceFunction 类，继承自 torch.autograd.Function
class InplaceFunction(Function):
    r"""
    仅用于向后兼容的目的。对于任何新用例，请使用 :class:`Function`。
    """

    # 初始化方法，接受一个 inplace 参数用于指示是否就地操作
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace


# _nested_map 函数，用于处理映射中的嵌套情况，条件性地应用函数 fn 到映射的元素上
def _nested_map(condition, fn, condition_msg=None):
    # 定义一个内部函数 _map，用于根据输入对象 obj 进行映射操作
    def _map(obj):
        # 如果条件函数 condition 返回 True，则对 obj 应用映射函数 fn
        if condition(obj):
            return fn(obj)
        # 如果 obj 是 None，则直接返回 None
        elif obj is None:
            return None
        # 如果 obj 是列表或元组类型
        elif isinstance(obj, (list, tuple)):
            # 使用生成器表达式对列表或元组中的每个元素应用 _map 函数
            mapped = (_map(x) for x in obj)
            # 如果 obj 是 namedtuple，根据其类型重新创建对象并返回
            if hasattr(obj, "_fields"):
                # obj 是 namedtuple
                return type(obj)(*mapped)
            # 否则根据原始类型重新创建对象并返回
            return type(obj)(mapped)
        # 如果 obj 是字典类型
        elif isinstance(obj, dict):
            # 对字典中的每个值应用 _map 函数并返回新的字典
            return {x: _map(obj[x]) for x in obj}
        # 如果 obj 不属于以上任何一种类型，则抛出 ValueError 异常
        else:
            raise ValueError(
                "Auto nesting doesn't know how to process "
                "an input object of type "
                + torch.typename(obj)
                + (
                    ". Accepted types: " + condition_msg + ", or lists/tuples of them"
                    if condition_msg
                    else ""
                )
            )

    # 返回定义好的映射函数 _map
    return _map
def _jit_unwrap_structured(obj):
    # 如果对象具有 "_jit_unwrap" 方法，则调用该方法进行对象解封
    if hasattr(obj, "_jit_unwrap"):
        return obj._jit_unwrap()
    # 否则直接返回对象本身
    return obj


def _iter_filter(condition, allow_unknown=False, condition_msg=None, conversion=None):
    # 返回一个内部函数 _iter，该函数根据条件迭代处理输入对象
    def _iter(obj):
        # 如果提供了转换函数，先对对象进行转换
        if conversion is not None:
            obj = conversion(obj)
        # 如果对象满足条件，生成该对象
        if condition(obj):
            yield obj
        # 如果对象为 None，直接返回
        elif obj is None:
            return
        # 如果对象是列表或元组，递归迭代处理每个元素
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                yield from _iter(o)
        # 如果对象是字典，迭代处理字典的每个值
        elif isinstance(obj, dict):
            # 我们只接受原始键类型，因此不需要检查它们
            for o in obj.values():
                yield from _iter(o)
        # 如果允许未知类型，生成该对象
        elif allow_unknown:
            yield obj
        # 否则抛出值错误异常，说明无法处理的输入对象类型
        else:
            raise ValueError(
                "Auto nesting doesn't know how to process "
                "an input object of type "
                + torch.typename(obj)
                + (
                    ". Accepted types: " + condition_msg + ", or lists/tuples of them"
                    if condition_msg
                    else ""
                )
            )

    return _iter


def _unflatten(input, proto):
    # 将输入的列表或元组按照 proto 指定的结构展开为嵌套的列表/元组结构
    def unflatten_helper(input, proto):
        res: List[Optional[torch.Tensor]] = []
        # 如果 proto 具有 "_jit_wrap" 方法，则调用该方法进行对象包装
        if hasattr(proto, "_jit_wrap"):
            return proto._jit_wrap(input)
        # 如果 proto 不是列表或元组，则返回输入的第一个元素和剩余部分
        if not isinstance(proto, (list, tuple)):
            return input[0], input[1:]
        # 否则对 proto 中的每个元素进行递归处理
        for e in proto:
            if e is None:
                res.append(e)
            else:
                res_e, input = unflatten_helper(input, e)
                res.append(res_e)
        return type(proto)(res), input

    return unflatten_helper(input, proto)[0]


_iter_jit_values = _iter_filter(
    lambda o: o is None or isinstance(o, torch._C.Value),
    condition_msg="jit's Values or None",
)
_iter_tensors = _iter_filter(
    lambda x: isinstance(x, torch.Tensor),
    condition_msg="Tensors",
    conversion=_jit_unwrap_structured,
)
_iter_tensors_permissive = _iter_filter(
    lambda x: isinstance(x, torch.Tensor),
    allow_unknown=True,
    condition_msg="Tensors (permissive)",
)
_iter_None_tensors = _iter_filter(
    lambda o: o is None or isinstance(o, torch.Tensor), condition_msg="Tensors or None"
)
_map_tensor_data = _nested_map(
    lambda x: isinstance(x, torch.Tensor), lambda o: o.data, condition_msg="Tensors"
)


class NestedIOFunction(Function):
    r"""
    This class is here only for backward compatibility reasons.
    Use :class:`Function` instead of this for any new use case.
    """
    # 'type: ignore' 语句在此处是必需的，因为这些函数在超类 (Function) 中声明为 '@staticmethod'，
    # 但在此处作为实例方法声明，这会导致 mypy 报告不兼容。
    # 将输入参数存储在对象的 _nested_input 属性中
    def _do_forward(self, *input):
        self._nested_input = input
        # 将输入参数展平为元组
        flat_input = tuple(_iter_tensors(input))
        # 调用父类的 _do_forward 方法进行前向传播，并获取展平后的输出
        flat_output = super()._do_forward(*flat_input)  # type: ignore[misc]
        # 获取对象的 _nested_output 属性，用于反向传播
        nested_output = self._nested_output
        # 使用 _unflatten 函数将展平的输出重新构造成嵌套结构
        nested_tensors = _unflatten(flat_output, self._nested_output)
        # 返回重构后的嵌套结构张量
        return nested_tensors

    # 执行反向传播操作
    def _do_backward(self, gradients, retain_variables):
        # 存储是否保留中间变量的信息
        self.retain_variables = retain_variables
        # 调用父类的 _do_backward 方法执行反向传播，并获取结果
        result = super()._do_backward(gradients, retain_variables)  # type: ignore[misc]
        # 如果不保留中间变量，则删除相关属性
        if not retain_variables:
            del self._nested_output
            del self._to_save_nested
        # 返回反向传播的结果
        return result

    # 执行反向传播的扩展方法
    def backward(self, *gradients: Any) -> Any:  # type: ignore[override]
        """
        Shared backward utility.
        """
        # 使用 _unflatten 函数将输入梯度重新构造成嵌套结构
        nested_gradients = _unflatten(gradients, self._nested_output)
        # 调用 backward_extended 方法执行反向传播的扩展操作，并获取结果
        result = self.backward_extended(*nested_gradients)  # type: ignore[func-returns-value]
        # 返回结果中的 None 张量
        return tuple(_iter_None_tensors(result))

    # 将对象的 __call__ 方法指向 _do_forward 方法，使对象可被调用
    __call__ = _do_forward

    # 执行前向传播操作
    def forward(self, *args: Any) -> Any:  # type: ignore[override]
        """
        Shared forward utility.
        """
        # 使用 _map_tensor_data 函数映射嵌套输入张量的数据
        nested_tensors = _map_tensor_data(self._nested_input)
        # 调用 forward_extended 方法执行前向传播的扩展操作，并获取结果
        result = self.forward_extended(*nested_tensors)  # type: ignore[func-returns-value]
        # 删除对象的 _nested_input 属性
        del self._nested_input
        # 将前向传播的结果存储在对象的 _nested_output 属性中
        self._nested_output = result
        # 返回结果中的张量
        return tuple(_iter_tensors(result))

    # 将输入参数保存为待反向传播时需要保存的张量
    def save_for_backward(self, *args: Any) -> None:
        """
        See :meth:`Function.save_for_backward`.
        """
        # 使用 _iter_tensors 函数迭代输入参数，并存储在对象的 to_save 属性中
        self.to_save = tuple(_iter_tensors(args))
        self._to_save_nested = args

    # 返回对象保存的张量
    @property
    def saved_tensors(self):
        """
        See :meth:`Function.saved_tensors`.
        """
        # 调用父类的 saved_tensors 方法获取展平后的张量
        flat_tensors = super().saved_tensors  # type: ignore[misc]
        # 使用 _unflatten 函数将展平的张量重新构造成嵌套结构
        return _unflatten(flat_tensors, self._to_save_nested)

    # 标记指定张量为脏数据
    def mark_dirty(self, *args: Any, **kwargs: Any) -> None:
        """
        See :meth:`Function.mark_dirty`.
        """
        # 使用 _iter_tensors 函数迭代输入参数，并存储在对象的 dirty_tensors 属性中
        self.dirty_tensors = tuple(_iter_tensors((args, kwargs)))

    # 标记指定张量为不可微分
    def mark_non_differentiable(self, *args: Any, **kwargs: Any) -> None:
        """
        See :meth:`Function.mark_non_differentiable`.
        """
        # 使用 _iter_tensors 函数迭代输入参数，并存储在对象的 non_differentiable 属性中
        self.non_differentiable = tuple(_iter_tensors((args, kwargs)))

    # 执行扩展的前向传播操作（用户自定义）
    def forward_extended(self, *input: Any) -> None:
        """
        User defined forward.
        """
        # 抛出未实现异常，提示用户实现具体的前向传播逻辑
        raise NotImplementedError

    # 执行扩展的反向传播操作（用户自定义）
    def backward_extended(self, *grad_output: Any) -> None:
        """
        User defined backward.
        """
        # 抛出未实现异常，提示用户实现具体的反向传播逻辑
        raise NotImplementedError
```