# `.\pytorch\torch\_tensor.py`

```py
# 引入必要的模块和函数
# 允许未类型化的函数定义，用于类型检查工具 mypy
import copyreg  # 提供对象注册和反注册功能
import enum  # 支持枚举类型的定义和操作
import functools  # 提供高阶函数：部分应用函数、函数包装
import warnings  # 提供警告相关的功能

from collections import OrderedDict  # 提供有序字典的数据结构
from copy import deepcopy  # 提供深度复制功能
from numbers import Number  # 提供数字类型相关的抽象基类
from typing import Any, Dict, Optional, Tuple, Union  # 提供类型提示相关的功能

import torch  # PyTorch 主库
import torch._C as _C  # PyTorch C++ 扩展模块
from torch._namedtensor_internals import (  # 导入命名张量相关的内部函数
    check_serializing_named_tensor,  # 检查命名张量的序列化
    is_ellipsis,  # 判断是否为省略号
    resolve_ellipsis,  # 解析省略号
    single_ellipsis_index,  # 获取单个省略号的索引
    unzip_namedshape,  # 解压命名形状
    update_names,  # 更新名称
)
from torch.overrides import (  # 导入重载相关的函数和类
    get_default_nowrap_functions,  # 获取默认的不包装函数
    handle_torch_function,  # 处理 Torch 函数的重载
    has_torch_function,  # 检查是否存在 Torch 函数的重载（一元）
    has_torch_function_unary,  # 检查是否存在 Torch 函数的一元重载
    has_torch_function_variadic,  # 检查是否存在 Torch 函数的可变参数重载
)


def _handle_torch_function_and_wrap_type_error_to_not_implemented(f):
    assigned = functools.WRAPPER_ASSIGNMENTS

    @functools.wraps(f, assigned=assigned)
    def wrapped(*args, **kwargs):
        try:
            # 检查是否有 Torch 函数的重载，如果有则调用处理函数
            if has_torch_function(args):
                return handle_torch_function(wrapped, args, *args, **kwargs)
            # 否则直接调用原始函数
            return f(*args, **kwargs)
        except TypeError:
            # 处理类型错误异常，返回未实现（NotImplemented）
            return NotImplemented

    return wrapped


# 应避免使用此函数，仅保留用于加载旧序列化的 Tensor 子类
def _rebuild_from_type(func, type, args, dict):
    if type is Tensor:
        return func(*args)

    # 对 func(*args) 执行子类化，再设置 __dict__ 属性
    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret


def _rebuild_from_type_v2(func, new_type, args, state):
    # 对 func(*args) 执行，如果返回值类型不是 new_type，则执行子类化
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)
    # Tensor 类定义了 __setstate__ 方法但未定义 __getstate__ 方法，
    # 所以仅在不等于 Tensor.__setstate__ 时使用 __setstate__
    if (
        getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
        is not Tensor.__setstate__
    ):
        ret.__setstate__(state)
    else:
        # 否则使用 torch._utils._set_obj_state 设置对象状态
        ret = torch._utils._set_obj_state(ret, state)
    return ret


# 注意事项：如果你子类化 Tensor，并希望在进程间共享子类化类，必须更新 torch/multiprocessing/reductions.py
# 以定义一个 ForkingPickler 序列化模式用于该类。
#
# 注意事项：如果你向 Tensor 添加新方法，必须更新 torch/_C/__init__.pyi.in
# 以添加你方法的类型注解，否则该方法不会出现在自动完成中。
class Tensor(torch._C.TensorBase):
    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)
        if type(self) is Tensor and not state:
            # 对于常规张量且无 Python 状态的快速路径
            return self._reduce_ex_internal(proto)
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reduce_ex__, (self,), self, proto)
        # 调用内部 _reduce_ex_internal 方法
        func, args = self._reduce_ex_internal(proto)
        return (_rebuild_from_type_v2, (func, type(self), args, state))
    # 返回当前张量的底层 TypedStorage 对象。
    def storage(self):
        # 如果当前张量有 torch function unary，则通过 torch function 处理并返回处理后的结果
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage, (self,), self)

        # 发出警告，指出 TypedStorage 即将被移除，推荐使用 UntypedStorage
        torch.storage._warn_typed_storage_removal(stacklevel=2)
        # 调用内部方法 _typed_storage() 返回 TypedStorage 对象
        return self._typed_storage()

    # 仅供内部使用，避免引发弃用警告
    def _typed_storage(self):
        # 获取非类型化的存储对象
        untyped_storage = self.untyped_storage()
        # 创建 TypedStorage 对象，使用非类型化存储、当前张量的数据类型和 _internal=True
        return torch.TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

    # 设置张量的状态，用于反序列化
    def __setstate__(self, state):
        # 如果当前张量有 torch function unary，则通过 torch function 处理并返回处理后的结果
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__setstate__, (self,), self, state)
        # 警告：当使用 torch.load() 加载张量时，不会调用此方法；而是由 _rebuild_tensor_v2 管理
        if not self.is_leaf:
            raise RuntimeError("__setstate__ 只能在叶子张量上调用")
        # 根据 state 的长度进行不同的反序列化处理
        if len(state) == 4:
            # 旧版张量的序列化方式
            self.set_(*state)
            return
        elif len(state) == 5:
            # 旧版 Variable 的序列化方式
            self.data = state[0]
            state = (state[3], state[4], state[2])
        # 设置 _backward_hooks 预期是一个无操作。
        # 参见注释 [不要序列化 hooks]
        self.requires_grad, _, self._backward_hooks = state

    # 返回张量的字符串表示形式，支持 tensor_contents 参数
    def __repr__(self, *, tensor_contents=None):
        # 如果当前张量有 torch function unary，则通过 torch function 处理并返回处理后的结果
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.__repr__, (self,), self, tensor_contents=tensor_contents
            )
        # 在 Python 3 中，所有字符串都是 Unicode。
        return torch._tensor_str._str(self, tensor_contents=tensor_contents)

    # 反向传播函数
    def backward(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
        ):
            r"""Computes the gradient of current tensor wrt graph leaves.

            The graph is differentiated using the chain rule. If the tensor is
            non-scalar (i.e. its data has more than one element) and requires
            gradient, the function additionally requires specifying a ``gradient``.
            It should be a tensor of matching type and shape, that represents
            the gradient of the differentiated function w.r.t. ``self``.

            This function accumulates gradients in the leaves - you might need to zero
            ``.grad`` attributes or set them to ``None`` before calling it.
            See :ref:`Default gradient layouts<default-grad-layouts>`
            for details on the memory layout of accumulated gradients.

            .. note::

                If you run any forward ops, create ``gradient``, and/or call ``backward``
                in a user-specified CUDA stream context, see
                :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

            .. note::

                When ``inputs`` are provided and a given input is not a leaf,
                the current implementation will call its grad_fn (though it is not strictly needed to get this gradients).
                It is an implementation detail on which the user should not rely.
                See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

            Args:
                gradient (Tensor, optional): The gradient of the function
                    being differentiated w.r.t. ``self``.
                    This argument can be omitted if ``self`` is a scalar.
                retain_graph (bool, optional): If ``False``, the graph used to compute
                    the grads will be freed. Note that in nearly all cases setting
                    this option to True is not needed and often can be worked around
                    in a much more efficient way. Defaults to the value of
                    ``create_graph``.
                create_graph (bool, optional): If ``True``, graph of the derivative will
                    be constructed, allowing to compute higher order derivative
                    products. Defaults to ``False``.
                inputs (sequence of Tensor, optional): Inputs w.r.t. which the gradient will be
                    accumulated into ``.grad``. All other tensors will be ignored. If not
                    provided, the gradient is accumulated into all the leaf Tensors that were
                    used to compute the :attr:`tensors`.
            """
            # 如果 self 有 torch function 的自定义处理方式，则调用 handle_torch_function 进行处理
            if has_torch_function_unary(self):
                return handle_torch_function(
                    Tensor.backward,
                    (self,),
                    self,
                    gradient=gradient,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                    inputs=inputs,
                )
            # 否则使用 torch.autograd.backward 进行自动求导计算梯度
            torch.autograd.backward(
                self, gradient, retain_graph, create_graph, inputs=inputs
            )
    r"""Registers a backward hook.

    The hook will be called every time a gradient with respect to the
    Tensor is computed. The hook should have the following signature::

        hook(grad) -> Tensor or None

    The hook should not modify its argument, but it can optionally return
    a new gradient which will be used in place of :attr:`grad`.

    This function returns a handle with a method ``handle.remove()``
    that removes the hook from the module.

    .. note::
        See :ref:`backward-hooks-execution` for more information on how when this hook
        is executed, and how its execution is ordered relative to other hooks.

    Example::

        >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
        >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
        >>> v.backward(torch.tensor([1., 2., 3.]))
        >>> v.grad

         2
         4
         6
        [torch.FloatTensor of size (3,)]

        >>> h.remove()  # removes the hook
    """
    # 如果操作支持 Torch 函数的单目操作，则调用 Torch 函数处理
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.register_hook, (self,), self, hook)
    # 如果张量不需要梯度，抛出运行时错误
    if not self.requires_grad:
        raise RuntimeError(
            "cannot register a hook on a tensor that doesn't require gradient"
        )
    # 如果没有定义过梯度回调钩子字典，则初始化为有序字典
    if self._backward_hooks is None:
        self._backward_hooks = OrderedDict()
        # 如果存在梯度函数，将当前对象注册到梯度函数的回调字典中
        if self.grad_fn is not None:
            self.grad_fn._register_hook_dict(self)

    # 导入可移除句柄类
    from torch.utils.hooks import RemovableHandle

    # 创建可移除句柄实例，将钩子函数添加到梯度回调字典中
    handle = RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    # 返回创建的句柄实例
    return handle
    def register_post_accumulate_grad_hook(self, hook):
        r"""Registers a backward hook that runs after grad accumulation.

        The hook will be called after all gradients for a tensor have been accumulated,
        meaning that the .grad field has been updated on that tensor. The post
        accumulate grad hook is ONLY applicable for leaf tensors (tensors without a
        .grad_fn field). Registering this hook on a non-leaf tensor will error!

        The hook should have the following signature::

            hook(param: Tensor) -> None

        Note that, unlike other autograd hooks, this hook operates on the tensor
        that requires grad and not the grad itself. The hook can in-place modify
        and access its Tensor argument, including its .grad field.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks. Since
            this hook runs during the backward pass, it will run in no_grad mode (unless
            create_graph is True). You can use torch.enable_grad() to re-enable autograd
            within the hook if you need it.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> lr = 0.01
            >>> # simulate a simple SGD update
            >>> h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad, alpha=-lr))
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v
            tensor([-0.0100, -0.0200, -0.0300], requires_grad=True)

            >>> h.remove()  # removes the hook
        """
        # 如果该张量具有 torch 函数的单目运算，则调用 torch 函数处理注册后累积梯度钩子
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.register_post_accumulate_grad_hook, (self,), self, hook
            )
        # 如果张量不需要梯度，则抛出运行时错误
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )
        # 如果张量有梯度函数，则抛出运行时错误，因为后累积梯度钩子只能注册在叶子张量上
        if self.grad_fn is not None:
            raise RuntimeError(
                "post accumulate grad hooks cannot be registered on non-leaf tensors"
            )
        # 如果尚未初始化后累积梯度钩子字典，则初始化为有序字典
        if self._post_accumulate_grad_hooks is None:
            self._post_accumulate_grad_hooks: Dict[Any, Any] = OrderedDict()

        # 引入可移除的句柄类
        from torch.utils.hooks import RemovableHandle

        # 创建一个可移除的句柄实例
        handle = RemovableHandle(self._post_accumulate_grad_hooks)
        # 将钩子函数添加到后累积梯度钩子字典中
        self._post_accumulate_grad_hooks[handle.id] = hook
        # 返回句柄实例，以便后续可以通过句柄移除该钩子
        return handle
    def reinforce(self, reward):
        # 定义内部函数 trim，用于去除字符串左右两边的空格并重新组合成一个新的字符串
        def trim(str):
            return "\n".join([line.strip() for line in str.split("\n")])

        # 抛出运行时错误，并格式化输出错误信息
        raise RuntimeError(
            trim(
                r"""reinforce() was removed.
            Use torch.distributions instead.
            See https://pytorch.org/docs/main/distributions.html

            Instead of:

            probs = policy_network(state)
            action = probs.multinomial()
            next_state, reward = env.step(action)
            action.reinforce(reward)
            action.backward()

            Use:

            probs = policy_network(state)
            # NOTE: categorical is equivalent to what used to be called multinomial
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            next_state, reward = env.step(action)
            loss = -m.log_prob(action) * reward
            loss.backward()
        """
            )
        )

    # 为 _C.TensorBase.detach 方法添加文档字符串
    detach = _C._add_docstr(
        _C.TensorBase.detach,
        r"""
    Returns a new Tensor, detached from the current graph.

    The result will never require gradient.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.

    .. note::

      Returned Tensor shares the same storage with the original one.
      In-place modifications on either of them will be seen, and may trigger
      errors in correctness checks.
    """,
    )

    # 为 _C.TensorBase.detach_ 方法添加文档字符串
    detach_ = _C._add_docstr(
        _C.TensorBase.detach_,
        r"""
    Detaches the Tensor from the graph that created it, making it a leaf.
    Views cannot be detached in-place.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.
    """,
    )

    def is_shared(self):
        r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        # 如果具有 torch 函数的一元操作，则调用 handle_torch_function 处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.is_shared, (self,), self)
        # 否则调用 _typed_storage 方法判断是否在共享内存中
        return self._typed_storage()._is_shared()

    def share_memory_(self):
        r"""Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.

        See :meth:`torch.UntypedStorage.share_memory_` for more details.
        """
        # 如果具有 torch 函数的一元操作，则调用 handle_torch_function 处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.share_memory_, (self,), self)
        # 否则调用 _typed_storage 方法将底层存储移动到共享内存
        self._typed_storage()._share_memory_()
        return self
    def module_load(self, other, assign=False):
        r"""Defines how to transform ``other`` when loading it into ``self`` in :meth:`~nn.Module.load_state_dict`.

        Used when :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        It is expected that ``self`` is a parameter or buffer in an ``nn.Module`` and ``other`` is the
        value in the state dictionary with the corresponding key, this method defines
        how ``other`` is remapped before being swapped with ``self`` via
        :func:`~torch.utils.swap_tensors`` in ``module.load_state_dict()``.

        .. note::
            This method should always return a new object that is not ``self`` or ``other``.
            For example, the default implementation returns ``self.copy_(other).detach()``
            if ``assign`` is ``False`` or ``other.detach()`` if ``assign`` is ``True``.

        Args:
            other (Tensor): value in state dict with key corresponding to ``self``
            assign (bool): the assign argument passed to :meth:`nn.Module.load_state_dict`

        """
        # 如果存在 torch function 变元，调用 torch function 处理
        if has_torch_function_variadic(self, other):
            return handle_torch_function(
                Tensor.module_load, (self, other), self, other, assign=assign
            )

        # 如果 assign 参数为 True，返回 other 的 detached 版本
        if assign:
            return other.detach()
        else:
            # 否则，使用 self 的 copy_ 方法复制 other，并且返回 detached 版本
            return self.copy_(other).detach()

    def __reversed__(self):
        r"""Reverses the tensor along dimension 0."""
        # 如果存在 torch function 一元操作，调用 torch function 处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reversed__, (self,), self)
        # 如果张量的维度为 0，返回自身
        if self.dim() == 0:
            return self
        else:
            # 否则，沿着第 0 维进行翻转
            return self.flip(0)

    def norm(
        self,
        p: Optional[Union[float, str]] = "fro",
        dim=None,
        keepdim=False,
        dtype=None,
    ):
        r"""See :func:`torch.norm`"""
        # 如果存在 torch function 一元操作，调用 torch function 处理
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.norm, (self,), self, p=p, dim=dim, keepdim=keepdim, dtype=dtype
            )
        # 调用 torch.norm 函数计算张量的范数
        return torch.norm(self, p, dim, keepdim, dtype=dtype)

    def solve(self, other):
        from torch._linalg_utils import solve

        # 调用 solve 函数解决线性系统 self @ x = other
        return solve(self, other)

    def lstsq(self, other):
        from torch._linalg_utils import lstsq

        # 调用 lstsq 函数计算最小二乘解
        return lstsq(self, other)

    def eig(self, eigenvectors=False):
        from torch._linalg_utils import eig

        # 调用 eig 函数计算自适应特征值和特征向量
        return eig(self, eigenvectors=eigenvectors)

    def symeig(self, eigenvectors=False):
        from torch._linalg_utils import _symeig

        # 调用 _symeig 函数计算对称特征值和特征向量
        return _symeig(self, eigenvectors=eigenvectors)
    def lu(self, pivot=True, get_infos=False):
        r"""See :func:`torch.lu`"""
        # 如果 get_infos 为 True，则不需要检查错误，反之亦然
        # 如果对象有 torch 函数的一元版本，则调用处理 torch 函数的函数
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.lu, (self,), self, pivot=pivot, get_infos=get_infos
            )

        # 使用 torch._lu_with_info 函数计算 LU 分解，并获取置换信息和错误信息（如果需要）
        LU, pivots, infos = torch._lu_with_info(
            self, pivot=pivot, check_errors=(not get_infos)
        )
        # 如果需要获取额外的信息，则返回 LU 矩阵、置换向量和信息向量
        if get_infos:
            return LU, pivots, infos
        else:
            # 否则，只返回 LU 矩阵和置换向量
            return LU, pivots

    def stft(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "Optional[Tensor]" = None,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: Optional[bool] = None,
        return_complex: Optional[bool] = None,
    ):
        r"""See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        """
        # 如果对象有 torch 函数的一元版本，则调用处理 torch 函数的函数
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.stft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
            )
        # 调用 torch.stft 函数计算短时傅里叶变换，并返回结果
        return torch.stft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex=return_complex,
        )

    def istft(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "Optional[Tensor]" = None,
        center: bool = True,
        normalized: bool = False,
        onesided: Optional[bool] = None,
        length: Optional[int] = None,
        return_complex: bool = False,
    ):
        # 调用 torch.istft 函数计算逆短时傅里叶变换，并返回结果
        return torch.istft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex=return_complex,
        )
    ):
        r"""See :func:`torch.istft`"""
        # 如果self对象有torch函数的自定义处理方法，使用torch函数处理
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.istft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                normalized=normalized,
                onesided=onesided,
                length=length,
                return_complex=return_complex,
            )
        # 否则调用torch库中的istft函数
        return torch.istft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex=return_complex,
        )

    def resize(self, *sizes):
        # 如果self对象有torch函数的自定义处理方法，使用torch函数处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.resize, (self,), self, *sizes)
        # 发出警告，非原地调整大小已经不建议使用
        warnings.warn("non-inplace resize is deprecated")
        from torch.autograd._functions import Resize

        # 使用Resize类的apply方法调整大小
        return Resize.apply(self, sizes)

    def resize_as(self, tensor):
        # 如果self对象和tensor对象有torch函数的自定义处理方法，使用torch函数处理
        if has_torch_function_variadic(self, tensor):
            return handle_torch_function(Tensor.resize_as, (self, tensor), self, tensor)
        # 发出警告，非原地调整大小已经不建议使用
        warnings.warn("non-inplace resize_as is deprecated")
        from torch.autograd._functions import Resize

        # 使用Resize类的apply方法调整大小为tensor的大小
        return Resize.apply(self, tensor.size())

    def split(self, split_size, dim=0):
        r"""See :func:`torch.split`"""
        # 如果self对象有torch函数的自定义处理方法，使用torch函数处理
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.split, (self,), self, split_size, dim=dim
            )
        # 如果split_size是Tensor类型，尝试转换为整数
        if isinstance(split_size, Tensor):
            try:
                split_size = int(split_size)
            except ValueError:
                pass

        # 如果split_size是整数或torch.SymInt类型，调用torch库中的split方法
        if isinstance(split_size, (int, torch.SymInt)):
            return torch._VF.split(self, split_size, dim)  # type: ignore[attr-defined]
        else:
            # 否则调用torch库中的split_with_sizes方法
            return torch._VF.split_with_sizes(self, split_size, dim)

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        r"""Returns the unique elements of the input tensor.

        See :func:`torch.unique`
        """
        # 如果self对象有torch函数的自定义处理方法，使用torch函数处理
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique,
                (self,),
                self,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        # 否则调用torch库中的unique函数返回输入张量的唯一元素
        return torch.unique(
            self,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )
    # 定义了一个方法用于去除每个连续等价元素组中除第一个元素外的所有元素
    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        # 如果对象有torch函数处理，则调用处理函数，否则调用torch库中的unique_consecutive函数
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique_consecutive,
                (self,),
                self,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        # 调用torch库中的unique_consecutive函数进行处理
        return torch.unique_consecutive(
            self, return_inverse=return_inverse, return_counts=return_counts, dim=dim
        )

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rsub__(self, other):
        # 调用C++实现的变量函数rsub，执行other - self
        return _C._VariableFunctions.rsub(self, other)

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rdiv__(self, other):
        # 返回self的倒数乘以other
        return self.reciprocal() * other

    # 将右除等同于右真除
    __rtruediv__ = __rdiv__
    # 与__rdiv__类似，但使用了C++ TensorBase类中的__idiv__
    __itruediv__ = _C.TensorBase.__idiv__

    # 处理torch函数和封装类型错误以不实现，使用C++ TensorBase类中的pow函数
    __pow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(
        _C.TensorBase.pow
    )
    # 处理torch函数和封装类型错误以不实现，使用C++ TensorBase类中的pow_函数
    __ipow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(
        _C.TensorBase.pow_
    )

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmod__(self, other):
        # 返回other对self取模
        return torch.remainder(other, self)

    # 定义格式化方法，根据对象类型和属性来确定返回值
    def __format__(self, format_spec):
        # 如果对象有torch函数处理，则调用处理函数，否则执行默认处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
        # 如果对象是标量并且非元数据，返回其item并格式化
        if self.dim() == 0 and not self.is_meta and type(self) is Tensor:
            return self.item().__format__(format_spec)
        # 否则执行默认的格式化处理
        return object.__format__(self, format_spec)

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rpow__(self, other):
        # 返回other对self求幂
        return torch.pow(other, self)

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __floordiv__(self, other):
        # 返回self除以other的整数部分
        return torch.floor_divide(self, other)

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rfloordiv__(self, other):
        # 返回other除以self的整数部分
        return torch.floor_divide(other, self)

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rlshift__(self, other):
        # 返回other左移self位
        return torch.bitwise_left_shift(other, self)

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rrshift__(self, other):
        # 返回other右移self位
        return torch.bitwise_right_shift(other, self)

    # 处理torch函数和封装类型错误以不实现
    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmatmul__(self, other):
        # 返回other与self的矩阵乘积
        return torch.matmul(other, self)

    # 对象正值的操作，即返回本身的正值
    __pos__ = _C.TensorBase.positive
    # 对象负值的操作，即返回本身的负值
    __neg__ = _C.TensorBase.neg
    # 对象的绝对值操作，即返回本身的绝对值
    __abs__ = _C.TensorBase.abs
    # 如果对象有定义 torch 函数的一元操作，使用 torch 函数处理该方法
    def __len__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__len__, (self,), self)
        # 如果张量是0维的，抛出类型错误异常
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        # 如果处于追踪状态，发出警告，说明使用 len() 获取张量形状可能导致追踪不正确
        if torch._C._get_tracing_state():
            warnings.warn(
                "Using len to get tensor shape might cause the trace to be incorrect. "
                "Recommended usage would be tensor.shape[0]. "
                "Passing a tensor of different shape might lead to errors or silently give "
                "incorrect results.",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        # 返回张量的第一个维度大小
        return self.shape[0]

    def __iter__(self):
        # 注意：这里使用 'imap' 而不是 'map'，这样在 Python 2 中我们会得到一个生成器，并且不会立即执行所有索引。
        # 这可以节省工作，并且有助于保持追踪顺序的确定性（例如，如果你 zip(*hiddens)，则急切映射将强制执行 hiddens[0] 的所有索引，然后是 hiddens[1]，而生成器映射将交错执行它们。）
        # 注意：我们故意跳过 __torch_function__ 分发。参见 gh-54457
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        # 如果处于追踪状态，发出警告，说明迭代张量可能导致追踪不正确
        if torch._C._get_tracing_state():
            warnings.warn(
                "Iterating over a tensor might cause the trace to be incorrect. "
                "Passing a tensor of different shape won't change the number of "
                "iterations executed (and might lead to errors or silently give "
                "incorrect results).",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        # 返回张量在第0维上解绑后的迭代器
        return iter(self.unbind(0))

    def __hash__(self):
        # 不要在此处处理 __torch_function__，因为用户默认的实现会处理大多数函数，而且很可能会做错。
        # 如果需要，用户可以在子类上定义这个方法来轻松覆盖它。
        # 返回张量对象的 ID 作为其哈希值
        return id(self)

    def __dir__(self):
        # 如果对象有定义 torch 函数的一元操作，使用 torch 函数处理该方法
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dir__, (self,), self)
        # 获取张量类的方法列表
        tensor_methods = dir(self.__class__)
        # 移除已弃用的 'volatile' 属性
        tensor_methods.remove("volatile")
        # 获取张量对象的实例属性列表
        attrs = list(self.__dict__.keys())
        # 合并方法和属性列表
        keys = tensor_methods + attrs

        # 仅在张量是 dense 且不在 CUDA 上时，移除 "__cuda_array_interface__" 属性
        if (not self.is_cuda) or self.is_sparse:
            keys.remove("__cuda_array_interface__")

        # 返回排序后的方法和属性列表
        return sorted(keys)

    # Numpy 数组接口，支持 `numpy.asarray(tensor) -> ndarray`
    __array_priority__ = 1000  # 优先使用张量操作而不是 numpy 操作
    # 当对象在使用__array__方法时，检查是否存在torch的函数处理单元，若存在则调用处理函数处理该方法
    def __array__(self, dtype=None):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__array__, (self,), self, dtype=dtype)
        # 若未指定dtype，则返回自身的numpy表示
        if dtype is None:
            return self.numpy()
        else:
            # 否则返回将numpy数组转换为指定dtype的torch张量
            return self.numpy().astype(dtype, copy=False)

    # 当使用__array_wrap__方法时，重新包装Numpy数组为合适的张量类型，以支持如`numpy.sin(tensor) -> tensor`或`numpy.greater(tensor, 0) -> ByteTensor`
    def __array_wrap__(self, array):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.__array_wrap__, (self,), self, array=array
            )
        # 若数组的dtype为布尔类型，进行类型转换为"uint8"，因为torch没有内置的布尔张量
        if array.dtype == bool:
            array = array.astype("uint8")
        # 返回从Numpy数组转换而来的torch张量
        return torch.from_numpy(array)

    # 检查张量中是否存在指定的元素
    def __contains__(self, element):
        r"""Check if `element` is present in tensor

        Args:
            element (Tensor or scalar): element to be checked
                for presence in current tensor"
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__contains__, (self,), self, element)
        # 如果元素是torch.Tensor、数字、torch的符号整数、torch的符号浮点数或torch的符号布尔类型之一，则返回是否存在的布尔值
        if isinstance(
            element, (torch.Tensor, Number, torch.SymInt, torch.SymFloat, torch.SymBool)
        ):
            return (element == self).any().item()  # type: ignore[union-attr]

        # 抛出错误，因为Tensor.__contains__仅支持Tensor或标量类型的元素
        raise RuntimeError(
            f"Tensor.__contains__ only supports Tensor or scalar, but you passed in a {type(element)}."
        )

    # 定义属性
    @property
    def __cuda_array_interface__(self):
        """Array view description for cuda tensors.

        See:
        https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
        """
        # 检查是否有 Torch 函数的一元操作
        if has_torch_function_unary(self):
            # 调用 Torch 函数处理 CUDA 数组接口，返回处理结果
            return handle_torch_function(Tensor.__cuda_array_interface__.__get__, (self,), self)  # type: ignore[attr-defined]

        # 抛出 AttributeError 来表示不支持的张量类型，使得 hasattr(cpu_tensor, "__cuda_array_interface__") 为 False
        if not self.is_cuda:
            raise AttributeError(
                f"Can't get __cuda_array_interface__ on non-CUDA tensor type: {self.type()} "
                "If CUDA data is required use tensor.cuda() to copy tensor to device memory."
            )

        # 对稀疏张量抛出 AttributeError
        if self.is_sparse:
            raise AttributeError(
                f"Can't get __cuda_array_interface__ on sparse type: {self.type()} "
                "Use Tensor.to_dense() to convert to a dense tensor first."
            )

        # 如果张量需要梯度，抛出 RuntimeError，与 tensor.__array__() 的行为一致
        if self.requires_grad:
            raise RuntimeError(
                "Can't get __cuda_array_interface__ on Variable that requires grad. "
                "If gradients aren't required, use var.detach() to get Variable that doesn't require grad."
            )

        # 根据张量的数据类型选择 CUDA 设备的字节顺序及本机字节顺序的映射关系
        typestr = {
            torch.complex64: "<c8",
            torch.complex128: "<c16",
            torch.float16: "<f2",
            torch.float32: "<f4",
            torch.float64: "<f8",
            torch.uint8: "|u1",
            torch.int8: "|i1",
            torch.int16: "<i2",
            torch.int32: "<i4",
            torch.int64: "<i8",
        }[self.dtype]

        itemsize = self.element_size()

        shape = tuple(self.shape)
        # 如果张量是连续的，则 __cuda_array_interface__ v2 要求步长为空（未设置或设置为 None）
        if self.is_contiguous():
            strides = None
        else:
            # 计算非连续张量的步长
            strides = tuple(s * itemsize for s in self.stride())
        data_ptr = self.data_ptr() if self.numel() > 0 else 0
        data = (data_ptr, False)  # 只读属性设置为 False

        # 返回描述 CUDA 数组视图的字典
        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=2)

    def storage_type(self):
        r"""storage_type() -> type

        Returns the type of the underlying storage.

        """
        # 检查是否有 Torch 函数的一元操作
        if has_torch_function_unary(self):
            # 处理 Torch 函数的调用
            return handle_torch_function(Tensor.storage_type, (self,), self)

        # 警告即将移除的有类型存储
        torch.storage._warn_typed_storage_removal()

        # 返回底层存储的类型
        return self._typed_storage()._get_legacy_storage_class()
    def refine_names(self, *names):
        r"""Refines the dimension names of :attr:`self` according to :attr:`names`.
        
        Refining is a special case of renaming that "lifts" unnamed dimensions.
        A ``None`` dim can be refined to have any name; a named dim can only be
        refined to have the same name.
        
        Because named tensors can coexist with unnamed tensors, refining names
        gives a nice way to write named-tensor-aware code that works with both
        named and unnamed tensors.
        
        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded greedily; it is expanded in-place to fill
        :attr:`names` to the same length as ``self.dim()`` using names from the
        corresponding indices of ``self.names``.
        
        Python 2 does not support Ellipsis but one may use a string literal
        instead (``'...'``).
        
        Args:
            names (iterable of str): The desired names of the output tensor. May
                contain up to one Ellipsis.
        
        Examples::
        
            >>> imgs = torch.randn(32, 3, 128, 128)
            >>> named_imgs = imgs.refine_names('N', 'C', 'H', 'W')
            >>> named_imgs.names
            ('N', 'C', 'H', 'W')
        
            >>> tensor = torch.randn(2, 3, 5, 7, 11)
            >>> tensor = tensor.refine_names('A', ..., 'B', 'C')
            >>> tensor.names
            ('A', None, None, 'B', 'C')
        
        .. warning::
            The named tensor API is experimental and subject to change.
        
        """
        # 如果有 Torch 函数的自定义处理，则调用 Torch 函数处理 refine_names
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.refine_names, (self,), self, *names)
        # 解析 names 中的 Ellipsis，根据 self.names 扩展 names 到相同长度
        names = resolve_ellipsis(names, self.names, "refine_names")
        # 调用父类的 refine_names 方法，传入解析后的 names
        return super().refine_names(names)
    def align_to(self, *names):
        r"""Permutes the dimensions of the :attr:`self` tensor to match the order
        specified in :attr:`names`, adding size-one dims for any new names.

        All of the dims of :attr:`self` must be named in order to use this method.
        The resulting tensor is a view on the original tensor.

        All dimension names of :attr:`self` must be present in :attr:`names`.
        :attr:`names` may contain additional names that are not in ``self.names``;
        the output tensor has a size-one dimension for each of those new names.

        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded to be equal to all dimension names of :attr:`self`
        that are not mentioned in :attr:`names`, in the order that they appear
        in :attr:`self`.

        Python 2 does not support Ellipsis but one may use a string literal
        instead (``'...'``).

        Args:
            names (iterable of str): The desired dimension ordering of the
                output tensor. May contain up to one Ellipsis that is expanded
                to all unmentioned dim names of :attr:`self`.

        Examples::

            >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)
            >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')

            # Move the F and E dims to the front while keeping the rest in order
            >>> named_tensor.align_to('F', 'E', ...)

        .. warning::
            The named tensor API is experimental and subject to change.

        """
        # 如果自身有单目运算，则使用 Torch 函数处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.align_to, (self,), self, *names)
        # 寻找 Ellipsis 的位置索引
        ellipsis_idx = single_ellipsis_index(names, "align_to")
        # 如果没有 Ellipsis，直接调用父类的 align_to 方法
        if ellipsis_idx is None:
            return super().align_to(names)
        # 否则，调用父类的 align_to 方法，展开 Ellipsis
        return super().align_to(
            [name for name in names if not is_ellipsis(name)], ellipsis_idx
        )

    def unflatten(self, dim, sizes):
        r"""
        unflatten(dim, sizes) -> Tensor

        See :func:`torch.unflatten`.

        """
        # 如果自身有单目运算，则使用 Torch 函数处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.unflatten, (self,), self, dim, sizes)

        # 如果 sizes 为空，则抛出 RuntimeError
        if not sizes:
            raise RuntimeError("unflatten: sizes must be non-empty")

        names = None
        # 如果 sizes 是 OrderedDict 或者 tuple/list of tuple/list 类型，则解压并使用 names
        if isinstance(sizes, OrderedDict) or (
            isinstance(sizes, (tuple, list)) and isinstance(sizes[0], (tuple, list))
        ):
            names, sizes = unzip_namedshape(sizes)
            # 调用父类的 unflatten 方法，使用 names
            return super().unflatten(dim, sizes, names)
        else:
            # 否则，直接调用父类的 unflatten 方法
            return super().unflatten(dim, sizes)
    def rename_(self, *names, **rename_map):
        """In-place version of :meth:`~Tensor.rename`."""

        if has_torch_function_unary(self):
            # 如果对象有torch函数的一元操作，调用torch函数处理
            return handle_torch_function(
                Tensor.rename_, (self,), self, *names, **rename_map
            )

        # Note [rename_ / rename API]
        # Python API与C++ API有所不同：
        # 1) tensor.rename(*names) 接受可变参数名列表
        # 2) tensor.rename(**rename_map) 接受一个映射用于重命名
        # C++是静态的，难以实现类似的行为。
        # 调用update_names函数，在原地更新维度名称
        return update_names(self, names, rename_map, inplace=True)

    def rename(self, *names, **rename_map):
        """Renames dimension names of :attr:`self`.

        There are two main usages:

        ``self.rename(**rename_map)`` 返回一个视图，其中维度按照映射 :attr:`rename_map` 指定的重命名。

        ``self.rename(*names)`` 返回一个视图，按位置重命名所有维度，使用 :attr:`names`。

        使用 ``self.rename(None)`` 可以在张量上删除名称。

        不能同时指定位置参数 :attr:`names` 和关键字参数 :attr:`rename_map`。

        Examples::

            >>> imgs = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
            >>> renamed_imgs = imgs.rename(N='batch', C='channels')
            >>> renamed_imgs.names
            ('batch', 'channels', 'H', 'W')

            >>> renamed_imgs = imgs.rename(None)
            >>> renamed_imgs.names
            (None, None, None, None)

            >>> renamed_imgs = imgs.rename('batch', 'channel', 'height', 'width')
            >>> renamed_imgs.names
            ('batch', 'channel', 'height', 'width')

        .. warning::
            命名张量API是实验性的，可能会有变化。

        """
        if has_torch_function_unary(self):
            # 如果对象有torch函数的一元操作，调用torch函数处理
            return handle_torch_function(
                Tensor.rename, (self,), self, *names, **rename_map
            )

        # See Note [rename_ / rename API]
        # 调用update_names函数，在不在原地更新维度名称
        return update_names(self, names, rename_map, inplace=False)

    def to_sparse_coo(self):
        """Convert a tensor to :ref:`coordinate format <sparse-coo-docs>`.

        Examples::

             >>> dense = torch.randn(5, 5)
             >>> sparse = dense.to_sparse_coo()
             >>> sparse._nnz()
             25

        """
        # 返回稀疏张量的COO格式表示
        return self.to_sparse()
    def dim_order(self):
        """
        dim_order() -> tuple
        
        Returns a tuple of int describing the dim order or physical layout of :attr:`self`.
        
        Args:
            None
        
        Dim order represents how dimensions are laid out in memory,
        starting from the outermost to the innermost dimension.
        
        Example::
            >>> torch.empty((2, 3, 5, 7)).dim_order()
            (0, 1, 2, 3)
            >>> torch.empty((2, 3, 5, 7), memory_format=torch.channels_last).dim_order()
            (0, 2, 3, 1)
        
        .. warning::
            The dim_order tensor API is experimental and subject to change.
        """
        # 如果对象具有自定义的 torch 函数，交由处理器处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.dim_order, (self,), self)
        
        # 导入 torch 内部的实用模块
        import torch._prims_common as utils
        
        # 返回描述对象维度顺序或内存布局的整数元组
        return tuple(utils.compute_elementwise_output_logical_to_physical_perm(self))

    def _update_names(self, names, inplace):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor._update_names, (self,), self, names, inplace
            )
        
        # 查看注释 [rename_ / rename API]
        if inplace:
            # 在原地重命名，并返回结果
            return super().rename_(names)
        else:
            # 创建重命名后的新对象，并返回结果
            return super().rename(names)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.
        
        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.
        
        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.
        
        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
        # 如果未传入 kwargs，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        
        # 如果不是所有参数类型都是当前类的子类，则返回未实现
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        
        # 禁用 torch 函数的子类处理
        with _C.DisableTorchFunctionSubclass():
            # 执行函数并返回结果
            ret = func(*args, **kwargs)
            # 如果函数在默认不包装函数列表中，则直接返回结果
            if func in get_default_nowrap_functions():
                return ret
            else:
                # 否则将结果转换为当前类的实例并返回
                return _convert(ret, cls)
    
    # 禁用 torch 分发实现
    __torch_dispatch__ = _C._disabled_torch_dispatch_impl
    # 创建一个 DLpack 'capsule'，用于将当前张量导出到其他库中使用。
    # 这个方法将被消费这个capsule的库的`from_dlpack`方法调用。
    # `from_dlpack`方法会将当前流作为规范的一部分传递给这个方法。

    def __dlpack__(self, stream=None):
        # 如果张量有torch函数的一元操作，调用torch函数处理逻辑
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dlpack__, (self,), self, stream)

        # DLPack capsules无法捕获所有PyTorch的语义，
        # 因此禁止导出那些可能丢失属性的张量，比如requires_grad或者有共轭位设置的。
        if self.requires_grad:
            raise RuntimeError("无法导出需要梯度的张量，请使用tensor.detach()")
        if self.is_conj():
            raise RuntimeError("无法导出带有共轭位设置的张量")
        if self.layout != torch.strided:
            raise RuntimeError("无法导出布局不是torch.strided的张量")

        if stream is not None and type(stream) is not int:
            # CUDA/ROCm中的流指针是唯一编号的，可以通过其整数值检索。
            raise TypeError("stream必须是``int``或``none``")
        elif stream is not None and stream != -1:
            if self.device.type == "cuda":
                # 注意：这段逻辑处理了默认流的特殊情况，必须与torch/utils/dlpack.py中的from_dlpack保持同步。
                if stream == 1 and torch.version.hip is None:
                    stream = torch.cuda.default_stream()
                elif stream == 0 and torch.version.hip is not None:
                    stream = torch.cuda.default_stream()
                else:
                    stream = torch.cuda.ExternalStream(stream)
                # 只在不同的流上进行同步
                sync_stream = torch.cuda.current_stream()
                if stream != sync_stream:
                    event = torch.cuda.Event()
                    event.record(sync_stream)
                    stream.wait_event(event)
        
        # 将张量转换为DLpack格式并返回
        return torch.to_dlpack(self)
    # 定义一个特殊方法 `__dlpack_device__`，返回一个元组，包含设备类型枚举和索引
    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        # 如果对象支持 Torch 函数的一元操作，调用 Torch 函数处理
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dlpack_device__, (self,), self)

        # 导入 DLDeviceType 类型，用于指定设备类型
        from torch.utils.dlpack import DLDeviceType

        # 获取对象的设备信息
        device = self.device
        # 如果设备索引不为 None，则使用索引；否则默认为 0
        idx = device.index if device.index is not None else 0
        # 获取 Torch 设备类型
        torch_device_type = device.type

        # 根据 Torch 设备类型确定 DLDeviceType 枚举值
        if torch_device_type == "cuda" and torch.version.hip is not None:
            device_type = DLDeviceType.kDLROCM  # 如果是 CUDA 设备且是 HIP 平台
        elif torch_device_type == "cpu" and self.is_pinned():
            device_type = DLDeviceType.kDLCPUPinned  # 如果是 CPU 设备且已经固定
        elif torch_device_type == "cuda":
            device_type = DLDeviceType.kDLGPU  # 如果是 CUDA 设备
        elif torch_device_type == "cpu":
            device_type = DLDeviceType.kDLCPU  # 如果是 CPU 设备
        elif self.device.type == "xpu":
            device_type = DLDeviceType.kDLOneAPI  # 如果是 XPU 设备
        else:
            # 如果设备类型未知，则抛出异常
            raise ValueError(f"Unknown device type {torch_device_type} for Dlpack")

        # 返回设备类型枚举值和设备索引
        return (device_type, idx)

    # 指定当前模块为 "torch"
    __module__ = "torch"
# 定义一个函数 `_convert`，用于将返回值 `ret` 转换为指定的类 `cls`
def _convert(ret, cls):
    # 如果目标类 `cls` 是 `Tensor`，则直接返回 `ret`，不进行转换
    if cls is Tensor:
        return ret

    # 如果 `ret` 是 `Tensor` 类的实例但不是 `cls` 类型的实例，则将其转换为 `cls` 类的子类
    if isinstance(ret, Tensor) and not isinstance(ret, cls):
        ret = ret.as_subclass(cls)

    # 如果 `ret` 是元组或列表类型
    if isinstance(ret, (tuple, list)):
        # 递归地对元组或列表中的每个元素进行 `_convert` 转换，并使用相同的类型构造返回结果
        ret = type(ret)(_convert(r, cls) for r in ret)

    # 返回转换后的结果 `ret`
    return ret
```