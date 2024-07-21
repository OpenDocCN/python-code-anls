# `.\pytorch\torch\nn\parameter.py`

```
from collections import OrderedDict  # 导入有序字典模块

import torch  # 导入 PyTorch 模块
from torch._C import _disabled_torch_function_impl  # 导入 _disabled_torch_function_impl 函数


# 用于结合 _TensorMeta 和 Parameter 的实例检查重写的元类。
class _ParameterMeta(torch._C._TensorMeta):
    # 使得 isinstance(t, Parameter) 对于具有 _is_param 标志的自定义张量实例返回 True。
    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance) or (
            isinstance(instance, torch.Tensor) and getattr(instance, "_is_param", False)
        )


class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            # 为了便于向后兼容性维护，保留此路径用于标准 Tensor。
            # 最终，我们应该改变标准 Tensor 的行为来匹配。
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # 自定义张量的路径：在实例上设置一个标志，指示其为参数。
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "Parameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        t._is_param = True
        return t

    # 注意：以下 3 个方法仅适用于标准 Tensor。自定义张量类型的参数仍被视为该自定义张量类型，这些方法不会对它们调用。
    # 实现深拷贝方法，用于复制对象及其属性
    def __deepcopy__(self, memo):
        # 如果对象已经在备忘录中存在，则直接返回备忘录中的对象
        if id(self) in memo:
            return memo[id(self)]
        else:
            # 创建对象的深拷贝，包括数据的克隆和是否需要梯度的信息
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            # 将新对象存入备忘录
            memo[id(self)] = result
            return result

    # 返回对象的字符串表示，包括类名和数据信息
    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()

    # 实现对象的序列化方法，用于将对象转换为可序列化的状态
    def __reduce_ex__(self, proto):
        # 获取对象的状态
        state = torch._utils._get_obj_state(self)

        # 不序列化对象的钩子（hooks）
        hooks = OrderedDict()
        if not state:
            # 如果没有额外状态，返回参数对象的重建方法和参数数据及梯度信息以及空的钩子
            return (
                torch._utils._rebuild_parameter,
                (self.data, self.requires_grad, hooks),
            )

        # 否则返回参数对象的带状态重建方法和参数数据及梯度信息以及钩子和状态
        return (
            torch._utils._rebuild_parameter_with_state,
            (self.data, self.requires_grad, hooks, state),
        )

    # 禁用 Torch 函数的特殊方法
    __torch_function__ = _disabled_torch_function_impl
class UninitializedTensorMixin:
    _allowed_methods = [
        torch.Tensor.__hash__,  # 允许哈希方法
        torch.Tensor.size,  # 允许获取张量大小的方法
        torch.Tensor.copy_,  # 允许复制张量的方法
        torch.Tensor.is_complex,  # 允许检查张量是否为复数的方法
        torch.Tensor.is_floating_point,  # 允许检查张量是否为浮点数的方法
        torch.Tensor.half,  # 允许将张量转换为半精度浮点数的方法
        torch.Tensor.float,  # 允许将张量转换为单精度浮点数的方法
        torch.Tensor.double,  # 允许将张量转换为双精度浮点数的方法
        torch.Tensor.char,  # 允许将张量转换为字符型的方法
        torch.Tensor.short,  # 允许将张量转换为短整型的方法
        torch.Tensor.int,  # 允许将张量转换为整型的方法
        torch.Tensor.long,  # 允许将张量转换为长整型的方法
        torch.Tensor.cuda,  # 允许将张量移动到 CUDA 设备的方法
        torch.Tensor.cpu,  # 允许将张量移动到 CPU 设备的方法
        torch.Tensor.to,  # 允许将张量转换为指定设备和数据类型的方法
        torch.Tensor.get_device,  # 允许获取张量所在设备的方法
        torch._has_compatible_shallow_copy_type,  # 允许检查张量是否有兼容的浅复制类型的方法
    ]

    def materialize(self, shape, device=None, dtype=None):
        r"""Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
        if device is None:
            device = self.data.device  # 如果设备为空，则使用当前数据的设备
        if dtype is None:
            dtype = self.data.dtype  # 如果数据类型为空，则使用当前数据的数据类型
        self.data = torch.empty(shape, device=device, dtype=dtype)  # 创建一个形状、设备和数据类型与当前一致的空张量
        self.__class__ = self.cls_to_become  # 修改当前实例的类为预定义的类

    @property
    def shape(self):
        raise RuntimeError(
            "Can't access the shape of an uninitialized parameter or buffer. "
            "This error usually happens in `load_state_dict` when trying to load "
            "an uninitialized parameter into an initialized one. "
            "Call `forward` to initialize the parameters before accessing their attributes."
        )

    def share_memory_(self):
        raise RuntimeError(
            "Can't share memory on an uninitialized parameter or buffer. "
            "Call `forward` to initialize the parameters before calling "
            "`module.share_memory()`."
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}>"  # 返回当前实例的类名的字符串表示形式

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (self.__class__, (self.requires_grad,))  # 序列化当前实例时返回一个元组，用于重新构造实例

    @classmethod
    # 实现自定义的 __torch_function__ 方法，用于处理对于 Torch 张量的函数调用
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 检查是否是允许调用的方法，或者是被包装在描述符中的 Tensor 属性访问
        if func in cls._allowed_methods or func.__class__.__name__ == "method-wrapper":
            # 如果 kwargs 为 None，初始化为空字典
            if kwargs is None:
                kwargs = {}
            # 调用父类的 __torch_function__ 方法处理函数调用
            return super().__torch_function__(func, types, args, kwargs)
        # 如果不是允许调用的方法，抛出 ValueError 异常
        raise ValueError(
            f"Attempted to use an uninitialized parameter in {func}. "
            "This error happens when you are using a `LazyModule` or "
            f"explicitly manipulating `torch.nn.parameter.{cls.__name__}` "
            "objects. When using LazyModules Call `forward` with a dummy batch "
            "to initialize the parameters before calling torch functions"
        )
# 检查参数是否为未初始化的张量混合类型
def is_lazy(param):
    return isinstance(param, UninitializedTensorMixin)


class UninitializedParameter(UninitializedTensorMixin, Parameter):
    r"""A parameter that is not initialized.

    Uninitialized Parameters are a a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = Parameter  # 指定这个类的默认类是 torch.nn.Parameter

    def __new__(cls, requires_grad=True, device=None, dtype=None) -> None:
        # 创建一个空张量，用于表示未初始化的参数
        factory_kwargs = {"device": device, "dtype": dtype}
        data = torch.empty(0, **factory_kwargs)
        # 返回一个未初始化的张量混合类型的实例
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            # 深度拷贝实现，用于复制未初始化参数的实例
            result = type(self)(self.requires_grad, self.data.device, self.data.dtype)
            memo[id(self)] = result
            return result


class UninitializedBuffer(UninitializedTensorMixin, torch.Tensor):
    r"""A buffer that is not initialized.

    Uninitialized Buffer is a a special case of :class:`torch.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = torch.Tensor  # 指定这个类的默认类是 torch.Tensor

    def __new__(cls, requires_grad=False, device=None, dtype=None) -> None:
        # 创建一个空张量，用于表示未初始化的缓冲区
        factory_kwargs = {"device": device, "dtype": dtype}
        data = torch.empty(0, **factory_kwargs)
        # 返回一个未初始化的张量混合类型的实例
        return torch.Tensor._make_subclass(cls, data, requires_grad)
```