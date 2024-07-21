# `.\pytorch\torch\_library\custom_ops.py`

```py
# mypy: allow-untyped-defs
# 引入 inspect 模块，用于获取函数签名信息
import inspect
# 引入 weakref 模块，用于创建弱引用对象
import weakref
# 引入类型提示模块
from typing import (
    Any,            # 任意类型
    Callable,       # 可调用对象类型
    Dict,           # 字典类型
    Iterable,       # 可迭代对象类型
    Iterator,       # 迭代器类型
    List,           # 列表类型
    Optional,       # 可选类型
    Sequence,       # 序列类型
    Tuple,          # 元组类型
    Union,          # 联合类型
)

# 从 torch.utils._exposed_in 模块中导入 exposed_in 函数
from torch.utils._exposed_in import exposed_in

# 从当前包的上级包中导入 _C, _library, _ops, autograd, library, Tensor 对象
from .. import _C, _library, _ops, autograd, library, Tensor
# 从当前包的子模块中导入 utils 模块
from . import utils

# 定义设备类型的类型别名
device_types_t = Optional[Union[str, Sequence[str]]]


@exposed_in("torch.library")
# 定义 custom_op 函数，用于创建自定义操作符
def custom_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Iterable[str],  # 传递给函数的参数名称列表，该函数会修改这些参数
    device_types: device_types_t = None,  # 函数适用的设备类型
    schema: Optional[str] = None,  # 操作符的模式字符串，用于描述操作符的签名
) -> Callable:
    """Wraps a function into custom operator.

    Reasons why you may want to create a custom op include:
    - Wrapping a third-party library or custom kernel to work with PyTorch
    subsystems like Autograd.
    - Preventing torch.compile/export/FX tracing from peeking inside your function.

    This API is used as a decorator around a function (please see examples).
    The provided function must have type hints; these are needed to interface
    with PyTorch's various subsystems.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_linear". The name is used as the op's stable identifier
            in PyTorch subsystems (e.g. torch.export, FX graphs).
            To avoid name collisions, please use your project name as the namespace;
            e.g. all custom ops in pytorch/fbgemm use "fbgemm" as the namespace.
        mutates_args (Iterable[str]): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined.
        device_types (None | str | Sequence[str]): The device type(s) the function
            is valid for. If no device type is provided, then the function
            is used as the default implementation for all device types.
            Examples: "cpu", "cuda".
        schema (None | str): A schema string for the operator. If None
            (recommended) we'll infer a schema for the operator from its type
            annotations. We recommend letting us infer a schema unless you
            have a specific reason not to.
            Example: "(Tensor x, int y) -> (Tensor, Tensor)".

    .. note::
        We recommend not passing in a ``schema`` arg and instead letting us infer
        it from the type annotations. It is error-prone to write your own schema.
        You may wish to provide your own schema if our interpretation of
        the type annotation is not what you want.
        For more info on how to write a schema string, see
        `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func>`_
    """
    pass
    def register_custom_op(name: str, fn=None, *, schema=None, mutates_args=()):
        """
        Register a custom operation with the specified name and function.
    
        Args:
            name (str): The name of the custom operation in the format 'namespace::opname'.
            fn (Callable, optional): The function implementing the custom operation. Defaults to None.
            schema (str, optional): The schema string describing the operation's behavior. Defaults to None.
            mutates_args (tuple, optional): Arguments mutated by the operation. Defaults to an empty tuple.
    
        Returns:
            CustomOpDef: An instance representing the registered custom operation.
    
        Examples::
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import custom_op
            >>> import numpy as np
            >>>
            >>> @custom_op("mylib::numpy_sin", mutates_args=())
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> x = torch.randn(3)
            >>> y = numpy_sin(x)
            >>> assert torch.allclose(y, x.sin())
            >>>
            >>> # Example of a custom op that only works for one device type.
            >>> @custom_op("mylib::numpy_sin_cpu", mutates_args=(), device_types="cpu")
            >>> def numpy_sin_cpu(x: Tensor) -> Tensor:
            >>>     x_np = x.numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np)
            >>>
            >>> x = torch.randn(3)
            >>> y = numpy_sin_cpu(x)
            >>> assert torch.allclose(y, x.sin())
            >>>
            >>> # Example of a custom op that mutates an input
            >>> @custom_op("mylib::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu")
            >>> def numpy_sin_inplace(x: Tensor) -> None:
            >>>     x_np = x.numpy()
            >>>     np.sin(x_np, out=x_np)
            >>>
            >>> x = torch.randn(3)
            >>> expected = x.sin()
            >>> numpy_sin_inplace(x)
            >>> assert torch.allclose(x, expected)
    
        """
    
        def inner(fn):
            import torch
    
            # Infer schema if not provided
            if schema is None:
                import torch._custom_op.impl
                schema_str = torch._custom_op.impl.infer_schema(fn, mutates_args)
            else:
                schema_str = schema
            
            # Extract namespace and operation name
            namespace, opname = name.split("::")
            
            # Create a CustomOpDef object with the given parameters
            result = CustomOpDef(namespace, opname, schema_str, fn)
    
            # Validate schema's alias annotations against mutates_args
            if schema is not None:
                expected = set()
                for arg in result._opoverload._schema.arguments:
                    if arg.alias_info is not None and arg.alias_info.is_write:
                        expected.add(arg.name)
                if expected != set(mutates_args):
                    raise ValueError(
                        f"Attempted to create a custom op with `mutates_args={mutates_args}` "
                        f"and `schema={schema}. The schema suggests that the op mutates {expected}"
                        f"which is different from what was provided to us in `mutates_args`. "
                        f"Please make these consistent."
                    )
            
            # Register the kernel for the specified device_types
            result.register_kernel(device_types)(fn)
            return result
    
        # If fn is None, return the inner function for deferred registration
        if fn is None:
            return inner
        return inner(fn)
class CustomOpDef:
    """CustomOpDef is a wrapper around a function that turns it into a custom op.

    It has various methods for registering additional behavior for this
    custom op.

    You should not instantiate CustomOpDef directly; instead, use the
    :func:`torch.library.custom_op` API.
    """

    def __init__(self, namespace: str, name: str, schema: str, fn: Callable) -> None:
        # Fields used to interface with the PyTorch dispatcher
        # 设置命名空间（namespace）、操作名称（name）、模式（schema）和初始化函数（fn）
        self._namespace = namespace
        self._name = name
        self._schema = schema

        # 初始化函数
        self._init_fn = fn

        # 用于存储后端函数的字典和其他可选函数的初始化
        self._backend_fns: Dict[Union[str, None], Callable] = {}
        self._abstract_fn: Optional[Callable] = None
        self._setup_context_fn: Optional[Callable] = None
        self._backward_fn: Optional[Callable] = None

        # 获取允许重写的库对象，并注册到分发器中
        self._lib = get_library_allowing_overwrite(self._namespace, self._name)
        self._register_to_dispatcher()
        
        # 将当前实例添加到全局操作定义字典中
        OPDEFS[self._qualname] = self

    @property
    def _qualname(self) -> str:
        # 返回操作的限定名称，格式为 "<namespace>::<name>"
        return f"{self._namespace}::{self._name}"

    def __repr__(self) -> str:
        # 返回自定义操作定义对象的字符串表示形式
        return f"<CustomOpDef({self._qualname})>"

    def register_kernel(
        self, device_types: device_types_t, fn: Optional[Callable] = None, /
    def register_autograd(
        self,
        backward: Callable,
        /,
        *,
        setup_context: Optional[Callable] = None,
    # 注册自定义操作到调度器
    def _register_to_dispatcher(self) -> None:
        # 获取私有属性 _lib
        lib = self._lib
        # 将操作名称和模式串拼接成字符串表示的模式
        schema_str = self._name + self._schema
        # 解析模式字符串为 C++ schema 对象
        cpp_schema = _C.parse_schema(schema_str)
        
        # 检查模式中是否存在仅限关键字参数的张量
        if utils.has_kwarg_only_tensors(cpp_schema):
            # 如果需要支持此功能，应逐步实现以下功能：
            # - 支持非可微的仅关键字参数张量
            # - 支持所有类型的仅关键字参数张量
            raise NotImplementedError(
                f"custom_op with kwarg-only Tensor args. Please make your "
                f"tensors not kwarg-only. Got: {schema_str}"
            )
        
        # 在库中定义操作，包括一些标签
        lib.define(
            schema_str,
            tags=[_C.Tag.pt2_compliant_tag, _C.Tag.needs_fixed_stride_order],
        )
        
        # 查找操作重载实现
        self._opoverload = _library.utils.lookup_op(self._qualname)
        
        # 定义一个虚拟实现函数
        def fake_impl(*args, **kwargs):
            # 如果没有抽象函数注册，尝试生成一个简单的虚拟实现
            if self._abstract_fn is None:
                if _library.utils.can_generate_trivial_fake_impl(self._opoverload):
                    return None
                # 抛出运行时错误，要求注册一个虚拟实现
                raise RuntimeError(
                    f"There was no fake impl registered for {self}. "
                    f"This is necessary for torch.compile/export/fx tracing to work. "
                    f"Please use `{self._init_fn.__name__}.register_fake` to add an "
                    f"fake impl."
                )
            # 调用抽象函数处理输入参数
            return self._abstract_fn(*args, **kwargs)
        
        # 在库中注册虚拟实现
        lib._register_fake(self._name, fake_impl, _stacklevel=4)
        
        # 创建自动求导实现
        autograd_impl = _library.autograd.make_autograd_impl(self._opoverload, self)
        # 在库中注册自动求导实现
        lib.impl(self._name, autograd_impl, "Autograd", with_keyset=True)
        
        # 获取操作重载的模式对象
        schema = self._opoverload._schema
        
        # 如果模式对象是可变的
        if schema.is_mutable:
            
            # 定义就地或视图自动求导实现
            def adinplaceorview_impl(keyset, *args, **kwargs):
                # 遍历模式参数和输入参数，处理就地更新的情况
                for arg, val in _library.utils.zip_schema(schema, args, kwargs):
                    if not arg.alias_info:
                        continue
                    if not arg.alias_info.is_write:
                        continue
                    if isinstance(val, Tensor):
                        autograd.graph.increment_version(val)
                    elif isinstance(val, (tuple, list)):
                        for v in val:
                            if isinstance(v, Tensor):
                                autograd.graph.increment_version(v)
                # 执行自动分发，确保在就地或视图操作下的自动分发处理
                with _C._AutoDispatchBelowADInplaceOrView():
                    return self._opoverload.redispatch(
                        keyset & _C._after_ADInplaceOrView_keyset, *args, **kwargs
                    )
            
            # 在库中注册就地或视图自动求导实现
            lib.impl(
                self._name,
                adinplaceorview_impl,
                "ADInplaceOrView",
                with_keyset=True,
            )

    # 实现调用操作的方法
    def __call__(self, *args, **kwargs):
        # 调用操作重载对象处理传入的参数
        return self._opoverload(*args, **kwargs)
# NOTE: [Supporting decorator and non-decorator usage]
#
# Some APIs may be both used as a decorator and not as a decorator.
# For example:
#
# >>> def fn(x):
# >>>     return x.sin()
# >>>
# >>> # Usage 1: not as a decorator
# >>> numpy_sin.register_kernel("cuda", fn)
# >>>
# >>> # Usage 2: as a decorator
# >>> @numpy_sin.register_kernel("cuda")
# >>> def fn2(x):
# >>>     return x.sin
#
# The way we support this is that `register_kernel` accepts an optional `fn`.
# If `fn` is provided (Usage 1), then we know that the user is using it not
# as a decorator.
# If `fn` is not provided (Usage 2), then `register_kernel` needs to return a
# decorator.

# 字典，用于存储操作定义到库的映射关系
OPDEF_TO_LIB: Dict[str, "library.Library"] = {}
# 弱引用字典，用于存储操作定义
OPDEFS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

# 获取允许覆盖的库
def get_library_allowing_overwrite(namespace: str, name: str) -> "library.Library":
    qualname = f"{namespace}::{name}"

    # 如果操作定义已存在，则销毁并删除
    if qualname in OPDEF_TO_LIB:
        OPDEF_TO_LIB[qualname]._destroy()
        del OPDEF_TO_LIB[qualname]

    # 创建新的库对象
    lib = library.Library(namespace, "FRAGMENT")
    OPDEF_TO_LIB[qualname] = lib
    return lib

# 迭代器，用于遍历张量
def iter_tensors(
    args: Tuple[Any], kwargs: Dict[str, Any], allowed_nesting: int = 1
) -> Iterator[Tensor]:
    # 检查参数是否为张量，支持嵌套
    def check(arg):
        if isinstance(arg, Tensor):
            yield arg
        elif allowed_nesting > 0 and isinstance(arg, (tuple, list)):
            yield from iter_tensors(tuple(arg), {}, allowed_nesting - 1)

    # 遍历参数列表
    for arg in args:
        yield from check(arg)
    # 遍历关键字参数值
    for kwarg in kwargs.values():
        yield from check(kwarg)

# 获取操作定义，支持自定义操作和操作重载
def _maybe_get_opdef(
    op: Union[CustomOpDef, _ops.OpOverload, str]
) -> Optional[CustomOpDef]:
    if isinstance(op, CustomOpDef):
        return op
    if isinstance(op, _ops.OpOverload):
        op = op._name
    assert isinstance(op, str)
    # 如果操作定义存在，则返回
    if op in OPDEFS:
        return OPDEFS[op]
    return None
```