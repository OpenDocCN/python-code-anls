# `.\pytorch\torch\_custom_op\impl.py`

```py
# 设置 mypy 来允许未类型化的定义
mypy: allow-untyped-defs

# 导入必要的模块
import dataclasses  # 用于数据类的装饰器
import functools   # 用于创建高阶函数和可缓存的装饰器
import inspect     # 提供了检查源码的工具
import sys         # 提供了与 Python 解释器交互的变量和函数
import typing      # 用于类型提示
import weakref     # 提供了弱引用对象的支持

# 导入 torch 相关模块和类
from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy  # 从 torchgen.model 模块中导入多个类

import torch         # 导入 PyTorch 核心库
import torch._C as _C  # 导入 PyTorch C++ API
import torch.library as library  # 导入 torch.library 模块
from torch.library import get_ctx  # 从 torch.library 模块导入 get_ctx 函数

# 导入自定义的 autograd 相关模块和函数
from .autograd import autograd_kernel_indirection, construct_autograd_kernel

# 导入 torch._library.infer_schema 模块和 infer_schema 函数
import torch._library.infer_schema
from torch._library.infer_schema import infer_schema

"""
用于自定义操作的详细指南，请参见
https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

这个文件包含了我们自定义操作 API 的部分实现。
"""

# 导出的符号列表
__all__ = ["custom_op", "CustomOp", "get_ctx"]

# 支持的设备类型与关键字的映射
SUPPORTED_DEVICE_TYPE_TO_KEY = {
    "cpu": "CPU",
    "cuda": "CUDA",
}

# 预留的命名空间集合，用于避免与 PyTorch 内部实现混淆
RESERVED_NS = {
    "prim",
    "prims",
    "aten",
    "at",
    "torch",
    "pytorch",
}

def custom_op(
    qualname: str, manual_schema: typing.Optional[str] = None
) -> typing.Callable:
    r"""Creates a new CustomOp object.

    警告：如果你是用户，请不要直接使用此函数
    （请使用 torch._custom_ops API）。
    详细的自定义操作指南请参见以下链接：
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    在 PyTorch 中，定义一个操作（简称 op）是一个两步过程：
    - 首先需要定义（创建）这个 op
    - 然后需要实现操作符如何与各种 PyTorch 子系统（如 CPU/CUDA 张量，Autograd 等）交互的行为

    这个函数定义了 CustomOp 对象（第一步）；
    你需要在 CustomOp 对象上调用各种方法来执行第二步操作。

    这是一个装饰器 API（详见示例）。

    参数:
        qualname (str): 应该是一个看起来像 "namespace::operator_name" 的字符串。
            在 PyTorch 中，操作符需要一个命名空间来避免名称冲突；
            一个给定的操作符只能被创建一次。
            如果你在写一个 Python 库，建议命名空间使用你顶层模块的名称。
            操作符的名字必须与你传递给 custom_op 的函数的名字相同（详见示例）。
        manual_schema (Optional[str]): 每个 PyTorch 操作符都需要一个模式，告诉 PyTorch 输入/输出的类型。
            如果为 None（默认），我们将根据函数的类型注解来推断模式（详见示例）。
            否则，如果你不想使用类型注解，可以提供模式字符串。

    Returns:
        Callable: 返回一个可调用对象，通常是一个函数或方法。
    # 定义装饰器函数 custom_op，用于创建自定义操作符
    def custom_op(qualname: str, *, manual_schema: Optional[str] = None) -> Callable[[Callable], CustomOp]:
        # 内部函数 inner 用于处理装饰的函数 func
        def inner(func):
            # 检查 func 是否为 Python 函数，如果不是则引发 ValueError 异常
            if not inspect.isfunction(func):
                raise ValueError(
                    f"custom_op(...)(func): Expected `func` to be a Python "
                    f"function, got: {type(func)}"
                )
    
            # 解析限定名称 qualname 的命名空间和名称
            ns, name = parse_qualname(qualname)
            # 验证命名空间的有效性
            validate_namespace(ns)
            # 检查 func 的名称是否与解析出的名称相符，不符则引发 ValueError 异常
            if func.__name__ != name:
                raise ValueError(
                    f"custom_op(qualname='{qualname}', ...)(func): expected `func` "
                    f"to have name '{name}' but got '{func.__name__}'. "
                    f"Please either change the name of `func` or the qualname that "
                    f"is passed to `custom_op`"
                )
    
            # 推断函数的 schema 或者使用手动提供的 schema
            schema = infer_schema(func) if manual_schema is None else manual_schema
            # 根据 schema 构建函数的 schema 字符串
            schema_str = f"{name}{schema}"
            # 解析并验证函数的 schema
            function_schema = FunctionSchema.parse(schema_str)
            validate_schema(function_schema)
            # 如果手动提供了 schema，则验证函数是否符合该 schema
            if manual_schema is not None:
                validate_function_matches_schema(function_schema, func)
    
            # 创建具有命名空间 ns 和 schema 的库对象 lib
            lib = library.Library(ns, "FRAGMENT")
            # 在库中定义函数的 schema_str
            lib.define(schema_str)
            # 查找或抛出与函数名 function_schema.name 对应的操作句柄 ophandle
            ophandle = find_ophandle_or_throw(ns, function_schema.name)
            # 创建 CustomOp 对象 result，使用库 lib、命名空间 ns、函数 schema、名称 name、操作句柄 ophandle
            result = CustomOp(lib, ns, function_schema, name, ophandle, _private_access=True)
    
            # 将 result 对象的名称、模块和文档字符串设置为与 func 相同
            result.__name__ = func.__name__
            result.__module__ = func.__module__
            result.__doc__ = func.__doc__
    
            # 在库 lib 中实现 result 对象的操作名 _opname 的 Autograd
            library.impl(lib, result._opname, "Autograd")(
                autograd_kernel_indirection(weakref.proxy(result))
            )
    
            # 设置 Torch 的调度错误回调函数，使用 ophandle 和 report_error_callback 处理函数
            torch._C._dispatch_set_report_error_callback(
                ophandle, functools.partial(report_error_callback, weakref.proxy(result))
            )
    
            # 返回创建的 CustomOp 对象 result
            return result
    
        # 返回内部函数 inner，用于处理被装饰的函数
        return inner
# 全局字典，保存所有 CustomOp 对象的引用
# 这个字典保持了所有 CustomOp 的存在（参见 NOTE [CustomOp lifetime]）
# 用于查询与特定 C++ 调度操作符相关联的 CustomOp 对象
# 一个示例用法是 FakeTensor：FakeTensor 通过 CustomOp API 检查特定操作符是否注册了实现。
# 字典的键是操作符的限定名（例如 aten::foo）
global_registry: typing.Dict[str, "CustomOp"] = {}

# CustomOp 类，用于定义 PyTorch 中的自定义操作符
class CustomOp:
    r"""Class for custom operators in PyTorch.

    Use the CustomOp API to create user-defined custom operators that behave
    just like regular PyTorch operators (e.g. torch.sin, torch.mm) when it
    comes to various PyTorch subsystems (like torch.compile).

    To construct a `CustomOp`, use `custom_op`.
    """

    def __init__(self, lib, cpp_ns, schema, operator_name, ophandle, *, _private_access=False):
        super().__init__()
        # 如果不是私有访问，则抛出运行时错误
        if not _private_access:
            raise RuntimeError(
                "The CustomOp constructor is private and we do not guarantee "
                "BC for it. Please use custom_op(...) to create a CustomOp object"
            )
        # 构造限定名，例如 "cpp_ns::operator_name"
        name = f"{cpp_ns}::{operator_name}"
        self._schema = schema
        self._cpp_ns = cpp_ns
        self._lib: library.Library = lib
        self._ophandle: _C._DispatchOperatorHandle = ophandle
        # 操作符的名称，例如 "foo"，这里为方便起见进行缓存
        self._opname: str = operator_name
        # 这是 _opname，但带有命名空间，例如 "custom::foo"
        self._qualname: str = name
        self.__name__ = None  # mypy 要求这样做
        # 一些实现注册为 DispatchKeys 的内核，直接修改 _impls 字典不会对其产生影响
        self._impls: typing.Dict[str, typing.Optional[FuncAndLocation]] = {}
        # 参见 NOTE [CustomOp autograd kernel indirection]
        self._registered_autograd_kernel_indirection = False

        # 将自身注册到全局注册表 global_registry 中
        global_registry[self._qualname] = self

    # 注册 autograd kernel 的间接调用
    def _register_autograd_kernel_indirection(self):
        assert not self._registered_autograd_kernel_indirection
        # 使用 lib.impl 注册 autograd kernel，使用 weakref.proxy(self) 作为参数
        self._lib.impl(self._opname, autograd_kernel_indirection(weakref.proxy(self)), "Autograd")
        self._registered_autograd_kernel_indirection = True

    # 记录实现和源位置到 self._impls 中
    # 注意，这并不会导致 torch.library 使用该实现，需要在单独的 self._lib.impl 调用中完成
    def _register_impl(self, kind, func, stacklevel=2):
        # 检查是否已经注册了指定类型的实现
        if self._has_impl(kind):
            # 如果已经注册，则抛出运行时错误，说明不支持重复注册
            func_and_location = self._impls[kind]
            assert func_and_location is not None  # Pacify mypy
            location = func_and_location.location
            raise RuntimeError(
                f"Attempting to register a {kind} impl for operator {self._qualname} "
                f"that already has a {kind} impl registered from Python at "
                f"{location}. This is not supported."
            )
        # 获取当前调用栈信息
        frame = inspect.getframeinfo(sys._getframe(stacklevel))
        location = f"{frame.filename}:{frame.lineno}"
        # 将函数及其位置信息注册到 _impls 字典中
        self._impls[kind] = FuncAndLocation(func, location)

    def _get_impl(self, kind):
        # 返回给定类型的实现函数及其位置信息
        return self._impls[kind]

    def _has_impl(self, kind):
        # 检查是否存在给定类型的实现
        return kind in self._impls

    def _destroy(self):
        # NOTE: [CustomOp lifetime]
        # 销毁 CustomOp 对象的方法。虽然删除了一些属性，但对象仍保留在全局注册表中。
        del self._lib

        opnamespace = getattr(torch.ops, self._cpp_ns)
        # 删除 torch.ops 下对应的操作符名称属性
        if hasattr(opnamespace, self._opname):
            delattr(opnamespace, self._opname)

        # 从全局注册表中删除当前对象的资格名称
        del global_registry[self._qualname]

    def __repr__(self):
        # 返回 CustomOp 对象的字符串表示形式，包含操作符的资格名称
        return f'<CustomOp(op="{self._qualname}")>'

    def __call__(self, *args, **kwargs):
        # 绕过 torch.ops.* 直接调用 OperatorHandle::callBoxed 方法来执行操作
        # 使用 torch.ops.* 有时会比较麻烦（可能会慢，并且由于缓存操作符的生命周期问题，使得测试 CustomOp 变得困难）
        result = _C._dispatch_call_boxed(self._ophandle, *args, **kwargs)
        return result

    def impl(
        self, device_types: typing.Union[str, typing.Iterable[str]], _stacklevel=2,
    ) -> typing.Callable:
        r"""Register an implementation for a device type for this CustomOp object.

        WARNING: if you're a user, please do not use this directly
        (instead use the torch._custom_ops APIs).
        Also please see the following for a detailed guide on custom ops.
        https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

        If the CustomOp is passed multiple Tensor inputs with different device
        types, it will dispatch to the registered implementation for the highest
        priority device type among those present.
        The supported device types, in order of priority, are {'cuda', 'cpu'}.

        This API is used as a decorator (see examples).

        Arguments:
            device_types (str or Iterable[str]): the device type(s) to register the function for.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> @custom_op("my_library::numpy_cos")
            >>> def numpy_cos(x: Tensor) -> Tensor:
            >>>     ...

            >>> # Register an implementation for CPU Tensors
            >>> @numpy_cos.impl('cpu')
            >>> def numpy_cos_impl_cpu(x):
            >>>     return torch.from_numpy(np.cos(x.numpy()))

            >>> # Register an implementation for CUDA Tensors
            >>> @numpy_cos.impl('cuda')
            >>> def numpy_cos_impl_cuda(x):
            >>>     return torch.from_numpy(np.cos(x.cpu().numpy())).to(x.device)

            >>> x = torch.randn(3)
            >>> numpy_cos(x)  # calls numpy_cos_impl_cpu

            >>> x_cuda = x.cuda()
            >>> numpy_cos(x)  # calls numpy_cos_impl_cuda

        """
        # 将实现注册为特定设备类型的装饰器函数
        if isinstance(device_types, str):
            device_types = [device_types]
        # 验证设备类型的有效性
        for device_type in device_types:
            validate_device_type(device_type)

        def inner(f):
            # 遍历设备类型集合，确保自定义操作未注册相同设备类型的实现
            for device_type in set(device_types):
                self._check_doesnt_have_library_impl(device_type)
                # 注册实现到CustomOp对象中
                self._register_impl(device_type, f, stacklevel=_stacklevel)
                # 使用torch.library机制将实现注册到库中
                dispatch_key = SUPPORTED_DEVICE_TYPE_TO_KEY[device_type]
                library.impl(self._lib, self._opname, dispatch_key)(f)
            return f

        return inner

    def _check_doesnt_have_library_impl(self, device_type):
        # 检查是否已经注册过特定设备类型的实现，避免重复注册
        if self._has_impl(device_type):
            return
        # 检查是否已经存在针对该设备类型的预定义实现
        key = SUPPORTED_DEVICE_TYPE_TO_KEY[device_type]
        if _C._dispatch_has_computed_kernel_for_dispatch_key(self._qualname, key):
            raise RuntimeError(
                f"impl(..., device_types={device_type}): the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing torch.library or TORCH_LIBRARY registration.")
    def impl_factory(self) -> typing.Callable:
        r"""Register an implementation for a factory function."""
        # 定义一个装饰器函数 inner，用于注册工厂函数的实现
        def inner(f):
            # 调用对象的 _register_impl 方法，注册工厂函数实现
            self._register_impl("factory", f)
            # 使用 library.impl 函数注册工厂函数 f 到指定的库和操作名，使用 "BackendSelect" 策略
            library.impl(self._lib, self._opname, "BackendSelect")(f)
            # 返回原始的函数 f
            return f

        # 返回装饰器函数 inner，用于注册工厂函数的实现
        return inner

    def _check_can_register_backward(self):
        # 定义一个内部函数 error，用于抛出运行时错误，指明无法注册反向传播公式的具体原因
        def error(detail):
            raise RuntimeError(
                f"Cannot use torch._custom_ops APIs to register backward "
                f"formula for {detail}. Got operator "
                f"{self._qualname} with schema: {schema}"
            )

        # 获取对象的 schema 属性作为本地变量
        schema = self._schema
        # 检查操作的类型，如果不是 functional 类型，则调用 error 函数抛出错误
        if schema.kind() != SchemaKind.functional:
            error("non-functional operator")

        # 获取操作的返回值列表
        rets = schema.returns
        # 如果返回值列表为空，则调用 error 函数抛出错误
        if not schema.returns:
            error("operator with no returns")

        # 断言返回值列表长度大于 0
        assert len(rets) > 0
        # 检查返回值中是否有非变异视图的情况
        is_non_mutating_view = any(
            r.annotation is not None and not r.annotation.is_write for r in rets
        )
        # 如果存在非变异视图的情况，则调用 error 函数抛出错误
        if is_non_mutating_view:
            error("operator that returns views")

        # 定义允许的返回值类型及其对应的字符串表示
        allowed_return_types = {
            BaseType(BaseTy.int): "int",
            BaseType(BaseTy.SymInt): "SymInt",
            BaseType(BaseTy.bool): "bool",
            BaseType(BaseTy.float): "float",
            BaseType(BaseTy.Tensor): "Tensor",
            ListType(BaseType(BaseTy.Tensor), None): "List[Tensor]",
        }
        # 遍历操作的返回值列表，检查是否符合允许的返回值类型
        for ret in schema.returns:
            if ret.type in allowed_return_types:
                continue
            # 如果返回值类型不在允许的类型集合中，则调用 error 函数抛出错误
            error(f"operator with return not in {list(allowed_return_types.values())} (got {ret.type})")
    # 检查是否已经注册了 Autograd 核心间接引用
    def _check_doesnt_have_library_autograd_impl(self):
        # 如果已经注册了 Autograd 核心间接引用，则直接返回
        if self._registered_autograd_kernel_indirection:
            return

        # 检查是否已经为 DispatchKey::CompositeImplicitAutograd 注册了操作
        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "CompositeImplicitAutograd"):
            # 如果是，则抛出运行时错误，指出不需要自动求导公式
            raise RuntimeError(
                f"impl_backward/impl_save_for_backward: the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing registration to DispatchKey::CompositeImplicitAutograd."
                f"CompositeImplicitAutograd operators do not need an autograd formula; "
                f"instead, the operator will decompose into its constituents and those "
                f"can have autograd formulas defined on them.")

        # 检查是否已经为某些 Autograd<BACKEND> 键注册了 Autograd 核心
        # 目前主要考虑 CPU 和 CUDA
        for key in ["Autograd", "AutogradCPU", "AutogradCUDA"]:
            if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, key):
                # 如果是，则抛出运行时错误，建议移除对应的 torch.library 或 TORCH_LIBRARY 注册
                raise RuntimeError(
                    f"impl_backward/impl_save_for_backward: "
                    f"the operator {self._qualname} already has an Autograd kernel "
                    f"registered to DispatchKey::{key} via a pre-existing "
                    f"torch.library or TORCH_LIBRARY registration. Please either "
                    f"remove those registrations or don't use the torch._custom_ops APIs")
    def _check_doesnt_have_library_meta_impl(self):
        # 如果已经实现了 "abstract"，则直接返回，不需要再处理
        if self._has_impl("abstract"):
            return

        # 如果用户的操作符是 CompositeExplicitAutograd，
        # 允许它们实现 "abstract"。这是一个实用性的考虑，
        # 因为现有的自定义操作可能有 CompositeExplicitAutograd 的注册，
        # 但这些注册可能不适用于 Meta 内核，所以这里给出一个逃生口。
        if (
            _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "CompositeExplicitAutograd")
            and not _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "Meta")
        ):
            return

        # 否则，如果用户已经有一个 Meta 内核或者他们的操作是 CompositeImplicitAutograd 或其他别名的 dispatch key，
        # 则抛出异常。

        # 对于 CompositeImplicitAutograd 的特殊情况
        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "CompositeImplicitAutograd"):
            raise RuntimeError(
                f"impl_abstract(...): 操作符 {self._qualname} "
                f"已经通过 DispatchKey::CompositeImplicitAutograd 的预先存在的注册 "
                f"拥有了此设备类型的实现。CompositeImplicitAutograd 操作符不需要抽象实现；"
                f"相反，该操作符将分解为其组成部分，这些部分可以在它们上面定义抽象实现。"
            )

        # 如果操作符已经有了 DispatchKey::Meta 的实现，则抛出异常
        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "Meta"):
            raise RuntimeError(
                f"impl_abstract(...): 操作符 {self._qualname} "
                f"已经通过 torch.library 或 TORCH_LIBRARY 的预先存在注册 "
                f"拥有了 DispatchKey::Meta 的实现。"
                f"请删除该注册或不要调用 impl_abstract。"
            )

    # 注册 autograd 内核函数
    # ["backward", "save_for_backward", and "autograd"]
    # 作为显式 autograd API 的一部分，用户必须提供 "save_for_backward" 和 "backward" 函数。
    # 当两者都提供时，我们会自动构建 "autograd" 内核。
    def _register_autograd_kernel(self):
        # 断言已经实现了 "backward" 和 "save_for_backward"
        assert self._has_impl("backward")
        assert self._has_impl("save_for_backward")
        
        # 构建 autograd 内核
        kernel = construct_autograd_kernel(
            self._schema,
            self._output_differentiability,
            self,
            get_op(self._qualname),
            self._get_impl("save_for_backward").func,
            self._get_impl("backward").func
        )
        
        # 注册 "autograd" 内核
        self._register_impl("autograd", kernel)
    def impl_save_for_backward(self, _stacklevel=2):
        r"""Register a function that tells us what to save for backward.

        Please see impl_backward for more details.
        """
        # 定义内部函数 inner，用于注册反向传播需要保存的函数
        def inner(f):
            # 检查是否可以注册反向传播函数
            self._check_can_register_backward()
            # 检查是否没有库自动求导实现
            self._check_doesnt_have_library_autograd_impl()
            # 如果还未注册自动求导内核间接引用，则注册它
            if not self._registered_autograd_kernel_indirection:
                self._register_autograd_kernel_indirection()
            # 注册实现函数到保存反向传播函数的列表中
            self._register_impl("save_for_backward", f, stacklevel=_stacklevel)
            # 如果已经有了 backward 实现，则注册自动求导内核
            if self._has_impl("backward"):
                self._register_autograd_kernel()
        # 返回内部函数 inner，用于注册反向传播需要保存的函数
        return inner
# 使用 dataclasses 模块定义一个名为 FuncAndLocation 的数据类，包含 func 和 location 两个字段
@dataclasses.dataclass
class FuncAndLocation:
    func: typing.Callable  # func 字段是一个可调用对象
    location: str  # location 字段是一个字符串


# 定义一个名为 find_ophandle_or_throw 的函数，接受两个参数：cpp_ns（字符串）和 operator_name（OperatorName 类型）
def find_ophandle_or_throw(cpp_ns: str, operator_name: OperatorName):
    # 根据 operator_name 的 overload_name 属性确定 overload_name 的值，如果为 None 则为空字符串
    overload_name = (
        "" if operator_name.overload_name is None else operator_name.overload_name
    )
    # 调用 _C._dispatch_find_schema_or_throw 方法查找并返回符合条件的模式或抛出异常
    return _C._dispatch_find_schema_or_throw(
        f"{cpp_ns}::{str(operator_name.name)}", overload_name
    )


# 定义一个名为 validate_namespace 的函数，用于验证命名空间字符串 ns 的合法性
def validate_namespace(ns: str) -> None:
    # 如果命名空间中包含 '.'，抛出 ValueError 异常，提示命名空间不应包含 '.' 且必须是有效的变量名
    if "." in ns:
        raise ValueError(
            f'custom_op(..., ns="{ns}"): expected ns to not contain any . (and be a '
            f"valid variable name)"
        )
    # 如果命名空间在保留的命名空间列表 RESERVED_NS 中，抛出 ValueError 异常，提示命名空间已被保留
    if ns in RESERVED_NS:
        raise ValueError(
            f"custom_op(..., ns='{ns}'): '{ns}' is a reserved namespace, "
            f"please choose something else. "
        )


# 定义一个名为 validate_schema 的函数，用于验证给定的 FunctionSchema 对象 schema 是否为 functional schema
def validate_schema(schema: FunctionSchema) -> None:
    # 如果 schema 不是 functional schema，抛出 ValueError 异常，提示 custom_op 仅支持 functional operators
    if not torch._library.utils.is_functional_schema(schema):
        raise ValueError(
            f"custom_op only supports functional operators "
            f"(ops that do not mutate any inputs, do not return "
            f"views of the inputs, and has at least one return). "
            f"Got the following non-functional schema: {schema}"
        )

    # 简单起见，不允许 schema.arguments 中有 self 参数，如果有则抛出 ValueError 异常
    if schema.arguments.self_arg is not None:
        raise ValueError(
            f"custom_op does not support arguments named 'self'. Please "
            f"rename your argument. Got: {schema}"
        )


# 定义一个名为 parse_qualname 的函数，用于解析限定名称 qualname，并返回命名空间和名称的元组
def parse_qualname(qualname: str) -> typing.Tuple[str, str]:
    # 根据 '::' 分割 qualname 字符串，得到命名空间和名称组成的列表 names
    names = qualname.split("::", 1)
    # 如果 names 列表长度不为 2，抛出 ValueError 异常，提示期望在 qualname 中有命名空间
    if len(names) != 2:
        raise ValueError(f"Expected there to be a namespace in {qualname}, i.e. The "
                         f"operator name should look something like ns::foo")
    # 如果名称部分 names[1] 中包含 '.'，抛出 ValueError 异常，提示不支持带有 '.' 的操作符名称
    if '.' in names[1]:
        raise ValueError(f"The torch.custom_ops APIs do not handle overloads, "
                         f"i.e. operator names with '.' in them. "
                         f"Please name your operator something like ns::foo. "
                         f"Got: {qualname}")
    # 返回命名空间和名称的元组
    return names[0], names[1]


# 定义一个名为 validate_device_type 的函数，用于验证设备类型字符串 device_type 是否被支持
def validate_device_type(device_type: str) -> None:
    # 如果 device_type 不在支持的设备类型列表 SUPPORTED_DEVICE_TYPE_TO_KEY 中，抛出 ValueError 异常
    if device_type not in SUPPORTED_DEVICE_TYPE_TO_KEY:
        raise ValueError(
            f"CustomOp.impl(device_types=[{device_type}, ...]): we only support device_type "
            f"in {SUPPORTED_DEVICE_TYPE_TO_KEY.keys()}."
        )


# 定义一个名为 supported_param 的函数，用于判断 inspect.Parameter 对象 param 的类型是否为支持的类型
def supported_param(param: inspect.Parameter) -> bool:
    # 如果 param 的类型为 POSITIONAL_OR_KEYWORD 或者 KEYWORD_ONLY，则返回 True，否则返回 False
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


# 定义一个名为 validate_function_matches_schema 的函数，用于验证给定的函数 func 是否符合给定的 FunctionSchema schema
def validate_function_matches_schema(
    schema: FunctionSchema, func: typing.Callable
) -> None:
    # 获取函数 func 的参数签名
    sig = inspect.signature(func)
    # 检查所有函数参数是否都支持指定的支持函数
    if not all(supported_param(p) for _, p in sig.parameters.items()):
        # 如果有不支持的参数，则抛出值错误异常，包括函数签名信息
        raise ValueError(
            f"custom_op(..., manual_schema)(func): positional-only args, "
            f"varargs, and kwargs are not supported. Please rewrite `func` "
            f"to not have them. Got `func` with signature: {sig}"
        )

    # 检查函数参数是否有类型注解或返回类型注解
    if (
        any(
            p.annotation is not inspect.Parameter.empty
            for _, p in sig.parameters.items()
        )
        or sig.return_annotation is not inspect.Signature.empty
    ):
        # 如果有类型注解，则抛出值错误异常，要求函数不包含类型注解，同时包含函数签名信息
        raise ValueError(
            f"custom_op(..., manual_schema)(func): When passing in a manual "
            f"schema, we expect `func` to have no type annotations to avoid "
            f"ambiguity. Got `func` with signature: {sig}"
        )

    # 获取所有位置或关键字参数
    positional = [
        (name, param)
        for name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

    # 获取所有仅关键字参数
    kwargonly = [
        (name, param)
        for name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]

    # 定义错误函数，用于抛出关于函数签名不匹配的值错误异常
    def error():
        raise ValueError(
            f"custom_op(..., manual_schema)(func): When passing in a manual "
            f"schema, we expect `func`'s signature to match `manual_schema` "
            f"(aside from type annotations). "
            f"func's signature: {sig}, manual_schema: {schema}"
        )

    # 定义错误函数，用于抛出关于默认参数不匹配的值错误异常
    def error_default_args():
        raise ValueError(
            f"custom_op(..., manual_schema)(func): "
            f"neither func nor manual_schema should have default "
            f"arguments. Got "
            f"func's signature: {sig}, manual_schema: {schema}"
        )

    # 比较函数签名参数与手动模式的参数，检查它们是否匹配
    def compare(sig_args, schema_args):
        # 如果参数数量不匹配，则调用错误函数
        if len(sig_args) != len(schema_args):
            error()
        # 逐个比较参数名称和默认值，如有不匹配则调用错误函数
        for (name, param), arg in zip(sig_args, schema_args):
            if name != arg.name:
                error()
            if param.default is not inspect.Parameter.empty or arg.default is not None:
                error_default_args()

    # 比较所有位置或关键字参数与手动模式的平铺位置参数
    compare(positional, schema.arguments.flat_positional)
    # 比较所有仅关键字参数与手动模式的平铺关键字参数
    compare(kwargonly, schema.arguments.flat_kwarg_only)
# 定义一个函数，用于报告错误回调，接受自定义操作和一个关键字作为参数，不返回任何结果
def report_error_callback(custom_op: typing.Any, key: str) -> None:
    # 如果关键字为 "Undefined"，则抛出未实现错误，提示没有张量输入给该操作符
    if key == "Undefined":
        raise NotImplementedError(
            f"{custom_op}: There were no Tensor inputs to this operator "
            f"(e.g. you passed an empty list of Tensors). If your operator is a "
            f"factory function (that is, it takes no Tensors and constructs "
            f"a new one), then please use CustomOp.impl_factory to register "
            f"an implementation for it"
        )
    # 如果关键字为 "Meta"，则抛出未实现错误，提示在使用设备为 'Meta' 时没有为此自定义操作注册抽象实现
    if key == "Meta":
        raise NotImplementedError(
            f"{custom_op}: when running with device='Meta' tensors: there is no "
            f"abstract impl registered for this CustomOp. Please register one via "
            f"CustomOp.impl_abstract to get this CustomOp to work with Meta tensors"
        )
    # 如果关键字为 "CPU" 或 "CUDA"，则抛出未实现错误，提示在特定设备上没有为此自定义操作注册具体实现
    if key in ("CPU", "CUDA"):
        device = key.lower()
        raise NotImplementedError(
            f"{custom_op}: when running with device='{device}' tensors: there is no "
            f"{device} impl registered for this CustomOp. Please register one via "
            f"CustomOp.impl(device_type='{device}')"
        )
    # 如果以上条件都不满足，则抛出未实现错误，提示没有为此分发键注册实现
    raise NotImplementedError(
        f"{custom_op}: No implementation for dispatch key {key}. It is likely "
        f"that we have not added this functionality yet, please either open an "
        f"issue or if you're feeling adventurous, use the low-level "
        f"torch.library API"
    )


# 定义一个函数，用于根据现有的操作创建自定义操作，接受一个操作符对象作为参数，返回一个CustomOp对象
def custom_op_from_existing(op):
    # 获取操作的命名空间
    ns = op.namespace
    # 创建一个torch.library.Library对象，用于存储自定义操作的库信息
    lib = torch.library.Library(ns, "FRAGMENT")
    # 获取操作的名称（去除命名空间前缀）
    name = op.name().split("::")[-1]
    # 获取操作的模式字符串表示，并去除命名空间前缀
    schema_str = str(op._schema)
    schema_str = schema_str.split("::")[-1]
    # 解析操作的函数模式，返回FunctionSchema对象
    schema = FunctionSchema.parse(schema_str)
    # 返回一个CustomOp对象，封装了自定义操作的相关信息
    return CustomOp(lib, ns, schema, name, op, _private_access=True)


# 定义一个函数，用于获取指定操作符的默认包装
def get_op(qualname):
    # 定义一个函数，用于抛出值错误，指示找不到指定的操作符
    def error_not_found():
        raise ValueError(
            f"Could not find the operator {qualname}. Please make sure you have "
            f"already registered the operator and (if registered from C++) "
            f"loaded it via torch.ops.load_library.")

    # 解析操作符的命名空间和名称
    ns, name = parse_qualname(qualname)
    # 如果torch.ops中没有该命名空间，则抛出值错误
    if not hasattr(torch.ops, ns):
        error_not_found()
    # 获取命名空间对应的操作符命名空间对象
    opnamespace = getattr(torch.ops, ns)
    # 如果操作符命名空间对象中没有指定名称的操作符，则抛出值错误
    if not hasattr(opnamespace, name):
        error_not_found()
    # 获取操作符对象的默认包装，并返回
    packet = getattr(opnamespace, name)
    return packet.default


# 定义一个函数，用于查找指定的自定义操作符
def _find_custom_op(qualname, also_check_torch_library=False):
    # 如果在全局注册表中找到指定的自定义操作符，则返回其对应的注册信息
    if qualname in global_registry:
        return global_registry[qualname]
    # 如果不需要检查torch库，则抛出运行时错误，指示找不到指定的自定义操作符
    if not also_check_torch_library:
        raise RuntimeError(
            f'Could not find custom op "{qualname}". Did you register it via '
            f"the torch._custom_ops API?")
    # 获取指定操作符的默认包装
    overload = get_op(qualname)
    # 基于现有的操作包装，创建自定义操作并返回
    result = custom_op_from_existing(overload)
    return result


# 定义一个函数，用于获取指定自定义操作符的抽象实现
def get_abstract_impl(qualname):
    # 如果给定的函数名 qualname 不在 Torch 的全局自定义操作注册表中，则返回 None
    if qualname not in torch._custom_op.impl.global_registry:
        return None
    
    # 从 Torch 的全局自定义操作注册表中获取 qualname 对应的自定义操作对象
    custom_op = torch._custom_op.impl.global_registry[qualname]
    
    # 如果获取的自定义操作对象为空，则返回 None
    if custom_op is None:
        return None
    
    # 如果 custom_op 对象没有实现 "abstract" 方法，返回 None
    if not custom_op._has_impl("abstract"):
        return None
    
    # 返回 custom_op 对象中实现了 "abstract" 方法的函数对象
    return custom_op._get_impl("abstract").func
def _custom_op_with_schema(qualname, schema, needs_fixed_stride_order=True):
    # 将限定名称分割成命名空间和名称部分
    ns, name = qualname.split("::")
    # 构建函数模式字符串，结合名称和给定的模式
    schema_str = f"{name}{schema}"
    # 解析函数模式字符串为FunctionSchema对象
    function_schema = FunctionSchema.parse(schema_str)
    # 验证函数模式的合法性
    validate_schema(function_schema)
    # 如果需要固定步幅顺序，创建标签列表
    tags = [torch._C.Tag.needs_fixed_stride_order] if needs_fixed_stride_order else []
    # 创建库对象，使用命名空间和"FRAGMENT"类型
    lib = library.Library(ns, "FRAGMENT")
    # 在库中定义给定模式的操作，附加标签
    lib.define(schema_str, tags=tags)
    # 查找或抛出操作句柄
    ophandle = find_ophandle_or_throw(ns, function_schema.name)
    # 创建自定义操作对象，传入库、命名空间、函数模式、名称、操作句柄
    result = CustomOp(lib, ns, function_schema, name, ophandle, _private_access=True)
    # 注册自动微分核心间接引用
    result._register_autograd_kernel_indirection()

    # 设置操作句柄的错误报告回调函数
    torch._C._dispatch_set_report_error_callback(
        ophandle, functools.partial(report_error_callback, weakref.proxy(result))
    )
    # 返回给定限定名称的操作
    return get_op(qualname)
```