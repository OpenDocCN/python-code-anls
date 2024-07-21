# `.\pytorch\torch\_inductor\virtualized.py`

```py
"""
This file provides a number of "global" variables/handlers that are actually
thread local and dynamically scoped, with Inductor patching them to various
implementations depending on the situation.

These handlers are interacted with in a fairly stylized way.  Typically,
we will import V from this module::

    from .virtualized import V

Various handlers are accessible as attributes on this module; for example,
you might access ``V.graph.sizevars.size_hint`` to resolve a size hint associated with
a number.

There are a few distinct usage patterns for virtualized global variables:

1. Implicit argument passing.  Examples: ``V.current_node``, ``V.aot_compilation``.
   Use ``V.set_current_node`` to change what the current node is while we're
   executing some region of code, so code inside that region can query ``V.current_node``
   to find out what it is.  This is often more convenient than manually threading
   the current node as an argument through all call stacks.

2. Per-compilation global state.  Examples: ``V.fake_mode``, ``V.graph``.  For a
   given ``compile_fx`` invocation, these typically don't change, but they are
   associated with some internal state so they cannot just be global functions.
   We install these objects at the beginning of compilation and then you can
   conveniently access them without having to pass them around.

3. Alternate define-by-run interpretations.  Examples: ``V.ops``, ``V.kernel``.
   A commonly used IR in Inductor is define-by-run: instead of maintaining
   explicit syntax data structures, we instead represent loop bodies as
   callable functions, which internally invoke operations defined on
   ``V.ops``.  To perform semantic analysis, print or code generate these
   operations, we dynamically patch ``V.ops`` with an alternate handler with
   the intended semantics and then run the callable function.  For example, to
   extract out a traditional (FX) graph representation of the define-by-run
   IR, simply install a handler that records each ``ops`` call to a graph.

   TODO: Define a parent class / protocol that defines all of the operations
   V.ops is expected to support.

It is typically an error to access a virtualized global without having installed
an appropriate handler (you will get a NullHandler), although in some cases we
provide a default implementation.

One last thing: although most virtualized globals are accessed via ``V``, ``ops`` is
ubiquitous enough to have its own top level variable, so you will typically see
``ops.constant(...)`` rather than ``V.ops.constant(...)``.  In fact, these are not
equivalent; the former interface supports arithmetic overloads like ``x + y``
instead of forcing ``ops.add(x, y)``, so it should be preferred.

Some operators are seemingly unused, but they are implicitly used by ops_wrapper.
In particular, we typically have an operator for every basic pointwise PyTorch operation
supported.
"""

# From the future module, import the __annotations__ feature
from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager
from threading import local
from typing import Any, Callable, Generic, List, Type, TYPE_CHECKING, TypeVar, Union

from .ops_handler import (  # noqa: F401
    KernelFormatterHandler,
    MockHandler,
    OpsHandler,
    ReductionType,
    StoreMode,
    WrapperHandler,
)

if TYPE_CHECKING:
    import torch
    from torch._inductor.debug import DebugContext
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import InterpreterShim
    from torch._subclasses import FakeTensorMode

# 创建一个本地线程对象
threadlocal = local()

# 定义一个泛型类型变量
T = TypeVar("T")

# Sentinel 类，用于指示未设置全局变量（类似于 None）
class NullHandler:
    """
    Sentinel indicating that a global variable is unset ala None.  Typically,
    attempting to access the global variable before it's set is an error, but with
    NullHandler it won't fail until you try to access an attribute on it.
    """
    pass


# 泛型类 Virtualized，实现一个通过线程本地变量重定向的全局变量
class Virtualized(Generic[T]):
    """
    Implements a global variable that redirects via thread local variable
    (NB: construct this class to create the global variable; this is not
    a singleton class!)

    This allows us to swap in different op implementations in codegen.

    NB: Despite the fact that we typically call these "handlers" (e.g., NullHandler is
    the default value of the variable), we sometimes use these variables to
    store other things, like booleans.
    """

    def __init__(self, vname: str, default: Union[Callable[[], T], Type[NullHandler]]):
        # 私有属性 _key，用于在线程本地存储中标识变量名
        self._key: str = f"__torchinductor_{vname}"
        # 初始默认值为给定的 default 参数
        self._default = default

    # 设置处理器的方法，并返回上下文管理器
    def _set_handler(self, value: T) -> AbstractContextManager[None]:
        # 获取当前的处理器
        prior = self._get_handler()
        # 将新值存储到线程本地存储中
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            try:
                yield
            finally:
                # 恢复之前的处理器值
                self._set_handler(prior)

        return ctx()

    # 获取当前处理器的方法
    def _get_handler(self) -> T:
        try:
            # 尝试从线程本地存储中获取处理器值
            return getattr(threadlocal, self._key)
        except AttributeError:
            # 如果属性错误，则返回默认值（调用 _default 方法）
            return self._default()  # type: ignore[return-value]

    # 重载 __getattr__ 方法，使得可以通过该类访问处理器的属性
    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_handler(), name)


# NullKernelHandler 类，继承自 NullHandler，用于处理没有内核上下文时的情况
class NullKernelHandler(NullHandler):
    """
    We need access `V.kernel.removed_buffers` in DeferredLine class when there
    is no kernel in the context. This happens when codegening the wrapper.
    Initialize `removed_buffers` and `inplaced_to_remove` explicitly so we don't
    need call 'getattr' with default value which is error prone to typo in
    attribute name.
    """

    def __init__(self):
        super().__init__()
        # 初始化 removed_buffers 和 inplaced_to_remove 属性，避免在访问属性时出现错误
        self.removed_buffers = set()
        self.inplaced_to_remove = set()
        self.index_dtype = "tl.int64"


# _ops 全局变量，实例化 Virtualized 泛型类，用于操作处理器
_ops: Virtualized[OpsHandler[Any]] = Virtualized("ops", MockHandler)
# 创建一个 Virtualized 对象 _graph，用于处理图形降级相关操作
_graph: Virtualized[GraphLowering] = Virtualized("graph", NullHandler)

# 创建一个 Virtualized 对象 _real_inputs，用于处理实际输入的列表
_real_inputs: Virtualized[List[torch.Tensor]] = Virtualized("real_inputs", NullHandler)

# 创建一个 Virtualized 对象 _fake_mode，用于处理虚拟张量模式相关操作
_fake_mode: Virtualized[FakeTensorMode] = Virtualized("fake_mode", NullHandler)

# 创建一个 Virtualized 对象 _kernel，用于处理内核相关操作，目前采用空处理程序
_kernel: Virtualized[NullKernelHandler] = Virtualized(
    "kernel", NullKernelHandler
)  # TODO: improve type

# 创建一个 Virtualized 对象 _debug，用于处理调试上下文相关操作
_debug: Virtualized[DebugContext] = Virtualized("debug", NullHandler)

# 创建一个 Virtualized 对象 _interpreter，用于处理解释器相关操作
_interpreter: Virtualized[InterpreterShim] = Virtualized("interpreter", NullHandler)

# 创建一个 Virtualized 对象 _aot_compilation，用于处理 AOT 编译相关操作
_aot_compilation: Virtualized[bool] = Virtualized("aot_compilation", NullHandler)

# 创建一个 Virtualized 对象 _current_node，用于处理当前节点相关操作
_current_node: Virtualized[torch.fx.Node] = Virtualized("current_node", NullHandler)


class OpsValue:
    """The return type of most ops calls.

    This exists so we can overload magic methods, and write mathematical
    expressions much more fluently. So instead of

        ops.add(ops.mul(ops.mul(ops.sub(ops.mul(_Ap2, x), _Ap3), x), x), _1)

    we can write

        (_Ap2 * x - _Ap3) * x * x + _1

    """

    value: Any

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"OpsValue({self.value!r})"

    # 以下为重载的数学运算符方法，用于处理 OpsValue 实例的运算

    def __add__(self, other):
        return ops.add(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __sub__(self, other):
        return ops.sub(self, other)

    def __neg__(self):
        return ops.neg(self)

    def __truediv__(self, other):
        return ops.truediv(self, other)

    def __floordiv__(self, other):
        return ops.floordiv(self, other)

    def __mod__(self, other):
        return ops.mod(self, other)

    def __pow__(self, other):
        return ops.pow(self, other)

    def __lt__(self, other):
        return ops.lt(self, other)

    def __le__(self, other):
        return ops.le(self, other)

    def __eq__(self, other):
        return ops.eq(self, other)

    def __ne__(self, other):
        return ops.ne(self, other)

    def __gt__(self, other):
        return ops.gt(self, other)

    def __ge__(self, other):
        return ops.ge(self, other)

    def __and__(self, other):
        return ops.bitwise_and(self, other)

    def __or__(self, other):
        return ops.bitwise_or(self, other)

    def __xor__(self, other):
        return ops.bitwise_xor(self, other)

    def __invert__(self):
        return ops.bitwise_not(self)

    def __rshfit__(self, n):
        return ops.bitwise_right_shift(self, n)

    def __lshift__(self, n):
        return ops.bitwise_left_shift(self, n)


class OpsWrapper:
    """This wraps any returned IR values into an `OpsValue` instance, so that we
    can overload the magic methods for writing mathematical expressions fluently.
    """
    # 当对象的属性不存在时，调用该方法，返回一个内部函数 inner
    def __getattr__(self, name):
        # 定义内部函数 inner，接受任意位置参数和关键字参数
        def inner(*args, **kwargs):
            # 对传入的所有位置参数进行解包操作，返回解包后的列表
            new_args = [OpsWrapper._unwrap(a) for a in args]
            # 对传入的所有关键字参数进行解包操作，返回解包后的字典
            new_kwargs = {k: OpsWrapper._unwrap(v) for k, v in kwargs.items()}
            # 调用 _ops 模块中的 name 方法，传入解包后的参数和关键字参数，再封装返回值
            return OpsWrapper._wrap(getattr(_ops, name)(*new_args, **new_kwargs))

        # 返回内部函数 inner，用于动态获取属性时执行相应的操作
        return inner

    # 静态方法，用于将特定类型的对象解包
    @staticmethod
    def _unwrap(x):
        # 如果 x 是列表或元组，递归地对其内部元素解包并返回元组
        if isinstance(x, (list, tuple)):
            return tuple(OpsWrapper._unwrap(v) for v in x)
        # 如果 x 是 OpsValue 类型的对象，直接返回其值
        if isinstance(x, OpsValue):
            return x.value
        # 否则直接返回 x
        return x

    # 静态方法，用于将特定类型的对象封装为 OpsValue 对象
    @staticmethod
    def _wrap(x):
        # 如果 x 是列表或元组，将其内部元素逐个封装为 OpsValue 对象并返回元组
        if isinstance(x, (list, tuple)):
            return tuple(OpsValue(v) for v in x)
        # 否则将 x 封装为 OpsValue 对象并返回
        return OpsValue(x)

    # 静态方法，实现间接索引功能，接受索引值、大小及是否检查有效性的标志作为参数
    @staticmethod
    def indirect_indexing(index, size, check=True):
        # 调用 _unwrap 方法对索引值进行解包
        index = OpsWrapper._unwrap(index)
        # 调用 _ops 模块中的 indirect_indexing 方法，传入解包后的索引值、大小及检查标志，并返回结果
        return _ops.indirect_indexing(index, size, check)
# 创建一个名为 ops 的实例，使用 OpsWrapper 类的实例化对象
ops = OpsWrapper()

# 定义一个名为 _V 的类
class _V:
    # 将 MockHandler 类指定为 _V 类的属性
    MockHandler = MockHandler
    # 将 KernelFormatterHandler 类指定为 _V 类的属性
    KernelFormatterHandler = KernelFormatterHandler
    # 将 WrapperHandler 类指定为 _V 类的属性
    WrapperHandler = WrapperHandler

    # 定义一系列函数类型的属性，并将相应的函数赋给这些属性
    set_ops_handler: Callable[[Any], Any] = _ops._set_handler
    get_ops_handler: Callable[[], Any] = _ops._get_handler
    set_graph_handler: Callable[[GraphLowering], Any] = _graph._set_handler
    set_real_inputs: Callable[[Any], Any] = _real_inputs._set_handler
    get_real_inputs: Callable[[], Any] = _real_inputs._get_handler
    set_fake_mode: Callable[[Any], Any] = _fake_mode._set_handler
    get_fake_mode: Callable[[], Any] = _fake_mode._get_handler
    set_kernel_handler: Callable[[Any], Any] = _kernel._set_handler
    set_debug_handler: Callable[[Any], Any] = _debug._set_handler
    set_interpreter_handler: Callable[[Any], Any] = _interpreter._set_handler
    set_aot_compilation: Callable[[bool], Any] = _aot_compilation._set_handler
    get_aot_compilation: Callable[[], Any] = _aot_compilation._get_handler
    set_current_node: Callable[[Any], Any] = _current_node._set_handler
    get_current_node: Callable[[], Any] = _current_node._get_handler

    # 定义 ops 属性为只读属性，并返回 _ops._get_handler() 的结果
    @property
    def ops(self) -> OpsHandler[Any]:
        """The operator handler specific to the current codegen task"""
        return _ops._get_handler()

    # 定义 graph 属性为只读属性，并返回 _graph._get_handler() 的结果
    @property
    def graph(self) -> GraphLowering:
        """The graph currently being generated"""
        return _graph._get_handler()

    # 定义 real_inputs 属性为只读属性，并返回 _real_inputs._get_handler() 的结果
    @property
    def real_inputs(self):
        """non-fake example inputs"""
        return _real_inputs._get_handler()

    # 定义 fake_mode 属性为只读属性，并返回 _fake_mode._get_handler() 的结果
    @property
    def fake_mode(self):
        """The graph currently being generated"""
        return _fake_mode._get_handler()

    # 定义 kernel 属性为只读属性，并返回 _kernel._get_handler() 的结果
    @property
    def kernel(self):
        """The kernel currently being generated"""
        return _kernel._get_handler()

    # 定义 debug 属性为只读属性，并返回 _debug._get_handler() 的结果
    @property
    def debug(self):
        return _debug._get_handler()

    # 定义 interpreter 属性为只读属性，并返回 _interpreter._get_handler() 的结果
    @property
    def interpreter(self):
        return _interpreter._get_handler()

    # 定义 aot_compilation 属性为只读属性，并返回 _aot_compilation._get_handler() 的结果
    @property
    def aot_compilation(self):
        return _aot_compilation._get_handler()

    # 定义 current_node 属性为只读属性，并返回 _current_node._get_handler() 的结果
    @property
    def current_node(self):
        return _current_node._get_handler()

# 将 _V 类的实例化对象赋给 V 变量
V = _V()
```