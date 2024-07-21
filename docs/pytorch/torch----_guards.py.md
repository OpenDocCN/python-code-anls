# `.\pytorch\torch\_guards.py`

```py
# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib  # 导入 contextlib 模块，用于支持上下文管理器
import dataclasses  # 导入 dataclasses 模块，支持数据类
import enum  # 导入 enum 模块，支持枚举类型
import functools  # 导入 functools 模块，用于高阶函数（如装饰器）
import logging  # 导入 logging 模块，支持日志记录
import threading  # 导入 threading 模块，支持多线程编程
import traceback  # 导入 traceback 模块，支持获取异常堆栈信息
import unittest.mock  # 导入 unittest.mock 模块，支持单元测试的模拟对象
import weakref  # 导入 weakref 模块，支持弱引用对象
from abc import abstractmethod  # 从 abc 模块中导入 abstractmethod 装饰器，用于定义抽象方法
from contextlib import contextmanager  # 从 contextlib 模块中导入 contextmanager 装饰器，简化上下文管理器的创建
from typing import (  # 导入 typing 模块中的多个类型和装饰器，用于类型提示
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

from torch.utils import _pytree as pytree  # 导入 torch.utils 模块下的 _pytree
from torch.utils._traceback import CapturedTraceback  # 导入 torch.utils._traceback 模块中的 CapturedTraceback 类
from torch.utils.weak import WeakTensorKeyDictionary  # 导入 torch.utils.weak 模块中的 WeakTensorKeyDictionary 类

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


if TYPE_CHECKING:
    import sympy  # 在类型检查模式下引入 sympy 模块，用于类型提示

    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import torch  # 在类型检查模式下引入 torch 模块，以启用代码智能功能

"""
torch._guards is the definitional source of truth for general purpose guard structures.

An important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,
and no guard installation notions here.
"""

# 定义 CompileId 命名元组，表示编译标识符，包含 frame_id 和 frame_compile_id 两个字段
class CompileId(NamedTuple):
    frame_id: int
    # This id is per-frame, and counts how many times we've compiled this
    # frame.  This could have been a global id but having this be per-frame
    # gives you a better intuitive sense for how many recompiles have occurred
    # so far.
    frame_compile_id: int
    # TODO: consider also tracking the recompilation count

    def __str__(self):
        return f"{self.frame_id}/{self.frame_compile_id}"


# 定义 TraceId 命名元组，表示跟踪标识符，包含 compile_id 和 attempt 两个字段
class TraceId(NamedTuple):
    compile_id: CompileId
    # This starts off as 0, and every time we restart analysis it goes
    # up by one
    attempt: int

    def __str__(self):
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f"{self.compile_id}_{self.attempt}"


# 定义 GuardSource 枚举类，表示守卫源，包含多种守卫来源类型
class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_NN_MODULE = 2
    GLOBAL_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6
    LOCAL_FSDP_MODULE = 7
    GLOBAL_FSDP_MODULE = 8
    BACKWARD_STATE = 9
    EPHEMERAL = 10
    SYNTHETIC_LOCAL = 11

    # 判断是否为 FSDP 模块类型的守卫
    def is_fsdp_module(self) -> bool:
        return self in (GuardSource.GLOBAL_FSDP_MODULE, GuardSource.LOCAL_FSDP_MODULE)

    # 判断是否为神经网络模块类型的守卫
    def is_nn_module(self) -> bool:
        return (
            self
            in (
                GuardSource.GLOBAL_NN_MODULE,
                GuardSource.LOCAL_NN_MODULE,
            )
            or self.is_fsdp_module()
        )

    # 判断是否为本地类型的守卫
    def is_local(self):
        return self in (
            GuardSource.LOCAL,
            GuardSource.LOCAL_NN_MODULE,
            GuardSource.LOCAL_FSDP_MODULE,
        )


"""
Base class for a "GuardBuilder" role.

The GuardBuilderBase role is to represent a scope within which to build a guard. The name is a little
"""
# 定义一个空的基础抽象类 GuardBuilderBase，用于作为所有具体 GuardBuilder 类的基类
class GuardBuilderBase:
    pass


# 定义一个命名元组 ShapeGuard，包含两个字段：expr 用于表达符号表达式，stack 用于捕获的回溯信息
class ShapeGuard(NamedTuple):
    expr: sympy.Expr
    stack: CapturedTraceback


# 使用 dataclasses.dataclass 装饰器定义 Guard 类
@dataclasses.dataclass
class Guard:
    # originating_source 是调用 make_guard 方法构造此 guard 对象的源头。
    # 名称的含义取决于 create_fn；必须查看 create_fn 内的使用点来确定名称的含义。
    #
    # 尽管 name 看似仅仅是一个名称，但通常它是一个任意的 Python 表达式，
    # 在 GuardBuilder.eval 中会使用所有的全局变量（如果创建了局部 guard，则还包括局部变量）来评估，
    # 以提取我们要进行 guard 测试的 Python 对象。这种评估通常发生在 originating_source.name() 中。
    #
    # 偶尔，name 不是一个有效的 Python 表达式；有时它是毫无意义的。
    # 像 GRAD_MODE 和 SHAPE_ENV 这样的 create_fn 就属于这种情况。
    originating_source: Source
    create_fn: Callable[[GuardBuilderBase, Guard], None]

    # 仅导出。这些值在创建 guard check_fn 时被写入。
    guard_types: Optional[List[str]] = None
    code_list: Optional[List[str]] = None
    obj_weakref: Optional[object] = None
    guarded_class_weakref: Optional[type] = None

    stack: Optional[CapturedTraceback] = None
    user_stack: Optional[traceback.StackSummary] = None
    _hash: Optional[int] = None

    def __hash__(self):
        # 如果 _hash 为 None，则计算其 hash 值，考虑 name、source 和 create_fn 的 id
        if self._hash is None:
            self._hash = hash((self.name, self.source, id(self.create_fn)))
        return self._hash

    def sort_key(self):
        # 将重复的输入 guard 放在最后。重复的 guard 具有两个源头，而 guard.name 仅考虑一个源头。
        from torch._dynamo.guards import GuardBuilder
        
        # 判断是否为重复的输入 guard
        is_duplicate_input = (
            isinstance(self.create_fn, functools.partial)
            and self.create_fn.func is GuardBuilder.DUPLICATE_INPUT
        )
        return (
            is_duplicate_input,
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.inner_create_fn().__code__.co_firstlineno,
        )

    def __lt__(self, other):
        # 通过 sort_key 方法来比较两个 Guard 对象的大小
        return self.sort_key() < other.sort_key()
    # 返回未装饰的创建函数（如果已经部分装饰）
    def inner_create_fn(self):
        if isinstance(self.create_fn, functools.partial):
            return self.create_fn.func
        else:
            return self.create_fn

    @property
    # 返回源对象的名称作为属性的名称
    def name(self) -> str:
        return self.originating_source.name()

    @property
    # 返回保护源对象作为属性的源对象
    def source(self) -> GuardSource:
        return self.originating_source.guard_source()

    @staticmethod
    # 转换弱引用对象为字符串，解决 Python 弱引用的一个 bug
    def weakref_to_str(obj_weakref):
        """
        This is a workaround of a Python weakref bug.

        `obj_weakref` is instance returned by `weakref.ref`,
        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:

            class MyConfig(dict):
                def __getattr__(self, x):
                    return self[x]

            obj = MyConfig(offset=5)
            obj_weakref = weakref.ref(obj)
            str(obj_weakref)  # raise error: KeyError: '__name__'
        """
        if isinstance(obj_weakref, weakref.ReferenceType):
            obj = obj_weakref()
            if obj is not None:
                return f"<weakref at {hex(id(obj_weakref))}; to '{obj.__class__.__name__}' at {hex(id(obj))}>"
            else:
                return f"<weakref at {hex(id(obj_weakref))}; dead>"
        else:
            return str(obj_weakref)

    def __repr__(self):
        # 返回对象的字符串表示形式，包括源名称、名称、创建函数名称、保护类型列表、代码列表、对象弱引用和受保护类弱引用
        s = f"""
        {self.source.name.lower() if self.source else ""} {repr(self.name)} {self.inner_create_fn().__name__}
        {{
            'guard_types': {self.guard_types},
            'code': {self.code_list},
            'obj_weakref': {self.weakref_to_str(self.obj_weakref)}
            'guarded_class': {self.guarded_class_weakref}
        }}
        """
        return s

    def __str__(self):
        # 返回对象的字符串表示形式，包括名称、源名称、创建函数名称、保护类型列表、代码列表、对象弱引用和受保护类弱引用
        output = f"Name: {repr(self.name)}\n"
        source = self.source.name.lower() if self.source else ""
        output += f"    Source: {source}\n"
        output += f"    Create Function: {self.inner_create_fn().__name__}\n"
        output += f"    Guard Types: {self.guard_types}\n"
        output += f"    Code List: {self.code_list}\n"
        output += f"    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n"
        output += f"    Guarded Class Weakref: {self.guarded_class_weakref}\n"
        return output

    def create(self, builder: GuardBuilderBase):
        # 尝试使用创建函数创建对象，捕获异常并记录日志
        try:
            return self.create_fn(builder, self)
        except Exception:
            log.exception("Error while creating guard:\n%s", str(self).rstrip())
            if self.stack:
                log.error("Created at:\n%s", "".join(self.stack.format()[-4:]).rstrip())
            raise

    def is_nn_module(self):
        # 返回源对象是否是 nn 模块
        return self.source.is_nn_module()

    def is_fsdp_module(self):
        # 返回源对象是否是 fsdp 模块
        return self.source.is_fsdp_module()

    def is_local(self):
        # 返回源对象是否是本地模块
        return self.source.is_local()
    # 设置导出信息，用于指定守卫类型、守卫的类、代码列表和对象弱引用
    def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref):
        # 如果 guard_types 列表为空，则初始化为空列表
        if not self.guard_types:
            self.guard_types = list()

        # 将 guard_type 添加到 guard_types 列表中
        self.guard_types.append(guard_type)

        # 断言 guarded_class_weakref 要么与 guarded_class 相同，要么为 None
        assert self.guarded_class_weakref in (
            guarded_class,
            None,
        ), "Guarded class id must be identical, or None"
        # 设置 guarded_class_weakref 为 guarded_class
        self.guarded_class_weakref = guarded_class

        # 如果 code_list 为空，则将其初始化为给定的 code_list；否则将 code_list 扩展到已有的列表中
        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)

        # 对象弱引用可能是短暂的，如 list[slice(1, 2)]。如果在多次调用 set_export_info 之间
        # weakref 失效，即使是 dead weakref 也是可以接受的。
        assert (
            self.obj_weakref
            in (
                obj_weakref,
                None,
            )
            or callable(self.obj_weakref)
            and self.obj_weakref() is None
        ), "Guarded object must be identical, None or ephemeral (dead weakref)"
        # 设置 obj_weakref 为 obj_weakref
        self.obj_weakref = obj_weakref
"""
Parent structure for guard env expressions.
A GuardEnvExpr can have any subtype.
Note: All subtypes must be handled exhaustively in
torch._dynamo.guards._parse_guard_env_guards to avoid a RuntimeError.
"""
@dataclasses.dataclass
class GuardEnvExpr:
    pass

"""
A class representing a pair of duplicate inputs.
input_pos_a and input_pos_b are input positions we have deduped.
"""
@dataclasses.dataclass
class DuplicateInputs(GuardEnvExpr):
    input_source_a: Source
    input_source_b: Source

    def __post_init__(self):
        assert self.input_source_a != self.input_source_b

"""
Checkpointable is an interface for driving state snapshotting, left purposely vague for now.

copy_graphstate() -> T, a somewhat legacy name, is expected to emit a snapshot of any type that
can also be taken in at restore_graphstate(T) calls.

When to snapshot, is, at the moment, an implementation detail of upstream callers. Checkpointable
does not provide any garuantees around consistency, idempotency, or safety of calling its APIs, yet.

In the future, it will have a closer coupling to a generic Checkpoint management system.
"""
class Checkpointable(Generic[T]):
    @abstractmethod
    def copy_graphstate(self) -> T:
        ...

    @abstractmethod
    def restore_graphstate(self, state: T):
        ...


class GuardsCheckpointState:
    """
    The GuardCheckpointState - it is the T of Checkpointable[T] for GuardsContext
    """
    dynamo_guards: Set[Guard] = set()

    def __init__(self, dynamo_guards):
        self.dynamo_guards = dynamo_guards

    def diff(self, other):
        """
        Produces a delta against another GuardsCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        Guard type objects.
        """
        r = self.dynamo_guards.difference(other.dynamo_guards)
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        return self.diff(other) is None


class ModuleContextCheckpointState:
    nn_modules: Dict[str, torch.nn.Module] = {}

    def __init__(self, nn_modules):
        self.nn_modules = nn_modules

    def diff(self, other):
        """
        Produces a delta against another ModuleContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        module key names.
        """
        r = set(self.nn_modules.keys()).difference(set(other.nn_modules.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        return self.diff(other) is None


class ModuleContext(Checkpointable[ModuleContextCheckpointState]):
    def __init__(self):
        self.nn_modules: Dict[str, Any] = {}

    def copy_graphstate(self):
        return ModuleContextCheckpointState(dict(self.nn_modules))
    # 定义一个方法用于恢复图状态，接受一个 ModuleContextCheckpointState 类型的参数 state
    def restore_graphstate(self, state):
        # 断言 state 是 ModuleContextCheckpointState 类型的对象
        assert isinstance(state, ModuleContextCheckpointState)
        # 将 self.nn_modules 设置为 state.nn_modules 的值，即恢复神经网络模块的状态
        self.nn_modules = state.nn_modules
class GlobalContextCheckpointState:
    # 全局状态字典，键为字符串，值为元组，元组包含多个可调用对象
    global_state: Dict[str, Tuple[Callable, ...]] = {}

    def __init__(self, global_states):
        # 初始化方法，接受全局状态字典作为参数
        self.global_state = global_states

    def diff(self, other):
        """
        对比另一个 GlobalContextCheckpointState 的差异。

        如果没有差异，则返回 None，否则返回一个集合，包含不匹配的全局键名。
        """
        # 找出当前实例与另一个实例的全局键名的差集
        r = set(self.global_state.keys()).difference(set(other.global_state.keys()))
        # 如果差集为空集，则返回 None
        if len(r) == 0:
            return None
        # 否则返回差集
        return r

    def __eq__(self, other):
        # 判断两个对象是否相等，基于 diff 方法的结果
        return self.diff(other) is None


class GlobalContext(Checkpointable[GlobalContextCheckpointState]):
    """
    追踪函数执行期间的全局 torch 状态。

    例如，torch.is_grad_enabled。
    """

    # 支持的全局状态集合
    _supported_global_states = {
        "grad_enabled",
        "torch_function_enabled",
        "autocast_enabled",
        "autocast_cpu_enabled",
        "autocast_gpu_dtype",
        "autocast_cpu_dtype",
        "autocast_cache_enabled",
    }

    def __init__(self):
        # 初始化方法，初始化全局状态字典为空字典
        self.global_state: Dict[str, Tuple[Callable, ...]] = {}

    def copy_graphstate(self):
        # 复制当前全局状态的方法
        return GlobalContextCheckpointState(dict(self.global_state))

    def restore_graphstate(self, state):
        # 恢复全局状态的方法，确保传入的参数是 GlobalContextCheckpointState 类的实例
        assert isinstance(state, GlobalContextCheckpointState)
        # 将传入状态的全局状态复制到当前实例的全局状态
        self.global_state = state.global_state
        # 断言当前全局状态与支持的全局状态集合一致
        assert (
            len(self.global_state) == len(self._supported_global_states)
            and set(self.global_state.keys()) == self._supported_global_states
        ), "Global state mismatch"
        # 针对全局状态字典中的每个条目，执行相应的函数和参数
        for func, args in self.global_state.values():
            func(args)


"""
GuardsContext 是当前追踪上下文中所有守卫的可检查表示。
其生命周期与追踪上下文绑定，不应在其外部直接实例化。
为了在内部传递此对象的状态表示，请使用 copy_graphstate 提取它们以生成 GuardsCheckpointState。
"""


# 类似于 Set[Guard]，但会记录安装在其目标位置的所有守卫在其上下文中的用户堆栈
class GuardsSet:
    def __init__(self, inner=None):
        # 初始化方法，如果未提供内部集合，则创建一个空集合
        if inner is None:
            inner = set()
        self.inner = inner

    def __iter__(self):
        # 迭代器方法，返回内部集合的迭代器
        return iter(self.inner)

    def __len__(self):
        # 返回内部集合的长度
        return len(self.inner)

    # 减法操作以及布尔运算通常用于确定高阶操作之间添加的守卫的差异
    def __sub__(self, other):
        # 减法操作，返回当前集合与另一个集合的差集作为新的 GuardsSet 对象
        return GuardsSet(self.inner - other.inner)

    def __bool__(self):
        # 布尔运算方法，判断内部集合是否为真（非空）
        return bool(self.inner)
    def add(self, guard: Guard, *, collect_debug_stack=True, skip=0):
        # 如果给定的 guard 已经存在于集合中，则直接返回，不进行添加
        if guard in self.inner:
            return
        
        # 如果需要收集调试堆栈信息
        if collect_debug_stack:
            # 如果 guard 的堆栈信息为空，则提取捕获的 traceback 信息
            if guard.stack is None:
                guard.stack = CapturedTraceback.extract(skip=1 + skip)
            # 如果 guard 的用户堆栈信息为空，则提取当前调用栈信息
            if guard.user_stack is None:
                guard.user_stack = TracingContext.extract_stack()
        
        # 将 guard 添加到集合中
        self.inner.add(guard)

    def update(self, *others: Set[Guard]):
        # 遍历所有传入的集合
        for o in others:
            # 遍历集合中的每个 guard
            for g in o:
                # 调用 add 方法将 guard 添加到 self.inner 中，跳过一层调用栈
                self.add(g, skip=1)

    def remove_guards_with_source(self, source):
        """Delete all guards with a given source"""
        # 使用集合推导式过滤掉来源为指定 source 的 guard
        self.inner = {g for g in self.inner if g.originating_source != source}
class GuardsContext(Checkpointable[GuardsCheckpointState]):
    # GuardsContext 类，实现了 Checkpointable 接口，处理 GuardsCheckpointState 状态
    def __init__(self):
        # 初始化方法，创建 dynamo_guards 和 aotautograd_guards 属性
        self.dynamo_guards: GuardsSet = GuardsSet()  # 使用 GuardsSet 创建 dynamo_guards
        self.aotautograd_guards: List[GuardEnvExpr] = []  # 创建空的 aotautograd_guards 列表

    def copy_graphstate(self):
        # 复制当前的图状态
        return GuardsCheckpointState(set(self.dynamo_guards.inner))

    def restore_graphstate(self, state):
        # 恢复图状态的方法，接受 GuardsCheckpointState 类型的 state 参数
        # 注意：这里 "steals" 了传入的 state
        assert isinstance(state, GuardsCheckpointState)
        self.dynamo_guards = GuardsSet(state.dynamo_guards)


_TLS = threading.local()

"""
TracingContext is the source of truth for all currently accumulated information
needed to trace. Its lifecycle is kept 1:1 when using TorchDynamo, but other systems
are open to managing their own TracingContext with that in mind.

The purpose of TracingContext is not to be a dumping ground, or god object, but rather to avoid
having to plumb complex subsystems across multiple verticals.

Ex: A common example is guard accumulation between dynamo, shape_env, aot_autograd, and inductor.
Accessing the current tracing context via
TracingContext.get() allows users to accumulate their own guards for processing, without needing to know how
to plumb objects back up to where frame interpretation happened.

Note that you can end up with multiple TracingContext for a single compilation
of a frame, as we reset the TracingContext whenever we restart analysis.
CompileContext is a more overarching context that encompasses multiple restarts.
"""


class CompileContext:
    # 编译上下文类，用于管理编译过程的上下文信息
    @staticmethod
    def get() -> CompileContext:
        # 静态方法，获取当前的 CompileContext 实例
        assert _TLS.compile_context is not None
        return _TLS.compile_context

    @staticmethod
    def try_get() -> Optional[CompileContext]:
        # 尝试获取当前的 CompileContext 实例，可能返回 None
        return getattr(_TLS, "compile_context", None)

    def __init__(self, compile_id):
        # 初始化方法，接受编译 ID 作为参数
        assert compile_id is None or isinstance(compile_id, CompileId)
        self.compile_id: Optional[CompileId] = compile_id  # 设置编译 ID
        self.attempt = 0  # 初始化尝试次数为 0

    @staticmethod
    def current_compile_id():
        # 获取当前的编译 ID 的静态方法
        self = CompileContext.try_get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id():
        # 获取当前的追踪 ID 的静态方法
        self = CompileContext.try_get()
        if self is None:
            return None
        if self.compile_id is None:
            return None
        return TraceId(self.compile_id, self.attempt)


class TracingContext:
    """
    Provides the currently installed TracingContext, or None.

    Note that it is a staticmethod, and invocations outside of `with tracing()` (see below), are valid but
    will return None.
    """
    # 追踪上下文类，提供当前安装的追踪上下文或 None

    @staticmethod
    def try_get() -> Optional[TracingContext]:
        # 尝试获取当前的 TracingContext 实例，可能返回 None
        return getattr(_TLS, "tracing_context", None)

    @staticmethod
    def get() -> TracingContext:
        # 获取当前的 TracingContext 实例，如果不存在则抛出运行时错误
        if ctx := TracingContext.try_get():
            return ctx
        raise RuntimeError(
            "TracingContext.get() must be called within an ongoing trace."
        )
    def __init__(self, fake_mode):
        # 初始化 GuardsContext 对象
        self.guards_context = GuardsContext()
        # 初始化 ModuleContext 对象
        self.module_context = ModuleContext()
        # 初始化 GlobalContext 对象
        self.global_context = GlobalContext()
        # 设置 fake_mode 属性
        self.fake_mode = fake_mode
        # 初始化空的 frame_summary_stack 列表，用于记录函数调用堆栈信息
        self.frame_summary_stack = []
        
        # 下面的变量是 frame_summary_stack 的一部分，但为了清晰起见，分开列出。
        # 它用于跟踪当前正在处理的函数内的行号。
        # 当我们进行函数调用时，这个变量会被清除，并且函数的位置会被推入 frame_summary_stack 中。
        self.loc_in_frame = None
        
        # 在 aot_autograd 之后设置的元数据
        self.fw_metadata = None
        # 在 aot_autograd 之后设置的 AOT 图的名称
        self.aot_graph_name = None
        self.params_flat = None
        
        # 用于从后端编译器到 aot_autograd 的扩展返回调用约定。
        # 每个输出的编译器指定的输出步幅，如果不知道步幅则为 None。
        # 这总是一个提示值，不是 SymInt（如果可以从 Inductor 方便地获取 SymInt，则更好）。
        # 在 aot_autograd.py 中更改此内容时要小心，确保不会意外引入 SymInt 的 guards。
        self.output_strides: Optional[List[Optional[Tuple[int, ...]]]] = None
        
        # 当为 True 时，当我们在 Dynamo 跟踪中遇到一个整数时，
        # 我们会（1）强制取消特化，并（2）将其强制作为类似大小的未支持整数。
        # 当处理某些已知为类似大小且可能具有 0/1 条目的整数列表时，目前正在使用此功能。
        self.force_unspec_int_unbacked_size_like = False
        
        # 参见备注【Tensor Fakification and Symbol Caching】
        # 使用 WeakTensorKeyDictionary 来映射张量到上下文的弱引用字典
        self.tensor_to_context = WeakTensorKeyDictionary()
        
        # 如果为 True，在第一次 AOT Autograd 调用时，Aot Autograd 将返回具有适当元数据的虚拟张量。
        # 参见备注【Returning Fake Tensors on First AOT Autograd Call】
        self.fakify_first_call = False

    def clear(self):
        # 查看 output_graph.py 中 save_global_state 函数的注释，了解清除全局上下文的背景。
        self.global_context.global_state = {}

    @staticmethod
    @contextmanager
    def patch(**kwargs):
        prior = {}
        ctx = TracingContext.get()

        for key in kwargs.keys():
            # 在无效条目时会引发 KeyError
            # 保存原始的上下文属性值
            prior[key] = getattr(ctx, key)
        
        for key, val in kwargs.items():
            # 设置新的上下文属性值
            setattr(ctx, key, val)
        
        try:
            yield
        finally:
            # 在退出上下文管理器后，恢复先前保存的上下文属性值
            for key, val in prior.items():
                setattr(ctx, key, val)
    def extract_stack():
        # 尝试获取当前的追踪上下文对象
        self = TracingContext.try_get()
        # 如果获取不到上下文对象，则返回一个空的堆栈摘要
        if self is None:
            return traceback.StackSummary()
        # 否则获取当前的帧摘要堆栈
        stack = self.frame_summary_stack
        # 如果存在具体的帧位置信息，则将其添加到堆栈中
        if self.loc_in_frame is not None:
            stack = stack + [self.loc_in_frame]
        # 从堆栈列表创建并返回堆栈摘要
        return traceback.StackSummary.from_list(stack)

    # 当需要调用与当前帧状态无关的某些代码时调用此函数
    @staticmethod
    @contextlib.contextmanager
    def clear_frame():
        # 获取当前的追踪上下文对象
        tc = TracingContext.get()
        # 使用 context manager 清空 frame_summary_stack 和 loc_in_frame
        with unittest.mock.patch.object(
            tc, "frame_summary_stack", []
        ), unittest.mock.patch.object(tc, "loc_in_frame", None):
            try:
                yield
            except Exception as e:
                # 防止 real_stack 被附加
                #
                # 不变条件是，如果一个异常有 real_stack，则我们已经适当地附加了用户堆栈，
                # 并且不再需要附加任何东西。因为我们无法方便地插入异常抛出时的位置，
                # 所以我们在每次设置用户堆栈的地方都插入。但是，我们的编译器堆栈执行 "尾调用"
                # （当它调用用户编译器时），此时父异常帧会错误地附加不正确的帧。
                #
                # 然而，如果某种方式，有人通过此范围引发了一个带有堆栈的异常
                # （例如，因为他们在处理节点时逐个恢复了用户堆栈状态），我们应该尊重它。
                # 因此，我们不能无条件地设置为 None。
                if not hasattr(e, "real_stack"):
                    e.real_stack = None  # type: ignore[attr-defined]
                raise

    @staticmethod
    @contextlib.contextmanager
    def current_frame(frame_summary):
        # frame_summary 可以为 None，仅利用 real_stack 附加到抛出的异常
        # 获取当前的追踪上下文对象
        tc = TracingContext.get()
        # 如果 frame_summary 不为 None，则将其添加到帧摘要堆栈中
        if frame_summary is not None:
            tc.frame_summary_stack.append(frame_summary)
        # 保存旧的 loc_in_frame 值，并将其设置为 None
        old = tc.loc_in_frame
        tc.loc_in_frame = None
        try:
            yield
        except Exception as e:
            # 如果异常 e 没有 real_stack 属性，则将当前堆栈摘要附加到 e 的 real_stack 属性中
            if not hasattr(e, "real_stack"):
                e.real_stack = tc.extract_stack()  # type: ignore[attr-defined]
            raise
        finally:
            # 如果 frame_summary 不为 None，则弹出帧摘要堆栈的最后一个元素
            if frame_summary is not None:
                tc.frame_summary_stack.pop()
            # 恢复旧的 loc_in_frame 值
            tc.loc_in_frame = old

    @staticmethod
    @contextlib.contextmanager
    # 定义一个生成器函数 report_output_strides，用于输出跟踪上下文的输出步幅
    def report_output_strides():
        # 尝试获取当前的跟踪上下文对象
        tc = TracingContext.try_get()
        # 如果未获取到跟踪上下文对象，则返回 None 并结束生成器
        if tc is None:
            yield None
            return
        # 保存旧的输出步幅设置
        old_output_strides = tc.output_strides
        # 清空当前跟踪上下文对象的输出步幅
        tc.output_strides = []
        try:
            # 生成器生成当前跟踪上下文对象的输出步幅，并返回
            yield tc.output_strides
        finally:
            # 恢复之前保存的旧的输出步幅设置
            tc.output_strides = old_output_strides

    # 静态方法：设置当前位置信息到跟踪上下文中
    @staticmethod
    def set_current_loc(filename, lineno, frame_name):
        # 获取当前的跟踪上下文对象，设置当前位置信息到其中
        TracingContext.get().loc_in_frame = traceback.FrameSummary(
            filename, lineno, frame_name, lookup_line=False
        )
@contextmanager
def compile_context(context: Optional[CompileContext]):
    # 获取当前线程局部存储中的旧编译上下文
    old_context = getattr(_TLS, "compile_context", None)
    # 将传入的编译上下文设置为线程局部存储中的当前编译上下文
    _TLS.compile_context = context
    try:
        # 返回传入的编译上下文，供上下文管理器使用
        yield context
    finally:
        # 恢复旧的编译上下文到线程局部存储中
        _TLS.compile_context = old_context


@contextmanager
def tracing(context: Optional[TracingContext]):
    """
    This function installs the passed in tracing context as a dynamic scoped
    global variable.

    Calls to TracingContext.get() while not under a `with tracing()` context
    will return None.
    """
    # 获取当前线程局部存储中的旧追踪上下文
    old_context = getattr(_TLS, "tracing_context", None)
    # 将传入的追踪上下文设置为线程局部存储中的当前追踪上下文
    _TLS.tracing_context = context
    try:
        # 返回传入的追踪上下文，供上下文管理器使用
        yield context
    except Exception as e:
        # 如果异常对象没有属性 "real_stack"，并且传入的追踪上下文不为 None，则设置异常的真实堆栈信息
        if not hasattr(e, "real_stack") and context is not None:
            e.real_stack = context.extract_stack()  # type: ignore[attr-defined]
        raise
    finally:
        # 如果追踪上下文不为 None，并且处于 fake 模式下，并且 shape_env 不为 None，则执行清理操作
        if (
            context is not None
            and context.fake_mode is not None
            and context.fake_mode.shape_env is not None
        ):
            context.fake_mode.shape_env.cleanup()
        # 恢复旧的追踪上下文到线程局部存储中
        _TLS.tracing_context = old_context


# Subclasses can be found in torch/_dynamo/source.py
# TODO(voz): Consider a toplevel torch/_source.py
@dataclasses.dataclass(frozen=True)
class Source:
    def is_dict_key(self):
        return False

    def is_ephemeral(self):
        return False

    def reconstruct(self, codegen):
        raise NotImplementedError

    def guard_source(self) -> GuardSource:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    def make_guard(self, fn) -> Guard:
        # 如果 guard_source() 返回 GuardSource.CONSTANT，则抛出 NotImplementedError
        if self.guard_source() is GuardSource.CONSTANT:
            raise NotImplementedError
        # 否则返回一个 Guard 对象，使用当前对象和传入的函数参数
        return Guard(self, fn)

    def is_nn_module(self) -> bool:
        # 返回 guard_source() 是否为 nn_module 的结果
        return self.guard_source().is_nn_module()

    def subguards_allowed(self):
        """True if you can guard on attributes of this"""
        # 如果 guard_source() 不是 GuardSource.SYNTHETIC_LOCAL，则返回 True
        return self.guard_source() != GuardSource.SYNTHETIC_LOCAL


# Subclasses can be found in torch/_dynamo/source.py
@dataclasses.dataclass(frozen=True)
class ChainedSource(Source):
    base: Source

    def is_dict_key(self):
        # 递归调用 base 的 is_dict_key() 方法，直到遇到 ConstDictKey 或 Source
        return self.base.is_dict_key()

    def is_ephemeral(self):
        # 返回 base 的 is_ephemeral() 方法的结果
        return self.base.is_ephemeral()


def detect_fake_mode(inputs: Any = None):
    """
    Attempts to "detect" what the current fake mode is.  If there is one ambiently
    available from TracingContext, we preferentially use that.  Otherwise, we
    heuristically detect the fake mode via the following sources, in order of
    priority:

        - Currently active fake mode on stack
        - Fake mode associated with passed in tensors (inputs does not
          have to be flattened)
    """
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    # 初始化空列表来存储检测到的 fake modes
    fake_modes = []
    # 尝试获取当前的追踪上下文对象
    if context := TracingContext.try_get():
        # 获取追踪上下文中的 fake_mode 属性
        fake_mode = context.fake_mode
        # 如果 fake_mode 不为 None，则将其添加到 fake_modes 列表中
        if fake_mode is not None:
            fake_modes.append((fake_mode, "tracing context", 0))

    # 导入 torch.utils._python_dispatch 模块中的 _get_current_dispatch_mode_stack 函数
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    # 遍历 _get_current_dispatch_mode_stack 返回的模式栈，使用 enumerate 逆序遍历
    for i, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        # 如果当前模式 m 是 FakeTensorMode 类型，则将其添加到 fake_modes 列表中
        if isinstance(m, FakeTensorMode):
            fake_modes.append((m, "active fake mode", i))

    # 将输入 inputs 展平为一维列表 flat_inputs
    flat_inputs = pytree.tree_leaves(inputs)
    # 遍历 flat_inputs 的索引和值 flat_input
    for i, flat_input in enumerate(flat_inputs):
        # 如果 flat_input 是 FakeTensor 类型，则将其 fake_mode 添加到 fake_modes 列表中
        if isinstance(flat_input, FakeTensor):
            fake_modes.append((flat_input.fake_mode, "fake tensor input", i))

    # 如果 fake_modes 列表非空，则进行模式一致性检查
    if fake_modes:
        # 取出 fake_modes 列表中的第一个元素的 fake_mode、desc1 和 i1
        fake_mode, desc1, i1 = fake_modes[0]
        # 遍历 fake_modes 列表中的其他元素 m、desc2 和 i2
        for m, desc2, i2 in fake_modes[1:]:
            # 断言当前 fake_mode 与后续元素的 fake_mode 相同，否则抛出异常
            assert fake_mode is m, (
                f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                f"fake mode from {desc1} {i1} allocated at:\n{fake_mode.stack}\n"
                f"fake mode from {desc2} {i2} allocated at:\n{m.stack}"
            )
        # 返回首个 fake_mode
        return fake_mode
    else:
        # 如果 fake_modes 列表为空，则返回 None
        return None
# 检查分发模式堆栈，寻找是否有激活的虚拟模式，并返回该模式对象
def active_fake_mode():
    # 导入需要的模块：FakeTensorMode 和 _get_current_dispatch_mode_stack
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    # 遍历反转后的分发模式堆栈
    for _, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        # 检查当前元素是否为 FakeTensorMode 类型的对象
        if isinstance(m, FakeTensorMode):
            # 如果找到激活的虚拟模式，则返回该模式对象
            return m

    # 如果没有找到激活的虚拟模式，返回 None
    return None
```