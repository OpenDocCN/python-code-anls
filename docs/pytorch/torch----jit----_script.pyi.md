# `.\pytorch\torch\jit\_script.pyi`

```py
# 设置类型检查器允许未定义的函数签名
# 禁用类型错误代码“type-arg”
from typing import Any, Callable, NamedTuple, overload, TypeVar
from typing_extensions import Never, TypeAlias

# 从_typeshed导入Incomplete类型
from _typeshed import Incomplete

# 导入torch库
import torch

# 从torch._classes中导入classes别名
from torch._classes import classes as classes

# 从torch._jit_internal中导入_qualified_name函数别名
from torch._jit_internal import _qualified_name as _qualified_name

# 从torch.jit._builtins中导入_register_builtin函数别名
from torch.jit._builtins import _register_builtin as _register_builtin

# 从torch.jit._fuser中导入_graph_for和_script_method_graph_for函数别名
from torch.jit._fuser import (
    _graph_for as _graph_for,
    _script_method_graph_for as _script_method_graph_for,
)

# 从torch.jit._monkeytype_config中导入JitTypeTraceConfig、JitTypeTraceStore和monkeytype_trace别名
from torch.jit._monkeytype_config import (
    JitTypeTraceConfig as JitTypeTraceConfig,
    JitTypeTraceStore as JitTypeTraceStore,
    monkeytype_trace as monkeytype_trace,
)

# 从torch.jit._recursive中导入_compile_and_register_class、infer_methods_to_compile、
# ScriptMethodStub和wrap_cpp_module别名
from torch.jit._recursive import (
    _compile_and_register_class as _compile_and_register_class,
    infer_methods_to_compile as infer_methods_to_compile,
    ScriptMethodStub as ScriptMethodStub,
    wrap_cpp_module as wrap_cpp_module,
)

# 从torch.jit._serialization中导入validate_map_location函数别名
from torch.jit._serialization import validate_map_location as validate_map_location

# 从torch.jit._state中导入_enabled、_set_jit_function_cache、_set_jit_overload_cache、
# _try_get_jit_cached_function和_try_get_jit_cached_overloads别名
from torch.jit._state import (
    _enabled as _enabled,
    _set_jit_function_cache as _set_jit_function_cache,
    _set_jit_overload_cache as _set_jit_overload_cache,
    _try_get_jit_cached_function as _try_get_jit_cached_function,
    _try_get_jit_cached_overloads as _try_get_jit_cached_overloads,
)

# 从torch.jit.frontend中导入get_default_args、get_jit_class_def和get_jit_def函数别名
from torch.jit.frontend import (
    get_default_args as get_default_args,
    get_jit_class_def as get_jit_class_def,
    get_jit_def as get_jit_def,
)

# 从torch.nn中导入Module类别名
from torch.nn import Module as Module

# 从torch.overrides中导入has_torch_function、has_torch_function_unary和has_torch_function_variadic函数别名
from torch.overrides import (
    has_torch_function as has_torch_function,
    has_torch_function_unary as has_torch_function_unary,
    has_torch_function_variadic as has_torch_function_variadic,
)

# 从torch.package中导入PackageExporter和PackageImporter类别名
from torch.package import (
    PackageExporter as PackageExporter,
    PackageImporter as PackageImporter,
)

# 从torch.utils中导入set_module函数别名
from torch.utils import set_module as set_module

# 定义torch._C.ScriptFunction类型别名
ScriptFunction = torch._C.ScriptFunction

# 定义JitTypeTraceStore类型的type_trace_db变量
type_trace_db: JitTypeTraceStore

# 定义ResolutionCallback类型别名，表示解析回调函数
# 该类型是一个接收字符串参数并返回Callable[..., Any]类型的函数
ResolutionCallback: TypeAlias = Callable[[str], Callable[..., Any]]

# 定义_ClassVar类型变量，限定为type类型
_ClassVar = TypeVar("_ClassVar", bound=type)

# 定义_reduce函数，参数为cls，返回None
def _reduce(cls) -> None: ...

# 定义Attribute类，继承自NamedTuple，包含value和type属性
class Attribute(NamedTuple):
    value: Incomplete
    type: Incomplete

# 定义_get_type_trace_db函数，返回类型为JitTypeTraceStore的类型追踪数据库
def _get_type_trace_db(): ...

# 定义_get_function_from_type函数，接受cls和name参数，返回函数对象
def _get_function_from_type(cls, name): ...

# 定义_is_new_style_class函数，接受cls参数，判断是否为新式类
def _is_new_style_class(cls): ...

# 定义OrderedDictWrapper类，用于包装有序字典
class OrderedDictWrapper:
    _c: Incomplete

    # 初始化方法，接受_c参数
    def __init__(self, _c) -> None: ...

    # keys方法，返回所有键
    def keys(self): ...

    # values方法，返回所有值
    def values(self): ...

    # __len__方法，返回字典长度
    def __len__(self) -> int: ...

    # __delitem__方法，删除指定键的项
    def __delitem__(self, k) -> None: ...

    # items方法，返回所有键值对
    def items(self): ...

    # __setitem__方法，设置指定键的值
    def __setitem__(self, k, v) -> None: ...

    # __contains__方法，检查是否包含指定键
    def __contains__(self, k) -> bool: ...

    # __getitem__方法，获取指定键的值
    def __getitem__(self, k): ...

# 定义OrderedModuleDict类，继承自OrderedDictWrapper，用于包装有序模块字典
class OrderedModuleDict(OrderedDictWrapper):
    _python_modules: Incomplete

    # 初始化方法，接受module和python_dict参数
    def __init__(self, module, python_dict) -> None: ...

    # items方法，返回所有键值对
    def items(self): ...

    # __contains__方法，检查是否包含指定键
    def __contains__(self, k) -> bool: ...

    # __setitem__方法，设置指定键的值
    def __setitem__(self, k, v) -> None: ...

    # __getitem__方法，获取指定键的值
    def __getitem__(self, k): ...
class ScriptMeta(type):
    # 元类 ScriptMeta，用于创建脚本类的元类
    def __init__(cls, name, bases, attrs) -> None:
        # 初始化方法，设置元类的基本属性

class _CachedForward:
    # 用于缓存的类 _CachedForward

    def __get__(self, obj, cls):
        # 实现描述符协议的 __get__ 方法，支持实例和类调用时的行为

class ScriptWarning(Warning):
    # 自定义的警告类 ScriptWarning，继承自内置的 Warning 类

def script_method(fn):
    # 装饰器函数 script_method，用于标记方法为脚本方法

class ConstMap:
    # 常量映射类 ConstMap

    const_mapping: Incomplete
    # 常量映射属性 const_mapping，类型为 Incomplete

    def __init__(self, const_mapping) -> None:
        # 初始化方法，接受常量映射作为参数

    def __getattr__(self, attr):
        # 实现 __getattr__ 方法，用于获取属性

def unpackage_script_module(
    importer: PackageImporter,
    script_module_id: str,
) -> torch.nn.Module:
    # 解包脚本模块的函数 unpackage_script_module

_magic_methods: Incomplete
# 魔术方法集合 _magic_methods，类型为 Incomplete

class RecursiveScriptClass:
    # 递归脚本类 RecursiveScriptClass

    _c: Incomplete
    # 属性 _c，类型为 Incomplete

    _props: Incomplete
    # 属性 _props，类型为 Incomplete

    def __init__(self, cpp_class) -> None:
        # 初始化方法，接受 C++ 类作为参数

    def __getattr__(self, attr):
        # 实现 __getattr__ 方法，用于获取属性

    def __setattr__(self, attr, value) -> None:
        # 实现 __setattr__ 方法，用于设置属性值

    def forward_magic_method(self, method_name, *args, **kwargs):
        # 执行特定的前向魔术方法

    def __getstate__(self) -> None:
        # 获取对象的状态信息

    def __iadd__(self, other):
        # 实现增量加法运算符 +=

def method_template(self, *args, **kwargs):
    # 方法模板函数，接受任意数量的位置参数和关键字参数

class ScriptModule(Module, metaclass=ScriptMeta):
    # 脚本模块类 ScriptModule，继承自 Module 类，使用 ScriptMeta 元类

    __jit_unused_properties__: Incomplete
    # 属性 __jit_unused_properties__，类型为 Incomplete

    def __init__(self) -> None:
        # 初始化方法

    forward: Callable[..., Any]
    # 前向方法 forward，类型为 Callable，接受任意类型的参数并返回任意类型的值

    def __getattr__(self, attr):
        # 实现 __getattr__ 方法，用于获取属性

    def __setattr__(self, attr, value) -> None:
        # 实现 __setattr__ 方法，用于设置属性值

    def define(self, src):
        # 定义方法，接受脚本代码作为参数

    def _replicate_for_data_parallel(self):
        # 为数据并行复制方法

    def __reduce_package__(self, exporter: PackageExporter):
        # 减少打包方法，接受 PackageExporter 对象作为参数

    @property
    def code(self) -> str:
        # 属性方法 code，返回字符串类型的代码

    @property
    def code_with_constants(self) -> tuple[str, ConstMap]:
        # 属性方法 code_with_constants，返回包含代码和常量映射的元组

    @property
    def graph(self) -> torch.Graph:
        # 属性方法 graph，返回 torch.Graph 对象

    @property
    def inlined_graph(self) -> torch.Graph:
        # 内联图属性方法，返回 torch.Graph 对象

    @property
    def original_name(self) -> str:
        # 原始名称属性方法，返回字符串类型的名称

class RecursiveScriptModule(ScriptModule):
    # 递归脚本模块类 RecursiveScriptModule，继承自 ScriptModule 类

    _disable_script_meta: bool
    # 禁用脚本元信息属性 _disable_script_meta，布尔类型

    _c: Incomplete
    # 属性 _c，类型为 Incomplete

    def __init__(self, cpp_module) -> None:
        # 初始化方法，接受 C++ 模块作为参数

    @staticmethod
    def _construct(cpp_module, init_fn):
        # 静态方法，用于构建对象，接受 C++ 模块和初始化函数作为参数

    @staticmethod
    def _finalize_scriptmodule(script_module) -> None:
        # 静态方法，用于最终化脚本模块，接受脚本模块作为参数

    _concrete_type: Incomplete
    # 具体类型属性 _concrete_type，类型为 Incomplete

    _modules: Incomplete
    # 模块属性 _modules，类型为 Incomplete

    _parameters: Incomplete
    # 参数属性 _parameters，类型为 Incomplete

    _buffers: Incomplete
    # 缓冲区属性 _buffers，类型为 Incomplete

    __dict__: Incomplete
    # 字典属性 __dict__，类型为 Incomplete

    def _reconstruct(self, cpp_module) -> None:
        # 重构方法，接受 C++ 模块作为参数

    def save(self, f, **kwargs):
        # 保存方法，接受文件对象和关键字参数

    def _save_for_lite_interpreter(self, *args, **kwargs):
        # 为轻量级解释器保存方法，接受任意数量的位置参数和关键字参数

    def _save_to_buffer_for_lite_interpreter(self, *args, **kwargs):
        # 为轻量级解释器保存到缓冲区方法，接受任意数量的位置参数和关键字参数

    def save_to_buffer(self, *args, **kwargs):
        # 保存到缓冲区方法，接受任意数量的位置参数和关键字参数

    def get_debug_state(self, *args, **kwargs):
        # 获取调试状态方法，接受任意数量的位置参数和关键字参数

    def extra_repr(self):
        # 额外表示方法

    def graph_for(self, *args, **kwargs):
        # 获取图方法，接受任意数量的位置参数和关键字参数

    def define(self, src) -> None:
        # 定义方法，接受脚本代码作为参数

    def __getattr__(self, attr):
        # 实现 __getattr__ 方法，用于获取属性

    def __setattr__(self, attr, value) -> None:
        # 实现 __setattr__ 方法，用于设置属性值

    def __copy__(self):
        # 复制方法

    def __deepcopy__(self, memo):
        # 深度复制方法

    def forward_magic_method(self, method_name, *args, **kwargs):
        # 执行特定的前向魔术方法

    def __iter__(self):
        # 迭代器方法

    def __getitem__(self, idx):
        # 获取项目方法，接受索引参数

    def __len__(self) -> int:
        # 获取长度方法，返回整数类型

    def __contains__(self, key) -> bool:
        # 包含方法，接受键参数，返回布尔类型

    def __dir__(self):
        # 获取属性名称列表方法

    def __bool__(self) -> bool:
        # 布尔转换方法，返回布尔类型

    def _replicate_for_data_parallel(self):
        # 为数据并行复制方法

def _get_methods(cls):
    # 获取类方法集合的函数 _get_methods
# _compiled_methods_allowlist: Incomplete
# 定义一个名为 _compiled_methods_allowlist 的变量，但其具体内容未完全列出或定义

def _make_fail(name):
    # 定义一个函数 _make_fail，接受一个参数 name，但函数体未完全列出或定义
    ...

def call_prepare_scriptable_func_impl(obj, memo):
    # 定义一个函数 call_prepare_scriptable_func_impl，接受两个参数 obj 和 memo，但函数体未完全列出或定义
    ...

def call_prepare_scriptable_func(obj):
    # 定义一个函数 call_prepare_scriptable_func，接受一个参数 obj，但函数体未完全列出或定义
    ...

def create_script_dict(obj):
    # 定义一个函数 create_script_dict，接受一个参数 obj，但函数体未完全列出或定义
    ...

def create_script_list(obj, type_hint: Incomplete | None = ...):
    # 定义一个函数 create_script_list，接受一个参数 obj 和一个可选的 type_hint 参数，
    # 但函数体未完全列出或定义

@overload
def script(
    obj: type[Module],
    optimize: bool | None = None,
    _frames_up: int = 0,
    _rcb: ResolutionCallback | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> Never:
    # 装饰器标记的函数重载，接受一个 type[Module] 类型的 obj 参数和多个可选参数，
    # 返回类型为 Never，但函数体未完全列出或定义

@overload
def script(  # type: ignore[misc]
    obj: dict,
    optimize: bool | None = None,
    _frames_up: int = 0,
    _rcb: ResolutionCallback | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> torch.ScriptDict:
    # 装饰器标记的函数重载，接受一个 dict 类型的 obj 参数和多个可选参数，
    # 返回类型为 torch.ScriptDict，但函数体未完全列出或定义

@overload
def script(  # type: ignore[misc]
    obj: list,
    optimize: bool | None = None,
    _frames_up: int = 0,
    _rcb: ResolutionCallback | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> torch.ScriptList:
    # 装饰器标记的函数重载，接受一个 list 类型的 obj 参数和多个可选参数，
    # 返回类型为 torch.ScriptList，但函数体未完全列出或定义

@overload
def script(  # type: ignore[misc]
    obj: Module,
    optimize: bool | None = None,
    _frames_up: int = 0,
    _rcb: ResolutionCallback | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> RecursiveScriptModule:
    # 装饰器标记的函数重载，接受一个 Module 类型的 obj 参数和多个可选参数，
    # 返回类型为 RecursiveScriptModule，但函数体未完全列出或定义

@overload
def script(  # type: ignore[misc]
    obj: _ClassVar,
    optimize: bool | None = None,
    _frames_up: int = 0,
    _rcb: ResolutionCallback | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> _ClassVar:
    # 装饰器标记的函数重载，接受一个 _ClassVar 类型的 obj 参数和多个可选参数，
    # 返回类型为 _ClassVar，但函数体未完全列出或定义

@overload
def script(  # type: ignore[misc]
    obj: Callable,
    optimize: bool | None = None,
    _frames_up: int = 0,
    _rcb: ResolutionCallback | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> ScriptFunction:
    # 装饰器标记的函数重载，接受一个 Callable 类型的 obj 参数和多个可选参数，
    # 返回类型为 ScriptFunction，但函数体未完全列出或定义

@overload
def script(
    obj: Any,
    optimize: bool | None = None,
    _frames_up: int = 0,
    _rcb: ResolutionCallback | None = None,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = None,
) -> RecursiveScriptClass:
    # 装饰器标记的函数重载，接受一个 Any 类型的 obj 参数和多个可选参数，
    # 返回类型为 RecursiveScriptClass，但函数体未完全列出或定义

@overload
def script(
    obj,
    optimize: Incomplete | None = ...,
    _frames_up: int = ...,
    _rcb: Incomplete | None = ...,
    example_inputs: list[tuple] | dict[Callable, list[tuple]] | None = ...,
):
    # 装饰器标记的函数重载，接受一个不明确类型的 obj 参数和多个可选参数，
    # 但函数体未完全列出或定义

def _check_overload_defaults(impl_defaults, overload_defaults, loc) -> None:
    # 定义一个函数 _check_overload_defaults，接受三个参数 impl_defaults, overload_defaults 和 loc，
    # 返回类型为 None，但函数体未完全列出或定义

def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    # 定义一个函数 _compile_function_with_overload，接受三个参数 overload_fn, qual_name 和 impl_fn，
    # 但函数体未完全列出或定义

def _get_overloads(obj):
    # 定义一个函数 _get_overloads，接受一个参数 obj，但函数体未完全列出或定义

def _check_directly_compile_overloaded(obj) -> None:
    # 定义一个函数 _check_directly_compile_overloaded，接受一个参数 obj，
    # 返回类型为 None，但函数体未完全列出或定义

def interface(obj):
    # 定义一个函数 interface，接受一个参数 obj，但函数体未完全列出或定义

def _recursive_compile_class(obj, loc):
    # 定义一个函数 _recursive_compile_class，接受两个参数 obj 和 loc，但函数体未完全列出或定义

# CompilationUnit: Incomplete
# 定义一个类 CompilationUnit，但其具体内容未完全列出或定义

def pad(s: str, padding: int, offset: int = ..., char: str = ...):
    # 定义一个函数 pad，接受四个参数 s, padding, offset 和 char，
    # 其中 offset 和 char 参数有默认值，但函数体未完全列出或定义

class _ScriptProfileColumn:
    header: Incomplete
    alignment: Incomplete
    offset: Incomplete
    rows: Incomplete

    def __init__(
        self,
        header: str,
        alignment: int = ...,
        offset: int = ...,
    ) -> None:
        # 定义一个类 _ScriptProfileColumn，包含四个属性 header, alignment, offset 和 rows，
        # 这些属性的具体内容未完全列出或定义
        # 定义类的初始化方法 __init__，接受三个必选参数 header, alignment 和 offset，
        # 返回类型为 None
        ...
    # 定义一个方法 `add_row`，接受两个参数 `lineno`（行号，整数类型）和 `value`（值，任意类型）
    def add_row(self, lineno: int, value: Any):
        ...
    
    # 定义一个方法 `materialize`，没有参数
    def materialize(self):
        ...
class _ScriptProfileTable:
    # 定义一个名为 _ScriptProfileTable 的类，用于表示脚本性能分析表格
    cols: Incomplete
    # 类属性 cols，类型未完全定义
    source_range: Incomplete
    # 类属性 source_range，类型未完全定义

    def __init__(
        self,
        cols: list[_ScriptProfileColumn],
        source_range: list[int],
    ) -> None:
        # _ScriptProfileTable 类的初始化方法，接受两个参数：
        # - cols: 一个 _ScriptProfileColumn 对象组成的列表
        # - source_range: 一个整数列表
        ...

    def dump_string(self):
        # 输出该对象的字符串表示，具体功能未详细说明
        ...


class _ScriptProfile:
    # 定义一个名为 _ScriptProfile 的类，用于表示脚本性能分析
    profile: Incomplete
    # 类属性 profile，类型未完全定义

    def __init__(self) -> None:
        # _ScriptProfile 类的初始化方法，没有参数
        ...

    def enable(self) -> None:
        # 启用脚本性能分析，具体实现未详细说明
        ...

    def disable(self) -> None:
        # 禁用脚本性能分析，具体实现未详细说明
        ...

    def dump_string(self) -> str:
        # 返回该对象的字符串表示，具体功能未详细说明
        ...

    def dump(self) -> None:
        # 输出该对象的内容或状态，具体功能未详细说明
        ...


def _unwrap_optional(x):
    # 定义一个名为 _unwrap_optional 的函数，接受一个参数 x
    # 用途是解开可能包装的可选值，具体实现未详细说明
    ...
```