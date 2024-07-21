# `.\pytorch\torch\jit\_script.py`

```py
"""TorchScript.

This module contains functionality to support the JIT's scripting frontend, notably:
    - torch.jit.script

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
# 导入所需的模块和函数
import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch._utils_internal import log_torchscript_usage
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for

from torch.jit._monkeytype_config import (
    JitTypeTraceConfig,
    JitTypeTraceStore,
    monkeytype_trace,
)
from torch.jit._recursive import (
    _compile_and_register_class,
    infer_methods_to_compile,
    ScriptMethodStub,
    wrap_cpp_module,
)
from torch.jit._state import (
    _enabled,
    _set_jit_function_cache,
    _set_jit_overload_cache,
    _try_get_jit_cached_function,
    _try_get_jit_cached_overloads,
)
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module

# 创建用于存储 MonkeyType 的调用跟踪的数据库
type_trace_db = JitTypeTraceStore()  # DB to hold all call traces from MonkeyType

# 将 graph_for 函数注册到 ScriptMethod 上
torch._C.ScriptMethod.graph_for = _script_method_graph_for  # type: ignore[attr-defined]
# 将 graph_for 函数注册到 ScriptFunction 上
torch._C.ScriptFunction.graph_for = _graph_for  # type: ignore[attr-defined]
ScriptFunction = torch._C.ScriptFunction
# 设置 ScriptFunction 的文档字符串
ScriptFunction.__doc__ = """
Functionally equivalent to a :class:`ScriptModule`, but represents a single
function and does not have any attributes or Parameters.
"""
# 将 ScriptFunction 设置到 torch.jit 模块中
set_module(ScriptFunction, "torch.jit")


# 如果尝试对 JIT 函数进行 pickle 操作，则抛出错误
def _reduce(cls):
    raise pickle.PickleError("ScriptFunction cannot be pickled")


# 设置 ScriptFunction 的 reduce 方法
ScriptFunction.__reduce__ = _reduce  # type: ignore[assignment]

# 如果 JIT 功能已启用，则定义 Attribute 类，否则定义一个具有相同行为的函数
if _enabled:
    Attribute = collections.namedtuple("Attribute", ["value", "type"])
else:
    # 如果 JIT 功能未启用，则定义 Attribute 函数
    def Attribute(value, type):  # type: ignore[no-redef]
        return value

# 设置 Attribute 函数的文档字符串
Attribute.__doc__ = """
    This method is a pass-through function that returns `value`, mostly
    used to indicate to the TorchScript compiler that the left-hand side
    expression is a class instance attribute with type of `type`. Note that
    `torch.jit.Attribute` should only be used in `__init__` method of `jit.ScriptModule`
    subclasses.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:
``` 
    # 空容器（如 `[]` 和 `{}`），TorchScript 假设它们包含 `Tensor` 类型的容器
    # 可选类型（如 `Optional[T]`），如果赋予了类型 `T` 的有效值，TorchScript 会假设它是类型 `T` 而不是 `Optional[T]`
    
    # 在 eager 模式下，这只是一个简单的传递函数，返回 `value`，没有其他影响。
    
    # 示例：
    
    # 导入 Torch 和字典类型
    import torch
    from typing import Dict
    
    # 定义一个继承自 ScriptModule 的 AttributeModule 类
    class AttributeModule(torch.jit.ScriptModule):
        def __init__(self):
            super().__init__()
            
            # 创建一个名为 foo 的 TorchScript 属性，初始值为 0.1，类型为 float
            self.foo = torch.jit.Attribute(0.1, float)
            
            # 可以在这里使用 self.foo 作为一个 float 类型
            assert 0.0 < self.foo
            
            # 创建一个名为 names_ages 的 TorchScript 属性，初始值为一个空字典，类型为 Dict[str, int]
            self.names_ages = torch.jit.Attribute({}, Dict[str, int])
            self.names_ages["someone"] = 20
            assert isinstance(self.names_ages["someone"], int)
    
    # 创建 AttributeModule 实例 m
    m = AttributeModule()
    
    # m 包含两个属性：
    # 1. 类型为 float 的 foo
    # 2. 类型为 Dict[str, int] 的 names_ages
    
    # 清理实例和类定义
    del AttributeModule
    del m
    
    # 注意：现在更推荐使用类型注释而不是 `torch.jit.Attribute`：
    
    # 示例：
    
    # 导入 Torch 和字典类型
    import torch
    from typing import Dict
    
    # 定义一个继承自 nn.Module 的 AttributeModule 类
    class AttributeModule(torch.nn.Module):
        # 定义一个名为 names 的类型为 Dict[str, int] 的属性
        names: Dict[str, int]
        
        def __init__(self):
            super().__init__()
            # 初始化 names 属性为一个空字典
            self.names = {}
    
    # 创建 AttributeModule 实例 m
    m = AttributeModule()
    
    # 清理实例和类定义
    del AttributeModule
    del m
    
    # 参数：
    # value: 要分配给属性的初始值
    # type: 一个 Python 类型
    
    # 返回：
    # 返回 `value`
# 获取类型跟踪数据库的函数
def _get_type_trace_db():
    # 这是一个私有API，不建议外部使用。
    return type_trace_db


# 根据类型和方法名获取函数对象
def _get_function_from_type(cls, name):
    return getattr(cls, name, None)


# 判断类是否是新式类
# 新式类必须具有 '__dict__' 属性或者定义了 '__slots__'
def _is_new_style_class(cls):
    if hasattr(cls, "__class__"):
        return "__dict__" in dir(cls) or hasattr(cls, "__slots__")


# 这些 OrderedDictWrapper 类用于替换模块中的实际 OrderedDict，
# 使得我们可以在 C++ 中存储数据的同时，重用大部分 nn.Module 的功能。
# 每个 OrderedDict 需要支持以下操作：
#  x not in view
#  x in view
#  view[name] = ...
#  view.values()
#  del view[name]
#  view.items()
#  view.keys()
#  len(view)
class OrderedDictWrapper:
    def __init__(self, _c):
        self._c = _c

    # 返回包装对象中的键列表
    def keys(self):
        return [k for k, v in self.items()]

    # 返回包装对象中的值列表
    def values(self):
        return [v for k, v in self.items()]

    # 返回包装对象中的元素数量
    def __len__(self):
        return len(self.values())

    # 禁止删除脚本模块的方法或参数
    def __delitem__(self, k):
        raise RuntimeError("cannot delete methods or parameters of a script module")

    # 返回包装对象中的元素键值对列表
    def items(self):
        return self._c.items()

    # 向包装对象中添加新的键值对，如果键已存在则抛出异常
    def __setitem__(self, k, v):
        if k not in self:
            raise RuntimeError(
                f"Can't add a new parameter after ScriptModule construction. Tried to add '{k}"
            )
        self._c.setattr(k, v)

    # 检查包装对象是否包含指定键
    def __contains__(self, k):
        return self._c.contains(k)

    # 根据键获取包装对象中的值
    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self._c.getattr(k)


# 继承自 OrderedDictWrapper 的 OrderedModuleDict 类
class OrderedModuleDict(OrderedDictWrapper):
    def __init__(self, module, python_dict):
        super().__init__(torch._C.ModuleDict(module))
        # 包含脚本模块和非脚本 Python 模块的字典

        # 因为脚本模块在 Python 中是子类化的，
        # C++ 模块类不会持有对它们的引用，
        # 为了确保在此处始终获取相同的 Python 值，
        # 我们也将其存储在 Python 字典中。
        self._python_modules = python_dict

    # 返回 Python 模块字典的键值对列表
    def items(self):
        r = self._python_modules.items()
        return r

    # 检查 Python 模块字典是否包含指定键
    def __contains__(self, k):
        return k in self._python_modules
    # 定义特殊方法 __setitem__，用于在 ScriptModule 中设置键值对
    def __setitem__(self, k, v):
        # 情况说明：允许在 ScriptModule 构建后重新分配子模块的情况
        # 1. 如果属性是一个模块接口类型，保证模块不会内联在图中，可以安全地交换一个新的 ScriptModule 进去。
        # 2. 如果新值是具有相同 JIT 类型的 ScriptModule，则 IR 不会改变，可以合法地交换一个新模块进去。
        # 在这两种情况下，允许交换一个新的脚本化模块，并更新相应的 Python 模块字典以保持同步。
        # 注意：要交换的值必须是 ScriptModule 而不是 nn.Module，否则是非法的，会抛出错误。
        if isinstance(v, ScriptModule):
            # 调用底层 C++ 接口设置属性 k 为 ScriptModule v
            self._c.setattr(k, v)
            # 更新 Python 模块字典，保持键 k 对应的值为 v
            self._python_modules[k] = v
        else:
            # 如果尝试用非脚本化模块替换现有模块，抛出运行时错误
            raise RuntimeError(
                "Cannot re-assign modules in a ScriptModule with non-scripted "
                f"module, tried to replace existing module '{k}': {v}"
            )

    # 定义特殊方法 __getitem__，用于在 ScriptModule 中获取键 k 对应的值
    def __getitem__(self, k):
        return self._python_modules[k]
# 对于每个继承 ScriptModule 的用户定义类，这个元类：
# (1) 查找所有标记为 @script_method 的方法，并从类属性中移除它们
# (2) 在类的 __init__ 方法周围放置一个包装器，用于在原始 __init__ 方法运行后递归地编译模块中的所有 script_method。
#     这必须发生在用户定义的 __init__ 方法之后，以便在脚本编译器解析 `self.param` 或 `self.module` 的引用之前初始化子模块和参数。
class ScriptMeta(type):
    def __init__(cls, name, bases, attrs):  # noqa: B902
        # 初始化函数，用于元类在创建类时进行初始化
        cls._methods: Dict[str, Any] = {}  # 初始化一个空字典，用于存储类的方法
        cls._constants_set = set(getattr(cls, "__constants__", ()))  # 获取类的常量集合或为空集合
        # 遍历所有父类，将父类的方法和常量集合聚合到当前类中
        for base in reversed(bases):
            for k, v in getattr(base, "_methods", {}).items():
                cls._methods[k] = v
            base_constants: Set = getattr(base, "_constants_set", set())
            cls._constants_set = cls._constants_set.union(base_constants)

        # 查找当前类中的所有脚本方法（ScriptMethod）
        for k, v in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)  # 删除原始属性
                cls._methods[v.original_method.__name__] = v  # 将脚本方法添加到_methods字典中

        if getattr(cls, "_disable_script_meta", False):
            # 如果禁用了脚本元信息，则返回父类的初始化方法
            return super().__init__(name, bases, attrs)

        original_init = getattr(cls, "__init__", lambda self: None)

        @functools.wraps(original_init)
        def init_then_script(self, *args, **kwargs):
            num_methods = len(cls._methods)
            original_init(self, *args, **kwargs)  # 调用原始的初始化方法
            added_methods_in_init = len(cls._methods) > num_methods

            if type(self) == cls:
                # 定义一个函数，用于创建脚本方法的存根
                def make_stubs(module):
                    cls = type(module)
                    if hasattr(cls, "_methods"):
                        return [v for k, v in sorted(cls._methods.items())]
                    else:
                        return infer_methods_to_compile(module)

                # 创建实际的脚本模块
                self.__dict__["_actual_script_module"] = torch.jit._recursive.create_script_module(
                    self, make_stubs, share_types=not added_methods_in_init
                )

                # 删除Python属性，以便__getattr__和__setattr__能够正确找到脚本版本
                concrete_type = self._actual_script_module._concrete_type
                for name in concrete_type.get_attributes():
                    delattr(self, name)
                for name, _ in concrete_type.get_modules():
                    delattr(self, name)
                for name in ("_parameters", "_buffers", "_modules"):
                    delattr(self, name)

        cls.__init__ = init_then_script  # 将初始化方法设置为新定义的init_then_script函数
        super().__init__(name, bases, attrs)  # 调用父类的初始化方法
class _CachedForward:
    # 定义一个特殊描述符类，用于延迟获取对象的'forward'属性
    def __get__(self, obj, cls):
        # 返回当前对象的'forward'属性，忽略类型检查提示
        return self.__getattr__("forward")  # type: ignore[attr-defined]


class ScriptWarning(Warning):
    # 定义一个空的警告类 ScriptWarning
    pass


def script_method(fn):
    # 如果未启用 _enabled，则直接返回原始函数 fn
    if not _enabled:
        return fn
    # 注意：在这里我们需要遍历两个帧，因为在 ScriptModule 的元类帧存在，
    # 而不是在函数上使用 @script 或在 CompilationUnit 上调用 define()。
    # 调用栈的结构如下：
    #
    # 0. createResolutionCallback()
    # 1. script_method()
    # 2. ScriptModule 的元类帧
    # 3. 包围作用域
    #
    # createResolutionCallback 内部添加 1 来到达调用函数的作用域（这个函数）。
    # 添加 2 可以到达正确的包围作用域。
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    # 获取 fn 函数的 JIT 定义语法树
    ast = get_jit_def(fn, fn.__name__, self_name="ScriptModule")
    # 返回一个 ScriptMethodStub 对象，包含解析回调和 JIT 定义的语法树 ast
    return ScriptMethodStub(_rcb, ast, fn)


class ConstMap:
    def __init__(self, const_mapping):
        # 初始化函数，接受一个常量映射 const_mapping
        self.const_mapping = const_mapping

    def __getattr__(self, attr):
        # 获取指定属性 attr 对应的常量映射值
        return self.const_mapping[attr]


def unpackage_script_module(
    importer: PackageImporter, script_module_id: str
) -> torch.nn.Module:
    """
    由 ``torch.package.PackageImporter`` 的 Pickler 的 ``persistent_load`` 函数调用。

    执行从 ``torch.package`` 存档中加载并返回一个 ScriptModule 的工作。
    """
    # 如果 importer.zip_reader 不是 torch._C.PyTorchFileReader 的实例，则抛出运行时错误
    if not isinstance(importer.zip_reader, torch._C.PyTorchFileReader):
        raise RuntimeError(
            "Loading ScriptObjects from a PackageImporter created from a "
            "directory is not supported. Use a package archive file instead."
        )
    # 创建一个 CompilationUnit 对象 cu
    cu = torch._C.CompilationUnit()
    # 从 package 中导入 IR 模块 cpp_module
    cpp_module = torch._C._import_ir_module_from_package(
        cu,
        importer.zip_reader,
        importer.storage_context,
        validate_map_location(importer.last_map_location),
        script_module_id,
    )
    # 将 cpp_module 封装成 Python 的 ScriptModule 并返回
    return wrap_cpp_module(cpp_module)


if _enabled:
    # 如果 _enabled 被启用，则定义一个魔法方法列表 _magic_methods
    _magic_methods = [
        "__iter__",
        "__len__",
        "__neg__",
        "__mul__",
        "__contains__",
        "__add__",
        "__sub__",
        "__pow__",
        "__truediv__",
        "__mod__",
        "__ne__",
        "__eq__",
        "__lt__",
        "__gt__",
        "__le__",
        "__ge__",
        "__and__",
        "__or__",
        "__xor__",
        "__getitem__",
        "__setitem__",
        "__call__",
        "__int__",
        "__float__",
        "__bool__",
        "__str__",
        "__enter__",
        "__exit__",
    ]
    class RecursiveScriptClass:
        """Wrapper for a TorchScript class instance for use in Python.

        An analogue of RecursiveScriptModule for regular objects that are not modules.
        This class is a wrapper around a torch._C.ScriptObject that represents an instance
        of a TorchScript class and allows it to be used in Python.

        Attributes:
            _c [torch._C.ScriptObject]: The C++ object to which attribute lookups and method
                calls are forwarded.
            _props [Dict[str, property]]: A dictionary of properties fetched from self._c and
                exposed on this wrapper.
        """

        def __init__(self, cpp_class):
            super().__init__()
            self.__dict__["_initializing"] = True
            self._c = cpp_class

            # Add wrapped object's properties to this class instance.
            self._props = {
                prop.name: property(prop.getter, prop.setter)
                for prop in self._c._properties()
            }

            self.__dict__["_initializing"] = False

        def __getattr__(self, attr):
            if self.__dict__.get("_initializing"):
                return super().__getattr__(attr)  # type: ignore[misc]

            if attr in self._props:
                return self._props[attr].fget()  # type: ignore[call-arg, misc]

            return getattr(self._c, attr)

        def __setattr__(self, attr, value):
            if self.__dict__.get("_initializing"):
                return super().__setattr__(attr, value)

            if attr in self._props:
                return self._props[attr].fset(value)  # type: ignore[call-arg, misc]

            setattr(self._c, attr, value)

        # Delegate calls to magic methods like __len__ to the C++ module backing the
        # RecursiveScriptClass.
        def forward_magic_method(self, method_name, *args, **kwargs):
            if not self._c._has_method(method_name):
                raise TypeError

            self_method = self.__getattr__(method_name)
            return self_method(*args, **kwargs)

        def __getstate__(self):
            raise pickle.PickleError("ScriptClasses cannot be pickled")

        def __iadd__(self, other):
            if self._c._has_method("__iadd__"):
                return self.forward_magic_method("__iadd__", other)
            else:
                return self.forward_magic_method("__add__", other)

    # Define a list of magic method names to be added as methods to RecursiveScriptClass
    for method_name in _magic_methods:
        
        # Define a method template that forwards calls to the respective magic method
        def method_template(self, *args, **kwargs):
            return self.forward_magic_method(method_name, *args, **kwargs)
        
        # Attach the method_template function as a method with method_name to RecursiveScriptClass
        setattr(RecursiveScriptClass, method_name, method_template)

    # this is a Python 'non-data descriptor' that causes the first access
    # to ScriptModule's forward to look up the forward method and stash
    # it in the objects dict. Due to the standard rules for attribute lookup,
    # subsequent lookups will just directly return the previously looked up method.
    # 这是必要的，因为 nn.Module 将 forward 定义为一个方法。如果我们什么都不做，
    # __getattr__ 将不会被调用。相反，我们会得到 nn.Module.forward
    # 这总是会抛出异常的。

    # 需要将所有 RecursiveScriptModule 的方法复制到 ScriptModule 中。
    #
    # 这是因为 `super().foo()` 不会使用 `__getattr__` 查找 `foo`。
    # 所以我们需要手动在 ScriptModule 上使每个方法可用。
    for name, item in RecursiveScriptModule.__dict__.items():
        if not callable(item) and not isinstance(item, property):
            continue
        if name.startswith("__") or hasattr(ScriptModule, name):
            continue
        # 我们可以整体复制实现，因为除了上面的 `super()` 问题之外，ScriptModule
        # 的行为与 RecursiveScriptModule 完全相同。
        setattr(ScriptModule, name, item)

    def _get_methods(cls):
        import inspect

        # 在 Python 3 中，未绑定的方法是函数，但在 Python 2 中它们是方法
        return inspect.getmembers(
            cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)
        )

    _compiled_methods_allowlist = {
        "forward",
        "register_buffer",
        "register_parameter",
        "register_module",
        "add_module",
        "_apply",
        "apply",
        "cuda",
        "cpu",
        "to",
        "type",
        "float",
        "double",
        "half",
        "state_dict",
        "_save_to_state_dict",
        "load_state_dict",
        "_load_from_state_dict",
        "_named_members",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "children",
        "named_children",
        "modules",
        "named_modules",
        "zero_grad",
        "share_memory",
        "_get_name",
        "extra_repr",
        "_slow_forward",
        "_tracing_name",
        "eval",
        "train",
        "get_extra_state",
        "set_extra_state",
    }

    def _make_fail(name):
        def fail(self, *args, **kwargs):
            raise RuntimeError(name + " is not supported on ScriptModules")

        return fail

    for name, method in _get_methods(torch.nn.Module):
        if name.startswith("__") or name.endswith("_call_impl"):
            continue
        if (
            name not in RecursiveScriptModule.__dict__
            and name not in _compiled_methods_allowlist
        ):
            setattr(RecursiveScriptModule, method.__name__, _make_fail(name))
else:
    # 如果不是上述情况，则进行以下操作

    # TODO MAKE SURE THAT DISABLING WORKS
    # 定义一个递归脚本类，用于忽略类型检查的重定义警告
    class RecursiveScriptClass:  # type: ignore[no-redef]
        pass

    # 定义一个脚本模块类，用于忽略类型检查的重定义警告
    class ScriptModule(torch.nn.Module):  # type: ignore[no-redef]
        def __init__(self, arg=None):
            super().__init__()

    # 定义一个递归脚本模块类，继承自ScriptModule类，用于忽略类型检查的重定义警告
    class RecursiveScriptModule(ScriptModule):  # type: ignore[no-redef]
        def __init__(self, arg=None):
            super().__init__()

def call_prepare_scriptable_func_impl(obj, memo):
    # 如果obj不是torch.nn.Module的实例，则直接返回obj
    if not isinstance(obj, torch.nn.Module):
        return obj

    # 获取obj的唯一标识符
    obj_id = id(obj)

    # 如果obj_id已经存在于memo中，则obj已经被处理过或者正在被处理
    if obj_id in memo:
        return memo[id(obj)]

    # 如果obj有__prepare_scriptable__方法，则调用该方法，否则直接返回obj
    obj = obj.__prepare_scriptable__() if hasattr(obj, "__prepare_scriptable__") else obj  # type: ignore[operator]

    # 将obj记录在memo中，避免在模块层次结构中出现循环时导致无限递归
    memo[obj_id] = obj

    new_obj_dict = {}

    # 遍历obj的属性字典
    for name, sub_module in obj.__dict__.items():
        # 如果属性名为"_modules"，则遍历其值中的每一个子模块
        if name == "_modules":
            for k, v in sub_module.items():
                sub_module[k] = call_prepare_scriptable_func_impl(v, memo)
            new_obj_dict[name] = sub_module
        # 如果子模块是torch.nn.Module的实例并且不是ScriptModule的实例，则递归调用call_prepare_scriptable_func_impl
        elif isinstance(sub_module, torch.nn.Module) and not isinstance(
            sub_module, ScriptModule
        ):
            new_obj_dict[name] = call_prepare_scriptable_func_impl(sub_module, memo)
        else:
            new_obj_dict[name] = sub_module

    # 将new_obj_dict中的值更新到obj的属性字典中
    for k, v in new_obj_dict.items():
        obj.__dict__[name] = v

    # 返回处理后的obj
    return obj

def call_prepare_scriptable_func(obj):
    # 定义一个memo字典，用于记录处理过的torch.nn.Module实例
    memo: Dict[int, torch.nn.Module] = {}
    # 调用call_prepare_scriptable_func_impl处理obj并返回结果
    return call_prepare_scriptable_func_impl(obj, memo)

def create_script_dict(obj):
    """
    Create a ``torch._C.ScriptDict`` instance with the data from ``obj``.

    Args:
        obj (dict): The Python dictionary that is used to initialize the ``ScriptDict``
                    returned by this function.

    Returns:
        An instance of ``torch._C.ScriptDict`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """
    # 使用obj数据创建一个torch._C.ScriptDict实例，并返回
    return torch._C.ScriptDict(obj)  # type: ignore[attr-defined]

def create_script_list(obj, type_hint=None):
    """
    Create a ``torch._C.ScriptList`` instance with the data from ``obj``.

    Args:
        obj (dict): The Python list that is used to initialize the ``ScriptList``
                    returned by this function.
    Returns:
        An instance of ``torch._C.ScriptList`` that has the same data as ``obj``
        and can be passed between Python and TorchScript with reference semantics and
        zero copy overhead.
    """
    # 使用obj数据创建一个torch._C.ScriptList实例，并返回
    return torch._C.ScriptList(obj)  # type: ignore[attr-defined]

# 定义一个全局变量_TOPLEVEL，表示当前代码处于顶层
_TOPLEVEL: bool = True

def _script_impl(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    # 声明一个变量 example_inputs，类型为 Union[List[Tuple], Dict[Callable, List[Tuple]], None]，初始值为 None
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
    global type_trace_db



    if optimize is not None:
        # 如果 optimize 参数不为空，发出警告并指出其已不再生效，建议使用 torch.jit.optimized_execution() 替代
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=3,
        )



    # 对于已经被脚本化的模块、函数、类实例，直接返回该对象
    if isinstance(obj, RecursiveScriptClass):
        return obj
    if isinstance(obj, ScriptModule):
        return obj
    if isinstance(obj, ScriptFunction):
        return obj



    if example_inputs:
        # 如果存在 example_inputs，根据 MonkeyType 是否安装来启用基于类型的注解分析
        # 检查 example_inputs 是否定义，并使用提供的示例输入运行方法的 eager 模式版本
        # 这会记录所有的调用堆栈到 type_trace_db 中
        type_trace_db = JitTypeTraceStore()
        if monkeytype_trace:
            # 配置 MonkeyType 的类型跟踪器
            monkeytype_config = JitTypeTraceConfig(type_trace_db)
            with monkeytype_trace(monkeytype_config):
                if isinstance(example_inputs, Dict):
                    # 如果 obj 是 nn.Module 或者类，则对每个方法使用提供的示例输入执行
                    # 这些示例输入的格式应为 Dict(class.method, (arguments))
                    # 用于推断那些在 MonkeyType 内部未直接调用的方法的类型注解
                    for module, example_input in example_inputs.items():
                        for example in example_input:
                            module(*example)
                elif isinstance(example_inputs, List):
                    # 如果 example_inputs 是 List 类型，则对 obj 使用提供的示例输入执行
                    for examples in example_inputs:
                        obj(*examples)
                else:
                    # 如果 example_inputs 的格式不符合预期，则抛出 ValueError
                    raise ValueError(
                        "Error: Unable to infer types. Please format the inputs to type `List[Tuple]`"
                        " or `Dict[Callable, List[Tuple]]` to be run with MonkeyType."
                    )
        else:
            # 如果未安装 MonkeyType，则发出警告
            warnings.warn(
                "Warning: monkeytype is not installed. Please install https://github.com/Instagram/MonkeyType "
                "to enable Profile-Directed Typing in TorchScript. Refer to "
                "https://github.com/Instagram/MonkeyType/blob/master/README.rst to install MonkeyType. "
            )



    if isinstance(obj, torch.nn.Module):
        # 如果 obj 是 torch.nn.Module，则调用 call_prepare_scriptable_func 函数进行准备
        obj = call_prepare_scriptable_func(obj)
        # 使用 torch.jit._recursive.create_script_module 创建脚本模块
        return torch.jit._recursive.create_script_module(
            obj, torch.jit._recursive.infer_methods_to_compile
        )
    else:
        # 如果 obj 有 __prepare_scriptable__ 方法，则调用该方法进行准备，否则直接返回 obj
        obj = obj.__prepare_scriptable__() if hasattr(obj, "__prepare_scriptable__") else obj  # type: ignore[operator]



    if isinstance(obj, dict):
        # 如果 obj 是 dict 类型，则调用 create_script_dict 函数创建脚本化的字典对象
        return create_script_dict(obj)
    if isinstance(obj, list):
        # 如果 obj 是 list 类型，则调用 create_script_list 函数创建脚本化的列表对象
        return create_script_list(obj)
    # 检查对象是否是类
    if inspect.isclass(obj):
        # 获取对象的完全限定名
        qualified_name = _qualified_name(obj)
        # 如果这个类型是 `nn.Module` 的子类，则可能意味着应该传递一个实例而不是模块
        if issubclass(obj, torch.nn.Module):
            # 抛出运行时错误，说明类型无法编译，因为它继承自 nn.Module，应传递一个实例
            raise RuntimeError(
                f"Type '{obj}' cannot be compiled since it inherits from nn.Module, pass an instance instead"
            )

        # 枚举类型在 TorchScript 中自动可用，显式脚本化不是必要的，但也不会有害
        if issubclass(obj, enum.Enum):
            return obj

        # 如果不是新式类，则抛出运行时错误，TorchScript 类必须是新式类，需要继承自 'object'
        if not _is_new_style_class(obj):
            raise RuntimeError(
                "TorchScript classes must be new-style classes. "
                "Please inherit from 'object'."
            )
        
        # 如果类的方法解析顺序超过 2 层，则抛出运行时错误，TorchScript 类不支持多重继承
        if len(obj.mro()) > 2:
            raise RuntimeError(
                "TorchScript classes does not support inheritance yet. "
                "Please directly inherit from 'object'."
            )
        
        # 如果 `_rcb` 为空，则从当前帧创建一个解析回调
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
        
        # 编译并注册该类
        _compile_and_register_class(obj, _rcb, qualified_name)
        return obj
    # 如果对象是函数或方法
    elif inspect.isfunction(obj) or inspect.ismethod(obj):
        # 获取对象的完全限定名
        qualified_name = _qualified_name(obj)
        
        # 如果这是一个装饰函数，则需要获取其原始函数及其解析回调
        if hasattr(obj, "__script_if_tracing_wrapper"):
            obj = obj.__original_fn  # type: ignore[union-attr]
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        
        # 一些函数被显式标记为在脚本模式下不支持
        if hasattr(obj, "__script_unsupported"):
            raise RuntimeError("TorchScript error: " + obj.__script_unsupported)
        
        # 检查是否可以直接编译重载的函数
        _check_directly_compile_overloaded(obj)
        
        # 尝试从 JIT 缓存中获取已编译的函数
        maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
        if maybe_already_compiled_fn:
            # 将函数标记为可以在 TorchDynamo 中内联
            maybe_already_compiled_fn._torchdynamo_inline = obj  # type: ignore[attr-defined]
            return maybe_already_compiled_fn
        
        # 获取函数的 JIT AST
        ast = get_jit_def(obj, obj.__name__)
        
        # 如果 `_rcb` 为空，则从闭包中创建一个解析回调
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        
        # 使用 TorchScript 编译函数
        fn = torch._C._jit_script_compile(
            qualified_name, ast, _rcb, get_default_args(obj)
        )
        
        # 将原始函数的文档字符串传递给编译后的函数
        fn.__doc__ = obj.__doc__
        
        # 允许 torch.compile() 内联该函数
        fn._torchdynamo_inline = obj  # type: ignore[attr-defined]
        
        # 将编译后的函数缓存起来
        _set_jit_function_cache(obj, fn)
        
        return fn
    else:
        # 对于其他类型的对象，创建其 TorchScript 脚本类
        return torch.jit._recursive.create_script_class(obj)
# 定义一个名为 script 的函数，用于对给定的对象进行 TorchScript 编译
def script(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
    r"""Script the function.

    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    Scripting a dictionary or list copies the data inside it into a TorchScript instance than can be
    subsequently passed by reference between Python and TorchScript with zero copy overhead.

    ``torch.jit.script`` can be used as a function for modules, functions, dictionaries and lists
     and as a decorator ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.

    Args:
        obj (Callable, class, or nn.Module):  The ``nn.Module``, function, class type,
                                                  dictionary, or list to compile.
        example_inputs (Union[List[Tuple], Dict[Callable, List[Tuple]], None]): Provide example inputs
            to annotate the arguments for a function or ``nn.Module``.

    Returns:
        If ``obj`` is ``nn.Module``, ``script`` returns
        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will
        have the same set of sub-modules and parameters as the
        original ``nn.Module``. If ``obj`` is a standalone function,
        a :class:`ScriptFunction` will be returned. If ``obj`` is a ``dict``, then
        ``script`` returns an instance of `torch._C.ScriptDict`. If ``obj`` is a ``list``,
        then ``script`` returns an instance of `torch._C.ScriptList`.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFunction

            # See the compiled graph as Python code
            print(foo.code)

            # Call the function using the TorchScript interpreter
            foo(torch.ones(2, 2), torch.ones(2, 2))

        .. testoutput::
            :hide:

            ...
    """
    如果未启用 TorchScript，则直接返回对象。
    """
    if not _enabled:
        return obj

    """
    标记全局变量 _TOPLEVEL，并记录 TorchScript 使用情况。
    """
    global _TOPLEVEL
    if _TOPLEVEL:
        log_torchscript_usage("script")
    prev = _TOPLEVEL
    _TOPLEVEL = False

    """
    尝试使用 _script_impl 函数对对象进行脚本化处理，传入参数包括：
    - obj: 待处理对象
    - optimize: 是否优化脚本化结果
    - _frames_up: 堆栈帧上升数
    - _rcb: 递归回调函数
    - example_inputs: 示例输入，用于脚本化时的参数注解
    """
    try:
        return _script_impl(
            obj=obj,
            optimize=optimize,
            _frames_up=_frames_up + 1,
            _rcb=_rcb,
            example_inputs=example_inputs,
        )
    finally:
        """
        恢复先前的 _TOPLEVEL 变量状态。
        """
        _TOPLEVEL = prev
# 重载在 _jit_internal 中注册，然后在此处编译，以便 nn/functional.py 中可以使用 _overload 而无需导入循环

def _check_overload_defaults(impl_defaults, overload_defaults, loc):
    # 检查重载函数的默认参数是否与实现函数的默认参数一致
    for name, overload_value in overload_defaults.items():
        if name not in impl_defaults or impl_defaults[name] != overload_value:
            raise torch.jit.frontend.FrontendError(
                loc,
                "Default parameters on overloads do not affect the runtime so they "
                "must equal to the default parameter on the implementation function. Found on "
                f"parameter {name}",
            )

def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    # 获取重载函数的定义声明
    overload_decl = get_jit_def(overload_fn, overload_fn.__name__).decl()
    # 获取重载函数的签名
    overload_signature = torch.jit.annotations.get_signature(
        overload_fn, None, None, inspect.ismethod(overload_fn)
    )
    # 获取实现函数的 AST
    impl_ast = get_jit_def(impl_fn, impl_fn.__name__)
    # 获取重载函数的默认参数
    overload_defaults = get_default_args(overload_fn)
    # 获取实现函数的默认参数
    implementation_defaults = get_default_args(impl_fn)
    # 从闭包中创建解析回调
    _rcb = _jit_internal.createResolutionCallbackFromClosure(impl_fn)
    # 检查重载函数的默认参数是否与实现函数的默认参数一致
    _check_overload_defaults(
        implementation_defaults, overload_defaults, overload_decl.range()
    )
    # 使用 Torch Script 编译重载函数
    fn = torch._C._jit_script_compile_overload(
        qual_name,
        overload_decl,
        impl_ast,
        _rcb,
        implementation_defaults,
        overload_signature,
    )
    return fn

def _get_overloads(obj):
    # 检查是否存在已缓存的编译函数
    existing_compiled_fns = _try_get_jit_cached_overloads(obj)
    # 获取对象的限定名
    qual_name = _qualified_name(obj)
    # 获取未编译的重载函数
    uncompiled_overloads = _jit_internal._get_fn_overloads(qual_name)
    if uncompiled_overloads is None:
        return existing_compiled_fns

    if obj in uncompiled_overloads:
        raise RuntimeError(
            _jit_internal.get_overload_no_implementation_error_message("function", obj)
        )

    compiled_fns = []
    # 编译所有未编译的重载函数
    for overload_fn in uncompiled_overloads:
        compiled_fns.append(
            _compile_function_with_overload(overload_fn, qual_name, obj)
        )

    if existing_compiled_fns:
        compiled_fns = existing_compiled_fns + compiled_fns

    # 缓存编译结果，清除用于编译的信息
    _set_jit_overload_cache(obj, compiled_fns)
    _jit_internal._clear_fn_overloads(qual_name)
    return compiled_fns

def _check_directly_compile_overloaded(obj):
    # 获取对象的限定名
    qual_name = _qualified_name(obj)
    # 检查是否存在未编译的重载函数或已缓存的编译函数
    if _jit_internal._get_fn_overloads(qual_name) or _try_get_jit_cached_overloads(obj):
        raise RuntimeError(
            f"Function {qual_name} cannot be directly compiled because it"
            " is overloaded. It must be used in a context of a function"
            " where its inputs can determine which overload to call."
        )

def interface(obj):
    r"""Decorate to annotate classes or modules of different types.
    """
    This decorator defines a TorchScript interface for annotating classes or modules 
    that can have different implementations or types implementing the same interface. 
    It allows defining callable functions or modules with varying implementations 
    that can be swapped at runtime.
    
    Example usage includes defining an interface `InterfaceType` with a method `run` 
    that must be implemented by different classes or modules. Two implementations are 
    demonstrated: `Impl1`, a scripted class using TorchScript, and `Impl2`, a subclass 
    of `torch.nn.Module` with an exported method `run`.
    
    A function `user_fn` is defined to accept a list of `InterfaceType` instances and 
    call their `run` method based on an index. The function `user_fn` is then scripted 
    using TorchScript for performance.
    
    Finally, instances of `Impl1` and a scripted version of `Impl2` are created and 
    passed to `user_fn_jit` to demonstrate calling different implementations of 
    the `InterfaceType`.
    
    The code checks if `obj` is a class, ensures it is a new-style class, and verifies 
    the inheritance structure for TorchScript compatibility. Depending on whether `obj` 
    inherits from `torch.nn.Module`, it determines whether to create a class interface 
    type or a module interface type. It then compiles the interface using TorchScript's 
    internal functions and assigns a mangled name to `obj.__torch_script_interface__`.
    
    Parameters:
    - obj: The class or module to which the interface is applied.
    
    Returns:
    - The modified class or module with TorchScript interface annotations.
    """
    if not inspect.isclass(obj):
        raise RuntimeError("interface must be applied to a class")
    if not _is_new_style_class(obj):
        raise RuntimeError("TorchScript interfaces must inherit from 'object'")
    
    # Expected MRO is:
    #   User module
    #   torch.nn.modules.module.Module
    #   object
    is_module_interface = issubclass(obj, torch.nn.Module) and len(obj.mro()) == 3
    
    if not is_module_interface and len(obj.mro()) > 2:
        raise RuntimeError(
            "TorchScript interface does not support inheritance yet. "
            "Please directly inherit from 'object' or 'nn.Module'."
        )
    
    qualified_name = _qualified_name(obj)
    rcb = _jit_internal.createResolutionCallbackFromFrame(1)
    # if this type is a `nn.Module` subclass, generate a module interface type
    # instead of a class interface type; a module interface type only compiles
    # the user provided methods as part of the interface
    ast = get_jit_class_def(obj, obj.__name__)
    mangled_classname = torch._C._jit_script_interface_compile(
        qualified_name, ast, rcb, is_module_interface
    )
    obj.__torch_script_interface__ = mangled_classname
    return obj
def _recursive_compile_class(obj, loc):
    # 获取对象的完全限定名称
    _qual_name = _qualified_name(obj)
    # 创建一个新的错误调用堆栈对象，以便在编译失败时使用
    error_stack = torch._C.CallStack(_qual_name, loc)
    # 获取对象的方法解析回调函数
    rcb = _jit_internal.createResolutionCallbackForClassMethods(obj)
    # 编译并注册类对象
    return _compile_and_register_class(obj, rcb, _qual_name)


CompilationUnit = torch._C.CompilationUnit
# 设置 CompilationUnit 对象的模块路径为 "torch.jit"
set_module(CompilationUnit, "torch.jit")


def pad(s: str, padding: int, offset: int = 0, char: str = " "):
    # 如果需要的填充长度大于等于字符串 s 的长度，则调整填充长度
    if padding >= len(s):
        padding -= len(s)
    # 构造填充后的字符串
    return "".join([char for _ in range(padding + offset)]) + s


class _ScriptProfileColumn:
    def __init__(self, header: str, alignment: int = 4, offset: int = 0):
        # 初始化列的标题、对齐方式和偏移量
        self.header = header
        self.alignment = alignment
        self.offset = offset
        # 创建一个空的行字典，用于存储行号到值的映射关系
        self.rows: Dict[int, Any] = {}

    def add_row(self, lineno: int, value: Any):
        # 向列中添加一行数据，指定行号和对应的值
        self.rows[lineno] = value

    def materialize(self):
        # 计算列中值的最大长度，并准备输出格式化后的行数据
        max_length = len(self.header)
        rows: List[Tuple[int, str]] = []
        for key, value in self.rows.items():
            cell = str(value)
            rows.append((key, cell))
            max_length = max(len(cell), max_length)

        # 计算所需的填充长度，根据对齐方式和偏移量
        if self.alignment > 0:
            padding = max_length + self.alignment
            padding -= padding % self.alignment
        else:
            padding = 0

        # 格式化每行的输出，包括标题和行数据
        rows = [(key, pad(cell, padding, self.offset)) for key, cell in rows]
        return pad(self.header, padding, self.offset), rows


class _ScriptProfileTable:
    def __init__(self, cols: List[_ScriptProfileColumn], source_range: List[int]):
        # 初始化表格，指定列和源码范围
        self.cols = cols
        self.source_range = source_range

    def dump_string(self):
        # 生成表格的字符串表示
        outputs: List[str] = []
        cells: List[Tuple[str, Dict[int, str]]] = []
        header_buffer = ""
        # 遍历列对象，生成列的标题和数据行
        for col in self.cols:
            header, rows = col.materialize()
            header_buffer += header
            cells.append((header, dict(rows)))

        # 将标题行添加到输出列表中
        outputs.append(header_buffer)
        # 添加分隔线，长度与标题行相同
        outputs.append(pad("", len(header_buffer), 0, "="))
        # 遍历源码范围中的每一行，生成对应的行数据
        for line in self.source_range:
            row_buffer = ""
            for header, rows in cells:
                # 获取指定行的单元格数据，如果不存在则使用空白填充
                cell = rows.get(line)
                if cell is None:
                    row_buffer += pad("", len(header))
                else:
                    row_buffer += cell
            # 将生成的行数据添加到输出列表中
            outputs.append(row_buffer)
        # 将所有行数据用换行符连接起来，形成最终的输出字符串
        return "\n".join(outputs)


class _ScriptProfile:
    def __init__(self):
        # 初始化脚本分析器对象
        self.profile = classes.profiling._ScriptProfile()

    def enable(self):
        # 启用脚本分析器
        self.profile.enable()

    def disable(self):
        # 禁用脚本分析器
        self.profile.disable()
    # 定义一个方法用于生成代表脚本文件的统计信息的字符串
    def dump_string(self) -> str:
        # 初始化一个空列表，用于存储每个脚本文件的统计信息表格字符串
        outputs: List[str] = []
        # 遍历每个脚本文件的统计信息
        for source_stats in self.profile._dump_stats():
            # 获取脚本文件的引用
            source_ref = source_stats.source()
            # 将脚本文件内容按行拆分为列表
            source_lines = source_ref.text().splitlines()
            # 计算每行开头的空格数，用于去除缩进
            dedent = min(len(line) - len(line.lstrip(" ")) for line in source_lines)
            # 去除每行开头的空格，保留内容
            source_lines = [line[dedent:] for line in source_lines]

            # 获取脚本文件起始行号和结束行号
            start_line = source_ref.starting_lineno()
            end_line = start_line + len(source_lines)
            # 创建起始行号到结束行号的范围
            source_range = range(start_line, end_line)
            # 定义不同列的名称
            lineno = _ScriptProfileColumn("Line #")
            hits = _ScriptProfileColumn("Hits")
            time_ns = _ScriptProfileColumn("Time (ns)")
            line_contents = _ScriptProfileColumn("Line Contents", 0, 1)
            # 获取每行的统计信息映射
            stats = source_stats.line_map()
            # 遍历脚本文件的每一行
            for line in source_range:
                # 添加行号到行号列
                lineno.add_row(line, line)
                # 添加行内容到行内容列
                line_contents.add_row(line, source_lines[line - start_line])
                # 获取当前行的统计信息
                stat = stats.get(line)
                if stat is not None:
                    # 如果存在统计信息，将命中次数添加到命中次数列
                    hits.add_row(line, stat.count())
                    # 将执行时间添加到执行时间列
                    time_ns.add_row(line, stat.duration_ns())

            # 创建脚本文件的统计信息表格对象
            table = _ScriptProfileTable(
                [lineno, hits, time_ns, line_contents], list(source_range)
            )
            # 将当前脚本文件的统计信息表格字符串添加到输出列表
            outputs.append(table.dump_string())
        # 将所有脚本文件的统计信息表格字符串用两个换行符连接成一个字符串并返回
        return "\n\n".join(outputs)

    # 定义一个方法用于打印脚本文件的统计信息
    def dump(self):
        # 调用生成统计信息字符串的方法并打印结果
        print(self.dump_string())
# 定义一个函数 _unwrap_optional，用于从可选值中获取真实值
def _unwrap_optional(x):
    # 断言 x 不为 None，如果为 None 则抛出异常 "Unwrapping null optional"
    assert x is not None, "Unwrapping null optional"
    # 返回真实的值 x
    return x

# 调用 _register_builtin 函数，注册 _unwrap_optional 函数，关联的原始字符串为 "aten::_unwrap_optional"
_register_builtin(_unwrap_optional, "aten::_unwrap_optional")

# 调用 _register_builtin 函数，注册 _jit_internal.is_scripting 函数，关联的原始字符串为 "aten::is_scripting"
_register_builtin(_jit_internal.is_scripting, "aten::is_scripting")

# 调用 _register_builtin 函数，注册 has_torch_function 函数，关联的原始字符串为 "aten::has_torch_function"
_register_builtin(has_torch_function, "aten::has_torch_function")

# 调用 _register_builtin 函数，注册 has_torch_function_unary 函数，关联的原始字符串为 "aten::has_torch_function"
_register_builtin(has_torch_function_unary, "aten::has_torch_function")

# 调用 _register_builtin 函数，注册 has_torch_function_variadic 函数，关联的原始字符串为 "aten::has_torch_function"
_register_builtin(has_torch_function_variadic, "aten::has_torch_function")
```