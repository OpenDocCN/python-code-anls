# `.\pytorch\torch\fx\_symbolic_trace.py`

```
# mypy: allow-untyped-defs
# 导入内置模块和第三方模块
import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections

# 导入 itertools 中的 chain 函数
from itertools import chain

# 导入 types 模块中的特定类型
from types import CodeType, FunctionType, ModuleType

# 导入 typing 中的类型注解
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

# 导入 torch 库及其子模块
import torch
import torch.utils._pytree as pytree

# 导入 torch._C 中的 ScriptObject 类
from torch._C import ScriptObject  # type: ignore[attr-defined]

# 导入 torch._library.fake_class_registry 中的 FakeScriptObject 类
from torch._library.fake_class_registry import FakeScriptObject

# 导入 _compatibility 模块中的 compatibility 装饰器
from ._compatibility import compatibility

# 导入 graph 模块中的类和函数
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph

# 导入 graph_module 模块中的类
from .graph_module import GraphModule

# 导入 _lazy_graph_module 模块中的函数
from ._lazy_graph_module import _make_graph_module

# 导入 node 模块中的类和函数
from .node import Argument, base_types, map_aggregate

# 导入 proxy 模块中的类和函数
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager

# 定义一个常量，表示 CO_VARARGS 和 CO_VARKEYWORDS 标志的组合
HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

# 定义全局变量，存储 torch.nn.Module 的 __call__ 方法
_orig_module_call: Callable = torch.nn.Module.__call__

# 定义全局变量，存储 torch.nn.Module 的 __getattr__ 方法
_orig_module_getattr: Callable = torch.nn.Module.__getattr__

# 存储可代理类的字典，键为类型，值为 None
_proxyable_classes: Dict[Type, None] = {}

# 标志位，指示当前是否处于 FX 跟踪状态
_is_fx_tracing_flag = False

# 函数，返回当前是否处于 FX 跟踪状态
def is_fx_tracing():
    return _is_fx_tracing_flag

# compatibility 装饰器修饰的类，用于代理类的元类定义
@compatibility(is_backward_compatible=True)
class ProxyableClassMeta(type):
    """
    ProxyableClassMeta 允许您使给定 Python 类的构造在符号跟踪中可追踪。
    例如::

        import torch
        import torch.fx

        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair_ctor(x : TensorPair, y : torch.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)

        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = torch.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : torch.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    从这个例子中，我们可以看到使用 ProxyableClassMeta 元类定义的类（TensorPair）
    的构造可以在符号跟踪中记录。
    """

    def __init__(cls, name, bases, attrs):
        # 将当前类添加到 _proxyable_classes 字典中
        _proxyable_classes.setdefault(cls)
        super().__init__(name, bases, attrs)
    # 定义类的调用方法，允许类被调用时创建实例
    def __call__(cls, *args, **kwargs):
        # 创建一个未初始化的类实例
        instance = cls.__new__(cls)  # type: ignore[call-overload]

        # 如果不在函数追踪中
        if not is_fx_tracing():
            # 使用给定的参数初始化实例
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance

        found_proxies = []

        # 检查参数中是否有代理对象，并记录下来
        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)

        # 检查所有位置参数中是否有代理对象
        map_aggregate(args, check_proxy)
        # 检查所有关键字参数中是否有代理对象
        map_aggregate(kwargs, check_proxy)

        # 如果找到了代理对象
        if len(found_proxies) != 0:
            # 获取第一个找到的代理对象的追踪器
            tracer = found_proxies[0].tracer
            # 使用追踪器创建一个代理对象，代表调用函数
            return tracer.create_proxy("call_function", cls, args, kwargs)
        else:
            # 如果没有找到代理对象，使用给定的参数初始化实例
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance
def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__  # 获取函数对象 fn 的代码对象
    co_flags = co.co_flags & ~HAS_VARSTUFF  # 清除变长参数标志位
    co_args: tuple
    if hasattr(co, "co_qualname"):
        # Python-3.11+ 代码签名
        co_args = (
            nargs,  # 参数个数
            0,
            0,
            co.co_nlocals,  # 局部变量数
            co.co_stacksize,  # 栈大小
            co_flags,  # 标志位
            co.co_code,  # 字节码
            co.co_consts,  # 常量
            co.co_names,  # 名称列表
            co.co_varnames,  # 变量名列表
            co.co_filename,  # 文件名
            co.co_name,  # 函数名
            co.co_qualname,  # 限定函数名（忽略类型检查）
            co.co_firstlineno,  # 函数起始行号
            co.co_lnotab,  # 行号表
            co.co_exceptiontable,  # 异常表（忽略类型检查）
            co.co_freevars,  # 自由变量
            co.co_cellvars,  # 单元变量
        )
    elif hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    else:
        co_args = (
            nargs,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    new_code = CodeType(*co_args)  # 创建新的代码对象
    return FunctionType(
        new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )

# 需要为 *args 和 **kwargs 插入占位符节点
# 不能正常调用此函数，否则会尝试解包它们
# 相反，让 Python 认为 args 和 kwargs 是普通变量


@compatibility(is_backward_compatible=False)
class PHBase:
    """
    代表 `concrete_args` 的输入占位符对象
    """

    def __repr__(self):
        return "PH"


PH = PHBase()  # 创建一个占位符对象


@compatibility(is_backward_compatible=False)
class PHWithMeta(PHBase):
    """
    代表 `concrete_args` 的带有元数据的输入占位符对象
    """

    def __init__(self, ph_key: Optional[str] = None):
        super().__init__()

        # 为用户提供一个标识占位符节点的键值
        self.ph_key = ph_key


def _transfer_attrs(fr, to):
    for attr_name in dir(fr):  # 遍历对象 fr 的属性名列表
        attr_val = getattr(fr, attr_name)  # 获取属性值
        if (
            not callable(attr_val)  # 排除可调用属性
            and not attr_name.startswith("__")  # 排除私有属性
            and not hasattr(to, attr_name)  # 如果目标对象 to 没有同名属性
        ):
            setattr(to, attr_name, attr_val)  # 设置目标对象 to 的属性值


@compatibility(is_backward_compatible=True)
class Tracer(TracerBase):
    # 这段注释解释了对类和初始化函数的文档字符串和参数的设置。它们用于 torch.fx.symbolic_trace 的符号追踪功能。

    """
    Tracer(autowrap_modules=(math,), autowrap_functions=())

    ``Tracer`` 是实现符号追踪功能的类，用于 ``torch.fx.symbolic_trace``。
    对 ``symbolic_trace(m)`` 的调用相当于 ``Tracer().trace(m)``。

    Tracer 可以被子类化以覆盖追踪过程的各种行为。可以通过类方法的文档字符串了解可以覆盖的不同行为。
    """

    # 这段注释指出，不检查此 API 的向后兼容性，因为 `autowrap_modules` 的默认值包含 `math` 模块的本地文件路径，
    # 这会在不同机器上产生波动。
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        autowrap_modules: Tuple[ModuleType] = (math,),
        autowrap_functions: Tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
        # This method defines the constructor for the Tracer class.
        # It initializes various attributes related to tracing and wrapping functions/modules.

        """
        Construct a Tracer object.

        Args:

            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap(). Backward-compatibility for
                this parameter is guaranteed.

            autowrap_functions (Tuple[Callable, ...]): defaults to `()`,
                Python functions that should be wrapped automatically without
                needing to use fx.wrap(). Backward compatibility for this
                parameter is guaranteed.

            param_shapes_constant (bool): When this flag is set,  calls to shape,
                size and a few other shape like attributes of a module's parameter
                will be evaluated directly, rather than returning a new Proxy value
                for an attribute access. Backward compatibility for this parameter
                is guaranteed.
        """

        # Call the constructor of the superclass (presumably `object` or another base class)
        super().__init__()

        # Initialize a set to store IDs of functions that should be eagerly wrapped during tracing
        # This includes functions from `autowrap_modules` and `autowrap_functions`
        self._autowrap_function_ids: Set[int] = {
            id(value)
            for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)
        }
        self._autowrap_function_ids.update({id(f) for f in autowrap_functions})

        # Initialize a list to store additional modules to apply autowrap to at the start
        # This includes `autowrap_modules` passed to the constructor
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)

        # Store the value of param_shapes_constant which determines attribute access behavior
        self.param_shapes_constant = param_shapes_constant

        # Initialize attributes related to module hierarchy and tracing
        self.submodule_paths: Optional[Dict[torch.nn.Module, str]] = None
        self.root_module_name: str = ""
        self.scope = Scope("", None)  # Initialize the scope with root values
        self.module_stack = collections.OrderedDict()  # Initialize an ordered dict for module call stack
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}  # Initialize dict mapping node names to module scope

    # Class-level attribute to keep track of function names and their occurrences
    _qualname_counter: Dict[str, int] = collections.defaultdict(int)

    # Decorator indicating backward compatibility for the following method
    @compatibility(is_backward_compatible=True)
    def get_fresh_qualname(self, prefix: str) -> str:
        """
        Gets a fresh name for a prefix and returns it. This function ensures
        that it will not clash with an existing attribute on the graph.
        """
        # 初始化一个带有数字后缀的名称
        qualname = f"{prefix}0"
        
        # 如果根对象没有这个名称属性，则重置计数器以从零开始
        if not hasattr(self.root, qualname):
            self._qualname_counter[prefix] = 0
            return qualname

        # 获取当前计数器值
        i = self._qualname_counter[prefix]
        
        # 循环直到找到一个未被使用的名称
        while True:
            qualname = f"{prefix}{i}"
            i += 1
            if not hasattr(self.root, qualname):
                break
        
        # 更新计数器值
        self._qualname_counter[prefix] = i

        return qualname

    @compatibility(is_backward_compatible=True)
    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        # 检查模块是否在torch.nn或torch.ao.nn命名空间下，并且不是torch.nn.Sequential的实例
        return (
            (m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn"))
            and not isinstance(m, torch.nn.Sequential)
        )

    @compatibility(is_backward_compatible=True)
    # 返回指定模块在模块层次结构中的完整名称路径
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        # 使用 O(1) 算法查找子模块路径
        if self.submodule_paths:
            # 尝试从预存的子模块路径字典中获取模块的路径
            path = self.submodule_paths.get(mod)
            # 如果路径为 None，抛出模块未安装为子模块的错误
            if path is None:
                raise NameError("module is not installed as a submodule")
            assert isinstance(path, str)
            return path
        # 如果没有预存的子模块路径，则使用 O(N^2) 算法进行回溯查找
        else:
            # 遍历根模块及其所有命名子模块，寻找与给定模块相符的命名
            for n, p in self.root.named_modules():
                if mod is p:
                    return n
            # 若未找到匹配的子模块，抛出模块未安装为子模块的错误
            raise NameError("module is not installed as a submodule")

    # 调用模块的兼容性包装函数，用于处理模块的前向计算
    @compatibility(is_backward_compatible=True)
    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        # 获取模块的完全限定名，用于标识模块在图中的位置
        module_qualified_name = self.path_of_module(m)
        # 在当前作用域下创建一个新的作用域，用于跟踪这次模块调用的状态
        with ScopeContextManager(self.scope, Scope(module_qualified_name, type(m))) as _scope:
            # 将模块的路径及其类型加入到模块堆栈中
            self.module_stack[_scope.module_path] = (module_qualified_name, _scope.module_type)
            # 如果模块不是叶子模块，则直接调用其 forward 方法
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                # 如果模块是叶子模块，则创建一个代理对象，表示调用了该模块
                ret_val = self.create_proxy("call_module", module_qualified_name, args, kwargs)
            # 从模块堆栈中弹出最后一个加入的模块信息，并进行断言验证
            key, _ = self.module_stack.popitem(last=True)
            assert key == _scope.module_path, f" Unexpected key {key}"

        return ret_val
    # 定义一个方法，用于在调用 nn.Module 实例的 getattr 方法时指定 Tracer 的行为
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        """
        Method that specifies the behavior of this ``Tracer`` when we call getattr
        on a call to an ``nn.Module`` instance.

        By default, the behavior is to return a proxy value for the attribute. It
        also stores the proxy value in the ``parameter_proxy_cache``, so that future
        calls will reuse the proxy rather than creating a new one.

        This method can be overridden to --for example-- not return proxies when
        querying parameters.

        Args:

            attr (str): The name of the attribute being queried
            attr_val (Any): The value of the attribute
            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies

        Return:

            The return value from the getattr call.
        """
        
        # 定义一个函数，用于尝试获取属性的代理对象
        def maybe_get_proxy_for_attr(
            attr_val, collection_to_search, parameter_proxy_cache
        ):
            for n, p in collection_to_search:
                if attr_val is p:
                    # 如果属性值在缓存中不存在，则创建并存储代理对象
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        # 检查是否需要传入代理工厂函数的参数
                        if (
                            "proxy_factory_fn"
                            in inspect.signature(self.create_proxy).parameters
                        ):
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node: ParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        # 调用 create_proxy 方法创建代理对象
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        # 将创建的代理对象存入缓存
                        parameter_proxy_cache[n] = val_proxy
                    # 返回已缓存的代理对象
                    return parameter_proxy_cache[n]
            return None

        # 如果属性值是 torch.nn.Parameter 类型，则尝试获取其代理对象
        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        # 如果启用了代理缓冲区属性，并且属性值是 torch.Tensor 类型，则尝试获取其代理对象
        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache
            )
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        # 默认情况下直接返回属性值
        return attr_val

    # This method will be refactored
    # 以下方法将进行重构
    @compatibility(is_backward_compatible=False)
    @compatibility(is_backward_compatible=True)
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    # 定义对象的深拷贝方法，用于创建当前对象的深层副本
    def __deepcopy__(self, memo):
        # 创建一个新的 Tracer 对象实例
        new_tracer = Tracer.__new__(Tracer)

        # 遍历当前对象的所有属性和对应的值
        for k, v in self.__dict__.items():
            # 如果属性名 k 在集合 {'_autowrap_search'} 中，则执行浅拷贝
            if k in {'_autowrap_search'}:
                new_obj = copy.copy(v)
            else:
                # 否则，执行深拷贝，传入 memo 字典以处理循环引用
                new_obj = copy.deepcopy(v, memo)

            # 将新属性值赋给新创建的 Tracer 对象的属性字典中
            new_tracer.__dict__[k] = new_obj

        # 返回创建的新的 Tracer 对象实例，完成深拷贝操作
        return new_tracer
    # 定义一个方法，用于处理占位符的代理
    def _proxy_placeholder(self, name, concrete_args, sig, fn_for_analysis):
        # 检查具体参数是否不为 None，并且指定名称存在于具体参数中
        if concrete_args is not None and name in concrete_args:
            cnt = 0  # 计数器，用于生成唯一的占位符名称

            # 定义替换占位符的函数
            def replace_ph(x):
                nonlocal cnt  # 使用非局部变量 cnt
                cnt += 1
                param = sig.parameters[name]  # 获取参数的签名信息
                # 根据参数是否有默认值，设置默认值
                default = (
                    ()
                    if param.default is inspect.Parameter.empty
                    else (param.default,)
                )
                # 创建一个占位符代理对象
                out = self.create_proxy(
                    "placeholder", f"{name}_{str(cnt)}", default, {}
                )
                # 如果 x 是 PHBase 类的实例
                if isinstance(x, PHBase):
                    if x != PH:
                        # 当使用除了单例 PH 外的其他占位符时，传输属性
                        # 从占位符传输元数据到底层节点
                        _transfer_attrs(fr=x, to=out.node)
                    return out
                # 如果 x 是布尔值或者是基本类型之一，但不是 torch.Tensor
                if (
                    type(x) == bool
                    or type(x) in base_types
                    and type(x) != torch.Tensor
                ):
                    # 断言确保 out 和 x 的值相等
                    torch._assert(
                        out == x,
                        f"{name} has been specialized to have value {x} but got another value",
                    )
                elif x is None:
                    # 如果 x 是 None，则创建一个断言函数
                    args = (
                        out,
                        f"{name} has been specialized to have value None but got another value",
                    )
                    self.create_proxy("call_function", _assert_is_none, args, {})
                else:
                    # 发出警告，提示无法添加断言以确保正确的输入
                    warnings.warn(
                        f"Was not able to add assertion to guarantee correct input {name} to "
                        f"specialized function. It is up to the user to make sure that your inputs match the "
                        f"inputs you specialized the function with."
                    )

                return x

            # 对具体参数中指定名称对应的值应用替换占位符的函数
            return pytree.tree_map(replace_ph, concrete_args[name])

        # 如果名称以 '*' 开头，设置默认值为空元组
        if name[0] == "*":
            default = ()
        else:
            # 否则，根据参数的签名信息设置默认值
            param = sig.parameters[name]
            default = () if param.default is inspect.Parameter.empty else (param.default,)  # type: ignore[assignment]

        # 创建一个占位符代理对象
        return self.create_proxy(
            "placeholder",
            name,
            default,
            {},
            type_expr=fn_for_analysis.__annotations__.get(name, None)
        )
# 用于存储需要修补的函数的全局字典的ID和函数名的元组，以便于 wrap() API 的目的
# 键是全局字典的ID和函数名，以确保只对给定函数进行一次包装
_wrapped_fns_to_patch: Dict[Tuple[int, str], dict] = {}

# 用于存储需要修补的类方法的列表，每个元素是类类型和函数名的元组
# 目前仅适用于未正确追踪的 Tensor.* 方法
_wrapped_methods_to_patch: List[Tuple[type, str]] = []

if os.environ.get("FX_PATCH_GETITEM") == "1":
    # 当需要跟踪像 BERT 中的 PositionalEmbedding 这样的模型时，需要进行此更改
    # 但会导致量化中的问题，详细记录在此处：
    # https://github.com/pytorch/pytorch/issues/50710
    # 一旦问题修复，可以将此变更设置为默认行为
    _wrapped_methods_to_patch.append((torch.Tensor, "__getitem__"))


def _find_proxy(*objects_to_search):
    """
    递归搜索数据结构以查找 Proxy()，找到后返回它，如果找不到则返回 None。
    """
    proxy = None

    def find_proxy(x):
        nonlocal proxy
        if isinstance(x, Proxy):
            proxy = x

    map_aggregate(objects_to_search, find_proxy)
    return proxy


def _create_wrapped_func(orig_fn):
    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        给定一个闭包的 ``orig_function`` 进行调用，搜索 args 和 kwargs 中是否有 Proxy 对象。
        如果有，发出一个 ``call_function`` 节点以保留对此叶子函数的直接调用。
        否则，只返回此函数调用的结果，因为此函数未被追踪。
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = proxy.tracer.create_proxy(
                "call_function", orig_fn, args, kwargs
            )
            return_proxy.node.meta["is_wrapped"] = True
            return return_proxy
        return orig_fn(*args, **kwargs)

    return wrapped


def _create_wrapped_method(cls, name):
    orig_fn = getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        搜索 args 和 kwargs 中是否有 Proxy 对象。如果有，发出一个 ``call_method`` 节点以保留对此方法的直接调用。
        否则，只返回此方法调用的结果，因为此函数未被追踪。
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return proxy.tracer.create_proxy("call_method", name, args, kwargs)
        return orig_fn(*args, **kwargs)

    return wrapped


class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any

    def revert(self):
        raise NotImplementedError


class _PatchedFnSetItem(_PatchedFn):
    # 这个类继承自 _PatchedFn 类，具有 frame_dict、fn_name 和 orig_fn 属性
    # 但未实现 revert 方法
    # 定义一个方法 revert，属于某个类的成员方法
    def revert(self):
        # 将 self.orig_fn 的值赋给 self.frame_dict 字典中的 self.fn_name 键
        self.frame_dict[self.fn_name] = self.orig_fn
class _PatchedFnDel(_PatchedFn):
    # 定义用于删除操作的特殊修补函数对象
    def revert(self):
        # 恢复原始状态，删除在构造时指定的函数名对应的键值对
        del self.frame_dict[self.fn_name]


class _PatchedFnSetAttr(_PatchedFn):
    # 定义用于设置属性操作的特殊修补函数对象
    def revert(self):
        # 恢复原始状态，设置在构造时指定的属性名为原始函数
        setattr(self.frame_dict, self.fn_name, self.orig_fn)


class _Patcher:
    # 修补器类，用于管理和执行函数和方法的临时替换
    def __init__(self):
        super().__init__()
        # 记录已经应用的修补函数列表
        self.patches_made: List[_PatchedFn] = []
        # 记录已访问过的对象的 ID，避免重复操作
        self.visited: Set[int] = set()

    def patch(
        self,
        frame_dict: Dict[str, Any],
        name: str,
        new_fn: Callable,
        deduplicate: bool = True,
    ):
        """
        替换 frame_dict[name] 为 new_fn，直到退出上下文管理器。
        """
        # 标记新函数已经被修补过
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        # 如果 name 不在 frame_dict 中，并且在内置函数中存在
        if name not in frame_dict and hasattr(builtins, name):
            # 添加删除操作的修补函数对象到列表中
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        # 如果 frame_dict[name] 已经被修补过，则直接返回
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # 已经修补过，无需再次操作
        else:
            # 否则，添加设置属性操作的修补函数对象到列表中
            self.patches_made.append(
                _PatchedFnSetAttr(frame_dict, name, frame_dict[name])
            )
        # 将 frame_dict[name] 设置为新的函数 new_fn
        frame_dict[name] = new_fn

    def patch_method(
        self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True
    ):
        """
        替换 cls.name 为 new_fn，直到退出上下文管理器。
        """
        # 标记新方法已经被修补过
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        # 获取原始方法
        orig_fn = getattr(cls, name)
        # 如果原始方法已经被修补过，则直接返回
        if getattr(orig_fn, "__fx_already_patched", False):
            return  # 已经修补过，无需再次操作
        # 添加设置属性操作的修补函数对象到列表中
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
        # 将 cls.name 设置为新的方法 new_fn
        setattr(cls, name, new_fn)

    def visit_once(self, thing: Any):
        """在第一次调用时返回 True，否则返回 False"""
        # 获取对象的 ID
        idx = id(thing)
        # 如果对象的 ID 已经在 visited 集合中，则返回 False
        if idx in self.visited:
            return False
        # 否则，将对象的 ID 添加到 visited 集合中，并返回 True
        self.visited.add(idx)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        撤销通过 self.patch() 和 self.patch_method() 所做的所有更改。
        """
        # 反向顺序撤销修补操作，以正确处理重复修补的情况
        while self.patches_made:
            self.patches_made.pop().revert()
        # 清空 visited 集合
        self.visited.clear()


def _patch_wrapped_functions(patcher: _Patcher):
    """
    遍历 _wrapped_fn_patch_table，对于每个 frame 对象，在 _create_wrapped_func 包装器中包装列出的全局函数。
    """
    for (_, name), frame_dict in _wrapped_fns_to_patch.copy().items():
        # 如果 name 不在 frame_dict 中，并且在内置函数中存在
        if name not in frame_dict and hasattr(builtins, name):
            # 获取内置函数的原始函数
            orig_fn = getattr(builtins, name)
        else:
            # 否则，获取 frame_dict[name] 的原始函数
            orig_fn = frame_dict[name]
        # 使用 patcher.patch() 方法替换 frame_dict 中的 name 键对应的函数为包装后的新函数
        patcher.patch(frame_dict, name, _create_wrapped_func(orig_fn))

    for cls, name in _wrapped_methods_to_patch:
        # 使用 patcher.patch_method() 方法替换类 cls 的方法 name 为包装后的新方法
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))


def _autowrap_check(
    patcher: _Patcher, frame_dict: Dict[str, Any], function_ids: Set[int]


# 声明函数的参数列表，包括三个参数：
# - patcher: 类型为 _Patcher，用于某种类型的修补程序
# - frame_dict: 类型为 Dict[str, Any]，用于存储字符串键和任意类型值的字典
# - function_ids: 类型为 Set[int]，用于存储整数类型的集合，代表函数的标识符集合
def wrap(fn_or_name: Union[str, Callable]):
    """
    This function can be called at module-level scope to register fn_or_name as a "leaf function".
    A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being
    traced through::

        # foo/bar/baz.py
        def my_custom_function(x, y):
            return x * x + y * y

        torch.fx.wrap('my_custom_function')

        def fn_to_be_traced(x, y):
            # When symbolic tracing, the below call to my_custom_function will be inserted into
            # the graph rather than tracing it.
            return my_custom_function(x, y)

    This function can also equivalently be used as a decorator::

        # foo/bar/baz.py
        @torch.fx.wrap
        def my_custom_function(x, y):
            return x * x + y * y

    A wrapped function can be thought of a "leaf function", analogous to the concept of
    "leaf modules", that is, they are functions that are left as calls in the FX trace
    rather than traced through.

    Args:

        fn_or_name (Union[str, Callable]): The function or name of the global function to insert into the
            graph when it's called
    """
    if not callable(fn_or_name) and not isinstance(fn_or_name, str):
        raise RuntimeError(
            "Unsupported type for global function! Must be either a callable or "
            "string name"
        )

    if callable(fn_or_name):
        assert not isinstance(fn_or_name, str)  # to make mypy happy
        fn_name = fn_or_name.__name__
    else:
        assert isinstance(
            fn_or_name, str
        ), "fn_or_name must be a global function or string name"
        fn_name = fn_or_name

    currentframe = inspect.currentframe()
    assert currentframe is not None
    f = currentframe.f_back
    assert f is not None
    if f.f_code.co_name != "<module>":
        raise NotImplementedError("wrap must be called at the top level of a module")

    # consider implementing Callable version of this via _autowrap_function_ids / _autowrap_search
    # semantics would be slightly different, but would add support `from x import wrapped_function`
    _wrapped_fns_to_patch[(id(f.f_globals), fn_name)] = f.f_globals
    return fn_or_name
    # 创建一个符号追踪 API 的入口函数，用于从给定的 nn.Module 或函数实例中追踪记录操作，构建成一个 GraphModule。
    def symbolic_trace(root, concrete_args=None):
        # 创建一个追踪器对象
        tracer = Tracer()
        # 使用追踪器对象追踪记录从 root 开始的操作，根据 concrete_args 部分特化输入
        graph = tracer.trace(root, concrete_args)
        # 获取 root 对象的类名或函数名，作为图模块的名称
        name = (
            root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
        )
        # 调用内部函数 _make_graph_module，将追踪到的操作转换为 GraphModule，并返回
        return _make_graph_module(tracer.root, graph, name)
# 使用装饰器 @wrap 将该函数包装起来，使其可以执行特定的包装逻辑
@wrap
# 断言函数参数 value 必须为 None，否则抛出异常并附带错误消息 msg
def _assert_is_none(value, msg):
    assert value is None, msg
```