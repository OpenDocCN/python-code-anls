# `.\pytorch\torch\fx\proxy.py`

```
# mypy: ignore-errors
# 导入所需的模块和库

import enum
import dis
import copy
import sys
import torch
import inspect
import operator
import collections

from dataclasses import is_dataclass, fields
# 导入 graph 模块中的特定函数和类
from .graph import magic_methods, reflectable_magic_methods, Graph
# 导入 torch.utils._traceback 中的 CapturedTraceback 类
from torch.utils._traceback import CapturedTraceback
# 导入类型提示
from typing import Tuple, Dict, OrderedDict, Optional, Any, Iterator, Callable
# 从 .node 模块中导入特定类和函数
from .node import Target, Node, Argument, base_types, map_aggregate
# 导入 _compatibility 模块中的 compatibility 类
from ._compatibility import compatibility
# 导入 operator_schemas 模块中的 check_for_mutable_operation 函数
from .operator_schemas import check_for_mutable_operation
# 导入 torch.fx.traceback 模块
import torch.fx.traceback as fx_traceback

# 指定该模块中可导出的类和函数名
__all__ = ['TracerBase', 'GraphAppendingTracer', 'TraceError',
           'Proxy', 'Attribute', 'ParameterProxy', 'Scope',
           'ScopeContextManager']


# Scope 类：记录模块路径和模块类型的作用域对象
@compatibility(is_backward_compatible=False)
class Scope:
    """ Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)

        class M(torch.nn.Module):
            def __init__(self):
                self.sub = Sub()

            def forward(self, x):
                # This will be a call_method Node as well,
                # scope for this would be (module_path="", None)
                x = x.transpose(1, 2)
                x = self.sub(x)
                return x

    """

    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        # 初始化模块路径和模块类型
        self.module_path = module_path
        self.module_type = module_type


# ScopeContextManager 类：在符号跟踪期间跟踪节点的作用域的上下文管理器
@compatibility(is_backward_compatible=False)
class ScopeContextManager:
    """ A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """

    def __init__(
        self,
        scope: Scope,
        current_scope: Scope,
    ):
        super().__init__()
        # 复制当前作用域以便在退出时恢复
        self._prev_scope = copy.copy(scope)
        # 更新作用域为当前作用域
        scope.module_path = current_scope.module_path
        scope.module_type = current_scope.module_type
        # 保存引用以便后续恢复
        self._scope = scope

    def __enter__(self):
        return self._scope

    def __exit__(self, *args):
        # 恢复到之前保存的作用域
        self._scope.module_path = self._prev_scope.module_path
        self._scope.module_type = self._prev_scope.module_type
        return


# _COPY_META_FIELDS 列表：包含需要复制的元数据字段名
_COPY_META_FIELDS = [
    "nn_module_stack",
    "torch_fn",
    "source_fn_stack",
    "original_aten",
    "recompute",
    "ac_graph_id",
    "from_node",
    "quantization_tag",
]
# 使用装饰器指定此类是向后兼容的
@compatibility(is_backward_compatible=True)
class TracerBase:
    # 类成员变量，表示追踪器所用的图形结构
    graph: Graph

    # 是否记录堆栈跟踪信息的特性标志
    record_stack_traces: bool = False

    # 可变模式检查的特性标志
    # 默认在版本1.12中启用
    check_mutable_operations: bool = False

    # 断言跟踪的特性标志
    trace_asserts: bool = False

    # 代理访问缓冲区值的特性标志
    proxy_buffer_attributes: bool = False

    # 要追踪的函数名称。仅在``root``是``nn.Module``实例时使用
    traced_func_name: str = "forward"

    # 将包含模块的名称映射到操作符名称的映射
    scope: Scope

    # 记录模块调用堆栈的有序字典
    module_stack: OrderedDict[str, Tuple[str, Any]]

    # 节点名称到模块范围的映射
    node_name_to_scope: Dict[str, Tuple[str, type]]

    # 使用装饰器指定此成员是向后兼容的
    @compatibility(is_backward_compatible=True)
    # 定义一个方法用于创建图节点，接受多个参数和可选参数，并返回创建的节点对象
    def create_node(self, kind: str, target: Target,
                    args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: Optional[str] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """
        # 如果节点类型是 'call_function' 并且启用了可变操作检查，调用函数检查函数调用是否是可变操作
        if kind == 'call_function' and self.check_mutable_operations:
            check_for_mutable_operation(target, args, kwargs)

        # 创建图节点对象，传入节点类型、目标、参数、关键字参数、名称和类型表达式
        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        
        # TODO node_name_to_scope 将会被废弃，使用 node.meta['nn_module_stack'] 替代
        # 将新创建的节点名称映射到作用域路径和模块类型上
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        
        # 可选地设置节点的堆栈跟踪以便调试
        if fx_traceback.has_preserved_node_meta():
            current_meta: Dict[str, Any] = fx_traceback.get_current_meta()

            # 获取当前元数据中的堆栈跟踪信息
            stack_trace = current_meta.get("stack_trace")
            if stack_trace:
                node.stack_trace = stack_trace
            
            # 显式设置节点的堆栈跟踪、nn_module_stack 和 source_fn 到 node.meta 中
            # 如果需要其他元数据字段，可以在此添加
            for field in _COPY_META_FIELDS:
                if field in current_meta:
                    node.meta[field] = copy.copy(current_meta[field])

            # 在跟踪 Atten 操作时，对序列号进行递减以修正
            new_seq_nr = torch.autograd._get_sequence_nr() - 1
            # 序列号在每次创建新的 autograd Node 时递增。在前向传播期间，我们存储最后一个 autograd Node 的序列号
            # 对应于此 fx 节点的 meta。单个 Atten 操作可以创建多个 autograd 节点，例如就地 foreach 操作
            # 在反向传播期间，我们从当前执行的 autograd Node 上检索存储的序列号。详见注释 [ Sequence Number ]。
            if current_meta.get("in_grad_fn", 0) > 0:
                new_seq_nr = current_meta["grad_fn_seq_nr"][-1]
            node.meta["seq_nr"] = new_seq_nr

        # 如果没有堆栈跟踪信息但存在模块堆栈，复制当前模块堆栈到节点的 nn_module_stack 中
        elif self.module_stack:
            node.meta['nn_module_stack'] = copy.copy(self.module_stack)
        
        # 返回创建的节点对象
        return node

    # 兼容性修饰符，声明该方法是向后兼容的
    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> 'Proxy':
        # 返回一个 Proxy 对象，代理给定的节点和当前对象
        return Proxy(node, self)

    # 兼容性修饰符，声明该方法是向后兼容的
    @compatibility(is_backward_compatible=True)
    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any],
                     name: Optional[str] = None, type_expr : Optional[Any] = None,
                     proxy_factory_fn: Callable[[Node], 'Proxy'] = None):
        '''
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        '''

        # 创建参数的元组表示
        args_ = self.create_arg(args)
        # 创建关键字参数的字典表示
        kwargs_ = self.create_arg(kwargs)
        # 确保args_是一个元组，kwargs_是一个字典
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)

        # 使用给定的参数创建节点
        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)

        # 根据是否提供了代理工厂函数来创建代理对象
        if not proxy_factory_fn:
            proxy = self.proxy(node)
        else:
            proxy = proxy_factory_fn(node)

        # 如果记录堆栈跟踪信息并且代理对象的节点没有堆栈跟踪信息，则添加堆栈跟踪信息
        if self.record_stack_traces and not proxy.node.stack_trace:
            proxy.node.stack_trace = ''.join(CapturedTraceback.extract().format())

        # 返回创建的代理对象
        return proxy

    def _find_user_frame(self):
        """
        Find the Python stack frame executing the user code during
        symbolic tracing.
        """
        # 获取当前正在执行的帧对象
        frame = inspect.currentframe()

        # 定义PyTorch源文件列表，用于过滤帧对象
        pt_files = ['torch/fx/proxy.py',
                    'torch/fx/_symbolic_trace.py',
                    'torch/fx/experimental/proxy_tensor.py',
                    'torch/_ops.py',
                    'torch/_tensor.py',
                    'torch/utils/_python_dispatch.py',
                    'torch/_prims_common/wrappers.py',
                    'torch/_refs/__init__.py',
                    'torch/_refs/nn/functional/__init__.py',
                    'torch/utils/_stats.py',
                    ]

        # 在调用堆栈中向上遍历，直到找到不在PyTorch源文件中的帧对象
        while frame:
            frame = frame.f_back
            if frame and all(not frame.f_code.co_filename.endswith(file) for file in pt_files):
                break

        # 如果找不到符合条件的帧对象，则返回None
        if not frame:
            return None

        # 返回找到的用户代码执行帧对象
        return frame

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Argument:
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.
        """
        # 如果对象不是 Proxy 类型，并且具有 '__fx_create_arg__' 方法
        if not isinstance(a, Proxy) and hasattr(a, '__fx_create_arg__'):
            # 调用对象的 '__fx_create_arg__' 方法，返回其创建的 Argument 对象
            return a.__fx_create_arg__(self)

        # 处理聚合类型
        elif isinstance(a, tuple) and hasattr(a, '_fields'):
            # 由于 NamedTuple 构造函数不支持生成器表达式作为参数，因此先构建中间元组
            # 然后将其解包到 NamedTuple 构造函数中
            args = tuple(self.create_arg(elem) for elem in a)
            return type(a)(*args)  # type: ignore[arg-type]

        # 处理 tuple 和 list 类型
        elif isinstance(a, (tuple, list)):
            # 递归调用 create_arg 处理元素，返回新的 tuple 或 list
            return type(a)(self.create_arg(elem) for elem in a)

        # 处理 dict 类型
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                # 对字典键进行处理，确保不包含 Proxy 对象
                k = self.create_arg(k)

                # 检查是否存在 Node 类型的键，如果存在则引发异常
                def no_node(arg):
                    if isinstance(arg, Node):
                        raise RuntimeError("Keys for dictionaries used as an argument cannot contain a "
                                           f"Node. Got key: {k}")
                map_aggregate(k, no_node)

                # 对字典值进行递归调用 create_arg 处理
                r[k] = self.create_arg(v)
            return r

        # 处理 slice 类型
        elif isinstance(a, slice):
            # 递归调用 create_arg 处理 slice 的 start、stop 和 step
            return slice(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))

        # 处理 range 类型
        elif isinstance(a, range):
            # 递归调用 create_arg 处理 range 的 start、stop 和 step
            return range(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))

        # 处理 torch._ops.OpOverload 和 torch._ops.HigherOrderOperator 类型
        elif isinstance(a, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            # 直接返回该类型对象
            return a

        # 处理 Proxy 类型的对象
        if isinstance(a, Proxy):
            # 返回 Proxy 对象的节点属性
            return a.node

        # 处理 dataclass 类型
        if is_dataclass(a):
            # 通过反射获取 dataclass 的字段，并递归调用 create_arg 处理每个字段的值
            kwargs = {field.name: self.create_arg(getattr(a, field.name)) for field in fields(a)}
            # 创建一个 call_function 节点，表示调用 dataclass 构造函数
            return self.create_node("call_function", a.__class__, (), kwargs)

        # 处理基本类型、enum 类型以及 None 和 Ellipsis
        elif isinstance(a, (*base_types, enum.Enum)) or a is None or a is ...:
            # 直接返回基本类型、enum 类型、None 或 Ellipsis
            return a

        # 处理未支持的类型，引发 NotImplementedError
        raise NotImplementedError(f"argument of type: {type(a)}")

    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: 'Proxy') -> bool:
        """Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.
        """
        # 抛出异常，表示无法在控制流中使用符号化跟踪的变量作为输入
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: 'Proxy') -> Iterator:
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        # 抛出异常，指示无法迭代代理对象。这可能发生在代理对象在循环中使用或作为 *args 或 **kwargs 函数参数时。
        # 查看 torch.fx 文档以获取有关可以跟踪的控制流类型的更详细解释，并查看 Proxy 的文档以获取帮助来排除代理迭代错误。
        raise TraceError('Proxy object cannot be iterated. This can be '
                         'attempted when the Proxy is used in a loop or'
                         ' as a *args or **kwargs function argument. '
                         'See the torch.fx docs on pytorch.org for a '
                         'more detailed explanation of what types of '
                         'control flow can be traced, and check out the'
                         ' Proxy docstring for help troubleshooting '
                         'Proxy iteration errors')

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: 'Proxy') -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.
        """
        # 返回通过调用 Attribute 类获取 obj 对象的 'keys' 属性的结果
        return Attribute(obj, 'keys')()
# 用于标记兼容性，表示这个类或函数在向后兼容方面符合预期
@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):
    def __init__(self, graph: Graph):
        # 调用父类构造函数初始化
        super().__init__()
        # 将传入的图对象赋值给实例变量graph
        self.graph = graph
        # 创建一个空的作用域实例，并赋值给实例变量scope
        self.scope = Scope("", None)
        # 创建一个有序字典，用于管理模块堆栈
        self.module_stack = collections.OrderedDict()
        # 创建一个空字典，用于映射节点名称到作用域对象
        self.node_name_to_scope = {}

# 标记为不向后兼容的函数
@compatibility(is_backward_compatible=False)
def assert_fn(x):
    # 断言函数，用于断言参数x的真实性

# 用于标记兼容性，表示这个类在向后兼容方面符合预期
@compatibility(is_backward_compatible=True)
class TraceError(ValueError):
    # TraceError 类，继承自 ValueError，用于表示追踪错误的异常类

# 用于标记兼容性，表示这个类在向后兼容方面符合预期
@compatibility(is_backward_compatible=True)
class Proxy:
    """
    ``Proxy`` 对象是 ``Node`` 包装器，在符号追踪期间流经程序，并记录所有它们接触到的操作
    （``torch`` 函数调用、方法调用、操作符），并将它们添加到增长的 FX 图中。

    如果您正在进行图形转换，可以将自己的 ``Proxy`` 方法包装在原始 ``Node`` 周围，
    以便您可以使用重载的操作符将其他内容添加到 ``Graph`` 中。

    ``Proxy`` 对象不能被迭代。换句话说，如果在循环中使用 ``Proxy`` 或作为
    ``*args``/``**kwargs`` 的函数参数，符号追踪器将抛出错误。

    有两种主要解决方法：
    1. 将不可追踪的逻辑提取到顶层函数中，并在其上使用 ``fx.wrap``。
    2. 如果控制流是静态的（即循环次数基于某些超参数），可以将代码保持在其原始位置，
       并重构为类似以下形式的内容：

        for i in range(self.some_hyperparameter):
            indexed_item = proxied_value[i]

    有关 Proxy 内部更详细的描述，请查阅 `torch/fx/README.md` 中的 "Proxy" 部分。
    """

    # 用于标记兼容性，表示这个方法在向后兼容方面符合预期
    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: 'Optional[TracerBase]' = None):
        # 如果没有传入追踪器对象，则使用 GraphAppendingTracer 创建一个新的追踪器
        if tracer is None:
            tracer = GraphAppendingTracer(node.graph)
        # 将追踪器和节点对象赋值给实例变量
        self.tracer = tracer
        self.node = node

    def __repr__(self) -> str:
        # 返回代表此对象的字符串表示形式，格式为 'Proxy(节点名称)'
        return f'Proxy({self.node.name})'

    def __getattr__(self, k) -> 'Attribute':
        # 如果是方法调用，将不会将其添加到图中，而是对方法调用进行优化
        return Attribute(self, k)

    def __call__(self, *args, **kwargs) -> 'Proxy':
        # 创建一个代理对象，表示方法调用，并将其添加到追踪器中
        return self.tracer.create_proxy('call_method', '__call__', (self,) + args, kwargs)
    # 定义一个迭代器方法，返回一个'Proxy'对象的迭代器
    def __iter__(self) -> Iterator['Proxy']:
        # 获取当前帧对象
        frame = inspect.currentframe()
        assert frame is not None
        # 获取调用帧对象
        calling_frame = frame.f_back
        assert calling_frame is not None
        # 获取调用帧中的指令列表
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        
        # 根据 Python 版本选择不同的索引方式获取当前指令索引
        if sys.version_info >= (3, 11):
            from bisect import bisect_left
            inst_idx = bisect_left(inst_list, calling_frame.f_lasti, key=lambda x: x.offset)
        else:
            inst_idx = calling_frame.f_lasti // 2
        
        # 获取当前指令对象
        inst = inst_list[inst_idx]
        
        # 如果当前指令为 'UNPACK_SEQUENCE'，返回一个生成器对象
        if inst.opname == 'UNPACK_SEQUENCE':
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        # 否则调用 tracer 对象的 iter 方法
        return self.tracer.iter(self)

    # 定义一个绝对值方法，返回一个调用函数的代理对象
    def __abs__(self):
        return self.tracer.create_proxy('call_function', operator.abs, (self,), {})

    # 定义一个布尔值方法，根据 tracer 中的 trace_asserts 属性做不同的处理
    def __bool__(self) -> bool:
        if self.tracer.trace_asserts:
            # 检查布尔值是否在断言中使用，通过字节码模式来判断
            frame = inspect.currentframe()
            assert frame is not None
            # 获取调用帧对象
            calling_frame = frame.f_back
            assert calling_frame is not None
            # 获取调用帧中的指令列表
            insts = list(dis.get_instructions(calling_frame.f_code))
            
            # 根据 Python 版本选择不同的索引方式获取当前指令索引
            if sys.version_info >= (3, 11):
                from bisect import bisect_left
                cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
            else:
                cur = calling_frame.f_lasti // 2
            
            # 获取当前指令对象
            inst = insts[cur]

            # 如果当前指令为 'POP_JUMP_IF_TRUE'，进一步检查是否符合断言的字节码模式
            if inst.opname == 'POP_JUMP_IF_TRUE':
                first = insts[cur + 1]
                assert inst.arg is not None
                last = insts[inst.arg // 2 - 1]
                # 判断字节码模式是否以断言错误开始并以 RAISE_VARARGS 结束
                starts_with_assert = (first.opname == 'LOAD_GLOBAL' and first.argval == 'AssertionError'
                                      or first.opname == 'LOAD_ASSERTION_ERROR')
                if starts_with_assert and last.opname == 'RAISE_VARARGS':
                    # 创建一个调用函数的代理对象并返回 True
                    self.tracer.create_proxy('call_function', assert_fn, (self,), {})
                    return True

        # 否则调用 tracer 对象的 to_bool 方法
        return self.tracer.to_bool(self)

    # 定义一个 keys 方法，返回 tracer 对象的 keys 方法的结果
    @compatibility(is_backward_compatible=True)
    def keys(self):
        return self.tracer.keys(self)

    # 定义一个长度方法，抛出运行时异常，不支持符号跟踪中的长度操作
    def __len__(self):
        raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
                           "this call to be recorded, please call torch.fx.wrap('len') at "
                           "module scope")
    # 定义用于 Torch 函数的特殊方法，支持函数的追踪和代理调用
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        # 如果 args 或 kwargs 为 None，则初始化为空元组或空字典
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        # tracers 字典用于存储找到的追踪器对象
        tracers : Dict[Any, None] = {}

        # 内部函数，用于遍历并记录 args 和 kwargs 中的追踪器对象
        def find_tracer(a):
            if isinstance(a, cls):
                tracers[a.tracer] = None
        
        # 遍历 args 和 kwargs，将其中的追踪器对象记录到 tracers 字典中
        torch.fx.node.map_aggregate(args, find_tracer)
        torch.fx.node.map_aggregate(kwargs, find_tracer)

        # 如果找到了多个不同的追踪器对象，则抛出 RuntimeError 异常
        if len(tracers) > 1:
            raise RuntimeError(f'Found multiple different tracers {list(tracers.keys())} while '
                               f'trying to trace operations {orig_method}')
        
        # 获取唯一的追踪器对象
        tracer = next(iter(tracers.keys()))

        # 根据不同的原始方法类型进行处理
        if isinstance(orig_method, torch._C.ScriptMethod):
            # 如果是 Torch 脚本方法，则调整 args 并返回代理对象
            args = (orig_method.owner,) + args
            return tracer.create_proxy('call_method', orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            # 如果是张量方法或属性，则返回相应的代理调用
            return tracer.create_proxy('call_method', orig_method.__name__, args, kwargs)
        else:
            # 如果是其他类型的方法，检查是否是高阶运算符，若是则抛出异常
            if isinstance(orig_method, torch._ops.HigherOrderOperator):
                # TODO: 定义如何符号化追踪高阶运算符
                raise RuntimeError("Unable to symbolically trace HigherOrderOperators")
            # 否则返回函数调用的代理对象
            return tracer.create_proxy('call_function', orig_method, args, kwargs,
                                       name=tracer.graph._target_to_str(orig_method.__name__))
# 使用装饰器兼容性修饰类，支持向后兼容
@compatibility(is_backward_compatible=True)
class Attribute(Proxy):
    # 使用兼容性修饰器初始化方法，接受根代理和属性名作为参数
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: Proxy, attr: str):
        # 设置根代理和属性名
        self.root = root
        self.attr = attr
        # 获取跟踪器对象并保存
        self.tracer = root.tracer
        # 初始化节点为 None
        self._node: Optional[Node] = None

    # 定义属性访问器 node
    @property
    def node(self):
        # 如果节点为 None，则创建节点，用于属性访问
        if self._node is None:
            # 创建代理对象的节点，用于调用函数
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    # 定义对象调用方法
    def __call__(self, *args, **kwargs):
        # 创建代理对象并调用方法
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)


# 使用兼容性修饰器标记类，表明不支持向后兼容
@compatibility(is_backward_compatible=False)
class ParameterProxy(Proxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing
    """
    # 初始化方法，接受跟踪器、节点、名称和参数对象
    def __init__(self, tracer: TracerBase, node: Node, name, param):
        # 调用父类初始化方法
        super().__init__(node, tracer)
        # 断言参数对象是 torch.nn.Parameter 类型
        assert isinstance(param, torch.nn.Parameter)
        # 保存参数对象和名称
        self.param = param
        self.name = name

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        return f'ParameterProxy({self.name})'

    # 返回参数对象的形状
    @property
    def shape(self):
        return self.param.shape

    # 返回参数对象的大小
    def size(self):
        return self.param.size()

    # 返回参数对象的维度
    def dim(self):
        return self.param.dim()

    # 返回参数对象的维度数
    @property
    def ndim(self):
        return self.param.ndim

    # 返回参数对象的元素数量
    def numel(self):
        return self.param.numel()

    # 返回参数对象的元素数量
    def nelement(self):
        return self.param.nelement()


# 遍历魔术方法列表，为每个方法动态创建函数并添加到 Proxy 类中
for method in magic_methods:
    def _scope(method):
        def impl(*args, **kwargs):
            # 获取第一个参数的跟踪器对象
            tracer = args[0].tracer
            # 获取操作符模块中指定方法的函数对象
            target = getattr(operator, method)
            # 创建代理对象并调用函数
            return tracer.create_proxy('call_function', target, args, kwargs)
        # 设置函数名称
        impl.__name__ = method
        # 构造魔术方法的名称
        as_magic = f'__{method.strip("_")}__'
        # 将函数绑定到 Proxy 类中作为魔术方法
        setattr(Proxy, as_magic, impl)


# 定义函数用于动态创建反射魔术方法，并将其添加到 Proxy 类中
def _define_reflectable(orig_method_name):
    # 构造反射魔术方法名称
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        # 获取操作符模块中指定方法的函数对象
        target = getattr(operator, orig_method_name)
        # 创建代理对象并调用函数，交换参数位置
        return self.tracer.create_proxy('call_function', target, (rhs, self), {})
    # 设置函数名称和限定名称
    impl.__name__ = method_name
    impl.__qualname__ = method_name
    # 将函数绑定到 Proxy 类中作为反射魔术方法
    setattr(Proxy, method_name, impl)


# 遍历反射魔术方法列表，为每个方法调用 _define_reflectable 函数定义反射魔术方法
for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)
```