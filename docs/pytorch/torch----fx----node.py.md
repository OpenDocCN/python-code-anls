# `.\pytorch\torch\fx\node.py`

```py
# mypy: ignore-errors

# 导入必要的类型
from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set

# 导入兼容性模块
from ._compatibility import compatibility

# 导入不可变集合
from .immutable_collections import immutable_dict, immutable_list

# 导入PyTorch库
import torch

# 导入内置的Python函数和变量
import builtins

# 导入Python类型相关的模块
import types

# 导入检查模块
import inspect

# 导入警告模块
import warnings

# 导入PyTorch FX中的运算模式规范化函数
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair

# 导入自定义操作
from .._ops import ops as _ops

# 导入PyTorch C++扩展中的_NodeBase类
from torch._C import _NodeBase

# 如果在类型检查环境下，则导入图(Graph)类
if TYPE_CHECKING:
    from .graph import Graph

# 声明本模块中公开的符号
__all__ = ['Node', 'map_arg', 'map_aggregate', "has_side_effect"]

# 定义基本参数类型的联合类型
BaseArgumentTypes = Union[str, int, float, bool, complex, torch.dtype,
                          torch.Tensor, torch.device, torch.memory_format, torch.layout, torch._ops.OpOverload]

# 获取BaseArgumentTypes的实际参数类型
base_types = BaseArgumentTypes.__args__  # type: ignore[attr-defined]

# 目标类型，可以是可调用对象或字符串
Target = Union[Callable[..., Any], str]

# 参数类型的联合类型，可以是元组、列表、字典等
Argument = Optional[Union[
    Tuple[Any, ...],  # 实际上是Argument类型，但mypy无法表示递归类型
    List[Any],        # 实际上是Argument类型
    Dict[str, Any],   # 实际上是Argument类型
    slice,            # 切片类型，但在typing中slice不是模板类型
    range,
    'Node',           # 节点类型
    BaseArgumentTypes # 基本参数类型
]]

# 需要在调度前保留的具有副作用的函数集合
_side_effectful_need_to_be_preserved_pre_dispatch: Set[Callable] = {
    torch._C._set_grad_enabled,
    torch.amp._enter_autocast,
    torch.amp._exit_autocast,
}

# 需要标记为具有副作用的函数集合，用于优化处理
_side_effectful_functions: Set[Callable] = {
    torch._assert,
    torch._assert_async,
    _ops.aten._assert_async.msg,
    _ops.aten._assert_scalar.default,
    _ops.aten.copy_.default,
    _ops.aten.set_.source_Tensor,
    _ops.aten.index_put_.default,
    _ops.aten.sym_constrain_range.default,
    _ops.aten.sym_constrain_range_for_size.default,
    _ops.profiler._record_function_enter,
    _ops.profiler._record_function_enter_new,
    _ops.profiler._record_function_exit,
    _ops.inductor.accumulate_grad_.default,
} | _side_effectful_need_to_be_preserved_pre_dispatch

# 如果_ops.inductor中存在"resize_storage_bytes_"函数，则添加到_side_effectful_functions集合中
if hasattr(_ops.inductor, "resize_storage_bytes_"):
    _side_effectful_functions.add(_ops.inductor.resize_storage_bytes_.default)


# 标记具有副作用的函数，用于后续处理
@compatibility(is_backward_compatible=False)
def has_side_effect(fn: Callable) -> Callable:
    _side_effectful_functions.add(fn)
    return fn


# 获取原始方法所在的模块名称
def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    name = orig_method.__name__
    module = orig_method.__module__
    # 如果模块已知，则返回模块名称
    if module is not None:
        return module
    # 否则在torch和torch.nn.functional中猜测
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    # 如果找不到模块，则引发异常
    raise RuntimeError(f'cannot find module for {orig_method}')

# 从CPython typing模块中借用的代码
# https://github.com/python/cpython/blob/f90dc36c15d7fee0efaf6d39e97be0bdf2683e93/Lib/typing.py#L156
# 返回对象的 repr() 表示，特别处理类型（内部辅助函数）。
# 如果 obj 是类型，则返回一个比默认的 type.__repr__ 更短的版本，基于模块和限定名，通常足以唯一标识一个类型。
# 对于其他所有情况，我们都会退回到 repr(obj)。
def _type_repr(obj):
    if isinstance(obj, type):
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is ...:
        return '...'
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)

# 获取函数的限定名称字符串。
# 如果 func 函数是内置的函数，如 getattr，则直接返回其名称。
# 如果 func 是 torch.Tensor 的方法描述符或包装器描述符，并且与 getattr(torch.Tensor, func.__name__, None) 相同，则返回形如 "torch.Tensor.{func.__name__}" 的字符串。
# 对于匿名函数（lambda），尝试获取其在模块中的定义名称。
# 调整模块名称，以适应特定的情况（如 "torch._ops" 被调整为 "torch.ops"）。
# 修正特定情况下的名称不匹配问题（如 module 是 "torch" 并且 name 是 "segment_reduce" 时，将 name 更改为 "_segment_reduce"）。
def _get_qualified_name(func: Callable[..., Any]) -> str:
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    if (isinstance(func, (types.MethodDescriptorType, types.WrapperDescriptorType))
       and func is getattr(torch.Tensor, func.__name__, None)):
        return f"torch.Tensor.{func.__name__}"
    name = func.__name__
    if name == "<lambda>":
        try:
            name = inspect.getsource(func).split("=")[0].strip()
        except Exception as e:
            raise RuntimeError("Unable to represent lambda") from e
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    if module == "torch" and name == "segment_reduce":
        name = "_" + name
    return f'{module}.{name}'

# 格式化参数的字符串表示。
# 如果参数具有 _custom_fx_repr_fn 属性，则调用该方法返回其自定义表示。
# 如果参数是列表，则格式化列表的元素。
# 如果参数是元组，则格式化元组的元素。
# 如果参数是字典，则格式化字典的键值对。
# 如果参数是 Node 类型，则返回以 '%' 开头的参数字符串。
# 否则，返回参数的字符串表示。
def _format_arg(arg, max_list_len=float('inf')) -> str:
    if hasattr(arg, '_custom_fx_repr_fn'):
        return arg._custom_fx_repr_fn()
    elif isinstance(arg, list):
        items = ', '.join(_format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len)
        maybe_len = '' if len(arg) < max_list_len + 1 else f', ...[total_len={len(arg)}]'
        return f'[{items}{maybe_len}]'
    elif isinstance(arg, tuple):
        items = ', '.join(_format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len)
        maybe_len = '' if len(arg) < max_list_len + 1 else f', ...[total_len={len(arg)}]'
        maybe_comma = ',' if len(arg) == 1 else ''
        return f'({items}{maybe_comma}{maybe_len})'
    elif isinstance(arg, dict):
        items_str = ', '.join(f'{k}: {_format_arg(v)}' for k, v in arg.items())
        return f'{{{items_str}}}'
    
    if isinstance(arg, Node):
        return '%' + str(arg)
    else:
        return str(arg)

# Node 类，表示图中单个操作的数据结构。
# 大部分情况下，Node 表示对各种实体的调用点。
@compatibility(is_backward_compatible=True)
class Node(_NodeBase):
    """
    ``Node`` 是表示 ``Graph`` 内个别操作的数据结构。大部分情况下，Node 表示对各种实体的调用点，
    """
    """
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
      ``args`` and ``kwargs`` are don't-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *excluding the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    """

    # 定义一个兼容性函数修饰器，支持向后兼容
    @compatibility(is_backward_compatible=True)
    def __getstate__(self):
        # 复制当前对象的字典状态
        state = self.__dict__.copy()
        # 添加额外的状态信息
        state["_erased"] = self._erased
        state["_prev"] = self._prev
        state["_next"] = self._next
        return state

    # 定义一个函数，用于设置对象的状态
    def __setstate__(self, state):
        # 弹出并获取特定的状态属性
        _erased = state.pop("_erased")
        _prev = state.pop("_prev")
        _next = state.pop("_next")
        # 更新对象的字典状态
        self.__dict__.update(state)
        # 恢复额外的状态信息
        self._erased = _erased
        self._prev = _prev
        self._next = _next

    # 定义一个属性方法，用于访问下一个节点
    @property
    def next(self) -> 'Node':
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
        return self._next

    # 定义一个属性方法，用于访问前一个节点
    @property
    def prev(self) -> 'Node':
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
        return self._prev
    # 使用装饰器标记方法兼容性，表明该方法是向后兼容的
    @compatibility(is_backward_compatible=True)
    def prepend(self, x: 'Node') -> None:
        """
        在图中节点列表中此节点之前插入节点 x。示例::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): 要插入在此节点之前的节点。必须是同一图中的成员节点。
        """
        # 断言节点 x 和当前节点属于同一个图
        assert self.graph == x.graph, "Attempting to move a Node into a different Graph"
        
        # 如果插入节点和当前节点是同一个节点，发出警告并返回
        if self == x:
            warnings.warn("Trying to prepend a node to itself. This behavior has no effect on the graph.")
            return
        
        # 从节点列表中移除节点 x
        x._remove_from_list()
        
        # 获取当前节点的前一个节点
        p = self._prev
        
        # 调整指针，将 x 插入到当前节点之前
        p._next, x._prev = x, p
        x._next, self._prev = self, x
        
        # 计算节点 x 的排序键
        psk = x._prev._sort_key
        nsk = x._next._sort_key
        
        # 根据相邻节点的排序键长度，调整节点 x 的排序键
        if len(psk) > len(nsk):
            idx: int
            *prefix, idx = psk[:len(nsk) + 1]
            x._sort_key = (*prefix, idx + 1)
        elif len(psk) < len(nsk):
            *prefix, idx = nsk[:len(psk) + 1]
            x._sort_key = (*prefix, idx - 1)
        else:  # 长度相同，增加长度1
            x._sort_key = (*psk, 0)

    # 比较运算符重载，实现节点之间按排序键进行比较
    def __gt__(self, other: 'Node'):
        return self._sort_key > other._sort_key

    # 比较运算符重载，实现节点之间按排序键进行比较
    def __lt__(self, other: 'Node'):
        return self._sort_key < other._sort_key

    # 比较运算符重载，实现节点之间按排序键进行比较
    def __ge__(self, other: 'Node'):
        return self > other or self == other

    # 比较运算符重载，实现节点之间按排序键进行比较
    def __le__(self, other: 'Node'):
        return self < other or self == other

    # 使用装饰器标记方法兼容性，表明该方法是向后兼容的
    @compatibility(is_backward_compatible=True)
    def append(self, x: 'Node') -> None:
        """
        在图中节点列表中此节点之后插入节点 x。
        等效于 ``self.next.prepend(x)``。

        Args:
            x (Node): 要插入在此节点之后的节点。必须是同一图中的成员节点。
        """
        # 调用当前节点的下一个节点的 prepend 方法，将 x 插入到当前节点之后
        self._next.prepend(x)

    # 从节点列表中移除当前节点
    def _remove_from_list(self):
        p, n = self._prev, self._next
        p._next, n._prev = n, p

    # 属性装饰器，返回此节点的参数元组
    @property
    def args(self) -> Tuple[Argument, ...]:
        """
        返回此 ``Node`` 的参数元组。参数的解释依赖于节点的操作码。
        更多信息请参见 :class:`Node` 的文档字符串。
        """
        return self._args

    # args 属性的 setter 方法，设置此节点的参数元组
    @args.setter
    def args(self, a : Tuple[Argument, ...]):
        """
        设置此节点的参数元组。参数的解释依赖于节点的操作码。
        更多信息请参见 ``fx.Graph`` 的文档字符串。
        """
        # 不要直接调用 `__update_args_kwargs`。正确的方式是通过直接赋值来设置 `args`，例如 `node.args = new_args`
        self.__update_args_kwargs(map_arg(a, lambda x: x), self._kwargs)  # type: ignore[arg-type]
    @property
    def kwargs(self) -> Dict[str, Argument]:
        """
        返回该节点的关键字参数字典。参数的解释取决于节点的操作码。
        更多信息请参阅:class:`Node`的文档字符串。
    
        允许对此属性进行赋值。赋值后会自动更新所有使用和用户的记录。
        """
        return self._kwargs
    
    @kwargs.setter
    def kwargs(self, k : Dict[str, Argument]):
        """
        设置该节点的关键字参数字典。参数的解释取决于节点的操作码。
        更多信息请参阅``fx.Graph``的文档字符串。
    
        注意：不要直接调用`__update_args_kwargs`。正确的设置`args`的方式是直接赋值，如 `node.kwargs = new_kwargs`
        """
        self.__update_args_kwargs(self._args, map_arg(k, lambda x: x))  # type: ignore[arg-type]
    
    @property
    def all_input_nodes(self) -> List['Node']:
        """
        返回所有作为该节点输入的节点列表。等同于遍历`args`和`kwargs`，仅收集是节点的值。
    
        返回：
            包含在该节点的`args`和`kwargs`中的节点列表，按照顺序返回。
        """
        return list(self._input_nodes.keys())
    
    @compatibility(is_backward_compatible=True)
    def update_arg(self, idx : int, arg : Argument) -> None:
        """
        更新现有的位置参数以包含新值`arg`。调用后，`self.args[idx] == arg`。
    
        Args:
            idx (int): 要更新的`self.args`中元素的索引。
            arg (Argument): 要写入`args`中的新参数值。
        """
        args = list(self.args)
        args[idx] = arg
        self.args = tuple(args)
    
    @compatibility(is_backward_compatible=True)
    def insert_arg(self, idx : int, arg : Argument) -> None:
        """
        在参数列表中的指定索引处插入位置参数。
    
        Args:
            idx (int): 要在`self.args`中插入新参数之前的元素索引。
            arg (Argument): 要插入到`args`中的新参数值。
        """
        assert 0 <= idx <= len(self.args), "insert_args index must be between 0 and len(self.args)"
        args_left = self.args[:idx]
        args_right = self.args[idx:]
    
        self._args = args_left + (arg,) + args_right
    
        _new_input_nodes: Dict[Node, None] = {}
        map_arg(arg, _new_input_nodes.setdefault)
    
        for new_use in _new_input_nodes.keys():
            if new_use not in self._input_nodes:
                self._input_nodes.setdefault(new_use)
                new_use.users.setdefault(self)
    # 更新对象的关键字参数，将指定键的值更新为新的参数值 arg
    def update_kwarg(self, key: str, arg: Argument) -> None:
        # 创建一个新的 kwargs 字典，复制自身的关键字参数
        kwargs = dict(self.kwargs)
        # 更新指定键的参数值为 arg
        kwargs[key] = arg
        # 将对象的关键字参数更新为新的 kwargs 字典
        self.kwargs = kwargs
    
    @property
    def stack_trace(self) -> Optional[str]:
        """
        返回在跟踪期间记录的 Python 堆栈跟踪（如果有的话）。
        在使用 fx.Tracer 进行跟踪时，通常由 Tracer.create_proxy 填充此属性。
        若要在调试目的下记录跟踪期间的堆栈跟踪，请在 Tracer 实例上设置 record_stack_traces = True。
        使用 dynamo 进行跟踪时，默认情况下，此属性将由 OutputGraph.create_proxy 填充。
    
        堆栈跟踪字符串的最内部帧位于字符串的末尾。
        """
        return self.meta.get("stack_trace", None)
    
    @stack_trace.setter
    def stack_trace(self, trace: Optional[str]):
        # 将给定的堆栈跟踪字符串设置为 meta 字典中的 "stack_trace" 键的值
        self.meta["stack_trace"] = trace
    
    def __update_args_kwargs(self, new_args: Tuple['Argument', ...], new_kwargs: Dict[str, 'Argument']):
        """
        此 API 是内部使用的。请不要直接调用它。
        """
        # 更新对象的位置参数（new_args）和关键字参数（new_kwargs）
        self._args = new_args
        self._kwargs = new_kwargs
    
        # 清除之前使用对象的引用关系
        for old_use in self._input_nodes.keys():
            old_use.users.pop(self)
    
        # 重置输入节点映射
        self._input_nodes = {}
        # 使用 map_arg 函数将位置参数和关键字参数映射到输入节点
        map_arg(self._args, self._input_nodes.setdefault)
        map_arg(self._kwargs, self._input_nodes.setdefault)
    
        # 更新新的使用对象的引用关系
        for new_use in self._input_nodes.keys():
            new_use.users.setdefault(self)
    
    def __repr__(self) -> str:
        # 如果定义了自定义的 repr_fn 函数，则调用它来生成对象的字符串表示形式
        if self._repr_fn:
            return self._repr_fn(self)
        # 否则返回对象的名称
        return self.name
    # 将目标对象的打印输出更加用户友好化
    def _pretty_print_target(self, target):
        """
        Make target printouts more user-friendly.
        1) builtins will be printed as `builtins.xyz`
        2) operators will be printed as `operator.xyz`
        3) other callables will be printed with qualified name, e.g. torch.add
        """
        # 如果目标是字符串，直接返回
        if isinstance(target, str):
            return target
        # 如果目标对象具有 '__module__' 属性
        if hasattr(target, '__module__'):
            # 如果目标对象没有 '__name__' 属性
            if not hasattr(target, '__name__'):
                # 为了防御性编程，如果没有 '__name__'，获取限定名称（qualified name）
                # 不确定这种情况是否会发生在 'operator' 或 'builtins' 的成员中
                return _get_qualified_name(target)
            # 如果目标对象的模块是 'builtins'
            if target.__module__ == 'builtins':
                return f'builtins.{target.__name__}'
            # 如果目标对象的模块是 '_operator'
            elif target.__module__ == '_operator':
                return f'operator.{target.__name__}'
        # 返回目标对象的限定名称
        return _get_qualified_name(target)

    # 标记此函数为向后兼容，并且参数 is_backward_compatible 设置为 True
    @compatibility(is_backward_compatible=True)
    def format_node(self,
                    placeholder_names: Optional[List[str]] = None,
                    maybe_return_typename: Optional[List[str]] = None) -> Optional[str]:
        """
        Return a descriptive string representation of ``self``.

        This method can be used with no arguments as a debugging
        utility.

        This function is also used internally in the ``__str__`` method
        of ``Graph``. Together, the strings in ``placeholder_names``
        and ``maybe_return_typename`` make up the signature of the
        autogenerated ``forward`` function in this Graph's surrounding
        GraphModule. ``placeholder_names`` and ``maybe_return_typename``
        should not be used otherwise.

        Args:
            placeholder_names: A list that will store formatted strings
                representing the placeholders in the generated
                ``forward`` function. Internal use only.
            maybe_return_typename: A single-element list that will store
                a formatted string representing the output of the
                generated ``forward`` function. Internal use only.

        Returns:
            str: If 1) we're using ``format_node`` as an internal helper
                in the ``__str__`` method of ``Graph``, and 2) ``self``
                is a placeholder Node, return ``None``. Otherwise,
                return a descriptive string representation of the
                current Node.
        """
        # 如果节点是占位符类型
        if self.op == 'placeholder':
            # 断言占位符的目标是字符串类型
            assert isinstance(self.target, str)
            # 构造节点的字符串表示形式
            arg_str = self.target
            arg_str += arg_str + f': {_type_repr(self.type)}' if self.type else ''
            # 如果传入了占位符列表，则将节点字符串表示形式添加到列表中并返回None
            if placeholder_names:
                placeholder_names.append(arg_str)
                return None
            # 否则，构造并返回节点的详细描述字符串
            maybe_typename = f'{_type_repr(self.type)} ' if self.type else ''
            default_val = '(default=' + str(self.args[0]) + ')' if self.args else ''
            return f'%{self.name} : {maybe_typename}[num_users={len(self.users)}] = {self.op}[target={self.target}]{default_val}'
        # 如果节点是获取属性类型
        elif self.op == 'get_attr':
            # 构造并返回节点的获取属性类型的详细描述字符串
            maybe_typename = f'{_type_repr(self.type)} ' if self.type is not None else ''
            return f'%{self.name} : {maybe_typename}[num_users={len(self.users)}] = ' \
                   f'{self.op}[target={self._pretty_print_target(self.target)}]'
        # 如果节点是输出类型
        elif self.op == 'output':
            # 如果节点有类型信息并且传入了返回类型列表，则在返回类型列表中存储返回类型的格式化字符串
            if self.type and maybe_return_typename:
                maybe_return_typename[0] = f' -> {_type_repr(self.type)}'
            # 返回输出节点的字符串表示形式
            return f'return {self.args[0]}'
        # 对于其他类型的节点
        else:
            # 构造并返回节点的一般描述字符串
            maybe_typename = f'{_type_repr(self.type)} ' if self.type is not None else ''
            return f'%{self.name} : {maybe_typename}[num_users={len(self.users)}] = ' \
                   f'{self.op}[target={self._pretty_print_target(self.target)}](' \
                   f'args = {_format_arg(self.args)}, kwargs = {_format_arg(self.kwargs)})'
    # 使用装饰器 @compatibility(is_backward_compatible=True) 标记函数为向后兼容
    def replace_all_uses_with(self,
                              replace_with : 'Node',
                              delete_user_cb: Callable[['Node'], bool] = lambda user: True,
                              *,
                              propagate_meta=False
                              ) -> List['Node']:
        """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.
    
        Args:
    
            replace_with (Node): The node to replace all uses of ``self`` with.
            delete_user_cb (Callable): Callback that is called to determine
              whether a given user of the self node should be removed.
            propagate_meta (bool): Whether or not to copy all properties
              on the .meta field of the original node onto the replacement node.
              For safety, this is only valid to do if the replacement node
              doesn't already have an existing .meta field.
    
        Returns:
    
            The list of Nodes on which this change was made.
        """
        # 如果 propagate_meta 为 True，则确保 replace_with 的 .meta 字段为空，以确保安全地复制原始节点的 .meta 字段
        if propagate_meta:
            assert len(replace_with.meta) == 0, \
                'Called node.replace_all_uses_with(replace_with, propagate_meta=True), ' \
                'but replace_with already has .meta keys'
            # 将原始节点 self 的所有 .meta 键值对复制到替换节点 replace_with 的 .meta 字段中
            for k, v in self.meta.items():
                replace_with.meta[k] = v
        # 获取所有使用当前节点 self 的节点列表，并准备处理
        to_process = list(self.users)
        # 存储跳过的节点列表
        skipped = []
        # 获取当前节点所属的模块 m
        m = self.graph.owning_module
        # 遍历所有使用当前节点 self 的节点
        for use_node in to_process:
            # 根据 delete_user_cb 回调函数决定是否删除当前使用节点 use_node
            if not delete_user_cb(use_node):
                skipped.append(use_node)
                continue
    
            # 定义一个函数 maybe_replace_node，用于替换节点中的 self 节点为 replace_with 节点
            def maybe_replace_node(n : Node) -> Node:
                if n == self:
                    return replace_with
                else:
                    return n
    
            # 如果模块 m 存在 _replace_hook 属性，则调用该属性进行节点替换的钩子操作
            if getattr(m, "_replace_hook", None):
                m._replace_hook(old=self, new=replace_with.name, user=use_node)
    
            # 使用 map_arg 函数将 use_node 的参数和关键字参数中的 self 节点替换为 replace_with 节点
            new_args = map_arg(use_node.args, maybe_replace_node)
            new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
            # 断言新参数和关键字参数的类型分别为 tuple 和 dict
            assert isinstance(new_args, tuple)
            assert isinstance(new_kwargs, dict)
            # 更新 use_node 的参数和关键字参数为新的替换值
            use_node.__update_args_kwargs(new_args, new_kwargs)
    
        # 断言处理后的使用节点数量与跳过节点的数量之和为 0，确保所有使用节点都已处理或跳过
        assert len(self.users) - len(skipped) == 0
        # 返回所有处理过的节点列表，排除了跳过的节点
        return [n for n in to_process if n not in skipped]
    def is_impure(self):
        """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Returns:

            bool: If the op is impure or not.
        """
        # 检查操作是否为占位符或输出，若是则返回 True
        if self.op in {"placeholder", "output"}:
            return True

        # 检查是否为不纯的函数调用
        if self.op == "call_function":
            return self.target in _side_effectful_functions

        # 检查是否为不纯的模块调用
        if self.op == "call_module":
            assert (
                self.graph.owning_module is not None
            ), "self.graph.owning_module not set for purity check"
            target_mod = self.graph.owning_module.get_submodule(self.target)
            assert (
                target_mod is not None
            ), f"Did not find expected submodule target {self.target}"
            # 返回目标模块的 _is_impure 属性，若不存在则默认为 False
            return getattr(target_mod, "_is_impure", False)

        # 默认情况下认为操作是纯净的
        return False

    @compatibility(is_backward_compatible=False)
    def normalized_arguments(
            self, root : torch.nn.Module, arg_types : Optional[Tuple[Any]] = None,
            kwarg_types : Optional[Dict[str, Any]] = None,
            normalize_to_only_use_kwargs : bool = False) -> Optional[ArgsKwargsPair]:
        """
        Returns normalized arguments to Python targets. This means that
        `args/kwargs` will be matched up to the module/functional's
        signature and return exclusively kwargs in positional order
        if `normalize_to_only_use_kwargs` is true.
        Also populates default values. Does not support positional-only
        parameters or varargs parameters.

        Supports module calls.

        May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

        Args:
            root (torch.nn.Module): Module upon which to resolve module targets.
            arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
            kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
            normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

        Returns:

            Returns NamedTuple ArgsKwargsPair, or `None` if not successful.
        """
        # 如果操作为函数调用
        if self.op == 'call_function':
            assert callable(self.target)
            # 调用 normalize_function 函数来规范化函数的参数
            return normalize_function(self.target, self.args, self.kwargs, arg_types, kwarg_types)  # type: ignore[arg-type]
        # 如果操作为模块调用
        elif self.op == 'call_module':
            assert isinstance(self.target, str)
            # 调用 normalize_module 函数来规范化模块的参数
            return normalize_module(root, self.target, self.args, self.kwargs)  # type: ignore[arg-type]

        # 如果操作既不是函数调用也不是模块调用，则返回 None
        return None

    @compatibility(is_backward_compatible=True)
    def replace_input_with(self, old_input: 'Node', new_input: 'Node'):
        """
        Loop through input nodes of ``self``, and replace all instances of
        ``old_input`` with ``new_input``.

        Args:
            old_input (Node): The old input node to be replaced.
            new_input (Node): The new input node to replace ``old_input``.
        """
        def maybe_replace_node(n : Node) -> Node:
            # Return new_input if the current node n is equal to old_input, otherwise return n unchanged
            return new_input if n == old_input else n

        # Access the module that owns the graph
        m = self.graph.owning_module
        # Check if the owning module has a "_replace_hook" attribute and call it with old_input, new_input's name, and self as arguments
        if getattr(m, "_replace_hook", None):
            m._replace_hook(old=old_input, new=new_input.name, user=self)

        # Map maybe_replace_node function over self.args to create new_args
        new_args = map_arg(self.args, maybe_replace_node)
        # Map maybe_replace_node function over self.kwargs to create new_kwargs
        new_kwargs = map_arg(self.kwargs, maybe_replace_node)
        # Assert that new_args is a tuple and new_kwargs is a dictionary
        assert isinstance(new_args, tuple)
        assert isinstance(new_kwargs, dict)
        # Update self's arguments and keyword arguments with new_args and new_kwargs
        self.__update_args_kwargs(new_args, new_kwargs)

    def _rename(self, candidate: str):
        # If candidate is the same as self's name, return without doing anything
        if candidate == self.name:
            return
        # Create a new name using the graph's namespace
        name = self.graph._graph_namespace.create_name(candidate, None)
        # Set self's name attribute to the newly created name
        self.name = name
        # Rename self within the graph's namespace
        self.graph._graph_namespace._rename_object(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # If setting the 'name' attribute and self already has a 'name' attribute
        if name == 'name' and hasattr(self, "name"):
            # Access the module that owns the graph
            m = self.graph.owning_module
            # Check if the owning module has a "_replace_hook" attribute
            if getattr(m, "_replace_hook", None):
                # Assert that value is a string and call _replace_hook for each user of self
                assert isinstance(value, str)
                for user in self.users:
                    m._replace_hook(old=self, new=value, user=user)
        update = False
        # If self has the attribute 'name' and the graph has a "_find_nodes_lookup_table" attribute
        if (
                hasattr(self, name) and
                hasattr(self.graph, "_find_nodes_lookup_table") and
                self in self.graph._find_nodes_lookup_table
        ):
            update = True
            # Remove self from the graph's _find_nodes_lookup_table
            self.graph._find_nodes_lookup_table.remove(self)
        # Set the attribute 'name' to value
        object.__setattr__(self, name, value)
        # If update is True, insert self into the graph's _find_nodes_lookup_table
        if update:
            self.graph._find_nodes_lookup_table.insert(self)
# 根据兼容性标记装饰器，声明这个函数在向后兼容的环境中有效
@compatibility(is_backward_compatible=True)
# 将函数 fn 应用于出现在参数 a 中的每个 Node 对象。a 可能是列表、元组、切片或带有字符串键的字典。
def map_arg(a: Argument, fn: Callable[[Node], Argument]) -> Argument:
    # 断言 fn 必须是可调用的，否则抛出异常
    assert callable(fn), "torch.fx.map_arg(a, fn): fn must be a callable"
    # 调用 map_aggregate 函数，对参数 a 进行处理，将 fn 应用于其中的每个 Node 对象
    return map_aggregate(a, lambda x: fn(x) if isinstance(x, Node) else x)

# 根据兼容性标记装饰器，声明这个函数在向后兼容的环境中有效
@compatibility(is_backward_compatible=True)
# 将函数 fn 应用于出现在参数 a 中的每个 Node 对象。a 可能是列表、元组、切片或带有字符串键的字典。
def map_aggregate(a: Argument, fn: Callable[[Argument], Argument]) -> Argument:
    # 如果 a 是元组类型，则对其中的每个元素递归调用 map_aggregate，并根据原始类型重新组装成元组（如果有 '_fields' 属性）
    if isinstance(a, tuple):
        t = tuple(map_aggregate(elem, fn) for elem in a)
        # 如果 a 是命名元组且有 '_fields' 属性，则重新使用原始类型进行打包
        return t if not hasattr(a, '_fields') else type(a)(*t)
    # 如果 a 是列表类型，则创建一个不可变列表，对其中的每个元素递归调用 map_aggregate
    elif isinstance(a, list):
        return immutable_list(map_aggregate(elem, fn) for elem in a)
    # 如果 a 是字典类型，则创建一个不可变字典，对其中的每个键值对递归调用 map_aggregate
    elif isinstance(a, dict):
        return immutable_dict((k, map_aggregate(v, fn)) for k, v in a.items())
    # 如果 a 是切片类型，则创建一个新的切片对象，对起始、终止和步长分别递归调用 map_aggregate
    elif isinstance(a, slice):
        return slice(map_aggregate(a.start, fn), map_aggregate(a.stop, fn), map_aggregate(a.step, fn))
    # 对于除以上类型之外的 a，直接将 fn 应用于 a 并返回结果
    else:
        return fn(a)
```