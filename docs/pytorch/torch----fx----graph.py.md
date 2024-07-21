# `.\pytorch\torch\fx\graph.py`

```py
# mypy: allow-untyped-defs
# 导入默认字典集合模块
from collections import defaultdict
# 从当前包中导入节点相关的模块
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
# 导入PyTorch中的PyTree模块
import torch.utils._pytree as pytree
# 从当前包中导入FX PyTree模块
from . import _pytree as fx_pytree
# 导入兼容性模块
from ._compatibility import compatibility
# 从torch._C模块导入_NodeIter
from torch._C import _NodeIter

# 导入操作系统模块
import os
# 导入上下文管理模块
import contextlib
# 导入类型检查相关模块
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type, Iterable
# 导入数据类模块
from dataclasses import dataclass
# 再次导入上下文管理模块
from contextlib import contextmanager
# 导入复制模块
import copy
# 导入枚举模块
import enum
# 导入PyTorch库
import torch
# 导入关键字模块
import keyword
# 导入正则表达式模块
import re
# 导入内建模块
import builtins
# 导入数学模块
import math
# 导入警告模块
import warnings
# 导入检查模块
import inspect

# 定义公开导出的模块成员列表
__all__ = ["PythonCode", "CodeGen", "Graph"]

# 如果类型检查开启，从图形模块中导入图形模块和追踪器
if TYPE_CHECKING:
    from .graph_module import GraphModule  # noqa: F401
    from ._symbolic_trace import Tracer   # noqa: F401


# 内建类型到其对应`typing`类型的映射
_origin_type_map = {
    list: List,
    dict: Dict,
    set: Set,
    frozenset: FrozenSet,
    tuple: Tuple,
}


# 函数签名类型，用于处理生成代码的主体列表
TransformCodeFunc = Callable[[List[str]], List[str]]


class _CustomBuiltin(NamedTuple):
    """每个图形全局变量中添加的额外对象集合。

    一些标准库对象的repr()在没有导入的情况下不是有效的Python代码。对于这类常见对象，
    我们将它们捆绑在每个FX图形的全局变量中。
    """
    # 从标准库中导入这个对象的方法字符串
    import_str: str
    # 由该导入字符串产生的实际对象
    obj: Any

# 自定义内建对象的字典
_custom_builtins: Dict[str, _CustomBuiltin] = {}


# 注册自定义内建对象的函数
def _register_custom_builtin(name: str, import_str: str, obj: Any):
    _custom_builtins[name] = _CustomBuiltin(import_str, obj)

# 注册一些常用的自定义内建对象
_register_custom_builtin('inf', 'from math import inf', math.inf)
_register_custom_builtin('nan', 'from math import nan', math.nan)
_register_custom_builtin('NoneType', 'NoneType = type(None)', type(None))
_register_custom_builtin('torch', 'import torch', torch)
_register_custom_builtin('device', 'from torch import device', torch.device)
_register_custom_builtin('fx_pytree', 'import torch.fx._pytree as fx_pytree', fx_pytree)
_register_custom_builtin('pytree', 'import torch.utils._pytree as pytree', pytree)


# 判断字符串是否是魔术方法（以双下划线开始和结束）
def _is_magic(x: str) -> bool:
    return x.startswith('__') and x.endswith('__')


# 将给定的字符串转换为Python风格的变量名（蛇形命名）
def _snake_case(s: str) -> str:
    """
    将给定的字符串 ``s`` 转换为Python风格的变量名

    示例:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    chars = []
    prev_lower = False
    for c in s:
        if prev_lower and c.isupper():
            chars.append('_')
        chars.append(c.lower())
        prev_lower = c.islower()
    return ''.join(chars)


# 判断对象是否来自于torch模块
def _is_from_torch(obj: Any) -> bool:
    module_name = getattr(obj, '__module__', None)
    # 如果 module_name 不是 None，则进行以下操作
    if module_name is not None:
        # 获取 module_name 的基础模块部分，即第一个点之前的部分
        base_module = module_name.partition('.')[0]
        # 检查基础模块是否为 'torch'，并且 module_name 不以指定的特定前缀开头
        return (
            base_module == 'torch' and
            not module_name.startswith("torch._dynamo.") and
            not module_name.startswith("torch._inductor.")
        )

    # 获取对象 obj 的名称
    name = getattr(obj, '__name__', None)
    # 如果名称不为 None 并且不等于 'torch'
    # 检查对象是否是 torch 或 torch.nn.functional 模块中的某个函数或类
    if name is not None and name != 'torch':
        for guess in [torch, torch.nn.functional]:
            if getattr(guess, name, None) is obj:
                return True

    # 如果以上条件都不满足，则返回 False
    return False
class _Namespace:
    """A context for associating names uniquely with objects.

    The following invariants are enforced:
    - Each object gets a single name.
    - Each name is unique within a given namespace.
    - Names generated do not shadow builtins, unless the object is indeed that builtin.
    """

    def __init__(self):
        # Dictionary to map objects to their associated unique names
        self._obj_to_name: Dict[Any, str] = {}
        # Set to track names that are not yet associated with any object
        self._unassociated_names = set()
        # Set to track all used names (ensures uniqueness)
        self._used_names: Set[str] = set()
        # Dictionary to store the count of base names to handle suffixes like `_1`, `_2`, etc.
        self._base_count: Dict[str, int] = defaultdict(int)

        # Regular expression to replace illegal characters in identifiers with '_'
        self._illegal_char_regex = re.compile('[^0-9a-zA-Z_]+')
        # Regular expression to match and handle existing suffixes like `_1`, `_2`, etc.
        self._name_suffix_regex = re.compile(r"(.*)_(\d+)$")

    def create_name(self, candidate: str, obj: Optional[Any]) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]

        # Delete illegal characters from the candidate name
        candidate = self._illegal_char_regex.sub('_', candidate)

        # If candidate becomes empty after cleaning, assign a default name
        if not candidate:
            candidate = '_unnamed'

        # If the candidate starts with a digit, prepend '_'
        if candidate[0].isdigit():
            candidate = f'_{candidate}'

        # Check if the candidate already has a numeric suffix
        match = self._name_suffix_regex.match(candidate)
        if match is None:
            base = candidate
            num = None
        else:
            base, num_str = match.group(1, 2)
            num = int(num_str)

        # If no numeric suffix exists, use the count from _base_count
        candidate = base if num is None else f'{base}_{num}'
        if not num:
            num = self._base_count[base]

        # Ensure the name is unique and not illegal
        while candidate in self._used_names or self._is_illegal_name(candidate, obj):
            num += 1
            candidate = f'{base}_{num}'

        # Mark the name as used
        self._used_names.add(candidate)
        self._base_count[base] = num

        # Associate the name with the object or mark it as unassociated
        if obj is None:
            self._unassociated_names.add(candidate)
        else:
            self._obj_to_name[obj] = candidate

        return candidate

    def associate_name_with_obj(self, name: str, obj: Any):
        """Associate a unique name with an object.

        Neither `name` nor `obj` should be associated already.
        """
        assert obj not in self._obj_to_name
        assert name in self._unassociated_names
        self._obj_to_name[obj] = name
        self._unassociated_names.remove(name)

    def _is_illegal_name(self, name: str, obj: Any) -> bool:
        # 1. Check if the name is a Python keyword
        if name in keyword.kwlist:
            return True

        # 2. Check if the name shadows a Python builtin, unless obj is that builtin
        if name in builtins.__dict__:
            return obj is not builtins.__dict__[name]

        # 3. Check if the name shadows a custom builtin
        if name in _custom_builtins:
            return obj is not _custom_builtins[name].obj

        return False
    # 定义一个方法 `_rename_object`，用于将对象重新命名为指定的名称
    def _rename_object(self, obj: Any, name: str):
        # 确保待重命名的对象在 `_obj_to_name` 字典中
        assert obj in self._obj_to_name
        # 更新 `_obj_to_name` 字典，将对象映射到新的名称
        self._obj_to_name[obj] = name
        # 将新的名称添加到 `_used_names` 集合中，表示该名称已被使用
        self._used_names.add(name)
# 定义了一个字典，将不同的 torch 数据类型映射为相应的缩写字符串
dtype_abbrs = {
    torch.bfloat16: 'bf16',  # torch.bfloat16 对应 'bf16'
    torch.float64: 'f64',    # torch.float64 对应 'f64'
    torch.float32: 'f32',    # torch.float32 对应 'f32'
    torch.float16: 'f16',    # torch.float16 对应 'f16'
    torch.float8_e4m3fn: 'f8e4m3fn',            # torch.float8_e4m3fn 对应 'f8e4m3fn'
    torch.float8_e5m2: 'f8e5m2',                # torch.float8_e5m2 对应 'f8e5m2'
    torch.float8_e4m3fnuz: 'f8e4m3fnuz',        # torch.float8_e4m3fnuz 对应 'f8e4m3fnuz'
    torch.float8_e5m2fnuz: 'f8e5m2fnuz',        # torch.float8_e5m2fnuz 对应 'f8e5m2fnuz'
    torch.complex32: 'c32',    # torch.complex32 对应 'c32'
    torch.complex64: 'c64',    # torch.complex64 对应 'c64'
    torch.complex128: 'c128',  # torch.complex128 对应 'c128'
    torch.int8: 'i8',          # torch.int8 对应 'i8'
    torch.int16: 'i16',        # torch.int16 对应 'i16'
    torch.int32: 'i32',        # torch.int32 对应 'i32'
    torch.int64: 'i64',        # torch.int64 对应 'i64'
    torch.bool: 'b8',          # torch.bool 对应 'b8'
    torch.uint8: 'u8',         # torch.uint8 对应 'u8'
    torch.uint32: 'u32',       # torch.uint32 对应 'u32'
    torch.uint64: 'u64',       # torch.uint64 对应 'u64'
}

@compatibility(is_backward_compatible=True)
@dataclass
class PythonCode:
    """
    Represents all the information necessary to exec or save a graph as Python code.
    """
    src: str  # 存储前向函数定义的 Python 源代码字符串
    globals: Dict[str, Any]  # 存储在执行 `src_def` 期间全局作用域内的值的字典
    _lineno_map: Optional[Dict[int, Optional[int]]]  # 可选的映射，从前向函数的行号到节点索引的字典


def _format_target(base: str, target: str) -> str:
    """
    根据 base 和 target 格式化目标字符串

    Args:
        base: 基础字符串
        target: 目标字符串

    Returns:
        格式化后的字符串
    """
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'  # 如果 e 不是标识符，使用 getattr 获取属性
        else:
            r = f'{r}.{e}'  # 否则直接拼接属性名
    return r


class _InsertPoint:
    """
    用于插入新节点的上下文管理器
    """
    def __init__(self, graph, new_insert):
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        self.graph._insert = self.orig_insert  # 恢复原始插入点


class _node_list:
    """
    表示图中节点列表的迭代器
    """
    def __init__(self, graph: 'Graph', direction: str = '_next'):
        assert direction in ['_next', '_prev']
        self.graph = graph
        self.direction = direction

    def __len__(self):
        return self.graph._len  # 返回图中节点的数量

    def __iter__(self):
        assert self.direction == "_prev" or self.direction == "_next"
        yield from _NodeIter(self.graph._root, self.direction == "_prev")  # 迭代图中节点的迭代器

    def __reversed__(self):
        return _node_list(self.graph, '_next' if self.direction == '_prev' else '_prev')  # 返回反向迭代器


class _PyTreeInfo(NamedTuple):
    """
    包含使用 Pytrees 时的额外信息
    """
    orig_args: List[str]  # 原始参数列表
    in_spec: pytree.TreeSpec  # 输入的树形结构规范
    out_spec: Optional[pytree.TreeSpec]  # 可选的输出的树形结构规范


@dataclass(frozen=True)
class _ParsedStackTrace:
    """
    表示解析的堆栈跟踪的最顶层帧
    """
    file: str  # 文件名
    lineno: str  # 行号
    name: str  # 名称
    code: str  # 代码片段

    def get_summary_str(self):
        """
        返回摘要字符串

        Returns:
            摘要字符串，包含文件名、行号、名称和代码片段
        """
        return f'File: {self.file}:{self.lineno} in {self.name}, code: {self.code}'


def _parse_stack_trace(stack_trace: str):
    """
    从堆栈跟踪中解析出文件名、行号和代码片段

    Args:
        stack_trace: 堆栈跟踪字符串

    Returns:
        解析得到的堆栈跟踪摘要字符串，或者 None（如果输入为 None）
    """
    if stack_trace is None:
        return None
    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    lines = stack_trace.strip().split('\n')
    # 堆栈跟踪应该从内向外，所以倒序查找以找到以 'File ' 开头的第一行
    summary_str = ""
    # 从倒数第二行开始向前遍历列表 lines
    for idx in range(len(lines) - 2, -1, -1):
        # 去除当前行两侧的空白字符，并存储到变量 line 中
        line = lines[idx].strip()
        # 使用正则表达式 pattern 匹配当前行
        matches = pattern.match(line)
        # 如果匹配成功
        if matches:
            # 提取匹配的文件名、行号和名称信息
            file = matches.group(1)
            lineno = matches.group(2)
            name = matches.group(3)
            # 下一行应该是出错的代码行，获取并去除两侧空白字符
            code = lines[idx + 1].strip()
            # 返回解析后的错误信息对象 _ParsedStackTrace
            return _ParsedStackTrace(file, lineno, name, code)
    # 如果未找到匹配的错误信息，则返回 None
    return None
@compatibility(is_backward_compatible=False)
class CodeGen:
    # 类的初始化方法，设置初始属性：_body_transformer为可选的代码转换函数，_func_name默认为"forward"
    def __init__(self):
        self._body_transformer: Optional[TransformCodeFunc] = None
        self._func_name: str = "forward"

    # 生成函数定义的方法，根据给定的自由变量和可能的返回注释生成FX函数的开头
    # 默认情况下，如果free_vars为['a', 'b']，maybe_return_annotation为''，则返回字符串形如'def {self._func_name}(a, b):'
    def gen_fn_def(self, free_vars: List[str], maybe_return_annotation: str) -> str:
        # 如果原始函数的第一个参数不是'self'，则在free_vars的开头添加'self'
        if len(free_vars) == 0 or free_vars[0] != 'self':
            free_vars.insert(0, 'self')
        return f"def {self._func_name}({', '.join(free_vars)}){maybe_return_annotation}:"

    # 生成输出语句的方法，根据给定的输出参数生成FX函数的返回语句
    # 注意：返回的语句不应该有缩进
    def generate_output(self, output_args: Argument) -> str:
        return f'return {repr(output_args)}'

    # 处理输入的方法，使得图能够接受它们作为参数
    # 非默认的代码生成可能导致函数的输入与图的输入不同
    def process_inputs(self, *args: Any) -> Any:
        return args

    # 处理图的输出，使其与代码生成的输出相同
    def process_outputs(self, outputs: Any) -> Any:
        return outputs

    # 返回额外的全局变量，如果代码生成使用额外的全局值
    # 例如，如果需要在全局上下文中使用`List`，则返回 [('List', typing.List)]
    def additional_globals(self) -> List[Tuple[str, Any]]:
        return []

    # 生成 Python 代码的内部方法，接受节点、根模块、命名空间等参数
    # 返回 PythonCode 对象，用于代码生成
    def _gen_python_code(
        self, nodes, root_module: str, namespace: _Namespace, *,
        verbose: bool = False, include_stride: bool = False, include_device: bool = False, colored: bool = False
    ) -> PythonCode:
        fn_code = ...
        globals_ = ...
        lineno_map = ...
        return PythonCode(fn_code, globals_, _lineno_map=lineno_map)


# 理想情况下，我们希望将所有的 pytree 逻辑重构到这个代码生成类中
# 不幸的是，目前有三个领域需要额外的逻辑在 FX 中
# 1. 在初始符号跟踪中，pytree 逻辑与 `concrete_args` 相关联
# 2. 在 FX 图中，我们需要访问 2 个属性 - in_spec 和 out_spec
#    由于我们无法在 FX 的 forward 中访问 `.graph`，因此需要将属性复制到模块中
# 3. 我们目前无法使用 `add_global` 注册 pytree 导入 - 原因不明
class _PyTreeCodeGen(CodeGen):
    # 初始化方法，接受一个 _PyTreeInfo 类型的参数 pytree_info
    def __init__(self, pytree_info: _PyTreeInfo):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 pytree_info 参数赋值给实例变量 self.pytree_info
        self.pytree_info: _PyTreeInfo = pytree_info

    # 处理输入数据的方法，接受任意数量的输入参数 *inputs，返回处理后的结果
    def process_inputs(self, *inputs: Any) -> Any:
        # 使用 pytree 模块的 arg_tree_leaves 函数将输入参数展平化处理
        flat_args = pytree.arg_tree_leaves(*inputs)
        # 返回处理后的结果
        return flat_args

    # 处理输出数据的方法，接受一个输出参数 out，返回处理后的结果
    def process_outputs(self, out: Any) -> Any:
        # 如果 self.pytree_info 为 None 或者 self.pytree_info.out_spec 为 None，则直接返回输出参数 out
        if self.pytree_info is None or self.pytree_info.out_spec is None:
            return out
        # 如果输出参数 out 不是列表或元组类型，则将其转换为包含单个元素的列表
        if not isinstance(out, (list, tuple)):
            out = [out]
        # 断言 self.pytree_info.out_spec 不为 None
        assert self.pytree_info.out_spec is not None
        # 使用 pytree 模块的 tree_unflatten 函数将输出参数 out 根据 self.pytree_info.out_spec 进行反展平化处理
        return pytree.tree_unflatten(out, self.pytree_info.out_spec)
        # Given a user function/model:
        #   myargs = [myargs0, myargs1]
        #   mykwargs = {'mykwargs0': ..., 'mykwargs1': ...}
        #   def forward(self, mypos, *myargs, mykey=None, **mykwargs):
        #
        # The generated code flattens all keywords into positional arguments for `forward()`
        #   e.g forward(self, mypos, myargs0, myargs1, mykey, mykwargs0, mykwargs1):
        #
        # Within `forward`, `tree_flatten_spec` still parses args and kwargs separately
        #   e.g. tree_flatten_spec(([mypos, myargs0, myargs1],
        #                          {'mykey':mykey, 'mykwargs0':mykwargs0, 'mykwargs1':mykwargs1}),
        #                          self._in_spec)
        #
        # If the user function/model does not have keywords, the dict is suppressed from tree_flatten_spec
        #   e.g. tree_flatten_spec([mypos, myargs0, myargs1]), self._in_spec)
        if self.pytree_info is None:
            # 如果没有 pytree_info，则调用父类的 gen_fn_def 方法
            return super().gen_fn_def(free_vars, maybe_return_annotation)

        fn_args = self.pytree_info.orig_args
        has_orig_self = (fn_args[0] == 'self') if len(fn_args) > 0 else False
        if has_orig_self:
            # 如果原始参数包含 'self'，则在 free_vars 列表中插入 'self'
            free_vars.insert(0, 'self')
        fn_definition = super().gen_fn_def(fn_args[:], maybe_return_annotation)

        if len(free_vars) > 0:  # pytree has placeholders in it
            # 当 free_vars 中有占位符时，处理 kwargs 的存在情况，in_spec 是一个元组 (args, kwargs)
            has_args_kwargs_tuple = self.pytree_info.in_spec.type == tuple and \
                self.pytree_info.in_spec.num_children == 2 and \
                self.pytree_info.in_spec.children_specs[0].type == tuple and \
                self.pytree_info.in_spec.children_specs[1].type == dict
            fn_kwargs = '{}'
            fn_signature = f"[{', '.join(fn_args)}], self._in_spec"
            if has_args_kwargs_tuple:
                # 如果有 args 和 kwargs 的元组形式
                count_args = self.pytree_info.in_spec.children_specs[0].num_children
                fn_args = self.pytree_info.orig_args[:count_args]
                fn_kwargs = '{' + ', '.join(f"'{k}':{v}" for k, v in zip(
                                  self.pytree_info.in_spec.children_specs[1].context,
                                  self.pytree_info.orig_args[count_args:])) + '}'
                fn_signature = f"([{', '.join(fn_args)}], {fn_kwargs}), self._in_spec"

            # Python 中 `var1: annotation1, var2: annotation2 = function_call()` 是无效的语法
            # 需要拆分成两行：
            # 一个是注解：`var1: annotation1; var2: annotation2;` （注意分号）
            # 一个是代码：`var1, var2, = function_call()`
            without_annotation = [x.split(":")[0] for x in free_vars]
            has_annotation = [x + "; " for x in free_vars if ":" in x]
            if len(has_annotation) > 0:
                fn_definition += "\n    " + "".join(has_annotation) + "\n"
            fn_definition += f"""
    {', '.join(without_annotation)}, = fx_pytree.tree_flatten_spec({fn_signature})"""
        return fn_definition



    # 将 without_annotation 列表中的元素用逗号和空格连接成字符串，然后解构赋值给左边的变量
    {', '.join(without_annotation)}, = fx_pytree.tree_flatten_spec({fn_signature})
    # 返回 fn_definition 变量的值作为函数的定义
    return fn_definition



    def generate_output(self, output_args):
        # 如果存在 pytree_info 并且 pytree_info.out_spec 不为空
        if self.pytree_info and self.pytree_info.out_spec:
            # 返回格式化字符串，调用 pytree 模块的 tree_unflatten 函数
            return f'return pytree.tree_unflatten({repr(output_args)}, self._out_spec)'
        else:
            # 否则调用父类的 generate_output 方法
            return super().generate_output(output_args)



        # 如果条件不满足，调用父类的 generate_output 方法
        return super().generate_output(output_args)
class _FindNodesLookupTable:
    """
    Side table for the graph for the purpose of doing fast queries
    用于图形快速查询的辅助表
    """
    
    def __init__(self):
        self.table: Dict[Tuple[str, Optional[Target]], Dict[Node, None]] = defaultdict(dict)
        # 初始化一个默认字典，用于存储特定键的节点集合

    def _key(self, node) -> Tuple[str, Optional[Target]]:
        return (node.op, node.target if node.op == "call_function" else None)
        # 返回节点的操作类型和目标（如果是"call_function"操作）

    def __contains__(self, node) -> bool:
        return node in self.table[self._key(node)]
        # 判断节点是否在存储结构中的特定键的集合中

    def insert(self, node: Node) -> None:
        self.table[self._key(node)][node] = None
        # 将节点插入到存储结构中的特定键的集合中

    def remove(self, node: Node) -> None:
        self.table[self._key(node)].pop(node)
        # 从存储结构中的特定键的集合中移除节点

    def find_nodes(self, *, op: str, target: Optional['Target'] = None):
        if op == "call_function":
            assert target is not None
            return dict(self.table[(op, target)]).keys()
            # 如果操作类型是"call_function"，返回具有指定目标的节点集合的键集合

        if target is None:
            return dict(self.table[(op, None)]).keys()
            # 如果目标为None，返回具有指定操作类型但无目标的节点集合的键集合

        # op 是 call_method, get_attr, call_module
        return [node for node in self.table[(op, None)].keys() if node.target == target]
        # 对于操作类型为 call_method, get_attr, call_module，返回具有指定目标的节点集合的节点列表

@compatibility(is_backward_compatible=True)
class Graph:
    """
    ``Graph`` is the main data structure used in the FX Intermediate Representation.
    It consists of a series of ``Node`` s, each representing callsites (or other
    syntactic constructs). The list of ``Node`` s, taken together, constitute a
    valid Python function.

    For example, the following code

    .. code-block:: python

        import torch
        import torch.fx

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return torch.topk(torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3)

        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

    Will produce the following Graph::

        print(gm.graph)

    .. code-block:: text

        graph(x):
            %linear_weight : [num_users=1] = self.linear.weight
            %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
            %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
            %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
            %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
            %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
            return topk_1

    For the semantics of operations represented in the ``Graph``, please see :class:`Node`.
    """

    @compatibility(is_backward_compatible=True)
    # 用于表示此类在向后兼容性方面的兼容性
    def __init__(self, owning_module: Optional["GraphModule"] = None, tracer_cls: Optional[Type["Tracer"]] = None,
                 tracer_extras: Optional[Dict[str, Any]] = None):
        """
        Construct an empty Graph.
        """
        # 初始化根节点，并设置默认值
        self._root : Node = Node(self, '', 'root', '', (), {})
        # 字典，用于记录已使用的名称和它们的计数
        self._used_names : Dict[str, int] = {}  # base name -> number
        # 设置节点插入函数，初始为根节点的前插入方法
        self._insert = self._root.prepend
        # 图中节点数量的初始值
        self._len = 0
        # 命名空间对象，用于管理命名空间相关操作
        self._graph_namespace = _Namespace()
        # 拥有该图的模块，可选参数
        self._owning_module = owning_module
        # 追踪器的类，用于跟踪图的操作
        self._tracer_cls = tracer_cls
        # 额外的追踪器参数
        self._tracer_extras = tracer_extras
        # 代码生成器，用于生成代码
        self._codegen = CodeGen()
        # 与图相关的字段
        self._co_fields : Dict[str, Any] = {}
        # 查找节点的查询表
        self._find_nodes_lookup_table = _FindNodesLookupTable()

    @property
    def owning_module(self):
        # 获取拥有该图的模块
        return self._owning_module

    @owning_module.setter
    def owning_module(self, mod: Optional["GraphModule"]):
        # 设置拥有该图的模块
        self._owning_module = mod

    @property
    def nodes(self) -> _node_list:
        """
        Get the list of Nodes that constitute this Graph.

        Note that this ``Node`` list representation is a doubly-linked list. Mutations
        during iteration (e.g. delete a Node, add a Node) are safe.

        Returns:

            A doubly-linked list of Nodes. Note that ``reversed`` can be called on
            this list to switch iteration order.
        """
        # 返回构成该图的节点列表
        return _node_list(self)

    @compatibility(is_backward_compatible=False)
    def find_nodes(self, *, op: str, target: Optional['Target'] = None, sort: bool = True):
        """
        Allows for fast query of nodes

        Args:

            op (str): the name of the operation

            target (Optional[Target]): the target of the node. For call_function,
                the target is required. For other ops, the target is optional.

            sort (bool): whether to return nodes in the order they appear on
                         on the graph.

        Returns:

            Iteratable of nodes with the requested op and target.
        """
        # 使用查找表查找具有特定操作和目标的节点列表
        node_list = self._find_nodes_lookup_table.find_nodes(op=op, target=target)
        # 如果需要排序，则对节点列表进行排序后返回
        if sort:
            return sorted(node_list)
        # 否则直接返回节点列表
        return node_list

    @compatibility(is_backward_compatible=True)
    # 定义一个方法 graph_copy，用于将给定图中的所有节点复制到当前图对象中
    # g: 源图对象，从中复制节点
    # val_map: 一个字典，用于存储从 g 中节点到当前图节点的映射关系
    # return_output_node: 是否返回输出节点，如果为 True，则同时返回输出节点
    # 返回值: 如果源图 g 具有输出节点，则返回当前图 self 中对应的输出值，否则返回 None
    def graph_copy(self, g : 'Graph', val_map : Dict[Node, Node], return_output_node=False) -> 'Optional[Argument]':
        """
        Copy all nodes from a given graph into ``self``.

        Args:

            g (Graph): The source graph from which to copy Nodes.

            val_map (Dict[Node, Node]): a dictionary that will be populated with a mapping
                from nodes in ``g`` to nodes in ``self``. Note that ``val_map`` can be passed
                in with values in it already to override copying of certain values.

        Returns:

            The value in ``self`` that is now equivalent to the output value in ``g``,
            if ``g`` had an ``output`` node. ``None`` otherwise.
        """
        # 遍历源图 g 中的所有节点
        for node in g.nodes:
            # 如果节点已经在映射表 val_map 中，跳过该节点
            if node in val_map:
                continue
            # 如果节点的操作是 'output'
            if node.op == 'output':
                # 映射节点的参数列表中的每个节点，并返回映射后的结果
                rv = map_arg(node.args[0], lambda n: val_map[n])
                # 如果 return_output_node 为 True，则返回映射后的结果和当前节点
                return rv if not return_output_node else (rv, node)
            # 复制节点到当前图 self 中，并更新映射表 val_map
            val_map[node] = self.node_copy(node, lambda n : val_map[n])
        # 如果没有找到输出节点，返回 None
        return None

    # 重写 __deepcopy__ 方法，用于深度复制当前图对象
    # memo: 用于记录已复制节点的字典，防止重复复制和循环引用
    # 返回值: 返回一个新的 Graph 对象，包含了当前图对象的深度复制
    def __deepcopy__(self, memo=None) -> 'Graph':
        """
        Explicitly implement __deepcopy__ to prevent excessive recursion depth
        from the default implementation. This uses graph_copy to copy the nodes
        in an iterative way, rather than recursive. It also populates the
        memoization table to prevent unnecessary copies (e.g. references to
        nodes or other parts of the Graph from a custom GraphModule implementation.
        """
        # 如果 memo 为空，则初始化为一个空字典
        memo = memo if memo else {}
        # 创建一个新的 Graph 对象 g，使用与当前对象相同的追踪器类
        g = Graph(tracer_cls=self._tracer_cls)
        # 使用 graph_copy 方法复制当前图 self 的节点到新图 g 中，并返回输出节点及其值
        output_vals = g.graph_copy(self, val_map=memo, return_output_node=True)
        # 复制当前对象的代码生成器信息到新图对象 g 中
        g._codegen = copy.deepcopy(self._codegen)
        # 断言输出值是一个元组
        assert isinstance(output_vals, tuple)
        # 解包输出值，获取输出值和原始输出节点
        output_val, old_output_node = output_vals
        # 在新图对象 g 中创建新的输出节点，使用复制后的输出值和原始输出节点的类型表达式
        new_output_node = g.output(output_val, type_expr=getattr(old_output_node, 'type', None))
        # 复制原始输出节点的元数据到新的输出节点中
        new_output_node.meta = copy.copy(old_output_node.meta)
        # 返回新创建的 Graph 对象 g
        return g

    # 定义一个装饰器 compatibility，用于标记当前方法或类是向后兼容的
    @compatibility(is_backward_compatible=True)
    def create_node(self, op: str, target: 'Target',
                    args: Optional[Tuple['Argument', ...]] = None,
                    kwargs: Optional[Dict[str, 'Argument']] = None,
                    name: Optional[str] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Create a ``Node`` and add it to the ``Graph`` at the current insert-point.
        Note that the current insert-point can be set via :meth:`Graph.inserting_before`
        and :meth:`Graph.inserting_after`.

        Args:
            op (str): the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
                'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
                described in the ``Graph`` docstring.

            args (Optional[Tuple[Argument, ...]]): is a tuple of arguments to this node.

            kwargs (Optional[Dict[str, Argument]]): the kwargs of this Node

            name (Optional[str]): an optional string name for the ``Node``.
                This will influence the name of the value assigned to in the
                Python generated code.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted node.
        """
        assert op in ('call_function', 'call_method', 'get_attr', 'call_module', 'placeholder', 'output')
        # 确保 op 是有效的操作类型

        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        # 如果 args 或 kwargs 是 None，则分别设为空元组和空字典

        assert isinstance(args, tuple), "args must be a tuple"
        assert isinstance(kwargs, dict), "kwargs must be a dict"
        # 确保 args 是元组，kwargs 是字典

        candidate = name if name is not None else self._target_to_str(target)
        # 如果提供了 name，则使用该名字；否则使用 target 转换后的字符串作为候选名字

        name = self._graph_namespace.create_name(candidate, None)
        # 使用候选名字在命名空间中创建一个唯一的名字

        n = Node(self, name, op, target, args, kwargs, type_expr)
        # 创建一个新的 Node 对象

        if self.owning_module is not None and getattr(self.owning_module, "_create_node_hooks", None) is not None:
            for f in self.owning_module._create_node_hooks:
                f(n)
        # 如果 owning_module 不为 None 并且其包含 _create_node_hooks 属性，则依次调用钩子函数并传入新创建的 Node 对象

        self._graph_namespace.associate_name_with_obj(name, n)
        # 在命名空间中将名字与新创建的 Node 对象关联起来

        self._insert(n)
        # 将新创建的 Node 对象插入到图中的当前插入点

        self._find_nodes_lookup_table.insert(n)
        # 将新创建的 Node 对象插入到查找表中，以便进行快速查找

        self._len += 1
        # 增加图中节点的计数器

        return n
        # 返回新创建并插入的 Node 对象

    @compatibility(is_backward_compatible=False)
    def process_inputs(self, *args):
        """
        Processes args so that they can be passed to the FX graph.
        """
        return self._codegen.process_inputs(*args)
        # 调用 _codegen 对象的 process_inputs 方法处理输入参数，并返回结果

    @compatibility(is_backward_compatible=False)
    def process_outputs(self, out):
        return self._codegen.process_outputs(out)
        # 调用 _codegen 对象的 process_outputs 方法处理输出参数，并返回结果


    @compatibility(is_backward_compatible=True)
    def erase_node(self, to_erase: Node) -> None:
        """
        Erases a ``Node`` from the ``Graph``. Throws an exception if
        there are still users of that node in the ``Graph``.

        Args:
            to_erase (Node): The ``Node`` to erase from the ``Graph``.
        """
        # 检查节点是否还有其他节点在使用，若有则抛出异常
        if len(to_erase.users) > 0:
            raise RuntimeError(f'Tried to erase Node {to_erase} but it still had {len(to_erase.users)} '
                               f'users in the graph: {to_erase.users}!')
        
        # 检查节点是否属于当前图，若不是则抛出异常
        if to_erase.graph != self:
            raise RuntimeError(f"Attempting to remove {to_erase} from wrong graph!")
        
        # 如果节点已经被标记为已删除，则发出警告并直接返回
        if to_erase._erased:
            warnings.warn(f"erase_node({to_erase}) on an already erased node")
            return
        
        # 如果当前图具有所有者模块，并且具有“_erase_node_hooks”属性，则对该节点执行挂钩函数
        if self.owning_module is not None and getattr(self.owning_module, "_erase_node_hooks", None) is not None:
            for f in self.owning_module._erase_node_hooks:
                f(to_erase)
        
        # 从查找节点的查找表中移除该节点
        self._find_nodes_lookup_table.remove(to_erase)
        
        # 从节点链表中移除该节点
        to_erase._remove_from_list()
        
        # 标记节点为已删除，因为迭代器可能仍然引用已删除的节点
        to_erase._erased = True
        
        # 减少当前图中节点的计数器
        self._len -= 1

        # 将该节点的参数节点设为None，以便被引用的节点可以相应地更新其“users”
        new_args = map_arg(to_erase.args, lambda n: None)
        assert isinstance(new_args, tuple)
        to_erase.args = new_args
        
        # 将该节点的关键字参数设为None，以便被引用的节点可以相应地更新其“users”
        new_kwargs = map_arg(to_erase.kwargs, lambda n: None)
        assert isinstance(new_kwargs, dict)
        to_erase.kwargs = new_kwargs

    @compatibility(is_backward_compatible=True)
    def inserting_before(self, n: Optional[Node] = None):
        """
        Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporary set the insert point and
        then restore it when the with statement exits::

            with g.inserting_before(n):
                ... # inserting before node n
            ... # insert point restored to what it was previously
            g.inserting_before(n) #  set the insert point permanently

        Args:
            n (Optional[Node]): The node before which to insert. If None this will insert before
                the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        # 如果节点为None，则在整个图的开头之前插入
        if n is None:
            return self.inserting_after(self._root)
        
        # 断言要插入的节点在当前图中，否则抛出异常
        assert n.graph == self, "Node to insert before is not in graph."
        
        # 返回一个_InsertPoint对象，该对象用于管理插入点的恢复
        return _InsertPoint(self, n.prepend)

    @compatibility(is_backward_compatible=True)
    def inserting_after(self, n: Optional[Node] = None):
        """
        Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporarily set the insert point and
        then restore it when the 'with' statement exits::

            with g.inserting_after(n):
                ... # 在节点 n 后插入
            ... # 插入点恢复到之前的状态
            g.inserting_after(n) # 永久设置插入点

        Args:
            n (Optional[Node]): 要插入位置的前一个节点。如果为 None，则在整个图的开头之后插入。

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        if n is None:
            return self.inserting_before(self._root)
        assert n.graph == self, "Node to insert after is not in graph."
        return _InsertPoint(self, n.append)

    @compatibility(is_backward_compatible=True)
    def placeholder(self, name: str, type_expr: Optional[Any] = None,
                    default_value : Any = inspect.Signature.empty) -> Node:
        """
        Insert a ``placeholder`` node into the Graph. A ``placeholder`` represents
        a function input.

        Args:
            name (str): 输入值的名称。这对应于函数中的位置参数的名称，该函数表示此 ``Graph``。
            type_expr (Optional[Any]): 表示此节点输出的可选类型注释，表示将具有的 Python 类型。
                在某些情况下，这对于正确的代码生成是必要的（例如，当函数随后用于 TorchScript 编译时）。
            default_value (Any): 此函数参数应采用的默认值。注意：为了允许 `None` 作为默认值，
                应将 `inspect.Signature.empty` 作为此参数传递，以指定参数没有默认值。

        .. note::
            对于此方法，与 ``Graph.create_node`` 相同的插入点和类型表达式规则适用。
        """
        args = () if default_value is inspect.Signature.empty else (default_value,)
        return self.create_node('placeholder', name, args=args, type_expr=type_expr)

    @compatibility(is_backward_compatible=True)


这些注释提供了每个方法的详细解释，包括方法的作用、参数的含义以及一些与使用相关的注意事项。
    # 定义一个方法 `get_attr`，用于向图中插入一个 `get_attr` 节点。`get_attr` 节点表示从 `Module` 层级中获取属性的操作。
    def get_attr(self, qualified_name: str, type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``get_attr`` node into the Graph. A ``get_attr`` ``Node`` represents the
        fetch of an attribute from the ``Module`` hierarchy.

        Args:

            qualified_name (str): the fully-qualified name of the attribute to be retrieved.
                For example, if the traced Module has a submodule named ``foo``, which has a
                submodule named ``bar``, which has an attribute named ``baz``, the qualified
                name ``foo.bar.baz`` should be passed as ``qualified_name``.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.


        Returns:

            The newly-created and inserted ``get_attr`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        # 定义一个内部函数 `_get_attr_reference_exists`，用于检查指定的 `qualified_name` 是否在 `mod` 模块中存在对应的属性引用。
        def _get_attr_reference_exists(mod: torch.nn.Module, qualified_name: str) -> bool:
            module_path, _, name = qualified_name.rpartition(".")
            
            # 尝试获取模块路径下的子模块
            try:
                submod: torch.nn.Module = mod.get_submodule(module_path)
            except AttributeError:
                # 捕获属性错误异常，提示获取模块路径失败，并返回 False
                warnings.warn(f"Failed to fetch module {module_path}!")
                return False
            
            # 检查子模块是否具有指定的属性名
            if not hasattr(submod, name):
                return False
            
            # 获取属性名对应的实际对象
            res = getattr(submod, name)
            
            # 检查对象是否为 `torch.nn.Module`、`torch.nn.Parameter` 或者存在于子模块的缓冲区中
            if (not isinstance(res, torch.nn.Module)
                    and not isinstance(res, torch.nn.Parameter)
                    and name not in submod._buffers):
                return False
            
            return True
        
        # 如果存在 `owning_module` 并且 `_get_attr_reference_exists` 返回 False，则发出警告提示信息
        if (self.owning_module and
                not _get_attr_reference_exists(self.owning_module, qualified_name)):
            warnings.warn("Attempted to insert a get_attr Node with no "
                          "underlying reference in the owning "
                          "GraphModule! Call "
                          "GraphModule.add_submodule to add the "
                          "necessary submodule, "
                          "GraphModule.add_parameter to add the "
                          "necessary Parameter, or "
                          "nn.Module.register_buffer to add the "
                          "necessary buffer", stacklevel=2)
        
        # 调用当前对象的 `create_node` 方法，创建并返回一个类型为 `get_attr` 的新节点，表示成功插入 `get_attr` 节点
        return self.create_node('get_attr', qualified_name, type_expr=type_expr)

    # 声明一个 `compatibility` 装饰器，标记此方法为向后兼容
    @compatibility(is_backward_compatible=True)
    # 将一个 `call_module` 节点插入到 `Graph` 中。`call_module` 节点表示对 `Module` 层级中某个 `Module` 的 `forward()` 函数的调用。
    def call_module(self,
                    module_name: str,
                    args: Optional[Tuple['Argument', ...]] = None,
                    kwargs: Optional[Dict[str, 'Argument']] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``call_module`` ``Node`` into the ``Graph``. A ``call_module`` node
        represents a call to the forward() function of a ``Module`` in the ``Module``
        hierarchy.

        Args:

            module_name (str): The qualified name of the ``Module`` in the ``Module``
                hierarchy to be called. For example, if the traced ``Module`` has a
                submodule named ``foo``, which has a submodule named ``bar``, the
                qualified name ``foo.bar`` should be passed as ``module_name`` to
                call that module.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this should *not* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted ``call_module`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        # 如果存在所属的 `owning_module` 并且指定的 `module_name` 子模块不存在，则发出警告
        if (self.owning_module and
                self.owning_module.get_submodule(module_name) is None):
            warnings.warn("Attempted to insert a call_module Node with "
                          "no underlying reference in the owning "
                          "GraphModule! Call "
                          "GraphModule.add_submodule to add the "
                          "necessary submodule")
        # 调用 Graph 对象的 `create_node` 方法，创建并返回一个 `call_module` 节点
        return self.create_node('call_module', module_name, args, kwargs, type_expr=type_expr)

    # 声明这个方法是向后兼容的，与之前版本兼容
    @compatibility(is_backward_compatible=True)
    # 定义一个方法，用于在图中插入一个“call_method”节点，表示在args的第一个元素上调用指定的方法。
    def call_method(self,
                    method_name: str,
                    args: Optional[Tuple['Argument', ...]] = None,
                    kwargs: Optional[Dict[str, 'Argument']] = None,
                    type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``call_method`` ``Node`` into the ``Graph``. A ``call_method`` node
        represents a call to a given method on the 0th element of ``args``.

        Args:

            method_name (str): The name of the method to apply to the self argument.
                For example, if args[0] is a ``Node`` representing a ``Tensor``,
                then to call ``relu()`` on that ``Tensor``, pass ``relu`` to ``method_name``.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this *should* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly created and inserted ``call_method`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node('call_method', method_name, args, kwargs, type_expr=type_expr)

    # 定义一个方法，用于在图中插入一个“call_function”节点，表示调用指定的Python可调用对象（如函数）。
    @compatibility(is_backward_compatible=True)
    def call_function(self,
                      the_function: Callable[..., Any],
                      args: Optional[Tuple['Argument', ...]] = None,
                      kwargs: Optional[Dict[str, 'Argument']] = None,
                      type_expr: Optional[Any] = None) -> Node:
        """
        Insert a ``call_function`` ``Node`` into the ``Graph``. A ``call_function`` node
        represents a call to a Python callable, specified by ``the_function``.

        Args:

            the_function (Callable[..., Any]): The function to be called. Can be any PyTorch
                operator, Python function, or member of the ``builtins`` or ``operator``
                namespaces.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called function.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called function

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly created and inserted ``call_function`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node('call_function', the_function, args, kwargs, type_expr=type_expr)
    # 使用装饰器标记方法兼容性为向后兼容
    @compatibility(is_backward_compatible=True)
    # 定义方法：将一个节点从一个图复制到另一个图中
    def node_copy(self, node: Node, arg_transform: Callable[[Node], 'Argument'] = lambda x: x) -> Node:
        """
        Copy a node from one graph into another. ``arg_transform`` needs to transform arguments from
        the graph of node to the graph of self. Example::

            # Copying all the nodes in `g` into `new_graph`
            g : torch.fx.Graph = ...
            new_graph = torch.fx.graph()
            value_remap = {}
            for node in g.nodes:
                value_remap[node] = new_graph.node_copy(node, lambda n : value_remap[n])

        Args:

            node (Node): The node to copy into ``self``.

            arg_transform (Callable[[Node], Argument]): A function that transforms
                ``Node`` arguments in node's ``args`` and ``kwargs`` into the
                equivalent argument in ``self``. In the simplest case, this should
                retrieve a value out of a table mapping Nodes in the original
                graph to ``self``.
        """
        # 转换节点的参数并返回
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        # 断言参数的类型
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        # 创建新节点并设置元数据
        result_node = self.create_node(node.op, node.target, args, kwargs, node.name, node.type)
        result_node.meta = copy.copy(node.meta)
        return result_node

    # 使用装饰器标记方法兼容性为向后兼容
    @compatibility(is_backward_compatible=True)
    # 定义方法：向图中插入一个输出节点，表示 Python 代码中的 return 语句
    def output(self, result: 'Argument', type_expr: Optional[Any] = None):
        """
        Insert an ``output`` ``Node`` into the ``Graph``. An ``output`` node represents
        a ``return`` statement in Python code. ``result`` is the value that should
        be returned.

        Args:

            result (Argument): The value to be returned.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        .. note::

            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        # 调用 Graph.create_node 方法创建一个输出节点并返回
        return self.create_node(op='output', target='output', args=(result,), type_expr=type_expr)

    # 定义方法：将目标转换为字符串表示形式
    def _target_to_str(self, target : Target) -> str:
        # 如果目标是可调用的，则获取其名称
        if callable(target):
            op = target.__name__
        else:
            assert isinstance(target, str)
            op = target
            # 如果目标是魔术方法，则去除双下划线并转换为蛇形命名
            if _is_magic(op):
                op = op[2:-2]
        # 将操作符转换为蛇形命名的字符串并返回
        op = _snake_case(op)
        return op

    # 使用装饰器标记方法兼容性为向后兼容
    @compatibility(is_backward_compatible=True)
    # 定义方法：生成 Python 代码表示
    def python_code(
        self, root_module: str, *,
        verbose: bool = False, include_stride: bool = False, include_device: bool = False, colored: bool = False
    ):
        """
        Generate Python code representation.

        Args:

            root_module (str): Root module name.

            verbose (bool): Whether to include verbose output.

            include_stride (bool): Whether to include stride information.

            include_device (bool): Whether to include device information.

            colored (bool): Whether to generate colored output.
        """
        # 实现生成 Python 代码的功能，具体细节未展示在此
    ) -> PythonCode:
        """
        Turn this ``Graph`` into valid Python code.

        Args:

            root_module (str): The name of the root module on which to look-up
                qualified name targets. This is usually 'self'.

        Returns:

            A PythonCode object, consisting of two fields:
                src: the Python source code representing the object
                globals: a dictionary of global names in `src` -> the objects that they reference.
        """
        # NOTE: [Graph Namespaces]
        #
        # There are two types of symbols in generated Python source code:
        # locals and globals.
        #   Locals are locally defined by the output of a node in the Graph.
        #   Globals are references to external objects, like functions or types.
        #
        # When generating Python code, we need to make sure to name things
        # appropriately. In particular:
        # - All names should be unique, to avoid weird shadowing bugs.
        # - These names need to be consistent, e.g. a object should always be
        #   referenced by the same name.
        #
        # To do this, we create a new namespace just for this source. All names
        # that get printed must come from this namespace.
        namespace = _Namespace()

        # Override Node's repr to generate a valid name within our namespace.
        # Since repr() is designed to produce a valid Python expression, it
        # makes sense to re-use it. This way, it's easy to print something like
        # Tuple[Node, Node] by simply calling repr() on it. Node's __repr__ is
        # implemented cooperatively to allow this.
        def node_repr(n: Node):
            return namespace.create_name(n.name, n)

        @contextmanager
        def override_node_repr(graph: Graph):
            orig_repr_fns = {}
            for node in graph.nodes:
                orig_repr_fns[node] = node._repr_fn
                node._repr_fn = node_repr
            try:
                yield None
            finally:
                # restore the original repr functions
                for node in graph.nodes:
                    node._repr_fn = orig_repr_fns[node]

        # Temporarily override the node representation function for all nodes in the graph
        with override_node_repr(self):
            # Generate Python code for the graph using specified parameters and namespace
            return self._python_code(
                root_module, namespace,
                verbose=verbose, include_stride=include_stride, include_device=include_device, colored=colored
            )

    def _python_code(
        self, root_module: str, namespace: _Namespace, *,
        verbose: bool = False, include_stride: bool = False, include_device: bool = False, colored: bool = False,
    ) -> PythonCode:
        return self._codegen._gen_python_code(
            self.nodes, root_module, namespace,
            verbose=verbose, include_stride=include_stride, include_device=include_device, colored=colored
        )

# 返回生成的 Python 代码对象，基于给定的节点、根模块和命名空间。可以选择性地包含详细信息、步长、设备信息以及是否使用彩色输出。


    def __str__(self) -> str:
        """
        Return a human-readable (not machine-readable) string representation
        of this Graph
        """
        placeholder_names : List[str] = []
        # This is a one-element array just so ``format_node`` can modify the closed
        # over value
        maybe_return_typename : List[str] = ['']

        node_strs = [node.format_node(placeholder_names) for node in self.nodes]
        param_str = ', '.join(placeholder_names)
        s = f'graph({param_str}){maybe_return_typename[0]}:'
        for node_str in node_strs:
            if node_str:
                s += '\n    ' + node_str
        return s

# 返回描述此图形的人类可读（非机器可读）字符串表示。其中包括占位符名称列表、节点的格式化字符串以及可能的返回类型名称。


    @compatibility(is_backward_compatible=True)
    def print_tabular(self):
        """
        Prints the intermediate representation of the graph in tabular
        format. Note that this API requires the ``tabulate`` module to be
        installed.
        """
        try:
            from tabulate import tabulate
        except ImportError:
            print("`print_tabular` relies on the library `tabulate`, "
                  "which could not be found on this machine. Run `pip "
                  "install tabulate` to install the library.")
            raise

        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs]
                      for n in self.nodes]
        print(tabulate(node_specs,
              headers=['opcode', 'name', 'target', 'args', 'kwargs']))

# 打印以表格形式显示的图形的中间表示。需要安装 `tabulate` 库。如果找不到该库，会提示用户安装。


    @compatibility(is_backward_compatible=True)
    @compatibility(is_backward_compatible=True)

# 标记此方法与旧版兼容，允许在后向兼容模式下使用。
    def eliminate_dead_code(self):
        """
        Remove all dead code from the graph, based on each node's number of
        users, and whether the nodes have any side effects. The graph must be
        topologically sorted before calling.

        Returns:
          bool: Whether the graph was changed as a result of the pass.

        Example:

        Before dead code is eliminated, `a` from `a = x + 1` below has no users
        and thus can be eliminated from the graph without having an effect.

        .. code-block:: python

            def forward(self, x):
                a = x + 1
                return x + self.attr_1

        After dead code is eliminated, `a = x + 1` has been removed, and the rest
        of `forward` remains.

        .. code-block:: python

            def forward(self, x):
                return x + self.attr_1

        .. warning::

            Dead code elimination has some heuristics to avoid removing
            side-effectful nodes (see Node.is_impure) but in general coverage
            is very bad, so you should assume that this method is not sound
            to call unless you know that your FX graph consists entirely
            of functional operations.
        """
        # Lint the graph first to make sure its topologically sorted, otherwise
        # DCE below will not behave as expected.
        self.lint()

        # Reverse iterate so that when we remove a node, any nodes used as an
        # input to that node have an updated user count that no longer reflects
        # the removed node.
        changed = False
        for node in reversed(self.nodes):
            # Check if the node is not impure and has no users
            if not node.is_impure() and len(node.users) == 0:
                # Remove the node from the graph
                self.erase_node(node)
                # Set flag indicating a change was made
                changed = True

        # Return whether the graph was changed
        return changed
# 可以通过反射调用的特殊方法，包括加法
reflectable_magic_methods = {
    'add': '{} + {}',            # 加法方法
    'sub': '{} - {}',            # 减法方法
    'mul': '{} * {}',            # 乘法方法
    'floordiv': '{} // {}',      # 地板除法方法
    'truediv': '{} / {}',        # 真除法方法
    'div': '{} / {}',            # 除法方法
    'mod': '{} % {}',            # 取模方法
    'pow': '{} ** {}',           # 幂运算方法
    'lshift': '{} << {}',        # 左移方法
    'rshift': '{} >> {}',        # 右移方法
    'and_': '{} & {}',           # 位与方法
    'or_': '{} | {}',            # 位或方法
    'xor': '{} ^ {}',            # 位异或方法
    'getitem': '{}[{}]',         # 索引访问方法
    'matmul': '{} @ {}',         # 矩阵乘法方法
}

# 包含可反射调用的特殊方法和常规比较方法的字典
magic_methods = dict({
    'eq': '{} == {}',            # 等于比较方法
    'ne': '{} != {}',            # 不等于比较方法
    'lt': '{} < {}',             # 小于比较方法
    'gt': '{} > {}',             # 大于比较方法
    'le': '{} <= {}',            # 小于等于比较方法
    'ge': '{} >= {}',            # 大于等于比较方法
    'pos': '+{}',                # 正数运算方法
    'neg': '-{}',                # 负数运算方法
    'invert': '~{}'},            # 按位取反方法
    **reflectable_magic_methods)

# 包含就地操作方法的字典，如就地加法、位与等
inplace_methods = {
    'iadd': '{} += {}',          # 就地加法方法
    'iand': '{} &= {}',          # 就地位与方法
    'ifloordiv': '{} //= {}',    # 就地地板除法方法
    'ilshift': '{} <<= {}',      # 就地左移方法
    'imod': '{} %= {}',          # 就地取模方法
    'imul': '{} *= {}',          # 就地乘法方法
    'imatmul': '{} @= {}',       # 就地矩阵乘法方法
    'ior': '{} |= {}',           # 就地位或方法
    'ipow': '{} **= {}',         # 就地幂运算方法
    'irshift': '{} >>= {}',      # 就地右移方法
    'isub': '{} -= {}',          # 就地减法方法
    'itruediv': '{} /= {}',      # 就地真除法方法
    'ixor': '{} ^= {}',          # 就地位异或方法
    'setitem': '{}[{}] = {}',    # 就地设置索引值方法
}
```