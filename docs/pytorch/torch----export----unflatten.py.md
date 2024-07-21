# `.\pytorch\torch\export\unflatten.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import abc  # 抽象基类模块
import copy  # 复制模块
import operator  # 操作符模块
from collections import defaultdict  # 默认字典集合模块
from copy import deepcopy  # 深拷贝模块
from enum import Enum  # 枚举类模块
from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union  # 类型提示模块

import torch  # PyTorch 模块
import torch.fx._pytree as fx_pytree  # Torch FX 下的 Pytree 模块
import torch.utils._pytree as pytree  # Torch Utils 下的 Pytree 模块
from torch._library.fake_class_registry import FakeScriptObject  # 假脚本对象模块
from torch.export._tree_utils import reorder_kwargs  # 重排序关键字参数模块
from torch.export.exported_program import (  # 导出程序模块
    ConstantArgument,  # 常数参数类
    ExportedProgram,  # 导出程序类
    InputKind,  # 输入类型枚举
    ModuleCallSignature,  # 模块调用签名类
    SymIntArgument,  # 符号整数参数类
    TensorArgument,  # 张量参数类
)
from torch.fx._symbolic_trace import is_fx_tracing  # 是否进行 FX 追踪模块
from torch.fx.experimental.proxy_tensor import py_sym_types  # 代理张量模块
from torch.utils._pytree import GetAttrKey, SequenceKey  # Pytree 下的属性键和序列键模块

from ._remove_effect_tokens_pass import _remove_effect_tokens  # 移除效果令牌的模块

__all__ = ["InterpreterModule", "UnflattenedModule", "unflatten", "FlatArgsAdapter"]

# 定义属性种类的枚举
class _AttrKind(Enum):
    PARAMETER = "parameter"  # 参数类型
    BUFFER = "buffer"  # 缓冲区类型
    CONSTANT = "constant"  # 常数类型

# 将 'from_obj' 的属性赋值给 'target' 的完全限定名称，安装空模块以确保子路径存在
def _assign_attr(
    from_obj: Union[torch.Tensor, torch.ScriptObject],  # 源对象可以是张量或脚本对象
    to_module: torch.nn.Module,  # 目标模块
    target: str,  # 目标属性的完全限定名称
    attr_kind: _AttrKind,  # 属性种类枚举
    persistent: bool = True,  # 是否持久化属性
):
    *prefix, field = target.split(".")  # 拆分目标名称为前缀和字段名
    for item in prefix:
        t = getattr(to_module, item, None)  # 获取目标模块的属性

        if t is None:
            t = torch.nn.Module()  # 如果属性不存在则创建新的模块
            setattr(to_module, item, t)  # 设置属性为新创建的模块
        to_module = t  # 更新目标模块为当前模块的属性

    if attr_kind == _AttrKind.PARAMETER:  # 如果是参数类型
        assert isinstance(from_obj, torch.nn.Parameter)
        to_module.register_parameter(field, from_obj)  # 注册参数
    elif attr_kind == _AttrKind.BUFFER:  # 如果是缓冲区类型
        assert isinstance(from_obj, torch.Tensor)
        to_module.register_buffer(field, from_obj, persistent=persistent)  # 注册缓冲区
    elif attr_kind == _AttrKind.CONSTANT:  # 如果是常数类型
        assert not isinstance(from_obj, FakeScriptObject), "FakeScriptObject should only exist during tracing."
        assert isinstance(
            from_obj,
            (
                torch.Tensor,
                torch.ScriptObject,
            ),
        )
        setattr(to_module, field, from_obj)  # 设置常数属性

# InterpreterModule 类，用于使用 torch.fx.Interpreter 执行，提供更好的堆栈跟踪信息和更容易调试的执行方式
class InterpreterModule(torch.nn.Module):
    """A module that uses torch.fx.Interpreter to execute instead of the usual
    codegen that GraphModule uses. This provides better stack trace information
    and makes it easier to debug execution.
    """

    def __init__(
        self,
        graph: torch.fx.Graph,  # Torch FX 图形对象
    ):
        super().__init__()
        self.graph = graph  # 设置模块的 FX 图形对象
        self.graph.owning_module = self  # 设置图形对象的拥有模块
    def forward(self, *args, **kwargs):
        # 确保图模块已经初始化，否则抛出异常
        assert self.graph_module is not None, "Didn't finalize this InterpreterModule"
        
        if torch.compiler.is_dynamo_compiling():
            # 如果正在使用 Dynamo 编译，则无法通过 torch.fx.Interpreter 跟踪，因此回退到 GraphModule 的代码生成。
            return self.graph_module(*args, **kwargs)
        else:
            if kwargs:
                # 处理 **kwargs。FX 只原生支持位置参数（通过占位符）。因此，为了传递 kwargs，必须将占位符的名称与 kwargs 字典中的键对应起来。
                arg_list = list(args)
                kwarg_names = self.arg_names[len(arg_list) :]
                for kwarg_name in kwarg_names:
                    if kwarg_name in kwargs:
                        arg_list.append(kwargs[kwarg_name])

                # 断言传入的 kwargs 必须完全匹配 GraphModule 指定的位置参数。这应该由展开过程来保证。
                assert len(kwarg_names) == len(kwargs)
                assert len(arg_list) == len(self.arg_names)
                args = tuple(arg_list)

            # 运行 torch.fx.Interpreter，禁用 IO 处理
            return torch.fx.Interpreter(self, graph=self.graph).run(
                *args, enable_io_processing=False
            )

    def finalize(self):
        # 需要“finalize”因为 GraphModule 基于图中观察到的 get_attrs 填充其自己的 state_dict。因此，我们需要完全构建图并调用 _sink_params，然后生成这个 GraphModule。

        # 直接在字典上设置 'graph_module'，以避免其被注册为子模块。
        self.__dict__["graph_module"] = torch.fx.GraphModule(self, self.graph)
        # 对图进行 lint 检查
        self.graph.lint()

        # 缓存参数名称以用于 kwargs 处理（参见 forward()）
        self.arg_names = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                self.arg_names.append(node.target)
class FlatArgsAdapter(abc.ABC):
    """
    Adapts input arguments with ``input_spec`` to align ``target_spec``.
    """

    @abc.abstractmethod
    def adapt(
        self,
        target_spec: pytree.TreeSpec,
        input_spec: pytree.TreeSpec,
        input_args: List[Any],
    ) -> List[Any]:
        """NOTE: This adapter may mutate given ``input_args_with_path``."""
        ...


class UnflattenedModule(torch.nn.Module):
    def __init__(
        self,
        export_module: ExportedProgram,
        flat_args_adapter: Optional[FlatArgsAdapter] = None,
    ):
        """Initialize an UnflattenedModule instance.

        Args:
            export_module (ExportedProgram): The exported program to initialize from.
            flat_args_adapter (Optional[FlatArgsAdapter]): Adapter for flat arguments, defaults to None.
        """
        super().__init__()
        self.export_module = export_module
        self.flat_args_adapter = flat_args_adapter

    def _print_graph(self):
        """Prints the graph of each module if available."""
        for fqn, mod in self.named_modules():
            print(fqn + ":")
            if hasattr(mod, "graph") and isinstance(mod.graph, torch.fx.Graph):
                print(mod.graph)


def unflatten(
    module: ExportedProgram, flat_args_adapter: Optional[FlatArgsAdapter] = None
) -> UnflattenedModule:
    """Unflatten an ExportedProgram, producing a module with the same module
    hierarchy as the original eager module.

    Args:
        module (ExportedProgram): The ExportedProgram to unflatten.
        flat_args_adapter (Optional[FlatArgsAdapter]): Adapter for flat arguments, defaults to None.

    Returns:
        UnflattenedModule: An instance of UnflattenedModule with the same module hierarchy as the original eager module.
    """
    module = _remove_effect_tokens(module)
    return UnflattenedModule(module, flat_args_adapter)


def _inplace_buffer_mutations(graph: torch.fx.Graph, graph_signature) -> None:
    """Transform buffer mutations from their functionalized form into a copy_
    node in the graph.

    Functionalization represents buffer mutation by passing the buffer as an input and output. So for example, the eager code:
        def forward(self, x):
            self.buffer += x
            return x * x

    Will become a graph that looks like:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            mul = aten.mul(x, x)
            return (mutated_buffer, mul)

    We want to inplace this into something that looks like the original eager code:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            buffer.copy_(mutated_buffer)
            mul = aten.mul(x, x)
            return (mul,)
    """
    output_node = next(iter(reversed(graph.nodes)))
    # 断言输出节点是"output"，且其参数列表长度为1
    assert output_node.op == "output" and len(output_node.args) == 1
    # 获取返回的参数列表
    return_args = output_node.args[0]
    
    # 获取图签名中要变异的缓冲区节点映射
    mutation_node_to_buffer = graph_signature.buffers_to_mutate
    # 获取要变异的节点列表
    mutations = return_args[: len(mutation_node_to_buffer)]
    # 将输入到缓冲区的映射关系反转，以便根据缓冲区名获取输入名
    buffers_to_inputs = {v: k for k, v in graph_signature.inputs_to_buffers.items()}
    # 创建输入名称到节点的映射，限定节点操作为"placeholder"
    input_name_to_node = {
        node.name: node for node in graph.nodes if node.op == "placeholder"
    }
    
    # 遍历每一个变异节点
    for mutation in mutations:
        # 获取与变异节点关联的缓冲区名
        buffer_name = mutation_node_to_buffer[mutation.name]
        # 根据缓冲区名找到对应的输入名
        input_name = buffers_to_inputs[buffer_name]
        # 根据输入名找到对应的输入节点
        input_node = input_name_to_node[input_name]
    
        # 在变异节点之后插入新节点
        with graph.inserting_after(mutation):
            # 创建一个调用函数节点，使用 torch.ops.aten.copy_ 函数复制输入节点和变异节点
            new_node = graph.create_node(
                "call_function", torch.ops.aten.copy_, (input_node, mutation)
            )
            # 复制变异节点的元数据到新节点
            for k, v in mutation.meta.items():
                new_node.meta[k] = v
        # 替换所有先前功能变异的用途为新的复制输出节点
        mutation.replace_all_uses_with(new_node, lambda x: x is not new_node)
    
    # 从图输出中移除已变异的缓冲区，因为不再需要传递它们。不需要处理输入，这将由 _sink_params 处理。
    user_outputs = tuple(
        return_args[len(mutation_node_to_buffer):],
    )
    # 更新输出节点的参数为用户输出
    output_node.args = ((user_outputs),)
# 检查 `candidate` 是否为 `target` 的前缀
def _is_prefix(candidate, target):
    return len(candidate) < len(target) and target[: len(candidate)] == candidate


# 计算子模块 `child_fqn` 相对于父模块 `parent_fqn` 的访问路径
def _compute_accessor(parent_fqn: str, child_fqn: str) -> str:
    if parent_fqn == "":
        # 处理根模块的情况
        return child_fqn

    # 拆分父模块和子模块的完全限定名
    parent_split = parent_fqn.split(".")
    child_split = child_fqn.split(".")

    # 断言子模块是父模块的后代模块
    assert (
        child_split[: len(parent_split)] == parent_split
    ), f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'"

    # 返回子模块相对于父模块的访问路径
    return ".".join(child_split[len(parent_split):])


# 验证两个神经网络模块 `x` 和 `y` 的计算图是否等价
def _verify_graph_equivalence(x: torch.nn.Module, y: torch.nn.Module):
    # 将计算图转换为字符串形式以进行比较
    def graph_dump(graph: torch.fx.Graph) -> str:
        ret = []
        nodes_idx: Dict[int, int] = {}

        # 辅助函数，用于转换节点的参数
        def arg_dump(arg) -> str:
            if isinstance(arg, torch.fx.Node):
                return "%" + str(nodes_idx[id(arg)])
            return str(arg)

        # 遍历计算图的每个节点，生成节点的字符串表示
        for i, node in enumerate(graph.nodes):
            args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
            args_dump += [
                f"{key}={value}"
                for key, value in pytree.tree_map(arg_dump, node.kwargs).items()
            ]
            target = node.target if node.op == "call_function" else ""
            ret.append(f"{i}: {node.op}[{target}]({', '.join(args_dump)})")
            nodes_idx[id(node)] = i
        return "\n".join(ret)

    # 断言两个模块的计算图是等价的
    assert graph_dump(x.graph) == graph_dump(y.graph)


# 向神经网络模块 `gm` 添加一个新的特定规范 `spec`
def _add_spec(gm: torch.nn.Module, spec) -> str:
    i = 0
    while hasattr(gm, f"_spec_{i}"):
        i += 1
    name = f"_spec_{i}"
    # 设置模块的属性来存储规范
    setattr(gm, name, spec)
    return name


# 生成一个将节点 `node` 展平的操作，并应用特定的规范 `spec`
def _generate_flatten(gm: torch.nn.Module, node, spec) -> torch.fx.Node:
    # 添加特定规范 `spec` 到模块 `gm` 中
    name = _add_spec(gm, spec)
    # 获取模块中的特定规范节点
    spec_node = gm.graph.get_attr(name)
    # 调用函数 `fx_pytree.tree_flatten_spec` 来执行节点展平操作
    return gm.graph.call_function(fx_pytree.tree_flatten_spec, (node, spec_node))


# 生成一个将节点列表 `nodes` 还原的操作，并应用特定的规范 `spec`
def _generate_unflatten(gm: torch.nn.Module, nodes, spec) -> torch.fx.Node:
    # 添加特定规范 `spec` 到模块 `gm` 中
    name = _add_spec(gm, spec)
    # 获取模块中的特定规范节点
    spec_node = gm.graph.get_attr(name)
    # 调用函数 `pytree.tree_unflatten` 来执行节点列表还原操作
    return gm.graph.call_function(pytree.tree_unflatten, (nodes, spec_node))


# 获取神经网络模块 `mod` 中指定名称 `target` 的子模块
def _get_submodule(mod: torch.nn.Module, target: str):
    # 拆分 `target` 字符串以获取模块的层级结构
    *prefix, field = target.split(".")

    # 遍历每个模块层级，逐级获取子模块
    for item in prefix:
        submod = getattr(mod, item, None)

        # 如果子模块不存在，返回 `None`
        if submod is None:
            return None

        # 如果子模块不是 `torch.nn.Module` 类型，返回 `None`
        if not isinstance(submod, torch.nn.Module):
            return None

        # 更新当前模块为当前子模块
        mod = submod

    # 返回指定名称的子模块
    return getattr(mod, field, None)


# 向神经网络模块 `mod` 中添加指定名称 `target` 的子模块 `module_to_add`
def _add_submodule(mod: torch.nn.Module, target: str, module_to_add: torch.nn.Module):
    # 拆分 `target` 字符串以获取模块的层级结构
    *prefix, field = target.split(".")

    # 遍历每个模块层级，逐级获取或添加子模块
    for item in prefix:
        submod = getattr(mod, item, None)

        # 如果子模块不存在，创建一个新的 `torch.nn.Module` 并添加到当前模块
        if submod is None:
            submod = torch.nn.Module()
            setattr(mod, item, submod)

        # 如果子模块不是 `torch.nn.Module` 类型，返回 `False`
        if not isinstance(submod, torch.nn.Module):
            return False

        # 更新当前模块为当前子模块
        mod = submod

    # 将 `module_to_add` 添加为指定名称的子模块
    mod.add_module(field, module_to_add)


class _ModuleFrame:
    # 初始化方法，接受多个参数来初始化对象的各个属性
    def __init__(
        self,
        flat_graph,              # 平面图形参数，表示平展的图形结构
        nodes,                   # 节点参数，表示节点集合
        seen_nodes,              # 已见节点参数，表示已经查看过的节点集合
        seen_modules,            # 已见模块参数，表示已经查看过的模块集合
        parent,                  # 父对象参数，表示父级对象
        module_stack,            # 模块堆栈参数，表示模块的堆栈结构
        module_id,               # 模块ID参数，表示模块的唯一标识符
        module_call_graph: Dict[str, ModuleCallSignature],  # 模块调用图参数，字符串到模块调用签名的字典
        module: Optional[torch.nn.Module] = None,  # 模块参数，表示一个可选的PyTorch模块，默认为None
    # 添加占位符方法，用于向对象添加占位符节点
    def add_placeholder(self, x):
        # 断言确保当前对象的全限定名称不为空，否则抛出异常
        assert self.fqn != "", f"Cannot add placeholder {x} to root module"
        # 断言确保x的图形与当前对象的平面图形相同，否则抛出异常
        assert x.graph is self.flat_graph
        # 在当前图形的插入位置之前插入一个新的占位符节点
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        # 复制所有元数据字段，即使某些字段对于占位符节点可能不相关
        placeholder_node.meta = copy.copy(x.meta)
        # 将原节点x映射到新创建的占位符节点
        self.node_to_placeholder[x] = placeholder_node

    # 复制符号调用函数方法，用于复制具有符号大小节点的调用函数节点
    def copy_sym_call_function(self, x):
        # 确保x的meta数据中的val字段是py_sym_types类型，否则会断言失败
        assert isinstance(x.meta["val"], py_sym_types)
        # 重新映射输入参数x中的所有参数和关键字参数
        args = tuple(
            self.remap_input(_x) if isinstance(_x, torch.fx.Node) else _x
            for _x in x.args
        )
        kwargs = {
            k: self.remap_input(_x) if isinstance(_x, torch.fx.Node) else _x
            for k, _x in x.kwargs.items()
        }
        # 在当前图形中创建一个调用函数节点，传入目标函数、参数和关键字参数
        node = self.graph.call_function(x.target, args, kwargs)
        # 复制节点x的所有元数据字段到新创建的节点
        node.meta = copy.copy(x.meta)
        # 将原节点x映射到新创建的调用函数节点
        self.node_map[x] = node
        # 返回新创建的节点
        return node
    # 重新映射输入节点到当前图形的等效节点
    def remap_input(self, x):
        # 断言输入节点的图形必须是当前平面图形
        assert x.graph is self.flat_graph
        # 如果节点已经在节点映射中，则返回映射后的节点
        if x in self.node_map:
            return self.node_map[x]
        # 打印调试信息，显示正在重新映射输入节点
        self.print(f"remap_input({x})")
        # 如果节点是已知的占位符节点，则返回其对应的占位符
        if x in self.node_to_placeholder:
            return self.node_to_placeholder[x]
        # 如果节点是占位符操作，或者模块调用图不保留模块调用签名，则创建新的占位符节点
        elif (
            x.op == "placeholder"
            or self.module_call_graph.get(self.fqn) is None
            # 允许在不保留模块调用签名的情况下创建占位符
        ):
            self.add_placeholder(x)
            # 如果存在父级调用模块，则在父级模块调用前插入占位符节点
            if self.parent_call_module is not None:
                # 重要：在插入占位符节点时，将输出*预置*以匹配插入方式
                with self.parent.graph.inserting_before(self.parent_call_module):
                    self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
            return self.node_to_placeholder[x]
        # 如果节点是函数调用操作，则复制对应的符号调用函数节点
        elif x.op == "call_function":
            # 导出符号调用函数节点的重复节点，如果需要保留模块调用签名，则重新复制它们
            self.copy_sym_call_function(x)
            return self.node_map[x]
        else:
            # 如果无法处理节点操作类型，则引发运行时错误
            raise RuntimeError(
                f"Could not run remap_input() on op type: {x.op} for node {x}"
            )
        return self.node_to_placeholder[x]  # 返回节点对应的占位符

```  
    # 复制给定节点，并在节点映射中记录复制后的节点
    def copy_node(self, node):
        # 打印调试信息，显示正在复制节点
        self.print("copying", node.format_node())
        # 使用图形对象的节点复制方法，根据重新映射输入函数创建节点的副本，并记录到节点映射中
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        # 记录已看到的节点，使用节点名称作为键
        self.seen_nodes[node.name] = node

```  
    # 从平面图形的节点列表中按顺序运行所有节点
    def run_outer(self):
        i = 0
        # 遍历平面图形中的所有节点，并显示节点的神经网络模块堆栈和格式化节点信息
        for node in self.flat_graph.nodes:
            self.print(i, node.meta.get("nn_module_stack"), node.format_node())
            i += 1

        # 复制所有图形输入节点
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == "placeholder":
            # 复制当前节点，并移动到下一个节点
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]

        # 从指定节点索引开始运行图形
        self.run_from(node_idx)

        # 复制图形输出节点
        for node in self.flat_graph.nodes:
            if node.op == "output":
                # 复制输出节点，并在节点映射中记录复制后的节点
                self.copy_node(node)

```  
    # 如果设置了详细输出模式，则打印信息
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
def _outline_submodules(orig_graph: torch.fx.Graph, root_module: UnflattenedModule):
    # 用于跟踪已经访问过的节点，以节点名称为键，节点对象为值的字典
    seen_nodes: Dict[str, torch.fx.Node] = {}

    # 用于跟踪已经访问过的模块，以模块对象的内存地址为键，模块对象为值的字典
    seen_modules: Dict[int, torch.nn.Module] = {}

    # 初始化 _ModuleFrame 对象，用于处理子模块的提取和整理
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        None,
        [""],
        "",
        {
            entry.fqn: entry.signature
            for entry in root_module.module_call_graph
            if entry.signature
        },
        module=root_module,
    ).run_outer()


def _reorder_submodules(
    parent: torch.nn.Module, fqn_order: Dict[str, int], prefix: str = ""
):
    # 如果前缀为空字符串，则尝试按照指定的顺序添加子模块
    if prefix == "":
        for fqn in list(fqn_order.keys())[1:]:
            if _get_submodule(parent, fqn) is None:
                _add_submodule(parent, fqn, torch.nn.Module())

    # 存储子模块的列表，用于进一步的排序和注册
    children = []

    # 遍历父模块下的每一个子模块，重新命名并添加到子模块列表中
    for name, child in list(parent._modules.items()):
        if child is None:
            continue
        fqn = prefix + name
        _reorder_submodules(child, fqn_order, prefix=fqn + ".")
        delattr(parent, name)
        children.append((fqn_order[fqn], name, child))

    # 根据 fqn_order 中的顺序对子模块进行排序
    children.sort(key=operator.itemgetter(0))

    # 注册已排序的子模块到父模块中
    for _, name, child in children:
        parent.register_module(name, child)


def _sink_params(
    module: torch.nn.Module,
    inputs_to_state: Dict[str, List[str]],
    scope: List[str],
):
    """Sink params, buffers, and constants from graph inputs into get_attr nodes.

    Exported modules are purely functional, so they pass their parameters and
    buffers in as inputs to the graph.

    To replicate eager's semantics, we need to get them from the module state
    via get_attr instead.

    module: GraphModule, potentially containining nested submodules.
    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.
    scope: tracks where we are in the module hierarchy, so that we can emit the
        right `getattr(self, "foo.bar")` calls, etc.
    """
    # 记录由子模块移除的输入的字典，以模块对象的内存地址为键，移除的节点名称列表为值
    module_id_to_inputs_removed: Dict[int, List[str]] = defaultdict(list)

    # 使用 _modules 而不是 named_children()，因为需要在遍历中显示重复的模块
    for name, submodule in module._modules.items():
        submod_id_to_inputs_removed = _sink_params(
            cast(torch.nn.Module, submodule), inputs_to_state, scope + [name]
        )
        for k, v in submod_id_to_inputs_removed.items():
            module_id_to_inputs_removed[k].extend(v)

    # 如果模块没有定义图形属性，则直接返回记录输入移除的字典
    if not hasattr(module, "graph"):
        return module_id_to_inputs_removed

    # 否则获取模块的图形属性
    graph = module.graph
    # 从图中筛选出所有操作为"placeholder"的节点，并存入列表中
    inputs = list(filter(lambda n: n.op == "placeholder", graph.nodes))
    # 获取最后一个placeholder节点
    the_last_input = inputs[-1]

    # 筛选出所有操作为"call_module"的节点，从图中移除对应的placeholder节点
    call_module_nodes = filter(lambda n: n.op == "call_module", graph.nodes)
    for node in call_module_nodes:
        # 通过节点的目标路径获取对应的子模块
        submodule = _recursive_getattr(module, node.target.split("."))
        # 如果子模块存在并且其ID在已移除节点的字典中，则更新节点的参数
        if submodule is not None and id(submodule) in module_id_to_inputs_removed:
            node.args = tuple(
                filter(
                    lambda n: n.name not in module_id_to_inputs_removed[id(submodule)],
                    node.args,
                )
            )

    # 筛选出图中所有与当前作用域相关的输入到状态的节点
    inputs_to_state_of_scope: Dict[torch.fx.Node, list[str]] = {}
    for node in inputs:
        if node.name not in inputs_to_state:
            continue
        
        state_name = None
        for sn in inputs_to_state[node.name]:
            sn_split = sn.split(".")
            # 如果状态名称与当前作用域匹配，则记录该节点与其状态名称
            if sn_split[: len(scope)] == scope:
                state_name = sn_split
                break
        
        # 如果找不到匹配的状态名称，则跳过该节点
        if state_name is None:
            continue
        
        inputs_to_state_of_scope[node] = state_name

    # 记录被移除的输入节点名称，以备返回
    inputs_removed: List[str] = []

    # 遍历作用域中每个节点及其对应的状态名称
    for node, state_name in inputs_to_state_of_scope.items():
        # 如果节点有用户使用，则获取状态属性并创建新的"get_attr"节点
        if len(node.users) > 0:
            attr_path = state_name[len(scope) :]
            state_attr = _recursive_getattr(module, attr_path)
            assert isinstance(state_attr, (torch.Tensor, torch.ScriptObject))

            # 确保新创建的"get_attr"节点位于最后一个placeholder节点之后
            with graph.inserting_after(the_last_input):
                new_node = graph.create_node("get_attr", ".".join(attr_path))

            # 用新节点替换当前节点的所有使用，并传播元数据
            node.replace_all_uses_with(new_node, propagate_meta=True)

        # 从图中移除当前节点
        graph.erase_node(node)
        # 记录已移除的输入节点名称
        inputs_removed.append(node.name)

    # 如果module是InterpreterModule的实例，则执行finalize方法
    if isinstance(module, InterpreterModule):
        module.finalize()

    # 返回模块ID到被移除输入节点名称的字典
    return {id(module): inputs_removed}
# 定义一个函数，用于递归地获取对象的属性值
def _recursive_getattr(obj, attr_path):
    # 遍历属性路径中的每一个属性名
    for attr in attr_path:
        # 检查对象是否具有当前属性
        if not hasattr(obj, attr):
            # 如果对象缺少当前属性，则返回 None
            return None
        # 获取对象的当前属性值，并将其作为新的对象
        obj = getattr(obj, attr)

    # 返回最终找到的对象属性值
    return obj
```