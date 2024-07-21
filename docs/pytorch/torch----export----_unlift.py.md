# `.\pytorch\torch\export\_unlift.py`

```py
# mypy: allow-untyped-defs
# 导入所需模块和类型声明
import copy  # 导入copy模块，用于对象的深复制
from itertools import chain  # 导入chain函数，用于迭代器的扁平化处理
from typing import Any, Dict, List, Optional, Tuple  # 导入类型声明，定义函数参数和返回类型

import torch  # 导入PyTorch库
import torch.utils._pytree as pytree  # 导入PyTorch的_pytree模块，用于处理树结构
from torch._export.utils import _check_input_constraints_for_graph  # 导入用于检查图输入约束的函数
from torch.export.unflatten import _assign_attr, _AttrKind  # 导入用于属性分配和属性种类的函数
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo  # 导入用于PyTree代码生成和信息的函数和类
from ._remove_effect_tokens_pass import _remove_effect_tokens  # 导入用于移除效果标记的函数

from .exported_program import (  # 导入导出程序相关的模块和类
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    OutputKind,
)


@torch._dynamo.disable  # 使用torch._dynamo.disable装饰器，禁用动态分析功能
def _check_input_constraints_pre_hook(self, *args, **kwargs):
    # 将输入参数args按照树的方式扁平化，并获取扁平化后的路径和接收到的规范
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path(args)

    # 如果接收到的规范与self._in_spec不匹配，则抛出ValueError异常
    if received_spec != self._in_spec:
        raise ValueError(  # noqa: B904
            "Trying to flatten user inputs with exported input tree spec: \n"
            f"{self._in_spec}\n"
            "but actually got inputs with tree spec of: \n"
            f"{received_spec}"
        )

    # 调用_check_input_constraints_for_graph函数，检查图中占位符节点的输入约束
    return _check_input_constraints_for_graph(
        [node for node in self.graph.nodes if node.op == "placeholder"],  # 获取所有操作为"placeholder"的节点列表
        flat_args_with_path,  # 扁平化后的参数列表和路径
        self.range_constraints,  # 范围约束
    )


def _unlift_inputs_as_getattr(
    gm: torch.fx.GraphModule,  # 输入类型为torch.fx.GraphModule
    lifted_inputs: List[Optional[str]],  # 提升后的输入名称列表，可选字符串
) -> Tuple[Dict[str, torch.fx.Node], Dict[str, torch.fx.Node]]:
    """
    将引用参数/缓冲区/常量的输入作为图中的getattr节点解除提升
    """
    unlifted_name_to_node = {}  # 未提升名称到节点的映射
    input_name_to_node = {}  # 输入名称到节点的映射

    # 获取所有占位符节点列表
    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    assert len(lifted_inputs) == len(placeholder_nodes)  # 断言提升后的输入名称列表长度与占位符节点列表长度相等

    # 遍历占位符节点和提升后的输入名称列表
    for input_node, lifted_node in zip(placeholder_nodes, lifted_inputs):
        if lifted_node is None:
            input_name_to_node[input_node.name] = input_node  # 如果提升节点为None，则将占位符节点加入输入名称到节点映射中
        else:
            with gm.graph.inserting_after(input_node):
                # 获取getattr节点，并用占位符节点替换所有使用该节点的地方
                getattr_node = gm.graph.get_attr(lifted_node)
                input_node.replace_all_uses_with(getattr_node)
                metadata = input_node.meta
                gm.graph.erase_node(input_node)  # 擦除占位符节点
                getattr_node.meta = metadata
                unlifted_name_to_node[lifted_node] = getattr_node  # 将未提升名称到节点的映射中加入提升节点和getattr节点的映射关系

    return unlifted_name_to_node, input_name_to_node  # 返回未提升名称到节点的映射和输入名称到节点的映射


def _insert_copy_for_mutations(
    gm: torch.fx.GraphModule,  # 输入类型为torch.fx.GraphModule
    mutated_outputs: List[Optional[str]],  # 突变输出的名称列表，可选字符串
    unlifted_name_to_node: Dict[str, torch.fx.Node],  # 未提升名称到节点的映射，字符串到节点
    input_name_to_node: Dict[str, torch.fx.Node],  # 输入名称到节点的映射，字符串到节点
) -> None:
    """
    找到所有被突变的缓冲区和输入，并插入copy_运算符以反映突变
    """
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":  # 如果节点操作为"output"
            output_node = node  # 将节点赋给output_node
            break
    assert output_node is not None  # 断言output_node不为None
    outputs = pytree.tree_flatten(output_node.args)[0]  # 将输出节点参数按树结构展平并获取第一个元素
    assert len(outputs) == len(mutated_outputs)  # 断言输出长度与突变输出名称列表长度相等

    user_output_nodes = []  # 用户输出节点列表
    return_nodes_to_copy = {}  # 返回节点到复制映射的字典
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        # 遍历每个输出节点及其对应的变异节点名
        if mutated_node_name is None:
            # 如果变异节点名为 None，则将返回节点添加到用户输出节点列表中，并继续下一个迭代
            user_output_nodes.append(return_node)
            continue

        if mutated_node_name in unlifted_name_to_node:
            # 如果变异节点名在未提升的名称到节点映射中存在，则使用对应的节点
            mutated_node = unlifted_name_to_node[mutated_node_name]
        elif mutated_node_name in input_name_to_node:
            # 如果变异节点名在输入名称到节点映射中存在，则使用对应的节点
            mutated_node = input_name_to_node[mutated_node_name]
        else:
            # 如果变异节点名既不在未提升的名称到节点映射中，也不在输入名称到节点映射中，则抛出运行时错误
            raise RuntimeError(
                f"Could not find {mutated_node_name} in either buffer or input nodes"
            )

        with gm.graph.inserting_before(output_node):
            # 在输出节点之前插入操作节点
            copy_node = gm.graph.call_function(
                torch.ops.aten.copy_.default, (mutated_node, return_node)
            )
            # 将返回节点到复制节点的映射保存起来
            return_nodes_to_copy[return_node] = copy_node

    output_args = [
        return_nodes_to_copy[node] if node in return_nodes_to_copy else node
        for node in user_output_nodes
    ]
    with gm.graph.inserting_before(output_node):
        # 仅返回用户输出节点
        new_output = gm.graph.output(tuple(output_args))
        # 用新的输出节点替换原始输出节点的所有使用
        output_node.replace_all_uses_with(new_output)
        # 在图中擦除原始输出节点
        gm.graph.erase_node(output_node)
def _get_codegen(
    in_spec: pytree.TreeSpec,
    out_spec: Optional[pytree.TreeSpec],
    forward_arg_names: Optional[List[str]] = None,
) -> _PyTreeCodeGen:
    """
    Create the codegen for the graph module based on the in/out specs
    """
    # 如果有指定前向参数的名称列表，则使用该列表作为参数名
    if forward_arg_names:
        names = forward_arg_names
    else:
        # 如果未指定前向参数名列表，则根据输入规范生成参数名列表
        if (
            in_spec.type == tuple
            and in_spec.num_children == 2
            and in_spec.children_specs[0].type == tuple
            and in_spec.children_specs[1].type == dict
        ):
            # 如果输入规范包含参数 (tuple) 和关键字参数 (dict)
            # 生成参数名列表，包括参数名和关键字参数名
            names = [f"arg_{i}" for i in range(in_spec.children_specs[0].num_children)]
            names.extend(in_spec.children_specs[1].context)
        else:
            # 否则，根据输入规范生成参数名列表
            names = [f"arg_{i}" for i in range(in_spec.num_children)]

    # 返回一个 _PyTreeCodeGen 对象，该对象包含生成代码所需的信息
    return _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )


def _unlift(
    gm: torch.fx.GraphModule,
    lifted_inputs: List[Optional[str]],
    mutated_outputs: List[Optional[str]],
    in_spec: pytree.TreeSpec,
    out_spec: Optional[pytree.TreeSpec],
    state_dict: Dict[str, Any],
    constants: Dict[str, Any],
    forward_arg_names: Optional[List[str]] = None,
):
    """
    Args:
        lifted_inputs: A list matching the graph module's input nodes. For
        an input node that is referring to a lifted parameter/buffer, this
        list will contain the fqn the corresponding attribute. Otherwise, this
        list will contain None. This is used to unlift the lifted parameters as
        get_attr nodes.

        mutated_outputs: A list matching the graph module's output nodes. For
        an output node that is referring to a mutated buffer or user input, this
        list will contain the name of the corresponding buffer or user input
        that needs to be mutated. Otherwise, this list will contain None. This
        is used to re-insert an inplace copy_ operator to copy the mutated
        values back to the original node.
    """
    # 解除 lifted_inputs 中的 lifted 参数，并用 get_attr 节点替换
    unlifted_name_to_node, input_name_to_node = _unlift_inputs_as_getattr(
        gm, lifted_inputs
    )
    # 为发生突变的输出插入复制操作符
    _insert_copy_for_mutations(
        gm, mutated_outputs, unlifted_name_to_node, input_name_to_node
    )
    # 为图形模块设置代码生成器
    gm.graph._codegen = _get_codegen(in_spec, out_spec, forward_arg_names)
    # 检查图的合法性
    gm.graph.lint()
    # 消除死代码
    gm.graph.eliminate_dead_code()
    # 重新编译图形模块
    gm.recompile()
    # 返回更新后的图形模块
    return gm


def _register_attrs_to_new_gm(
    new_gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    state_dict: Dict[str, Any],
    constants: Dict[str, Any],
) -> None:
    # 将图形模块的非持久化缓冲区注册到新图形模块中
    non_persistent_buffers = set(graph_signature.non_persistent_buffers)
    # 遍历图形签名中的缓冲区名称列表
    for name in graph_signature.buffers:
        # 如果名称在非持久性缓冲区列表中
        if name in non_persistent_buffers:
            # 将持久性标记设为False
            persistent = False
            # 从常量字典中获取对应的值
            value = constants[name]
        else:
            # 否则将持久性标记设为True
            persistent = True
            # 从状态字典中获取对应的值
            value = state_dict[name]
        # 调用_assign_attr函数，将值赋给新的图形模型，并指定属性种类为BUFFER
        _assign_attr(
            value, new_gm, name, attr_kind=_AttrKind.BUFFER, persistent=persistent
        )

    # 遍历图形签名中的参数名称列表
    for name in graph_signature.parameters:
        # 从状态字典中获取对应的参数值
        value = state_dict[name]
        # 调用_assign_attr函数，将参数值赋给新的图形模型，并指定属性种类为PARAMETER
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )

    # 遍历提升的自定义对象和提升的张量常量名称列表的合并结果
    for name in chain(
        graph_signature.lifted_custom_objs, graph_signature.lifted_tensor_constants
    ):
        # 从常量字典中获取对应的值
        value = constants[name]
        # 调用_assign_attr函数，将值赋给新的图形模型，并指定属性种类为CONSTANT
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.CONSTANT,
        )
class _StatefulGraphModuleFactory(type):
    """
    Metaclass that ensures a private constructor for _StatefulGraphModule
    """

    def __call__(cls, *args, **kwargs):
        # 禁止直接实例化该类，抛出 TypeError 异常
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
        )

    def _create(cls, root, graph, range_constraints=None):
        # 使用父类的 __call__ 方法创建实例，并传入参数
        return super().__call__(
            root,
            graph,
            range_constraints=range_constraints,
        )


class _StatefulGraphModule(torch.fx.GraphModule, metaclass=_StatefulGraphModuleFactory):
    def __init__(self, root, graph, range_constraints=None):
        # 调用父类的初始化方法，传入 root 和 graph 参数
        super().__init__(root, graph)
        # 初始化 range_constraints，若未提供则为空列表
        self.range_constraints = range_constraints or []


def _create_stateful_graph_module(
    plain_graph_module: torch.fx.GraphModule,
    range_constraints,
    # TODO(suo) this should not be optional, but is since we still ahve
    # capture_pre_autograd_graph grr
    graph_signature: Optional[ExportGraphSignature] = None,
):
    # 使用 _StatefulGraphModule 的 _create 方法创建状态化的图模块
    stateful_gm = _StatefulGraphModule._create(
        plain_graph_module,
        plain_graph_module.graph,
        range_constraints=range_constraints,
    )
    # 注册前向预钩子函数 _check_input_constraints_pre_hook，并传入关键字参数
    stateful_gm.register_forward_pre_hook(
        _check_input_constraints_pre_hook, with_kwargs=True
    )

    if graph_signature is None:
        return stateful_gm

    # 对非持久化缓冲区进行修复
    # torch.fx 不区分持久化和非持久化缓冲区，需要在此处恢复该区分
    for buffer in graph_signature.non_persistent_buffers:
        # 获取 plain_graph_module 中的缓冲区并赋值给 stateful_gm
        _assign_attr(
            plain_graph_module.get_buffer(buffer),
            stateful_gm,
            buffer,
            attr_kind=_AttrKind.BUFFER,
            persistent=False,
        )

    return stateful_gm


def _unlift_exported_program_lifted_states(ep: ExportedProgram) -> torch.nn.Module:
    # 移除效果标记后的 ExportedProgram
    ep = _remove_effect_tokens(ep)
    # 创建新的 GraphModule 实例 new_gm，复制 ep 的图结构
    new_gm = torch.fx.GraphModule(ep.graph_module, copy.deepcopy(ep.graph))
    # 将 ep 的图签名、状态字典和常量注册到 new_gm
    _register_attrs_to_new_gm(new_gm, ep.graph_signature, ep.state_dict, ep.constants)
    # 获取 forward_arg_names 元数据
    forward_arg_names = ep.graph_module.meta.get("forward_arg_names")

    # 创建 lifted_inputs 列表，包含输入规范的目标值或 None
    lifted_inputs: List[Optional[str]] = [
        (
            in_spec.target
            if in_spec.kind
            in (
                InputKind.BUFFER,
                InputKind.CONSTANT_TENSOR,
                InputKind.PARAMETER,
                InputKind.CUSTOM_OBJ,
            )
            else None
        )
        for in_spec in ep.graph_signature.input_specs
    ]

    # 创建 mutated_outputs 列表，包含输出规范的目标值或 None
    mutated_outputs: List[Optional[str]] = [
        (
            out_spec.target
            if out_spec.kind
            in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
            else None
        )
        for out_spec in ep.graph_signature.output_specs
    ]
    # 使用 _unlift 函数处理新的图形模块 new_gm，传入 lifted_inputs、mutated_outputs、
    # ep.call_spec.in_spec、ep.call_spec.out_spec、ep.state_dict、ep.constants 以及 forward_arg_names 参数
    new_gm = _unlift(
        new_gm,
        lifted_inputs,
        mutated_outputs,
        ep.call_spec.in_spec,
        ep.call_spec.out_spec,
        ep.state_dict,
        ep.constants,
        forward_arg_names=forward_arg_names,
    )
    
    # 使用 _create_stateful_graph_module 函数创建一个带有状态的图形模块 unlift_gm，
    # 使用 new_gm、ep.range_constraints 和 ep.graph_signature 作为参数
    unlift_gm = _create_stateful_graph_module(
        new_gm, ep.range_constraints, ep.graph_signature
    )
    
    # 将 unlift_gm 的元数据更新为 ep.graph_module 的元数据
    unlift_gm.meta.update(ep.graph_module.meta)
    
    # 返回更新后的带有状态的图形模块 unlift_gm
    return unlift_gm
```