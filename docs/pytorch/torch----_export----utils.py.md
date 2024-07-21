# `.\pytorch\torch\_export\utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import ast                    # 提供抽象语法树操作的功能
import dataclasses           # 提供数据类（dataclass）的支持
import inspect               # 提供获取对象信息的函数
import math                  # 提供数学函数
import operator              # 提供操作符函数
import re                    # 提供正则表达式操作

from inspect import Parameter  # 导入参数对象
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type  # 导入类型提示

import torch  # 导入 PyTorch 库
from torch._subclasses.fake_tensor import FakeTensor  # 导入假张量类

from torch.export import ExportedProgram  # 导入导出程序类
from torch.export.exported_program import (  # 导入导出程序相关功能
    _name_hoo_subgraph_placeholders,
    _rename_without_collisions,
)
from torch.export.graph_signature import InputKind, OutputKind  # 导入图签名相关类别
from torch.utils._pytree import (  # 导入 PyTree 相关功能
    _register_pytree_node,
    Context,
    FlattenFunc,
    FromDumpableContextFn,
    GetAttrKey,
    KeyPath,
    keystr,
    MappingKey,
    SequenceKey,
    ToDumpableContextFn,
    tree_flatten_with_path,
    UnflattenFunc,
)

# 定义不同输入类型的前缀映射关系
placeholder_prefixes = {
    InputKind.USER_INPUT: "",           # 用户输入类型前缀为空
    InputKind.PARAMETER: "p_",          # 参数类型前缀为 'p_'
    InputKind.BUFFER: "b_",             # 缓冲区类型前缀为 'b_'
    InputKind.CONSTANT_TENSOR: "c_",    # 常量张量类型前缀为 'c_'
    InputKind.CUSTOM_OBJ: "obj_",       # 自定义对象类型前缀为 'obj_'
    InputKind.TOKEN: "token",           # 标记类型前缀为 'token'
}


def _check_input_constraints_for_graph(
    input_placeholders: List[torch.fx.Node],  # 输入占位符列表
    flat_args_with_path,                     # 扁平化的参数路径
    range_constraints                        # 范围约束
):
    def get_keystr(key_path: KeyPath) -> str:
        """For a given index into the flat_args, return a human readable string
        describing how to access it, e.g. "*args["foo"][0].bar"
        """
        # 返回扁平化参数路径对应的可读字符串表示
        # 如果索引为0，表示是 *args 对象；否则是 **kwargs 对象
        args_kwargs_key_path = key_path[0]
        assert isinstance(args_kwargs_key_path, SequenceKey)
        if args_kwargs_key_path.idx == 0:
            return f"*args{keystr(key_path[1:])}"
        else:
            kwarg_key = key_path[1]
            assert isinstance(kwarg_key, MappingKey)
            name = str(kwarg_key)[1:-1]  # 去掉括号，获取名称字符串
            return f"{name}{keystr(key_path[2:])}"

    import sympy  # 导入 sympy 库，用于符号计算

    from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
        _convert_range_to_int,  # 导入将范围转换为整数的函数
    )
    from torch.utils._sympy.solve import try_solve  # 导入符号求解功能

    # 检查输入占位符列表与扁平化参数路径的长度是否一致
    if len(flat_args_with_path) != len(input_placeholders):
        raise RuntimeError(
            "Unexpected number of inputs "
            f"(expected {len(input_placeholders)}, got {len(flat_args_with_path)})"
        )
    # 注意：导出程序已经保证了所有由等式约束相关的 InputDims 使用相同符号，
    # 因此我们可以统一使用给定输入维度值的符号来检查等式约束。
    unification_map: Dict[sympy.Symbol, Any] = {}


def register_dataclass_as_pytree_node(
    cls: Type[Any],                                # 数据类类型
    flatten_fn: Optional[FlattenFunc] = None,       # 扁平化函数（可选）
    unflatten_fn: Optional[UnflattenFunc] = None,   # 反扁平化函数（可选）
    *,
    serialized_type_name: Optional[str] = None,     # 序列化类型名称（可选）
    to_dumpable_context: Optional[ToDumpableContextFn] = None,   # 转换为可转储上下文的函数（可选）
    from_dumpable_context: Optional[FromDumpableContextFn] = None,   # 从可转储上下文中恢复的函数（可选）
):
    pass  # 注册数据类作为 PyTree 节点的函数，暂无具体实现
    # 定义一个布尔类型的参数 return_none_fields，并初始化为 False
    return_none_fields: bool = False,
) -> None:
    # 断言检查是否为数据类，否则抛出异常
    assert dataclasses.is_dataclass(
        cls
    ), f"Only dataclasses can be registered with this function: {cls}"

    def default_flatten_fn(obj: Any) -> Tuple[List[Any], Context]:
        # 初始化空列表和名称列表
        flattened = []
        flat_names = []
        none_names = []
        # 遍历数据类的字段
        for f in dataclasses.fields(obj):
            name, val = f.name, getattr(obj, f.name)
            # 如果字段值不为 None 或者设置了返回 None 字段时，添加到列表中
            if val is not None or return_none_fields:
                flattened.append(val)
                flat_names.append(name)
            else:
                none_names.append(name)
        return flattened, [flat_names, none_names]

    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, none_names = context
        # 使用给定的值和上下文信息重新构造数据类对象
        return cls(**dict(zip(flat_names, values)), **dict.fromkeys(none_names))

    def default_flatten_fn_with_keys(obj: Any) -> Tuple[List[Any], Context]:
        # 使用指定的 flatten_fn 函数对对象进行扁平化处理
        flattened, (flat_names, none_names) = flatten_fn(obj)  # type: ignore[misc]
        # 将字段名和值以 MappingKey 包装后返回
        return [(MappingKey(k), v) for k, v in zip(flat_names, flattened)], flat_names

    # 如果未指定 flatten_fn，则使用默认的 flatten_fn 函数
    flatten_fn = flatten_fn if flatten_fn is not None else default_flatten_fn
    # 如果未指定 unflatten_fn，则使用默认的 unflatten_fn 函数
    unflatten_fn = unflatten_fn if unflatten_fn is not None else default_unflatten_fn

    # 检查 to_dumpable_context 和 from_dumpable_context 是否同时为 None 或者同时已注册
    if (to_dumpable_context is None) ^ (from_dumpable_context is None):
        raise ValueError(
            f"Both to_dumpable_context and from_dumpable_context for {cls} must "
            "be None or registered."
        )

    # 注册 PyTree 节点信息
    _register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        flatten_with_keys_fn=default_flatten_fn_with_keys,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def is_param(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a parameter within the exported program
    """

    # 检查节点名称是否在输入参数的签名字典中
    return node.name in program.graph_signature.inputs_to_parameters


def get_param(
    program: ExportedProgram,
    node: torch.fx.Node,
) -> Optional[torch.nn.Parameter]:
    """
    Returns the parameter associated with the given node in the exported program.
    Returns None if the node is not a parameter within the exported program
    """

    # 如果节点是参数，则返回该节点在状态字典中对应的参数值，否则返回 None
    if is_param(program, node):
        parameter_name = program.graph_signature.inputs_to_parameters[node.name]
        return program.state_dict[parameter_name]

    return None


def is_buffer(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a buffer within the exported program
    """

    # 检查节点名称是否在输入缓冲区的签名字典中
    return node.name in program.graph_signature.inputs_to_buffers


def get_buffer(
    program: ExportedProgram,
    node: torch.fx.Node,
) -> Optional[torch.Tensor]:
    """
    Returns the buffer associated with the given node in the exported program.
    Returns None if the node is not a buffer within the exported program
    """

    # 如果节点是缓冲区，则返回该节点在状态字典中对应的张量，否则返回 None
    # 如果节点是缓冲区，则根据节点名称从程序的输入到缓冲区的映射中获取缓冲区的名称
    if is_buffer(program, node):
        # 从程序的图形签名中检索非持久性缓冲区的名称
        buffer_name = program.graph_signature.inputs_to_buffers[node.name]
        # 如果缓冲区名称在非持久性缓冲区列表中，则返回该缓冲区在程序常量中的值
        if buffer_name in program.graph_signature.non_persistent_buffers:
            return program.constants[buffer_name]
        else:
            # 否则，返回该缓冲区在程序状态字典中的值
            return program.state_dict[buffer_name]

    # 如果节点不是缓冲区，则返回空值
    return None
def is_lifted_tensor_constant(
    program: ExportedProgram,
    node: torch.fx.Node,
) -> bool:
    """
    Checks if the given node is a lifted tensor constant within the exported program.
    Returns True if the node is a lifted tensor constant, otherwise False.
    """

    return node.name in program.graph_signature.inputs_to_lifted_tensor_constants


def get_lifted_tensor_constant(
    program: ExportedProgram,
    node: torch.fx.Node,
) -> Optional[torch.Tensor]:
    """
    Returns the lifted tensor constant associated with the given node in the exported program.
    Returns None if the node is not a lifted tensor constant within the exported program.
    """

    if is_lifted_tensor_constant(program, node):
        lifted_tensor_name = program.graph_signature.inputs_to_lifted_tensor_constants[
            node.name
        ]
        return program.constants[lifted_tensor_name]

    return None


def sequential_split(gm: torch.fx.GraphModule, node_call_back) -> torch.fx.GraphModule:
    """
    Creates a new graph module that splits the input graph module into multiple submodules
    based on the node_call_back function.
    """

    from torch.fx.passes.split_module import split_module

    # Assign a split_id to each node based on the node_call_back function
    split_map = {}
    split_id = 0
    for node in gm.graph.nodes:
        if node_call_back(node):
            split_id += 1
        split_map[node] = split_id

    # Split the graph module using split_module function with the split_map
    new_gm = split_module(
        gm,
        gm,
        lambda node: split_map[node],
        keep_original_order=True,
        keep_original_node_name=True,
    )
    # Preserve code generation info from the original graph module
    new_gm.graph._codegen = gm.graph._codegen
    new_gm.recompile()
    return new_gm


def nodes_filter(nodes: List[torch.fx.Node], node_call_back) -> List[torch.fx.Node]:
    """
    Returns a list of nodes that match the node_call_back condition.
    """

    return [node for node in nodes if node_call_back(node)]


def nodes_first(
    nodes: List[torch.fx.Node], node_call_back=None
) -> Optional[torch.fx.Node]:
    """
    Returns the first node in the list that matches the node_call_back condition.
    If node_call_back is None, returns the first node in the list.
    Returns None if no matching node is found.
    """

    ret = nodes_filter(nodes, node_call_back if node_call_back else lambda node: True)
    if len(ret) > 0:
        return ret[0]
    return None


def nodes_count(nodes: List[torch.fx.Node], node_call_back) -> int:
    """
    Returns the number of nodes in the list that match the node_call_back condition.
    """

    return len(nodes_filter(nodes, node_call_back))


def nodes_map(nodes: List[torch.fx.Node], node_call_back) -> List[torch.fx.Node]:
    """
    Applies the node_call_back function to each node in the list sequentially.
    Returns the list of nodes after applying node_call_back to each element.
    """

    for node in nodes:
        node_call_back(node)
    return nodes


def node_replace_(
    # 定义函数的参数：旧节点（类型为 torch.fx.Node）、新节点（类型为 torch.fx.Node）、是否删除旧节点的标志（默认为 False）
    old_node: torch.fx.Node, new_node: torch.fx.Node, delete_old: bool = False
def replace_all_uses_with(old_node: torch.fx.Node, new_node: torch.fx.Node, delete_old: bool) -> None:
    """
    Replace all uses of old_node with new_node.
    """
    # 替换所有使用 old_node 的地方为 new_node
    old_node.replace_all_uses_with(new_node)
    if delete_old:
        # 如果需要删除旧节点，则清空旧节点的用户列表，并从图中删除旧节点
        old_node.users.clear()
        old_node.graph.erase_node(old_node)


def node_inline_(call_mod_node: torch.fx.Node) -> None:
    """
    Inline the submodule of the given node into the parent module.
    Note: we only support the case where submodule takes tensors inputs.
    """
    # 断言调用节点的操作为 "call_module"
    assert call_mod_node.op == "call_module"
    gm = call_mod_node.graph.owning_module

    # 断言调用节点的目标为字符串类型
    assert isinstance(call_mod_node.target, str)
    sub_gm = getattr(gm, call_mod_node.target)

    # 获取子模块中所有的占位符节点和非占位符节点（body），以及输出节点
    phs = (node for node in sub_gm.graph.nodes if node.op == "placeholder")
    body = (
        node for node in sub_gm.graph.nodes if node.op not in ("placeholder", "output")
    )
    output = [node for node in sub_gm.graph.nodes if node.op == "output"]

    # 替换占位符节点为调用节点的实际参数节点
    for ph, arg in zip(phs, call_mod_node.args):
        assert isinstance(arg, torch.fx.Node)
        replace_all_uses_with(ph, arg, delete_old=True)

    # 在调用节点之前插入新节点
    with gm.graph.inserting_before(call_mod_node):
        # 复制子模块中的所有非占位符节点到父模块中，并替换原节点为新节点
        for node in body:
            new_node = gm.graph.node_copy(node)
            replace_all_uses_with(node, new_node, delete_old=True)

        # 如果子模块有输出节点
        if len(output) > 0:
            assert len(output) == 1 and len(output[0].args) == 1
            new_output = output[0].args[0]

            if isinstance(new_output, torch.fx.Node):
                replace_all_uses_with(call_mod_node, new_output, delete_old=True)
            elif isinstance(new_output, (list, tuple)):
                # 内联输出节点的 get_item 调用
                get_item_users = nodes_filter(
                    list(call_mod_node.users.keys()),
                    lambda node: node.op == "call_function"
                    and node.target == operator.getitem,
                )
                # get_item_node.args[1] 是索引，指向 new_output[idx]
                nodes_map(
                    get_item_users,
                    lambda get_item_node: replace_all_uses_with(
                        get_item_node,
                        new_output[get_item_node.args[1]],
                        delete_old=True,
                    ),
                )
                call_mod_node.graph.erase_node(call_mod_node)
            else:
                raise NotImplementedError(
                    f"Unsupported output type {type(new_output)}. Expect it to be a Node or a list/tuple of Nodes."
                )
        else:
            call_mod_node.graph.erase_node(call_mod_node)

    # 删除所有未使用的子模块，并重新编译父模块
    gm.delete_all_unused_submodules()
    gm.recompile()
    return gm


def _get_torch_jit_trace_forward_signature(mod: torch.nn.Module):
    """
    Get source code and parse argument names using AST. The function returns
    a signature of the forward() function.

    # TODO: Directly provide inspect.signature compatible TS-d module.
    """
    # 使用 AST 获取模块的源代码并解析参数名，返回 forward() 函数的签名
    ast_mod = ast.parse(mod.code)
    ast_func_def: ast.FunctionDef = ast_mod.body[0]  # type: ignore[assignment]
    # 定义一个字典，指定参数类型映射，目前只允许位置参数或关键字参数
    arg_type_map = {"args": Parameter.POSITIONAL_OR_KEYWORD}

    # 遍历 AST 树中的所有参数类型，并创建相应的参数
    param_list = []
    for arg_type, param_type in arg_type_map.items():
        # 获取 AST 函数定义中特定类型的参数列表
        arg_name_list = [a.arg for a in getattr(ast_func_def.args, arg_type)]
        for arg_name in arg_name_list:
            if arg_name == "self":
                continue  # 跳过 self 参数
            # 创建参数对象并加入到参数列表中
            param_list.append(inspect.Parameter(arg_name, param_type))

    # 返回根据参数列表创建的函数签名对象
    return inspect.Signature(parameters=param_list)
# 绑定签名到输入参数
def _bind_signature_to_inputs(mod, fake_args, fake_kwargs):
    # 如果模块是 TorchScript 的 ScriptModule 或 TracedModule 类型，则获取其前向传播函数的签名
    if isinstance(mod, (torch.jit.ScriptModule, torch.jit.TracedModule)):
        sig = _get_torch_jit_trace_forward_signature(mod)

        # 对于来自 TorchScript 的占位符名称进行健全性检查
        assert len(sig.parameters) == len(fake_args) + len(fake_kwargs), (
            "Arguments other than POSITIONAL_OR_KEYWORD kinds in forward() "
            "are not supported in _get_torch_jit_trace_forward_signature"
        )
    else:
        # 否则获取普通模块的 forward 方法的签名
        sig = inspect.signature(mod.forward)

    # 使用假参数绑定到签名，并返回参数字典
    return sig.bind(*fake_args, **fake_kwargs).arguments


def placeholder_naming_pass(
    gm: torch.fx.GraphModule,
    export_graph_signature: torch.export.ExportGraphSignature,
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constants: Dict[str, Any],
) -> None:
    """
    This pass is run at the end of _export_non_strict() to assign better placeholder node names:
        - User inputs:
            These follow the signature of mod.forward(), e.g. forward(x, y) produces nodes x, y.
            For nested inputs from dictionaries, lists, tuples, or dataclasses,
            the names are a concatenation of the path to the tensor.
                e.g. x = {
                    'a': torch.randn(),
                    'b': [torch.randn(), torch.randn()]
                }
            produces nodes x_a, x_b_0, x_b_1.
        - Parameters/buffers/constants/custom objects:
            These follow the FQN of the object, prefixed by "p", "b", "c", "obj" respectively.
                e.g. self.bar.l0.weight produces "p_bar_l0_weight".
        - Effect tokens:
            These are named token, token_1, ...
    """

    def _strip_name(x):
        # 移除特定前缀并将非字母数字字符替换为下划线
        if x.startswith("L__self___"):
            x = x[len("L__self___") :]
        x = re.sub(r"[^a-zA-Z0-9]", "_", x)
        return x

    def _extract_pytree_key(x):
        # 根据输入的类型提取 Pytree 的键
        if isinstance(x, MappingKey):
            x = re.sub(r"[^a-zA-Z0-9]", "_", str(x.key))
            return x
        elif isinstance(x, SequenceKey):
            return str(x.idx)
        elif isinstance(x, GetAttrKey):
            return x.name
        else:
            raise RuntimeError(f"Pytree key of type {type(x)} not handled for {x}")

    # 存储从路径到名称的映射
    name_map: Dict[str, str] = {}

    # 将模型的输入参数绑定到模型的 forward 方法的签名
    combined_args = _bind_signature_to_inputs(mod, fake_args, fake_kwargs)

    # 使用树形展开方法获取平铺后的参数及其路径
    flat_args_with_path, _ = tree_flatten_with_path(combined_args)

    # 获取用户输入的名称列表，这些名称来自导出图的输入规格
    user_input_names = [
        spec.arg.name
        for spec in export_graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]

    # 使用 Pytree 路径为嵌套的用户输入命名
    # 使用 zip 函数将参数列表 flat_args_with_path 和用户输入名称列表 user_input_names 一一对应起来进行处理
    for (arg_path, arg), user_input_name in zip(flat_args_with_path, user_input_names):
        # 如果用户提供了输入名称 user_input_name，则调用 _rename_without_collisions 函数重命名，避免命名冲突
        if user_input_name:
            # 构建新的名称，包括路径中的元素以及用户提供的名称，并使用下划线连接
            _rename_without_collisions(
                name_map,
                user_input_name,
                placeholder_prefixes[InputKind.USER_INPUT] + "_".join(_extract_pytree_key(x).lower() for x in arg_path),
                is_placeholder=True,
            )

    # 使用图签名的输入规范来映射参数/缓冲区/常量的名称
    # 将 token 参数命名为 token、token_1 等（这些对用户不可见）
    for spec in export_graph_signature.input_specs:
        # 如果是用户输入，则跳过
        if spec.kind == InputKind.USER_INPUT:
            continue
        # 如果是 token 类型，则 base_name 为空字符串；否则，使用目标名称的小写形式
        if spec.kind == InputKind.TOKEN:
            base_name = ""
        else:
            base_name = _strip_name(spec.target).lower()
        # 移除 base_name 中的非字母数字字符，用下划线替换
        base_name = re.sub(r"[^a-zA-Z0-9]", "_", base_name)

        # 调用 _rename_without_collisions 函数重命名参数名称
        _rename_without_collisions(
            name_map,
            spec.arg.name,
            placeholder_prefixes[spec.kind] + base_name,
            is_placeholder=True,
        )

    # 处理调用函数/get_attr 输入的命名冲突
    # 在此处，我们希望优先使用用户输入的名称覆盖调用函数的名称
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        # 调用 _rename_without_collisions 函数确保不会发生命名冲突
        _rename_without_collisions(name_map, node.name, node.name)

    # 为图中的节点分配新的名称
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # 断言确保 name_map 中存在对应的映射关系
            assert node.name in name_map
            # 更新节点的名称和目标名称为 name_map 中的映射值
            node.name = node.target = name_map[node.name]
        elif node.name in name_map:
            # 更新节点的名称为 name_map 中的映射值
            node.name = name_map[node.name]

    # 传播名称到高阶操作子图中的占位符
    _name_hoo_subgraph_placeholders(gm)

    # 重新编译图模块代码
    gm.recompile()

    # 修改图签名（输入规范、输出规范、用户输入变异）
    for spec in export_graph_signature.input_specs:
        # 断言确保 spec.arg.name 存在于 name_map 中
        assert spec.arg.name in name_map
        # 更新 spec.arg.name 为 name_map 中的映射值
        spec.arg.name = name_map[spec.arg.name]
        # 如果是 CUSTOM_OBJ 类型且 spec.target 存在于 name_map 中，则去除 obj_ 前缀
        if (spec.kind == InputKind.CUSTOM_OBJ and spec.target in name_map):
            spec.target = name_map[spec.target][4:]  # 去除 obj_ 前缀

    for spec in export_graph_signature.output_specs:
        # 如果 spec.arg.name 存在于 name_map 中，则更新为 name_map 中的映射值
        if spec.arg.name in name_map:
            spec.arg.name = name_map[spec.arg.name]
        # 如果是 USER_INPUT_MUTATION 类型且 spec.target 存在于 name_map 中，则更新为 name_map 中的映射值
        if spec.kind == OutputKind.USER_INPUT_MUTATION and spec.target in name_map:
            spec.target = name_map[spec.target]

    # 为自定义对象常量字典中的键重命名
    # 遍历常量字典中的所有键值对
    for name in list(constants.keys()):
        # 获取当前常量的值
        constant = constants[name]
        # 如果当前常量名存在于名称映射表中，并且常量不是 torch.Tensor 类型
        if name in name_map and not isinstance(
            constant, torch.Tensor
        ):
            # 对自定义对象使用通用名称进行重命名
            new_name = name_map[name]
            # 检查新旧名称不同，并且旧名称匹配特定模式 "arg(\d+)_1"，
            # 同时新名称不是指定输入类型的自定义对象前缀加上旧名称
            if (
                new_name != name
                and re.match(r"arg(\d+)_1", name)
                and new_name != placeholder_prefixes[InputKind.CUSTOM_OBJ] + name
            ):
                # 将新名称映射到当前常量的值
                constants[new_name] = constant
                # 删除原始名称的常量条目
                del constants[name]
```