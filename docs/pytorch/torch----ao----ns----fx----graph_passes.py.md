# `.\pytorch\torch\ao\ns\fx\graph_passes.py`

```
# mypy: allow-untyped-defs
# 导入需要的模块和函数
import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

# 从本地工具函数中导入多个函数
from .utils import (
    get_node_first_input_and_output_type,
    getattr_from_fqn,
    NodeInputOrOutputType,
    return_first_non_observer_node,
    get_number_of_non_param_args,
    get_target_type_str,
    get_arg_indices_of_inputs_to_log,
    get_node_input_qparams,
    op_type_supports_shadowing,
    get_normalized_nth_input,
)

# 从特定模块导入多个自定义类型
from .ns_types import (
    NSSingleResultValuesType,
    NSSubgraph,
    NSNodeTargetType,
)
# 从指定模块导入函数映射
from torch.ao.ns.fx.mappings import (
    get_node_type_to_io_type_map,
)
# 从量化观察器模块导入特定函数
from torch.ao.quantization.observer import _is_activation_post_process

# 导入类型提示
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set

# 定义一个函数，用于获取节点的完全限定名称（Fully Qualified Name）
def _maybe_get_fqn(node: Node, gm: GraphModule) -> Optional[str]:
    fqn = None
    # 检查 GraphModule 对象是否具有属性 '_node_name_to_scope'
    if hasattr(gm, '_node_name_to_scope'):
        # 如果节点是观察器，则获取其观察的节点的完全限定名称
        # 在跟踪期间创建完全限定名称时，观察器节点并不存在。
        # 如果节点是调用模块，且目标是激活后处理模块，则使用其第一个输入节点作为完全限定名称的节点。
        node_to_use_for_fqn = node
        if node.op == 'call_module':
            assert isinstance(node.target, str)
            module = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(module):
                node_to_use_for_fqn = get_normalized_nth_input(node, gm, 0)
        # 获取节点在 GraphModule 对象中的完全限定名称
        fqn = gm._node_name_to_scope[node_to_use_for_fqn.name][0]  # type: ignore[index]
    return fqn  # type: ignore[return-value]

# 定义一个函数，用于在特定节点后插入日志记录器
def _insert_logger_after_node(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    logger_node_name_suffix: str,
    ref_node_name: str,
    model_name: str,
    ref_name: str,
    ref_node_target_type: str,
    results_type: str,
    index_within_arg: int,
    index_of_arg: int,
    fqn: Optional[str],
) -> Node:
    """
    给定图形的起始节点结构为

    prev_node -> node -> next_node

    该函数创建一个新的 logger_cls 对象，并将其添加在 node 后面，
    结果如下

    prev_node -> node -> logger_obj -> next_node
    """
    # 创建新的日志记录器节点名称
    logger_node_name = \
        get_new_attr_name_with_prefix(node.name + logger_node_name_suffix)(gm)
    # 获取节点的目标类型字符串表示
    target_type = get_target_type_str(node, gm)
    # 创建日志记录器对象
    logger_obj = logger_cls(
        ref_node_name, node.name, model_name, ref_name, target_type,
        ref_node_target_type,
        results_type, index_within_arg, index_of_arg, fqn)
    # 将日志记录器对象附加到父模块中
    setattr(gm, logger_node_name, logger_obj)
    # 在图中创建日志记录器节点
    logger_node = node.graph.create_node(
        'call_module', logger_node_name, (node,), {})
    return logger_node

# 定义一个函数，用于向模型添加日志记录器
def add_loggers_to_model(
    gm: GraphModule,
    node_to_instrument_inputs_to_ref_node_name: Dict[Node, Tuple[str, str]],
    node_to_instrument_outputs_to_ref_node_name: Dict[Node, Tuple[str, str]],
    logger_cls: Callable,
    logger_node_name_suffix: str,
    model_name: str,
) -> None:
    """
    向 GraphModule 中的特定节点添加日志记录器
    """
    model_name: str,


# 定义一个变量 model_name，类型为字符串
# 定义函数_insert_quantize_per_tensor_node，用于在图形中的特定节点后插入量化操作节点
def _insert_quantize_per_tensor_node(
    prev_node_c: Node,  # 前一个节点 prev_node_c
    node_a: Node,       # 当前节点 node_a
    gm_b: GraphModule,  # 图形模块 gm_b
    graph_c: Graph,     # 图形 graph_c
    scale: Union[torch.Tensor, float],  # 量化操作的比例因子
    zero_point: Union[torch.Tensor, int],  # 量化操作的零点
    dtype_cast_name: str,  # 数据类型转换的名称
) -> Node:  # 函数返回一个节点对象

    # 复制比例因子
    scale_node_name = \
        get_new_attr_name_with_prefix(
            node_a.name + '_input_scale_')(gm_b)  # 创建新的比例因子节点名称
    setattr(gm_b, scale_node_name, scale)  # 将比例因子设置为 gm_b 的属性
    scale_node = graph_c.create_node(
        'get_attr', scale_node_name, (), {}, scale_node_name)  # 创建比例因子节点对象

    # 复制零点
    zero_point_node_name = \
        get_new_attr_name_with_prefix(
            node_a.name + '_input_zero_point_')(gm_b)  # 创建新的零点节点名称
    setattr(gm_b, zero_point_node_name, zero_point)  # 将零点设置为 gm_b 的属性
    zero_point_node = graph_c.create_node(
        'get_attr', zero_point_node_name, (), {}, zero_point_node_name)  # 创建零点节点对象

    # 创建 quantize_per_tensor 调用节点
    return graph_c.create_node(
        'call_function', torch.quantize_per_tensor,  # 调用 torch.quantize_per_tensor 函数
        (prev_node_c, scale_node, zero_point_node, torch.quint8), {},  # 设置调用函数的参数
        dtype_cast_name)  # 返回创建的节点对象

# 定义函数_insert_dtype_cast_after_node，用于在特定节点后插入数据类型转换节点
def _insert_dtype_cast_after_node(
    node_a: Node,  # 当前节点 node_a
    node_c: Node,  # 需要插入转换节点的节点 node_c
    prev_node_c: Union[Node, List[Node]],  # 前一个节点 prev_node_c 或节点列表
    gm_a: GraphModule,  # 图形模块 gm_a
    gm_b: GraphModule,  # 图形模块 gm_b
    graph_c: Graph,     # 图形 graph_c
    node_name_prefix: str,  # 节点名称的前缀
    logger_cls: Callable,  # 日志记录器类
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],  # 节点类型到 I/O 类型的映射字典
) -> Union[Node, List[Node]]:  # 函数返回一个节点对象或节点列表

    """
    给定从图 B 派生的起始图 C：

    ... -> prev_node_c -> node_c -> ...

    和相关的 node_a，插入正确的数据类型转换节点，使得 prev_node_c
    转换为 node_a 期望的数据类型，最终结果为：

                          dtype_cast
                        /
    ... -> prev_node_c -> node_c -> ...

    例如，如果 node_c 是一个 int8 操作节点，而 node_a 是一个 fp32 操作节点，
    此函数将插入一个反量化操作。
    """

    # 初始化数据类型转换操作相关变量
    dtype_cast_op = None
    dtype_cast_mod_cls = None
    dtype_cast_method = None
    dtype_cast_method_dtype = None
    dtype_cast_scale = None
    dtype_cast_zero_point = None

    # 获取 node_a 的输入和输出数据类型
    node_input_type_a, _node_output_type_a = \
        get_node_first_input_and_output_type(
            node_a, gm_a, logger_cls, node_type_to_io_type_map)

    # 获取 node_c 的输入和输出数据类型
    node_input_type_c, _node_output_type_c = \
        get_node_first_input_and_output_type(
            node_c, gm_b, logger_cls, node_type_to_io_type_map)
    # 检查节点输入类型，确定需要进行的数据类型转换操作
    if (
        (node_input_type_a == NodeInputOrOutputType.FP32 and
         node_input_type_c == NodeInputOrOutputType.INT8) or
        (node_input_type_a == NodeInputOrOutputType.FP32 and
         node_input_type_c == NodeInputOrOutputType.FP16) or
        # TODO(future PR): determine the actual dtype of node_c,
        # the current code only works because dequantize works with
        # multiple input dtypes.
        (node_input_type_a == NodeInputOrOutputType.FP32 and
         node_input_type_c == NodeInputOrOutputType.FP32_OR_INT8)
    ):
        # 如果节点 A 的输入类型是 FP32，节点 C 的类型是 INT8、FP16 或 FP32_OR_INT8，
        # 使用 torch.dequantize 进行数据类型转换
        dtype_cast_op = torch.dequantize
    elif (
        node_input_type_a == node_input_type_c and
        node_input_type_a != NodeInputOrOutputType.UNKNOWN
    ):
        # 如果节点 A 和节点 C 的输入类型相同且不是 UNKNOWN 类型，
        # 则使用 torch.nn.Identity，表示数据类型保持不变
        dtype_cast_mod_cls = torch.nn.Identity
    elif (
        node_input_type_a == NodeInputOrOutputType.INT8 and
        node_input_type_c == NodeInputOrOutputType.FP32
    ):
        # 如果节点 A 的输入类型是 INT8，节点 C 的类型是 FP32，
        # 需要进行量化操作以保持数据类型的正确性
        # 获取节点 A 的输入量化参数
        node_a_input_qparams = get_node_input_qparams(
            node_a, gm_a, node_type_to_io_type_map)
        if node_a_input_qparams is not None:
            # 如果量化参数不为空，使用 torch.quantize_per_tensor 进行量化
            dtype_cast_op = torch.quantize_per_tensor  # type: ignore[assignment]
            dtype_cast_scale, dtype_cast_zero_point = node_a_input_qparams
    elif (
        node_input_type_a == NodeInputOrOutputType.FP16 and
        node_input_type_c == NodeInputOrOutputType.FP32
    ):
        # 如果节点 A 的输入类型是 FP16，节点 C 的类型是 FP32，
        # 使用 'to' 方法进行数据类型转换到 torch.float16
        dtype_cast_method = 'to'
        dtype_cast_method_dtype = torch.float16
    else:
        # 如果未匹配到任何情况，抛出断言错误，要求实现从 node_c 的类型到 node_a 的类型的数据类型转换
        raise AssertionError(
            f"dtype cast from {node_input_type_c} {node_c.format_node()} to " +
            f"{node_input_type_a} {node_a.format_node()} needs to be implemented")

    if isinstance(prev_node_c, Node):
        # 如果 prev_node_c 是 Node 类型
        # 生成一个新的 dtype cast 的名称，确保其唯一性
        new_dtype_cast_name = \
            get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        if dtype_cast_op:
            if dtype_cast_scale is not None and dtype_cast_zero_point is not None:
                # 如果存在量化的 scale 和 zero_point，调用 _insert_quantize_per_tensor_node
                return _insert_quantize_per_tensor_node(
                    prev_node_c, node_a, gm_b, graph_c, dtype_cast_scale,
                    dtype_cast_zero_point, new_dtype_cast_name)
            else:
                # 否则，创建一个调用函数节点，使用 dtype_cast_op 进行数据类型转换
                return graph_c.create_node(
                    'call_function', dtype_cast_op, (prev_node_c,), {},
                    new_dtype_cast_name)
        elif dtype_cast_method:
            # 如果存在 dtype_cast_method，创建一个调用方法节点，使用 dtype_cast_method 进行数据类型转换
            return graph_c.create_node(
                'call_method', dtype_cast_method,
                (prev_node_c, dtype_cast_method_dtype), {}, new_dtype_cast_name)
        else:
            # 否则，使用 dtype_cast_mod_cls 创建一个新的 dtype_cast 模块，并添加到 gm_b 中
            assert dtype_cast_mod_cls
            dtype_cast_mod = dtype_cast_mod_cls()
            setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
            return graph_c.create_node(
                'call_module', new_dtype_cast_name, (prev_node_c,), {},
                new_dtype_cast_name)
    # 如果 prev_node_c 是 list 类型，则进入条件分支
    elif isinstance(prev_node_c, list):
        # 初始化结果列表
        results = []
        # 遍历 prev_node_c 列表中的每个元素
        for prev_node_c_inner in prev_node_c:
            # 根据 node_name_prefix 获取带有前缀的新属性名
            new_dtype_cast_name = \
                get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
            # 如果存在 dtype_cast_op，则创建一个新的 dtype_cast 节点并添加到结果列表中
            if dtype_cast_op:
                # TODO(future PR): add handling for quantize_per_tensor
                new_dtype_cast_node = graph_c.create_node(
                    'call_function', dtype_cast_op, (prev_node_c_inner,), {},
                    new_dtype_cast_name)
                results.append(new_dtype_cast_node)
            # 否则，假设存在 dtype_cast_mod_cls
            else:
                assert dtype_cast_mod_cls
                # 实例化 dtype_cast_mod_cls 并将其设置为 gm_b 的属性
                dtype_cast_mod = dtype_cast_mod_cls()
                setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
                # 创建一个新的 dtype_cast 节点并添加到结果列表中
                new_dtype_cast_node = graph_c.create_node(
                    'call_module', new_dtype_cast_name, (prev_node_c_inner,), {},
                    new_dtype_cast_name)
                results.append(new_dtype_cast_node)
        # 返回结果列表
        return results
    # 如果 prev_node_c 的类型不在预期范围内，则引发 AssertionError
    else:
        raise AssertionError(f"type f{type(prev_node_c)} is not handled")
# TODO(future PR): look into using copy_node API instead
# 定义一个函数 `_copy_node_from_a_to_c`，从图模块 `gm_a` 复制节点 `node_a` 到图 `graph_c` 中，并返回复制后的节点。
def _copy_node_from_a_to_c(
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    graph_c: Graph,
) -> Node:
    """
    Simple copy of node_a to graph_c.
    """
    # 如果节点操作为 'get_attr'
    if node_a.op == 'get_attr':
        # 创建一个新的属性名，确保唯一性
        node_a_copy_name = \
            get_new_attr_name_with_prefix(node_a.name + '_shadow_copy_')(gm_b)
        # 获取节点 `node_a` 的属性对象
        node_a_obj = getattr_from_fqn(gm_a, node_a.target)  # type: ignore[arg-type]
        # 如果属性对象是张量，则进行分离操作
        if torch.is_tensor(node_a_obj):
            node_a_obj = node_a_obj.detach()
        # 将属性对象设置到 `gm_b` 中
        setattr(gm_b, node_a_copy_name, node_a_obj)
        # 在图 `graph_c` 中创建节点 `node_a_copy`，表示复制的节点
        node_a_copy = graph_c.create_node(
            node_a.op, node_a_copy_name, (), {}, node_a_copy_name)
        return node_a_copy
    # 如果节点操作为 'call_method'
    elif node_a.op == 'call_method':
        # 断言节点目标为 'dequantize' 或 'to'
        assert node_a.target in ('dequantize', 'to'), \
            f"target {node_a.target} is not implemented"
        # 如果是 'dequantize' 操作
        if node_a.target == 'dequantize':
            # 递归复制节点 `node_a` 的第一个标准化输入参数
            arg_copy = _copy_node_from_a_to_c(
                get_normalized_nth_input(node_a, gm_a, 0),
                gm_a, gm_b, graph_c)  # type: ignore[arg-type]
            # 创建一个新的属性名，确保唯一性
            node_a_copy_name = \
                get_new_attr_name_with_prefix(node_a.name + '_shadow_copy_')(gm_b)
            # 在图 `graph_c` 中创建节点 `node_a_copy`，表示复制的节点
            node_a_copy = graph_c.create_node(
                node_a.op, node_a.target, (arg_copy,), {}, node_a_copy_name)
            return node_a_copy
        else:  # 如果是 'to' 操作
            # 递归复制节点 `node_a` 的第一个和第二个标准化输入参数
            arg_copy = _copy_node_from_a_to_c(
                get_normalized_nth_input(node_a, gm_a, 0), gm_a, gm_b, graph_c)  # type: ignore[arg-type]
            # 创建一个新的属性名，确保唯一性
            node_a_copy_name = \
                get_new_attr_name_with_prefix(node_a.name + '_shadow_copy_')(gm_b)
            # 在图 `graph_c` 中创建节点 `node_a_copy`，表示复制的节点
            node_a_copy = graph_c.create_node(
                node_a.op, node_a.target,
                (arg_copy, get_normalized_nth_input(node_a, gm_a, 1)),
                {}, node_a_copy_name)
            return node_a_copy

    else:
        # 抛出断言错误，表明未实现节点操作 `node_a.op` 的处理
        raise AssertionError(
            f"handling of node {node_a.format_node()} with op {node_a.op} is not implemented")

# 定义一个函数 `_can_insert_copy_of_subgraph_a`，判断是否可以将子图 `subgraph_a` 复制插入到图模块 `gm_a` 中
def _can_insert_copy_of_subgraph_a(
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    num_non_param_args_node_a: int,
) -> bool:
    """
    This function returns `False` if the input subgraph cannot be copied by
    `_insert_copy_of_subgraph_a_after_input_node_c`. This usually means
    that there is a corner case logic for which copy is not yet implemented.
    """
    # 构建需要检查的节点列表
    nodes = []
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        nodes.append(cur_node)
        # 获取节点 `cur_node` 的第一个标准化输入参数，并进行类型忽略
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)  # type: ignore[assignment]
    nodes.append(cur_node)
    # 反转节点列表，确保从起始节点到结束节点的顺序
    nodes.reverse()
    # 定义一个内部函数 `_can_insert`，用于检查是否可以插入节点
    def _can_insert(node_a_arg, gm_a):
        # 如果 `node_a_arg` 是 `Node` 类型
        if isinstance(node_a_arg, Node):
            # 调用辅助函数 `return_first_non_observer_node`，获取非观察节点
            arg_a = return_first_non_observer_node(node_a_arg, gm_a)
            # 如果节点操作为 'call_method'，检查目标是否为 'dequantize' 或 'to'
            if arg_a.op == 'call_method':
                return arg_a.target in ('dequantize', 'to')
            # 如果节点操作为 'get_attr'，直接返回 True
            elif arg_a.op == 'get_attr':
                return True
            else:
                return False
        # 如果 `node_a_arg` 是列表或元组类型
        elif isinstance(node_a_arg, (list, tuple)):
            # 遍历列表或元组中的每个元素 `el`
            for el in node_a_arg:
                # 如果元素不是 `Node` 类型，则返回 False
                if not isinstance(el, Node):
                    return False
        # 其他情况下返回 True
        return True

    # 对每个节点 `node_a` 进行处理，检查是否处理复制行为，按照 `_insert_copy_of_subgraph_a_after_input_node_c` 中的逻辑
    for node_a in nodes:

        # 计算非参数节点数目，如果是第一个节点，则使用 `num_non_param_args_node_a`，否则为 1
        local_num_non_param_args_node_a = num_non_param_args_node_a \
            if node_a is nodes[0] else 1

        # 获取节点 `node_a` 的规范化参数和关键字参数，仅使用关键字参数进行规范化
        norm_args_kwargs = node_a.normalized_arguments(
            gm_a, normalize_to_only_use_kwargs=True)
        # 如果规范化后的参数和关键字参数不为 None，则分别赋值给 `norm_args` 和 `norm_kwargs`
        if norm_args_kwargs is not None:
            norm_args, norm_kwargs = norm_args_kwargs
        # 否则直接使用节点 `node_a` 的 args 和 kwargs
        else:
            norm_args, norm_kwargs = node_a.args, node_a.kwargs

        # 初始化当前索引 `cur_idx`
        cur_idx = 0

        # 遍历规范化后的参数 `norm_args`
        while cur_idx < len(norm_args):
            # 如果当前索引 `cur_idx` 为 0，则跳过
            if cur_idx == 0:
                pass
            # 如果当前索引 `cur_idx` 为 1，并且 `local_num_non_param_args_node_a` 为 2，则跳过
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            else:
                # 调用 `_can_insert` 函数，检查是否可以插入 `norm_args[cur_idx]`
                if not _can_insert(norm_args[cur_idx], gm_a):
                    return False
            # 增加当前索引 `cur_idx`
            cur_idx += 1

        # 遍历规范化后的关键字参数 `norm_kwargs` 的值
        for kwarg_val in norm_kwargs.values():
            # 如果当前索引 `cur_idx` 为 0，则跳过
            if cur_idx == 0:
                pass
            # 如果当前索引 `cur_idx` 为 1，并且 `local_num_non_param_args_node_a` 为 2，则跳过
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            else:
                # 调用 `_can_insert` 函数，检查是否可以插入关键字参数的值 `kwarg_val`
                if not _can_insert(kwarg_val, gm_a):
                    return False
            # 增加当前索引 `cur_idx`
            cur_idx += 1

    # 所有节点处理完成，返回 True
    return True
def _insert_copy_of_subgraph_a_after_input_node_c(
    input_node_c: Union[Node, List[Node]],
    input_node_c_2: Optional[Union[Node, List[Node]]],
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    TODO(before land): real docblock
    """
    # Determine the graph_c to which nodes will be added
    if isinstance(input_node_c, Node):
        graph_c = input_node_c.graph
    else:
        assert isinstance(input_node_c, list)
        graph_c = input_node_c[0].graph

    # Create a sequential list of nodes from subgraph_a, from end to start
    nodes_of_a = [subgraph_a.end_node]
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        # Traverse through the graph_a to get the next node in subgraph_a
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)  # type: ignore[assignment]
        nodes_of_a.insert(0, cur_node)

    # Insert nodes_of_a into graph_c sequentially
    cur_node_a = nodes_of_a[0]
    cur_node_c = _insert_copy_of_node_a_after_input_node_c(
        input_node_c,
        input_node_c_2,
        cur_node_a,
        gm_a,
        gm_b,
        node_name_prefix)
    for cur_idx_a in range(1, len(nodes_of_a)):
        cur_node_a = nodes_of_a[cur_idx_a]
        prev_node_c = cur_node_c  # The last added node is the input for the next node
        cur_node_c = _insert_copy_of_node_a_after_input_node_c(
            prev_node_c,
            # TODO(future PR): enable multiple inputs for nodes which are not at start of subgraph
            None,
            cur_node_a,
            gm_a,
            gm_b,
            node_name_prefix)
    # Return the last inserted node into graph_c
    return cur_node_c


def _insert_copy_of_node_a_after_input_node_c(
    input_node_c: Union[Node, List[Node]],
    input_node_c_2: Optional[Union[Node, List[Node]]],
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    Assume that node_a from graph_a has
      args (input, (input2)?, arg1, ...), and
      kwargs {kw0: kwarg0, ...}

    Note: input2 is optional. If it equals to None, we assume that the op
    has a single non-param input.  If it is specified, we assume that the op
    has two non-param inputs.

    Copies the underlying values of arg1..argn and kwarg0..kwargn into gm_b,
    and creates the corresponding nodes in graph_c. Note: observers are ignored,
    so if an arg is an observer we navigate up until we find a non-observer parent.

    If node_a is a call_module, points the module pointed to by node_a to gm_b.

    Creates the copy of node_a in graph_c, with input as the first arg,
    and all other args and kwargs pointing to the copies of the objects
    in gm_b created above.

    An example in pictures:

    graph A:
    ========

    input -------------> node_a
                         / / /
    (input_2)?----------/ / /
                         / /
    weight -> weight_obs  /
                         /

    """
    bias ----------------

    graph C (derived from B):
    =========================

    input_node_c --> node_a_copy
                     / / /
    (input_node_c_2)? / /
                     / /
    weight_copy ----/ /
                     /
    bias_copy ------/
    """
    # 检查输入节点 input_node_c 的类型，确定要使用的图 graph_c
    if isinstance(input_node_c, Node):
        graph_c = input_node_c.graph
    else:
        assert isinstance(input_node_c, list)
        graph_c = input_node_c[0].graph

    # 获取节点 node_a 的规范化参数，仅使用关键字参数进行规范化
    norm_args_kwargs = node_a.normalized_arguments(
        gm_a, normalize_to_only_use_kwargs=True)
    if norm_args_kwargs is not None:
        norm_args, norm_kwargs = norm_args_kwargs
    else:
        norm_args, norm_kwargs = node_a.args, node_a.kwargs

    new_args = []  # 存储新的参数列表
    new_kwargs = {}  # 存储新的关键字参数字典

    def _copy_arg(arg):
        # 复制来自另一个图的其他输入节点
        if isinstance(arg, Node):
            arg = return_first_non_observer_node(arg, gm_a)
            arg = _copy_node_from_a_to_c(arg, gm_a, gm_b, graph_c)
            return arg
        elif isinstance(arg, (int, float, torch.dtype)):
            return arg
        elif isinstance(kwarg_val, (list, tuple)):
            for el in kwarg_val:
                assert not isinstance(el, Node), \
                    "handling of Node inside list is not implemented"
            return arg
        else:
            raise AssertionError(
                f"handling for kwarg of type {type(kwarg_val)} is not implemented")

    cur_idx = 0  # 当前参数索引

    # 处理规范化参数列表
    while cur_idx < len(norm_args):
        if cur_idx == 0:
            new_arg = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_arg = input_node_c_2
        else:
            new_arg = _copy_arg(norm_args[cur_idx])
        new_args.append(new_arg)
        cur_idx += 1

    # 处理规范化关键字参数
    for kwarg_name, kwarg_val in norm_kwargs.items():
        # 拼接来自基础图的输入
        if cur_idx == 0:
            new_kwargs[kwarg_name] = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_kwargs[kwarg_name] = input_node_c_2
        else:
            new_kwargs[kwarg_name] = _copy_arg(kwarg_val)
        cur_idx += 1

    new_args = tuple(new_args)  # 将新的参数列表转换为元组

    # 为节点 a 在图 C 中创建一个新的名称
    node_a_shadows_c_name = \
        get_new_attr_name_with_prefix(node_name_prefix)(gm_b)

    if node_a.op == 'call_module':
        # 如果目标是一个模块，则从 gm_a 中获取对应模块并在 gm_b 中设置副本
        new_mod_copy_name = \
            get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        setattr(gm_b, new_mod_copy_name, mod_a)
        # 在图 C 中创建节点 a 的阴影节点
        node_a_shadows_c = graph_c.create_node(
            node_a.op, new_mod_copy_name, new_args,
            new_kwargs, node_a_shadows_c_name)
        return node_a_shadows_c
    else:
        # 断言节点 A 的操作为 'call_function' 或 'call_method'
        assert node_a.op in ('call_function', 'call_method')
        # 在图 C 中创建一个新节点，操作为节点 A 的操作，目标为节点 A 的目标，
        # 新参数为 new_args，新关键字参数为 new_kwargs，节点名为 node_a_shadows_c_name
        node_a_shadows_c = graph_c.create_node(
            node_a.op, node_a.target, new_args,
            new_kwargs, node_a_shadows_c_name)
        # 返回新创建的节点 node_a_shadows_c
        return node_a_shadows_c
# 创建一个新的 GraphModule，该模块由图 C 组成，其中图 A 的有意义节点作为图 B 对应节点的影子。例如，

# 图 A:
# a0 -> op0_fp32 -> a1 -> op1_fp32 -> a2

# 图 B:
# b0 -> op0_int8 -> b1 -> op1_int8 -> b2

# 匹配的节点对: {'op0': (op0_fp32, op0_int8), 'op1': (op1_fp32, op1_int8)}

# 图 C (A 影子 B):

#    / dequant0 -> op0_fp32 -> logger_a_0  / dequant_1 -> op1_fp32 -> logger_a_1
#   /                                     /
# b0 -------------> op0_int8 -> logger_b_0 --------------> op1_int8 -> logger_b_1

# 简言之，该函数对每个节点对执行以下操作：
# * 从 gm_a 复制必要的属性和模块到 gm_b，保持名称唯一
# * 添加一个 dtype 转换操作 (dequant, quant 等)
# * 在 gm_b 的图中添加 node_a 的副本
# * 在 node_a 和 node_b 的输出上添加记录器

def create_a_shadows_b(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    matched_subgraph_pairs: Dict[str, Tuple[NSSubgraph, NSSubgraph]],
    logger_cls: Callable,
    should_log_inputs: bool,
    node_type_to_io_type_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
) -> GraphModule:
    """
    创建一个新的 GraphModule，其中包含图 C，其中图 A 的有意义节点作为图 B 对应节点的影子。
    """

    # 如果未提供 node_type_to_io_type_map，则获取默认的节点类型到 I/O 类型映射
    if node_type_to_io_type_map is None:
        node_type_to_io_type_map = get_node_type_to_io_type_map()

    # graph_c 是从复制 graph_b 的节点并插入从 graph_a 复制的影子节点创建的图
    graph_c = Graph()
    env_c: Dict[str, Any] = {}  # 环境字典，用于跟踪节点到其在 graph_c 中对应的表示
    modules = dict(gm_b.named_modules())  # gm_b 中的模块字典，用于复制模块信息和属性

    # 定义一个函数，用于加载参数，将图中的节点映射到 env_c 中对应的节点
    def load_arg(a):
        return map_arg(a, lambda node: env_c[node.name])

    # 初始化 start_node_b_to_matched_subgraph_a_and_name 和 end_node_b_to_matched_subgraph_a_and_name 字典
    # 用于存储匹配的子图 A 和子图 B 的开始和结束节点的映射关系及其相关信息
    start_node_b_to_matched_subgraph_a_and_name = {}
    end_node_b_to_matched_subgraph_a_and_name = {}

    # 遍历匹配的子图对，将子图 A 和子图 B 的信息存储到对应的映射字典中
    for match_name, match in matched_subgraph_pairs.items():
        subgraph_a, subgraph_b = match
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)
        start_node_b_to_matched_subgraph_a_and_name[subgraph_b.start_node] = \
            (subgraph_a, match_name, ref_node_type_a, ref_node_type_b)
        end_node_b_to_matched_subgraph_a_and_name[subgraph_b.end_node] = \
            (subgraph_a, match_name, ref_node_type_a, ref_node_type_b)

    # 使用 gm_b 和 graph_c 创建一个新的 GraphModule gm_c，并返回它
    gm_c = GraphModule(gm_b, graph_c)
    return gm_c
```