# `.\pytorch\torch\ao\quantization\fx\convert.py`

```
# 忽略类型检查错误的标志
# 从 typing 模块导入各种类型注解，以便在代码中使用
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
# 从 torch.ao.quantization.quant_type 模块导入 QuantType 类型
from torch.ao.quantization.quant_type import QuantType
# 导入 torch 库
import torch
# 导入 copy 模块，用于对象的深复制操作
import copy
# 导入 warnings 模块，用于发出警告
import warnings
# 从 torch.fx 模块导入 GraphModule 类
from torch.fx import (
    GraphModule,
)
# 从 torch.fx.graph 模块中导入 Graph, Node, Argument 类
from torch.fx.graph import (
    Graph,
    Node,
    Argument,
)
# 从相对路径导入一些实用工具函数
from ..utils import (
    activation_is_statically_quantized,
    weight_is_quantized,
    get_qparam_dict,
    _parent_name,
    get_swapped_custom_module_class,
)
# 从 ..qconfig 模块导入 QConfigAny 类和 qconfig_equals 函数
from ..qconfig import (
    QConfigAny,
    qconfig_equals
)
# 从 ..qconfig_mapping 模块导入 QConfigMapping 类
from ..qconfig_mapping import QConfigMapping
# 从 .qconfig_mapping_utils 模块导入一些函数
from .qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
    _compare_prepare_convert_qconfig_mappings,
    _update_qconfig_for_fusion,
    _is_qconfig_supported_by_dtype_configs,
    _update_qconfig_for_qat,
)
# 从 torch.ao.quantization.backend_config.utils 模块导入一些函数
from torch.ao.quantization.backend_config.utils import (
    get_root_module_to_quantized_reference_module,
    get_pattern_to_dtype_configs,
    get_fused_module_classes,
    get_qat_module_classes,
)
# 从 torch.ao.quantization.backend_config 模块导入 BackendConfig 类和 get_native_backend_config 函数
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
)
# 从 torch.ao.quantization.observer 模块导入 _is_activation_post_process 函数
from torch.ao.quantization.observer import _is_activation_post_process
# 从当前模块中的 graph_module 导入 _is_observed_module 和 _is_observed_standalone_module 函数
from .graph_module import (
    _is_observed_module,
    _is_observed_standalone_module,
)
# 从当前模块中的 _equalize 模块导入 update_obs_for_equalization 和 convert_eq_obs 函数
from ._equalize import update_obs_for_equalization, convert_eq_obs
# 从 torch.nn.utils.parametrize 模块导入 type_before_parametrizations 函数
from torch.nn.utils.parametrize import type_before_parametrizations
# 从当前模块中的 utils 导入一些函数
from .utils import (
    _get_module,
    _is_custom_module_lstm,
    _is_custom_module_mha,
    assert_and_get_unique_device,
    get_custom_module_class_keys,
    create_getattr_from_value,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    node_arg_is_weight,
)
# 从 torch.ao.quantization.utils 模块导入一些函数
from torch.ao.quantization.utils import (
    is_per_channel,
    to_underlying_dtype,
)
# 从 torch.ao.quantization.quantize 模块导入 _remove_qconfig 函数
from torch.ao.quantization.quantize import (
    _remove_qconfig,
)
# 从当前模块中的 custom_config 模块导入 ConvertCustomConfig 和 PrepareCustomConfig 类
from .custom_config import (
    ConvertCustomConfig,
    PrepareCustomConfig,
)
# 从当前模块中的 lower_to_fbgemm 模块导入 lower_to_fbgemm 函数
from .lower_to_fbgemm import lower_to_fbgemm
# 导入 _decomposed 模块，注册量化分解运算符，不做其他使用
from ._decomposed import quantized_decomposed_lib  # noqa: F401
# 导入 operator 模块
import operator

# 定义一个公开的变量列表，指定了模块导出的公共接口
__all__ = [
    "convert",
    "convert_custom_module",
    "convert_standalone_module",
    "convert_weighted_module",
]

# 支持的量化数据类型列表
SUPPORTED_QDTYPES = [
    torch.quint8,
    torch.qint8,
    torch.qint32,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
]

# 量化方案到选择量化参数操作的映射字典
_QSCHEME_TO_CHOOSE_QPARAMS_OP = {
    torch.per_tensor_affine: torch.ops.quantized_decomposed.choose_qparams.tensor,
    torch.per_tensor_symmetric: torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor,
}

# 替换观察器节点为量化-去量化节点的分解版本
def _replace_observer_with_quantize_dequantize_node_decomposed(
        model: torch.fx.GraphModule,
        node: Node,
        modules: Dict[str, torch.nn.Module],
        node_name_to_scope: Dict[str, Tuple[str, type]],
        node_name_to_qconfig: Dict[str, QConfigAny]) -> None:
    """替换 activation_post_process 模块调用节点为使用分解张量进行量化和反量化的节点

    Before:
    ... -> observer_0(x) -> ...
    After:
    ... -> torch.ops.quantized_decomposed.quantize_per_tensor(x, ...) ->
    torch.ops.quantized_decomposed.dequantize_per_tensor() -> ...

    或者使用 quantize_per_channel 和 dequantize_per_channel
    """
    # 获取模型的计算图
    graph = model.graph
    # 断言模块列表不为空
    assert modules is not None
    # 断言节点的目标是字符串类型
    assert isinstance(node.target, str)
    # 获取模块路径和前缀
    module_path, prefix = _get_module_path_and_prefix(node, node_name_to_scope, node_name_to_qconfig)
    # 获取观察器对象
    activation_post_process = modules[node.target]
    # 如果观察器对象有 'convert' 属性，则调用 convert 方法并返回
    if hasattr(activation_post_process, "convert"):
        activation_post_process.convert(model, node)
        return
    # 如果所有使用该观察器的消费者和生产者的量化配置都为 None，则跳过替换为量化/反量化节点的步骤
    skip_replacement = all(_has_none_qconfig(n, node_name_to_qconfig) for n in
                           list(node.args) + list(node.users.keys()))
    # 如果需要跳过替换或者 activation_post_process 不支持转换，则移除观察器节点
    if skip_replacement or not _is_conversion_supported(activation_post_process):
        # 在计算图中节点之前插入操作
        with graph.inserting_before(node):
            # 用节点的参数替换其所有使用的地方
            node.replace_all_uses_with(node.args[0])
            # 从计算图中擦除节点
            graph.erase_node(node)
        return

    # 否则，可以将 activation_post_process 模块调用转换为量化/反量化节点

    # 1. 从 activation_post_process 模块中提取信息，以生成量化和反量化运算符
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    # 如果 activation_post_process 有 'is_dynamic' 属性，则获取其值
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[assignment]

    # 如果 dtype 是 torch.float16，则抛出未实现的错误
    elif dtype == torch.float16:
        raise NotImplementedError("decomposed to float16 op not implemented yet")

    # 应该不会执行到这里，因为我们在开头有检查确保 activation_post_process 是受支持的
    """ Replace activation_post_process module call node with quantize and
    dequantize node

    Before:
    ... -> observer_0(x) -> ...
    After:
    ... -> torch.quantize_per_tensor(x, ...) -> x.dequantize() -> ...
    """
    # 断言确保 modules 不为空
    assert modules is not None
    # 断言确保 node.target 是字符串类型
    assert isinstance(node.target, str)
    # 获取模型的计算图
    graph = model.graph
    # 获取节点所属模块路径和前缀
    module_path, prefix = _get_module_path_and_prefix(node, node_name_to_scope, node_name_to_qconfig)
    # 获取 activation_post_process 对象，即当前节点的观察器模块
    activation_post_process = modules[node.target]
    # 如果所有与该观察器节点相关的消费者和生产者节点的 qconfig 都为 None，则跳过替换为量化/反量化节点
    skip_replacement = all(_has_none_qconfig(n, node_name_to_qconfig) for n in
                           list(node.args) + list(node.users.keys()))
    # 如果跳过替换或者 activation_post_process 不支持转换为量化操作
    if skip_replacement or not _is_conversion_supported(activation_post_process):
        # 没有找到相应的量化操作和信息，因此移除该观察器节点
        with graph.inserting_before(node):
            # 替换当前节点的所有使用处为其参数中的第一个节点
            node.replace_all_uses_with(node.args[0])
            # 从计算图中擦除该节点
            graph.erase_node(node)
        return

    # 否则，可以将 activation_post_process 模块调用转换为量化/反量化节点
    # 获取 activation_post_process 模块的数据类型
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    # 判断 activation_post_process 模块是否是动态的
    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[attr-defined, assignment]
    # 检查 dtype 是否在静态量化支持的类型列表中，并且不是动态量化
    if dtype in [torch.quint8, torch.qint8, torch.qint32, torch.float8_e5m2, torch.float8_e4m3fn] and \
            (not is_dynamic):
        # TODO: 可能需要清理此条件检查，它很难理解这个 if 语句和下面的 elif 语句

        # uint8/int8/int32 静态量化分支

        # 1. 从 activation_post_process 模块中提取信息，用于生成量化和反量化操作符
        node_type = "call_function"
        quantize_op: Optional[Callable] = None
        scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]

        # 如果是按通道量化
        if is_per_channel(activation_post_process.qscheme):  # type: ignore[attr-defined]
            ch_axis = int(activation_post_process.ch_axis)  # type: ignore[attr-defined, arg-type]
            qparams = {"_scale_": scale, "_zero_point_": zero_point, "_axis_": ch_axis, "_dtype_": dtype}
            quantize_op = torch.quantize_per_channel
        else:
            scale = float(scale)
            zero_point = int(zero_point)
            qparams = {"_scale_": scale, "_zero_point_": zero_point, "_dtype_": dtype}
            quantize_op = torch.quantize_per_tensor

        # 2. 替换 activation_post_process 节点为量化和反量化操作
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]

            # 遍历 qparams 字典中的键值对
            for key, value_or_node in qparams.items():
                # TODO: 可以在 qparams 字典本身添加值是否需要注册为属性的信息
                if key in ['_scale_', '_zero_point_']:
                    # 对于 scale 和 zero_point 值，将它们作为 buffer 注册在根模块中
                    # TODO: 或许这里需要更复杂的属性名称
                    qparam_node = create_getattr_from_value(
                        model, graph, module_path + prefix + key, value_or_node)
                    quantize_op_inputs.append(qparam_node)
                else:
                    # 对于不是 scale/zero_point 的 qparams（如 axis、dtype），将它们作为字面量存储在图中
                    quantize_op_inputs.append(value_or_node)

            # 创建量化节点并添加到图中
            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            # 创建反量化节点
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            # 用反量化节点替换原始节点的所有使用
            node.replace_all_uses_with(dequantized_node)
            # 删除原始节点
            graph.erase_node(node)
    elif is_dynamic:
        # 如果是动态量化分支

        node_type = "call_function"
        quantize_op = torch.quantize_per_tensor_dynamic
        # TODO: 从观察器中获取减少范围的信息
        # reduce_range = activation_post_process.reduce_range
        reduce_range = torch.backends.quantized.engine in ("fbgemm", "x86")
        qparams = {"_dtype_": dtype, "_reduce_range_": reduce_range}

        with graph.inserting_before(node):
            # 在节点之前插入操作
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value in qparams.items():
                quantize_op_inputs.append(value)

            # 创建量化节点
            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            # 调用反量化方法
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            # 替换所有使用该节点的地方为反量化节点
            node.replace_all_uses_with(dequantized_node)
            # 删除原始节点
            graph.erase_node(node)
    elif dtype == torch.float16:
        # 如果数据类型是 torch.float16

        node_type = "call_method"
        quantize_op = "to"  # type: ignore[assignment]
        qparams = {"_dtype_": dtype}
        with graph.inserting_before(node):
            # 在节点之前插入操作
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value in qparams.items():
                # TODO: 可以添加值是否需要在 qparams 字典中注册为属性的信息
                quantize_op_inputs.append(value)

            # 创建量化节点
            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            # 调用反量化方法
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            # 替换所有使用该节点的地方为反量化节点
            node.replace_all_uses_with(dequantized_node)
            # 删除原始节点

            graph.erase_node(node)

    # 不应该到达这里，因为我们在开始时已经确保 activation_post_process 得到支持
# this is a temporary hack for custom module, we may want to implement
# this properly after the custom module class design is finalized
# TODO: DeQuantStubs are currently inserted only after custom module LSTM, while observers are inserted
# after all other custom modules. In the future, we should simply insert QuantStubs before and DeQuantStubs
# after custom modules in general, and replace these with "quantize" and "dequantize" nodes respectively.
def _replace_observer_or_dequant_stub_with_dequantize_node(node: Node, graph: Graph) -> None:
    # 获取自定义模块调用节点
    call_custom_module_node = node.args[0]
    assert isinstance(call_custom_module_node, Node), \
        f"Expecting the for call custom module node to be a Node, but got {call_custom_module_node}"
    # 用自定义模块调用节点替换当前节点的所有使用
    node.replace_all_uses_with(call_custom_module_node)
    # 从图中删除当前节点
    graph.erase_node(node)
    # 在自定义模块调用节点前插入反量化节点
    _insert_dequantize_node(call_custom_module_node, graph)

def _is_conversion_supported(activation_post_process: torch.nn.Module) -> bool:
    # 获取激活后处理对象的数据类型
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        # 检查激活后处理对象是否为动态量化
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[attr-defined, assignment]

    # 检查是否支持当前的量化数据类型和是否为动态量化或者为浮点数16位
    return (
        (dtype in SUPPORTED_QDTYPES and (not is_dynamic)) or  # type: ignore[return-value]
        is_dynamic or
        dtype == torch.float16
    )

def _has_none_qconfig(node: Argument, node_name_to_qconfig: Dict[str, QConfigAny]) -> bool:
    """ Check if a node has a qconfig of None, i.e. user requested to not quantize
    the node
    """
    # 检查节点是否具有空的量化配置，即用户请求不对该节点进行量化
    return isinstance(node, Node) and node.name in node_name_to_qconfig and node_name_to_qconfig[node.name] is None

def _run_weight_observers(observed: GraphModule, backend_config: BackendConfig) -> None:
    """ Extract the subgraph that produces the weight for dynamic quant
    or weight only quant node and run the subgraph to observe the weight.
    Note that the observers of dynamic quant or weight only quant ops are
    run during the convert step.
    """
    # 遍历观察模块的所有节点
    for node in observed.graph.nodes:
        if node.op != "call_function":
            continue
        # 遍历节点的所有参数
        for node_arg in node.args:
            # 如果参数为权重
            if node_arg and node_arg_is_weight(node, node_arg):
                # 收集生成节点的权重观察者节点
                weight_observer_nodes = collect_producer_nodes(node_arg)
                if weight_observer_nodes is None:
                    continue
                # 从生成节点创建图模块
                weight_observer_module = \
                    graph_module_from_producer_nodes(
                        observed, weight_observer_nodes)
                # 运行权重观察者
                weight_observer_module()

def _maybe_recursive_remove_dequantize(arg: Any, node: Node, graph: Graph) -> None:
    """ If the arg is a dequantize Node, or a list/tuple/dict of dequantize Node,
    we'll recursively remove the dequantize Node
    """
    # 如果参数是反量化节点，或者是包含反量化节点的列表/元组/字典，则递归删除反量化节点
    # 检查参数是否为 Node 类型，并且操作为 "call_method"，目标方法为 "dequantize"
    if isinstance(arg, Node) and \
       arg.op == "call_method" and \
       arg.target == "dequantize":
        # 获取 quantize_node，它是 dequantize 方法的第一个参数
        quantize_node = arg.args[0]
        # 仅替换特定使用的 dequantize 方法，因为其他节点可能也使用了 dequantize 方法
        node.replace_input_with(arg, quantize_node)
    
    # 如果参数是 list 或者 tuple 类型，则递归处理每个元素
    elif isinstance(arg, (list, tuple)):
        for arg_element in arg:
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    
    # 如果参数是 dict 类型，则递归处理每个值
    elif isinstance(arg, dict):
        for arg_element in arg.values():
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    
    # 如果参数类型不支持，则发出警告，指明不支持的节点类型
    else:
        warnings.warn(f"Unsupported node type in recursive remove dequantize: {type(arg)}")
# 给定观察节点，根据节点名称到作用域的映射和节点名称到量化配置的映射，返回模块路径和前缀。
# 当观察节点是 F.linear 操作的输入时（而不是其他量化操作的输出），返回前缀 "_input"。
# TODO: 此逻辑有些笨拙，应考虑如何移除或通用化。
def _get_module_path_and_prefix(
        obs_node: Node,
        node_name_to_scope: Dict[str, Tuple[str, type]],
        node_name_to_qconfig: Dict[str, QConfigAny]) -> Tuple[str, str]:
    """ Given and observer node, get the `Scope` or the fully qualified name for
    the submodule containing the observed node, also return a prefix of "_input"
    when the observed node is an input of a F.linear op, and not the output of another
    quantized op.
    TODO: this logic is hacky, we should think about how to remove it or make it more
    general
    """
    # 获取被观察节点
    observed_node = obs_node.args[0]
    
    # 确定观察器是因为被观察节点是下一个运算符的输入而插入的标志
    assert isinstance(observed_node, Node), \
        f"Expecting observed node to be a Node, but got {observed_node}"
    is_input_observer_only = node_name_to_qconfig[observed_node.name] is None \
        if observed_node.name in node_name_to_qconfig else None
    
    # 如果是因为输入运算符而插入的观察器
    if is_input_observer_only:
        # 找到使用观察节点的第一个用户来获取路径，如果用户列表中有线性调用函数，则返回第一个线性节点的完全限定名
        users = list(obs_node.users)
        first_linear_use_or_first_use = users[0] if users else None
        linear_node = None
        for n in users:
            if n.op == "call_function" and n.target == torch.nn.functional.linear:
                linear_node = n
                break
        if linear_node:
            first_linear_use_or_first_use = linear_node
        prefix = "_input"
    else:
        # 如果量化函数在操作的输出处，使用观察输入节点来获取路径
        first_linear_use_or_first_use = observed_node
        prefix = ""

    # 如果第一个线性使用节点或第一个使用节点存在，并且存在于节点名称到作用域的映射中
    if first_linear_use_or_first_use and first_linear_use_or_first_use.name in node_name_to_scope:
        module_path, _ = node_name_to_scope[first_linear_use_or_first_use.name]
    else:
        # TODO: 它没有被使用，所以我们可以跳过量化，但这需要改变量化节点的返回类型
        # 如果需要，我们稍后可以修复它
        module_path = ""
    return module_path, prefix

# 在图中为节点插入反量化节点
def _insert_dequantize_node(
        node: Node,
        graph: Graph) -> None:
    """ Inserts dequantize node for `node` in `graph`
    """
    with graph.inserting_after(node):
        dequantize_node = graph.call_method("dequantize", (node,))
        # 替换除了 dequantize 节点以外的所有用户节点的输入
        for user_node in dict(node.users):
            if user_node is not dequantize_node:
                user_node.replace_input_with(node, dequantize_node)

# 获取节点可能的观察器
def _maybe_get_observer_for_node(
        node: Node,
        modules: Dict[str, torch.nn.Module]
) -> Optional[torch.nn.Module]:
    """
    """
    # 对于给定节点，如果它被观察到（即有观察者），返回观察者实例；否则返回 None。
    """
    # 遍历节点的所有用户（使用该节点作为输入的其他节点）
    for maybe_obs_node in node.users.keys():
        # 检查用户节点的操作类型是否为 'call_module'
        if maybe_obs_node.op == 'call_module':
            # 从模块字典中获取目标节点的字符串表示，即模块的名称
            maybe_obs = modules[str(maybe_obs_node.target)]
            # 检查该模块是否为激活后处理模块
            if _is_activation_post_process(maybe_obs):
                # 如果是激活后处理模块，则返回该模块实例
                return maybe_obs
    # 如果没有找到符合条件的观察者模块，返回 None
    return None
def convert_standalone_module(
        node: Node,
        modules: Dict[str, torch.nn.Module],
        model: torch.fx.GraphModule,
        is_reference: bool,
        backend_config: Optional[BackendConfig]) -> None:
    """ Converts a observed standalone module to a quantized standalone module by calling
    the fx convert api, currently using the same `is_reference` flag as parent, but we may
    changing this behavior in the future (e.g. separating quantization and lowering for
    standalone module as well)

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - model: original model
      - is_reference: a flag from parent provided by user to decide if we want to
        produce a reference model or a fbgemm/qnnpack model
      - backend_config: backend configuration of the target backend of quantization
    """
    # TODO: remove is_reference flag

    # Determine the appropriate conversion function based on the is_reference flag
    if is_reference:
        convert_fn = torch.ao.quantization.quantize_fx.convert_to_reference_fx
    else:
        convert_fn = torch.ao.quantization.quantize_fx.convert_fx  # type: ignore[attr-defined]

    # Access the observed standalone module from the modules dictionary
    observed_standalone_module: GraphModule = modules[str(node.target)]  # type: ignore[assignment]

    # Retrieve the input quantized indices for the standalone module
    sm_input_quantized_idxs = \
        observed_standalone_module \
        .meta["_observed_graph_module_attrs"].standalone_module_input_quantized_idxs

    # Remove dequantize nodes for inputs in the node arguments
    args = list(node.args)
    for idx in range(len(args)):
        if idx in sm_input_quantized_idxs:
            arg = args[idx]
            if arg.op == "call_method" and arg.target == "dequantize":  # type: ignore[union-attr]
                quantize_node = arg.args[0]  # type: ignore[union-attr]
                node.replace_input_with(arg, quantize_node)
                if len(arg.users) == 0:  # type: ignore[union-attr]
                    model.graph.erase_node(arg)

    # Determine if a dequantize node should be added for the output
    sm_output_quantized_idxs = \
        observed_standalone_module \
        .meta["_observed_graph_module_attrs"].standalone_module_output_quantized_idxs
    if len(sm_output_quantized_idxs) > 0:
        assert sm_output_quantized_idxs[0] == 0, "Currently only quantized output idxs = [0] is supported"

        # Insert a dequantize node after the current node if output is kept quantized
        _insert_dequantize_node(node, model.graph)

    # TODO: allow convert_custom_config to override backend_config
    # for standalone module

    # Perform the conversion using the determined conversion function
    quantized_standalone_module = convert_fn(
        observed_standalone_module,
        backend_config=backend_config)

    # Update the original model's modules dictionary with the quantized module
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, quantized_standalone_module)
    modules[str(node.target)] = quantized_standalone_module
# 将带权重的模块转换为模型中的参考量化模块
# 如果 QAT 模块的 QConfig 未设置，则该模块仍将转换为浮点模块。
def convert_weighted_module(
        node: Node,
        modules: Dict[str, torch.nn.Module],
        observed_node_names: Set[str],
        node_name_to_qconfig: Dict[str, QConfigAny],
        backend_config: BackendConfig,
        is_decomposed: bool = False,
        is_reference: bool = False,
) -> None:
    """ Convert a weighted module to reference quantized module in the model
    If the QConfig of a QAT module is not set, the module will still be converted to
    a float module.

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - observed_node_names: names for the set of observed fx node, we can skip
        this conversion if the node is not observed
    """
    # 获取原始模块
    original_module = modules[str(node.target)]
    # 获取模块的 QConfig
    qconfig: QConfigAny = original_module.qconfig  # type: ignore[assignment]
    # 初始化权重后处理为 None
    weight_post_process = None
    # 获取 QAT 模块类
    qat_module_classes = get_qat_module_classes(backend_config)

    # 如果原始模块是 QAT 模块类的实例
    if isinstance(
            original_module,
            qat_module_classes):
        # 将 QAT 模块转换为浮点模块，并将其设置为原始模块
        weight_post_process = original_module.weight_fake_quant
        original_module = original_module.to_float()  # type: ignore[operator]
        # 将 QAT 模块设置为浮点模块
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, original_module)

    # 检查当前节点是否在观察节点集合中
    is_observed = node.name in observed_node_names
    # 如果此节点没有定义 QConfig，或者节点名称到 QConfig 的映射中有 None，或者节点不在观察节点集合中，则返回
    if qconfig is None or _has_none_qconfig(node, node_name_to_qconfig) or not is_observed:
        return

    # 获取后端配置中特定类型模块的模式到数据类型配置映射
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs = pattern_to_dtype_configs.get(type(original_module), [])
    # 如果 QConfig 不受数据类型配置支持，则返回
    if not _is_qconfig_supported_by_dtype_configs(qconfig, dtype_configs):
        return

    # TODO: 将 weight_is_statically_quantized 重命名为 weight_is_int8_quantized
    # 检查权重是否量化
    is_weight_quantized = weight_is_quantized(qconfig)

    # 如果权重没有被量化，则返回
    if not is_weight_quantized:
        return

    # 初始化融合模块为 None，浮点模块为原始模块
    fused_module = None
    float_module = original_module
    # 如果原始模块是融合模块的实例，则分离出浮点模块和融合模块
    if isinstance(original_module, torch.ao.nn.intrinsic._FusedModule):
        fused_module = float_module
        float_module = fused_module[0]  # type: ignore[index]

    # TODO: 将此内容移动到参考量化模块中
    # 初始化权重 QParams 或权重 QParams 字典
    wq_or_wq_dict = {"is_decomposed": is_decomposed}
    # 如果 float_module 是 torch.nn.RNNCellBase 类型
    if isinstance(float_module, torch.nn.RNNCellBase):
        # 使用量化配置获取权重后处理器对象，并应用于权重参数
        weight_post_process_ih = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_hh = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_ih(float_module.weight_ih)
        weight_post_process_hh(float_module.weight_hh)
        
        # 获取权重参数的量化参数字典
        weight_qparams_ih = get_qparam_dict(weight_post_process_ih)
        weight_qparams_hh = get_qparam_dict(weight_post_process_hh)
        
        # 更新权重量化参数字典
        wq_or_wq_dict.update({
            "weight_ih": weight_qparams_ih,
            "weight_hh": weight_qparams_hh,
        })
    
    # 如果 float_module 是 torch.nn.LSTM 或 torch.nn.GRU 类型
    elif isinstance(float_module, (torch.nn.LSTM, torch.nn.GRU)):
        # 遍历所有扁平化权重的名称
        for wn in float_module._flat_weights_names:
            # 检查 float_module 是否具有该属性并且属性名以 "weight" 开头
            if hasattr(float_module, wn) and wn.startswith("weight"):
                # 获取权重对象
                weight = getattr(float_module, wn)
                
                # 使用量化配置获取权重后处理器对象
                weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]
                
                # 如果权重后处理器的数据类型是 torch.qint8
                if weight_post_process.dtype == torch.qint8:  # type: ignore[union-attr]
                    weight_post_process(weight)  # type: ignore[operator, misc]
                
                # 获取权重后处理器对象的量化参数字典
                wq_or_wq_dict[wn] = get_qparam_dict(weight_post_process)
    else:
        # 如果 weight_post_process 为 None，说明原始模块不是量化感知训练（QAT）模块
        # 在这种情况下，我们需要从 qconfig 中获取 weight_post_process
        is_ptq = weight_post_process is None
        
        # 如果是 PTQ 模式，则从 qconfig 中获取 weight_post_process，并确保它位于正确的设备上
        if is_ptq:
            weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]
            device = assert_and_get_unique_device(float_module)
            if device:
                weight_post_process.to(device)

        # 至少调用一次权重观察器或伪量化器，以确保尺度和零点的形状正确
        # 注意：有两种情况下我们不需要在此处调用它们：
        #
        # (1) QAT：模型的 forward 方法已经调用了权重观察器或伪量化器，
        #     并且这通常发生在训练过程中，因此我们不需要在此处再次调用。
        #
        # (2) 非参考（低化）情况：量化模块的 from_float 方法已经调用了权重观察器或伪量化器，
        #     因此我们也不需要在此处再次调用。
        #
        # 目前我们忽略了这两种情况，无论如何都在此处调用权重观察器或伪量化器，
        # 这在技术上是不正确的。对于（1），这主要是为了在测试代码中保留 BC（向后兼容性），
        # 可能并不总是在转换之前进行训练。未来，我们应该为这两种情况中断 BC。
        # 参见 https://github.com/pytorch/pytorch/issues/73941。
        #
        # 然而，对于 PT2，我们不需要在这里保留 BC，所以可以跳过这个 hack 对于 QAT。
        # 我们将这种情况标识为（is_decomposed + is_reference + is_qat）。
        # 注意，尽管如此，对于 PTQ 在 PT2 流程中我们仍然需要它，
        # 因为模型的 forward 方法不会调用权重观察器。
        is_qat = not is_ptq
        
        # 如果不是（分解 + 参考 + QAT）的情况，则调用权重观察器或伪量化器
        if not (is_decomposed and is_reference and is_qat):
            weight_post_process(float_module.weight)  # type: ignore[operator]

        # 更新权重量化或权重量化字典
        wq_or_wq_dict.update(get_qparam_dict(weight_post_process))

    # 对于所有量化模式（静态、动态、仅权重），我们使用相同的参考模块：根（浮点）模块类到量化参考模块类的映射
    # root_module_to_quantized_reference_module：从根（浮点）模块类到量化参考模块类的映射，例如 nn.Conv2d 到 nn.quantized._reference.Conv2d
    root_module_to_quantized_reference_module = get_root_module_to_quantized_reference_module(backend_config)
    
    # 获取相应的量化参考模块类
    ref_qmodule_cls = root_module_to_quantized_reference_module.get(type_before_parametrizations(float_module), None)
    
    # 断言确保找到了对应的参考量化模块类
    assert (
        ref_qmodule_cls is not None
    ), f"No reference quantized module class configured for {type_before_parametrizations(float_module)}"
    
    # 使用浮点模块和权重量化或权重量化字典创建参考量化模块
    ref_qmodule = ref_qmodule_cls.from_float(float_module, wq_or_wq_dict)  # type: ignore[attr-defined]
    
    # 如果融合模块不为 None，则更新它为参考量化模块
    if fused_module is not None:
        fused_module[0] = ref_qmodule  # type: ignore[operator]
    else:
        # 否则，更新模块的属性以指向参考量化模块
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, ref_qmodule)
def _remove_previous_dequantize_in_custom_module(node: Node, prev_node: Node, graph: Graph) -> None:
    """
    Given a custom module `node`, if the previous node is a dequantize, reroute the custom as follows:

    Before: quantize - dequantize - custom_module
    After: quantize - custom_module
                 \\ - dequantize
    """
    # 确保前一个节点是 Node 类型，用于自定义模块节点
    assert isinstance(prev_node, Node), \
        f"Expecting the argument for custom module node to be a Node, but got {prev_node}"
    
    # 如果前一个节点是 dequantize 操作
    if prev_node.op == "call_method" and prev_node.target == "dequantize":
        # 将 custom_module 的输入替换为 dequantize 的输入
        node.replace_input_with(prev_node, prev_node.args[0])
        
        # 如果 dequantize 节点没有其他使用者，则移除该节点
        if len(prev_node.users) == 0:
            graph.erase_node(prev_node)

def convert_custom_module(
        node: Node,
        graph: Graph,
        modules: Dict[str, torch.nn.Module],
        custom_module_class_mapping: Dict[QuantType, Dict[Type, Type]],
        statically_quantized_custom_module_nodes: Set[Node]) -> None:
    """ Converts an observed custom module to a quantized custom module based on
    `custom_module_class_mapping`
    For static quantization, we'll also remove the previous `dequantize` node and
    attach the observer node for output to the module, the observer for the node
    will be converted to a dequantize node instead of quantize-dequantize pairs
    later in the graph. In the end we would have a quantized custom module that
    has the same interface as a default quantized module in nn.quantized namespace,
    i.e. quantized input and quantized output.

    Args:
      - node: The call_module node of the observed standalone module
      - graph: The graph containing the node
      - modules: named_module of original model
      - custom_module_class_mapping: mapping from observed custom module class to
        quantized custom module class, used to swap custom modules
      - statically_quantized_custom_module_nodes: we'll add the custom module node
        if we find it is statically quantized, this will be used later when converting
        observers to quant/dequant node pairs, if the observed node is a statically
        quantized custom module nodes, we'll convert the observer to a dequantize node,
        this is to keep the interface the same as the default quantized module.
        TODO: maybe we want to redesign this part to align with reference model design
        as well, but there has been some discussions around the interface, so we can do
        it later.
    """
    # 获取观察到的自定义模块
    observed_custom_module = modules[str(node.target)]
    
    # 获取节点的观察器（如果存在）
    maybe_obs = _maybe_get_observer_for_node(node, modules)
    
    # 获取观察到的自定义模块的量化配置
    qconfig = observed_custom_module.qconfig
    # 检查激活是否静态量化，如果是，则处理静态量化的自定义模块节点
    if activation_is_statically_quantized(qconfig):
        # 将节点添加到静态量化的自定义模块节点集合中
        statically_quantized_custom_module_nodes.add(node)
        
        # 如果当前节点是自定义模块 LSTM
        if _is_custom_module_lstm(node, modules):
            # 输入应该是元组形式 (input, (hidden0, hidden1))
            # 确保所有三个输入节点都已经量化
            assert (
                len(node.args) == 2 and
                isinstance(node.args[1], tuple) and
                len(node.args[1]) == 2
            )
            # 解包输入节点
            (inputs, (hidden0, hidden1)) = node.args  # type: ignore[misc]
            assert isinstance(inputs, Node)
            assert isinstance(hidden0, Node)
            assert isinstance(hidden1, Node)
            # 移除自定义模块中先前的去量化操作
            _remove_previous_dequantize_in_custom_module(node, inputs, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden0, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden1, graph)
        
        # 如果当前节点是自定义模块 MHA (Multihead Attention)
        elif _is_custom_module_mha(node, modules):
            # 输入应该是 (query, key, value) 的形式
            # TODO: 这是启用 MultiheadAttention 的完整量化路径的第一步，目前仅覆盖模块的输入部分。
            # 对于输出部分，类似于 LSTM 自定义模块，还需添加额外的处理。
            assert len(node.args) == 3
            query, key, value = node.args
            assert isinstance(query, Node)
            assert isinstance(key, Node)
            assert isinstance(value, Node)
            # 移除自定义模块中先前的去量化操作
            _remove_previous_dequantize_in_custom_module(node, query, graph)
            _remove_previous_dequantize_in_custom_module(node, key, graph)
            _remove_previous_dequantize_in_custom_module(node, value, graph)
        
        # 否则处理其他自定义模块，移除先前的去量化节点并处理激活后处理
        else:
            # 确保输入节点已经量化，移除之前的去量化节点
            arg = node.args[0]
            assert isinstance(arg, Node)
            _remove_previous_dequantize_in_custom_module(node, arg, graph)
            # 将后续的观察器合并到模块转换中
            activation_post_process = _maybe_get_observer_for_node(node, modules)
            assert activation_post_process is not None
            observed_custom_module.activation_post_process = activation_post_process

    # 将观察的自定义模块交换为量化的自定义模块
    quantized_custom_module_class = get_swapped_custom_module_class(
        observed_custom_module, custom_module_class_mapping, qconfig)
    quantized_custom_module = \
        quantized_custom_module_class.from_observed(observed_custom_module)
    # 获取父级名称和名称
    parent_name, name = _parent_name(node.target)
    # 设置模块的属性为量化后的自定义模块
    setattr(modules[parent_name], name, quantized_custom_module)
# 定义一个函数，用于将观察模型（包含观察器调用的模块）转换为参考量化模型。
def convert(
        model: GraphModule, is_reference: bool = False,
        convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
        is_standalone_module: bool = False,
        _remove_qconfig_flag: bool = True,
        qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None,
        backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
        is_decomposed: bool = False) -> GraphModule:
    """
    We will convert an observed model (a module with observer calls) to a reference
    quantized model, the rule is simple:
    1. for each observer module call in the graph, we'll convert it to calls to
       quantize and dequantize functions based on the observer instance
    2. for weighted operations like linear/conv, we need to convert them to reference
       quantized module, this requires us to know whether the dtype configured for the
       weight is supported in the backend, this is done in prepare step and the result
       is stored in observed_node_names, we can decide whether we need to swap the
       module based on this set

    Args:
       * `is_standalone_module`: when this flag is True, it means we are quantizing
       a submodule that is not inlined in parent module, and will be quantized
       separately as one unit.

       * `is_decomposed`: a boolean flag to indicate whether we want to use the
        quantize operator for decomposed quantized tensor
        (torch.ops.quantized_decomposed.quantize_per_tensor) or default/standalone
        quantized tensor (torch.quantize_per_tensor)

    Returns:
         a quantized standalone module, whether input/output is quantized is
         specified by prepare_custom_config, with
         input_quantized_idxs, output_quantized_idxs, please
         see docs for :func:`~torch.ao.quantization.prepare_fx` for details
    """

    # 如果没有提供转换定制配置，使用默认的ConvertCustomConfig对象
    if convert_custom_config is None:
        convert_custom_config = ConvertCustomConfig()

    # 如果传入的convert_custom_config是字典类型，发出警告并转换为ConvertCustomConfig对象
    if isinstance(convert_custom_config, dict):
        warnings.warn(
            "Passing a convert_custom_config_dict to convert is deprecated and will not be supported "
            "in a future version. Please pass in a ConvertCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config)

    # 如果传入的qconfig_mapping是字典类型，发出警告并转换为QConfigMapping对象
    if isinstance(qconfig_mapping, dict):
        warnings.warn(
            "Passing a QConfig dictionary to convert is deprecated and will not be supported "
            "in a future version. Please pass in a QConfigMapping instead.",
            FutureWarning,
            stacklevel=2,
        )
        qconfig_mapping = QConfigMapping.from_dict(qconfig_mapping) if qconfig_mapping else None
    
    # 深度拷贝qconfig_mapping以确保不会在后续处理中修改原始对象
    qconfig_mapping = copy.deepcopy(qconfig_mapping)
    # 确保qconfig_mapping为None或者是QConfigMapping类型的实例
    assert qconfig_mapping is None or isinstance(qconfig_mapping, QConfigMapping)
    # 如果 backend_config 是一个字典类型，则发出警告信息，表明传递 backend_config_dict 给 prepare 函数已经过时，
    # 在未来版本中将不再支持。建议传入 BackendConfig 对象。
    if isinstance(backend_config, dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        # 从字典创建 BackendConfig 对象
        backend_config = BackendConfig.from_dict(backend_config)

    # 如果 backend_config 为 None，则调用 get_native_backend_config 函数获取默认的 backend_config
    if backend_config is None:
        backend_config = get_native_backend_config()

    # 确保传入的 model 是通过 prepare_fx 函数生成的观察模块
    assert _is_observed_module(model), \
        'incoming model must be produced by prepare_fx'

    # 获取 model 对应的 "_observed_graph_module_attrs" 元数据
    observed_graph_module_attrs = model.meta["_observed_graph_module_attrs"]

    # 从 observed_graph_module_attrs 中获取相关属性
    node_name_to_scope: Dict[str, Tuple[str, type]] = observed_graph_module_attrs.node_name_to_scope
    prepare_custom_config: PrepareCustomConfig = observed_graph_module_attrs.prepare_custom_config
    observed_node_names: Set[str] = observed_graph_module_attrs.observed_node_names
    node_name_to_qconfig: Dict[str, QConfigAny] = observed_graph_module_attrs.node_name_to_qconfig  # type: ignore[assignment]

    # 创建一个字典，将 model 中每个模块的完全限定模块名映射到模块实例
    # 例如：
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    # 这里使用 remove_duplicate=False 是因为 torch.cat 使用相同的 activation_post_process 模块实例但有不同的名称
    modules = dict(model.named_modules(remove_duplicate=False))

    # TODO 当我们更新 prepare 逻辑以获取有关哪些图节点已被观察并与 convert 共享信息时，重构此代码。
    # 目前暂时未完善的计划。
    # 如果存在 qconfig_mapping，则进行以下操作
    if qconfig_mapping:
        # 获取 observed_graph_module_attrs 中的 qconfig_mapping，类型为 QConfigMapping
        prepare_qconfig_mapping: QConfigMapping = observed_graph_module_attrs.qconfig_mapping  # type: ignore[assignment]
        # 深拷贝 modules
        modules_copy = copy.deepcopy(modules)

        # 如果模型是量化训练模型，更新 qconfig_mapping 和 backend_config
        if observed_graph_module_attrs.is_qat:
            _update_qconfig_for_qat(qconfig_mapping, backend_config)
        
        # 更新模型中的量化配置信息
        _update_qconfig_for_fusion(model, qconfig_mapping)

        # 比较 prepare_qconfig_mapping 和 qconfig_mapping 的转换映射
        _compare_prepare_convert_qconfig_mappings(prepare_qconfig_mapping, qconfig_mapping)  # type: ignore[arg-type]

        # 生成模型中节点名称到量化配置的映射 convert_node_name_to_qconfig
        convert_node_name_to_qconfig = _generate_node_name_to_qconfig(
            model, modules_copy, model.graph, qconfig_mapping, node_name_to_scope)

        # 检查生成的 convert_node_name_to_qconfig，确保所有值要么与 prepare 中的一致，要么为 None
        for k, v in node_name_to_qconfig.items():
            assert k in convert_node_name_to_qconfig, f'Expected key {k} in convert node_name_to_qconfig'
            if convert_node_name_to_qconfig[k] is not None:
                assert qconfig_equals(v, convert_node_name_to_qconfig[k]), \
                    f"Expected k {k} to have the same value in prepare and convert QConfigMappings, " \
                    f"but {v} was updated to {convert_node_name_to_qconfig[k]}"
        
        # 更新 node_name_to_qconfig
        node_name_to_qconfig = convert_node_name_to_qconfig

    # 获取自定义模块类的键列表
    custom_module_classes = get_custom_module_class_keys(convert_custom_config.observed_to_quantized_mapping)
    # 获取观察到的到量化映射的自定义模块类映射
    custom_module_class_mapping = convert_custom_config.observed_to_quantized_mapping

    # 如果存在均衡化节点名称到量化配置的映射
    if observed_graph_module_attrs.equalization_node_name_to_qconfig is not None:
        # 执行均衡化相关操作：更新观察器以进行缩放输入，缩放权重
        weight_eq_obs_dict = update_obs_for_equalization(model, modules)
        convert_eq_obs(model, modules, weight_eq_obs_dict)

    # 在顶层前向方法中始终运行权重观察器，用于动态量化操作或仅权重量化操作
    _run_weight_observers(model, backend_config)

    # 初始化图输入节点名称列表
    graph_inputs: List[str] = []
    # 遍历模型图中的节点
    for node in model.graph.nodes:
        # 如果节点操作是 'placeholder'，将节点名称添加到图输入列表中
        if node.op == 'placeholder':
            graph_inputs.append(node.name)

    # 如果用户指定了要量化的输入，覆盖输入量化的附加状态
    placeholder_node_seen_cnt = 0
    # 获取准备定制配置中的输入量化索引列表
    input_quantized_idxs: List[int] = prepare_custom_config.input_quantized_indexes
    # 获取准备定制配置中的输出量化索引列表
    output_quantized_idxs: List[int] = prepare_custom_config.output_quantized_indexes

    # 获取根模块到量化参考模块的映射
    root_module_to_quantized_reference_module = get_root_module_to_quantized_reference_module(backend_config)
    # 将映射的键转换为元组以支持 isinstance(module, tuple_of_classes)
    root_module_classes = tuple(root_module_to_quantized_reference_module.keys())
    # 获取量化训练模块类列表
    qat_module_classes = get_qat_module_classes(backend_config)
    # 获取使用后端配置获取融合模块的类集合
    fused_module_classes = get_fused_module_classes(backend_config)
    # 创建一个空的集合，用于存储静态量化的自定义模块节点
    statically_quantized_custom_module_nodes: Set[Node] = set()

    # 将模型图中与观察器转换为量化/反量化操作后的死代码删除
    model.graph.eliminate_dead_code()
    # 创建一个新的图模块对象，使用经过死代码删除的模型图
    model = GraphModule(model, model.graph)

    # TODO: 或许将这部分代码移到quantize_fx.py中
    # 如果不是参考模式，则将模型降低到fbgemm格式
    if not is_reference:
        model = lower_to_fbgemm(model, node_name_to_qconfig, node_name_to_scope)

    # TODO: 这看起来有些巧妙，我们需要检查为什么需要这个，并查看是否可以移除它
    # 移除qconfig和activation_post_process模块
    if _remove_qconfig_flag:
        _remove_qconfig(model)
    # 删除模型中所有未使用的子模块
    model.delete_all_unused_submodules()
    # 从模型的元数据中删除"_observed_graph_module_attrs"属性（如果存在的话）
    model.meta.pop("_observed_graph_module_attrs", None)
    # 返回经过处理后的模型对象
    return model
```