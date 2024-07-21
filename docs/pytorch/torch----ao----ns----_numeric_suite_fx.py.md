# `.\pytorch\torch\ao\ns\_numeric_suite_fx.py`

```
"""
This module contains tooling to compare weights and activations
across models. Example usage::

    import copy
    import torch
    import torch.ao.quantization.quantize_fx as quantize_fx
    import torch.ao.ns._numeric_suite_fx as ns

    m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1)).eval()
    mp = quantize_fx.prepare_fx(m, {'': torch.ao.quantization.default_qconfig})
    # We convert a copy because we need the original prepared model
    # to be available for comparisons, and `quantize_fx.convert_fx` is inplace.
    mq = quantize_fx.convert_fx(copy.deepcopy(mp))

    #
    # Comparing weights
    #

    # extract weight pairs
    weight_comparison = ns.extract_weights('a', mp, 'b', mq)

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        weight_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
        'sqnr')

    # weight_comparison contains the weights from `mp` and `mq` stored
    # in pairs, and can be used for further analysis.


    #
    # Comparing activations, with error propagation
    #

    # add loggers
    mp_ns, mq_ns = ns.add_loggers(
        'a', copy.deepcopy(mp),
        'b', copy.deepcopy(mq),
        ns.OutputLogger)

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_ns(datum)
    mq_ns(datum)

    # extract intermediate activations
    act_comparison = ns.extract_logger_info(
        mp_ns, mq_ns, ns.OutputLogger, 'b')

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        act_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
        'sqnr')

    # act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

    #
    # Comparing activations, without error propagation
    #

    # create shadow model
    mp_shadows_mq = ns.add_shadow_loggers(
        'a', copy.deepcopy(mp),
        'b', copy.deepcopy(mq),
        ns.OutputLogger)

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_shadows_mq(datum)

    # extract intermediate activations
    shadow_act_comparison = ns.extract_shadow_logger_info(
        mp_shadows_mq, ns.OutputLogger, 'b')

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        shadow_act_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
        'sqnr')

    # shadow_act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

"""

import collections  # 导入 collections 模块

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.ao.quantization.quantize_fx as quantize_fx  # 导入 PyTorch 的量化模块
from torch.fx import GraphModule  # 从 torch.fx 导入 GraphModule 类
from torch.fx.graph import Node  # 从 torch.fx.graph 导入 Node 类
from torch.ao.ns.fx.mappings import (
    get_base_name_to_sets_of_related_ops,  # 从 torch.ao.ns.fx.mappings 导入函数
)
from torch.ao.ns.fx.graph_matcher import (
    get_matching_subgraph_pairs,  # 从 torch.ao.ns.fx.graph_matcher 导入函数
)
    get_type_a_related_to_b,


注释：


    # 获取与类型A相关联的B的函数或方法的引用


这行代码看起来像是从某处获取了一个函数或方法的引用，其名称表明它与某种类型A相关联。
# 从权重工具模块中导入提取节点权重的函数
from .fx.weight_utils import (
    extract_weight_from_node,
)

# 从图遍历模块中导入函数，用于将日志记录器添加到模型中
from .fx.graph_passes import (
    add_loggers_to_model,
    create_a_shadows_b,
)

# 从工具模块中导入多个辅助函数
from .fx.utils import (
    rekey_logger_info_on_node_name_of_model,
    maybe_add_missing_fqns,
    get_target_type_str,
)

# 从命名空间类型模块中导入多个特定类型
from .fx.ns_types import (
    NSSingleResultValuesType,
    NSResultsType,
    NSNodeTargetType,
)

# 导入用于获取融合模式到根节点获取器的函数
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_root_node_getter

# 导入后端配置模块
from torch.ao.quantization.backend_config import BackendConfig

# 导入匹配工具模块中的函数
from torch.ao.quantization.fx.match_utils import _find_matches

# 导入图模块中的函数
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr

# 导入量化配置映射工具模块中的函数
from torch.ao.quantization.fx.qconfig_mapping_utils import _generate_node_name_to_qconfig

# 导入量化处理程序模块中的函数
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers

# 导入量化配置映射类
from torch.ao.quantization import QConfigMapping

# 从命名空间中的阴影工具模块导入多个函数和常量
from torch.ao.ns.fx.n_shadows_utils import (
    OutputProp,
    _get_dedup_subgraphs,
    SHADOW_WRAPPER_NODE_NAME_PREFIX,
    group_results_by_subgraph,
    create_results_comparison,
    print_n_shadows_summary,
    create_n_transformed_and_logged_copies_of_subgraph,
    create_add_loggers_graph,
    extract_weight_comparison,
)

# 导入多重量化配置映射类
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping

# 导入类型检查相关
from typing import Dict, Tuple, Callable, List, Optional, Set, Any, Type, TYPE_CHECKING

# 如果类型检查开启，导入更加具体的量化配置类型
if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfigAny

# 定义 RNN 返回类型为元组类型
RNNReturnType = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

class OutputLogger(nn.Module):
    """
    Base class for capturing intermediate values.
    """
    # 统计信息列表
    stats: List[torch.Tensor]
    # RNN 统计信息列表
    stats_rnn: List[RNNReturnType]

    # 将其标记为不纯的，以避免在 DCE 过程中移除其调用
    _is_impure = True

    def __init__(
        self,
        ref_node_name: str,
        prev_node_name: str,
        model_name: str,
        ref_name: str,
        prev_node_target_type: str,
        ref_node_target_type: str,
        results_type: str,
        index_within_arg: int,
        index_of_arg: int,
        fqn: Optional[str],
        qconfig_str: Optional[str] = '',
        ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个空列表，用于存储统计数据的张量
        self.stats: List[torch.Tensor] = []
        # 初始化一个空列表，用于存储统计数据的循环神经网络返回类型
        self.stats_rnn: List[RNNReturnType] = []

        # 节点名称，负责添加此记录器的节点名称
        # 注意：
        # - 如果记录节点输出，则与 prev_node_name 相同
        # - 如果记录节点输入，则是该记录器记录的节点输入的节点名称
        #
        # 示例：logger1 记录 op1 的输入，logger2 记录 op1 的输出：
        #
        #  x1 -> logger1 -> op1 -> logger2 -> x2
        #
        # 在此示例中：
        #   - logger1 的 prev_node_name 是 x1，ref_node_name 是 op1
        #   - logger2 的 prev_node_name 是 op1，ref_node_name 是 op1
        self.ref_node_name = ref_node_name
        # 记录器捕获的节点输出的节点名称
        self.prev_node_name = prev_node_name

        # 模型的名称，节点来源的模型名称
        self.model_name = model_name
        # 引用名称，用于将来自不同模型的记录器匹配到一起
        self.ref_name = ref_name
        # 节点输出目标类型的类型
        self.prev_node_target_type = prev_node_target_type
        # 负责添加此记录器的节点目标类型的类型
        self.ref_node_target_type = ref_node_target_type
        # stats 中包含的值的类型
        self.results_type = results_type
        # 此节点在输入/输出节点的参数中的索引，例如在 cat([x1, x2, x3], dim=0) 中，x2 的 index_within_arg 为 1
        self.index_within_arg = index_within_arg
        # 此节点在输入/输出节点的参数中的索引，例如在 add(x1, x2) 中，x2 的 index_of_arg 为 1
        self.index_of_arg = index_of_arg
        # 完全限定名称
        self.fqn = fqn
        # 如果在 prepare_fx 之前添加记录器，但不希望收集校准结果，只希望在 convert_fx 后收集结果，则添加一个标志来控制此记录器是否收集数据
        self.enabled = True
        # qconfig 的字符串表示
        self.qconfig_str = qconfig_str
        # 可以关闭此选项以在校准期间减少内存使用
        self.save_activations = True

    # 注意：无法注释 x 的类型，因为 TorchScript 不支持 Union 类型。
    # 定义一个前向传播方法，接受输入参数 x
    def forward(self, x):
        """
        """  # 空白的文档块，以便于文档自动生成工具处理

        # 如果未启用记录，则直接返回输入 x
        if not self.enabled:
            return x
        
        # 如果未启用保存激活状态，则直接返回输入 x
        if not self.save_activations:
            return x
        
        # 如果 x 是 torch.Tensor 类型，则将其分离后加入到统计列表中
        if isinstance(x, torch.Tensor):
            self.stats.append(x.detach())
        
        # 如果 x 是一个长度为 2 的元组，并且第二个元素也是长度为 2 的元组，
        # 则将其各项分离后加入到 RNN 统计列表中
        elif isinstance(x, tuple) and len(x) == 2 and len(x[1]) == 2:
            new_res = (x[0].detach(), (x[1][0].detach(), x[1][1].detach()))
            self.stats_rnn.append(new_res)
        
        # 返回处理后的 x
        return x

    # 定义对象的字符串表示方法
    def __repr__(self):
        # 创建一个清理后的字典，用于对象的字符串表示
        clean_dict = {
            k: v
            for k, v in self.__dict__.items()
            # 跳过 nn.Module 的键和以 '_' 开头的键
            if (k != 'training') and not k.startswith('_')
        }
        # 返回对象的字符串表示，格式为 "OutputLogger({...})"
        return f"OutputLogger({clean_dict})"
class OutputComparisonLogger(OutputLogger):
    """
    Same as OutputLogger, but also requires the original activation
    in order to calculate the comparison at calibration time
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO(future PR): make the comparison function configurable
        self.comparison_fn = torch.ao.ns.fx.utils.compute_sqnr  # 设置比较函数为计算信噪比的函数
        self.comparison_fn_name = 'sqnr'  # 比较函数名称为'sqnr'
        self.comparisons = []  # 预先计算的输出与参考值比较结果列表

    def forward(self, x, x_ref):
        """
        """  # 空白文档块以确保自动生成文档的正常运行
        if not self.enabled:
            return x  # 如果未启用，直接返回输入张量x
        assert isinstance(x, torch.Tensor), 'non-tensor inputs not yet supported'  # 断言输入x为张量
        if self.save_activations:
            # save the activation, for debugging
            self.stats.append(x.detach())  # 如果保存激活状态，将当前张量x的分离版本添加到统计列表中
        self.comparisons.append(self.comparison_fn(x, x_ref))  # 将使用比较函数计算x与参考张量x_ref的比较结果添加到比较结果列表中
        return x  # 返回输入张量x

    def __repr__(self):
        clean_dict = {
            k: v
            for k, v in self.__dict__.items()
            # skip nn.Module keys
            if (k != 'training') and not k.startswith('_')
        }
        return f"OutputComparisonLogger({clean_dict})"


class NSTracer(quantize_fx.QuantizationTracer):
    """
    Just like a regular FX quantization tracer, but treats observers and fake_quantize
    modules as leaf modules.
    """
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        """
        """  # 空白文档块以确保自动生成文档的正常运行
        if isinstance(m, torch.ao.quantization.ObserverBase):
            return True  # 如果模块是观察器基类，认为是叶子模块
        elif isinstance(m, torch.ao.quantization.FakeQuantizeBase):
            return True  # 如果模块是伪量化基类，认为是叶子模块
        return super().is_leaf_module(m, module_qualified_name)


def _extract_weights_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: List[Tuple[Node, str]],
    results: NSResultsType,
    op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]] = None,
) -> None:
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._extract_weights_one_model")
    for node, ref_name in nodes_and_names_to_instrument:
        res_type = NSSingleResultValuesType.WEIGHT.value
        extracted_weight = extract_weight_from_node(
            node, model, op_to_type_to_weight_extraction_fn)
        if extracted_weight:
            if ref_name not in results:
                results[ref_name] = {res_type: {}}
            results[ref_name][res_type][model_name] = [extracted_weight]


def _extract_weights_impl(
    model_name_a: str,
    gm_a: GraphModule,
    model_name_b: str,
    gm_b: GraphModule,
    base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
):
    """
    Helper function to extract weights from two models and store them in results.

    This function extracts weights from nodes specified in nodes_and_names_to_instrument
    for both models gm_a and gm_b, and stores them in the results dictionary.

    Parameters:
    - model_name_a: Name of the first model
    - gm_a: GraphModule object of the first model
    - model_name_b: Name of the second model
    - gm_b: GraphModule object of the second model
    - base_name_to_sets_of_related_ops: Optional dictionary mapping base names to sets of related operation types
    - unmatchable_types_map: Optional dictionary mapping base names to sets of unmatchable operation types
    """
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._extract_weights_impl")
    for node, ref_name in nodes_and_names_to_instrument:
        res_type = NSSingleResultValuesType.WEIGHT.value
        # Extract weights from the node using the provided extraction function
        extracted_weight = extract_weight_from_node(node, gm_a, base_name_to_sets_of_related_ops)
        if extracted_weight:
            if ref_name not in results:
                results[ref_name] = {res_type: {}}
            # Store the extracted weights in the results dictionary under the appropriate model name
            results[ref_name][res_type][model_name_a] = [extracted_weight]
        # Extract weights from the node using the provided extraction function for the second model
        extracted_weight_b = extract_weight_from_node(node, gm_b, base_name_to_sets_of_related_ops)
        if extracted_weight_b:
            if ref_name not in results:
                results[ref_name] = {res_type: {}}
            # Store the extracted weights in the results dictionary under the appropriate model name for the second model
            results[ref_name][res_type][model_name_b] = [extracted_weight_b]
    op_to_type_to_weight_extraction_fn: Optional[Dict[str, Dict[Callable, Callable]]] = None,


op_to_type_to_weight_extraction_fn 是一个可选的变量，类型为 Optional[Dict[str, Dict[Callable, Callable]]]，默认为 None。

它表示一个字典结构，其中：
- 键（key）是字符串（str），用于操作（operation）的标识符。
- 值（value）是另一个字典，其中：
  - 键是一个可调用对象（Callable），用于表示数据类型（type）的标识符。
  - 值是一个可调用对象（Callable），用于提取权重（weight）的函数。

这个变量用于存储特定操作和数据类型之间的权重提取函数映射关系，如果没有显式赋值，将默认为 None。
    # 记录 API 使用情况到 Torch 的内部日志
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._extract_weights_impl")
    # 获取匹配的子图对
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops,
        unmatchable_types_map)

    # 将匹配的子图对分成两个数据结构，分别对应两个模型
    nodes_and_names_to_instrument_a: List[Tuple[Node, str]] = []
    nodes_and_names_to_instrument_b: List[Tuple[Node, str]] = []
    for match_name, match in matched_subgraph_pairs.items():
        subgraph_a, subgraph_b = match
        # 将模型A中需要仪器化的节点和对应的名称加入列表
        nodes_and_names_to_instrument_a.append((subgraph_a.base_op_node, match_name))
        # 将模型B中需要仪器化的节点和对应的名称加入列表
        nodes_and_names_to_instrument_b.append((subgraph_b.base_op_node, match_name))

    # 初始化结果字典，用于存放权重比较结果
    results: NSResultsType = {}
    # 提取模型A的权重信息，并更新到结果字典中
    _extract_weights_one_model(
        model_name_a, gm_a, nodes_and_names_to_instrument_a, results,
        op_to_type_to_weight_extraction_fn)
    # 提取模型B的权重信息，并更新到结果字典中
    _extract_weights_one_model(
        model_name_b, gm_b, nodes_and_names_to_instrument_b, results,
        op_to_type_to_weight_extraction_fn)

    # 补充缺失的全限定名（Fully Qualified Name）条目
    maybe_add_missing_fqns(results)

    # 根据模型B中节点的名称重新组织日志信息
    results = rekey_logger_info_on_node_name_of_model(results, model_name_b)

    # 返回最终的权重比较结果
    return results
    # 使用 GraphModule 创建 GraphModule 对象 gm_a，同时追踪模型 model_a 的执行过程
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    
    # 获取模型 model_a 的 'node_name_to_scope' 属性，如果存在则赋值给 gm_a 的 '_node_name_to_scope'
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(model_a, 'node_name_to_scope')
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    
    # 使用 GraphModule 创建 GraphModule 对象 gm_b，同时追踪模型 model_b 的执行过程
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    
    # 获取模型 model_b 的 'node_name_to_scope' 属性，如果存在则赋值给 gm_b 的 '_node_name_to_scope'
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(model_b, 'node_name_to_scope')
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    
    # 调用 _extract_weights_impl 函数，提取模型权重信息并返回结果
    return _extract_weights_impl(
        model_name_a, gm_a, model_name_b, gm_b, base_name_to_sets_of_related_ops,
        unmatchable_types_map, op_to_type_to_weight_extraction_fn)
def _add_loggers_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument_inputs: List[Tuple[Node, str, str]],
    nodes_and_names_to_instrument_outputs: List[Tuple[Node, str, str]],
    logger_cls: Callable,
) -> nn.Module:
    # 记录API使用情况，用于性能分析和追踪
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._add_loggers_one_model")

    # 创建字典以将节点与要记录其输入的名称和类型关联起来
    node_to_instrument_inputs_to_ref_name: Dict[Node, Tuple[str, str]] = {}
    # 创建字典以将节点与要记录其输出的名称和类型关联起来
    node_to_instrument_outputs_to_ref_name: Dict[Node, Tuple[str, str]] = {}

    # 遍历输入节点和相关信息的列表，并将其存入对应的字典
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_inputs:
        node_to_instrument_inputs_to_ref_name[node] = (ref_name, ref_node_type)
    
    # 遍历输出节点和相关信息的列表，并将其存入对应的字典
    for node, ref_name, ref_node_type in nodes_and_names_to_instrument_outputs:
        node_to_instrument_outputs_to_ref_name[node] = (ref_name, ref_node_type)

    # 将记录器添加到模型中，记录器将记录给定节点的输入和输出
    model = add_loggers_to_model(
        model, node_to_instrument_inputs_to_ref_name,
        node_to_instrument_outputs_to_ref_name, logger_cls, model_name)
    
    # 返回添加了记录器的模型
    return model


def _add_loggers_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
    base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
) -> Tuple[nn.Module, nn.Module]:
    # 记录API使用情况，用于性能分析和追踪
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._add_loggers_impl")

    # 获取匹配的子图对，这些子图对应于两个模型中相匹配的操作
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b,
        base_name_to_sets_of_related_ops, unmatchable_types_map)

    # 初始化用于记录输入和输出节点信息的列表
    nodes_and_names_to_instrument_inputs_a = []
    nodes_and_names_to_instrument_inputs_b = []
    nodes_and_names_to_instrument_outputs_a = []
    nodes_and_names_to_instrument_outputs_b = []

    # 遍历匹配的子图对，将相关节点和信息添加到相应的列表中
    for match_name, (subgraph_a, subgraph_b) in matched_subgraph_pairs.items():
        # 获取子图A和B的目标类型字符串
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)

        # 如果应记录输入，则将起始节点添加到输入记录列表中
        if should_log_inputs:
            nodes_and_names_to_instrument_inputs_a.append(
                (subgraph_a.start_node, match_name, ref_node_type_a))
            nodes_and_names_to_instrument_inputs_b.append(
                (subgraph_b.start_node, match_name, ref_node_type_b))
        
        # 总是将结束节点添加到输出记录列表中，用于记录激活函数的输出等信息
        nodes_and_names_to_instrument_outputs_a.append(
            (subgraph_a.end_node, match_name, ref_node_type_a))
        nodes_and_names_to_instrument_outputs_b.append(
            (subgraph_b.end_node, match_name, ref_node_type_b))
    # 使用指定的函数将日志记录器添加到模型 A 中，并返回添加日志后的新模型 A
    new_model_a = _add_loggers_one_model(
        name_a,                   # 模型 A 的名称
        gm_a,                     # 模型 A 的计算图
        nodes_and_names_to_instrument_inputs_a,   # 需要记录输入的节点和名称的映射
        nodes_and_names_to_instrument_outputs_a,  # 需要记录输出的节点和名称的映射
        logger_cls                # 日志记录器的类
    )
    
    # 使用指定的函数将日志记录器添加到模型 B 中，并返回添加日志后的新模型 B
    new_model_b = _add_loggers_one_model(
        name_b,                   # 模型 B 的名称
        gm_b,                     # 模型 B 的计算图
        nodes_and_names_to_instrument_inputs_b,   # 需要记录输入的节点和名称的映射
        nodes_and_names_to_instrument_outputs_b,  # 需要记录输出的节点和名称的映射
        logger_cls                # 日志记录器的类
    )
    
    # 返回包含新模型 A 和新模型 B 的元组
    return (new_model_a, new_model_b)
# 定义函数 add_loggers，用于给两个模型添加日志记录器
def add_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs : bool = False,
    base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    Instrument model A and model B with loggers.

    Args:
        name_a: string name of model A to use in results  模型 A 的名称，用于结果中使用
        model_a: model A  模型 A
        name_b: string name of model B to use in results  模型 B 的名称，用于结果中使用
        model_b: model B  模型 B
        logger_cls: class of Logger to use  要使用的 Logger 类
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change  可选参数，子图基本节点的覆盖，可能会更改
        unmatchable_types_map: optional override of unmatchable types, subject to change  可选参数，无法匹配类型的覆盖，可能会更改

    Return:
        Returns a tuple of (model_a_with_loggers, model_b_with_loggers).  Modifies both models inplace.
        返回一个元组，包含已添加日志记录器的模型 A 和模型 B。会直接修改两个模型。
    """

    # 记录 API 使用情况，一次性
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.add_loggers")
    
    # TODO(future PR): expose these  TODO:（未来的 PR）：公开这些
    # 被跳过的模块名称和类的列表
    skipped_module_names: List[str] = []
    skipped_module_classes: List[Callable] = []
    
    # 创建两个 NSTracer 对象用于追踪模型 A 和模型 B
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    
    # 使用 NSTracer 对象对模型 A 和模型 B 进行追踪，创建 GraphModule 对象
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    
    # 获取模型 A 和模型 B 的被观察图模块属性 node_name_to_scope，如果存在则设置给 gm_a 和 gm_b
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(model_a, 'node_name_to_scope')
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
        
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(model_b, 'node_name_to_scope')
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    
    # 调用 _add_loggers_impl 函数，为模型 A 和模型 B 添加日志记录器
    return _add_loggers_impl(
        name_a, gm_a, name_b, gm_b, logger_cls,
        should_log_inputs=should_log_inputs,
        base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
        unmatchable_types_map=unmatchable_types_map)


# 定义函数 _extract_logger_info_one_model，用于提取单个模型的日志记录器信息
def _extract_logger_info_one_model(
    model: nn.Module,
    results: NSResultsType,
    logger_cls: Callable,
) -> None:
    # 记录 API 使用情况，一次性
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._extract_logger_info_one_model")
    # 遍历模型中命名的所有模块和它们的名称
    for gm_name, mod in model.named_modules():
        # TODO(future PR): better check when scripted
        # 检查当前模块是否是日志记录器类的实例，或者是torch.jit.RecursiveScriptModule的实例且原始名称是'OutputLogger'
        is_logger = (
            isinstance(mod, logger_cls)  # type: ignore[arg-type]
            or (
                isinstance(mod, torch.jit.RecursiveScriptModule)
                and mod.original_name == 'OutputLogger'
            )
        )
        # 如果是日志记录器
        if is_logger:
            # 获取日志记录器的引用名称
            key = mod.ref_name
            # 如果结果中不存在这个引用名称，创建一个空字典
            if key not in results:
                results[key] = {}
            # 断言当前模型名称在结果中还不存在，确保结果的唯一性
            assert mod.model_name not in results[key], \
                f"{mod.model_name} is already present in results"
            # 如果结果中对应的类型还不存在，创建一个空字典
            if mod.results_type not in results[key]:
                results[key][mod.results_type] = {}
            # 如果模型名称在结果中对应的类型里还不存在，创建一个空列表
            if mod.model_name not in results[key][mod.results_type]:
                results[key][mod.results_type][mod.model_name] = []
            # 选择要使用的统计数据，如果模块具有RNN统计数据，则使用RNN统计数据
            stats_to_use = mod.stats
            if len(mod.stats_rnn) > 0:
                stats_to_use = mod.stats_rnn
            # 构建包含模块各种属性的数据字典
            data = {
                'type': mod.results_type,
                'values': stats_to_use,
                'ref_node_name': mod.ref_node_name,
                'ref_node_target_type': mod.ref_node_target_type,
                'prev_node_name': mod.prev_node_name,
                'prev_node_target_type': mod.prev_node_target_type,
                'index_within_arg': mod.index_within_arg,
                'index_of_arg': mod.index_of_arg,
                'fqn': mod.fqn,
                'qconfig_str': mod.qconfig_str,
            }
            # 如果模块具有'comparisons'属性，将其添加到数据字典中
            if hasattr(mod, 'comparisons'):
                data['comparisons'] = mod.comparisons
                data['comparison_fn_name'] = mod.comparison_fn_name
            else:
                # 否则将空列表和空字符串添加到数据字典中
                data['comparisons'] = []
                data['comparison_fn_name'] = ''
            # 将数据字典添加到结果中的适当位置
            results[key][mod.results_type][mod.model_name].append(data)
            # 确保列表按特定规则排序
            results[key][mod.results_type][mod.model_name].sort(
                key=lambda res:
                f"{res['index_of_arg']}:{res['index_within_arg']}"
            )
# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_outputs`
# 提取日志信息的功能，用于比较模型输出部分
def extract_logger_info(
    model_a: nn.Module,
    model_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType:
    """
    Traverse all loggers in `model_a` and `model_b`, and extract the logged
    information.

    Args:
        model_a: model A
        model_b: model B
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
    # 记录 API 使用情况，仅调用一次
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.extract_logger_info")
    # 初始化结果字典
    results: NSResultsType = {}
    # 遍历模型 A 和模型 B
    for model in (model_a, model_b):
        # 调用辅助函数，提取每个模型的日志信息
        _extract_logger_info_one_model(model, results, logger_cls)
    # 填充缺失的完全限定名条目
    maybe_add_missing_fqns(results)
    # 根据模型 B 的节点名称重新组织日志信息的键
    results = rekey_logger_info_on_node_name_of_model(
        results, model_name_to_use_for_layer_names)
    # 返回提取的日志信息结果
    return results


# 添加影子日志记录器的实现函数
def _add_shadow_loggers_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
    base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    node_type_to_io_type_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
) -> nn.Module:
    # 记录 API 使用情况，仅调用一次
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx._add_shadow_loggers_impl")
    # 获取匹配的子图对
    matched_subgraph_pairs = get_matching_subgraph_pairs(
        gm_a, gm_b, base_name_to_sets_of_related_ops,
        unmatchable_types_map)
    # 创建模型 A 对模型 B 的影子版本
    gm_a_shadows_b = create_a_shadows_b(
        name_a, gm_a, name_b, gm_b, matched_subgraph_pairs, logger_cls,
        should_log_inputs=should_log_inputs,
        node_type_to_io_type_map=node_type_to_io_type_map)
    # 返回创建的影子模型
    return gm_a_shadows_b


# 添加影子日志记录器的外部函数接口
def add_shadow_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = False,
    base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    node_type_to_io_type_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
) -> nn.Module:
    """
    Instrument model A and model B with shadow loggers.
    """
    # 调用内部函数实现添加影子日志记录器的功能
    return _add_shadow_loggers_impl(
        name_a, model_a, name_b, model_b, logger_cls, should_log_inputs,
        base_name_to_sets_of_related_ops, node_type_to_io_type_map,
        unmatchable_types_map)
    Args:
        name_a: string name of model A to use in results
        model_a: model A
        name_b: string name of model B to use in results
        model_b: model B
        logger_cls: class of Logger to use
        should_log_inputs: whether to log inputs
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
    """
    # 记录使用量，这里是 Torch 的 API 使用量记录
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.add_shadow_loggers")
    
    # TODO(future PR): expose these
    # 未来的 Pull Request 中可能会公开这些信息
    skipped_module_names: List[str] = []
    skipped_module_classes: List[Callable] = []
    
    # 创建两个 NSTracer 对象用于跟踪模型 A 和模型 B
    tracer_a = NSTracer(skipped_module_names, skipped_module_classes)
    tracer_b = NSTracer(skipped_module_names, skipped_module_classes)
    
    # 使用 NSTracer 对象追踪模型 A 和模型 B 的计算图
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    
    # 如果模型 A 中有观察到的图模块属性 'node_name_to_scope'，则将其赋给 gm_a
    maybe_model_a_node_name_to_scope = _get_observed_graph_module_attr(model_a, 'node_name_to_scope')
    if maybe_model_a_node_name_to_scope is not None:
        gm_a._node_name_to_scope = maybe_model_a_node_name_to_scope
    
    # 如果模型 B 中有观察到的图模块属性 'node_name_to_scope'，则将其赋给 gm_b
    maybe_model_b_node_name_to_scope = _get_observed_graph_module_attr(model_b, 'node_name_to_scope')
    if maybe_model_b_node_name_to_scope is not None:
        gm_b._node_name_to_scope = maybe_model_b_node_name_to_scope
    
    # 调用内部函数 _add_shadow_loggers_impl 添加影子记录器，并返回结果
    return _add_shadow_loggers_impl(
        name_a, gm_a, name_b, gm_b, logger_cls,
        should_log_inputs=should_log_inputs,
        base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
        node_type_to_io_type_map=node_type_to_io_type_map,
        unmatchable_types_map=unmatchable_types_map)
# 提取影子模型中所有日志记录的信息
def extract_shadow_logger_info(
    model_a_shadows_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType:
    """
    Traverse all loggers in a shadow model, and extract the logged
    information.

    Args:
        model_a_shadows_b: shadow model，影子模型，它会记录数据
        logger_cls: class of Logger to use，用于记录日志的 Logger 类
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output，用于输出中层名称的模型名称字符串

    Return:
        NSResultsType, containing the logged comparisons
    """
    # 记录使用量统计信息
    torch._C._log_api_usage_once("quantization_api._numeric_suite_fx.extract_shadow_logger_info")
    # 创建一个默认字典作为结果容器
    results: NSResultsType = collections.defaultdict(dict)
    # 调用函数，从模型中提取日志信息，将结果存入 results 中
    _extract_logger_info_one_model(model_a_shadows_b, results, logger_cls)
    # 填补可能缺少的全限定名条目
    maybe_add_missing_fqns(results)
    # 根据模型 b 的节点名称重新组织日志信息
    results = rekey_logger_info_on_node_name_of_model(
        results, model_name_to_use_for_layer_names)
    # 返回标准字典形式的结果
    return dict(results)


# 使用比较函数扩展日志记录结果
def extend_logger_results_with_comparison(
    results: NSResultsType,
    model_name_1: str,
    model_name_2: str,
    comparison_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    comparison_name: str,
) -> None:
    """
    Compares the logged values from `model_name_2` against the corresponding
    values in `model_name_1`, using `comparison_fn`. Records the result
    in `model_name_2`'s results under `comparison_name`. Modifies `results` inplace.

    Args:
        results: the result data structure from `extract_logger_info` or
          `extract_shadow_logger_info`. 结果数据结构，来自 `extract_logger_info` 或 `extract_shadow_logger_info`
        model_name_1: string name of model 1，模型1的名称字符串
        model_name_2: string name of model 2，模型2的名称字符串
        comparison_fn: function to compare two Tensors，用于比较两个张量的函数
        comparison_name: string name of model to use for
          layer names in the output，输出中层名称的模型名称字符串
    """
    # 遍历结果字典中的值，该值为字典类型
    for results_type_to_results in results.values():
        # 遍历结果字典中的值，该值为字典类型
        for model_name_to_results in results_type_to_results.values():
            # 断言 model_name_1 在 model_name_to_results 中，否则抛出异常
            assert model_name_1 in model_name_to_results, \
                f"{model_name_1} not found in results"
            # 断言 model_name_2 在 model_name_to_results 中，否则抛出异常
            assert model_name_2 in model_name_to_results, \
                f"{model_name_2} not found in results"

            # 获取 model_name_1 和 model_name_2 对应的结果
            results_1 = model_name_to_results[model_name_1]
            results_2 = model_name_to_results[model_name_2]

            # 遍历 results_2 中的结果
            for result_2 in results_2:
                # 获取 result_2 中的索引信息
                index_within_arg_2 = result_2['index_within_arg']
                index_of_arg_2 = result_2['index_of_arg']
                # 初始化 result_1
                result_1 = None
                # 在 results_1 中查找与 result_2 对应的 result_1
                for cur_result_1 in results_1:
                    index_within_arg_1 = cur_result_1['index_within_arg']
                    index_of_arg_1 = cur_result_1['index_of_arg']
                    if (
                        (index_within_arg_1 == index_within_arg_2) and
                        (index_of_arg_1 == index_of_arg_2)
                    ):
                        result_1 = cur_result_1
                        break
                # 断言找到对应的 result_1
                assert result_1 is not None

                # 获取 result_1 和 result_2 中的值
                values_1 = result_1['values']
                values_2 = result_2['values']
                # 初始化 result_2 中的比较结果列表
                result_2[comparison_name] = []
                # 遍历 values_1 和 values_2 进行比较
                for value_1, value_2 in zip(values_1, values_2):
                    # 使用比较函数对 value_1 和 value_2 进行比较，并将结果添加到 result_2 中
                    comparison_result = comparison_fn(value_1, value_2)
                    result_2[comparison_name].append(comparison_result)
def prepare_n_shadows_model(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_multi_mapping: QConfigMultiMapping,
    backend_config: BackendConfig,
    custom_prepare_fn: Optional[Callable] = None,
    custom_prepare_kwargs: Optional[Dict[str, Any]] = None,
    custom_tracer: Any = None,
) -> GraphModule:
    """
    Given a model with a graph with M ops such as


      args_kwargs_m -> op_m -> output_m


    And a set of N qconfigs for each op, creates a new model, with
    each of the subgraph of `op_m` transformed into

    .. code::

           |---------> op_m_n -> log_m_n
           |                     /
      args_kwargs_m ---------> op_m -> log_m_0

    Where op_m_n is op_m wrapped in a submodule and transformed with
    qconfig_n, and its inner graph looks like

    .. code::

      args_m -------- op_m_prepared_with_qconfig_n -> out_m_n
                  /
      kwargs_m ---

    This is useful for testing different quantization of multiple layers in
    a single pass through the model.

    High level TODOs for future PRs:
    * figure out a better way to name the output structure
    * return a results data structure instead of printing it out
    * add examples to docblocks
    """

    # Initialize a QuantizationTracer object for tracing quantization information
    if custom_tracer is None:
        tracer = quantize_fx.QuantizationTracer([], [])
    else:
        tracer = custom_tracer
    # Convert the original model into a GraphModule using the QuantizationTracer
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    # Populate logger FQNs to ensure correct logging
    mt._node_name_to_scope = tracer.node_name_to_scope

    # Run example input propagation to determine outputs of the model
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)

    # Identify all modules in the model graph and define quantization patterns
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = \
        get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: List[str] = []
    standalone_module_classes: List[Type] = []
    custom_module_classes: List[Type] = []
    # Find and categorize potential quantization matches within the model
    matches = _find_matches(
        mt.graph, modules, patterns, root_node_getter_mapping,
        standalone_module_names, standalone_module_classes, custom_module_classes)
    # Deduplicate subgraphs that match quantization patterns
    subgraphs_dedup: Dict[str, List[Node]] = \
        _get_dedup_subgraphs(matches)

    # Generate mapping of node names to qconfigs for each qconfig set
    list_of_node_name_to_qconfig: List[Dict[str, QConfigAny]] = []
    for qconfig_mapping in qconfig_multi_mapping.qconfig_mappings_list:
        # Generate node to qconfig mapping for the current qconfig set
        node_name_to_qconfig = _generate_node_name_to_qconfig(
            mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope)
        list_of_node_name_to_qconfig.append(node_name_to_qconfig)

    # For each region in the model, prepare for quantization:
    #   For each qconfig for that region, prepare the corresponding subgraph:
    # 遍历去重后的子图集合，每个子图以索引和匹配名称为元组形式返回
    for (subgraph_idx, (match_name, nodes_in_this_subgraph)) in \
            enumerate(subgraphs_dedup.items()):
        # 对每个子图创建转换后的副本，并记录日志
        create_n_transformed_and_logged_copies_of_subgraph(
            mt, subgraph_idx, match_name, nodes_in_this_subgraph,
            qconfig_multi_mapping.qconfig_mappings_list, list_of_node_name_to_qconfig,
            custom_prepare_fn, custom_prepare_kwargs  # type: ignore[arg-type]
        )

    # 返回修改后的模型
    return mt
# TODO(future PR): we should rethink the names of all the PNP APIs
# 定义一个名为 _prepare_n_shadows_add_loggers_model 的函数，用于准备一个模型
# 该函数接受以下参数：
# - model: torch.nn.Module，需要准备的模型
# - example_inputs: Any，示例输入，用于传播
# - qconfig_mapping: QConfigMapping，量化配置映射
# - backend_config: BackendConfig，后端配置
# 返回类型为 torch.nn.Module
def _prepare_n_shadows_add_loggers_model(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_mapping: QConfigMapping,
    backend_config: BackendConfig,
) -> torch.nn.Module:
    r"""
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.

    This creates a model which provides logging for the following
    problem: if we quantize `model` with `qconfig_mapping` and feed
    the same input through both models, log the comparisons of
    corresponding intermediate layers.

    The problem is solved with a single model.  Specifically, we
    partition `model` into N subgraphs, create a copy of each relevant
    subgraph, wrap it in a module, apply the quantization API to that
    module, and hook up loggers to measure the comparisons.

    Example starting graph:

      x0 -> op0 -> x1 -> op1 -> x2

    Example config: quantize op0 to int8, do nothing to op1.
    The following graph will be created:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \                        \                           \       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_0 -> x2_1 ----> clog

    Where op0_0 is op0, op0_1 is op0 wrapped in a submodule and quantized
    to int8, op1_0 is op1 (appearing in the graph twice), log is a logger,
    and clog is a comparison logger.
    """

    # 创建一个量化追踪器对象，用于跟踪量化过程中的信息
    tracer = quantize_fx.QuantizationTracer([], [])
    # 使用量化追踪器追踪模型的图结构
    mt = torch.fx.GraphModule(model, tracer.trace(model))
    # 必要步骤，确保记录器的完全限定名称（FQN）被填充
    mt._node_name_to_scope = tracer.node_name_to_scope

    # 运行示例输入传播，需要这一步调用 prepare_fx 来处理各个子图
    output_prop = OutputProp(mt)
    output_prop.propagate(*example_inputs)

    # 查找原始图中需要考虑的子图集合
    modules = dict(mt.named_modules(remove_duplicate=False))
    patterns = _get_pattern_to_quantize_handlers(backend_config)
    root_node_getter_mapping = \
        get_fusion_pattern_to_root_node_getter(backend_config)
    standalone_module_names: List[str] = []
    standalone_module_classes: List[Type] = []
    custom_module_classes: List[Type] = []
    # 查找匹配的子图集合
    matches = _find_matches(
        mt.graph, modules, patterns, root_node_getter_mapping,
        standalone_module_names, standalone_module_classes, custom_module_classes)
    # 获取去重后的子图集合
    subgraphs_dedup: Dict[str, List[Node]] = \
        _get_dedup_subgraphs(matches)

    # 为每个子图生成节点到量化配置的映射
    node_name_to_qconfig = _generate_node_name_to_qconfig(
        mt, modules, mt.graph, qconfig_mapping, tracer.node_name_to_scope)

    # 现在，将图变异为带有记录器的图，并进行传播错误处理
    create_add_loggers_graph(
        mt, subgraphs_dedup, qconfig_mapping, node_name_to_qconfig)

    # 返回变异后的模型
    return mt
# TODO(future PR): we should rethink the names of all the PNP APIs
def _n_shadows_compare_weights(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_mapping: QConfigMapping,
    backend_config: BackendConfig,
) -> NSResultsType:
    """
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.
    """
    # 根据传入的 qconfig_mapping 创建 QConfigMultiMapping 对象
    qconfig_multi_mapping = \
        QConfigMultiMapping.from_list_qconfig_mapping([qconfig_mapping])
    # 准备 n_shadows_model，使用给定的输入样本和 QConfigMultiMapping 对象
    mp = prepare_n_shadows_model(
        model, example_inputs, qconfig_multi_mapping, backend_config)
    # 通过模型传递输入数据以填充观察权重实际值的观察器
    mp(*example_inputs)
    # 将准备好的 n_shadows_model 转换为量化模型
    mq = convert_n_shadows_model(mp)
    # 提取权重比较结果
    weight_comparison = extract_weight_comparison(mq)
    return weight_comparison

# TODO(future PR): consider aligning API signature with other similar quantization
# functions (enable_fake_quant, etc)
def loggers_set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    """
    Sets the `enabled` setting on a `model`'s loggers
    """
    # 遍历模型的所有模块，找到类型为 OutputLogger 的子模块并设置其 enabled 属性
    for name, child in model.named_modules():
        if isinstance(child, OutputLogger):
            child.enabled = enabled

# TODO(future PR): consider aligning API signature with other similar quantization
# functions (enable_fake_quant, etc)
def loggers_set_save_activations(
    model: torch.nn.Module,
    save_activations: bool,
) -> None:
    """
    Sets the `save_activations` setting on a `model`'s loggers
    """
    # 遍历模型的所有模块，找到类型为 OutputLogger 的子模块并设置其 save_activations 属性
    for name, child in model.named_modules():
        if isinstance(child, OutputLogger):
            child.save_activations = save_activations

def convert_n_shadows_model(
    model: GraphModule,
    custom_convert_fn: Optional[Callable] = None,
    custom_convert_kwargs: Optional[Dict[str, Any]] = None
) -> GraphModule:
    """
    Given a model from `prepare_n_shadows_model`, runs `convert_fx`
    on each shadow submodule.
    """
    # 遍历模型的所有节点，对每个以 SHADOW_WRAPPER_NODE_NAME_PREFIX 开头的节点进行量化转换
    for node in model.graph.nodes:
        # TODO(future PR): consider matching in a safer way than
        # node name string match
        if node.name.startswith(SHADOW_WRAPPER_NODE_NAME_PREFIX):
            orig_mod = getattr(model, node.name)
            # 如果未提供自定义转换函数，则使用默认的 convert_fx 进行转换
            if custom_convert_fn is None:
                converted_mod = torch.ao.quantization.quantize_fx.convert_fx(
                    orig_mod)
            else:
                # 否则，使用自定义的转换函数及参数进行转换
                if custom_convert_kwargs is None:
                    custom_convert_kwargs = {}
                converted_mod = custom_convert_fn(orig_mod, **custom_convert_kwargs)
            setattr(model, node.name, converted_mod)

    return model

def extract_results_n_shadows_model(model: torch.nn.Module) -> NSResultsType:
    """
    Extracts logger results from `model`.
    """
    # 初始化结果字典
    results: NSResultsType = {}
    # 提取模型中的 logger 结果信息
    _extract_logger_info_one_model(model, results, OutputLogger)
    return results

def print_comparisons_n_shadows_model(results: NSResultsType) -> None:
    # 待实现，用于打印 n_shadows_model 的比较结果
    pass
    """
    Prints a summary of extracted `results`.
    """
    # 根据子图对结果进行分组
    results_grouped = group_results_by_subgraph(results)
    # 创建结果比较的摘要
    results_comparison = create_results_comparison(results_grouped)
    # 打印阴影摘要的数量
    print_n_shadows_summary(results_comparison)
```