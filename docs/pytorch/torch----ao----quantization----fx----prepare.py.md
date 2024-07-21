# `.\pytorch\torch\ao\quantization\fx\prepare.py`

```
# 设置 mypy 选项，允许未标注类型的函数定义
mypy: allow-untyped-defs

# 导入所需的模块和类
import copy
import torch
import warnings
from torch.fx import (
    GraphModule,
)
from torch.fx.graph import (
    Graph,
    Node,
)
from torch.fx.node import Argument

# 导入量化相关的模块和函数
from ..quantize import (
    propagate_qconfig_,
)
from ..observer import (
    _is_activation_post_process,
    _PartialWrapper,
)
from ..qconfig import (
    _is_reuse_input_qconfig,
    QConfigAny,
)
from ..qconfig_mapping import (
    QConfigMapping,
)
from .qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
    _update_qconfig_for_fusion,
    _get_flattened_qconfig_dict,
    _update_qconfig_for_qat,
)

# 导入量化处理相关的函数和类
from .quantize_handler import (
    _default_root_node_getter,
    _get_pattern_to_quantize_handlers,
    QuantizeHandler,
)

# 导入量化观察器相关的类
from torch.ao.quantization import (
    ObserverBase,
    FixedQParamsObserver,
    FixedQParamsFakeQuantize,
    _DerivedObserverOrFakeQuantize,
)

# 导入量化工具函数和类
from torch.ao.quantization.utils import (
    Pattern,
    NodePattern,
)

# 导入均衡化相关的函数
from ._equalize import (
    is_equalization_observer,
    node_supports_equalization,
)

# 导入模式工具函数
from .pattern_utils import (
    _sorted_patterns_dict,
)

# 导入匹配工具函数
from .match_utils import (
    _MatchResultWithQConfig,
    _find_matches,
)

# 导入工具函数
from .utils import (
    _insert_dequant_stubs_for_custom_module_lstm_output,
    _is_custom_module_lstm,
    _maybe_get_custom_module_lstm_from_node_arg,
    _qconfig_satisfies_dtype_config_constraints,
    get_custom_module_class_keys,
    all_node_args_have_no_tensors,
    assert_and_get_unique_device,
    get_non_observable_arg_indexes_and_types,
    get_new_attr_name_with_prefix,
    node_arg_is_weight,
    node_arg_is_bias,
    NON_QUANTIZABLE_WEIGHT_OPS,
    ObservedGraphModuleAttrs,
)

# 导入量化观察器和占位符观察器
from torch.ao.quantization import (
    PlaceholderObserver
)
from torch.ao.quantization.quantize import (
    convert
)

# 导入工具函数
from ..utils import (
    _parent_name,
    get_qconfig_dtypes,
    get_swapped_custom_module_class,
)

# 导入后端配置相关函数
from ..backend_config.utils import (
    get_pattern_to_dtype_configs,
    get_module_to_qat_module,
    get_fusion_pattern_to_root_node_getter,
)
from ..backend_config import (
    BackendConfig,
    DTypeConfig,
    get_native_backend_config,
)

# 导入自定义配置相关类
from .custom_config import (
    PrepareCustomConfig,
    StandaloneModuleConfigEntry,
)

# 导入量化器相关类
from torch.ao.quantization.quantizer import (
    EdgeOrNode,
    QuantizationSpec,
    QuantizationSpecBase,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
    DerivedQuantizationSpec,
)
from torch.ao.quantization import ObserverOrFakeQuantize

# 导入 FakeTensor 类
from torch._subclasses import FakeTensor

# 导入类型提示
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict

# 定义不需要添加观察器的数据类型列表
_DO_NOT_OBS_DTYPE_LIST = [int, float, torch.bool, None]
# 定义需要添加观察器的数据类型列表
_OBS_DTYPE_LIST = [
    torch.quint8,
    torch.qint8,
    torch.qint32,
    torch.float16,
    torch.uint8,
    torch.int8,
    # 定义一个 torch 张量的数据类型：16位有符号整数
    torch.int16,
    # 定义一个 torch 张量的数据类型：32位有符号整数
    torch.int32,
    # 定义一个 torch 张量的数据类型：带有指定精度和范围的浮点数，精度为 8 位，指数为 5，尾数为 2
    torch.float8_e5m2,
    # 定义一个 torch 张量的数据类型：带有指定精度和范围的浮点数，精度为 8 位，指数为 4，尾数为 3，符号位为无符号
    torch.float8_e4m3fn,
# 默认的浮点观察器或伪量化控制器
_DEFAULT_FP32_OBS_OR_FQ_CTR = PlaceholderObserver.with_args(dtype=torch.float)

# 注意：以下默认目标数据类型信息字典是临时的，应尽快移至新的可编程 API 类中
# 默认的用于目标数据类型信息的浮点32位量化配置
_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation
}

# 默认的用于目标数据类型信息的8位量化配置
_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation
}

def _get_observer_kwargs(quant_spec: Union[QuantizationSpec, FixedQParamsQuantizationSpec]):
    # 将量化规范转换为参数字典并深度复制
    kwargs_dict = asdict(quant_spec)
    return copy.deepcopy(kwargs_dict)

def _get_qspec_for_arg(
    arg: Node,
    input_qspec_map: Dict[Node, QuantizationSpecBase],
    named_modules: Dict[str, torch.nn.Module]
) -> Optional[QuantizationSpecBase]:
    # 获取参数的量化规范，跳过后处理激活函数节点
    while _is_activation_post_process_node(arg, named_modules):
        arg = arg.args[0]  # type: ignore[assignment]
    return input_qspec_map.get(arg, None)

def _create_obs_or_fq_from_qspec(
    quantization_spec: Optional[QuantizationSpecBase],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
):
    """ 根据量化规范创建观察器或伪量化对象

    Args:
       quantization_spec: 用于存储创建观察器或伪量化器的参数
       obs_or_fq_map: 这是从边/输出到相应观察器/伪量化实例的映射，可以根据配置重用不同的边/输出
       is_qat: 是否为量化感知训练
    """
    if quantization_spec is None:
        return None
    if isinstance(quantization_spec, SharedQuantizationSpec):
        edge_or_node = quantization_spec.edge_or_node
        assert edge_or_node in obs_or_fq_map, \
            "请确保只引用已插入观察器/伪量化器的边或节点：" \
            f"'{edge_or_node}' 不在\n{obs_or_fq_map.keys()} 中"
        return obs_or_fq_map[edge_or_node]
    elif isinstance(quantization_spec, DerivedQuantizationSpec):
        # 不能使用asdict，因此这里不调用get_observer_kwargs
        kwargs = {
            "dtype": quantization_spec.dtype,
            "derive_qparams_fn": quantization_spec.derive_qparams_fn,
            "quant_min": quantization_spec.quant_min,
            "quant_max": quantization_spec.quant_max,
            "qscheme": quantization_spec.qscheme,
            "ch_axis": quantization_spec.ch_axis,
        }
        edge_or_nodes = quantization_spec.derived_from
        obs_or_fqs = [obs_or_fq_map[k] for k in edge_or_nodes]
        kwargs["obs_or_fqs"] = obs_or_fqs
        return _DerivedObserverOrFakeQuantize.with_args(**kwargs)()
    elif isinstance(quantization_spec, FixedQParamsQuantizationSpec):
        # 如果 quantization_spec 是 FixedQParamsQuantizationSpec 类型
        kwargs = _get_observer_kwargs(quantization_spec)
        # 获取观察器的参数
        observer_ctr = FixedQParamsObserver.with_args(**kwargs)
        # 创建 FixedQParamsObserver 实例
        if is_qat:
            # 如果是量化感知训练模式，返回 FixedQParamsFakeQuantize 实例
            return FixedQParamsFakeQuantize.with_args(observer=observer_ctr)()
        else:
            # 否则返回观察器实例
            return observer_ctr()

    assert isinstance(quantization_spec, QuantizationSpec)
    # 断言 quantization_spec 是 QuantizationSpec 类型
    observer_or_fake_quant_ctr = quantization_spec.observer_or_fake_quant_ctr
    # 获取 observer_or_fake_quant_ctr 属性
    kwargs = _get_observer_kwargs(quantization_spec)
    # 获取观察器的参数
    kwargs.pop("observer_or_fake_quant_ctr")
    # 移除 kwargs 中的 "observer_or_fake_quant_ctr" 键
    # we will remove is_dynamic from QuantizationSpec because
    # it seems that dynamic range quantization
    # 从 QuantizationSpec 中删除 is_dynamic，因为似乎是动态范围量化
    obs_or_fq_class = observer_or_fake_quant_ctr
    # 将 observer_or_fake_quant_ctr 赋值给 obs_or_fq_class
    if isinstance(observer_or_fake_quant_ctr, _PartialWrapper):
        # 如果 observer_or_fake_quant_ctr 是 _PartialWrapper 类型
        obs_or_fq_class = observer_or_fake_quant_ctr.p.func  # type: ignore[union-attr, assignment]
        # 获取 _PartialWrapper 的 func 属性作为 obs_or_fq_class
    if "PerChannel" not in obs_or_fq_class.__name__:  # type: ignore[operator, union-attr]
        # 如果观察器或伪量化器类名中不包含 "PerChannel"
        kwargs.pop("ch_axis")
        # 移除 kwargs 中的 "ch_axis" 键
    return observer_or_fake_quant_ctr.with_args(**kwargs)()
    # 使用 kwargs 创建 observer_or_fake_quant_ctr 实例并返回
# 定义一个函数 _needs_obs_or_fq，用于判断是否需要插入观察器或伪量化节点
# 根据上一个输出的数据类型、是否动态、当前目标的数据类型、是否动态、是否重用输入的观察器或伪量化节点以及是否是第零个参数来确定
def _needs_obs_or_fq(
        prev_output_dtype: Any,
        prev_output_is_dynamic: bool,
        cur_target_dtype: Any,
        cur_target_is_dynamic: bool,
        reuse_input_obs_or_fq: bool,
        is_zeroth_arg: bool = False) -> bool:
    """
    note: we will treat "not specified" as torch.float for now
    utility function that checks if we should insert an observer or fake quant node
    base on the requested dtype for the nodes from user

    is_zeroth_arg: we only dynamically quantize the first arg of the node right now
      this should be removed when we enable configuring dynamic quantization
      for a specific argument, this can be removed if we deprecate fx graph mode
      quantization

    """

    # 如果当前目标是动态的，则需要插入一个占位观察器，以便在转换步骤中选择参数 -> 量化 -> 动态量化
    if cur_target_is_dynamic:
        assert cur_target_dtype in _OBS_DTYPE_LIST, \
            f"Expected cur_target_dtype to be torch.float, but got: {cur_target_dtype}"
        assert prev_output_dtype not in _DO_NOT_OBS_DTYPE_LIST
        return is_zeroth_arg
    # 如果重用输入的观察器或伪量化节点，则不需要插入新的观察器或伪量化节点
    if reuse_input_obs_or_fq:
        return False
    # 对于非动态量化的情况
    # 如果当前目标数据类型在观察器的数据类型列表中
    if cur_target_dtype in _OBS_DTYPE_LIST:
        # 则需要检查前一个输出的数据类型是否在观察器数据类型列表中，同时当前目标数据类型不等于前一个输出数据类型
        return prev_output_dtype in _OBS_DTYPE_LIST + [torch.float] and cur_target_dtype != prev_output_dtype

    # 目前跳过了许多错误检查
    return False


# 定义一个函数 _is_activation_post_process_node，用于判断节点是否是激活后处理节点
def _is_activation_post_process_node(node: Node, named_modules: Dict[str, torch.nn.Module]) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "call_module" and \
        _is_activation_post_process(named_modules[str(node.target)])


# 定义一个函数 _get_dtype_and_is_dynamic，根据观察器或伪量化模块的构造函数，返回数据类型和是否动态的元组
def _get_dtype_and_is_dynamic(obs_or_fq: Optional[ObserverOrFakeQuantize]) -> Tuple[Optional[torch.dtype], bool]:
    """ Given a constructor for observer or fake quant module, returns
    a Tuple of dtype and is_dynamic
    """
    # TODO: instead of instantiating the instance, we can use inspect to get the default args
    if obs_or_fq is None:
        return None, False
    else:
        return obs_or_fq.dtype, getattr(obs_or_fq, "is_dynamic", False)  # type: ignore[return-value]


# 定义一个函数 _is_input_arg_dtype_supported_by_backend，检查配置的参数是否被后端支持
def _is_input_arg_dtype_supported_by_backend(
    arg: Argument,
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
    backend_config: BackendConfig,
) -> bool:
    """ Check if the configured qconfig for the argument
    is supported by the backend or not
    """
    if isinstance(arg, (list, tuple)):
        return all(_is_input_arg_dtype_supported_by_backend(
            a, node, qconfig,
            dtype_config, backend_config) for a in arg)
    if not isinstance(arg, Node):
        return True
    # TODO: support check for standalone module
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias
    # 如果节点需要激活，则处理激活相关逻辑
    if is_activation:
        # 从节点的元数据中获取输入激活观测或频次计数器
        input_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("input_act_obs_or_fq_ctr")
        # 如果计数器存在，则调用它获取具体值，否则为 None
        input_act_obs_or_fq = input_act_obs_or_fq_ctr() if input_act_obs_or_fq_ctr else None
        # 获取量化配置的数据类型和是否动态标志
        qconfig_dtype, qconfig_is_dynamic = _get_dtype_and_is_dynamic(input_act_obs_or_fq)
        # TODO(future PR): 在弄清楚 backend_config 在某些情况下为 None 的原因后，移除以下转换为 bool 的操作
        # 检查输入数据类型是否符合预期配置，包括动态性
        return (dtype_config.input_dtype is None) or (
            dtype_config.input_dtype == qconfig_dtype and
            bool(dtype_config.is_dynamic) == bool(qconfig_is_dynamic) and
            _qconfig_satisfies_dtype_config_constraints(qconfig, dtype_config.input_dtype_with_constraints)
        )
    # 如果节点为权重处理
    elif is_weight:
        # 从节点的元数据中获取权重观测或频次计数器
        weight_obs_or_fq_ctr = node.meta["target_dtype_info"].get("weight_obs_or_fq_ctr", None)
        # 如果计数器存在，则调用它获取具体值，否则为 None
        weight_obs_or_fq = weight_obs_or_fq_ctr() if weight_obs_or_fq_ctr else None
        # 获取量化配置的权重数据类型和是否动态标志
        qconfig_weight_dtype, _ = _get_dtype_and_is_dynamic(weight_obs_or_fq)
        # 获取后端配置的权重数据类型
        backend_config_weight_dtype = dtype_config.weight_dtype
        # 检查权重数据类型是否符合预期配置
        dtype_matches = qconfig_weight_dtype == backend_config_weight_dtype
        # 检查量化配置是否满足数据类型约束
        qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
            qconfig, dtype_config.weight_dtype_with_constraints, is_activation=False)
        return backend_config_weight_dtype is None or (dtype_matches and qconfig_satisfies_constraints)
    else:  # 处理偏置
        # 从节点的元数据中获取偏置观测或频次计数器
        bias_obs_or_fq_ctr = node.meta["target_dtype_info"].get("bias_obs_or_fq_ctr", None)
        # 如果计数器存在，则调用它获取具体值，否则为 None
        bias_obs_or_fq = bias_obs_or_fq_ctr() if bias_obs_or_fq_ctr else None
        # 获取量化配置的偏置数据类型和是否动态标志
        qconfig_bias_dtype, _ = _get_dtype_and_is_dynamic(bias_obs_or_fq)
        # 获取后端配置的偏置数据类型
        backend_config_bias_dtype = dtype_config.bias_dtype
        # 检查偏置数据类型是否符合预期配置
        return backend_config_bias_dtype is None or qconfig_bias_dtype == backend_config_bias_dtype
def _is_output_dtype_supported_by_backend(
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
) -> bool:
    """ Check if the configured qconfig for the output
    is supported by the backend or not
    """
    # 获取后端配置中的输出数据类型
    backend_config_output_dtype = dtype_config.output_dtype
    # 获取节点的目标数据类型信息，可能包含输出激活观察器或伪量化控制器
    output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("output_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR)
    # 如果有输出激活观察器或伪量化控制器，则调用它来获取实际值
    output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    # 获取 qconfig 的输出数据类型和是否为动态类型
    qconfig_output_dtype, qconfig_output_is_dynamic = _get_dtype_and_is_dynamic(output_act_obs_or_fq)
    # 如果 qconfig 指定输出为动态类型，则强制设为 torch.float32
    if qconfig_output_is_dynamic:
        qconfig_output_dtype = torch.float32
    # 检查 qconfig 的输出数据类型是否与后端配置的输出数据类型匹配
    dtype_matches = qconfig_output_dtype == backend_config_output_dtype
    # 检查 qconfig 是否满足与输出数据类型相关的约束条件
    qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
        qconfig, dtype_config.output_dtype_with_constraints)
    # 如果后端配置的输出数据类型为空，或者满足类型匹配且约束条件，则返回 True
    return backend_config_output_dtype is None or (dtype_matches and qconfig_satisfies_constraints)

def _is_observer_in_same_graph(
    node: Node,
    named_modules: Dict[str, torch.nn.Module],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat,
):
    """ Check if observer in same graph
    when the node output is not fp32 and input is 'placeholder'
    the input is assumed to be quantized, so it is observed
    in a different place rather than not observed.
    """
    # 获取节点输出的目标数据类型
    node_output_dtype = _get_arg_target_dtype_as_output(node, named_modules, obs_or_fq_map, is_qat)
    # 如果节点有参数，并且第一个参数是节点类型，并且节点输出类型为 torch.quint8 或 torch.uint8
    # 并且第一个参数是 'placeholder' 操作类型，则返回 False
    if len(node.args) > 0 and isinstance(node.args[0], Node):
        if node_output_dtype in [torch.quint8, torch.uint8] and node.args[0].op == 'placeholder':
            return False
    # 默认返回 True
    return True

def _is_pattern_dtype_config_and_qconfig_supported_by_backend(
    pattern: Optional[Pattern],
    matched_node_pattern: Optional[List[Node]],
    qconfig: QConfigAny,
    backend_config: BackendConfig,
) -> bool:
    """ Check if the dtype configuration of a pattern is supported by
    the backend or not, and whether the qconfig satisfies constraints
    specified in the corresponding dtype config.
    """
    # 如果后端配置或者模式为空，则返回 True
    if backend_config is None or pattern is None:
        return True
    # 断言：匹配的节点模式不为空并且长度至少为1
    assert matched_node_pattern is not None and len(matched_node_pattern) >= 1
    # 获取模式到数据类型配置的映射
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    # 获取模式对应的数据类型配置列表
    dtype_configs: List[DTypeConfig] = pattern_to_dtype_configs.get(pattern, [])
    # 使用后端配置获取模式到根节点获取器的映射
    pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)
    
    # 根据给定模式从映射中获取根节点获取器，如果没有则使用默认的根节点获取器
    root_node_getter = pattern_to_root_node_getter.get(pattern, _default_root_node_getter)
    
    # 使用获取到的根节点获取器获取匹配节点模式对应的根节点
    root_node = root_node_getter(matched_node_pattern)
    
    # 将根节点作为输入节点
    input_node = root_node
    
    # 将匹配节点模式的第一个节点作为输出节点
    output_node = matched_node_pattern[0]
    
    # 遍历每个数据类型配置
    for dtype_config in dtype_configs:
        # 检查输入参数的数据类型是否被后端支持
        supported = True
        for arg in list(input_node.args) + list(input_node.kwargs.values()):
            supported = supported and _is_input_arg_dtype_supported_by_backend(
                arg, input_node, qconfig, dtype_config, backend_config)
        
        # 检查输出节点的数据类型是否被后端支持
        supported = supported and _is_output_dtype_supported_by_backend(
            output_node, qconfig, dtype_config)
        
        # 如果所有数据类型都被支持，则返回True
        if supported:
            return True
    
    # 如果没有找到支持的数据类型配置，则返回False
    return False
# 返回给定节点的独立模块的 QConfigMapping 和 PrepareCustomConfig
def _get_standalone_module_configs(
    node: Node,
    named_modules: Dict[str, torch.nn.Module],
    prepare_custom_config: PrepareCustomConfig,
    parent_qconfig: QConfigAny,
    parent_backend_config: Optional[BackendConfig],
) -> Tuple[QConfigMapping, Tuple[Any, ...], PrepareCustomConfig, Optional[BackendConfig]]:
    """
    Returns the standalone module QConfigMapping and PrepareCustomConfig
    for `node`, assuming that the module pointed to by `node` is
    a standalone modules.
    """
    # 获取节点对应的模块名和模块类型
    module_name = str(node.target)
    module_type = type(named_modules[module_name])  # type: ignore[index]
    # name config 优先于 type config
    config_entry = StandaloneModuleConfigEntry(None, (), None, None)
    config_entry = prepare_custom_config.standalone_module_classes.get(module_type, config_entry)
    config_entry = prepare_custom_config.standalone_module_names.get(module_name, config_entry)
    # 如果用户未指定 qconfig 字典，则使用父模块的 qconfig
    qconfig_mapping = config_entry.qconfig_mapping or QConfigMapping().set_global(parent_qconfig)
    # 获取配置中的示例输入
    example_inputs = config_entry.example_inputs
    # 获取或创建自定义配置
    prepare_custom_config = config_entry.prepare_custom_config or PrepareCustomConfig()
    # 获取或设置后端配置
    backend_config = config_entry.backend_config or parent_backend_config
    return (qconfig_mapping, example_inputs, prepare_custom_config, backend_config)


# 将 root 中匹配到的模块转换为量化训练模块
def _qat_swap_modules(
        root: torch.nn.Module,
        module_to_qat_module: Dict[Pattern, Type[torch.nn.Module]]) -> None:
    convert(root, mapping=module_to_qat_module, inplace=True, remove_qconfig=False)


# 将匹配到的节点名称添加到集合中
def _add_matched_node_name_to_set(matched_node_pattern: NodePattern, s: Set[str]):
    if isinstance(matched_node_pattern, Node):
        s.add(matched_node_pattern.name)
    elif isinstance(matched_node_pattern, (list, tuple)):
        for maybe_node in matched_node_pattern:
            _add_matched_node_name_to_set(maybe_node, s)


# 将 ObserverOrFakeQuantize 实例附加到模型中，并创建调用节点
def _insert_obs_or_fq(
    node: Node,
    obs_or_fq: ObserverOrFakeQuantize,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Attaches `obs_or_fq` to `model`, and creates a node which calls
    `obs_or_fq` on the output of `node`.

    obs_or_fq: an instance of Observer or FakeQuantize module
    """
    # 获取模型的设备并将 obs_or_fq 移动到设备上
    model_device = assert_and_get_unique_device(model)
    if model_device:
        obs_or_fq.to(model_device)
    # 将 obs_or_fq 模块作为模型的属性添加
    if is_equalization_observer(obs_or_fq):
        prefix = node.name + '_equalization_process_'
    else:
        prefix = 'activation_post_process_'
    get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
    obs_or_fq_name = get_new_obs_or_fq_name(model)
    setattr(model, obs_or_fq_name, obs_or_fq)
    named_modules[obs_or_fq_name] = obs_or_fq
    # 在节点后插入新的观察节点到图中
    with graph.inserting_after(node):
        new_obs = graph.create_node(
            'call_module', obs_or_fq_name, (node,), {})
    return new_obs
def _set_target_dtype_info_for_matched_node_pattern(
    matched_node_pattern: NodePattern,
    last_node: Node,
    qconfig: QConfigAny,
    qhandler: Optional[QuantizeHandler],
    backend_config: BackendConfig,
    named_modules: Dict[str, torch.nn.Module],
    cache_for_no_tensor_check: Dict[Node, bool],
    processed_nodes: Set[Node],
) -> None:
    """ Sets the target_dtype_info for each node in matched_node_pattern
    Note: processed_nodes is used to ensure we only process each node once
    """
    # 如果 matched_node_pattern 是列表或元组，则逐个处理其中的 node_pattern
    if isinstance(matched_node_pattern, (list, tuple)):
        for node_pattern in matched_node_pattern:
            _set_target_dtype_info_for_matched_node_pattern(
                node_pattern,
                last_node,
                qconfig,
                qhandler,
                backend_config,
                named_modules,
                cache_for_no_tensor_check,
                processed_nodes
            )

    # 如果 matched_node_pattern 是 Node 类型，则设置其 target_dtype_info
    # 其他类型的匹配对象，如 int、float 字面量，会被忽略
    elif isinstance(matched_node_pattern, Node):
        # 强制类型断言为 Node
        assert isinstance(matched_node_pattern, Node)
        node = matched_node_pattern
        # 如果 node 已经处理过，则直接返回
        if node in processed_nodes:
            return
        # 将 node 加入已处理节点集合中
        processed_nodes.add(node)

        # 如果 qconfig 为 None，则直接返回
        if qconfig is None:
            return

        # 获取当前节点的目标激活数据类型信息
        target_dtype_info: Dict[str, Any] = (
            _get_target_activation_dtype_for_node(
                node,
                qconfig,
                qhandler,
                named_modules,
                backend_config,
                cache_for_no_tensor_check,
            )
        )
        # 将目标激活数据类型信息存储在节点的 meta 字典中的 target_dtype_info 键下
        node.meta["target_dtype_info"] = target_dtype_info

def _get_target_activation_dtype_for_node(
    node: Node,
    qconfig: QConfigAny,
    qhandler: Optional[QuantizeHandler],
    named_modules: Dict[str, torch.nn.Module],
    backend_config: BackendConfig,
    cache_for_no_tensor_check: Dict[Node, bool],
) -> Dict[str, Any]:
    """
    返回节点 op 属性中输入激活、输出激活、权重、偏置的 dtype 和 is_dynamic 设置。
    用于在参考模型表示中的 `quantize` 调用中。
    如果不需要 `quantize` 调用，则返回 None。

    例如，如果节点对应于以下结构中的 `op0`：

      x0 -> op0 -> x1

    而我们希望参考量化表示为：

      x0 -> quant_static -> dequant -> op0 -> quant_dynamic -> dequant -> x1
    """
    # 如果所有节点参数都不包含张量，则返回空观察器字典
    args_have_no_tensors = \
        all_node_args_have_no_tensors(
            node, named_modules, cache_for_no_tensor_check)
    if args_have_no_tensors:
        return {
            "input_act_obs_or_fq_ctr": None,  # 输入激活观察器或量化控制器为空
            "output_act_obs_or_fq_ctr": None,  # 输出激活观察器或量化控制器为空
        }
    # 获取量化配置以确定节点最终的数据类型
    if qconfig is not None:
        act_dtype, weight_dtype, input_act_is_dynamic = \
            get_qconfig_dtypes(qconfig)

        # 当前 `QConfig` 只有一个 `activation` 字段。
        # 对于静态量化，它被用于输入和输出激活。对于动态量化，
        # 此字段目前仅用于输入激活，输出激活为 fp32。
        # 未来随着我们向 `QConfig` 对象添加更多字段，可能会发生变化。
        output_act_dtype = act_dtype \
            if (not input_act_is_dynamic) else torch.float

        # 如果激活数据类型是 torch.float16 并且权重数据类型也是 torch.float16
        # 并且输入激活不是动态的，则偏置数据类型为 torch.float16，否则为 torch.float
        bias_dtype = torch.float16 \
            if (
                act_dtype == torch.float16
                and weight_dtype == torch.float16
                and (not input_act_is_dynamic)
            ) else torch.float

        # 检查是否是通用张量值操作
        is_general_tensor_value_op = \
            (qhandler is not None and qhandler.is_general_tensor_value_op())

        # 检查是否是独立模块
        _is_standalone_module = (
            qhandler is not None and qhandler.is_standalone_module()
        )

        weight_index = None
        # 如果节点是 Node 实例，并且操作为 "call_function"，
        # 并且目标在后端配置的复杂格式映射中，
        # 则尝试获取权重索引
        if isinstance(node, Node) and node.op == "call_function" and \
           node.target in backend_config._pattern_complex_format_to_config:
            weight_index = backend_config._pattern_complex_format_to_config[node.target]._input_type_to_index.get("weight")

        bias_index = None
        # 如果节点是 Node 实例，并且操作为 "call_function"，
        # 并且目标在后端配置的复杂格式映射中，
        # 则尝试获取偏置索引
        if isinstance(node, Node) and node.op == "call_function" and \
           node.target in backend_config._pattern_complex_format_to_config:
            bias_index = backend_config._pattern_complex_format_to_config[node.target]._input_type_to_index.get("bias")

        # 返回量化配置相关信息的字典
        return {
            "input_act_obs_or_fq_ctr": qconfig.activation,  # 输入激活观察器或量化控制器
            "weight_obs_or_fq_ctr": qconfig.weight,  # 权重观察器或量化控制器
            "bias_obs_or_fq_ctr": PlaceholderObserver.with_args(dtype=bias_dtype),  # 偏置观察器或量化控制器
            "weight_index": weight_index,  # 权重索引
            "bias_index": bias_index,  # 偏置索引
            "output_act_obs_or_fq_ctr": qconfig.activation,  # 输出激活观察器或量化控制器
            "reuse_input_obs_or_fq": _is_reuse_input_qconfig(qconfig),  # 重用输入观察器或量化控制器
            "input_output_share_observers": is_general_tensor_value_op,  # 输入输出共享观察器或量化控制器
            "_is_standalone_module": _is_standalone_module,  # 是否是独立模块
        }
    # 如果没有指定量化配置，则返回默认的 FP32 目标数据类型信息的副本
    return copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)
# 获取参数节点对应的输出激活观察器或者伪量化对象的构造器
def _get_output_act_obs_or_fq(
    arg: Node,
    named_modules: Dict[str, torch.nn.Module],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> ObserverOrFakeQuantize:
    """ Get the constructor for observer or fake quant object for
    the argument in the original graph as the output of previous node,
    skipping inserted observers

    We are assuming that the observers are inserted correctly, and the dtype for
    argument in quantized graph will match what is specified by the qconfig
    """
    assert isinstance(arg, Node)  # 断言参数 arg 是 Node 类型的对象

    # 如果参数节点的元数据中包含 "quantization_annotation" 键
    if "quantization_annotation" in arg.meta:
        # 从参数节点的量化注释中获取输出量化规格，使用它创建对应的观察器或者伪量化对象
        return _create_obs_or_fq_from_qspec(arg.meta["quantization_annotation"].output_qspec, obs_or_fq_map, is_qat)

    # 自定义模块 LSTM 的输出是一个元组，我们将其拆分成内部节点，以插入 DeQuantStubs
    custom_module_lstm_node = _maybe_get_custom_module_lstm_from_node_arg(arg, named_modules)
    output_act_obs_or_fq_ctr = None
    
    # 如果存在自定义模块 LSTM 节点
    if custom_module_lstm_node is not None:
        # 获取自定义模块 LSTM 节点的目标数据类型信息中的输出激活观察器或伪量化对象的构造器
        output_act_obs_or_fq_ctr = custom_module_lstm_node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"]
        # 如果构造器存在，使用它创建对应的激活观察器或伪量化对象
        output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    # 如果参数节点是激活后处理节点
    elif _is_activation_post_process_node(arg, named_modules):
        observed_arg = arg.args[0]  # 获取观察到的参数
        assert isinstance(observed_arg, Node), "Currently we only support observing Node"
        
        # 如果观察到的参数节点的元数据中包含 "quantization_annotation" 键
        if "quantization_annotation" in observed_arg.meta:
            # 从观察到的参数节点的量化注释中获取输出量化规格，使用它创建对应的观察器或伪量化对象
            output_act_obs_or_fq = \
                _create_obs_or_fq_from_qspec(
                    observed_arg.meta["quantization_annotation"].output_qspec, obs_or_fq_map, is_qat)
        else:
            assert "target_dtype_info" in observed_arg.meta
            # 获取观察到的参数节点的目标数据类型信息中的输出激活观察器或伪量化对象的构造器
            output_act_obs_or_fq_ctr = observed_arg.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"]
            # 如果构造器存在，使用它创建对应的激活观察器或伪量化对象
            output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    else:
        # 如果参数节点的元数据中存在 "target_dtype_info" 键
        if "target_dtype_info" in arg.meta:
            # 获取参数节点的目标数据类型信息中的输出激活观察器或伪量化对象的构造器
            output_act_obs_or_fq_ctr = \
                arg.meta["target_dtype_info"].get("output_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR)
        else:
            output_act_obs_or_fq_ctr = _DEFAULT_FP32_OBS_OR_FQ_CTR
        # 如果构造器存在，使用它创建对应的激活观察器或伪量化对象
        output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None

    return output_act_obs_or_fq
    # 调用函数 _get_output_act_obs_or_fq 处理参数 arg，得到输出的激活、观察或量化值
    arg_as_output_act_obs_or_fq = _get_output_act_obs_or_fq(arg, named_modules, obs_or_fq_map, is_qat)
    # 调用函数 _get_dtype_and_is_dynamic 处理 arg_as_output_act_obs_or_fq，获取目标数据类型和是否为动态类型
    arg_as_output_target_dtype, _ = _get_dtype_and_is_dynamic(arg_as_output_act_obs_or_fq)
    # 返回计算得到的目标数据类型
    return arg_as_output_target_dtype
def _get_arg_as_input_act_obs_or_fq(
    arg: Node,
    node: Node,
    named_modules: Dict[str, torch.nn.Module],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Optional[ObserverOrFakeQuantize]:
    """ Get the observer or fake quant constructor for the Argument `arg`, as input
    to Node `node`
    """
    # 确保参数 `arg` 是 Node 类型
    assert isinstance(arg, Node)

    # 如果节点 `node` 的元数据中包含量化注释
    if "quantization_annotation" in node.meta:
        # 从量化注释中获取输入参数节点到观察器或伪量化构造器的映射
        input_qspec_map = node.meta["quantization_annotation"].input_qspec_map
        # 获取参数 `arg` 对应的量化规范
        input_arg_qspec = _get_qspec_for_arg(arg, input_qspec_map, named_modules)
        # 如果量化规范为空，使用默认的 FP32 观察器或伪量化构造器
        if input_arg_qspec is None:
            input_arg_obs_or_fq = _DEFAULT_FP32_OBS_OR_FQ_CTR()
        else:
            # 根据量化规范创建观察器或伪量化器
            input_arg_obs_or_fq = _create_obs_or_fq_from_qspec(input_arg_qspec, obs_or_fq_map, is_qat)
        return input_arg_obs_or_fq

    # 如果节点 `node` 的元数据中没有量化注释，根据参数类型判断是否为权重、偏置或激活函数
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias

    obs_or_fq_ctr = None
    # 根据参数类型选择相应的观察器或伪量化构造器
    if is_activation:
        obs_or_fq_ctr = node.meta["target_dtype_info"].get("input_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR)
    elif is_weight:
        if node.target not in NON_QUANTIZABLE_WEIGHT_OPS:
            obs_or_fq_ctr = node.meta["target_dtype_info"].get("weight_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR)
    else:
        obs_or_fq_ctr = node.meta["target_dtype_info"].get("bias_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR)

    # 返回选定的观察器或伪量化构造器的实例，如果没有选择到则返回 None
    return obs_or_fq_ctr() if obs_or_fq_ctr else None

def _maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any],
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config: PrepareCustomConfig,
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
    backend_config: Optional[BackendConfig] = None,
) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    # 对于诸如 torch.cat([x0, x1]) 这样的操作，
    # 遍历列表进行处理
    # 如果参数 arg 是列表或元组类型，则处理每个元素
    if isinstance(arg, (list, tuple)):
        # 初始化一个空列表，用于存放处理后的每个元素
        new_arg_to_return = []
        # 遍历每个内部元素 inner_arg
        for inner_arg in arg:
            # 调用函数 _maybe_insert_input_observer_for_arg_or_kwarg 处理每个元素，并将返回结果添加到新列表中
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
                node, inner_arg, qconfig, model, named_modules,
                graph,
                qhandler,
                prepare_custom_config,
                obs_or_fq_map,
                is_qat,
                backend_config)
            new_arg_to_return.append(new_inner_arg)
        # 根据原始参数的类型构造新的对象（列表或元组），并返回处理后的结果
        return type(arg)(new_arg_to_return)

    # 如果参数 arg 不是 Node 类型，则直接返回 arg
    if not isinstance(arg, Node):
        return arg
    assert isinstance(arg, Node)
    # 默认情况下（没有观察器），将新参数设置为原始参数 arg
    new_arg = arg

    # 检查是否为独立模块
    is_standalone_module = qhandler is not None and qhandler.is_standalone_module()
    # TODO: 将此部分移动到单独的函数中

    # 如果不是独立模块，则执行以下操作
    if not is_standalone_module:
        # 注意：在此分支中，qconfig 可能为 None，因此我们现在从 node.meta 获取 act/fq
        # 对于大多数节点来说，这是常规流程，除了独立模块

        # 检查节点的 meta 数据中是否存在 "quantization_annotation"
        if "quantization_annotation" in node.meta:
            # 如果存在，获取 reuse_input_obs_or_fq 值
            reuse_input_obs_or_fq = node.meta["quantization_annotation"]._reuse_input_obs_or_fq
        else:
            assert "target_dtype_info" in node.meta
            # TODO: 在这里我们假设 "target_dtype_info" 是存在的，也许需要提供一个默认值
            target_dtype_info = node.meta["target_dtype_info"]
            # 对于没有配置 `reuse_input_obs_or_fq` 的节点，默认为 False，
            # 这使得用户可以选择是否配置此字段
            reuse_input_obs_or_fq = target_dtype_info.get("reuse_input_obs_or_fq", False)
        
        # 获取参数 arg 作为输入的激活/观察器或量化对象
        arg_as_input_act_obs_or_fq = _get_arg_as_input_act_obs_or_fq(arg, node, named_modules, obs_or_fq_map, is_qat)
        # 获取参数 arg 作为输入的目标数据类型和是否为动态类型
        arg_as_input_target_dtype, arg_as_input_target_is_dynamic = _get_dtype_and_is_dynamic(arg_as_input_act_obs_or_fq)

        # 获取参数 arg 作为输出的激活/观察器或量化对象
        arg_as_output_act_obs_or_fq = _get_output_act_obs_or_fq(arg, named_modules, obs_or_fq_map, is_qat)
        # 获取参数 arg 作为输出的目标数据类型和是否为动态类型
        arg_as_output_target_dtype, arg_as_output_target_is_dynamic = _get_dtype_and_is_dynamic(arg_as_output_act_obs_or_fq)

        # 确定是否需要激活/观察器或量化对象
        needs_obs_or_fq = _needs_obs_or_fq(
            arg_as_output_target_dtype,
            arg_as_output_target_is_dynamic,
            arg_as_input_target_dtype,
            arg_as_input_target_is_dynamic,
            reuse_input_obs_or_fq,
            is_zeroth_arg=len(node.args) > 0 and arg is node.args[0],
        )
    else:
        # 确保 qconfig 不为 None
        assert qconfig is not None
        # 对于独立模块的自定义流程
        # 获取独立模块的配置信息
        _, _, sm_prepare_custom_config, _ = \
            _get_standalone_module_configs(
                node, named_modules, prepare_custom_config, qconfig, backend_config)
        # 获取独立模块输入量化索引
        sm_input_quantized_idxs = sm_prepare_custom_config.input_quantized_indexes
    
        # 遍历当前节点的所有参数，确定当前参数的索引
        # 对于 args，设置为当前参数的索引
        # 对于 kwargs，保持为 None
        cur_input_idx = None
        for arg_idx, arg_to_check in enumerate(node.args):
            if arg_to_check is arg:
                cur_input_idx = arg_idx
                break
    
        # 如果 cur_input_idx 为 None，则不需要 observer 或者 fake quantization
        if cur_input_idx is None:
            needs_obs_or_fq = False
        else:
            # 获取当前参数作为输出目标类型的 dtype
            arg_as_output_target_dtype = _get_arg_target_dtype_as_output(arg, named_modules, obs_or_fq_map, is_qat)
            # 根据当前参数索引确定当前参数作为输入目标类型的 dtype
            arg_as_input_target_dtype = torch.quint8 if cur_input_idx in sm_input_quantized_idxs \
                else torch.float
            # 判断是否需要 observer 或者 fake quantization
            needs_obs_or_fq = (
                (arg_as_output_target_dtype != arg_as_input_target_dtype) and
                (arg_as_input_target_dtype != torch.float)
            )
    
        # 获取激活后处理函数
        act_post_process_ctr = qconfig.activation
        # 获取当前参数作为输入的激活 observer 或者 fake quantization
        arg_as_input_act_obs_or_fq = act_post_process_ctr() if act_post_process_ctr else None
    
    if needs_obs_or_fq:
    
        existing_obs_node = None
    
        # 使用新的 observer 之前，检查是否已存在正确类型的 observer
        # 如果存在，则使用已存在的 observer，避免重复插入
        # TODO: 这段代码在未来可能会被移除，目前用于确保数值在未来的使用中
        # 应该删除这段代码
        # 如果移除这段代码，意味着每个使用都插入一个 observer，即使 dtype 相同，也可能会有额外的处理来移除多余的 observers
        for maybe_obs_node in arg.users.keys():
            if maybe_obs_node.op == 'call_module':
                maybe_obs_mod = named_modules[maybe_obs_node.target]  # type: ignore[index]
                if (
                    type(maybe_obs_mod) == type(arg_as_input_act_obs_or_fq) and
                    maybe_obs_mod.dtype == arg_as_input_target_dtype  # type: ignore[possibly-undefined]
                ):
                    arg_as_input_act_obs_or_fq = maybe_obs_mod  # type: ignore[assignment]
                    existing_obs_node = maybe_obs_node
                    break
    
        # 确保当前参数作为输入的激活 observer 或者 fake quantization 不为 None
        assert arg_as_input_act_obs_or_fq is not None
        # 将当前参数与节点映射到 observer 或者 fake quantization
        obs_or_fq_map[(arg, node)] = arg_as_input_act_obs_or_fq
        # 如果不存在已存在的 observer 节点，则插入新的 observer 或者 fake quantization
        if existing_obs_node is None:
            new_obs_node = _insert_obs_or_fq(
                arg, arg_as_input_act_obs_or_fq, model, named_modules, graph)
            # 将当前参数覆盖为被观察的参数
            new_arg = new_obs_node
        else:
            # 否则，将当前参数设置为已存在的 observer 节点
            new_arg = existing_obs_node
    
    # 返回新的参数
    return new_arg
# 为给定节点 `node` 可能插入输入观察器。
def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config: PrepareCustomConfig,
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
    backend_config: Optional[BackendConfig] = None
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node

    Note: backend_config only needed for standalone_module node
    """
    # 遍历每个输入参数。如果该参数的目标数据类型与当前节点的目标数据类型不匹配，则插入观察器。
    new_args = []
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node, arg, qconfig, model, named_modules, graph,
            qhandler,
            prepare_custom_config,
            obs_or_fq_map,
            is_qat,
            backend_config)
        new_args.append(new_arg)

    new_kwargs = {}
    for k, kwarg in node.kwargs.items():
        # 对每个关键字参数进行相同的处理
        new_kwarg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node, kwarg, qconfig, model, named_modules, graph,
            qhandler,
            prepare_custom_config,
            obs_or_fq_map,
            is_qat,
            backend_config)
        new_kwargs[k] = new_kwarg

    # 将新的参数和关键字参数分配给节点，直接修改原节点
    node.args = tuple(new_args)
    node.kwargs = new_kwargs

# 为给定节点 `node` 可能插入输入均衡观察器。
def _maybe_insert_input_equalization_observers_for_node(
    node: Node,
    equalization_qconfig: Any,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    is_branch: bool,
) -> None:
    """
    If `node` needs to be equalized, find the input/weight observers it needs in
    `equalization_qconfig`, creates them, and inserts it into `graph`.

    If `node` does not need an equalization observer, returns None.
    """
    # 如果 equalization_qconfig 为 None 或者当前节点不支持均衡化，则直接返回
    if equalization_qconfig is None or not node_supports_equalization(node, named_modules):
        return

    # 如果节点处于分支中，则发出警告并返回
    if is_branch:
        warnings.warn(
            f"Cannot equalize {node} because it is part of a branch."
        )
        return

    new_args = []
    for arg in node.args:
        # 如果参数不是节点或者是偏置节点，则直接添加到新参数列表中
        if not isinstance(arg, Node) or node_arg_is_bias(node, arg):
            new_args.append(arg)
            continue

        is_weight = node_arg_is_weight(node, arg)

        # 根据是否是权重决定是使用权重均衡器还是输入激活均衡器
        act_eq_process_ctr = equalization_qconfig.weight if is_weight else \
            equalization_qconfig.input_activation

        # 创建新的均衡观察器模块并插入到图中，并用新的节点替换原参数
        new_eq_obs_mod = act_eq_process_ctr()
        new_eq_obs_node = _insert_obs_or_fq(
            arg, new_eq_obs_mod, model, named_modules, graph)

        new_args.append(new_eq_obs_node)
    # 将新的参数和关键字参数赋值给节点的 args 属性，直接修改原对象
    node.args = tuple(new_args)
def _maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Optional[Node]:
    """
    If `node` needs an output observer, creates it, inserts it into `graph`
    and returns it.

    If `node` does not need an output observer, returns None.

    Note: inserting dynamic quantization ops for output is not supported in fx graph mode
    quantization code path right now
    """
    # Assert that the node's operation is not 'output', as handling for outputs is elsewhere
    assert node.op != 'output', 'observer insertion for outputs is handled elsewhere'

    is_standalone_module = False
    # Check if there is a "quantization_annotation" in the node's metadata
    if "quantization_annotation" in node.meta:
        # Create output activation observer or fake quantize from quantization specification
        output_act_obs_or_fq = _create_obs_or_fq_from_qspec(
            node.meta["quantization_annotation"].output_qspec, obs_or_fq_map, is_qat
        )
    else:
        # If no "quantization_annotation" is present, assert that "target_dtype_info" exists
        assert "target_dtype_info" in node.meta
        # Check if the node is a standalone module based on its metadata
        is_standalone_module = node.meta["target_dtype_info"].get("_is_standalone_module", False)
        # Get the callable to create output activation observer or fake quantize
        output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("output_act_obs_or_fq_ctr")
        output_act_obs_or_fq = output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    
    # Determine the target dtype and if it is dynamic from the created output observer or fake quantize
    target_dtype, target_is_dynamic = _get_dtype_and_is_dynamic(output_act_obs_or_fq)
    
    # Uncommenting this section will enable reuse of input observer or fake quantize for output
    # reuse_input_obs_or_fq = node.meta["target_dtype_info"].get("reuse_input_obs_or_fq", False)
    # Set reuse_input_obs_or_fq to False for now due to current implementation limitations
    
    reuse_input_obs_or_fq = False

    # Check if output observer or fake quantize is needed based on target dtype and dynamic status
    needs_obs_or_fq = _needs_obs_or_fq(torch.float, False, target_dtype, target_is_dynamic, reuse_input_obs_or_fq)
    
    # Currently, activation in QConfig applies to both input and output; dynamic quantization of input
    # is supported, but not for output independently
    #
    # Limitation exists in specifying different observers for input and output activations through QConfig
    # in current API; this restriction may change in future PyTorch versions
    # 如果目标是动态的，那么不需要观察器或量化器
    # 可以改变 QConfig 以支持输入/输出激活，如果我们想要移除以下检查，或者如果我们可以弃用 fx 图模式的量化
    if target_is_dynamic:
        needs_obs_or_fq = False

    # 我们从不向独立模块的输出插入观察器，我们假设如果需要，它们已经在独立模块内部插入了
    needs_obs_or_fq = needs_obs_or_fq and \
        (not is_standalone_module)

    # 如果需要观察器或量化器
    if needs_obs_or_fq:
        # 将节点与其输出激活的观察器或量化器映射存入 obs_or_fq_map
        obs_or_fq_map[node] = output_act_obs_or_fq
        # 调用 _insert_obs_or_fq 函数插入观察器或量化器到模型的节点中
        return _insert_obs_or_fq(node, output_act_obs_or_fq, model, named_modules, graph)
    else:
        # 否则返回 None
        return None
# 在图输出节点之前可能插入观察者的函数，用于量化
def _maybe_insert_observers_before_graph_output(
    # 输出的图节点
    graph_output_node: Node,
    # PyTorch 模型
    model: torch.nn.Module,
    # 命名模块字典
    named_modules: Dict[str, torch.nn.Module],
    # 图结构
    graph: Graph,
    # 边或节点到观察器或伪量化对象的映射
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    # 是否为量化训练模式
    is_qat: bool,
) -> None:
    """
    如果输出需要量化，并且输出中存在任何尚未观察的节点，
    则为这些节点插入观察者。
    """

    def _recursive_maybe_replace_node_with_obs(
        # 可能是节点的参数
        maybe_node: Argument,
        # PyTorch 模型
        model: torch.nn.Module,
        # 命名模块字典
        named_modules: Dict[str, torch.nn.Module],
        # 图结构
        graph: Graph,
        # 观察器或伪量化对象的映射
        obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
        # 已经处理的节点集合
        already_observed_nodes: Set[Node],
    ) -> Optional[Node]:
        """
        递归地检查并替换节点为观察者。
        如果节点需要被观察，但是尚未被观察，则插入观察者。
        如果节点已经被观察，则继续递归处理其后续节点。
        如果节点不需要被观察，则直接返回节点本身。
        """
    ) -> Argument:
        """
        Navigate an arbitrary data structure of lists, tuples, dicts.
        For each container type, recurse on all inputs. Once any Node
        is found, insert an observer if needed and do not recurse further.

        Returns the data structure with all nodes needing observation being
        replaced by their observers.
        """
        # 如果 maybe_node 是 Node 类型
        if isinstance(maybe_node, Node):
            # 获取该节点的目标数据类型作为输出
            arg_as_output_target_dtype = _get_arg_target_dtype_as_output(maybe_node, named_modules, obs_or_fq_map, is_qat)
            observer_mod = None
            arg_as_input_target_dtype = torch.float
            # 如果节点的元数据中包含目标数据类型信息
            if "target_dtype_info" in maybe_node.meta:
                # 获取输入激活观察器或量化器类
                observer_cls = maybe_node.meta["target_dtype_info"].get("input_act_obs_or_fq_ctr", None)
                if observer_cls is not None:
                    # 创建观察器对象
                    observer_mod = observer_cls()
                    arg_as_input_target_dtype = observer_mod.dtype
            # 检查是否需要插入观察器
            need_obs = (
                arg_as_output_target_dtype != arg_as_input_target_dtype and
                arg_as_input_target_dtype != torch.float
            )
            if need_obs:
                assert observer_mod is not None
                # 插入观察器节点
                observer_node = _insert_obs_or_fq(
                    maybe_node, observer_mod, model, named_modules, graph)
                return observer_node
            else:
                return maybe_node
        # 如果 maybe_node 是 list 或 tuple 类型
        elif isinstance(maybe_node, (list, tuple)):
            results = []
            # 递归处理每个内部节点
            for inner_node in maybe_node:
                results.append(_recursive_maybe_replace_node_with_obs(
                    inner_node, model, named_modules, graph))
            # 根据原始类型返回结果
            if isinstance(maybe_node, list):
                return results
            else:
                return tuple(results)
        # 如果 maybe_node 是 dict 类型
        elif isinstance(maybe_node, dict):
            results_dict = {}
            # 递归处理字典中的每个值
            for k, inner_v in maybe_node.items():
                results_dict[k] = _recursive_maybe_replace_node_with_obs(
                    inner_v, model, named_modules, graph)
            return results_dict
        # 如果 maybe_node 是 None 类型
        elif maybe_node is None:
            return None
        else:
            # 抛出异常，处理未知类型的节点
            raise Exception("Unhandled type for returned node:", maybe_node)  # noqa: TRY002

    # 遍历图输出节点的每个旧参数
    new_args = []
    for old_arg in graph_output_node.args:
        # 调用递归函数处理旧参数并添加到新参数列表中
        new_args.append(
            _recursive_maybe_replace_node_with_obs(
                old_arg, model, named_modules, graph))

    # 将新参数列表转换为元组并赋值给图输出节点的参数
    graph_output_node.args = tuple(new_args)  # type: ignore[assignment]
def _maybe_propagate_dtype_for_node(
    node: Node,
    target_dtype: Union[torch.dtype, type],
    node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig],
) -> None:
    """
    Assigns `target_dtype` to `node`, setting `is_dynamic` to False. If `node`
    is a general tensor shape op, also call this function recursively on
    the first argument, to propagate the dtype to the caller.
    """
    # 设置节点的目标数据类型信息
    node.meta["target_dtype_info"]["input_act_obs_or_fq_ctr"] = None
    node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"] = None

    # 如果这是一个复制节点，将目标数据类型传播到第一个参数
    root_node, _, pattern, qhandler, qconfig = node_name_to_match_result_with_qconfig.get(
        node.name, (None, None, None, None, None))
    
    # 如果存在量化处理器并且是通用张量形状操作
    if qhandler is not None and qhandler.is_general_tensor_value_op():
        prev_node = node.args[0]
        # 如果前一个节点是 Node 类型，递归调用 _maybe_propagate_dtype_for_node
        if isinstance(prev_node, Node):
            _maybe_propagate_dtype_for_node(
                prev_node, target_dtype, node_name_to_match_result_with_qconfig)

def propagate_dtypes_for_known_nodes(
    graph: Graph,
    node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig],
) -> None:
    """
    Currently we assume that inputs to the graph are either `torch.float` or
    `torch.quint8`, which is not always correct. For ops such as
    `x.masked_fill(mask, value)`, we know that the dtype of  `mask` is a
    `BoolTensor`. Propagate this information throughout the graph.

    Note: not all dtypes in the graph will be correct after this pass, but a
    higher percentage of them will be correct. Hopefully in the future we can
    replace this with a better way to reason about dtypes of tensors.
    """
    # 遍历图中的每个节点
    for node in graph.nodes:
        # 获取节点中的非可观测参数索引及其类型
        non_observable_arg_dict = get_non_observable_arg_indexes_and_types(node)

        # 对于每种参数类型
        for arg_type in non_observable_arg_dict:
            # 获取非可观测参数的索引
            non_observable_indices = non_observable_arg_dict[arg_type](node)

            # 对于每个索引
            for index in non_observable_indices:
                arg = node.args[index]

                # 当参数是元组时，手动处理其所有元素
                if isinstance(arg, (tuple, list)):
                    arg_list = list(arg)
                else:
                    arg_list = [arg]

                # 对于参数列表中的每个参数
                for cur_arg in arg_list:
                    # 对于 Node 类型的参数，调用 _maybe_propagate_dtype_for_node
                    if isinstance(cur_arg, torch.fx.node.Node):
                        _maybe_propagate_dtype_for_node(
                            cur_arg, arg_type, node_name_to_match_result_with_qconfig)

def _maybe_make_input_output_share_observers(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
) -> bool:
    """
    Ensures that we share an observer
    """
    """
    for all input arguments as well as the output argument. In detail, given
    a graph of

      x0 -> obs0 -> op -> x2
                  /
      x1 -> obs1 /

    where node obs0 points to observer instance observer0,
    obs1 points to observer1 and obs2 points to observer2, we make nodes obs1
    and ob2 point to observer0.
    Returns: whether the operation succeeded or not
    """
    first_arg = None
    # 找到第一个非张量参数
    for i in range(len(node.args)):
        if isinstance(node.args[i], (Node, list, tuple)):
            first_arg = node.args[i]
            break

    # 如果没有非张量参数，直接返回 False
    if first_arg is None:
        return False

    if isinstance(first_arg, (list, tuple)):
        first_arg_arg = first_arg[0]
    elif isinstance(first_arg, Node):
        first_arg_arg = first_arg
    else:
        return False

    # 如果我们有这样的图形
    #   observed_node -> non_observed_node -> cat
    # 我们需要追溯到第一个观察器
    iteration_guard = 0
    while not _is_activation_post_process_node(first_arg_arg, named_modules):
        if not isinstance(first_arg_arg, Node):
            return False
        # 没有找到操作的后处理激活器
        if first_arg_arg.op == "placeholder":
            return False
        # 追溯参数直到找到第一个张量/节点
        trace_back_node = None
        for i in range(len(first_arg_arg.args)):
            trace_back_node = first_arg_arg.args[i]
            if isinstance(trace_back_node, Node):
                break
        if trace_back_node is None:
            return False
        first_arg_arg = trace_back_node

        iteration_guard += 1
        if iteration_guard > 10000:
            raise AssertionError('Unable to find observer of previous node')

    assert isinstance(first_arg_arg, Node)
    target_to_use = first_arg_arg.target
    assert isinstance(target_to_use, str)
    obs_mod_to_use = named_modules[target_to_use]

    if isinstance(first_arg, (list, tuple)):
        # 设置所有其他输入观察器节点来使用该模块
        for input_idx, input_arg in enumerate(first_arg):
            if input_idx == 0:
                continue
            iteration_guard = 0
            while not _is_activation_post_process_node(input_arg, named_modules):
                # 由于当前节点没有输入参数，追溯失败
                if len(input_arg.args) < 1:
                    return False
                input_arg = input_arg.args[0]
                iteration_guard += 1
                if iteration_guard > 10000:
                    raise AssertionError('Unable to find observer of previous node')

            parent_name, name = _parent_name(input_arg.target)
            setattr(named_modules[parent_name], name, obs_mod_to_use)

    # 设置输出观察器节点来使用该模块
    # 对于每个使用当前节点作为输入的输出观察节点，进行处理
    for output_obs_node in node.users.keys():
        # 确保输出观察节点是激活后处理节点
        assert _is_activation_post_process_node(output_obs_node, named_modules)
        # 获取输出观察节点的父模块名称和节点名称
        parent_name, name = _parent_name(output_obs_node.target)
        # 将处理后的观察模块应用于父模块的对应属性上
        setattr(named_modules[parent_name], name, obs_mod_to_use)

    # TODO(未来的PR): 删除孤立的观察模块
    # 返回操作成功的标志True
    return True
# 从图模型中移除输出观察者节点
def _remove_output_observer(
        node: Node,
        model: torch.nn.Module,
        named_modules: Dict[str, torch.nn.Module]):
    # 获取节点的用户列表，并转换为列表
    items = list(node.users.items())
    # 遍历节点的每一个输出观察者节点及其索引
    for output_obs_node, _ in items:
        # 断言输出观察者节点是激活后处理节点
        assert _is_activation_post_process_node(output_obs_node, named_modules)
        # 将输出观察者节点的所有使用替换为当前节点
        output_obs_node.replace_all_uses_with(node)
        # 从模型图中删除输出观察者节点
        model.graph.erase_node(output_obs_node)  # type: ignore[union-attr, operator]

# 将自定义模块替换为被观察的模块
def _swap_custom_module_to_observed(
        node: Node,
        qconfig: QConfigAny,
        named_modules: Dict[str, torch.nn.Module],
        prepare_custom_config: PrepareCustomConfig):
    # 获取节点目标对应的自定义模块
    custom_module = named_modules[node.target]  # type: ignore[index]
    # 获取准备自定义配置中的浮点到观察映射
    custom_module_class_mapping = prepare_custom_config.float_to_observed_mapping
    # 获取替换为被观察的自定义模块类
    observed_custom_module_class = \
        get_swapped_custom_module_class(
            custom_module, custom_module_class_mapping, qconfig)
    # 根据浮点模块创建被观察的自定义模块实例
    observed_custom_module = \
        observed_custom_module_class.from_float(custom_module)
    # 获取父节点和名称
    parent_name, name = _parent_name(node.target)
    # 设置父模块中的属性为被观察的自定义模块
    setattr(named_modules[parent_name], name, observed_custom_module)

# 为模型插入观察者节点
def insert_observers_for_model(
    model: GraphModule,
    node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig],
    node_name_to_qconfig: Dict[str, QConfigAny],
    prepare_custom_config: PrepareCustomConfig,
    equalization_config_map: Dict[str, Any],
    backend_config: BackendConfig,
    observed_node_names: Set[str],
    is_qat: bool,
) -> Optional[Node]:
    """
    插入观察者节点，使用以下高级算法：

    对于图中的每个节点：
      1. 确定量化图中此节点的目标数据类型，并保存以备后续步骤使用
      2. 确定此节点所有参数和关键字参数的目标数据类型
      3. 如果任何参数或关键字参数的目标数据类型与当前节点的数据类型不匹配，则插入观察者
      4. 如果当前节点需要输出观察者，则插入输出观察者

    例如：

    - 起始图：
        x0 -> linear -> x1

    - 处理 x0 后的观察图：
        x0(fp32)

    - 处理 linear 后的观察图：
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8)

    - 处理 x1 后的观察图：
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8) -> x1

    处理完一个节点后，简单的观察者放置保证了该节点及其所有前驱节点的完整性。
    未来可能会有优化图的操作，如去重观察者等。
    """

    # node.meta["target_dtype_info"] 存储了从 qconfig 推导出的节点目标数据类型信息，
    # 例如，如果我们有一个带有 qconfig 的 conv2d 节点
    # qconfig = QConfig(activation=..., weight=...)
    # # input 和 bias 节点的信息被省略
    # # 对于 getattr 节点
    # # weight = getattr(self, 'weight')
    # weight.meta["target_dtype_info"] = {
    # Cache dictionary to store whether each Node has tensors (True/False)
    cache_for_no_tensor_check: Dict[Node, bool] = {}

    # Initialize named_modules with all modules in the model, including duplicates
    named_modules = dict(model.named_modules(remove_duplicate=False))

    # Lists to store indexes of quantized inputs and outputs based on custom configuration
    input_quantized_idxs: List[int] = prepare_custom_config.input_quantized_indexes
    output_quantized_idxs: List[int] = prepare_custom_config.output_quantized_indexes

    # Set to track processed nodes during dtype mapping initialization
    processed_nodes: Set[Node] = set()

    # Initialize default dtype info for all nodes in the model graph
    for node in model.graph.nodes:
        node.meta["target_dtype_info"] = copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)

    # Counters to assign indexes to placeholder nodes and output nodes
    inputs_seen_counter = 0
    outputs_seen_counter = 0

    # Dictionaries to map placeholder nodes to their input indexes and output nodes to their output indexes
    placeholder_node_to_input_index: Dict[Node, int] = {}
    output_node_to_output_index: Dict[Node, int] = {}

    # Iterate through all nodes in the model graph to populate input and output indexes
    for node in model.graph.nodes:
        if node.op == "placeholder":
            placeholder_node_to_input_index[node] = inputs_seen_counter
            inputs_seen_counter += 1
        if node.op == "output":
            output_node_to_output_index[node] = outputs_seen_counter
            outputs_seen_counter += 1

    # Step 1: Set the observer or fake quantize module constructor for each node in the matched_node_pattern
    for match_res_with_qconfig in node_name_to_match_result_with_qconfig.values():
        last_node, matched_node_pattern, pattern, qhandler, qconfig = match_res_with_qconfig
        assert qhandler is not None
        # Set target dtype information for nodes matching the specified pattern
        _set_target_dtype_info_for_matched_node_pattern(
            matched_node_pattern,
            last_node,
            qconfig,
            qhandler,
            backend_config,
            named_modules,
            cache_for_no_tensor_check,
            processed_nodes
        )

    # Step 2: Special cases handling dtype information for specific operators

    # Step 2.1: Process each node individually for settings not based on patterns
    #           This step may be removed in the future with improved node dtype information
    # 遍历模型图中的每个节点
    for node in model.graph.nodes:
        # 检查节点操作是否为 "placeholder" 并且对应的输入索引在 input_quantized_idxs 中
        if node.op == "placeholder" and placeholder_node_to_input_index[node] in input_quantized_idxs:
            # 用户不应调用 PlaceholderObserver 的 calculate_qparams 方法，这里用于编码输入张量的数据类型信息
            # 我们不会将这些观察器实际插入图中，也不会调用 calculate_qparams 方法
            node.meta["target_dtype_info"] = copy.copy(_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO)
        
        # 如果节点操作为 "call_module", "call_method", "call_function"
        elif node.op in ("call_module", "call_method", "call_function"):
            # 检查节点的所有参数是否都不包含张量
            args_have_no_tensors = \
                all_node_args_have_no_tensors(
                    node, named_modules, cache_for_no_tensor_check)
            if args_have_no_tensors:
                # 如果所有参数都不包含张量，则设置目标数据类型信息为 None
                node.meta["target_dtype_info"] = {
                    "input_act_obs_or_fq_ctr": None,
                    "output_act_obs_or_fq_ctr": None,
                }
        
        # 如果节点操作为 "output" 并且对应的输出索引在 output_quantized_idxs 中
        elif node.op == "output" and output_node_to_output_index[node] in output_quantized_idxs:
            # TODO(未来的PR): 更新 output_quantized_idxs 的API以匹配任意的数据结构。
            # 总是只有一个输出，且该输出可以有任意嵌套的值。List[int] 不是这种情况的正确数据类型。

            # TODO(未来的PR): 如果必要，支持模型输出中的更多数据类型
            # 复制默认的 quint8 目标数据类型配置信息到节点的 meta 属性中
            node.meta["target_dtype_info"] = copy.copy(_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO)

    # Step 2.2, 对于已知输入数据类型的节点，在整个图中传播这些类型。
    # 例如，如果有一个调用如下：
    #   x1 = x0.masked_fill(mask, 1)
    # 我们将 mask 的类型传播为 torch.bool
    propagate_dtypes_for_known_nodes(model.graph, node_name_to_match_result_with_qconfig)

    # Step 3, 检查请求的目标数据类型信息是否被后端支持
    # 如果不支持，将重置目标数据类型信息以使用默认值（float Tensor）
    
    # 重置节点处理过的计数器和已处理节点的集合
    processed_nodes: Set[Node] = set()
    # 遍历节点名称到匹配结果与量化配置的映射的所有值
    for match_res_with_qconfig in node_name_to_match_result_with_qconfig.values():
        # 解包匹配结果与量化配置
        last_node, matched_node_pattern, pattern, qhandler, qconfig = match_res_with_qconfig
        # 检查模式、匹配节点模式、量化配置是否被后端支持
        is_supported_by_backend = _is_pattern_dtype_config_and_qconfig_supported_by_backend(
            pattern, matched_node_pattern, qconfig, backend_config)
        # 断言量化处理器不为空
        assert qhandler is not None

        # 获取输出激活数据类型，以便不重置特殊类型节点
        # TODO: 可能需要更统一地处理这些节点
        # 如果可以使用 node.meta["val"]，则可以改进此处
        output_act_or_fq_ctr = node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"]
        output_act_or_fq = output_act_or_fq_ctr() if output_act_or_fq_ctr else None
        output_act_dtype, _ = _get_dtype_and_is_dynamic(output_act_or_fq)
        # 如果不被后端支持且输出激活数据类型不在 [None, int, float, torch.bool] 中
        if not is_supported_by_backend and output_act_dtype not in [None, int, float, torch.bool]:
            # 如果不被后端支持，则将目标数据类型信息恢复为默认值
            _set_target_dtype_info_for_matched_node_pattern(
                matched_node_pattern,
                last_node,
                torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig,
                None,
                backend_config,
                named_modules,
                cache_for_no_tensor_check,
                processed_nodes
            )

    # 在此之后，当前节点及其所有参数都已分配目标数据类型信息。现在，我们为此节点的输入（如果需要）和输出（如果需要）插入观察器。

    # 由于我们在进行图变异时，我们遍历原始节点而不是 model.graph.nodes。
    nodes_before_observation = list(model.graph.nodes)

    # 避免为具有相同目标的多个节点重复自定义模块交换
    custom_module_names_already_swapped: Set[str] = set()

    # 重置输入/输出计数器
    inputs_seen_counter = 0
    outputs_seen_counter = 0
    results_node = None
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}

    # 返回结果节点
    return results_node
def _run_prepare_fx_on_standalone_modules(
    model: torch.nn.Module,
    is_qat: bool,
    named_modules: Dict[str, torch.nn.Module],
    node_name_to_match_result_with_qconfig: Any,
    prepare_custom_config: PrepareCustomConfig,
    backend_config: BackendConfig,
) -> None:
    """
    Runs prepare_fx on each standalone module. Note: this does
    not modify the graph, it just replaces the unobserved modules with
    their observed versions.
    """
    # 遍历所有与量化配置匹配的节点结果
    for (root_node, _, pattern, qhandler, qconfig) in node_name_to_match_result_with_qconfig.values():
        # 如果量化处理器为None，则跳过
        if qhandler is None:
            continue
        # 如果量化处理器不是独立模块，则跳过
        elif not qhandler.is_standalone_module():
            continue

        # 获取独立模块的配置信息、示例输入、自定义配置和后端配置
        sm_qconfig_mapping, sm_example_inputs, sm_prepare_custom_config, \
            sm_backend_config = _get_standalone_module_configs(
                root_node, named_modules, prepare_custom_config, qconfig, backend_config)

        # 获取当前独立模块
        standalone_module = named_modules[root_node.target]
        # 准备独立模块的量化函数
        prepare = \
            torch.ao.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore[attr-defined]
        # 应用量化函数，返回观察后的独立模块
        observed_standalone_module = \
            prepare(
                standalone_module,
                sm_qconfig_mapping,
                is_qat,
                example_inputs=sm_example_inputs,
                prepare_custom_config=sm_prepare_custom_config,
                backend_config=sm_backend_config)
        
        # 获取父模块和模块名称
        parent_name, name = _parent_name(root_node.target)
        # 将观察后的独立模块设置为父模块的属性
        setattr(named_modules[parent_name], name, observed_standalone_module)
        # 更新命名模块中的独立模块
        named_modules[root_node.target] = observed_standalone_module

def _save_state(
    observed: GraphModule,
    node_name_to_qconfig: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]],
    prepare_custom_config: PrepareCustomConfig,
    equalization_node_name_to_qconfig: Dict[str, Any],
    qconfig_mapping: QConfigMapping,
    is_qat: bool,
    observed_node_names: Set[str],
) -> None:
    # 将观察后的图模块属性保存到元数据中
    observed.meta["_observed_graph_module_attrs"] = (
        ObservedGraphModuleAttrs(
            node_name_to_qconfig=node_name_to_qconfig,
            node_name_to_scope=node_name_to_scope,
            prepare_custom_config=prepare_custom_config,
            equalization_node_name_to_qconfig=equalization_node_name_to_qconfig,
            qconfig_mapping=qconfig_mapping,
            is_qat=is_qat,
            observed_node_names=observed_node_names,
        )
    )

def prepare(
        model: GraphModule,
        qconfig_mapping: Union[QConfigMapping, Dict[str, Any]],
        is_qat: bool,
        node_name_to_scope: Dict[str, Tuple[str, type]],
        example_inputs: Tuple[Any, ...],
        prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None] = None,
        _equalization_config: Union[QConfigMapping, Dict[str, Any], None] = None,
        backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
        is_standalone_module: bool = False) -> GraphModule:
    """
    standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module
    Args:
        node_name_to_scope: mapping from node name to the scope of the module which contains the node.
        The scope is a tuple of fully qualified path of the module and the type of the module
    Returns:
        model(GraphModule): prepared standalone module
        attributes related to standalone module
        in model.meta["_observed_graph_module_attrs"]:
            is_observed_standalone_module (bool): boolean value that shows whether the
            current model is a observed standalone module or not
            standalone_module_input_quantized_idxs(List[Int]): a list of
                indexes for the graph input that is expected to be quantized,
                same as input_quantized_idxs configuration provided
                for the standalone module
            standalone_module_output_quantized_idxs(List[Int]): a list of
                indexs for the graph output that is quantized
                same as input_quantized_idxs configuration provided
                for the standalone module
    """

    # 如果 prepare_custom_config 为 None，则使用默认的 PrepareCustomConfig 对象
    if prepare_custom_config is None:
        prepare_custom_config = PrepareCustomConfig()

    # 如果 _equalization_config 为 None，则使用空的 QConfigMapping 对象
    if _equalization_config is None:
        _equalization_config = QConfigMapping()

    # 如果 qconfig_mapping 是 dict 类型，则发出警告，并将其转换为 QConfigMapping 对象
    if isinstance(qconfig_mapping, dict):
        warnings.warn(
            "Passing a QConfig dictionary to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a QConfigMapping instead.",
            FutureWarning,
            stacklevel=2,
        )
        qconfig_mapping = QConfigMapping.from_dict(qconfig_mapping)

    # 如果 _equalization_config 是 dict 类型，则发出警告，并将其转换为 QConfigMapping 对象
    if isinstance(_equalization_config, dict):
        warnings.warn(
            "Passing a QConfig dictionary to prepare for equalization is deprecated and will not "
            "be supported in a future version. Please pass in a QConfigMapping instead.",
            FutureWarning,
            stacklevel=2,
        )
        _equalization_config = QConfigMapping.from_dict(_equalization_config)

    # 如果 prepare_custom_config 是 dict 类型，则发出警告，并将其转换为 PrepareCustomConfig 对象
    if isinstance(prepare_custom_config, dict):
        warnings.warn(
            "Passing a prepare_custom_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a PrepareCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config)
    # 如果 backend_config 是字典类型，发出警告，并将其转换为 BackendConfig 对象
    if isinstance(backend_config, dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        backend_config = BackendConfig.from_dict(backend_config)

    # 确保 qconfig_mapping 和 _equalization_config 是 QConfigMapping 类型
    assert isinstance(qconfig_mapping, QConfigMapping)
    assert isinstance(_equalization_config, QConfigMapping)

    # 深拷贝 qconfig_mapping 和 _equalization_config，防止修改原始对象
    qconfig_mapping = copy.deepcopy(qconfig_mapping)
    _equalization_config = copy.deepcopy(_equalization_config)

    # 创建空字典 pattern_to_quantize_handler，用于存储节点模式到 QuantizeHandler 的映射关系
    # 这些映射关系将被用于量化处理
    pattern_to_quantize_handler: Dict[Pattern, QuantizeHandler] = {}

    # 如果 backend_config 为 None，则获取本地后端配置
    if backend_config is None:
        backend_config = get_native_backend_config()

    # 获取模式到 QuantizeHandler 的映射关系
    pattern_to_quantize_handler = _get_pattern_to_quantize_handlers(backend_config)
    # 对模式进行排序
    pattern_to_quantize_handler = _sorted_patterns_dict(pattern_to_quantize_handler)

    # 获取融合模式到根节点获取器的映射关系
    root_node_getter_mapping = \
        get_fusion_pattern_to_root_node_getter(backend_config)

    # 更新模型的量化配置，用于融合
    _update_qconfig_for_fusion(model, qconfig_mapping)
    _update_qconfig_for_fusion(model, _equalization_config)

    # 获取扁平化的量化配置字典
    flattened_qconfig_dict = _get_flattened_qconfig_dict(qconfig_mapping)

    # 将量化配置传播到模型中
    # TODO: 支持正则表达式
    propagate_qconfig_(model, flattened_qconfig_dict, prepare_custom_config.to_dict())

    # 如果是量化训练，获取模块到量化训练模块的映射，并替换模型中的模块
    if is_qat:
        module_to_qat_module = get_module_to_qat_module(backend_config)
        _qat_swap_modules(model, module_to_qat_module)
        _update_qconfig_for_qat(qconfig_mapping, backend_config)

    # 创建 named_modules 字典，记录模块的全限定名到模块实例的映射关系
    named_modules = dict(model.named_modules(remove_duplicate=False))

    # 创建 equalization_node_name_to_qconfig 字典，记录节点名称到均衡化配置的映射关系
    equalization_node_name_to_qconfig = _generate_node_name_to_qconfig(
        model, named_modules, model.graph, _equalization_config, node_name_to_scope)

    # 创建 node_name_to_qconfig 字典，记录节点名称到量化配置的映射关系
    node_name_to_qconfig = _generate_node_name_to_qconfig(
        model, named_modules, model.graph, qconfig_mapping, node_name_to_scope)

    # 获取需要单独处理的模块名称和模块类别列表
    standalone_module_names = list(prepare_custom_config.standalone_module_names.keys())
    standalone_module_classes = list(prepare_custom_config.standalone_module_classes.keys())
    # 获取自定义模块类的键集合，用于后续处理
    custom_module_classes = get_custom_module_class_keys(prepare_custom_config.float_to_observed_mapping)
    
    # 在模型的计算图中查找与量化处理器模式匹配的结果，但不包含量化配置信息
    matches_without_qconfig = _find_matches(
        model.graph, named_modules, pattern_to_quantize_handler, root_node_getter_mapping,
        standalone_module_names, standalone_module_classes, custom_module_classes)
    
    # 将节点名称与匹配结果关联的量化配置实例映射起来
    node_name_to_match_result_with_qconfig = {}
    for node_name, match_without_qconfig in matches_without_qconfig.items():
        match_with_qconfig = (*match_without_qconfig, node_name_to_qconfig[node_name])
        node_name_to_match_result_with_qconfig[node_name] = match_with_qconfig
    
    # 在独立模块上运行准备函数，可能涉及量化训练等
    _run_prepare_fx_on_standalone_modules(
        model, is_qat, named_modules, node_name_to_match_result_with_qconfig, prepare_custom_config, backend_config)
    
    # 用于记录观察节点的名称集合，以便在转换阶段决定是否需要将浮点模块转换为参考量化模块
    observed_node_names: Set[str] = set()
    
    # 为模型插入观察器以实现量化
    result_node = insert_observers_for_model(
        model,
        node_name_to_match_result_with_qconfig,
        node_name_to_qconfig,
        prepare_custom_config,
        equalization_node_name_to_qconfig,
        backend_config,
        observed_node_names,
        is_qat,
    )
    
    # 将模型封装为图形模块，以便进一步处理
    model = GraphModule(model, model.graph)
    
    # 保存模型的状态和配置信息，包括量化配置、观察节点信息等
    _save_state(model, node_name_to_qconfig, node_name_to_scope,
                prepare_custom_config, equalization_node_name_to_qconfig,
                qconfig_mapping, is_qat, observed_node_names)
    
    # 如果是独立模块，则进行必要的断言和属性设置
    if is_standalone_module:
        assert result_node is not None
        assert isinstance(result_node.args[0], Node), \
            "standalone module only supports returning simple value currently"\
            "(not tuple, dict etc.)"
        # 在父模块中观察这些输入
        # 将 List[int] 转换为 Tensor，因为模块属性是 Union[Tensor, Module]
        input_quantized_idxs: List[int] = prepare_custom_config.input_quantized_indexes
        output_quantized_idxs: List[int] = prepare_custom_config.output_quantized_indexes
        observed_graph_module_attrs = model.meta["_observed_graph_module_attrs"]
        # 进行就地修改
        observed_graph_module_attrs.is_observed_standalone_module = True
        observed_graph_module_attrs.standalone_module_input_quantized_idxs = \
            input_quantized_idxs
        observed_graph_module_attrs.standalone_module_output_quantized_idxs = \
            output_quantized_idxs
    
    # 返回经过处理和量化的模型
    return model
```