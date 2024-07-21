# `.\pytorch\torch\ao\quantization\backend_config\utils.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型声明
from typing import Dict, Any, List, Callable, Union, Tuple, Type

# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入本地模块和类
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
)
from ..utils import Pattern
from ..fuser_method_mappings import (
    _reverse2,
    _reverse3,
)

# __all__列表定义，用于模块导入时的限制
__all__ = [
    "get_pattern_to_dtype_configs",
    "get_qat_module_classes",
    "get_fused_module_classes",
    "get_pattern_to_input_type_to_index",
    "get_root_module_to_quantized_reference_module",
    "get_fuser_method_mapping",
    "get_module_to_qat_module",
    "get_fusion_pattern_to_root_node_getter",
    "get_fusion_pattern_to_extra_inputs_getter",
    "remove_boolean_dispatch_from_name",
    "pattern_to_human_readable",
    "entry_to_pretty_str",
]

# 根据后端配置获取模式到数据类型配置列表的映射
def get_pattern_to_dtype_configs(backend_config: BackendConfig) -> Dict[Pattern, List[DTypeConfig]]:
    pattern_to_dtype_configs: Dict[Pattern, List[DTypeConfig]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_dtype_configs[pattern] = config.dtype_configs
    return pattern_to_dtype_configs

# 根据后端配置获取量化训练模块类的元组
def get_qat_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    qat_module_classes = []
    for config in backend_config.configs:
        if config.qat_module is not None:
            qat_module_classes.append(config.qat_module)
    return tuple(set(qat_module_classes))

# 根据后端配置获取融合模块类的元组
def get_fused_module_classes(backend_config: BackendConfig) -> Tuple[type, ...]:
    fused_module_classes = []
    for config in backend_config.configs:
        if config.fused_module is not None:
            fused_module_classes.append(config.fused_module)
    return tuple(set(fused_module_classes))

# 根据后端配置获取模式到输入类型到索引的字典映射
def get_pattern_to_input_type_to_index(backend_config: BackendConfig) -> Dict[Pattern, Dict[str, int]]:
    pattern_to_input_type_to_index: Dict[Pattern, Dict[str, int]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        pattern_to_input_type_to_index[pattern] = config._input_type_to_index
    return pattern_to_input_type_to_index

# 根据后端配置获取根模块到量化参考模块的映射字典
def get_root_module_to_quantized_reference_module(
        backend_config: BackendConfig) -> Dict[Type[torch.nn.Module], Type[torch.nn.Module]]:
    mapping: Dict[Type[torch.nn.Module], Type[torch.nn.Module]] = {}
    for config in backend_config.configs:
        if config.root_module is not None and config.reference_quantized_module is not None:
            mapping[config.root_module] = config.reference_quantized_module
    return mapping

# 根据后端配置获取模式到融合方法映射的字典
def get_fuser_method_mapping(backend_config: BackendConfig) -> Dict[Pattern, Union[nn.Sequential, Callable]]:
    fuser_method_mapping : Dict[Pattern, Union[nn.Sequential, Callable]] = {}
    # 遍历 backend_config._pattern_complex_format_to_config 字典的每个键值对
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        # 检查 config 中的 fuser_method 是否存在
        if config.fuser_method is not None:
            # 注意：在 BackendConfig 中，fuser_method 和 pattern 都按照正向顺序指定，
            # 但是内部的模式匹配代码使用了反向嵌套元组的格式，因此我们需要将两者都转换成内部格式
            # 获取反向嵌套元组格式的 fuser_method
            fuser_method = _get_fuser_method_in_reversed_nested_tuple_format(config)
            # 将 pattern 映射到相应的 fuser_method
            fuser_method_mapping[pattern] = fuser_method
    # 返回 pattern 到 fuser_method 的映射字典
    return fuser_method_mapping
# 根据后端配置获取模块到量化模块的映射，返回一个字典，键为模式（Pattern），值为对应的 torch.nn.Module 类型
def get_module_to_qat_module(backend_config: BackendConfig) -> Dict[Pattern, Type[torch.nn.Module]]:
    # 初始化空字典，用于存储模块到量化模块的映射关系
    module_to_qat_module: Dict[Pattern, Type[torch.nn.Module]] = {}
    # 遍历后端配置中的模式复杂格式到配置的映射项
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        # 如果配置中存在量化模块，则将模式与量化模块的映射添加到字典中
        if config.qat_module is not None:
            module_to_qat_module[pattern] = config.qat_module
    # 返回模块到量化模块的映射字典
    return module_to_qat_module

# 根据后端配置获取融合模式到获取根节点函数的映射，返回一个字典，键为模式（Pattern），值为一个可调用对象
def get_fusion_pattern_to_root_node_getter(backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    """ Get a map from fusion pattern to a function that returns the root node
    from the fusion pattern, e.g. the most common one is:
    def get_root_node(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]
    This can work for all patterns whose root node is the "last node" in the pattern,
    e.g. (torch.add, MatchAllNode, (torch.ReLU, torch.Conv2d))
    """
    # 初始化空字典，用于存储融合模式到获取根节点函数的映射关系
    root_node_getter_mapping: Dict[Pattern, Callable] = {}
    # 遍历后端配置中的模式复杂格式到配置的映射项
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        # 如果配置中存在根节点获取函数，则将模式与函数的映射添加到字典中
        if config._root_node_getter is not None:
            root_node_getter_mapping[pattern] = config._root_node_getter
    # 返回融合模式到获取根节点函数的映射字典
    return root_node_getter_mapping

# 根据后端配置获取融合模式到获取额外输入节点函数的映射，返回一个字典，键为模式（Pattern），值为一个可调用对象
def get_fusion_pattern_to_extra_inputs_getter(backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    """ Get a map from fusion pattern to a function that returns extra input nodes
    from the fusion pattern, in the order required by the root node. This is optional,
    if not specified, we will not copy over any extra inputs for the root node.
    Example:
    # Let's say we have the pattern (torch.add, MatchAllNode, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
    # and root node is torch.nn.Conv2d, and the node in MatchAllNode would be an extra
    # argument to the fused module, we can unpack the pattern and return the node at
    # MatchAllNode here
    # we can implement extra_inputs_getter as follows:
    def extra_inputs_getter(pattern) -> List[Any]:
        add, extra_input, conv_pattern = pattern
        return [extra_input]
    """
    # 初始化空字典，用于存储融合模式到获取额外输入节点函数的映射关系
    extra_inputs_getter_mapping: Dict[Pattern, Callable] = {}
    # 遍历后端配置中的模式复杂格式到配置的映射项
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        # 如果配置中存在额外输入获取函数，则将模式与函数的映射添加到字典中
        if config._extra_inputs_getter is not None:
            extra_inputs_getter_mapping[pattern] = config._extra_inputs_getter
    # 返回融合模式到获取额外输入节点函数的映射字典
    return extra_inputs_getter_mapping

# 根据操作对象去除名称中的布尔分发，返回一个对象
def remove_boolean_dispatch_from_name(p) -> Any:
    """
    Some ops have a default string representation such as
    '<function boolean_dispatch.<locals>.fn at 0x7ff1106bf280>',
    this function replaces them with the hardcoded function names.
    """
    # 根据操作对象 p 进行名称的处理，将特定操作映射为固定的函数名称字符串
    if p is F.fractional_max_pool2d:
        return "torch.nn.functional.fractional_max_pool2d"
    elif p is F.fractional_max_pool3d:
        return "torch.nn.functional.fractional_max_pool3d"
    elif p is F.max_pool1d:
        return "torch.nn.functional.max_pool1d"
    # 如果函数 p 是 torch.nn.functional.max_pool2d，则返回对应的字符串表示
    elif p is F.max_pool2d:
        return "torch.nn.functional.max_pool2d"
    # 如果函数 p 是 torch.nn.functional.max_pool3d，则返回对应的字符串表示
    elif p is F.max_pool3d:
        return "torch.nn.functional.max_pool3d"
    # 如果函数 p 是 torch.nn.functional.adaptive_max_pool1d，则返回对应的字符串表示
    elif p is F.adaptive_max_pool1d:
        return "torch.nn.functional.adaptive_max_pool1d"
    # 如果函数 p 是 torch.nn.functional.adaptive_max_pool2d，则返回对应的字符串表示
    elif p is F.adaptive_max_pool2d:
        return "torch.nn.functional.adaptive_max_pool2d"
    # 如果函数 p 是 torch.nn.functional.adaptive_max_pool3d，则返回对应的字符串表示
    elif p is F.adaptive_max_pool3d:
        return "torch.nn.functional.adaptive_max_pool3d"
    # 如果函数 p 不是量化文档中的布尔调度，抛出断言错误，指出 p 没有人类可读的表示方式
    assert "boolean_dispatch" not in str(p), \
        f"{p} does not have a human readable representation in " + \
        "quantization documentation"
    # 返回函数 p 本身，即使未找到匹配的字符串表示
    return p
def pattern_to_human_readable(p) -> Any:
    if isinstance(p, tuple):
        # 如果输入是元组，则递归处理内部的每个元素
        return tuple(pattern_to_human_readable(inner_p) for inner_p in p)
    elif isinstance(p, str):
        # 如果输入是字符串，则直接返回，因为方法名已经是人类可读的
        return p
    else:
        # 如果不是元组也不是字符串，先移除名称中的布尔分发，然后返回
        p = remove_boolean_dispatch_from_name(p)
        return p

# TODO(future PR): move backend_config_dict to use dataclass and move this logic to
# the corresponding __str__ function
def entry_to_pretty_str(entry) -> str:
    """
    给定 backend_config_dict 条目，返回其人类可读的字符串表示形式。
    """
    s = "{\n"

    # 总是先输出 pattern
    if "pattern" in entry:
        pattern_str = pattern_to_human_readable(entry["pattern"])

        s += f"  'pattern': {pattern_str},\n"

    # dtype_configs 的定制输出，使其看起来更整洁
    if "dtype_configs" in entry:
        s += "  'dtype_configs': [\n"
        for dtype_config in entry["dtype_configs"]:
            s += "    {\n"
            for k, v in dtype_config.items():
                s += f"      '{k}': {v},\n"
            s += "    },\n"
        s += "  ],\n"

    # num_tensor_args_to_observation_type 的定制输出，使其看起来更整洁
    if "num_tensor_args_to_observation_type" in entry:
        s += "  'num_tensor_args_to_observation_type': {\n"
        for k, v in entry["num_tensor_args_to_observation_type"].items():
            s += f"    {k}: {v},\n"
        s += "  },\n"

    # 输出所有其他字段
    custom_handled_fields = [
        "pattern",
        "dtype_configs",
        "num_tensor_args_to_observation_type",
    ]
    for field_name in entry:
        if field_name in custom_handled_fields:
            continue
        s += f"  '{field_name}': {entry[field_name]},\n"

    s += "}"
    return s

def _get_pattern_in_reversed_nested_tuple_format(config: BackendPatternConfig) -> Pattern:
    """
    在给定的配置中返回模式的反转嵌套元组格式，该格式在量化模式匹配代码中内部使用。

    如果模式不是元组，或者模式已经以反转嵌套元组格式指定，则直接返回模式。
    否则：
    - 对于 2-元组 (a, b)，返回 (b, a)。
    - 对于 3-元组 (a, b, c)，返回 (c, (b, a))。
    """
    # 实现略，未提供完整的代码块
    """
    如果配置中指定了复杂格式的模式匹配方式，则返回该复杂格式。

    如果配置中的 'pattern' 为 None，则抛出数值错误异常，要求必须指定 'pattern' 或 'pattern_complex_format'。

    如果配置中的 'pattern' 不是元组类型，则直接返回 'pattern'。

    如果配置中的 'pattern' 是简单元组格式，则进行转换：
        - 如果元组长度为2，将其转换为 (b, a) 格式。
        - 如果元组长度为3，将其转换为 (c, (b, a)) 格式。
        - 其他长度将引发数值错误异常。

    返回转换后的模式格式。
    """
    if config._pattern_complex_format is not None:
        return config._pattern_complex_format
    if config.pattern is None:
        raise ValueError("Either 'pattern' or 'pattern_complex_format' must be specified")
    if not isinstance(config.pattern, tuple):
        return config.pattern

    # Pattern is specified in the simple tuple format, need to convert
    if len(config.pattern) == 2:
        (a, b) = config.pattern
        return (b, a)
    elif len(config.pattern) == 3:
        (a, b, c) = config.pattern
        return (c, (b, a))
    else:
        raise ValueError("Expected a tuple with 2 or 3 elements, got: ", config.pattern)
# 返回根据给定配置中指定的融合方法，以反向嵌套元组格式返回的融合方法。
def _get_fuser_method_in_reversed_nested_tuple_format(config: BackendPatternConfig) -> Callable:
    """
    根据给定配置返回以反向嵌套元组格式表示的融合方法。
    
    如果模式在反向嵌套元组格式中指定，则假定融合方法也以此格式指定，并直接返回。
    否则，根据以下规则转换融合方法：

        * 对于 f(is_qat, conv, relu)，返回 f'(is_qat, relu, conv)
        * 对于 f(is_qat, conv, bn, relu)，返回 f'(is_qat, relu, bn_conv)，其中 bn_conv 是一个二元组 (bn, conv)

    融合方法的第一个参数始终是 `is_qat`，在转换中不受影响。当前仅支持具有 3 或 4 个参数的函数。
    """
    assert config.fuser_method is not None
    # 如果配置指定了复杂格式的模式，则直接返回配置中的融合方法
    if config._pattern_complex_format is not None:
        return config.fuser_method
    # 如果模式不是元组，则抛出 ValueError
    if not isinstance(config.pattern, tuple):
        raise ValueError("Expected pattern to be a tuple, got: ", config.pattern)

    # 如果模式是简单元组格式，则需要进行转换
    if len(config.pattern) == 2:
        return _reverse2(config.fuser_method)
    elif len(config.pattern) == 3:
        return _reverse3(config.fuser_method)
    else:
        # 如果模式元素个数不是 2 或 3，则抛出 ValueError
        raise ValueError("Expected a tuple with 2 or 3 elements, got: ", config.pattern)
```