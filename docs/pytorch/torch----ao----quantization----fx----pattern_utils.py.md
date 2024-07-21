# `.\pytorch\torch\ao\quantization\fx\pattern_utils.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
from collections import OrderedDict
from typing import Dict, Any
from torch.ao.quantization.utils import Pattern
from ..fake_quantize import FixedQParamsFakeQuantize
from ..observer import ObserverBase
import copy

# 定义模块公开的符号列表
__all__ = [
    "get_default_fusion_patterns",
    "get_default_quant_patterns",
    "get_default_output_activation_post_process_map",
]

# TODO(future PR): fix the typing on QuantizeHandler (currently a circular dependency)
# 定义 QuantizeHandler 类型为 Any，因为目前存在循环依赖问题
QuantizeHandler = Any

# 用于卷积和批归一化融合的默认模式字典
_DEFAULT_FUSION_PATTERNS: Dict[Pattern, QuantizeHandler] = OrderedDict()

# 注册融合模式的装饰器函数
def _register_fusion_pattern(pattern):
    def insert(fn):
        _DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn
    return insert

# 返回默认的融合模式字典
def get_default_fusion_patterns() -> Dict[Pattern, QuantizeHandler]:
    return copy.copy(_DEFAULT_FUSION_PATTERNS)

# 用于静态量化和量化感知训练的默认量化模式字典
_DEFAULT_QUANTIZATION_PATTERNS: Dict[Pattern, QuantizeHandler] = OrderedDict()

# 注册量化模式的装饰器函数，支持固定量化参数观察器
def _register_quant_pattern(pattern, fixed_qparams_observer=None):
    def insert(fn):
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if fixed_qparams_observer is not None:
            # 将模式与固定量化参数伪量化器关联起来
            _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP[pattern] = FixedQParamsFakeQuantize.with_args(observer=fixed_qparams_observer)
            # 将模式与固定量化参数观察器关联起来
            _DEFAULT_OUTPUT_OBSERVER_MAP[pattern] = fixed_qparams_observer
        return fn
    return insert

# 返回默认的量化模式字典
def get_default_quant_patterns() -> Dict[Pattern, QuantizeHandler]:
    return copy.copy(_DEFAULT_QUANTIZATION_PATTERNS)

# 用于输出激活后处理构造函数的默认映射字典
_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP: Dict[Pattern, QuantizeHandler] = {}
_DEFAULT_OUTPUT_OBSERVER_MAP: Dict[Pattern, QuantizeHandler] = {}

# 返回默认的输出激活后处理映射字典，根据是否在训练中选择不同的映射
def get_default_output_activation_post_process_map(is_training) -> Dict[Pattern, ObserverBase]:
    if is_training:
        return copy.copy(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP)
    else:
        return copy.copy(_DEFAULT_OUTPUT_OBSERVER_MAP)

# 定义一个函数，对模式字典按照模式长度排序，以便于匹配更长的模式优先
def _sorted_patterns_dict(patterns_dict: Dict[Pattern, QuantizeHandler]) -> Dict[Pattern, QuantizeHandler]:
    """
    Return a sorted version of the patterns dictionary such that longer patterns are matched first,
    e.g. match (F.relu, F.linear) before F.relu.
    This works for current use cases, but we may need to have a more clever way to sort
    things to address more complex patterns
    """
    # 定义一个函数，用于计算给定模式的长度，通过统计所有条目的数量来实现。
    # 这确保了 (nn.ReLU, (nn.BatchNorm, nn.Conv2d)) 在 (nn.BatchNorm, nn.Conv2d) 之前，
    # 以便我们能够首先匹配前者。
    def get_len(pattern):
        len = 0  # 初始化长度计数器
        if isinstance(pattern, tuple):  # 如果模式是一个元组
            for item in pattern:  # 遍历元组中的每个项
                len += get_len(item)  # 递归调用 get_len 函数，累加每个项的长度
        else:
            len += 1  # 如果模式不是元组，直接将长度加1
        return len  # 返回计算得到的长度

    # 返回一个排序后的有序字典，按照模式的长度进行降序排列
    return OrderedDict(sorted(patterns_dict.items(), key=lambda kv: -get_len(kv[0]) if isinstance(kv[0], tuple) else 1))
```