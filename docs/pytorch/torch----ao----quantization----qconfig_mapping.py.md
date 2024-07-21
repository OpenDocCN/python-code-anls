# `.\pytorch\torch\ao\quantization\qconfig_mapping.py`

```
# mypy: allow-untyped-defs
# Import necessary modules and functions
from __future__ import annotations
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union, List

# Import torch module
import torch

# Import specific functions and classes from local modules
from .fake_quantize import (
    default_weight_fake_quant,
    FixedQParamsFakeQuantize,
)
from .observer import (
    _PartialWrapper,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    default_placeholder_observer,
    default_weight_observer,
)
from .qconfig import (
    default_reuse_input_qconfig,
    default_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
    get_default_qconfig,
    get_default_qat_qconfig,
    QConfig,
    QConfigAny,
    default_quint8_weight_qconfig
)

# Define public symbols that will be exported when using "from module import *"
__all__ = [
    "get_default_qconfig_mapping",
    "get_default_qat_qconfig_mapping",
    "QConfigMapping",
]

# TODO: replace all usages with these constants
# Define global constants for dictionary keys
_GLOBAL_DICT_KEY = ""
_OBJECT_TYPE_DICT_KEY = "object_type"
_MODULE_NAME_REGEX_DICT_KEY = "module_name_regex"
_MODULE_NAME_DICT_KEY = "module_name"
_MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY = "module_name_object_type_order"

# TODO: derive this map from the BackendConfig
# Dictionary mapping operations to their respective observer functions or callables
_FIXED_QPARAMS_OP_TO_OBSERVER: Dict[Union[Callable, str], _PartialWrapper] = {
    torch.nn.Hardsigmoid: default_fixed_qparams_range_0to1_observer,
    torch.nn.functional.hardsigmoid: default_fixed_qparams_range_0to1_observer,
    "hardsigmoid": default_fixed_qparams_range_0to1_observer,
    "hardsigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Sigmoid: default_fixed_qparams_range_0to1_observer,
    torch.sigmoid: default_fixed_qparams_range_0to1_observer,
    "sigmoid": default_fixed_qparams_range_0to1_observer,
    "sigmoid_": default_fixed_qparams_range_0to1_observer,
    torch.nn.Softmax: default_fixed_qparams_range_0to1_observer,
    torch.nn.Tanh: default_fixed_qparams_range_neg1to1_observer,
    torch.tanh: default_fixed_qparams_range_neg1to1_observer,
    "tanh": default_fixed_qparams_range_neg1to1_observer,
    "tanh_": default_fixed_qparams_range_neg1to1_observer,
}

def _get_default_qconfig_mapping(is_qat: bool, backend: str, version: int) -> QConfigMapping:
    """
    Return the default QConfigMapping for the given quantization type and backend.
    """
    # Determine the appropriate qconfig based on whether it's quantization-aware training (qat) or not
    if is_qat:
        qconfig = get_default_qat_qconfig(backend, version)
    else:
        qconfig = get_default_qconfig(backend, version)
    
    # Choose the default weight observer based on whether it's qat or not
    default_weight = default_weight_fake_quant if is_qat else default_weight_observer

    # Modify the qconfig_transpose based on backend compatibility issues
    if backend in ("fbgemm", "x86"):
        qconfig_transpose = QConfig(activation=qconfig.activation, weight=default_weight)
    else:
        qconfig_transpose = qconfig

    # Currently layernorm only supports float weights
    # Return the computed QConfigMapping
    return qconfig_transpose
    # 创建一个 QConfig 对象，用于 LayerNorm 操作，设置激活函数和权重的量化配置
    qconfig_layernorm = QConfig(activation=qconfig.activation, weight=default_placeholder_observer)

    # 创建一个 QConfigMapping 对象，用于配置量化参数映射
    qconfig_mapping = QConfigMapping() \
        .set_global(qconfig) \  # 设置全局的量化配置
        .set_object_type("reshape", default_reuse_input_qconfig) \  # 设置特定类型操作的量化配置
        .set_object_type(torch.nn.ConvTranspose1d, qconfig_transpose) \
        .set_object_type(torch.nn.ConvTranspose2d, qconfig_transpose) \
        .set_object_type(torch.nn.ConvTranspose3d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose1d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose2d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.conv_transpose3d, qconfig_transpose) \
        .set_object_type(torch.nn.functional.layer_norm, qconfig_layernorm) \  # 设置特定函数的量化配置
        .set_object_type(torch.nn.LayerNorm, qconfig_layernorm) \
        .set_object_type(torch.nn.PReLU, default_quint8_weight_qconfig) \  # 设置特定类型操作的量化配置

    # 使用特定的观察器为固定量化参数的操作设置量化配置
    fixed_qparams_observer_to_qconfig: Dict[Any, QConfigAny] = {}
    for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        if observer in fixed_qparams_observer_to_qconfig:
            fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[observer]
        else:
            if is_qat:
                # 如果是量化感知训练，使用 FixedQParamsFakeQuantize 观察器
                activation = FixedQParamsFakeQuantize.with_args(observer=observer)
            else:
                activation = observer
            fixed_qparams_qconfig = QConfig(activation=activation, weight=default_weight)
            fixed_qparams_observer_to_qconfig[observer] = fixed_qparams_qconfig
        # 设置固定量化参数操作的量化配置
        qconfig_mapping.set_object_type(fixed_qparams_op, fixed_qparams_qconfig)

    # 返回配置好的 qconfig_mapping 对象，用于量化模型
    return qconfig_mapping
# 返回用于后训练量化的默认 QConfigMapping。
def get_default_qconfig_mapping(backend="x86", version=0) -> QConfigMapping:
    """
    Return the default QConfigMapping for post training quantization.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    # TODO: add assert for backend choices
    # 调用内部函数 _get_default_qconfig_mapping，返回默认的 QConfigMapping
    return _get_default_qconfig_mapping(False, backend, version)

# 返回用于量化感知训练的默认 QConfigMapping。
def get_default_qat_qconfig_mapping(backend="x86", version=1) -> QConfigMapping:
    """
    Return the default QConfigMapping for quantization aware training.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    # 调用内部函数 _get_default_qconfig_mapping，返回用于量化感知训练的默认 QConfigMapping
    return _get_default_qconfig_mapping(True, backend, version)

# 返回一个使用 torch.ao.quantization.default_symmetric_qnnpack_qconfig 的 QConfigMapping。
def _get_symmetric_qnnpack_qconfig_mapping() -> QConfigMapping:
    """
    Return a QConfigMapping that uses `torch.ao.quantization.default_symmetric_qnnpack_qconfig`
    as the default QConfig.
    """
    # 获取默认的对称 QNNPACK QConfig
    default_qconfig = default_symmetric_qnnpack_qconfig
    # 调用内部函数 _get_default_qconfig_mapping_with_default_qconfig，返回使用默认 QConfig 的 QConfigMapping
    return _get_default_qconfig_mapping_with_default_qconfig(False, "qnnpack", default_qconfig)

# 返回一个使用 torch.ao.quantization.default_symmetric_qnnpack_qat_qconfig 的 QConfigMapping。
def _get_symmetric_qnnpack_qat_qconfig_mapping() -> QConfigMapping:
    """
    Return a QConfigMapping that uses `torch.ao.quantization.default_symmetric_qnnpack_qat_qconfig`
    as the default QConfig.
    """
    # 获取默认的对称 QNNPACK QAT QConfig
    default_qconfig = default_symmetric_qnnpack_qat_qconfig
    # 调用内部函数 _get_default_qconfig_mapping_with_default_qconfig，返回使用默认 QConfig 的 QConfigMapping
    return _get_default_qconfig_mapping_with_default_qconfig(True, "qnnpack", default_qconfig)

# 返回一个使用提供的 qconfig 作为默认 QConfig 的 QConfigMapping。
def _get_default_qconfig_mapping_with_default_qconfig(
    is_qat: bool,
    backend: str,
    default_qconfig: QConfig,
) -> QConfigMapping:
    """
    Return a QConfigMapping that uses the provided qconfig as the default QConfig.
    """
    # 如果是量化感知训练，获取相应的 QConfigMapping
    if is_qat:
        qconfig_mapping = get_default_qat_qconfig_mapping(backend)
    else:
        qconfig_mapping = get_default_qconfig_mapping(backend)
    # 设置全局的默认 QConfig
    qconfig_mapping.set_global(default_qconfig)
    # 遍历对象类型的 QConfig 映射，如果映射不在 _FIXED_QPARAMS_OP_TO_OBSERVER 中，则设置为默认 QConfig
    for pattern in qconfig_mapping.object_type_qconfigs.keys():
        if pattern not in _FIXED_QPARAMS_OP_TO_OBSERVER:
            qconfig_mapping.set_object_type(pattern, default_qconfig)
    # 返回设置后的 QConfigMapping
    return qconfig_mapping

# 定义 QConfigMapping 对象，映射模型操作到 torch.ao.quantization.QConfig。
_QCONFIG_STYLE_ORDER: List[str] = [
    "global_qconfig",
    "object_type_qconfigs",
    "module_name_regex_qconfigs",
    "module_name_qconfigs",
    "module_name_object_type_order_qconfigs",
]

class QConfigMapping:
    """
    Mapping from model ops to :class:`torch.ao.quantization.QConfig` s.
    """
    def __init__(self):
        # 初始化 QConfigMapping 类的实例，定义各种类型的配置字典，按优先级递增排列：
        self.global_qconfig: QConfigAny = None  # 全局默认的 QConfig
        self.object_type_qconfigs: OrderedDict[Union[Callable, str], QConfigAny] = OrderedDict()  # 按对象类型指定的 QConfig
        self.module_name_regex_qconfigs: OrderedDict[str, QConfigAny] = OrderedDict()  # 按模块名正则表达式指定的 QConfig
        self.module_name_qconfigs: OrderedDict[str, QConfigAny] = OrderedDict()  # 按精确模块名指定的 QConfig
        self.module_name_object_type_order_qconfigs: OrderedDict[Tuple[str, Callable, int], QConfigAny] =\
            OrderedDict()  # 按模块名、对象类型和出现顺序指定的 QConfig

    def set_global(self, global_qconfig: QConfigAny) -> QConfigMapping:
        """
        设置全局默认的 QConfig。
        """
        self.global_qconfig = global_qconfig
        return self

    def set_object_type(self, object_type: Union[Callable, str], qconfig: QConfigAny) -> QConfigMapping:
        """
        为特定的模块类型、函数或方法名设置 QConfig。
        如果已经为现有对象类型设置了 QConfig，则新的 QConfig 将覆盖旧的。
        """
        self.object_type_qconfigs[object_type] = qconfig
        return self
    # 设置匹配给定模块名正则表达式的模块的量化配置（QConfig）。
    # 如果已经设置了现有模块名正则表达式的QConfig，则新的QConfig将覆盖旧的QConfig，
    # 同时保留最初注册正则表达式的顺序。
    def set_module_name_regex(self, module_name_regex: str, qconfig: QConfigAny) -> QConfigMapping:
        self.module_name_regex_qconfigs[module_name_regex] = qconfig
        return self

    # 设置匹配给定模块名的模块的量化配置（QConfig）。
    # 如果已经设置了现有模块名的QConfig，则新的QConfig将覆盖旧的QConfig。
    def set_module_name(self, module_name: str, qconfig: QConfigAny) -> QConfigMapping:
        self.module_name_qconfigs[module_name] = qconfig
        return self

    # 设置匹配给定模块名、对象类型和模块出现的索引的组合的模块的量化配置（QConfig）。
    # 如果已经设置了现有（模块名、对象类型、索引）的QConfig，则新的QConfig将覆盖旧的QConfig。
    def set_module_name_object_type_order(
            self,
            module_name: str,
            object_type: Callable,
            index: int,
            qconfig: QConfigAny) -> QConfigMapping:
        self.module_name_object_type_order_qconfigs[(module_name, object_type, index)] = qconfig
        return self

    # 返回此对象的字符串表示形式，用于调试和打印。
    # 输出包含类名和其持有的量化配置（QConfig），按预定义顺序进行格式化。
    def __repr__(self) -> str:
        output = self.__class__.__name__ + " ("
        for style_name in _QCONFIG_STYLE_ORDER:
            output += f"\n {style_name}"
            qconfigs = getattr(self, style_name)
            if isinstance(qconfigs, OrderedDict) and len(qconfigs) > 0:
                for key, qconfig in qconfigs.items():
                    output += f"\n  {key}: {qconfig}"
            else:
                output += f"\n  {qconfigs}"
        return output + "\n)"

    # TODO: remove this
    # 将当前的 QConfigMapping 对象转换为字典形式并返回
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``QConfigMapping`` to a dictionary with the following keys:

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are lists of tuples.
        """
        return {
            _GLOBAL_DICT_KEY: self.global_qconfig,  # 存储全局 QConfig 的键值对列表
            _OBJECT_TYPE_DICT_KEY: list(self.object_type_qconfigs.items()),  # 存储对象类型到 QConfig 的映射列表
            _MODULE_NAME_REGEX_DICT_KEY: list(self.module_name_regex_qconfigs.items()),  # 存储模块名称正则到 QConfig 的映射列表
            _MODULE_NAME_DICT_KEY: list(self.module_name_qconfigs.items()),  # 存储模块名称到 QConfig 的映射列表
            _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY: [
                (*k, v) for k, v in self.module_name_object_type_order_qconfigs.items()
            ],  # 存储模块名称、对象类型、顺序索引到 QConfig 的映射列表
        }

    # TODO: remove this
    @classmethod
    def from_dict(cls, qconfig_dict: Dict[str, Any]) -> QConfigMapping:
        """
        Create a ``QConfigMapping`` from a dictionary with the following keys (all optional):

            "" (for global QConfig)

            "object_type"

            "module_name_regex"

            "module_name"

            "module_name_object_type_order"

        The values of this dictionary are expected to be lists of tuples.
        """
        # 创建一个 QConfigMapping 对象
        conf = cls()
        # 如果全局 QConfig 存在于输入字典中，则设置全局 QConfig
        if _GLOBAL_DICT_KEY in qconfig_dict:
            conf.set_global(qconfig_dict[_GLOBAL_DICT_KEY])
        # 遍历并设置对象类型到 QConfig 的映射
        for object_type, qconfig in qconfig_dict.get(_OBJECT_TYPE_DICT_KEY, []):
            conf.set_object_type(object_type, qconfig)
        # 遍历并设置模块名称正则到 QConfig 的映射
        for module_name_regex, qconfig in qconfig_dict.get(_MODULE_NAME_REGEX_DICT_KEY, []):
            conf.set_module_name_regex(module_name_regex, qconfig)
        # 遍历并设置模块名称到 QConfig 的映射
        for module_name, qconfig in qconfig_dict.get(_MODULE_NAME_DICT_KEY, []):
            conf.set_module_name(module_name, qconfig)
        # 遍历并设置模块名称、对象类型、顺序索引到 QConfig 的映射
        for module_name, object_type, index, qconfig in qconfig_dict.get(_MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY, []):
            conf.set_module_name_object_type_order(module_name, object_type, index, qconfig)
        return conf
```