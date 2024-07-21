# `.\pytorch\torch\ao\quantization\backend_config\backend_config.py`

```py
# mypy: allow-untyped-defs
# 从未来导入类型注解支持，允许未标记的函数定义
from __future__ import annotations
# 导入 dataclass 用于定义数据类，以及类型提示相关的模块
from dataclasses import dataclass
# 导入 Any、Callable、Dict、List、Optional、Type、Union 等类型
from typing import Any, Callable, Dict, List, Optional, Type, Union, TYPE_CHECKING
# 导入 PyTorch 深度学习库
import torch
# 导入枚举类型支持
from enum import Enum

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入用于量化的 Torch 工具库中的 Pattern 类型
    from torch.ao.quantization.utils import Pattern

# __all__ 列表声明，指定在使用 from module import * 时导入的符号
__all__ = [
    "BackendConfig",
    "BackendPatternConfig",
    "DTypeConfig",
    "DTypeWithConstraints",
    "ObservationType",
]

# DTypeConfig 字典键值
INPUT_DTYPE_DICT_KEY = "input_dtype"
OUTPUT_DTYPE_DICT_KEY = "output_dtype"
WEIGHT_DTYPE_DICT_KEY = "weight_dtype"
BIAS_DTYPE_DICT_KEY = "bias_dtype"
IS_DYNAMIC_DICT_KEY = "is_dynamic"

# BackendConfig 字典键值
NAME_DICT_KEY = "name"
CONFIGS_DICT_KEY = "configs"

# BackendPatternConfig 字典键值
PATTERN_DICT_KEY = "pattern"
PATTERN_COMPLEX_FORMAT_DICT_KEY = "pattern_complex_format"
OBSERVATION_TYPE_DICT_KEY = "observation_type"
DTYPE_CONFIGS_DICT_KEY = "dtype_configs"
ROOT_MODULE_DICT_KEY = "root_module"
QAT_MODULE_DICT_KEY = "qat_module"
REFERENCE_QUANTIZED_MODULE_DICT_KEY = "reference_quantized_module_for_root"
FUSED_MODULE_DICT_KEY = "fused_module"
FUSER_METHOD_DICT_KEY = "fuser_method"
ROOT_NODE_GETTER_DICT_KEY = "root_node_getter"
EXTRA_INPUTS_GETTER_DICT_KEY = "extra_inputs_getter"
NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY = "num_tensor_args_to_observation_type"
INPUT_TYPE_TO_INDEX_DICT_KEY = "input_type_to_index"

# TODO: maybe rename this to something that's not related to observer
# e.g. QParamsType
# 观察类型的枚举，用于表示运算符或运算符模式的不同观察方式
class ObservationType(Enum):
    """ An enum that represents different ways of how an operator/operator pattern
    should be observed
    """

    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    """this means input and output are observed with different observers, based
    on qconfig.activation
    example: conv, linear, softmax
    """

    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    """this means the output will use the same observer instance as input, based
    on qconfig.activation
    example: torch.cat, maxpool
    """

    INPUT_OUTPUT_NOT_OBSERVED = 2
    """this means the input and output are never observed
    example: x.shape, x.size
    """

# 数据类，用于指定给定数据类型的额外约束的配置，例如量化值范围、缩放值范围和固定量化参数
@dataclass
class DTypeWithConstraints:
    """
    Config for specifying additional constraints for a given dtype, such as quantization
    value ranges, scale value ranges, and fixed quantization params, to be used in
    :class:`~torch.ao.quantization.backend_config.DTypeConfig`.

    The constraints currently supported are:

    * `quant_min_lower_bound` and `quant_max_upper_bound`: Lower and upper
      bounds for the minimum and maximum quantized values respectively. If
      the QConfig's `quant_min` and `quant_max` fall outside this range,
      then the QConfig will be ignored.
    """
    # 数据类型（dtype）：用于指定张量的数据类型，可选，默认为None。
    dtype: Optional[torch.dtype] = None
    
    # 最小量化值`
    # Optional data type for the configuration parameters, defaults to None
    dtype: Optional[torch.dtype] = None
    
    # Lower bound for the minimum quantization value allowed in the QConfig
    quant_min_lower_bound: Union[int, float, None] = None
    
    # Upper bound for the maximum quantization value allowed in the QConfig
    quant_max_upper_bound: Union[int, float, None] = None
    
    # Lower bound for the minimum scale value in the QConfig
    # If the minimum scale value (eps) falls below this bound, the QConfig will be ignored
    scale_min_lower_bound: Union[int, float, None] = None
    
    # Upper bound for the maximum scale value in the QConfig
    # Currently not enforced
    scale_max_upper_bound: Union[int, float, None] = None
    
    # Exact scale value required for operators with fixed quantization parameters
    # Examples include sigmoid and tanh
    scale_exact_match: Optional[float] = None
    
    # Exact zero point value required for operators with fixed quantization parameters
    zero_point_exact_match: Optional[int] = None
# 使用 dataclass 装饰器创建一个数据类 DTypeConfig，用于配置量化操作中支持的数据类型
@dataclass
class DTypeConfig:
    """
    Config object that specifies the supported data types passed as arguments to
    quantize ops in the reference model spec, for input and output activations,
    weights, and biases.

    For example, consider the following reference model:

      quant1 - [dequant1 - fp32_linear - quant2] - dequant2

    The pattern in the square brackets refers to the reference pattern of
    statically quantized linear. Setting the input dtype as `torch.quint8`
    in the DTypeConfig means we pass in `torch.quint8` as the dtype argument
    to the first quantize op (quant1). Similarly, setting the output dtype as
    `torch.quint8` means we pass in `torch.quint8` as the dtype argument to
    the second quantize op (quant2).

    Note that the dtype here does not refer to the interface dtypes of the
    op. For example, the "input dtype" here is not the dtype of the input
    tensor passed to the quantized linear op. Though it can still be the
    same as the interface dtype, this is not always the case, e.g. the
    interface dtype is fp32 in dynamic quantization but the "input dtype"
    specified in the DTypeConfig would still be quint8. The semantics of
    dtypes here are the same as the semantics of the dtypes specified in
    the observers.

    These dtypes are matched against the ones specified in the user's
    QConfig. If there is a match, and the QConfig satisfies the constraints
    specified in the DTypeConfig (if any), then we will quantize the given
    pattern using this DTypeConfig. Otherwise, the QConfig is ignored and
    the pattern will not be quantized.

    Example usage::

        >>> # xdoctest: +SKIP(failing)
        >>> dtype_config1 = DTypeConfig(
        ...     input_dtype=torch.quint8,
        ...     output_dtype=torch.quint8,
        ...     weight_dtype=torch.qint8,
        ...     bias_dtype=torch.float)

        >>> dtype_config2 = DTypeConfig(
        ...     input_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     output_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     weight_dtype=DTypeWithConstraints(
        ...         dtype=torch.qint8,
        ...         quant_min_lower_bound=-128,
        ...         quant_max_upper_bound=127,
        ...     ),
        ...     bias_dtype=torch.float)

        >>> dtype_config1.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype_with_constraints
        DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, \
scale_min_lower_bound=None, scale_max_upper_bound=None)
    """
    # 定义属性 input_dtype_with_constraints，用于存储输入数据类型和相关约束的配置信息
    input_dtype_with_constraints: DTypeWithConstraints
    # 声明一个实例变量，用于存储输入数据类型和约束条件
    output_dtype_with_constraints: DTypeWithConstraints
    # 声明一个实例变量，用于存储输出数据类型和约束条件
    weight_dtype_with_constraints: DTypeWithConstraints
    # 声明一个实例变量，用于存储偏置项数据类型（可选）
    bias_dtype: Optional[torch.dtype]
    # 声明一个实例变量，用于标记是否为动态模型（可选）
    is_dynamic: Optional[bool]

    # 初始化方法，接受多个参数来设定数据类型和约束条件
    def __init__(
        self,
        input_dtype: Union[torch.dtype, DTypeWithConstraints, None] = None,
        output_dtype: Union[torch.dtype, DTypeWithConstraints, None] = None,
        weight_dtype: Union[torch.dtype, DTypeWithConstraints, None] = None,
        bias_dtype: Optional[torch.dtype] = None,
        is_dynamic: Optional[bool] = None,
    ):
        # 如果输入数据类型是 DTypeWithConstraints 类型，则直接赋值给实例变量
        if isinstance(input_dtype, DTypeWithConstraints):
            self.input_dtype_with_constraints = input_dtype
        # 否则，用输入数据类型创建一个新的 DTypeWithConstraints 对象，并赋值给实例变量
        else:
            self.input_dtype_with_constraints = DTypeWithConstraints(dtype=input_dtype)

        # 类似地处理输出数据类型
        if isinstance(output_dtype, DTypeWithConstraints):
            self.output_dtype_with_constraints = output_dtype
        else:
            self.output_dtype_with_constraints = DTypeWithConstraints(dtype=output_dtype)

        # 类似地处理权重数据类型
        if isinstance(weight_dtype, DTypeWithConstraints):
            self.weight_dtype_with_constraints = weight_dtype
        else:
            self.weight_dtype_with_constraints = DTypeWithConstraints(dtype=weight_dtype)

        # 将偏置数据类型和是否为动态模型的标志直接赋值给实例变量
        self.bias_dtype = bias_dtype
        self.is_dynamic = is_dynamic

    # 获取输入数据类型的属性方法
    @property
    def input_dtype(self) -> Optional[torch.dtype]:
        return self.input_dtype_with_constraints.dtype

    # 获取输出数据类型的属性方法
    @property
    def output_dtype(self) -> Optional[torch.dtype]:
        return self.output_dtype_with_constraints.dtype

    # 获取权重数据类型的属性方法
    @property
    def weight_dtype(self) -> Optional[torch.dtype]:
        return self.weight_dtype_with_constraints.dtype

    # 定义一个类方法，用于实现类级别的操作
    @classmethod
    def from_dict(cls, dtype_config_dict: Dict[str, Any]) -> DTypeConfig:
        """
        从给定的字典创建一个 ``DTypeConfig`` 对象，字典包含以下可选项：
            "input_dtype": torch.dtype 或 ``DTypeWithConstraints``
            "output_dtype": torch.dtype 或 ``DTypeWithConstraints``
            "weight_dtype": torch.dtype 或 ``DTypeWithConstraints``
            "bias_dtype": torch.dtype
            "is_dynamic": bool
        """
        # 从字典中获取输入数据类型，如果不存在则为 None
        input_dtype = dtype_config_dict.get(INPUT_DTYPE_DICT_KEY, None)
        # 如果输入数据类型存在且不是 torch.dtype 或 DTypeWithConstraints 类型，则抛出 ValueError 异常
        if input_dtype is not None and not isinstance(input_dtype, (torch.dtype, DTypeWithConstraints)):
            raise ValueError("Expected input_dtype to be a torch.dtype or DTypeWithConstraints")
        
        # 从字典中获取输出数据类型，如果不存在则为 None
        output_dtype = dtype_config_dict.get(OUTPUT_DTYPE_DICT_KEY, None)
        # 如果输出数据类型存在且不是 torch.dtype 或 DTypeWithConstraints 类型，则抛出 ValueError 异常
        if output_dtype is not None and not isinstance(output_dtype, (torch.dtype, DTypeWithConstraints)):
            raise ValueError("Expected output_dtype to be a torch.dtype or DTypeWithConstraints")
        
        # 从字典中获取权重数据类型，如果不存在则为 None
        weight_dtype = dtype_config_dict.get(WEIGHT_DTYPE_DICT_KEY, None)
        # 如果权重数据类型存在且不是 torch.dtype 或 DTypeWithConstraints 类型，则抛出 ValueError 异常
        if weight_dtype is not None and not isinstance(weight_dtype, (torch.dtype, DTypeWithConstraints)):
            raise ValueError("Expected weight_dtype to be a torch.dtype or DTypeWithConstraints")
        
        # 从字典中获取偏置数据类型，如果不存在则为 None
        bias_dtype = dtype_config_dict.get(BIAS_DTYPE_DICT_KEY, None)
        
        # 从字典中获取是否为动态数据类型，如果不存在则为 None
        is_dynamic = dtype_config_dict.get(IS_DYNAMIC_DICT_KEY, None)
        
        # 使用类方法创建并返回 DTypeConfig 对象，传入各种数据类型参数
        return cls(input_dtype, output_dtype, weight_dtype, bias_dtype, is_dynamic)

    def to_dict(self) -> Dict[str, Any]:
        """
        将当前 ``DTypeConfig`` 对象转换为一个包含在 :func:`~torch.ao.quantization.backend_config.DTypeConfig.from_dict`
        中描述的项的字典。
        """
        # 创建一个空字典，用于存储转换后的数据类型配置信息
        dtype_config_dict: Dict[str, Any] = {}
        
        # 如果输入数据类型不为 None，则将其加入字典中
        if self.input_dtype is not None:
            dtype_config_dict[INPUT_DTYPE_DICT_KEY] = self.input_dtype_with_constraints
        
        # 如果输出数据类型不为 None，则将其加入字典中
        if self.output_dtype is not None:
            dtype_config_dict[OUTPUT_DTYPE_DICT_KEY] = self.output_dtype_with_constraints
        
        # 如果权重数据类型不为 None，则将其加入字典中
        if self.weight_dtype is not None:
            dtype_config_dict[WEIGHT_DTYPE_DICT_KEY] = self.weight_dtype_with_constraints
        
        # 如果偏置数据类型不为 None，则将其加入字典中
        if self.bias_dtype is not None:
            dtype_config_dict[BIAS_DTYPE_DICT_KEY] = self.bias_dtype
        
        # 如果动态数据类型不为 None，则将其加入字典中
        if self.is_dynamic is not None:
            dtype_config_dict[IS_DYNAMIC_DICT_KEY] = self.is_dynamic
        
        # 返回包含数据类型配置信息的字典
        return dtype_config_dict
class BackendConfig:
    # TODO: refer to NativeBackendConfig once that is implemented
    """Config that defines the set of patterns that can be quantized on a given backend, and how reference
    quantized models can be produced from these patterns.

    A pattern in this context refers to a module, a functional, an operator, or a directed acyclic graph
    of the above. Each pattern supported on the target backend can be individually configured through
    :class:`~torch.ao.quantization.backend_config.BackendPatternConfig` in terms of:

    (1) The supported input/output activation, weight, and bias data types

    (2) How observers and quant/dequant ops are inserted in order to construct the reference pattern, and

    (3) (Optionally) Fusion, QAT, and reference module mappings.

    The format of the patterns is described in:
    https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md
    """



# TODO: refer to NativeBackendConfig once that is implemented
"""Config that defines the set of patterns that can be quantized on a given backend, and how reference
quantized models can be produced from these patterns.

A pattern in this context refers to a module, a functional, an operator, or a directed acyclic graph
of the above. Each pattern supported on the target backend can be individually configured through
:class:`~torch.ao.quantization.backend_config.BackendPatternConfig` in terms of:

(1) The supported input/output activation, weight, and bias data types

(2) How observers and quant/dequant ops are inserted in order to construct the reference pattern, and

(3) (Optionally) Fusion, QAT, and reference module mappings.

The format of the patterns is described in:
https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md
"""
    """
    构造函数，初始化一个 BackendConfig 对象，可以指定后端名称。

    Parameters:
    - name: str, 后端的名称，默认为空字符串

    Attributes:
    - name: str, 后端的名称

    Notes:
    - 使用了一个字典来存储 BackendPatternConfig，以处理可能存在的重复配置。
    - 键值在这个字典中使用了复杂的反转元组格式，这仅用于内部使用；
      用户想要访问原始模式应通过 `self.configs` 属性来获取。

    Returns:
    - None
    """
    def __init__(self, name: str = ""):
        self.name = name
        # Store all BackendPatternConfigs in a map to handle duplicates
        # Note: the key in this map uses the complex reversed tuple format.
        # This is intended only for internal use; users who wish to access
        # the original patterns should go through `self.configs` instead.
        self._pattern_complex_format_to_config: Dict[Pattern, BackendPatternConfig] = {}

    """
    返回一个字符串，用于描述 BackendConfig 对象的内容。

    Parameters:
    - None

    Returns:
    - str, 描述该 BackendConfig 对象的字符串表示形式，包括其属性和状态
    """
    def __repr__(self):
        return f"BackendConfig({self.__dict__})"

    """
    设置目标后端的名称。

    Parameters:
    - name: str, 后端的名称

    Returns:
    - BackendConfig, 返回修改后的 BackendConfig 对象自身
    """
    def set_name(self, name: str) -> BackendConfig:
        """
        Set the name of the target backend.
        """
        self.name = name
        return self
    def set_backend_pattern_config(self, config: BackendPatternConfig) -> BackendConfig:
        """
        Set the config for a pattern that can be run on the target backend.
        This overrides any existing config for the given pattern.
        """
        # 获取模式的复杂格式，以便于处理循环依赖
        pattern_complex_format = torch.ao.quantization.backend_config.utils \
            ._get_pattern_in_reversed_nested_tuple_format(config)  # type: ignore[attr-defined]
        # 将模式的复杂格式映射到配置对象，覆盖现有配置
        self._pattern_complex_format_to_config[pattern_complex_format] = config
        # 返回当前实例，支持方法链调用
        return self

    def set_backend_pattern_configs(self, configs: List[BackendPatternConfig]) -> BackendConfig:
        """
        Set the configs for patterns that can be run on the target backend.
        This overrides any existing config for a given pattern if it was previously registered already.
        """
        # 遍历传入的配置列表，依次调用set_backend_pattern_config方法
        for conf in configs:
            self.set_backend_pattern_config(conf)
        # 返回当前实例，支持方法链调用
        return self

    @property
    def configs(self) -> List[BackendPatternConfig]:
        """
        Return a copy of the list of configs set in this `BackendConfig`.
        """
        # 返回当前实例中存储的配置对象列表的副本
        return list(self._pattern_complex_format_to_config.values())

    @classmethod
    def from_dict(cls, backend_config_dict: Dict[str, Any]) -> BackendConfig:
        """
        Create a ``BackendConfig`` from a dictionary with the following items:

            "name": the name of the target backend

            "configs": a list of dictionaries that each represents a `BackendPatternConfig`

        """
        # 从给定的字典创建BackendConfig实例
        conf = cls(backend_config_dict.get(NAME_DICT_KEY, ""))
        # 遍历configs键对应的值（配置列表），将每个配置添加到BackendConfig实例中
        for d in backend_config_dict.get(CONFIGS_DICT_KEY, []):
            if isinstance(d, BackendPatternConfig):
                conf.set_backend_pattern_config(d)
            elif isinstance(d, Dict):
                conf.set_backend_pattern_config(BackendPatternConfig.from_dict(d))
            else:
                # 如果配置不是BackendPatternConfig或字典，则引发值错误
                raise ValueError(f"Expected backend_config_dict['{CONFIGS_DICT_KEY}'] to be a dictionary")
        # 返回配置好的BackendConfig实例
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``BackendConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendConfig.from_dict`.
        """
        # 将当前BackendConfig实例转换为字典形式
        return {
            NAME_DICT_KEY: self.name,
            CONFIGS_DICT_KEY: [c.to_dict() for c in self.configs],
        }
class BackendPatternConfig:
    """
    Config object that specifies quantization behavior for a given operator pattern.
    For a detailed example usage, see :class:`~torch.ao.quantization.backend_config.BackendConfig`.
    """

    def __init__(self, pattern: Optional[Pattern] = None):
        # 初始化方法，设置模式（可以为空）
        self.pattern: Optional[Pattern] = pattern
        # 设置观察类型为：输出使用不同观察者作为输入
        self.observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
        # 初始化空的数据类型配置列表
        self.dtype_configs: List[DTypeConfig] = []
        # 根模块、量化训练后模块、参考量化模块、融合模块初始设为空
        self.root_module: Optional[Type[torch.nn.Module]] = None
        self.qat_module: Optional[Type[torch.nn.Module]] = None
        self.reference_quantized_module: Optional[Type[torch.nn.Module]] = None
        self.fused_module: Optional[Type[torch.nn.Module]] = None
        # 融合方法初始设为空
        self.fuser_method: Optional[Callable] = None

        # 临时/内部配置
        self._root_node_getter: Optional[Callable] = None
        self._extra_inputs_getter: Optional[Callable] = None
        self._num_tensor_args_to_observation_type: Dict[int, ObservationType] = {}
        self._input_type_to_index: Dict[str, int] = {}
        self._pattern_complex_format: Optional[Pattern] = None

    def __repr__(self):
        # 返回一个表示对象的字符串，仅包括非空字段
        dict_nonempty = {
            k: v for k, v in self.__dict__.items()
            if (
                (not isinstance(v, (list, dict)) and v is not None)
                or (isinstance(v, (list, dict)) and len(v) > 0)
            )
        }
        return f"BackendPatternConfig({dict_nonempty})"

    def set_pattern(self, pattern: Pattern) -> BackendPatternConfig:
        """
        Set the pattern to configure.

        The pattern can be a float module, functional operator, pytorch operator, or a tuple
        combination of the above. Tuple patterns are treated as sequential patterns, and
        currently only tuples of 2 or 3 elements are supported.
        """
        # 设置要配置的模式，如果已有复杂格式模式，则引发值错误异常
        if self._pattern_complex_format is not None:
            raise ValueError("Only one of 'pattern' or 'pattern_complex_format' can be set")
        self.pattern = pattern
        return self
    def set_observation_type(self, observation_type: ObservationType) -> BackendPatternConfig:
        """
        Set how observers should be inserted in the graph for this pattern.

        Observation type here refers to how observers (or quant-dequant ops) will be placed
        in the graph. This is used to produce the desired reference patterns understood by
        the backend. Weighted ops such as linear and conv require different observers
        (or quantization parameters passed to quantize ops in the reference model) for the
        input and the output.

        There are two observation types:

            `OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT` (default): the output observer instance
            will be different from the input. This is the most common observation type.

            `OUTPUT_SHARE_OBSERVER_WITH_INPUT`: the output observer instance will be the
            same as the input. This is useful for operators like `cat`.

        Note: This will be renamed in the near future, since we will soon insert QuantDeQuantStubs
        with observers (and fake quantizes) attached instead of observers themselves.
        """
        # 设置观察类型，决定如何在图中插入观察器（或量化-反量化操作）
        self.observation_type = observation_type
        return self

    def add_dtype_config(self, dtype_config: DTypeConfig) -> BackendPatternConfig:
        """
        Add a set of supported data types passed as arguments to quantize ops in the
        reference model spec.
        """
        # 添加一组支持的数据类型配置，作为参数传递给参考模型规范中的量化操作
        self.dtype_configs.append(dtype_config)
        return self

    def set_dtype_configs(self, dtype_configs: List[DTypeConfig]) -> BackendPatternConfig:
        """
        Set the supported data types passed as arguments to quantize ops in the
        reference model spec, overriding all previously registered data types.
        """
        # 设置支持的数据类型配置，作为参数传递给参考模型规范中的量化操作，并覆盖之前注册的所有数据类型
        self.dtype_configs = dtype_configs
        return self

    def set_root_module(self, root_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        Set the module that represents the root for this pattern.

        When we construct the reference quantized model during the convert phase,
        the root modules (e.g. torch.nn.Linear for torch.ao.nn.intrinsic.LinearReLU)
        will be swapped to the corresponding reference quantized modules (e.g.
        torch.ao.nn.reference.quantized.Linear). This allows custom backends to
        specify custom reference quantized module implementations to match the
        numerics of their lowered operators. Since this is a one-to-one mapping,
        both the root module and the reference quantized module must be specified
        in the same BackendPatternConfig in order for the conversion to take place.
        """
        # 设置代表此模式根节点的模块。

        # 在转换阶段构建参考量化模型时，根模块（例如 torch.nn.Linear 对应于 torch.ao.nn.intrinsic.LinearReLU）
        # 将会被替换为相应的参考量化模块（例如 torch.ao.nn.reference.quantized.Linear）。
        # 这允许自定义后端指定自定义参考量化模块实现，以匹配其降低操作的数值计算。因为这是一对一的映射，
        # 根模块和参考量化模块必须在同一个 BackendPatternConfig 中指定，以便进行转换。
        self.root_module = root_module
        return self
    def set_qat_module(self, qat_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        设置表示该模式的量化感知训练（QAT）实现的模块。
        """
        self.qat_module = qat_module
        return self

    def set_reference_quantized_module(self, reference_quantized_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        设置表示该模式根模块的参考量化实现的模块。

        更多详情，请参阅 :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module`。
        """
        self.reference_quantized_module = reference_quantized_module
        return self

    def set_fused_module(self, fused_module: Type[torch.nn.Module]) -> BackendPatternConfig:
        """
        设置表示该模式的融合实现的模块。
        """
        self.fused_module = fused_module
        return self

    def set_fuser_method(self, fuser_method: Callable) -> BackendPatternConfig:
        """
        设置用于融合此 BackendPatternConfig 模式的函数。

        该函数的第一个参数应为 `is_qat`，其余参数应为元组模式中的项目。
        函数的返回值应为生成的融合模块。

        例如，对于模式 `(torch.nn.Linear, torch.nn.ReLU)` 的融合方法可以是：

            def fuse_linear_relu(is_qat, linear, relu):
                return torch.ao.nn.intrinsic.LinearReLU(linear, relu)

        更复杂的示例，请参阅 https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6。
        """
        self.fuser_method = fuser_method
        return self

    def _set_root_node_getter(self, root_node_getter: Callable) -> BackendPatternConfig:
        """
        设置获取根节点的函数。
        """
        self._root_node_getter = root_node_getter
        return self

    def _set_extra_inputs_getter(self, extra_inputs_getter: Callable) -> BackendPatternConfig:
        """
        设置获取额外输入的函数。
        """
        self._extra_inputs_getter = extra_inputs_getter
        return self

    def _set_num_tensor_args_to_observation_type(
            self, num_tensor_args_to_observation_type: Dict[int, ObservationType]) -> BackendPatternConfig:
        """
        设置将张量参数数量映射到观察类型的字典。
        """
        self._num_tensor_args_to_observation_type = num_tensor_args_to_observation_type
        return self

    def _set_input_type_to_index(self, input_type_to_index: Dict[str, int]) -> BackendPatternConfig:
        """
        设置将输入类型映射到索引的字典。
        """
        self._input_type_to_index = input_type_to_index
        return self
    # 将复杂格式的模式设置到配置中，使用反向嵌套元组格式

    def _set_pattern_complex_format(self, pattern: Pattern) -> BackendPatternConfig:
        """
        Set the pattern to configure, using the reversed nested tuple format.

        See the BackendConfig README for more detail:
        https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md#advanced-pattern-specification
        """
        # 如果已经设置了 pattern，则抛出数值错误异常
        if self.pattern is not None:
            raise ValueError("Only one of 'pattern' or 'pattern_complex_format' can be set")
        # 将 pattern 设置为指定的 pattern
        self._pattern_complex_format = pattern
        # 返回当前对象的引用，以便支持链式调用
        return self

    @classmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``BackendPatternConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.from_dict`.
        """
        # 创建一个空字典来存储 BackendPatternConfig 对象的属性
        backend_pattern_config_dict: Dict[str, Any] = {
            OBSERVATION_TYPE_DICT_KEY: self.observation_type,
            DTYPE_CONFIGS_DICT_KEY: [c.to_dict() for c in self.dtype_configs],
        }
        # 如果设置了 pattern 属性，则将其添加到字典中
        if self.pattern is not None:
            backend_pattern_config_dict[PATTERN_DICT_KEY] = self.pattern
        # 如果设置了 root_module 属性，则将其添加到字典中
        if self.root_module is not None:
            backend_pattern_config_dict[ROOT_MODULE_DICT_KEY] = self.root_module
        # 如果设置了 qat_module 属性，则将其添加到字典中
        if self.qat_module is not None:
            backend_pattern_config_dict[QAT_MODULE_DICT_KEY] = self.qat_module
        # 如果设置了 reference_quantized_module 属性，则将其添加到字典中
        if self.reference_quantized_module is not None:
            backend_pattern_config_dict[REFERENCE_QUANTIZED_MODULE_DICT_KEY] = self.reference_quantized_module
        # 如果设置了 fused_module 属性，则将其添加到字典中
        if self.fused_module is not None:
            backend_pattern_config_dict[FUSED_MODULE_DICT_KEY] = self.fused_module
        # 如果设置了 fuser_method 属性，则将其添加到字典中
        if self.fuser_method is not None:
            backend_pattern_config_dict[FUSER_METHOD_DICT_KEY] = self.fuser_method
        # 如果设置了 _root_node_getter 属性，则将其添加到字典中
        if self._root_node_getter is not None:
            backend_pattern_config_dict[ROOT_NODE_GETTER_DICT_KEY] = self._root_node_getter
        # 如果设置了 _extra_inputs_getter 属性，则将其添加到字典中
        if self._extra_inputs_getter is not None:
            backend_pattern_config_dict[EXTRA_INPUTS_GETTER_DICT_KEY] = self._extra_inputs_getter
        # 如果 _num_tensor_args_to_observation_type 字典不为空，则将其添加到字典中
        if len(self._num_tensor_args_to_observation_type) > 0:
            backend_pattern_config_dict[NUM_TENSOR_ARGS_TO_OBSERVATION_TYPE_DICT_KEY] = self._num_tensor_args_to_observation_type
        # 如果 _input_type_to_index 字典不为空，则将其添加到字典中
        if len(self._input_type_to_index) > 0:
            backend_pattern_config_dict[INPUT_TYPE_TO_INDEX_DICT_KEY] = self._input_type_to_index
        # 如果设置了 _pattern_complex_format 属性，则将其添加到字典中
        if self._pattern_complex_format is not None:
            backend_pattern_config_dict[PATTERN_COMPLEX_FORMAT_DICT_KEY] = self._pattern_complex_format
        # 返回包含 BackendPatternConfig 对象所有属性的字典
        return backend_pattern_config_dict
```