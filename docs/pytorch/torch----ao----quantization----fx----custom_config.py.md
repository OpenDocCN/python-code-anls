# `.\pytorch\torch\ao\quantization\fx\custom_config.py`

```py
# mypy: allow-untyped-defs
# 从未来导入注解，允许未类型化的定义
from __future__ import annotations
# 导入用于数据类的装饰器
from dataclasses import dataclass
# 导入类型相关的工具
from typing import Any, Dict, List, Optional, Tuple, Type

# 导入量化相关的模块和类
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str

# 模块的公开接口列表
__all__ = [
    "ConvertCustomConfig",
    "FuseCustomConfig",
    "PrepareCustomConfig",
    "StandaloneModuleConfigEntry",
]

# 常量定义，用于替换现有代码中的字符串
STANDALONE_MODULE_NAME_DICT_KEY = "standalone_module_name"
STANDALONE_MODULE_CLASS_DICT_KEY = "standalone_module_class"
FLOAT_TO_OBSERVED_DICT_KEY = "float_to_observed_custom_module_class"
OBSERVED_TO_QUANTIZED_DICT_KEY = "observed_to_quantized_custom_module_class"
NON_TRACEABLE_MODULE_NAME_DICT_KEY = "non_traceable_module_name"
NON_TRACEABLE_MODULE_CLASS_DICT_KEY = "non_traceable_module_class"
INPUT_QUANTIZED_INDEXES_DICT_KEY = "input_quantized_idxs"
OUTPUT_QUANTIZED_INDEXES_DICT_KEY = "output_quantized_idxs"
PRESERVED_ATTRIBUTES_DICT_KEY = "preserved_attributes"


@dataclass
# 数据类，用于定义单独模块的配置条目
class StandaloneModuleConfigEntry:
    # qconfig_mapping用于子模块中prepare函数的配置，None表示使用父级qconfig_mapping中的qconfig
    qconfig_mapping: Optional[QConfigMapping]
    # 示例输入，作为元组存储
    example_inputs: Tuple[Any, ...]
    # 自定义配置的准备函数
    prepare_custom_config: Optional[PrepareCustomConfig]
    # 后端配置
    backend_config: Optional[BackendConfig]


# 准备自定义配置类，用于torch.ao.quantization.quantize_fx.prepare_fx和torch.ao.quantization.quantize_fx.prepare_qat_fx
class PrepareCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.prepare_fx` and
    :func:`~torch.ao.quantization.quantize_fx.prepare_qat_fx`.

    Example usage::

        prepare_custom_config = PrepareCustomConfig() \
            .set_standalone_module_name("module1", qconfig_mapping, example_inputs, \
                child_prepare_custom_config, backend_config) \
            .set_standalone_module_class(MyStandaloneModule, qconfig_mapping, example_inputs, \
                child_prepare_custom_config, backend_config) \
            .set_float_to_observed_mapping(FloatCustomModule, ObservedCustomModule) \
            .set_non_traceable_module_names(["module2", "module3"]) \
            .set_non_traceable_module_classes([NonTraceableModule1, NonTraceableModule2]) \
            .set_input_quantized_indexes([0]) \
            .set_output_quantized_indexes([0]) \
            .set_preserved_attributes(["attr1", "attr2"])
    """
    # 初始化函数，设置空字典和列表作为类的初始属性，用于存储配置信息
    def __init__(self):
        self.standalone_module_names: Dict[str, StandaloneModuleConfigEntry] = {}
        self.standalone_module_classes: Dict[Type, StandaloneModuleConfigEntry] = {}
        self.float_to_observed_mapping: Dict[QuantType, Dict[Type, Type]] = {}
        self.non_traceable_module_names: List[str] = []
        self.non_traceable_module_classes: List[Type] = []
        self.input_quantized_indexes: List[int] = []
        self.output_quantized_indexes: List[int] = []
        self.preserved_attributes: List[str] = []

    # 返回对象的字符串表示，仅包含非空的属性
    def __repr__(self):
        dict_nonempty = {
            k: v for k, v in self.__dict__.items()
            if len(v) > 0
        }
        return f"PrepareCustomConfig({dict_nonempty})"

    # 设置运行独立模块的配置，使用模块名作为键
    def set_standalone_module_name(
            self,
            module_name: str,
            qconfig_mapping: Optional[QConfigMapping],
            example_inputs: Tuple[Any, ...],
            prepare_custom_config: Optional[PrepareCustomConfig],
            backend_config: Optional[BackendConfig]) -> PrepareCustomConfig:
        """
        设置运行由“module_name”标识的独立模块的配置。

        如果“qconfig_mapping”为None，则使用父“qconfig_mapping”。
        如果“prepare_custom_config”为None，则使用空的“PrepareCustomConfig”。
        如果“backend_config”为None，则使用父“backend_config”。
        """
        # 将模块名和对应的配置信息存储到self.standalone_module_names字典中
        self.standalone_module_names[module_name] = \
            StandaloneModuleConfigEntry(qconfig_mapping, example_inputs, prepare_custom_config, backend_config)
        # 返回当前对象的引用
        return self

    # 设置运行独立模块的配置，使用模块类作为键
    def set_standalone_module_class(
            self,
            module_class: Type,
            qconfig_mapping: Optional[QConfigMapping],
            example_inputs: Tuple[Any, ...],
            prepare_custom_config: Optional[PrepareCustomConfig],
            backend_config: Optional[BackendConfig]) -> PrepareCustomConfig:
        """
        设置运行由“module_class”标识的独立模块的配置。

        如果“qconfig_mapping”为None，则使用父“qconfig_mapping”。
        如果“prepare_custom_config”为None，则使用空的“PrepareCustomConfig”。
        如果“backend_config”为None，则使用父“backend_config”。
        """
        # 将模块类和对应的配置信息存储到self.standalone_module_classes字典中
        self.standalone_module_classes[module_class] = \
            StandaloneModuleConfigEntry(qconfig_mapping, example_inputs, prepare_custom_config, backend_config)
        # 返回当前对象的引用
        return self
    # 设置从自定义浮点模块类到自定义观察模块类的映射关系，并返回当前配置对象
    def set_float_to_observed_mapping(
            self,
            float_class: Type,
            observed_class: Type,
            quant_type: QuantType = QuantType.STATIC) -> PrepareCustomConfig:
        """
        Set the mapping from a custom float module class to a custom observed module class.

        The observed module class must have a ``from_float`` class method that converts the float module class
        to the observed module class. This is currently only supported for static quantization.
        """
        # 如果量化类型不是静态的，抛出异常，因为目前仅支持静态量化
        if quant_type != QuantType.STATIC:
            raise ValueError("set_float_to_observed_mapping is currently only supported for static quantization")
        # 如果该量化类型还未在映射字典中，则创建一个空的映射字典
        if quant_type not in self.float_to_observed_mapping:
            self.float_to_observed_mapping[quant_type] = {}
        # 将浮点模块类映射到观察模块类
        self.float_to_observed_mapping[quant_type][float_class] = observed_class
        # 返回当前配置对象
        return self

    # 设置不可追踪模块的名称列表，并返回当前配置对象
    def set_non_traceable_module_names(self, module_names: List[str]) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by name.
        """
        self.non_traceable_module_names = module_names
        return self

    # 设置不可追踪模块的类列表，并返回当前配置对象
    def set_non_traceable_module_classes(self, module_classes: List[Type]) -> PrepareCustomConfig:
        """
        Set the modules that are not symbolically traceable, identified by class.
        """
        self.non_traceable_module_classes = module_classes
        return self

    # 设置应该进行量化的图输入的索引列表，并返回当前配置对象
    def set_input_quantized_indexes(self, indexes: List[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the inputs of the graph that should be quantized.
        Inputs are otherwise assumed to be in fp32 by default instead.
        """
        self.input_quantized_indexes = indexes
        return self

    # 设置应该进行量化的图输出的索引列表，并返回当前配置对象
    def set_output_quantized_indexes(self, indexes: List[int]) -> PrepareCustomConfig:
        """
        Set the indexes of the outputs of the graph that should be quantized.
        Outputs are otherwise assumed to be in fp32 by default instead.
        """
        self.output_quantized_indexes = indexes
        return self

    # 设置应该在图模块中保留的属性名称列表，并返回当前配置对象
    def set_preserved_attributes(self, attributes: List[str]) -> PrepareCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``PrepareCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.PrepareCustomConfig.from_dict`.
        """
        # 定义一个内部函数，将 StandaloneModuleConfigEntry 转换为元组
        def _make_tuple(key: Any, e: StandaloneModuleConfigEntry):
            # 如果存在 qconfig_mapping，则将其转换为字典形式；否则为 None
            qconfig_dict = e.qconfig_mapping.to_dict() if e.qconfig_mapping else None
            # 如果存在 prepare_custom_config，则将其转换为字典形式；否则为 None
            prepare_custom_config_dict = e.prepare_custom_config.to_dict() if e.prepare_custom_config else None
            # 返回元组，包含键、qconfig 字典、example_inputs、prepare_custom_config 字典和 backend_config
            return (key, qconfig_dict, e.example_inputs, prepare_custom_config_dict, e.backend_config)

        # 创建空字典 d，用于存储转换后的配置信息
        d: Dict[str, Any] = {}

        # 遍历 standalone_module_names 中的每个模块名和对应的配置条目
        for module_name, sm_config_entry in self.standalone_module_names.items():
            # 如果字典中不存在 STANDALONE_MODULE_NAME_DICT_KEY 键，则创建空列表
            if STANDALONE_MODULE_NAME_DICT_KEY not in d:
                d[STANDALONE_MODULE_NAME_DICT_KEY] = []
            # 将转换后的元组追加到对应键的列表中
            d[STANDALONE_MODULE_NAME_DICT_KEY].append(_make_tuple(module_name, sm_config_entry))

        # 遍历 standalone_module_classes 中的每个模块类和对应的配置条目
        for module_class, sm_config_entry in self.standalone_module_classes.items():
            # 如果字典中不存在 STANDALONE_MODULE_CLASS_DICT_KEY 键，则创建空列表
            if STANDALONE_MODULE_CLASS_DICT_KEY not in d:
                d[STANDALONE_MODULE_CLASS_DICT_KEY] = []
            # 将转换后的元组追加到对应键的列表中
            d[STANDALONE_MODULE_CLASS_DICT_KEY].append(_make_tuple(module_class, sm_config_entry))

        # 遍历 float_to_observed_mapping 中的每个量化类型和对应的映射关系
        for quant_type, float_to_observed_mapping in self.float_to_observed_mapping.items():
            # 如果字典中不存在 FLOAT_TO_OBSERVED_DICT_KEY 键，则创建空字典
            if FLOAT_TO_OBSERVED_DICT_KEY not in d:
                d[FLOAT_TO_OBSERVED_DICT_KEY] = {}
            # 将量化类型和对应的映射关系添加到字典中
            d[FLOAT_TO_OBSERVED_DICT_KEY][_get_quant_type_to_str(quant_type)] = float_to_observed_mapping

        # 如果 non_traceable_module_names 非空，则将其添加到结果字典中
        if len(self.non_traceable_module_names) > 0:
            d[NON_TRACEABLE_MODULE_NAME_DICT_KEY] = self.non_traceable_module_names

        # 如果 non_traceable_module_classes 非空，则将其添加到结果字典中
        if len(self.non_traceable_module_classes) > 0:
            d[NON_TRACEABLE_MODULE_CLASS_DICT_KEY] = self.non_traceable_module_classes

        # 如果 input_quantized_indexes 非空，则将其添加到结果字典中
        if len(self.input_quantized_indexes) > 0:
            d[INPUT_QUANTIZED_INDEXES_DICT_KEY] = self.input_quantized_indexes

        # 如果 output_quantized_indexes 非空，则将其添加到结果字典中
        if len(self.output_quantized_indexes) > 0:
            d[OUTPUT_QUANTIZED_INDEXES_DICT_KEY] = self.output_quantized_indexes

        # 如果 preserved_attributes 非空，则将其添加到结果字典中
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes

        # 返回最终的字典表示，包含了所有配置信息
        return d
    # 定义 ConvertCustomConfig 类，用于定制量化转换的配置
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.convert_fx`.

    Example usage::

        convert_custom_config = ConvertCustomConfig() \
            .set_observed_to_quantized_mapping(ObservedCustomModule, QuantizedCustomModule) \
            .set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self):
        # 初始化 observed_to_quantized_mapping 字典，用于存储观察模块类到量化模块类的映射关系
        self.observed_to_quantized_mapping: Dict[QuantType, Dict[Type, Type]] = {}
        # 初始化 preserved_attributes 列表，用于存储要保留的属性名称
        self.preserved_attributes: List[str] = []

    def __repr__(self):
        # 生成类的字符串表示，只包含非空字典项
        dict_nonempty = {
            k: v for k, v in self.__dict__.items()
            if len(v) > 0
        }
        return f"ConvertCustomConfig({dict_nonempty})"

    def set_observed_to_quantized_mapping(
            self,
            observed_class: Type,
            quantized_class: Type,
            quant_type: QuantType = QuantType.STATIC) -> ConvertCustomConfig:
        """
        Set the mapping from a custom observed module class to a custom quantized module class.

        The quantized module class must have a ``from_observed`` class method that converts the observed module class
        to the quantized module class.
        """
        # 如果 quant_type 不存在于 observed_to_quantized_mapping 中，创建一个空字典
        if quant_type not in self.observed_to_quantized_mapping:
            self.observed_to_quantized_mapping[quant_type] = {}
        # 将 observed_class 映射到 quantized_class，存储在对应的 quant_type 中
        self.observed_to_quantized_mapping[quant_type][observed_class] = quantized_class
        return self

    def set_preserved_attributes(self, attributes: List[str]) -> ConvertCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        # 设置要保留的属性列表
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(cls, convert_custom_config_dict: Dict[str, Any]) -> ConvertCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "observed_to_quantized_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from observed module classes to quantized module classes, e.g.::
            {
            "static": {FloatCustomModule: ObservedCustomModule},
            "dynamic": {FloatCustomModule: ObservedCustomModule},
            "weight_only": {FloatCustomModule: ObservedCustomModule}
            }
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        # 创建一个空的 ConvertCustomConfig 实例
        conf = cls()
        # 遍历 observed_to_quantized_custom_module_class 字典中的每个条目
        for quant_type_name, custom_module_mapping in convert_custom_config_dict.get(OBSERVED_TO_QUANTIZED_DICT_KEY, {}).items():
            # 将 quant_type_name 转换为对应的量化类型
            quant_type = _quant_type_from_str(quant_type_name)
            # 遍历 custom_module_mapping 中的每个条目，设置观察模块类到量化模块类的映射关系
            for observed_class, quantized_class in custom_module_mapping.items():
                conf.set_observed_to_quantized_mapping(observed_class, quantized_class, quant_type)
        # 设置保留的属性列表
        conf.set_preserved_attributes(convert_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        # 返回配置实例
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``ConvertCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        # 创建一个空的字典 d
        d: Dict[str, Any] = {}
        # 遍历 observed_to_quantized_mapping 字典中的每个条目
        for quant_type, observed_to_quantized_mapping in self.observed_to_quantized_mapping.items():
            # 如果 d 中没有 OBSERVED_TO_QUANTIZED_DICT_KEY 键，则创建一个空的字典
            if OBSERVED_TO_QUANTIZED_DICT_KEY not in d:
                d[OBSERVED_TO_QUANTIZED_DICT_KEY] = {}
            # 将量化类型 quant_type 转换为字符串形式，并将映射关系存储在 d 中
            d[OBSERVED_TO_QUANTIZED_DICT_KEY][_get_quant_type_to_str(quant_type)] = observed_to_quantized_mapping
        # 如果保留的属性列表不为空，则将其存储在 d 中
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        # 返回生成的字典
        return d
class FuseCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.fuse_fx`.

    Example usage::

        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self):
        # 初始化实例变量 preserved_attributes，用于存储需要保留的属性列表
        self.preserved_attributes: List[str] = []

    def __repr__(self):
        # 生成该对象的字符串表示，只包含非空的实例变量
        dict_nonempty = {
            k: v for k, v in self.__dict__.items()
            if len(v) > 0
        }
        return f"FuseCustomConfig({dict_nonempty})"

    def set_preserved_attributes(self, attributes: List[str]) -> FuseCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        # 设置需要保留的属性列表，并返回当前对象以支持链式调用
        self.preserved_attributes = attributes
        return self

    # TODO: remove this
    @classmethod
    def from_dict(cls, fuse_custom_config_dict: Dict[str, Any]) -> FuseCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        # 从字典中创建一个 FuseCustomConfig 对象，设置其中的 preserved_attributes
        conf = cls()
        conf.set_preserved_attributes(fuse_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``FuseCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        # 将当前对象转换为字典形式，包含 preserved_attributes
        d: Dict[str, Any] = {}
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d
```