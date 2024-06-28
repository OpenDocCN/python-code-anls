# `.\quantizers\auto.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入类型提示相关模块
from typing import Dict, Optional, Union

# 导入自动配置模块
from ..models.auto.configuration_auto import AutoConfig
# 导入量化配置相关类和方法
from ..utils.quantization_config import (
    AqlmConfig,
    AwqConfig,
    BitsAndBytesConfig,
    GPTQConfig,
    QuantizationConfigMixin,
    QuantizationMethod,
    QuantoConfig,
)
# 导入各种量化器类
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_gptq import GptqHfQuantizer
from .quantizer_quanto import QuantoHfQuantizer

# 自动量化器与量化器类的映射关系
AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "bitsandbytes_4bit": Bnb4BitHfQuantizer,
    "bitsandbytes_8bit": Bnb8BitHfQuantizer,
    "gptq": GptqHfQuantizer,
    "aqlm": AqlmHfQuantizer,
    "quanto": QuantoHfQuantizer,
}

# 自动量化配置与量化配置类的映射关系
AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "awq": AwqConfig,
    "bitsandbytes_4bit": BitsAndBytesConfig,
    "bitsandbytes_8bit": BitsAndBytesConfig,
    "gptq": GPTQConfig,
    "aqlm": AqlmConfig,
    "quanto": QuantoConfig,
}

# 自动量化配置类，用于根据给定的量化配置自动分发到正确的量化配置
class AutoQuantizationConfig:
    """
    The Auto-HF quantization config class that takes care of automatically dispatching to the correct
    quantization config given a quantization config stored in a dictionary.
    """

    @classmethod
    # 从给定的字典构建一个类方法，用于反序列化量化配置
    def from_dict(cls, quantization_config_dict: Dict):
        # 从量化配置字典中获取量化方法，如果不存在则设为 None
        quant_method = quantization_config_dict.get("quant_method", None)
        # 对于 bnb 模型，需要特别处理以确保兼容性
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            # 如果配置中指定了 load_in_4bit，则使用 4 位量化方法后缀；否则使用 8 位后缀
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            # 构建量化方法字符串，结合 BITS_AND_BYTES 常量
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        # 如果未指定量化方法，则抛出 ValueError 异常
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized"
            )

        # 如果量化方法不在自动量化配置映射的键中，则抛出 ValueError 异常
        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        # 根据量化方法从自动量化配置映射中获取目标类
        target_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
        # 使用目标类的类方法将量化配置字典反序列化为对象
        return target_cls.from_dict(quantization_config_dict)

    # 类方法：从预训练模型中加载模型配置，并返回相应的量化配置
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 从预训练模型名称或路径加载模型配置
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 如果模型配置中没有量化配置，则抛出 ValueError 异常
        if getattr(model_config, "quantization_config", None) is None:
            raise ValueError(
                f"Did not found a `quantization_config` in {pretrained_model_name_or_path}. Make sure that the model is correctly quantized."
            )
        # 获取模型配置中的量化配置字典
        quantization_config_dict = model_config.quantization_config
        # 使用 from_dict 方法将量化配置字典反序列化为量化配置对象
        quantization_config = cls.from_dict(quantization_config_dict)
        # 将传递自 from_pretrained 的额外参数更新到量化配置对象中
        quantization_config.update(kwargs)
        # 返回构建好的量化配置对象
        return quantization_config
    """
     The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`.
    """

    @classmethod
    # 类方法：从给定的量化配置创建一个实例
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        # 如果 quantization_config 是字典类型，则转换为 QuantizationConfig
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        # 获取量化方法
        quant_method = quantization_config.quant_method

        # 对 BITS_AND_BYTES 方法进行特殊处理，因为我们有一个单独的量化配置类用于 4-bit 和 8-bit 量化
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        # 检查量化方法是否在自动量化映射中
        if quant_method not in AUTO_QUANTIZER_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        # 根据量化方法选择对应的类
        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        # 使用选定的类创建实例并返回
        return target_cls(quantization_config, **kwargs)

    @classmethod
    # 类方法：从预训练模型名称或路径创建一个实例
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 从预训练模型获取量化配置
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 使用量化配置创建实例
        return cls.from_config(quantization_config)

    @classmethod
    # 类方法：合并两个量化配置
    def merge_quantization_configs(
        cls,
        quantization_config: Union[dict, QuantizationConfigMixin],
        quantization_config_from_args: Optional[QuantizationConfigMixin],
        """
        处理同时存在来自参数和模型配置中的量化配置的情况。
        """
        # 如果参数中有 quantization_config，则生成警告消息
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        # 如果 quantization_config 是字典类型，则转换为 AutoQuantizationConfig 对象
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        # 对于 GPTQConfig 或 AwqConfig 类型的 quantization_config，并且 quantization_config_from_args 不为空的特殊情况处理
        if isinstance(quantization_config, (GPTQConfig, AwqConfig)) and quantization_config_from_args is not None:
            # 获取 quantization_config_from_args 中的加载属性，并设置到 quantization_config 对象中
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)
            # 更新警告消息，说明加载属性将被传入的参数覆盖，其余属性将被忽略
            warning_msg += f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."

        # 如果有警告消息，发出警告
        if warning_msg != "":
            warnings.warn(warning_msg)

        # 返回处理后的 quantization_config 对象
        return quantization_config
```