# `.\transformers\models\vision_encoder_decoder\configuration_vision_encoder_decoder.py`

```
# 设置文件编码为 UTF-8
# 版权声明版权 2021 年 HuggingFace 公司。
# 版权声明版权 2018 年，NVIDIA 公司。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非根据适用法律或书面同意，否则根据许可下分发的软件是基于"原样"基础分发的，
# 没有任何明示或暗示的担保或条件。请查看许可证获取明确的授权和限制。

# 导入必要的类型检查及其它模块
from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
# 导入版本管理模块
from packaging import version
# 导入配置相关函数
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置
from ...onnx import OnnxConfig
# 导入日志模块
from ...utils import logging
# 导入自动配置
from ..auto.configuration_auto import AutoConfig

# 如果支持类型检查
if TYPE_CHECKING:
    # 导入预训练标记化基类和张量类型
    from ... import PreTrainedTokenizerBase, TensorType

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 VisionEncoderDecoderConfig 类，继承自预训练配置类
class VisionEncoderDecoderConfig(PretrainedConfig):
    r"""
    [`VisionEncoderDecoderConfig`] 是用于存储一个 [`VisionEncoderDecoderModel`] 的配置类。
    用于根据指定的参数实例化一个 Vision-Encoder-Text-Decoder 模型，定义编码器和解码器配置。

    配置对象继承自 [`PretrainedConfig`] 类，可用于控制模型输出。更多信息请阅读 [`PretrainedConfig`] 的文档。

    Args:
        kwargs (*optional*):
            Keyword 参数字典。特别是:

                - **encoder** ([`PretrainedConfig`], *optional*) -- 定义编码器配置的配置对象的一个实例。
                - **decoder** ([`PretrainedConfig`], *optional*) -- 定义解码器配置的配置对象的一个实例。

    Examples:

    ```python
    >>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    >>> # 初始化一个 ViT & BERT 风格的配置
    >>> config_encoder = ViTConfig()
    >>> config_decoder = BertConfig()

    >>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # 从 ViT 和 bert-base-uncased 风格配置初始化一个 ViTBert 模型（带有随机权重）
    >>> model = VisionEncoderDecoderModel(config=config)

    >>> # 访问模型配置
    >>> config_encoder = model.config.encoder
    >>> config_decoder = model.config.decoder
    >>> # 将解码器配置设置为 causal lm
    >>> config_decoder.is_decoder = True
    >>> config_decoder.add_cross_attention = True

    >>> # 保存模型，包括其配置
    >>> model.save_pretrained("my-model")

    >>> # 从预训练文件夹加载模型和配置

    ```
    # 从预训练模型加载配置
    encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("my-model")
    # 从预训练模型加载模型
    model = VisionEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
    
    # 设置模型类型为“vision-encoder-decoder”
    model_type = "vision-encoder-decoder"
    # 标记为复合模型
    is_composition = True
    
    # 定义模型的初始化方法
    def __init__(self, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 检查是否传入了“encoder”和“decoder”子配置
        if "encoder" not in kwargs or "decoder" not in kwargs:
            # 抛出数值错误
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}"
            )
    
        # 弹出并获取编码器和解码器配置
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")
    
        # 根据配置创建编码器和解码器
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        # 标记为编码-解码器模型
        self.is_encoder_decoder = True
    
    # 类方法：从编码器和解码器配置实例化一个 VisionEncoderDecoderConfig 对象
    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        # 设置解码器配置的解码器标志和交叉注意力标志
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
    
        # 返回一个 VisionEncoderDecoderConfig 实例
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)
class VisionEncoderDecoderEncoderOnnxConfig(OnnxConfig):
    # 定义一个名为 VisionEncoderDecoderEncoderOnnxConfig 的类，它继承自 OnnxConfig 类

    torch_onnx_minimum_version = version.parse("1.11")
    # 设置一个静态属性 torch_onnx_minimum_version 为版本号 1.11

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义一个名为 inputs 的属性，返回一个字符串到整数和字符串的映射
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )
        # 返回一个按指定顺序排列的输入映射

    @property
    def atol_for_validation(self) -> float:
        # 定义一个名为 atol_for_validation 的属性，返回一个浮点数
        return 1e-4
        # 返回一个浮点数 0.0001 用于验证的绝对误差值

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义一个名为 outputs 的属性，返回一个字符串到整数和字符串的映射
        return OrderedDict({"last_hidden_state": {0: "batch", 1: "encoder_sequence"}})
        # 返回一个按指定顺序排列的输出映射


class VisionEncoderDecoderDecoderOnnxConfig(OnnxConfig):
    # 定义一个名为 VisionEncoderDecoderDecoderOnnxConfig 的类，它继承自 OnnxConfig 类

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义一个名为 inputs 的属性，返回一个字符串到整数和字符串的映射
        common_inputs = OrderedDict()
        # 创建一个有序字典 common_inputs

        common_inputs["input_ids"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        # 在 common_inputs 中添加键值对，键是 input_ids，值是一个整数到字符串的映射

        common_inputs["attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        # 在 common_inputs 中添加键值对，键是 attention_mask，值是一个整数到字符串的映射

        common_inputs["encoder_hidden_states"] = {0: "batch", 1: "encoder_sequence"}
        # 在 common_inputs 中添加键值对，键是 encoder_hidden_states，值是一个整数到字符串的映射

        return common_inputs
        # 返回 common_inputs


    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        # 定义一个名为 generate_dummy_inputs 的方法，接受一些参数并返回字符串到任意类型的映射
        import torch
        # 导入 torch 模块

        common_inputs = OrderedDict()
        # 创建一个有序字典 common_inputs

        dummy_input = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )
        # 调用父类的 generate_dummy_inputs 方法并将结果保存在 dummy_input 中

        batch, encoder_sequence = dummy_input["input_ids"].shape
        # 获取 dummy_input 中 input_ids 的形状，并分别赋给 batch 和 encoder_sequence

        encoder_hidden_states_shape = (batch, encoder_sequence, self._config.encoder_hidden_size)
        # 定义一个名为 encoder_hidden_states_shape 的变量，存储一个元组

        common_inputs["input_ids"] = dummy_input.pop("input_ids")
        # 弹出并赋值给 common_inputs 中的 input_ids 键

        common_inputs["attention_mask"] = dummy_input.pop("attention_mask")
        # 弹出并赋值给 common_inputs 中的 attention_mask 键

        common_inputs["encoder_hidden_states"] = torch.zeros(encoder_hidden_states_shape)
        # 在 common_inputs 中添加 encoder_hidden_states 键，并赋值为一个全 0 的 tensor

        return common_inputs
        # 返回 common_inputs


class VisionEncoderDecoderOnnxConfig(OnnxConfig):
    # 定义一个名为 VisionEncoderDecoderOnnxConfig 的类，它继承自 OnnxConfig 类

    @property
    def inputs(self) -> None:
        # 定义一个名为 inputs 的属性，返回 None
        pass
        # 什么也不做

    def get_encoder_config(self, encoder_config: PretrainedConfig) -> OnnxConfig:
        # 定义一个名为 get_encoder_config 的方法，接受一个 PretrainedConfig 类型的参数并返回 OnnxConfig 类型的变量
        r"""
        Returns ONNX encoder config for `VisionEncoderDecoder` model.

        Args:
            encoder_config (`PretrainedConfig`):
                The encoder model's configuration to use when exporting to ONNX.

        Returns:
            [`VisionEncoderDecoderEncoderOnnxConfig`]: An instance of the ONNX configuration object
        """
        return VisionEncoderDecoderEncoderOnnxConfig(encoder_config)
        # 返回一个 VisionEncoderDecoderEncoderOnnxConfig 类的实例对象，传入参数 encoder_config

    def get_decoder_config(
        self, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, feature: str = "default"
    ) -> OnnxConfig:
        r"""
        返回用于`VisionEncoderDecoder`模型的ONNX解码器配置。

        Args:
            encoder_config (`PretrainedConfig`):
                导出到ONNX时要使用的编码器模型配置。
            decoder_config (`PretrainedConfig`):
                导出到ONNX时要使用的解码器模型配置。
            feature (`str`, *optional*):
                要导出模型的特征类型。

        Returns:
            [`VisionEncoderDecoderDecoderOnnxConfig`]: ONNX配置对象的实例。
        """
        # 将解码器配置的编码器隐藏层大小设置为编码器配置的隐藏层大小
        decoder_config.encoder_hidden_size = encoder_config.hidden_size
        # 返回ONNX配置对象的实例
        return VisionEncoderDecoderDecoderOnnxConfig(decoder_config, feature)
```