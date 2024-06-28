# `.\models\vision_encoder_decoder\configuration_vision_encoder_decoder.py`

```py
# 引入需要的模块和类
from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
# 引入版本控制的模块
from packaging import version
# 引入日志记录工具
from ...utils import logging
# 引入自动配置模块
from ..auto.configuration_auto import AutoConfig

# 如果是类型检查阶段，则导入必要的类型
if TYPE_CHECKING:
    from ... import PreTrainedTokenizerBase, TensorType

# 获取全局日志记录器
logger = logging.get_logger(__name__)


class VisionEncoderDecoderConfig(PretrainedConfig):
    r"""
    [`VisionEncoderDecoderConfig`] 是配置类，用于存储 [`VisionEncoderDecoderModel`] 的配置信息。
    用于根据指定的参数实例化一个 Vision-Encoder-Text-Decoder 模型，定义编码器和解码器的配置。

    配置对象继承自 [`PretrainedConfig`]，用于控制模型的输出。查阅 [`PretrainedConfig`] 的文档获取更多信息。

    Args:
        kwargs (*optional*):
            关键字参数的字典。特别是:

                - **encoder** ([`PretrainedConfig`], *optional*) -- 定义编码器配置的配置对象实例。
                - **decoder** ([`PretrainedConfig`], *optional*) -- 定义解码器配置的配置对象实例。

    Examples:

    ```
    >>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    >>> # 初始化 ViT 和 BERT 风格的配置
    >>> config_encoder = ViTConfig()
    >>> config_decoder = BertConfig()

    >>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

    >>> # 初始化一个 ViTBert 模型（具有随机权重），从 ViT 和 google-bert/bert-base-uncased 风格的配置开始
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
    """
    pass  # VisionEncoderDecoderConfig 类定义结束
    # 使用预训练模型名称加载视觉编码-解码器配置
    encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("my-model")
    # 使用预训练模型名称加载视觉编码-解码器模型，传入相应的配置
    model = VisionEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)



    # 定义模型类型为视觉编码-解码器
    model_type = "vision-encoder-decoder"
    # 标记该类为组合类
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 检查是否传入了编码器和解码器的配置，否则抛出异常
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}"
            )

        # 弹出并获取编码器配置和模型类型
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        # 弹出并获取解码器配置和模型类型
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        # 根据编码器配置创建自动配置对象
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        # 根据解码器配置创建自动配置对象
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        # 标记该模型为编码-解码器结构
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        """
        从预训练的编码器模型配置和解码器模型配置实例化一个 `VisionEncoderDecoderConfig`（或其派生类）。

        返回:
            [`VisionEncoderDecoderConfig`]: 配置对象的一个实例
        """
        # 记录日志信息，设置解码器配置为True和添加交叉注意力机制为True
        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        # 返回使用编码器和解码器配置实例化的类的实例
        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)
class VisionEncoderDecoderEncoderOnnxConfig(OnnxConfig):
    # 定义 Torch ONNX 的最低版本要求为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回输入规范化的顺序字典，定义了各个输入的维度信息
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        # 返回用于验证的绝对误差容限
        return 1e-4

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回输出规范化的顺序字典，定义了各个输出的维度信息
        return OrderedDict({"last_hidden_state": {0: "batch", 1: "encoder_sequence"}})


class VisionEncoderDecoderDecoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回输入规范化的顺序字典，定义了各个公共输入的维度信息
        common_inputs = OrderedDict()
        common_inputs["input_ids"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        common_inputs["attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        common_inputs["encoder_hidden_states"] = {0: "batch", 1: "encoder_sequence"}

        return common_inputs

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        import torch

        common_inputs = OrderedDict()

        # 调用父类方法生成虚拟输入
        dummy_input = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 提取 input_ids 的 batch 和 encoder_sequence 的值
        batch, encoder_sequence = dummy_input["input_ids"].shape
        # 创建 encoder_hidden_states 的形状 (batch, encoder_sequence, encoder_hidden_size) 的零张量
        encoder_hidden_states_shape = (batch, encoder_sequence, self._config.encoder_hidden_size)
        common_inputs["input_ids"] = dummy_input.pop("input_ids")
        common_inputs["attention_mask"] = dummy_input.pop("attention_mask")
        common_inputs["encoder_hidden_states"] = torch.zeros(encoder_hidden_states_shape)

        return common_inputs


class VisionEncoderDecoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> None:
        # 空实现，表示没有特定的输入定义
        pass

    def get_encoder_config(self, encoder_config: PretrainedConfig) -> OnnxConfig:
        r"""
        返回用于 `VisionEncoderDecoder` 模型的 ONNX 编码器配置。

        Args:
            encoder_config (`PretrainedConfig`):
                导出到 ONNX 时使用的编码器模型配置。

        Returns:
            [`VisionEncoderDecoderEncoderOnnxConfig`]: ONNX 配置对象的实例
        """
        return VisionEncoderDecoderEncoderOnnxConfig(encoder_config)

    def get_decoder_config(
        self, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, feature: str = "default"
        # 返回用于 `VisionEncoderDecoder` 模型的 ONNX 解码器配置
    ) -> OnnxConfig:
        r"""
        Returns ONNX decoder config for `VisionEncoderDecoder` model.

        Args:
            encoder_config (`PretrainedConfig`):
                The encoder model's configuration to use when exporting to ONNX.
            decoder_config (`PretrainedConfig`):
                The decoder model's configuration to use when exporting to ONNX
            feature (`str`, *optional*):
                The type of feature to export the model with.

        Returns:
            [`VisionEncoderDecoderDecoderOnnxConfig`]: An instance of the ONNX configuration object.
        """
        # 设置解码器配置的隐藏状态大小为编码器配置的隐藏状态大小
        decoder_config.encoder_hidden_size = encoder_config.hidden_size
        # 返回一个包含解码器配置和特征的 ONNX 配置对象实例
        return VisionEncoderDecoderDecoderOnnxConfig(decoder_config, feature)
```