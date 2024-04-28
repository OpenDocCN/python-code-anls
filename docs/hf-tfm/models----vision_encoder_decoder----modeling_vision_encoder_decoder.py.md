# `.\transformers\models\vision_encoder_decoder\modeling_vision_encoder_decoder.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，
# 没有任何明示或暗示的保证或条件，查看许可证以获取特定语言的权限和限制
""" 支持视觉编码器-文本解码器架构的类"""


import gc
import os
import tempfile
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig


# 从transformers.models.encoder_decoder.modeling_encoder_decoder中复制的shift_tokens_right函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的标记向右移动一个标记。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 用`pad_token_id`替换标签中可能存在的-100值
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VisionEncoderDecoderConfig"

VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    此类可用于使用任何预训练的视觉自编码模型作为编码器和任何预训练的文本自回归模型作为解码器初始化图像到文本序列模型。
    编码器通过[`~AutoModel.from_pretrained`]函数加载，解码器通过[`~AutoModelForCausalLM.from_pretrained`]函数加载。
    交叉注意力层会自动添加到解码器，并应在下游生成任务（如图像字幕）上进行微调。

    使用预训练检查点初始化序列到序列模型以进行序列生成的有效性
    # 在 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) 中展示了如何利用预训练的检查点进行序列生成任务，作者为Sascha Rothe, Shashi Narayan, Aliaksei Severyn, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu。
    
    # 此外，在 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) 中展示了如何利用大型预训练视觉模型进行光学字符识别（OCR），从而显著提高性能。
    
    # 当 Vision-Encoder-Text-Decoder 模型经过训练/微调后，可以像其他模型一样保存/加载（查看示例以获取更多信息）。
    
    # 此模型继承自 [`PreTrainedModel`]。查看超类文档以了解库为所有模型实现的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。
    
    # 此模型还是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。将其用作常规 PyTorch 模块，并参考 PyTorch 文档以获取与一般用法和行为相关的所有信息。
    
    # 参数:
    #     config ([`VisionEncoderDecoderConfig`]): 包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# 定义一个文档字符串常量，用于描述视觉编码器解码器模型的输入
VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""

# 使用装饰器添加起始文档字符串到视觉编码器解码器模型类上
@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
class VisionEncoderDecoderModel(PreTrainedModel):
    r"""
    [`VisionEncoderDecoderModel`] 是一个通用的模型类，当使用 :meth*~transformers.AutoModel.from_pretrained* 类方法为编码器创建一个基础视觉模型类，并为解码器创建另一个基础视觉模型类时，将实例化为一个变压器架构。
    """

    # 设置配置类为 VisionEncoderDecoderConfig
    config_class = VisionEncoderDecoderConfig
    # 设置基础模型前缀为 "vision_encoder_decoder"
    base_model_prefix = "vision_encoder_decoder"
    # 设置主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        ):
            # 如果没有提供配置信息并且编码器或解码器为空，则抛出数值错误
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        # 如果没有提供配置信息，则使用编码器和解码器的配置信息创建视觉编码器解码器配置对象
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            # 如果配置信息不是指定的配置类，则抛出数值错误
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # 如果解码器的交叉注意力隐藏大小不为空
        if config.decoder.cross_attention_hidden_size is not None:
            # 如果解码器的交叉注意力隐藏大小不等于编码器的隐藏大小，则抛出数值错误
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 使用配置信息初始化，确保输入和输出嵌入不是绑定的
        config.tie_word_embeddings = False
        super().__init__(config)

        # 如果编码器为空，则根据配置信息创建自动模型
        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        # 如果解码器为空，则根据配置信息创建自动模型
        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        # 如果编码器的配置信息与共享的编码器配置信息不同，则发出警告
        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        # 如果解码器的配置信息与共享的解码器配置信息不同，则发出警告
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # 确保各个模型的配置信息引用共享的配置信息，以便配置的更新会同步
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # 如果编码器输出可能需要投影到不同维度以供解码器使用
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        # 如果编码器有输出嵌入，则抛出数值错误
        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder
    # 返回解码器的输出嵌入
    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    # 从预训练的编码器和解码器创建模型
    @classmethod
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    
    # 前向传播函数
    @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    
    # 根据标签准备解码器输入的标识符
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    # 调整标记嵌入
    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported.Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    # 重新排序缓存
    def _reorder_cache(self, past_key_values, beam_idx):
        # 在这里应用解码器缓存重新排序
        return self.decoder._reorder_cache(past_key_values, beam_idx)
```