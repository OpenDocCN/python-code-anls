# `.\models\encoder_decoder\modeling_encoder_decoder.py`

```py
# 设置编码为 UTF-8
# 版权声明及许可证信息
# 该类支持编码-解码架构

# 导入所需的库和模块
import gc
import inspect
import os
import tempfile
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入配置相关的类和函数
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig

# 设置 logger
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

# 版本更新提示信息
DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
``` 
    # 该库为其所有模型实现了一些功能，比如下载或保存、调整输入嵌入大小、剪枝头等。

    # 这个模型也是 PyTorch 的 [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类。
    # 您可以将其用作常规的 PyTorch 模块，并参考 PyTorch 文档以了解所有与一般使用和行为相关的事项。

    # 参数：
    #    config ([`EncoderDecoderConfig`])：包含模型所有参数的模型配置类。
    #        使用配置文件初始化不会加载与模型相关的权重，只会加载配置。 查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# 定义了一个文档字符串常量，用于描述编码器-解码器模型的输入
ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的标记向右移动一个标记位置。
    """
    # 创建一个与输入形状相同的零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入的标记除第一个位置外全部向右移动一位
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 检查解码器起始标记是否为 None
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 将第一个位置的标记设置为解码器的起始标记
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 检查填充标记是否为 None
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将标签中可能存在的 -100 值替换为填充标记
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class EncoderDecoderModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] 是一个通用的模型类，当使用 :meth:`~transformers.AutoModel.from_pretrained` 类方法
    为编码器选择一个库中的基本模型类时，将被实例化为一个变换器架构，并为解码器选择另一个基本模型类，
    并使用 :meth:`~transformers.AutoModelForCausalLM.from_pretrained` 类方法创建。
    """

    # 配置类为 EncoderDecoderConfig
    config_class = EncoderDecoderConfig
    # 基本模型前缀为 "encoder_decoder"
    base_model_prefix = "encoder_decoder"
    # 主输入名称为 "input_ids"
    main_input_name = "input_ids"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    # 绑定权重
    def tie_weights(self):
        # 如果需要，绑定编码器和解码器
        if self.config.tie_encoder_decoder:
            # 绑定编码器和解码器的基本模型
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    # 从预训练的编码器-解码器模型创建
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    # 添加文档字符串到模型前向传递
    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 forward 方法，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入数据的标识符
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入数据的标识符
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力遮罩
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,  # 编码器输出
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,  # 过去的关键值
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入向量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        **kwargs,  # 其他关键字参数
    # 根据标签准备解码器输入的方法
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签向右移动，并填充起始标记
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 准备进行生成的输入方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # 准备解码器的输入
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # 如果解码器输入中包含注意力遮罩，则复制到新的变量中
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 创建包含不同输入参数的字典
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    # 调整标记嵌入大小的方法
    def resize_token_embeddings(self, *args, **kwargs):
        # 抛出未实现错误
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    # 重新排序缓存的方法
    def _reorder_cache(self, past_key_values, beam_idx):
        # 在此应用解码器缓存重排序
        return self.decoder._reorder_cache(past_key_values, beam_idx)
```