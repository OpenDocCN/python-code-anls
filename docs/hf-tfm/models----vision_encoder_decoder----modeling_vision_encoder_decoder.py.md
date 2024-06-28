# `.\models\vision_encoder_decoder\modeling_vision_encoder_decoder.py`

```py
# 设置文件的编码格式为 UTF-8

# 版权声明，指明版权归 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求，否则不得使用此文件
# 可在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件按"原样"分发，无任何担保或条件
# 请参阅许可证了解具体权限和限制

""" 用于支持 Vision-Encoder-Text-Decoder 结构的类"""

# 引入模块
import gc  # Python 垃圾回收模块
import os  # 操作系统模块
import tempfile  # 临时文件和目录模块
from typing import Optional, Tuple, Union  # 引入类型提示

import torch  # 引入 PyTorch 模块
from torch import nn  # 引入 PyTorch 中的神经网络模块
from torch.nn import CrossEntropyLoss  # 引入交叉熵损失函数

# 引入 Transformers 库中的一些模块和函数
from ...configuration_utils import PretrainedConfig  # 引入预训练配置相关函数
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput  # 引入基础模型输出和 Seq2SeqLM 输出
from ...modeling_utils import PreTrainedModel  # 引入预训练模型基类
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 引入辅助函数和日志记录函数
from ..auto.configuration_auto import AutoConfig  # 引入自动配置函数
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM  # 引入自动模型加载函数
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig  # 引入视觉编码解码器配置类


# 从 Transformers 库中的 encoder_decoder 模块中复制的函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 向右移动一个位置。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建一个与输入形状相同的全零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将输入向右移动一个位置
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id  # 设置起始 token

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将标签中可能存在的 -100 值替换为 `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "VisionEncoderDecoderConfig"

VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    此类可用于初始化一个图像到文本序列模型，其中编码器是任何预训练的视觉自编码模型，解码器是任何预训练的文本自回归模型。
    编码器通过 [`~AutoModel.from_pretrained`] 函数加载，解码器通过 [`~AutoModelForCausalLM.from_pretrained`] 函数加载。
    交叉注意力层会自动添加到解码器中，并应在下游生成任务（如图像字幕）中进行微调。

    初始化序列到序列模型时使用预训练检查点的有效性
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.



    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
    Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical
    character recognition (OCR) yields a significant performance improvement.



    After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any
    other models (see the examples for more information).



    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)



    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.



    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""

@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
class VisionEncoderDecoderModel(PreTrainedModel):
    r"""
    [`VisionEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base vision model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    # 设置配置类为 VisionEncoderDecoderConfig
    config_class = VisionEncoderDecoderConfig
    # 指定基础模型前缀为 "vision_encoder_decoder"
    base_model_prefix = "vision_encoder_decoder"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        ):
            # 如果未提供配置且未同时提供编码器和解码器，则抛出数值错误
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            # 如果未提供配置，则从编码器和解码器的配置中创建视觉编码器解码器配置
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            # 如果提供的配置不是预期的配置类类型，则抛出数值错误
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            # 如果解码器配置中指定了交叉注意力的隐藏大小
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                # 则要求解码器的交叉注意力隐藏大小必须与编码器的隐藏大小相等，否则抛出数值错误
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

        # 初始化配置，确保输入和输出嵌入不被绑定
        config.tie_word_embeddings = False
        # 调用父类初始化方法，传入配置
        super().__init__(config)

        if encoder is None:
            # 如果未提供编码器，则从配置中创建自动模型
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            # 如果未提供解码器，则从配置中创建自动因果语言模型
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        # 将编码器和解码器存储在实例变量中
        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            # 如果编码器的配置不等于共享配置，则记录警告信息
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            # 如果解码器的配置不等于共享配置，则记录警告信息
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # 确保各自模型的配置引用了共享的配置，以便配置的更新能够同步
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # 如果编码器和解码器的隐藏大小不相等且解码器未指定交叉注意力隐藏大小，则需要对编码器输出进行投影以适配解码器
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            # 如果编码器具有语言模型头部，则抛出数值错误
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

    def get_encoder(self):
        # 返回存储的编码器模型
        return self.encoder

    def get_decoder(self):
        # 返回存储的解码器模型
        return self.decoder
    # 返回当前模型的解码器的输出嵌入层
    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    # 设置当前模型的解码器的输出嵌入层为新的嵌入层
    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    # 从预训练的编码器和解码器模型名或路径创建一个模型实例
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ):
        pass

    # 前向传播函数，执行模型的正向运算
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
    ):
        pass

    # 根据标签准备解码器的输入标识，用于生成序列的输入
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 准备生成阶段的输入，整理输入数据用于模型生成
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

    # 调整标记嵌入的大小（目前未实现）
    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported.Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    # 重新排序缓存中的过去键值对，用于Beam搜索中的解码器缓存重排
    def _reorder_cache(self, past_key_values, beam_idx):
        # 在这里执行解码器缓存的重新排序
        return self.decoder._reorder_cache(past_key_values, beam_idx)
```