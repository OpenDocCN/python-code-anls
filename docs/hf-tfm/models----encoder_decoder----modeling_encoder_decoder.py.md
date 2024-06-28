# `.\models\encoder_decoder\modeling_encoder_decoder.py`

```
# 设置编码方式为 UTF-8，确保脚本可以正确处理各种字符
# 版权声明和许可证信息，表明此代码遵循 Apache License, Version 2.0
# 详细许可证信息可以通过指定的 URL 获取
# 除非符合许可证中的规定，否则不得使用此文件
# 引入必要的模块和库
import gc  # Python 的垃圾回收模块，用于手动控制内存的释放
import inspect  # 用于获取对象信息的模块，如获取函数或类的源代码
import os  # 提供了许多与操作系统交互的函数
import tempfile  # 用于创建临时文件和目录的模块
import warnings  # 用于处理警告的模块
from typing import Optional, Tuple, Union  # Python 的类型提示模块，指定函数参数和返回值的类型

import torch  # PyTorch 深度学习库
from torch import nn  # PyTorch 中的神经网络模块
from torch.nn import CrossEntropyLoss  # 交叉熵损失函数

# 从 Hugging Face 的 Transformers 库中导入相关的模块和函数
from ...configuration_utils import PretrainedConfig  # 预训练配置文件类
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput  # 模型输出类
from ...modeling_utils import PreTrainedModel  # 预训练模型基类
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 辅助函数和日志模块
from ..auto.configuration_auto import AutoConfig  # 自动配置模块
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM  # 自动模型和自动语言模型模块
from .configuration_encoder_decoder import EncoderDecoderConfig  # 编码器-解码器配置类

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置对象名称
_CONFIG_FOR_DOC = "EncoderDecoderConfig"

# 弃用警告信息，指出新版本的变化和使用建议
DEPRECATION_WARNING = (
    "Version v4.12.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.12.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

# Encoder-Decoder 模型文档字符串的起始部分，使用原始字符串表示
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
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)



    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.



    Parameters:
        config ([`EncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""

# 定义一个函数，用于将输入的 token ids 向右移动一位
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    将输入的 token ids 向右移动一位。
    """
    # 创建一个与 input_ids 相同形状的零张量 shifted_input_ids
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将 input_ids 的除第一列外的数据复制到 shifted_input_ids 的第二列开始
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 如果 decoder_start_token_id 为 None，则抛出 ValueError
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 将 shifted_input_ids 的第一列设置为 decoder_start_token_id
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果 pad_token_id 为 None，则抛出 ValueError
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将 shifted_input_ids 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    # 返回向右移动后的 input ids
    return shifted_input_ids


@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
# 定义 EncoderDecoderModel 类，继承自 PreTrainedModel
class EncoderDecoderModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    
    EncoderDecoderModel 是一个通用的模型类，当使用 :meth:`~transformers.AutoModel.from_pretrained` 方法为编码器和
    :meth:`~transformers.AutoModelForCausalLM.from_pretrained` 方法为解码器创建时，它将被实例化为一个转换器架构。
    """

    # 类变量，指定配置类为 EncoderDecoderConfig
    config_class = EncoderDecoderConfig
    # 类变量，指定基础模型前缀为 "encoder_decoder"
    base_model_prefix = "encoder_decoder"
    # 类变量，主输入名称为 "input_ids"
    main_input_name = "input_ids"
    # 类变量，支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化方法
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        """
        Initialize the EncoderDecoderModel.
        初始化 EncoderDecoderModel。
        """
        # 如果需要，将编码器和解码器的权重绑定在一起
        def tie_weights(self):
            """
            Tie encoder & decoder if needed.
            如果需要，将编码器和解码器的权重绑定在一起。
            """
            if self.config.tie_encoder_decoder:
                # 获取解码器基础模型的前缀
                decoder_base_model_prefix = self.decoder.base_model_prefix
                # 调用 _tie_encoder_decoder_weights 方法，将编码器和解码器的权重绑定在一起
                self._tie_encoder_decoder_weights(
                    self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
                )

    # 获取编码器模型的方法
    def get_encoder(self):
        """
        Get the encoder model.
        获取编码器模型。
        """
        return self.encoder

    # 获取解码器模型的方法
    def get_decoder(self):
        """
        Get the decoder model.
        获取解码器模型。
        """
        return self.decoder

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        """
        Get the input embeddings.
        获取输入嵌入。
        """
        return self.encoder.get_input_embeddings()

    # 获取输出嵌入的方法
    def get_output_embeddings(self):
        """
        Get the output embeddings.
        获取输出嵌入。
        """
        return self.decoder.get_output_embeddings()

    # 设置输出嵌入的方法
    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings.
        设置输出嵌入。
        """
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ):
        """
        Instantiate an encoder-decoder model from pretrained model configurations.
        从预训练模型配置实例化一个编码器-解码器模型。
        """
        pass

    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，用于生成模型的输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # 此处是模型前向传播的方法，接收多个输入参数，包括输入的 token IDs、注意力掩码等
        # 返回模型的输出结果，如生成的 token IDs、注意力分布等

    # 根据标签准备解码器的输入 token IDs
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 准备用于生成的输入参数，构建生成过程所需的输入字典
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # 调用解码器对象的准备输入方法，获取解码器的输入信息
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # 如果解码器输入中包含注意力掩码，则获取之
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 构建输入字典，包括输入的注意力掩码、解码器的注意力掩码、解码器的输入 token IDs、编码器的输出等
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    # 调整 token embeddings 大小的方法，目前尚未实现
    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    # 重新排序缓存数据的方法，用于束搜索时的缓存重排
    def _reorder_cache(self, past_key_values, beam_idx):
        # 调用解码器对象的缓存重排方法
        return self.decoder._reorder_cache(past_key_values, beam_idx)
```