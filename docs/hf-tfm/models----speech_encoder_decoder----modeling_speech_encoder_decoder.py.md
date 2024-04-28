# `.\transformers\models\speech_encoder_decoder\modeling_speech_encoder_decoder.py`

```
# 引入 typing 库，用于类型提示
from typing import Optional, Tuple, Union
# 引入 torch 库
import torch
# 从 torch 库中引入 nn 模块
from torch import nn
# 从 torch.nn 库中引入 CrossEntropyLoss 类
from torch.nn import CrossEntropyLoss
# 从 Huggingface 库中引入预训练配置模块
from ...configuration_utils import PretrainedConfig
# 从 Huggingface 库中引入模型输出模块
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
# 从 Huggingface 库中引入模型模块
from ...modeling_utils import PreTrainedModel
# 从 Huggingface 库中引入工具模块
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 从预训练配置模块中引入 AutoConfig 类
from ..auto.configuration_auto import AutoConfig
# 从预训练模型模块中引入 AutoModel 和 AutoModelForCausalLM 类
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
# 从语音编码器解码器配置模块中引入 SpeechEncoderDecoderConfig 类
from .configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig
# 从 logging 模块中引入 logger 对象
logger = logging.get_logger(__name__)

# 文档指定的配置
_CONFIG_FOR_DOC = "SpeechEncoderDecoderConfig"
# 文档指定的起始文档字符串
SPEECH_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a speech-sequence-to-text-sequence model with any pretrained speech
    autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is
    loaded via [`~AutoModel.from_pretrained`] function and the decoder is loaded via
    [`~AutoModelForCausalLM.from_pretrained`] function. Cross-attention layers are automatically added to the decoder
    and should be fine-tuned on a downstream generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [Large-Scale Self- and Semi-Supervised Learning for Speech
    Translation](https://arxiv.org/abs/2104.06678) it is shown how leveraging large pretrained speech models for speech
    translation yields a significant performance improvement.

    After such an Speech-Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
``` 
    # 这个模型也是一个 PyTorch 的 torch.nn.Module 子类。
    # 可以像使用常规的 PyTorch 模块一样使用它，并参考 PyTorch 文档了解有关一般用法和行为的所有事项。
    
    # 参数:
    #     config ([`SpeechEncoderDecoderConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型相关的权重，只加载配置。
    #         查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
# 将输入的 token 向右移动一个位置
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个与输入相同形状的全零张量
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入的 token 向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 设置首个 token 为 decoder_start_token_id
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将 labels 中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@add_start_docstrings(SPEECH_ENCODER_DECODER_START_DOCSTRING)
# `SpeechEncoderDecoderModel` 是一个通用的模型类，创建时会用库中的一个基础模型类作为编码器，另一个作为解码器。
class SpeechEncoderDecoderModel(PreTrainedModel):
    r"""
    [`SpeechEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    config_class = SpeechEncoderDecoderConfig
    base_model_prefix = "speech_encoder_decoder"
    main_input_name = "inputs"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    # 设置输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    # 冻结特征编码器
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        """
        self.encoder.freeze_feature_encoder()

    @classmethod
    # 从预训练模型中实例化对象
    def from_pretrained(cls, *args, **kwargs):
        # 当前不支持快速初始化
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for SpeechEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    # 类方法，用于从预训练的编码器和解码器模型中加载模型
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ):
    # 前向传播函数，执行模型的前向计算
    @add_start_docstrings_to_model_forward(SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
    # 根据标签准备解码器输入的标识符
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # 准备解码器的输入
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # 获取解码器注意力掩码
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 构建输入字典
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        # 返回输入字典
        return input_dict

    # 调整标记嵌入大小的方法（未实现）
    def resize_token_embeddings(self, *args, **kwargs):
        # 抛出未实现错误
        raise NotImplementedError(
            "Resizing the embedding layers via the SpeechEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    # 重新排列缓存的方法
    def _reorder_cache(self, past_key_values, beam_idx):
        # 应用解码器缓存重新排序
        return self.decoder._reorder_cache(past_key_values, beam_idx)
```