# `.\models\speech_encoder_decoder\modeling_speech_encoder_decoder.py`

```
# 导入必要的库和模块
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入配置相关的模块和函数
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "SpeechEncoderDecoderConfig"

# 文档字符串，描述了 Speech-Encoder-Text-Decoder 架构的类
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
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.



    Parameters:
        config ([`SpeechEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.


These comments provide context and explanations for each part of the code block, as requested.
"""

SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
# 将输入的 token ids 向右移动一个位置
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    将输入的 token ids 向右移动一个位置。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将除了第一列外的所有列替换为当前列的前一列的值，实现向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    # 设置第一列的值为 decoder_start_token_id
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # 将 labels 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@add_start_docstrings(SPEECH_ENCODER_DECODER_START_DOCSTRING)
# SpeechEncoderDecoderModel 类，用于包装 Transformer 架构中的编码器和解码器
class SpeechEncoderDecoderModel(PreTrainedModel):
    r"""
    [`SpeechEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    
    [`SpeechEncoderDecoderModel`] 是一个通用的模型类，将会被实例化为一个 Transformer 架构，其编码器和解码器是基于库中的基础模型类创建的，
    可以通过 :meth*~transformers.AutoModel.from_pretrained* 方法创建编码器，以及 :meth*~transformers.AutoModelForCausalLM.from_pretrained* 
    方法创建解码器。
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
    ):
        super().__init__(config)
        # 初始化 SpeechEncoderDecoderModel 实例
        # encoder 和 decoder 是预训练的编码器和解码器模型
        self.encoder = encoder
        self.decoder = decoder

    # 返回当前实例的编码器模型
    def get_encoder(self):
        return self.encoder

    # 返回当前实例的解码器模型
    def get_decoder(self):
        return self.decoder

    # 返回当前实例解码器的输出嵌入
    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    # 设置当前实例解码器的输出嵌入
    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    # 冻结特征编码器，禁用特征编码器的梯度计算，使其在训练过程中不会更新参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        调用此函数将禁用语音编码器的特征编码器的梯度计算，使其在训练过程中不会更新参数。
        """
        self.encoder.freeze_feature_encoder()

    @classmethod
    # 从预训练模型加载 SpeechEncoderDecoderModel 实例
    def from_pretrained(cls, *args, **kwargs):
        # 目前不支持快速初始化复合模型
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for SpeechEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    # 从预训练的编码器和解码器模型名称或路径加载模型
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
        # 准备解码器输入的 token ids，从标签中右移，用于生成过程的初始化
        def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
            return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # 为生成准备输入，包括解码器的准备
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 构建输入字典，供模型生成使用
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        # 不支持通过 SpeechEncoderDecoderModel 直接调整嵌入层大小
        raise NotImplementedError(
            "Resizing the embedding layers via the SpeechEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # 在这里重新排序缓存，应用解码器缓存重排
        return self.decoder._reorder_cache(past_key_values, beam_idx)
```