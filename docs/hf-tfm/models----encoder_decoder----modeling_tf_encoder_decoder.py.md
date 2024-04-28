# `.\models\encoder_decoder\modeling_tf_encoder_decoder.py`

```py
# 导入必要的库和模块
from __future__ import annotations
import inspect
import re
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf

# 导入 Hugging Face 库中的相关模块和函数
from ...configuration_utils import PretrainedConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    unpack_inputs,
)
from ...tf_utils import shape_list
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "EncoderDecoderConfig"

# 废弃警告消息
DEPRECATION_WARNING = (
    "Version v4.17.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.17.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

# Encoder-Decoder 模型开始文档字符串
ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the
    encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via
    [`~TFAutoModel.from_pretrained`] function and the decoder is loaded via [`~TFAutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like summarization.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models
    (see the examples for more information).

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个文档字符串用于记录编码器解码器的输入
ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""

# 将输入的 token 向右移动，用指定的 pad_token_id 和 decoder_start_token_id 进行填充
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 检查是否设置了 pad_token_id 属性
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)

    # 检查是否设置了 decoder_start_token_id 属性
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)

    # 创建一个 tensor，将 decoder_start_token_id 填充到指定形状
    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    # 将 input_ids 右移一位，加入 decoder_start_token_id
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 替换 labels 中可能存在的 -100 值为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
    )

    # 断言 labels 只包含正值和 -100
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作被调用，通过包装结果在无操作中调用
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


# TFEncoderDecoderModel 类，用于创建一个包含编码器和解码器的 transformer 模型
@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class TFEncoderDecoderModel(TFPreTrainedModel, TFCausalLanguageModelingLoss):
    r"""
    [`TFEncoderDecoderModel`] 是一个通用的模型类，当使用 [`~TFAutoModel.from_pretrained`] 类方法为编码器和
    [`~TFAutoModelForCausalLM.from_pretrained`] 类方法为解码器创建时，将被实例化为一个 transformer 架构的模型。
    """

    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"
    load_weight_prefix = "tf_encoder_decoder_model"

    # 初始化 TFEncoderDecoderModel，接受配置、编码器和解码器作为参数
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[TFPreTrainedModel] = None,
        decoder: Optional[TFPreTrainedModel] = None,
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
    def tf_to_pt_weight_rename(self, tf_weight):
        # 将 TensorFlow 和 PyTorch 的权重进行重命名以对齐
        # 我们的 TensorFlow 基类与 PyTorch 模型相比多了一个额外的层
        # （主模型结构在 MainLayer 类中）。如果我们删除该层，那么权重名称将正常对齐。
        # 然而，该额外层的名称与基本模型中的 MainLayer 的名称相同。
        # 我们在这里假设配置模型类型与 MainLayer 的名称相同。
        # 我不知道任何不是这种情况的地方，也不确定如何从配置中获取到正确的 MainLayer 名称！

        # 只有在我们从 PyTorch 中交叉加载权重时才需要这种重命名。
        # 但是，由于权重现在经常是 SafeTensor，我们不知道我们是否将会进行交叉加载，直到我们检查权重文件。
        # 因此，我们仍然指定 tf_to_pt_weight_rename，让超级方法自行判断是否需要它。
        encoder_model_type = self.config.encoder.model_type
        if "encoder" in tf_weight and "decoder" not in tf_weight:
            # 如果权重中包含 "encoder" 且不包含 "decoder"，则执行重命名
            return (re.sub(rf"encoder\.{encoder_model_type}\.", "encoder.", tf_weight),)
        else:
            # 否则返回原始权重名称
            return (tf_weight,)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    # 以下是装饰器
    @unpack_inputs
    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    # 以下是 call 方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # 准备 decoder 的输入数据用于生成，包括过去的键值对
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # 获取 decoder 的注意力掩码
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 获取过去的键值对
        past_key_values = decoder_inputs.get("past_key_values")
        # 如果过去的键值对为空，则使用 "past" 中的值，例如在 TF GPT2 中
        if past_key_values is None:
            past_key_values = decoder_inputs.get("past")
        # 构建输入字典，包括 input_ids、attention_mask、decoder_attention_mask 等
        input_dict = {
            "input_ids": None,  # needs to be passed to make Keras.layer.__call__ happy
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            # TODO (joao): the `TFBaseModelOutput` wrapper should not be needed after the generate refactor is complete
            "encoder_outputs": TFBaseModelOutput(last_hidden_state=encoder_outputs[0]),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        return input_dict

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 通过标签准备 decoder 的输入 ids
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        # 抛出未实现的错误，因为直接通过 TFEncoderDecoderModel 调整嵌入层不受支持
        raise NotImplementedError(
            "Resizing the embedding layers via the TFEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # 在这里重新排序缓存
        return self.decoder._reorder_cache(past, beam_idx)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "enc_to_dec_proj", None) is not None:
            with tf.name_scope(self.enc_to_dec_proj.name):
                self.enc_to_dec_proj.build([None, None, self.encoder.config.hidden_size])
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
```