# `.\models\vision_encoder_decoder\modeling_tf_vision_encoder_decoder.py`

```
# 设置编码为 UTF-8，确保脚本能够正确处理各种字符
# 版权声明，指出代码的版权归属及使用许可
# 导入必要的模块和类型声明
# 引入警告模块，用于显示编码警告信息
# 导入 NumPy 库，用于处理数组和矩阵数据
# 导入 TensorFlow 库，用于构建和训练深度学习模型

# 导入配置工具相关模块和类
from ...configuration_utils import PretrainedConfig
# 导入 TF 模型输出相关模块和类
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput
# 导入 TF 实用工具相关模块和类
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFPreTrainedModel, get_initializer, keras, unpack_inputs
# 导入 TensorFlow 实用工具，包括形状处理函数
from ...tf_utils import shape_list
# 导入通用工具模块，包括模型输出、文档字符串处理、日志记录等功能
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入自动配置相关类
from ..auto.configuration_auto import AutoConfig
# 导入 TensorFlow 自动化模型相关类
from ..auto.modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM
# 导入视觉编码器-解码器配置类
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 文档字符串中用到的配置名称
_CONFIG_FOR_DOC = "VisionEncoderDecoderConfig"

# 弃用警告信息，提醒版本更新带来的变更
DEPRECATION_WARNING = (
    "Version v4.17.0 introduces a better way to train encoder-decoder models by computing the loss inside the"
    " encoder-decoder framework rather than in the decoder itself. You may observe training discrepancies if"
    " fine-tuning a model trained with versions anterior to 4.17.0. The decoder_input_ids are now created based on the"
    " labels, no need to pass them yourself anymore."
)

# 视觉编码器-解码器类的起始文档字符串，详细说明其功能和用法
VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model
    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via
    [`~TFAutoModel.from_pretrained`] function and the decoder is loaded via [`~TFAutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like image captioning.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
    # 在论文[Large Pretrained Vision Models](https://arxiv.org/abs/2109.10282)中展示了如何利用大型预训练视觉模型进行光学字符识别（OCR），从而显著提高性能。
    #
    # 训练/微调了这样的视觉-编码器-文本-解码器模型后，可以像处理其他模型一样保存/加载它（参见示例以获取更多信息）。
    #
    # 这个模型继承自[`TFPreTrainedModel`]。请查阅超类文档，了解库为所有模型实现的通用方法（例如下载或保存、调整输入嵌入、剪枝头等）。
    #
    # 这个模型也是一个[keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)子类。可以将其作为常规的 TF 2.0 Keras 模型使用，并参考 TF 2.0 的文档了解所有与一般使用和行为相关的事项。
    #
    # 参数:
    #     config ([`VisionEncoderDecoderConfig`]): 包含模型所有参数的配置类。
    #         使用配置文件初始化模型不会加载与模型关联的权重，只加载配置。查看[`~TFPreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
"""


# Copied from transformers.models.encoder_decoder.modeling_tf_encoder_decoder.shift_tokens_right
# 将输入的 token 向右移动一位，用于生成 decoder 的输入序列
def shift_tokens_right(input_ids: tf.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # 检查 pad_token_id 是否为 None，如果是则抛出数值错误异常
    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    pad_token_id = tf.cast(pad_token_id, input_ids.dtype)

    # 检查 decoder_start_token_id 是否为 None，如果是则抛出数值错误异常
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    decoder_start_token_id = tf.cast(decoder_start_token_id, input_ids.dtype)

    # 创建一个形状为 (batch_size, 1) 的张量，用 decoder_start_token_id 填充
    start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
    # 将 start_tokens 和 input_ids 的前 n-1 列拼接起来，形成向右移动后的输入序列
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
    # 将 labels 中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids = tf.where(
        shifted_input_ids == -100, tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids
    )

    # 确保 shifted_input_ids 中的值大于等于 0，并添加调试信息
    assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=input_ids.dtype))

    # 确保断言操作被调用，通过将结果包装在 identity 操作中
    with tf.control_dependencies([assert_gte0]):
        shifted_input_ids = tf.identity(shifted_input_ids)

    return shifted_input_ids


@add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING)
# TFVisionEncoderDecoderModel 是一个通用的模型类，用于将库中的一个基本视觉模型类作为编码器，另一个基本模型类作为解码器
class TFVisionEncoderDecoderModel(TFPreTrainedModel, TFCausalLanguageModelingLoss):
    r"""
    [`TFVisionEncoderDecoderModel`] 是一个通用模型类，当使用 [`~TFAutoModel.from_pretrained`] 类方法为编码器创建一个基本视觉模型类，
    并使用 [`~TFAutoModelForCausalLM.from_pretrained`] 类方法为解码器创建另一个基本模型类时，它将被实例化为一个转换器架构。
    """

    config_class = VisionEncoderDecoderConfig  # 配置类为 VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"  # 基础模型前缀为 "vision_encoder_decoder"
    load_weight_prefix = "tf_vision_encoder_decoder_model"  # 加载权重前缀为 "tf_vision_encoder_decoder_model"
    main_input_name = "pixel_values"  # 主输入名称为 "pixel_values"

    # 初始化函数，接受配置、编码器和解码器作为参数
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[TFPreTrainedModel] = None,
        decoder: Optional[TFPreTrainedModel] = None,
        ):
            # 检查配置是否为 None，并且编码器和解码器必须同时提供，否则抛出数值错误异常
            if config is None and (encoder is None or decoder is None):
                raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
            # 如果配置为 None，则从提供的编码器和解码器配置创建 VisionEncoderDecoderConfig 对象
            if config is None:
                config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
            else:
                # 如果提供的配置不是 self.config_class 类型，则抛出数值错误异常
                if not isinstance(config, self.config_class):
                    raise ValueError(f"config: {config} has to be of type {self.config_class}")

            # 如果解码器配置中的交叉注意力隐藏大小不为 None
            if config.decoder.cross_attention_hidden_size is not None:
                # 检查解码器的交叉注意力隐藏大小是否等于编码器的隐藏大小，否则抛出数值错误异常
                if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                    raise ValueError(
                        "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                        f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                        f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                        " `config.encoder.hidden_size`."
                    )

            # 使用给定的配置初始化父类
            super().__init__(config)

            # 如果编码器为 None，则从配置创建 TFAutoModel 对象，并命名为 "encoder"
            if encoder is None:
                encoder = TFAutoModel.from_config(config.encoder, name="encoder")

            # 如果解码器为 None，则从配置创建 TFAutoModelForCausalLM 对象，并命名为 "decoder"
            if decoder is None:
                decoder = TFAutoModelForCausalLM.from_config(config.decoder, name="decoder")

            # 将编码器和解码器设置为类的属性
            self.encoder = encoder
            self.decoder = decoder

            # 如果编码器的配置与类的配置不同，发出警告信息
            if self.encoder.config.to_dict() != self.config.encoder.to_dict():
                logger.warning(
                    f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                    f" {self.config.encoder}"
                )
            # 如果解码器的配置与类的配置不同，发出警告信息
            if self.decoder.config.to_dict() != self.config.decoder.to_dict():
                logger.warning(
                    f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                    f" {self.config.decoder}"
                )

            # 确保各模型的配置与共享配置保持同步
            self.encoder.config = self.config.encoder
            self.decoder.config = self.config.decoder

            # 如果编码器输出具有嵌入层，则抛出数值错误异常
            if (
                self.encoder.get_output_embeddings() is not None:
                raise ValueError(
                    f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
                )
    def input_signature(self):
        # 获取视觉编码器的配置
        vision_config = self.config.encoder
        # 检查是否存在额外的视觉配置，如果有则使用它
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config
        # 检查视觉配置中是否定义了图像尺寸，如果没有则使用输入尺寸作为默认值
        if hasattr(vision_config, "image_size"):
            image_size = vision_config.image_size
        else:
            image_size = vision_config.input_size
        # 返回输入签名字典，包括像素值和解码器输入 ID 的 TensorSpec
        return {
            "pixel_values": tf.TensorSpec(
                shape=(
                    None,
                    vision_config.num_channels,
                    image_size,
                    image_size,
                ),
                dtype=tf.float32,
            ),
            "decoder_input_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="decoder_input_ids"),
        }

    def get_encoder(self):
        # 返回当前对象的编码器
        return self.encoder

    def get_decoder(self):
        # 返回当前对象的解码器
        return self.decoder

    def get_input_embeddings(self):
        # 返回编码器的输入嵌入
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        # 返回解码器的输出嵌入
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        # 设置解码器的输出嵌入
        return self.decoder.set_output_embeddings(new_embeddings)

    def tf_to_pt_weight_rename(self, tf_weight):
        # 根据不同的情况，重命名 TensorFlow 到 PyTorch 的权重名称
        # 这是为了解决 TensorFlow 和 PyTorch 模型结构不完全对齐的问题
        encoder_model_type = self.config.encoder.model_type
        if "encoder" in tf_weight and "decoder" not in tf_weight:
            return (re.sub(rf"encoder\.{encoder_model_type}\.", "encoder.", tf_weight),)
        else:
            return (tf_weight,)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ):
        # 从预训练的编码器和解码器模型构建一个新的对象
        # 这是一个类方法，用于初始化对象
        pass  # Placeholder for method implementation

    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        VISION_ENCODER_DECODER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, **kwargs):
        # 在模型前向传播时执行一些预处理和文档化操作
        pass  # Placeholder for method implementation
    # 定义一个方法 `call`，接受多个参数：
    # - pixel_values: 像素值，可以是 numpy 数组、Tensor 或 None
    # - decoder_input_ids: 解码器输入的 ID，可以是 numpy 数组、Tensor 或 None
    # - decoder_attention_mask: 解码器注意力掩码，可以是 numpy 数组、Tensor 或 None
    # - encoder_outputs: 编码器输出，可以是元组或 TFBaseModelOutput 类型的可选项
    # - past_key_values: 缓存的键值对，是一个元组，包含 numpy 数组或 Tensor 的元组的可选项
    # - decoder_inputs_embeds: 解码器输入的嵌入，可以是 numpy 数组、Tensor 或 None
    # - labels: 标签，可以是 numpy 数组、Tensor 或 None
    # - use_cache: 是否使用缓存，布尔类型的可选项
    # - output_attentions: 是否输出注意力权重，布尔类型的可选项
    # - output_hidden_states: 是否输出隐藏状态，布尔类型的可选项
    # - return_dict: 是否返回字典形式的结果，布尔类型的可选项
    # - training: 是否处于训练模式，布尔类型，默认为 False
    # - **kwargs: 其他关键字参数

    def serving_output(self, output):
        # 如果配置指定使用缓存，则提取输出中的 past_key_values 的第二个元素作为 pkv，否则设为 None
        pkv = tf.tuple(output.past_key_values)[1] if self.config.decoder.use_cache else None
        # 如果配置要求输出解码器隐藏状态，则转换输出中的 decoder_hidden_states 为 Tensor，否则设为 None
        dec_hs = (
            tf.convert_to_tensor(output.decoder_hidden_states) if self.config.decoder.output_hidden_states else None
        )
        # 如果配置要求输出解码器注意力权重，则转换输出中的 decoder_attentions 为 Tensor，否则设为 None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.decoder.output_attentions else None
        # 如果配置要求输出编码器隐藏状态，则转换输出中的 encoder_hidden_states 为 Tensor，否则设为 None
        enc_hs = (
            tf.convert_to_tensor(output.encoder_hidden_states) if self.config.encoder.output_hidden_states else None
        )
        # 如果配置要求输出编码器注意力权重，则转换输出中的 encoder_attentions 为 Tensor，否则设为 None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.encoder.output_attentions else None
        # 如果配置要求输出交叉注意力权重，并且输出中有 cross_attentions，则转换输出中的 cross_attentions 为 Tensor，否则设为 None
        cross_attns = (
            tf.convert_to_tensor(output.cross_attentions)
            if self.config.decoder.output_attentions and output.cross_attentions is not None
            else None
        )

        # 返回 TFSeq2SeqLMOutput 类的实例，包括输出的逻辑 logits、缓存的 past_key_values、解码器隐藏状态、解码器注意力权重、
        # 编码器最后的隐藏状态、编码器隐藏状态、编码器注意力权重和交叉注意力权重
        return TFSeq2SeqLMOutput(
            logits=output.logits,
            past_key_values=pkv,
            decoder_hidden_states=dec_hs,
            decoder_attentions=dec_attns,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=enc_hs,
            encoder_attentions=enc_attns,
            cross_attentions=cross_attns,
        )

    # 定义一个方法 `prepare_inputs_for_generation`，用于为生成准备输入
    # - input_ids: 输入的 ID
    # - past_key_values: 缓存的键值对，可选项
    # - attention_mask: 注意力掩码，可选项
    # - use_cache: 是否使用缓存，可选项
    # - encoder_outputs: 编码器输出，可选项
    # - **kwargs: 其他关键字参数
        ):
        # 准备解码器的输入，使用当前的输入 ID 和过去的键值对
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # 获取解码器的注意力掩码，如果存在的话
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        # 获取过去的键值对
        past_key_values = decoder_inputs.get("past_key_values")
        # 构建输入字典，包括像素值（传递以确保 Keras.layer.__call__ 正常工作）、注意力掩码、解码器的注意力掩码、解码器的输入 ID
        input_dict = {
            "pixel_values": None,  # 需要传递以确保 Keras.layer.__call__ 正常工作
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            # TODO (joao): 在生成重构完成后，应该不再需要 `TFBaseModelOutput` 包装器
            "encoder_outputs": TFBaseModelOutput(last_hidden_state=encoder_outputs[0]),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        # 返回构建好的输入字典
        return input_dict

    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        # 根据标签准备解码器的输入 ID，右移标签以适应解码器的输入要求
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        # 抛出未实现错误，因为不支持通过 TFVisionEncoderDecoderModel 直接调整嵌入层大小
        raise NotImplementedError(
            "Resizing the embedding layers via the TFVisionEncoderDecoderModel directly is not supported. "
            "Please use the respective methods of the wrapped objects (model.decoder.resize_token_embeddings(...))"
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 enc_to_dec_proj 属性，则构建它的计算图
        if getattr(self, "enc_to_dec_proj", None) is not None:
            with tf.name_scope(self.enc_to_dec_proj.name):
                self.enc_to_dec_proj.build([None, None, self.encoder.config.hidden_size])
        # 如果存在 encoder 属性，则构建它的计算图
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在 decoder 属性，则构建它的计算图
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
```