# `.\transformers\models\xglm\modeling_tf_xglm.py`

```py
# 导入必要的库和模块
import math
import random
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFSharedEmbeddings,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_xglm import XGLMConfig

# 获取日志器
logger = logging.get_logger(__name__)

# 定义一些常量
_CHECKPOINT_FOR_DOC = "facebook/xglm-564M"
_CONFIG_FOR_DOC = "XGLMConfig"

# 预训练模型列表
TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xglm-564M",
    # See all XGLM models at https://huggingface.co/models?filter=xglm
]

# 定义一个很大的负数值
LARGE_NEGATIVE = -1e8

# 创建正弦波位置编码
def create_sinusoidal_positions(num_positions: int, embedding_dim: int, padding_idx: Optional[int]) -> tf.Tensor:
    # 计算正弦波参数
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    
    # 根据位置和参数计算正弦波
    emb = tf.expand_dims(tf.range(num_positions, dtype=tf.float32), axis=1) * tf.expand_dims(emb, axis=0)
    emb = tf.reshape(tf.concat([tf.sin(emb), tf.cos(emb)], axis=1), (num_positions, -1))
    
    # 如果embedding_dim是奇数，则在最后添加一个0
    if embedding_dim % 2 == 1:
        emb = tf.concat([emb, tf.zeros((num_positions, 1))], axis=1)
    
    # 如果指定了padding_idx，则对应位置的向量设为0
    if padding_idx is not None:
        _padding_mask = tf.concat(
            [
                tf.ones((padding_idx, shape_list(emb)[1])),
                tf.zeros((1, shape_list(emb)[1])),
                tf.ones((shape_list(emb)[0] - padding_idx - 1, shape_list(emb)[1])),
            ],
            axis=0,
        )
        emb *= _padding_mask
    
    return tf.constant(emb, name="embed_positions")

# 根据输入ID创建位置ID
def _create_position_ids_from_input_ids(
    input_ids: tf.Tensor, past_key_values_length: int, padding_idx: Optional[int]
) -> tf.Tensor:
    """
    """
    将非填充符号替换为它们的位置数。位置数从padding_idx + 1开始。忽略填充符号。这是从fairseq的 `utils.make_positions`修改而来。
    """
    # 根据输入的input_ids和padding_idx生成一个掩码，1表示非填充符号，0表示填充符号
    mask = tf.where(input_ids != padding_idx, 1, 0)
    # 使用cumsum函数计算掩码的累加和，并加上past_key_values_length，再乘以mask，得到增量索引
    incremental_indices = (tf.cast(tf.cumsum(mask, axis=1), dtype=mask.dtype) + past_key_values_length) * mask
    # 将增量索引转换为int64类型，并加上padding_idx作为最终的位置数
    return tf.cast(incremental_indices, dtype=tf.int64) + padding_idx
# 根据输入的嵌入向量和过去的 key-value 长度计算位置 ID
def _create_position_ids_from_inputs_embeds(
    inputs_embeds: tf.Tensor, past_key_values_length: int, padding_idx: Optional[int]
) -> tf.Tensor:
    # 获取输入嵌入的形状
    input_shape = shape_list(inputs_embeds)[:-1]
    # 获取序列长度
    sequence_length = input_shape[1]
    # 根据 padding 索引和序列长度创建位置 ID 序列
    position_ids = tf.range(padding_idx + 1, sequence_length + padding_idx + 1, dtype=tf.int64)
    # 将位置 ID 扩展到和输入嵌入形状一致
    return tf.broadcast_to(tf.expand_dims(position_ids, axis=0), input_shape) + past_key_values_length


# 创建用于自注意力的因果掩码
def _make_causal_mask(input_ids_shape: tf.TensorShape, past_key_values_length: int = 0):
    # 获取批量大小和目标序列长度
    bsz = input_ids_shape[0]
    tgt_len = input_ids_shape[1]
    # 创建一个 tgt_len x tgt_len 的掩码矩阵，初始值为 LARGE_NEGATIVE
    mask = tf.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    # 创建一个 tgt_len 长度的索引序列
    mask_cond = tf.range(shape_list(mask)[-1])
    # 将掩码矩阵中小于当前索引的位置设为 0
    mask = tf.where(mask_cond < tf.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)
    # 如果有过去的 key-value 长度，则在掩码前面添加 0 区域
    if past_key_values_length > 0:
        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length)), mask], axis=-1)
    # 将掩码复制到批量维度
    return tf.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


# 扩展注意力掩码到合适的形状
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    # 获取掩码的源序列长度
    src_len = shape_list(mask)[1]
    # 如果没有指定目标序列长度，则使用源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 将掩码从 2D 扩展到 4D
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    # 返回扩展后的掩码，其中 1 表示被遮蔽，LARGE_NEGATIVE 表示未被遮蔽
    return (1.0 - expanded_mask) * LARGE_NEGATIVE


# XGLM 注意力层的实现
class TFXGLMAttention(tf.keras.layers.Layer):
    # 初始化注意力层，包括嵌入维度、头数、dropout 等参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        # 调用父类的构造方法，传入任意额外关键字参数
        super().__init__(**kwargs)
        # 设置嵌入维度
        self.embed_dim = embed_dim

        # 设置注意力头数
        self.num_heads = num_heads
        # 创建丢弃层
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 计算每个注意力头的维度
        self.head_dim = embed_dim // num_heads
        # 检查是否可以均分嵌入维度
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放系数
        self.scaling = self.head_dim**-0.5
        # 判断是否为解码器
        self.is_decoder = is_decoder

        # 创建 K 投影层
        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="k_proj")
        # 创建 Q 投影层
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        # 创建 V 投影层
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        # 创建输出投影层
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int, bsz: int):
        # 重新排列张量形状，以适应多头自注意力的计算
        return tf.transpose(tf.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states: tf.Tensor,
        key_value_states: tf.Tensor | None = None,
        past_key_value: Tuple[Tuple[tf.Tensor]] | None = None,
        attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 构建网络层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建 K 投影层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 构建 Q 投影层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 构建 V 投影层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 构建输出投影层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 定义 TF 函数式 API 的自定义层，继承自 tf.keras.layers.Layer
class TFXGLMDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: XGLMConfig, **kwargs: Any) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 获取配置中的嵌入维度作为自定义层的嵌入维度
        self.embed_dim = config.d_model
        # 创建自注意力层，用于处理自注意力机制
        self.self_attn = TFXGLMAttention(
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            name="self_attn",
        )
        # 添加随机失活层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数，并添加随机失活层
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)

        # 如果配置中需要跨注意力，创建相应的编码器注意力层和层归一化层
        if config.add_cross_attention:
            self.encoder_attn = TFXGLMAttention(
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
                name="encoder_attn",
            )
            self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(
                epsilon=1e-5, name="encoder_attn_layer_norm"
            )

        # 创建自注意力层的归一化层
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="self_attn_layer_norm")
        # 创建全连接层 fc1
        self.fc1 = tf.keras.layers.Dense(config.ffn_dim, name="fc1")
        # 创建全连接层 fc2
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        # 创建最终的归一化层
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")
        # 将配置保存到当前自定义层的配置中
        self.config = config

    # 从 transformers 中的相关模块中复制代码，定义自定义层的调用方法
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        encoder_hidden_states: tf.Tensor | None = None,
        encoder_attention_mask: tf.Tensor | None = None,
        layer_head_mask: tf.Tensor | None = None,
        cross_attn_layer_head_mask: tf.Tensor | None = None,
        past_key_value: Tuple[tf.Tensor] | None = None,
        training: Optional[bool] = False,
    # 定义神经网络结构的构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在self_attn属性，执行下面的代码块
        if getattr(self, "self_attn", None) is not None:
            # 在TensorFlow中创建一个命名空间
            with tf.name_scope(self.self_attn.name):
                # 调用self_attn的build方法
                self.self_attn.build(None)
        # 如果存在self_attn_layer_norm属性，执行下面的代码块
        if getattr(self, "self_attn_layer_norm", None) is not None:
            # 在TensorFlow中创建一个命名空间
            with tf.name_scope(self.self_attn_layer_norm.name):
                # 调用self_attn_layer_norm的build方法，传入参数[None, None, self.embed_dim]
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        # 如果存在fc1属性，执行下面的代码块
        if getattr(self, "fc1", None) is not None:
            # 在TensorFlow中创建一个命名空间
            with tf.name_scope(self.fc1.name):
                # 调用fc1的build方法，传入参数[None, None, self.embed_dim]
                self.fc1.build([None, None, self.embed_dim])
        # 如果存在fc2属性，执行下面的代码块
        if getattr(self, "fc2", None) is not None:
            # 在TensorFlow中创建一个命名空间
            with tf.name_scope(self.fc2.name):
                # 调用fc2的build方法，传入参数[None, None, self.config.ffn_dim]
                self.fc2.build([None, None, self.config.ffn_dim])
        # 如果存在final_layer_norm属性，执行下面的代码块
        if getattr(self, "final_layer_norm", None) is not None:
            # 在TensorFlow中创建一个命名空间
            with tf.name_scope(self.final_layer_norm.name):
                # 调用final_layer_norm的build方法，传入参数[None, None, self.embed_dim]
                self.final_layer_norm.build([None, None, self.embed_dim])
        # 如果存在encoder_attn属性，执行下面的代码块
        if getattr(self, "encoder_attn", None) is not None:
            # 在TensorFlow中创建一个命名空间
            with tf.name_scope(self.encoder_attn.name):
                # 调用encoder_attn的build方法
                self.encoder_attn.build(None)
        # 如果存在encoder_attn_layer_norm属性，执行下面的代码块
        if getattr(self, "encoder_attn_layer_norm", None) is not None:
            # 在TensorFlow中创建一个命名空间
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                # 调用encoder_attn_layer_norm的build方法，传入参数[None, None, self.embed_dim]
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
# 定义一个可序列化的 keras 层 TFXGLMMainLayer
@keras_serializable
class TFXGLMMainLayer(tf.keras.layers.Layer):
    # 设置配置类为 XGLMConfig
    config_class = XGLMConfig

    # 初始化函数，接受配置和词嵌入作为输入
    def __init__(
        self, config: XGLMConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, *inputs, **kwargs: Any
    ) -> None:
        super().__init__(*inputs, **kwargs)
        
        # 初始化配置信息
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 初始化词嵌入
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = TFSharedEmbeddings(
                config.vocab_size, config.d_model, self.padding_idx, name="embed_tokens"
            )

        # 初始化偏移量
        self.offset = 2
        # 创建正弦位置编码权重
        self._embed_positions_weights = create_sinusoidal_positions(
            num_positions=config.max_position_embeddings + self.offset,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # 初始化 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 创建多层解码器层
        self.layers = [TFXGLMDecoderLayer(config, name=f"layers.{i}") for i in range(config.num_layers)]
        self.layerdrop = config.layerdrop
        # 初始化层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    # 获取输入词嵌入
    def get_input_embeddings(self) -> TFSharedEmbeddings:
        return self.embed_tokens

    # 设置输入词嵌入
    def set_input_embeddings(self, value: TFSharedEmbeddings) -> None:
        self.embed_tokens = value

    # 准备解码器注意力掩码
    def _prepare_decoder_attention_mask(
        self,
        attention_mask: tf.Tensor | None,
        input_shape: tf.TensorShape,
        past_key_values_length: int,
    ) -> tf.Tensor:
        # 创建因果关系掩码
        combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length)
        combined_attention_mask = tf.cond(
            input_shape[-1] > 1, lambda: combined_attention_mask, lambda: tf.ones_like(combined_attention_mask)
        )
        if attention_mask is None:
            return combined_attention_mask
        expand_attention_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
        return expand_attention_mask + combined_attention_mask

    # 嵌入位置信息
    def embed_positions(self, position_ids: np.ndarray | tf.Tensor | None = None) -> tf.Tensor:
        position_ids += self.offset
        positions = tf.gather(self._embed_positions_weights, position_ids, axis=0)
        return positions

    @unpack_inputs
    # 定义一个函数call，用于处理模型输入及参数，返回结果
    def call(
        # 定义参数input_ids，类型为TFModelInputType或None，表示输入的token ID序列
        self,
        input_ids: TFModelInputType | None = None,
        # 定义参数attention_mask，类型为np.ndarray或tf.Tensor或None，表示attention mask矩阵
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 定义参数position_ids，类型为np.ndarray或tf.Tensor或None，表示位置ID序列
        position_ids: np.ndarray | tf.Tensor | None = None,
        # 定义参数encoder_hidden_states，类型为np.ndarray或tf.Tensor或None，表示编码器的隐藏状态
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        # 定义参数encoder_attention_mask，类型为np.ndarray或tf.Tensor或None，表示编码器的attention mask
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 定义参数head_mask，类型为np.ndarray或tf.Tensor或None，表示注意力头部的mask
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 定义参数cross_attn_head_mask，类型为np.ndarray或tf.Tensor或None，表示交叉注意力头部的mask
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 定义参数past_key_values，类型为Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]或None，表示过去的key-value对
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 定义参数inputs_embeds，类型为np.ndarray或tf.Tensor或None，表示嵌入输入的特征
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 定义参数use_cache，类型为Optional[bool]或None，表示是否使用缓存
        use_cache: Optional[bool] = None,
        # 定义参数output_attentions，类型为Optional[bool]或None，表示是否输出attention信息
        output_attentions: Optional[bool] = None,
        # 定义参数output_hidden_states，类型为Optional[bool]或None，表示是否输出隐藏状态信息
        output_hidden_states: Optional[bool] = None,
        # 定义参数return_dict，类型为Optional[bool]或None，表示是否返回字典形式的输出
        return_dict: Optional[bool] = None,
        # 定义参数training，类型为Optional[bool]或False，表示是否处于训练模式
        training: Optional[bool] = False,
        # 定义额外的关键字参数
        **kwargs: Any,
        
    # 定义一个build方法，用于构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置已构建标志为True
        self.built = True
        # 如果存在layer_norm属性，则构建layer_norm
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        # 如果存在embed_tokens属性，则构建embed_tokens
        if getattr(self, "embed_tokens", None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        # 如果存在layers属性，则逐层构建layers
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 定义一个名为TFXGLMPreTrainedModel的类，继承自TFPreTrainedModel类
class TFXGLMPreTrainedModel(TFPreTrainedModel):
    # 指定配置类为XGLMConfig
    config_class = XGLMConfig
    # 指定基础模型前缀为"model"
    base_model_prefix = "model"

# 原始文档字符串，包含有关此模型的详细说明和用法
XGLM_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    ...
    ... # 此处为文档字符串的详细说明, 包括模型的输入格式，参数等
    ...
"""

XGLM_INPUTS_DOCSTRING = r"""
"""

# 为XGLMModel添加文档字符串，描述此模型的用途和详细说明
@add_start_docstrings(
    "The bare XGLM Model transformer outputting raw hidden-states without any specific head on top.",
    XGLM_START_DOCSTRING,
)
class TFXGLMModel(TFXGLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_layers* layers. Each layer is a [`TFXGLMDecoderLayer`]
    ...
    ... # 这里可能包含更多的模型描述和用法说明
    ...
    # 初始化函数，接受 XGLMConfig 和可选的 TFSharedEmbeddings 输入，并继承父类的属性和方法
    def __init__(
        self, config: XGLMConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, *inputs: Any, **kwargs: Any
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFXGLMMainLayer 模型层，并传入配置和嵌入标记
        self.model = TFXGLMMainLayer(config, embed_tokens=embed_tokens, name="model")

    # 调用函数装饰器，将 call 函数进行装饰，添加额外的文档字符串和代码示例文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播函数，接受多种输入参数，返回模型输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        # 调用模型的前向传播函数，传入所有输入参数，并获取模型的输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型的输出
        return outputs

    # 构建模型，如果已经构建过则直接返回，否则进行构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 检查模型是否存在，如果存在则使用 tf.name_scope 进行构建
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
# 为了添加起始文档字符串，创建一个 XGLM Model 转换器，带有放在顶部的语言建模头部 (线性层，其权重与输入嵌入层相连)
@add_start_docstrings(
    """
    The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XGLM_START_DOCSTRING,
)
class TFXGLMForCausalLM(TFXGLMPreTrainedModel, TFCausalLanguageModelingLoss):
    # 指定基本模型前缀
    base_model_prefix = "model"
    # 在加载时忽略的键列表 (当缺少时)
    _keys_to_ignore_on_load_missing = [
        r"model.embed_positions.weights",
        r"lm_head.weight",
    ]
    # 在保存时忽略的键列表
    _keys_to_ignore_on_save = [
        r"model.embed_positions.weights",
    ]

    # 初始化函数
    def __init__(
        self, config: XGLMConfig, embed_tokens: Optional[TFSharedEmbeddings] = None, *inputs: Any, **kwargs: Any
    ) -> None:
        # 调用父类初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 创建 XGLM 主模型层
        self.model = TFXGLMMainLayer(config, embed_tokens=embed_tokens, name="model")
        # 创建语言建模头部 (Dense 层)
        self.lm_head = tf.keras.layers.Dense(
            config.vocab_size,
            use_bias=False,
            kernel_initializer=get_initializer(config.init_std),
            name="lm_head",
        )
        self.config = config

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 为生成器准备输入
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 如果 past 在 kwargs 中定义，则仅使用最后一个 token 作为 inputs_ids
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)

        # 获取 position_ids 和 attention_mask
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        # 如果 attention_mask 存在且 position_ids 不存在，则根据 attention_mask 累积生成 position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        # 返回输入参数字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 添加输入解包装，开始模型前向传播文档字符串，返回类型替换文档字符串和代码示例文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个函数，接受多个参数，其中大部分参数的类型为TFModelInputType或者None
    def call(
        self,
        # 输入的 token IDs，数据类型为TFModelInputType或者None，默认为None
        input_ids: TFModelInputType | None = None,
        # 注意力掩码，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        attention_mask: np.ndarray | tf.Tensor | None = None,
        # 位置编码，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        position_ids: np.ndarray | tf.Tensor | None = None,
        # 编码器隐藏状态，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        # 编码器注意力掩码，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        # 头掩码，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        head_mask: np.ndarray | tf.Tensor | None = None,
        # 跨注意力头掩码，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        cross_attn_head_mask: np.ndarray | tf.Tensor | None = None,
        # 过去的键值对，数据类型为Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]或者None，默认为None
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        # 输入的嵌入数据，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        # 标签数据，数据类型为np.ndarray或者tf.Tensor或者None，默认为None
        labels: np.ndarray | tf.Tensor | None = None,
        # 是否使用缓存，数据类型为Optional[bool]或者None，默认为None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，数据类型为Optional[bool]或者None，默认为None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，数据类型为Optional[bool]或者None，默认为None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的结果，数据类型为Optional[bool]或者None，默认为None
        return_dict: Optional[bool] = None,
        # 是否在训练模式，数据类型为Optional[bool]，默认为False
        training: Optional[bool] = False,
        # 其他参数，数据类型为Any
        **kwargs: Any,
    def call(self, input_ids: tf.Tensor, attention_mask: tf.Tensor = None, position_ids: tf.Tensor = None,
             encoder_hidden_states: tf.Tensor = None, encoder_attention_mask: tf.Tensor = None,
             head_mask: tf.Tensor = None, cross_attn_head_mask: tf.Tensor = None, past_key_values: Union[Dict, None] = None,
             inputs_embeds: tf.Tensor = None, use_cache: bool = None, output_attentions: bool = None,
             output_hidden_states: bool = None, return_dict: bool = None, training: bool = None,
             labels: Union[np.ndarray, tf.Tensor] = None) -> Union[TFCausalLMOutputWithCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]`. All labels set to `-100`
            are ignored (masked), and the loss is only computed for labels in `[0, ..., config.vocab_size]`.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # shift labels to the left and cut the last logit token
            labels = tf.concat(
                [labels[:, 1:], tf.fill((labels.shape[0], 1), tf.cast(self.config.pad_token_id, labels.dtype))],
                axis=-1,
            )
            loss = self.hf_compute_loss(labels, lm_logits)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "model", None) is not None:
            with tf.name_scope(self.model.name):
                self.model.build(None)
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.hidden_size])

    def tf_to_pt_weight_rename(self, tf_weight):
        if tf_weight == "lm_head.weight":
            return tf_weight, "model.embed_tokens.weight"
        else:
            return (tf_weight,)
```