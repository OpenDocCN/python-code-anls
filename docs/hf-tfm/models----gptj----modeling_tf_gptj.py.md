# `.\models\gptj\modeling_tf_gptj.py`

```
# 设置文件编码为UTF-8，确保可以正确处理中文和其他特殊字符
# 版权声明，声明代码版权归EleutherAI和HuggingFace团队所有
#
# 根据Apache许可证2.0版，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据此许可证分发的软件是基于“按原样”分发的，
# 不附带任何明示或暗示的保证或条件。请参阅许可证获取更多详情。
""" TF 2.0 GPT-J模型 """

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 导入自定义模块和函数
from ...activations_tf import get_tf_activation
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPast,
    TFCausalLMOutputWithPast,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFSharedEmbeddings,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_gptj import GPTJConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置
_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-j-6B"
_CONFIG_FOR_DOC = "GPTJConfig"

# 预训练模型存档列表
GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-j-6B",
    # 更多GPT-J模型详见 https://huggingface.co/models?filter=gptj
]
    # 初始化方法，接受一个GPTJConfig对象和其他关键字参数
    def __init__(self, config: GPTJConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_attention_heads
        # 检查embed_dim是否能被num_attention_heads整除
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        # 设置注意力的缩放因子
        self.scale_attn = self.head_dim**0.5
        # 设置旋转维度
        self.rotary_dim = config.rotary_dim

        # 设置注意力的dropout层
        self.attn_dropout = keras.layers.Dropout(config.attn_pdrop)
        # 设置残差连接的dropout层
        self.resid_dropout = keras.layers.Dropout(config.resid_pdrop)

        # 初始化查询投影层
        self.q_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="q_proj",
        )
        # 初始化键投影层
        self.k_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="k_proj",
        )
        # 初始化值投影层
        self.v_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="v_proj",
        )
        # 初始化输出投影层
        self.out_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj",
        )

        # 设置最大位置编码
        self.max_positions = config.max_position_embeddings
        # 创建一个下三角形的掩码矩阵
        self.lower_triangle_mask = tf.reshape(
            tf.cast(tf.experimental.numpy.tril(tf.ones((self.max_positions, self.max_positions))), tf.int8),
            (1, 1, self.max_positions, self.max_positions),
        )
        # 确定位置编码的维度
        pos_embd_dim = self.rotary_dim or self.embed_dim
        # 创建正弦位置编码
        self.embed_positions = create_sinusoidal_positions(self.max_positions, pos_embd_dim)

    # 获取因果掩码，用于自注意力机制
    def get_causal_mask(self, key_length, query_length) -> tf.Tensor:
        return tf.cast(self.lower_triangle_mask[:, :, key_length - query_length : key_length, :key_length], tf.bool)

    # 静态方法，返回一个用于掩码的偏置
    @staticmethod
    def get_masked_bias(dtype: tf.DType) -> tf.Tensor:
        return tf.cast(tf.constant(-1e9), dtype)
    def _split_heads(self, hidden_states: tf.Tensor, rotary: bool) -> tf.Tensor:
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # Compute the new shape for splitting heads
        new_shape = shape_list(hidden_states)[:-1] + [self.num_attention_heads, self.head_dim]
        # Reshape the tensor to split heads
        hidden_states = tf.reshape(hidden_states, new_shape)
        if rotary:
            return hidden_states
        # Transpose tensor dimensions based on its rank
        if len(shape_list(hidden_states)) == 4:
            return tf.transpose(hidden_states, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)
        elif len(shape_list(hidden_states)) == 5:
            return tf.transpose(hidden_states, (0, 1, 3, 2, 4))  # (batch, blocks, head, block_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}")

    def _merge_heads(self, hidden_states: tf.Tensor) -> tf.Tensor:
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # Transpose tensor dimensions to merge heads back
        if len(shape_list(hidden_states)) == 4:
            hidden_states = tf.transpose(hidden_states, (0, 2, 1, 3))
        elif len(shape_list(hidden_states)) == 5:
            hidden_states = tf.transpose(hidden_states, (0, 1, 3, 2, 4))
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}")
        # Compute the new shape after merging heads
        new_shape = shape_list(hidden_states)[:-2] + [self.num_attention_heads * self.head_dim]
        return tf.reshape(hidden_states, new_shape)

    def _attn(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # compute causal mask from causal mask buffer
        query_length, key_length = shape_list(query)[-2], shape_list(key)[-2]
        # Generate a causal mask for self-attention
        causal_mask = self.get_causal_mask(key_length, query_length)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)

        # Compute attention weights
        attn_weights = tf.matmul(query, key, transpose_b=True)
        # Apply causal mask to attention weights
        attn_weights = tf.where(causal_mask, attn_weights, self.get_masked_bias(attn_weights.dtype))

        # Scale attention weights
        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply additional attention mask
            attn_weights = attn_weights + attention_mask

        # Apply stable softmax to compute attention probabilities
        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, value.dtype)
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if specified
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # Compute attention output by weighted sum of values
        attn_output = tf.matmul(attn_weights, value)

        return attn_output, attn_weights
    # 定义一个方法，用于处理自注意力机制的计算，输入包括隐藏状态、过去的键值对、注意力掩码、位置编码、头掩码、缓存使用标志和是否输出注意力权重
    def call(
        self,
        hidden_states: tf.Tensor,  # 输入的隐藏状态张量
        layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,  # 可选的过去层的键值对
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量，可为None
        position_ids: tf.Tensor | None = None,  # 位置编码张量，可为None
        head_mask: tf.Tensor | None = None,  # 头掩码张量，可为None
        use_cache: bool = False,  # 是否使用缓存，默认为False
        output_attentions: bool = False,  # 是否输出注意力权重，默认为False
    ):
        # 使用三个不同的线性投影来生成查询、键和值
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 将查询、键和值张量分割成多头
        query = self._split_heads(query, True)
        key = self._split_heads(key, True)
        value = self._split_heads(value, False)

        # 根据位置编码应用旋转位置嵌入（如果提供了旋转维度）
        sincos = tf.cast(tf.gather(self.embed_positions, position_ids, axis=0), hidden_states.dtype)
        sincos = tf.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            # 合并旋转后的部分和传递部分的键和查询
            key = tf.concat((k_rot, k_pass), axis=-1)
            query = tf.concat((q_rot, q_pass), axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        # 转置键和查询张量的维度
        key = tf.transpose(key, (0, 2, 1, 3))
        query = tf.transpose(query, (0, 2, 1, 3))

        # 如果提供了过去的键值对，则将当前键和值与过去的键值对连接起来
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = tf.concat((past_key, key), axis=-2)
            value = tf.concat((past_value, value), axis=-2)

        # 如果设置了使用缓存，则将当前的键值对作为“present”返回，否则返回None
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 计算自注意力机制的输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并多头注意力的输出
        attn_output = self._merge_heads(attn_output)
        
        # 通过输出投影层处理注意力输出
        attn_output = self.out_proj(attn_output)
        
        # 应用残差连接和dropout到注意力输出
        attn_output = self.resid_dropout(attn_output)

        # 构造最终输出元组，包括注意力输出和可能的“present”和注意力权重（如果设置输出注意力权重）
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # 返回最终的输出元组：注意力输出、可能的“present”和（如果设置输出注意力权重）注意力权重
    # 如果模型已经建立，则直接返回，不进行重复建立
    if self.built:
        return
    
    # 标记模型为已建立状态
    self.built = True
    
    # 如果存在查询投影层对象，建立查询投影层，并指定输入形状为 [None, None, self.embed_dim]
    if getattr(self, "q_proj", None) is not None:
        with tf.name_scope(self.q_proj.name):
            self.q_proj.build([None, None, self.embed_dim])
    
    # 如果存在键投影层对象，建立键投影层，并指定输入形状为 [None, None, self.embed_dim]
    if getattr(self, "k_proj", None) is not None:
        with tf.name_scope(self.k_proj.name):
            self.k_proj.build([None, None, self.embed_dim])
    
    # 如果存在值投影层对象，建立值投影层，并指定输入形状为 [None, None, self.embed_dim]
    if getattr(self, "v_proj", None) is not None:
        with tf.name_scope(self.v_proj.name):
            self.v_proj.build([None, None, self.embed_dim])
    
    # 如果存在输出投影层对象，建立输出投影层，并指定输入形状为 [None, None, self.embed_dim]
    if getattr(self, "out_proj", None) is not None:
        with tf.name_scope(self.out_proj.name):
            self.out_proj.build([None, None, self.embed_dim])
class TFGPTJMLP(keras.layers.Layer):
    # 初始化函数，定义了模型层的各个组件和参数
    def __init__(self, intermediate_size: int, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        embed_dim = config.n_embd

        # 输入层全连接层，用于将输入向量映射到中间维度
        self.fc_in = keras.layers.Dense(
            intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="fc_in"
        )
        # 输出层全连接层，将中间维度映射回原始嵌入维度
        self.fc_out = keras.layers.Dense(
            embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="fc_out"
        )

        # 激活函数，根据配置选择合适的激活函数
        self.act = get_tf_activation(config.activation_function)
        # Dropout 层，用于防止过拟合
        self.dropout = keras.layers.Dropout(config.embd_pdrop)
        self.embed_dim = config.n_embd
        self.intermediate_size = intermediate_size

    # 前向传播函数，定义了层的计算流程
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 输入向量经过输入层全连接层
        hidden_states = self.fc_in(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 经过输出层全连接层
        hidden_states = self.fc_out(hidden_states)
        # 应用 Dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    # 构建函数，用于构建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果输入层存在，构建输入层
        if getattr(self, "fc_in", None) is not None:
            with tf.name_scope(self.fc_in.name):
                self.fc_in.build([None, None, self.embed_dim])
        # 如果输出层存在，构建输出层
        if getattr(self, "fc_out", None) is not None:
            with tf.name_scope(self.fc_out.name):
                self.fc_out.build([None, None, self.intermediate_size])


class TFGPTJBlock(keras.layers.Layer):
    # 初始化函数，定义了模型层的各个组件和参数
    def __init__(self, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        # 内部维度，用于确定 MLP 层的中间维度大小
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        # 第一层的 LayerNormalization 层
        self.ln_1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        # 自注意力层
        self.attn = TFGPTJAttention(config, name="attn")
        # MLP 层，用于处理经过自注意力层后的隐藏状态
        self.mlp = TFGPTJMLP(inner_dim, config, name="mlp")
        self.config = config

    # 前向传播函数，定义了层的计算流程
    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        # 使用自注意力的输出和输入的张量、位置的 ID 的张量和头部面具的张量的张量，以及缓存的缓存的布尔值使用的缓存的注意
        ):
            residual = hidden_states
            # 将隐藏状态进行 LayerNormalization
            hidden_states = self.ln_1(hidden_states)
            # 使用注意力机制进行计算
            attn_outputs = self.attn(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )  # attn_outputs: attn_output, present, (attentions)
            # 获取注意力输出
            attn_output = attn_outputs[0]
            # 剩余连接和前馈神经网络
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = attn_output + feed_forward_hidden_states + residual

            # 如果使用缓存，则输出包含隐藏状态
            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                # 否则，输出中去除第一个元素
                outputs = (hidden_states,) + outputs[1:]
            return outputs  # hidden_states, present, (attentions)

        def build(self, input_shape=None):
            if self.built:
                return
            self.built = True
            # 构建 LayerNormalization 层
            if getattr(self, "ln_1", None) is not None:
                with tf.name_scope(self.ln_1.name):
                    self.ln_1.build([None, None, self.config.n_embd])
            # 构建注意力机制
            if getattr(self, "attn", None) is not None:
                with tf.name_scope(self.attn.name):
                    self.attn.build(None)
            # 构建前馈神经网络
            if getattr(self, "mlp", None) is not None:
                with tf.name_scope(self.mlp.name):
                    self.mlp.build(None)
@keras_serializable
class TFGPTJMainLayer(keras.layers.Layer):
    # 使用 keras_serializable 装饰器，表明这是一个可以序列化的 Keras 层
    config_class = GPTJConfig
    # 设置类属性 config_class 为 GPTJConfig，这是用于配置模型的类

    def __init__(self, config: GPTJConfig, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # 调用父类的初始化方法

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict
        # 初始化一些配置参数和控制输出的标志

        self.num_hidden_layers = config.n_layer
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range
        # 从配置中获取模型的隐藏层数、嵌入维度、位置编码数以及初始化范围等属性

        self.wte = TFSharedEmbeddings(
            config.vocab_size, config.hidden_size, initializer_range=config.initializer_range, name="wte"
        )
        # 初始化共享嵌入层对象，用于将输入的 token 序列转换为向量表示
        self.drop = keras.layers.Dropout(config.embd_pdrop)
        # 初始化 Dropout 层，用于在训练过程中随机丢弃部分嵌入层输出
        self.h = [TFGPTJBlock(config, name=f"h_._{i}") for i in range(config.n_layer)]
        # 初始化 GPTJBlock 的列表，每个 block 是 GPTJ 模型中的一个处理块，用于构建完整的模型
        self.ln_f = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        # 初始化 LayerNormalization 层，用于对最终输出进行归一化处理
        self.embed_dim = config.n_embd
        # 设置嵌入维度属性

    def get_input_embeddings(self):
        return self.wte
        # 返回输入嵌入层对象

    def set_input_embeddings(self, value: tf.Tensor):
        self.wte.weight = value
        self.wte.vocab_size = shape_list(value)[0]
        # 设置输入嵌入层的权重和词汇大小属性

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError
        # 剪枝模型中的注意力头部，heads_to_prune 是一个字典，表示每个层需要剪枝的注意力头部列表

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 模型的前向传播函数，接收多个输入参数，并返回模型的输出

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果模型已经构建过，则直接返回

        if getattr(self, "wte", None) is not None:
            with tf.name_scope(self.wte.name):
                self.wte.build(None)
        # 如果存在 wte 属性，则调用其 build 方法构建嵌入层

        if getattr(self, "ln_f", None) is not None:
            with tf.name_scope(self.ln_f.name):
                self.ln_f.build([None, None, self.embed_dim])
        # 如果存在 ln_f 属性，则调用其 build 方法构建 LayerNormalization 层

        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)
        # 如果存在 h 属性，则遍历每个 GPTJBlock 层并调用其 build 方法构建模型中的处理块
    # 此模型继承自 `TFPreTrainedModel`。查看超类文档以了解库实现的通用方法，如下载或保存模型、调整输入嵌入大小、修剪头等。
    
    # 此模型也是一个 `keras.Model` 的子类。可以像普通的 TF 2.0 Keras 模型一样使用它，并参考 TF 2.0 文档了解一般用法和行为。
    
    # <Tip> 标签中的内容是关于 `transformers` 中 TensorFlow 模型和层接受输入的两种格式的说明：
    # - 使用关键字参数作为所有输入（类似于 PyTorch 模型）
    # - 将所有输入作为列表、元组或字典的第一个位置参数
    # 第二种格式的支持是因为 Keras 方法在传递输入给模型和层时更喜欢这种格式。因此，在使用 `model.fit()` 等方法时，只需以 `model.fit()` 支持的任何格式传递输入和标签即可正常工作！然而，如果想在 Keras 方法之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能性可以用来收集所有输入张量到第一个位置参数中：
    # - 仅使用 `input_ids` 作为单个张量：`model(input_ids)`
    # - 使用变长列表，包含按照文档字符串中给定的顺序的一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 使用字典，将一个或多个输入张量与文档字符串中给定的输入名称相关联：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    # 注意，当使用子类化创建模型和层时，无需担心这些，可以像对待任何其他 Python 函数一样传递输入！
    
    # Parameters: 部分描述了模型的参数：
    # - config (`GPTJConfig` 类型)：包含模型所有参数的模型配置类。
    #   初始化配置文件时不会加载与模型关联的权重，仅加载配置。查看 `~TFPreTrainedModel.from_pretrained` 方法以加载模型权重。
"""

GPTJ_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare GPT-J Model transformer outputting raw hidden-states without any specific head on top.",
    GPTJ_START_DOCSTRING,
)
class TFGPTJModel(TFGPTJPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 初始化时设置 GPT-J 主层
        self.transformer = TFGPTJMainLayer(config, name="transformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """
        # 调用 GPT-J 主层的前向传播函数，返回输出结果
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                # 构建 GPT-J 主层
                self.transformer.build(None)


@add_start_docstrings(
    """
    The GPT-J Model transformer with a language modeling head on top.
    """,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForCausalLM(TFGPTJPreTrainedModel, TFCausalLanguageModelingLoss):
    # 这里会定义带有语言建模头部的 GPT-J 模型
    # 初始化方法，接收配置和其他输入参数，并调用父类的初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建一个名为transformer的TFGPTJMainLayer对象，使用给定的配置
        self.transformer = TFGPTJMainLayer(config, name="transformer")
        # 创建一个全连接层，称为lm_head，用于语言模型的输出预测
        self.lm_head = keras.layers.Dense(
            config.vocab_size, kernel_initializer=get_initializer(config.initializer_range), name="lm_head"
        )
        # 将配置信息保存在实例变量中
        self.config = config

    # 返回lm_head，用于获取输出的嵌入表示
    def get_output_embeddings(self):
        return self.lm_head

    # 设置lm_head的值为new_embeddings，用于更新输出的嵌入表示
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 准备用于生成的输入数据，根据传入的参数设置不同的输入
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果past_key_values存在，只使用输入的最后一个token
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            # 如果存在token_type_ids，则也只使用最后一个token的类型
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        # 如果attention_mask存在而position_ids不存在，则根据attention_mask计算position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            # 如果past_key_values存在，只使用计算后的position_ids的最后一个值

        # 返回一个包含所有生成输入的字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    # 调用模型的方法，接收多种输入参数，并进行前向传播计算
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        # 可选的训练参数，控制是否返回字典形式的输出
    ) -> Union[TFCausalLMOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        # 获取transformer模型的输出，根据传入的参数进行调用
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从transformer的输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 使用语言模型头部生成logits
        lm_logits = self.lm_head(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果提供了标签（labels）
        if labels is not None:
            # 将logits向左移动一位并截断最后一个logit token
            shifted_logits = lm_logits[:, :-1]
            # 将标签向右移动一位以匹配shifted_logits的长度
            labels = labels[:, 1:]
            # 计算损失函数
            loss = self.hf_compute_loss(labels, shifted_logits)

        # 如果不需要返回字典，则按照非字典格式返回输出
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，创建TFCausalLMOutputWithPast对象
        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 如果存在transformer模型，则构建transformer模型
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果存在lm_head模型，则构建lm_head模型
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.n_embd])
    """
    The `build` method for constructing the TFGPTJForSequenceClassification model architecture.

    Ensures the model is built correctly by initializing necessary layers based on input shape and configuration.
    """
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        
        # 如果存在 transformer 层，则构建 transformer 层
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        
        # 如果存在 score 层，则构建 score 层，其中输出的形状为 [None, None, self.config.n_embd]
        if getattr(self, "score", None) is not None:
            with tf.name_scope(self.score.name):
                self.score.build([None, None, self.config.n_embd])
    ```
    The GPT-J Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPTJ_START_DOCSTRING,
    )
    # TFGPTJForQuestionAnswering 类的定义，继承自 TFGPTJPreTrainedModel 和 TFQuestionAnsweringLoss
    class TFGPTJForQuestionAnswering(TFGPTJPreTrainedModel, TFQuestionAnsweringLoss):
        # 加载时忽略的键列表，用于处理缺失情况
        _keys_to_ignore_on_load_missing = [r"h.\d+.attn.masked_bias", r"h.\d+.attn.bias", r"lm_head.weight"]

        def __init__(self, config, *inputs, **kwargs):
            super().__init__(config, *inputs, **kwargs)
            # 初始化时设置类别数量
            self.num_labels = config.num_labels
            # 创建 GPT-J 主层实例，并命名为 "transformer"
            self.transformer = TFGPTJMainLayer(config, name="transformer")
            # 初始化问答输出层，使用 Dense 层，内核初始化方式根据配置的初始化范围确定
            self.qa_outputs = keras.layers.Dense(
                self.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
            )
            # 保存配置对象的引用
            self.config = config

        @unpack_inputs
        # 将输入解包并添加到模型前向传播的文档字符串中，描述了输入的格式
        @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        # 添加代码示例的文档字符串，描述了如何使用模型进行问答任务
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=TFQuestionAnsweringModelOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        # 模型的前向传播函数，接收多个输入参数，包括输入的特征、位置编码、注意力掩码等
        def call(
            self,
            input_ids: TFModelInputType | None = None,
            past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            token_type_ids: np.ndarray | tf.Tensor | None = None,
            position_ids: np.ndarray | tf.Tensor | None = None,
            head_mask: np.ndarray | tf.Tensor | None = None,
            inputs_embeds: np.ndarray | tf.Tensor | None = None,
            start_positions: np.ndarray | tf.Tensor | None = None,
            end_positions: np.ndarray | tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            training: Optional[bool] = False,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        定义函数的签名和返回类型注解，指定函数返回的类型是 TFQuestionAnsweringModelOutput 或者 (tf.Tensor, tf.Tensor) 的元组。
        """

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取transformer模型的输出
        sequence_output = transformer_outputs[0]

        # 通过qa_outputs模型计算起始位置和结束位置的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 去除最后一个维度为1的维度
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果提供了起始和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不返回字典，则按照元组的方式返回输出
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则构造 TFQuestionAnsweringModelOutput 对象
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果模型已经构建，则直接返回
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                # 构建transformer模型
                self.transformer.build(None)
        # 如果qa_outputs存在，则构建qa_outputs模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```