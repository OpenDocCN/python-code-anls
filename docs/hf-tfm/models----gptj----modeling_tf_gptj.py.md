# `.\models\gptj\modeling_tf_gptj.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 此文件版权归 EleutherAI 团队和 HuggingFace 团队所有
#
# 根据 Apache 许可证 2.0 版（“许可证”）进行许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”分发，不提供任何形式的明示或暗示保证。
# 有关特定语言的权限，请参阅许可证。
"""TF 2.0 GPT-J 模型。"""

# 引入必要的库
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# 从相关文件中导入函数和类
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
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_gptj import GPTJConfig

# 设置日志记录器
logger = logging.get_logger(__name__)

# 以下两个常量用于文档生成
_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-j-6B"
_CONFIG_FOR_DOC = "GPTJConfig"

# 预训练模型列表
GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-j-6B",
    # 查看所有 GPT-J 模型: https://huggingface.co/models?filter=gptj
]


def create_sinusoidal_positions(num_pos: int, dim: int) -> tf.Tensor:
    # 创建正弦波位置编码
    inv_freq = tf.cast(1.0 / (10000 ** (tf.range(0, dim, 2) / dim)), tf.float32)
    # 计算正弦和余弦部分
    sinusoid_inp = tf.cast(tf.einsum("i , j -> i j", tf.range(num_pos, dtype=tf.float32), inv_freq), tf.float32)
    sin, cos = tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)
    # 合并正弦和余弦部分
    out = tf.concat((sin, cos), axis=1)
    return out


def rotate_every_two(x: tf.Tensor) -> tf.Tensor:
    # 沿着最后一个维度旋转张量的每两个元素
    rotate_half_tensor = tf.stack((-x[:, :, :, 1::2], x[:, :, :, ::2]), axis=-1)
    new_shape = shape_list(rotate_half_tensor)[:-2] + [tf.math.reduce_prod(shape_list(rotate_half_tensor)[-2:])]
    rotate_half_tensor = tf.reshape(rotate_half_tensor, new_shape)
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor: tf.Tensor, sincos: tf.Tensor) -> tf.Tensor:
    # 应用旋转位置编码到输入张量
    sin_pos, cos_pos = sincos
    sin_pos = tf.repeat(sin_pos[:, :, None, :], 2, 3)
    cos_pos = tf.repeat(cos_pos[:, :, None, :], 2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


# 定义 GPTJAttention 类
class TFGPTJAttention(tf.keras.layers.Layer):
    # 初始化方法，接受配置和其他关键字参数
    def __init__(self, config: GPTJConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 从配置中获取隐藏层大小作为嵌入维度
        self.embed_dim = config.hidden_size
        # 从配置中获取注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_attention_heads
        # 检查嵌入维度是否能被注意力头的数量整除
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            # 如果不能整除，则抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        # 计算缩放因子
        self.scale_attn = self.head_dim**0.5
        # 从配置中获取旋转维度
        self.rotary_dim = config.rotary_dim

        # 初始化注意力和残差的丢弃层
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)

        # 初始化查询投影层
        self.q_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="q_proj",
        )
        # 初始化键投影层
        self.k_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="k_proj",
        )
        # 初始化值投影层
        self.v_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="v_proj",
        )
        # 初始化输出投影层
        self.out_proj = tf.keras.layers.Dense(
            self.embed_dim,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="out_proj",
        )

        # 从配置中获取最大位置嵌入
        self.max_positions = config.max_position_embeddings
        # 生成下三角的掩码
        self.lower_triangle_mask = tf.reshape(
            tf.cast(tf.experimental.numpy.tril(tf.ones((self.max_positions, self.max_positions))), tf.int8),
            (1, 1, self.max_positions, self.max_positions),
        )
        # 计算位置嵌入维度，如果旋转维度存在则使用旋转维度，否则使用嵌入维度
        pos_embd_dim = self.rotary_dim or self.embed_dim
        # 创建正弦位置编码
        self.embed_positions = create_sinusoidal_positions(self.max_positions, pos_embd_dim)

    # 获取因果遮罩的方法
    def get_causal_mask(self, key_length, query_length) -> tf.Tensor:
        # 返回由掩码生成的布尔类型张量
        return tf.cast(self.lower_triangle_mask[:, :, key_length - query_length : key_length, :key_length], tf.bool)

    # 静态方法，获取掩码偏置
    @staticmethod
    def get_masked_bias(dtype: tf.DType) -> tf.Tensor:
        # 返回常数生成的张量，并转换为指定数据类型
        return tf.cast(tf.constant(-1e9), dtype)
    def _split_heads(self, hidden_states: tf.Tensor, rotary: bool) -> tf.Tensor:
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # 计算新的形状，将隐藏状态的维度分割成注意力头的大小和注意力头的数量
        new_shape = shape_list(hidden_states)[:-1] + [self.num_attention_heads, self.head_dim]
        # 重塑隐藏状态张量的形状
        hidden_states = tf.reshape(hidden_states, new_shape)
        # 如果使用旋转注意力机制，则直接返回隐藏状态张量
        if rotary:
            return hidden_states
        # 如果隐藏状态张量的秩为4，则交换轴以匹配 Transformer 模型的输入格式
        if len(shape_list(hidden_states)) == 4:
            return tf.transpose(hidden_states, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)
        # 如果隐藏状态张量的秩为5，则交换轴以匹配 Transformer 模型的输入格式
        if len(shape_list(hidden_states)) == 5:
            return tf.transpose(hidden_states, (0, 1, 3, 2, 4))  # (batch, blocks, head, block_length, head_features)
        # 如果隐藏状态张量的秩既不是4也不是5，则引发 ValueError 异常
        raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}")

    def _merge_heads(self, hidden_states: tf.Tensor) -> tf.Tensor:
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # 如果隐藏状态张量的秩为4，则交换轴以合并注意力头的维度和注意力头数量的维度
        if len(shape_list(hidden_states)) == 4:
            hidden_states = tf.transpose(hidden_states, (0, 2, 1, 3))
        # 如果隐藏状态张量的秩为5，则交换轴以合并注意力头的维度和注意力头数量的维度
        elif len(shape_list(hidden_states)) == 5:
            hidden_states = tf.transpose(hidden_states, (0, 1, 3, 2, 4))
        else:
            # 如果隐藏状态张量的秩既不是4也不是5，则引发 ValueError 异常
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}")
        # 计算新的形状，将隐藏状态的维度合并成一个维度
        new_shape = shape_list(hidden_states)[:-2] + [self.num_attention_heads * self.head_dim]
        # 重塑隐藏状态张量的形状
        return tf.reshape(hidden_states, new_shape)

    def _attn(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # 从因果掩码缓冲区计算因果掩码
        query_length, key_length = shape_list(query)[-2], shape_list(key)[-2]
        causal_mask = self.get_causal_mask(key_length, query_length)

        # 将注意力权重计算保持在 fp32 中，以避免溢出问题
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)

        # 计算注意力权重
        attn_weights = tf.matmul(query, key, transpose_b=True)
        # 应用因果掩码
        attn_weights = tf.where(causal_mask, attn_weights, self.get_masked_bias(attn_weights.dtype))

        # 缩放注意力权重
        attn_weights = attn_weights / self.scale_attn

        # 如果存在注意力掩码，则应用它
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行稳定的 softmax 操作
        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对注意力头进行屏蔽
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        attn_output = tf.matmul(attn_weights, value)

        return attn_output, attn_weights
    # 定义一个方法用于执行 self-attention 操作
    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 将隐藏状态通过投影矩阵进行线性变换，得到查询(query)、键(key)、值(value)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 将查询、键、值分割成多头，进行相关操作
        query = self._split_heads(query, True)
        key = self._split_heads(key, True)
        value = self._split_heads(value, False)

        # 从位置嵌入矩阵中取出与位置相对应的信息
        sincos = tf.cast(tf.gather(self.embed_positions, position_ids, axis=0), hidden_states.dtype)
        sincos = tf.split(sincos, 2, axis=-1)

        # 如果存在旋转维度(rotary_dim)，则进行相应计算
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            key = tf.concat((k_rot, k_pass), axis=-1)
            query = tf.concat((q_rot, q_pass), axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        # 调整键、查询的维度排列
        key = tf.transpose(key, (0, 2, 1, 3))
        query = tf.transpose(query, (0, 2, 1, 3))

        # 处理过去的层(layer_past)
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = tf.concat((past_key, key), axis=-2)
            value = tf.concat((past_value, value), axis=-2)

        # 如果需要使用缓存，则保存当前的键值对
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 计算 self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并多头注意力，通过输出投影层
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # 准备输出，并根据需要添加注意力权重
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # 返回输出结果：attn_output, present, (attentions)
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型为已构建
        self.built = True
        # 如果有查询投影层，则构建查询投影层
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                # 构建查询投影层的结构
                self.q_proj.build([None, None, self.embed_dim])
        # 如果有键投影层，则构建键投影层
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                # 构建键投影层的结构
                self.k_proj.build([None, None, self.embed_dim])
        # 如果有值投影层，则构建值投影层
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                # 构建值投影层的结构
                self.v_proj.build([None, None, self.embed_dim])
        # 如果有输出投影层，则构建输出投影层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                # 构建输出投影层的结构
                self.out_proj.build([None, None, self.embed_dim])
class TFGPTJMLP(tf.keras.layers.Layer):
    # 定义一个名为 TFGPTJMLP 的类，继承自 tf.keras.layers.Layer
    def __init__(self, intermediate_size: int, config: GPTJConfig, **kwargs):
        # 构造函数，初始化函数，接受中间大小和配置参数
        super().__init__(**kwargs)
        embed_dim = config.n_embd

        self.fc_in = tf.keras.layers.Dense(
            intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="fc_in"
        )
        # 创建一个全连接层 fc_in，设置初始权重和名称
        self.fc_out = tf.keras.layers.Dense(
            embed_dim, kernel_initializer=get_initializer(config.initializer_range), name="fc_out"
        )
        # 创建一个全连接层 fc_out，设置初始权重和名称

        self.act = get_tf_activation(config.activation_function)
        # 获取激活函数
        self.dropout = tf.keras.layers.Dropout(config.embd_pdrop)
        # 创建一个 dropout 层，设置 drop 率
        self.embed_dim = config.n_embd
        self.intermediate_size = intermediate_size

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 定义 call 方法，传入隐藏状态并返回输出张量
        hidden_states = self.fc_in(hidden_states)
        # 使用全连接层 fc_in 处理隐藏状态
        hidden_states = self.act(hidden_states)
        # 使用激活函数处理隐藏状态
        hidden_states = self.fc_out(hidden_states)
        # 使用全连接层 fc_out 处理隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 使用 dropout 处理隐藏状态
        return hidden_states

    def build(self, input_shape=None):
        # 构建模型
        if self.built:
            return
        self.built = True
        if getattr(self, "fc_in", None) is not None:
            with tf.name_scope(self.fc_in.name):
                self.fc_in.build([None, None, self.embed_dim])
        # 构建全连接层 fc_in
        if getattr(self, "fc_out", None) is not None:
            with tf.name_scope(self.fc_out.name):
                self.fc_out.build([None, None, self.intermediate_size])
        # 构建全连接层 fc_out


class TFGPTJBlock(tf.keras.layers.Layer):
    # 定义一个名为 TFGPTJBlock 的类，继承自 tf.keras.layers.Layer
    def __init__(self, config: GPTJConfig, **kwargs):
        # 构造函数，初始化函数，接受配置参数
        super().__init__(**kwargs)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        # 内部维度设置为配置内部维度，否则设置为 4 倍的嵌入维度
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        # 创建一个层标准化层 ln_1
        self.attn = TFGPTJAttention(config, name="attn")
        # 创建一个 GPTJ 注意力层 attn
        self.mlp = TFGPTJMLP(inner_dim, config, name="mlp")
        # 创建一个 GPTJ MLP 层 mlp
        self.config = config

    def call(
        self,
        hidden_states: tf.Tensor,
        layer_past: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        position_ids: tf.Tensor | None = None,
        head_mask: tf.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        ):
            # 将当前隐藏状态作为残差保存
            residual = hidden_states
            # 对当前隐藏状态进行 LayerNormalization
            hidden_states = self.ln_1(hidden_states)
            # 使用注意力机制处理当前隐藏状态
            attn_outputs = self.attn(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )  # attn_outputs: attn_output, present, (attentions)
            # 获取注意力机制的输出
            attn_output = attn_outputs[0]
            # 输出除了注意力机制输出之外的其他内容
            outputs = attn_outputs[1:]

            # 对当前隐藏状态进行前馈神经网络处理
            feed_forward_hidden_states = self.mlp(hidden_states)
            # 将注意力机制输出、前馈神经网络输出和残差相加，更新隐藏状态
            hidden_states = attn_output + feed_forward_hidden_states + residual

            # 如果使用缓存，将隐藏状态添加到输出中
            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                # 否则，只将隐藏状态和注意力机制之外的其他内容添加到输出中
                outputs = (hidden_states,) + outputs[1:]
            # 返回更新后的输出，包括隐藏状态、present、(attentions)
            return outputs  # hidden_states, present, (attentions)

        def build(self, input_shape=None):
            # 如果已经构建过，则直接返回
            if self.built:
                return
            # 标记为已构建
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
# 使用 keras_serializable 装饰器将类标记为可以序列化
@keras_serializable
# 创建 TFGPTJMainLayer 类，继承自 tf.keras.layers.Layer
class TFGPTJMainLayer(tf.keras.layers.Layer):
    # 将 config_class 属性设置为 GPTJConfig 类
    config_class = GPTJConfig

    # 初始化函数，接受 config 和其他输入参数
    def __init__(self, config: GPTJConfig, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(*inputs, **kwargs)

        # 将传入的 config 分配给 self.config
        self.config = config
        # 从 config 中获取其他属性并分配给对应的实例变量
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict
        self.num_hidden_layers = config.n_layer
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        # 创建 TFSharedEmbeddings 实例并赋值给 self.wte
        self.wte = TFSharedEmbeddings(
            config.vocab_size, config.hidden_size, initializer_range=config.initializer_range, name="wte"
        )
        # 创建 tf.keras.layers.Dropout 实例并赋值给 self.drop
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        # 使用列表推导式创建 TFGPTJBlock 实例列表并赋值给 self.h
        self.h = [TFGPTJBlock(config, name=f"h_._{i}") for i in range(config.n_layer)]
        # 创建 tf.keras.layers.LayerNormalization 实例并赋值给 self.ln_f
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        # 将 config 中的 n_embd 赋值给 self.embed_dim
        self.embed_dim = config.n_embd

    # 获取输入嵌入，返回 self.wte
    def get_input_embeddings(self):
        return self.wte

    # 设置输入嵌入，将值赋给 self.wte.weight 和 self.wte.vocab_size
    def set_input_embeddings(self, value: tf.Tensor):
        self.wte.weight = value
        self.wte.vocab_size = shape_list(value)[0]

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    # 调用函数，接受多个输入参数并执行相应的操作
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
    # 构建函数，用于构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 检查是否存在 self.wte，并构建它
        if getattr(self, "wte", None) is not None:
            with tf.name_scope(self.wte.name):
                self.wte.build(None)
        # 检查是否存在 self.ln_f，并构建它
        if getattr(self, "ln_f", None) is not None:
            with tf.name_scope(self.ln_f.name):
                self.ln_f.build([None, None, self.embed_dim])
        # 检查是否存在 self.h，并为每个实例构建它们
        if getattr(self, "h", None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 创建 TFGPTJPreTrainedModel 类，继承自 TFPreTrainedModel
class TFGPTJPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 将 config_class 属性设置为 GPTJConfig 类
    config_class = GPTJConfig
    # 将 base_model_prefix 属性设置为 "transformer"
    base_model_prefix = "transformer"
    # 定义 _keys_to_ignore_on_load_unexpected 属性
    # 用于在加载 TF 模型时忽略预期之外的/丢失的层
    _keys_to_ignore_on_load_unexpected = [r"h.\d+.attn.bias"]


# 定义 GPTJ_START_DOCSTRING
GPTJ_START_DOCSTRING = r"""
    # 这个模型继承自 [`TFPreTrainedModel`]。检查超类文档以获取库实现的所有通用方法（如下载或保存、调整输入嵌入、修剪头等）。
    
    # 这个模型也是一个 [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 子类。将其用作常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档以获取有关一般用法和行为的所有相关信息。
    
    # TensorFlow 模型和 `transformers` 中的层接受两种输入格式：
    # - 将所有输入作为关键字参数（如 PyTorch 模型一样），或者
    # - 将所有输入作为列表、元组或字典放在第一个位置参数中。
    
    # 支持第二种格式的原因是，Keras 方法在将输入传递给模型和层时更喜欢这种格式。由于这种支持，当使用 `model.fit()` 等方法时，您应该能很容易地完成工作 - 只要以 `model.fit()` 支持的任何格式传递输入和标签即可！但是，如果希望在 Keras 方法之外使用第二种格式，比如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能的方式可以用于收集第一个位置参数中的所有输入张量：
    
    # - 只有一个 Tensor 内有 `input_ids`，没有其他内容：`model(input_ids)`
    # - 一个长度不固定的列表，包含按照文档字符串给定顺序的一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 一个字典，包含按照文档字符串中给定的输入名称关联的一个或多个输入张量：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    
    # 请注意，当使用 [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，你不需要担心这些，因为你可以像对待任何其他 Python 函数一样传递输入！
    
    # 参数:
    #     config ([`GPTJConfig`]): 具有模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 [`~TFPreTrainedModel.from_pretrained`] 方法以加载模型权重。
# GPTJ 输入的文档字符串
GPTJ_INPUTS_DOCSTRING = r"""
"""


# 创建 GPT-J 模型，输出原始隐藏状态，没有特定的头部
@add_start_docstrings(
    "The bare GPT-J Model transformer outputting raw hidden-states without any specific head on top.",
    GPTJ_START_DOCSTRING,
)
class TFGPTJModel(TFGPTJPreTrainedModel):
    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPTJMainLayer(config, name="transformer")

    # 定义调用函数
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

        # 调用 transformer 层
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

    # 构建函数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)


# GPT-J 模型 transformer，带有语言建模头部
@add_start_docstrings(
    """
    The GPT-J Model transformer with a language modeling head on top.
    """,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForCausalLM(TFGPTJPreTrainedModel, TFCausalLanguageModelingLoss):
    # 初始化方法，接受配置和可变数量的输入参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建一个 GPTJ 主层的实例，使用给定的配置，并命名为"transformer"
        self.transformer = TFGPTJMainLayer(config, name="transformer")
        # 创建一个全连接层，用于生成语言模型的输出，权重初始化采用给定的范围
        self.lm_head = tf.keras.layers.Dense(
            config.vocab_size, kernel_initializer=get_initializer(config.initializer_range), name="lm_head"
        )
        # 将配置保存在实例中
        self.config = config

    # 获取输出嵌入的方法
    def get_output_embeddings(self):
        # 返回语言模型头部
        return self.lm_head

    # 设置输出嵌入的方法
    def set_output_embeddings(self, new_embeddings):
        # 将新的嵌入设置为语言模型头部
        self.lm_head = new_embeddings

    # 为生成准备输入的方法
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 获取 token_type_ids 参数，如果不存在则为 None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果 past_key_values 存在
        if past_key_values:
            # 只保留输入 ids 中的最后一个 token
            inputs = tf.expand_dims(inputs[:, -1], -1)
            # 如果 token_type_ids 存在，则也只保留最后一个 token 的类型 id
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        # 获取 position_ids、attention_mask 参数
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        # 如果 attention_mask 存在且 position_ids 不存在
        if attention_mask is not None and position_ids is None:
            # 根据 attention_mask 计算 position_ids
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            # 如果 past_key_values 存在，则只保留 position_ids 中的最后一个值
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        # 返回输入字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    # 调用模型的方法
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
```  
    ) -> Union[TFCausalLMOutputWithPast, Tuple[tf.Tensor]]:
        r"""
        labels (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        # 使用 transformer 处理输入数据，获取模型输出
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
        # 从 transformer 输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 将隐藏状态传入 lm_head 获取最终的 logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 将 logits 向左移动一位，并且截取最后一个标记之前的内容作为预测结果
            shifted_logits = lm_logits[:, :-1]
            # 将标签向左移动一位，与预测结果对齐
            labels = labels[:, 1:]
            # 计算损失
            loss = self.hf_compute_loss(labels, shifted_logits)

        if not return_dict:
            # 如果不返回字典，则将结果组成元组返回
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则将结果组成 TFCausalLMOutputWithPast 类返回
        return TFCausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 transformer 已构建，则构建 transformer
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 如果 lm_head 已构建，则构建 lm_head
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build([None, None, self.config.n_embd])
@add_start_docstrings(
    """
    The GPT-J Model transformer with a sequence classification head on top (linear layer).

    [`GPTJForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT, GPT-2, GPT-Neo) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPTJ_START_DOCSTRING,
)
class TFGPTJForSequenceClassification(TFGPTJPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = [r"h.\d+.attn.masked_bias", r"h.\d+.attn.bias", r"lm_head.weight"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.transformer = TFGPTJMainLayer(config, name="transformer")  # 初始化 GPT-J 主层
        self.score = tf.keras.layers.Dense(
            self.num_labels,
            use_bias=False,
            kernel_initializer=get_initializer(config.initializer_range),
            name="score",
        )  # 初始化分类得分层
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutputWithPast,
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
    ):  # 定义模型调用方法
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)  # 构建 GPT-J 主层
        if getattr(self, "score", None) is not None:
            with tf.name_scope(self.score.name):
                self.score.build([None, None, self.config.n_embd])  # 构建分类得分层


@add_start_docstrings(
    """
    # GPT-J 模型转换器，顶部带有用于抽取式问答任务（如 SQuAD）的跨度分类头部
    #（在隐藏状态输出的顶部添加线性层以计算“跨度起始 logits”和“跨度结束 logits”）
    """,
    # GPT-J 的起始文档字符串
    GPTJ_START_DOCSTRING,
# 定义了一个名为TFGPTJForQuestionAnswering的类，该类继承自TFGPTJPreTrainedModel和TFQuestionAnsweringLoss
class TFGPTJForQuestionAnswering(TFGPTJPreTrainedModel, TFQuestionAnsweringLoss):
    # 在加载过程中要忽略的键列表
    _keys_to_ignore_on_load_missing = [r"h.\d+.attn.masked_bias", r"h.\d+.attn.bias", r"lm_head.weight"]

    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置num_labels属性为config.num_labels的值
        self.num_labels = config.num_labels
        # 创建一个TFGPTJMainLayer对象，命名为transformer
        self.transformer = TFGPTJMainLayer(config, name="transformer")
        # 创建一个Dense层，输出维度为num_labels，使用config.initializer_range进行初始化，命名为qa_outputs
        self.qa_outputs = tf.keras.layers.Dense(
            self.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 设置config属性为输入的config
        self.config = config

    # call方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
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
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        接收输入参数，并返回包含答案的模型输出或者包含 TensorFlow 张量的元组
        start_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            开始位置的标签，用于计算标记分类损失的标记位置（索引）。位置被限制在序列长度 (`sequence_length`) 内。超出序列的位置不计入损失计算。
        end_positions (`np.ndarray` or `tf.Tensor` of shape `(batch_size,)`, *optional*):
            结束位置的标签，用于计算标记分类损失的标记位置（索引）。位置被限制在序列长度 (`sequence_length`) 内。超出序列的位置不计入损失计算。
        """

        # 使用 Transformer 处理输入，获取 transformer_outputs
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
        # 从 transformer_outputs 中获取序列输出
        sequence_output = transformer_outputs[0]

        # 将序列输出传入 QA 输出层获取 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 沿着最后一个轴（axis=-1）分割为 start_logits 和 end_logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        # 移除 start_logits 和 end_logits 中的维度为 1 的轴
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果 start_positions 和 end_positions 不为 None，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果 return_dict 为 False，则返回元组
        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TFQuestionAnsweringModelOutput 类型
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 构建 Transformer 层
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        # 构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```