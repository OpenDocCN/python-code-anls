# `.\models\gpt2\modeling_tf_gpt2.py`

```py
# 设置编码格式为 utf-8
# 版权声明，版权归属于 The OpenAI Team Authors 和 HuggingFace Inc. team 以及 NVIDIA CORPORATION
# 根据 Apache 2.0 许可证，可以在遵守许可的情况下使用此文件
# 获取许可证副本的链接
# 根据适用法律或书面同意的情况下，根据许可证分发的软件是基于"AS IS"的基础，没有任何形式的担保或条件，无论是明示或隐含的
# 查看特定语言的限制权限和限制
# The OpenAI GPT-2 模型的 TensorFlow 2.0 版本
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFSequenceClassifierOutputWithPast,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFConv1D,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "gpt2"
# 用于文档的配置
_CONFIG_FOR_DOC = "GPT2Config"

# 预训练模型的存档列表
TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # 可在 https://huggingface.co/models?filter=gpt2 查看所有 GPT-2 模型
]


# 定义 TFAttention 类
class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, config, scale=False, is_cross_attention=False, **kwargs):
        # 初始化函数，用于创建一个 Transformer 层的注意力机制
        super().__init__(**kwargs)

        n_state = nx  # 在 Attention 中，n_state=768（nx=n_embd）
        # [在 Block 中到 Attention 中切换 nx => n_state，以保持与 TF 实现的一致性]
        assert n_state % config.n_head == 0
        # 检查 n_state 是否可以被 config.n_head 整除
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions

        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            # 如果是跨注意力，则使用两个 TFConv1D 层
            self.c_attn = TFConv1D(n_state * 2, nx, initializer_range=config.initializer_range, name="c_attn")
            self.q_attn = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="q_attn")
        else:
            # 如果不是跨注意力，则使用一个 TFConv1D 层
            self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name="c_attn")

        # 用于将注意力权重乘以 V，得到最终输出
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_proj")
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.pruned_heads = set()
        self.embed_dim = n_state

    def prune_heads(self, heads):
        pass

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """
        返回一个对角线以下为1的矩阵，其余为0。与 tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd) 相同，但不会在 TPUs 上产生垃圾数据。
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        # q, k, v 的形状为 [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # 缩放注意力分数
            w = w / tf.math.sqrt(dk)

        if not self.is_cross_attention:
            # 如果只有“普通”注意力层实现了因果掩码
            # w 的形状为 [batch, heads, dst_sequence, src_sequence]，信息从 src 流向 dst。
            _, _, nd, ns = shape_list(w)
            b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
            b = tf.reshape(b, [1, 1, nd, ns])
            w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # 应用注意力掩码
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        w = stable_softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # 如果需要，对头进行掩码
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs
    # 将输入张量 x 的维度重新排列，调换第二维和第三维的顺序
    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        # 获取调整后张量的形状信息
        x_shape = shape_list(x)
        # 构建新的形状信息，将原张量倒数第二和倒数第三维合并
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        # 返回按新形状排列的张量
        return tf.reshape(x, new_x_shape)

    # 将输入张量 x 拆分成多个头部
    def split_heads(self, x):
        # 获取输入张量的形状信息
        x_shape = shape_list(x)
        # 构建新的形状信息，将最后一维分成多个头部
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        # 将输入张量按新形状拆分成多个头部，并进行转置操作
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    # 定义模型的调用方法
    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
        training=False,
    ):
        # 如果存在编码器的隐藏状态
        if encoder_hidden_states is not None:
            # 检查是否定义了交叉注意力的权重
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            # 使用查询张量计算注意力
            query = self.q_attn(x)
            # 计算键值对输出
            kv_out = self.c_attn(encoder_hidden_states)
            key, value = tf.split(kv_out, 2, axis=2)
            # 将注意力遮盖设为编码器的注意力遮盖
            attention_mask = encoder_attention_mask
        else:
            # 对输入张量进行注意力计算
            x = self.c_attn(x)
            query, key, value = tf.split(x, 3, axis=2)

        # 将查询、键和值张量拆分成多个头部
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        # 如果存在过去的键值对
        if layer_past is not None:
            # 拆分过去的键和值
            past_key, past_value = tf.unstack(layer_past, axis=0, num=2)
            # 将当前键和值与过去的键和值连接起来
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        # 用于 keras 序列化的处理
        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        # 执行注意力计算
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]

        # 合并多个头部
        a = self.merge_heads(a)
        # 应用投影层
        a = self.c_proj(a)
        # 应用残差连接和 dropout
        a = self.resid_dropout(a, training=training)

        # 返回输出结果
        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建了模型，则直接返回
        if self.built:
            return
        self.built = True
        # 根据是否为交叉注意力，确定 c_attn 的形状
        if self.is_cross_attention:
            c_attn_shape = 2 * self.embed_dim
        else:
            c_attn_shape = 3 * self.embed_dim
        # 如果存在 c_proj 层，则构建它
        if getattr(self, "c_proj", None) is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.embed_dim])
        # 如果存在 c_attn 层，则构建它
        if getattr(self, "c_attn", None) is not None:
            with tf.name_scope(self.c_attn.name):
                self.c_attn.build([None, None, c_attn_shape])
        # 如果存在 q_attn 层，则构建它
        if getattr(self, "q_attn", None) is not None:
            with tf.name_scope(self.q_attn.name):
                self.q_attn.build([None, None, self.embed_dim])
# 定义一个 TFMLP 类，继承自 tf.keras.layers.Layer
class TFMLP(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        # 获取配置中嵌入的维度
        nx = config.n_embd
        # 创建一个 TFConv1D 实例，用于处理输入数据
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name="c_fc")
        # 创建一个 TFConv1D 实例，用于投影数据到指定维度
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name="c_proj")
        # 使用激活函数
        self.act = get_tf_activation(config.activation_function)
        # 添加一个 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        # 内部变量
        self.intermediate_size = n_state
        self.embed_dim = nx

    # 调用方法
    def call(self, x, training=False):
        # 应用激活函数
        h = self.act(self.c_fc(x))
        # 投影处理后的数据
        h2 = self.c_proj(h)
        # 在训练模式下使用 dropout 层
        h2 = self.dropout(h2, training=training)
        # 返回结果
        return h2

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 构建 self.c_fc
        if getattr(self, "c_fc", None) is not None:
            with tf.name_scope(self.c_fc.name):
                self.c_fc.build([None, None, self.intermediate_size])
        # 构建 self.c_proj
        if getattr(self, "c_proj", None) is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.embed_dim])

# 定义一个 TFBlock 类，继承自 tf.keras.layers.Layer
class TFBlock(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        # 获取配置中嵌入的维度
        nx = config.n_embd
        # 如果配置中指定了内部维度，则使用配置中的值，否则使用默认值
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        # 创建一个 LayerNormalization 层
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_1")
        # 创建一个 TFAttention 实例
        self.attn = TFAttention(nx, config, scale, name="attn")
        # 创建一个 LayerNormalization 层
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_2")

        # 如果配置中指定了添加交叉注意力，则创建交叉注意力实例和 LayerNormalization 层
        if config.add_cross_attention:
            self.crossattention = TFAttention(nx, config, scale, name="crossattention", is_cross_attention=True)
            self.ln_cross_attn = tf.keras.layers.LayerNormalization(
                epsilon=config.layer_norm_epsilon, name="ln_cross_attn"
            )

        # 创建一个 TFMLP 实例
        self.mlp = TFMLP(inner_dim, config, name="mlp")
        self.hidden_size = config.hidden_size

    # 调用方法
    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
        training=False,
    # 对输入进行 self-attention 处理
    ):
        # 通过 ln_1 层对输入进行预处理
        a = self.ln_1(x)
        # 使用 self-attention 模块处理输入
        output_attn = self.attn(
            a,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 从 self-attention 输出中获取 a，并更新 x
        a = output_attn[0]  # output_attn: a, present, (attentions)
        outputs = output_attn[1:]
        x = x + a

        # Cross-Attention Block
        # 如果存在 encoder_hidden_states，进行 cross-attention 处理
        if encoder_hidden_states is not None:
            # 添加一个自注意力块用于跨注意力
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # 通过 ln_cross_attn 层处理输入
            ca = self.ln_cross_attn(x)
            # 使用 cross-attention 模块处理输入
            output_cross_attn = self.crossattention(
                ca,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                training=training,
            )
            # 从 cross-attention 输出中获取 ca，并更新 x
            ca = output_cross_attn[0]  # output_attn: a, present, (cross_attentions)
            x = x + ca
            # 如果需要输出 attention 权重，则添加 cross attentions 到 outputs
            outputs = outputs + output_cross_attn[2:]

        # 通过 ln_2 层处理 x
        m = self.ln_2(x)
        # 使用 mlp 处理 m
        m = self.mlp(m, training=training)
        x = x + m

        # 将结果输出到 outputs
        outputs = [x] + outputs
        return outputs  # 返回 x, present, (attentions, cross_attentions)

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型则退出
        if self.built:
            return
        self.built = True
        # 如果存在 ln_1 层，对其进行构建
        if getattr(self, "ln_1", None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.hidden_size])
        # 如果存在 attn 层，对其进行构建
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        # 如果存在 ln_2 层，对其进行构建
        if getattr(self, "ln_2", None) is not None:
            with tf.name_scope(self.ln_2.name):
                self.ln_2.build([None, None, self.hidden_size])
        # 如果存在 mlp 层，对其进行构建
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 如果存在 crossattention 层，对其进行构建
        if getattr(self, "crossattention", None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)
        # 如果存在 ln_cross_attn 层，对其进行构建
        if getattr(self, "ln_cross_attn", None) is not None:
            with tf.name_scope(self.ln_cross_attn.name):
                self.ln_cross_attn.build([None, None, self.hidden_size])
# 标记类为可序列化的 Keras 层，用于 GPT2 主层
@keras_serializable
class TFGPT2MainLayer(tf.keras.layers.Layer):
    # 使用 GPT2 配置类
    config_class = GPT2Config

    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类初始化函数
        super().__init__(*inputs, **kwargs)

        # 保存配置信息
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        # 初始化模型参数
        self.num_hidden_layers = config.n_layer
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        # 初始化词嵌入层
        self.wte = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_size,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="wte",
        )
        # 初始化位置编码层
        self.wpe = tf.keras.layers.Embedding(
            input_dim=config.n_positions,
            output_dim=config.n_embd,
            embeddings_initializer=get_initializer(config.initializer_range),
            name="wpe",
        )
        # 初始化 Dropout 层
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        # 初始化 Transformer 块
        self.h = [TFBlock(config, scale=True, name=f"h_._{i}") for i in range(config.n_layer)]
        # 初始化 LayerNormalization 层
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="ln_f")
        # 初始化嵌入维度
        self.embed_dim = config.hidden_size

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.wte

    # 设置输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 剪枝模型头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    # 调用函数
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    # 构建函数，用于构建模型，根据输入形状来构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 wte 属性，则构建 wte 层
        if getattr(self, "wte", None) is not None:
            # 使用 wte 层的名字为当前操作创建一个命名空间
            with tf.name_scope(self.wte.name):
                # 构建 wte 层
                self.wte.build(None)
        # 如果存在 wpe 属性，则构建 wpe 层
        if getattr(self, "wpe", None) is not None:
            # 使用 wpe 层的名字为当前操作创建一个命名空间
            with tf.name_scope(self.wpe.name):
                # 构建 wpe 层
                self.wpe.build(None)
        # 如果存在 ln_f 属性，则构建 ln_f 层
        if getattr(self, "ln_f", None) is not None:
            # 使用 ln_f 层的名字为当前操作创建一个命名空间
            with tf.name_scope(self.ln_f.name):
                # 构建 ln_f 层，输入形状为 [None, None, self.embed_dim]
                self.ln_f.build([None, None, self.embed_dim])
        # 如果存在 h 属性，则遍历 h 列表
        if getattr(self, "h", None) is not None:
            # 对于 h 列表中的每一层，构建该层
            for layer in self.h:
                # 使用当前层的名字为当前操作创建一个命名空间
                with tf.name_scope(layer.name):
                    # 构建当前层
                    layer.build(None)
# 定义 TFGPT2PreTrainedModel 类，它是 TFPreTrainedModel 的子类
class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 GPT2Config
    config_class = GPT2Config
    # 指定基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 在加载模型时忽略预期之外的未授权层或缺失层
    # 用正则表达式表示，例如 "h.\d+.attn.bias" 表示匹配 h 下数字以及 attn.bias
    _keys_to_ignore_on_load_unexpected = [r"h.\d+.attn.bias", r"h.\d+.crossattention.bias"]

    @property
    def input_signature(self):
        # 定义输入签名，包括 input_ids 和 attention_mask
        # input_ids 是 tf.Tensor 类型，形状为 (None, None)，类型为 tf.int32
        return {
            "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            # attention_mask 是 tf.Tensor 类型，形状为 (None, None)，类型为 tf.int32
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
        }


# 定义 TFGPT2DoubleHeadsModelOutput 类，它是 ModelOutput 的子类
@dataclass
class TFGPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 模型输出类，包括 logits, mc_logits, past_key_values, hidden_states, attentions
    logits: tf.Tensor = None
    # 定义一个变量 mc_logits，用于存储 TensorFlow 的张量，初始值为 None
    mc_logits: tf.Tensor = None
    # 定义一个变量 past_key_values，用于存储 TensorFlow 的张量列表或者 None，初始值为 None
    past_key_values: List[tf.Tensor] | None = None
    # 定义一个变量 hidden_states，用于存储 TensorFlow 的张量元组或者 None，初始值为 None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义一个变量 attentions，用于存储 TensorFlow 的张量元组或者 None，初始值为 None
    attentions: Tuple[tf.Tensor] | None = None
# GPT2 模型的起始文档字符串，详细描述了模型的继承关系和一般用法
GPT2_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Parameters:
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# GPT2 模型的输入文档字符串
GPT2_INPUTS_DOCSTRING = r"""
"""

# 在 GPT2Model 类上添加起始文档字符串和自定义文档字符串
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class TFGPT2Model(TFGPT2PreTrainedModel):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 GPT2 主要层对象
        self.transformer = TFGPT2MainLayer(config, name="transformer")

    # 装饰器，用于将输入解包
    @unpack_inputs
    # 添加前向传播的文档说明
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串，指定文档字符串的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 文档字符串检查点
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 配置类
    )
    # 定义模型的调用方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的token IDs
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的键值对
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # token类型IDs
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置IDs
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头掩码
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入嵌入
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,  # 编码器隐藏状态
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 编码器注意力掩码
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        training: Optional[bool] = False,  # 是否处于训练模式
```  
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past` are used, the user can optionally input only the last `decoder_input_ids` (those that don't have
            their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """

        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 调用transformer模型进行处理，返回输出结果

        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        # 如果已经构建过，则直接返回
        self.built = True
        # 将状态标记为已构建
        if getattr(self, "transformer", None) is not None:
            # 检查是否存在transformer属性
            with tf.name_scope(self.transformer.name):
                # 使用transformer的名称创建一个tf域
                self.transformer.build(None)
                # 调用transformer对象的build��法
# 导入必要的库
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
# 定义 TFGPT2LMHeadModel 类，继承自 TFGPT2PreTrainedModel 和 TFCausalLanguageModelingLoss
class TFGPT2LMHeadModel(TFGPT2PreTrainedModel, TFCausalLanguageModelingLoss):
    # 初始化函数，接受配置和其他参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 创建 GPT2 主体层
        self.transformer = TFGPT2MainLayer(config, name="transformer")

    # 获取输出嵌入
    def get_output_embeddings(self):
        # 返回输入嵌入
        return self.get_input_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, value):
        # 设置输入嵌入
        self.set_input_embeddings(value)

    # 为生成准备输入
    def prepare_inputs_for_generation(self, inputs, past_key_values=None, use_cache=None, **kwargs):
        # 获取 token_type_ids
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果定义了过去的键值，只取最后一个 token 作为输入
        if past_key_values:
            inputs = tf.expand_dims(inputs[:, -1], -1)
            if token_type_ids is not None:
                token_type_ids = tf.expand_dims(token_type_ids[:, -1], -1)

        # 获取位置 ids 和注意力 mask
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)

        # 如果有注意力 mask 但没有位置 ids，则根据注意力 mask 计算位置 ids
        if attention_mask is not None and position_ids is None:
            position_ids = tf.math.cumsum(attention_mask, axis=-1, exclusive=True)
            if past_key_values:
                position_ids = tf.expand_dims(position_ids[:, -1], -1)

        # 返回准备好的输入字典
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "token_type_ids": token_type_ids,
        }

    # 调用函数，处理输入参数和输出结果
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFCausalLMOutputWithCrossAttentions,
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
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 检查模型是否已经构建，若已构建则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 检查是否存在转换器，若存在则使用其名称作为 TensorFlow 命名空间并构建
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                # 调用转换器的构建方法
                self.transformer.build(None)
# 为自定义的 TFGPT2DoubleHeadsModel 类添加文档字符串
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """,
    GPT2_START_DOCSTRING,
)
class TFGPT2DoubleHeadsModel(TFGPT2PreTrainedModel):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置 num_labels 属性为 1
        config.num_labels = 1
        # 创建 transformer 层
        self.transformer = TFGPT2MainLayer(config, name="transformer")
        # 创建 multiple_choice_head 层
        self.multiple_choice_head = TFSequenceSummary(
            config, initializer_range=config.initializer_range, name="multiple_choice_head"
        )

    # 调用前的预处理装饰器，将函数名，文档字符串，返回值限定到 model 输入
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFGPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 call 方法，接受输入参数，返回模型输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        mc_token_ids: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    # 定义 input_signature 属性，指定输入参数的签名
    @property
    def input_signature(self):
        return {
            "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),
            "mc_token_ids": tf.TensorSpec((None, None), tf.int32, name="mc_token_ids"),
        }

    # 构建方法，在第一次使用之前构建模型的层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
        if getattr(self, "multiple_choice_head", None) is not None:
            with tf.name_scope(self.multiple_choice_head.name):
                self.multiple_choice_head.build(None)



# 为自定义的 TFGPT2ForSequenceClassification 类添加文档字符串
@add_start_docstrings(
    """
    The GPT2 Model transformer with a sequence classification head on top (linear layer).

    [`TFGPT2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    # `pad_token_id` 是在配置中定义的，它找到每一行中最后一个不是填充标记的 token。如果没有定义 `pad_token_id`，则只需取每一行中的最后一个值。由于在传递 `inputs_embeds` 而不是 `input_ids` 时无法猜测填充标记，因此会进行相同的操作（取每一行中的最后一个值）。
    """,
    # 使用 GPT2_START_DOCSTRING 进行文档字符串的起始标记
    GPT2_START_DOCSTRING,
# 定义一个用于序列分类任务的 TensorFlow 模型，继承自TFGPT2PreTrainedModel和TFSequenceClassificationLoss类
class TFGPT2ForSequenceClassification(TFGPT2PreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建一个密集层，用于计算分类得分
        self.score = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="score",
            use_bias=False,
        )
        # 创建一个 GPT2 主层，用于进行转换
        self.transformer = TFGPT2MainLayer(config, name="transformer")
        # 保存配置
        self.config = config

    # 定义前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="microsoft/DialogRPT-updown",
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
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        ):
```py  
    # 定义一个方法，返回结果为 TFSequenceClassifierOutputWithPast 或 Tuple[tf.Tensor]
    def forward(
        self,
        input_ids: tf.Tensor,
        past_key_values: Optional[Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        token_type_ids: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        head_mask: Optional[tf.Tensor] = None,
        inputs_embeds: Optional[tf.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        training: bool = False,
        labels: Optional[tf.Tensor] = None
    ) -> Union[TFSequenceClassifierOutputWithPast, Tuple[tf.Tensor]]:
        # 调用transformer模型进行预测
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
        
        # 获取transformer模型的输出结果中的隐藏状态
        hidden_states = transformer_outputs[0]
        # 通过隐藏状态得到逻辑回归的输出
        logits = self.score(hidden_states)
        # 获取logits的shape
        logits_shape = shape_list(logits)
        in_logits = None
        # 如果没有定义pad_token_id，则所有序列长度为-1
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 如果存在input_ids，则计算序列长度
            if input_ids is not None:
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            # 如果不存在input_ids，则警告并将所有序列长度定义为-1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        loss = None

        # 如果存在标签，则计算损失
        if labels is not None:
            assert (
                self.config.pad_token_id is not None or logits_shape[0] == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0 : logits_shape[0], sequence_lengths]

            loss = self.hf_compute_loss(tf.reshape(labels, [-1]), tf.reshape(in_logits, [-1, self.num_labels]))
        pooled_logits = in_logits if in_logits is not None else logits

        # 如果return_dict为False，则返回非字典格式的输出
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回字典格式的输出
        return TFSequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果模型中存在"score"属性
        if getattr(self, "score", None) is not None:
            # 在 TensorFlow 中给该操作指定命名空间
            with tf.name_scope(self.score.name):
                # 构建"score"操作，维度为[None, None, self.config.n_embd]
                self.score.build([None, None, self.config.n_embd])
        # 如果模型中存在"transformer"属性
        if getattr(self, "transformer", None) is not None:
            # 在 TensorFlow 中给该操作指定命名空间
            with tf.name_scope(self.transformer.name):
                # 构建"transformer"操作，输入形状为 None
                self.transformer.build(None)
```