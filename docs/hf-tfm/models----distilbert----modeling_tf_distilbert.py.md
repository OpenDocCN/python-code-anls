# `.\models\distilbert\modeling_tf_distilbert.py`

```
# 导入必要的库和模块
# 引入警告模块，用于处理警告信息
# 引入类型提示模块，用于类型提示
import warnings
from typing import Optional, Tuple, Union

# 引入 NumPy 库，用于数值计算
import numpy as np
# 引入 TensorFlow 库
import tensorflow as tf

# 从相应的模块中引入各种输出类型的模型输出类
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
# 引入各种损失函数和模型输入类型
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
# 引入工具函数和辅助函数
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
# 从 DistilBERT 配置文件中导入配置类
from .configuration_distilbert import DistilBertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档用的预训练模型和配置文件
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"

# 文档中的预训练模型列表
TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # See all DistilBERT models at https://huggingface.co/models?filter=distilbert
]

# 定义 TFEmbeddings 类，用于构建来自单词、位置和标记类型嵌入的嵌入层
class TFEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        # 初始化嵌入层
        super().__init__(**kwargs)
        # 获取配置信息
        self.config = config
        # 设置嵌入维度
        self.dim = config.dim
        # 设置初始化范围
        self.initializer_range = config.initializer_range
        # 设置最大位置嵌入
        self.max_position_embeddings = config.max_position_embeddings
        # 使用层归一化层，进行归一化
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
        # 设置丢弃率
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout)
    # 构建嵌入层
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建权重张量
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.dim],
                # 初始化权重张量
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 在 "position_embeddings" 命名空间下创建位置嵌入张量
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
                # 初始化位置嵌入张量
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 LayerNorm，则构建 LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNorm
                self.LayerNorm.build([None, None, self.config.dim])

    # 调用嵌入层
    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言输入张量不为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果存在输入张量，则根据输入张量索引获取嵌入向量
        if input_ids is not None:
            # 检查输入张量是否在合理范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 从权重张量中根据输入张量索引获取嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果位置张量为空，则创建默认位置张量
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 根据位置张量索引获取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 最终嵌入向量等于输入嵌入向量加上位置嵌入向量
        final_embeddings = inputs_embeds + position_embeds
        # 对最终嵌入向量进行 LayerNorm
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 在训练时应用 dropout
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
# 定义 TFMultiHeadSelfAttention 类，继承自 tf.keras.layers.Layer 类
class TFMultiHeadSelfAttention(tf.keras.layers.Layer):
    # 初始化方法，接受配置参数 config 和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置多头注意力的头数为配置参数中的 n_heads
        self.n_heads = config.n_heads
        # 设置维度为配置参数中的 dim
        self.dim = config.dim
        # 使用 tf.keras.layers.Dropout 创建一个丢弃层，丢弃率为配置参数中的 attention_dropout
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
        # 设置是否输出注意力权重，值为配置参数中的 output_attentions
        self.output_attentions = config.output_attentions

        # 断言条件，确保隐藏大小 self.dim 能被头数 self.n_heads 整除
        assert self.dim % self.n_heads == 0, f"Hidden size {self.dim} not dividable by number of heads {self.n_heads}"

        # 使用 tf.keras.layers.Dense 创建一个全连接层，输出维度为配置参数中的 dim
        self.q_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="q_lin"
        )
        # 使用 tf.keras.layers.Dense 创建一个全连接层，输出维度为配置参数中的 dim
        self.k_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="k_lin"
        )
        # 使用 tf.keras.layers.Dense 创建一个全连接层，输出维度为配置参数中的 dim
        self.v_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="v_lin"
        )
        # 使用 tf.keras.layers.Dense 创建一个全连接层，输出维度为配置参数中的 dim
        self.out_lin = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="out_lin"
        )

        # 初始化一个空的头集合
        self.pruned_heads = set()
        # 保存配置参数
        self.config = config

    # 剪枝头部的方法，抛出未实现的错误
    def prune_heads(self, heads):
        raise NotImplementedError
    def call(self, query, key, value, mask, head_mask, output_attentions, training=False):
        """
        Parameters:
            query: tf.Tensor(bs, seq_length, dim)  # 输入的查询张量，形状为 (batch size, 序列长度, 维度)
            key: tf.Tensor(bs, seq_length, dim)  # 输入的键张量，形状为 (batch size, 序列长度, 维度)
            value: tf.Tensor(bs, seq_length, dim)  # 输入的值张量，形状为 (batch size, 序列长度, 维度)
            mask: tf.Tensor(bs, seq_length)  # 掩码张量，形状为 (batch size, 序列长度)

        Returns:
            weights: tf.Tensor(bs, n_heads, seq_length, seq_length) Attention weights context: tf.Tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = shape_list(query)  # 获取 query 的形状信息
        k_length = shape_list(key)[1]  # 获取 key 的长度信息
        dim_per_head = int(self.dim / self.n_heads)  # 计算每个 head 的维度
        dim_per_head = tf.cast(dim_per_head, dtype=tf.int32)  # 将维度转换为整型
        mask_reshape = [bs, 1, 1, k_length]  # 定义 mask 的形状

        def shape(x):
            """separate heads"""
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))  # 将张量进行形状变换，拆分多个头

        def unshape(x):
            """group heads"""
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))  # 将多个头合并成一个张量

        q = shape(self.q_lin(query))  # 将查询张量进行线性变换并分离头部
        k = shape(self.k_lin(key))  # 将键张量进行线性变换并分离头部
        v = shape(self.v_lin(value))  # 将值张量进行线性变换并分离头部
        q = tf.cast(q, dtype=tf.float32)  # 将查询张量转换为浮点数类型
        q = tf.multiply(q, tf.math.rsqrt(tf.cast(dim_per_head, dtype=tf.float32)))  # 对查询张量进行归一化
        k = tf.cast(k, dtype=q.dtype)  # 将键张量转换为和查询张量相同的类型
        scores = tf.matmul(q, k, transpose_b=True)  # 计算注意力分数
        mask = tf.reshape(mask, mask_reshape)  # 调整掩码的形状

        mask = tf.cast(mask, dtype=scores.dtype)  # 将掩码转换为和注意力分数相同的类型
        scores = scores - 1e30 * (1.0 - mask)  # 将掩码应用到注意力分数上
        weights = stable_softmax(scores, axis=-1)  # 使用稳定的 softmax 函数计算注意力权重
        weights = self.dropout(weights, training=training)  # 使用 dropout 进行注意力权重的调整

        # 如果存在头部掩码，则将它应用到权重上
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)  # 计算上下文向量
        context = unshape(context)  # 将多头注意力结果合并
        context = self.out_lin(context)  # 输出线性变换后的上下文

        if output_attentions:
            return (context, weights)  # 如果需要输出注意力权重，则返回上下文和权重
        else:
            return (context,)  # 否则只返回上下文
    # 构建自定义层的方法，如果已经构建过，则直接返回
    def build(self, input_shape=None):
        # 检查是否已经构建过，如果是，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果存在 q_lin 层，则构建 q_lin 层
        if getattr(self, "q_lin", None) is not None:
            # 使用 tf 的命名作用域创建 q_lin 层
            with tf.name_scope(self.q_lin.name):
                # 构建 q_lin 层
                self.q_lin.build([None, None, self.config.dim])
        # 如果存在 k_lin 层，则构建 k_lin 层
        if getattr(self, "k_lin", None) is not None:
            # 使用 tf 的命名作用域创建 k_lin 层
            with tf.name_scope(self.k_lin.name):
                # 构建 k_lin 层
                self.k_lin.build([None, None, self.config.dim])
        # 如果存在 v_lin 层，则构建 v_lin 层
        if getattr(self, "v_lin", None) is not None:
            # 使用 tf 的命名作用域创建 v_lin 层
            with tf.name_scope(self.v_lin.name):
                # 构建 v_lin 层
                self.v_lin.build([None, None, self.config.dim])
        # 如果存在 out_lin 层，则构建 out_lin 层
        if getattr(self, "out_lin", None) is not None:
            # 使用 tf 的命名作用域创建 out_lin 层
            with tf.name_scope(self.out_lin.name):
                # 构建 out_lin 层
                self.out_lin.build([None, None, self.config.dim])
class TFFFN(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)
        # 初始化dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 初始化全连接层1
        self.lin1 = tf.keras.layers.Dense(
            config.hidden_dim, kernel_initializer=get_initializer(config.initializer_range), name="lin1"
        )
        # 初始化全连接层2
        self.lin2 = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="lin2"
        )
        # 获取激活函数
        self.activation = get_tf_activation(config.activation)
        # 保存配置
        self.config = config

    def call(self, input, training=False):
        # 全连接层1前向传播
        x = self.lin1(input)
        # 激活函数
        x = self.activation(x)
        # 全连接层2前向传播
        x = self.lin2(x)
        # dropout层
        x = self.dropout(x, training=training)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "lin1", None) is not None:
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.config.dim])
        if getattr(self, "lin2", None) is not None:
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.config.hidden_dim])


class TFTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        # 初始化dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 获取激活函数
        self.activation = config.activation
        self.output_attentions = config.output_attentions

        assert (
            config.dim % config.n_heads == 0
        ), f"Hidden size {config.dim} not dividable by number of heads {config.n_heads}"
        # 初始化多头自注意力层
        self.attention = TFMultiHeadSelfAttention(config, name="attention")
        # 初始化self-attention层的LayerNormalization层
        self.sa_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="sa_layer_norm")

        # 初始化前馈神经网络
        self.ffn = TFFFN(config, name="ffn")
        # 初始化输出层的LayerNormalization层
        self.output_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="output_layer_norm")
        # 保存配置
        self.config = config
    # 定义一个方法用于调用Transformer模块，接收输入x，注意力掩码attn_mask，头部掩码head_mask，是否输出注意力权重output_attentions，是否训练training
    def call(self, x, attn_mask, head_mask, output_attentions, training=False):  # removed: src_enc=None, src_len=None
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim)
            attn_mask: tf.Tensor(bs, seq_length)

        Outputs: sa_weights: tf.Tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
        tf.Tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        # 使用自注意力机制处理输入x，得到sa_output
        sa_output = self.attention(x, x, x, attn_mask, head_mask, output_attentions, training=training)
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # 处理返回元组的情况
            # assert type(sa_output) == tuple
            sa_output = sa_output[0]
        # 对sa_output进行残差连接和Layer Normalization
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        # 使用前馈神经网络处理sa_output，得到ffn_output
        ffn_output = self.ffn(sa_output, training=training)  # (bs, seq_length, dim)
        # 对ffn_output进行残差连接和Layer Normalization
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

    # 构建Transformer模块
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建Self-Attention层的Layer Normalization
        if getattr(self, "sa_layer_norm", None) is not None:
            with tf.name_scope(self.sa_layer_norm.name):
                self.sa_layer_norm.build([None, None, self.config.dim])
        # 构建前馈神经网络层
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)
        # 构建输出层的Layer Normalization
        if getattr(self, "output_layer_norm", None) is not None:
            with tf.name_scope(self.output_layer_norm.name):
                self.output_layer_norm.build([None, None, self.config.dim])
# 定义一个自定义的 Transformer 层，继承自 tf.keras.layers.Layer
class TFTransformer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化 Transformer 层的参数
        self.n_layers = config.n_layers
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        # 创建多个 TransformerBlock 层，存储在 self.layer 列表中
        self.layer = [TFTransformerBlock(config, name=f"layer_._{i}") for i in range(config.n_layers)]

    # 定义 Transformer 层的前向传播函数
    def call(self, x, attn_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=False):
        # docstyle-ignore
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: tf.Tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: tf.Tensor(bs, seq_length, dim)
                Sequence of hidden states in the last (top) layer
            all_hidden_states: Tuple[tf.Tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[tf.Tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        # 初始化存储隐藏状态和注意力权重的变量
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        # 遍历每个 TransformerBlock 层
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            # 调用 TransformerBlock 层的前向传播函数
            layer_outputs = layer_module(hidden_state, attn_mask, head_mask[i], output_attentions, training=training)
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1, f"Incorrect number of outputs {len(layer_outputs)} instead of 1"

        # 添加最后一层的隐藏状态到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # 根据 return_dict 决定返���结果的形式
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 构建 Transformer 层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建每个 TransformerBlock 层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义一个 DistilBert 主层，继承自 tf.keras.layers.Layer
@keras_serializable
class TFDistilBertMainLayer(tf.keras.layers.Layer):
    config_class = DistilBertConfig
    # 初始化函数，接受配置和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 将配置信息保存到对象中
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        # 创建嵌入层对象
        self.embeddings = TFEmbeddings(config, name="embeddings")  # Embeddings
        # 创建变换器对象
        self.transformer = TFTransformer(config, name="transformer")  # Encoder

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    # 剪枝头部
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    # 调用函数，处理输入数据
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 检查输入数据是否合法
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果没有指定注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)  # (bs, seq_length)

        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        # 准备头部掩码（暂未实现）
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers

        # 获取嵌入层输出
        embedding_output = self.embeddings(input_ids, inputs_embeds=inputs_embeds)  # (bs, seq_length, dim)
        # 将嵌入层输出传入变换器
        tfmr_output = self.transformer(
            embedding_output,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=training,
        )

        # 返回变换器输出
        return tfmr_output  # last-layer hidden-state, (all hidden_states), (all attentions)
    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 在命名空间下构建嵌入层
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在变换器层，则构建变换器层
        if getattr(self, "transformer", None) is not None:
            # 在命名空间下构建变换器层
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
# 为编码器和任务特定模型提供接口
class TFDistilBertPreTrainedModel(TFPreTrainedModel):
    """
    一个处理权重初始化和一个简单接口用于下载和加载预训练模型的抽象类。
    """

    # 配置类为 DistilBertConfig
    config_class = DistilBertConfig
    # 基础模型前缀为 "distilbert"


DISTILBERT_START_DOCSTRING = r"""

    此模型继承自 [`TFPreTrainedModel`]。查看超类文档以了解库实现的所有模型的通用方法（如下载或保存、调整输入嵌入、修剪头等）。

    此模型还是 [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 的子类。将其用作常规的 TF 2.0 Keras 模型，并参考 TF 2.0 文档以获取有关一般用法和行为的所有相关信息。

    <Tip>

    `transformers` 中的 TensorFlow 模型和层接受两种格式的输入：

    - 将所有输入作为关键字参数（类似于 PyTorch 模型），或
    - 将所有输入作为列表、元组或字典的第一个位置参数。

    支持第二种格式的原因是，当将输入传递给模型和层时，Keras 方法更喜欢此格式。由于这种支持，当使用 `model.fit()` 等方法时，应该可以正常工作 - 只需以 `model.fit()` 支持的任何格式传递输入和标签即可！但是，如果要在 Keras 方法之外使用第二种格式，例如在使用 Keras `Functional` API 创建自己的层或模型时，有三种可能性可用于在第一个位置参数中收集所有输入张量：

    - 仅具有 `input_ids` 的单个张量，没有其他内容：`model(input_ids)`
    - 长度不同的列表，其中包含一个或多个输入张量，按照文档字符串中给定的顺序：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    - 一个字典，其中包含一个或多个与文档字符串中给定的输入名称相关联的输入张量：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    请注意，当使用 [子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层时，您无需担心任何这些，因为您可以像对待任何其他 Python 函数一样传递输入！

    </Tip>

    参数:
        config ([`DistilBertConfig`]): 包含模型所有参数的模型配置类。
            使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.__call__`] 和 [`PreTrainedTokenizer.encode`] 获取详情。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块中的特定头部失效的掩码。掩码值选在 `[0, 1]` 之间：
            # - 1 表示头部**未被掩码**，
            # - 0 表示头部**被掩码**。
        inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示而不是传递 `input_ids`。如果您想要更多控制如何将 `input_ids` 索引转换为相关向量，这将很有用。
            # 如果您想要更多控制如何将 `input_ids` 索引转换为相关向量，这将很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的 `attentions`。此参数仅在急切模式下可用，在图模式下将使用配置中的值。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的 `hidden_states`。此参数仅在急切模式下可用，在图模式下将使用配置中的值。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。此参数可以在急切模式下使用，在图模式下该值将始终设置为 True。
        training (`bool`, *optional*, defaults to `False`):
            # 是否在训练模式下使用模型（一些模块，如 dropout 模块，在训练和评估之间有不同的行为）。
# 导入必要的库
from transformers.modeling_tf_distilbert import TFDistilBertPreTrainedModel, TFDistilBertMainLayer
from transformers.modeling_tf_utils import TFModelInputType, TFBaseModelOutput
from transformers.utils import add_start_docstrings, unpack_inputs, add_start_docstrings_to_model_forward, add_code_sample_docstrings

# 定义 TFDistilBertModel 类，继承自 TFDistilBertPreTrainedModel
@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertModel(TFDistilBertPreTrainedModel):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 distilbert 层
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")  # Embeddings

    # 定义 call 方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用 distilbert 层
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    # 构建方法
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)


# 定义 TFDistilBertLMHead 类，继承自 tf.keras.layers.Layer
class TFDistilBertLMHead(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.dim = config.dim

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.input_embeddings = input_embeddings

    # 构建方法
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.input_embeddings

    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置
    def get_bias(self):
        return {"bias": self.bias}

    # 设置偏置
    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 定义一个方法，接受隐藏状态作为输入
    def call(self, hidden_states):
        # 获取隐藏状态的序列长度
        seq_length = shape_list(tensor=hidden_states)[1]
        # 将隐藏状态重塑为二维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.dim])
        # 使用输入嵌入权重矩阵与隐藏状态进行矩阵乘法
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将结果重塑为三维张量
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置项到隐藏状态
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回处理后的隐藏状态
        return hidden_states
# 添加起始文档字符串，描述了DistilBert模型及其在顶部的`masked language modeling`头部
# 继承自TFDistilBertPreTrainedModel和TFMaskedLanguageModelingLoss
class TFDistilBertForMaskedLM(TFDistilBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        # 初始化DistilBert主层
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        # 初始化词汇转换层
        self.vocab_transform = tf.keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="vocab_transform"
        )
        self.act = get_tf_activation(config.activation)
        # 初始化词汇层规范化
        self.vocab_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="vocab_layer_norm")
        # 初始化词汇投影层
        self.vocab_projector = TFDistilBertLMHead(config, self.distilbert.embeddings, name="vocab_projector")

    # 获取语言模型头部
    def get_lm_head(self):
        return self.vocab_projector

    # 获取前缀偏置名称
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.vocab_projector.name

    # 调用方法，接收输入并执行前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 获取 DistilBERT 模型的输出
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取隐藏状态
        hidden_states = distilbert_output[0]  # (bs, seq_length, dim)
        # 对隐藏状态进行词汇转换
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        # 对预测 logits 进行激活函数处理
        prediction_logits = self.act(prediction_logits)  # (bs, seq_length, dim)
        # 对预测 logits 进行词汇层归一化
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        # 对预测 logits 进行词汇投影
        prediction_logits = self.vocab_projector(prediction_logits)

        # 如果没有标签，则损失为 None；否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_logits)

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (prediction_logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFMaskedLMOutput 对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建 DistilBERT 模型
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)
        # 构建词汇转换层
        if getattr(self, "vocab_transform", None) is not None:
            with tf.name_scope(self.vocab_transform.name):
                self.vocab_transform.build([None, None, self.config.dim])
        # 构建词汇层归一化层
        if getattr(self, "vocab_layer_norm", None) is not None:
            with tf.name_scope(self.vocab_layer_norm.name):
                self.vocab_layer_norm.build([None, None, self.config.dim])
        # 构建词汇投影层
        if getattr(self, "vocab_projector", None) is not None:
            with tf.name_scope(self.vocab_projector.name):
                self.vocab_projector.build(None)
# 定义一个 DistilBert 模型转换器，顶部带有一个序列分类/回归头部（在汇总输出之上的线性层），例如用于 GLUE 任务
@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForSequenceClassification(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建 DistilBert 主层
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        # 创建预分类器
        self.pre_classifier = tf.keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )
        # 创建分类器
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)
        # 保存配置
        self.config = config

    # 调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        r"""
        定义了一个方法，用于构建模型，根据输入形状构建相关的层和参数。

        Parameters:
            input_shape (tuple of int, *optional*):
                输入的形状。如果为 `None`，则不执行任何操作。

        Returns:
            None
        """
        if self.built:
            # 如果已经构建过模型，则直接返回，不执行后续操作
            return
        # 将 built 标志设置为 True，表示模型已构建
        self.built = True
        # 如果存在 distilbert 层，则构建 distilbert 层
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)
        # 如果存在 pre_classifier 层，则构建 pre_classifier 层
        if getattr(self, "pre_classifier", None) is not None:
            with tf.name_scope(self.pre_classifier.name):
                self.pre_classifier.build([None, None, self.config.dim])
        # 如果存在 classifier 层，则构建 classifier 层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.dim])
# 为了在DistilBERT模型上构建一个标记分类头部，在隐藏状态输出的顶部添加一个线性层，用于命名实体识别（NER）任务
# 继承TFDistilBertPreTrainedModel和TFTokenClassificationLoss类
class TFDistilBertForTokenClassification(TFDistilBertPreTrainedModel, TFTokenClassificationLoss):

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 将配置中的标签数量赋给实例变量
        self.num_labels = config.num_labels

        # 初始化DistilBERT模型的主层
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        # 添加丢弃层
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        # 添加分类器
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 将配置信息保存在实例变量中
        self.config = config

    # 定义call方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        # 标签用于计算标记分类损失
        # labels的形状为(batch_size, sequence_length)，可选
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用DistilBERT模型，获取输出
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出并应用丢弃层
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        # 将序列输出应用于分类器，生成logits
        logits = self.classifier(sequence_output)
        # 如果存在标签，计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典，返回(logits, outputs[1:])或者仅返回outputs[1:]（如果损失为None）
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFTokenClassifierOutput对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义一个方法用于构建模型，输入形状为可选参数
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果模型中存在distilbert属性，则进行以下操作
        if getattr(self, "distilbert", None) is not None:
            # 在TensorFlow中为distilbert添加命名空间，用于组织相关操作
            with tf.name_scope(self.distilbert.name):
                # 构建distilbert模型
                self.distilbert.build(None)
        # 如果模型中存在classifier属性，则进行以下操作
        if getattr(self, "classifier", None) is not None:
            # 在TensorFlow中为classifier添加命名空间，用于组织相关操作
            with tf.name_scope(self.classifier.name):
                # 为classifier构建模型，输入形状为[None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 导入必要的库
@add_start_docstrings(
    """
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
# 定义 TFDistilBertForMultipleChoice 类，继承自 TFDistilBertPreTrainedModel 和 TFMultipleChoiceLoss
class TFDistilBertForMultipleChoice(TFDistilBertPreTrainedModel, TFMultipleChoiceLoss):
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 实例化 TFDistilBertMainLayer，用于处理 DistilBERT 主体部分
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        # 添加 dropout 层，用于防止过拟合
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)
        # 添加全连接层，用于特征提取和非线性变换
        self.pre_classifier = tf.keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )
        # 添加输出分类层，用于多选分类
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 存储配置信息
        self.config = config

    # 定义前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(
        DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 定义函数的输入类型和返回类型
        if input_ids is not None:
            # 如果存在 input_ids，则计算 num_choices 和 seq_length
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            # 如果不存在 input_ids，则计算 num_choices 和 seq_length
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将 input_ids 展平为二维张量 (batch_size * num_choices, seq_length)，若 input_ids 不存在则为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 展平为二维张量 (batch_size * num_choices, seq_length)，若 attention_mask 不存在则为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将 inputs_embeds 展平为三维张量 (batch_size * num_choices, seq_length, embedding_size)，若 inputs_embeds 不存在则为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 调用 DistilBERT 模型进行前向传播
        distilbert_output = self.distilbert(
            flat_input_ids,
            flat_attention_mask,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取 DistilBERT 输出的隐藏状态 (bs, seq_len, dim)
        hidden_state = distilbert_output[0]
        # 提取每个序列的第一个 token 的输出作为池化的输出 (bs, dim)
        pooled_output = hidden_state[:, 0]
        # 使用预分类器对池化的输出进行处理 (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)
        # 对处理后的结果进行 dropout (bs, dim)
        pooled_output = self.dropout(pooled_output, training=training)
        # 使用分类器得到 logits (bs * num_choices, num_choices)
        logits = self.classifier(pooled_output)
        # 重塑 logits 的形状为 (batch_size, num_choices)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果没有提供 labels，则 loss 为 None；否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 根据 return_dict 的值返回结果
        if not return_dict:
            # 若不返回字典，则返回预测 logits 和 DistilBERT 输出的其他结果
            output = (reshaped_logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 若返回字典，则返回包括 loss、logits、DistilBERT 输出的隐藏状态和注意力的字典
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 标记模型为已构建状态
        self.built = True
        # 构建 DistilBERT、预分类器和分类器
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)
        if getattr(self, "pre_classifier", None) is not None:
            with tf.name_scope(self.pre_classifier.name):
                self.pre_classifier.build([None, None, self.config.dim])
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.dim])
# 使用 DistilBert 模型进行抽取式问答任务的模型定义，该模型在隐藏状态输出之上具有一个用于计算 `span start logits` 和 `span end logits` 的线性层
class TFDistilBertForQuestionAnswering(TFDistilBertPreTrainedModel, TFQuestionAnsweringLoss):
    
    # 初始化方法
    def __init__(self, config, *inputs, **kwargs):
        # 调用继承类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 使用 DistilBert 主要层定义 distilbert
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        
        # 定义输出的全连接层 qa_outputs
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        
        # 检查标签数量是否为 2
        assert config.num_labels == 2, f"Incorrect number of labels {config.num_labels} instead of 2"
        
        # 定义 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.qa_dropout)
        
        # 保存配置
        self.config = config

    # call 方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    # 函数定义，接受输入并返回模型输出或包含答案起始和结束位置的元组
    def call(self, input_ids: tf.Tensor, attention_mask: tf.Tensor, head_mask: Optional[tf.Tensor] = None,
             inputs_embeds: Optional[tf.Tensor] = None, output_attentions: Optional[bool] = None,
             output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
             training: Optional[bool] = None, start_positions: Optional[tf.Tensor] = None,
             end_positions: Optional[tf.Tensor] = None) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:

        # 调用 DistilBERT 模型，获取输出
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states, training=training)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            # 创建包含开始和结束位置标签的字典
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 计算损失
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            # 返回模型输出
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 类
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    # 模型构建函数
    def build(self, input_shape=None):
        # 如果已构建，直接返回
        if self.built:
            return
        self.built = True
        # 构建 DistilBERT 模型和 QA 输出层
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.dim])
``` 
```