# `.\models\distilbert\modeling_tf_distilbert.py`

```py
# 导入警告模块，用于处理警告信息
import warnings
# 导入类型提示相关模块
from typing import Optional, Tuple, Union
# 导入 NumPy 库
import numpy as np
# 导入 TensorFlow 库
import tensorflow as tf

# 从 Hugging Face 库中导入相关模块和函数
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_distilbert import DistilBertConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 文档中使用的预训练模型名称
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
# 文档中使用的配置文件名称
_CONFIG_FOR_DOC = "DistilBertConfig"

# 可用的预训练模型列表
TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # 更多 DistilBERT 模型列表可见 https://huggingface.co/models?filter=distilbert
]

class TFEmbeddings(keras.layers.Layer):
    """构建由单词、位置和标记类型嵌入组成的嵌入层。"""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 嵌入层配置
        self.config = config
        # 嵌入维度
        self.dim = config.dim
        # 初始化器范围
        self.initializer_range = config.initializer_range
        # 最大位置嵌入数量
        self.max_position_embeddings = config.max_position_embeddings
        # 层归一化层对象
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
        # Dropout 层对象
        self.dropout = keras.layers.Dropout(rate=config.dropout)
    # 在神经网络层的建立过程中，用于构建层的方法，设置输入形状（如果有）
    def build(self, input_shape=None):
        # 在 "word_embeddings" 命名空间下创建一个权重张量，用于词嵌入
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.dim],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 在 "position_embeddings" 命名空间下创建一个权重张量，用于位置嵌入
        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.dim],
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 如果网络层已经建立，直接返回
        if self.built:
            return
        self.built = True
        
        # 如果存在 LayerNorm 层，则构建 LayerNorm 层，设置输入形状
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.dim])

    # 在调用时应用嵌入操作，基于输入张量进行嵌入处理
    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言输入张量不为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果提供了 input_ids，使用权重张量对应位置的嵌入向量
        if input_ids is not None:
            # 检查 input_ids 是否在合法范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 根据 input_ids 从权重张量中获取对应的嵌入向量
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状，排除最后一个维度（批次维度）
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 position_ids，则创建默认的位置 id 张量
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 根据 position_ids 从位置嵌入张量中获取位置嵌入向量
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)

        # 将输入嵌入向量和位置嵌入向量相加，得到最终的嵌入向量
        final_embeddings = inputs_embeds + position_embeds

        # 对最终嵌入向量应用 LayerNorm 层
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        
        # 在训练时，对最终嵌入向量应用 Dropout 层
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        # 返回最终的嵌入向量
        return final_embeddings
# 定义一个名为 TFMultiHeadSelfAttention 的自定义层，继承自 keras 的 Layer 类
class TFMultiHeadSelfAttention(keras.layers.Layer):
    # 初始化方法，接受一个 config 参数和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置多头注意力的头数和每个头的维度
        self.n_heads = config.n_heads
        self.dim = config.dim
        # 创建一个 Dropout 层，用于注意力机制的dropout
        self.dropout = keras.layers.Dropout(config.attention_dropout)
        # 是否输出注意力权重的标志
        self.output_attentions = config.output_attentions

        # 断言确保隐藏层大小能被头数整除
        assert self.dim % self.n_heads == 0, f"Hidden size {self.dim} not dividable by number of heads {self.n_heads}"

        # 创建 Dense 层来处理查询、键和值的线性变换
        self.q_lin = keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="q_lin"
        )
        self.k_lin = keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="k_lin"
        )
        self.v_lin = keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="v_lin"
        )
        # 创建 Dense 层用于最终输出的线性变换
        self.out_lin = keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="out_lin"
        )

        # 初始化一个空集合，用于记录需要剪枝的注意力头
        self.pruned_heads = set()
        # 保存传入的配置对象
        self.config = config

    # 剪枝注意力头的方法，抛出未实现的错误
    def prune_heads(self, heads):
        raise NotImplementedError
    def call(self, query, key, value, mask, head_mask, output_attentions, training=False):
        """
        Parameters:
            query: tf.Tensor(bs, seq_length, dim)
            key: tf.Tensor(bs, seq_length, dim)
            value: tf.Tensor(bs, seq_length, dim)
            mask: tf.Tensor(bs, seq_length)

        Returns:
            weights: tf.Tensor(bs, n_heads, seq_length, seq_length) Attention weights context: tf.Tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = shape_list(query)  # 获取query张量的形状信息
        k_length = shape_list(key)[1]  # 获取key张量的长度信息
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()
        dim_per_head = int(self.dim / self.n_heads)  # 计算每个注意力头的维度
        dim_per_head = tf.cast(dim_per_head, dtype=tf.int32)  # 转换为整数类型
        mask_reshape = [bs, 1, 1, k_length]  # 定义用于重塑mask张量的形状

        def shape(x):
            """将张量按照注意力头分离"""
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            """将张量的注意力头重新组合"""
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))

        q = shape(self.q_lin(query))  # 使用线性层处理query张量并按照注意力头分离 (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # 使用线性层处理key张量并按照注意力头分离 (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # 使用线性层处理value张量并按照注意力头分离 (bs, n_heads, k_length, dim_per_head)
        q = tf.cast(q, dtype=tf.float32)  # 将q张量转换为float32类型
        q = tf.multiply(q, tf.math.rsqrt(tf.cast(dim_per_head, dtype=tf.float32)))  # 对q张量进行归一化处理
        k = tf.cast(k, dtype=q.dtype)  # 将k张量转换为和q相同的数据类型
        scores = tf.matmul(q, k, transpose_b=True)  # 计算注意力分数 (bs, n_heads, q_length, k_length)
        mask = tf.reshape(mask, mask_reshape)  # 重塑mask张量的形状为 (bs, n_heads, qlen, klen)
        # scores.masked_fill_(mask, -float('inf'))            # (bs, n_heads, q_length, k_length)

        mask = tf.cast(mask, dtype=scores.dtype)  # 将mask张量转换为和scores相同的数据类型
        scores = scores - 1e30 * (1.0 - mask)  # 使用mask进行注意力分数的屏蔽处理
        weights = stable_softmax(scores, axis=-1)  # 对注意力分数进行稳定的softmax操作，得到注意力权重 (bs, n_heads, qlen, klen)
        weights = self.dropout(weights, training=training)  # 对注意力权重进行dropout操作，用于训练阶段 (bs, n_heads, qlen, klen)

        # 如果需要，对注意力头进行屏蔽操作
        if head_mask is not None:
            weights = weights * head_mask

        context = tf.matmul(weights, v)  # 计算上下文张量 (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # 将上下文张量按照注意力头重新组合成原始形状 (bs, q_length, dim)
        context = self.out_lin(context)  # 使用线性层处理上下文张量，得到最终输出 (bs, q_length, dim)

        if output_attentions:
            return (context, weights)  # 如果需要输出注意力权重，返回上下文和注意力权重
        else:
            return (context,)  # 否则，仅返回上下文张量
    # 构建函数，用于构建模型的组件
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志表示已经构建过
        self.built = True
        
        # 如果存在查询线性层，则构建查询线性层
        if getattr(self, "q_lin", None) is not None:
            with tf.name_scope(self.q_lin.name):
                # 使用输入维度构建查询线性层
                self.q_lin.build([None, None, self.config.dim])
        
        # 如果存在键线性层，则构建键线性层
        if getattr(self, "k_lin", None) is not None:
            with tf.name_scope(self.k_lin.name):
                # 使用输入维度构建键线性层
                self.k_lin.build([None, None, self.config.dim])
        
        # 如果存在值线性层，则构建值线性层
        if getattr(self, "v_lin", None) is not None:
            with tf.name_scope(self.v_lin.name):
                # 使用输入维度构建值线性层
                self.v_lin.build([None, None, self.config.dim])
        
        # 如果存在输出线性层，则构建输出线性层
        if getattr(self, "out_lin", None) is not None:
            with tf.name_scope(self.out_lin.name):
                # 使用输入维度构建输出线性层
                self.out_lin.build([None, None, self.config.dim])
# 定义一个名为 TFFFN 的自定义 Keras 层
class TFFFN(keras.layers.Layer):
    # 初始化方法，接受一个配置对象和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 设置 dropout 层，使用配置对象中的 dropout 参数
        self.dropout = keras.layers.Dropout(config.dropout)
        # 设置第一个全连接层，使用配置对象中的 hidden_dim 参数和特定的初始化器
        self.lin1 = keras.layers.Dense(
            config.hidden_dim, kernel_initializer=get_initializer(config.initializer_range), name="lin1"
        )
        # 设置第二个全连接层，使用配置对象中的 dim 参数和特定的初始化器
        self.lin2 = keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="lin2"
        )
        # 设置激活函数，根据配置对象中的 activation 参数选择对应的 TensorFlow 激活函数
        self.activation = get_tf_activation(config.activation)
        # 保存配置对象
        self.config = config

    # 定义 call 方法，处理输入数据，并进行层间传递
    def call(self, input, training=False):
        # 第一层全连接操作
        x = self.lin1(input)
        # 应用激活函数
        x = self.activation(x)
        # 第二层全连接操作
        x = self.lin2(x)
        # 应用 dropout 操作，根据 training 参数判断是否进行训练模式
        x = self.dropout(x, training=training)
        # 返回处理后的数据
        return x

    # build 方法，用于构建层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记该层已经构建
        self.built = True
        # 检查并构建第一层全连接层
        if getattr(self, "lin1", None) is not None:
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.config.dim])
        # 检查并构建第二层全连接层
        if getattr(self, "lin2", None) is not None:
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.config.hidden_dim])


# 定义一个名为 TFTransformerBlock 的自定义 Keras 层
class TFTransformerBlock(keras.layers.Layer):
    # 初始化方法，接受一个配置对象和其他关键字参数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 设置注意力头数
        self.n_heads = config.n_heads
        # 设置向量维度
        self.dim = config.dim
        # 设置隐藏层维度
        self.hidden_dim = config.hidden_dim
        # 设置 dropout 层，使用配置对象中的 dropout 参数
        self.dropout = keras.layers.Dropout(config.dropout)
        # 设置激活函数类型
        self.activation = config.activation
        # 是否输出注意力权重
        self.output_attentions = config.output_attentions

        # 确保向量维度可以被注意力头数整除
        assert (
            config.dim % config.n_heads == 0
        ), f"Hidden size {config.dim} not dividable by number of heads {config.n_heads}"

        # 设置自注意力层
        self.attention = TFMultiHeadSelfAttention(config, name="attention")
        # 设置自注意力层后的 LayerNormalization 层
        self.sa_layer_norm = keras.layers.LayerNormalization(epsilon=1e-12, name="sa_layer_norm")

        # 设置前馈神经网络层
        self.ffn = TFFFN(config, name="ffn")
        # 设置前馈神经网络层后的 LayerNormalization 层
        self.output_layer_norm = keras.layers.LayerNormalization(epsilon=1e-12, name="output_layer_norm")
        # 保存配置对象
        self.config = config
    def call(self, x, attn_mask, head_mask, output_attentions, training=False):  # removed: src_enc=None, src_len=None
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim)
                输入张量，形状为(batch_size, 序列长度, 维度)
            attn_mask: tf.Tensor(bs, seq_length)
                注意力掩码张量，形状为(batch_size, 序列长度)，用于屏蔽无效位置的注意力
            head_mask: Not used in this function
                该参数在本函数中未使用
            output_attentions: bool
                是否输出注意力权重张量
            training: bool, optional
                是否处于训练模式，默认为False

        Outputs:
            sa_weights: tf.Tensor(bs, n_heads, seq_length, seq_length)
                注意力权重张量，形状为(batch_size, 注意力头数, 序列长度, 序列长度)
            ffn_output: tf.Tensor(bs, seq_length, dim)
                变换器块的输出张量，形状为(batch_size, 序列长度, 维度)
        """
        # Self-Attention
        sa_output = self.attention(x, x, x, attn_mask, head_mask, output_attentions, training=training)
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            # assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output, training=training)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

    def build(self, input_shape=None):
        """
        构建模型的方法，用于初始化相关层的参数和变量。

        Parameters:
            input_shape: Not used in this function
                该参数在本函数中未使用
        """
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "sa_layer_norm", None) is not None:
            with tf.name_scope(self.sa_layer_norm.name):
                self.sa_layer_norm.build([None, None, self.config.dim])
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)
        if getattr(self, "output_layer_norm", None) is not None:
            with tf.name_scope(self.output_layer_norm.name):
                self.output_layer_norm.build([None, None, self.config.dim])
class TFTransformer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = config.n_layers  # 初始化层数
        self.output_hidden_states = config.output_hidden_states  # 是否输出所有隐藏层状态
        self.output_attentions = config.output_attentions  # 是否输出所有注意力权重

        # 初始化每一层的TransformerBlock
        self.layer = [TFTransformerBlock(config, name=f"layer_._{i}") for i in range(config.n_layers)]

    def call(self, x, attn_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=False):
        # docstyle-ignore
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim) 输入序列的嵌入表示
            attn_mask: tf.Tensor(bs, seq_length) 序列的注意力掩码

        Returns:
            hidden_state: tf.Tensor(bs, seq_length, dim)
                最后（顶层）层的隐藏状态序列
            all_hidden_states: Tuple[tf.Tensor(bs, seq_length, dim)]
                长度为n_layers的元组，包含每一层的隐藏状态序列
                可选：仅在output_hidden_states=True时返回
            all_attentions: Tuple[tf.Tensor(bs, n_heads, seq_length, seq_length)]
                长度为n_layers的元组，包含每一层的注意力权重
                可选：仅在output_attentions=True时返回
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(hidden_state, attn_mask, head_mask[i], output_attentions, training=training)
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1, f"Incorrect number of outputs {len(layer_outputs)} instead of 1"

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # 如果return_dict为False，则返回非None的元组
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        # 如果return_dict为True，则返回TFBaseModelOutput对象
        return TFBaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


@keras_serializable
class TFDistilBertMainLayer(keras.layers.Layer):
    config_class = DistilBertConfig
    # 初始化函数，接受一个配置对象和额外的关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将配置对象保存在实例中
        self.config = config
        # 从配置对象中获取隐藏层的数量
        self.num_hidden_layers = config.num_hidden_layers
        # 是否输出注意力权重
        self.output_attentions = config.output_attentions
        # 是否输出隐藏层状态
        self.output_hidden_states = config.output_hidden_states
        # 是否使用返回字典
        self.return_dict = config.use_return_dict

        # 创建嵌入层对象并保存到实例中，命名为“embeddings”
        self.embeddings = TFEmbeddings(config, name="embeddings")  # Embeddings
        # 创建Transformer编码器对象并保存到实例中，命名为“transformer”
        self.transformer = TFTransformer(config, name="transformer")  # Encoder

    # 返回嵌入层对象的方法
    def get_input_embeddings(self):
        return self.embeddings

    # 设置嵌入层对象的权重和词汇大小的方法
    def set_input_embeddings(self, value):
        # 设置嵌入层对象的权重
        self.embeddings.weight = value
        # 设置嵌入层对象的词汇大小
        self.embeddings.vocab_size = value.shape[0]

    # 抽象方法，用于剪枝头部（未实现）
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    # 装饰器函数，用于解压输入参数
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
        # 如果同时指定了input_ids和inputs_embeds，则抛出错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了input_ids，则获取其形状
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果指定了inputs_embeds，则获取其形状（去除最后一个维度）
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        # 如果没有指定input_ids或inputs_embeds，则抛出错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果未提供attention_mask，则创建全1的注意力掩码
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)  # (bs, seq_length)

        # 将attention_mask转换为float32类型
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        # 准备头部掩码（目前未实现）
        # head_mask的形状为[num_hidden_layers]或[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            # 创建与隐藏层数相同数量的None列表作为头部掩码
            head_mask = [None] * self.num_hidden_layers

        # 获取嵌入层的输出，输入到Transformer编码器中
        embedding_output = self.embeddings(input_ids, inputs_embeds=inputs_embeds)  # (bs, seq_length, dim)
        # 将嵌入层的输出传递给Transformer编码器
        tfmr_output = self.transformer(
            embedding_output,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=training,
        )

        # 返回Transformer编码器的输出，包括最后一层的隐藏状态、所有隐藏状态和所有注意力权重
        return tfmr_output  # last-layer hidden-state, (all hidden_states), (all attentions)
    # 构建方法用于构造模型的层次结构，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型具有嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            # 在命名空间下构建嵌入层，使用嵌入层的名称
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果模型具有变换器层，则构建变换器层
        if getattr(self, "transformer", None) is not None:
            # 在命名空间下构建变换器层，使用变换器层的名称
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
# 接口用于编码器和特定任务模型
class TFDistilBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 DistilBertConfig 作为配置类
    config_class = DistilBertConfig
    # base_model_prefix 指定为 "distilbert"
    base_model_prefix = "distilbert"


# DISTILBERT_START_DOCSTRING 是一个包含多行字符串的文档字符串，描述了模型的继承关系和基本用法
DISTILBERT_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
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
        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# DISTILBERT_INPUTS_DOCSTRING 是一个包含多行字符串的文档字符串，用于描述输入的格式和使用方法
DISTILBERT_INPUTS_DOCSTRING = r"""
    # Args: 输入参数说明开始
    input_ids (`Numpy array` or `tf.Tensor` of shape `({0})`):
        # 输入序列标记在词汇表中的索引
        Indices of input sequence tokens in the vocabulary.

        # 通过 [`AutoTokenizer`] 可以获取输入的索引。参见 [`PreTrainedTokenizer.__call__`] 和 [`PreTrainedTokenizer.encode`] 获取详细信息。
        Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
        [`PreTrainedTokenizer.encode`] for details.

        # [What are input IDs?](../glossary#input-ids) 输入 ID 是什么？
    attention_mask (`Numpy array` or `tf.Tensor` of shape `({0})`, *optional*):
        # 注意力掩码，用于避免对填充的标记索引执行注意力操作。掩码值在 `[0, 1]` 之间：

        # - 1 表示**未掩码**的标记，
        # - 0 表示**已掩码**的标记。
        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

        # [What are attention masks?](../glossary#attention-mask) 注意力掩码是什么？
    head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
        # 自注意力模块中要屏蔽的头部的掩码。掩码值在 `[0, 1]` 之间：

        # - 1 表示**未掩码**的头部，
        # - 0 表示**已掩码**的头部。
        Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

        # inputs_embeds (`tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
        # 可选项，可以直接传递嵌入表示而不是传递 `input_ids`。如果希望更好地控制如何将 `input_ids` 索引转换为关联向量，这很有用。
        # This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
        # model's internal embedding lookup matrix.
    output_attentions (`bool`, *optional*):
        # 是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 获取更多细节。此参数仅在 eager 模式下使用，在图模式下将使用配置中的值。
        Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
        tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
        config will be used instead.
    output_hidden_states (`bool`, *optional*):
        # 是否返回所有层的隐藏状态。查看返回张量中的 `hidden_states` 获取更多细节。此参数仅在 eager 模式下使用，在图模式下将使用配置中的值。
        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
        more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
        used instead.
    return_dict (`bool`, *optional*):
        # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。此参数可以在 eager 模式下使用，在图模式下该值将始终为 True。
        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
        eager mode, in graph mode the value will always be set to True.
    training (`bool`, *optional*, defaults to `False`):
        # 是否使用模型处于训练模式（某些模块如 dropout 在训练和评估之间有不同的行为）。
        Whether or not to use the model in training mode (some modules like dropout modules have different
        behaviors between training and evaluation).
"""
TFDistilBertModel 类定义了一个基于 DistilBERT 模型的编码器/转换器，不添加特定的输出头部。

@parameters：config - DistilBERT 模型的配置
             *inputs - 输入参数
             **kwargs - 额外的关键字参数

@returns：DistilBERT 模型的输出结果

"""

@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertModel(TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")  # Embeddings

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

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)


class TFDistilBertLMHead(keras.layers.Layer):
    """
    TFDistilBertLMHead 类定义了 DistilBERT 的语言模型头部。

    @paramters：config - DistilBERT 的配置
                input_embeddings - 输入的嵌入层

    """

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.dim = config.dim

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        """
        建立语言模型头部的权重。
        
        @paramters：input_shape - 输入形状

        """
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self):
        """
        获取输出嵌入层。

        @returns：输入嵌入层

        """
        return self.input_embeddings

    def set_output_embeddings(self, value):
        """
        设置输出嵌入层。

        @paramters：value - 新的嵌入层权重

        """
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        """
        获取偏置项。

        @returns：偏置项字典

        """
        return {"bias": self.bias}

    def set_bias(self, value):
        """
        设置偏置项。

        @paramters：value - 新的偏置项值

        """
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 定义一个方法 `call`，接受 `hidden_states` 参数
    def call(self, hidden_states):
        # 获取 `hidden_states` 张量的序列长度
        seq_length = shape_list(tensor=hidden_states)[1]
        # 将 `hidden_states` 张量重塑为二维张量，形状为 [-1, self.dim]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.dim])
        # 对重塑后的张量 `hidden_states` 与模型的输入嵌入权重矩阵进行矩阵乘法，转置模型的输入嵌入权重矩阵
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 将矩阵乘法的结果重新塑形为三维张量，形状为 [-1, seq_length, self.config.vocab_size]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 在张量 `hidden_states` 上添加偏置项，偏置项为模型的偏置
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        # 返回处理后的张量 `hidden_states`
        return hidden_states
# 添加模型文档字符串，描述该类为带有 `masked language modeling` 头部的 DistilBERT 模型
@add_start_docstrings(
    """DistilBert Model with a `masked language modeling` head on top.""",
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForMaskedLM(TFDistilBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config

        # 初始化 DistilBERT 主层
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        
        # 创建词汇转换层，用于预测词汇的分布
        self.vocab_transform = keras.layers.Dense(
            config.dim, kernel_initializer=get_initializer(config.initializer_range), name="vocab_transform"
        )
        
        # 获取激活函数并应用于模型
        self.act = get_tf_activation(config.activation)
        
        # 添加词汇层归一化层
        self.vocab_layer_norm = keras.layers.LayerNormalization(epsilon=1e-12, name="vocab_layer_norm")
        
        # 初始化 DistilBERT 语言模型头部
        self.vocab_projector = TFDistilBertLMHead(config, self.distilbert.embeddings, name="vocab_projector")

    def get_lm_head(self):
        # 返回语言模型头部
        return self.vocab_projector

    def get_prefix_bias_name(self):
        # 警告：方法 get_prefix_bias_name 已废弃，请使用 `get_bias` 替代
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        # 返回带有语言模型头部名字的前缀
        return self.name + "/" + self.vocab_projector.name

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
        # 神经网络模型的前向传播函数，用于执行推断或训练步骤
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
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
        # 提取DistilBERT模型的输出隐藏状态
        hidden_states = distilbert_output[0]  # (bs, seq_length, dim)
        # 将隐藏状态映射为预测的logits（对应于词汇表大小）
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        # 应用激活函数到预测的logits
        prediction_logits = self.act(prediction_logits)  # (bs, seq_length, dim)
        # 对预测的logits进行层归一化
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        # 投影到词汇表维度的空间
        prediction_logits = self.vocab_projector(prediction_logits)

        # 如果没有提供标签，则损失为None；否则使用预测的logits计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_logits)

        # 如果不要求返回字典，则返回一组输出
        if not return_dict:
            output = (prediction_logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有命名属性的TFMaskedLMOutput对象，包括损失、logits、隐藏状态和注意力权重
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在DistilBERT模型，则构建它
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)
        # 如果存在词汇转换层，则构建它
        if getattr(self, "vocab_transform", None) is not None:
            with tf.name_scope(self.vocab_transform.name):
                self.vocab_transform.build([None, None, self.config.dim])
        # 如果存在词汇层归一化，则构建它
        if getattr(self, "vocab_layer_norm", None) is not None:
            with tf.name_scope(self.vocab_layer_norm.name):
                self.vocab_layer_norm.build([None, None, self.config.dim])
        # 如果存在词汇投影层，则构建它
        if getattr(self, "vocab_projector", None) is not None:
            with tf.name_scope(self.vocab_projector.name):
                self.vocab_projector.build(None)
@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForSequenceClassification(TFDistilBertPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels  # 初始化分类标签数量

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")  # 初始化 DistilBERT 主层

        # 预分类器，用于准备输入特征
        self.pre_classifier = keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )

        # 分类器，用于分类任务，输出为 num_labels 个类别
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

        # Dropout 层，用于防止过拟合
        self.dropout = keras.layers.Dropout(config.seq_classif_dropout)

        self.config = config  # 保存配置信息

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
        ):
        """
        根据输入调用模型进行前向传播计算。

        Args:
            input_ids (TFModelInputType | None): 输入序列的 token IDs
            attention_mask (np.ndarray | tf.Tensor | None): 注意力遮罩，掩盖无意义的位置
            head_mask (np.ndarray | tf.Tensor | None): 多头注意力掩码
            inputs_embeds (np.ndarray | tf.Tensor | None): 替代输入的嵌入向量
            output_attentions (Optional[bool]): 是否输出注意力权重
            output_hidden_states (Optional[bool]): 是否输出隐藏状态
            return_dict (Optional[bool]): 是否返回字典格式的输出
            labels (np.ndarray | tf.Tensor | None): 标签 IDs
            training (Optional[bool]): 是否处于训练模式

        Returns:
            输出字典或对象，包含预测结果或损失值
        """
        # 省略部分代码
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 调用 DistilBERT 模型处理输入数据，获取模型输出
        distilbert_output = self.distilbert(
            input_ids=input_ids,                # 输入的 token IDs
            attention_mask=attention_mask,      # 注意力掩码，指示哪些 token 是填充的
            head_mask=head_mask,                # 头部掩码，用于控制哪些注意力头部是有效的
            inputs_embeds=inputs_embeds,        # 嵌入的输入张量
            output_attentions=output_attentions,    # 是否输出注意力权重
            output_hidden_states=output_hidden_states,    # 是否输出隐藏状态
            return_dict=return_dict,            # 是否以字典形式返回结果
            training=training,                  # 是否处于训练模式
        )
        hidden_state = distilbert_output[0]    # 获取 DistilBERT 输出的隐藏状态 (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]     # 获取池化的输出 (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # 对池化输出进行预分类
        pooled_output = self.dropout(pooled_output, training=training)  # 对预分类输出进行 dropout 处理
        logits = self.classifier(pooled_output)    # 使用分类器获取 logits (bs, dim)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)  # 计算损失，若无标签则为 None

        if not return_dict:
            output = (logits,) + distilbert_output[1:]   # 如果不返回字典，则输出 logits 和其他 DistilBERT 输出
            return ((loss,) + output) if loss is not None else output  # 返回损失和输出或者仅输出

        # 返回 TFSequenceClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return  # 如果模型已经构建过，则直接返回

        self.built = True  # 标记模型已经构建

        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)  # 构建 DistilBERT 模型

        if getattr(self, "pre_classifier", None) is not None:
            with tf.name_scope(self.pre_classifier.name):
                self.pre_classifier.build([None, None, self.config.dim])  # 构建预分类器模型

        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.dim])  # 构建分类器模型
@add_start_docstrings(
    """
    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForTokenClassification(TFDistilBertPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 设置分类任务的标签数量
        self.num_labels = config.num_labels

        # 初始化 DistilBERT 主层
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        # Dropout 层，用于防止过拟合
        self.dropout = keras.layers.Dropout(config.dropout)
        # 分类器，输出层，用于将隐藏状态输出映射到标签空间
        self.classifier = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
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
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 DistilBERT 主层，获取模型的输出
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
        # 获取序列输出
        sequence_output = outputs[0]
        # 应用 Dropout 层以防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将 Dropout 后的序列输出传入分类器，得到预测的 logits
        logits = self.classifier(sequence_output)
        # 如果有标签，计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不要求返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则返回 TFTokenClassifierOutput 格式的输出
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 如果模型已经构建完成，则直接返回，不进行重复构建
    if self.built:
        return
    
    # 标记模型为已构建状态
    self.built = True
    
    # 如果模型包含名为'distilbert'的属性且不为None，则构建distilbert部分
    if getattr(self, "distilbert", None) is not None:
        # 在TensorFlow中使用名称作用域来管理命名空间，这里创建distilbert的名称作用域
        with tf.name_scope(self.distilbert.name):
            # 调用distilbert的build方法来构建模型
            self.distilbert.build(None)
    
    # 如果模型包含名为'classifier'的属性且不为None，则构建classifier部分
    if getattr(self, "classifier", None) is not None:
        # 在TensorFlow中使用名称作用域来管理命名空间，这里创建classifier的名称作用域
        with tf.name_scope(self.classifier.name):
            # 调用classifier的build方法来构建模型，传入输入形状作为参数
            self.classifier.build([None, None, self.config.hidden_size])
@add_start_docstrings(
    """
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForMultipleChoice(TFDistilBertPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 DistilBERT 主层，作为模型的主体部分
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        
        # Dropout 层，用于随机断开输入神经元，防止过拟合
        self.dropout = keras.layers.Dropout(config.seq_classif_dropout)
        
        # 预分类器层，包含一个 Dense 层用于降维和激活函数为 ReLU
        self.pre_classifier = keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )
        
        # 分类器层，输出为单个值，用于多选题的分类
        self.classifier = keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        
        # 存储配置信息
        self.config = config

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
        **kwargs,
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        """
        Forward pass for TFDistilBertForMultipleChoice.
        
        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length)`, optional):
                Mask to avoid performing attention on padding token indices.
            head_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, optional):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length, hidden_size)`, optional):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            output_attentions (:obj:`bool`, optional):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (:obj:`bool`, optional):
                Whether or not to return the hidden states of all layers.
            return_dict (:obj:`bool`, optional):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            labels (:obj:`np.ndarray` or :obj:`tf.Tensor` of shape :obj:`(batch_size,)`, optional):
                Labels for computing the multiple choice classification loss.
            training (:obj:`bool`, optional):
                Whether to set the model to training mode (dropout active).
        
        Returns:
            :obj:`TFMultipleChoiceModelOutput` or :obj:`Tuple` comprising various elements depending on the configuration
            (config_class, output_attentions, output_hidden_states).
        """
        # 实现模型的前向传播
        # 省略部分代码以保持注释紧凑
        pass
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果提供了 input_ids，则获取 num_choices 和 seq_length
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]  # 获取 input_ids 的第二维大小，即选项数
            seq_length = shape_list(input_ids)[2]   # 获取 input_ids 的第三维大小，即序列长度
        else:
            num_choices = shape_list(inputs_embeds)[1]  # 获取 inputs_embeds 的第二维大小，即选项数
            seq_length = shape_list(inputs_embeds)[2]   # 获取 inputs_embeds 的第三维大小，即序列长度

        # 将 input_ids 展开成二维张量，如果 input_ids 不为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 展开成二维张量，如果 attention_mask 不为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将 inputs_embeds 展开成三维张量，如果 inputs_embeds 不为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 调用 DistilBERT 模型进行前向传播，获取输出
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
        hidden_state = distilbert_output[0]  # 获取 DistilBERT 输出的隐藏状态 (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # 获取隐藏状态的首个位置，作为池化输出 (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # 经过预分类器处理 (bs, dim)
        pooled_output = self.dropout(pooled_output, training=training)  # 应用 dropout (bs, dim)
        logits = self.classifier(pooled_output)  # 经过分类器处理，得到预测 logits

        reshaped_logits = tf.reshape(logits, (-1, num_choices))  # 重新调整 logits 的形状为 (batch_size, num_choices)

        # 计算损失，如果提供了 labels
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不需要返回字典形式的输出，则返回元组形式的输出
        if not return_dict:
            output = (reshaped_logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则返回 TFMultipleChoiceModelOutput 对象
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True

        # 如果模型中包含 DistilBERT 层，则构建 DistilBERT 层
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)

        # 如果模型中包含预分类器层，则构建预分类器层
        if getattr(self, "pre_classifier", None) is not None:
            with tf.name_scope(self.pre_classifier.name):
                self.pre_classifier.build([None, None, self.config.dim])

        # 如果模型中包含分类器层，则构建分类器层
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.dim])
@add_start_docstrings(
    """
    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DISTILBERT_START_DOCSTRING,
)
class TFDistilBertForQuestionAnswering(TFDistilBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 DistilBERT 主层，使用给定的配置和名称
        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        
        # 初始化输出层，一个全连接层用于预测起始和结束位置的 logits
        self.qa_outputs = keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        
        # 断言确保标签数目为2，用于检查是否正确配置了模型
        assert config.num_labels == 2, f"Incorrect number of labels {config.num_labels} instead of 2"
        
        # 初始化 dropout 层，用于在训练时进行随机失活
        self.dropout = keras.layers.Dropout(config.qa_dropout)
        
        # 保存配置对象到实例中
        self.config = config

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播方法，接受多个输入参数并返回输出
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
        ):
        """
        此方法用于模型的前向传播，接受多个输入参数并返回输出结果。
        """
        # 以下是方法体的代码，包括输入参数和具体的处理逻辑。
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 获取 DistilBERT 的输出，包括隐藏状态和注意力权重等
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
        # 对隐藏状态应用 dropout，用于防止过拟合
        hidden_states = self.dropout(hidden_states, training=training)  # (bs, max_query_len, dim)
        # 通过线性层计算起始和结束位置的 logits
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        # 如果给定了起始和结束位置，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不要求返回字典，则根据是否存在损失返回相应的输出
        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 类的对象，包含损失、起始和结束 logits、隐藏状态和注意力权重
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 DistilBERT 存在，则构建其层次结构
        if getattr(self, "distilbert", None) is not None:
            with tf.name_scope(self.distilbert.name):
                self.distilbert.build(None)
        # 如果 QA 输出层存在，则构建其层次结构
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.dim])
```