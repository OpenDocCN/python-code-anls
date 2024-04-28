# `.\models\funnel\modeling_tf_funnel.py`

```
# 引入必要的库和模块
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
# 引入相对路径下的模块和函数
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "FunnelConfig"

# Funnel 模型的预训练模型存档列表
TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

# 定义正无穷常量
INF = 1e6

# 定义 FunnelEmbeddings 类，用于构建来自单词、位置和令牌类型嵌入的嵌入
class TFFunnelEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
```  
    # 初始化函数，接受配置和其他关键字参数，并调用父类初始化方法
    def __init__(self, config, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
    
        # 设置模型配置
        self.config = config
        # 设置隐藏层尺寸
        self.hidden_size = config.hidden_size
        # 设置初始化器标准差，默认为1.0，如果配置中未指定，则使用1.0
        self.initializer_std = 1.0 if config.initializer_std is None else config.initializer_std
    
        # 使用配置中的参数创建 LayerNormalization 层，命名为 "layer_norm"
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 创建 Dropout 层，设置丢弃率为配置中指定的隐藏层丢弃率
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout)
    
    # 构建模型的函数，在此函数中定义模型的参数和层
    def build(self, input_shape=None):
        # 在命名作用域 "word_embeddings" 下执行以下操作
        with tf.name_scope("word_embeddings"):
            # 添加权重张量，命名为 "weight"，形状为 [词汇大小, 隐藏层大小]，使用给定标准差的初始化器
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(initializer_range=self.initializer_std),
            )
    
        # 如果模型已构建，则直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建
        self.built = True
        # 如果存在 LayerNormalization 层，则在指定的命名作用域下构建该层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNormalization 层，输入形状为 [None, None, 配置中的模型维度]
                self.LayerNorm.build([None, None, self.config.d_model])
    
    # 模型的调用函数，用于应用嵌入到输入张量
    def call(self, input_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.
    
        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言确保输入张量不为空
        assert not (input_ids is None and inputs_embeds is None)
        # 断言确保只有一个输入张量不为空
        assert not (input_ids is not None and inputs_embeds is not None)
    
        # 如果提供了 input_ids，则根据权重张量和输入索引进行嵌入
        if input_ids is not None:
            # 检查输入索引是否在合法范围内
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            # 使用权重张量和输入索引进行嵌入
            inputs_embeds = tf.gather(self.weight, input_ids)
    
        # 对嵌入结果应用 LayerNormalization 层
        final_embeddings = self.LayerNorm(inputs=inputs_embeds)
        # 对输出结果应用 Dropout 层，用于防止过拟合
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
    
        # 返回最终嵌入结果张量
        return final_embeddings
class TFFunnelAttentionStructure:
    """
    Contains helpers for `TFFunnelRelMultiheadAttention `.
    """

    cls_token_type_id: int = 2  # 定义 <cls> 的 token 类型 ID 为 2

    def __init__(self, config):
        self.d_model = config.d_model  # 初始化模型的维度
        self.attention_type = config.attention_type  # 初始化注意力机制的类型
        self.num_blocks = config.num_blocks  # 初始化块的数量
        self.separate_cls = config.separate_cls  # 初始化是否分离 <cls> 的参数
        self.truncate_seq = config.truncate_seq  # 初始化是否截断序列的参数
        self.pool_q_only = config.pool_q_only  # 初始化是否只对查询进行池化的参数
        self.pooling_type = config.pooling_type  # 初始化池化的类型

        self.sin_dropout = tf.keras.layers.Dropout(config.hidden_dropout)  # 初始化 sinusoidal position embedding 的 dropout 层
        self.cos_dropout = tf.keras.layers.Dropout(config.hidden_dropout)  # 初始化 cosine position embedding 的 dropout 层
        # 跟踪从原始输入中进行池化的进度，例如序列长度是多少倍降低的
        self.pooling_mult = None  # 初始化池化的倍数

    def init_attention_inputs(self, inputs_embeds, attention_mask=None, token_type_ids=None, training=False):
        """Returns the attention inputs associated to the inputs of the model."""
        # 输入的嵌入形状为 batch_size x seq_len x d_model
        # 注意力掩码和 token 类型 ID 的形状为 batch_size x seq_len
        self.pooling_mult = 1  # 初始化池化的倍数为 1
        self.seq_len = seq_len = shape_list(inputs_embeds)[1]  # 获取输入嵌入的序列长度
        position_embeds = self.get_position_embeds(seq_len, training=training)  # 根据序列长度获取位置嵌入
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None  # 将 token 类型 ID 转换为矩阵
        cls_mask = (
            tf.pad(tf.ones([seq_len - 1, seq_len - 1], dtype=inputs_embeds.dtype), [[1, 0], [1, 0]])
            if self.separate_cls
            else None
        )  # 如果分离 <cls>，则创建用于屏蔽 <cls> 的掩码
        return (position_embeds, token_type_mat, attention_mask, cls_mask)  # 返回注意力输入

    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        token_type_mat = tf.equal(tf.expand_dims(token_type_ids, -1), tf.expand_dims(token_type_ids, -2))  # 将 token 类型 ID 转换为矩阵
        # 将 <cls> 视为与 A 和 B 相同部分的一部分
        cls_ids = tf.equal(token_type_ids, tf.constant([self.cls_token_type_id], dtype=token_type_ids.dtype))  # 获取所有 <cls> 的位置
        cls_mat = tf.logical_or(tf.expand_dims(cls_ids, -1), tf.expand_dims(cls_ids, -2))  # 将 <cls> 转换为矩阵
        return tf.logical_or(cls_mat, token_type_mat)  # 返回 token 类型 ID 的矩阵形式

    def stride_pool_pos(self, pos_id, block_index):
        """
        Pool `pos_id` while keeping the cls token separate (if `self.separate_cls=True`).
        """
        if self.separate_cls:  # 如果分离了 <cls>
            # 在分离 <cls> 的情况下，我们将 <cls> 视为第一个真实块的上一个块的一部分
            # 由于第一个真实块总是位置为 1，上一个块的位置将为 `1 - 2 ** block_index`。
            cls_pos = tf.constant([-(2**block_index) + 1], dtype=pos_id.dtype)  # 计算第一个块的位置
            pooled_pos_id = pos_id[1:-1] if self.truncate_seq else pos_id[1:]  # 截取序列并进行池化
            return tf.concat([cls_pos, pooled_pos_id[::2]], 0)  # 将 <cls> 和经过池化的位置连接起来
        else:
            return pos_id[::2]  # 对位置序列进行池化
    def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
        """
        构建位置向量，表示`pos`和`pooled_pos`之间的相对位置关系。
        """
        # 如果未提供pooled_pos，则默认为pos
        if pooled_pos is None:
            pooled_pos = pos

        # 计算参考点的位置
        ref_point = pooled_pos[0] - pos[0]
        # 计算需要移除的数量
        num_remove = shift * shape_list(pooled_pos)[0]
        # 计算最大距离和最小距离
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        # 返回一个从最大距离到最小距离的等差序列，步长为-stride
        return tf.range(max_dist, min_dist - 1, -stride)

    def stride_pool(self, tensor, axis):
        """
        沿着给定轴对张量进行分层切片。
        """
        if tensor is None:
            return None

        # 如果axis是int型列表或元组，对张量进行递归切片
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # 如果张量是列表或元组类型，则对每个子张量进行递归切片
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # 处理负轴
        axis %= len(shape_list(tensor))

        # 对轴进行分层切片
        axis_slice = slice(None, -1, 2) if self.separate_cls and self.truncate_seq else slice(None, None, 2)
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = tf.concat([tensor[cls_slice], tensor], axis)
        return tensor[enc_slice]

    def pool_tensor(self, tensor, mode="mean", stride=2):
        """对尺寸为[B x T (x H)]的张量应用1D池化。"""
        if tensor is None:
            return None

        # 如果张量是列表或元组类型，则对每个子张量进行递归池化
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        if self.separate_cls:
            suffix = tensor[:, :-1] if self.truncate_seq else tensor
            tensor = tf.concat([tensor[:, :1], suffix], axis=1)

        ndim = len(shape_list(tensor))
        if ndim == 2:
            tensor = tensor[:, :, None]

        # 根据mode参数进行池化
        if mode == "mean":
            tensor = tf.nn.avg_pool1d(tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        elif mode == "max":
            tensor = tf.nn.max_pool1d(tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        elif mode == "min":
            tensor = -tf.nn.max_pool1d(-tensor, stride, strides=stride, data_format="NWC", padding="SAME")
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        # 对张量进行挤压，去除为1的维度
        return tf.squeeze(tensor, 2) if ndim == 2 else tensor
    def pre_attention_pooling(self, output, attention_inputs):
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        # 获取注意力层之前的输出和注意力输入的适当部分
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        # 如果只对查询进行池化
        if self.pool_q_only:
            # 如果使用分解的注意力类型
            if self.attention_type == "factorized":
                # 对位置嵌入进行池化，并保持其它部分不变
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            # 对标记类型矩阵进行池化
            token_type_mat = self.stride_pool(token_type_mat, 1)
            # 对 CLS 掩码进行池化
            cls_mask = self.stride_pool(cls_mask, 0)
            # 对输出进行张量池化
            output = self.pool_tensor(output, mode=self.pooling_type)
        else:
            # 将池化倍数乘以 2
            self.pooling_mult *= 2
            # 如果使用分解的注意力类型
            if self.attention_type == "factorized":
                # 对位置嵌入进行池化
                position_embeds = self.stride_pool(position_embeds, 0)
            # 对标记类型矩阵进行池化
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            # 对 CLS 掩码进行池化
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            # 对注意力掩码进行池化
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            # 对输出进行张量池化
            output = self.pool_tensor(output, mode=self.pooling_type)
        # 更新注意力输入
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs):
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        # 获取注意力层之后的注意力输入的适当部分
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        # 如果只对查询进行池化
        if self.pool_q_only:
            # 将池化倍数乘以 2
            self.pooling_mult *= 2
            # 如果使用分解的注意力类型
            if self.attention_type == "factorized":
                # 对位置嵌入进行池化，并保持其它部分不变
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            # 对标记类型矩阵进行池化
            token_type_mat = self.stride_pool(token_type_mat, 2)
            # 对 CLS 掩码进行池化
            cls_mask = self.stride_pool(cls_mask, 1)
            # 对注意力掩码进行池化
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        # 更新注意力输入
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs
# 定义一个函数用于相对位置的注意力机制中的偏移收集，该函数实现了一种特定的注意力机制。
def _relative_shift_gather(positional_attn, context_len, shift):
    # 获取 positional_attn 的形状信息
    batch_size, n_head, seq_len, max_rel_len = shape_list(positional_attn)
    # max_rel_len = 2 * context_len + shift -1 是可能的相对位置 i-j 的数量

    # 下面的操作与在 PyTorch 中执行以下 gather 相同，这样可能是更清晰但效率较低的代码。
    # idxs = context_len + torch.arange(0, context_len).unsqueeze(0) - torch.arange(0, seq_len).unsqueeze(1)
    # # context_len + i-j 的矩阵
    # return positional_attn.gather(3, idxs.expand([batch_size, n_head, context_len, context_len]))

    # 将 positional_attn 重新塑造成 [batch_size, n_head, max_rel_len, seq_len] 的形状
    positional_attn = tf.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    # 从 shift 开始，丢弃前 shift 列
    positional_attn = positional_attn[:, :, shift:, :]
    # 重新塑造为 [batch_size, n_head, seq_len, max_rel_len - shift] 的形状
    positional_attn = tf.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    # 保留前 context_len 列
    positional_attn = positional_attn[..., :context_len]
    # 返回处理后的 positional_attn
    return positional_attn

# 定义 TFFunnelRelMultiheadAttention 类，继承自 tf.keras.layers.Layer
class TFFunnelRelMultiheadAttention(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        # 从 config 中获取参数
        self.attention_type = config.attention_type
        self.n_head = n_head = config.n_head
        self.d_head = d_head = config.d_head
        self.d_model = d_model = config.d_model
        self.initializer_range = config.initializer_range
        self.block_index = block_index

        # 使用 dropout 进行隐藏层和注意力层的随机失活
        self.hidden_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.attention_dropout = tf.keras.layers.Dropout(config.attention_dropout)

        # 获取初始化器
        initializer = get_initializer(config.initializer_range)

        # 定义全连接层，用于计算 Q、K、V
        self.q_head = tf.keras.layers.Dense(
            n_head * d_head, use_bias=False, kernel_initializer=initializer, name="q_head"
        )
        self.k_head = tf.keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name="k_head")
        self.v_head = tf.keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name="v_head")

        # 定义后置全连接层，用于将多头注意力的输出映射回原始维度
        self.post_proj = tf.keras.layers.Dense(d_model, kernel_initializer=initializer, name="post_proj")
        # LayerNormalization 层，用于归一化
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 缩放因子，用于缩放注意力得分
        self.scale = 1.0 / (d_head**0.5)
    # 构建注意力头数、头部维度和模型总维度的变量
    def build(self, input_shape=None):
        n_head, d_head, d_model = self.n_head, self.d_head, self.d_model
        # 获取初始化器
        initializer = get_initializer(self.initializer_range)

        # 初始化相对位置编码的参数
        self.r_w_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_w_bias"
        )
        # 初始化相对位置编码的参数
        self.r_r_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_r_bias"
        )
        # 初始化相对位置编码的参数
        self.r_kernel = self.add_weight(
            shape=(d_model, n_head, d_head), initializer=initializer, trainable=True, name="r_kernel"
        )
        # 初始化相对位置编码的参数
        self.r_s_bias = self.add_weight(
            shape=(n_head, d_head), initializer=initializer, trainable=True, name="r_s_bias"
        )
        # 初始化段嵌入的参数
        self.seg_embed = self.add_weight(
            shape=(2, n_head, d_head), initializer=initializer, trainable=True, name="seg_embed"
        )

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标志为已构建
        self.built = True
        # 构建 Q、K、V 的注意力头、后处理投影和层归一化
        if getattr(self, "q_head", None) is not None:
            with tf.name_scope(self.q_head.name):
                self.q_head.build([None, None, d_model])
        if getattr(self, "k_head", None) is not None:
            with tf.name_scope(self.k_head.name):
                self.k_head.build([None, None, d_model])
        if getattr(self, "v_head", None) is not None:
            with tf.name_scope(self.v_head.name):
                self.v_head.build([None, None, d_model])
        if getattr(self, "post_proj", None) is not None:
            with tf.name_scope(self.post_proj.name):
                self.post_proj.build([None, None, n_head * d_head])
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, d_model])
```  
    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        # q_head has shape batch_size x sea_len x n_head x d_head
        
        # 如果使用的是分解型相对注意力机制
        if self.attention_type == "factorized":
            # 从论文中获取的符号表示，参考附录 A.2.2 的最终公式（https://arxiv.org/abs/2006.03236）
            # phi 和 pi 的形状为 seq_len x d_model，psi 和 omega 的形状为 context_len x d_model
            phi, pi, psi, omega = position_embeds
            # 形状为 n_head x d_head
            u = self.r_r_bias * self.scale
            # 形状为 d_model x n_head x d_head
            w_r = self.r_kernel

            # 形状为 batch_size x sea_len x n_head x d_model
            q_r_attention = tf.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]

            # 形状为 batch_size x n_head x seq_len x context_len
            positional_attn = tf.einsum("bind,jd->bnij", q_r_attention_1, psi) + tf.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        else:
            # 如果使用的是一般的相对注意力机制
            # 从论文中获取的符号表示，参考附录 A.2.1 的最终公式（https://arxiv.org/abs/2006.03236）
            # 获取正确的位置编码，形状为 max_rel_len x d_model
            if shape_list(q_head)[1] != context_len:
                shift = 2
                r = position_embeds[self.block_index][1]
            else:
                shift = 1
                r = position_embeds[self.block_index][0]
            # 形状为 n_head x d_head
            v = self.r_r_bias * self.scale
            # 形状为 d_model x n_head x d_head
            w_r = self.r_kernel

            # 形状为 max_rel_len x n_head x d_model
            r_head = tf.einsum("td,dnh->tnh", r, w_r)
            # 形状为 batch_size x n_head x seq_len x max_rel_len
            positional_attn = tf.einsum("binh,tnh->bnit", q_head + v, r_head)
            # 形状为 batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

        # 如果有类别掩码，则将位置注意力乘以类别掩码
        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn
    # 计算相对于token_type_ids的相对注意力分数
    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        # 如果token_type_mat为空，则返回0
        if token_type_mat is None:
            return 0
        # 获取batch_size, seq_len, context_len的形状
        batch_size, seq_len, context_len = shape_list(token_type_mat)
        # q_head的形状为 batch_size x seq_len x n_head x d_head
        # 形状为 n_head x d_head
        r_s_bias = self.r_s_bias * self.scale

        # 形状为 batch_size x n_head x seq_len x 2
        token_type_bias = tf.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        # 形状为 batch_size x n_head x seq_len x context_len
        token_type_mat = tf.tile(token_type_mat[:, None], [1, shape_list(q_head)[2], 1, 1])
        # 分裂成形状为 batch_size x n_head x seq_len
        diff_token_type, same_token_type = tf.split(token_type_bias, 2, axis=-1)
        # 形状为 batch_size x n_head x seq_len x context_len
        token_type_attn = tf.where(
            token_type_mat,
            tf.tile(same_token_type, [1, 1, 1, context_len]),
            tf.tile(diff_token_type, [1, 1, 1, context_len]),
        )

        # 如果存在cls_mask，则将token_type_attn与cls_mask相乘
        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn
    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        # 调用函数，传入查询query、关键字key和数值value，注意力输入attention_inputs，是否输出注意力output_attentions和是否训练training参数
        # query的形状为 batch_size x seq_len x d_model
        # key和value的形状为 batch_size x context_len x d_model
        # attention_inputs包括 position_embeds, token_type_mat, attention_mask, cls_mask

        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        # 从 attention_inputs 中解包得到 position_embeds, token_type_mat, attention_mask, cls_mask

        batch_size, seq_len, _ = shape_list(query)
        # 获取 query 的 batch_size 和 seq_len
        context_len = shape_list(key)[1]
        # 获取 key 的 context_len
        n_head, d_head = self.n_head, self.d_head
        # 获取 n_head 和 d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = tf.reshape(self.q_head(query), [batch_size, seq_len, n_head, d_head])
        # 将 query 经过 self.q_head 处理后的结果 reshape 成 batch_size x seq_len x n_head x d_head
        # Shapes batch_size x context_len x n_head x d_head
        k_head = tf.reshape(self.k_head(key), [batch_size, context_len, n_head, d_head])
        # 将 key 经过 self.k_head 处理后的结果 reshape 成 batch_size x context_len x n_head x d_head
        v_head = tf.reshape(self.v_head(value), [batch_size, context_len, n_head, d_head])
        # 将 value 经过 self.v_head 处理后的结果 reshape 成 batch_size x context_len x n_head x d_head

        q_head = q_head * self.scale
        # 将 q_head 乘以 self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # 将 self.r_w_bias 乘以 self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        content_score = tf.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        # 通过数学运算获取 content_score
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        # 获取 positional_attn
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)
        # 获取 token_type_attn

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn
        # 合并注意力分数

        # perform masking
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=attn_score.dtype)
            # 将 attention_mask 转换为和 attn_score 相同的数据类型
            attn_score = attn_score - (INF * (1 - attention_mask[:, None, None]))
            # 如果存在 attention_mask，则对 attn_score 进行掩盖操作

        # attention probability
        attn_prob = stable_softmax(attn_score, axis=-1)
        # attention_prob 是 attn_score 经过 softmax 处理后得到的概率值
        attn_prob = self.attention_dropout(attn_prob, training=training)
        # 对 attn_prob 进行 attention_dropout 操作，根据训练参数判断是否进行训练

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = tf.einsum("bnij,bjnd->bind", attn_prob, v_head)
        # 获取 attention output

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(tf.reshape(attn_vec, [batch_size, seq_len, n_head * d_head]))
        # 使用 post_proj 对 attn_vec 进行处理并 reshape 成 batch_size x seq_len x d_model
        attn_out = self.hidden_dropout(attn_out, training=training)
        # 对 attn_out 进行 hidden_dropout 操作，根据训练参数判断是否进行训练

        output = self.layer_norm(query + attn_out)
        # 对 query 和 attn_out 进行相加后使用 layer_norm 处理
        return (output, attn_prob) if output_attentions else (output,)
        # 如果 output_attentions 为 True，则返回 (output, attn_prob)，否则只返回 (output,)
# 定义一个基于 TensorFlow 的自定义层 TFFunnelPositionwiseFFN，用于 Transformer 模型中的前馈神经网络部分
class TFFunnelPositionwiseFFN(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 根据配置参数初始化权重初始化器
        initializer = get_initializer(config.initializer_range)
        # 创建第一个全连接层，用于 FFN 的第一层
        self.linear_1 = tf.keras.layers.Dense(config.d_inner, kernel_initializer=initializer, name="linear_1")
        # 获取激活函数
        self.activation_function = get_tf_activation(config.hidden_act)
        # 创建激活函数的 dropout 层
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        # 创建第二个全连接层，用于 FFN 的第二层
        self.linear_2 = tf.keras.layers.Dense(config.d_model, kernel_initializer=initializer, name="linear_2")
        # 创建 dropout 层，用于第二个全连接层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 创建层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 保存配置参数
        self.config = config

    # 定义前向传播方法
    def call(self, hidden, training=False):
        # 第一层全连接
        h = self.linear_1(hidden)
        # 应用激活函数
        h = self.activation_function(h)
        # 应用激活函数的 dropout
        h = self.activation_dropout(h, training=training)
        # 第二层全连接
        h = self.linear_2(h)
        # 应用 dropout
        h = self.dropout(h, training=training)
        # 使用残差连接和层归一化
        return self.layer_norm(hidden + h)

    # 构建层的方法，用于构建子层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建第一个全连接层
        if getattr(self, "linear_1", None) is not None:
            with tf.name_scope(self.linear_1.name):
                self.linear_1.build([None, None, self.config.d_model])
        # 构建第二个全连接层
        if getattr(self, "linear_2", None) is not None:
            with tf.name_scope(self.linear_2.name):
                self.linear_2.build([None, None, self.config.d_inner])
        # 构建层归一化层
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])


# 定义一个基于 TensorFlow 的自定义层 TFFunnelLayer，用于 Transformer 模型中的单层
class TFFunnelLayer(tf.keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        # 创建注意力层
        self.attention = TFFunnelRelMultiheadAttention(config, block_index, name="attention")
        # 创建前馈网络层
        self.ffn = TFFunnelPositionwiseFFN(config, name="ffn")

    # 定义前向传播方法
    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        # 调用注意力层的前向传播
        attn = self.attention(
            query, key, value, attention_inputs, output_attentions=output_attentions, training=training
        )
        # 调用前馈网络层的前向传播
        output = self.ffn(attn[0], training=training)
        # 返回输出和注意力结果
        return (output, attn[1]) if output_attentions else (output,)

    # 构建层的方法，用于构建子层的权重
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建注意力层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建前馈网络层
        if getattr(self, "ffn", None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)


# 定义一个基于 TensorFlow 的自定义层 TFFunnelEncoder，用于 Transformer 模型中的编码器层
class TFFunnelEncoder(tf.keras.layers.Layer):
    # 初始化函数，用于创建一个新的 TFFunnel 模型实例
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 从配置中获取是否分离类别（CLS）信息的标志
        self.separate_cls = config.separate_cls
        # 从配置中获取是否仅使用池化输出的标志
        self.pool_q_only = config.pool_q_only
        # 从配置中获取每个块的重复次数
        self.block_repeats = config.block_repeats
        # 创建 TFFunnelAttentionStructure 对象，用于处理注意力结构
        self.attention_structure = TFFunnelAttentionStructure(config)
        # 创建多个 TFFunnelLayer 对象，组成模型的各个块
        self.blocks = [
            # 根据配置中的块大小和索引创建对应数量的 TFFunnelLayer 对象
            [TFFunnelLayer(config, block_index, name=f"blocks_._{block_index}_._{i}") for i in range(block_size)]
            # 遍历配置中的块大小列表，同时记录块的索引
            for block_index, block_size in enumerate(config.block_sizes)
        ]

    # 模型的调用方法，用于前向传播计算
    def call(
        self,
        inputs_embeds,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=False,
        ):
            # 未实现在长张量上进行池化，因此需要转换这个掩码
            # attention_mask = tf.cast(attention_mask, inputs_embeds.dtype)
            # 初始化注意力输入
            attention_inputs = self.attention_structure.init_attention_inputs(
                inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                training=training,
            )
            # 将初始隐状态设置为输入嵌入
            hidden = inputs_embeds

            # 初始化保存所有隐藏状态的列表，如果要输出隐藏状态
            all_hidden_states = (inputs_embeds,) if output_hidden_states else None
            # 初始化保存所有注意力矩阵的列表，如果要输出注意力
            all_attentions = () if output_attentions else None

            # 遍历所有的 Transformer block
            for block_index, block in enumerate(self.blocks):
                # 判断当前隐藏状态长度是否大于 2（不包括[CLS]）且当前不是第一个 block
                pooling_flag = shape_list(hidden)[1] > (2 if self.separate_cls else 1)
                pooling_flag = pooling_flag and block_index > 0
                # 初始化存储 pooled_hidden 的变量
                pooled_hidden = tf.zeros(shape_list(hidden))

                # 如果需要池化
                if pooling_flag:
                    # 执行前注意力池化
                    pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                        hidden, attention_inputs
                    )

                # 遍历当前 block 中的所有 Transformer 层
                for layer_index, layer in enumerate(block):
                    # 根据 block 中每个层的重复次数进行处理
                    for repeat_index in range(self.block_repeats[block_index]):
                        # 判断是否需要进行池化
                        do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                        if do_pooling:
                            query = pooled_hidden
                            key = value = hidden if self.pool_q_only else pooled_hidden
                        else:
                            query = key = value = hidden
                        # 计算每个层的输出，并更新 hidden
                        layer_output = layer(
                            query, key, value, attention_inputs, output_attentions=output_attentions, training=training
                        )
                        hidden = layer_output[0]
                        # 如果需要池化，执行后注意力池化
                        if do_pooling:
                            attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                        # 如果需要输出注意力矩阵
                        if output_attentions:
                            all_attentions = all_attentions + layer_output[1:]
                        # 如果需要输出隐藏状态
                        if output_hidden_states:
                            all_hidden_states = all_hidden_states + (hidden,)

            # 如果不需要以字典形式返回结果
            if not return_dict:
                return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
            # 以 TFBaseModelOutput 对象的形式返回结果
            return TFBaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)

        # 构建模型层
        def build(self, input_shape=None):
            # 如果模型已经构建则直接返回
            if self.built:
                return
            self.built = True
            # 遍历所有的 Transformer block，构建每个层
            for block in self.blocks:
                for layer in block:
                    with tf.name_scope(layer.name):
                        layer.build(None)
def upsample(x, stride, target_len, separate_cls=True, truncate_seq=False):
    """
    将张量 `x` 上采样以匹配 `target_len`，通过在序列长度维度上重复 `stride` 次令牌。
    """
    # 如果步幅为1，则不进行上采样，直接返回输入张量
    if stride == 1:
        return x
    # 如果需要分离CLS标记，则提取CLS标记，并将其从输入张量中去除
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    # 在序列长度维度上重复令牌
    output = tf.repeat(x, repeats=stride, axis=1)
    # 如果需要分离CLS标记
    if separate_cls:
        # 如果需要截断序列
        if truncate_seq:
            # 在输出张量末尾填充0以实现截断
            output = tf.pad(output, [[0, 0], [0, stride - 1], [0, 0]])
        # 截断序列到目标长度减1，然后将CLS标记重新连接到输出张量前面
        output = output[:, : target_len - 1]
        output = tf.concat([cls, output], axis=1)
    else:
        # 如果不需要分离CLS标记，则直接截断序列到目标长度
        output = output[:, :target_len]
    return output


class TFFunnelDecoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化时设置是否分离CLS标记和是否截断序列的参数
        self.separate_cls = config.separate_cls
        self.truncate_seq = config.truncate_seq
        # 计算上采样步幅，根据配置的块大小计算得到
        self.stride = 2 ** (len(config.block_sizes) - 1)
        # 初始化注意力结构和解码器层
        self.attention_structure = TFFunnelAttentionStructure(config)
        self.layers = [TFFunnelLayer(config, 0, name=f"layers_._{i}") for i in range(config.num_decoder_layers)]

    def call(
        self,
        final_hidden,
        first_block_hidden,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        training=False,
    ):
        # 对最终隐藏状态进行上采样，并与第一个块的隐藏状态相加
        upsampled_hidden = upsample(
            final_hidden,
            stride=self.stride,
            target_len=shape_list(first_block_hidden)[1],
            separate_cls=self.separate_cls,
            truncate_seq=self.truncate_seq,
        )
        hidden = upsampled_hidden + first_block_hidden
        # 初始化存储所有隐藏状态和注意力的变量
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # 初始化注意力输入
        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
        )
        # 遍历所有解码器层
        for layer in self.layers:
            # 调用解码器层的call方法进行解码
            layer_output = layer(
                hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions, training=training
            )
            hidden = layer_output[0]
            # 如果需要输出注意力权重，则记录每一层的注意力权重
            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            # 如果需要输出隐藏状态，则记录每一层的隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)
        # 如果不返回字典，则返回一个元组
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        # 返回TFBaseModelOutput字典对象
        return TFBaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置为已构建状态
        self.built = True
        # 遍历所有解码器层，构建每一层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
# 引入 keras_serializable 装饰器，用于指示类是可序列化的 Keras 模型
@keras_serializable
# 定义 TFFunnelBaseLayer 类，继承自 tf.keras.layers.Layer 类
class TFFunnelBaseLayer(tf.keras.layers.Layer):
    """Base model without decoder"""  # 类的简要描述信息

    # 模型的配置类
    config_class = FunnelConfig

    # 类的初始化方法
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)  # 调用父类的初始化方法

        # 保存传入的配置
        self.config = config
        # 设置是否输出注意力权重
        self.output_attentions = config.output_attentions
        # 设置是否输出隐藏状态
        self.output_hidden_states = config.output_hidden_states
        # 设置是否返回字典格式的输出
        self.return_dict = config.use_return_dict

        # 创建嵌入层对象
        self.embeddings = TFFunnelEmbeddings(config, name="embeddings")
        # 创建编码器对象
        self.encoder = TFFunnelEncoder(config, name="encoder")

    # 获取输入嵌入层对象
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入嵌入层对象
    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 头部剪枝方法，用于剪枝注意力头
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # 在 TF 2.0 模型库中尚未实现

    # 调用方法，用于模型前向传播
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 检查输入参数的合法性
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 若未提供注意力掩码，则创建一个全 1 的掩码
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # 若未提供标记类型 ID，则创建一个全 0 的标记类型 ID
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # 若未提供输入嵌入向量，则通过嵌入层获取
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, training=training)

        # 调用编码器进行编码
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回编码器输出
        return encoder_outputs

    # 构建方法，用于构建层
    def build(self, input_shape=None):
        if self.built:  # 若已经构建，则直接返回
            return
        self.built = True  # 标记已构建
        if getattr(self, "embeddings", None) is not None:  # 若嵌入层存在
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)  # 构建嵌入层
        if getattr(self, "encoder", None) is not None:  # 若编码器存在
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)  # 构建编码器


# 引入 keras_serializable 装饰器，用于指示类是可序列化的 Keras 模型
@keras_serializable
# 定义 TFFunnelMainLayer 类，继承自 tf.keras.layers.Layer 类
class TFFunnelMainLayer(tf.keras.layers.Layer):
    """Base model with decoder"""  # 类的简要描述信息

    # 模型的配置类
    config_class = FunnelConfig
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.block_sizes = config.block_sizes   # 获取配置信息中的 block_sizes
        self.output_attentions = config.output_attentions   # 获取配置信息中的 output_attentions
        self.output_hidden_states = config.output_hidden_states   # 获取配置信息中的 output_hidden_states
        self.return_dict = config.use_return_dict   # 获取配置信息中的 use_return_dict

        self.embeddings = TFFunnelEmbeddings(config, name="embeddings")   # 创建 TFFunnelEmbeddings 对象并赋值给 self.embeddings
        self.encoder = TFFunnelEncoder(config, name="encoder")   # 创建 TFFunnelEncoder 对象并赋值给 self.encoder
        self.decoder = TFFunnelDecoder(config, name="decoder")   # 创建 TFFunnelDecoder 对象并赋值给 self.decoder

    def get_input_embeddings(self):
        return self.embeddings    # 返回 self.embeddings 对象

    def set_input_embeddings(self, value):
        self.embeddings.weight = value   # 设置 self.embeddings 对象的 weight 属性为 value
        self.embeddings.vocab_size = shape_list(value)[0]   # 设置 self.embeddings 对象的 vocab_size 属性为 value 的第一个维度大小

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # 抛出 NotImplementedError 异常，表明该方法还未在 TF 2.0 模型库中实现

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,


注释：



    def __init__(self, config, **kwargs):
        # 调用父类的 __init__ 方法
        super().__init__(**kwargs)

        self.config = config  # 保存配置信息
        self.block_sizes = config.block_sizes  # 获取配置信息中的 block_sizes
        self.output_attentions = config.output_attentions  # 获取配置信息中的 output_attentions
        self.output_hidden_states = config.output_hidden_states  # 获取配置信息中的 output_hidden_states
        self.return_dict = config.use_return_dict  # 获取配置信息中的 use_return_dict

        self.embeddings = TFFunnelEmbeddings(config, name="embeddings")  # 创建 TFFunnelEmbeddings 对象并赋值给 self.embeddings
        self.encoder = TFFunnelEncoder(config, name="encoder")  # 创建 TFFunnelEncoder 对象并赋值给 self.encoder
        self.decoder = TFFunnelDecoder(config, name="decoder")  # 创建 TFFunnelDecoder 对象并赋值给 self.decoder

    def get_input_embeddings(self):
        return self.embeddings  # 返回 self.embeddings 对象

    def set_input_embeddings(self, value):
        self.embeddings.weight = value  # 设置 self.embeddings 对象的 weight 属性为 value
        self.embeddings.vocab_size = shape_list(value)[0]  # 设置 self.embeddings 对象的 vocab_size 属性为 value 的第一个维度大小

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # 抛出 NotImplementedError 异常，表明该方法还未在 TF 2.0 模型库中实现

    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,


注释：
        ):
        # 如果同时指定了input_ids和inputs_embeds，则抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了input_ids，则获取其形状
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果只指定了inputs_embeds，则获取其形状并去掉最后一个维度
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        # 如果既没有指定input_ids也没有指定inputs_embeds，则抛出数值错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果未指定attention_mask，则填充为1
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)

        # 如果未指定token_type_ids，则填充为0
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # 如果未指定inputs_embeds，则使用self.embeddings方法对input_ids进行embedding处理
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, training=training)

        # 使用self.encoder方法对inputs_embeds进行编码处理
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            training=training,
        )

        # 使用self.decoder方法对编码结果进行解码处理
        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],
            first_block_hidden=encoder_outputs[1][self.block_sizes[0]],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 如果不返回字典形式的结果
        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            # 如果需要输出hidden_states，则将encoder和decoder的hidden_states拼接输出
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            # 如果需要输出attentions，则将encoder和decoder的attentions拼接输出
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        # 返回TFBaseModelOutput类型的结果，包括最终hidden_state、hidden_states和attentions
        return TFBaseModelOutput(
            last_hidden_state=decoder_outputs[0],
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 设置构建标记为True
        self.built = True
        # 如果embeddings存在，则构建embeddings
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果encoder存在，则构建encoder
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果decoder存在，则构建decoder
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 定义一个用于判别器的预测模块，由两个密集层组成
class TFFunnelDiscriminatorPredictions(tf.keras.layers.Layer):
    
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 从配置中获取初始化范围，并用它来初始化权重
        initializer = get_initializer(config.initializer_range)
        # 创建一个密集层，输出维度为config.d_model，使用之前定义的initializer进行初始化
        self.dense = tf.keras.layers.Dense(config.d_model, kernel_initializer=initializer, name="dense")
        # 获取激活函数
        self.activation_function = get_tf_activation(config.hidden_act)
        # 创建一个密集层，输出维度为1，使用之前定义的initializer进行初始化
        self.dense_prediction = tf.keras.layers.Dense(1, kernel_initializer=initializer, name="dense_prediction")
        self.config = config

    def call(self, discriminator_hidden_states):
        # 通过密集层处理判别器的隐藏状态
        hidden_states = self.dense(discriminator_hidden_states)
        # 使用激活函数处理隐藏状态
        hidden_states = self.activation_function(hidden_states)
        # 压缩结果并返回Logits
        logits = tf.squeeze(self.dense_prediction(hidden_states))
        return logits

    def build(self, input_shape=None):
        # 如果已经构建，直接返回
        if self.built:
            return
        self.built = True
        # 如果已经定义了dense层，则构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.d_model])
        # 如果已经定义了dense_prediction层，则构建它
        if getattr(self, "dense_prediction", None) is not None:
            with tf.name_scope(self.dense_prediction.name):
                self.dense_prediction.build([None, None, self.config.d_model])

# 定义一个用于掩码语言模型头部的类
class TFFunnelMaskedLMHead(tf.keras.layers.Layer):

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        # 添加一个偏置项，shape为(config.vocab_size,)
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.input_embeddings

    # 设置输出嵌入层
    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置项
    def get_bias(self):
        return {"bias": self.bias}

    # 设置偏置项
    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    # 处理隐藏状态
    def call(self, hidden_states, training=False):
        # 获取序列长度
        seq_length = shape_list(tensor=hidden_states)[1]
        # 重塑隐藏状态
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        # 矩阵相乘得到结果
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        # 重塑结果
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        # 添加偏置项
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states


class TFFunnelClassificationHead(tf.keras.layers.Layer):
    # 初始化方法，接收配置、标签数量和其他关键字参数
    def __init__(self, config, n_labels, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 根据配置中的初始化范围获取初始化器
        initializer = get_initializer(config.initializer_range)
        # 创建一个全连接层，输出维度为配置中的模型维度，使用指定的初始化器，命名为"linear_hidden"
        self.linear_hidden = tf.keras.layers.Dense(
            config.d_model, kernel_initializer=initializer, name="linear_hidden"
        )
        # 创建一个Dropout层，dropout率为配置中的隐藏层dropout率
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 创建一个全连接层，输出维度为标签数量，使用指定的初始化器，命名为"linear_out"
        self.linear_out = tf.keras.layers.Dense(n_labels, kernel_initializer=initializer, name="linear_out")
        # 保存配置
        self.config = config

    # 前向传播方法，接收隐藏层输入和训练标志
    def call(self, hidden, training=False):
        # 输入经过全连接层"linear_hidden"
        hidden = self.linear_hidden(hidden)
        # 使用双曲正切激活函数
        hidden = tf.keras.activations.tanh(hidden)
        # 使用Dropout层进行dropout，训练中使用
        hidden = self.dropout(hidden, training=training)
        # 最终输出经过全连接层"linear_out"
        return self.linear_out(hidden)

    # 构建方法，接收输入形状
    def build(self, input_shape=None):
        # 如果已经构建过了，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果"linear_hidden"存在，则构建它，形状为[None, None, 配置中的模型维度]
        if getattr(self, "linear_hidden", None) is not None:
            with tf.name_scope(self.linear_hidden.name):
                self.linear_hidden.build([None, None, self.config.d_model])
        # 如果"linear_out"存在，则构建它，形状为[None, None, 配置中的模型维度]
        if getattr(self, "linear_out", None) is not None:
            with tf.name_scope(self.linear_out.name):
                self.linear_out.build([None, None, self.config.d_model])
class TFFunnelPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义类属性config_class为FunnelConfig
    config_class = FunnelConfig
    # 定义基模型的前缀为"funnel"
    base_model_prefix = "funnel"

    @property
    def dummy_inputs(self):
        # 重写dummy_inputs方法，由于Funnel在处理非常小的输入时会出现问题，因此这里将输入稍微放大一点
        return {"input_ids": tf.ones((1, 3), dtype=tf.int32)}


@dataclass
class TFFunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FunnelForPreTraining`].

    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
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

    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


FUNNEL_START_DOCSTRING = r"""

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

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
    # 将输入和标签以任何`model.fit()`支持的格式传递！
    # 如果您想在Keras方法之外（如在使用Keras`Functional`API创建自己的层或模型时）使用第二种格式，
    # 例如`fit()`和`predict()`，则有三种可能性可用于收集第一个位置参数中的所有输入张量：
    # - 仅具有`input_ids`的单个张量，没有其他内容：`model(input_ids)`
    # - 长度不等的列表，包含按照文档字符串中给定的顺序的一个或多个输入张量：
    # `model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 一个字典，其中包含一个或多个与文档字符串中给定的输入名称相关联的输入张量：
    # `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    # 请注意，当使用[子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，您就不必担心任何这些，
    # 因为您可以像对待任何其它Python函数一样传递输入！

    # 参数：
    # config ([`XxxConfig`]): 模型配置类，包含模型的所有参数。
    # 用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
# 基于Funnel Transformer模型的基础Transformer模型，输出没有上采样头（也称为解码器）或任何特定任务头部的原始隐藏状态。
# 这里包含了Funnel Transformer模型的基础说明文档

# 创建TFFunnelBaseModel类，继承自TFFunnelPreTrainedModel类
class TFFunnelBaseModel(TFFunnelPreTrainedModel):
    # 初始化函数，接受 FunnelConfig 配置和输入参数，调用父类的初始化函数，并创建一个 TFFunnelBaseLayer 对象
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.funnel = TFFunnelBaseLayer(config, name="funnel")

    # 调用函数装饰器，给函数添加文档字符串说明和代码示例
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 拆解输入参数，定义调用函数，传入各种输入参数并调用 self.funnel 对象
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutput]:
        return self.funnel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

    # 定义服务输出函数，输出最后的隐藏状态、隐藏状态和注意力
    def serving_output(self, output):
        # 不使用 tf.convert_to_tensor 转换隐藏状态和注意力，因为它们的维度不同
        return TFBaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    # 构建模型，如果已经构建过则直接返回，否则构建 self.funnel 对象
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
# 使用自定义的文档字符串添加模型描述和文档
@add_start_docstrings(
    "The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.",
    FUNNEL_START_DOCSTRING,
)
# 定义一个 Funnel Transformer 模型类继承自 TFFunnelPreTrainedModel
class TFFunnelModel(TFFunnelPreTrainedModel):
    # 初始化方法
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建 Funnel 主体层
        self.funnel = TFFunnelMainLayer(config, name="funnel")

    # 调用方法，根据输入参数调用 Funnel 模型
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutput]:
        # 调用 Funnel 模型并返回结果
        return self.funnel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

    # 服务输出方法，返回基础模型输出
    def serving_output(self, output):
        # 返回基础模型输出
        return TFBaseModelOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    # 构建方法，构建 Funnel 模型
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        # 设置已构建标志
        self.built = True
        # 如果存在 Funnel 则构建
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)


# 使用自定义的文档字符串添加模型描述和文档
@add_start_docstrings(
    """
    Funnel model with a binary classification head on top as used during pretraining for identifying generated tokens.
    """,
    FUNNEL_START_DOCSTRING,
)
# 定义一个用于预训练的 Funnel 模型类
class TFFunnelForPreTraining(TFFunnelPreTrainedModel):
    # 初始化方法
    def __init__(self, config: FunnelConfig, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 创建 Funnel 主体层和鉴别器预测
        self.funnel = TFFunnelMainLayer(config, name="funnel")
        self.discriminator_predictions = TFFunnelDiscriminatorPredictions(config, name="discriminator_predictions")

    # 调用方法，根据输入参数调用 Funnel 模型
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换输出文档字符串
    @replace_return_docstrings(output_type=TFFunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 定义调用方法，接受多种输入参数，返回预训练输出
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,
    ) -> Union[Tuple[tf.Tensor], TFFunnelForPreTrainingOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, TFFunnelForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small")
        >>> model = TFFunnelForPreTraining.from_pretrained("funnel-transformer/small")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> logits = model(inputs).logits
        ```"""
        # 使用Funnel模型对输入进行预训练
        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        # 获取判别器预测
        logits = self.discriminator_predictions(discriminator_sequence_output)

        # 如果不需要返回dict，则返回logits和隐藏状态
        if not return_dict:
            return (logits,) + discriminator_hidden_states[1:]

        # 否则返回预训练输出
        return TFFunnelForPreTrainingOutput(
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    # 定义serving_output方法，返回预测输出
    def serving_output(self, output):
        # 隐藏状态和注意力不会转换为张量，因为它们的维度不同
        return TFFunnelForPreTrainingOutput(
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        if getattr(self, "funnel", None) is not None:
            # 使用Funnel模型进行构建
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        if getattr(self, "discriminator_predictions", None) is not None:
            # 使用判别器进行构建
            with tf.name_scope(self.discriminator_predictions.name):
                self.discriminator_predictions.build(None)
# 使用装饰器添加模型的开始文档字符串和Funnel开始文档字符串
@add_start_docstrings("""Funnel Model with a `language modeling` head on top.""", FUNNEL_START_DOCSTRING)
# 定义TFFunnelForMaskedLM类，继承自TFFunnelPreTrainedModel和TFMaskedLanguageModelingLoss
class TFFunnelForMaskedLM(TFFunnelPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 初始化方法
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        # 定义funnel属性为TFFunnelMainLayer对象
        self.funnel = TFFunnelMainLayer(config, name="funnel")
        # 定义lm_head属性为TFFunnelMaskedLMHead对象
        self.lm_head = TFFunnelMaskedLMHead(config, self.funnel.embeddings, name="lm_head")

    # 获取lm_head属性的方法
    def get_lm_head(self) -> TFFunnelMaskedLMHead:
        return self.lm_head

    # 获取前缀偏置名称的方法
    def get_prefix_bias_name(self) -> str:
        # 发出警告，方法已废弃，建议使用get_bias方法
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    # 调用方法，实现模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFMaskedLMOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用funnel模型的前向传播
        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]
        # 使用lm_head对序列输出进行预测
        prediction_scores = self.lm_head(sequence_output, training=training)

        # 如果labels不为None则计算损失，否则损失为None
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果return_dict为False，则返回output中的内容，否则返回TFMaskedLMOutput对象
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 定义一个方法，接受TFMaskedLMOutput类型的参数，并返回TFMaskedLMOutput类型的结果
    def serving_output(self, output: TFMaskedLMOutput) -> TFMaskedLMOutput:
        # hidden_states和attentions没有使用tf.convert_to_tensor转换为张量，因为它们的维度不同
        # 返回一个新的TFMaskedLMOutput对象，包含logits、hidden_states和attentions
        return TFMaskedLMOutput(logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions)

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建好了，就直接返回
        if self.built:
            return
        self.built = True
        # 如果存在名为"funnel"的属性，则在名为"funnel"的作用域下构建模型
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        # 如果存在名为"lm_head"的属性，则在名为"lm_head"的作用域下构建模型
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                self.lm_head.build(None)
# 使用 Funnel 模型进行序列分类/回归任务，顶部有一个线性层（线性层在汇总输出之上），例如用于 GLUE 任务
@add_start_docstrings(
    """
    Funnel Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForSequenceClassification(TFFunnelPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化方法，接收 FunnelConfig 类型的 config 参数和其他输入
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        # 获取类别数量
        self.num_labels = config.num_labels

        # 创建 FunnelBaseLayer 和 FunnelClassificationHead 对象
        self.funnel = TFFunnelBaseLayer(config, name="funnel")
        self.classifier = TFFunnelClassificationHead(config, config.num_labels, name="classifier")

    # 定义 call 方法，接收多个输入参数和返回值
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFSequenceClassifierOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 使用 Funnel 模型处理输入数据
        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]  # 获取汇总的输出
        logits = self.classifier(pooled_output, training=training)  # 使用分类器进行分类

        # 根据标签和预测结果计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不返回字典，则将结果包装到元组中返回
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则以 TFSequenceClassifierOutput 格式返回结果
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def serving_output(self, output: TFSequenceClassifierOutput) -> TFSequenceClassifierOutput:
        # 将输出中的隐藏状态和注意力矩阵作为非张量返回，因为它们维度不同
        return TFSequenceClassifierOutput(
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)  # 构建funnel模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)  # 构建classifier模型
# 为多选分类任务添加一个漏斗模型，该模型在汇总输出的基础上有一个多选分类头部（池化输出上方的线性层和一个 softmax 层），例如用于 RocStories/SWAG 任务。
@add_start_docstrings(
    """
    Funnel Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
# 定义 TFFunnelForMultipleChoice 类，继承自 TFFunnelPreTrainedModel 和 TFMultipleChoiceLoss
class TFFunnelForMultipleChoice(TFFunnelPreTrainedModel, TFMultipleChoiceLoss):
    # 定义初始化函数，输入参数包括 config、inputs 和 kwargs
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)

        # 初始化漏斗模型
        self.funnel = TFFunnelBaseLayer(config, name="funnel")
        # 初始化分类器
        self.classifier = TFFunnelClassificationHead(config, 1, name="classifier")

    # 定义 dummy_inputs 属性，返回一个字典
    @property
    def dummy_inputs(self):
        return {"input_ids": tf.ones((3, 3, 4), dtype=tf.int32)}

    # 解包输入参数，为模型前向传播添加文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 call 函数，接收多个输入参数
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFMultipleChoiceModelOutput]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果输入中包含 input_ids
        if input_ids is not None:
            # 获取 input_ids 的第二维度大小，即选择数量
            num_choices = shape_list(input_ids)[1]
            # 获取 input_ids 的第三维度大小，即序列长度
            seq_length = shape_list(input_ids)[2]
        else:
            # 获取 inputs_embeds 的第二维度大小，即选择数量
            num_choices = shape_list(inputs_embeds)[1]
            # 获取 inputs_embeds 的第三维度大小，即序列长度
            seq_length = shape_list(inputs_embeds)[2]

        # 将 input_ids 展开成二维数组，形状为 (-1, seq_length)，如果 input_ids 不为 None
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        # 将 attention_mask 展开成二维数组，形状为 (-1, seq_length)，如果 attention_mask 不为 None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        # 将 token_type_ids 展开成二维数组，形状为 (-1, seq_length)，如果 token_type_ids 不为 None
        flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None
        # 将 inputs_embeds 展开成三维数组，形状为 (-1, seq_length, shape_list(inputs_embeds)[3])，如果 inputs_embeds 不为 None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )

        # 对 funnel 模型进行前向传播
        outputs = self.funnel(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取最后一层隐藏状态
        last_hidden_state = outputs[0]
        # 获取池化后的输出
        pooled_output = last_hidden_state[:, 0]
        # 对池化后的输出进行分类
        logits = self.classifier(pooled_output, training=training)
        # 重塑分类结果的形状
        reshaped_logits = tf.reshape(logits, (-1, num_choices))

        # 如果 labels 不为 None��则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不以字典形式返回结果
        if not return_dict:
            # 组装输出结果
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 以 TFMultipleChoiceModelOutput 格式返回结果
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 用于服务输出的函数
    def serving_output(self, output: TFMultipleChoiceModelOutput) -> TFMultipleChoiceModelOutput:
        # hidden_states 和 attentions 不会转换为张量，因为它们的维度都不同
        return TFMultipleChoiceModelOutput(
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果模型中有funnel属性
        if getattr(self, "funnel", None) is not None:
            # 使用funnel的名称创建一个命名空间
            with tf.name_scope(self.funnel.name):
                # 构建funnel模型
                self.funnel.build(None)
        # 如果模型中有classifier属性
        if getattr(self, "classifier", None) is not None:
            # 使用classifier的名称创建一个命名空间
            with tf.name_scope(self.classifier.name):
                # 构建classifier模型
                self.classifier.build(None)
# 使用 Token Classification 头部的 Funnel 模型，用于命名实体识别(Named Entity Recognition, NER)等任务
class TFFunnelForTokenClassification(TFFunnelPreTrainedModel, TFTokenClassificationLoss):
    
    # 初始化函数，接收 FunnelConfig 对象和额外的输入参数
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        
        # 设置 num_labels 属性为配置中的标签数量
        self.num_labels = config.num_labels
        
        # 创建 Funnel 主层对象
        self.funnel = TFFunnelMainLayer(config, name="funnel")
        # 创建 Dropout 层对象
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        # 创建分类器 Dense 层对象
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置对象
        self.config = config
    
    # 调用方法，处理模型输入并执行前向传播
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFTokenClassifierOutput]:
        
        # 调用 Funnel 主层对象进行前向传播
        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        
        # 获取序列输出
        sequence_output = outputs[0]
        
        # 对序列输出应用 Dropout
        sequence_output = self.dropout(sequence_output, training=training)
        # 通过分类器获取 logits
        logits = self.classifier(sequence_output)
        
        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, logits)
        
        # 如果不返回字典，则返回结果元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # 如果返回字典，则构建 TFTokenClassifierOutput 对象并返回
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 为输出结果添加服务格式化
    def serving_output(self, output: TFTokenClassifierOutput) -> TFTokenClassifierOutput:
        # 不将 hidden_states 和 attentions 转换为张量，因为它们的维度各不相同
        return TFTokenClassifierOutput(
            # 返回原始输出的 logits、hidden_states 和 attentions
            logits=output.logits, hidden_states=output.hidden_states, attentions=output.attentions
        )

    # 构建模型的内部结构
    def build(self, input_shape=None):
        # 如果模型已经被构建过，则不执行任何操作
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果模型包含漏斗结构
        if getattr(self, "funnel", None) is not None:
            # 在特定的命名作用域内构建漏斗层
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)
        # 如果模型包含分类器
        if getattr(self, "classifier", None) is not None:
            # 在特定的命名作用域内构建分类器层，设置输入形状为 (batch_size, sequence_length, hidden_size)
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 使用 Funnel 模型进行抽取式问答任务的模型定义，其顶部有一个用于 span 分类的头部（在隐藏状态输出的基础上有线性层，用于计算“span 起始 logits”和“span 结束 logits”）
@add_start_docstrings(
    """
    Funnel Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FUNNEL_START_DOCSTRING,
)
class TFFunnelForQuestionAnswering(TFFunnelPreTrainedModel, TFQuestionAnsweringLoss):
    # 初始化方法，接受 FunnelConfig 实例对象作为参数，以及其他可选输入
    def __init__(self, config: FunnelConfig, *inputs, **kwargs) -> None:
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 设置模型的标签数量
        self.num_labels = config.num_labels

        # 创建 Funnel 主层对象，并命名为“funnel”
        self.funnel = TFFunnelMainLayer(config, name="funnel")
        # 创建输出层，使用全连接层将隐藏状态输出转换为问题回答的标签数量，使用指定的初始化器初始化权重，命名为“qa_outputs”
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 将配置对象保存到模型中
        self.config = config

    # 定义模型的调用方法，处理输入并返回模型输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small",
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        ) -> Union[Tuple[tf.Tensor], TFQuestionAnsweringModelOutput]:
        r"""
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs = self.funnel(
            input_ids,
            attention_mask,
            token_type_ids,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]  # 从模型输出中提取序列输出

        logits = self.qa_outputs(sequence_output)  # 使用序列输出生成答案的logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)  # 将logits分割成开始和结束位置的logits
        start_logits = tf.squeeze(start_logits, axis=-1)  # 压缩开始位置的logits
        end_logits = tf.squeeze(end_logits, axis=-1)  # 压缩结束位置的logits

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))  # 计算损失函数

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]  # 输出包含开始和结束位置的logits以及其他输出
            return ((loss,) + output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,  # 返回loss
            start_logits=start_logits,  # 返回开始位置的logits
            end_logits=end_logits,  # 返回结束位置的logits
            hidden_states=outputs.hidden_states,  # 返回隐藏状态
            attentions=outputs.attentions,  # 返回注意力权重
        )

    def serving_output(self, output: TFQuestionAnsweringModelOutput) -> TFQuestionAnsweringModelOutput:
        # hidden_states and attentions not converted to Tensor with tf.convert_to_tensor as they are all of
        # different dimensions
        return TFQuestionAnsweringModelOutput(
            start_logits=output.start_logits,  # 返回开始位置���logits
            end_logits=output.end_logits,  # 返回结束位置的logits
            hidden_states=output.hidden_states,  # 返回隐藏状态
            attentions=output.attentions,  # 返回注意力权重
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "funnel", None) is not None:
            with tf.name_scope(self.funnel.name):
                self.funnel.build(None)  # 构建模型
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])  # 构建回答的输出模型
```