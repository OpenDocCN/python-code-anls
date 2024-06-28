# `.\models\deprecated\transfo_xl\modeling_tf_transfo_xl.py`

```
"""
 TF 2.0 Transformer XL model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ....modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    get_initializer,
    keras,
    keras_serializable,
    unpack_inputs,
)
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "transfo-xl/transfo-xl-wt103"
_CONFIG_FOR_DOC = "TransfoXLConfig"

# 预训练模型存档列表
TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "transfo-xl/transfo-xl-wt103",
    # 查看所有 Transformer XL 模型的列表：https://huggingface.co/models?filter=transfo-xl
]


class TFPositionalEmbedding(keras.layers.Layer):
    def __init__(self, demb, **kwargs):
        super().__init__(**kwargs)

        # 初始化逆频率矩阵，用于位置编码
        self.inv_freq = 1 / (10000 ** (tf.range(0, demb, 2.0) / demb))

    def call(self, pos_seq, bsz=None):
        # 将逆频率转换为与位置序列相同的数据类型
        self.inv_freq = tf.cast(self.inv_freq, dtype=pos_seq.dtype)
        # 计算正弦和余弦函数输入，形成位置编码矩阵
        sinusoid_inp = tf.einsum("i,j->ij", pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)

        if bsz is not None:
            # 如果提供了批大小，对位置编码进行扩展
            return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
        else:
            # 否则返回单个位置编码矩阵
            return pos_emb[:, None, :]


class TFPositionwiseFF(keras.layers.Layer):
    # 初始化函数，用于创建一个新的自定义层对象
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5, init_std=0.02, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置模型参数
        self.d_model = d_model    # 模型维度
        self.d_inner = d_inner    # 内部层维度
        self.dropout = dropout    # 丢弃率

        # 第一个全连接层，用于非线性变换
        self.layer_1 = keras.layers.Dense(
            d_inner, kernel_initializer=get_initializer(init_std), activation=tf.nn.relu, name="CoreNet_._0"
        )
        # 第一个丢弃层，用于正则化
        self.drop_1 = keras.layers.Dropout(dropout)
        # 第二个全连接层，用于映射回原始模型维度
        self.layer_2 = keras.layers.Dense(d_model, kernel_initializer=get_initializer(init_std), name="CoreNet_._3")
        # 第二个丢弃层，用于正则化
        self.drop_2 = keras.layers.Dropout(dropout)

        # 层标准化，用于规范化每个样本的特征
        self.layer_norm = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layer_norm")

        # 是否使用预层标准化
        self.pre_lnorm = pre_lnorm

    def call(self, inp, training=False):
        # 如果使用预层标准化
        if self.pre_lnorm:
            # 层标准化 + 位置感知前向反馈
            core_out = self.layer_norm(inp)
            core_out = self.layer_1(core_out)     # 第一个全连接层
            core_out = self.drop_1(core_out, training=training)   # 第一个丢弃层
            core_out = self.layer_2(core_out)     # 第二个全连接层
            core_out = self.drop_2(core_out, training=training)   # 第二个丢弃层

            # 残差连接
            output = core_out + inp   # 输出等于核心输出加上输入
        else:
            # 位置感知前向反馈
            core_out = self.layer_1(inp)   # 第一个全连接层
            core_out = self.drop_1(core_out, training=training)   # 第一个丢弃层
            core_out = self.layer_2(core_out)   # 第二个全连接层
            core_out = self.drop_2(core_out, training=training)   # 第二个丢弃层

            # 残差连接 + 层标准化
            output = self.layer_norm(inp + core_out)   # 输出等于输入加上核心输出后进行层标准化

        return output
# 定义 Transformer 中解码器层的自定义 Keras 层
class TFRelPartialLearnableDecoderLayer(keras.layers.Layer):
    def __init__(
        self,
        n_head,                      # 注意力头的数量
        d_model,                     # 模型的维度
        d_head,                      # 每个注意力头的维度
        d_inner,                     # 内部前馈网络的维度
        dropout,                     # 注意力和前馈网络中的dropout率
        dropatt=0.0,                 # 注意力机制中的额外dropout率
        pre_lnorm=False,             # 是否在 LayerNormalization 前应用注意力和前馈网络
        r_w_bias=None,               # 注意力机制中的位置偏置
        r_r_bias=None,               # 注意力机制中的位置偏置
        layer_norm_epsilon=1e-5,     # LayerNormalization 的 epsilon 参数
        init_std=0.02,               # 权重初始化的标准差
        output_attentions=False,     # 是否输出注意力权重
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_head = n_head                           # 初始化注意力头的数量
        self.d_model = d_model                         # 初始化模型的维度
        self.d_head = d_head                           # 初始化每个注意力头的维度
        self.dropout = dropout                         # 初始化dropout率
        self.output_attentions = output_attentions     # 是否输出注意力权重

        # 定义一个全连接层，用于计算查询、键和值
        self.qkv_net = keras.layers.Dense(
            3 * n_head * d_head,                        # 输出维度是 3 * 注意力头数 * 注意力头维度
            kernel_initializer=get_initializer(init_std),
            use_bias=False,
            name="qkv_net"
        )

        self.drop = keras.layers.Dropout(dropout)       # 定义dropout层，用于注意力权重和前馈网络
        self.dropatt = keras.layers.Dropout(dropatt)   # 定义额外的dropout层，用于注意力权重
        self.o_net = keras.layers.Dense(
            d_model,                                    # 输出维度为模型的维度
            kernel_initializer=get_initializer(init_std),
            use_bias=False,
            name="o_net"
        )

        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name="layer_norm"
        )

        self.scale = 1 / (d_head**0.5)                  # 缩放因子为注意力头维度的平方根

        self.pre_lnorm = pre_lnorm                      # 是否在LayerNormalization之前应用注意力和前馈网络

        if r_r_bias is not None and r_w_bias is not None:  # 如果提供了位置偏置，则共享位置偏置
            self.r_r_bias = r_r_bias                     # 初始化相对位置的重排偏置
            self.r_w_bias = r_w_bias                     # 初始化相对位置的重排偏置
        else:
            self.r_r_bias = None                         # 否则初始化为None
            self.r_w_bias = None                         # 否则初始化为None

        # 定义一个全连接层，用于计算相对位置的重排
        self.r_net = keras.layers.Dense(
            self.n_head * self.d_head,                   # 输出维度为 注意力头数 * 注意力头维度
            kernel_initializer=get_initializer(init_std),
            use_bias=False,
            name="r_net"
        )

    def build(self, input_shape):
        if self.r_r_bias is None or self.r_w_bias is None:  # 如果未提供位置偏置，则创建并添加可训练的位置偏置
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head),
                initializer="zeros",
                trainable=True,
                name="r_r_bias"
            )
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head),
                initializer="zeros",
                trainable=True,
                name="r_w_bias"
            )
        super().build(input_shape)

    def _rel_shift(self, x):
        x_size = shape_list(x)

        # 在第二个维度上填充1行0列，用于相对位置的重排
        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        # 重新整形张量以便进行切片操作
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        # 进行切片操作以去除填充的部分
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        # 恢复原来的形状
        x = tf.reshape(x, x_size)

        return x
    ):
        super().__init__(**kwargs)
        
        self.dec_attn = TFRelPartialLearnableMultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=dropatt,
            pre_lnorm=pre_lnorm,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            init_std=init_std,
            layer_norm_epsilon=layer_norm_epsilon,
            output_attentions=output_attentions,
            name="dec_attn",
        )
        self.pos_ff = TFPositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm=pre_lnorm,
            init_std=init_std,
            layer_norm_epsilon=layer_norm_epsilon,
            name="pos_ff",
        )


# 初始化函数，用于创建一个新的实例
def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.0, pre_lnorm=False,
             r_w_bias=None, r_r_bias=None, init_std=0.02, layer_norm_epsilon=1e-12,
             output_attentions=False, **kwargs):
    # 调用父类的初始化方法，传入额外的关键字参数
    super().__init__(**kwargs)

    # 创建多头注意力机制对象
    self.dec_attn = TFRelPartialLearnableMultiHeadAttn(
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=dropatt,
        pre_lnorm=pre_lnorm,
        r_w_bias=r_w_bias,
        r_r_bias=r_r_bias,
        init_std=init_std,
        layer_norm_epsilon=layer_norm_epsilon,
        output_attentions=output_attentions,
        name="dec_attn",
    )
    
    # 创建位置前馈神经网络对象
    self.pos_ff = TFPositionwiseFF(
        d_model,
        d_inner,
        dropout,
        pre_lnorm=pre_lnorm,
        init_std=init_std,
        layer_norm_epsilon=layer_norm_epsilon,
        name="pos_ff",
    )



    def call(self, dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=False):
        # 使用 self.dec_attn 对象进行调用，计算注意力输出
        attn_outputs = self.dec_attn(dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=training)
        
        # 使用 self.pos_ff 对象进行调用，计算位置前馈网络输出
        ff_output = self.pos_ff(attn_outputs[0], training=training)

        # 将位置前馈网络输出与注意力输出列表合并为一个输出列表
        outputs = [ff_output] + attn_outputs[1:]

        # 返回最终输出列表
        return outputs


# 定义 call 方法，用于执行模型的前向传播过程，接收多个参数并返回多个输出
def call(self, dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=False):
    # 调用 self.dec_attn 对象，计算注意力输出
    attn_outputs = self.dec_attn(dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=training)
    
    # 调用 self.pos_ff 对象，计算位置前馈网络输出
    ff_output = self.pos_ff(attn_outputs[0], training=training)

    # 将位置前馈网络的输出和注意力输出列表合并为一个最终输出列表
    outputs = [ff_output] + attn_outputs[1:]

    # 返回最终的输出列表
    return outputs
# 定义 TFTransfoEmbeddings 类，继承自 keras.layers.Layer
class TFTransfoEmbeddings(keras.layers.Layer):
    # 初始化方法，接受词汇量大小、嵌入维度、初始化标准差等参数
    def __init__(self, vocab_size, emb_size, init_std, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size  # 设置词汇量大小
        self.emb_size = emb_size      # 设置嵌入维度
        self.init_std = init_std      # 设置初始化标准差

    # 构建方法，在此处创建权重
    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.vocab_size, self.emb_size),  # 设置权重的形状为 (词汇量大小, 嵌入维度)
            initializer=get_initializer(self.init_std),  # 使用给定的初始化器初始化权重
            name="embeddings",  # 设置权重的名称为 "embeddings"
        )

        super().build(input_shape)  # 调用父类的 build 方法

    # 调用方法，用于获取给定输入的嵌入表示
    def call(self, inputs):
        return tf.gather(self.weight, inputs)  # 返回权重中对应输入索引的嵌入表示


# 定义 TFAdaptiveEmbedding 类，继承自 keras.layers.Layer
class TFAdaptiveEmbedding(keras.layers.Layer):
    # 初始化方法，接受词汇量、嵌入维度、投影维度、截断列表等参数
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, init_std=0.02, sample_softmax=False, **kwargs):
        super().__init__(**kwargs)

        self.n_token = n_token      # 设置词汇量
        self.d_embed = d_embed      # 设置嵌入维度
        self.init_std = init_std    # 设置初始化标准差

        self.cutoffs = cutoffs + [n_token]  # 设置截断列表，并加入最大词汇量
        self.div_val = div_val      # 设置除法因子
        self.d_proj = d_proj        # 设置投影维度

        self.emb_scale = d_proj**0.5  # 计算嵌入缩放因子

        self.cutoff_ends = [0] + self.cutoffs  # 计算截断结束点列表

        self.emb_layers = []  # 初始化嵌入层列表
        self.emb_projs = []   # 初始化嵌入投影列表

        # 如果除法因子为 1，抛出未实现错误，否则创建嵌入层和投影
        if div_val == 1:
            raise NotImplementedError
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(
                    TFTransfoEmbeddings(
                        r_idx - l_idx,
                        d_emb_i,
                        init_std,
                        name=f"emb_layers_._{i}",  # 设置嵌入层的名称
                    )
                )

    # 构建方法，在此处创建投影权重
    def build(self, input_shape):
        for i in range(len(self.cutoffs)):
            d_emb_i = self.d_embed // (self.div_val**i)
            self.emb_projs.append(
                self.add_weight(
                    shape=(d_emb_i, self.d_proj),  # 设置投影权重的形状为 (当前嵌入维度 // 当前除法因子^i, 投影维度)
                    initializer=get_initializer(self.init_std),  # 使用给定的初始化器初始化投影权重
                    trainable=True,
                    name=f"emb_projs_._{i}",  # 设置投影权重的名称
                )
            )

        super().build(input_shape)  # 调用父类的 build 方法
    # 定义一个方法 `call`，接受一个输入参数 `inp`
    def call(self, inp):
        # 如果 `div_val` 等于 1，抛出未实现错误
        if self.div_val == 1:
            raise NotImplementedError  # 这里抛出错误，因为在我们的预训练检查点中这些代码未使用
        else:
            # 将输入 `inp` 展平成一维数组 `inp_flat`
            inp_flat = tf.reshape(inp, (-1,))
            # 创建一个全零张量 `emb_flat`，形状为 (inp_flat 的长度, self.d_proj)
            emb_flat = tf.zeros([shape_list(inp_flat)[0], self.d_proj])
            # 遍历 `self.cutoffs` 列表中的每个元素
            for i in range(len(self.cutoffs)):
                # 获取当前分段的左右索引
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                # 创建一个布尔掩码 `mask_i`，标记出位于当前分段内的元素
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)

                # 根据掩码 `mask_i` 从 `inp_flat` 中获取子集 `inp_i`，并将其归一化到从 0 开始
                inp_i = tf.boolean_mask(inp_flat, mask_i) - l_idx
                # 使用 `self.emb_layers[i]` 对 `inp_i` 进行嵌入操作
                emb_i = self.emb_layers[i](inp_i)
                # 将嵌入向量 `emb_i` 与对应的投影矩阵 `self.emb_projs[i]` 进行点乘
                emb_i = tf.einsum("id,de->ie", emb_i, self.emb_projs[i])

                # 根据 `mask_idx` 的索引在 `emb_flat` 上执行散列更新
                mask_idx = tf.where(mask_i)
                scatter = tf.scatter_nd(mask_idx, emb_i, shape_list(emb_flat))
                emb_flat = tf.cast(emb_flat, dtype=scatter.dtype)
                emb_flat += scatter

            # 将 `emb_flat` 重新整形为与 `inp` 相同形状的张量 `embed`
            embed_shape = shape_list(inp) + [self.d_proj]
            embed = tf.reshape(emb_flat, embed_shape)

        # 将嵌入张量 `embed` 乘以 `emb_scale`
        embed *= self.emb_scale

        # 返回嵌入张量 `embed` 作为方法的输出结果
        return embed
@keras_serializable
class TFTransfoXLMainLayer(keras.layers.Layer):
    # 指定配置类为 TransfoXLConfig
    config_class = TransfoXLConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 将传入的配置对象保存到实例中
        self.config = config
        self.output_hidden_states = config.output_hidden_states  # 是否输出隐藏状态
        self.output_attentions = config.output_attentions  # 是否输出注意力权重
        self.return_dict = config.use_return_dict  # 是否使用字典形式返回结果

        self.n_token = config.vocab_size  # 词汇表大小

        self.d_embed = config.d_embed  # 嵌入维度
        self.d_model = config.d_model  # 模型维度
        self.n_head = config.n_head  # 注意力头数
        self.d_head = config.d_head  # 每个注意力头的维度
        self.untie_r = config.untie_r  # 是否解开注意力头的参数

        # 创建自适应嵌入层
        self.word_emb = TFAdaptiveEmbedding(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            div_val=config.div_val,
            init_std=config.init_std,
            name="word_emb",
        )

        # Dropout 层
        self.drop = keras.layers.Dropout(config.dropout)

        self.n_layer = config.n_layer  # 层数
        self.mem_len = config.mem_len  # 记忆长度
        self.attn_type = config.attn_type  # 注意力类型

        self.layers = []  # 初始化层列表
        if config.attn_type == 0:  # 如果是默认的注意力类型
            # 创建多层自定义解码器层
            for i in range(config.n_layer):
                self.layers.append(
                    TFRelPartialLearnableDecoderLayer(
                        config.n_head,
                        config.d_model,
                        config.d_head,
                        config.d_inner,
                        config.dropout,
                        dropatt=config.dropatt,
                        pre_lnorm=config.pre_lnorm,
                        r_w_bias=None if self.untie_r else self.r_w_bias,
                        r_r_bias=None if self.untie_r else self.r_r_bias,
                        layer_norm_epsilon=config.layer_norm_epsilon,
                        init_std=config.init_std,
                        output_attentions=self.output_attentions,
                        name=f"layers_._{i}",
                    )
                )
        else:  # 如果是其他类型的注意力（这部分未实现）
            raise NotImplementedError  # 已删除这些代码以避免维护死代码 - 在预训练检查点中未使用

        self.same_length = config.same_length  # 是否长度相同
        self.clamp_len = config.clamp_len  # 限制长度

        if self.attn_type == 0:  # 如果是默认的注意力类型
            # 创建位置嵌入层
            self.pos_emb = TFPositionalEmbedding(self.d_model, name="pos_emb")
        else:  # 如果是其他类型的注意力（这部分未实现）
            raise NotImplementedError  # 已删除这些代码以避免维护死代码 - 在预训练检查点中未使用
    # 在构建模型时，根据输入形状设置权重参数，如果未指定则设置r_w_bias和r_r_bias为全零向量
    def build(self, input_shape):
        if not self.untie_r:
            # 添加r_w_bias权重，形状为(n_head, d_head)，初始化为全零，可训练
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
            # 添加r_r_bias权重，形状为(n_head, d_head)，初始化为全零，可训练
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
        # 调用父类的build方法
        super().build(input_shape)

    # 返回输入嵌入向量
    def get_input_embeddings(self):
        return self.word_emb

    # 设置输入嵌入向量，但此方法未实现，抛出未实现错误
    def set_input_embeddings(self, value):
        raise NotImplementedError

    # 向后兼容性方法，设置sample_softmax为-1
    def backward_compatible(self):
        self.sample_softmax = -1

    # 重置记忆长度为给定的mem_len
    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    # 剪枝特定头部的方法，但此方法未实现，抛出未实现错误
    def _prune_heads(self, heads):
        raise NotImplementedError

    # 初始化记忆数组，若mem_len大于0，则创建长度为mem_len的空记忆列表，每个元素形状为(mem_len, bsz, d_model)
    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer):
                empty = tf.zeros([self.mem_len, bsz, self.d_model])
                mems.append(empty)

            return mems
        else:
            return None

    # 更新记忆数组，将隐藏状态hids缓存到mems中，mlen为之前的记忆长度，qlen为当前查询长度
    def _update_mems(self, hids, mems, mlen, qlen):
        # 如果mems为None，则直接返回None
        if mems is None:
            return None

        # 断言hids和mems的长度相等
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # 计算新的记忆长度范围
        new_mems = []
        end_idx = mlen + tf.math.maximum(0, qlen)
        beg_idx = tf.math.maximum(0, end_idx - tf.convert_to_tensor(self.mem_len))
        for i in range(len(hids)):
            # 将mems[i]转换为与hids[i]相同的数据类型
            mems[i] = tf.cast(mems[i], dtype=hids[i].dtype)
            # 将hids[i]与mems[i]拼接在一起
            cat = tf.concat([mems[i], hids[i]], axis=0)
            # 停止梯度计算拼接的结果cat
            tf.stop_gradient(cat)
            # 将拼接后的结果按照计算得到的索引范围截取，并加入到new_mems列表中
            new_mems.append(cat[beg_idx:end_idx])

        return new_mems

    # 调用模型，接受多种输入参数，使用装饰器unpack_inputs来解包输入
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        mems: List[tf.Tensor] | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
# TFTransfoXLPreTrainedModel 类的定义，继承自 TFPreTrainedModel，用于处理权重初始化以及预训练模型的下载和加载接口。
class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 TransfoXLConfig
    config_class = TransfoXLConfig
    # 基础模型的前缀名为 "transformer"
    base_model_prefix = "transformer"


# 使用 dataclass 装饰器定义 TFTransfoXLModelOutput 类
@dataclass
class TFTransfoXLModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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

    # 最终隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)` 的 tf.Tensor
    last_hidden_state: tf.Tensor = None
    # mems 是一个长度为 `config.n_layers` 的 List，包含预先计算的隐藏状态（在注意力块中的键和值），用于加速顺序解码。
    mems: List[tf.Tensor] = None
    # hidden_states 是一个可选的元组，当 `output_hidden_states=True` 时返回，或者 `config.output_hidden_states=True` 时返回。
    # 包含每层模型输出的 tf.Tensor（嵌入层输出和每层输出各一个），形状为 `(batch_size, sequence_length, hidden_size)`。
    hidden_states: Tuple[tf.Tensor] | None = None
    # attentions 是一个可选的元组，当 `output_attentions=True` 时返回，或者 `config.output_attentions=True` 时返回。
    # 包含每层注意力权重的 tf.Tensor（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFTransfoXLLMHeadModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """
    """
    Args:
        losses (`tf.Tensor` of shape *(batch_size, sequence_length-1)*, *optional*, returned when `labels` is provided):
            Language modeling losses (not reduced).
            语言建模的损失（未归并），形状为 *(batch_size, sequence_length-1)*，在提供 `labels` 时返回。
        prediction_scores (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
            语言建模头部的预测分数，形状为 `(batch_size, sequence_length, config.vocab_size)`，
            表示每个词汇标记的预测分数（经过 SoftMax 后的分数）。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
            包含预先计算的隐藏状态（注意力块中的键和值）。长度为 `config.n_layers` 的列表，
            可用于加速序列解码。已经计算过的 token id 不应该作为输入传递给模型。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每一层输出的隐藏状态，以及初始嵌入输出的元组。当设置 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均。当设置 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    """

    prediction_scores: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
@dataclass
class TFTransfoXLSequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            be passed as input ids as they have already been computed.
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

    loss: tf.Tensor | None = None  # 损失值，如果提供了 `labels` 参数，则返回（可选）
    logits: tf.Tensor = None  # 分类（或回归，如果 `config.num_labels==1`）得分，未经 SoftMax 处理前的张量
    mems: List[tf.Tensor] = None  # 长度为 `config.n_layers` 的张量列表，包含预先计算的隐藏状态（注意力块中的键和值）
                                  # 可以用于加速顺序解码
    hidden_states: Tuple[tf.Tensor] | None = None  # 可选，当 `output_hidden_states=True` 时返回，模型在每个层的输出和初始嵌入输出的元组
                                                  # 形状为 `(batch_size, sequence_length, hidden_size)`
    attentions: Tuple[tf.Tensor] | None = None  # 可选，当 `output_attentions=True` 时返回，注意力 softmax 后的注意力权重
                                               # 用于计算自注意力头中的加权平均值



TRANSFO_XL_START_DOCSTRING = r"""

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
    # 在使用 `model.fit()` 支持的任何格式传递输入和标签！然而，如果你想在 Keras 方法之外使用第二种格式，比如在使用 Keras 的 `Functional` API 创建自定义层或模型时，可以使用以下三种方法来收集所有输入张量到第一个位置参数中：
    
    # - 仅包含 `input_ids` 的单个张量：`model(input_ids)`
    # - 包含不同长度列表，按照文档字符串中给定的顺序包含一个或多个输入张量：`model([input_ids, attention_mask])` 或 `model([input_ids, attention_mask, token_type_ids])`
    # - 包含一个或多个输入张量，并与文档字符串中给定的输入名称关联的字典：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
    
    # 注意，当使用 [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) 创建模型和层时，不需要担心这些问题，因为可以像传递到任何其他 Python 函数一样传递输入！
    
    # Parameters:
    #     config ([`TransfoXLConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型关联的权重，只加载配置。可以查看 [`~PreTrainedModel.from_pretrained`] 方法加载模型权重。
"""

TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as `input_ids` as they have already been computed.
        head_mask (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLModel(TFTransfoXLPreTrainedModel):
    """
    TFTransfoXLModel 类的定义，继承自 TFTransfoXLPreTrainedModel 类。

    使用 @add_start_docstrings 装饰器添加了类的文档字符串，说明此类是一个不带顶层头的原始 TransfoXL 模型。

    """
    # 初始化方法，用于创建一个新的TransfoXL模型实例
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法，传递配置和其他可选输入
        super().__init__(config, *inputs, **kwargs)
        # 创建一个TransfoXL的主层实例，命名为"transformer"
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")

    # 将函数输入进行解包，以便在调用模型前添加文档字符串描述
    @unpack_inputs
    # 添加模型前向传播的文档字符串描述，使用TRANSFO_XL_INPUTS_DOCSTRING作为参数
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串描述，包括checkpoint、output_type、config_class等参数
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的调用方法，接受多个输入参数并返回输出结果
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        mems: List[tf.Tensor] | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
    ) -> TFTransfoXLModelOutput | Tuple[tf.Tensor]:
        # 调用TransfoXL主层的前向传播方法，传递输入参数并接收输出
        outputs = self.transformer(
            input_ids=input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型的输出结果
        return outputs
"""
Transformer-XL 模型，在顶部有一个语言建模头部（自适应 softmax，其权重与自适应输入嵌入层相结合）。
"""
@add_start_docstrings(
    """
    Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化 Transformer-XL 主层，并命名为 "transformer"
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")
        # 是否使用采样 softmax
        self.sample_softmax = config.sample_softmax
        # 断言确保 sample_softmax 小于等于 0，因为采样 softmax 的实现尚未完成
        assert self.sample_softmax <= 0, (
            "Sampling from the softmax is not implemented yet. Please look at issue: #3310:"
            " https://github.com/huggingface/transformers/issues/3310"
        )

        # 创建自适应 softmax，使用 TFAdaptiveSoftmaxMask 类
        self.crit = TFAdaptiveSoftmaxMask(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, name="crit"
        )

    # 重置 token embeddings 大小的方法，但未实现
    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError()

    # 获取输出 embeddings 的方法
    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        # 如果存在输出层，则返回最后一层的输出层
        if len(self.crit.out_layers) > 0:
            return self.crit.out_layers[-1]
        # 否则返回 None
        return None

    # 重置记忆长度的方法
    def reset_memory_length(self, mem_len):
        self.transformer.reset_memory_length(mem_len)

    # 初始化记忆的方法
    def init_mems(self, bsz):
        return self.transformer.init_mems(bsz)

    # call 方法，处理模型的前向传播
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLLMHeadModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        mems: List[tf.Tensor] | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: bool = False,
        **kwargs,
    ):
        # 在这里实现模型的前向传播
        pass  # Placeholder to indicate forward propagation implementation is expected elsewhere
    ) -> TFTransfoXLLMHeadModelOutput | Tuple[tf.Tensor]:
        # 定义函数签名，指定返回类型为 TFTransfoXLLMHeadModelOutput 或 tf.Tensor 元组
        if input_ids is not None:
            # 如果 input_ids 不为 None，则获取其形状的前两个维度大小
            bsz, tgt_len = shape_list(input_ids)[:2]
        else:
            # 如果 input_ids 为 None，则获取 inputs_embeds 的形状的前两个维度大小
            bsz, tgt_len = shape_list(inputs_embeds)[:2]

        # 使用 Transformer 模型进行前向传播计算
        transformer_outputs = self.transformer(
            input_ids,
            mems,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=training,
        )

        # 获取 Transformer 模型输出的最后一个隐藏层
        last_hidden = transformer_outputs[0]
        # 截取最后一个隐藏层的后 tgt_len 长度作为预测的隐藏状态
        pred_hid = last_hidden[:, -tgt_len:]

        # 使用 crit 对预测的隐藏状态进行 softmax 计算，得到预测分数
        softmax_output = self.crit(pred_hid, labels, training=training)
        # 如果 labels 为 None，则返回 softmax_output 作为预测分数
        prediction_scores = softmax_output if labels is None else ()

        # 如果不要求返回字典，则返回预测分数和 transformer_outputs 的其它部分
        if not return_dict:
            return (prediction_scores,) + transformer_outputs[1:]

        # 返回 TFTransfoXLLMHeadModelOutput 类型的对象，包含预测分数和其它 Transformer 模型输出的相关部分
        return TFTransfoXLLMHeadModelOutput(
            prediction_scores=prediction_scores,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **model_kwargs):
        # 准备输入用于生成过程
        inputs = {}

        # 如果 past_key_values 不为 None，则使用 input_ids 的最后一个位置的扩展维度作为输入
        if past_key_values:
            input_ids = tf.expand_dims(input_ids[:, -1], axis=-1)
        else:
            # 否则，直接使用原始的 input_ids
            input_ids = input_ids

        return inputs

    # 根据 torch 的 tie_weights 函数进行调整
    def tf_to_pt_weight_rename(self, tf_weight):
        # 如果配置中要求绑定词嵌入且在 tf_weight 中包含 "crit.out_layers"
        if self.config.tie_word_embeddings and "crit.out_layers" in tf_weight:
            # 返回 tf_weight，并将 "crit.out_layers" 替换为 "transformer.word_emb.emb_layers"
            return tf_weight, tf_weight.replace("crit.out_layers", "transformer.word_emb.emb_layers")
        # 如果配置中要求绑定投影且在 tf_weight 中包含 "crit.out_projs"
        elif self.config.tie_projs and "crit.out_projs" in tf_weight:
            for i, tie_proj in enumerate(self.config.tie_projs):
                # 如果 tie_proj 为真且配置参数符合要求，则替换相应的 tf_weight 部分
                if tie_proj and self.config.div_val == 1 and self.config.d_model != self.config.d_embed:
                    return tf_weight, tf_weight.replace(f"crit.out_projs.{i}", "transformer.word_emb.emb_projs.0")
                # 如果 tie_proj 为真且配置参数符合要求，则替换相应的 tf_weight 部分
                elif tie_proj and self.config.div_val != 1:
                    return tf_weight, tf_weight.replace("crit.out_projs", "transformer.word_emb.emb_projs")
        else:
            # 如果不满足以上条件，则返回原始的 tf_weight
            return (tf_weight,)
"""
The Transfo XL Model transformer with a sequence classification head on top (linear layer).

[`TFTransfoXLForSequenceClassification`] uses the last token in order to do the classification, as other causal
models (e.g. GPT-1,GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
"""
@add_start_docstrings(
    """
    Decorator to add docstrings to the model's constructor (__init__ method) for `TFTransfoXLForSequenceClassification`.

    Args:
        config (:class:`~transformers.TransfoXLConfig`):
            The configuration class to instantiate the model with.

    This initializes the sequence classifier by setting the number of labels and creating a Dense layer for scoring,
    and instantiates the main Transformer layer (`TFTransfoXLMainLayer`).

    This model supports sequence classification tasks based on the transformer's last token.
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLForSequenceClassification(TFTransfoXLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        # Define a Dense layer (`score`) for predicting sequence classifications
        self.score = keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.init_range),
            name="score",
            use_bias=False,
        )
        # Initialize the main Transformer layer (`transformer`) for sequence processing
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")

    def get_output_embeddings(self):
        """
        Retrieve the output embeddings from the transformer's word embeddings.

        This method warns that sequence classification models do not have output embeddings and that
        `.get_output_embeddings` will be removed in future versions of transformers.
        """
        logger.warning(
            "Sequence classification models do not have output embeddings. `.get_output_embeddings` will be removed "
            "in transformers v4.32."
        )
        # Return the word embeddings from the transformer layer
        return self.transformer.word_emb

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLSequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        mems: List[tf.Tensor] | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        **kwargs,
    ):
        """
        Perform the forward pass of the TFTransfoXLForSequenceClassification model.

        This method processes inputs through the Transformer XL model and computes logits for sequence classification.

        Args:
            input_ids (:obj:`tf.Tensor` or :obj:`np.ndarray`, `optional`):
                The input IDs of shape `[batch_size, sequence_length]`.
            mems (:obj:`List[tf.Tensor]` or :obj:`None`, `optional`):
                List of memory states from previous batches to speed up sequential decoding.
            head_mask (:obj:`np.ndarray` or :obj:`tf.Tensor` or :obj:`None`, `optional`):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (:obj:`np.ndarray` or :obj:`tf.Tensor` or :obj:`None`, `optional`):
                Instead of input IDs, directly pass embeddings. Shape should be `[batch_size, sequence_length, hidden_size]`.
            output_attentions (:obj:`bool`, `optional`):
                Whether to return attention weights.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether to return hidden states.
            return_dict (:obj:`bool`, `optional`):
                Whether to return outputs as a dictionary instead of tuple.
            labels (:obj:`np.ndarray` or :obj:`tf.Tensor` or :obj:`None`, `optional`):
                Labels for computing the sequence classification loss. Shape should be `[batch_size]`.
            training (:obj:`bool`, `optional`):
                Whether to run in training mode. Defaults to `False`.

        Returns:
            :obj:`Union[TFTransfoXLSequenceClassifierOutput, Tuple]`:
                The sequence classifier output, which includes logits, hidden states, and/or attention weights,
                depending on the configuration and optional outputs.

        """
        ) -> Union[Tuple, TFTransfoXLSequenceClassifierOutputWithPast]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 调用 Transformer 模型进行前向传播，并返回输出结果
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取 Transformer 输出的隐藏状态
        hidden_states = transformer_outputs[0]
        
        # 使用得分函数对隐藏状态进行分类得到预测的 logits
        logits = self.score(hidden_states)
        
        # 初始化用于选择 logits 的变量
        in_logits = None
        
        # 如果没有定义填充标记，则将序列长度设置为 -1
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 如果输入了 input_ids，则计算每个样本的序列长度
            if input_ids is not None:
                # 计算每个样本的实际序列长度
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                # 使用序列长度从 logits 中选择相应的 logits
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            else:
                # 如果没有输入 input_ids，则警告并设置序列长度为 -1
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        
        # 初始化损失变量
        loss = None
        
        # 如果有提供标签，则计算损失
        if labels is not None:
            # 如果有输入 input_ids，则获取 batch_size 和 sequence_length
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            
            # 如果没有填充标记，并且 batch_size 大于 1，则会报错
            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            # 如果序列长度不是 Tensor，则从 logits 中选择相应的 logits
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            # 计算损失
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))

        # 如果没有 return_dict，则返回非字典形式的输出
        pooled_logits = in_logits if in_logits is not None else logits

        if not return_dict:
            # 如果不返回字典，则返回元组形式的输出
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTransfoXLSequenceClassifierOutputWithPast 类型的字典形式输出
        return TFTransfoXLSequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```