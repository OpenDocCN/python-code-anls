# `.\models\deprecated\transfo_xl\modeling_tf_transfo_xl.py`

```py
# 设置文件编码格式为 utf-8
# 版权声明
# 引入必要的库和模块
# TF 2.0 Transformer XL model.
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
# 获取 logger
logger = logging.get_logger(__name__)
# 定义常量
_CHECKPOINT_FOR_DOC = "transfo-xl-wt103"
_CONFIG_FOR_DOC = "TransfoXLConfig"
# 预训练模型列表
TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "transfo-xl-wt103",
    # See all Transformer XL models at https://huggingface.co/models?filter=transfo-xl
]

# 自定义位置嵌入层
class TFPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, demb, **kwargs):
        # 初始化方法
        super().__init__(**kwargs)
        # 初始化频率
        self.inv_freq = 1 / (10000 ** (tf.range(0, demb, 2.0) / demb))

    # 前向传播方法
    def call(self, pos_seq, bsz=None):
        # 将频率转换为输入序列的数据类型
        self.inv_freq = tf.cast(self.inv_freq, dtype=pos_seq.dtype)
        # 计算正弦和余弦输入
        sinusoid_inp = tf.einsum("i,j->ij", pos_seq, self.inv_freq)
        # 计算位置嵌入
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
        if bsz is not None:
            # 如果有 batch size，则将位置嵌入进行扩展
            return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
        else:
            # 否则直接返回位置嵌入
            return pos_emb[:, None, :]

# 自定义位置前馈全连接层
class TFPositionwiseFF(tf.keras.layers.Layer):
    # 初始化函数，设置模型参数，包括输入维度、隐藏层维度、dropout概率等
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, layer_norm_epsilon=1e-5, init_std=0.02, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
    
        # 设置模型参数
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
    
        # 创建第一个全连接层，设置神经元数量、初始化方法、激活函数和层名称
        self.layer_1 = tf.keras.layers.Dense(
            d_inner, kernel_initializer=get_initializer(init_std), activation=tf.nn.relu, name="CoreNet_._0"
        )
        # 创建第一个 dropout 层
        self.drop_1 = tf.keras.layers.Dropout(dropout)
        # 创建第二个全连接层，设置神经元数量、初始化方法和层名称
        self.layer_2 = tf.keras.layers.Dense(d_model, kernel_initializer=get_initializer(init_std), name="CoreNet_._3")
        # 创建第二个 dropout 层
        self.drop_2 = tf.keras.layers.Dropout(dropout)
    
        # 创建 LayerNormalization 层，设置 epsilon 和层名称
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layer_norm")
    
        # 设置是否在 layer normalization 之前应用
        self.pre_lnorm = pre_lnorm
    
    # 模型的前向传播函数
    def call(self, inp, training=False):
        # 如果在 layer normalization 之前应用
        if self.pre_lnorm:
            # 应用 layer normalization 和位置前馈网络
            core_out = self.layer_norm(inp)
            core_out = self.layer_1(core_out)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)
    
            # 残差连接
            output = core_out + inp
        else:
            # 应用位置前馈网络
            core_out = self.layer_1(inp)
            core_out = self.drop_1(core_out, training=training)
            core_out = self.layer_2(core_out)
            core_out = self.drop_2(core_out, training=training)
    
            # 残差连接和 layer normalization
            output = self.layer_norm(inp + core_out)
    
        return output
# 定义一个名为 TFRelPartialLearnableMultiHeadAttn 的自定义 Layer 类
class TFRelPartialLearnableMultiHeadAttn(tf.keras.layers.Layer):
    # 初始化函数，设置各种参数和属性
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0.0,
        pre_lnorm=False,
        r_r_bias=None,
        r_w_bias=None,
        layer_norm_epsilon=1e-5,
        init_std=0.02,
        output_attentions=False,
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置各个参数的初始数值
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.output_attentions = output_attentions

        # 创建一个 Dense 层，用于计算查询、键、值
        self.qkv_net = tf.keras.layers.Dense(
            3 * n_head * d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name="qkv_net"
        )

        # 定义一个 Dropout 层，用于计算 drop 操作
        self.drop = tf.keras.layers.Dropout(dropout)
        
        # 定义一个 Dropout 层，用于计算 dropatt 操作
        self.dropatt = tf.keras.layers.Dropout(dropatt)
        
        # 创建一个 Dense 层，用于计算输出
        self.o_net = tf.keras.layers.Dense(
            d_model, kernel_initializer=get_initializer(init_std), use_bias=False, name="o_net"
        )

        # 创建一个 LayerNormalization 层，用于对输出进行归一化
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name="layer_norm")

        # 定义一个缩放因子，用于缩放计算出的注意力分数
        self.scale = 1 / (d_head**0.5)

        # 设置是否进行预 LayerNormalization 操作的标识
        self.pre_lnorm = pre_lnorm

        # 如果给定了 r_r_bias 和 r_w_bias，则共享存在，否则为 None
        if r_r_bias is not None and r_w_bias is not None:  # Biases are shared
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = None
            self.r_w_bias = None

        # 创建一个 Dense 层，用于计算 r_net
        self.r_net = tf.keras.layers.Dense(
            self.n_head * self.d_head, kernel_initializer=get_initializer(init_std), use_bias=False, name="r_net"
        )

    # 构建函数，用于构建 Layer 的参数
    def build(self, input_shape):
        # 如果 r_r_bias 和 r_w_bias 为 None，则创建参数并加入到 Layer
        if self.r_r_bias is None or self.r_w_bias is None:  # Biases are not shared
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
        # 调用父类的构建函数
        super().build(input_shape)

    # 定义一个方法 _rel_shift，用于实现相对位置编码
    def _rel_shift(self, x):
        x_size = shape_list(x)

        # 在 x 的第二个维度上进行填充
        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        # 重新调整张量的形状
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        # 进行切片操作，去除填充的部分
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        # 重新调整张量的形状
        x = tf.reshape(x, x_size)

        return x


注释太多了，超过了字符限制，请自行完成。
    # 定义 Transformer 的 Decoder 层，继承自 TransformerDecoderLayer
    def __init__(
        self, 
        n_head, 
        d_model, 
        d_head, 
        dropout, 
        dropatt=0, 
        pre_lnorm=False, 
        r_w_bias=True, 
        r_r_bias=True, 
        init_std=0.02, 
        layer_norm_epsilon=1e-5, 
        output_attentions=False, 
        **kwargs
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 定义 Decoder 层的注意力机制
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
        
        # 定义 Decoder 层的前馈神经网络
        self.pos_ff = TFPositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm=pre_lnorm,
            init_std=init_std,
            layer_norm_epsilon=layer_norm_epsilon,
            name="pos_ff",
        )

    # 定义 Decoder 层的前向传播逻辑
    def call(self, dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=False):
        # 使用注意力机制处理输入
        attn_outputs = self.dec_attn(dec_inp, r, dec_attn_mask, mems, head_mask, output_attentions, training=training)
        
        # 使用前馈神经网络处理注意力机制的输出
        ff_output = self.pos_ff(attn_outputs[0], training=training)

        # 返回前馈神经网络的输出和注意力机制的附加信息
        outputs = [ff_output] + attn_outputs[1:]

        return outputs
# 定义一个继承自 tf.keras.layers.Layer 的 TFTransfoEmbeddings 类
class TFTransfoEmbeddings(tf.keras.layers.Layer):
    # 初始化函数，接收参数：词汇大小，嵌入维度，初始化标准差和其他参数
    def __init__(self, vocab_size, emb_size, init_std, **kwargs):
        super().__init__(**kwargs)

        # 将输入的参数保存为成员变量
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.init_std = init_std

    # 在模型构建过程中构建层的权重
    def build(self, input_shape):
        # 添加权重变量到层中，shape 为 (vocab_size, emb_size)
        self.weight = self.add_weight(
            shape=(self.vocab_size, self.emb_size),
            initializer=get_initializer(self.init_std),
            name="embeddings",
        )

        super().build(input_shape)

    # 执行层的前向传播操作
    def call(self, inputs):
        # 根据 inputs 从权重中获取对应的嵌入向量并返回
        return tf.gather(self.weight, inputs)


# 定义一个继承自 tf.keras.layers.Layer 的 TFAdaptiveEmbedding 类
class TFAdaptiveEmbedding(tf.keras.layers.Layer):
    # 初始化函数，接收参数：token 数量，嵌入维度，投影维度，cutoffs，除数，初始化标准差等
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, init_std=0.02, sample_softmax=False, **kwargs):
        super().__init__(**kwargs)

        # 将输入的参数保存为成员变量
        self.n_token = n_token
        self.d_embed = d_embed
        self.init_std = init_std

        # 根据 cutoffs 和 n_token 构建 cutoff_ends，用于后续计算
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj**0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = []

        # 如果 div_val 为 1，抛出 NotImplementedError 异常
        if div_val == 1:
            raise NotImplementedError  # Removed these to avoid maintaining dead code - They are not used in our pretrained checkpoint
        else:
            # 根据 cutoffs 构建不同的 TFTransfoEmbeddings 层
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(
                    TFTransfoEmbeddings(
                        r_idx - l_idx,
                        d_emb_i,
                        init_std,
                        name=f"emb_layers_._{i}",
                    )
                )

    # 在模型构建过程中构建层的权重
    def build(self, input_shape):
        self.emb_projs = []

        # 根据 emb_layers 和 d_proj 构建不同的权重
        for i in range(len(self.cutoffs)):
            d_emb_i = self.d_embed // (self.div_val**i)
            self.emb_projs.append(
                self.add_weight(
                    shape=(d_emb_i, self.d_proj),
                    initializer=get_initializer(self.init_std),
                    trainable=True,
                    name=f"emb_projs_._{i}",
                )
            )

        super().build(input_shape)
    # 定义一个方法用于对输入进行处理并返回嵌入向量
    def call(self, inp):
        # 如果除以的值为1，则抛出未实现错误
        if self.div_val == 1:
            raise NotImplementedError  # 已删除这些内容以避免维护死代码 - 它们在我们的预训练检查点中没有使用
        else:
            # 将输入展平为一维数组
            inp_flat = tf.reshape(inp, (-1,))
            # 创建一个全零张量作为嵌入向量的初始值
            emb_flat = tf.zeros([shape_list(inp_flat)[0], self.d_proj])
            # 遍历每一个截断点
            for i in range(len(self.cutoffs)):
                # 获取当前截断点的左右边界
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                # 根据当前截断点的范围创建一个布尔掩码
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)

                # 根据掩码从输入中提取对应部分
                inp_i = tf.boolean_mask(inp_flat, mask_i) - l_idx
                # 通过嵌入层获取嵌入向量
                emb_i = self.emb_layers[i](inp_i)
                # 利用嵌入投影将嵌入向量投影到指定维度
                emb_i = tf.einsum("id,de->ie", emb_i, self.emb_projs[i])

                # 根据掩码和嵌入向量的位置进行散列操作并更新嵌入向量
                mask_idx = tf.where(mask_i)
                scatter = tf.scatter_nd(mask_idx, emb_i, shape_list(emb_flat))
                emb_flat = tf.cast(emb_flat, dtype=scatter.dtype)
                emb_flat += scatter

            # 将平坦的嵌入向量重新整形为与输入相同的形状，并乘以嵌入缩放比例
            embed_shape = shape_list(inp) + [self.d_proj]
            embed = tf.reshape(emb_flat, embed_shape)

        # 最终的嵌入向量乘以嵌入缩放参数
        embed *= self.emb_scale

        # 返回嵌入结果
        return embed
# 定义 TFTransfoXLMainLayer 类，并添加 keras 序列化装饰器
@keras_serializable
class TFTransfoXLMainLayer(tf.keras.layers.Layer):
    # 将 TransfoXLConfig 类赋值给 config_class
    config_class = TransfoXLConfig

    # 初始化方法，接收 config 参数和其他关键字参数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 将 config 参数赋值给 self.config
        self.config = config
        # 根据 config 的配置设置输出隐藏层、输出注意力和返回字典结果等属性
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.use_return_dict

        # 设置词汇表大小、嵌入维度、模型维度、头数、头维度和是否解开 r
        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.untie_r = config.untie_r

        # 创建 TFAdaptiveEmbedding 层用于词嵌入
        self.word_emb = TFAdaptiveEmbedding(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            div_val=config.div_val,
            init_std=config.init_std,
            name="word_emb",
        )

        # 添加 Dropout 层
        self.drop = tf.keras.layers.Dropout(config.dropout)

        # 设置层数、记忆长度和注意力类型
        self.n_layer = config.n_layer
        self.mem_len = config.mem_len
        self.attn_type = config.attn_type

        # 初始化层列表
        self.layers = []
        if config.attn_type == 0:  # 默认注意力类型
            # 根据层数添加 TFRelPartialLearnableDecoderLayer 层到 layers 列表中
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
        else:  # learnable embeddings 和 absolute embeddings
            raise NotImplementedError  # 移除这部分代码以避免维护死代码 - 我们的预训练检查点中未使用

        # 设置是否相同长度和截断长度
        self.same_length = config.same_length
        self.clamp_len = config.clamp_len

        if self.attn_type == 0:  # 默认注意力类型
            # 创建 TFPositionalEmbedding 层用于位置编码
            self.pos_emb = TFPositionalEmbedding(self.d_model, name="pos_emb")
        else:  # learnable embeddings 和 absolute embeddings
            raise NotImplementedError  # 移除这部分代码以避免维护死代码 - 我们的预训练检查点中未使用
    # 定义一个方法，用于构建模型
    def build(self, input_shape):
        # 如果不是解开 r，那么添加 r_w_bias 权重，并初始化为零
        if not self.untie_r:
            self.r_w_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_w_bias"
            )
            # 添加 r_r_bias 权重，并初始化为零
            self.r_r_bias = self.add_weight(
                shape=(self.n_head, self.d_head), initializer="zeros", trainable=True, name="r_r_bias"
            )
        # 调用父类的 build 方法
        super().build(input_shape)

    # 获取输入的嵌入
    def get_input_embeddings(self):
        return self.word_emb

    # 设置输入的嵌入，抛出未实现的错误
    def set_input_embeddings(self, value):
        raise NotImplementedError

    # 向后兼容处理
    def backward_compatible(self):
        self.sample_softmax = -1

    # 重置记忆长度
    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    # 对头部进行修剪
    def _prune_heads(self, heads):
        raise NotImplementedError

    # 初始化记忆
    def init_mems(self, bsz):
        # 如果记忆长度大于 0
        if self.mem_len > 0:
            mems = []
            # 遍历每一层，生成全零的记忆
            for i in range(self.n_layer):
                empty = tf.zeros([self.mem_len, bsz, self.d_model])
                mems.append(empty)
            return mems
        else:
            return None

    # 更新记忆
    def _update_mems(self, hids, mems, mlen, qlen):
        # 不处理 None
        if mems is None:
            return None

        # mems 不是 None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # 可以缓存到 mems 的步长
        new_mems = []
        end_idx = mlen + tf.math.maximum(0, qlen)
        beg_idx = tf.math.maximum(0, end_idx - tf.convert_to_tensor(self.mem_len))
        for i in range(len(hids)):
            # 转换类型
            mems[i] = tf.cast(mems[i], dtype=hids[i].dtype)
            # 拼接隐藏状态和记忆
            cat = tf.concat([mems[i], hids[i]], axis=0)
            # 停止梯度传播
            tf.stop_gradient(cat)
            new_mems.append(cat[beg_idx:end_idx])

        return new_mems

    # 模型的调用方法
    @unpack_inputs
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
class TFTransfoXLPreTrainedModel(TFPreTrainedModel):
    """
    一个处理权重初始化和一个简单接口以下载和加载预训练模型的抽象类。
    """

    # 指定配置类
    config_class = TransfoXLConfig
    # 模型的基本前缀
    base_model_prefix = "transformer"


@dataclass
class TFTransfoXLModelOutput(ModelOutput):
    """
    模型输出的基类，可能还包含一个过去的键/值（用于加速序列解码）。

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态的序列。
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含预计算的隐藏状态（注意力块中的键和值）。可用于加速序列解码。已经计算过过去的令牌 ID 不应该作为输入 ID 传递给该模型。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `tf.Tensor` 的元组（一个用于嵌入的输出 + 一个用于每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层的输出隐藏状态加上初始嵌入输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `tf.Tensor` 的元组（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    last_hidden_state: tf.Tensor = None
    mems: List[tf.Tensor] = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFTransfoXLLMHeadModelOutput(ModelOutput):
    """
    模型输出的基类，可能还包含一个过去的键/值（用于加速序列解码）。
    """
    # losses 参数是一个 tf.Tensor 类型的张量，形状为 (batch_size, sequence_length-1)，可选，仅当提供 labels 参数时返回
    # 它代表语言建模损失（未缩减的）
    Args:
        losses (`tf.Tensor` of shape *(batch_size, sequence_length-1)*, *optional*, returned when `labels` is provided):
            # 是语言模型头部预测分数，形状为 (batch_size, sequence_length, config.vocab_size)
            Language modeling losses (not reduced).
        
        # prediction_scores 参数是一个 tf.Tensor 类型的张量，形状为 (batch_size, sequence_length, config.vocab_size)
        # 它代表语言模型头部的预测分数（经过 SoftMax 后的每个词汇的分数）
        prediction_scores (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            # 预测分数的语言模型头部（SoftMax 后每个词汇的分数）
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        
        # mems 参数是一个包含多个 tf.Tensor 的列表，长度为 config.n_layers
        # 它包含预先计算的隐藏状态（注意力块中的键和值），可以用于加速序列解码
        # 应用模型时不应将它们传递为输入 ID，因为这些 ID 已经被计算过
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            # 包含预计算的隐藏状态（注意力块中的键和值）。可以用于加速序列解码
            Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems`
            # 当给定 `mems` 输入时，可用于加速序列解码
            input) to speed up sequential decoding. The token ids which have their past given to this model should not
            # 已经给出过去的 token ids 不应传递为输入 id，因为它们已经被计算过
            be passed as input ids as they have already been computed.
        
        # hidden_states 参数是一个包含 tf.Tensor 的元组，可选，仅当传递 output_hidden_states=True 或 config.output_hidden_states=True 时返回
        # 它是每个层输出的隐藏状态，加上初始嵌入输出
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 是包含 tf.Tensor 的元组，一个用于嵌入层输出，一个用于每层输出，形状为 (batch_size, sequence_length, hidden_size)
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            # 模型在每一层的输出加上初始嵌入层的隐藏状态
            `(batch_size, sequence_length, hidden_size)`.

            # 模型在每一层的输出加上初始嵌入层的隐藏状态
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        
        # attentions 参数是一个包含 tf.Tensor 的元组，可选，仅当传递 output_attentions=True 或 config.output_attentions=True 时返回
        # 它是每一层的注意力权重，形状为 (batch_size, num_heads, sequence_length, sequence_length)
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 是包含 tf.Tensor 的元组，分别对应每一层注意力层，形状为 (batch_size, num_heads, sequence_length, sequence_length)
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            # 注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值
            sequence_length)`.

            # 注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            # 自注意力头中的加权平均值
            heads.

    # 初始化 prediction_scores 参数，默认值为 None，类型为 tf.Tensor
    prediction_scores: tf.Tensor = None
    # 初始化 mems 参数，默认值为 None，类型为包含 tf.Tensor 的列表
    mems: List[tf.Tensor] = None
    # 初始化 hidden_states 参数，默认值为 None，类型为包含 tf.Tensor 的元组
    hidden_states: Tuple[tf.Tensor] | None = None
    # 初始化 attentions 参数，默认值为 None，类型为包含 tf.Tensor 的元组
    attentions: Tuple[tf.Tensor] | None = None
# 这是一个用于序列分类模型输出的基类

@dataclass
# 使用 dataclass 装饰器，这是一个 Python 3.7 引入的新特性，可以自动生成类的一些方法，如 __init__、__repr__ 等
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

    # 分类 (或回归) 的损失
    loss: tf.Tensor | None = None
    # 分类 (或回归) 的分数（SoftMax 之前）
    logits: tf.Tensor = None
    # 存储预计算的隐藏状态（注意力块中的键和值），可用于加速序列解码
    mems: List[tf.Tensor] = None
    # 模型在每个层的隐藏状态的元组，以及初始嵌入输出
    hidden_states: Tuple[tf.Tensor] | None = None
    # 在每个层之后的注意力权重的元组，用于计算自注意力头中的加权平均值
    attentions: Tuple[tf.Tensor] | None = None


TRANSFO_XL_START_DOCSTRING = r"""

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
"""
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
        config ([`TransfoXLConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TRANSFO_XL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。

            可以使用 [`AutoTokenizer`] 来获取这些索引。详见 [`PreTrainedTokenizer.__call__`] 和 [`PreTrainedTokenizer.encode`]。

            [什么是输入 ID?](../glossary#input-ids)
        mems (`List[tf.Tensor]` of length `config.n_layers`):
            包含模型计算的预先计算的隐藏状态（注意力模块中的键和值）。可用于加速顺序解码。这个模型的输入中不应包含已经给出其记忆的令牌 ID。
        head_mask (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于将自注意力模块的特定头部置零的遮罩。遮罩值取在 `[0, 1]`：

            - 1 表示该头部 **未被遮罩**，
            - 0 表示该头部 **被遮罩**。
        inputs_embeds (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            可选，您可以选择直接传递嵌入表示，而不是传递 `input_ids`。如果您想要更多地控制如何将 `input_ids` 索引转换为关联向量，而不是使用模型的内部嵌入查找矩阵，则这将非常有用。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量中的 `attentions`。此参数仅可在 eager 模式下使用，在图模式中，将使用配置中的值。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量中的 `hidden_states`。此参数仅可在 eager 模式下使用，在图模式中，将使用配置中的值。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。此参数可以在 eager 模式下使用，在图模式中，值将始终设置为 True。
        training (`bool`, *optional*, defaults to `False`):
            是否在训练模式下使用模型（某些模块，如 dropout 模块，在训练和评估之间具有不同的行为）。
"""


@add_start_docstrings(
    "不带特定顶层的原始 Bert 模型变压器输出的输出。",
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLModel(TFTransfoXLPreTrainedModel):
    # 初始化方法，用于创建一个新的TFTransfoXLMainLayer对象，并将其存储在self.transformer属性中
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)
        # 创建一个TFTransfoXLMainLayer对象，并命名为"transformer"，存储在self.transformer属性中
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")

    # 使用装饰器添加注释到模型的前向传播方法上
    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTransfoXLModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法，接受一系列输入参数，并返回模型输出
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
        # 调用self.transformer对象的call方法进行前向传播
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

        # 返回模型输出
        return outputs
@add_start_docstrings(
    """
    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive
    input embeddings)
    """,
    TRANSFO_XL_START_DOCSTRING,
)
# 使用 TFTransfoXLPreTrainedModel 为基类创建 TFTransfoXLLMHeadModel 类
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):
    def __init__(self, config):
        # 调用基类的构造函数
        super().__init__(config)
        # 创建 Transformer-XL 主层对象，并命名为“transformer”
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")
        # 设置是否使用抽样 softmax，如果设置为正值则抛出异常
        self.sample_softmax = config.sample_softmax
        assert self.sample_softmax <= 0, (
            "Sampling from the softmax is not implemented yet. Please look at issue: #3310:"
            " https://github.com/huggingface/transformers/issues/3310"
        )
        # 创建自适应 softmax 对象 crit，用于语言模型的输出层
        self.crit = TFAdaptiveSoftmaxMask(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, name="crit"
        )

    # 重置 token embeddings 的尺寸
    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError()

    # 获取输出 embeddings
    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        # 检查是否使用了自适应 softmax，如果是，则返回输出层的最后一层
        if len(self.crit.out_layers) > 0:
            return self.crit.out_layers[-1]
        return None

    # 重置记忆长度
    def reset_memory_length(self, mem_len):
        # 调用 Transformer-XL 主层对象的重置记忆长度方法
        self.transformer.reset_memory_length(mem_len)

    # 初始化记忆
    def init_mems(self, bsz):
        # 调用 Transformer-XL 主层对象的初始化记忆方法
        return self.transformer.init_mems(bsz)

    # 模型调用方法，实现模型的前向传播
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
        **kwargs
    ):
        # 调用 Transformer-XL 主层对象的前向传播方法，返回结果
        return self.transformer(
            input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
            **kwargs
        )
    # 定义函数返回类型为 TFTransfoXLLMHeadModelOutput 或者 Tuple[tf.Tensor]
    def test_step(self, input_ids: tf.Tensor, mems: tf.Tensor | List[tf.Tensor] | None, labels: tf.Tensor | None = None, return_dict: bool = False, training: bool = False) -> TFTransfoXLLMHeadModelOutput | Tuple[tf.Tensor]:
        # 如果输入的 input_ids 不为 None，获取其形状的前两个维度作为 bsz 和 tgt_len
        if input_ids is not None:
            bsz, tgt_len = shape_list(input_ids)[:2]
        else:
            # 如果 input_ids 为 None，获取 inputs_embeds 的形状的前两个维度作为 bsz 和 tgt_len
            bsz, tgt_len = shape_list(inputs_embeds)[:2]

        # 运行 Transformer 模块，并获取输出结果
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

        # 获取 Transformer 模块输出的最后一个隐藏层
        last_hidden = transformer_outputs[0]
        # 从最后一个隐藏层中获取预测的隐藏态
        pred_hid = last_hidden[:, -tgt_len:]

        # 使用 crit 模块对预测的隐藏态进行 softmax 操作，得到 prediction_scores，如果 labels 为 None 则为空元组 
        softmax_output = self.crit(pred_hid, labels, training=training)
        prediction_scores = softmax_output if labels is None else ()

        # 如果 return_dict 为 False，则返回 prediction_scores 和 transformer_outputs[1:] 中的元素
        if not return_dict:
            return (prediction_scores,) + transformer_outputs[1:]

        # 否则以 TFTransfoXLLMHeadModelOutput 的形式返回预测结果
        return TFTransfoXLLMHeadModelOutput(
            prediction_scores=prediction_scores,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # 准备生成的输入，在模型的 keyword 参数中使用 past_key_values
    def prepare_inputs_for_generation(self, input_ids: tf.Tensor, past_key_values=None, **model_kwargs):
        inputs = {}

        # 如果 past_key_values 有值，则在 input_ids 中选择最后一个元素作为输入
        if past_key_values:
            input_ids = tf.expand_dims(input_ids[:, -1], axis=-1)
        else:
            # 否则使用原始的 input_ids
            input_ids = input_ids

        return inputs

    # 类中的权重重命名函数,根据配置参数进行相应的替换
    def tf_to_pt_weight_rename(self, tf_weight):
        if self.config.tie_word_embeddings and "crit.out_layers" in tf_weight:
            # 替换相应���权重名称
            return tf_weight, tf_weight.replace("crit.out_layers", "transformer.word_emb.emb_layers")
        elif self.config.tie_projs and "crit.out_projs" in tf_weight:
            for i, tie_proj in enumerate(self.config.tie_projs):
                if tie_proj and self.config.div_val == 1 and self.config.d_model != self.config.d_embed:
                    return tf_weight, tf_weight.replace(f"crit.out_projs.{i}", "transformer.word_emb.emb_projs.0")
                elif tie_proj and self.config.div_val != 1:
                    return tf_weight, tf_weight.replace("crit.out_projs", "transformer.word_emb.emb_projs")
        else:
            return (tf_weight,)
@add_start_docstrings(
    """
    The Transfo XL Model transformer with a sequence classification head on top (linear layer).

    [`TFTransfoXLForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-1,GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    TRANSFO_XL_START_DOCSTRING,
)
class TFTransfoXLForSequenceClassification(TFTransfoXLPreTrainedModel, TFSequenceClassificationLoss):
    # 初始化函数，接受配置和其他参数，设置模型的标签数量，初始化线性层和转换器
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.score = tf.keras.layers.Dense(
            config.num_labels,
            kernel_initializer=get_initializer(config.init_range),
            name="score",
            use_bias=False,
        )
        self.transformer = TFTransfoXLMainLayer(config, name="transformer")

    # 获取输出嵌入信息，该方法将在transformers v4.32后移除
    def get_output_embeddings(self):
        # Remove after transformers v4.32. Fix this model's `test_model_common_attributes` test too.
        logger.warning(
            "Sequence classification models do not have output embeddings. `.get_output_embeddings` will be removed "
            "in transformers v4.32."
        )
        return self.transformer.word_emb

    # 模型调用方法，接受一系列输入参数，进行模型的前向传播
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
    ) -> Union[Tuple, TFTransfoXLSequenceClassifierOutputWithPast]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss. Indices should be in `[0, ...,
            config.vocab_size - 1]`.
        """


        # 使用 Transformer 处理输入序列，返回 Transformer 的输出
        transformer_outputs = self.transformer(
            input_ids=input_ids,  # 输入的标识符序列
            mems=mems,  # 用于记忆的 Transformer 输出
            head_mask=head_mask,  # 头部注意力屏蔽掩码
            inputs_embeds=inputs_embeds,  # 输入嵌入向量
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏层状态
            return_dict=return_dict,  # 是否返回字典格式的输出
            training=training,  # 是否处于训练模式
        )

        # 提取 Transformer 的输出中的隐藏状态
        hidden_states = transformer_outputs[0]

        # 使用 self.score 对隐藏状态进行评分
        logits = self.score(hidden_states)

        # 初始化 in_logits
        in_logits = None

        # 如果没有定义 pad_token_id，则将 sequence_lengths 设为 -1
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            # 如果 input_ids 不为空，则找到标识符序列中的 padding token，并计算序列长度
            if input_ids is not None:
                # 找到标识符序列中的 padding token，并计算序列长度
                sequence_lengths = (
                    tf.argmax(tf.cast(tf.math.equal(input_ids, self.config.pad_token_id), input_ids.dtype), axis=-1)
                    - 1
                )
                # 如果序列长度大于等于 0，则不进行处理；否则，将序列长度设为 input_ids 的长度减去 1
                sequence_lengths = tf.where(sequence_lengths >= 0, sequence_lengths, input_ids.shape[-1] - 1)
                # 从 logits 中按 sequence_lengths 的值抽取对应位置的元素，并存储到 in_logits 中
                in_logits = tf.gather(logits, sequence_lengths, batch_dims=1, axis=1)
            # 如果 input_ids 为空，则将 sequence_lengths 设为 -1，并输出警告信息
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        # 初始化 loss
        loss = None

        # 如果 labels 不为空，则计算损失函数值
        if labels is not None:
            # 如果 input_ids 不为空，则获取 batch_size 和 sequence_length
            if input_ids is not None:
                batch_size, sequence_length = shape_list(input_ids)[:2]
            # 如果 input_ids 为空，则获取 batch_size 和 sequence_length
            else:
                batch_size, sequence_length = shape_list(inputs_embeds)[:2]
            # 如果没有定义 pad_token_id，则 batch_size 必须为 1；否则，抛出异常
            assert (
                self.config.pad_token_id is not None or batch_size == 1
            ), "Cannot handle batch sizes > 1 if no padding token is defined."

            # 如果 sequence_lengths 不是 Tensor，则从 logits 中抽取 in_logits
            if not tf.is_tensor(sequence_lengths):
                in_logits = logits[0:batch_size, sequence_lengths]

            # 根据 labels 和 in_logits 计算损失函数值
            loss = self.hf_compute_loss(tf.reshape(labels, [-1, 1]), tf.reshape(in_logits, [-1, self.num_labels]))

        # 如果 in_logits 不为空，则将 pooled_logits 设为 in_logits；否则，将 pooled_logits 设为 logits
        pooled_logits = in_logits if in_logits is not None else logits

        # 如果 return_dict 为 False，则将结果以元组形式返回
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则将结果以字典形式返回
        return TFTransfoXLSequenceClassifierOutputWithPast(
```