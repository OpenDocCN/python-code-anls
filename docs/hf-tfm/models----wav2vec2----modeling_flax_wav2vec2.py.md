# `.\transformers\models\wav2vec2\modeling_flax_wav2vec2.py`

```
# 设置编码格式为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，被授权者除了按照该许可证遵守行为之外不得使用此文件
# 可以在以下网址获取许可证的一份拷贝
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按照该许可证分发的软件都是按"原样"的基础分发的
# 没有任何种类的担保或条件，无论是明示的还是暗示的，参见许可证中关于具体语言的权限以及限制
# 看许可证获取具体的语言资源和限制
""" Flax Wav2Vec2 模型。"""

# 导入必要的库
from functools import partial
from typing import Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

# 导入相关的类和方法
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import (
    ACT2FN,
    FlaxPreTrainedModel,
    append_replace_return_docstrings,
    overwrite_call_docstring,
)
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_wav2vec2 import Wav2Vec2Config

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 FlaxWav2Vec2BaseModelOutput 类，用于输出模型输出结果，包括隐藏状态和注意力机制
@flax.struct.dataclass
class FlaxWav2Vec2BaseModelOutput(ModelOutput):
    """
    FlaxWav2Vec2BaseModelOutput 的输出类型，包括潜在的隐藏状态和注意力信息。

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        extract_features (`jnp.ndarray` of shape `(batch_size, sequence_length, last_conv_dim)`):
            模型最后一层的卷积层提取的特征向量序列，`last_conv_dim` 是最后一个卷积层的维度。
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态的元组 `jnp.ndarray`（一个用于嵌入输出 + 一个用于每一层的输出），形状为
            `(batch_size, sequence_length, hidden_size)`。

            模型的每一层的隐藏状态以及初始嵌入输出。
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组 `jnp.ndarray`（每一层一个）的形状为 `(batch_size, num_heads, sequence_length,
            sequence_length)`。

            注意力权重在经过注意力 softmax 后的结果，用于在自注意力头中计算加权平均值。
    """
    # 定义变量last_hidden_state，类型为jnp.ndarray，默认值为None
    last_hidden_state: jnp.ndarray = None
    # 定义变量extract_features，类型为jnp.ndarray，默认值为None
    extract_features: jnp.ndarray = None
    # 定义变量hidden_states，类型为Tuple[jnp.ndarray]，可选类型为None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    # 定义变量attentions，类型为Tuple[jnp.ndarray]，可选类型为None
    attentions: Optional[Tuple[jnp.ndarray]] = None
# 使用装饰器定义类的一个数据结构,表示[`FlaxWav2Vec2ForPreTrainingOutput`]的输出类型，并包括潜在的隐藏状态和注意力
class FlaxWav2Vec2ForPreTrainingOutput(ModelOutput):
    # 定义全局变量的类型和默认值
    projected_states: jnp.ndarray = None
    projected_quantized_states: jnp.ndarray = None
    codevector_perplexity: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None

# 定义一个函数，用于计算给定形状的随机掩码范围。用于实现[SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)。
# 注意，此方法没有经过优化以在TPU上运行，应该作为训练期间预处理的一部分在CPU上运行。
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[np.ndarray] = None,
    min_masks: int = 0,
) -> np.ndarray:
    Args:
        shape: 用于计算蒙版的形状。
            应该是大小为2的数组，第一个元素是批处理大小，第二个是时间步长
        mask_prob:
            每个令牌被选为蒙版起始的概率。这将与时间步长数乘以蒙版跨度长度相除，以蒙版大约这个百分比的所有元素。
            但是由于重叠，实际数量会更小（除非 no_overlap 为 True）
        mask_length: 蒙版的大小
        min_masks: 最小蒙版数

    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` 必须大于 0。")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` 必须小于 `sequence_length`，但是得到了 `mask_length`：{mask_length} 和"
            f" `sequence_length`：{sequence_length}`"
        )

    # 计算批处理中蒙版的数量
    num_masked_spans = int(mask_prob * sequence_length / mask_length + np.random.rand(1).item())
    num_masked_spans = max(num_masked_spans, min_masks)

    # 确保蒙版索引数量小于等于序列长度
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment 蒙版填充
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)

    # 获取要蒙版的随机索引
    spec_aug_mask_idxs = np.array(
        [
            np.random.choice(np.arange(sequence_length - (mask_length - 1)), num_masked_spans, replace=False)
            for _ in range(batch_size)
        ]
    )

    # 将蒙版的索引扩展为蒙版跨度
    spec_aug_mask_idxs = np.broadcast_to(spec_aug_mask_idxs[:, :, None], (batch_size, num_masked_spans, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, num_masked_spans * mask_length)

    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, num_masked_spans, mask_length)).reshape(
        batch_size, num_masked_spans * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 散点索引以蒙版
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    if attention_mask is not None:
        # 确保填充的输入 id 不能被蒙版
        spec_aug_mask = np.where(attention_mask, spec_aug_mask, False)

    return spec_aug_mask
# 从特征向量中随机采样 `num_negatives` 个负向量的索引
def _sample_negative_indices(features_shape: Tuple, num_negatives: int, attention_mask: Optional[np.ndarray] = None):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    # 获取特征向量的形状信息
    batch_size, sequence_length, hidden_size = features_shape
    # 检查序列长度是否小于等于1，若是，则引发 ValueError 异常
    if sequence_length <= 1:
        raise ValueError(
            "`features should have `sequence_length` > 1, but are of shape "
            f"(batch_size, sequence_length, hidden_size) = ({batch_size, sequence_length, hidden_size})."
        )

    # 从同一句话中获取 `num_negatives` 个随机向量的索引
    sampled_negative_indices = []
    for batch_idx in range(batch_size):
        # 计算上限，若存在注意力掩码则取其和，否则取序列长度减1
        high = attention_mask[batch_idx].sum() - 1 if attention_mask is not None else sequence_length - 1
        # 从0到high之间随机采样 `num_negatives * sequence_length` 个索引
        sampled_indices_slice = np.random.randint(0, high, size=(num_negatives * sequence_length,))
        # 将采样到的索引添加到列表中
        sampled_negative_indices.append(sampled_indices_slice)

    # 转换成 numpy 数组
    sampled_negative_indices = np.asarray(sampled_negative_indices, dtype=np.int32)

    # 生成正向量本身的索引，并将其重复 `num_negatives` 次
    feature_indices = np.broadcast_to(np.arange(sequence_length)[:, None], (sequence_length, num_negatives)).flatten()

    # 避免采样相同的正向量，但保持分布均匀
    sampled_negative_indices[sampled_negative_indices >= feature_indices] += 1

    # 根据批次大小进行修正
    for batch_idx in range(1, batch_size):
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


WAV_2_VEC_2_START_DOCSTRING = r"""
    Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)


"""
    Parameters:
        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float32`):
            The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).

            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.

            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**

            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
# WAV_2_VEC_2_INPUTS_DOCSTRING 定义了函数 read_zip 的文档字符串，描述了函数的参数和用法
WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `jnp.ndarray`. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask) .. warning:: `attention_mask` should only be passed
            if the corresponding processor has `config.return_attention_mask == True`. For all models whose processor
            has `config.return_attention_mask == False`, such as
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), `attention_mask` should **not** be
            passed to avoid degraded performance when doing batched inference. For such models `input_values` should
            simply be padded with 0 and passed without `attention_mask`. Be aware that these models also yield slightly
            different results depending on whether `input_values` is padded or not.
        mask_time_indices (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义 FlaxWav2Vec2LayerNormConvLayer 类，继承自 nn.Module
class FlaxWav2Vec2LayerNormConvLayer(nn.Module):
    # 接受 Wav2Vec2Config 对象作为配置参数
    config: Wav2Vec2Config
    # 定义层的 ID，初始值为 0
    layer_id: int = 0
    # 定义变量的数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 设置方法，用于初始化各种属性
    def setup(self):
        # 如果当前层不是第一层，则输入卷积维度为配置文件中对应层的卷积维度，否则为1
        self.in_conv_dim = self.config.conv_dim[self.layer_id] if self.layer_id > 0 else 1
        # 输出卷积维度为配置文件中对应层的卷积维度
        self.out_conv_dim = self.config.conv_dim[self.layer_id]

        # 初始化卷积层
        self.conv = nn.Conv(
            features=self.config.conv_dim[self.layer_id],  # 输入特征维度为配置文件中对应层的卷积维度
            kernel_size=(self.config.conv_kernel[self.layer_id],),  # 卷积核大小为配置文件中对应层的卷积核大小
            strides=(self.config.conv_stride[self.layer_id],),  # 卷积步幅为配置文件中对应层的卷积步幅
            use_bias=self.config.conv_bias,  # 是否使用偏置，根据配置文件决定
            kernel_init=jax.nn.initializers.he_normal(),  # 卷积核初始化方式为 He 正态分布初始化
            padding="VALID",  # 不使用填充
            dtype=self.dtype,  # 数据类型
        )
        # 初始化 LayerNormalization 层
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化激活函数，根据配置文件中指定的激活函数名称从预定义字典中获取对应的激活函数
        self.activation = ACT2FN[self.config.feat_extract_activation]

    # 对象被调用时执行的方法，用于执行前向传播
    def __call__(self, hidden_states):
        # 输入经过卷积层
        hidden_states = self.conv(hidden_states)
        # 卷积结果经过 LayerNormalization 层
        hidden_states = self.layer_norm(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的结果
        return hidden_states
class FlaxConvWithWeightNorm(nn.Module):
    config: Wav2Vec2Config  # 存储模型配置信息
    dtype: jnp.dtype = jnp.float32  # 默认数据类型为 jnp.float32

    def setup(self):  # 定义模型初始化方法
        self.conv = nn.Conv(  # 创建卷积层对象
            features=self.config.hidden_size,  # 输出通道数等于隐藏尺寸
            kernel_size=(self.config.num_conv_pos_embeddings,),  # 卷积核大小为位置嵌入数量
            kernel_init=jax.nn.initializers.he_normal(),  # 卷积核参数初始化
            padding="VALID",  # 边缘填充方式为有效填充
            feature_group_count=self.config.num_conv_pos_embedding_groups,  # 分组卷积的组数
            dtype=self.dtype,  # 指定数据类型
        )
        weight_shape = (  # 计算权重形状
            self.conv.features,  # 权重数量等于输出通道数
            self.conv.features // self.conv.feature_group_count,  # 每个分组的权重数量
            self.conv.kernel_size[0],  # 卷积核的大小
        )
        self.weight_v = self.param("weight_v", jax.nn.initializers.he_normal(), weight_shape)  # 初始化权重 v
        self.weight_g = self.param("weight_g", lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :])  # 初始化权重 g
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.conv.features,))  # 初始化偏置
        self.prev_padding = self.conv.kernel_size[0] // 2  # 计算卷积前的填充大小

    def _get_normed_weights(self):  # 定义获取归一化权重的方法
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]  # 计算权重 v 的范数
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)  # 归一化权重 v
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)  # 使用权重 g 缩放权重 v
        return normed_kernel  # 返回归一化后的卷积核

    def __call__(self, hidden_states):  # 定义模型调用方法
        kernel = self._get_normed_weights()  # 获取归一化的卷积核
        hidden_states = jnp.pad(hidden_states, ((0, 0), (self.prev_padding, self.prev_padding), (0, 0)))  # 对隐藏状态进行填充
        hidden_states = self.conv.apply({"params": {"kernel": kernel.T, "bias": self.bias}}, hidden_states)  # 应用卷积操作
        return hidden_states  # 返回卷积后的隐藏状态


class FlaxWav2Vec2PositionalConvEmbedding(nn.Module):
    config: Wav2Vec2Config  # 存储模型配置信息
    dtype: jnp.dtype = jnp.float32  # 默认数据类型为 jnp.float32

    def setup(self):  # 定义模型初始化方法
        self.conv = FlaxConvWithWeightNorm(self.config, dtype=self.dtype)  # 创建带权重归一化的卷积层对象
        self.activation = ACT2FN[self.config.feat_extract_activation]  # 获取激活函数
        self.num_pad_remove = 1 if self.config.num_conv_pos_embeddings % 2 == 0 else 0  # 计算需要移除的填充数目

    def __call__(self, hidden_states):  # 定义模型调用方法
        hidden_states = hidden_states.transpose((0, 1, 2))  # 调整输入张量的维度顺序

        hidden_states = self.conv(hidden_states)  # 应用卷积操作

        if self.num_pad_remove > 0:  # 如果需要移除填充
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]  # 移除填充
        hidden_states = self.activation(hidden_states)  # 应用激活函数

        hidden_states = hidden_states.transpose((0, 1, 2))  # 恢复张量维度顺序
        return hidden_states  # 返回隐藏状态


class FlaxConvLayersCollection(nn.Module):
    config: Wav2Vec2Config  # 存储模型配置信息
    dtype: jnp.dtype = jnp.float32  # 默认数据类型为 jnp.float32
```  
    # 初始化特征提取的设置
    def setup(self):
        # 如果特征提取使用层归一化
        if self.config.feat_extract_norm == "layer":
            # 创建一系列层归一化卷积层
            self.layers = [
                FlaxWav2Vec2LayerNormConvLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_feat_extract_layers)
            ]
        # 如果特征提取使用组归一化，但目前不支持
        elif self.config.feat_extract_norm == "group":
            raise NotImplementedError("At the moment only ``config.feat_extact_norm == 'layer'`` is supported")
        # 如果特征提取归一化方式不符合要求
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {self.config.feat_extract_norm}, but has to be one of ['group',"
                " 'layer']"
            )
    
    # 特征提取前向传播
    def __call__(self, hidden_states):
        # 依次通过各个卷积层
        for i, conv_layer in enumerate(self.layers):
            hidden_states = conv_layer(hidden_states)
        # 返回最终的特征
        return hidden_states
class FlaxWav2Vec2FeatureEncoder(nn.Module):
    """从原始音频波形构造特征"""

    config: Wav2Vec2Config  # Wav2Vec2 模型的配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型，默认为 jnp.float32

    def setup(self):
        # 初始化卷积层集合
        self.conv_layers = FlaxConvLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, input_values, freeze_feature_encoder=False):
        # 将输入值扩展一个维度，用于卷积操作
        hidden_states = input_values[:, :, None]
        # 通过卷积层集合处理输入值
        hidden_states = self.conv_layers(hidden_states)
        if freeze_feature_encoder:
            # 如果需要冻结特征编码器，停止梯度传播
            hidden_states = jax.lax.stop_gradient(hidden_states)
        return hidden_states


class FlaxWav2Vec2FeatureProjection(nn.Module):
    """Wav2Vec2 的特征投影模块"""

    config: Wav2Vec2Config  # Wav2Vec2 模型的配置
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型，默认为 jnp.float32

    def setup(self):
        # 初始化层归一化
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化全连接层，用于特征投影
        self.projection = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化 dropout 层
        self.dropout = nn.Dropout(rate=self.config.feat_proj_dropout)

    def __call__(self, hidden_states, deterministic=True):
        # 执行层归一化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 执行特征投影
        hidden_states = self.projection(norm_hidden_states)
        # 执行 dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states, norm_hidden_states


class FlaxWav2Vec2Attention(nn.Module):
    """Wav2Vec2 的注意力模块"""

    config: Wav2Vec2Config  # Wav2Vec2 模型的配置
    embed_dim: int  # 嵌入维度
    num_heads: int  # 头的数量
    dropout: float = 0.0  # dropout 概率，默认为 0.0
    bias: bool = True  # 是否使用偏置，默认为 True
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型，默认为 jnp.float32

    def setup(self) -> None:
        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            # 如果嵌入维度不能被头的数量整除，抛出错误
            raise ValueError(
                f"embed_dim 必须能够被 num_heads 整除 (得到 `embed_dim`: {self.embed_dim} 和 `num_heads`:"
                f" {self.num_heads})."
            )

        # 部分函数应用，用于创建全连接层
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=self.bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 初始化查询、键、值投影
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        # 初始化输出投影
        self.out_proj = dense()

        # 初始化 dropout 层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def _split_heads(self, hidden_states):
        # 将隐藏状态分割成多个头
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        # 合并多个头的隐藏状态
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        
        # 获取查询投影
        query_states = self.q_proj(hidden_states)

        # 获取键投影
        key_states = self.k_proj(hidden_states)
        
        # 获取值投影
        value_states = self.v_proj(hidden_states)

        # 对查询投影、键投影和值投影进行头部分割
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # 如果存在注意力掩码，则扩展维度以匹配张量形状
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        # 将布尔类型的注意力掩码转换为注意力偏置
        if attention_mask is not None:
            # 注意力掩码转换为注意力偏置形式
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        # 初始化 dropout_rng
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # 计算注意力权重
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # 计算注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)

        # 返回注意力输出和注意力权重
        return attn_output, attn_weights
# 定义一个FlaxWav2Vec2FeedForward类，继承自nn.Module类
class FlaxWav2Vec2FeedForward(nn.Module):
    # 定义config属性，存储Wav2Vec2Config实例
    config: Wav2Vec2Config
    # 定义dtype属性，默认值为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，用来设置网络层结构
    def setup(self):
        # 初始化中间层的dropout
        self.intermediate_dropout = nn.Dropout(rate=self.config.activation_dropout)

        # 初始化中间层的稠密层，应用配置的初始化方法和数据类型
        self.intermediate_dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 判断配置中隐藏层激活函数是否为字符串，选择对应的激活函数
        if isinstance(self.config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.intermediate_act_fn = self.config.hidden_act

        # 初始化输出层的稠密层，应用配置的初始化方法和数据类型
        self.output_dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化输出层的dropout
        self.output_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    # 实现__call__方法，用来定义前向传播过程
    def __call__(self, hidden_states, deterministic=True):
        # 中间层前向计算
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, deterministic=deterministic)

        # 输出层前向计算
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, deterministic=deterministic)
        # 返回输出结果
        return hidden_states


# 定义FlaxWav2Vec2EncoderLayerStableLayerNorm类，继承自nn.Module类
class FlaxWav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    # 定义config属性，存储Wav2Vec2Config实例
    config: Wav2Vec2Config
    # 定义dtype属性，默认值为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法，用来设置网络层结构
    def setup(self):
        # 初始化注意力层
        self.attention = FlaxWav2Vec2Attention(
            config=self.config,
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            dtype=self.dtype,
        )
        # 初始化dropout层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        # 初始化Layer Norm层
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化前馈网络结构
        self.feed_forward = FlaxWav2Vec2FeedForward(self.config, dtype=self.dtype)
        # 初始化最终Layer Norm层
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 实现__call__方法，用来定义前向传播过程
    def __call__(self, hidden_states, attention_mask=None, deterministic=True, output_attentions=False):
        # 保存注意力计算前的隐藏状态
        attn_residual = hidden_states
        # 应用Layer Norm到隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 计算注意力权重并进行注意力计算
        hidden_states, attn_weights = self.attention(
            hidden_states, attention_mask=attention_mask, deterministic=deterministic
        )
        # 应用dropout层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 加上注意力计算前的隐藏状态，构成残差连接
        hidden_states = attn_residual + hidden_states
        # 应用前馈网络
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states), deterministic=deterministic
        )

        # 构建输出元组
        outputs = (hidden_states,)

        # 根据输出注意力权重标志，决定是否返回注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出结果
        return outputs


# 定义FlaxWav2Vec2EncoderLayerStableLayerNormCollection类，继承自nn.Module类
class FlaxWav2Vec2EncoderLayerStableLayerNormCollection(nn.Module):
    # 定义config属性，存储Wav2Vec2Config实例
    config: Wav2Vec2Config
    # 定义数据类型为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # 创建一个列表,其中包含指定数量的 FlaxWav2Vec2EncoderLayerStableLayerNorm 层
        self.layers = [
            FlaxWav2Vec2EncoderLayerStableLayerNorm(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
    
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 初始化用于存储注意力和隐藏状态的元组
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
    
        # 遍历所有的层
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态,将当前隐藏状态添加到元组中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
    
            # 在当前层上执行前向传播,获得输出
            layer_outputs = layer(
                hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
            )
    
            # 更新隐藏状态
            hidden_states = layer_outputs[0]
    
            # 如果需要输出注意力,将当前注意力添加到元组中
            if output_attentions:
                all_attentions += (layer_outputs[1],)
    
        # 如果需要输出隐藏状态,将最后一个隐藏状态添加到元组中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
    
        # 将输出组成一个元组
        outputs = (hidden_states, all_hidden_states, all_attentions)
    
        # 如果不需要返回字典,将元组中的非 None 元素返回
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
    
        # 如果需要返回字典,创建并返回 FlaxBaseModelOutput
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 定义一个基于 Flax 的 Wav2Vec2 的稳定层归一化编码器类
class FlaxWav2Vec2StableLayerNormEncoder(nn.Module):
    # Wav2Vec2的配置
    config: Wav2Vec2Config
    # 数据类型设置为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化各个组件
    def setup(self):
        # 初始化位置卷积嵌入
        self.pos_conv_embed = FlaxWav2Vec2PositionalConvEmbedding(self.config, dtype=self.dtype)
        # 初始化层归一化
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化丢弃层
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        # 初始化编码器层
        self.layers = FlaxWav2Vec2EncoderLayerStableLayerNormCollection(self.config, dtype=self.dtype)

    # 调用方法，处理输入hidden_states，应用注意力掩码等，并返回结果
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 如果存在注意力掩码
        if attention_mask is not None:
            # 确保填充的标记不被关注
            hidden_states = jnp.where(
                jnp.broadcast_to(attention_mask[:, :, None], hidden_states.shape), hidden_states, 0
            )

        # 计算位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)

        # 加上位置嵌入
        hidden_states = hidden_states + position_embeddings
        # 应用丢弃层
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 处理输入hidden_states，应用注意力掩码等，返回结果
        outputs = self.layers(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 对最后的隐藏状态进行层归一化
        last_hidden_state = self.layer_norm(outputs[0])

        # 更新`hidden_states`中的最后一个元素，在上述应用`layernorm`后
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_state,)

        # 如果不返回字典
        if not return_dict:
            outputs = (last_hidden_state, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput
        return FlaxBaseModelOutput(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=outputs.attentions
        )


# 定义一个基于Flax的Wav2Vec2的Gumbel向量量化器类
class FlaxWav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    使用Gumbel softmax进行向量量化。更多信息详见[CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf)。
    """

    # Wav2Vec2的配置
    config: Wav2Vec2Config
    # 数据类型设置为jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 原因和步骤：初始化对象时，设置属性
    def setup(self):
        # 设置编码向量的分组数和每组的编码向量数
        self.num_groups = self.config.num_codevector_groups
        self.num_vars = self.config.num_codevectors_per_group

        # 如果编码向量维度不能被分组数整除，则抛出异常
        if self.config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {self.config.codevector_dim} must be divisible by"
                f" `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # 为编码向量变量（码字）提供存储空间
        self.codevectors = self.param(
            "codevectors",
            jax.nn.initializers.uniform(),
            (1, self.num_groups * self.num_vars, self.config.codevector_dim // self.num_groups),
        )

        # 权重投影层，用于投影到编码向量总数的维度
        self.weight_proj = nn.Dense(
            self.num_groups * self.num_vars,
            kernel_init=jax.nn.initializers.normal(1.0),
            dtype=self.dtype,
        )

    # 静态方法：计算困惑度
    @staticmethod
    def _compute_perplexity(probs, mask=None):
        # 如果给定了掩码，则将掩码扩展到与概率形状相同
        if mask is not None:
            mask_extended = jnp.broadcast_to(mask.flatten()[:, None, None], probs.shape)
            # 将掩码应用于概率，将非掩码位置上的概率置为零
            probs = jnp.where(mask_extended, probs, jnp.zeros_like(probs))
            # 计算边际概率，即各个位置上的概率的和除以掩码位置的总数
            marginal_probs = probs.sum(axis=0) / mask.sum()
        else:
            # 如果没有给定掩码，则计算平均概率
            marginal_probs = probs.mean(axis=0)

        # 计算困惑度 = exp(-sum(边际概率 * log(边际概率 + 1e-7))) 的和
        perplexity = jnp.exp(-jnp.sum(marginal_probs * jnp.log(marginal_probs + 1e-7), axis=-1)).sum()
        # 返回困惑度
        return perplexity
    # 定义一个方法，接受隐藏状态、时间索引掩码、是否确定性、温度参数作为输入
    def __call__(self, hidden_states, mask_time_indices=None, deterministic=True, temperature=1):
        # 获取隐藏状态的形状信息
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到码向量维度
        hidden_states = self.weight_proj(hidden_states)
        # 将隐藏状态重塑为(batch_size * sequence_length * self.num_groups, -1)的形状
        hidden_states = hidden_states.reshape(batch_size * sequence_length * self.num_groups, -1)

        # 如果不是确定性的，则执行以下操作
        if not deterministic:
            # 在可微分的方式中通过Gumbel分布采样码向量概率
            gumbel_rng = self.make_rng("gumbel")
            gumbels = jax.random.gumbel(gumbel_rng, hidden_states.shape)
            codevector_probs = nn.softmax((hidden_states + gumbels) / temperature)

            # 计算困惑度
            codevector_soft_dist = nn.softmax(
                hidden_states.reshape(batch_size * sequence_length, self.num_groups, -1), axis=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        # 如果是确定性的，则执行以下操作
        else:
            # 以不可微分的方式取argmax，计算硬码向量分布（one hot）
            codevector_idx = hidden_states.argmax(axis=-1)
            codevector_probs = jax.nn.one_hot(codevector_idx, hidden_states.shape[-1]) * 1.0
            codevector_probs = codevector_probs.reshape(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        # 重塑码向量概率的形状为(batch_size * sequence_length, -1)
        codevector_probs = codevector_probs.reshape(batch_size * sequence_length, -1)
        # 使用概率来检索码向量
        codevectors_per_group = jnp.expand_dims(codevector_probs, axis=-1) * self.codevectors
        codevectors = codevectors_per_group.reshape(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).reshape(batch_size, sequence_length, -1)

        # 返回码向量和困惑度
        return codevectors, perplexity
class FlaxWav2Vec2Adapter(nn.Module):
    # 定义一个适配器模块，继承自 nn.Module
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 如果隐藏状态需要下投影，如果特征维度不匹配
        if self.config.output_hidden_size != self.config.hidden_size:
            # 创建一个具有指定输出大小的全连接层，并初始化权重
            self.proj = nn.Dense(
                self.config.output_hidden_size,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                dtype=self.dtype,
            )
            # 创建一个 LayerNorm 层
            self.proj_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        else:
            self.proj = self.proj_layer_norm = None

        # 创建一个适配器层的集合
        self.layers = FlaxWav2Vec2AdapterLayersCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        # 如果需要，进行隐藏状态的下投影
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 对隐藏状态进行处理
        hidden_states = self.layers(hidden_states)

        return hidden_states


class FlaxWav2Vec2AdapterLayer(nn.Module):
    # 定义一个适配器层，继承自 nn.Module
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建一个卷积层
        self.conv = nn.Conv(
            features=2 * self.config.output_hidden_size,
            kernel_size=(self.config.adapter_kernel_size,),
            strides=(self.config.adapter_stride,),
            padding=((1, 1),),
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # 对隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 使用门控线性单元激活函数处理隐藏状态
        hidden_states = nn.glu(hidden_states, axis=2)

        return hidden_states


class FlaxWav2Vec2AdapterLayersCollection(nn.Module):
    # 定义一个适配器层的集合，继承自 nn.Module
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # 创建适配器层的列表
        self.layers = [
            FlaxWav2Vec2AdapterLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_adapter_layers)
        ]

    def __call__(self, hidden_states):
        # 遍历适配器层列表，对隐藏状态进行处理
        for conv_layer in self.layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


class FlaxWav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Wav2Vec2 模型的抽象类，用于处理权重初始化和预训练模型的下载和加载
    config_class = Wav2Vec2Config
    base_model_prefix: str = "wav2vec2"
    main_input_name = "input_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Wav2Vec2Config,
        input_shape: Tuple = (1, 1024),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
       
    ):
        # 实例化一个自定义的模块对象
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 调用父类的构造函数进行初始化
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_values = jnp.zeros(input_shape, dtype="i4")
        # 创建一个与input_values形状相同的全1张量
        attention_mask = jnp.ones_like(input_values)
        # 将rng切分为两部分，params_rng 和 dropout_rng
        params_rng, dropout_rng = jax.random.split(rng, 2)
        # 将切分后的rng存放在rngs字典中
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 通过随机初始化方式初始化自定义模块的参数
        random_params = self.module.init(rngs, input_values, attention_mask, return_dict=False)["params"]

        if params is not None:
            # 将随机初始化的参数和已有的参数展平为一维张量
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            # 将缺失的键添加到已有的参数字典中
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            # 冻结参数字典并返回
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        freeze_feature_encoder: bool = False,
        return_dict: Optional[bool] = None,
    ):
        # 判断是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 判断是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 判断是否返回字典格式结果
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入值的batch size和序列长度
        batch_size, sequence_length = input_values.shape

        # 如果attention_mask为None，创建与input_values形状相同的全1张量作为attention_mask
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 创建用于存放PRNGKey的字典
        rngs = {}
        # 如果dropout_rng不为None，将其加入rngs字典
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 创建inputs字典并存储参数
        inputs = {"params": params or self.params}

        # 对自定义模块的apply方法进行调用
        return self.module.apply(
            inputs,
            jnp.array(input_values, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            mask_time_indices,
            not train,
            output_attentions,
            output_hidden_states,
            freeze_feature_encoder,
            return_dict,
            rngs=rngs,
        )

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        return self.module._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)
# 定义 FlaxWav2Vec2Module 类，继承自 nn.Module
class FlaxWav2Vec2Module(nn.Module):
    # 配置信息
    config: Wav2Vec2Config
    # 数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建 FlaxWav2Vec2FeatureEncoder 对象
        self.feature_extractor = FlaxWav2Vec2FeatureEncoder(self.config, dtype=self.dtype)
        # 创建 FlaxWav2Vec2FeatureProjection 对象
        self.feature_projection = FlaxWav2Vec2FeatureProjection(self.config, dtype=self.dtype)
        # 创建 masked_spec_embed 参数
        self.masked_spec_embed = self.param(
            "masked_spec_embed", jax.nn.initializers.uniform(), (self.config.hidden_size,)
        )

        # 根据 config.do_stable_layer_norm 的值创建不同的编码器
        if self.config.do_stable_layer_norm:
            self.encoder = FlaxWav2Vec2StableLayerNormEncoder(self.config, dtype=self.dtype)
        else:
            raise NotImplementedError("``config.do_stable_layer_norm is False`` is currently not supported.")

        # 如果配置中需要 adapter，则创建 FlaxWav2Vec2Adapter 对象
        self.adapter = FlaxWav2Vec2Adapter(self.config, dtype=self.dtype) if self.config.add_adapter else None

    # 前向传播方法
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        deterministic=True,
        output_attentions=None,
        output_hidden_states=None,
        freeze_feature_encoder=False,
        return_dict=None,
    ):
        # 提取特征
        extract_features = self.feature_extractor(input_values, freeze_feature_encoder=freeze_feature_encoder)

        # 根据 attention_mask 计算相应的注意力掩码
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 进行特征投射
        hidden_states, extract_features = self.feature_projection(extract_features, deterministic=deterministic)

        # 如果给定了 mask_time_indices，则应用 SpecAugment 增强
        if mask_time_indices is not None:
            hidden_states = jnp.where(
                jnp.broadcast_to(mask_time_indices[:, :, None], hidden_states.shape),
                jnp.broadcast_to(self.masked_spec_embed[None, None, :], hidden_states.shape),
                hidden_states,
            )

        # 通过编码器进行编码
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的输出
        hidden_states = encoder_outputs[0]

        # 如果配置中需要 adapter，则应用 adapter
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 根据 return_dict 的值返回不同的输出
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return FlaxWav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 获取特征提取输出长度的方法
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        ...
    ):
        """
        计算卷积层的输出长度
        """

        # 设置是否添加适配器，默认为config中设定的值
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 获取的1D卷积层输出长度公式
            return (input_length - kernel_size) // stride + 1

        # 遍历卷积核大小和步长，计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要添加适配器层
        if add_adapter:
            # 遍历适配器层数量，计算输出长度
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None
    ):
        # 根据attention_mask的累积值计算非填充长度
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]

        # 获取特征提取输出长度
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)

        batch_size = attention_mask.shape[0]

        # 初始化全零的注意力掩码
        attention_mask = jnp.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        # 确保所有输出长度之前的值都被关注
        attention_mask = attention_mask.at[jnp.arange(attention_mask.shape[0]), output_lengths - 1].set(1)
        # 翻转计算出的注意力掩码
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype("bool")
        return attention_mask
# 添加起始文档字符串，描述Wav2Vec2模型输出原始隐藏状态的基本模型，以及其它相关信息
@add_start_docstrings(
    "The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class FlaxWav2Vec2Model(FlaxWav2Vec2PreTrainedModel):
    # 使用FlaxWav2Vec2Module作为模块类
    module_class = FlaxWav2Vec2Module


# 添加文档字符串描述返回、示例等信息，并将其替换为FlaxWav2Vec2Model的调用文档字符串
FLAX_WAV2VEC2_MODEL_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> from transformers import AutoProcessor, FlaxWav2Vec2Model
    >>> from datasets import load_dataset
    >>> import soundfile as sf

    >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60")
    >>> model = FlaxWav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60")


    >>> def map_to_array(batch):
    ...     speech, _ = sf.read(batch["file"])
    ...     batch["speech"] = speech
    ...     return batch


    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> ds = ds.map(map_to_array)

    >>> input_values = processor(
    ...     ds["speech"][0], sampling_rate=16_000, return_tensors="np"
    ... ).input_values  # Batch size 1
    >>> hidden_states = model(input_values).last_hidden_state
    ```
"""

# 重写FlaxWav2Vec2Model的调用文档字符串
overwrite_call_docstring(
    FlaxWav2Vec2Model,
    WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_MODEL_DOCSTRING,
)

# 添加替换返回文档字符串
append_replace_return_docstrings(
    FlaxWav2Vec2Model, output_type=FlaxWav2Vec2BaseModelOutput, config_class=Wav2Vec2Config
)


# 创建FlaxWav2Vec2ForCTCModule类
class FlaxWav2Vec2ForCTCModule(nn.Module):
    # 定义config属性为Wav2Vec2Config，dtype属性为jnp.float32
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 创建wav2vec2属性，使用FlaxWav2Vec2Module和指定的config和dtype
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)
        # 创建dropout属性，使用指定的config.final_dropout
        self.dropout = nn.Dropout(rate=self.config.final_dropout)
        # 创建lm_head属性，使用指定的config.vocab_size和kernel_init等参数
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 调用方法
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        deterministic=True,
        output_attentions=None,
        output_hidden_states=None,
        freeze_feature_encoder=False,
        return_dict=None,
    ):
        # 对input_values执行wav2vec2操作，获取输出结果
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            freeze_feature_encoder=freeze_feature_encoder,
            return_dict=return_dict,
        )

        # 获取隐藏状态，并对其进行dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 计算logits
        logits = self.lm_head(hidden_states)

        # 如果return_dict为False，则返回logits和outputs[2:]
        if not return_dict:
            return (logits,) + outputs[2:]

        # 如果return_dict为True，则返回FlaxCausalLMOutput类型的结果
        return FlaxCausalLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    # 定义一个方法来计算卷积层的输出长度
    def _get_feat_extract_output_lengths(
        self,
        input_lengths: Union[jnp.ndarray, int],
        add_adapter: Optional[bool] = None,
    ):
        """
        Computes the output length of the convolutional layers
        """

        # 如果 add_adapter 为 None，则使用配置中的 add_adapter
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        # 定义一个方法来计算卷积层输出长度
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层的输出长度公式来自于 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 根据配置中的卷积核大小和步长计算输入长度的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果要添加适配器层，则根据配置中的适配器层数量计算输入长度的输出长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        # 返回最终的输入长度
        return input_lengths
# 添加文档字符串描述 Wav2Vec2 模型与用于 CTC 的语言建模头部
@add_start_docstrings(
    "Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).",
    WAV_2_VEC_2_START_DOCSTRING,
)
# 定义FlaxWav2Vec2ForCTC类，继承自FlaxWav2Vec2PreTrainedModel类
class FlaxWav2Vec2ForCTC(FlaxWav2Vec2PreTrainedModel):
    # 模型类别为FlaxWav2Vec2ForCTCModule
    module_class = FlaxWav2Vec2ForCTCModule

# FlaxWav2Vec2ForCTC类的文档字符串
FLAX_WAV2VEC2_FOR_CTC_DOCSTRING = """
    Returns:

    Example:

    # 示例代码，使用模型进行预测解码
    ```python
    >>> import jax.numpy as jnp
    >>> from transformers import AutoProcessor, FlaxWav2Vec2ForCTC
    >>> from datasets import load_dataset
    >>> import soundfile as sf

    >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
    >>> model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")

    # 自定义函数map_to_array，用于处理batch数据
    >>> def map_to_array(batch):
    ...     speech, _ = sf.read(batch["file"])
    ...     batch["speech"] = speech
    ...     return batch

    # 加载数据集并映射处理
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> ds = ds.map(map_to_array)

    # 处理输入数据并生成预测结果
    >>> input_values = processor(
    ...     ds["speech"][0], sampling_rate=16_000, return_tensors="np"
    ... ).input_values  # Batch size 1
    >>> logits = model(input_values).logits
    >>> predicted_ids = jnp.argmax(logits, axis=-1)

    # 解码预测结果
    >>> transcription = processor.decode(predicted_ids[0])
    >>> # should give:  "A MAN SAID TO THE UNIVERSE SIR I EXIST"
    ```

"""

# 覆盖FlaxWav2Vec2ForCTC类的调用文档字符串
overwrite_call_docstring(
    FlaxWav2Vec2ForCTC,
    WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_FOR_CTC_DOCSTRING,
)
# 追加替换返回文档字符串
append_replace_return_docstrings(FlaxWav2Vec2ForCTC, output_type=FlaxCausalLMOutput, config_class=Wav2Vec2Config)


# 定义FlaxWav2Vec2ForPreTrainingModule类，继承自nn.Module类
class FlaxWav2Vec2ForPreTrainingModule(nn.Module):
    # 定义config属性，类型为Wav2Vec2Config类
    config: Wav2Vec2Config
    # 定义dtype属性，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 初始化wav2vec2属性，使用FlaxWav2Vec2Module类
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)
        # 初始化dropout_features属性，使用nn.Dropout类
        self.dropout_features = nn.Dropout(self.config.feat_quantizer_dropout)

        # 初始化quantizer属性，使用FlaxWav2Vec2GumbelVectorQuantizer类
        self.quantizer = FlaxWav2Vec2GumbelVectorQuantizer(self.config, dtype=self.dtype)
        # 初始化project_q属性，使用nn.Dense类，进行压缩编码向量的投影转换
        self.project_q = nn.Dense(
            self.config.proj_codevector_dim,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 初始化project_hid属性，使用nn.Dense类，进行隐藏状态的投影转换
        self.project_hid = nn.Dense(
            self.config.proj_codevector_dim,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 定义调用方法，接受多个参数
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        gumbel_temperature: int = 1,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        freeze_feature_encoder=False,
        return_dict=None,
   # 该方法是一个私有方法，用于获取特征提取层的输出长度
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """
        # 如果add_adapter为None，则使用配置中的add_adapter值
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        # 定义计算卷积层输出长度的函数
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D卷积层输出长度公式来自于https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        # 遍历配置中的卷积核大小和步幅，更新输入长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要添加适配器，再次根据适配器的参数更新输入长度
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        # 返回更新后的输入长度
        return input_lengths
# 使用'add_start_docstrings'装饰器为FlaxWav2Vec2ForPreTraining类添加注释和文档字符串
@add_start_docstrings("""Wav2Vec2 Model with a quantizer and `VQ` head on top.""", WAV_2_VEC_2_START_DOCSTRING)
class FlaxWav2Vec2ForPreTraining(FlaxWav2Vec2PreTrainedModel):
    # 设置模型类别为FlaxWav2Vec2ForPreTrainingModule
    module_class = FlaxWav2Vec2ForPreTrainingModule

    # 使用'add_start_docstrings_to_model_forward'装饰器为__call__方法添加注释和文档字符串
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    # 重写__call__方法，添加'gumbel_temperature'输入参数
    def __call__(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        gumbel_temperature: int = 1,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        gumbel_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        freeze_feature_encoder: bool = False,
        return_dict: Optional[bool] = None,
    ):
        # 设置output_attentions, output_hidden_states和return_dict默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取input_values的batch_size和sequence_length
        batch_size, sequence_length = input_values.shape

        # 如果attention_mask是None，则创建一个全为1的矩阵
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 处理需要的PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        if gumbel_rng is not None:
            rngs["gumbel"] = gumbel_rng

        # 设置inputs为params或self.params
        inputs = {"params": params or self.params}

        # 调用module的apply方法，传入各种参数和参数值
        return self.module.apply(
            inputs,
            jnp.array(input_values, dtype="f4"),
            jnp.array(attention_mask, dtype="i4"),
            mask_time_indices,
            gumbel_temperature,
            not train,
            output_attentions,
            output_hidden_states,
            freeze_feature_encoder,
            return_dict,
            rngs=rngs,
        )


# 给FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING添加注释和文档字符串
FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING = """
    Returns:

    Example:

    ```python
    >>> import optax
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> from transformers import AutoFeatureExtractor, FlaxWav2Vec2ForPreTraining
    >>> from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices
    >>> from datasets import load_dataset
    >>> import soundfile as sf

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")
    >>> model = FlaxWav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-lv60")


    >>> def map_to_array(batch):
    ...     speech, _ = sf.read(batch["file"])
    ...     batch["speech"] = speech
    ...     return batch


    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> ds = ds.map(map_to_array)




注释：
    >>> input_values = feature_extractor(ds["speech"][0], return_tensors="np").input_values  # Batch size 1

    >>> # 计算被掩盖的索引
    >>> batch_size, raw_sequence_length = input_values.shape
    >>> # 通过模型获取特征提取后的输出长度
    >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
    >>> # 计算需要被掩盖的时间索引
    >>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)

    >>> outputs = model(input_values, mask_time_indices=mask_time_indices)

    >>> # 计算预测（=projected_states）和目标（=projected_quantized_states）之间的余弦相似度
    >>> cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states)

    >>> # 证明余弦相似度远高于随机
    >>> assert np.asarray(cosine_sim)[mask_time_indices].mean() > 0.5
```  
# 覆盖 FlaxWav2Vec2ForPreTraining 的文档字符串
overwrite_call_docstring(
    FlaxWav2Vec2ForPreTraining,
    WAV_2_VEC_2_INPUTS_DOCSTRING + FLAX_WAV2VEC2_FOR_PRETRAINING_DOCSTRING,
)
# 追加并替换 FlaxWav2Vec2ForPreTraining 的返回文档字符串
append_replace_return_docstrings(
    FlaxWav2Vec2ForPreTraining, output_type=FlaxWav2Vec2ForPreTrainingOutput, config_class=Wav2Vec2Config
)
```