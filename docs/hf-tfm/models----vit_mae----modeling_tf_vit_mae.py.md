# `.\transformers\models\vit_mae\modeling_tf_vit_mae.py`

```
# 声明文件编码为 UTF-8
# 版权声明和许可信息
# 本模型基于 Apache 许可证 2.0 进行许可
# 除非符合许可协议要求或书面同意，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0
# 本软件按“原样”提供，不提供任何形式的担保或条件
# 查看许可证以获取更多信息
""" TF 2.0 ViT MAE (masked autoencoder) model."""
# 导入所需的库和模块
from __future__ import annotations  # 允许在类型注释中使用类型本身

import collections.abc  # 导入抽象基类模块
import math  # 导入数学模块
from copy import deepcopy  # 导入深度拷贝函数
from dataclasses import dataclass  # 导入 dataclass 装饰器
from typing import Optional, Tuple, Union  # 导入类型提示

import numpy as np  # 导入 NumPy 库
import tensorflow as tf  # 导入 TensorFlow 库

# 导入 Hugging Face 库中的其他模块和函数
from ...activations_tf import get_tf_activation  # 导入获取 TensorFlow 激活函数的函数
from ...file_utils import (  # 导入文件相关工具函数
    ModelOutput,  # 导入模型输出类
    add_start_docstrings,  # 导入添加文档字符串的函数
    add_start_docstrings_to_model_forward,  # 导入给模型前向方法添加文档字符串的函数
    replace_return_docstrings,  # 导入替换返回文档字符串的函数
)
from ...modeling_tf_outputs import TFBaseModelOutput  # 导入 TensorFlow 模型输出基类
from ...modeling_tf_utils import (  # 导入 TensorFlow 模型相关工具函数
    TFModelInputType,  # 导入 TensorFlow 模型输入类型
    TFPreTrainedModel,  # 导入 TensorFlow 预训练模型基类
    get_initializer,  # 导入获取初始化器的函数
    keras_serializable,  # 导入 keras 可序列化装饰器
    unpack_inputs,  # 导入解包输入的函数
)
from ...tf_utils import shape_list, stable_softmax  # 导入 TensorFlow 相关工具函数
from ...utils import logging  # 导入日志记录工具
from .configuration_vit_mae import ViTMAEConfig  # 导入 ViT MAE 模型配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

_CONFIG_FOR_DOC = "ViTMAEConfig"  # 用于文档字符串的配置信息
_CHECKPOINT_FOR_DOC = "facebook/vit-mae-base"  # 用于文档字符串的检查点信息

@dataclass
class TFViTMAEModelOutput(ModelOutput):  # TFViTMAEModel 的输出类，继承自模型输出类
    """
    Class for TFViTMAEModel's outputs, with potential hidden states and attentions.
    """
```  
    """
    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    # 定义变量并初始化为None，用于接收传入的参数值
    last_hidden_state: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None
```  
@dataclass
class TFViTMAEDecoderOutput(ModelOutput):
    """
    TFViTMAEDecoderOutput类的输出，包含潜在的隐藏状态和注意力。

    参数:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            像素重建logits。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `tf.Tensor`元组（一个用于嵌入输出 + 每层输出的一个）的形状为`(batch_size, sequence_length, hidden_size)`。 模型在每一层输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `tf.Tensor`元组（每个层一个）的形状为`(batch_size, num_heads, sequence_length,
            sequence_length)`。 在注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    attentions: Tuple[tf.Tensor] | None = None


@dataclass
class TFViTMAEForPreTrainingOutput(ModelOutput):
    """
    TFViTMAEForPreTrainingOutput类的输出，包含潜在的隐藏状态和注意力。

    参数:
        loss (`tf.Tensor` of shape `(1,)`):
            像素重建损失。
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            像素重建logits。
        mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            指示哪些补丁被遮罩（1）哪些没有（0）的张量。
        ids_restore (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            包含（混洗的）被遮罩补丁的原始索引的张量。
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `tf.Tensor`元组（一个用于嵌入输出 + 每层输出的一个）的形状为`(batch_size, sequence_length, hidden_size)`。 模型在每一层输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `tf.Tensor`元组（每个层一个）的形状为`(batch_size, num_heads, sequence_length,
            sequence_length)`。 在注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    mask: tf.Tensor = None
    ids_restore: tf.Tensor = None
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义一个变量 attentions，类型为 Tuple[tf.Tensor] 或者 None，初始值为 None
# 创建 2D 正弦/余弦位置嵌入
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    创建 2D 正弦/余弦位置嵌入。

    Args:
        embed_dim (int): 嵌入维度。
        grid_size (int): 网格的高度和宽度。
        add_cls_token (bool, 可选，默认为 False): 是否添加分类 (CLS) token。

    Returns:
        一个形状为 (grid_size * grid_size, embed_dim) 或 (1 + grid_size * grid_size, embed_dim) 的张量：包含位置嵌入
    """
    # 生成高度和宽度的 1D 坐标范围
    grid_h = tf.range(grid_size, dtype=tf.float32)
    grid_w = tf.range(grid_size, dtype=tf.float32)
    
    # 创建 2D 网格，生成网格坐标
    grid = tf.meshgrid(grid_w, grid_h)  # 这里 w 在前
    grid = tf.stack(grid, axis=0)  # 将网格堆叠在一起
    
    # 调整网格形状为 2x1xgrid_size x grid_size
    grid = tf.reshape(grid, [2, 1, grid_size, grid_size])
    
    # 从网格获取 2D 正弦/余弦位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    # 如果需要添加 CLS token，将其添加到位置嵌入的前面
    if add_cls_token:
        pos_embed = tf.concat([tf.zeros((1, embed_dim)), pos_embed], axis=0)  # 添加 CLS token
    
    # 返回位置嵌入
    return pos_embed


# 从网格中生成 2D 正弦/余弦位置嵌入
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # 确保嵌入维度是偶数
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim 必须是偶数")
    
    # 使用一半的维度来编码高度
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    
    # 使用另一半的维度来编码宽度
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    # 将高度和宽度的嵌入连接起来
    emb = tf.concat([emb_h, emb_w], axis=1)  # (H*W, D)
    
    # 返回合并的 2D 位置嵌入
    return emb


# 从 1D 网格生成正弦/余弦位置嵌入
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    获取指定嵌入维度的 1D 正弦/余弦位置嵌入。

    embed_dim: 输出维度
    pos: 待编码的位置，大小为 (M,)
    输出: (M, D)
    """
    # 确保嵌入维度是偶数
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim 必须是偶数")
    
    # 计算正弦/余弦频率
    omega = tf.range(embed_dim // 2, dtype="float32")  # 创建范围
    omega /= embed_dim / 2.0  # 缩放到 0 到 1 的范围
    omega = 1.0 / 10000 ** omega  # 转换为频率

    # 将位置调整为一维数组
    pos = tf.reshape(pos, [-1])  # (M,)
    
    # 计算外积以得到正弦/余弦输入
    out = tf.einsum("m,d->md", pos, omega)  # (M, D/2), 外积
    
    # 一半维度应用正弦模式
    emb_sin = tf.sin(out)  # (M, D/2)
    
    # 另一半维度应用余弦模式
    emb_cos = tf.cos(out)  # (M, D/2)
    
    # 将正弦和余弦嵌入连接起来
    emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    
    # 返回最终嵌入
    return emb


# 创建用于 CLS token、位置和补丁嵌入的类
class TFViTMAEEmbeddings(tf.keras.layers.Layer):
    """
    构建 CLS token、位置和补丁嵌入。
    """
    
    # 初始化类，接受配置参数和其他可选参数
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
    
    # 创建补丁嵌入层并获取补丁数量
    self.patch_embeddings = TFViTMAEPatchEmbeddings(config, name="patch_embeddings")  # 创建补丁嵌入
    self.num_patches = self.patch_embeddings.num_patches  # 获取补丁数量
    
    # 保存配置以供后续使用
    self.config = config  # 存储配置参数
    # 构建模型，设置初始输入形状
    def build(self, input_shape=None):
        # 添加用于表示类别的 token，形状为 (1, 1, 隐藏大小)
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )
        # 添加位置嵌入，形状为 (1, patch 数量加 1, 隐藏大小)
        self.position_embeddings = self.add_weight(
            shape=(1, self.num_patches + 1, self.config.hidden_size),
            initializer="zeros",
            trainable=False,  # 固定的 sin-cos 嵌入
            name="position_embeddings",
        )
        # 获取 2D sin-cos 位置嵌入
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=True,
        )[None, ...]
        # 分配位置嵌入
        self.position_embeddings.assign(pos_embed)

        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果 patch_embeddings 已存在，则构建 patch_embeddings
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)

    # 执行每个样本的随机 masking，通过每个样本的乱序来实现
    def random_masking(self, sequence: tf.Tensor, noise: tf.Tensor | None = None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`tf.Tensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        # 获取 batch_size, seq_length, dim
        batch_size, seq_length, dim = shape_list(sequence)
        # 计算需要保留的长度
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        # 如果未提供噪声，则从 [0, 1) 的均匀分布中随机生成噪声
        if noise is None:
            noise = tf.random.uniform(shape=(batch_size, seq_length), minval=0.0, maxval=1.0)

        # 对每个样本的噪声进行排序
        ids_shuffle = tf.argsort(noise, axis=1)  # 升序：小的是保留，大的是移除
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # 保留第一个子集
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = tf.gather(
            sequence,
            axis=1,
            batch_dims=1,
            indices=ids_keep,
        )

        # 生成二进制 mask：0 表示保留，1 表示移除
        mask_keep = tf.zeros((batch_size, len_keep))
        mask_remove = tf.ones((batch_size, seq_length - len_keep))
        mask = tf.concat([mask_keep, mask_remove], axis=-1)

        # 还原到原始顺序，得到二进制 mask
        mask = tf.gather(mask, axis=1, batch_dims=1, indices=ids_restore)

        return sequence_unmasked, mask, ids_restore
    # 定义一个方法call，接受像素值和噪声作为输入并返回一个张量
    def call(self, pixel_values: tf.Tensor, noise: tf.Tensor = None) -> tf.Tensor:
        # 使用patch_embeddings方法将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values)

        # 添加位置嵌入但不包括cls令牌
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # 对嵌入进行掩码处理，长度为长度*config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # 添加cls令牌
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = tf.tile(cls_token, (shape_list(embeddings)[0], 1, 1))
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)

        # 返回嵌入向量、掩码和ids_restore
        return embeddings, mask, ids_restore
# 定义一个名为TFViTMAEPatchEmbeddings的类，用于将像素值转换为初始隐藏状态（patch嵌入）以供Transformer消费
class TFViTMAEPatchEmbeddings(tf.keras.layers.Layer):

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)
        # 从配置中获取图像大小和patch大小，通道数和隐藏大小，并做相应的赋值
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 将获取的值保存为类的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config

        # 创建一个卷积层用于将输入的像素值转换为嵌入向量
        self.projection = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            kernel_initializer="glorot_uniform",  # following torch.nn.Linear
            bias_initializer="zeros",
            name="projection",
        )

    # 定义call方法，根据输入的像素值生成嵌入向量
    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        
        # 若为动态执行模式，检查通道数和图像大小是否和配置中的一致，如不一致则报错
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the"
                    " configuration."
                )
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        
        # 在CPU上运行时，`tf.keras.layers.Conv2D`不支持`NCHW`格式，所以将输入格式从`NCHW`改为`NHWC`
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 使用卷积层对输入的像素值进行投影
        projection = self.projection(pixel_values)

        # 将二维空间维度��换为单一的时间维度
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        x = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return x
    # 定义 build 方法，用于构建模型结构，可接受输入形状参数，默认为 None
    def build(self, input_shape=None):
        # 如果模型已经构建过，则直接返回，不重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 检查是否存在 projection 属性，并且其值不为 None
        if getattr(self, "projection", None) is not None:
            # 在 TensorFlow 中创建一个命名空间，用于组织模型结构
            with tf.name_scope(self.projection.name):
                # 构建 projection 层，指定输入形状为 [None, None, None, self.num_channels]
                self.projection.build([None, None, None, self.num_channels])
# 从transformers.models.vit.modeling_tf_vit.TFViTSelfAttention复制的代码，并将ViT替换为ViTMAE
class TFViTMAESelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 如果隐藏大小（config.hidden_size）不能被注意力头数（config.num_attention_heads）整除，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        # 设置注意力头数、每个注意力头的大小和总头大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 实例化查询、键、值和dropout层
        self.query = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # 从[batch_size, seq_length, all_head_size]重新塑造为[batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # 将张量从[batch_size, seq_length, num_attention_heads, attention_head_size]转置为[batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    # 定义函数并指定返回类型为包含 tf.Tensor 的元组
    def call(self, hidden_states: tf.Tensor, head_mask: Optional[tf.Tensor] = None, training: bool = False, output_attentions: bool = False) -> Tuple[tf.Tensor]:
        # 获取隐藏状态的批处理大小
        batch_size = shape_list(hidden_states)[0]
        # 使用 query 神经网络层处理隐藏状态
        mixed_query_layer = self.query(inputs=hidden_states)
        # 使用 key 神经网络层处理隐藏状态
        mixed_key_layer = self.key(inputs=hidden_states)
        # 使用 value 神经网络层处理隐藏状态
        mixed_value_layer = self.value(inputs=hidden_states)
        # 调用 transpose_for_scores 方法对处理后的结果进行排列，以备用于计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 通过 "query" 和 "key" 的点积计算原始的注意力分数
        # 结果形状为 (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 将注意力分数归一化为概率
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 使用 dropout 方法对注意力概率进行处理，可能会随机丢弃整个Token的注意力
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # 如果有需要，对头部进行掩码处理
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        # 计算注意力输出
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 将结果重新塑形为 (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        # 如果需要输出注意力，则返回注意力输出和注意力概率，否则只返回注意力输出
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        return outputs

    # 构建神经网络层
    def build(self, input_shape=None):
        # 如果已构建则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 query 层，则进行构建
        if getattr(self, "query", None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.h]idden_size])
        # 如果存在 key 层，则进行构建
        if getattr(self, "key", None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        # 如果存在 value 层，则进行构建
        if getattr(self, "value", None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
# 从 transformers.models.vit.modeling_tf_vit.TFViTSelfOutput 复制并将 ViT 改为 ViTMAE
class TFViTMAESelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为 config.hidden_size，初始化方式为 config.initializer_range，命名为"dense"
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个 Dropout 层，丢失率为 config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    # 此层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(inputs=hidden_states)
        # 在训练时使用 Dropout 层
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.vit.modeling_tf_vit.TFViTAttention 复制并将 ViT 改为 ViTMAE
class TFViTMAEAttention(tf.keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建 TFViTMAESelfAttention 层并命名为"attention"
        self.self_attention = TFViTMAESelfAttention(config, name="attention")
        # 创建 TFViTMAESelfOutput 层并命名为"output"
        self.dense_output = TFViTMAESelfOutput(config, name="output")

    # 剪枝一些 attention heads（注意力头）暂未实现
    def prune_heads(self, heads):
        raise NotImplementedError

    # 此层的前向传播逻辑
    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用 self_attention 层
        self_outputs = self.self_attention(
            hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training
        )
        # 调用 dense_output 层
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        outputs = (attention_output,) + self_outputs[1:]  # 若输出注意力信息则需要添加到 outputs 中

        return outputs

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "self_attention", None) is not None:
            with tf.name_scope(self.self_attention.name):
                # 为 self_attention 层构建
                self.self_attention.build(None)
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                # 为 dense_output 层构建
                self.dense_output.build(None)


# 从 transformers.models.vit.modeling_tf_vit.TFViTIntermediate 复制并将 ViT 改为 ViTMAE
class TFViTMAEIntermediate(tf.keras.layers.Layer):
    # 初始化函数，接收ViTMAEConfig对象和其他关键字参数
        def __init__(self, config: ViTMAEConfig, **kwargs):
            # 调用父类初始化函数
            super().__init__(**kwargs)
    
            # 创建一个全连接层，指定单元数和内核初始化方式
            self.dense = tf.keras.layers.Dense(
                units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
            )
    
            # 如果config.hidden_act是字符串，则将其转换为激活函数
            if isinstance(config.hidden_act, str):
                self.intermediate_act_fn = get_tf_activation(config.hidden_act)
            else:
                self.intermediate_act_fn = config.hidden_act
            self.config = config
    
        # 用于调用模型的前向传播函数，接收隐藏状态张量并返回转换后的张量
        def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
            # 进行全连接操作
            hidden_states = self.dense(inputs=hidden_states)
            # 使用中间激活函数处理隐藏状态
            hidden_states = self.intermediate_act_fn(hidden_states)
    
            return hidden_states
    
        # 用于构建模型，指定输入形状
        def build(self, input_shape=None):
            # 如果已经构建过，则直接返回
            if self.built:
                return
            self.built = True
            # 如果存在self.dense属性，则建立全连接层
            if getattr(self, "dense", None) is not None:
                # 在命名作用域内建立全连接层
                with tf.name_scope(self.dense.name):
                    self.dense.build([None, None, self.config.hidden_size])
# 从transformers.models.vit.modeling_tf_vit.TFViTOutput复制到ViT->ViTMAE
class TFViTMAEOutput(tf.keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，用于转换隐藏状态
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个dropout层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    # 定义该层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)  # 对隐藏状态应用全连接层
        hidden_states = self.dropout(inputs=hidden_states, training=training)  # 对隐藏状态应用dropout
        hidden_states = hidden_states + input_tensor  # 将处理后的隐藏状态与输入张量相加

        return hidden_states  # 返回处理后的隐藏状态

    # 构建层的内部变量
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
                
# 从transformers.models.vit.modeling_tf_vit.TFViTLayer复制到ViT->ViTMAE
class TFViTMAELayer(tf.keras.layers.Layer):
    """这对应于timm实现中的Block类。"""

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 定义ViTMAEAttention、ViTMAEIntermediate和ViTMAEOutput
        self.attention = TFViTMAEAttention(config, name="attention")
        self.intermediate = TFViTMAEIntermediate(config, name="intermediate")
        self.vit_output = TFViTMAEOutput(config, name="output")

        # 创建LayerNormalization层
        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_before"
        )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layernorm_after"
        )
        self.config = config

    # 定义该层的前向传播逻辑
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 使用自注意力机制处理输入张量，在 ViTMAE 中在自注意力之前应用 layernorm
        attention_outputs = self.attention(
            input_tensor=self.layernorm_before(inputs=hidden_states),  # 将输入张量经过 layernorm 处理后传入自注意力
            head_mask=head_mask,  # 注意力头部遮罩
            output_attentions=output_attentions,  # 是否输出注意力权重
            training=training,  # 是否处于训练模式
        )
        attention_output = attention_outputs[0]  # 取出注意力输出

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 ViTMAE 中，在自注意力之后也要应用 layernorm
        layer_output = self.layernorm_after(inputs=hidden_states)  # 使用 layernorm 处理自注意力输出

        intermediate_output = self.intermediate(hidden_states=layer_output)  # 中间层处理

        # 第二个残差连接
        layer_output = self.vit_output(
            hidden_states=intermediate_output,  # 经过中间层处理的输出
            input_tensor=hidden_states,  # 输入张量
            training=training  # 是否处于训练模式
        )
        outputs = (layer_output,) + attention_outputs[1:]  # 如果输出注意力权重，添加到输出结果中

        return outputs  # 返回结果

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)  # 构建自注意力层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)  # 构建中间层
        if getattr(self, "vit_output", None) is not None:
            with tf.name_scope(self.vit_output.name):
                self.vit_output.build(None)  # 构建输出层
        if getattr(self, "layernorm_before", None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])  # 构建 layernorm 在自注意力之前的处理
        if getattr(self, "layernorm_after", None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])  # 构建 layernorm 在自注意力之后的处理
# 从transformers.models.vit.modeling_tf_vit.TFViTEncoder中复制代码并将ViT替换为ViTMAE
class TFViTMAEEncoder(tf.keras.layers.Layer):
    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建包含多个TFViTMAELayer的列表，用于构建Encoder层
        self.layer = [TFViTMAELayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]

    # 定义call方法，用于模型的前向传播
    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 根据是否输出隐藏状态和注意力矩阵初始化空元组
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 循环遍历每个layer_module，计算每层的输出
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力矩阵，则将当前层的注意力加入all_attentions
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态加入all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不是以返回字典形式返回结果，则返回隐藏状态、所有隐藏状态和所有注意力矩阵
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 以TFBaseModelOutput的格式返回结果
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 构建Encoder层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义TFViTMAEMainLayer类，继承自tf.keras.layers.Layer
@keras_serializable
class TFViTMAEMainLayer(tf.keras.layers.Layer):
    config_class = ViTMAEConfig

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        # 初始化TFViTMAEEmbeddings、TFViTMAEEncoder和LayerNormalization层
        self.embeddings = TFViTMAEEmbeddings(config, name="embeddings")
        self.encoder = TFViTMAEEncoder(config, name="encoder")
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")

    # 获取输入嵌入
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings.patch_embeddings

    # 剪枝模型的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 解包输入
    @unpack_inputs
    # 定义一个方法，接受像素值、噪音、头部掩码等参数，并返回模型输出
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        noise: tf.Tensor = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        # 调用嵌入层方法，获得嵌入输出，掩码和ID回复
        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values=pixel_values, training=training, noise=noise
        )

        # 准备头部掩码，如果需要
        # 在head_mask中的1.0表示保留头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            # 如果需要头部掩码，则抛出未实现错误
            raise NotImplementedError
        else:
            # 否则，创建一个列表用于存放None，列表的长度为 self.config.num_hidden_layers
            head_mask = [None] * self.config.num_hidden_layers

        # 使用头部掩码和其他参数，调用编码器方法
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获得编码器输出的序列
        sequence_output = encoder_outputs[0]
        # 对序列输出进行层标准化
        sequence_output = self.layernorm(inputs=sequence_output)

        # 如果不需要返回字典，则返回元组
        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        # 返回 TFViTMAEModelOutput 对象
        return TFViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 检查是否有嵌入层，如果有则构建
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 检查是否有编码器，如果有则构建
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 检查是否有层标准化层，如果有则构建
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])
class TFViTMAEPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTMAEConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"



VIT_MAE_START_DOCSTRING = r"""
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

    - a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`ViTMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
    # 接受像素值，可以是 numpy 数组、tf.Tensor、tf.Tensor 列表、字典等形式，
    # 每个示例的形状必须为 (batch_size, num_channels, height, width)
    pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):

    # 头部遮罩，用于屏蔽自注意力模块的选定头部。
    # 遮罩值选择在 [0, 1] 范围内：
    # - 1 表示头部**未被屏蔽**，
    # - 0 表示头部**被屏蔽**。
    head_mask (`np.ndarray` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*): 

    # 是否返回所有注意力层的注意力张量
    # 该参数仅在 eager 模式下可用，在图模式下将使用配置中的值
    output_attentions (`bool`, *optional*): 

    # 是否返回所有层的隐藏状态
    # 该参数仅在 eager 模式下可用，在图模式下将使用配置中的值
    output_hidden_states (`bool`, *optional`): 

    # 是否返回[`~file_utils.ModelOutput`]而不是普通元组
    # 该参数可以在 eager 模式下使用，在图模式下该值将始终设置为 True
    return_dict (`bool`, *optional`): 

    # 是否在训练模式下使用模型
    # 一些模块（例如 dropout 模块）在训练和评估之间具有不同的行为
    training (`bool`, *optional*, defaults to `False``): 
"""
Transformer 模型的一个简单示例，该模型输出原始隐藏状态而不带特定头部。

:param config: ViTMAE 模型的配置
:param inputs: TFViTMAEModel 的输入
:param kwargs: TFViTMAEModel 的其他参数

Returns:
    TFViTMAEModelOutput 或者 tf.Tensor 元组

Examples:


>>> from transformers import AutoImageProcessor, TFViTMAEModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
>>> model = TFViTMAEModel.from_pretrained("facebook/vit-mae-base")

>>> inputs = image_processor(images=image, return_tensors="tf")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state

"""
@add_start_docstrings(
    "The bare ViTMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_MAE_START_DOCSTRING,
)
class TFViTMAEModel(TFViTMAEPreTrainedModel):
    def __init__(self, config: ViTMAEConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 ViT 主层
        self.vit = TFViTMAEMainLayer(config, name="vit")

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.vit.get_input_embeddings()

    # 模型调用方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFViTMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        noise: tf.Tensor = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFViTMAEModelOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, TFViTMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = TFViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="tf")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```
        """
        # 调用 ViT 主层
        outputs = self.vit(
            pixel_values=pixel_values,
            noise=noise,
            head_mask=head_mask,
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
        if getattr(self, "vit", None) is not None:
            with tf.name_scope(self.vit.name):
                self.vit.build(None)


class TFViTMAEDecoder(tf.keras.layers.Layer):
    # 初始化函数，接受配置、补丁数量和其他关键字参数
    def __init__(self, config, num_patches, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 创建一个全连接层，用于解码器的嵌入
        self.decoder_embed = tf.keras.layers.Dense(config.decoder_hidden_size, name="decoder_embed")

        # 复制配置对象，修改其中一些参数
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        
        # 创建多个 ViT 解码层对象
        self.decoder_layers = [
            TFViTMAELayer(decoder_config, name=f"decoder_layers.{j}") for j in range(config.decoder_num_hidden_layers)
        ]
        
        # 创建一个层归一化层，用于解码器
        self.decoder_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="decoder_norm")
        
        # 创建一个全连接层，用于解码器的预测
        self.decoder_pred = tf.keras.layers.Dense(
            config.patch_size**2 * config.num_channels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="decoder_pred",
        )  # encoder to decoder
        # 保存配置和补丁数量
        self.config = config
        self.num_patches = num_patches
        
    # 构建函数，用于构建层的权重
    def build(self, input_shape=None):
        # 创建一个可训练的掩码标记
        self.mask_token = self.add_weight(
            shape=(1, 1, self.config.decoder_hidden_size),
            initializer=tf.random_normal_initializer(stddev=self.config.initializer_range),
            trainable=True,
            name="mask_token",
        )
        # 创建一个不可训练的解码器位置嵌入
        self.decoder_pos_embed = self.add_weight(
            shape=(1, self.num_patches + 1, self.config.decoder_hidden_size),
            initializer="zeros",
            trainable=False,
            name="decoder_pos_embed",
        )
        # 用特定的函数生成解码器位置嵌入并赋值
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            add_cls_token=True,
        )[None, ...]
        self.decoder_pos_embed.assign(decoder_pos_embed)

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在解码器嵌入、层归一化、预测等属性，则构建它们的权重
        if getattr(self, "decoder_embed", None) is not None:
            with tf.name_scope(self.decoder_embed.name):
                self.decoder_embed.build([None, None, self.config.hidden_size])
        if getattr(self, "decoder_norm", None) is not None:
            with tf.name_scope(self.decoder_norm.name):
                self.decoder_norm.build([None, None, self.config.decoder_hidden_size])
        if getattr(self, "decoder_pred", None) is not None:
            with tf.name_scope(self.decoder_pred.name):
                self.decoder_pred.build([None, None, self.config.decoder_hidden_size])
        if getattr(self, "decoder_layers", None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 调用函数，用于执行层的正向传播
    def call(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
        # embed tokens
        x = self.decoder_embed(hidden_states)  # 使用解码器嵌入层将隐藏状态转换为嵌入表示

        # append mask tokens to sequence
        mask_tokens = tf.tile(
            self.mask_token,
            (shape_list(x)[0], shape_list(ids_restore)[1] + 1 - shape_list(x)[1], 1),
        )
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # 将mask tokens添加到序列中
        x_ = tf.gather(x_, axis=1, batch_dims=1, indices=ids_restore)  # 取消打乱
        x = tf.concat([x[:, :1, :], x_], axis=1)  # 追加cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed  # 将位置嵌入添加到隐藏状态中

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):  # 遍历解码器层模块
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                head_mask=None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]  # 更新隐藏状态

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  # 收集self注意力

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # 收集所有隐藏状态

        hidden_states = self.decoder_norm(hidden_states)  # 对隐藏状态进行归一化处理

        # predictor projection
        logits = self.decoder_pred(hidden_states)  # 通过解码器预测层获得logits

        # remove cls token
        logits = logits[:, 1:, :]  # 移除cls token

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)  # 如果不返回字典类型，则以元组的形式返回结果
        return TFViTMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)  # 返回解码器输出
@add_start_docstrings(
    "The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.",
    VIT_MAE_START_DOCSTRING,
)
class TFViTMAEForPreTraining(TFViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化函数，接受一个配置参数，并调用父类的初始化方法
        self.config = config

        # 创建 ViT 主层对象，用于特征提取
        self.vit = TFViTMAEMainLayer(config, name="vit")
        # 创建 ViT-MAE 解码器对象，用于自监督预训练
        self.decoder = TFViTMAEDecoder(
            config,
            num_patches=self.vit.embeddings.num_patches,
            name="decoder",
        )

    def get_input_embeddings(self):
        # 返回 ViT 主层对象的输入嵌入
        return self.vit.get_input_embeddings()

    def _prune_heads(self, heads_to_prune):
        # 剪枝特定头部的注意力头
        raise NotImplementedError

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)` or `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        # 获取 patch 大小和通道数
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # 确保通道在最后的维度上
        if shape_list(pixel_values)[1] == num_channels:
            pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 断言：确保像素值具有平方大小
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1],
            shape_list(pixel_values)[2],
            message="Make sure the pixel values have a squared size",
        )
        # 断言：确保像素值的大小可被 patch 大小整除
        tf.debugging.assert_equal(
            shape_list(pixel_values)[1] % patch_size,
            0,
            message="Make sure the pixel values have a size that is divisible by the patch size",
        )
        # 断言：确保像素值的通道数与配置中设置的通道数相等
        tf.debugging.assert_equal(
            shape_list(pixel_values)[3],
            num_channels,
            message=(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            ),
        )

        # patch 化
        batch_size = shape_list(pixel_values)[0]
        num_patches_one_direction = shape_list(pixel_values)[2] // patch_size
        patchified_pixel_values = tf.reshape(
            pixel_values,
            (batch_size, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size, num_channels),
        )
        patchified_pixel_values = tf.einsum("nhpwqc->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = tf.reshape(
            patchified_pixel_values,
            (batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels),
        )
        return patchified_pixel_values
        def unpatchify(self, patchified_pixel_values):
            """
            Args:
                patchified_pixel_values (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                    Patchified pixel values.

            Returns:
                `tf.Tensor` of shape `(batch_size, height, width, num_channels)`:
                    Pixel values.
            """
            # 获取patch的大小和通道数
            patch_size, num_channels = self.config.patch_size, self.config.num_channels
            # 计算每个方向的补丁数量
            num_patches_one_direction = int(shape_list(patchified_pixel_values)[1] ** 0.5)
            # 检查
            tf.debugging.assert_equal(
                num_patches_one_direction * num_patches_one_direction,
                shape_list(patchified_pixel_values)[1],
                message="Make sure that the number of patches can be squared",
            )

            # 反向拆解
            batch_size = shape_list(patchified_pixel_values)[0]
            patchified_pixel_values = tf.reshape(
                patchified_pixel_values,
                (batch_size, num_patches_one_direction, num_patches_one_direction, patch_size, patch_size, num_channels),
            )
            patchified_pixel_values = tf.einsum("nhwpqc->nhpwqc", patchified_pixel_values)
            pixel_values = tf.reshape(
                patchified_pixel_values,
                (batch_size, num_patches_one_direction * patch_size, num_patches_one_direction * patch_size, num_channels),
            )
            return pixel_values

        def forward_loss(self, pixel_values, pred, mask):
            """
            Args:
                pixel_values (`tf.Tensor` of shape `(batch_size, height, width, num_channels)`):
                    Pixel values.
                pred (`tf.Tensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                    Predicted pixel values.
                mask (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                    Tensor indicating which patches are masked (1) and which are not (0).

            Returns:
                `tf.Tensor`: Pixel reconstruction loss.
            """
            target = self.patchify(pixel_values)
            # 如果进行像素值标准化
            if self.config.norm_pix_loss:
                mean = tf.reduce_mean(target, axis=-1, keepdims=True)
                var = tf.math.reduce_variance(target, axis=-1, keepdims=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            # 计算损失
            loss = (pred - target) ** 2
            loss = tf.reduce_mean(loss, axis=-1)  # [batch_size, num_patches], mean loss per patch

            loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)  # mean loss on removed patches
            loss = tf.reshape(loss, (1,))
            return loss

        @unpack_inputs
        @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=TFViTMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于调用模型，并设置输入参数的类型与默认值
    def call(
        self,
        pixel_values: TFModelInputType | None = None,  # 像素值的类型与默认值
        noise: tf.Tensor = None,  # 噪音张量
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部遮罩的类型与默认值
        output_attentions: Optional[bool] = None,  # 是否输出注意力矩阵的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典的布尔值
        training: bool = False,  # 是否为训练模式的布尔值
    ) -> Union[TFViTMAEForPreTrainingOutput, Tuple[tf.Tensor]]:  # 返回类型的注释
        r"""
        Returns:  # 方法的返回值说明
        Examples:  # 例子的说明
        ```python  # 代码块开始
        >>> from transformers import AutoImageProcessor, TFViTMAEForPreTraining  # 导入模型和处理器
        >>> from PIL import Image  # 导入图像处理库
        >>> import requests  # 导入请求库
        
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 图片的 URL
        >>> image = Image.open(requests.get(url, stream=True).raw)  # 获取并打开图片
        
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")  # 加载图像处理器
        >>> model = TFViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")  # 加载预训练模型
        
        >>> inputs = image_processor(images=image, return_tensors="pt")  # 处理图片并返回张量
        >>> outputs = model(**inputs)  # 使用模型进行推理
        >>> loss = outputs.loss  # 损失值
        >>> mask = outputs.mask  # 掩码
        >>> ids_restore = outputs.ids_restore  # 恢复的 ID
        ```"""  # 代码块结束
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 设置返回值的字典
        
        outputs = self.vit(  # 调用 Transformer 模型进行推理
            pixel_values=pixel_values,  # 像素值
            noise=noise,  # 噪音
            head_mask=head_mask,  # 头部遮罩
            output_attentions=output_attentions,  # 输出注意力矩阵
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            return_dict=return_dict,  # 返回值的字典
            training=training,  # 是否训练模式
        )

        latent = outputs.last_hidden_state  # 隐层状态
        ids_restore = outputs.ids_restore  # 恢复的 ID
        mask = outputs.mask  # 掩码

        decoder_outputs = self.decoder(latent, ids_restore)  # 解码器输出
        logits = decoder_outputs.logits  # 逻辑值

        loss = self.forward_loss(pixel_values, logits, mask)  # 计算损失值

        if not return_dict:  # 如果不返回字典
            output = (logits, mask, ids_restore) + outputs[2:]  # 输出结果
            return ((loss,) + output) if loss is not None else output  # 返回结果

        return TFViTMAEForPreTrainingOutput(  # 返回预训练模型的输出
            loss=loss,  # 损失值
            logits=logits,  # 逻辑值
            mask=mask,  # 掩码
            ids_restore=ids_restore,  # 恢复的 ID
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力矩阵
        )

    def build(self, input_shape=None):  # 定义构建模型的方法
        if self.built:  # 如果已经构建过
            return  # 直接返回
        self.built = True  # 设置为已构建
        if getattr(self, "vit", None) is not None:  # 如果存在 vit
            with tf.name_scope(self.vit.name):  # 使用 vit 的名称作用域
                self.vit.build(None)  # 构建 vit
        if getattr(self, "decoder", None) is not None:  # 如果存在 decoder
            with tf.name_scope(self.decoder.name):  # 使用 decoder 的名称作用域
                self.decoder.build(None)  # 构建 decoder
```