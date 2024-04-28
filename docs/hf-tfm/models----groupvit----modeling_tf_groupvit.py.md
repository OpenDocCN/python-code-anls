# `.\models\groupvit\modeling_tf_groupvit.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
""" TF 2.0 GroupViT 模型。"""

# 导入必要的库
from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

# 导入自定义的模块
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
    TFModelInputType,
    TFPreTrainedModel,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_tensorflow_probability_available,
    logging,
    replace_return_docstrings,
)
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 软依赖
if is_tensorflow_probability_available():
    try:
        import tensorflow_probability as tfp

        # 在第一次调用时，检查是否安装了兼容版本的 TensorFlow
        # TensorFlow Probability 依赖于最新稳定版本的 TensorFlow
        _ = tfp.distributions.Normal(loc=0.0, scale=1.0)
    except ImportError:
        logger.error(
            "GroupViT 模型无法使用，因为无法加载 `tensorflow_probability`。"
            "看起来您安装了错误版本的 `tensorflow_probability`。"
            "请尝试按照以下说明重新安装：https://github.com/tensorflow/probability。"
        )

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "nvidia/groupvit-gcc-yfcc"

# 预训练模型存档列表
TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/groupvit-gcc-yfcc",
    # 查看所有 GroupViT 模型：https://huggingface.co/models?filter=groupvit
]

# 定义一个大负数常量
LARGE_NEGATIVE = -1e8

# 从 transformers.models.bart.modeling_tf_bart._expand_mask 复制的函数
def _expand_mask(mask: tf.Tensor, tgt_len: Optional[int] = None):
    """
    将��意力掩码从 `[bsz, seq_len]` 扩展为 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tf.constant(1.0)
    mask = tf.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))
    # 返回一个计算结果，计算方式为 (one_cst - expanded_mask) * LARGE_NEGATIVE
# 对比损失函数，从 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html 改编而来
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    # 计算平均值
    return tf.math.reduce_mean(
        # 计算稀疏分类交叉熵
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


# 从 transformers.models.clip.modeling_tf_clip.clip_loss 复制并修改为 groupvit_loss
def groupvit_loss(similarity: tf.Tensor) -> tf.Tensor:
    # 计算标题损失
    caption_loss = contrastive_loss(similarity)
    # 计算图像损失
    image_loss = contrastive_loss(tf.transpose(similarity))
    # 返回标题损失和图像损失的平均值
    return (caption_loss + image_loss) / 2.0


def hard_softmax(logits: tf.Tensor, dim: int) -> tf.Tensor:
    y_soft = stable_softmax(logits, dim)
    # 直通
    index = tf.argmax(y_soft, dim)
    y_hard = tf.one_hot(
        index,
        depth=shape_list(logits)[dim],
        # TensorFlow 期望轴为 -1 或在 [0, 3) 之间。但收到的是 -2
        # 这就是为什么使用以下代码片段的原因。
        axis=range(len(shape_list(logits)))[dim],
        dtype=y_soft.dtype,
    )
    ret = y_hard - tf.stop_gradient(y_soft) + y_soft

    return ret


def gumbel_softmax(logits: tf.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> tf.Tensor:
    gumbel_dist = tfp.distributions.Gumbel(0.0, 1.0)
    gumbels = gumbel_dist.sample(tf.shape(logits), dtype=logits.dtype)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = stable_softmax(gumbels, dim)

    if hard:
        # 直通
        index = tf.argmax(y_soft, dim)
        y_hard = tf.one_hot(
            index,
            depth=shape_list(logits)[dim],
            # TensorFlow 期望轴为 -1 或在 [0, 3) 之间。但收到的是 -2
            # 这就是为什么使用以下代码片段的原因。
            axis=range(len(shape_list(logits)))[dim],
            dtype=y_soft.dtype,
        )
        ret = y_hard - tf.stop_gradient(y_soft) + y_soft
    else:
        # 重新参数化技巧
        ret = y_soft
    return ret


def resize_attention_map(attentions: tf.Tensor, height: int, width: int, align_corners: bool = False) -> tf.Tensor:
    """
    Args:
        attentions (`tf.Tensor`): 注意力图，形状为 [batch_size, groups, feat_height*feat_width]
        height (`int`): 输出注意力图的高度
        width (`int`): 输出注意力图的宽度
        align_corners (`bool`, *optional*): `nn.functional.interpolate` 的 `align_corner` 参数。

    Returns:
        `tf.Tensor`: 调整大小后的注意力图，形状为 [batch_size, groups, height, width]
    """

    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = shape_list(attentions)[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = shape_list(attentions)[2] // feat_height
    # 获取注意力张量的批量大小
    batch_size = shape_list(attentions)[0]
    # 获取注意力张量的分组数，即组合标记的数量
    groups = shape_list(attentions)[1]  # number of group token
    # 将注意力张量重塑为 [batch_size, groups, height, width] 的形状
    attentions = tf.reshape(attentions, (batch_size, groups, feat_height, feat_width))
    # 调换张量的维度顺序，变为 [batch_size, height, width, groups] 的形状
    attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
    # 如果 align_corners 为 True，则使用双线性插值调整注意力张量的大小
    if align_corners:
        attentions = tf.compat.v1.image.resize(
            attentions,
            size=(height, width),
            method="bilinear",
            align_corners=align_corners,
        )
    # 如果 align_corners 为 False，则使用双线性插值调整注意力张量的大小
    else:
        attentions = tf.image.resize(attentions, size=(height, width), method="bilinear")
    # 再次调换张量的维度顺序，变为 [batch_size, groups, height, width] 的形状
    attentions = tf.transpose(attentions, perm=(0, 3, 1, 2))
    # 返回调整后的注意力张量
    return attentions
# 从注意力矩阵中获取分组信息
def get_grouping_from_attentions(attentions: Tuple[tf.Tensor], hw_shape: Tuple[int]) -> tf.Tensor:
    """
    Args:
        attentions (`tuple(tf.Tensor)`: TFGroupViTVisionTransformer 返回的注意力图的元组
        hw_shape (`tuple(int)`): 输出注意力图的高度和宽度
    Returns:
        `tf.Tensor`: 形状为 [batch_size, groups, height, width] 的注意力图
    """

    attn_maps = []  # 存储每个时间步的注意力图
    prev_attn_masks = None  # 前一个时间步的注意力图
    for attn_masks in attentions:
        # 调整维度顺序，[batch_size, num_groups, height x width] -> [batch_size, height x width, num_groups]
        attn_masks = tf.transpose(attn_masks, perm=(0, 2, 1))
        if prev_attn_masks is None:
            prev_attn_masks = attn_masks
        else:
            prev_attn_masks = tf.matmul(prev_attn_masks, attn_masks)
        # 调整维度顺序，[batch_size, height x width, num_groups] -> [batch_size, num_groups, height x width] -> [batch_size, num_groups, height, width]
        cur_attn_map = resize_attention_map(tf.transpose(prev_attn_masks, perm=(0, 2, 1)), *hw_shape)
        attn_maps.append(cur_attn_map)

    # 最终的分组信息，形状为 [batch_size, num_groups, height, width]
    final_grouping = attn_maps[-1]

    return tf.stop_gradient(final_grouping)


@dataclass
class TFGroupViTModelOutput(ModelOutput):
    """
    TFGroupViTModelOutput 类，继承自 ModelOutput
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        segmentation_logits (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        text_embeds (`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTTextModel`].
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`TFGroupViTVisionModel`].
        text_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTTextModel`].
        vision_model_output (`TFBaseModelOutputWithPooling`):
            The output of the [`TFGroupViTVisionModel`].
    """

    # 初始化变量
    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    segmentation_logits: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    # 将当前对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果键不是"text_model_output"或"vision_model_output"，则返回当前键对应的值；否则返回对应属性的元组形式
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
class TFGroupViTCrossAttentionLayer(tf.keras.layers.Layer):
    # 定义 TFGroupViTCrossAttentionLayer 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        # 初始化函数，接受 GroupViTVisionConfig 类型的 config 参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        self.attn = TFGroupViTAttention(config, name="attn")
        # 创建 TFGroupViTAttention 对象，并赋值给 self.attn
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm2")
        # 创建 LayerNormalization 层，并赋值给 self.norm2
        self.mlp = TFGroupViTMLP(config, name="mlp")
        # 创建 TFGroupViTMLP 对象，并赋值给 self.mlp
        self.norm_post = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_post")
        # 创建 LayerNormalization 层，并赋值给 self.norm_post
        self.config = config
        # 将 config 参数赋值给 self.config

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义 call 方法，接受 query、key 和 training 参数，返回 tf.Tensor 类型的结果
        x = query
        # 将 query 赋值给 x
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        # 将 self.attn(query, encoder_hidden_states=key)[0] 的结果加到 x 上
        x = x + self.mlp(self.norm2(x))
        # 将 self.mlp(self.norm2(x)) 的结果加到 x 上
        x = self.norm_post(x)
        # 对 x 进行 LayerNormalization
        return x
        # 返回 x

    def build(self, input_shape=None):
        # 定义 build 方法，接受 input_shape 参数
        if self.built:
            return
        # 如果已经构建过，则直接返回
        self.built = True
        # 将 self.built 标记为 True
        if getattr(self, "attn", None) is not None:
            # 如果 self.attn 存在
            with tf.name_scope(self.attn.name):
                # 使用 self.attn 的名称创建命名空间
                self.attn.build(None)
                # 调用 self.attn 的 build 方法
        if getattr(self, "norm2", None) is not None:
            # 如果 self.norm2 存在
            with tf.name_scope(self.norm2.name):
                # 使用 self.norm2 的名称创建命名空间
                self.norm2.build([None, None, self.config.hidden_size])
                # 调用 self.norm2 的 build 方法
        if getattr(self, "mlp", None) is not None:
            # 如果 self.mlp 存在
            with tf.name_scope(self.mlp.name):
                # 使用 self.mlp 的名称创建命���空间
                self.mlp.build(None)
                # 调用 self.mlp 的 build 方法
        if getattr(self, "norm_post", None) is not None:
            # 如果 self.norm_post 存在
            with tf.name_scope(self.norm_post.name):
                # 使用 self.norm_post 的名称创建命名空间
                self.norm_post.build([None, None, self.config.hidden_size])
                # 调用 self.norm_post 的 build 方法

class TFGroupViTAssignAttention(tf.keras.layers.Layer):
    # 定义 TFGroupViTAssignAttention 类，继承自 tf.keras.layers.Layer
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        # 初始化函数，接受 GroupViTVisionConfig 类型的 config 参数和其他关键字参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        self.scale = config.hidden_size**-0.5
        # 计算 scale 值
        self.q_proj = tf.keras.layers.Dense(config.hidden_size, name="q_proj")
        # 创建 Dense 层，并赋值给 self.q_proj
        self.k_proj = tf.keras.layers.Dense(config.hidden_size, name="k_proj")
        # 创建 Dense 层，并赋值给 self.k_proj
        self.v_proj = tf.keras.layers.Dense(config.hidden_size, name="v_proj")
        # 创建 Dense 层，并赋值给 self.v_proj
        self.proj = tf.keras.layers.Dense(config.hidden_size, name="proj")
        # 创建 Dense 层，并赋值给 self.proj
        self.assign_eps = config.assign_eps
        # 将 config.assign_eps 赋值给 self.assign_eps
        self.config = config
        # 将 config 参数赋值给 self.config

    def get_attn(self, attn: tf.Tensor, gumbel: bool = True, hard: bool = True, training: bool = False) -> tf.Tensor:
        # 定义 get_attn 方法，接受 attn、gumbel、hard 和 training 参数，返回 tf.Tensor 类型的结果
        if gumbel and training:
            # 如果 gumbel 为 True 且 training 为 True
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
            # 调用 gumbel_softmax 函数对 attn 进行处理
        else:
            if hard:
                # 如果 hard 为 True
                attn = hard_softmax(attn, dim=-2)
                # 调用 hard_softmax 函数对 attn 进行处理
            else:
                attn = stable_softmax(attn, axis=-2)
                # 调用 stable_softmax 函数对 attn 进行处理
        return attn
        # 返回处理后的 attn
    # 定义一个方法，用于计算注意力机制
    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool = False):
        # 将 key 作为 value
        value = key
        # 对 query 进行投影，得到 [batch_size, query_length, channels] 的结果
        query = self.q_proj(query)

        # 对 key 进行投影，得到 [batch_size, key_length, channels] 的结果
        key = self.k_proj(key)

        # 对 value 进行投影，得到 [batch_size, key_length, channels] 的结果
        value = self.v_proj(value)

        # 计算原始注意力分数，得到 [batch_size, query_length, key_length] 的结果
        raw_attn = tf.matmul(query, key, transpose_b=True) * self.scale

        # 获取经过处理后的注意力分数
        attn = self.get_attn(raw_attn, training=training)
        # 获取经过处理后的软注意力分数
        soft_attn = self.get_attn(raw_attn, training=training, gumbel=False, hard=False)

        # 对注意力分数进行归一化处理
        attn = attn / (tf.math.reduce_sum(attn, axis=-1, keepdims=True) + self.assign_eps)

        # 根据注意力分数计算输出
        out = tf.matmul(attn, value)

        # 对输出进行投影
        out = self.proj(out)

        # 返回输出和软注意力分数
        return out, soft_attn

    # 构建注意力层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 检查并构建 q_proj
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.config.hidden_size])
        # 检查并构建 k_proj
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.config.hidden_size])
        # 检查并构建 v_proj
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.config.hidden_size])
        # 检查并构建 proj
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.config.hidden_size])
class TFGroupViTTokenAssign(tf.keras.layers.Layer):
    # 初始化函数，接受 GroupViTVisionConfig 对象、组 token 数量、输出组数量等参数
    def __init__(self, config: GroupViTVisionConfig, num_group_token: int, num_output_group: int, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 设置输出组数量
        self.num_output_group = num_output_group
        # 对组 token 进行归一化处理
        self.norm_tokens = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_tokens")
        # 计算 MLP 比率
        assign_mlp_ratio = (
            config.assign_mlp_ratio
            if isinstance(config.assign_mlp_ratio, collections.abc.Iterable)
            else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        )
        # 计算 tokens 维度和 channels 维度
        tokens_dim, channels_dim = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        # 创建 MLP 层
        self.mlp_inter = TFGroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group, name="mlp_inter")
        # 对输出组 token 进行归一化处理
        self.norm_post_tokens = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="norm_post_tokens"
        )
        # 对 x 进行归一化处理
        self.norm_x = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_x")
        # 创建预分配注意力层
        self.pre_assign_attn = TFGroupViTCrossAttentionLayer(config, name="pre_assign_attn")

        # 创建分配注意力层
        self.assign = TFGroupViTAssignAttention(config, name="assign")
        # 对新的 x 进行归一化处理
        self.norm_new_x = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="norm_new_x")
        # 创建 MLP 通道层
        self.mlp_channels = TFGroupViTMLP(
            config, config.hidden_size, channels_dim, config.hidden_size, name="mlp_channels"
        )
        # 保存配置信息
        self.config = config

    # 投影组 token
    def project_group_token(self, group_tokens: tf.Tensor) -> tf.Tensor:
        """
        Args:
            group_tokens (tf.Tensor): group tokens, [batch_size, num_group_tokens, channels]

        Returns:
            projected_group_tokens (tf.Tensor): [batch_size, num_output_groups, channels]
        """
        # 使用 MLP 层对组 token 进行投影
        projected_group_tokens = self.mlp_inter(group_tokens)
        # 对投影后的组 token 进行归一化处理
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens
    def call(self, image_tokens: tf.Tensor, group_tokens: tf.Tensor, training: bool = False):
        """
        Args:
            image_tokens (`tf.Tensor`): image tokens, of shape [batch_size, input_length, channels]
            group_tokens (`tf.Tensor`): group tokens, [batch_size, num_group_tokens, channels]
        """

        # 对组 tokens 进行规范化处理
        group_tokens = self.norm_tokens(group_tokens)
        # 对图像 tokens 进行规范化处理
        image_tokens = self.norm_x(image_tokens)
        # 投影组 tokens 到指定维度
        projected_group_tokens = self.project_group_token(group_tokens)
        # 在分配注意力之前对投影的组 tokens 进行预处理
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        # 分配注意力，得到新的图像 tokens 和注意力分布
        new_image_tokens, attention = self.assign(projected_group_tokens, image_tokens)
        # 将新的图像 tokens 和投影的组 tokens 相加
        new_image_tokens += projected_group_tokens

        # 对新的图像 tokens 进行通道 MLP 处理
        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))

        return new_image_tokens, attention

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建各个层
        if getattr(self, "norm_tokens", None) is not None:
            with tf.name_scope(self.norm_tokens.name):
                self.norm_tokens.build([None, None, self.config.hidden_size])
        if getattr(self, "mlp_inter", None) is not None:
            with tf.name_scope(self.mlp_inter.name):
                self.mlp_inter.build(None)
        if getattr(self, "norm_post_tokens", None) is not None:
            with tf.name_scope(self.norm_post_tokens.name):
                self.norm_post_tokens.build([None, None, self.config.hidden_size])
        if getattr(self, "norm_x", None) is not None:
            with tf.name_scope(self.norm_x.name):
                self.norm_x.build([None, None, self.config.hidden_size])
        if getattr(self, "pre_assign_attn", None) is not None:
            with tf.name_scope(self.pre_assign_attn.name):
                self.pre_assign_attn.build(None)
        if getattr(self, "assign", None) is not None:
            with tf.name_scope(self.assign.name):
                self.assign.build(None)
        if getattr(self, "norm_new_x", None) is not None:
            with tf.name_scope(self.norm_new_x.name):
                self.norm_new_x.build([None, None, self.config.hidden_size])
        if getattr(self, "mlp_channels", None) is not None:
            with tf.name_scope(self.mlp_channels.name):
                self.mlp_channels.build(None)
# 从transformers.models.vit.modeling_tf_vit.TFViTPatchEmbeddings适配而来，将ViT->GroupViT
class TFGroupViTPatchEmbeddings(tf.keras.layers.Layer):
    """
    这个类将形状为`(batch_size, num_channels, height, width)`的`pixel_values`转换为形状为`(batch_size, seq_length, hidden_size)`的初始隐藏状态（patch embeddings），以供Transformer消费。
    """

    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)
        image_size, patch_size = config.image_size, config.patch_size
        num_channels = config.num_channels
        # hidden_size是一个成员，因为在调用方法中会用到
        self.hidden_size = config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config

        self.projection = tf.keras.layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format="channels_last",
            use_bias=True,
            kernel_initializer=get_initializer(self.config.initializer_range),
            bias_initializer="zeros",
            name="projection",
        )

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    # 定义函数的返回类型为 TensorFlow 张量
    ) -> tf.Tensor:
        # 获取像素值的形状信息
        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 如果在即时执行模式下，并且通道数不匹配配置中设置的通道数，则引发值错误
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果不插值位置编码，并且在即时执行模式下，并且图像大小不匹配模型大小，则引发值错误
        if (
            not interpolate_pos_encoding
            and tf.executing_eagerly()
            and (height != self.image_size[0] or width != self.image_size[1])
        ):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # 在 CPU 上运行时，`tf.keras.layers.Conv2D` 不支持 `NCHW` 格式
        # 所以将输入格式从 `NCHW` 更改为 `NHWC`
        # 形状为 (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))

        # 对像素值进行投影
        projection = self.projection(pixel_values)

        # 将二维空间维度转换为单个时间维度
        # 形状为 (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        # 在 TFGroupViTVisionEmbeddings 中，此层的嵌入将进行层归一化
        # LayerNormalization 层需要具有静态的最后一个维度（否则测试 keras 保存加载将失败，因为有符号张量）
        # 这就是为什么在 reshape 方法中使用 hidden_size 的原因
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, num_patches, self.hidden_size))

        # 返回嵌入结果
        return embeddings

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在投影层，则构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
# 从 transformers.vit.modeling_tf_vit.TFViTEmbeddings 适配而来的 TFGroupViTVisionEmbeddings 类
class TFGroupViTVisionEmbeddings(tf.keras.layers.Layer):
    """
    构建位置和补丁嵌入。
    """

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化补丁嵌入
        self.patch_embeddings = TFGroupViTPatchEmbeddings(config, name="patch_embeddings")
        # 初始化丢弃层
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout, name="dropout")
        # 初始化层归一化
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        self.config = config

    def build(self, input_shape=None):
        # 获取补丁数量
        num_patches = self.patch_embeddings.num_patches
        # 添加位置嵌入权重
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches, self.config.hidden_size),
            initializer="zeros",
            trainable=True,
            name="position_embeddings",
        )

        if self.built:
            return
        self.built = True
        # 构建补丁嵌入
        if getattr(self, "patch_embeddings", None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        # 构建丢弃层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        # 构建层归一化
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        这个方法允许插值预训练的位置编码，以便在更高分辨率图像上使用模型。
        
        来源：
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, num_patches, dim = shape_list(embeddings)
        num_positions = shape_list(self.position_embeddings)[1]

        if num_patches == num_positions and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
            ),
            size=(h0, w0),
            method="bicubic",
        )
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return patch_pos_embed

    def call(
        self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool = False, training: bool = False
    ) -> tf.Tensor:
        # 获取输入张量的形状信息
        _, _, height, width = shape_list(pixel_values)
        # 将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        # 对嵌入向量进行 LayerNormalization
        embeddings = self.layernorm(embeddings)

        # 为每个令牌添加位置编码
        if interpolate_pos_encoding:
            # 如果需要插值位置编码，则将插值后的位置编码添加到嵌入向量中
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，将固定位置编码添加到嵌入向量中
            embeddings = embeddings + self.position_embeddings

        # 对嵌入向量进行 Dropout 操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入向量
        return embeddings
# 从transformers.models.clip.modeling_tf_clip.TFCLIPTextEmbeddings复制并修改为GroupViTTextEmbeddings
class TFGroupViTTextEmbeddings(tf.keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = config.hidden_size

        self.config = config

    def build(self, input_shape: tf.TensorShape = None):
        # 创建token embedding权重矩阵
        with tf.name_scope("token_embedding"):
            self.weight = self.add_weight(
                shape=(self.config.vocab_size, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="weight",
            )

        # 创建position embedding权重矩阵
        with tf.name_scope("position_embedding"):
            self.position_embedding = self.add_weight(
                shape=(self.config.max_position_embeddings, self.embed_dim),
                initializer=get_initializer(self.config.initializer_factor * self.config.initializer_range),
                trainable=True,
                name="embeddings",
            )

        super().build(input_shape)

    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
    ) -> tf.Tensor:
        """
        根据输入张量应用嵌入。

        返回:
            final_embeddings (`tf.Tensor`): 输出嵌入张量。
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        final_embeddings = inputs_embeds + position_embeds

        return final_embeddings


class TFGroupViTStage(tf.keras.layers.Layer):
    """这对应于GroupViT实现中的`GroupingLayer`类。"""

    def __init__(
        self,
        config: GroupViTVisionConfig,
        depth: int,
        num_prev_group_token: int,
        num_group_token: int,
        num_output_group: int,
        **kwargs,
    # 初始化函数，接受关键字参数并初始化实例属性
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置配置属性
        self.config = config
        # 设置深度属性
        self.depth = depth
        # 设置组标记数量属性
        self.num_group_token = num_group_token
        # 创建编码器层列表
        self.layers = [TFGroupViTEncoderLayer(config, name=f"layers_._{i}") for i in range(depth)]

        # 如果存在组标记数量
        if num_group_token > 0:
            # 创建下采样对象
            self.downsample = TFGroupViTTokenAssign(
                config=config,
                num_group_token=num_group_token,
                num_output_group=num_output_group,
                name="downsample",
            )
        else:
            self.downsample = None

        # 如果存在前一组标记数量和组标记数量
        if num_prev_group_token > 0 and num_group_token > 0:
            # 创建组投影器
            self.group_projector = [
                tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="group_projector.0"),
                TFGroupViTMixerMLP(
                    config, num_prev_group_token, config.hidden_size // 2, num_group_token, name="group_projector.1"
                ),
            ]
        else:
            self.group_projector = None

    # 构建方法
    def build(self, input_shape=None):
        # 如果存在组标记数量
        if self.num_group_token > 0:
            # 添加组标记权重
            self.group_token = self.add_weight(
                shape=(1, self.num_group_token, self.config.hidden_size),
                initializer="zeros",
                trainable=True,
                name="group_token",
            )
        else:
            self.group_token = None

        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 构建下采样对象
        if getattr(self, "downsample", None) is not None:
            with tf.name_scope(self.downsample.name):
                self.downsample.build(None)
        # 构建编码器层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        # 构建组投影器
        if getattr(self, "group_projector", None) is not None:
            with tf.name_scope(self.group_projector[0].name):
                self.group_projector[0].build([None, None, self.config.hidden_size])
            with tf.name_scope(self.group_projector[1].name):
                self.group_projector[1].build(None)

    # 返回是否存在组标记的属性
    @property
    def with_group_token(self):
        return self.group_token is not None

    # 分割输入张量
    def split_x(self, x: tf.Tensor) -> tf.Tensor:
        if self.with_group_token:
            return x[:, : -self.num_group_token], x[:, -self.num_group_token :]
        else:
            return x, None

    # 连接输入张量和组标记张量
    def concat_x(self, x: tf.Tensor, group_token: tf.Tensor | None = None) -> tf.Tensor:
        if group_token is None:
            return x
        return tf.concat([x, group_token], axis=1)

    # 调用方法，接受隐藏状态、前一组标记、是否输出注意力权重、是否训练等参数
    def call(
        self,
        hidden_states: tf.Tensor,
        prev_group_token: tf.Tensor | None = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the grouping tensors of Grouping block.
        """
        # 如果存在 group token，则复制并创建与 hidden_states 相同形状的 group token
        if self.with_group_token:
            group_token = tf.tile(self.group_token, multiples=(shape_list(hidden_states)[0], 1, 1))
            # 如果存在 group_projector，则对 group token 进行处理
            if self.group_projector is not None:
                for layer in self.group_projector:
                    prev_group_token = layer(prev_group_token)
                group_token = group_token + prev_group_token
        else:
            group_token = None

        x = hidden_states

        # 将 hidden_states 和 group_token 进行拼接
        cat_x = self.concat_x(x, group_token)
        # 遍历每个层并进行处理
        for layer in self.layers:
            # 对 cat_x 进行处理
            layer_out = layer(
                cat_x,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
            )
            cat_x = layer_out[0]

        # 将处理后的结果拆分为 x 和 group_token
        x, group_token = self.split_x(cat_x)

        attention = None
        # 如果存在 downsample，则对 x 和 group_token 进行下采样
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)

        outputs = (x, group_token)
        # 如果需要输出 attention，则将 attention 加入到输出中
        if output_attentions:
            outputs = outputs + (attention,)

        return outputs
class TFGroupViTMLP(tf.keras.layers.Layer):
    # 定义 TFGroupViTMLP 类，继承自 tf.keras.layers.Layer
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs,
    ):
        # 初始化函数，接受配置参数和可选的隐藏层大小、中间层大小、输出层大小
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        self.config = config
        # 保存配置参数
        self.activation_fn = get_tf_activation(config.hidden_act)
        # 获取激活函数
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        # 如果隐藏层大小不为空，则使用传入的值，否则使用配置参数中的值
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        # 如果中间层大小不为空，则使用传入的值，否则使用配置参数中的值
        output_size = output_size if output_size is not None else hidden_size
        # 如果输出层大小不为空，则使用传入的值，否则使用隐藏层大小
        self.fc1 = tf.keras.layers.Dense(intermediate_size, name="fc1")
        # 创建全连接层 fc1
        self.fc2 = tf.keras.layers.Dense(output_size, name="fc2")
        # 创建全连接层 fc2
        self.intermediate_size = intermediate_size
        # 保存中间层大小
        self.hidden_size = hidden_size
        # 保存隐藏层大小

    def call(self, hidden_states: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 定义 call 方法，接受隐藏状态和训练标志
        hidden_states = self.fc1(hidden_states)
        # 使用全连接层 fc1 处理隐藏状态
        hidden_states = self.activation_fn(hidden_states)
        # 使用激活函数处理隐藏状态
        hidden_states = self.fc2(hidden_states)
        # 使用全连接层 fc2 处理隐藏状态
        return hidden_states
        # 返回处理后的隐藏状态

    def build(self, input_shape=None):
        # 定义 build 方法，接受输入形状
        if self.built:
            return
        # 如果已经构建过，则直接返回
        self.built = True
        # 标记为已构建
        if getattr(self, "fc1", None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.hidden_size])
        # 如果存在 fc1 层，则构建 fc1 层
        if getattr(self, "fc2", None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.intermediate_size])
        # 如果存在 fc2 层，则构建 fc2 层


class TFGroupViTMixerMLP(TFGroupViTMLP):
    # 定义 TFGroupViTMixerMLP 类，继承自 TFGroupViTMLP
    def call(self, x, training: bool = False):
        # 定义 call 方法，接受输入 x 和训练标志
        x = super().call(hidden_states=tf.transpose(x, perm=(0, 2, 1)))
        # 调用父类的 call 方法，对输入进行转置操作
        return tf.transpose(x, perm=(0, 2, 1))
        # 返回转置后的结果


# Adapted from transformers.models.clip.modeling_tf_clip.TFCLIPAttention
class TFGroupViTAttention(tf.keras.layers.Layer):
    # 定义 TFGroupViTAttention 类，继承自 tf.keras.layers.Layer
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 多头注意力机制，来源于《Attention Is All You Need》论文
    # 初始化函数，接受配置和其他参数
    def __init__(self, config: GroupViTConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = self.embed_dim // self.num_attention_heads
        # 检查嵌入维度是否能被注意力头数量整除
        if self.attention_head_size * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_attention_heads})."
            )

        # 计算初始化标准差
        factor = config.initializer_factor
        in_proj_std = (self.embed_dim**-0.5) * ((2 * config.num_hidden_layers) ** -0.5) * factor
        out_proj_std = (self.embed_dim**-0.5) * factor

        # 计算注意力头大小的平方根
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        # 初始化查询、键、值的投影层
        self.q_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="q_proj"
        )
        self.k_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="k_proj"
        )
        self.v_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(in_proj_std), name="v_proj"
        )

        # 初始化 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_dropout)

        # 初始化输出投影层
        self.out_proj = tf.keras.layers.Dense(
            units=self.embed_dim, kernel_initializer=get_initializer(out_proj_std), name="out_proj"
        )

    # 从 [batch_size, seq_length, all_head_size] 转置为 [batch_size, seq_length, num_attention_heads, attention_head_size]
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        # 转置张量从 [batch_size, seq_length, num_attention_heads, attention_head_size] 到 [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 模型调用函数
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor = None,
        causal_attention_mask: tf.Tensor = None,
        output_attentions: bool = None,
        encoder_hidden_states: tf.Tensor = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """定义一个函数，接受隐藏状态和编码器隐藏状态作为输入，返回注意力输出和注意力权重"""

        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]
        # 判断是否为交叉注意力
        is_cross_attention = encoder_hidden_states is not None

        # 使用查询投影层处理隐藏状态
        mixed_query_layer = self.q_proj(inputs=hidden_states)
        if is_cross_attention:
            # 如果是交叉注意力，使用编码器隐藏状态进行键值投影
            mixed_key_layer = self.k_proj(inputs=encoder_hidden_states)
            mixed_value_layer = self.v_proj(inputs=encoder_hidden_states)
        else:
            # 否则，使用隐藏状态进行键值投影
            mixed_key_layer = self.k_proj(inputs=hidden_states)
            mixed_value_layer = self.v_proj(inputs=hidden_states)

        # 对查询、键、值进行维度转换
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # 计算原始注意力分数，即查询和键的点积
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # 先应用因果注意力掩码
        if causal_attention_mask is not None:
            # 应用因果注意力掩码
            attention_scores = tf.add(attention_scores, causal_attention_mask)

        if attention_mask is not None:
            # 应用注意力掩码
            attention_scores = tf.add(attention_scores, attention_mask)

        # 将注意力分数归一化为概率
        _attention_probs = stable_softmax(logits=attention_scores, axis=-1)

        # 对注意力概率进行dropout
        attention_probs = self.dropout(inputs=_attention_probs)

        # 计算注意力输出
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # 重塑注意力输出的形状
        # (batch_size, seq_len_q, embed_dim)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.embed_dim))

        # 使用输出投影层处理注意力输出
        attention_output = self.out_proj(attention_output)
        # 在TFBert中，注意力权重在dropout之后返回
        # 但在CLIP中，注意力权重在dropout之前返回
        outputs = (attention_output, _attention_probs) if output_attentions else (attention_output,)

        return outputs
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 q_proj 属性，则构建 q_proj
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        # 如果存在 k_proj 属性，则构建 k_proj
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        # 如果存在 v_proj 属性，则构建 v_proj
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        # 如果存在 out_proj 属性，则构建 out_proj
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])
# 从transformers.models.clip.modeling_tf_clip.TFCLIPEncoderLayer复制代码，并将CLIP->GroupViT
class TFGroupViTEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化编码器层，设置隐藏层大小为config中的hidden_size
        self.embed_dim = config.hidden_size
        # 初始化自注意力层
        self.self_attn = TFGroupViTAttention(config, name="self_attn")
        # 初始化LayerNormalization层1
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        # 初始化MLP层
        self.mlp = TFGroupViTMLP(config, name="mlp")
        # 初始化LayerNormalization层2
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            causal_attention_mask (`tf.Tensor`): causal attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`):
                Whether or not to return the attentions tensors of all attention layers. See `outputs` under returned
                tensors for more detail.
        """
        residual = hidden_states

        # 对输入的hidden_states进行LayerNormalization
        hidden_states = self.layer_norm1(inputs=hidden_states)
        # 调用自注意力层
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            training=training,
        )
        hidden_states = attention_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        # 对hidden_states进行LayerNormalization
        hidden_states = self.layer_norm2(inputs=hidden_states)
        # 调用MLP层
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + attention_outputs[1:]  # 如果输出注意力，则添加注意力

        return outputs
    # 构建模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在 self_attn 属性，则构建 self_attn
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        # 如果存在 layer_norm1 属性，则构建 layer_norm1
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, self.embed_dim])
        # 如果存在 mlp 属性，则构建 mlp
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        # 如果存在 layer_norm2 属性，则构建 layer_norm2
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, self.embed_dim])
# 定义 TFGroupViTTextEncoder 类，继承自 tf.keras.layers.Layer
class TFGroupViTTextEncoder(tf.keras.layers.Layer):
    # 初始化函数，接受 GroupViTTextConfig 类型的 config 参数
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建 TFGroupViTEncoderLayer 对象列表，根据 config.num_hidden_layers 的数量
        self.layers = [TFGroupViTEncoderLayer(config, name=f"layers_._{i}") for i in range(config.num_hidden_layers)]

    # call 方法，接受多个参数，返回 Union[Tuple, TFBaseModelOutput] 类型的结果
    def call(
        self,
        hidden_states,
        attention_mask: tf.Tensor,
        causal_attention_mask: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 初始化 encoder_states 和 all_attentions，根据 output_hidden_states 和 output_attentions 的值
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 遍历 self.layers 列表
        for idx, encoder_layer in enumerate(self.layers):
            # 如果 output_hidden_states 为 True，则将 hidden_states 添加到 encoder_states 中
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # 调用 encoder_layer 的 call 方法，更新 hidden_states
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            # 如果 output_attentions 为 True，则将当前层的注意力矩阵添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果 output_hidden_states 为 True，则将最终的 hidden_states 添加到 encoder_states 中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回非空的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 如果 return_dict 为 True，则返回 TFBaseModelOutput 对象
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # build 方法，构建层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 遍历 self.layers 列表，构建每个层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)


# 定义 TFGroupViTVisionEncoder 类，继承自 tf.keras.layers.Layer
class TFGroupViTVisionEncoder(tf.keras.layers.Layer):
    # 初始化函数，接受 GroupViTVisionConfig 类型的 config 参数
    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建 TFGroupViTStage 对象列表，根据 config.depths 的数量
        self.stages = [
            TFGroupViTStage(
                config=config,
                depth=config.depths[i],
                num_group_token=config.num_group_tokens[i],
                num_output_group=config.num_output_groups[i],
                num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0,
                name=f"stages_._{i}",
            )
            for i in range(len(config.depths))
        ]

    # call 方法，接受多个参数，返回 Union[Tuple, TFBaseModelOutput] 类型的结果
    def call(
        self,
        hidden_states: tf.Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        return_dict: bool,
        training: bool = False,
    # 定义函数的返回类型为元组或 TFBaseModelOutput 类型
    ) -> Union[tuple, TFBaseModelOutput]:
        # 如果不输出隐藏状态，则初始化为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化为空元组
        all_groupings = () if output_attentions else None

        # 初始化 group_tokens 为 None
        group_tokens = None

        # 遍历模型的各个阶段
        for stage in self.stages:
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前阶段的处理函数，获取该阶段的输出
            layer_outputs = stage(hidden_states, group_tokens, output_attentions)

            # 更新隐藏状态为当前阶段的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 更新 group_tokens 为当前阶段的输出的第二个元素
            group_tokens = layer_outputs[1]

            # 如果输出注意力权重且当前阶段的输出中包含注意力权重，则将其添加到 all_groupings 中
            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回包含隐藏状态、所有隐藏状态和所有注意力权重的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None)
        # 返回 TFBaseModelOutput 对象，包含最终隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在阶段，则为每个阶段构建模型
        if getattr(self, "stages", None) is not None:
            for layer in self.stages:
                # 在命名空间下构建每个阶段的模型
                with tf.name_scope(layer.name):
                    layer.build(None)
# 从transformers.models.clip.modeling_tf_clip.TFCLIPTextTransformer复制代码，并将CLIPText->GroupViTText，CLIPEncoder->GroupViTTextEncoder
class TFGroupViTTextTransformer(tf.keras.layers.Layer):
    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化GroupViTTextEmbeddings层
        self.embeddings = TFGroupViTTextEmbeddings(config, name="embeddings")
        # 初始化GroupViTTextEncoder层
        self.encoder = TFGroupViTTextEncoder(config, name="encoder")
        # 初始化LayerNormalization层，用于最终的层归一化
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="final_layer_norm"
        )

        # 用于计算`pooled_output`的结束标记ID
        self.eos_token_id = config.eos_token_id
        # 嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size

    def call(
        self,
        input_ids: TFModelInputType,
        attention_mask: tf.Tensor,
        position_ids: tf.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        # 获取输入张量的形状
        input_shape = shape_list(input_ids)

        # 使用嵌入层处理输入的 token ids 和位置 ids
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        batch_size, seq_length = input_shape
        # CLIP 的文本模型使用因果注意力掩码，这里准备它
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)

        # 检查注意力掩码并反转
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask)

        # 将嵌入输出传入编码器
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出的序列输出
        sequence_output = encoder_outputs[0]
        # 应用最终的层归一化
        sequence_output = self.final_layer_norm(inputs=sequence_output)

        if self.eos_token_id == 2:
            # 在 PR #24773 之前，`eos_token_id` 是不正确的：让我们保留这里的更改
            # 具有这种 `eos_token_id` 的 CLIP 模型在配置中无法正确处理额外添加的新标记
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, n_ctx, transformer.width]
            # 从 eot 嵌入中获取特征（eot_token 是每个序列中的最高数字）
            pooled_output = tf.gather_nd(
                params=sequence_output,
                indices=tf.stack(
                    values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(input_ids, axis=-1)), axis=1
                ),
            )
        else:
            # 配置从 PR #24773 中更新了 `eos_token_id`（因此可以使用额外的新标记）
            pooled_output = tf.gather_nd(
                params=sequence_output,
                indices=tf.stack(
                    values=(
                        tf.range(input_shape[0], dtype=tf.int64),
                        tf.math.argmax(tf.cast(input_ids == self.eos_token_id, dtype=tf.int8), axis=-1),
                    ),
                    axis=1,
                ),
            )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 构建因果注意力掩码，用于在自注意力机制中屏蔽未来信息
    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        # 如果 seq_length 是运行时值，tf.constant 不支持，使用 tf.fill 处理动态形状
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)

        # 创建一个二维加性注意力掩码，所有位置都被屏蔽
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)

        # 将对角线和下三角部分设置为0（即不屏蔽的位置）
        # 提示：将二维矩阵视为（查询序列，键序列）的空间
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        # to_mask = tf.linalg.band_part(to_mask, -1, 0)
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)

        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在最终层归一化，则构建最终层归一化
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPVisionTransformer 适配而来的 TFGroupViTVisionTransformer 类
class TFGroupViTVisionTransformer(tf.keras.layers.Layer):
    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)

        # 初始化 TFGroupViTVisionEmbeddings 对象
        self.embeddings = TFGroupViTVisionEmbeddings(config, name="embeddings")
        # 初始化 TFGroupViTVisionEncoder 对象
        self.encoder = TFGroupViTVisionEncoder(config, name="encoder")
        # 初始化 LayerNormalization 层
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layernorm")
        # 获取隐藏层大小
        self.embed_dim = config.hidden_size

    def call(
        self,
        pixel_values: TFModelInputType,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[Tuple, TFBaseModelOutputWithPooling]:
        # 获取嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 获取编码器输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 对最后一个隐藏状态进行归一化
        last_hidden_state = self.layernorm(last_hidden_state)
        # 对最后一个隐藏状态进行平均池化
        pooled_output = tf.math.reduce_mean(last_hidden_state, axis=1)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return TFBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "layernorm", None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.embed_dim])


@keras_serializable
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPTextMainLayer 复制而来，将 CLIP->GroupViT
class TFGroupViTTextMainLayer(tf.keras.layers.Layer):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 初始化 TFGroupViTTextTransformer 对象
        self.text_model = TFGroupViTTextTransformer(config, name="text_model")

    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    # 定义一个方法，用于调用模型
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入的文本序列的标识符
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 注意力掩码，指示哪些标记需要被关注
        position_ids: np.ndarray | tf.Tensor | None = None,  # 位置标识符
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
        training: bool = False,  # 是否处于训练模式
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:  # 返回值的类型注解
        # 如果输入的文本序列标识符为空，则抛出数值错误
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # 获取输入的文本序列的形状
        input_shape = shape_list(input_ids)

        # 如果注意力掩码为空，则用值为1填充
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 调用文本模型，传入相应参数
        text_model_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回文本模型的输出
        return text_model_outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在文本模型，则构建文本模型
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
# 使用 keras_serializable 装饰器标记该类可序列化
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPVisionMainLayer 复制代码，并将 CLIP->GroupViT
class TFGroupViTVisionMainLayer(tf.keras.layers.Layer):
    # 设置配置类为 GroupViTVisionConfig
    config_class = GroupViTVisionConfig

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        # 创建 TFGroupViTVisionTransformer 对象作为 vision_model
        self.vision_model = TFGroupViTVisionTransformer(config, name="vision_model")

    # 获取输入嵌入层
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.vision_model.embeddings

    # 调用函数，接收像素值、是否输出注意力、是否输出隐藏状态、是否返回字典、训练标志，返回 TFBaseModelOutputWithPooling 或 tf.Tensor 元组
    @unpack_inputs
    def call(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用 vision_model 处理像素值等参数
        vision_model_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return vision_model_outputs

    # 构建函数，如果已构建则直接返回，否则构建 vision_model
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)


# 使用 keras_serializable 装饰器标记该类可序列化
# 从 transformers.models.clip.modeling_tf_clip.TFCLIPMainLayer 调整而来
class TFGroupViTMainLayer(tf.keras.layers.Layer):
    # 设置配置类为 GroupViTConfig
    config_class = GroupViTConfig
    # 初始化函数，接受配置参数和其他关键字参数
    def __init__(self, config: GroupViTConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 检查文本配置是否为 GroupViTTextConfig 类型
        if not isinstance(config.text_config, GroupViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type GroupViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查视觉配置是否为 GroupViTVisionConfig 类型
        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type GroupViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将配置参数保存到对象中
        self.config = config

        # 获取文本配置和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度和中间维度
        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建文本模型和视觉模型
        self.text_model = TFGroupViTTextTransformer(text_config, name="text_model")
        self.vision_model = TFGroupViTVisionTransformer(vision_config, name="vision_model")

        # 创建视觉投影层
        self.visual_projection = [
            tf.keras.layers.Dense(self.projection_intermediate_dim, name="visual_projection.0"),
            tf.keras.layers.BatchNormalization(name="visual_projection.1", momentum=0.9, epsilon=1e-5),
            tf.keras.layers.ReLU(name="visual_projection.2"),
            tf.keras.layers.Dense(self.projection_dim, name="visual_projection.3"),
        ]
        # 创建文本投影层
        self.text_projection = [
            tf.keras.layers.Dense(self.projection_intermediate_dim, name="text_projection.0"),
            tf.keras.layers.BatchNormalization(name="text_projection.1", momentum=0.9, epsilon=1e-5),
            tf.keras.layers.ReLU(name="text_projection.2"),
            tf.keras.layers.Dense(self.projection_dim, name="text_projection.3"),
        ]
    # 构建模型，设置输入形状
    def build(self, input_shape=None):
        # 添加可训练的 logit_scale 参数
        self.logit_scale = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.config.logit_scale_init_value),
            trainable=True,
            name="logit_scale",
        )

        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在文本模型，则构建文本模型
        if getattr(self, "text_model", None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        # 如果存在视觉模型，则构建视觉模型
        if getattr(self, "vision_model", None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        # 如果存在视觉投影，则构建视觉投影
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection[0].name):
                self.visual_projection[0].build([None, None, None, self.vision_embed_dim])
            with tf.name_scope(self.visual_projection[1].name):
                self.visual_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.visual_projection[3].name):
                self.visual_projection[3].build([None, None, None, self.projection_intermediate_dim])
        # 如果存在文本投影，则构建文本投影
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection[0].name):
                self.text_projection[0].build([None, None, None, self.text_embed_dim])
            with tf.name_scope(self.text_projection[1].name):
                self.text_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.text_projection[3].name):
                self.text_projection[3].build([None, None, None, self.projection_intermediate_dim])

    # 获取文本特征
    @unpack_inputs
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 如果没有输入文本序列，则抛出数值错误
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = shape_list(input_ids)

        # 如果没有指定注意力掩码，则使用全为1的注意力掩码
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 获取文本模型的输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取池化输出
        pooled_output = text_outputs[1]
        # 对文本投影中的每一层进行处理
        for layer in self.text_projection:
            pooled_output = layer(pooled_output)

        # 返回文本特征
        text_features = pooled_output
        return text_features

    @unpack_inputs
    # 获取图像特征的方法，接受像素值、是否输出注意力、隐藏状态、返回字典、训练标志等参数
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 如果像素值为空，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用视觉模型获取视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取池化后的输出
        pooled_output = vision_outputs[1]
        # 对视觉投影层进行迭代处理
        for layer in self.visual_projection:
            pooled_output = layer(pooled_output)

        # 图像特征为池化后的输出
        image_features = pooled_output
        return image_features

    # 调用方法，接受输入ID、像素值、注意力掩码、位置ID、返回损失、输出注意力、隐藏状态、分割等参数
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        pixel_values: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_segmentation: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
class TFGroupViTPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 GroupViTConfig
    config_class = GroupViTConfig
    # 设置基础模型前缀为 "groupvit"


GROUPVIT_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
    first positional argument :

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
      `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
      `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Args:
        config ([`GroupViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GROUPVIT_TEXT_INPUTS_DOCSTRING = r"""
# 空白，没有代码，用于后续添加注释
    # 定义函数参数说明
    Args:
        # 输入序列标记的索引，可以是 np.ndarray、tf.Tensor、List[tf.Tensor]、Dict[str, tf.Tensor] 或 Dict[str, np.ndarray] 类型，
        # 每个示例必须具有形状为({0})的形状
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        # 避免在填充标记索引上执行注意力的掩码。掩码值选择在[0, 1]之间：
        # - 1表示**未屏蔽**的标记，
        # - 0表示**已屏蔽**的标记。
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        # 输入序列标记在位置嵌入中的每个位置的索引。选择范围为[0, config.max_position_embeddings - 1]。
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            
            [What are position IDs?](../glossary#position-ids)
        # 是否返回所有注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        # 是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        # 是否返回一个`~utils.ModelOutput`而不是一个普通元组
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        # 是否在训练模式下使用模型
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

# 定义了一个文档字符串，用于描述输入参数的含义和可选参数
GROUPVIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
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
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""

# 定义了一个文档字符串，用于描述输入参数的含义和可选参数
GROUPVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` ``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.
            [What are input IDs?](../glossary#input-ids)
        
        pixel_values (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
        
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
        
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
        
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
        
        training (`bool`, *optional*, defaults to `False``):
            Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
# 定义 TFGroupViTTextModel 类，继承自 TFGroupViTPreTrainedModel
class TFGroupViTTextModel(TFGroupViTPreTrainedModel):
    # 设置配置类为 GroupViTTextConfig
    config_class = GroupViTTextConfig
    # 设置主要输入名称为 "input_ids"

    def __init__(self, config: GroupViTTextConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFGroupViTTextMainLayer 实例，命名为 groupvit
        self.groupvit = TFGroupViTTextMainLayer(config, name="groupvit")

    # 定义 call 方法，接收输入参数并返回模型输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, TFGroupViTTextModel

        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = TFGroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        # 调用 groupvit 的方法，传入参数并获取输出
        outputs = self.groupvit(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型输出
        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 groupvit 属性，则在���名空间下构建 groupvit
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)


# 定义 TFGroupViTVisionModel 类，继承自 TFGroupViTPreTrainedModel
class TFGroupViTVisionModel(TFGroupViTPreTrainedModel):
    # 设置配置类为 GroupViTVisionConfig
    config_class = GroupViTVisionConfig
    # 设置主要输入名称为 "pixel_values"

    def __init__(self, config: GroupViTVisionConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFGroupViTVisionMainLayer 实例，命名为 groupvit
        self.groupvit = TFGroupViTVisionMainLayer(config, name="groupvit")

    # 定义 call 方法，接收输入参数并返回模型输出
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    # 定义一个方法，用于调用模型
    def call(
        self,
        pixel_values: TFModelInputType | None = None,  # 像素值，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
        training: bool = False,  # 是否处于训练模式，默认为False
    ) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        r"""
        Returns: 返回值说明

        Examples: 示例代码

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTVisionModel

        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = TFGroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        # 调用groupvit模型，传入参数
        outputs = self.groupvit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回模型输出
        return outputs

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在groupvit模型，则构建该模型
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)
# 定义 TFGroupViTModel 类，继承自 TFGroupViTPreTrainedModel
@add_start_docstrings(GROUPVIT_START_DOCSTRING)
class TFGroupViTModel(TFGroupViTPreTrainedModel):
    # 指定配置类为 GroupViTConfig
    config_class = GroupViTConfig

    # 初始化方法，接受 GroupViTConfig 类型的配置参数
    def __init__(self, config: GroupViTConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 创建 TFGroupViTMainLayer 实例，传入配置参数和名称
        self.groupvit = TFGroupViTMainLayer(config, name="groupvit")

    # 装饰器，将输入解包并添加文档字符串到模型前向方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 获取文本特征的方法
    def get_text_features(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFGroupViTTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, TFGroupViTModel

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""

        # 调用 TFGroupViTMainLayer 实例的 get_text_features 方法，传入参数并获取文本特征
        text_features = self.groupvit.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 返回文本特征
        return text_features

    # 装饰器，将输入解包并添加文档字符串到模型前向方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    # 获取图像特征的方法
    def get_image_features(
        self,
        pixel_values: TFModelInputType | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> tf.Tensor:
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFGroupViTVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTModel

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""

        # 获取图像特征，通过将像素值应用于投影层到 TFGroupViTVisionModel 的池化输出
        image_features = self.groupvit.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GROUPVIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFGroupViTModelOutput, config_class=GroupViTConfig)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        pixel_values: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_segmentation: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFGroupViTModelOutput, Tuple[tf.Tensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFGroupViTModel
        >>> import tensorflow as tf

        >>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.math.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""

        # 调用 groupvit 方法进行模型推理
        outputs = self.groupvit(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_segmentation=output_segmentation,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    # 用于服务端输出
    def serving_output(self, output: TFGroupViTModelOutput) -> TFGroupViTModelOutput:
        # TODO: As is this currently fails with saved_model=True, because
        # TensorFlow cannot trace through nested dataclasses. Reference:
        # https://github.com/huggingface/transformers/pull/16886
        return output

    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 groupvit 属性，则构建 groupvit 模型
        if getattr(self, "groupvit", None) is not None:
            with tf.name_scope(self.groupvit.name):
                self.groupvit.build(None)
```