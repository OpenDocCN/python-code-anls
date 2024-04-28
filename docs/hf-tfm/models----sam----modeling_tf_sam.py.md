# `.\transformers\models\sam\modeling_tf_sam.py`

```py
# 该文件主要用于定义 TensorFlow 版本的自注意力掩码 (SAM) 模型
# 该文件大部分是通过从 PyTorch 版本自动翻译生成的，如果有任何不一致，以 PyTorch 版本为准

# 引入必要的库
from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义文档中使用的常量
_CONFIG_FOR_DOC = "SamConfig"
_CHECKPOINT_FOR_DOC = "facebook/sam-vit-huge"
TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam-vit-base",
    # See all SAM models at https://huggingface.co/models?filter=sam
]

# 定义 TFSamVisionEncoderOutput 类，用于保存 SAM 视觉编码器的输出
@dataclass
class TFSamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.
    """
    Args:
        image_embeds (`tf.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义变量image_embeds，类型为tf.Tensor或None，默认值为None，表示图像嵌入（image embeddings）
    image_embeds: tf.Tensor | None = None 
    # 定义变量last_hidden_state，类型为tf.Tensor，表示模型最后一层的隐藏状态序列
    last_hidden_state: tf.Tensor = None
    # 定义变量hidden_states，类型为元组Union[tf.Tensor]或None，默认值为None，表示模型每一层的隐藏状态序列
    hidden_states: Tuple[tf.Tensor] | None = None
    # 定义变量attentions，类型为元组Union[tf.Tensor]或None，默认值为None，表示模型每一层的注意力权重矩阵
    attentions: Tuple[tf.Tensor] | None = None
from dataclasses import dataclass
from typing import Tuple
import tensorflow as tf

@dataclass
class TFSamImageSegmentationOutput(ModelOutput):
    """
    Segment-Anything 模型输出的基类

    Args:
        iou_scores (`tf.Tensor` of shape `(batch_size, num_masks)`):
            预测掩码的 IoU 分数。
        pred_masks (`tf.Tensor` of shape `(batch_size, num_masks, height, width)`):
            预测的低分辨率掩码。需要由处理器进行后处理。
        vision_hidden_states  (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config
    # 初始化方法，接受配置信息和其他参数
    def __init__(self, config, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 从配置信息中获取图片尺寸和路径尺寸
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置信息中获取通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size
        # 如果图片尺寸和路径尺寸不是可迭代的对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像划分成块的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 保存计算结果到对象属性中
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用卷积层构建投影
        self.projection = tf.keras.layers.Conv2D(
            hidden_size, kernel_size=patch_size, strides=patch_size, name="projection"
        )

    # 前向传播方法，接受像素值作为输入，返回嵌入向量
    def call(self, pixel_values):
        # 获取像素值的形状
        batch_size, num_channels, height, width = shape_list(pixel_values)
        # 检查通道数是否与配置中的一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 检查输入图片的尺寸是否与配置中的一致
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # 对像素值进行投影，得到嵌入向量
        embeddings = self.projection(tf.transpose(pixel_values, perm=[0, 2, 3, 1]))
        return embeddings

    # 构建方法，用于构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记已经构建
        self.built = True
        # 如果投影层存在，则构建投影层
        if getattr(self, "projection", None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
class TFSamMLPBlock(tf.keras.layers.Layer):
    # 初始化MLPBlock层
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建第一个全连接层
        self.lin1 = tf.keras.layers.Dense(config.mlp_dim, name="lin1")
        # 创建第二个全连接层
        self.lin2 = tf.keras.layers.Dense(config.hidden_size, name="lin2")
        # 选择激活函数
        self.act = ACT2FN[config.hidden_act]
        # 保存配置
        self.config = config

    # 前向传播函数
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 第一层全连接
        hidden_states = self.lin1(hidden_states)
        # 激活函数
        hidden_states = self.act(hidden_states)
        # 第二层全连接
        hidden_states = self.lin2(hidden_states)
        return hidden_states

    # 构建层
    def build(self, input_shape=None):
        # 如果已经构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果lin1存在，则构建lin1层
        if getattr(self, "lin1", None) is not None:
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.config.hidden_size])
        # 如果lin2存在，则构建lin2层
        if getattr(self, "lin2", None) is not None:
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.config.mlp_dim])


class TFSamLayerNorm(tf.keras.layers.Layer):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """
    
    # 初始化LayerNorm层
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        # 初始化epsilon
        self.eps = eps
        # 初始化数据格式
        self.data_format = data_format
        # 初始化标准化形状
        self.normalized_shape = normalized_shape
        # 如果数据格式不是"channels_last"或"channels_first"，则抛出异常
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")

    # 构建层
    def build(self, input_shape):
        # 添加权重参数，初始化为1
        self.weight = self.add_weight(shape=self.normalized_shape, initializer="ones", name="weight")
        # 添加偏置参数，初始化为0
        self.bias = self.add_weight(shape=self.normalized_shape, initializer="zeros", name="bias")
        super().build(input_shape)

    # 前向传播函数
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # 如果数据格式为"channels_last"，则使用functional_layernorm对最后一个维度进行标准化
        if self.data_format == "channels_last":
            x = functional_layernorm(x, weight=self.weight, bias=self.bias, epsilon=self.eps, axis=-1)
        # 如果数据格式为"channels_first"，则使用functional_layernorm对第二个维度进行标准化
        elif self.data_format == "channels_first":
            x = functional_layernorm(x, weight=self.weight, bias=self.bias, epsilon=self.eps, axis=1)
        return x


class TFSamAttention(tf.keras.layers.Layer):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """
```  
    # 这是一个自注意力层的实现
    def __init__(self, config, downsample_rate=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
    
        # 如果未指定下采样率，则使用配置中的attention_downsample_rate，否则使用传入的downsample_rate
        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate
    
        # 计算内部维度，即隐藏层大小除以下采样率
        self.internal_dim = config.hidden_size // downsample_rate
        # 获取注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 如果内部维度无法被注意力头数量整除，则抛出ValueError异常
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")
    
        # 定义查询、键、值和输出的线性层
        self.q_proj = tf.keras.layers.Dense(self.internal_dim, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.internal_dim, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.internal_dim, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(self.hidden_size, name="out_proj")
    
    # 将隐藏状态分成多个注意力头
    def _separate_heads(self, hidden_states: tf.Tensor, num_attention_heads: int) -> tf.Tensor:
        # 获取批次大小、点批次大小、标记数量和通道数
        batch, point_batch_size, n_tokens, channel = shape_list(hidden_states)
        # 计算每个注意力头的通道数
        c_per_head = channel // num_attention_heads
        # 将隐藏状态重塑成 (batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        hidden_states = tf.reshape(
            hidden_states, (batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        )
        # 将最后两个维度交换位置，得到 (batch * point_batch_size, num_attention_heads, n_tokens, c_per_head)
        return tf.transpose(hidden_states, perm=[0, 2, 1, 3])
    
    # 将注意力头合并回原始形状
    def _recombine_heads(self, hidden_states: tf.Tensor, point_batch_size: int) -> tf.Tensor:
        # 获取批次大小、注意力头数量、标记数量和每个头的通道数
        batch, n_heads, n_tokens, c_per_head = shape_list(hidden_states)
        # 将最后两个维度交换位置，得到 (batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        hidden_states = tf.transpose(hidden_states, perm=[0, 2, 1, 3])
        # 将隐藏状态重塑成 (batch // max(1, point_batch_size), point_batch_size, n_tokens, n_heads * c_per_head)
        return tf.reshape(
            hidden_states,
            (batch // tf.reduce_max([1, point_batch_size]), point_batch_size, n_tokens, n_heads * c_per_head),
        )
    
    # 前向传播
    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        # 将输入通过线性层映射到内部维度
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
    
        # 获取点批次大小
        point_batch_size = shape_list(query)[1]
    
        # 将输入分成多个注意力头
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)
    
        # 计算注意力权重
        _, _, _, c_per_head = shape_list(query)
        attn = tf.matmul(
            query, tf.transpose(key, perm=[0, 1, 3, 2])
        )  # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens
        attn = attn / tf.math.sqrt(float(c_per_head))
        attn = tf.nn.softmax(attn, axis=-1)
    
        # 根据注意力权重计算输出
        out = tf.matmul(attn, value)
        # 将输出合并回原始形状
        out = self._recombine_heads(out, point_batch_size)
        # 通过最后一个线性层输出
        out = self.out_proj(out)
    
        return out
    
    
    这个代码实现了一个自注意力层。主要步骤包括:
    1. 初始化时获取配置参数,如隐藏层大小、注意力头数量等,并定义相关的线性层。
    2. 实现 `_separate_heads` 和 `_recombine_heads` 方法,用于将输入分成多个注意力头以及将注意力头输出合并回原始形状。
    3. 在 `call` 方法中,首先将输入通过线性层映射到内部维度,然后分成多个注意力头计算注意力权重,最后根据注意力权重计算输出并通过最后一个线性层输出。
    # 构建输入层
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果有 q_proj 层，则构建它
        if getattr(self, "q_proj", None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.hidden_size])
        # 如果有 k_proj 层，则构建它
        if getattr(self, "k_proj", None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.hidden_size])
        # 如果有 v_proj 层，则构建它
        if getattr(self, "v_proj", None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.hidden_size])
        # 如果有 out_proj 层，则构建它
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.internal_dim])
class TFSamTwoWayAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False, **kwargs):
        """
        初始化函数，定义一个具有四个层的Transformer块，包括：(1)稀疏输入的自注意力 (2)稀疏输入到密集输入的交叉注意力 
        (3)稀疏输入上的MLP块 (4)密集输入到稀疏输入的交叉注意力

        参数:
            config (`SamMaskDecoderConfig`):
                用于实例化该块的配置文件
            attention_downsample_rate (*可选*, int, 默认为2):
                用于降低注意力内部维度的块的下采样比率
            skip_first_layer_pe (*可选*, bool, 默认为 `False`):
                是否跳过在第一层添加查询点嵌入
        """
        super().__init__(**kwargs)

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        # 创建自注意力层
        self.self_attn = TFSamAttention(config, downsample_rate=1, name="self_attn")
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name="layer_norm1")

        # 创建从令牌到图像的交叉注意力层
        self.cross_attn_token_to_image = TFSamAttention(
            config, downsample_rate=attention_downsample_rate, name="cross_attn_token_to_image"
        )
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name="layer_norm2")

        # 创建MLP块
        self.mlp = TFSamMLPBlock(config, name="mlp")
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name="layer_norm3")

        self.layer_norm4 = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name="layer_norm4")

        # 创建从图像到令牌的交叉注意力层
        self.cross_attn_image_to_token = TFSamAttention(
            config, downsample_rate=attention_downsample_rate, name="cross_attn_image_to_token"
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def call(
        self,
        queries: tf.Tensor,
        keys: tf.Tensor,
        query_point_embedding: tf.Tensor,
        key_point_embedding: tf.Tensor,
        output_attentions: bool = False,
    ):
        # Self-attention block where the input attends to itself
        if self.skip_first_layer_pe:
            # If positional embeddings are skipped, directly apply self-attention
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            # If not skipped, add positional embedding to the queries and apply self-attention
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            # Add the self-attention output back to the original queries
            queries = queries + attn_out
        # Normalize the output with the first layer normalization
        queries = self.layer_norm1(queries)

        # Cross-attention block for token-to-image interaction
        query = queries + query_point_embedding  # Add positional embedding to the queries
        key = keys + key_point_embedding  # Add positional embedding to the keys

        # Apply cross-attention, where tokens attend to the image embeddings
        attn_out = self.cross_attn_token_to_image(query=query, key=key, value=keys)
        # Add the cross-attention output back to the original queries
        queries = queries + attn_out

        # Normalize the output with the second layer normalization
        queries = self.layer_norm2(queries)

        # Multi-Layer Perceptron (MLP) block for additional processing
        mlp_out = self.mlp(queries)  # Apply the MLP on the queries
        # Add the MLP output back to the original queries
        queries = queries + mlp_out
        # Normalize the output with the third layer normalization
        queries = self.layer_norm3(queries)

        # Cross-attention block for image-to-token interaction
        query = queries + query_point_embedding  # Add positional embedding to the queries
        key = keys + key_point_embedding  # Add positional embedding to the keys

        # Apply cross-attention, where image embeddings attend to the tokens
        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        # Add the cross-attention output back to the original keys
        keys = keys + attn_out

        # Normalize the output with the fourth layer normalization
        keys = self.layer_norm4(keys)

        # Prepare the final output tuple with queries and keys
        outputs = (queries, keys)

        # If attention outputs are requested, include them in the result tuple
        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)  # Otherwise, include a placeholder None

        # Return the final output tuple
        return outputs

    def build(self, input_shape=None):
        # Early exit if the build method has already been called
        if self.built:
            return
        # Mark the build method as having been called
        self.built = True
        
        # Check and build the self-attention layer if it exists
        if getattr(self, "self_attn", None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)

        # Check and build the first layer normalization if it exists
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, None, self.hidden_size])
        
        # Check and build the cross-attention from token to image if it exists
        if getattr(self, "cross_attn_token_to_image", None) is not None:
            with tf.name_scope(self.cross_attn_token_to_image.name):
                self.cross_attn_token_to_image.build(None)
        
        # Check and build the second layer normalization if it exists
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, None, self.hidden_size])

        # Check and build the Multi-Layer Perceptron (MLP) if it exists
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        
        # Check and build the third layer normalization if it exists
        if getattr(self, "layer_norm3", None) is not None:
            with tf.name_scope(self.layer_norm3.name):
                self.layer_norm3.build([None, None, None, self.hidden_size])

        # Check and build the fourth layer normalization if it exists
        if getattr(self, "layer_norm4", None) is not None:
            with tf.name_scope(self.layer_norm4.name):
                self.layer_norm4.build([None, None, None, self.hidden_size])

        # Check and build the cross-attention from image to token if it exists
        if getattr(self, "cross_attn_image_to_token", None) is not None:
            with tf.name_scope(self.cross_attn_image_to_token.name):
                self.cross_attn_image_to_token.build(None)
class TFSamTwoWayTransformer(tf.keras.layers.Layer):
    def __init__(self, config: SamMaskDecoderConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers  # 存储配置中的隐藏层数量
        self.layers = []  # 初始化存储层对象的列表

        # 遍历隐藏层数量，创建并添加多个两向注意力块层对象到层列表中
        for i in range(self.num_hidden_layers):
            self.layers.append(TFSamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0), name=f"layers_._{i}"))

        # 创建最终的注意力层对象，用于点到图像的注意力
        self.final_attn_token_to_image = TFSamAttention(config, name="final_attn_token_to_image")
        # 创建最终的层标准化层对象，用于最终的注意力层的输出
        self.layer_norm_final_attn = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layer_norm_final_attn"
        )

    def call(
        self,
        point_embeddings: tf.Tensor,
        image_embeddings: tf.Tensor,
        image_positional_embeddings: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TFBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_attentions = ()  # 初始化存储所有注意力权重的元组

        # 如果图像嵌入为空，则引发错误
        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        # 将图像嵌入和图像位置嵌入进行转置和展平，以备用于注意力计算
        image_embeddings = tf.transpose(flatten(image_embeddings, 2), perm=(0, 2, 1))[:, None]
        image_positional_embeddings = tf.transpose(flatten(image_positional_embeddings, 2), (0, 2, 1))[:, None]

        # 准备查询向量
        queries = point_embeddings
        keys = image_embeddings

        # 应用Transformer块和最终层标准化
        for layer in self.layers:
            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)  # 将每个注意力权重输出添加到存储中

        # 应用从点到图像的最终注意力层
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)  # 应用最终的层标准化
        return queries, keys, all_attentions  # 返回查询、键、所有注意力权重
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过则直接返回
        if self.built:
            return
        # 设置 built 标志为 True，表示模型已经构建
        self.built = True
        # 如果存在 final_attn_token_to_image 属性，则在 final_attn_token_to_image 作用域下构建它
        if getattr(self, "final_attn_token_to_image", None) is not None:
            with tf.name_scope(self.final_attn_token_to_image.name):
                self.final_attn_token_to_image.build(None)
        # 如果存在 layer_norm_final_attn 属性，则在 layer_norm_final_attn 作用域下构建它，并指定输入形状
        if getattr(self, "layer_norm_final_attn", None) is not None:
            with tf.name_scope(self.layer_norm_final_attn.name):
                self.layer_norm_final_attn.build([None, None, None, self.config.hidden_size])
        # 遍历 layers 列表，在每个 layer 的作用域下构建它
        for layer in self.layers:
            with tf.name_scope(layer.name):
                layer.build(None)
class TFSamFeedForward(tf.keras.layers.Layer):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False, **kwargs
    ):
        # 初始化函数，定义了神经网络的结构和参数
        super().__init__(**kwargs)
        self.num_layers = num_layers  # 神经网络的层数
        self.activation = tf.keras.layers.ReLU()  # 激活函数为 ReLU
        self.proj_in = tf.keras.layers.Dense(hidden_dim, input_shape=(input_dim,), name="proj_in")  # 输入层到隐藏层的全连接层
        self.proj_out = tf.keras.layers.Dense(output_dim, input_shape=(hidden_dim,), name="proj_out")  # 隐藏层到输出层的全连接层
        self.layers = [
            tf.keras.layers.Dense(hidden_dim, input_shape=(hidden_dim,), name=f"layers_._{i}")
            for i in range(num_layers - 2)
        ]  # 中间隐藏层
        self.sigmoid_output = sigmoid_output  # 输出是否经过 sigmoid 激活
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.input_dim = input_dim  # 输入维度

    def call(self, hidden_states):
        # 前向传播函数，定义了神经网络的计算过程
        hidden_states = self.proj_in(hidden_states)  # 输入层到隐藏层的计算
        hidden_states = self.activation(hidden_states)  # 使用激活函数激活隐藏层输出
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))  # 多层隐藏层的计算

        hidden_states = self.proj_out(hidden_states)  # 隐藏层到输出层的计算
        if self.sigmoid_output:
            hidden_states = tf.sigmoid(hidden_states)  # 如果输出需要 sigmoid 激活，则进行激活操作
        return hidden_states  # 返回神经网络的输出

    def build(self, input_shape=None):
        # 构建函数，用于构建神经网络的结构
        if self.built:
            return
        self.built = True  # 标记神经网络已构建
        if getattr(self, "proj_in", None) is not None:
            with tf.name_scope(self.proj_in.name):
                self.proj_in.build([None, None, self.input_dim])  # 构建输入层到隐藏层的全连接层
        if getattr(self, "proj_out", None) is not None:
            with tf.name_scope(self.proj_out.name):
                self.proj_out.build([None, None, self.hidden_dim])  # 构建隐藏层到输出层的全连接层
        if getattr(self, "layers", None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build([None, None, self.hidden_dim])  # 构建多层隐藏层
```py  
    # 初始化函数，接收 SamMaskDecoderConfig 配置和其他关键字参数
    def __init__(self, config: SamMaskDecoderConfig, **kwargs):
        # 调用父类构造函数
        super().__init__(**kwargs)
    
        # 从配置中获取隐藏层大小
        self.hidden_size = config.hidden_size
    
        # 获取多个 mask 输出的数量和 mask 令牌的数量
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1
    
        # 创建一个名为 "transformer" 的 TFSamTwoWayTransformer 对象
        self.transformer = TFSamTwoWayTransformer(config, name="transformer")
    
        # 创建两个反向卷积层，用于上采样
        self.upscale_conv1 = tf.keras.layers.Conv2DTranspose(
            self.hidden_size // 4, kernel_size=2, strides=2, name="upscale_conv1", data_format="channels_first"
        )
        self.upscale_conv2 = tf.keras.layers.Conv2DTranspose(
            self.hidden_size // 8, kernel_size=2, strides=2, name="upscale_conv2", data_format="channels_first"
        )
    
        # 创建一个 TFSamLayerNorm 层，用于上采样
        self.upscale_layer_norm = TFSamLayerNorm(
            self.hidden_size // 4, data_format="channels_first", name="upscale_layer_norm"
        )
    
        # 激活函数选择为 gelu 函数
        self.activation = tf.nn.gelu
    
        # 创建多个 feedforward 网络，组成 output_hypernetworks_mlps 列表
        mlps_list = []
        for i in range(self.num_mask_tokens):
            mlps_list += [
                TFSamFeedForward(
                    self.hidden_size,
                    self.hidden_size,
                    self.hidden_size // 8,
                    3,
                    name=f"output_hypernetworks_mlps_._{i}",
                )
            ]
        self.output_hypernetworks_mlps = mlps_list
    
        # 创建一个预测头，用于预测 IOU
        self.iou_prediction_head = TFSamFeedForward(
            self.hidden_size,
            config.iou_head_hidden_dim,
            self.num_mask_tokens,
            config.iou_head_depth,
            name="iou_prediction_head",
        )
    
    # 构建函数，用于构建模型的参数
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
    
        # 添加一个形状为 (1, hidden_size) 的权重，作为 IOU 令牌
        self.iou_token = self.add_weight(shape=(1, self.hidden_size), name="iou_token.weight", trainable=True)
        
        # 添加形状为 (num_mask_tokens, hidden_size) 的权重，作为 mask 令牌
        self.mask_tokens = self.add_weight(
            shape=(self.num_mask_tokens, self.hidden_size), name="mask_tokens.weight", trainable=True
        )
    
        # 若存在 transformer 层，则构建它
        if getattr(self, "transformer", None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
    
        # 若存在 upscale_conv1 层，则构建它
        if getattr(self, "upscale_conv1", None) is not None:
            with tf.name_scope(self.upscale_conv1.name):
                self.upscale_conv1.build([None, self.hidden_size, None, None])
    
        # 若存在 upscale_conv2 层，则构建它
        if getattr(self, "upscale_conv2", None) is not None:
            with tf.name_scope(self.upscale_conv2.name):
                self.upscale_conv2.build([None, self.hidden_size // 4, None, None])
    
        # 若存在 upscale_layer_norm 层，则构建它
        if getattr(self, "upscale_layer_norm", None) is not None:
            with tf.name_scope(self.upscale_layer_norm.name):
                self.upscale_layer_norm.build(None)
    
        # 若存在 iou_prediction_head 层，则构建它
        if getattr(self, "iou_prediction_head", None) is not None:
            with tf.name_scope(self.iou_prediction_head.name):
                self.iou_prediction_head.build(None)
    
        # 遍历 output_hypernetworks_mlps 列表，构建每个 feedforward 网络
        for mlp in self.output_hypernetworks_mlps:
            with tf.name_scope(mlp.name):
                mlp.build(None)
    # 定义一个方法，用于调用模型
    def call(
        self,
        # 图像嵌入向量，是一个张量
        image_embeddings: tf.Tensor,
        # 图像位置嵌入向量，是一个张量
        image_positional_embeddings: tf.Tensor,
        # 稀疏提示嵌入向量，是一个张量
        sparse_prompt_embeddings: tf.Tensor,
        # 密集提示嵌入向量，是一个张量
        dense_prompt_embeddings: tf.Tensor,
        # 是否生成多掩码输出，布尔值
        multimask_output: bool,
        # 输出注意力信息的选择性参数，默认为 None
        output_attentions: Optional[bool] = None,
class TFSamPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        # 初始化函数，设置缩放比例和配置信息
        super().__init__(**kwargs)
        self.scale = config.hidden_size // 2
        self.config = config

    def build(self, input_shape):
        # 构建函数，添加位置编码的权重矩阵
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(2, self.config.num_pos_feats),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.scale),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = tf.identity(input_coords)

        if input_shape is not None:
            # 将坐标归一化到[0,1]，并组成坐标矩阵
            coordinates = tf.stack(
                [
                    tf.cast(coordinates[:, :, :, 0], tf.float32) / input_shape[1],
                    tf.cast(coordinates[:, :, :, 1], tf.float32) / input_shape[0],
                ],
                axis=-1,
            )

        # 对坐标进行缩放和翻转，并与位置编码矩阵相乘
        coordinates = 2 * coordinates - 1
        coordinates = tf.cast(coordinates, self.positional_embedding.dtype)
        coordinates = tf.matmul(coordinates, self.positional_embedding)
        coordinates = 2 * np.pi * coordinates
        # 输出正弦和余弦的坐标编码
        # 输出形状为 d_1 x ... x d_n x 2 * num_pos_feats
        return tf.concat([tf.sin(coordinates), tf.cos(coordinates)], axis=-1)


class TFSamMaskEmbedding(tf.keras.layers.Layer):
    def __init__(self, config: SamPromptEncoderConfig, **kwargs):
        # 初始化函数，设置遮罩输入通道数、激活函数和卷积层等信息
        super().__init__(**kwargs)
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = tf.keras.layers.Conv2D(self.mask_input_channels, kernel_size=2, strides=2, name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(config.mask_input_channels, kernel_size=2, strides=2, name="conv2")
        self.conv3 = tf.keras.layers.Conv2D(config.hidden_size, kernel_size=1, name="conv3")
        self.layer_norm1 = TFSamLayerNorm(self.mask_input_channels, config.layer_norm_eps, name="layer_norm1")
        self.layer_norm2 = TFSamLayerNorm(self.mask_input_channels * 4, config.layer_norm_eps, name="layer_norm2")
        self.config = config
    def call(self, masks):
        # 将输入的 masks 转置，变成通道在最后的格式
        masks = tf.transpose(masks, perm=(0, 2, 3, 1))  # Convert to channels-last
        # 经过第一个卷积层
        hidden_states = self.conv1(masks)
        # 对结果进行 Layer Normalization 处理
        hidden_states = self.layer_norm1(hidden_states)
        # 使用激活函数处理结果
        hidden_states = self.activation(hidden_states)

        # 经过第二个卷积层
        hidden_states = self.conv2(hidden_states)
        # 对结果进行 Layer Normalization 处理
        hidden_states = self.layer_norm2(hidden_states)
        # 使用激活函数处理结果
        hidden_states = self.activation(hidden_states)
        
        # 经过第三个卷积层
        dense_embeddings = self.conv3(hidden_states)
        # 将结果转置，变成通道在最前的格式
        dense_embeddings = tf.transpose(dense_embeddings, perm=(0, 3, 1, 2))  # Convert back to channels-first
        # 返回处理后的结果
        return dense_embeddings

    def build(self, input_shape=None):
        # 因为不是通过标准的虚拟输入调用，需要显式调用 build 方法
        # 如果已经构建过网络结构则直接返回
        if self.built:
            return
        self.built = True
        with tf.name_scope("conv1"):
            # 构建第一层卷积层
            self.conv1.build([None, None, None, 1])
        with tf.name_scope("conv2"):
            # 构建第二层卷积层
            self.conv2.build([None, None, None, self.mask_input_channels])
        with tf.name_scope("conv3"):
            # 构建第三层卷积层
            self.conv3.build([None, None, None, self.mask_input_channels * 4])
        with tf.name_scope("layer_norm1"):
            # 构建第一个 Layer Normalization 层
            self.layer_norm1.build([None, None, None, self.mask_input_channels])
        with tf.name_scope("layer_norm2"):
            # 构建第二个 Layer Normalization 层
            self.layer_norm2.build([None, None, None, self.mask_input_channels * 4])
# TFSamPromptEncoder 是一个 TensorFlow Keras 图层,用于对图像提示进行编码
class TFSamPromptEncoder(tf.keras.layers.Layer):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding, **kwargs):
        super().__init__(**kwargs)
        # 初始化共享的 patch 嵌入层
        self.shared_embedding = shared_patch_embedding
        # 初始化掩码嵌入层
        self.mask_embed = TFSamMaskEmbedding(config, name="mask_embed")
        self.no_mask_embed = None

        # 设置图像嵌入尺寸和输入图像尺寸
        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        self.point_embed = []
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = None
        self.config = config

    def build(self, input_shape=None):
        # 创建没有掩码的嵌入向量
        self.no_mask_embed = self.add_weight(
            name="no_mask_embed.weight",
            shape=(1, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )
        # 创建多个点嵌入向量
        self.point_embed = [
            self.add_weight(
                name=f"point_embed_._{i}.weight",
                shape=(1, self.hidden_size),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                trainable=True,
            )
            for i in range(self.config.num_point_embeddings)
        ]
        # 创建不是点的嵌入向量
        self.not_a_point_embed = self.add_weight(
            name="not_a_point_embed.weight",
            shape=(1, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )
        # 构建掩码嵌入层
        with tf.name_scope("mask_embed"):
            self.mask_embed.build(
                (None, self.config.mask_input_channels, self.config.image_size, self.config.image_size)
            )

        # 如果已经构建过了,直接返回
        if self.built:
            return
        self.built = True
        # 为掩码嵌入层构建网络
        if getattr(self, "mask_embed", None) is not None:
            with tf.name_scope(self.mask_embed.name):
                self.mask_embed.build(None)
    def _embed_points(self, points: tf.Tensor, labels: tf.Tensor, pad: bool) -> tf.Tensor:
        """Embeds point prompts."""
        # 将点移到像素中心
        points = points + 0.5  
        if pad:
            #创建填充后的点形状和标签形状
            target_point_shape = (shape_list(points)[0], shape_list(points)[1], 1, shape_list(points)[-1])
            target_labels_shape = (shape_list(points)[0], shape_list(points)[1], 1)
            #创建填充点和标签
            padding_point = tf.zeros(target_point_shape, dtype=points.dtype)
            padding_label = -tf.ones(target_labels_shape, dtype=labels.dtype)
            #连接原始点和填充点
            points = tf.concat([points, padding_point], axis=2)
            labels = tf.concat([labels, padding_label], axis=2)
        input_shape = (self.input_image_size, self.input_image_size)
        #共享嵌入点
        point_embedding = self.shared_embedding(points, input_shape)

        # 替换不是点的值为not_a_point_embed[0]
        point_embedding = tf.where(labels[..., None] == -1, self.not_a_point_embed[0], point_embedding)

        # 替换特定值为0的点（标签!= -10）
        point_embedding = tf.where(
            labels[..., None] != -10,
            point_embedding,
            tf.zeros_like(point_embedding),
        )
        # 替换标签为0的点为point_embed[0]，使用逐点加法
        point_embedding = tf.where(
            (labels == 0)[:, :, :, None], point_embedding + self.point_embed[0], point_embedding
        )
        # 替换标签为1的点为point_embed[1]，使用逐点加法
        point_embedding = tf.where(
            (labels == 1)[:, :, :, None], point_embedding + self.point_embed[1], point_embedding
        )
        return point_embedding

    def _embed_boxes(self, boxes: tf.Tensor) -> tf.Tensor:
        """Embeds box prompts."""
        # 将框移到像素中心
        boxes = boxes + 0.5  
        batch_size, nb_boxes = shape_list(boxes)[:2]
        coords = tf.reshape(boxes, (batch_size, nb_boxes, 2, 2))
        input_shape = (self.input_image_size, self.input_image_size)
        #共享嵌入角落
        corner_embedding = self.shared_embedding(coords, input_shape)
        #使用条件表达式为每个点添加角落嵌入
        corner_embedding += tf.where(
            tf.range(shape_list(corner_embedding)[2])[None, None, :, None] == 0,
            self.point_embed[2][0],
            self.point_embed[3][0],
        )
        return corner_embedding

    def call(
        self,
        batch_size: Optional[int],
        input_points: Optional[Tuple[tf.Tensor, tf.Tensor]],
        input_labels: tf.Tensor | None,
        input_boxes: tf.Tensor | None,
        input_masks: tf.Tensor | None,
    # 定义一个方法，用于嵌入不同类型的提示，并返回稀疏和密集嵌入
    def embed(self, input_points: tf.Tensor = None, input_boxes: tf.Tensor = None, input_masks: tf.Tensor = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            input_points (`tf.Tensor`, *optional*):
                point coordinates and labels to embed.
            input_boxes (`tf.Tensor`, *optional*):
                boxes to embed
            input_masks (`tf.Tensor`, *optional*):
                masks to embed
        """

        sparse_embeddings = None
        # 如果存在输入的点坐标
        if input_points is not None:
            # 获取批量大小和点坐标的批量大小
            batch_size, point_batch_size = shape_list(input_points)[:2]
            # 如果输入标签为None，则引发值错误
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            # 调用_embed_points方法嵌入点坐标
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            # 创建一个形状为(batch_size, point_batch_size, 0, self.hidden_size)的零稀疏嵌入
            sparse_embeddings = tf.zeros(
                (batch_size, point_batch_size, 0, self.hidden_size), dtype=point_embeddings.dtype
            )
            # 将嵌入的点坐标添加到稀疏嵌入中
            sparse_embeddings = tf.concat([sparse_embeddings, point_embeddings], axis=2)
        
        # 如果存在输入的框
        if input_boxes is not None:
            # 获取批量大小
            batch_size = shape_list(input_boxes)[0]
            # 调用_embed_boxes方法嵌入框
            box_embeddings = self._embed_boxes(input_boxes)
            # 如果稀疏嵌入为空，则将框嵌入设置为稀疏嵌入
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                # 否则将框嵌入与稀疏嵌入连接起来
                sparse_embeddings = tf.concat([sparse_embeddings, box_embeddings], axis=2)
        
        # 如果存在输入的口罩
        if input_masks is not None:
            # 使用mask_embed方法嵌入密集嵌入
            dense_embeddings = self.mask_embed(input_masks)
        else:
            # 否则使用no_mask_embed的第一个元素作为密集嵌入
            dense_embeddings = self.no_mask_embed[0]
            # 重新调整密集嵌入的形状
            dense_embeddings = tf.reshape(dense_embeddings, (1, -1, 1, 1))
            # 使用瓷砖复制密集嵌入以匹配batch_size和图像嵌入大小
            dense_embeddings = tf.tile(
                dense_embeddings, (batch_size, 1, self.image_embedding_size[0], self.image_embedding_size[1])
            )
        
        # 如果稀疏嵌入为空，则创建一个形状为(batch_size, 0, 1, self.hidden_size)的零稀疏嵌入
        if sparse_embeddings is None:
            sparse_embeddings = tf.zeros((batch_size, 0, 1, self.hidden_size), dtype=dense_embeddings.dtype)

        # 返回稀疏嵌入和密集嵌入
        return sparse_embeddings, dense_embeddings
class TFSamVisionAttention(tf.keras.layers.Layer):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size, **kwargs):
        # 继承父类的初始化方法
        super().__init__(**kwargs)
        # 根据配置信息和窗口大小计算输入大小
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )
        self.input_size = input_size

        # 初始化注意力头数和注意力头维度等参数
        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        # 创建qkv layer，用于计算query, key, value
        self.qkv = tf.keras.layers.Dense(config.hidden_size * 3, use_bias=config.qkv_bias, name="qkv")
        # 创建proj layer，用于将attention输出映射回隐藏层大小
        self.proj = tf.keras.layers.Dense(config.hidden_size, name="proj")

        # 根据配置信息判断是否使用相对位置编码
        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")
        self.config = config

    def build(self, input_shape=None):
        if self.input_size is not None:
            # 初始化相对位置编码的嵌入矩阵
            self.rel_pos_h = self.add_weight(
                shape=(2 * self.input_size[0] - 1, self.head_dim), initializer="zeros", name="rel_pos_h"
            )
            self.rel_pos_w = self.add_weight(
                shape=(2 * self.input_size[1] - 1, self.head_dim), initializer="zeros", name="rel_pos_w"
            )

        if self.built:
            return
        self.built = True
        if getattr(self, "qkv", None) is not None:
            with tf.name_scope(self.qkv.name):
                # 构建qkv layer
                self.qkv.build([None, None, self.config.hidden_size])
        if getattr(self, "proj", None) is not None:
            with tf.name_scope(self.proj.name):
                # 构建proj layer
                self.proj.build([None, None, self.config.hidden_size])
    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: tf.Tensor) -> tf.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`tf.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        # 计算相对位置的最大距离
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # 如果 rel_pos 的形状与最大距离不符，则进行插值处理
        if rel_pos.shape[0] != max_rel_dist:
            # 插值处理 rel_pos
            rel_pos_resized = tf.image.resize(
                tf.reshape(rel_pos, (1, rel_pos.shape[0], -1)),
                size=(max_rel_dist, rel_pos.shape[1]),
                method="bilinear",
            )
            rel_pos_resized = tf.reshape(rel_pos_resized, (-1, max_rel_dist))
        else:
            # 如果形状符合要求，则直接使用 rel_pos
            rel_pos_resized = rel_pos

        # 根据短边长度对坐标进行缩放，以处理 q 和 k 形状不同的情况
        q_coords = tf.expand_dims(tf.range(q_size, dtype=tf.float32), 1) * max(k_size / q_size, 1.0)
        k_coords = tf.expand_dims(tf.range(k_size, dtype=tf.float32), 0) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        # 根据相对坐标从 rel_pos_resized 中提取位置嵌入
        return tf.gather(rel_pos_resized, tf.cast(relative_coords, tf.int32))

    def add_decomposed_rel_pos(
        self,
        attn: tf.Tensor,
        query: tf.Tensor,
        rel_pos_h: tf.Tensor,
        rel_pos_w: tf.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> tf.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`tf.Tensor`):
                attention map. 注意力图
            query (`tf.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel). 查询
            rel_pos_h (`tf.Tensor`):
                relative position embeddings (Lh, channel) for height axis. 高度轴的相对位置嵌入
            rel_pos_w (`tf.Tensor`):
                relative position embeddings (Lw, channel) for width axis. 宽度轴的相对位置嵌入
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width). 查询 q 的空间序列大小
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width). 键 k 的空间序列大小

        Returns:
            attn (`tf.Tensor`):
                attention map with added relative positional embeddings. 添加了相对位置嵌入的注意力图
        """
        # 获取查询的高度和宽度
        query_height, query_width = q_size
        # 获取键的高度和宽度
        key_height, key_width = k_size
        # 计算高度轴的相对位置嵌入
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        # 计算宽度轴的相对位置嵌入
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        # 获取查询的形状
        batch_size, _, dim = shape_list(query)
        # 重塑查询
        reshaped_query = tf.reshape(query, (batch_size, query_height, query_width, dim))
        # 计算高度轴的相对位置嵌入与查询的乘积
        rel_h = tf.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        # 计算宽度轴的相对位置嵌入与查询的乘积
        rel_w = tf.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        # 重塑注意力图
        attn = tf.reshape(attn, (batch_size, query_height, query_width, key_height, key_width))
        # 在注意力图中添加高度轴的相对位置嵌入
        attn = attn + tf.expand_dims(rel_h, axis=-1) + tf.expand_dims(rel_w, axis=-2)
        # 重新塑形注意力图
        attn = tf.reshape(attn, (batch_size, query_height * query_width, key_height * key_width))
        return attn
```  
    # 定义 call 方法，用于执行自注意力机制
    def call(self, hidden_states: tf.Tensor, output_attentions=False, training=False) -> tf.Tensor:
        # 获取隐藏状态的形状信息
        batch_size, height, width, _ = shape_list(hidden_states)
        # 将隐藏状态传入 qkv 网络层，并reshape为 (batch_size, height * width, 3, num_attention_heads, -1)
        qkv = tf.reshape(self.qkv(hidden_states), (batch_size, height * width, 3, self.num_attention_heads, -1))
        # 转置以便后续处理，变为 (3, batch_size, num_attention_heads, height * width, -1)
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        # 将 qkv tensor 按第一个维度解包为 query, key, value
        query, key, value = tf.unstack(
            tf.reshape(qkv, (3, batch_size * self.num_attention_heads, height * width, -1)), axis=0
        )
        # 计算注意力权重，使用矩阵乘法
        attn_weights = tf.matmul(query * self.scale, key, transpose_b=True)

        # 如果使用相对位置编码
        if self.use_rel_pos:
            # 添加分解的相对位置编码
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        # 对注意力权重进行 softmax 归一化
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        # 如果处于训练模式，则进行 dropout 处理
        if training:
            attn_probs = tf.nn.dropout(attn_weights, rate=self.dropout)
        else:
            attn_probs = attn_weights

        # 计算注意力输出，利用注意力权重和 value
        attn_output = tf.reshape(attn_probs @ value, (batch_size, self.num_attention_heads, height, width, -1))
        # 调整输出形状
        attn_output = tf.transpose(attn_output, perm=(0, 2, 3, 1, 4))
        attn_output = tf.reshape(attn_output, (batch_size, height, width, self.config.hidden_size))

        # 经过投影层处理
        attn_output = self.proj(attn_output)

        # 如果需要输出注意力权重，则将其包含在输出中
        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        # 返回输出
        return outputs
    # 定义 TFSamVisionLayer 类，继承自 tf.keras.layers.Layer
class TFSamVisionLayer(tf.keras.layers.Layer):
    # 初始化方法，接受配置信息 config 和窗口大小 window_size，以及其他参数
    def __init__(self, config, window_size, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建 LayerNormalization 层，用于层归一化
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm1")
        # 创建 TFSamVisionAttention 层，用于注意力计算
        self.attn = TFSamVisionAttention(config, window_size, name="attn")
        # 创建 LayerNormalization 层，用于层归一化
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm2")
        # 创建 TFSamMLPBlock 层，用于多层感知机
        self.mlp = TFSamMLPBlock(config, name="mlp")
        # 保存窗口大小和配置信息
        self.window_size = window_size
        self.config = config

    # 定义将隐藏状态进行窗口划分的方法
    def window_partition(self, hidden_states: tf.Tensor, window_size: int) -> Tuple[tf.Tensor, Tuple[int, int]]:
        # 获取隐藏状态的形状信息
        batch_size, height, width, channel = shape_list(hidden_states)

        # 计算需要进行填充的高度和宽度
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        # 如果需要填充，则进行填充操作
        if pad_h > 0 or pad_w > 0:
            hidden_states = tf.pad(hidden_states, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        pad_height, pad_width = height + pad_h, width + pad_w

        # 将隐藏状态重新调整形状，以便进行窗口划分
        hidden_states = tf.reshape(
            hidden_states,
            [batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel],
        )
        # 调整维度顺序，再次调整形状
        windows = tf.reshape(
            tf.transpose(hidden_states, perm=[0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, channel]
        )
        # 返回划分后的窗口和填充后的高度、宽度信息
        return windows, (pad_height, pad_width)

    # 定义将窗口恢复为原始隐藏状态的方法
    def window_unpartition(
        self, windows: tf.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> tf.Tensor:
        # 获取填充后的高度和宽度信息，以及原始的高度和宽度信息
        pad_height, pad_width = padding_shape
        height, width = original_shape
        # 计算 batch_size
        batch_size = shape_list(windows)[0] // (pad_height * pad_width // window_size // window_size)
        # 将窗口重新调整形状，以便恢复为原始隐藏状态
        hidden_states = tf.reshape(
            windows, [batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1]
        )
        hidden_states = tf.reshape(
            tf.transpose(hidden_states, perm=[0, 1, 3, 2, 4, 5]), [batch_size, pad_height, pad_width, -1]
        )

        # 如果填充高度或宽度大于原始高度或宽度，则进行裁剪
        if pad_height > height or pad_width > width:
            hidden_states = hidden_states[:, :height, :width, :]
        # 返回恢复后的隐藏状态
        return hidden_states

    # 定义调用方法
    def call(
        self,
        hidden_states: tf.Tensor,
        output_attentions: Optional[bool] = False,
        training: Optional[bool] = False,
    # 这是一个 Transformer 层的实现，包含了自注意力机制和前馈神经网络层
    def call(self, hidden_states: tf.Tensor, output_attentions: bool = False, training: bool = False
            ) -> Tuple[tf.Tensor]:
        # 保存输入的 hidden_states 作为残差连接
        residual = hidden_states
    
        # 进行第一个层归一化
        hidden_states = self.layer_norm1(hidden_states)
    
        # 如果启用了窗口自注意力，则进行窗口划分
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)
    
        # 计算自注意力，返回新的 hidden_states 和注意力权重
        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            training=training,
        )
    
        # 如果启用了窗口自注意力，则进行窗口合并
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))
    
        # 进行残差连接
        hidden_states = residual + hidden_states
    
        # 进行第二个层归一化
        layernorm_output = self.layer_norm2(hidden_states)
    
        # 进行前馈神经网络
        hidden_states = hidden_states + self.mlp(layernorm_output)
    
        # 返回 hidden_states 及可能的注意力权重
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
    
        return outputs
    
    # 构建模型层
    def build(self, input_shape=None):
        # 如果已经构建好则直接返回
        if self.built:
            return
        self.built = True
    
        # 构建第一个层归一化层
        if getattr(self, "layer_norm1", None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, None, self.config.hidden_size])
    
        # 构建自注意力层
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
    
        # 构建第二个层归一化层
        if getattr(self, "layer_norm2", None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, None, self.config.hidden_size])
    
        # 构建前馈神经网络层
        if getattr(self, "mlp", None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
# 创建 TFSamVisionNeck 类
class TFSamVisionNeck(tf.keras.layers.Layer):
    # 初始化函数，接收配置和其他关键字参数
    def __init__(self, config: SamVisionConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 保存配置信息到实例变量
        self.config = config

        # 创建一个卷积层，用于处理输入数据
        self.conv1 = tf.keras.layers.Conv2D(
            config.output_channels,
            kernel_size=1,
            use_bias=False,
            name="conv1",
        )
        # 创建 LayerNorm 层，用于标准化数据
        self.layer_norm1 = TFSamLayerNorm(config.output_channels, name="layer_norm1")
        # 创建另一个卷积层
        self.conv2 = tf.keras.layers.Conv2D(
            config.output_channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="conv2",
        )
        # 创建另一个 LayerNorm 层
        self.layer_norm2 = TFSamLayerNorm(config.output_channels, name="layer_norm2")

    # 定义调用函数，处理输入数据
    def call(self, hidden_states):
        # 通过第一个卷积层处理输入数据
        hidden_states = self.conv1(hidden_states)
        # 通过第一个 LayerNorm 层标准化数据
        hidden_states = self.layer_norm1(hidden_states)
        # 通过第二个卷积层处理标准化后的数据
        hidden_states = self.conv2(hidden_states)
        # 通过第二个 LayerNorm 层标准化处理后的数据
        hidden_states = self.layer_norm2(hidden_states)
        # 调换数据维度
        hidden_states = tf.transpose(hidden_states, perm=[0, 3, 1, 2])
        # 返回处理后的数据
        return hidden_states

    # 定义构建函数，构建网络层
    def build(self, input_shape=None):
        # 如果已经构建过网络层，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果存在第一个卷积层
        if getattr(self, "conv1", None) is not None:
            # 使用 TensorFlow 的命名空间构建卷积层
            with tf.name_scope(self.conv1.name):
                self.conv1.build([None, None, None, self.config.hidden_size])
        # 如果存在第一个 LayerNorm 层
        if getattr(self, "layer_norm1", None) is not None:
            # 使用 TensorFlow 的命名空间构建 LayerNorm 层
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build(None)
        # 如果存在第二个卷积层
        if getattr(self, "conv2", None) is not None:
            # 使用 TensorFlow 的命名空间构建卷积层
            with tf.name_scope(self.conv2.name):
                self.conv2.build([None, None, None, self.config.output_channels])
        # 如果存在第二个 LayerNorm 层
        if getattr(self, "layer_norm2", None) is not None:
            # 使用 TensorFlow 的命名空间构建 LayerNorm 层
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build(None)


# 创建 TFSamVisionEncoder 类
class TFSamVisionEncoder(tf.keras.layers.Layer):
    # 初始化函数，接收配置和其他关键字参数
    def __init__(self, config: SamVisionConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 保存配置信息到实例变量
        self.config = config
        # 保存图像大小到实例变量
        self.image_size = config.image_size

        # 创建补丁嵌入层
        self.patch_embed = TFSamPatchEmbeddings(config, name="patch_embed")

        # 初始化位置嵌入
        self.pos_embed = None

        # 创建一系列层
        self.layers = []
        for i in range(config.num_hidden_layers):
            # 创建视觉编码层
            layer = TFSamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
                name=f"layers_._{i}",
            )
            self.layers.append(layer)

        # 创建视觉颈部
        self.neck = TFSamVisionNeck(config, name="neck")
    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建过了，直接返回
        if self.built:
            return
        # 标记模型已经构建完成
        self.built = True
        
        # 如果使用绝对位置嵌入
        if self.config.use_abs_pos:
            # 初始化绝对位置嵌入，大小为预训练图像大小除以patch大小
            self.pos_embed = self.add_weight(
                shape=[
                    1,
                    self.config.image_size // self.config.patch_size,
                    self.config.image_size // self.config.patch_size,
                    self.config.hidden_size,
                ],
                initializer="zeros",
                trainable=True,
                name="pos_embed",
            )
    
        # 如果存在 patch embedding 层
        if getattr(self, "patch_embed", None) is not None:
            # 构建 patch embedding 层
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        # 如果存在 neck 层
        if getattr(self, "neck", None) is not None:
            # 构建 neck 层
            with tf.name_scope(self.neck.name):
                self.neck.build(None)
        # 构建所有 layers
        for layer in self.layers:
            with tf.name_scope(layer.name):
                layer.build(None)
    
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.patch_embed
    
    # 前向传播
    def call(
        self,
        pixel_values: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ):
    ) -> Union[Tuple, TFSamVisionEncoderOutput]:
        # 设置是否输出注意力权重，默认从配置中获取
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态，默认从配置中获取
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出，默认从配置中获取
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查是否指定了像素值，若未指定则引发异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值通过补丁嵌入层处理后作为隐藏状态的初始值
        hidden_states = self.patch_embed(pixel_values)
        # 若位置编码不为空，则将其加到隐藏状态上
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        # 初始化用于存储所有隐藏状态的元组，若不输出隐藏状态则为空
        all_hidden_states = () if output_hidden_states else None
        # 初始化用于存储所有自注意力权重的元组，若不输出注意力权重则为空
        all_self_attentions = () if output_attentions else None

        # 遍历所有编码器层
        for i, layer_module in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入元组
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用编码器层进行前向传播
            layer_outputs = layer_module(hidden_states, output_attentions=output_attentions, training=training)

            # 更新隐藏状态为编码器层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的自注意力权重加入元组
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 若需要输出隐藏状态，则将最终隐藏状态加入元组
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 将最终隐藏状态通过颈部层处理
        hidden_states = self.neck(hidden_states)

        # 若不以字典形式返回结果，则将结果组合成元组形式返回
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        # 以字典形式返回结果
        return TFSamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义 TFSamPreTrainedModel 类，继承自 TFPreTrainedModel
class TFSamPreTrainedModel(TFPreTrainedModel):
    # 配置类为 SamConfig
    config_class = SamConfig
    # 基础模型前缀为 "sam"
    base_model_prefix = "sam"
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

# SAM_START_DOCSTRING 为模型的文档字符串模板
SAM_START_DOCSTRING = r"""
    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a TensorFlow [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
    subclass. Use it as a regular TensorFlow Model and refer to the TensorFlow documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`SamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""

# SAM_INPUTS_DOCSTRING 为空

# 添加文档字符串说明至 TFSamModel 类
@add_start_docstrings(
    "Segment Anything Model (SAM) for generating segmentation masks, given an input image and ",
    " optional 2D location and bounding boxes.",
    SAM_START_DOCSTRING,
)
# TFSamModel 类继承自 TFSamPreTrainedModel 类
class TFSamModel(TFSamPreTrainedModel):
    # 在加载模型中忽略的键列表
    _keys_to_ignore_on_load_missing = [r"prompt_encoder.shared_embedding.positional_embedding"]

    # 初始化方法
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # 共享图像嵌入层
        self.shared_image_embedding = TFSamPositionalEmbedding(config.vision_config, name="shared_image_embedding")
        # 视觉编码器
        self.vision_encoder = TFSamVisionEncoder(config.vision_config, name="vision_encoder")
        # 提示编码器
        self.prompt_encoder = TFSamPromptEncoder(
            config.prompt_encoder_config, self.shared_image_embedding, name="prompt_encoder"
        )
        # 掩码解码器
        self.mask_decoder = TFSamMaskDecoder(config.mask_decoder_config, name="mask_decoder")
        self.config = config

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    # 获取图像整体位置嵌入
    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        grid = tf.ones((size, size))
        y_embed = tf.math.cumsum(grid, axis=0) - 0.5
        x_embed = tf.math.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(tf.stack([x_embed, y_embed], axis=-1))
        return tf.expand_dims(tf.transpose(positional_embedding, perm=[2, 0, 1]), axis=0)  # channel x height x width

    # 获取图像嵌入
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.TFModelOutput`] instead of a plain tuple.

        """
        # 通过视觉编码器传递像素值，返回图像嵌入
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取图像嵌入
        image_embeddings = vision_output[0]
        return image_embeddings

    def get_prompt_embeddings(
        self,
        input_points: tf.Tensor | None = None,
        input_labels: tf.Tensor | None = None,
        input_boxes: tf.Tensor | None = None,
        input_masks: tf.Tensor | None = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`tf.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`tf.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`tf.Tensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`tf.Tensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        # 通过提示编码器传递输入的点、标签、框和掩码，返回提示嵌入
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    # 定义一个方法，用于调用模型
    def call(
        self,
        # 输入像素值，默认为 None
        pixel_values: TFModelInputType | None = None,
        # 输入点，默认为 None
        input_points: tf.Tensor | None = None,
        # 输入标签，默认为 None
        input_labels: tf.Tensor | None = None,
        # 输入框，默认为 None
        input_boxes: tf.Tensor | None = None,
        # 输入掩码，默认为 None
        input_masks: tf.Tensor | None = None,
        # 图像嵌入，默认为 None
        image_embeddings: tf.Tensor | None = None,
        # 是否输出多掩码，默认为 True
        multimask_output: bool = True,
        # 是否输出注意力，默认为 None
        output_attentions: bool | None = None,
        # 是否输出隐藏状态，默认为 None
        output_hidden_states: bool | None = None,
        # 是否返回字典，默认为 None
        return_dict: bool | None = None,
        # 是否训练，默认为 False
        training: bool = False,
        # 其他参数
        **kwargs,
    # 定义一个方法，用于服务输出
    def serving_output(self, output: TFSamImageSegmentationOutput) -> TFSamImageSegmentationOutput:
        # 根据配置输出隐藏状态的 Tensor，否则为 None
        hs = tf.convert_to_tensor(output.vision_hidden_states) if self.config.output_hidden_states else None
        # 根据配置输出注意力的 Tensor，否则为 None
        attns = tf.convert_to_tensor(output.vision_attentions) if self.config.output_attentions else None

        # 返回服务输出对象
        return TFSamImageSegmentationOutput(
            # 返回 IOU 分数
            iou_scores=output.iou_scores,
            # 返回预测掩码
            pred_masks=output.pred_masks,
            # 返回隐藏状态的 Tensor 或 None
            vision_hidden_states=hs if self.config.output_hidden_states else None,
            # 返回注意力的 Tensor 或 None
            vision_attentions=attns if self.config.output_attentions else None,
            # 返回掩码解码器注意力或 None（如果配置要求输出注意力）
            mask_decoder_attentions=output.mask_decoder_attentions if self.config.output_attentions else None,
        )

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建，则返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在共享图像嵌入
        if getattr(self, "shared_image_embedding", None) is not None:
            # 在共享图像嵌入的作用域内构建它
            with tf.name_scope(self.shared_image_embedding.name):
                self.shared_image_embedding.build(None)
        # 如果存在视觉编码器
        if getattr(self, "vision_encoder", None) is not None:
            # 在视觉编码器的作用域内构建它
            with tf.name_scope(self.vision_encoder.name):
                self.vision_encoder.build(None)
        # 如果存在提示编码器
        if getattr(self, "prompt_encoder", None) is not None:
            # 在提示编码器的作用域内构建它
            with tf.name_scope(self.prompt_encoder.name):
                self.prompt_encoder.build(None)
        # 如果存在掩码解码器
        if getattr(self, "mask_decoder", None) is not None:
            # 在掩码解码器的作用域内构建它
            with tf.name_scope(self.mask_decoder.name):
                self.mask_decoder.build(None)
```py  
```