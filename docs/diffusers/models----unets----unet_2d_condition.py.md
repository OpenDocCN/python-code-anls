# `.\diffusers\models\unets\unet_2d_condition.py`

```
# 版权声明，标明版权信息和使用许可
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 按照 Apache License 2.0 版本进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 你不得在未遵守许可的情况下使用此文件
# you may not use this file except in compliance with the License.
# 你可以在以下网址获得许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，软件以“按原样”方式分发，不提供任何形式的担保或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定语言所适用的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from dataclasses import dataclass
# 导入所需的类型注释
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库和相关模块
import torch
import torch.nn as nn
import torch.utils.checkpoint

# 从配置和加载器模块中导入所需的类和函数
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from ..activations import get_activation
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,  # 导入与注意力机制相关的处理器
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from ..embeddings import (
    GaussianFourierProjection,  # 导入多种嵌入方法
    GLIGENTextBoundingboxProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from ..modeling_utils import ModelMixin  # 导入模型混合类
from .unet_2d_blocks import (
    get_down_block,  # 导入下采样块的构造函数
    get_mid_block,   # 导入中间块的构造函数
    get_up_block,    # 导入上采样块的构造函数
)

# 创建一个日志记录器，用于记录模型相关信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义 UNet2DConditionOutput 数据类，用于存储 UNet2DConditionModel 的输出
@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    UNet2DConditionModel 的输出。

    参数:
        sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`):
            基于 `encoder_hidden_states` 输入的隐藏状态输出，模型最后一层的输出。
    """

    sample: torch.Tensor = None  # 定义一个样本属性，默认为 None

# 定义 UNet2DConditionModel 类，表示一个条件 2D UNet 模型
class UNet2DConditionModel(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin
):
    r"""
    一个条件 2D UNet 模型，接受一个噪声样本、条件状态和时间步，并返回样本形状的输出。

    该模型继承自 [`ModelMixin`]。查看超类文档以获取其为所有模型实现的通用方法
    （例如下载或保存）。
    """

    _supports_gradient_checkpointing = True  # 表示该模型支持梯度检查点
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D", "CrossAttnUpBlock2D"]  # 不进行拆分的模块列表

    @register_to_config  # 将该方法注册到配置中
    # 初始化方法，设置类的基本属性
        def __init__(
            # 样本大小，默认为 None
            self,
            sample_size: Optional[int] = None,
            # 输入通道数，默认为 4
            in_channels: int = 4,
            # 输出通道数，默认为 4
            out_channels: int = 4,
            # 是否将输入样本中心化，默认为 False
            center_input_sample: bool = False,
            # 是否将正弦函数翻转为余弦函数，默认为 True
            flip_sin_to_cos: bool = True,
            # 频率偏移量，默认为 0
            freq_shift: int = 0,
            # 向下采样的块类型，包含多种块类型
            down_block_types: Tuple[str] = (
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            # 中间块的类型，默认为 UNet 的中间块类型
            mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
            # 向上采样的块类型，包含多种块类型
            up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            # 是否仅使用交叉注意力，默认为 False
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            # 每个块的输出通道数
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            # 每个块的层数，默认为 2
            layers_per_block: Union[int, Tuple[int]] = 2,
            # 下采样时的填充大小，默认为 1
            downsample_padding: int = 1,
            # 中间块的缩放因子，默认为 1
            mid_block_scale_factor: float = 1,
            # dropout 概率，默认为 0.0
            dropout: float = 0.0,
            # 激活函数类型，默认为 "silu"
            act_fn: str = "silu",
            # 归一化的组数，默认为 32
            norm_num_groups: Optional[int] = 32,
            # 归一化的 epsilon 值，默认为 1e-5
            norm_eps: float = 1e-5,
            # 交叉注意力的维度，默认为 1280
            cross_attention_dim: Union[int, Tuple[int]] = 1280,
            # 每个块的变换层数，默认为 1
            transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
            # 反向变换层的块数，默认为 None
            reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
            # 编码器隐藏层的维度，默认为 None
            encoder_hid_dim: Optional[int] = None,
            # 编码器隐藏层类型，默认为 None
            encoder_hid_dim_type: Optional[str] = None,
            # 注意力头的维度，默认为 8
            attention_head_dim: Union[int, Tuple[int]] = 8,
            # 注意力头的数量，默认为 None
            num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
            # 是否使用双交叉注意力，默认为 False
            dual_cross_attention: bool = False,
            # 是否使用线性投影，默认为 False
            use_linear_projection: bool = False,
            # 类嵌入类型，默认为 None
            class_embed_type: Optional[str] = None,
            # 附加嵌入类型，默认为 None
            addition_embed_type: Optional[str] = None,
            # 附加时间嵌入维度，默认为 None
            addition_time_embed_dim: Optional[int] = None,
            # 类嵌入数量，默认为 None
            num_class_embeds: Optional[int] = None,
            # 是否上溯注意力，默认为 False
            upcast_attention: bool = False,
            # ResNet 时间缩放偏移类型，默认为 "default"
            resnet_time_scale_shift: str = "default",
            # ResNet 是否跳过时间激活，默认为 False
            resnet_skip_time_act: bool = False,
            # ResNet 输出缩放因子，默认为 1.0
            resnet_out_scale_factor: float = 1.0,
            # 时间嵌入类型，默认为 "positional"
            time_embedding_type: str = "positional",
            # 时间嵌入维度，默认为 None
            time_embedding_dim: Optional[int] = None,
            # 时间嵌入激活函数，默认为 None
            time_embedding_act_fn: Optional[str] = None,
            # 时间步后激活函数，默认为 None
            timestep_post_act: Optional[str] = None,
            # 时间条件投影维度，默认为 None
            time_cond_proj_dim: Optional[int] = None,
            # 输入卷积核大小，默认为 3
            conv_in_kernel: int = 3,
            # 输出卷积核大小，默认为 3
            conv_out_kernel: int = 3,
            # 投影类嵌入输入维度，默认为 None
            projection_class_embeddings_input_dim: Optional[int] = None,
            # 注意力类型，默认为 "default"
            attention_type: str = "default",
            # 类嵌入是否拼接，默认为 False
            class_embeddings_concat: bool = False,
            # 中间块是否仅使用交叉注意力，默认为 None
            mid_block_only_cross_attention: Optional[bool] = None,
            # 交叉注意力归一化类型，默认为 None
            cross_attention_norm: Optional[str] = None,
            # 附加嵌入类型的头数量，默认为 64
            addition_embed_type_num_heads: int = 64,
    # 定义一个私有方法，用于检查配置参数
        def _check_config(
            self,
            # 定义下行块类型的元组，表示模型的结构
            down_block_types: Tuple[str],
            # 定义上行块类型的元组，表示模型的结构
            up_block_types: Tuple[str],
            # 定义仅使用交叉注意力的标志，可以是布尔值或布尔值的元组
            only_cross_attention: Union[bool, Tuple[bool]],
            # 定义每个块的输出通道数的元组，表示层的宽度
            block_out_channels: Tuple[int],
            # 定义每个块的层数，可以是整数或整数的元组
            layers_per_block: Union[int, Tuple[int]],
            # 定义交叉注意力维度，可以是整数或整数的元组
            cross_attention_dim: Union[int, Tuple[int]],
            # 定义每个块的变换器层数，可以是整数、整数的元组或元组的元组
            transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]],
            # 定义是否反转变换器层的布尔值
            reverse_transformer_layers_per_block: bool,
            # 定义注意力头的维度，表示注意力的分辨率
            attention_head_dim: int,
            # 定义注意力头的数量，可以是可选的整数或整数的元组
            num_attention_heads: Optional[Union[int, Tuple[int]],
    ):
        # 检查 down_block_types 和 up_block_types 的长度是否相同
        if len(down_block_types) != len(up_block_types):
            # 如果不同，抛出值错误并提供详细信息
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        # 检查 block_out_channels 和 down_block_types 的长度是否相同
        if len(block_out_channels) != len(down_block_types):
            # 如果不同，抛出值错误并提供详细信息
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # 检查 only_cross_attention 是否为布尔值且长度与 down_block_types 相同
        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            # 如果不满足条件，抛出值错误并提供详细信息
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        # 检查 num_attention_heads 是否为整数且长度与 down_block_types 相同
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            # 如果不满足条件，抛出值错误并提供详细信息
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        # 检查 attention_head_dim 是否为整数且长度与 down_block_types 相同
        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            # 如果不满足条件，抛出值错误并提供详细信息
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        # 检查 cross_attention_dim 是否为列表且长度与 down_block_types 相同
        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            # 如果不满足条件，抛出值错误并提供详细信息
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        # 检查 layers_per_block 是否为整数且长度与 down_block_types 相同
        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            # 如果不满足条件，抛出值错误并提供详细信息
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        # 检查 transformer_layers_per_block 是否为列表且 reverse_transformer_layers_per_block 为 None
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            # 遍历 transformer_layers_per_block 中的每个层
            for layer_number_per_block in transformer_layers_per_block:
                # 检查每个层是否为列表
                if isinstance(layer_number_per_block, list):
                    # 如果是，则抛出值错误，提示需要提供 reverse_transformer_layers_per_block
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

    # 定义设置时间投影的私有方法
    def _set_time_proj(
        self,
        # 时间嵌入类型
        time_embedding_type: str,
        # 块输出通道数
        block_out_channels: int,
        # 是否翻转正弦和余弦
        flip_sin_to_cos: bool,
        # 频率偏移
        freq_shift: float,
        # 时间嵌入维度
        time_embedding_dim: int,
    # 返回时间嵌入维度和时间步输入维度的元组
    ) -> Tuple[int, int]:
        # 判断时间嵌入类型是否为傅里叶
        if time_embedding_type == "fourier":
            # 计算时间嵌入维度，默认为 block_out_channels[0] * 2
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            # 确保时间嵌入维度为偶数
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            # 初始化高斯傅里叶投影，设定相关参数
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            # 设置时间步输入维度为时间嵌入维度
            timestep_input_dim = time_embed_dim
        # 判断时间嵌入类型是否为位置编码
        elif time_embedding_type == "positional":
            # 计算时间嵌入维度，默认为 block_out_channels[0] * 4
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
            # 初始化时间步对象，设定相关参数
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            # 设置时间步输入维度为 block_out_channels[0]
            timestep_input_dim = block_out_channels[0]
        # 如果时间嵌入类型不合法，抛出错误
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )
    
        # 返回时间嵌入维度和时间步输入维度
        return time_embed_dim, timestep_input_dim
    
    # 定义设置编码器隐藏投影的方法
    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ):
        # 如果编码器隐藏维度类型为空且隐藏维度已定义
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            # 默认将编码器隐藏维度类型设为'text_proj'
            encoder_hid_dim_type = "text_proj"
            # 注册编码器隐藏维度类型到配置中
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            # 记录信息日志
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")
    
        # 如果编码器隐藏维度为空且隐藏维度类型已定义，抛出错误
        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )
    
        # 判断编码器隐藏维度类型是否为'text_proj'
        if encoder_hid_dim_type == "text_proj":
            # 初始化线性投影层，输入维度为encoder_hid_dim，输出维度为cross_attention_dim
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        # 判断编码器隐藏维度类型是否为'text_image_proj'
        elif encoder_hid_dim_type == "text_image_proj":
            # 初始化文本-图像投影对象，设定相关参数
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        # 判断编码器隐藏维度类型是否为'image_proj'
        elif encoder_hid_dim_type == "image_proj":
            # 初始化图像投影对象，设定相关参数
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        # 如果编码器隐藏维度类型不合法，抛出错误
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        # 如果都不符合，将编码器隐藏投影设为None
        else:
            self.encoder_hid_proj = None
    # 设置类嵌入的私有方法
        def _set_class_embedding(
            self,
            class_embed_type: Optional[str],  # 嵌入类型，可能为 None 或特定字符串
            act_fn: str,  # 激活函数的名称
            num_class_embeds: Optional[int],  # 类嵌入数量，可能为 None
            projection_class_embeddings_input_dim: Optional[int],  # 投影类嵌入输入维度，可能为 None
            time_embed_dim: int,  # 时间嵌入的维度
            timestep_input_dim: int,  # 时间步输入的维度
        ):
            # 如果嵌入类型为 None 且类嵌入数量不为 None
            if class_embed_type is None and num_class_embeds is not None:
                # 创建嵌入层，大小为类嵌入数量和时间嵌入维度
                self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
            # 如果嵌入类型为 "timestep"
            elif class_embed_type == "timestep":
                # 创建时间步嵌入对象
                self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
            # 如果嵌入类型为 "identity"
            elif class_embed_type == "identity":
                # 创建恒等层，输入和输出维度相同
                self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
            # 如果嵌入类型为 "projection"
            elif class_embed_type == "projection":
                # 如果投影类嵌入输入维度为 None，抛出错误
                if projection_class_embeddings_input_dim is None:
                    raise ValueError(
                        "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                    )
                # 创建投影时间步嵌入对象
                self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
            # 如果嵌入类型为 "simple_projection"
            elif class_embed_type == "simple_projection":
                # 如果投影类嵌入输入维度为 None，抛出错误
                if projection_class_embeddings_input_dim is None:
                    raise ValueError(
                        "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                    )
                # 创建线性层作为简单投影
                self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
            # 如果没有匹配的嵌入类型
            else:
                # 将类嵌入设置为 None
                self.class_embedding = None
    
        # 设置附加嵌入的私有方法
        def _set_add_embedding(
            self,
            addition_embed_type: str,  # 附加嵌入类型
            addition_embed_type_num_heads: int,  # 附加嵌入类型的头数
            addition_time_embed_dim: Optional[int],  # 附加时间嵌入维度，可能为 None
            flip_sin_to_cos: bool,  # 是否翻转正弦到余弦
            freq_shift: float,  # 频率偏移量
            cross_attention_dim: Optional[int],  # 交叉注意力维度，可能为 None
            encoder_hid_dim: Optional[int],  # 编码器隐藏维度，可能为 None
            projection_class_embeddings_input_dim: Optional[int],  # 投影类嵌入输入维度，可能为 None
            time_embed_dim: int,  # 时间嵌入维度
    ):
        # 检查附加嵌入类型是否为 "text"
        if addition_embed_type == "text":
            # 如果编码器隐藏维度不为 None，则使用该维度
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            # 否则使用交叉注意力维度
            else:
                text_time_embedding_from_dim = cross_attention_dim

            # 创建文本时间嵌入对象
            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        # 检查附加嵌入类型是否为 "text_image"
        elif addition_embed_type == "text_image":
            # text_embed_dim 和 image_embed_dim 不必是 `cross_attention_dim`，为了避免 __init__ 过于繁杂
            # 在这里设置为 `cross_attention_dim`，因为这是当前唯一使用情况的所需维度 (Kandinsky 2.1)
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        # 检查附加嵌入类型是否为 "text_time"
        elif addition_embed_type == "text_time":
            # 创建时间投影对象
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            # 创建时间嵌入对象
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        # 检查附加嵌入类型是否为 "image"
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            # 创建图像时间嵌入对象
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        # 检查附加嵌入类型是否为 "image_hint"
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            # 创建图像提示时间嵌入对象
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        # 检查附加嵌入类型是否为 None 以外的值
        elif addition_embed_type is not None:
            # 抛出值错误，提示无效的附加嵌入类型
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

    # 定义一个属性方法，用于设置位置网络
    def _set_pos_net_if_use_gligen(self, attention_type: str, cross_attention_dim: int):
        # 检查注意力类型是否为 "gated" 或 "gated-text-image"
        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768  # 默认的正向长度
            # 如果交叉注意力维度是整数，则使用该值
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            # 如果交叉注意力维度是列表或元组，则使用第一个值
            elif isinstance(cross_attention_dim, (list, tuple)):
                positive_len = cross_attention_dim[0]

            # 根据注意力类型确定特征类型
            feature_type = "text-only" if attention_type == "gated" else "text-image"
            # 创建 GLIGEN 文本边界框投影对象
            self.position_net = GLIGENTextBoundingboxProjection(
                positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type
            )

    # 定义一个属性
    @property
    # 定义一个方法，返回一个字典，包含模型中所有的注意力处理器
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 初始化一个空字典，用于存储注意力处理器
        processors = {}

        # 定义一个递归函数，用于添加处理器到字典
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否有获取处理器的方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，键为名称，值为处理器
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，处理子模块
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回更新后的处理器字典
            return processors

        # 遍历当前模块的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，添加处理器
            fn_recursive_add_processors(name, module, processors)

        # 返回包含所有处理器的字典
        return processors

    # 定义一个方法，设置用于计算注意力的处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        # 获取当前处理器的数量
        count = len(self.attn_processors.keys())

        # 如果传入的是字典，且字典长度与注意力层数量不匹配，抛出错误
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        # 定义一个递归函数，用于设置处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，设置子模块的处理器
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前模块的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，设置处理器
            fn_recursive_attn_processor(name, module, processor)
    # 定义设置默认注意力处理器的方法
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器并设置默认的注意力实现。
        """
        # 检查所有注意力处理器是否属于添加的键值注意力处理器类
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 创建添加键值注意力处理器的实例
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否属于交叉注意力处理器类
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 创建标准注意力处理器的实例
            processor = AttnProcessor()
        else:
            # 如果注意力处理器类型不匹配，则抛出错误
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        # 设置选定的注意力处理器
        self.set_attn_processor(processor)

    # 定义设置梯度检查点的方法
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块具有梯度检查点属性，则设置其值
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # 定义启用 FreeU 机制的方法
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""启用 FreeU 机制，详细信息请见 https://arxiv.org/abs/2309.11497。

        在缩放因子后面的后缀表示它们被应用的阶段块。

        请参考 [官方仓库](https://github.com/ChenyangSi/FreeU) 以获取已知在不同管道（如 Stable Diffusion v1、v2 和 Stable Diffusion XL）中效果良好的值组合。

        参数：
            s1 (`float`):
                阶段 1 的缩放因子，用于减弱跳跃特征的贡献，以减轻增强去噪过程中的“过平滑效应”。
            s2 (`float`):
                阶段 2 的缩放因子，用于减弱跳跃特征的贡献，以减轻增强去噪过程中的“过平滑效应”。
            b1 (`float`): 阶段 1 的缩放因子，用于增强骨干特征的贡献。
            b2 (`float`): 阶段 2 的缩放因子，用于增强骨干特征的贡献。
        """
        # 遍历上采样块并设置相应的缩放因子
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)  # 设置阶段 1 的缩放因子
            setattr(upsample_block, "s2", s2)  # 设置阶段 2 的缩放因子
            setattr(upsample_block, "b1", b1)  # 设置阶段 1 的骨干缩放因子
            setattr(upsample_block, "b2", b2)  # 设置阶段 2 的骨干缩放因子

    # 定义禁用 FreeU 机制的方法
    def disable_freeu(self):
        """禁用 FreeU 机制。"""
        freeu_keys = {"s1", "s2", "b1", "b2"}  # 定义 FreeU 相关的键
        # 遍历上采样块
        for i, upsample_block in enumerate(self.up_blocks):
            # 遍历每个 FreeU 键
            for k in freeu_keys:
                # 如果上采样块具有该键的属性或其值不为 None，则将其值设置为 None
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)
    # 定义一个方法，用于启用融合的 QKV 投影
    def fuse_qkv_projections(self):
        """
        启用融合的 QKV 投影。对于自注意力模块，所有投影矩阵（即查询、键、值）都被融合。
        对于交叉注意力模块，键和值的投影矩阵被融合。

        <Tip warning={true}>

        此 API 是 🧪 实验性的。

        </Tip>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None

        # 遍历注意力处理器，检查是否包含“Added”字样
        for _, attn_processor in self.attn_processors.items():
            # 如果发现添加的 KV 投影，抛出错误
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        # 保存当前的注意力处理器
        self.original_attn_processors = self.attn_processors

        # 遍历模块，查找类型为 Attention 的模块
        for module in self.modules():
            if isinstance(module, Attention):
                # 启用投影融合
                module.fuse_projections(fuse=True)

        # 设置注意力处理器为融合的处理器
        self.set_attn_processor(FusedAttnProcessor2_0())

    # 定义一个方法，用于禁用已启用的融合 QKV 投影
    def unfuse_qkv_projections(self):
        """禁用已启用的融合 QKV 投影。

        <Tip warning={true}>

        此 API 是 🧪 实验性的。

        </Tip>

        """
        # 如果原始注意力处理器不为 None，则恢复到原始处理器
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    # 定义一个方法，用于获取时间嵌入
    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        # 将时间步长赋值给 timesteps
        timesteps = timestep
        # 如果 timesteps 不是张量
        if not torch.is_tensor(timesteps):
            # TODO: 这需要在 CPU 和 GPU 之间同步。因此，如果可以的话，尽量将 timesteps 作为张量传递
            # 这将是使用 `match` 语句的好例子（Python 3.10+）
            is_mps = sample.device.type == "mps"  # 检查设备类型是否为 MPS
            # 根据时间步长类型设置数据类型
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64  # 浮点数类型
            else:
                dtype = torch.int32 if is_mps else torch.int64  # 整数类型
            # 将 timesteps 转换为张量
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        # 如果 timesteps 是标量（零维张量），则扩展维度
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)  # 增加一个维度并转移到样本设备

        # 将 timesteps 广播到与样本批次维度兼容的方式
        timesteps = timesteps.expand(sample.shape[0])  # 扩展到批次大小

        # 通过时间投影获得时间嵌入
        t_emb = self.time_proj(timesteps)
        # `Timesteps` 不包含任何权重，总是返回 f32 张量
        # 但时间嵌入可能实际在 fp16 中运行，因此需要进行类型转换。
        # 可能有更好的方法来封装这一点。
        t_emb = t_emb.to(dtype=sample.dtype)  # 转换 t_emb 的数据类型
        # 返回时间嵌入
        return t_emb
    # 获取类嵌入的方法，接受样本张量和可选的类标签
        def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            # 初始化类嵌入为 None
            class_emb = None
            # 检查类嵌入是否存在
            if self.class_embedding is not None:
                # 如果类标签为 None，抛出错误
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")
    
                # 检查类嵌入类型是否为时间步
                if self.config.class_embed_type == "timestep":
                    # 将类标签通过时间投影处理
                    class_labels = self.time_proj(class_labels)
    
                    # `Timesteps` 不包含权重，总是返回 f32 张量
                    # 可能有更好的方式来封装这一点
                    class_labels = class_labels.to(dtype=sample.dtype)
    
                # 获取类嵌入并转换为与样本相同的数据类型
                class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
            # 返回类嵌入
            return class_emb
    
        # 获取增强嵌入的方法，接受嵌入张量、编码器隐藏状态和额外条件参数
        def get_aug_embed(
            self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
        # 处理编码器隐藏状态的方法，接受编码器隐藏状态和额外条件参数
        def process_encoder_hidden_states(
            self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    # 定义返回类型为 torch.Tensor
        ) -> torch.Tensor:
            # 检查是否存在隐藏层投影，并且配置为 "text_proj"
            if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
                # 使用文本投影对编码隐藏状态进行转换
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
            # 检查是否存在隐藏层投影，并且配置为 "text_image_proj"
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
                # 检查条件中是否包含 "image_embeds"
                if "image_embeds" not in added_cond_kwargs:
                    # 抛出错误提示缺少必要参数
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
    
                # 获取传入的图像嵌入
                image_embeds = added_cond_kwargs.get("image_embeds")
                # 对编码隐藏状态和图像嵌入进行投影转换
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
            # 检查是否存在隐藏层投影，并且配置为 "image_proj"
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
                # 检查条件中是否包含 "image_embeds"
                if "image_embeds" not in added_cond_kwargs:
                    # 抛出错误提示缺少必要参数
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
                # 获取传入的图像嵌入
                image_embeds = added_cond_kwargs.get("image_embeds")
                # 使用图像嵌入对编码隐藏状态进行投影转换
                encoder_hidden_states = self.encoder_hid_proj(image_embeds)
            # 检查是否存在隐藏层投影，并且配置为 "ip_image_proj"
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
                # 检查条件中是否包含 "image_embeds"
                if "image_embeds" not in added_cond_kwargs:
                    # 抛出错误提示缺少必要参数
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
    
                # 如果存在文本编码器的隐藏层投影，则对编码隐藏状态进行投影转换
                if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                    encoder_hidden_states = self.text_encoder_hid_proj(encoder_hidden_states)
    
                # 获取传入的图像嵌入
                image_embeds = added_cond_kwargs.get("image_embeds")
                # 对图像嵌入进行投影转换
                image_embeds = self.encoder_hid_proj(image_embeds)
                # 将编码隐藏状态和图像嵌入打包成元组
                encoder_hidden_states = (encoder_hidden_states, image_embeds)
            # 返回最终的编码隐藏状态
            return encoder_hidden_states
    # 定义前向传播函数
    def forward(
            # 输入的样本数据，类型为 PyTorch 张量
            sample: torch.Tensor,
            # 当前时间步，类型可以是张量、浮点数或整数
            timestep: Union[torch.Tensor, float, int],
            # 编码器的隐藏状态，类型为 PyTorch 张量
            encoder_hidden_states: torch.Tensor,
            # 可选的类别标签，类型为 PyTorch 张量
            class_labels: Optional[torch.Tensor] = None,
            # 可选的时间步条件，类型为 PyTorch 张量
            timestep_cond: Optional[torch.Tensor] = None,
            # 可选的注意力掩码，类型为 PyTorch 张量
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的交叉注意力参数，类型为字典，包含额外的关键字参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的附加条件参数，类型为字典，键为字符串，值为 PyTorch 张量
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            # 可选的下层块附加残差，类型为元组，包含 PyTorch 张量
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            # 可选的中间块附加残差，类型为 PyTorch 张量
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            # 可选的下层内部块附加残差，类型为元组，包含 PyTorch 张量
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            # 可选的编码器注意力掩码，类型为 PyTorch 张量
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 返回结果的标志，布尔值，默认值为 True
            return_dict: bool = True,
```