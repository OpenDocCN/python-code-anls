# `.\diffusers\pipelines\deprecated\versatile_diffusion\modeling_text_unet.py`

```
# 从 typing 模块导入各种类型注解
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 numpy 库，用于数组和矩阵操作
import numpy as np
# 导入 PyTorch 库，进行深度学习模型的构建和训练
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的功能性模块，提供常用操作
import torch.nn.functional as F

# 从 diffusers.utils 模块导入 deprecate 函数，用于处理弃用警告
from diffusers.utils import deprecate

# 导入配置相关的类和函数
from ....configuration_utils import ConfigMixin, register_to_config
# 导入模型相关的基类
from ....models import ModelMixin
# 导入激活函数获取工具
from ....models.activations import get_activation
# 导入注意力处理器相关组件
from ....models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,  # 额外键值注意力处理器
    CROSS_ATTENTION_PROCESSORS,      # 交叉注意力处理器
    Attention,                       # 注意力机制类
    AttentionProcessor,              # 注意力处理器基类
    AttnAddedKVProcessor,            # 额外键值注意力处理器类
    AttnAddedKVProcessor2_0,         # 版本 2.0 的额外键值注意力处理器
    AttnProcessor,                   # 基础注意力处理器
)
# 导入嵌入层相关组件
from ....models.embeddings import (
    GaussianFourierProjection,        # 高斯傅里叶投影类
    ImageHintTimeEmbedding,           # 图像提示时间嵌入类
    ImageProjection,                  # 图像投影类
    ImageTimeEmbedding,               # 图像时间嵌入类
    TextImageProjection,              # 文本图像投影类
    TextImageTimeEmbedding,           # 文本图像时间嵌入类
    TextTimeEmbedding,                # 文本时间嵌入类
    TimestepEmbedding,                # 时间步嵌入类
    Timesteps,                        # 时间步类
)
# 导入 ResNet 相关组件
from ....models.resnet import ResnetBlockCondNorm2D
# 导入 2D 双重变换器模型
from ....models.transformers.dual_transformer_2d import DualTransformer2DModel
# 导入 2D 变换器模型
from ....models.transformers.transformer_2d import Transformer2DModel
# 导入 2D 条件 UNet 输出类
from ....models.unets.unet_2d_condition import UNet2DConditionOutput
# 导入工具函数和常量
from ....utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
# 导入 PyTorch 相关工具函数
from ....utils.torch_utils import apply_freeu

# 创建日志记录器实例
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义获取下采样块的函数
def get_down_block(
    down_block_type,                    # 下采样块类型
    num_layers,                         # 层数
    in_channels,                        # 输入通道数
    out_channels,                       # 输出通道数
    temb_channels,                      # 时间嵌入通道数
    add_downsample,                    # 是否添加下采样
    resnet_eps,                         # ResNet 中的 epsilon 值
    resnet_act_fn,                     # ResNet 激活函数
    num_attention_heads,               # 注意力头数量
    transformer_layers_per_block,      # 每个块中的变换器层数
    attention_type,                    # 注意力类型
    attention_head_dim,                # 注意力头维度
    resnet_groups=None,                 # ResNet 组数（可选）
    cross_attention_dim=None,           # 交叉注意力维度（可选）
    downsample_padding=None,            # 下采样填充（可选）
    dual_cross_attention=False,         # 是否使用双重交叉注意力
    use_linear_projection=False,        # 是否使用线性投影
    only_cross_attention=False,         # 是否仅使用交叉注意力
    upcast_attention=False,             # 是否上升注意力
    resnet_time_scale_shift="default",  # ResNet 时间缩放偏移
    resnet_skip_time_act=False,         # ResNet 是否跳过时间激活
    resnet_out_scale_factor=1.0,       # ResNet 输出缩放因子
    cross_attention_norm=None,          # 交叉注意力归一化（可选）
    dropout=0.0,                       # dropout 概率
):
    # 如果下采样块类型以 "UNetRes" 开头，则去掉前缀
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    # 如果下采样块类型为 "DownBlockFlat"，则返回相应的块实例
    if down_block_type == "DownBlockFlat":
        return DownBlockFlat(
            num_layers=num_layers,        # 层数
            in_channels=in_channels,      # 输入通道数
            out_channels=out_channels,    # 输出通道数
            temb_channels=temb_channels,  # 时间嵌入通道数
            dropout=dropout,              # dropout 概率
            add_downsample=add_downsample, # 是否添加下采样
            resnet_eps=resnet_eps,        # ResNet 中的 epsilon 值
            resnet_act_fn=resnet_act_fn,  # ResNet 激活函数
            resnet_groups=resnet_groups,   # ResNet 组数（可选）
            downsample_padding=downsample_padding, # 下采样填充（可选）
            resnet_time_scale_shift=resnet_time_scale_shift, # ResNet 时间缩放偏移
        )
    # 检查下采样块类型是否为 CrossAttnDownBlockFlat
    elif down_block_type == "CrossAttnDownBlockFlat":
        # 如果没有指定 cross_attention_dim，则抛出值错误
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlockFlat")
        # 创建并返回 CrossAttnDownBlockFlat 实例，传入所需参数
        return CrossAttnDownBlockFlat(
            # 设置网络层数
            num_layers=num_layers,
            # 设置输入通道数
            in_channels=in_channels,
            # 设置输出通道数
            out_channels=out_channels,
            # 设置时间嵌入通道数
            temb_channels=temb_channels,
            # 设置 dropout 比率
            dropout=dropout,
            # 设置是否添加下采样层
            add_downsample=add_downsample,
            # 设置 ResNet 中的 epsilon 参数
            resnet_eps=resnet_eps,
            # 设置 ResNet 激活函数
            resnet_act_fn=resnet_act_fn,
            # 设置 ResNet 组的数量
            resnet_groups=resnet_groups,
            # 设置下采样的填充参数
            downsample_padding=downsample_padding,
            # 设置交叉注意力维度
            cross_attention_dim=cross_attention_dim,
            # 设置注意力头的数量
            num_attention_heads=num_attention_heads,
            # 设置是否使用双交叉注意力
            dual_cross_attention=dual_cross_attention,
            # 设置是否使用线性投影
            use_linear_projection=use_linear_projection,
            # 设置是否仅使用交叉注意力
            only_cross_attention=only_cross_attention,
            # 设置 ResNet 的时间尺度偏移
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 如果下采样块类型不被支持，则抛出值错误
    raise ValueError(f"{down_block_type} is not supported.")
# 根据给定参数创建上采样块的函数
def get_up_block(
    # 上采样块类型
    up_block_type,
    # 网络层数
    num_layers,
    # 输入通道数
    in_channels,
    # 输出通道数
    out_channels,
    # 上一层输出通道数
    prev_output_channel,
    # 条件嵌入通道数
    temb_channels,
    # 是否添加上采样
    add_upsample,
    # ResNet 的 epsilon 值
    resnet_eps,
    # ResNet 的激活函数
    resnet_act_fn,
    # 注意力头数
    num_attention_heads,
    # 每个块的 Transformer 层数
    transformer_layers_per_block,
    # 分辨率索引
    resolution_idx,
    # 注意力类型
    attention_type,
    # 注意力头维度
    attention_head_dim,
    # ResNet 组数，可选参数
    resnet_groups=None,
    # 跨注意力维度，可选参数
    cross_attention_dim=None,
    # 是否使用双重跨注意力
    dual_cross_attention=False,
    # 是否使用线性投影
    use_linear_projection=False,
    # 是否仅使用跨注意力
    only_cross_attention=False,
    # 是否上溯注意力
    upcast_attention=False,
    # ResNet 时间尺度偏移，默认为 "default"
    resnet_time_scale_shift="default",
    # ResNet 是否跳过时间激活
    resnet_skip_time_act=False,
    # ResNet 输出缩放因子
    resnet_out_scale_factor=1.0,
    # 跨注意力归一化类型，可选参数
    cross_attention_norm=None,
    # dropout 概率
    dropout=0.0,
):
    # 如果上采样块类型以 "UNetRes" 开头，去掉前缀
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    # 如果块类型是 "UpBlockFlat"，则返回相应的实例
    if up_block_type == "UpBlockFlat":
        return UpBlockFlat(
            # 传入各个参数
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 如果块类型是 "CrossAttnUpBlockFlat"
    elif up_block_type == "CrossAttnUpBlockFlat":
        # 检查跨注意力维度是否指定
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlockFlat")
        # 返回相应的跨注意力上采样块实例
        return CrossAttnUpBlockFlat(
            # 传入各个参数
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    # 如果块类型不支持，抛出异常
    raise ValueError(f"{up_block_type} is not supported.")


# 定义一个 Fourier 嵌入器类，继承自 nn.Module
class FourierEmbedder(nn.Module):
    # 初始化方法，设置频率和温度
    def __init__(self, num_freqs=64, temperature=100):
        # 调用父类构造函数
        super().__init__()

        # 保存频率数
        self.num_freqs = num_freqs
        # 保存温度
        self.temperature = temperature

        # 计算频率带
        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        # 扩展维度以便后续操作
        freq_bands = freq_bands[None, None, None]
        # 注册频率带为缓冲区，设为非持久性
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    # 定义调用方法，用于处理输入
    def __call__(self, x):
        # 将输入与频率带相乘
        x = self.freq_bands * x.unsqueeze(-1)
        # 返回处理后的结果，包含正弦和余弦
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 1, 3, 4, 2).reshape(*x.shape[:2], -1)


# 定义 GLIGEN 文本边界框投影类，继承自 nn.Module
class GLIGENTextBoundingboxProjection(nn.Module):
    # 初始化方法，设置对象的基本参数
        def __init__(self, positive_len, out_dim, feature_type, fourier_freqs=8):
            # 调用父类的初始化方法
            super().__init__()
            # 存储正样本的长度
            self.positive_len = positive_len
            # 存储输出的维度
            self.out_dim = out_dim
    
            # 初始化傅里叶嵌入器，设置频率数量
            self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
            # 计算位置特征的维度，包含 sin 和 cos
            self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy
    
            # 如果输出维度是元组，取第一个元素
            if isinstance(out_dim, tuple):
                out_dim = out_dim[0]
    
            # 根据特征类型设置线性层
            if feature_type == "text-only":
                self.linears = nn.Sequential(
                    # 第一层线性变换，输入为正样本长度加位置维度
                    nn.Linear(self.positive_len + self.position_dim, 512),
                    # 激活函数使用 SiLU
                    nn.SiLU(),
                    # 第二层线性变换
                    nn.Linear(512, 512),
                    # 激活函数使用 SiLU
                    nn.SiLU(),
                    # 输出层
                    nn.Linear(512, out_dim),
                )
                # 定义一个全为零的参数，用于文本特征的空值处理
                self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
    
            # 处理文本和图像的特征类型
            elif feature_type == "text-image":
                self.linears_text = nn.Sequential(
                    # 第一层线性变换
                    nn.Linear(self.positive_len + self.position_dim, 512),
                    # 激活函数使用 SiLU
                    nn.SiLU(),
                    # 第二层线性变换
                    nn.Linear(512, 512),
                    # 激活函数使用 SiLU
                    nn.SiLU(),
                    # 输出层
                    nn.Linear(512, out_dim),
                )
                self.linears_image = nn.Sequential(
                    # 第一层线性变换
                    nn.Linear(self.positive_len + self.position_dim, 512),
                    # 激活函数使用 SiLU
                    nn.SiLU(),
                    # 第二层线性变换
                    nn.Linear(512, 512),
                    # 激活函数使用 SiLU
                    nn.SiLU(),
                    # 输出层
                    nn.Linear(512, out_dim),
                )
                # 定义文本特征的空值处理参数
                self.null_text_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
                # 定义图像特征的空值处理参数
                self.null_image_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
    
            # 定义位置特征的空值处理参数
            self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
    
        # 前向传播方法定义
        def forward(
            self,
            boxes,
            masks,
            positive_embeddings=None,
            phrases_masks=None,
            image_masks=None,
            phrases_embeddings=None,
            image_embeddings=None,
    ):
        # 在最后一维增加一个维度，便于后续操作
        masks = masks.unsqueeze(-1)

        # 通过傅里叶嵌入函数生成 boxes 的嵌入表示
        xyxy_embedding = self.fourier_embedder(boxes)
        # 获取空白位置的特征，并调整形状为 (1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        # 计算加权嵌入，结合 masks 和空白位置特征
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        # 如果存在正样本嵌入
        if positive_embeddings:
            # 获取正样本的空白特征，并调整形状为 (1, 1, -1)
            positive_null = self.null_positive_feature.view(1, 1, -1)
            # 计算正样本嵌入的加权，结合 masks 和空白特征
            positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null

            # 将正样本嵌入与 xyxy 嵌入连接并通过线性层处理
            objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        else:
            # 在最后一维增加一个维度，便于后续操作
            phrases_masks = phrases_masks.unsqueeze(-1)
            image_masks = image_masks.unsqueeze(-1)

            # 获取文本和图像的空白特征，并调整形状为 (1, 1, -1)
            text_null = self.null_text_feature.view(1, 1, -1)
            image_null = self.null_image_feature.view(1, 1, -1)

            # 计算文本嵌入的加权，结合 phrases_masks 和空白特征
            phrases_embeddings = phrases_embeddings * phrases_masks + (1 - phrases_masks) * text_null
            # 计算图像嵌入的加权，结合 image_masks 和空白特征
            image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null

            # 将文本嵌入与 xyxy 嵌入连接并通过文本线性层处理
            objs_text = self.linears_text(torch.cat([phrases_embeddings, xyxy_embedding], dim=-1))
            # 将图像嵌入与 xyxy 嵌入连接并通过图像线性层处理
            objs_image = self.linears_image(torch.cat([image_embeddings, xyxy_embedding], dim=-1))
            # 将文本和图像的处理结果在维度 1 上连接
            objs = torch.cat([objs_text, objs_image], dim=1)

        # 返回最终的对象结果
        return objs
# 定义一个名为 UNetFlatConditionModel 的类，继承自 ModelMixin 和 ConfigMixin
class UNetFlatConditionModel(ModelMixin, ConfigMixin):
    r"""
    一个条件 2D UNet 模型，它接收一个有噪声的样本、条件状态和时间步，并返回一个样本形状的输出。

    该模型继承自 [`ModelMixin`]。请查看父类文档以了解其为所有模型实现的通用方法（例如下载或保存）。

    """

    # 设置该模型支持梯度检查点
    _supports_gradient_checkpointing = True
    # 定义不进行拆分的模块名称列表
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlockFlat", "CrossAttnUpBlockFlat"]

    # 注册到配置的装饰器
    @register_to_config
    # 初始化方法，设置类的基本参数
        def __init__(
            # 样本大小，可选参数
            self,
            sample_size: Optional[int] = None,
            # 输入通道数，默认为4
            in_channels: int = 4,
            # 输出通道数，默认为4
            out_channels: int = 4,
            # 是否将输入样本居中，默认为False
            center_input_sample: bool = False,
            # 是否将正弦函数翻转为余弦函数，默认为True
            flip_sin_to_cos: bool = True,
            # 频率偏移量，默认为0
            freq_shift: int = 0,
            # 向下采样块的类型，默认为三个CrossAttnDownBlockFlat和一个DownBlockFlat
            down_block_types: Tuple[str] = (
                "CrossAttnDownBlockFlat",
                "CrossAttnDownBlockFlat",
                "CrossAttnDownBlockFlat",
                "DownBlockFlat",
            ),
            # 中间块的类型，默认为UNetMidBlockFlatCrossAttn
            mid_block_type: Optional[str] = "UNetMidBlockFlatCrossAttn",
            # 向上采样块的类型，默认为一个UpBlockFlat和三个CrossAttnUpBlockFlat
            up_block_types: Tuple[str] = (
                "UpBlockFlat",
                "CrossAttnUpBlockFlat",
                "CrossAttnUpBlockFlat",
                "CrossAttnUpBlockFlat",
            ),
            # 是否仅使用交叉注意力，默认为False
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            # 块输出通道数，默认为320, 640, 1280, 1280
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            # 每个块的层数，默认为2
            layers_per_block: Union[int, Tuple[int]] = 2,
            # 向下采样时的填充大小，默认为1
            downsample_padding: int = 1,
            # 中间块的缩放因子，默认为1
            mid_block_scale_factor: float = 1,
            # dropout比例，默认为0.0
            dropout: float = 0.0,
            # 激活函数类型，默认为silu
            act_fn: str = "silu",
            # 归一化的组数，可选参数，默认为32
            norm_num_groups: Optional[int] = 32,
            # 归一化的epsilon值，默认为1e-5
            norm_eps: float = 1e-5,
            # 交叉注意力的维度，默认为1280
            cross_attention_dim: Union[int, Tuple[int]] = 1280,
            # 每个块的变换器层数，默认为1
            transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
            # 反向变换器层数的可选配置
            reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
            # 编码器隐藏维度的可选参数
            encoder_hid_dim: Optional[int] = None,
            # 编码器隐藏维度类型的可选参数
            encoder_hid_dim_type: Optional[str] = None,
            # 注意力头的维度，默认为8
            attention_head_dim: Union[int, Tuple[int]] = 8,
            # 注意力头数量的可选参数
            num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
            # 是否使用双交叉注意力，默认为False
            dual_cross_attention: bool = False,
            # 是否使用线性投影，默认为False
            use_linear_projection: bool = False,
            # 类嵌入类型的可选参数
            class_embed_type: Optional[str] = None,
            # 附加嵌入类型的可选参数
            addition_embed_type: Optional[str] = None,
            # 附加时间嵌入维度的可选参数
            addition_time_embed_dim: Optional[int] = None,
            # 类嵌入数量的可选参数
            num_class_embeds: Optional[int] = None,
            # 是否向上投射注意力，默认为False
            upcast_attention: bool = False,
            # ResNet时间缩放偏移的默认值
            resnet_time_scale_shift: str = "default",
            # ResNet跳过时间激活的设置，默认为False
            resnet_skip_time_act: bool = False,
            # ResNet输出缩放因子，默认为1.0
            resnet_out_scale_factor: int = 1.0,
            # 时间嵌入类型，默认为positional
            time_embedding_type: str = "positional",
            # 时间嵌入维度的可选参数
            time_embedding_dim: Optional[int] = None,
            # 时间嵌入激活函数的可选参数
            time_embedding_act_fn: Optional[str] = None,
            # 时间步后激活的可选参数
            timestep_post_act: Optional[str] = None,
            # 时间条件投影维度的可选参数
            time_cond_proj_dim: Optional[int] = None,
            # 输入卷积核的大小，默认为3
            conv_in_kernel: int = 3,
            # 输出卷积核的大小，默认为3
            conv_out_kernel: int = 3,
            # 投影类嵌入输入维度的可选参数
            projection_class_embeddings_input_dim: Optional[int] = None,
            # 注意力类型，默认为default
            attention_type: str = "default",
            # 类嵌入是否连接，默认为False
            class_embeddings_concat: bool = False,
            # 中间块是否仅使用交叉注意力的可选参数
            mid_block_only_cross_attention: Optional[bool] = None,
            # 交叉注意力的归一化类型的可选参数
            cross_attention_norm: Optional[str] = None,
            # 附加嵌入类型的头数量，默认为64
            addition_embed_type_num_heads=64,
        # 声明该方法为属性
        @property
    # 定义一个返回注意力处理器字典的方法
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回值:
            `dict` 的注意力处理器: 一个字典，包含模型中使用的所有注意力处理器，以其权重名称为索引。
        """
        # 初始化一个空字典以递归存储处理器
        processors = {}

        # 定义一个递归函数来添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块有获取处理器的方法，则添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历模块的子模块，递归调用函数
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回处理器字典
            return processors

        # 遍历当前模块的子模块，并调用递归函数
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        # 返回所有注意力处理器的字典
        return processors

    # 定义一个设置注意力处理器的方法
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。

        参数:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                实例化的处理器类或处理器类的字典，将被设置为所有 `Attention` 层的处理器。

                如果 `processor` 是字典，则键需要定义对应的交叉注意力处理器的路径。
                在设置可训练的注意力处理器时，强烈推荐这种做法。
        """
        # 计算当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 如果传入的是字典且数量不匹配，则引发错误
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入的是处理器字典，但处理器的数量 {len(processor)} 与注意力层的数量 {count} 不匹配。"
                f" 请确保传入 {count} 个处理器类。"
            )

        # 定义一个递归函数来设置注意力处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块有设置处理器的方法，则根据传入的处理器设置
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的子模块，递归调用函数
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前模块的子模块，并调用递归函数
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)
    # 设置默认的注意力处理器
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器，并设置默认的注意力实现。
        """
        # 检查所有注意力处理器是否属于已添加的 KV 注意力处理器
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 使用 AttnAddedKVProcessor 作为处理器
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否属于交叉注意力处理器
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 使用 AttnProcessor 作为处理器
            processor = AttnProcessor()
        else:
            # 如果处理器类型不匹配，则引发值错误
            raise ValueError(
                f"当注意力处理器的类型为 {next(iter(self.attn_processors.values()))} 时，无法调用 `set_default_attn_processor`"
            )

        # 设置选定的注意力处理器
        self.set_attn_processor(processor)

    # 设置梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块具有 gradient_checkpointing 属性，则设置其值
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # 启用 FreeU 机制
    def enable_freeu(self, s1, s2, b1, b2):
        r"""启用来自 https://arxiv.org/abs/2309.11497 的 FreeU 机制。

        缩放因子的后缀表示应用的阶段块。

        请参考 [官方库](https://github.com/ChenyangSi/FreeU) 以获取已知在不同管道（如 Stable Diffusion v1、v2 和 Stable Diffusion XL）中表现良好的值组合。

        参数:
            s1 (`float`):
                阶段 1 的缩放因子，用于减弱跳过特征的贡献。这是为了减轻增强去噪过程中的“过平滑效应”。
            s2 (`float`):
                阶段 2 的缩放因子，用于减弱跳过特征的贡献。这是为了减轻增强去噪过程中的“过平滑效应”。
            b1 (`float`): 阶段 1 的缩放因子，用于增强主干特征的贡献。
            b2 (`float`): 阶段 2 的缩放因子，用于增强主干特征的贡献。
        """
        # 遍历上采样块并设置相应的缩放因子
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)  # 设置阶段 1 的缩放因子
            setattr(upsample_block, "s2", s2)  # 设置阶段 2 的缩放因子
            setattr(upsample_block, "b1", b1)  # 设置阶段 1 的主干特征缩放因子
            setattr(upsample_block, "b2", b2)  # 设置阶段 2 的主干特征缩放因子

    # 禁用 FreeU 机制
    def disable_freeu(self):
        """禁用 FreeU 机制。"""
        freeu_keys = {"s1", "s2", "b1", "b2"}  # FreeU 机制的关键字集合
        # 遍历上采样块并将关键字的值设置为 None
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                # 如果上采样块具有该属性或属性值不为 None，则将其设置为 None
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)
    # 定义一个用于融合 QKV 投影的函数
    def fuse_qkv_projections(self):
        # 文档字符串，描述该函数的作用及实验性质
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.
    
        <Tip warning={true}>
    
        This API is 🧪 experimental.
    
        </Tip>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None
    
        # 遍历所有注意力处理器
        for _, attn_processor in self.attn_processors.items():
            # 检查处理器的类名是否包含 "Added"
            if "Added" in str(attn_processor.__class__.__name__):
                # 如果是，抛出错误提示不支持融合
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")
    
        # 保存原始的注意力处理器
        self.original_attn_processors = self.attn_processors
    
        # 遍历所有模块
        for module in self.modules():
            # 检查模块是否是 Attention 类的实例
            if isinstance(module, Attention):
                # 融合投影
                module.fuse_projections(fuse=True)
    
    # 定义一个用于取消 QKV 投影融合的函数
    def unfuse_qkv_projections(self):
        # 文档字符串，描述该函数的作用及实验性质
        """Disables the fused QKV projection if enabled.
    
        <Tip warning={true}>
    
        This API is 🧪 experimental.
    
        </Tip>
    
        """
        # 检查原始注意力处理器是否不为 None
        if self.original_attn_processors is not None:
            # 恢复到原始的注意力处理器
            self.set_attn_processor(self.original_attn_processors)
    
    # 定义一个用于卸载 LoRA 权重的函数
    def unload_lora(self):
        # 文档字符串，描述该函数的作用
        """Unloads LoRA weights."""
        # 发出卸载的弃用警告
        deprecate(
            "unload_lora",
            "0.28.0",
            "Calling `unload_lora()` is deprecated and will be removed in a future version. Please install `peft` and then call `disable_adapters().",
        )
        # 遍历所有模块
        for module in self.modules():
            # 检查模块是否具有 set_lora_layer 属性
            if hasattr(module, "set_lora_layer"):
                # 将 LoRA 层设置为 None
                module.set_lora_layer(None)
    
    # 定义前向传播函数
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
# 定义一个继承自 nn.Linear 的线性多维层
class LinearMultiDim(nn.Linear):
    # 初始化方法，接受输入特征、输出特征及其他参数
    def __init__(self, in_features, out_features=None, second_dim=4, *args, **kwargs):
        # 如果 in_features 是整数，则将其转换为包含三个维度的列表
        in_features = [in_features, second_dim, 1] if isinstance(in_features, int) else list(in_features)
        # 如果未提供 out_features，则将其设置为 in_features
        if out_features is None:
            out_features = in_features
        # 如果 out_features 是整数，则转换为包含三个维度的列表
        out_features = [out_features, second_dim, 1] if isinstance(out_features, int) else list(out_features)
        # 保存输入特征的多维信息
        self.in_features_multidim = in_features
        # 保存输出特征的多维信息
        self.out_features_multidim = out_features
        # 调用父类的初始化方法，计算输入和输出特征的总数量
        super().__init__(np.array(in_features).prod(), np.array(out_features).prod())

    # 定义前向传播方法
    def forward(self, input_tensor, *args, **kwargs):
        # 获取输入张量的形状
        shape = input_tensor.shape
        # 获取输入特征的维度数量
        n_dim = len(self.in_features_multidim)
        # 将输入张量重塑为适合线性层的形状
        input_tensor = input_tensor.reshape(*shape[0:-n_dim], self.in_features)
        # 调用父类的前向传播方法，得到输出张量
        output_tensor = super().forward(input_tensor)
        # 将输出张量重塑为目标形状
        output_tensor = output_tensor.view(*shape[0:-n_dim], *self.out_features_multidim)
        # 返回输出张量
        return output_tensor


# 定义一个平坦的残差块类，继承自 nn.Module
class ResnetBlockFlat(nn.Module):
    # 初始化方法，接受多个参数，包括通道数、丢弃率等
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        time_embedding_norm="default",
        use_in_shortcut=None,
        second_dim=4,
        **kwargs,
    # 初始化方法的结束，接收参数
        ):
            # 调用父类的初始化方法
            super().__init__()
            # 是否进行预归一化，设置为传入的值
            self.pre_norm = pre_norm
            # 将预归一化设置为 True
            self.pre_norm = True
    
            # 如果输入通道是整数，则构造一个包含三个维度的列表
            in_channels = [in_channels, second_dim, 1] if isinstance(in_channels, int) else list(in_channels)
            # 计算输入通道数的乘积
            self.in_channels_prod = np.array(in_channels).prod()
            # 保存输入通道的多维信息
            self.channels_multidim = in_channels
    
            # 如果输出通道不为 None
            if out_channels is not None:
                # 如果输出通道是整数，构造一个包含三个维度的列表
                out_channels = [out_channels, second_dim, 1] if isinstance(out_channels, int) else list(out_channels)
                # 计算输出通道数的乘积
                out_channels_prod = np.array(out_channels).prod()
                # 保存输出通道的多维信息
                self.out_channels_multidim = out_channels
            else:
                # 如果输出通道为 None，则输出通道乘积等于输入通道乘积
                out_channels_prod = self.in_channels_prod
                # 输出通道的多维信息与输入通道相同
                self.out_channels_multidim = self.channels_multidim
            # 保存时间嵌入的归一化状态
            self.time_embedding_norm = time_embedding_norm
    
            # 如果输出组数为 None，使用传入的组数
            if groups_out is None:
                groups_out = groups
    
            # 创建第一个归一化层，使用组归一化
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=self.in_channels_prod, eps=eps, affine=True)
            # 创建第一个卷积层，使用输入通道和输出通道乘积
            self.conv1 = torch.nn.Conv2d(self.in_channels_prod, out_channels_prod, kernel_size=1, padding=0)
    
            # 如果时间嵌入通道不为 None
            if temb_channels is not None:
                # 创建时间嵌入投影层
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels_prod)
            else:
                # 如果时间嵌入通道为 None，则不进行投影
                self.time_emb_proj = None
    
            # 创建第二个归一化层，使用输出组数和输出通道乘积
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels_prod, eps=eps, affine=True)
            # 创建丢弃层，使用传入的丢弃率
            self.dropout = torch.nn.Dropout(dropout)
            # 创建第二个卷积层，使用输出通道乘积
            self.conv2 = torch.nn.Conv2d(out_channels_prod, out_channels_prod, kernel_size=1, padding=0)
    
            # 设置非线性激活函数为 SiLU
            self.nonlinearity = nn.SiLU()
    
            # 检查是否使用输入短路，如果短路使用参数为 None，则根据通道数判断
            self.use_in_shortcut = (
                self.in_channels_prod != out_channels_prod if use_in_shortcut is None else use_in_shortcut
            )
    
            # 初始化快捷连接卷积为 None
            self.conv_shortcut = None
            # 如果使用输入短路
            if self.use_in_shortcut:
                # 创建快捷连接卷积层
                self.conv_shortcut = torch.nn.Conv2d(
                    self.in_channels_prod, out_channels_prod, kernel_size=1, stride=1, padding=0
                )
    # 定义前向传播方法，接收输入张量和时间嵌入
        def forward(self, input_tensor, temb):
            # 获取输入张量的形状
            shape = input_tensor.shape
            # 获取多维通道的维度数
            n_dim = len(self.channels_multidim)
            # 调整输入张量形状，合并通道维度并增加两个维度
            input_tensor = input_tensor.reshape(*shape[0:-n_dim], self.in_channels_prod, 1, 1)
            # 将张量视图转换为指定形状，保持通道数并增加两个维度
            input_tensor = input_tensor.view(-1, self.in_channels_prod, 1, 1)
    
            # 初始化隐藏状态为输入张量
            hidden_states = input_tensor
    
            # 对隐藏状态进行归一化处理
            hidden_states = self.norm1(hidden_states)
            # 应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
            # 通过第一个卷积层处理隐藏状态
            hidden_states = self.conv1(hidden_states)
    
            # 如果时间嵌入不为空
            if temb is not None:
                # 对时间嵌入进行非线性处理并调整形状
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
                # 将时间嵌入与隐藏状态相加
                hidden_states = hidden_states + temb
    
            # 对隐藏状态进行第二次归一化处理
            hidden_states = self.norm2(hidden_states)
            # 再次应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
    
            # 对隐藏状态应用 dropout 操作
            hidden_states = self.dropout(hidden_states)
            # 通过第二个卷积层处理隐藏状态
            hidden_states = self.conv2(hidden_states)
    
            # 如果存在短路卷积层
            if self.conv_shortcut is not None:
                # 通过短路卷积层处理输入张量
                input_tensor = self.conv_shortcut(input_tensor)
    
            # 将输入张量与隐藏状态相加，生成输出张量
            output_tensor = input_tensor + hidden_states
    
            # 将输出张量调整为指定形状，去掉多余的维度
            output_tensor = output_tensor.view(*shape[0:-n_dim], -1)
            # 再次调整输出张量的形状，匹配输出通道的多维结构
            output_tensor = output_tensor.view(*shape[0:-n_dim], *self.out_channels_multidim)
    
            # 返回最终的输出张量
            return output_tensor
# 定义一个名为 DownBlockFlat 的类，继承自 nn.Module
class DownBlockFlat(nn.Module):
    # 初始化方法，接受多个参数用于配置模型
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # ResNet 层数
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值
        resnet_time_scale_shift: str = "default",  # ResNet 的时间缩放偏移
        resnet_act_fn: str = "swish",  # ResNet 的激活函数
        resnet_groups: int = 32,  # ResNet 的分组数
        resnet_pre_norm: bool = True,  # 是否在 ResNet 前进行归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_downsample: bool = True,  # 是否添加下采样层
        downsample_padding: int = 1,  # 下采样时的填充
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个空列表，用于存放 ResNet 层
        resnets = []

        # 循环创建指定数量的 ResNet 层
        for i in range(num_layers):
            # 第一层使用输入通道，之后的层使用输出通道
            in_channels = in_channels if i == 0 else out_channels
            # 将 ResNet 层添加到列表中
            resnets.append(
                ResnetBlockFlat(
                    in_channels=in_channels,  # 当前层的输入通道数
                    out_channels=out_channels,  # 当前层的输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 值
                    groups=resnet_groups,  # 分组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                    non_linearity=resnet_act_fn,  # 激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否前归一化
                )
            )

        # 将 ResNet 层列表转为 nn.ModuleList 以便于管理
        self.resnets = nn.ModuleList(resnets)

        # 根据参数决定是否添加下采样层
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    LinearMultiDim(
                        out_channels,  # 输入通道数
                        use_conv=True,  # 使用卷积
                        out_channels=out_channels,  # 输出通道数
                        padding=downsample_padding,  # 填充
                        name="op"  # 下采样层名称
                    )
                ]
            )
        else:
            # 如果不添加下采样层，设置为 None
            self.downsamplers = None

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False

    # 定义前向传播方法
    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None  # 输入的隐藏状态和可选的时间嵌入
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 初始化输出状态为一个空元组
        output_states = ()

        # 遍历所有的 ResNet 层
        for resnet in self.resnets:
            # 如果在训练模式且开启了梯度检查点
            if self.training and self.gradient_checkpointing:
                # 定义一个创建自定义前向传播的方法
                def create_custom_forward(module):
                    # 定义自定义前向传播函数
                    def custom_forward(*inputs):
                        return module(*inputs)  # 调用模块进行前向传播

                    return custom_forward

                # 检查 PyTorch 版本，使用不同的调用方式
                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False  # 进行梯度检查点的前向传播
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb  # 进行梯度检查点的前向传播
                    )
            else:
                # 正常调用 ResNet 层进行前向传播
                hidden_states = resnet(hidden_states, temb)

            # 将当前隐藏状态添加到输出状态中
            output_states = output_states + (hidden_states,)

        # 如果存在下采样层
        if self.downsamplers is not None:
            # 遍历所有下采样层
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)  # 对隐藏状态进行下采样

            # 将下采样后的隐藏状态添加到输出状态中
            output_states = output_states + (hidden_states,)

        # 返回最终的隐藏状态和所有输出状态
        return hidden_states, output_states
# 定义一个名为 CrossAttnDownBlockFlat 的类，继承自 nn.Module
class CrossAttnDownBlockFlat(nn.Module):
    # 初始化方法，定义类的属性
    def __init__(
        # 输入通道数
        self,
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # 每个块的变换器层数，可以是整数或整数元组，默认为 1
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ResNet 的 epsilon 值，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 的时间尺度偏移设置，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 的组数，默认为 32
        resnet_groups: int = 32,
        # 是否使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头的数量，默认为 1
        num_attention_heads: int = 1,
        # 交叉注意力的维度，默认为 1280
        cross_attention_dim: int = 1280,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 下采样的填充大小，默认为 1
        downsample_padding: int = 1,
        # 是否添加下采样层，默认为 True
        add_downsample: bool = True,
        # 是否使用双重交叉注意力，默认为 False
        dual_cross_attention: bool = False,
        # 是否使用线性投影，默认为 False
        use_linear_projection: bool = False,
        # 是否只使用交叉注意力，默认为 False
        only_cross_attention: bool = False,
        # 是否上溯注意力，默认为 False
        upcast_attention: bool = False,
        # 注意力类型，默认为 "default"
        attention_type: str = "default",
    ):
        # 调用父类的构造函数以初始化基类
        super().__init__()
        # 初始化存储 ResNet 块的列表
        resnets = []
        # 初始化存储注意力模型的列表
        attentions = []

        # 设置是否使用交叉注意力的标志
        self.has_cross_attention = True
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 如果 transformer_layers_per_block 是一个整数，则将其转换为列表形式
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # 为每一层构建 ResNet 块和注意力模型
        for i in range(num_layers):
            # 设置当前层的输入通道数，第一层使用 in_channels，其他层使用 out_channels
            in_channels = in_channels if i == 0 else out_channels
            # 向 resnets 列表添加一个 ResNet 块
            resnets.append(
                ResnetBlockFlat(
                    # 设置 ResNet 块的输入通道数
                    in_channels=in_channels,
                    # 设置 ResNet 块的输出通道数
                    out_channels=out_channels,
                    # 设置时间嵌入通道数
                    temb_channels=temb_channels,
                    # 设置 ResNet 块的 epsilon 值
                    eps=resnet_eps,
                    # 设置 ResNet 块的组数
                    groups=resnet_groups,
                    # 设置 dropout 概率
                    dropout=dropout,
                    # 设置时间嵌入的归一化方法
                    time_embedding_norm=resnet_time_scale_shift,
                    # 设置激活函数
                    non_linearity=resnet_act_fn,
                    # 设置输出缩放因子
                    output_scale_factor=output_scale_factor,
                    # 设置是否在前面进行归一化
                    pre_norm=resnet_pre_norm,
                )
            )
            # 如果不使用双交叉注意力
            if not dual_cross_attention:
                # 向 attentions 列表添加一个 Transformer 2D 模型
                attentions.append(
                    Transformer2DModel(
                        # 设置注意力头的数量
                        num_attention_heads,
                        # 设置每个注意力头的输出通道数
                        out_channels // num_attention_heads,
                        # 设置输入通道数
                        in_channels=out_channels,
                        # 设置当前层的 Transformer 层数
                        num_layers=transformer_layers_per_block[i],
                        # 设置交叉注意力的维度
                        cross_attention_dim=cross_attention_dim,
                        # 设置归一化的组数
                        norm_num_groups=resnet_groups,
                        # 设置是否使用线性投影
                        use_linear_projection=use_linear_projection,
                        # 设置是否仅使用交叉注意力
                        only_cross_attention=only_cross_attention,
                        # 设置是否提高注意力精度
                        upcast_attention=upcast_attention,
                        # 设置注意力类型
                        attention_type=attention_type,
                    )
                )
            else:
                # 向 attentions 列表添加一个双 Transformer 2D 模型
                attentions.append(
                    DualTransformer2DModel(
                        # 设置注意力头的数量
                        num_attention_heads,
                        # 设置每个注意力头的输出通道数
                        out_channels // num_attention_heads,
                        # 设置输入通道数
                        in_channels=out_channels,
                        # 固定层数为 1
                        num_layers=1,
                        # 设置交叉注意力的维度
                        cross_attention_dim=cross_attention_dim,
                        # 设置归一化的组数
                        norm_num_groups=resnet_groups,
                    )
                )
        # 将注意力模型列表转换为 PyTorch 的 ModuleList
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 块列表转换为 PyTorch 的 ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 如果需要添加下采样层
        if add_downsample:
            # 初始化下采样层为 ModuleList
            self.downsamplers = nn.ModuleList(
                [
                    LinearMultiDim(
                        # 设置输出通道数
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            # 如果不添加下采样层，将其设为 None
            self.downsamplers = None

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
    # 定义前向传播函数，接收隐藏状态和其他可选参数
        def forward(
            self,
            hidden_states: torch.Tensor,  # 当前隐藏状态的张量
            temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态张量
            attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数
            encoder_attention_mask: Optional[torch.Tensor] = None,  # 可选的编码器注意力掩码
            additional_residuals: Optional[torch.Tensor] = None,  # 可选的额外残差张量
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:  # 返回隐藏状态和输出状态元组
            output_states = ()  # 初始化输出状态元组
    
            blocks = list(zip(self.resnets, self.attentions))  # 将残差网络和注意力模块配对成块
    
            for i, (resnet, attn) in enumerate(blocks):  # 遍历每个块及其索引
                if self.training and self.gradient_checkpointing:  # 检查是否在训练且启用梯度检查点
    
                    def create_custom_forward(module, return_dict=None):  # 定义自定义前向传播函数
                        def custom_forward(*inputs):  # 自定义前向传播逻辑
                            if return_dict is not None:  # 如果提供了返回字典
                                return module(*inputs, return_dict=return_dict)  # 返回带字典的结果
                            else:
                                return module(*inputs)  # 否则返回普通结果
    
                        return custom_forward  # 返回自定义前向传播函数
    
                    # 设置检查点参数，如果 PyTorch 版本大于等于 1.11.0，则使用非重入模式
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    # 通过检查点机制计算当前块的隐藏状态
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 创建自定义前向函数的检查点
                        hidden_states,  # 输入当前隐藏状态
                        temb,  # 输入时间嵌入
                        **ckpt_kwargs,  # 传递检查点参数
                    )
                    # 通过注意力模块处理隐藏状态并获取输出
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                        cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                        attention_mask=attention_mask,  # 注意力掩码
                        encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                        return_dict=False,  # 不返回字典格式
                    )[0]  # 取出第一个输出
                else:  # 如果不启用梯度检查
                    # 直接通过残差网络处理隐藏状态
                    hidden_states = resnet(hidden_states, temb)
                    # 通过注意力模块处理隐藏状态并获取输出
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                        cross_attention_kwargs=cross_attention_kwargs,  # 交叉注意力参数
                        attention_mask=attention_mask,  # 注意力掩码
                        encoder_attention_mask=encoder_attention_mask,  # 编码器注意力掩码
                        return_dict=False,  # 不返回字典格式
                    )[0]  # 取出第一个输出
    
                # 如果是最后一个块并且提供了额外残差，则将其添加到隐藏状态
                if i == len(blocks) - 1 and additional_residuals is not None:
                    hidden_states = hidden_states + additional_residuals  # 加上额外残差
    
                output_states = output_states + (hidden_states,)  # 将当前隐藏状态添加到输出状态元组中
    
            if self.downsamplers is not None:  # 如果存在下采样器
                for downsampler in self.downsamplers:  # 遍历每个下采样器
                    hidden_states = downsampler(hidden_states)  # 处理当前隐藏状态
    
                output_states = output_states + (hidden_states,)  # 将当前隐藏状态添加到输出状态元组中
    
            return hidden_states, output_states  # 返回最终的隐藏状态和输出状态元组
# 从 diffusers.models.unets.unet_2d_blocks 中复制，替换 UpBlock2D 为 UpBlockFlat，ResnetBlock2D 为 ResnetBlockFlat，Upsample2D 为 LinearMultiDim
class UpBlockFlat(nn.Module):
    # 初始化函数，定义输入输出通道及其他参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        prev_output_channel: int,  # 前一层输出通道数
        out_channels: int,  # 当前层输出通道数
        temb_channels: int,  # 时间嵌入通道数
        resolution_idx: Optional[int] = None,  # 分辨率索引
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        resnet_eps: float = 1e-6,  # ResNet 中的 epsilon 值
        resnet_time_scale_shift: str = "default",  # 时间尺度偏移设置
        resnet_act_fn: str = "swish",  # 激活函数类型
        resnet_groups: int = 32,  # 分组数
        resnet_pre_norm: bool = True,  # 是否进行预归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_upsample: bool = True,  # 是否添加上采样
    ):
        # 调用父类构造函数
        super().__init__()
        # 初始化一个空列表存储 ResNet 块
        resnets = []

        # 遍历层数，构建每一层的 ResNet 块
        for i in range(num_layers):
            # 根据层数决定残差跳跃通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 根据当前层数决定输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 将 ResNet 块添加到列表中
            resnets.append(
                ResnetBlockFlat(
                    in_channels=resnet_in_channels + res_skip_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 值
                    groups=resnet_groups,  # 分组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化
                    non_linearity=resnet_act_fn,  # 激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 预归一化
                )
            )

        # 将 ResNet 块列表转换为模块列表
        self.resnets = nn.ModuleList(resnets)

        # 如果需要添加上采样层，则创建上采样模块
        if add_upsample:
            self.upsamplers = nn.ModuleList([LinearMultiDim(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            # 否则设置为 None
            self.upsamplers = None

        # 初始化梯度检查点标志
        self.gradient_checkpointing = False
        # 设置分辨率索引
        self.resolution_idx = resolution_idx

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 残差隐藏状态元组
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        upsample_size: Optional[int] = None,  # 可选的上采样大小
        *args,  # 可变参数
        **kwargs,  # 可变关键字参数
    ) -> torch.Tensor:  # 定义一个返回 torch.Tensor 类型的函数
        # 如果参数列表 args 长度大于 0 或 kwargs 中的 scale 参数不为 None
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 设置弃用消息，提醒用户 scale 参数已弃用且将被忽略
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用 deprecate 函数，记录 scale 参数的弃用
            deprecate("scale", "1.0.0", deprecation_message)

        # 检查 FreeU 是否启用，取决于 s1, s2, b1 和 b2 的值
        is_freeu_enabled = (
            getattr(self, "s1", None)  # 获取 self 中的 s1 属性
            and getattr(self, "s2", None)  # 获取 self 中的 s2 属性
            and getattr(self, "b1", None)  # 获取 self 中的 b1 属性
            and getattr(self, "b2", None)  # 获取 self 中的 b2 属性
        )

        # 遍历 self.resnets 中的每个 ResNet 模型
        for resnet in self.resnets:
            # 弹出 res 隐藏状态的最后一个元素
            res_hidden_states = res_hidden_states_tuple[-1]  
            # 移除 res 隐藏状态元组的最后一个元素
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]  

            # FreeU: 仅在前两个阶段进行操作
            if is_freeu_enabled:
                # 应用 FreeU 操作，返回更新后的 hidden_states 和 res_hidden_states
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,  # 当前分辨率索引
                    hidden_states,  # 当前隐藏状态
                    res_hidden_states,  # 之前的隐藏状态
                    s1=self.s1,  # s1 参数
                    s2=self.s2,  # s2 参数
                    b1=self.b1,  # b1 参数
                    b2=self.b2,  # b2 参数
                )

            # 将当前的 hidden_states 和 res_hidden_states 在维度 1 上拼接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)  

            # 如果处于训练模式并且开启了梯度检查点
            if self.training and self.gradient_checkpointing:
                # 定义一个创建自定义前向函数的函数
                def create_custom_forward(module):
                    # 定义自定义前向函数，接收输入并调用模块
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                # 如果 PyTorch 版本大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 使用梯度检查点来计算 hidden_states
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 使用自定义前向函数
                        hidden_states,  # 当前隐藏状态
                        temb,  # 传入的额外输入
                        use_reentrant=False  # 禁用重入检查
                    )
                else:
                    # 对于早期版本，使用梯度检查点计算 hidden_states
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 使用自定义前向函数
                        hidden_states,  # 当前隐藏状态
                        temb  # 传入的额外输入
                    )
            else:
                # 在非训练模式下直接调用 resnet 处理 hidden_states
                hidden_states = resnet(hidden_states, temb)  

        # 如果存在上采样器
        if self.upsamplers is not None:
            # 遍历所有上采样器
            for upsampler in self.upsamplers:
                # 使用上采样器对 hidden_states 进行处理，指定上采样尺寸
                hidden_states = upsampler(hidden_states, upsample_size)  

        # 返回处理后的 hidden_states
        return hidden_states  
# 从 diffusers.models.unets.unet_2d_blocks 中复制的代码，修改了类名和一些组件
class CrossAttnUpBlockFlat(nn.Module):
    # 初始化方法，定义类的基本属性和参数
    def __init__(
        # 输入通道数
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 上一层输出的通道数
        prev_output_channel: int,
        # 额外的时间嵌入通道数
        temb_channels: int,
        # 可选的分辨率索引
        resolution_idx: Optional[int] = None,
        # dropout 概率
        dropout: float = 0.0,
        # 层数
        num_layers: int = 1,
        # 每个块的变换器层数，可以是单个整数或元组
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ResNet 的 epsilon 值
        resnet_eps: float = 1e-6,
        # ResNet 时间尺度偏移的类型
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数的类型
        resnet_act_fn: str = "swish",
        # ResNet 的组数
        resnet_groups: int = 32,
        # 是否在 ResNet 中使用预归一化
        resnet_pre_norm: bool = True,
        # 注意力头的数量
        num_attention_heads: int = 1,
        # 交叉注意力的维度
        cross_attention_dim: int = 1280,
        # 输出缩放因子
        output_scale_factor: float = 1.0,
        # 是否添加上采样步骤
        add_upsample: bool = True,
        # 是否使用双交叉注意力
        dual_cross_attention: bool = False,
        # 是否使用线性投影
        use_linear_projection: bool = False,
        # 是否仅使用交叉注意力
        only_cross_attention: bool = False,
        # 是否上溯注意力
        upcast_attention: bool = False,
        # 注意力类型
        attention_type: str = "default",
    # 定义构造函数的结束部分
        ):
            # 调用父类的构造函数
            super().__init__()
            # 初始化一个空列表用于存储残差网络块
            resnets = []
            # 初始化一个空列表用于存储注意力模型
            attentions = []
    
            # 设置是否使用交叉注意力标志为真
            self.has_cross_attention = True
            # 设置注意力头的数量
            self.num_attention_heads = num_attention_heads
    
            # 如果 transformer_layers_per_block 是整数，则将其转换为相同长度的列表
            if isinstance(transformer_layers_per_block, int):
                transformer_layers_per_block = [transformer_layers_per_block] * num_layers
    
            # 遍历每一层以构建残差网络和注意力模型
            for i in range(num_layers):
                # 设置残差跳过通道数，最后一层使用输入通道，否则使用输出通道
                res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
                # 设置残差网络输入通道数，第一层使用前一层输出通道，否则使用当前输出通道
                resnet_in_channels = prev_output_channel if i == 0 else out_channels
    
                # 添加一个残差网络块到 resnets 列表中
                resnets.append(
                    ResnetBlockFlat(
                        # 设置残差网络输入通道数
                        in_channels=resnet_in_channels + res_skip_channels,
                        # 设置残差网络输出通道数
                        out_channels=out_channels,
                        # 设置时间嵌入通道数
                        temb_channels=temb_channels,
                        # 设置残差网络的 epsilon 值
                        eps=resnet_eps,
                        # 设置残差网络的组数
                        groups=resnet_groups,
                        # 设置丢弃率
                        dropout=dropout,
                        # 设置时间嵌入的归一化方法
                        time_embedding_norm=resnet_time_scale_shift,
                        # 设置非线性激活函数
                        non_linearity=resnet_act_fn,
                        # 设置输出缩放因子
                        output_scale_factor=output_scale_factor,
                        # 设置是否进行预归一化
                        pre_norm=resnet_pre_norm,
                    )
                )
                # 如果不使用双重交叉注意力
                if not dual_cross_attention:
                    # 添加一个普通的 Transformer2DModel 到 attentions 列表中
                    attentions.append(
                        Transformer2DModel(
                            # 设置注意力头的数量
                            num_attention_heads,
                            # 设置每个注意力头的输出通道数
                            out_channels // num_attention_heads,
                            # 设置输入通道数
                            in_channels=out_channels,
                            # 设置层数
                            num_layers=transformer_layers_per_block[i],
                            # 设置交叉注意力维度
                            cross_attention_dim=cross_attention_dim,
                            # 设置归一化组数
                            norm_num_groups=resnet_groups,
                            # 设置是否使用线性投影
                            use_linear_projection=use_linear_projection,
                            # 设置是否仅使用交叉注意力
                            only_cross_attention=only_cross_attention,
                            # 设置是否上溯注意力
                            upcast_attention=upcast_attention,
                            # 设置注意力类型
                            attention_type=attention_type,
                        )
                    )
                else:
                    # 添加一个双重 Transformer2DModel 到 attentions 列表中
                    attentions.append(
                        DualTransformer2DModel(
                            # 设置注意力头的数量
                            num_attention_heads,
                            # 设置每个注意力头的输出通道数
                            out_channels // num_attention_heads,
                            # 设置输入通道数
                            in_channels=out_channels,
                            # 设置层数为 1
                            num_layers=1,
                            # 设置交叉注意力维度
                            cross_attention_dim=cross_attention_dim,
                            # 设置归一化组数
                            norm_num_groups=resnet_groups,
                        )
                    )
            # 将注意力模型列表转换为 nn.ModuleList
            self.attentions = nn.ModuleList(attentions)
            # 将残差网络块列表转换为 nn.ModuleList
            self.resnets = nn.ModuleList(resnets)
    
            # 如果需要添加上采样层
            if add_upsample:
                # 将上采样器添加到 nn.ModuleList 中
                self.upsamplers = nn.ModuleList([LinearMultiDim(out_channels, use_conv=True, out_channels=out_channels)])
            else:
                # 否则将上采样器设置为 None
                self.upsamplers = None
    
            # 设置梯度检查点标志为假
            self.gradient_checkpointing = False
            # 设置分辨率索引
            self.resolution_idx = resolution_idx
    # 定义前向传播函数，接收多个输入参数
        def forward(
            self,
            # 隐藏状态，类型为 PyTorch 的张量
            hidden_states: torch.Tensor,
            # 包含残差隐藏状态的元组，元素类型为 PyTorch 张量
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            # 可选的时间嵌入，类型为 PyTorch 的张量
            temb: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态，类型为 PyTorch 的张量
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的交叉注意力参数，类型为字典，包含任意键值对
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的上采样大小，类型为整数
            upsample_size: Optional[int] = None,
            # 可选的注意力掩码，类型为 PyTorch 的张量
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的编码器注意力掩码，类型为 PyTorch 的张量
            encoder_attention_mask: Optional[torch.Tensor] = None,
# 从 diffusers.models.unets.unet_2d_blocks 中复制的 UNetMidBlock2D 代码，替换了 UNetMidBlock2D 为 UNetMidBlockFlat，ResnetBlock2D 为 ResnetBlockFlat
class UNetMidBlockFlat(nn.Module):
    """
    2D UNet 中间块 [`UNetMidBlockFlat`]，包含多个残差块和可选的注意力块。

    参数：
        in_channels (`int`): 输入通道的数量。
        temb_channels (`int`): 时间嵌入通道的数量。
        dropout (`float`, *可选*, 默认值为 0.0): dropout 比率。
        num_layers (`int`, *可选*, 默认值为 1): 残差块的数量。
        resnet_eps (`float`, *可选*, 默认值为 1e-6): resnet 块的 epsilon 值。
        resnet_time_scale_shift (`str`, *可选*, 默认值为 `default`):
            应用于时间嵌入的归一化类型。这可以帮助提高模型在长范围时间依赖任务上的性能。
        resnet_act_fn (`str`, *可选*, 默认值为 `swish`): resnet 块的激活函数。
        resnet_groups (`int`, *可选*, 默认值为 32):
            resnet 块的分组归一化层使用的组数。
        attn_groups (`Optional[int]`, *可选*, 默认值为 None): 注意力块的组数。
        resnet_pre_norm (`bool`, *可选*, 默认值为 `True`):
            是否在 resnet 块中使用预归一化。
        add_attention (`bool`, *可选*, 默认值为 `True`): 是否添加注意力块。
        attention_head_dim (`int`, *可选*, 默认值为 1):
            单个注意力头的维度。注意力头的数量基于此值和输入通道的数量确定。
        output_scale_factor (`float`, *可选*, 默认值为 1.0): 输出缩放因子。

    返回：
        `torch.Tensor`: 最后一个残差块的输出，是一个形状为 `(batch_size, in_channels,
        height, width)` 的张量。

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # 默认，空间
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        # 初始化 UNetMidBlockFlat 类，设置各参数的默认值
        super().__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 保存输入通道数
        self.temb_channels = temb_channels  # 保存时间嵌入通道数
        self.dropout = dropout  # 保存 dropout 比率
        self.num_layers = num_layers  # 保存残差块的数量
        self.resnet_eps = resnet_eps  # 保存 resnet 块的 epsilon 值
        self.resnet_time_scale_shift = resnet_time_scale_shift  # 保存时间缩放偏移类型
        self.resnet_act_fn = resnet_act_fn  # 保存激活函数类型
        self.resnet_groups = resnet_groups  # 保存分组数
        self.attn_groups = attn_groups  # 保存注意力组数
        self.resnet_pre_norm = resnet_pre_norm  # 保存是否使用预归一化
        self.add_attention = add_attention  # 保存是否添加注意力块
        self.attention_head_dim = attention_head_dim  # 保存注意力头的维度
        self.output_scale_factor = output_scale_factor  # 保存输出缩放因子

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 定义前向传播方法，接受隐藏状态和可选的时间嵌入
        hidden_states = self.resnets[0](hidden_states, temb)  # 通过第一个残差块处理隐藏状态
        for attn, resnet in zip(self.attentions, self.resnets[1:]):  # 遍历后续的注意力块和残差块
            if attn is not None:  # 如果注意力块存在
                hidden_states = attn(hidden_states, temb=temb)  # 通过注意力块处理隐藏状态
            hidden_states = resnet(hidden_states, temb)  # 通过残差块处理隐藏状态

        return hidden_states  # 返回处理后的隐藏状态
# 从 diffusers.models.unets.unet_2d_blocks 中复制，替换 UNetMidBlock2DCrossAttn 为 UNetMidBlockFlatCrossAttn，ResnetBlock2D 为 ResnetBlockFlat
class UNetMidBlockFlatCrossAttn(nn.Module):
    # 初始化方法，定义模型参数
    def __init__(
        self,
        # 输入通道数
        in_channels: int,
        # 时间嵌入通道数
        temb_channels: int,
        # 输出通道数，默认为 None
        out_channels: Optional[int] = None,
        # Dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 层数，默认为 1
        num_layers: int = 1,
        # 每个块的 Transformer 层数，默认为 1
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ResNet 的 epsilon 值，默认为 1e-6
        resnet_eps: float = 1e-6,
        # ResNet 的时间尺度偏移，默认为 "default"
        resnet_time_scale_shift: str = "default",
        # ResNet 的激活函数类型，默认为 "swish"
        resnet_act_fn: str = "swish",
        # ResNet 的分组数，默认为 32
        resnet_groups: int = 32,
        # 输出的 ResNet 分组数，默认为 None
        resnet_groups_out: Optional[int] = None,
        # 是否使用预归一化，默认为 True
        resnet_pre_norm: bool = True,
        # 注意力头数，默认为 1
        num_attention_heads: int = 1,
        # 输出缩放因子，默认为 1.0
        output_scale_factor: float = 1.0,
        # 交叉注意力维度，默认为 1280
        cross_attention_dim: int = 1280,
        # 是否使用双交叉注意力，默认为 False
        dual_cross_attention: bool = False,
        # 是否使用线性投影，默认为 False
        use_linear_projection: bool = False,
        # 是否上升注意力计算精度，默认为 False
        upcast_attention: bool = False,
        # 注意力类型，默认为 "default"
        attention_type: str = "default",
    # 前向传播方法，定义模型的前向计算逻辑
    def forward(
        self,
        # 隐藏状态张量
        hidden_states: torch.Tensor,
        # 可选的时间嵌入张量，默认为 None
        temb: Optional[torch.Tensor] = None,
        # 可选的编码器隐藏状态张量，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 可选的注意力掩码，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的交叉注意力参数字典，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 可选的编码器注意力掩码，默认为 None
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # 定义函数的返回类型为 torch.Tensor
        if cross_attention_kwargs is not None:  # 检查 cross_attention_kwargs 是否为 None
            if cross_attention_kwargs.get("scale", None) is not None:  # 检查 scale 是否在 cross_attention_kwargs 中
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")  # 发出警告，提示 scale 参数已过时

        hidden_states = self.resnets[0](hidden_states, temb)  # 使用第一个残差网络处理隐藏状态和时间嵌入
        for attn, resnet in zip(self.attentions, self.resnets[1:]):  # 遍历注意力层和后续的残差网络
            if self.training and self.gradient_checkpointing:  # 检查是否在训练模式且开启了梯度检查点

                def create_custom_forward(module, return_dict=None):  # 定义一个函数以创建自定义前向传播
                    def custom_forward(*inputs):  # 定义实际的前向传播函数
                        if return_dict is not None:  # 检查是否需要返回字典形式的输出
                            return module(*inputs, return_dict=return_dict)  # 调用模块并返回字典
                        else:  # 如果不需要字典形式的输出
                            return module(*inputs)  # 直接调用模块并返回结果

                    return custom_forward  # 返回自定义前向传播函数

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}  # 根据 PyTorch 版本设置检查点参数
                hidden_states = attn(  # 使用注意力层处理隐藏状态
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 输入编码器的隐藏状态
                    cross_attention_kwargs=cross_attention_kwargs,  # 传递交叉注意力参数
                    attention_mask=attention_mask,  # 传递注意力掩码
                    encoder_attention_mask=encoder_attention_mask,  # 传递编码器注意力掩码
                    return_dict=False,  # 不返回字典
                )[0]  # 取出输出的第一个元素
                hidden_states = torch.utils.checkpoint.checkpoint(  # 使用检查点保存内存
                    create_custom_forward(resnet),  # 创建自定义前向传播
                    hidden_states,  # 输入隐藏状态
                    temb,  # 输入时间嵌入
                    **ckpt_kwargs,  # 解包检查点参数
                )
            else:  # 如果不在训练模式或不使用梯度检查点
                hidden_states = attn(  # 使用注意力层处理隐藏状态
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 输入编码器的隐藏状态
                    cross_attention_kwargs=cross_attention_kwargs,  # 传递交叉注意力参数
                    attention_mask=attention_mask,  # 传递注意力掩码
                    encoder_attention_mask=encoder_attention_mask,  # 传递编码器注意力掩码
                    return_dict=False,  # 不返回字典
                )[0]  # 取出输出的第一个元素
                hidden_states = resnet(hidden_states, temb)  # 使用残差网络处理隐藏状态和时间嵌入

        return hidden_states  # 返回处理后的隐藏状态
# 从 diffusers.models.unets.unet_2d_blocks.UNetMidBlock2DSimpleCrossAttn 复制，替换 UNetMidBlock2DSimpleCrossAttn 为 UNetMidBlockFlatSimpleCrossAttn，ResnetBlock2D 为 ResnetBlockFlat
class UNetMidBlockFlatSimpleCrossAttn(nn.Module):
    # 初始化方法，设置各层的输入输出参数
    def __init__(
        # 输入通道数
        in_channels: int,
        # 条件嵌入通道数
        temb_channels: int,
        # Dropout 概率
        dropout: float = 0.0,
        # 网络层数
        num_layers: int = 1,
        # ResNet 的 epsilon 值
        resnet_eps: float = 1e-6,
        # ResNet 的时间缩放偏移方式
        resnet_time_scale_shift: str = "default",
        # ResNet 激活函数类型
        resnet_act_fn: str = "swish",
        # ResNet 中组的数量
        resnet_groups: int = 32,
        # 是否使用 ResNet 前归一化
        resnet_pre_norm: bool = True,
        # 注意力头的维度
        attention_head_dim: int = 1,
        # 输出缩放因子
        output_scale_factor: float = 1.0,
        # 交叉注意力的维度
        cross_attention_dim: int = 1280,
        # 是否跳过时间激活
        skip_time_act: bool = False,
        # 是否仅使用交叉注意力
        only_cross_attention: bool = False,
        # 交叉注意力的归一化方式
        cross_attention_norm: Optional[str] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 设置是否使用交叉注意力机制
        self.has_cross_attention = True

        # 设置注意力头的维度
        self.attention_head_dim = attention_head_dim
        # 确定 ResNet 的组数，若未提供则使用默认值
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # 计算头的数量
        self.num_heads = in_channels // self.attention_head_dim

        # 确保至少有一个 ResNet 块
        resnets = [
            # 创建一个 ResNet 块
            ResnetBlockFlat(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                skip_time_act=skip_time_act,
            )
        ]
        # 初始化注意力列表
        attentions = []

        # 根据层数创建对应的注意力机制
        for _ in range(num_layers):
            # 根据是否支持缩放点积注意力选择处理器
            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )

            # 添加注意力机制到列表
            attentions.append(
                Attention(
                    query_dim=in_channels,
                    cross_attention_dim=in_channels,
                    heads=self.num_heads,
                    dim_head=self.attention_head_dim,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    only_cross_attention=only_cross_attention,
                    cross_attention_norm=cross_attention_norm,
                    processor=processor,
                )
            )
            # 添加 ResNet 块到列表
            resnets.append(
                ResnetBlockFlat(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    skip_time_act=skip_time_act,
                )
            )

        # 将注意力层存入模块列表
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 块存入模块列表
        self.resnets = nn.ModuleList(resnets)

    def forward(
        # 定义前向传播的方法
        self,
        hidden_states: torch.Tensor,
        # 可选的时间嵌入张量
        temb: Optional[torch.Tensor] = None,
        # 可选的编码器隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 可选的注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的交叉注意力参数字典
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 可选的编码器注意力掩码张量
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 如果传入的 cross_attention_kwargs 为 None，则初始化为空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 检查 cross_attention_kwargs 中是否有 'scale'，如果有则发出警告，说明该参数已弃用
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # 如果 attention_mask 为 None
        if attention_mask is None:
            # 如果 encoder_hidden_states 被定义：表示我们在进行交叉注意力，因此应该使用交叉注意力掩码
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # 当 attention_mask 被定义时：我们不检查 encoder_attention_mask
            # 这是为了与 UnCLIP 兼容，UnCLIP 使用 'attention_mask' 参数作为交叉注意力掩码
            # TODO: UnCLIP 应通过 encoder_attention_mask 参数而不是 attention_mask 参数来表达交叉注意力掩码
            #       然后我们可以简化整个 if/else 块为：
            #         mask = attention_mask if encoder_hidden_states is None else encoder_attention_mask
            mask = attention_mask

        # 使用第一个残差网络处理隐藏状态和时间嵌入
        hidden_states = self.resnets[0](hidden_states, temb)
        # 遍历所有注意力层和对应的残差网络
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # 使用注意力层处理隐藏状态
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,  # 传递编码器隐藏状态
                attention_mask=mask,  # 传递掩码
                **cross_attention_kwargs,  # 传递交叉注意力参数
            )

            # 使用残差网络处理隐藏状态
            hidden_states = resnet(hidden_states, temb)

        # 返回最终的隐藏状态
        return hidden_states
```