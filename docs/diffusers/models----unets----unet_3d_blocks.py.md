# `.\diffusers\models\unets\unet_3d_blocks.py`

```py
# 版权所有 2024 HuggingFace 团队。所有权利保留。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件是按“原样”基础分发，
# 不提供任何形式的保证或条件，无论是明示或暗示。
# 有关许可证的具体权限和限制，请参阅许可证。

# 导入类型提示中的任何类型
from typing import Any, Dict, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 PyTorch 导入神经网络模块
from torch import nn

# 导入实用工具函数，包括弃用和日志记录
from ...utils import deprecate, is_torch_version, logging
# 导入 PyTorch 相关的工具函数
from ...utils.torch_utils import apply_freeu
# 导入注意力机制相关的类
from ..attention import Attention
# 导入 ResNet 相关的类
from ..resnet import (
    Downsample2D,  # 导入 2D 下采样模块
    ResnetBlock2D,  # 导入 2D ResNet 块
    SpatioTemporalResBlock,  # 导入时空 ResNet 块
    TemporalConvLayer,  # 导入时间卷积层
    Upsample2D,  # 导入 2D 上采样模块
)
# 导入 2D 变换器模型
from ..transformers.transformer_2d import Transformer2DModel
# 导入时间相关的变换器模型
from ..transformers.transformer_temporal import (
    TransformerSpatioTemporalModel,  # 导入时空变换器模型
    TransformerTemporalModel,  # 导入时间变换器模型
)
# 导入运动模型的 UNet 相关类
from .unet_motion_model import (
    CrossAttnDownBlockMotion,  # 导入交叉注意力下块运动类
    CrossAttnUpBlockMotion,  # 导入交叉注意力上块运动类
    DownBlockMotion,  # 导入下块运动类
    UNetMidBlockCrossAttnMotion,  # 导入中间块交叉注意力运动类
    UpBlockMotion,  # 导入上块运动类
)

# 创建一个日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义 DownBlockMotion 类，继承自 DownBlockMotion
class DownBlockMotion(DownBlockMotion):
    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 设置弃用消息，提醒用户变更
        deprecation_message = "Importing `DownBlockMotion` from `diffusers.models.unets.unet_3d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_motion_model import DownBlockMotion` instead."
        # 调用弃用函数，记录弃用信息
        deprecate("DownBlockMotion", "1.0.0", deprecation_message)
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

# 定义 CrossAttnDownBlockMotion 类，继承自 CrossAttnDownBlockMotion
class CrossAttnDownBlockMotion(CrossAttnDownBlockMotion):
    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 设置弃用消息，提醒用户变更
        deprecation_message = "Importing `CrossAttnDownBlockMotion` from `diffusers.models.unets.unet_3d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_motion_model import CrossAttnDownBlockMotion` instead."
        # 调用弃用函数，记录弃用信息
        deprecate("CrossAttnDownBlockMotion", "1.0.0", deprecation_message)
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

# 定义 UpBlockMotion 类，继承自 UpBlockMotion
class UpBlockMotion(UpBlockMotion):
    # 初始化方法，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 设置弃用消息，提醒用户变更
        deprecation_message = "Importing `UpBlockMotion` from `diffusers.models.unets.unet_3d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_motion_model import UpBlockMotion` instead."
        # 调用弃用函数，记录弃用信息
        deprecate("UpBlockMotion", "1.0.0", deprecation_message)
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

# 定义 CrossAttnUpBlockMotion 类，继承自 CrossAttnUpBlockMotion
class CrossAttnUpBlockMotion(CrossAttnUpBlockMotion):
    # 初始化方法，用于创建类的实例
        def __init__(self, *args, **kwargs):
            # 定义一个关于导入的弃用警告信息
            deprecation_message = "Importing `CrossAttnUpBlockMotion` from `diffusers.models.unets.unet_3d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_motion_model import CrossAttnUpBlockMotion` instead."
            # 调用弃用警告函数，记录该功能的弃用信息及版本
            deprecate("CrossAttnUpBlockMotion", "1.0.0", deprecation_message)
            # 调用父类的初始化方法，传递参数以初始化父类部分
            super().__init__(*args, **kwargs)
# 定义一个名为 UNetMidBlockCrossAttnMotion 的类，继承自同名父类
class UNetMidBlockCrossAttnMotion(UNetMidBlockCrossAttnMotion):
    # 初始化方法，接收可变参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 定义弃用警告信息，提示用户更新导入路径
        deprecation_message = "Importing `UNetMidBlockCrossAttnMotion` from `diffusers.models.unets.unet_3d_blocks` is deprecated and this will be removed in a future version. Please use `from diffusers.models.unets.unet_motion_model import UNetMidBlockCrossAttnMotion` instead."
        # 触发弃用警告
        deprecate("UNetMidBlockCrossAttnMotion", "1.0.0", deprecation_message)
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)


# 定义一个函数，返回不同类型的下采样块
def get_down_block(
    # 定义参数，类型和含义
    down_block_type: str,  # 下采样块的类型
    num_layers: int,  # 层数
    in_channels: int,  # 输入通道数
    out_channels: int,  # 输出通道数
    temb_channels: int,  # 时间嵌入通道数
    add_downsample: bool,  # 是否添加下采样
    resnet_eps: float,  # ResNet 的 epsilon 参数
    resnet_act_fn: str,  # ResNet 的激活函数
    num_attention_heads: int,  # 注意力头数
    resnet_groups: Optional[int] = None,  # ResNet 的分组数，可选
    cross_attention_dim: Optional[int] = None,  # 交叉注意力维度，可选
    downsample_padding: Optional[int] = None,  # 下采样填充，可选
    dual_cross_attention: bool = False,  # 是否使用双重交叉注意力
    use_linear_projection: bool = True,  # 是否使用线性投影
    only_cross_attention: bool = False,  # 是否仅使用交叉注意力
    upcast_attention: bool = False,  # 是否提升注意力精度
    resnet_time_scale_shift: str = "default",  # ResNet 时间尺度偏移
    temporal_num_attention_heads: int = 8,  # 时间注意力头数
    temporal_max_seq_length: int = 32,  # 时间序列最大长度
    transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 每个块的变换器层数
    temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 时间变换器每块层数
    dropout: float = 0.0,  # dropout 概率
) -> Union[
    "DownBlock3D",  # 返回的可能类型之一：3D 下采样块
    "CrossAttnDownBlock3D",  # 返回的可能类型之二：交叉注意力下采样块
    "DownBlockSpatioTemporal",  # 返回的可能类型之三：时空下采样块
    "CrossAttnDownBlockSpatioTemporal",  # 返回的可能类型之四：时空交叉注意力下采样块
]:
    # 检查下采样块类型是否为 DownBlock3D
    if down_block_type == "DownBlock3D":
        # 创建并返回 DownBlock3D 实例
        return DownBlock3D(
            num_layers=num_layers,  # 传入层数
            in_channels=in_channels,  # 传入输入通道数
            out_channels=out_channels,  # 传入输出通道数
            temb_channels=temb_channels,  # 传入时间嵌入通道数
            add_downsample=add_downsample,  # 传入是否添加下采样
            resnet_eps=resnet_eps,  # 传入 ResNet 的 epsilon 参数
            resnet_act_fn=resnet_act_fn,  # 传入激活函数
            resnet_groups=resnet_groups,  # 传入分组数
            downsample_padding=downsample_padding,  # 传入下采样填充
            resnet_time_scale_shift=resnet_time_scale_shift,  # 传入时间尺度偏移
            dropout=dropout,  # 传入 dropout 概率
        )
    # 检查下采样块类型是否为 CrossAttnDownBlock3D
    elif down_block_type == "CrossAttnDownBlock3D":
        # 如果交叉注意力维度未指定，抛出错误
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        # 创建并返回 CrossAttnDownBlock3D 实例
        return CrossAttnDownBlock3D(
            num_layers=num_layers,  # 传入层数
            in_channels=in_channels,  # 传入输入通道数
            out_channels=out_channels,  # 传入输出通道数
            temb_channels=temb_channels,  # 传入时间嵌入通道数
            add_downsample=add_downsample,  # 传入是否添加下采样
            resnet_eps=resnet_eps,  # 传入 ResNet 的 epsilon 参数
            resnet_act_fn=resnet_act_fn,  # 传入激活函数
            resnet_groups=resnet_groups,  # 传入分组数
            downsample_padding=downsample_padding,  # 传入下采样填充
            cross_attention_dim=cross_attention_dim,  # 传入交叉注意力维度
            num_attention_heads=num_attention_heads,  # 传入注意力头数
            dual_cross_attention=dual_cross_attention,  # 传入是否使用双重交叉注意力
            use_linear_projection=use_linear_projection,  # 传入是否使用线性投影
            only_cross_attention=only_cross_attention,  # 传入是否仅使用交叉注意力
            upcast_attention=upcast_attention,  # 传入是否提升注意力精度
            resnet_time_scale_shift=resnet_time_scale_shift,  # 传入时间尺度偏移
            dropout=dropout,  # 传入 dropout 概率
        )
    # 检查下一个块的类型是否为时空下采样块
    elif down_block_type == "DownBlockSpatioTemporal":
        # 为 SDV 进行了添加
        # 返回一个时空下采样块的实例
        return DownBlockSpatioTemporal(
            # 设置层数
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # 是否添加下采样
            add_downsample=add_downsample,
        )
    # 检查下一个块的类型是否为交叉注意力时空下采样块
    elif down_block_type == "CrossAttnDownBlockSpatioTemporal":
        # 为 SDV 进行了添加
        # 如果没有指定交叉注意力维度，抛出错误
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlockSpatioTemporal")
        # 返回一个交叉注意力时空下采样块的实例
        return CrossAttnDownBlockSpatioTemporal(
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # 设置层数
            num_layers=num_layers,
            # 每个块的变换层数
            transformer_layers_per_block=transformer_layers_per_block,
            # 是否添加下采样
            add_downsample=add_downsample,
            # 设置交叉注意力维度
            cross_attention_dim=cross_attention_dim,
            # 注意力头数
            num_attention_heads=num_attention_heads,
        )

    # 如果块类型不匹配，则抛出错误
    raise ValueError(f"{down_block_type} does not exist.")
# 定义函数 get_up_block，返回不同类型的上采样模块
def get_up_block(
    # 上采样块类型
    up_block_type: str,
    # 层数
    num_layers: int,
    # 输入通道数
    in_channels: int,
    # 输出通道数
    out_channels: int,
    # 上一层的输出通道数
    prev_output_channel: int,
    # 时间嵌入通道数
    temb_channels: int,
    # 是否添加上采样
    add_upsample: bool,
    # ResNet 的 epsilon 值
    resnet_eps: float,
    # ResNet 的激活函数类型
    resnet_act_fn: str,
    # 注意力头的数量
    num_attention_heads: int,
    # 分辨率索引（可选）
    resolution_idx: Optional[int] = None,
    # ResNet 组数（可选）
    resnet_groups: Optional[int] = None,
    # 跨注意力维度（可选）
    cross_attention_dim: Optional[int] = None,
    # 是否使用双重跨注意力
    dual_cross_attention: bool = False,
    # 是否使用线性投影
    use_linear_projection: bool = True,
    # 是否仅使用跨注意力
    only_cross_attention: bool = False,
    # 是否提升注意力计算
    upcast_attention: bool = False,
    # ResNet 时间尺度移位的设置
    resnet_time_scale_shift: str = "default",
    # 时间上的注意力头数量
    temporal_num_attention_heads: int = 8,
    # 时间上的跨注意力维度（可选）
    temporal_cross_attention_dim: Optional[int] = None,
    # 时间序列最大长度
    temporal_max_seq_length: int = 32,
    # 每个块的变换器层数
    transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    # 每个块的时间变换器层数
    temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    # dropout 概率
    dropout: float = 0.0,
) -> Union[
    # 返回类型为不同的上采样块
    "UpBlock3D",
    "CrossAttnUpBlock3D",
    "UpBlockSpatioTemporal",
    "CrossAttnUpBlockSpatioTemporal",
]:
    # 判断上采样块类型是否为 UpBlock3D
    if up_block_type == "UpBlock3D":
        # 创建并返回 UpBlock3D 实例
        return UpBlock3D(
            # 传入参数设置
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            dropout=dropout,
        )
    # 判断上采样块类型是否为 CrossAttnUpBlock3D
    elif up_block_type == "CrossAttnUpBlock3D":
        # 检查是否提供跨注意力维度
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        # 创建并返回 CrossAttnUpBlock3D 实例
        return CrossAttnUpBlock3D(
            # 传入参数设置
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            dropout=dropout,
        )
    # 检查上升块类型是否为 "UpBlockSpatioTemporal"
    elif up_block_type == "UpBlockSpatioTemporal":
        # 为 SDV 添加的内容
        # 返回 UpBlockSpatioTemporal 实例，使用指定的参数
        return UpBlockSpatioTemporal(
            # 层数参数
            num_layers=num_layers,
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 前一个输出通道数
            prev_output_channel=prev_output_channel,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # 分辨率索引
            resolution_idx=resolution_idx,
            # 是否添加上采样
            add_upsample=add_upsample,
        )
    # 检查上升块类型是否为 "CrossAttnUpBlockSpatioTemporal"
    elif up_block_type == "CrossAttnUpBlockSpatioTemporal":
        # 为 SDV 添加的内容
        # 如果没有指定交叉注意力维度，抛出错误
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlockSpatioTemporal")
        # 返回 CrossAttnUpBlockSpatioTemporal 实例，使用指定的参数
        return CrossAttnUpBlockSpatioTemporal(
            # 输入通道数
            in_channels=in_channels,
            # 输出通道数
            out_channels=out_channels,
            # 前一个输出通道数
            prev_output_channel=prev_output_channel,
            # 时间嵌入通道数
            temb_channels=temb_channels,
            # 层数参数
            num_layers=num_layers,
            # 每个块的变换层数
            transformer_layers_per_block=transformer_layers_per_block,
            # 是否添加上采样
            add_upsample=add_upsample,
            # 交叉注意力维度
            cross_attention_dim=cross_attention_dim,
            # 注意力头数
            num_attention_heads=num_attention_heads,
            # 分辨率索引
            resolution_idx=resolution_idx,
        )

    # 如果上升块类型不符合任何已知类型，抛出错误
    raise ValueError(f"{up_block_type} does not exist.")
# 定义一个名为 UNetMidBlock3DCrossAttn 的类，继承自 nn.Module
class UNetMidBlock3DCrossAttn(nn.Module):
    # 初始化方法，设置类的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        resnet_eps: float = 1e-6,  # ResNet 中的小常数，避免除零
        resnet_time_scale_shift: str = "default",  # ResNet 时间缩放偏移
        resnet_act_fn: str = "swish",  # ResNet 激活函数类型
        resnet_groups: int = 32,  # ResNet 分组数
        resnet_pre_norm: bool = True,  # 是否在 ResNet 前进行归一化
        num_attention_heads: int = 1,  # 注意力头数
        output_scale_factor: float = 1.0,  # 输出缩放因子
        cross_attention_dim: int = 1280,  # 交叉注意力维度
        dual_cross_attention: bool = False,  # 是否使用双交叉注意力
        use_linear_projection: bool = True,  # 是否使用线性投影
        upcast_attention: bool = False,  # 是否使用上采样注意力
    ):
        # 省略具体初始化代码，通常会在这里初始化各个层和参数

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
        num_frames: int = 1,  # 帧数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 交叉注意力的额外参数
    ) -> torch.Tensor:  # 返回一个张量
        # 通过第一个 ResNet 层处理隐藏状态
        hidden_states = self.resnets[0](hidden_states, temb)
        # 通过第一个时间卷积层处理隐藏状态
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        # 遍历所有的注意力层、时间注意力层、ResNet 层和时间卷积层
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            # 通过当前注意力层处理隐藏状态
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]  # 只取返回的第一个元素
            # 通过当前时间注意力层处理隐藏状态
            hidden_states = temp_attn(
                hidden_states,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]  # 只取返回的第一个元素
            # 通过当前 ResNet 层处理隐藏状态
            hidden_states = resnet(hidden_states, temb)
            # 通过当前时间卷积层处理隐藏状态
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        # 返回最终的隐藏状态
        return hidden_states


# 定义一个名为 CrossAttnDownBlock3D 的类，继承自 nn.Module
class CrossAttnDownBlock3D(nn.Module):
    # 初始化方法，设置类的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        resnet_eps: float = 1e-6,  # ResNet 中的小常数，避免除零
        resnet_time_scale_shift: str = "default",  # ResNet 时间缩放偏移
        resnet_act_fn: str = "swish",  # ResNet 激活函数类型
        resnet_groups: int = 32,  # ResNet 分组数
        resnet_pre_norm: bool = True,  # 是否在 ResNet 前进行归一化
        num_attention_heads: int = 1,  # 注意力头数
        cross_attention_dim: int = 1280,  # 交叉注意力维度
        output_scale_factor: float = 1.0,  # 输出缩放因子
        downsample_padding: int = 1,  # 下采样的填充大小
        add_downsample: bool = True,  # 是否添加下采样层
        dual_cross_attention: bool = False,  # 是否使用双交叉注意力
        use_linear_projection: bool = False,  # 是否使用线性投影
        only_cross_attention: bool = False,  # 是否只使用交叉注意力
        upcast_attention: bool = False,  # 是否使用上采样注意力
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化残差块列表
        resnets = []
        # 初始化注意力层列表
        attentions = []
        # 初始化临时注意力层列表
        temp_attentions = []
        # 初始化临时卷积层列表
        temp_convs = []

        # 设置是否使用交叉注意力
        self.has_cross_attention = True
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads

        # 根据层数创建各层模块
        for i in range(num_layers):
            # 设置输入通道数，第一层使用传入的输入通道数，后续层使用输出通道数
            in_channels = in_channels if i == 0 else out_channels
            # 创建残差块并添加到列表中
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # 残差块的 epsilon 值
                    groups=resnet_groups,  # 组卷积的组数
                    dropout=dropout,  # Dropout 比例
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入的归一化方式
                    non_linearity=resnet_act_fn,  # 非线性激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否进行预归一化
                )
            )
            # 创建时间卷积层并添加到列表中
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,  # 输入通道数
                    out_channels,  # 输出通道数
                    dropout=0.1,  # Dropout 比例
                    norm_num_groups=resnet_groups,  # 归一化的组数
                )
            )
            # 创建二维变换器模型并添加到列表中
            attentions.append(
                Transformer2DModel(
                    out_channels // num_attention_heads,  # 每个注意力头的通道数
                    num_attention_heads,  # 注意力头的数量
                    in_channels=out_channels,  # 输入通道数
                    num_layers=1,  # 变换器层数
                    cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                    norm_num_groups=resnet_groups,  # 归一化的组数
                    use_linear_projection=use_linear_projection,  # 是否使用线性映射
                    only_cross_attention=only_cross_attention,  # 是否只使用交叉注意力
                    upcast_attention=upcast_attention,  # 是否上溢注意力
                )
            )
            # 创建时间变换器模型并添加到列表中
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // num_attention_heads,  # 每个注意力头的通道数
                    num_attention_heads,  # 注意力头的数量
                    in_channels=out_channels,  # 输入通道数
                    num_layers=1,  # 变换器层数
                    cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                    norm_num_groups=resnet_groups,  # 归一化的组数
                )
            )
        # 将残差块列表转换为模块列表
        self.resnets = nn.ModuleList(resnets)
        # 将临时卷积层列表转换为模块列表
        self.temp_convs = nn.ModuleList(temp_convs)
        # 将注意力层列表转换为模块列表
        self.attentions = nn.ModuleList(attentions)
        # 将临时注意力层列表转换为模块列表
        self.temp_attentions = nn.ModuleList(temp_attentions)

        # 如果需要添加下采样层
        if add_downsample:
            # 创建下采样模块列表
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,  # 输出通道数
                        use_conv=True,  # 是否使用卷积进行下采样
                        out_channels=out_channels,  # 输出通道数
                        padding=downsample_padding,  # 下采样时的填充
                        name="op",  # 模块名称
                    )
                ]
            )
        else:
            # 如果不需要下采样，设置为 None
            self.downsamplers = None

        # 初始化梯度检查点设置为 False
        self.gradient_checkpointing = False
    # 定义前向传播方法，接受多个输入参数并返回张量或元组
    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            num_frames: int = 1,
            cross_attention_kwargs: Dict[str, Any] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
            # TODO(Patrick, William) - 注意力掩码未使用
            output_states = ()  # 初始化输出状态为元组
    
            # 遍历所有的残差网络、临时卷积、注意力和临时注意力层
            for resnet, temp_conv, attn, temp_attn in zip(
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions
            ):
                # 使用残差网络处理隐状态和时间嵌入
                hidden_states = resnet(hidden_states, temb)
                # 使用临时卷积处理隐状态，考虑帧数
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                # 使用注意力层处理隐状态，返回字典设为 False
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]  # 取返回的第一个元素
                # 使用临时注意力层处理隐状态，返回字典设为 False
                hidden_states = temp_attn(
                    hidden_states,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]  # 取返回的第一个元素
    
                # 将当前隐状态添加到输出状态中
                output_states += (hidden_states,)
    
            # 如果存在下采样器，则逐个应用
            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)  # 应用下采样器
    
                # 将下采样后的隐状态添加到输出状态中
                output_states += (hidden_states,)
    
            # 返回最终的隐状态和所有输出状态
            return hidden_states, output_states
# 定义一个 3D 下采样模块，继承自 nn.Module
class DownBlock3D(nn.Module):
    # 初始化方法，设置各个参数及模块
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        resnet_eps: float = 1e-6,  # ResNet 中的 epsilon 值
        resnet_time_scale_shift: str = "default",  # 时间尺度偏移方式
        resnet_act_fn: str = "swish",  # ResNet 激活函数类型
        resnet_groups: int = 32,  # ResNet 中的组数
        resnet_pre_norm: bool = True,  # 是否使用预归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_downsample: bool = True,  # 是否添加下采样层
        downsample_padding: int = 1,  # 下采样的填充大小
    ):
        # 调用父类构造函数
        super().__init__()
        # 初始化 ResNet 模块列表
        resnets = []
        # 初始化时间卷积层列表
        temp_convs = []

        # 遍历层数，构建 ResNet 模块和时间卷积层
        for i in range(num_layers):
            # 第一层使用输入通道数，后续层使用输出通道数
            in_channels = in_channels if i == 0 else out_channels
            # 添加 ResNet 模块到列表中
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # ResNet 中的 epsilon 值
                    groups=resnet_groups,  # 组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化方式
                    non_linearity=resnet_act_fn,  # 激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 是否预归一化
                )
            )
            # 添加时间卷积层到列表中
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,  # 输入通道数
                    out_channels,  # 输出通道数
                    dropout=0.1,  # dropout 概率
                    norm_num_groups=resnet_groups,  # 组数
                )
            )

        # 将 ResNet 模块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)
        # 将时间卷积层列表转换为 nn.ModuleList
        self.temp_convs = nn.ModuleList(temp_convs)

        # 如果需要添加下采样层
        if add_downsample:
            # 初始化下采样模块列表
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,  # 输出通道数
                        use_conv=True,  # 使用卷积
                        out_channels=out_channels,  # 输出通道数
                        padding=downsample_padding,  # 填充大小
                        name="op",  # 模块名称
                    )
                ]
            )
        else:
            # 不添加下采样层
            self.downsamplers = None

        # 初始化梯度检查点开关为 False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        num_frames: int = 1,  # 帧数
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # 初始化输出状态元组
        output_states = ()

        # 遍历 ResNet 模块和时间卷积层
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # 通过 ResNet 模块处理隐藏状态
            hidden_states = resnet(hidden_states, temb)
            # 通过时间卷积层处理隐藏状态
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

            # 将当前的隐藏状态添加到输出状态元组中
            output_states += (hidden_states,)

        # 如果存在下采样层
        if self.downsamplers is not None:
            # 遍历下采样层
            for downsampler in self.downsamplers:
                # 通过下采样层处理隐藏状态
                hidden_states = downsampler(hidden_states)

            # 将当前的隐藏状态添加到输出状态元组中
            output_states += (hidden_states,)

        # 返回最终的隐藏状态和输出状态元组
        return hidden_states, output_states


# 定义一个 3D 交叉注意力上采样模块，继承自 nn.Module
class CrossAttnUpBlock3D(nn.Module):
    # 初始化方法，用于设置类的属性
        def __init__(
            # 输入通道数
            self,
            in_channels: int,
            # 输出通道数
            out_channels: int,
            # 前一个输出通道数
            prev_output_channel: int,
            # 时间嵌入通道数
            temb_channels: int,
            # dropout比率，默认为0.0
            dropout: float = 0.0,
            # 网络层数，默认为1
            num_layers: int = 1,
            # ResNet的epsilon值，默认为1e-6
            resnet_eps: float = 1e-6,
            # ResNet的时间尺度偏移，默认为"default"
            resnet_time_scale_shift: str = "default",
            # ResNet激活函数，默认为"swish"
            resnet_act_fn: str = "swish",
            # ResNet的组数，默认为32
            resnet_groups: int = 32,
            # ResNet是否使用预归一化，默认为True
            resnet_pre_norm: bool = True,
            # 注意力头的数量，默认为1
            num_attention_heads: int = 1,
            # 交叉注意力的维度，默认为1280
            cross_attention_dim: int = 1280,
            # 输出缩放因子，默认为1.0
            output_scale_factor: float = 1.0,
            # 是否添加上采样，默认为True
            add_upsample: bool = True,
            # 双重交叉注意力的开关，默认为False
            dual_cross_attention: bool = False,
            # 是否使用线性投影，默认为False
            use_linear_projection: bool = False,
            # 仅使用交叉注意力的开关，默认为False
            only_cross_attention: bool = False,
            # 是否上采样注意力，默认为False
            upcast_attention: bool = False,
            # 分辨率索引，默认为None
            resolution_idx: Optional[int] = None,
    ):
        # 调用父类的构造函数进行初始化
        super().__init__()
        # 初始化用于存储 ResNet 块的列表
        resnets = []
        # 初始化用于存储时间卷积层的列表
        temp_convs = []
        # 初始化用于存储注意力模型的列表
        attentions = []
        # 初始化用于存储时间注意力模型的列表
        temp_attentions = []

        # 设置是否使用交叉注意力
        self.has_cross_attention = True
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads

        # 遍历每一层以构建网络结构
        for i in range(num_layers):
            # 确定残差跳过通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 确定 ResNet 的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 添加一个 ResNet 块到列表中
            resnets.append(
                ResnetBlock2D(
                    # 设置输入通道数，包括残差跳过通道
                    in_channels=resnet_in_channels + res_skip_channels,
                    # 设置输出通道数
                    out_channels=out_channels,
                    # 设置时间嵌入通道数
                    temb_channels=temb_channels,
                    # 设置 epsilon 值
                    eps=resnet_eps,
                    # 设置组数
                    groups=resnet_groups,
                    # 设置 dropout 概率
                    dropout=dropout,
                    # 设置时间嵌入归一化方式
                    time_embedding_norm=resnet_time_scale_shift,
                    # 设置非线性激活函数
                    non_linearity=resnet_act_fn,
                    # 设置输出缩放因子
                    output_scale_factor=output_scale_factor,
                    # 设置是否在预归一化
                    pre_norm=resnet_pre_norm,
                )
            )
            # 添加一个时间卷积层到列表中
            temp_convs.append(
                TemporalConvLayer(
                    # 设置输出通道数
                    out_channels,
                    # 设置输入通道数
                    out_channels,
                    # 设置 dropout 概率
                    dropout=0.1,
                    # 设置组数
                    norm_num_groups=resnet_groups,
                )
            )
            # 添加一个 2D 转换器模型到列表中
            attentions.append(
                Transformer2DModel(
                    # 设置每个注意力头的通道数
                    out_channels // num_attention_heads,
                    # 设置注意力头的数量
                    num_attention_heads,
                    # 设置输入通道数
                    in_channels=out_channels,
                    # 设置层数
                    num_layers=1,
                    # 设置交叉注意力维度
                    cross_attention_dim=cross_attention_dim,
                    # 设置组数
                    norm_num_groups=resnet_groups,
                    # 是否使用线性投影
                    use_linear_projection=use_linear_projection,
                    # 是否仅使用交叉注意力
                    only_cross_attention=only_cross_attention,
                    # 是否提升注意力精度
                    upcast_attention=upcast_attention,
                )
            )
            # 添加一个时间转换器模型到列表中
            temp_attentions.append(
                TransformerTemporalModel(
                    # 设置每个注意力头的通道数
                    out_channels // num_attention_heads,
                    # 设置注意力头的数量
                    num_attention_heads,
                    # 设置输入通道数
                    in_channels=out_channels,
                    # 设置层数
                    num_layers=1,
                    # 设置交叉注意力维度
                    cross_attention_dim=cross_attention_dim,
                    # 设置组数
                    norm_num_groups=resnet_groups,
                )
            )
        # 将 ResNet 块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)
        # 将时间卷积层列表转换为 nn.ModuleList
        self.temp_convs = nn.ModuleList(temp_convs)
        # 将注意力模型列表转换为 nn.ModuleList
        self.attentions = nn.ModuleList(attentions)
        # 将时间注意力模型列表转换为 nn.ModuleList
        self.temp_attentions = nn.ModuleList(temp_attentions)

        # 如果需要上采样，则初始化上采样层
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            # 否则将上采样层设置为 None
            self.upsamplers = None

        # 设置梯度检查点标志
        self.gradient_checkpointing = False
        # 设置分辨率索引
        self.resolution_idx = resolution_idx
    # 定义前向传播函数，接受多个参数并返回张量
        def forward(
            self,
            hidden_states: torch.Tensor,  # 当前层的隐藏状态
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 以前层的隐藏状态元组
            temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态
            upsample_size: Optional[int] = None,  # 可选的上采样大小
            attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
            num_frames: int = 1,  # 帧数，默认值为1
            cross_attention_kwargs: Dict[str, Any] = None,  # 可选的交叉注意力参数
        ) -> torch.Tensor:  # 返回一个张量
            # 检查 FreeU 是否启用，基于多个属性的存在性
            is_freeu_enabled = (
                getattr(self, "s1", None)  # 获取属性 s1
                and getattr(self, "s2", None)  # 获取属性 s2
                and getattr(self, "b1", None)  # 获取属性 b1
                and getattr(self, "b2", None)  # 获取属性 b2
            )
    
            # TODO(Patrick, William) - 注意力掩码尚未使用
            for resnet, temp_conv, attn, temp_attn in zip(  # 遍历网络模块
                self.resnets, self.temp_convs, self.attentions, self.temp_attentions  # 从各模块提取
            ):
                # 从元组中弹出最后一个残差隐藏状态
                res_hidden_states = res_hidden_states_tuple[-1]
                # 更新元组，去掉最后一个隐藏状态
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
    
                # FreeU：仅对前两个阶段操作
                if is_freeu_enabled:
                    hidden_states, res_hidden_states = apply_freeu(  # 应用 FreeU 操作
                        self.resolution_idx,  # 当前分辨率索引
                        hidden_states,  # 当前隐藏状态
                        res_hidden_states,  # 残差隐藏状态
                        s1=self.s1,  # 属性 s1
                        s2=self.s2,  # 属性 s2
                        b1=self.b1,  # 属性 b1
                        b2=self.b2,  # 属性 b2
                    )
    
                # 将当前隐藏状态与残差隐藏状态在维度1上拼接
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    
                # 通过 ResNet 模块处理隐藏状态
                hidden_states = resnet(hidden_states, temb)
                # 通过临时卷积模块处理隐藏状态
                hidden_states = temp_conv(hidden_states, num_frames=num_frames)
                # 通过注意力模块处理隐藏状态，并提取第一个返回值
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,  # 传递编码器隐藏状态
                    cross_attention_kwargs=cross_attention_kwargs,  # 传递交叉注意力参数
                    return_dict=False,  # 不返回字典形式的结果
                )[0]  # 提取第一个返回值
                # 通过临时注意力模块处理隐藏状态，并提取第一个返回值
                hidden_states = temp_attn(
                    hidden_states,
                    num_frames=num_frames,  # 传递帧数
                    cross_attention_kwargs=cross_attention_kwargs,  # 传递交叉注意力参数
                    return_dict=False,  # 不返回字典形式的结果
                )[0]  # 提取第一个返回值
    
            # 如果存在上采样模块
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:  # 遍历上采样模块
                    hidden_states = upsampler(hidden_states, upsample_size)  # 应用上采样模块
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个名为 UpBlock3D 的类，继承自 nn.Module
class UpBlock3D(nn.Module):
    # 初始化函数，接受多个参数以配置网络层
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        prev_output_channel: int,  # 前一层的输出通道数
        out_channels: int,  # 当前层的输出通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 参数，用于数值稳定
        resnet_time_scale_shift: str = "default",  # ResNet 时间缩放偏移设置
        resnet_act_fn: str = "swish",  # ResNet 激活函数
        resnet_groups: int = 32,  # ResNet 的组数
        resnet_pre_norm: bool = True,  # 是否使用前置归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_upsample: bool = True,  # 是否添加上采样层
        resolution_idx: Optional[int] = None,  # 分辨率索引，默认为 None
    ):
        # 调用父类构造函数
        super().__init__()
        # 创建一个空列表，用于存储 ResNet 层
        resnets = []
        # 创建一个空列表，用于存储时间卷积层
        temp_convs = []

        # 根据层数创建 ResNet 和时间卷积层
        for i in range(num_layers):
            # 确定跳跃连接的通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 确定当前 ResNet 层的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 创建 ResNet 层，并添加到 resnets 列表中
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # epsilon 参数
                    groups=resnet_groups,  # 组数
                    dropout=dropout,  # dropout 概率
                    time_embedding_norm=resnet_time_scale_shift,  # 时间嵌入归一化
                    non_linearity=resnet_act_fn,  # 激活函数
                    output_scale_factor=output_scale_factor,  # 输出缩放因子
                    pre_norm=resnet_pre_norm,  # 前置归一化
                )
            )
            # 创建时间卷积层，并添加到 temp_convs 列表中
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,  # 输入通道数
                    out_channels,  # 输出通道数
                    dropout=0.1,  # dropout 概率
                    norm_num_groups=resnet_groups,  # 归一化组数
                )
            )

        # 将 ResNet 层的列表转为 nn.ModuleList，以便于管理
        self.resnets = nn.ModuleList(resnets)
        # 将时间卷积层的列表转为 nn.ModuleList，以便于管理
        self.temp_convs = nn.ModuleList(temp_convs)

        # 如果需要添加上采样层，则创建并添加
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            # 如果不需要上采样层，则设置为 None
            self.upsamplers = None

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False
        # 设置分辨率索引
        self.resolution_idx = resolution_idx

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 跳跃连接的隐藏状态元组
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        upsample_size: Optional[int] = None,  # 可选的上采样尺寸
        num_frames: int = 1,  # 帧数
    ) -> torch.Tensor:
        # 判断是否启用 FreeU，检查相关属性是否存在且不为 None
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )
        # 遍历自定义的 resnets 和 temp_convs，进行逐对处理
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # 从 res_hidden_states_tuple 中弹出最后一个隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新 res_hidden_states_tuple，去掉最后一个隐藏状态
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # 如果启用了 FreeU，则仅对前两个阶段进行操作
            if is_freeu_enabled:
                # 调用 apply_freeu 函数，处理隐藏状态
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            # 将当前的隐藏状态与残差隐藏状态在维度 1 上连接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # 通过当前的 resnet 处理隐藏状态和 temb
            hidden_states = resnet(hidden_states, temb)
            # 通过当前的 temp_conv 处理隐藏状态，传入 num_frames 参数
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        # 如果存在上采样器，则对每个上采样器进行处理
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                # 通过当前的 upsampler 处理隐藏状态，传入 upsample_size 参数
                hidden_states = upsampler(hidden_states, upsample_size)

        # 返回最终的隐藏状态
        return hidden_states
# 定义一个中间块时间解码器类，继承自 nn.Module
class MidBlockTemporalDecoder(nn.Module):
    # 初始化函数，定义输入输出通道、注意力头维度、层数等参数
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_head_dim: int = 512,
        num_layers: int = 1,
        upcast_attention: bool = False,
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化 ResNet 和 Attention 列表
        resnets = []
        attentions = []
        # 根据层数创建相应数量的 ResNet
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            # 将 SpatioTemporalResBlock 实例添加到 ResNet 列表中
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                    merge_strategy="learned",
                    switch_spatial_to_temporal_mix=True,
                )
            )

        # 添加 Attention 实例到 Attention 列表中
        attentions.append(
            Attention(
                query_dim=in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                eps=1e-6,
                upcast_attention=upcast_attention,
                norm_num_groups=32,
                bias=True,
                residual_connection=True,
            )
        )

        # 将 Attention 和 ResNet 列表转换为 ModuleList
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    # 前向传播函数，定义输入的隐藏状态和图像指示器的处理
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_only_indicator: torch.Tensor,
    ):
        # 处理第一个 ResNet 的输出
        hidden_states = self.resnets[0](
            hidden_states,
            image_only_indicator=image_only_indicator,
        )
        # 遍历剩余的 ResNet 和 Attention，交替处理
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            hidden_states = attn(hidden_states)  # 应用注意力机制
            # 处理 ResNet 的输出
            hidden_states = resnet(
                hidden_states,
                image_only_indicator=image_only_indicator,
            )

        # 返回最终的隐藏状态
        return hidden_states


# 定义一个上采样块时间解码器类，继承自 nn.Module
class UpBlockTemporalDecoder(nn.Module):
    # 初始化函数，定义输入输出通道、层数等参数
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        add_upsample: bool = True,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化 ResNet 列表
        resnets = []
        # 根据层数创建相应数量的 ResNet
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            # 将 SpatioTemporalResBlock 实例添加到 ResNet 列表中
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                    merge_strategy="learned",
                    switch_spatial_to_temporal_mix=True,
                )
            )
        # 将 ResNet 列表转换为 ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 如果需要，初始化上采样层
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
    # 定义前向传播函数，接收隐藏状态和图像指示器作为输入，返回处理后的张量
        def forward(
            self,
            hidden_states: torch.Tensor,
            image_only_indicator: torch.Tensor,
        ) -> torch.Tensor:
            # 遍历每个 ResNet 模块，更新隐藏状态
            for resnet in self.resnets:
                hidden_states = resnet(
                    hidden_states,
                    image_only_indicator=image_only_indicator,
                )
    
            # 如果存在上采样模块，则对隐藏状态进行上采样处理
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states)
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个名为 UNetMidBlockSpatioTemporal 的类，继承自 nn.Module
class UNetMidBlockSpatioTemporal(nn.Module):
    # 初始化方法，接收多个参数以配置该模块
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        temb_channels: int,  # 时间嵌入通道数
        num_layers: int = 1,  # 层数，默认为 1
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 每个块的变换层数，默认为 1
        num_attention_heads: int = 1,  # 注意力头数，默认为 1
        cross_attention_dim: int = 1280,  # 交叉注意力维度，默认为 1280
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 设置是否使用交叉注意力标志
        self.has_cross_attention = True
        # 存储注意力头的数量
        self.num_attention_heads = num_attention_heads

        # 支持每个块的变换层数为可变的
        if isinstance(transformer_layers_per_block, int):
            # 如果是整数，则将其转换为包含 num_layers 个相同元素的列表
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # 至少有一个 ResNet 块
        resnets = [
            # 创建第一个时空残差块
            SpatioTemporalResBlock(
                in_channels=in_channels,  # 输入通道数
                out_channels=in_channels,  # 输出通道数与输入相同
                temb_channels=temb_channels,  # 时间嵌入通道数
                eps=1e-5,  # 小常数用于数值稳定性
            )
        ]
        # 初始化注意力模块列表
        attentions = []

        # 遍历层数以添加注意力和残差块
        for i in range(num_layers):
            # 添加时空变换模型到注意力列表
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,  # 注意力头数
                    in_channels // num_attention_heads,  # 每个头的通道数
                    in_channels=in_channels,  # 输入通道数
                    num_layers=transformer_layers_per_block[i],  # 当前层的变换层数
                    cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                )
            )

            # 添加另一个时空残差块到残差列表
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,  # 输入通道数
                    out_channels=in_channels,  # 输出通道数与输入相同
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=1e-5,  # 小常数用于数值稳定性
                )
            )

        # 将注意力模块列表转换为 nn.ModuleList
        self.attentions = nn.ModuleList(attentions)
        # 将残差模块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态
        image_only_indicator: Optional[torch.Tensor] = None,  # 可选的图像指示张量
    # 返回类型为 torch.Tensor 的函数
    ) -> torch.Tensor:
        # 使用第一个残差网络处理隐藏状态，传入时间嵌入和图像指示器
        hidden_states = self.resnets[0](
            hidden_states,
            temb,
            image_only_indicator=image_only_indicator,
        )
    
        # 遍历注意力层和后续残差网络的组合
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            # 检查是否在训练中且开启了梯度检查点
            if self.training and self.gradient_checkpointing:  # TODO
    
                # 创建自定义前向传播函数
                def create_custom_forward(module, return_dict=None):
                    # 定义自定义前向传播内部函数
                    def custom_forward(*inputs):
                        # 根据返回字典参数决定是否使用返回字典
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
    
                    return custom_forward
    
                # 设置检查点参数，取决于 PyTorch 版本
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # 使用注意力层处理隐藏状态，传入编码器的隐藏状态和图像指示器
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                # 使用检查点进行残差网络的前向传播
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
            else:
                # 使用注意力层处理隐藏状态，传入编码器的隐藏状态和图像指示器
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
                # 使用残差网络处理隐藏状态
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
    
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个下采样的时空块，继承自 nn.Module
class DownBlockSpatioTemporal(nn.Module):
    # 初始化方法，设置输入输出通道和层数等参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        num_layers: int = 1,  # 层数，默认为1
        add_downsample: bool = True,  # 是否添加下采样
    ):
        super().__init__()  # 调用父类的初始化方法
        resnets = []  # 初始化一个空列表以存储残差块

        # 根据层数创建相应数量的 SpatioTemporalResBlock
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels  # 确定当前层的输入通道数
            resnets.append(
                SpatioTemporalResBlock(  # 添加一个新的时空残差块
                    in_channels=in_channels,  # 设置输入通道
                    out_channels=out_channels,  # 设置输出通道
                    temb_channels=temb_channels,  # 设置时间嵌入通道
                    eps=1e-5,  # 设置 epsilon 值
                )
            )

        self.resnets = nn.ModuleList(resnets)  # 将残差块列表转化为 ModuleList

        # 如果需要下采样，创建下采样模块
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(  # 添加一个下采样层
                        out_channels,  # 设置输入通道
                        use_conv=True,  # 是否使用卷积进行下采样
                        out_channels=out_channels,  # 设置输出通道
                        name="op",  # 下采样层名称
                    )
                ]
            )
        else:
            self.downsamplers = None  # 不添加下采样模块

        self.gradient_checkpointing = False  # 初始化梯度检查点为 False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态输入
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入
        image_only_indicator: Optional[torch.Tensor] = None,  # 可选的图像指示器
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()  # 初始化输出状态元组
        for resnet in self.resnets:  # 遍历每个残差块
            if self.training and self.gradient_checkpointing:  # 如果在训练且启用了梯度检查点

                # 定义一个创建自定义前向传播的方法
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)  # 返回模块的前向输出

                    return custom_forward

                # 根据 PyTorch 版本进行检查点操作
                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 使用自定义前向传播
                        hidden_states,  # 输入隐藏状态
                        temb,  # 输入时间嵌入
                        image_only_indicator,  # 输入图像指示器
                        use_reentrant=False,  # 不使用重入
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 使用自定义前向传播
                        hidden_states,  # 输入隐藏状态
                        temb,  # 输入时间嵌入
                        image_only_indicator,  # 输入图像指示器
                    )
            else:
                hidden_states = resnet(  # 直接通过残差块进行前向传播
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )

            output_states = output_states + (hidden_states,)  # 将当前隐藏状态添加到输出状态中

        if self.downsamplers is not None:  # 如果存在下采样模块
            for downsampler in self.downsamplers:  # 遍历下采样模块
                hidden_states = downsampler(hidden_states)  # 对隐藏状态进行下采样

            output_states = output_states + (hidden_states,)  # 将下采样后的状态添加到输出状态中

        return hidden_states, output_states  # 返回最终的隐藏状态和输出状态元组


# 定义一个交叉注意力下采样时空块，继承自 nn.Module
class CrossAttnDownBlockSpatioTemporal(nn.Module):
    # 初始化方法，设置模型的基本参数
        def __init__(
            # 输入通道数
            self,
            in_channels: int,
            # 输出通道数
            out_channels: int,
            # 时间嵌入通道数
            temb_channels: int,
            # 层数，默认为1
            num_layers: int = 1,
            # 每个块的变换层数，可以是整数或元组，默认为1
            transformer_layers_per_block: Union[int, Tuple[int]] = 1,
            # 注意力头数，默认为1
            num_attention_heads: int = 1,
            # 交叉注意力的维度，默认为1280
            cross_attention_dim: int = 1280,
            # 是否添加下采样，默认为True
            add_downsample: bool = True,
        ):
            # 调用父类初始化方法
            super().__init__()
            # 初始化残差网络列表
            resnets = []
            # 初始化注意力层列表
            attentions = []
    
            # 设置是否使用交叉注意力为True
            self.has_cross_attention = True
            # 设置注意力头数
            self.num_attention_heads = num_attention_heads
            # 如果变换层数是整数，则扩展为列表
            if isinstance(transformer_layers_per_block, int):
                transformer_layers_per_block = [transformer_layers_per_block] * num_layers
    
            # 遍历层数以创建残差块和注意力层
            for i in range(num_layers):
                # 如果是第一层，使用输入通道数，否则使用输出通道数
                in_channels = in_channels if i == 0 else out_channels
                # 添加残差块到列表
                resnets.append(
                    SpatioTemporalResBlock(
                        # 输入通道数
                        in_channels=in_channels,
                        # 输出通道数
                        out_channels=out_channels,
                        # 时间嵌入通道数
                        temb_channels=temb_channels,
                        # 防止除零的微小值
                        eps=1e-6,
                    )
                )
                # 添加注意力模型到列表
                attentions.append(
                    TransformerSpatioTemporalModel(
                        # 注意力头数
                        num_attention_heads,
                        # 每个头的输出通道数
                        out_channels // num_attention_heads,
                        # 输入通道数
                        in_channels=out_channels,
                        # 该层的变换层数
                        num_layers=transformer_layers_per_block[i],
                        # 交叉注意力的维度
                        cross_attention_dim=cross_attention_dim,
                    )
                )
    
            # 将注意力层转换为nn.ModuleList以支持PyTorch模型
            self.attentions = nn.ModuleList(attentions)
            # 将残差层转换为nn.ModuleList以支持PyTorch模型
            self.resnets = nn.ModuleList(resnets)
    
            # 如果需要添加下采样层
            if add_downsample:
                # 添加下采样层到nn.ModuleList
                self.downsamplers = nn.ModuleList(
                    [
                        Downsample2D(
                            # 输出通道数
                            out_channels,
                            # 是否使用卷积
                            use_conv=True,
                            # 输出通道数
                            out_channels=out_channels,
                            # 填充大小
                            padding=1,
                            # 操作名称
                            name="op",
                        )
                    ]
                )
            else:
                # 如果不需要下采样层，则设置为None
                self.downsamplers = None
    
            # 初始化梯度检查点为False
            self.gradient_checkpointing = False
    
        # 前向传播方法
        def forward(
            # 隐藏状态的张量
            hidden_states: torch.Tensor,
            # 时间嵌入的可选张量
            temb: Optional[torch.Tensor] = None,
            # 编码器隐藏状态的可选张量
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 仅图像指示的可选张量
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:  # 定义返回类型为一个元组，包含一个张量和多个张量的元组
        output_states = ()  # 初始化一个空元组，用于存储输出状态

        blocks = list(zip(self.resnets, self.attentions))  # 将自定义的残差网络和注意力模块打包成一个列表
        for resnet, attn in blocks:  # 遍历每个残差网络和对应的注意力模块
            if self.training and self.gradient_checkpointing:  # 如果处于训练模式并且启用了梯度检查点

                def create_custom_forward(module, return_dict=None):  # 定义一个创建自定义前向传播函数的辅助函数
                    def custom_forward(*inputs):  # 定义自定义前向传播函数，接受任意数量的输入
                        if return_dict is not None:  # 如果提供了返回字典参数
                            return module(*inputs, return_dict=return_dict)  # 使用返回字典参数调用模块
                        else:  # 如果没有提供返回字典
                            return module(*inputs)  # 直接调用模块并返回结果

                    return custom_forward  # 返回自定义前向传播函数

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}  # 根据 PyTorch 版本设置检查点参数
                hidden_states = torch.utils.checkpoint.checkpoint(  # 使用检查点功能计算隐藏状态以节省内存
                    create_custom_forward(resnet),  # 将残差网络传入自定义前向函数
                    hidden_states,  # 将当前隐藏状态作为输入
                    temb,  # 传递时间嵌入
                    image_only_indicator,  # 传递图像指示器
                    **ckpt_kwargs,  # 解包检查点参数
                )

                hidden_states = attn(  # 使用注意力模块处理隐藏状态
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 传递编码器隐藏状态
                    image_only_indicator=image_only_indicator,  # 传递图像指示器
                    return_dict=False,  # 不返回字典形式的结果
                )[0]  # 获取输出的第一个元素
            else:  # 如果不处于训练模式或未启用梯度检查点
                hidden_states = resnet(  # 直接使用残差网络处理隐藏状态
                    hidden_states,  # 输入当前隐藏状态
                    temb,  # 传递时间嵌入
                    image_only_indicator=image_only_indicator,  # 传递图像指示器
                )
                hidden_states = attn(  # 使用注意力模块处理隐藏状态
                    hidden_states,  # 输入隐藏状态
                    encoder_hidden_states=encoder_hidden_states,  # 传递编码器隐藏状态
                    image_only_indicator=image_only_indicator,  # 传递图像指示器
                    return_dict=False,  # 不返回字典形式的结果
                )[0]  # 获取输出的第一个元素

            output_states = output_states + (hidden_states,)  # 将当前的隐藏状态添加到输出状态元组中

        if self.downsamplers is not None:  # 如果存在下采样模块
            for downsampler in self.downsamplers:  # 遍历每个下采样模块
                hidden_states = downsampler(hidden_states)  # 使用下采样模块处理隐藏状态

            output_states = output_states + (hidden_states,)  # 将处理后的隐藏状态添加到输出状态元组中

        return hidden_states, output_states  # 返回当前的隐藏状态和所有输出状态
# 定义一个上采样的时空块类，继承自 nn.Module
class UpBlockSpatioTemporal(nn.Module):
    # 初始化方法，定义类的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        prev_output_channel: int,  # 前一层输出通道数
        out_channels: int,  # 当前层输出通道数
        temb_channels: int,  # 时间嵌入通道数
        resolution_idx: Optional[int] = None,  # 可选的分辨率索引
        num_layers: int = 1,  # 层数，默认为1
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值
        add_upsample: bool = True,  # 是否添加上采样层
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个空的列表，用于存储 ResNet 模块
        resnets = []

        # 根据层数创建对应的 ResNet 块
        for i in range(num_layers):
            # 如果是最后一层，使用输入通道数，否则使用输出通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 第一层的输入通道为前一层输出通道，后续层为输出通道
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 创建时空 ResNet 块并添加到列表中
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,  # 输入通道数
                    out_channels=out_channels,  # 输出通道数
                    temb_channels=temb_channels,  # 时间嵌入通道数
                    eps=resnet_eps,  # ResNet 的 epsilon 值
                )
            )

        # 将 ResNet 模块列表转换为 nn.ModuleList，以便在模型中管理
        self.resnets = nn.ModuleList(resnets)

        # 如果需要添加上采样层，则创建对应的 nn.ModuleList
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            # 如果不添加上采样层，设置为 None
            self.upsamplers = None

        # 初始化梯度检查点标志
        self.gradient_checkpointing = False
        # 设置分辨率索引
        self.resolution_idx = resolution_idx

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 之前层的隐藏状态元组
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        image_only_indicator: Optional[torch.Tensor] = None,  # 可选的图像指示器张量
    ) -> torch.Tensor:  # 定义函数返回类型为 PyTorch 的张量
        for resnet in self.resnets:  # 遍历当前对象中的所有 ResNet 模型
            # pop res hidden states  # 从隐藏状态元组中提取最后一个隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]  # 获取最后的 ResNet 隐藏状态
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]  # 更新元组，移除最后一个隐藏状态

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)  # 将当前隐藏状态与 ResNet 隐藏状态在维度 1 上拼接

            if self.training and self.gradient_checkpointing:  # 如果处于训练状态并启用了梯度检查点
                def create_custom_forward(module):  # 定义用于创建自定义前向传播函数的内部函数
                    def custom_forward(*inputs):  # 自定义前向传播，接收任意数量的输入
                        return module(*inputs)  # 调用原始模块的前向传播

                    return custom_forward  # 返回自定义前向传播函数

                if is_torch_version(">=", "1.11.0"):  # 检查当前 PyTorch 版本是否大于等于 1.11.0
                    hidden_states = torch.utils.checkpoint.checkpoint(  # 使用检查点机制保存内存
                        create_custom_forward(resnet),  # 传入自定义前向传播函数
                        hidden_states,  # 传入当前的隐藏状态
                        temb,  # 传入时间嵌入
                        image_only_indicator,  # 传入图像指示器
                        use_reentrant=False,  # 禁用重入
                    )
                else:  # 如果 PyTorch 版本小于 1.11.0
                    hidden_states = torch.utils.checkpoint.checkpoint(  # 使用检查点机制保存内存
                        create_custom_forward(resnet),  # 传入自定义前向传播函数
                        hidden_states,  # 传入当前的隐藏状态
                        temb,  # 传入时间嵌入
                        image_only_indicator,  # 传入图像指示器
                    )
            else:  # 如果不是训练状态或没有启用梯度检查点
                hidden_states = resnet(  # 直接调用 ResNet 模型处理隐藏状态
                    hidden_states,  # 传入当前的隐藏状态
                    temb,  # 传入时间嵌入
                    image_only_indicator=image_only_indicator,  # 传入图像指示器
                )

        if self.upsamplers is not None:  # 如果存在上采样模块
            for upsampler in self.upsamplers:  # 遍历所有上采样模块
                hidden_states = upsampler(hidden_states)  # 调用上采样模块处理隐藏状态

        return hidden_states  # 返回处理后的隐藏状态
# 定义一个时空交叉注意力上采样块类，继承自 nn.Module
class CrossAttnUpBlockSpatioTemporal(nn.Module):
    # 初始化方法，设置网络的各个参数
    def __init__(
        # 输入通道数
        in_channels: int,
        # 输出通道数
        out_channels: int,
        # 前一层输出通道数
        prev_output_channel: int,
        # 时间嵌入通道数
        temb_channels: int,
        # 分辨率索引，可选
        resolution_idx: Optional[int] = None,
        # 层数
        num_layers: int = 1,
        # 每个块的变换器层数，支持单个整数或元组
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        # ResNet 的 epsilon 值，防止除零错误
        resnet_eps: float = 1e-6,
        # 注意力头的数量
        num_attention_heads: int = 1,
        # 交叉注意力维度
        cross_attention_dim: int = 1280,
        # 是否添加上采样层
        add_upsample: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 存储 ResNet 层的列表
        resnets = []
        # 存储注意力层的列表
        attentions = []

        # 指示是否使用交叉注意力
        self.has_cross_attention = True
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads

        # 如果是整数，将其转换为列表，包含 num_layers 个相同的元素
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # 遍历每一层，构建 ResNet 和注意力层
        for i in range(num_layers):
            # 根据当前层是否是最后一层来决定跳过连接的通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 根据当前层决定 ResNet 的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 添加时空 ResNet 块到列表中
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                )
            )
            # 添加时空变换器模型到列表中
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                )
            )

        # 将注意力层列表转换为 nn.ModuleList，以便于管理
        self.attentions = nn.ModuleList(attentions)
        # 将 ResNet 层列表转换为 nn.ModuleList，以便于管理
        self.resnets = nn.ModuleList(resnets)

        # 如果需要，添加上采样层
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            # 否则上采样层设置为 None
            self.upsamplers = None

        # 设置梯度检查点标志为 False
        self.gradient_checkpointing = False
        # 存储分辨率索引
        self.resolution_idx = resolution_idx

    # 前向传播方法，定义输入和输出
    def forward(
        # 隐藏状态张量
        hidden_states: torch.Tensor,
        # 上一层隐藏状态的元组
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        # 可选的时间嵌入张量
        temb: Optional[torch.Tensor] = None,
        # 可选的编码器隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 可选的图像指示器张量
        image_only_indicator: Optional[torch.Tensor] = None,
    # 返回一个 torch.Tensor 类型的结果
    ) -> torch.Tensor:
        # 遍历每个 resnet 和 attention 模块的组合
        for resnet, attn in zip(self.resnets, self.attentions):
            # 从隐藏状态元组中弹出最后一个 res 隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新隐藏状态元组，去掉最后一个元素
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
    
            # 在指定维度连接当前的 hidden_states 和 res_hidden_states
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    
            # 如果处于训练模式且开启了梯度检查点
            if self.training and self.gradient_checkpointing:  # TODO
                # 定义一个用于创建自定义前向传播函数的函数
                def create_custom_forward(module, return_dict=None):
                    # 定义自定义前向传播逻辑
                    def custom_forward(*inputs):
                        # 根据是否返回字典，调用模块的前向传播
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
    
                    return custom_forward
    
                # 根据 PyTorch 版本选择 checkpoint 的参数
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # 使用检查点保存内存，调用自定义前向传播
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **ckpt_kwargs,
                )
                # 通过 attention 模块处理 hidden_states
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
            else:
                # 如果不使用检查点，直接通过 resnet 模块处理 hidden_states
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                # 通过 attention 模块处理 hidden_states
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]
    
        # 如果存在上采样模块，逐个应用于 hidden_states
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
    
        # 返回处理后的 hidden_states
        return hidden_states
```